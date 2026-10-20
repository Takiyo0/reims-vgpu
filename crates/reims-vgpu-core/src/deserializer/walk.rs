//! The segment walk: each segment of an exec's command buffers goes through the decoder of its
//! encoder type, and a decoder may be carried from one segment into the next. Also the record
//! loop every decoder shares ([`next_record`]).

use std::sync::Arc;

use crate::deserializer::decoder::Decoder;
use crate::deserializer::records::{CmdHeader, SegmentHeader};
use crate::deserializer::{DeathKind, Deserializer, RaiseKind};
use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::platform::command::CommandBuffer as _;
use crate::raise::{OnFault, Raise, RaiseResult};
use crate::rt::Hold;
use crate::sync::Mutex;

/// A decoder as the walk holds it; [`Decoders`] holds it too, past the walk.
pub(crate) type Carried<P> = Arc<Mutex<Decoder<P>>>;

/// The decoders a walk made, which its caller holds past its commit: a decoder left open (still
/// carried at the walk's end, or let go un-ended by an envelope) keeps its encoder open through the
/// commit and while the walk makes the next encoder on the same command buffer. They go most recent
/// first.
#[must_use = "the decoders must outlive the caller's commit"]
pub struct Decoders<P: Platform>(Vec<Carried<P>>);

impl<P: Platform> std::fmt::Debug for Decoders<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Decoders({})", self.0.len())
    }
}

impl<P: Platform> Drop for Decoders<P> {
    fn drop(&mut self) {
        while let Some(decoder) = self.0.pop() {
            drop(decoder);
        }
    }
}

/// The encoder type of the envelope segment, which sets the command buffer's protection options
/// and carries the real header.
const ENVELOPE: u8 = 5;

/// A record header's size: a decoder's loop runs while this many bytes remain.
pub(crate) const RECORD_HEADER_SIZE: u64 = 8;
const _: () = assert!(size_of::<CmdHeader>() == 8);

/// The next record of a segment: its header and a cursor over its payload, which `segment`
/// moves past; `None` when fewer than a header's bytes remain (a 1-7 byte tail ends the loop
/// unread).
pub(crate) fn next_record(segment: &mut Cursor) -> RaiseResult<Option<(CmdHeader, Cursor)>> {
    if segment.remaining() < RECORD_HEADER_SIZE {
        return Ok(None);
    }
    let header: CmdHeader = segment.read_or_raise(OnFault::Return)?;
    // A record length under 8 is an underflow.
    let Some(length) = u64::from(header.length).checked_sub(RECORD_HEADER_SIZE) else {
        crate::raise!(
            RaiseKind::RecordUnderflow,
            format_args!("record length {} is shorter than its header", header.length)
        );
    };
    // A record overrunning the segment fails here, before dispatch.
    match segment.sub_cursor(length) {
        Ok(record) => Ok(Some((header, record))),
        Err(error) => Err(Raise::new(RaiseKind::RecordOverrun)
            .with_detail(format_args!("record not readable: {error}"))),
    }
}

impl<P: Platform> Deserializer<P> {
    /// Decodes an exec's segments, in order. Each element is a cursor over one segment; the walk
    /// takes the ones it decodes and leaves the others in place. `command_buffer` is `None` when
    /// the caller's queue answered none: its sends then do nothing, and the decoders take it as
    /// it is.
    ///
    /// A raise out of a segment faults the decoder carried into that segment (possibly the one
    /// that raised, which has already faulted itself) and goes on; a raise out of that fault
    /// goes on in its place, and the segment's ends there. On a raise the decoders it made go as
    /// it returns; the caller commits nothing then. On a return, the caller holds them past its
    /// commit ([`Decoders`]).
    pub fn decode_segments(
        &self,
        hold: Hold<'_>,
        segments: &mut [Option<Cursor>],
        command_buffer: Option<&P::CommandBuffer>,
    ) -> RaiseResult<Decoders<P>> {
        let mut made = Decoders(Vec::new());
        let mut carried: Option<Carried<P>> = None;
        let mut walked = (|| -> RaiseResult<_> {
            for slot in segments.iter_mut() {
                // A missing cursor is dereferenced: the process dies after the earlier segments.
                let Some(mut segment) = slot.take() else {
                    crate::fatal!(DeathKind::NullSegment)
                };
                // Fewer than 8 bytes are skipped, and the carried decoder is kept.
                let Ok(mut header) = segment.read::<SegmentHeader>() else {
                    *slot = Some(segment);
                    continue;
                };
                if let Err(error) = segment.truncate_to(u64::from(header.length)) {
                    *slot = Some(segment);
                    return Err(Raise::new(RaiseKind::SegmentLength)
                        .with_detail(format_args!("segment length not usable: {error}")));
                }
                // `carried` keeps the decoder carried into this segment until the decode
                // returns, so the fault below reaches it if the decode raises.
                carried = self.decode_segment(
                    hold,
                    &mut header,
                    segment,
                    carried.clone(),
                    command_buffer,
                    &mut made,
                )?;
            }
            // A decoder still carried here keeps its encoder open until the caller lets `made` go,
            // after its commit.
            Ok(())
        })();
        if walked.is_err()
            && let Some(decoder) = &carried
        {
            // The decoder's lock goes before the walk's raise can end
            let faulted = decoder.lock().fault();
            hold.cleanup_step(&mut walked, faulted)?;
        }
        match walked {
            Ok(()) => Ok(made),
            Err(raise) => Err(raise),
        }
    }

    /// One segment, whose cursor is already truncated to its header's length. `header` is the
    /// walk's, which an envelope replaces with the one it carries. Returns the decoder to carry
    /// into the next segment.
    fn decode_segment(
        &self,
        hold: Hold<'_>,
        header: &mut SegmentHeader,
        mut segment: Cursor,
        carried: Option<Carried<P>>,
        command_buffer: Option<&P::CommandBuffer>,
        made: &mut Decoders<P>,
    ) -> RaiseResult<Option<Carried<P>>> {
        if header.encoder_type == ENVELOPE {
            // A failed read ends the segment silently (a carried decoder is dropped un-ended); the
            // options stay set if only the inner header fails; the inner length is not applied; the
            // envelope is not recursive.
            let Ok(options) = segment.read::<u64>() else {
                return Ok(None);
            };
            if let Some(buffer) = command_buffer {
                buffer.set_protection_options(OnFault::Return, options)?;
            }
            let Ok(inner) = segment.read::<SegmentHeader>() else {
                return Ok(None);
            };
            *header = inner;
        }
        let decoder = if header.continues_previous != 0 {
            // The type is compared first, and no carried decoder counts as type 0, so only type 0
            // reaches the missing-decoder check (measured, testing.md 7.1).
            let carried_type = carried.as_ref().map_or(0, |d| d.lock().encoder_type());
            if header.encoder_type != carried_type {
                crate::raise!(
                    RaiseKind::ContinuationType,
                    format_args!(
                        "a continuation of encoder type {} after encoder type {carried_type}",
                        header.encoder_type
                    )
                );
            }
            let Some(decoder) = carried else {
                crate::raise!(
                    RaiseKind::ContinuationMissing,
                    "a continuation with no encoder to continue"
                );
            };
            decoder
        } else if let Some(decoder) = carried {
            // A carried decoder is reused whatever the type says (measured, testing.md 7.1).
            decoder
        } else {
            let fresh = Arc::new(Mutex::new(Decoder::for_segment(
                self.weak(),
                header.encoder_type,
                command_buffer,
            )?));
            made.0.push(Arc::clone(&fresh));
            fresh
        };
        crate::perfstats::SEGMENTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::Instant::now();
        let decoded = decoder.lock().decode(hold, header, segment);
        crate::perfstats::DECODE_SEGMENT[usize::from(header.encoder_type).min(4)].add(t.elapsed());
        decoded?;
        // Tested after the decode, on the header an envelope may have replaced
        Ok((header.continued_by_next != 0).then_some(decoder))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deserializer::decoder::mock;
    use crate::deserializer::test_lookups::Lookups;
    use crate::host::HostPtr;
    use crate::memory::cursor::CursorRange;
    use crate::platform::Handle;
    use crate::raise::death::DeathKind as DeathClass;
    use crate::raise::{Origin, RaiseId};
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::Model;
    use crate::testkit::{Arg, FakePlatform, Fk, Step, kind};

    type P = FakePlatform;

    struct Setup {
        p: Arc<FakePlatform>,
        r: Arc<Lookups>,
        d: Arc<Deserializer<P>>,
    }

    fn setup() -> Setup {
        let p = Arc::new(FakePlatform::new());
        let r = Arc::new(Lookups::new(&p));
        let d = Deserializer::new(p.clone(), Some(p.device()), r.lookup());
        p.call_log().clear();
        Setup { p, r, d }
    }

    fn id(r: &Raise) -> RaiseId {
        r.id()
    }

    fn kind_id(k: RaiseKind) -> RaiseId {
        RaiseId::from(k)
    }

    /// Guest memory for segments: each range is a boxed buffer kept alive by the test.
    struct Mem(Vec<Box<[u8]>>);

    impl Mem {
        fn new() -> Mem {
            Mem(Vec::new())
        }

        /// A cursor over `bytes`.
        fn cursor(&mut self, bytes: Vec<u8>) -> Option<Cursor> {
            let mut b = bytes.into_boxed_slice();
            let c = Cursor::with_ranges(vec![CursorRange {
                address: HostPtr::from_ptr(b.as_mut_ptr()),
                length: b.len() as u64,
            }]);
            self.0.push(b);
            Some(c)
        }
    }

    /// A segment: its 8-byte header (`length` as given) and `words` as payload.
    fn seg(length: u32, t: u8, prev: u8, next: u8, words: &[u32]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&length.to_le_bytes());
        v.extend_from_slice(&[t, prev, next, 0]);
        for w in words {
            v.extend_from_slice(&w.to_le_bytes());
        }
        v
    }

    /// A well-formed segment of `words`.
    fn full(t: u8, prev: u8, next: u8, words: &[u32]) -> Vec<u8> {
        seg(8 + 4 * words.len() as u32, t, prev, next, words)
    }

    fn events(e: &mock::Events) -> Vec<String> {
        e.lock().clone()
    }

    fn cb(s: &Setup) -> Fk<kind::CommandBuffer> {
        s.p.world().new_obj(Model::None)
    }

    /// Runs the walk under a handler, as the task's exec does.
    fn run(
        s: &Setup,
        segments: &mut [Option<Cursor>],
        cb: Option<&Fk<kind::CommandBuffer>>,
    ) -> RaiseResult<Decoders<P>> {
        s.d.decode_segments(Hold::NONE, segments, cb)
    }

    fn platform_sel(r: &Raise) -> Option<&'static str> {
        match r.origin() {
            Origin::Platform(sel) => Some(sel),
            Origin::Boundary(site) => Some(site),
            Origin::Core(_) => None,
        }
    }

    #[test]
    fn decodes_each_segment_in_order_with_fresh_decoders() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(0, 0, 0, &[1])),
            m.cursor(full(2, 0, 0, &[2, 3])),
            m.cursor(full(4, 0, 0, &[])),
        ];
        let c = cb(&s);
        let decoders = run(&s, &mut segs, Some(&c)).unwrap();
        // Taken: every decoded element is empty afterwards.
        assert!(segs.iter().all(Option::is_none));
        assert_eq!(
            events(&e),
            vec![
                "new 1 0",
                "decode 1 0 0 0 4 Some(1)",
                "new 2 2",
                "decode 2 2 0 0 8 Some(2)",
                "new 3 4",
                "decode 3 4 0 0 0 None",
            ]
        );
        // The decoders live until the caller lets them go, newest first.
        drop(decoders);
        assert_eq!(
            events(&e)[6..],
            [
                "drop 3".to_owned(),
                "drop 2".to_owned(),
                "drop 1".to_owned()
            ]
        );
        mock::uninstall();
    }

    #[test]
    fn truncates_each_cursor_to_its_header_length() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        // Header says 12: the decoder sees 4 of the 12 payload bytes.
        let mut segs = vec![m.cursor(seg(12, 1, 0, 0, &[5, 6, 7]))];
        drop(run(&s, &mut segs, Some(&cb(&s))).unwrap());
        assert_eq!(events(&e)[1], "decode 1 1 0 0 4 Some(5)");
        mock::uninstall();
    }

    #[test]
    fn a_short_range_is_skipped_and_keeps_the_carried_decoder() {
        // A range of fewer than 8 bytes is skipped silently
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(0, 0, 1, &[1])),
            m.cursor(vec![1, 2, 3, 4, 5, 6, 7]),
            m.cursor(full(0, 1, 0, &[2])),
        ];
        let _decoders = run(&s, &mut segs, Some(&cb(&s))).unwrap();
        assert_eq!(
            events(&e),
            vec![
                "new 1 0",
                "decode 1 0 0 1 4 Some(1)",
                "decode 1 0 1 0 4 Some(2)"
            ]
        );
        // The skipped element stays in the list.
        assert!(segs[0].is_none());
        assert!(segs[1].is_some());
        assert!(segs[2].is_none());
        mock::uninstall();
    }

    #[test]
    fn a_null_element_dies_after_the_earlier_segments() {
        // A missing cursor is dereferenced: the process dies
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(2, 0, 0, &[1])),
            None,
            m.cursor(full(2, 0, 0, &[])),
        ];
        let c = cb(&s);
        let r = catch_death(|| run(&s, &mut segs, Some(&c)));
        assert_eq!(
            r.err(),
            Some(crate::raise::DeathId::from(DeathKind::NullSegment).site())
        );
        assert_eq!(
            events(&e)[..2],
            ["new 1 2".to_owned(), "decode 1 2 0 0 4 Some(1)".to_owned()]
        );
        mock::uninstall();
    }

    #[test]
    fn truncate_failures_raise_and_fault_the_carried_decoder() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        // Longer than the range.
        let mut segs = vec![
            m.cursor(full(0, 0, 1, &[1])),
            m.cursor(seg(64, 0, 1, 0, &[])),
        ];
        let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::SegmentLength));
        assert_eq!(
            err.detail(),
            Some("segment length not usable: length 0x40 is past the cursor's 0x8")
        );
        // The walk's decoders go as it returns the raise.
        assert_eq!(
            events(&e),
            vec!["new 1 0", "decode 1 0 0 1 4 Some(1)", "fault 1", "drop 1"]
        );
        // The failing element is left in place.
        assert!(segs[1].is_some());
        mock::uninstall();
    }

    #[test]
    fn a_length_below_the_header_fails_to_truncate() {
        let s = setup();
        let _e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![m.cursor(seg(4, 0, 0, 0, &[1]))];
        let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
        assert_eq!(
            err.detail(),
            Some(
                "segment length not usable: length 0x4 of a cursor of 0xc is before its offset 0x8"
            )
        );
        mock::uninstall();
    }

    #[test]
    fn a_decoder_raise_faults_it_then_the_carried_one() {
        // The decoder's own fault, then the walk's on the decoder carried INTO the failing
        // segment (here the same one: a second fault).
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(1, 0, 1, &[1])),
            m.cursor(full(1, 1, 0, &[mock::RAISE])),
            m.cursor(full(1, 0, 0, &[3])),
        ];
        let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::Mock("mock.decode")));
        assert_eq!(
            events(&e),
            vec![
                "new 1 1",
                "decode 1 1 0 1 4 Some(1)",
                format!("decode 1 1 1 0 4 Some({})", mock::RAISE).as_str(),
                "fault 1",
                "fault 1",
                "drop 1"
            ]
        );
        assert!(segs[2].is_some());
        mock::uninstall();
    }

    #[test]
    fn a_fresh_decoder_raise_faults_only_itself() {
        // No decoder was carried into the segment.
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(2, 0, 0, &[1])),
            m.cursor(full(0, 0, 0, &[mock::RAISE])),
        ];
        assert!(run(&s, &mut segs, Some(&cb(&s))).is_err());
        let ev = events(&e);
        assert_eq!(ev.iter().filter(|x| x.starts_with("fault")).count(), 1);
        assert_eq!(ev[ev.len() - 3..], ["fault 2", "drop 2", "drop 1"]);
        mock::uninstall();
    }

    #[test]
    fn a_raise_goes_on_to_the_callers_handler_with_its_site_after_both_faults() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(4, 0, 1, &[1])),
            m.cursor(full(4, 1, 0, &[mock::RAISE])),
        ];
        let c = cb(&s);
        let r = s.d.decode_segments(Hold::NONE, &mut segs, Some(&c));
        assert_eq!(
            r.err().map(|r| (r.site().kind, r.site().file)),
            Some((DeathClass::UnhandledRaise, "mock.decode"))
        );
        assert_eq!(
            events(&e).iter().filter(|x| x.starts_with("fault")).count(),
            2
        );
        mock::uninstall();
    }

    #[test]
    fn a_raise_in_the_carried_decoders_fault_replaces_the_one_in_flight() {
        // The fault has no handler of its own, so its raise leaves the walk in place of the
        // segment-length raise in flight.
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(1, 0, 1, &[mock::FAULT_RAISES])),
            m.cursor(seg(64, 1, 1, 0, &[])),
        ];
        let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::Mock("mock.fault")));
        assert_eq!(events(&e)[events(&e).len() - 2..], ["fault 1", "drop 1"]);
        mock::uninstall();
    }

    /// The raise a fault's raise replaces ends in the walk: what it carries lands in the
    /// caller's holder, released at the holder's end, after the walk's decoders go
    #[test]
    fn a_raise_the_faults_raise_replaces_lands_what_it_carries() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(1, 0, 1, &[mock::FAULT_RAISES])),
            m.cursor(full(1, 1, 0, &[mock::RAISE_CARRYING])),
        ];
        let c = cb(&s);
        let held = crate::rt::Held::new();
        let hold = Hold::of(&held);
        let err = s.d.decode_segments(hold, &mut segs, Some(&c)).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::Mock("mock.fault")));
        hold.land(err);
        assert_eq!(
            events(&e)[events(&e).len() - 3..],
            ["fault 1", "fault 1", "drop 1"]
        );
        drop(held);
        assert_eq!(events(&e).last().map(String::as_str), Some("release 1"));
        mock::uninstall();
    }

    #[test]
    fn a_fault_raise_after_the_walks_raise_goes_on_with_the_faults_site() {
        let s = setup();
        let _e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(2, 0, 1, &[mock::FAULT_RAISES])),
            m.cursor(full(1, 1, 0, &[1])),
        ];
        let c = cb(&s);
        let r = s.d.decode_segments(Hold::NONE, &mut segs, Some(&c));
        // The continuation-type raise is caught by the walk; the fault's raise, a decoder's
        // `endEncoding` with `OnFault::Return`, then goes on in its place with its own site.
        assert_eq!(
            r.err().map(|r| (r.site().kind, r.site().file)),
            Some((DeathClass::UnhandledRaise, "mock.fault"))
        );
        mock::uninstall();
    }

    /// The real decoder of `t` opens its encoder and continues; the next segment overruns its
    /// range; the carried decoder's fault sends `endEncoding`, which raises, and that raise is
    /// what the caller gets.
    fn a_carried_end_encoding_raise_replaces(first: Vec<u8>, t: u8) {
        let s = setup();
        let mut m = Mem::new();
        let mut segs = vec![m.cursor(first), m.cursor(seg(64, t, 1, 0, &[]))];
        s.p.script().push(
            "endEncoding",
            Step::raise("MTLDebugException", "endEncoding raised"),
        );
        let c = cb(&s);
        let err = run(&s, &mut segs, Some(&c)).unwrap_err();
        assert_eq!(s.p.call_log().count("endEncoding"), 1);
        assert_eq!(platform_sel(&err), Some("endEncoding"), "got {err}");
    }

    #[test]
    fn a_carried_compute_decoders_end_encoding_raise_replaces_the_one_in_flight() {
        a_carried_end_encoding_raise_replaces(full(1, 0, 1, &[]), 1);
    }

    #[test]
    fn a_carried_render_decoders_end_encoding_raise_replaces_the_one_in_flight() {
        // A pass prefix of one all-zero 0x1a record (header {0x1a, 8 + 0x248}).
        let mut pass = vec![0x1a_u32, 8 + 0x248];
        pass.extend_from_slice(&[0u32; 0x248 / 4]);
        a_carried_end_encoding_raise_replaces(full(0, 0, 1, &pass), 0);
    }

    #[test]
    fn a_carried_blit_decoders_end_encoding_raise_replaces_the_one_in_flight() {
        a_carried_end_encoding_raise_replaces(full(2, 0, 1, &[]), 2);
    }

    #[test]
    fn a_carried_decoder_left_open_lives_until_the_caller_lets_the_decoders_go() {
        // A last segment with `continued_by_next` leaves the encoder open until the caller lets the
        // walk's decoders go
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![m.cursor(full(0, 0, 1, &[1]))];
        let decoders = run(&s, &mut segs, Some(&cb(&s))).unwrap();
        assert_eq!(events(&e), vec!["new 1 0", "decode 1 0 0 1 4 Some(1)"]);
        drop(decoders);
        assert_eq!(events(&e).last().map(String::as_str), Some("drop 1"));
        mock::uninstall();
    }

    #[test]
    fn continuation_without_a_decoder_depends_on_the_type() {
        // No carried decoder counts as type 0.
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![m.cursor(full(0, 1, 0, &[]))];
        let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::ContinuationMissing));
        for t in [1u8, 2, 4, 3, 9] {
            let mut segs = vec![m.cursor(full(t, 1, 0, &[]))];
            let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
            assert_eq!(id(&err), kind_id(RaiseKind::ContinuationType), "type {t}");
        }
        assert!(events(&e).is_empty());
        mock::uninstall();
    }

    #[test]
    fn continuation_of_another_type_faults_the_carried_decoder() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![m.cursor(full(2, 0, 1, &[1])), m.cursor(full(0, 1, 0, &[2]))];
        let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::ContinuationType));
        assert_eq!(
            events(&e),
            vec!["new 1 2", "decode 1 2 0 1 4 Some(1)", "fault 1", "drop 1"]
        );
        mock::uninstall();
    }

    #[test]
    fn an_invalid_type_is_refused_before_any_decoder() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        for t in [3u8, 6, 0xff] {
            let mut segs = vec![m.cursor(full(t, 0, 0, &[1]))];
            let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
            assert_eq!(id(&err), kind_id(RaiseKind::EncoderType));
        }
        assert!(events(&e).is_empty());
        mock::uninstall();
    }

    #[test]
    fn a_carried_decoder_is_reused_whatever_the_type() {
        // A carried decoder is reused for a non-continuation segment of any type
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut segs = vec![
            m.cursor(full(4, 0, 1, &[1])),
            m.cursor(full(0, 0, 1, &[2])),
            m.cursor(full(9, 0, 0, &[3])),
        ];
        let _decoders = run(&s, &mut segs, Some(&cb(&s))).unwrap();
        assert_eq!(
            events(&e),
            vec![
                "new 1 4",
                "decode 1 4 0 1 4 Some(1)",
                "decode 1 0 0 1 4 Some(2)",
                "decode 1 9 0 0 4 Some(3)"
            ]
        );
        mock::uninstall();
    }

    #[test]
    fn envelope_sets_options_then_decodes_the_inner_header() {
        // The inner length (0x99) is not applied, and the inner flags decide the carry.
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut env = seg(32, 5, 0, 0, &[]);
        env.extend_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        env.extend_from_slice(&seg(0x99, 1, 0, 1, &[7, 8]));
        let mut segs = vec![m.cursor(env), m.cursor(full(1, 1, 0, &[9]))];
        let c = cb(&s);
        let _decoders = run(&s, &mut segs, Some(&c)).unwrap();
        let calls = s.p.call_log().calls_to("setProtectionOptions:");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].obj, c.id());
        assert_eq!(calls[0].args, vec![Arg::U64(0x1234_5678_9abc_def0)]);
        assert_eq!(
            events(&e),
            vec![
                "new 1 1",
                "decode 1 1 0 1 8 Some(7)",
                "decode 1 1 1 0 4 Some(9)"
            ]
        );
        mock::uninstall();
    }

    #[test]
    fn envelope_without_options_returns_nil_silently() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        // A carried decoder, then an envelope too short for its options: the walk lets go of the
        // carried decoder un-ended, so the continuation after it has none. It is still held
        // (`Decoders`), and goes as the walk returns the raise.
        let mut segs = vec![
            m.cursor(full(0, 0, 1, &[1])),
            m.cursor(seg(12, 5, 0, 0, &[0])),
            m.cursor(full(0, 1, 0, &[2])),
        ];
        let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::ContinuationMissing));
        assert_eq!(s.p.call_log().count("setProtectionOptions:"), 0);
        assert_eq!(
            events(&e),
            vec!["new 1 0", "decode 1 0 0 1 4 Some(1)", "drop 1"]
        );
        mock::uninstall();
    }

    #[test]
    fn envelope_options_stay_set_when_the_inner_header_fails() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut env = seg(20, 5, 0, 0, &[]);
        env.extend_from_slice(&3u64.to_le_bytes());
        env.extend_from_slice(&[1, 2, 3, 4]);
        let mut segs = vec![m.cursor(env)];
        drop(run(&s, &mut segs, Some(&cb(&s))).unwrap());
        assert_eq!(s.p.call_log().count("setProtectionOptions:"), 1);
        assert!(events(&e).is_empty());
        mock::uninstall();
    }

    #[test]
    fn envelope_is_not_recursive_and_nil_buffer_takes_no_options() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut env = seg(32, 5, 0, 0, &[]);
        env.extend_from_slice(&1u64.to_le_bytes());
        env.extend_from_slice(&seg(16, 5, 0, 0, &[]));
        env.extend_from_slice(&2u64.to_le_bytes());
        let mut segs = vec![m.cursor(env)];
        let err = run(&s, &mut segs, None).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::EncoderType));
        // No command buffer: no options sent.
        assert_eq!(s.p.call_log().count("setProtectionOptions:"), 0);
        assert!(events(&e).is_empty());
        mock::uninstall();
    }

    #[test]
    fn envelope_options_raise_faults_the_carried_decoder() {
        let s = setup();
        let e = mock::install();
        let mut m = Mem::new();
        let mut env = seg(24, 5, 0, 0, &[]);
        env.extend_from_slice(&1u64.to_le_bytes());
        env.extend_from_slice(&seg(8, 0, 1, 0, &[]));
        let mut segs = vec![m.cursor(full(0, 0, 1, &[1])), m.cursor(env)];
        s.p.script().push(
            "setProtectionOptions:",
            Step::unrecognized("setProtectionOptions:"),
        );
        let err = run(&s, &mut segs, Some(&cb(&s))).unwrap_err();
        assert_eq!(platform_sel(&err), Some("setProtectionOptions:"));
        assert_eq!(events(&e)[events(&e).len() - 2..], ["fault 1", "drop 1"]);
        mock::uninstall();
    }

    #[test]
    fn next_record_splits_a_segment_into_records() {
        let mut m = Mem::new();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[7, 0, 0, 0, 12, 0, 0, 0, 0xaa, 0xbb, 0xcc, 0xdd]);
        bytes.extend_from_slice(&[9, 0, 0, 0, 8, 0, 0, 0]);
        bytes.extend_from_slice(&[1, 2, 3]);
        let mut c = m.cursor(bytes).unwrap();
        let (h, r) = next_record(&mut c).unwrap().unwrap();
        assert_eq!((h.command_id, r.remaining()), (7, 4));
        let (h, r) = next_record(&mut c).unwrap().unwrap();
        assert_eq!((h.command_id, r.remaining()), (9, 0));
        // A 3-byte tail ends the loop unread.
        assert!(next_record(&mut c).unwrap().is_none());
        assert_eq!(c.remaining(), 3);
    }

    #[test]
    fn next_record_refuses_underflow_and_overrun() {
        let mut m = Mem::new();
        let mut c = m.cursor(vec![7, 0, 0, 0, 7, 0, 0, 0, 0]).unwrap();
        let err = next_record(&mut c).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::RecordUnderflow));
        let mut c = m
            .cursor(vec![7, 0, 0, 0, 0x20, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        let err = next_record(&mut c).unwrap_err();
        assert_eq!(id(&err), kind_id(RaiseKind::RecordOverrun));
        assert_eq!(
            err.detail(),
            Some("record not readable: a sub-cursor of 0x18 bytes, with 0x4 left")
        );
    }

    /// The decoders hold the deserializer weakly and the caller holds them, so the task's
    /// release after the exec returns is the deserializer's last: its device, delegate and
    /// listener are released there, on the tearing-down thread, and the caller's letting go
    /// later releases only the decoders' own fields.
    mod teardown {
        use super::*;
        use crate::deserializer::render::test_rig::describe_empty;
        use std::sync::mpsc;

        /// One segment of each class, none continued: render (an empty pass), compute, blit,
        /// info.
        fn one_of_each(m: &mut Mem) -> Vec<Option<Cursor>> {
            let pass = describe_empty();
            let mut render = seg(8 + pass.len() as u32, 0, 0, 0, &[]);
            render.extend(pass);
            vec![
                m.cursor(render),
                m.cursor(full(1, 0, 0, &[])),
                m.cursor(full(2, 0, 0, &[])),
                m.cursor(full(4, 0, 0, &[])),
            ]
        }

        /// The live encoders: render, compute, blit.
        fn encoders(p: &FakePlatform) -> [usize; 3] {
            [
                "MTLRenderCommandEncoder",
                "MTLComputeCommandEncoder",
                "MTLBlitCommandEncoder",
            ]
            .map(|class| p.world().live(class).len())
        }

        #[test]
        fn the_tasks_release_is_the_last_while_the_caller_holds_the_decoders() {
            let Setup { p, r, d } = setup();
            let dev = p.device();
            let device_refs = dev.retain_count();
            let mut m = Mem::new();
            let mut segs = one_of_each(&mut m);
            let c = p.world().new_obj::<kind::CommandBuffer>(Model::None);
            let decoders = d.decode_segments(Hold::NONE, &mut segs, Some(&c)).unwrap();
            assert_eq!(Arc::strong_count(&d), 1, "the decoders hold no reference");
            assert_eq!(encoders(&p), [1, 1, 1]);
            let gone = Arc::downgrade(&d);
            // The task's teardown releases the deserializer.
            drop(d);
            assert!(gone.upgrade().is_none());
            assert_eq!(dev.retain_count(), device_refs - 1);
            assert_eq!(Arc::strong_count(&r), 1);
            assert!(p.world().live("MTLSharedEventListener").is_empty());
            // The decoders, their encoders, are still held.
            assert_eq!(encoders(&p), [1, 1, 1]);
            drop(decoders);
            assert_eq!(encoders(&p), [0, 0, 0]);
            assert_eq!(dev.retain_count(), device_refs - 1);
        }

        #[test]
        fn a_teardown_before_the_decoding_thread_lets_the_decoders_go_ends_on_its_own_thread() {
            // FIFO 1 is in an exec of task T, holding the walk's decoders; the root FIFO's
            // delete of T tears T down before it lets them go.
            let Setup { p, r, d } = setup();
            let dev = p.device();
            let device_refs = dev.retain_count();
            let gone = Arc::downgrade(&d);
            let task = Arc::new(Mutex::new(Some(d)));
            let (decoded_tx, decoded_rx) = mpsc::channel::<()>();
            let (torn_down_tx, torn_down_rx) = mpsc::channel::<()>();
            let fifo = {
                let task = Arc::clone(&task);
                let p = Arc::clone(&p);
                #[allow(clippy::disallowed_methods)] // test scaffolding: the decoding FIFO's thread
                std::thread::spawn(move || {
                    let decoders = {
                        let d = task.lock().clone().unwrap();
                        let mut m = Mem::new();
                        let mut segs = one_of_each(&mut m);
                        let c = p.world().new_obj::<kind::CommandBuffer>(Model::None);
                        d.decode_segments(Hold::NONE, &mut segs, Some(&c)).unwrap()
                    };
                    decoded_tx.send(()).unwrap();
                    torn_down_rx.recv().unwrap();
                    let before_pop = encoders(&p);
                    drop(decoders);
                    (before_pop, encoders(&p))
                })
            };
            decoded_rx.recv().unwrap();
            // The root FIFO's teardown: the deserializer's last release.
            let parts = task.lock().take();
            drop(parts);
            assert!(gone.upgrade().is_none());
            assert_eq!(dev.retain_count(), device_refs - 1);
            assert_eq!(Arc::strong_count(&r), 1);
            assert!(p.world().live("MTLSharedEventListener").is_empty());
            torn_down_tx.send(()).unwrap();
            // FIFO 1's letting go of the decoders releases their encoders and nothing of the
            // deserializer's.
            assert_eq!(fifo.join().unwrap(), ([1, 1, 1], [0, 0, 0]));
            assert_eq!(dev.retain_count(), device_refs - 1);
        }
    }
}
