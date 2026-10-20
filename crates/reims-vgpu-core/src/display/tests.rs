//! The display's tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use parking_lot::Mutex;

use super::*;
use crate::device::test_support::{Opts, TestDevice, device};
use crate::display::pipeline::DisplayPipelineCache;
use crate::guest::GuestPtr;
use crate::platform::Handle;
use crate::platform::command::CommandQueue;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Buffer;
use crate::platform::types::{
    CommandBufferStatus, MtlOrigin, MtlRegion, MtlSize, MtlViewport, NsRange,
};
use crate::platform::values::TextureDesc;
use crate::raise::RaiseResult;
use crate::rt::Hold;
use crate::tasks::local::LocalTask;
use crate::testkit::calllog::Arg;
use crate::testkit::capture::catch_death;
use crate::testkit::host::{FakeHostQueue, QueueMode};
use crate::testkit::objects::{Fk, Model, kind};
use crate::testkit::{CompleteOn, FakePlatform, Step};

type P = FakePlatform;

/// The display area's lines `logs` took since its last take, with their levels, in order.
pub(crate) fn display_lines(logs: &crate::testkit::logs::Capture) -> Vec<(log::Level, String)> {
    logs.take()
        .into_iter()
        .filter(|(_, target, _)| target.starts_with(concat!(env!("CARGO_CRATE_NAME"), "::display")))
        .map(|(level, _, message)| (level, message))
        .collect()
}

/// Whether any of `lines` says a display was destroyed.
fn destroyed(lines: &[(log::Level, String)]) -> bool {
    lines
        .iter()
        .any(|(_, l)| l.starts_with("display on port ") && l.contains(" destroyed"))
}

/// A device with its root task and a kernel task (task 0, local): the displays' host.
struct Env {
    p: Arc<P>,
    td: TestDevice,
    q: Arc<FakeHostQueue>,
    frames: Arc<AtomicUsize>,
}

fn env() -> Env {
    let td = device(Opts::default()).with_root_page();
    td.task_root_node(0x55, 0x4000);
    {
        {
            let hold = Hold::NONE;
            td.device
                .create_task_id(hold, 0, 0x55, 0x4000, false)
                .unwrap();
        }
    }
    let p = td.platform.clone();
    p.call_log().clear();
    Env {
        p,
        td,
        q: FakeHostQueue::new(QueueMode::Manual),
        frames: Arc::new(AtomicUsize::new(0)),
    }
}

/// The device's kernel task.
fn kernel(e: &Env) -> Arc<LocalTask<P>> {
    match e.td.device.get_task_id(OnFault::Return, 0).unwrap() {
        crate::tasks::Task::Local(t) => t,
        _ => panic!("no local kernel task"),
    }
}

/// Another device on the same platform, and its handle (its drop shuts that device down).
fn other_device(e: &Env) -> (Arc<DeviceCore<P>>, Arc<crate::device::Handle<P>>) {
    let mut c = crate::device::config::DeviceConfig::new(Some(e.p.device()));
    c.display_port_count = 1;
    c.blocks = e.td.host.blocks();
    let other = DeviceCore::create(e.p.clone(), c).unwrap();
    let core: Arc<DeviceCore<P>> = Arc::clone(&other);
    (core, other)
}

/// A descriptor with the manual queue and a counting newFrame handler.
fn desc(e: &Env) -> DisplayDescriptor {
    let mut d = DisplayDescriptor::sdk_defaults();
    d.queue = Some(e.q.clone());
    let f = e.frames.clone();
    d.new_frame_event_handler = Some(Arc::new(move || {
        f.fetch_add(1, Ordering::SeqCst);
    }));
    d
}

fn host(e: &Env) -> Arc<DeviceCore<P>> {
    Arc::clone(&e.td.device)
}

fn make_with(e: &Env, d: &DisplayDescriptor, serial: u32) -> Arc<Display<P>> {
    Display::new(&e.p, host(e), d, 0, serial).unwrap().unwrap()
}

fn make(e: &Env) -> Arc<Display<P>> {
    make_with(e, &desc(e), 0x1234)
}

fn texture_on(
    dev: &Fk<kind::Device>,
    e: &Env,
    w: u64,
    h: u64,
    usage: u64,
    fmt: u64,
) -> Fk<kind::Texture> {
    let d = e.p.texture_descriptor(&TextureDesc {
        width: w,
        height: h,
        usage,
        pixel_format: fmt,
        ..TextureDesc::default()
    });
    dev.new_texture(OnFault::Die, &d).unwrap().unwrap()
}

fn texture(e: &Env, w: u64, h: u64) -> Fk<kind::Texture> {
    texture_on(&e.p.device(), e, w, h, 4 | 1, 80)
}

fn texture_with_surface(e: &Env, w: u64, h: u64) -> Fk<kind::Texture> {
    let surface = e.p.fake_surfaces().make_simple(w, h, 0x4247_5241, 4);
    let descriptor = e.p.texture_descriptor(&TextureDesc {
        width: w,
        height: h,
        usage: 4 | 1,
        pixel_format: 80,
        ..TextureDesc::default()
    });
    e.p.device()
        .new_texture_on_surface(&descriptor, Some(&surface), 0)
        .unwrap()
        .unwrap()
}

fn cmdbuf_on(dev: &Fk<kind::Device>) -> Fk<kind::CommandBuffer> {
    dev.new_command_queue()
        .unwrap()
        .command_buffer(OnFault::Die)
        .unwrap()
        .unwrap()
}

fn cmdbuf(e: &Env) -> Fk<kind::CommandBuffer> {
    cmdbuf_on(&e.p.device())
}

fn region(w: u64, h: u64) -> MtlRegion {
    MtlRegion::new(MtlOrigin::new(1, 2, 3), MtlSize::new(w, h, 1))
}

fn present(hold: Hold<'_>, d: &Display<P>, frame: Option<Fk<kind::Texture>>) -> RaiseResult<()> {
    d.present_frame(
        hold,
        OnFault::Die,
        frame,
        0x4247_5241,
        0,
        None,
        GammaTableArgs::default(),
        None,
        Box::new(|| {}),
    )
}

fn complete(e: &Env, cb: &Fk<kind::CommandBuffer>, status: CommandBufferStatus) {
    assert!(
        e.p.world()
            .complete(cb.id(), status, None, CompleteOn::Caller)
    );
}

/// The fragment property buffer's 24 floats.
fn props(d: &Display<P>) -> [f32; 24] {
    let g = d.display_mutex.lock();
    let b = g.frag_shader_property_buf.as_ref().unwrap();
    let mut bytes = [0u8; 96];
    GuestPtr::new(b.contents()).copy_to(0, &mut bytes, DeathKind::NullContents.site());
    let mut out = [0f32; 24];
    for (o, c) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *o = f32::from_ne_bytes(*c);
    }
    out
}

/// A name in a log line: none prints `none`; a name prints itself, quoted
#[test]
fn name_arg_prints_the_name() {
    assert_eq!(name_arg(None), "none");
    let fixed = DisplayName::new("QEMU display");
    assert_eq!(name_arg(Some(&fixed)), "\"QEMU display\"");
}

/// A display name compares by its contents
#[test]
fn display_name_equality_and_conversions() {
    let a = DisplayName::from(String::from("Main"));
    let b = DisplayName::from("Main");
    assert!(a == b);
    assert!(a == "Main");
    assert!(a != "Other");
    assert_eq!(format!("{a:?}"), "\"Main\"");
}

/// The dotted ids, and the site a typed death names
#[test]
fn typed_ids() {
    assert_eq!(
        crate::raise::RaiseId::from(RaiseKind::GammaCorrupt).to_string(),
        "display.gamma.corrupt"
    );
    let site = DeathKind::GammaOneEntry.site();
    assert_eq!(
        (site.kind, site.file, site.line),
        (
            crate::raise::death::DeathKind::OutOfBounds,
            "display.gamma.one_entry",
            0
        )
    );
}

mod serial_numbers {
    use super::*;

    /// Unique per process: a duplicate warns and fails; unreserving frees it.
    #[test]
    fn unique_per_process() {
        let e = env();
        let logs = crate::testkit::logs::capture();
        assert!(DisplayCore::reserve_serial_num(&*e.p, 7));
        assert!(!DisplayCore::reserve_serial_num(&*e.p, 7));
        assert_eq!(
            display_lines(&logs),
            vec![(
                log::Level::Warn,
                "display serial number 0x7 is already in use in this process".to_owned()
            )]
        );
        assert!(DisplayCore::reserve_serial_num(&*e.p, 8));
        DisplayCore::unreserve_serial_num(&*e.p, 7);
        assert!(DisplayCore::reserve_serial_num(&*e.p, 7));
        // Unreserving an unknown number does nothing.
        DisplayCore::unreserve_serial_num(&*e.p, 99);
        assert!(!DisplayCore::reserve_serial_num(&*e.p, 8));
    }

    /// A second device in the same process cannot reuse it (testing.md 7.3).
    #[test]
    fn shared_across_devices() {
        let e = env();
        let (other, _o1) = other_device(&e);
        let a = make_with(&e, &desc(&e), 5);
        let r = Display::new(&e.p, other, &desc(&e), 0, 5).unwrap();
        assert_eq!(r.err(), Some(DisplayRefusal::SerialInUse));
        drop(a);
        let (other, _o2) = other_device(&e);
        assert!(Display::new(&e.p, other, &desc(&e), 0, 5).unwrap().is_ok());
    }
}

mod display_init {
    use super::*;

    /// A duplicate serial: nil, never attached; the source is cancelled by the partial display's
    /// end.
    #[test]
    fn duplicate_serial() {
        let e = env();
        assert!(DisplayCore::reserve_serial_num(&*e.p, 3));
        let r = Display::new(&e.p, host(&e), &desc(&e), 0, 3).unwrap();
        assert_eq!(r.err(), Some(DisplayRefusal::SerialInUse));
        assert_eq!(e.p.call_log().count("attach_display"), 0);
        assert_eq!(e.p.call_log().count("detach_display"), 0);
        // Still reserved by the first owner.
        assert!(!DisplayCore::reserve_serial_num(&*e.p, 3));
    }

    /// A glyph handler is fine.
    #[test]
    fn one_glyph_handler() {
        let e = env();
        let mut d = desc(&e);
        d.cursor_glyph_handler = Some(Arc::new(|_, _| {}));
        assert!(Display::new(&e.p, host(&e), &d, 0, 1).unwrap().is_ok());
    }

    /// A buffer Metal refuses: the assert at its line.
    #[test]
    fn buffer_asserts() {
        for (skip, kind) in [
            (0u64, DeathKind::PositionBufferRefused),
            (1, DeathKind::TexCoordBufferRefused),
            (2, DeathKind::PropertyBufferRefused),
        ] {
            let e = env();
            e.p.script()
                .after("newBufferWithLength:options:", skip, Step::Nil);
            let r = catch_death(|| Display::new(&e.p, host(&e), &desc(&e), 0, 1).map(|_| ()));
            assert_eq!(r.err(), Some(kind.site()));
        }
    }

    /// A failed pre-warm pipeline is only not cached.
    #[test]
    fn prewarm_failure_ignored() {
        let e = env();
        e.p.script().push_n(
            "newRenderPipelineStateWithDescriptor:error:",
            2,
            Step::Error(Default::default()),
        );
        let d = make(&e);
        assert!(
            d.render_pipeline_cache
                .as_ref()
                .unwrap()
                .table
                .lock()
                .is_empty()
        );
    }

    /// No queue: no source, even with a handler; the display still comes up.
    #[test]
    fn no_queue_no_source() {
        let e = env();
        let mut d = desc(&e);
        d.queue = None;
        let d = make_with(&e, &d, 1);
        assert!(d.new_frame_event_source.lock().is_none());
        assert!(d.queue().is_none());
    }

    /// The plain source calls the captured handler; merges before it runs coalesce.
    #[test]
    fn plain_source_coalesces() {
        let e = env();
        let d = make(&e);
        present(Hold::NONE, &d, Some(texture(&e, 8, 8))).unwrap();
        e.q.run_pending();
        let before = e.frames.load(Ordering::SeqCst);
        d.signal_current_frame();
        d.signal_current_frame();
        d.signal_current_frame();
        e.q.run_pending();
        assert_eq!(e.frames.load(Ordering::SeqCst), before + 1);
        assert_eq!(d.guest_present_count(), 3);
    }

    /// The copies: every property the descriptor gives, read back through the getters.
    #[test]
    fn copies_descriptor() {
        let e = env();
        let mut dd = desc(&e);
        dd.name = Some("N".into());
        dd.size_in_millimeters = (1.5, 2.5);
        dd.config_epoch = 9;
        dd.origin = DisplayCoord { x: 3, y: 4 };
        dd.scale_factor = 2;
        dd.connection_type = 5;
        dd.display_uuid = Some("U".into());
        dd.min_nits = 0.5;
        dd.max_nits = 1000.0;
        dd.max_sdr_nits = 200.0;
        let d = make_with(&e, &dd, 0x42);
        assert_eq!(d.name().map(DisplayName::current).as_deref(), Some("N"));
        assert_eq!(d.size_in_millimeters(), (1.5, 2.5));
        assert_eq!(d.config_epoch(), 9);
        assert_eq!(d.origin(), DisplayCoord { x: 3, y: 4 });
        assert_eq!(d.scale_factor(), 2);
        assert_eq!(d.connection_type(), 5);
        assert_eq!(d.display_uuid(), Some("U"));
        assert_eq!(
            (d.min_nits(), d.max_nits(), d.max_sdr_nits()),
            (0.5, 1000.0, 200.0)
        );
        assert_eq!(d.serial_num(), 0x42);
        assert_eq!(d.port(), 0);
        assert_eq!(d.color_space(), dd.color_space);
    }
}

/// The display takes a `dispatch_retain` of the host's queue of its own, beside the
/// descriptor's, and `-unplug` releases it; a fake queue has no dispatch source, so the
/// newFrame source is core's model on it
#[test]
fn the_display_retains_the_hosts_queue() {
    let e = env();
    let dd = desc(&e);
    let before = Arc::strong_count(&e.q);
    let d = make_with(&e, &dd, 0x7710);
    assert_eq!(e.q.dispatch_retains(), 1);
    // The display's reference, and the source's on its target.
    assert_eq!(Arc::strong_count(&e.q), before + 2);
    assert!(d.new_frame_event_source.lock().is_some());
    d.unplug();
    // The display's is released; the cancelled source holds its target until its cancel
    // callout has run on it.
    assert_eq!(Arc::strong_count(&e.q), before + 1);
    e.q.run_pending();
    assert_eq!(Arc::strong_count(&e.q), before);
    drop(d);
    drop(dd);
}

/// An unretained pointer to a display (`DeviceCore::display_core`: the device's
/// `display_client[port]` and the guest display's client) reaches the display's core without a
/// reference: the host's last release runs the display's end at once, on its own thread, though the
/// core is held, and after its detach the pointer reaches nothing.
mod unretained_pointer {
    use super::*;

    /// A pointer to no object reaches nothing.
    #[test]
    fn nothing_without_an_object() {
        let e = env();
        assert!(host(&e).display_core(&Weak::<Display<P>>::new()).is_none());
    }

    /// A display's pointer reaches its core through its device from its creation until its end
    /// has detached it, and through no other device.
    #[test]
    fn its_device_reaches_it_until_its_end() {
        let e = env();
        let d = make(&e);
        let weak = Arc::downgrade(&d);
        let core = host(&e)
            .display_core(&weak)
            .expect("the display's device reaches its core");
        assert!(Arc::ptr_eq(&core, d.core()));
        drop(core);
        let (other, _other_handle) = other_device(&e);
        assert!(other.display_core(&weak).is_none());
        drop(d);
        assert!(host(&e).display_core(&weak).is_none());
    }
}

/// The display's end ends with a line naming its port and serial number
#[test]
fn the_end_logs_the_port_and_serial_number() {
    let e = env();
    let d = make(&e);
    let logs = crate::testkit::logs::capture();
    drop(d);
    assert_eq!(
        display_lines(&logs).last(),
        Some(&(
            log::Level::Info,
            "display on port 0 destroyed, serial number 0x1234".to_owned()
        ))
    );
}

/// A raise in the pre-warm dies at the send: no handler is live on any path into a display's
/// creation.
#[test]
fn a_raise_in_the_prewarm_dies_at_the_send() {
    let e = env();
    e.p.script().push(
        "newRenderPipelineStateWithDescriptor:error:",
        Step::raise("NSGenericException", "x"),
    );
    let r = catch_death(|| Display::new(&e.p, host(&e), &desc(&e), 0, 0x55).map(|_| ()));
    assert_eq!(
        r.err().map(|s| (s.kind, s.file, s.line)),
        Some((
            crate::raise::DeathKind::UncaughtObjc,
            "newRenderPipelineStateWithDescriptor:error:",
            0
        ))
    );
}

/// Records its own drop (a host block's destroy callback), with the queue's strong count
/// at that moment.
struct DropProbe {
    name: &'static str,
    log: Arc<Mutex<Vec<(&'static str, usize)>>>,
    queue: Weak<FakeHostQueue>,
}
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.log.lock().push((self.name, self.queue.strong_count()));
    }
}

/// The non-opt-in newFrame source's block holds its own copy of the host's handler, and
/// `dispatch_source_cancel` + `dispatch_release` free neither it nor the source's target queue:
/// libdispatch frees the handler in the cancel callout on the queue, then releases the queue
/// when the source is disposed. So the handler's destroy runs on the queue after the display's
/// end and the device's release, and before the queue's last release.
#[test]
fn unplug_frees_new_frame_handler_and_queue_on_the_queue() {
    let e = env();
    let log = Arc::new(Mutex::new(Vec::new()));
    let probe = DropProbe {
        name: "newFrame",
        log: log.clone(),
        queue: Arc::downgrade(&e.q),
    };
    let mut dd = desc(&e);
    dd.new_frame_event_handler = Some(Arc::new(move || {
        let _ = &probe;
    }));
    let d = make_with(&e, &dd, 0x1234);
    drop(dd); // the host releases its descriptor
    e.q.run_pending(); // the source's activation: nothing pending
    let base = Arc::strong_count(&e.q);
    let logs = crate::testkit::logs::capture();
    drop(d);
    assert!(destroyed(&display_lines(&logs)));
    assert!(
        log.lock().is_empty(),
        "the source's handler copy is freed on the queue, not in the teardown"
    );
    assert_eq!(
        e.q.pending(),
        1,
        "the cancellation is processed on the queue"
    );
    assert!(
        Arc::strong_count(&e.q) > 1,
        "the source still holds its queue"
    );
    e.q.run_pending();
    let l = log.lock().clone();
    assert_eq!(l.len(), 1);
    // The handler goes while the source's queue reference is still held.
    assert!(
        l[0].1 >= 2,
        "handler freed before the source releases its queue: {l:?}"
    );
    assert!(Arc::strong_count(&e.q) < base);
    assert_eq!(
        Arc::strong_count(&e.q),
        1,
        "only the test's own reference is left"
    );
}

/// A raise out of the display's pipeline cache dies at the send: no handler is live on any path
/// into a display's creation.
#[test]
fn a_raise_in_the_pipeline_cache_dies_at_the_send() {
    let e = env();
    e.p.script().push(
        "newLibraryWithSource:options:error:",
        Step::raise("NSGenericException", "x"),
    );
    let r = catch_death(|| Display::new(&e.p, host(&e), &desc(&e), 0, 0x56).map(|_| ()));
    assert_eq!(
        r.err().map(|s| (s.kind, s.file, s.line)),
        Some((
            crate::raise::DeathKind::UncaughtObjc,
            "newLibraryWithSource:options:error:",
            0
        ))
    );
}

/// The getters read their fields without the display's mutex, so they do not wait for it.
#[test]
fn getters_do_not_wait_for_display_mutex() {
    let e = env();
    let d = make(&e);
    let g = d.display_mutex.lock();
    let d2 = d.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread
    let t = std::thread::spawn(move || {
        let _ = d2.guest_color_space();
        let _ = d2.protection_requirements();
        tx.send(()).unwrap();
    });
    let r = rx.recv_timeout(std::time::Duration::from_millis(300));
    drop(g);
    t.join().unwrap();
    assert!(
        r.is_ok(),
        "getter blocked on display_mutex; it takes no lock"
    );
}

mod unplug {
    use super::*;

    /// The display holds one queue reference, `_queue`, and `-unplug` releases it: the
    /// descriptor copy keeps none that outlives the unplug.
    #[test]
    fn unplug_releases_the_only_queue_reference() {
        let e = env();
        let mut dd = desc(&e);
        dd.new_frame_event_handler = None;
        let before = Arc::strong_count(&e.q);
        let d = make_with(&e, &dd, 0x1234);
        drop(dd);
        assert_eq!(Arc::strong_count(&e.q), before);
        d.unplug();
        assert_eq!(Arc::strong_count(&e.q), before - 1);
        drop(d);
        assert_eq!(Arc::strong_count(&e.q), before - 1);
    }

    /// After an unplug a present asserts (no queue), and a signal with a frame dies at the
    /// nil source.
    #[test]
    fn present_after_unplug_asserts() {
        let e = env();
        let d = make(&e);
        present(Hold::NONE, &d, Some(texture(&e, 8, 8))).unwrap();
        d.unplug();
        let r = catch_death(|| present(Hold::NONE, &d, None));
        assert_eq!(
            r.err(),
            Some(DeathKind::PresentWithoutNewFrameHandler.site())
        );
        let r = catch_death(|| d.signal_current_frame());
        assert_eq!(r.err(), Some(DeathKind::NoNewFrameSource.site()));
    }
}

mod set_mode_list {
    use super::*;

    fn mode(x: u16) -> DisplayMode {
        DisplayMode {
            size_in_pixels: DisplayCoord { x, y: x },
            refresh_rate_in_hz: 60.0,
            supports_edr: false,
            supports_10bpc: false,
        }
    }

    /// An empty list is ignored; a list is stored (no guest display: nothing else).
    #[test]
    fn stores_without_guest_display() {
        let e = env();
        let d = make(&e);
        d.set_mode_list(&[mode(640)]).unwrap();
        d.set_mode_list(&[]).unwrap();
        assert_eq!(d.mode_list(), vec![mode(640)]);
    }
}

mod encode_current_frame {
    use super::*;

    /// Another device's buffer or texture, or a texture without render-target usage: NO,
    /// nothing encoded.
    #[test]
    fn refusals() {
        let e = env();
        let d = make(&e);
        let other = e.p.world().new_obj::<kind::Device>(Model::None);
        let dst = texture(&e, 8, 8);
        assert!(
            !d.encode_current_frame(&cmdbuf_on(&other), &dst, region(8, 8))
                .unwrap()
        );
        let odst = texture_on(&other, &e, 8, 8, 4, 80);
        assert!(
            !d.encode_current_frame(&cmdbuf(&e), &odst, region(8, 8))
                .unwrap()
        );
        let ro = texture_on(&e.p.device(), &e, 8, 8, 1, 80);
        let cb = cmdbuf(&e);
        let m = e.p.call_log().mark();
        assert!(!d.encode_current_frame(&cb, &ro, region(8, 8)).unwrap());
        assert!(e.p.call_log().since(m).is_empty());
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 0);
    }

    /// No frame: a clear pass only (an encoder ended at once), one frame in flight until
    /// completion, and no host present counted.
    #[test]
    fn clear_only_without_frame() {
        let e = env();
        let d = make(&e);
        let cb = cmdbuf(&e);
        let dst = texture(&e, 8, 8);
        let m = e.p.call_log().mark();
        assert!(d.encode_current_frame(&cb, &dst, region(4, 4)).unwrap());
        let sels: Vec<_> = e.p.call_log().since(m).iter().map(|c| c.sel).collect();
        assert_eq!(
            sels,
            [
                "renderCommandEncoderWithDescriptor:",
                "endEncoding",
                "addCompletedHandler:"
            ]
        );
        assert_eq!(d.host_present_count(), 0);
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 1);
        complete(&e, &cb, CommandBufferStatus::COMPLETED);
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 0);
    }

    /// The internal entry encodes the current frame with the region's viewport.
    #[test]
    fn internal_entry() {
        let e = env();
        let d = make(&e);
        present(Hold::NONE, &d, Some(texture(&e, 8, 8))).unwrap();
        let cb = cmdbuf(&e);
        d.internal_encode_current_frame(&cb, &texture(&e, 8, 8), region(8, 8))
            .unwrap();
        assert_eq!(
            e.p.call_log()
                .count("drawPrimitives:vertexStart:vertexCount:"),
            1
        );
        assert_eq!(d.host_present_count(), 0);
    }
}

mod encode_render_frame {
    use super::*;

    fn vp() -> MtlViewport {
        MtlViewport {
            origin_x: 0.0,
            origin_y: 0.0,
            width: 8.0,
            height: 8.0,
            znear: 0.0,
            zfar: 0.0,
        }
    }

    /// Protection requirements go to the command buffer before the encoder.
    #[test]
    fn protection_requirements() {
        let e = env();
        let d = make(&e);
        d.mode_change(8, 8, 0, 0x5);
        let cb = cmdbuf(&e);
        let m = e.p.call_log().mark();
        d.encode_render_frame(&texture(&e, 8, 8), &texture(&e, 8, 8), &cb, vp())
            .unwrap();
        let calls = e.p.call_log().since(m);
        let i = calls
            .iter()
            .position(|c| c.sel == "setProtectionOptions:")
            .unwrap();
        assert_eq!(calls[i].args, vec![Arg::U64(5)]);
        assert_eq!(calls[i + 1].sel, "renderCommandEncoderWithDescriptor:");
    }

    /// An integer target's pipeline fails; nil is set, and the draw is Metal's crash
    /// (testing.md 7.4).
    #[test]
    fn nil_pipeline_is_set_and_metal_dies_at_draw() {
        let e = env();
        let d = make(&e);
        e.p.script().push(
            "newRenderPipelineStateWithDescriptor:error:",
            Step::Error(Default::default()),
        );
        e.p.script()
            .push("drawPrimitives:vertexStart:vertexCount:", Step::crash());
        let dst = texture_on(&e.p.device(), &e, 8, 8, 4, 73);
        let cb = cmdbuf(&e);
        let r = catch_death(|| d.encode_render_frame(&texture(&e, 8, 8), &dst, &cb, vp()));
        assert_eq!(
            r.err(),
            Some(crate::testkit::script::metal_crash_site(
                "drawPrimitives:vertexStart:vertexCount:"
            ))
        );
        let set = e.p.call_log().calls_to("setRenderPipelineState:");
        assert_eq!(set.last().unwrap().args, vec![Arg::Obj(None)]);
        assert_eq!(d.display_mutex.lock().pipeline.format, 73);
    }

    /// The completion: an error status logs; the count drops under the mutex and wakes a
    /// waiter at zero.
    #[test]
    fn completion_error_logs() {
        let e = env();
        let d = make(&e);
        let logs = crate::testkit::logs::capture();
        let cb = cmdbuf(&e);
        d.encode_render_frame(&texture(&e, 8, 8), &texture(&e, 8, 8), &cb, vp())
            .unwrap();
        let cb2 = cmdbuf(&e);
        d.encode_render_frame(&texture(&e, 8, 8), &texture(&e, 8, 8), &cb2, vp())
            .unwrap();
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 2);
        assert!(display_lines(&logs).is_empty());
        complete(&e, &cb, CommandBufferStatus::ERROR);
        // The buffer's error, none here.
        assert_eq!(
            display_lines(&logs),
            vec![(
                log::Level::Error,
                "a frame's command buffer failed, with no error".to_owned()
            )]
        );
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 1);
        complete(&e, &cb2, CommandBufferStatus::COMPLETED);
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 0);
        assert!(display_lines(&logs).is_empty());
    }

    /// The completion's error: the `NSError`'s description.
    #[test]
    fn completion_error_logs_the_ns_error() {
        let e = env();
        let d = make(&e);
        let cb = cmdbuf(&e);
        d.encode_render_frame(&texture(&e, 8, 8), &texture(&e, 8, 8), &cb, vp())
            .unwrap();
        let logs = crate::testkit::logs::capture();
        let error = crate::platform::types::NsErrorText {
            domain: "MTLCommandBufferErrorDomain".into(),
            code: 1,
            description: "Error Domain=MTLCommandBufferErrorDomain Code=1 \"x\" \
                          UserInfo={NSLocalizedDescription=x}"
                .into(),
        };
        assert!(e.p.world().complete(
            cb.id(),
            CommandBufferStatus::ERROR,
            Some(error.clone()),
            CompleteOn::Caller
        ));
        assert_eq!(
            display_lines(&logs),
            vec![(
                log::Level::Error,
                format!("a frame's command buffer failed: {}", error.description)
            )]
        );
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 0);
    }

    /// A complete IOSurface-backed macOS frame supplies its own background.
    #[test]
    fn guest_composite_frame_is_not_blended_over_host_backdrop() {
        let e = env();
        let d = make(&e);
        let cb = cmdbuf(&e);
        let dst = texture(&e, 1024, 768);
        let frame = texture_with_surface(&e, 800, 600);
        let vp_guest = MtlViewport {
            origin_x: 0.0,
            origin_y: 0.0,
            width: 800.0,
            height: 600.0,
            znear: 0.0,
            zfar: 0.0,
        };
        let m = e.p.call_log().mark();
        d.encode_render_frame(&frame, &dst, &cb, vp_guest).unwrap();
        let calls = e.p.call_log().since(m);
        assert!(!calls.iter().any(
            |c| c.sel == "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"
        ));
        let draw_calls: Vec<_> = calls
            .iter()
            .filter(|c| c.sel == "drawPrimitives:vertexStart:vertexCount:")
            .collect();
        assert_eq!(draw_calls.len(), 1);
        assert!(d.display_mutex.lock().wallpaper_backdrop.is_none());
        assert!(!d.display_mutex.lock().pipeline.blend);

        // A cached firmware fallback must also be skipped after the guest takes over.
        d.encode_render_frame(&texture(&e, 800, 600), &dst, &cb, vp_guest)
            .unwrap();
        assert!(d.display_mutex.lock().wallpaper_backdrop.is_some());
        let m = e.p.call_log().mark();
        d.encode_render_frame(&frame, &dst, &cb, vp_guest).unwrap();
        assert_eq!(
            e.p.call_log()
                .since(m)
                .iter()
                .filter(|c| c.sel == "drawPrimitives:vertexStart:vertexCount:")
                .count(),
            1
        );
        assert!(!d.display_mutex.lock().pipeline.blend);
    }

    /// An EFI frame without an IOSurface can still use source-over to reveal the host fallback.
    #[test]
    fn efi_frame_uses_backdrop_blending() {
        let e = env();
        let d = make(&e);
        let cb = cmdbuf(&e);
        let dst = texture(&e, 1024, 768);
        let frame = texture(&e, 800, 600);
        d.encode_render_frame(&frame, &dst, &cb, vp()).unwrap();
        assert!(d.display_mutex.lock().pipeline.blend);
    }
}

mod wait_for_pending_frames {
    use super::*;

    /// A held encode delays a present by 2 s of wall clock, then it logs an error and proceeds
    /// (testing.md 7.3).
    #[test]
    fn times_out_after_two_seconds() {
        let e = env();
        let d = make(&e);
        let logs = crate::testkit::logs::capture();
        let cb = cmdbuf(&e);
        d.encode_current_frame(&cb, &texture(&e, 8, 8), region(8, 8))
            .unwrap();
        std::thread::scope(|sc| {
            let t = sc.spawn(|| present(Hold::NONE, &d, None).unwrap());
            assert!(
                e.p.virtual_clock()
                    .wait_for_waiters(1, Duration::from_secs(10))
            );
            e.p.virtual_clock().advance(Duration::from_millis(1999));
            std::thread::sleep(Duration::from_millis(20));
            assert!(!t.is_finished());
            e.p.virtual_clock().advance(Duration::from_millis(1));
            t.join().unwrap();
        });
        let errors: Vec<_> = display_lines(&logs)
            .into_iter()
            .filter(|(level, _)| *level == log::Level::Error)
            .collect();
        assert_eq!(
            errors,
            vec![(
                log::Level::Error,
                "frames still in flight after 2s; going on without them".to_owned()
            )]
        );
    }

    /// A completion wakes it before the deadline, silently.
    #[test]
    fn completion_wakes() {
        let e = env();
        let d = make(&e);
        let logs = crate::testkit::logs::capture();
        let cb = cmdbuf(&e);
        d.encode_current_frame(&cb, &texture(&e, 8, 8), region(8, 8))
            .unwrap();
        std::thread::scope(|sc| {
            let t = sc.spawn(|| present(Hold::NONE, &d, None).unwrap());
            assert!(
                e.p.virtual_clock()
                    .wait_for_waiters(1, Duration::from_secs(10))
            );
            complete(&e, &cb, CommandBufferStatus::COMPLETED);
            t.join().unwrap();
        });
        assert!(
            !display_lines(&logs)
                .iter()
                .any(|(level, _)| *level == log::Level::Error)
        );
    }
}

mod signal_responsiveness {
    use super::*;

    /// A queueless display dispatches nothing (docs/testing.md, section 6.21).
    #[test]
    fn queueless_dispatches_nothing() {
        let e = env();
        let mut dd = desc(&e);
        dd.queue = None;
        dd.new_frame_event_handler = None;
        let d = make_with(&e, &dd, 3);
        d.signal_responsiveness();
        assert_eq!(d.mode_list_responsiveness(), 0);
    }
}

mod update_gamma_table {
    use super::*;

    fn args(n: u32, len: u64, sum: u32) -> GammaTableArgs {
        GammaTableArgs {
            virtual_offset: 0x40,
            mapped_length: len,
            entry_count: n,
            sum,
        }
    }

    /// A mapped length past the largest local task raises before any copy (reachable: the guest
    /// display passes mappedLength 2^62); the largest one reaches the task's copy, whose range
    /// check raises for this task
    #[test]
    fn gamma_mapped_length_past_any_task_faults() {
        use crate::device::tasks::MAX_LOCAL_TASK_LENGTH;
        let e = env();
        let d = make(&e);
        let hold = Hold::NONE;
        for len in [1 << 62, MAX_LOCAL_TASK_LENGTH + 1, u64::MAX] {
            let err = d
                .update_gamma_table(hold, &*kernel(&e), args(2, len, 0))
                .unwrap_err();
            assert_eq!(err.id(), RaiseKind::GammaLength.into());
        }
        let err = d
            .update_gamma_table(hold, &*kernel(&e), args(2, MAX_LOCAL_TASK_LENGTH, 0))
            .unwrap_err();
        assert_ne!(err.id(), RaiseKind::GammaLength.into());
    }
}

/// Reads the matrix as the image does: `ldp s0, s1, [m]` (m[0][0] and m[0][1] as one 64-bit
/// access), m[1][0], m[2][0], m[1][1], m[2][1], `ldp s2, s3, [m, #8]` (m[0][2] and m[0][3]),
/// m[1][2], m[2][2], m[1][3], m[2][3]; column j of the buffer is {m[0][j], m[1][j], m[2][j]}.
#[test]
fn update_color_matrix_read_order() {
    struct Recorder(Mutex<Vec<String>>);
    impl ColorMatrix for Recorder {
        fn at(&self, row: u8, col: u8) -> f32 {
            self.0.lock().push(format!("m{row}{col}"));
            f32::from(row * 4 + col)
        }
        fn pair(&self, row: u8, col: u8) -> (f32, f32) {
            self.0.lock().push(format!("m{row}{col}+m{row}{}", col + 1));
            (f32::from(row * 4 + col), f32::from(row * 4 + col + 1))
        }
    }
    let e = env();
    let d = make(&e);
    let m = Recorder(Mutex::new(Vec::new()));
    {
        let g = d.display_mutex.lock();
        d.update_color_matrix_locked(OnFault::Die, &g, &m).unwrap();
    }
    assert_eq!(
        *m.0.lock(),
        vec![
            "m00+m01", "m10", "m20", "m11", "m21", "m02+m03", "m12", "m22", "m13", "m23"
        ]
    );
    assert_eq!(
        &props(&d)[..16],
        &[
            0., 4., 8., 0., 1., 5., 9., 0., 2., 6., 10., 0., 3., 7., 11., 0.
        ]
    );
}

mod present_frame {
    use super::*;

    /// No newFrame handler (or no queue) asserts (testing.md 7.4).
    #[test]
    fn asserts_without_handler() {
        let e = env();
        let mut dd = desc(&e);
        dd.new_frame_event_handler = None;
        let d = make_with(&e, &dd, 1);
        let r = catch_death(|| present(Hold::NONE, &d, None));
        assert_eq!(
            r.err(),
            Some(DeathKind::PresentWithoutNewFrameHandler.site())
        );
    }

    /// The first present changes mode; the same one again does not; the frame is kept
    /// (a retain, not a copy) and the completion runs under the display mutex.
    #[test]
    fn mode_change_frame_and_completion() {
        let e = env();
        let mut dd = desc(&e);
        let changes = Arc::new(AtomicUsize::new(0));
        let c2 = changes.clone();
        dd.mode_change_handler = Some(Arc::new(move |_, _| {
            c2.fetch_add(1, Ordering::SeqCst);
        }));
        let d = make_with(&e, &dd, 1);
        let f = texture(&e, 8, 8);
        let held = f.retain_count();
        let d2 = d.clone();
        let locked = Arc::new(AtomicBool::new(false));
        let l2 = locked.clone();
        d.present_frame(
            Hold::NONE,
            OnFault::Die,
            Some(f.clone_ref()),
            0x42,
            0,
            None,
            GammaTableArgs::default(),
            None,
            Box::new(move || l2.store(d2.display_mutex.try_lock().is_none(), Ordering::SeqCst)),
        )
        .unwrap();
        assert!(locked.load(Ordering::SeqCst));
        assert_eq!(f.retain_count(), held + 1);
        assert!(d.has_current_frame.load(Ordering::SeqCst));
        present_same(&d, &f);
        e.q.run_pending();
        assert_eq!(changes.load(Ordering::SeqCst), 1);
        assert_eq!(f.retain_count(), held + 1);
        // A 0x10010-wide frame is 0x10 wide.
        let wide = texture(&e, 0x1_0010, 8);
        present_fmt(&d, Some(wide), 0x42);
        let g = d.display_mutex.lock();
        assert_eq!((g.width, g.height), (0x10, 8));
    }

    fn present_fmt(d: &Display<P>, f: Option<Fk<kind::Texture>>, fmt: u32) {
        d.present_frame(
            Hold::NONE,
            OnFault::Die,
            f,
            fmt,
            0,
            None,
            GammaTableArgs::default(),
            None,
            Box::new(|| {}),
        )
        .unwrap();
    }

    fn present_same(d: &Display<P>, f: &Fk<kind::Texture>) {
        present_fmt(d, Some(f.clone_ref()), 0x42);
    }

    /// A colour matrix goes into the buffer column by column.
    #[test]
    fn color_matrix() {
        let e = env();
        let d = make(&e);
        let m = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];
        d.present_frame(
            Hold::NONE,
            OnFault::Die,
            None,
            0,
            0,
            None,
            GammaTableArgs::default(),
            Some(&m),
            Box::new(|| {}),
        )
        .unwrap();
        assert_eq!(
            &props(&d)[..16],
            &[
                1., 5., 9., 0., 2., 6., 10., 0., 3., 7., 11., 0., 4., 8., 12., 0.
            ]
        );
    }

    /// A dirty guest colour space: re-read from the guest display (none here: nil), cleared, and it
    /// forces a mode change.
    #[test]
    fn dirty_color_space() {
        let e = env();
        let d = make(&e);
        let logs = crate::testkit::logs::capture();
        present(Hold::NONE, &d, None).unwrap();
        let m = e.p.call_log().mark();
        present(Hold::NONE, &d, None).unwrap();
        assert!(e.p.call_log().since(m).is_empty());
        d.set_guest_color_space_dirty(true);
        assert!(d.guest_color_space_dirty());
        present(Hold::NONE, &d, None).unwrap();
        assert!(!d.guest_color_space_dirty());
        assert!(d.guest_color_space().is_none());
        assert_eq!(
            display_lines(&logs)
                .iter()
                .filter(|(_, s)| s.starts_with("display on port 0: mode "))
                .count(),
            2
        );
    }
}

#[test]
fn matrix_and_slopes_write_the_buffer() {
    let e = env();
    let d = make(&e);
    let m0 = e.p.call_log().mark();
    d.update_color_matrix(OnFault::Die, &[[1.0, 2.0, 3.0, 4.0], [0.0; 4], [0.0; 4]])
        .unwrap();
    d.update_gamma_slopes([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        .unwrap();
    let p = props(&d);
    assert_eq!(
        &p[..16],
        &[
            1., 0., 0., 0., 2., 0., 0., 0., 3., 0., 0., 0., 4., 0., 0., 0.
        ]
    );
    assert_eq!(&p[16..22], &[1., 2., 3., 4., 5., 6.]);
    let mods = e.p.call_log().since(m0);
    let mods: Vec<_> = mods.iter().filter(|c| c.sel == "didModifyRange:").collect();
    assert_eq!(mods.len(), 2);
    assert!(
        mods.iter()
            .all(|c| c.args == vec![Arg::val(&NsRange::new(0, 96))])
    );
}

mod signal_current_frame {
    use super::*;

    /// Callable with `display_mutex` held (EFI's completion).
    #[test]
    fn unlocked() {
        let e = env();
        let d = make(&e);
        present(Hold::NONE, &d, Some(texture(&e, 8, 8))).unwrap();
        let _g = d.display_mutex.lock();
        d.signal_current_frame();
        assert_eq!(d.guest_present_count(), 1);
    }
}

/// No guest display: {0xffff, 0xffff}
#[test]
fn cursor_position_without_guest_display() {
    let e = env();
    let d = make(&e);
    assert_eq!(
        d.cursor_position(),
        DisplayCoord {
            x: 0xffff,
            y: 0xffff
        }
    );
}

#[test]
fn protection_options() {
    let e = env();
    let d = make(&e);
    let logs = crate::testkit::logs::capture();
    d.set_protection_options(3);
    assert_eq!(d.protection_options(), 3);
    assert_eq!(
        display_lines(&logs),
        vec![(
            log::Level::Info,
            "display on port 0: protection options 0x0 changed to 0x3".to_owned()
        )]
    );
    d.set_protection_options(3);
    assert!(display_lines(&logs).is_empty(), "no change, no line");
}

#[test]
fn handler_getters() {
    let e = env();
    let mut dd = desc(&e);
    dd.cursor_show_handler = Some(Arc::new(|_| {}));
    dd.cursor_move_handler = Some(Arc::new(|| {}));
    let d = make_with(&e, &dd, 1);
    assert!(d.new_frame_event_handler().is_some());
    assert!(d.cursor_show_handler().is_some());
    assert!(d.cursor_move_handler().is_some());
    assert!(d.mode_change_handler().is_none());
    assert!(d.cursor_glyph_handler().is_none());
    assert!(d.queue().is_some());
    assert_eq!(d.mode_list_responsiveness(), 0);
    assert_eq!((d.guest_present_count(), d.host_present_count()), (0, 0));
    assert_eq!(d.protection_requirements(), 0);
    assert!(!d.guest_color_space_dirty());
}

/// `DisplayFragmentShaderProperties`: the layout the shaders read
#[test]
fn fragment_properties_layout() {
    assert_eq!(size_of::<DisplayFragmentShaderProperties>(), 96);
    assert_eq!(
        std::mem::offset_of!(DisplayFragmentShaderProperties, gamma_table_slope_at_zero),
        64
    );
    assert_eq!(
        std::mem::offset_of!(DisplayFragmentShaderProperties, gamma_table_slope_at_one),
        76
    );
}

/// `_mtlDevice = [_device mtlDevice]` keeps no reference (the `assign` getter is a plain load); the
/// pipeline cache's retain is the display side's only one. So a live display adds to the host's
/// `MTLDevice` exactly what the display's calls add: the cache, and what its Metal objects and the
/// three buffers hold.
#[test]
fn the_display_holds_no_mtl_device_of_its_own() {
    let e = env();
    let dev = e.p.device();
    let before = dev.retain_count();
    let expected = {
        let cache = DisplayPipelineCache::<P>::new(dev.clone_ref()).unwrap();
        let mut pd = PipelineKey {
            format: 80,
            gamma: true,
            blend: false,
        };
        drop(cache.pipeline_for(pd));
        pd.gamma = false;
        drop(cache.pipeline_for(pd));
        let bufs = (
            dev.new_buffer_of_length(OnFault::Die, 64, 0).unwrap(),
            dev.new_buffer_of_length(OnFault::Die, 32, 0).unwrap(),
            dev.new_buffer_of_length(OnFault::Die, 96, 16).unwrap(),
        );
        let n = dev.retain_count() - before;
        drop((bufs, cache));
        n
    };
    assert_eq!(dev.retain_count(), before);
    let d = make(&e);
    assert_eq!(dev.retain_count() - before, expected);
    assert!(d.mtl_device().is_some_and(|m| m.id() == dev.id()));
    drop(d);
    assert_eq!(dev.retain_count(), before);
}

/// The display's real pools: the initializer's and the encode's.
mod pools {
    use super::*;
    use crate::rt::pool::test_native::{events, with_test_native};

    fn vp() -> MtlViewport {
        MtlViewport {
            origin_x: 0.0,
            origin_y: 0.0,
            width: 8.0,
            height: 8.0,
            znear: 0.0,
            zfar: 0.0,
        }
    }

    /// The pool opens after the asserts and pops before the frame is counted in flight; the
    /// cache's lookup opens none of its own.
    #[test]
    fn encode_render_frame_pops_its_pool() {
        let e = env();
        let d = make(&e);
        let (frame, dst, cb) = (texture(&e, 8, 8), texture(&e, 8, 8), cmdbuf(&e));
        with_test_native(|| {
            d.encode_render_frame(&frame, &dst, &cb, vp()).unwrap();
            assert_eq!(events(), vec!["push 0x100", "pop 0x100"]);
        });
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 1);
    }

    /// A raise inside the pool dies at its send (the host's encode has no handler), leaving the
    /// pool pushed (no EH cleanup for `@autoreleasepool`) and the in-flight count unraised.
    #[test]
    fn encode_render_frame_raise_leaves_its_pool_pushed() {
        let e = env();
        let d = make(&e);
        let (frame, dst, cb) = (texture(&e, 8, 8), texture(&e, 8, 8), cmdbuf(&e));
        e.p.script().push(
            "renderCommandEncoderWithDescriptor:",
            Step::raise("NSGenericException", "x"),
        );
        with_test_native(|| {
            let site = catch_death(|| d.encode_render_frame(&frame, &dst, &cb, vp())).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (
                    crate::raise::death::DeathKind::UncaughtObjc,
                    "renderCommandEncoderWithDescriptor:"
                )
            );
            assert_eq!(events(), vec!["push 0x100"]);
        });
        assert_eq!(d.in_flight_frame_count.load(Ordering::SeqCst), 0);
    }

    /// The initializer's one pool holds the pre-warm, the cache and the serial number's
    /// reservation, and pops before the initializer returns.
    #[test]
    fn init_pools_nest_and_pop() {
        let e = env();
        let mut got = Vec::new();
        with_test_native(|| {
            let d = make(&e);
            got = events();
            drop(d);
        });
        assert_eq!(got, ["push 0x100", "pop 0x100"]);
    }

    /// A failed initializer pops its pool, then releases the partial display, whose end runs
    /// `unplug` (its serial-number release's pool outside the first).
    #[test]
    fn goto_fail_releases_after_the_pop() {
        let e = env();
        assert!(DisplayCore::reserve_serial_num(&*e.p, 9));
        let logs = crate::testkit::logs::capture();
        let mut got = Vec::new();
        with_test_native(|| {
            assert!(
                Display::new(&e.p, host(&e), &desc(&e), 0, 9)
                    .unwrap()
                    .is_err()
            );
            got = events();
        });
        // The serial number was never this display's, so the partial display's end runs no pool of
        // its own: the initializer's pop is the last event, and the destroyed line follows it.
        assert_eq!(got.last().map(String::as_str), Some("pop 0x100"), "{got:?}");
        assert_eq!(
            display_lines(&logs).last(),
            Some(&(
                log::Level::Info,
                "display on port 0 destroyed, serial number 0x9".to_owned()
            ))
        );
    }
}
