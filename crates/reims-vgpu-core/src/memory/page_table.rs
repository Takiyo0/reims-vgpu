//! A task's guest page table, walked at use time: 1-3 levels of one guest page of `u32` entries
//! each ([`ENTRIES`]: 4096, 1024 on x86-64), each a guest page number with the writable bit on
//! top.

use std::sync::Arc;

use crate::guest::records::{APV_PAGE_MASK, APV_PAGE_SHIFT, APV_PAGE_SIZE};
use crate::memory::{DeathKind, RaiseKind};
use crate::num::{ToUsize, Truncate};
use crate::platform::Platform;
use crate::raise::{OnFault, RaiseResult};
use crate::rt::Hold;

/// The walker's read block: fills `dst` from guest physical memory at the address; the answer
/// is ignored. `Err` is the block's raise (the local task's read can raise "Read past end
/// of page"), which goes on through the walker to its caller. The walk's [`OnFault`] says what
/// that raise does.
pub type PageTableRead =
    Arc<dyn Fn(Hold<'_>, OnFault, u64, &mut [u8]) -> RaiseResult<bool> + Send + Sync>;

/// Where the walk's two error lines go, with their one number. The walker holds no platform, so
/// its owner hands it this sink; [`host_error_log`] builds it from a platform.
pub type PageTableLog = Arc<dyn Fn(WalkLog, u64) + Send + Sync>;

/// The walk's error lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkLog {
    /// A table read from an index past the table; the number is the index.
    StartOutOfTable,
    /// A table read of more entries than the table has from its index; the number is the count.
    CountOutOfTable,
}

/// A [`PageTableLog`] that writes the lines through `log`. The platform is not used.
pub fn host_error_log<P: Platform>(_platform: Arc<P>) -> PageTableLog {
    Arc::new(move |line: WalkLog, n: u64| match line {
        WalkLog::StartOutOfTable => {
            log::warn!("page table read from entry {n}, past the table");
        }
        WalkLog::CountOutOfTable => {
            log::warn!("page table read of {n} entries, past the table");
        }
    })
}

/// A walk of a task's page table (1-3 levels) from its root, reading the tables at use time.
pub struct PageTableWalker {
    root: u64,
    depth: u32,
    allow_holes: bool,
    /// Released with the walker.
    read: PageTableRead,
    log: PageTableLog,
}

impl PageTableWalker {
    /// DIAG(walker-root): the root table's guest physical address and the depth it walks.
    pub fn root_and_depth(&self) -> (u64, u32) {
        (self.root, self.depth)
    }
}

impl std::fmt::Debug for PageTableWalker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageTableWalker")
            .field("root", &self.root)
            .field("depth", &self.depth)
            .field("allow_holes", &self.allow_holes)
            .finish_non_exhaustive()
    }
}

/// One level of the walk. `entries` holds the table's [`ENTRIES`] little-endian `u32`s as the read
/// block fills them. They start zeroed, so an entry the block writes nothing into reads as 0, a
/// hole.
struct Level {
    entries: Vec<u8>,
    index: u64,
    end: u64,
    base: u64,
}

/// A table's index bits: a table is one guest page of `u32` entries.
pub(crate) const INDEX_BITS: u32 = APV_PAGE_SHIFT.wrapping_sub(2);
/// Entries per table: 4096 (1024 on x86-64).
pub(crate) const ENTRIES: u64 = 1_u64.wrapping_shl(INDEX_BITS);

impl Level {
    /// Entry `offset`. An offset past the table's last only arises from the stale index of a walk
    /// that allows holes or from level indices truncated to 32 bits; the walk dies there.
    fn entry(&self, offset: u64) -> u32 {
        let at = offset.to_usize().wrapping_mul(4);
        match self.entries.get(at..at.wrapping_add(4)) {
            Some(&[a, b, c, d]) if offset < ENTRIES => u32::from_le_bytes([a, b, c, d]),
            _ => crate::fatal!(DeathKind::WalkEntryPastTable),
        }
    }
}

/// Level `level`: below the depth by the walk's construction (depth 0 is refused before).
fn level_mut(levels: &mut [Level], level: u32) -> &mut Level {
    match levels.get_mut(level.to_usize()) {
        Some(l) => l,
        None => crate::fatal!(DeathKind::WalkEntryPastTable),
    }
}

/// Reads `count` entries of the table at guest page `pa` from index `start` into
/// `level.entries`. False when the indices fall outside the table (logged). The read block's
/// raise is as `on` says.
#[expect(
    clippy::too_many_arguments,
    reason = "the walk's hold and fault, its read and log, and one table read's values"
)]
fn read_range(
    hold: Hold<'_>,
    on: OnFault,
    read: &PageTableRead,
    log: &PageTableLog,
    pa: u32,
    level: &mut Level,
    start: u64,
    count: u64,
) -> RaiseResult<bool> {
    // A zero page (a task root of 0) dies here, at the first walk.
    crate::fatal_unless!(pa != 0, DeathKind::WalkZeroTable);
    let address = u64::from(pa).wrapping_shl(APV_PAGE_SHIFT);
    if start >= ENTRIES {
        log(WalkLog::StartOutOfTable, start);
        return Ok(false);
    }
    if start.wrapping_add(count) > ENTRIES {
        log(WalkLog::CountOutOfTable, count);
        return Ok(false);
    }
    let from = start.wrapping_shl(2).to_usize();
    let to = start.wrapping_add(count).wrapping_shl(2).to_usize();
    // `start + count` can only wrap below `ENTRIES` with a count that would copy past the table;
    // the walk dies at the copy instead.
    let Some(dst) = level.entries.get_mut(from..to) else {
        crate::fatal!(DeathKind::WalkReadPastTable)
    };
    // The block's result is ignored.
    let _ = read(hold, on, address.wrapping_add(start.wrapping_shl(2)), dst)?;
    Ok(true)
}

impl PageTableWalker {
    /// A walker of the table at guest page `root`, `depth` levels deep. No validation: root 0
    /// and depth 0 are accepted. `log` is where the walk's error lines go (normally
    /// [`host_error_log`]).
    pub fn new(
        root: u32,
        depth: u32,
        allow_holes: bool,
        read: PageTableRead,
        log: PageTableLog,
    ) -> PageTableWalker {
        PageTableWalker {
            root: u64::from(root).wrapping_shl(APV_PAGE_SHIFT),
            depth,
            allow_holes,
            read,
            log,
        }
    }

    /// Visits each mapped page of `[start, start + length)` in address order: `visitor(address,
    /// page, writable bit)`, whose answer the walker ignores. Raises the walk's refusals, the read
    /// block's raises and the visitor's (a task's copy visitor reads guest memory, which can
    /// raise); visits made before a raise stand. Its callers reach it both under a handler and
    /// with none, and say which by `on`.
    pub fn visit_pages_in_range(
        &self,
        hold: Hold<'_>,
        on: OnFault,
        start: u64,
        length: u64,
        visitor: &mut dyn FnMut(u64, u32, u32) -> RaiseResult<bool>,
    ) -> RaiseResult<()> {
        if start & APV_PAGE_MASK != 0 {
            // No guest reaches this raise: every caller passes a page-aligned start.
            return Err(RaiseKind::WalkMisalignedStart
                .raise(on, &format!("start {start:#x} is not page aligned")));
        }
        if length & APV_PAGE_MASK != 0 {
            return Err(RaiseKind::WalkMisalignedLength
                .raise(on, &format!("length {length:#x} is not whole pages")));
        }
        if length == 0 {
            return Err(RaiseKind::WalkZeroLength.raise(on, "length 0"));
        }
        let (end, overflow) = start.overflowing_add(length.wrapping_sub(1));
        if overflow {
            // An over-release here too (not modelled, as above).
            return Err(RaiseKind::WalkEndOverflow.raise(
                on,
                &format!("range at {start:#x} of {length:#x} bytes overflows"),
            ));
        }
        let start_page = start.wrapping_shr(APV_PAGE_SHIFT);
        let end_page = end.wrapping_shr(APV_PAGE_SHIFT);
        let mut levels: Vec<Level> = (0..self.depth)
            .map(|i| {
                let shift = INDEX_BITS.wrapping_mul(self.depth.wrapping_sub(1).wrapping_sub(i));
                // Both indices are truncated to 32 bits.
                Level {
                    entries: vec![0; APV_PAGE_SIZE.to_usize()],
                    index: u64::from(start_page.wrapping_shr(shift).low_u32()),
                    end: u64::from(end_page.wrapping_shr(shift).low_u32()),
                    base: 0,
                }
            })
            .collect();
        // Depth 0 has no level to read: undefined, and no caller reaches it.
        let Some(root_level) = levels.first_mut() else {
            crate::fatal!(DeathKind::WalkNoLevels)
        };
        let (first, last) = (root_level.index, root_level.end);
        // The root read covers only the indices the range touches.
        if !read_range(
            hold,
            on,
            &self.read,
            &self.log,
            self.root.wrapping_shr(APV_PAGE_SHIFT).low_u32(),
            root_level,
            first,
            last.wrapping_sub(first).wrapping_add(1),
        )? {
            return Err(RaiseKind::WalkRootUnread.raise(on, "the root table could not be read"));
        }
        let mut level: u32 = 0;
        let mut index = first;
        let mut base: u64 = 0;
        loop {
            let offset = index.wrapping_sub(base);
            let entry = level_mut(&mut levels, level).entry(offset);
            let pa = entry & 0x7fff_ffff;
            if pa == 0 {
                if !self.allow_holes {
                    // Visits made before the hole stand.
                    return Err(RaiseKind::WalkHole.raise(on, "a hole in the page table"));
                }
            } else if level < self.depth.wrapping_sub(1) {
                let current = level_mut(&mut levels, level);
                current.index = index;
                current.base = base;
                base = index.wrapping_shl(INDEX_BITS);
                level = level.wrapping_add(1);
                let child = level_mut(&mut levels, level);
                index = child.index;
                // The result is ignored; the child table is re-read on every entry.
                let _ = read_range(hold, on, &self.read, &self.log, pa, child, 0, ENTRIES)?;
                continue;
            } else {
                // The visitor's result is ignored.
                let _ = visitor(
                    index.wrapping_shl(APV_PAGE_SHIFT),
                    pa,
                    entry.wrapping_shr(31),
                )?;
            }
            index = index.wrapping_add(1);
            if index > level_mut(&mut levels, level).end {
                break;
            }
            // Tested on the old offset.
            if offset.wrapping_add(1) >= ENTRIES {
                loop {
                    if level == 0 {
                        return Ok(());
                    }
                    level_mut(&mut levels, level).index = index;
                    level = level.wrapping_sub(1);
                    let parent = level_mut(&mut levels, level);
                    index = parent.index.wrapping_add(1);
                    base = parent.base;
                    if index.wrapping_sub(base) < ENTRIES {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use parking_lot::Mutex;

    use super::*;
    use crate::raise::Raise;
    use crate::testkit::capture::catch_death;

    const PAGE: u64 = APV_PAGE_SIZE;
    /// The last entry of a table.
    const LAST: u64 = ENTRIES - 1;

    /// Guest memory made of page tables: `(guest page, ENTRIES entries)`; everything else reads
    /// as zero. Records every read as `(address, length)`.
    #[derive(Default)]
    struct Guest {
        tables: Vec<(u32, Vec<u32>)>,
        reads: Mutex<Vec<(u64, usize)>>,
        raise_on: Option<u64>,
    }

    impl Guest {
        fn table(&mut self, page: u32, entries: &[(usize, u32)]) {
            let mut t = vec![0u32; ENTRIES.to_usize()];
            for &(i, e) in entries {
                t[i] = e;
            }
            self.tables.push((page, t));
        }

        fn read_fn(self: &Arc<Self>) -> PageTableRead {
            let g = Arc::clone(self);
            Arc::new(
                move |_: Hold<'_>,
                      on: OnFault,
                      address: u64,
                      dst: &mut [u8]|
                      -> RaiseResult<bool> {
                    g.reads.lock().push((address, dst.len()));
                    if g.raise_on == Some(address) {
                        return Err(crate::testkit::capture::raise(
                            on,
                            "read",
                            "Read past end of page",
                        ));
                    }
                    let page = (address >> APV_PAGE_SHIFT) as u32;
                    let first = ((address & APV_PAGE_MASK) / 4) as usize;
                    if let Some((_, t)) = g.tables.iter().find(|(p, _)| *p == page) {
                        for (k, chunk) in dst.chunks_mut(4).enumerate() {
                            chunk.copy_from_slice(&t[first + k].to_le_bytes());
                        }
                    }
                    Ok(false) // ignored
                },
            )
        }
    }

    /// A walker log that drops its lines.
    fn no_log() -> PageTableLog {
        Arc::new(|_, _| {})
    }

    /// The calls a [`recording_log`] saw: the line and its number.
    type LogCalls = Arc<Mutex<Vec<(WalkLog, u64)>>>;

    /// A walker log that records each call.
    fn recording_log() -> (PageTableLog, LogCalls) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let l = Arc::clone(&calls);
        let log: PageTableLog = Arc::new(move |line: WalkLog, n: u64| {
            l.lock().push((line, n));
        });
        (log, calls)
    }

    fn caught<T>(f: impl FnOnce(Hold<'_>) -> RaiseResult<T>) -> RaiseResult<T> {
        f(Hold::NONE)
    }

    /// `t` is memory's raise `kind` with `reason`, as a handler gets it.
    fn thrown_at(t: &Raise, kind: RaiseKind, reason: &str) {
        assert!(matches!(t.origin(), crate::raise::Origin::Core(_)));
        assert_eq!(t.detail().unwrap_or_default(), reason);
        assert_eq!((t.site().file, t.site().line), (kind.name(), 0));
    }

    fn walk(w: &PageTableWalker, start: u64, length: u64) -> RaiseResult<Vec<(u64, u32, u32)>> {
        let mut visits = Vec::new();
        caught(|hold| {
            w.visit_pages_in_range(
                hold,
                OnFault::Return,
                start,
                length,
                &mut |va, ppn, writable| {
                    visits.push((va, ppn, writable));
                    Ok(false) // the visitor's result is ignored
                },
            )
        })?;
        Ok(visits)
    }

    #[test]
    fn new_holds_the_read_block_until_dropped() {
        let g = Arc::new(Guest::default());
        let w = PageTableWalker::new(7, 2, true, g.read_fn(), no_log());
        assert_eq!(w.root, 7 << APV_PAGE_SHIFT);
        assert_eq!(w.depth, 2);
        assert!(w.allow_holes);
        assert_eq!(
            format!("{w:?}"),
            format!(
                "PageTableWalker {{ root: {}, depth: 2, allow_holes: true, .. }}",
                7 << APV_PAGE_SHIFT
            )
        );
        assert_eq!(Arc::strong_count(&g), 2);
        drop(w); // the read block's release
        assert_eq!(Arc::strong_count(&g), 1);
        // No validation: root 0 and depth 0 are accepted.
        let _ = PageTableWalker::new(0, 0, false, g.read_fn(), no_log());
    }

    mod visit_pages_in_range {
        use super::*;

        #[test]
        fn refusals_in_order() {
            let g = Arc::new(Guest::default());
            // Root 0: the checks come before the root assert.
            let w = PageTableWalker::new(0, 1, false, g.read_fn(), no_log());
            thrown_at(
                &walk(&w, 1, 3).unwrap_err(),
                RaiseKind::WalkMisalignedStart,
                "start 0x1 is not page aligned",
            );
            thrown_at(
                &walk(&w, 0, 3).unwrap_err(),
                RaiseKind::WalkMisalignedLength,
                "length 0x3 is not whole pages",
            );
            thrown_at(
                &walk(&w, 0, 0).unwrap_err(),
                RaiseKind::WalkZeroLength,
                "length 0",
            );
            thrown_at(
                &walk(&w, u64::MAX - APV_PAGE_MASK, 2 * PAGE).unwrap_err(),
                RaiseKind::WalkEndOverflow,
                &format!(
                    "range at {:#x} of {:#x} bytes overflows",
                    u64::MAX - APV_PAGE_MASK,
                    2 * PAGE
                ),
            );
            assert!(g.reads.lock().is_empty());
        }

        #[test]
        fn an_out_of_bounds_root_read_logs_then_throws() {
            // Warns, then raises.
            use crate::testkit::platform::FakePlatform;
            let p = Arc::new(FakePlatform::new());
            let logs = crate::testkit::logs::capture();
            let g = Arc::new(Guest::default());
            let w = PageTableWalker::new(1, 1, false, g.read_fn(), host_error_log(p.clone()));
            thrown_at(
                &walk(&w, ENTRIES * PAGE, PAGE).unwrap_err(),
                RaiseKind::WalkRootUnread,
                "the root table could not be read",
            );
            assert_eq!(
                logs.take(),
                vec![(
                    log::Level::Warn,
                    concat!(env!("CARGO_CRATE_NAME"), "::memory::page_table").to_owned(),
                    format!("page table read from entry {ENTRIES}, past the table")
                )]
            );
            assert!(g.reads.lock().is_empty());
            // Depth 1 over pages [LAST, LAST + 2): the count runs past the table.
            let w = PageTableWalker::new(1, 1, false, g.read_fn(), host_error_log(p.clone()));
            thrown_at(
                &walk(&w, LAST * PAGE, 2 * PAGE).unwrap_err(),
                RaiseKind::WalkRootUnread,
                "the root table could not be read",
            );
            assert_eq!(
                logs.take().into_iter().map(|l| l.2).collect::<Vec<_>>(),
                vec!["page table read of 2 entries, past the table".to_owned()]
            );
        }

        #[test]
        fn a_zero_root_dies_at_the_first_walk() {
            // testing.md 7.4 `smoke_crash`, `tasks_service_crash`.
            let g = Arc::new(Guest::default());
            let w = PageTableWalker::new(0, 1, false, g.read_fn(), no_log());
            let d = catch_death(|| walk(&w, 0, PAGE)).unwrap_err();
            assert_eq!(d, DeathKind::WalkZeroTable.site());
        }

        #[test]
        fn depth_one_reads_only_the_touched_entries() {
            // Level indices are truncated to 32 bits
            let mut g = Guest::default();
            g.table(9, &[(3, 0x8000_0011), (4, 0x12), (5, 0x13)]);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 1, false, g.read_fn(), no_log());
            let v = walk(&w, 3 * PAGE, 3 * PAGE).unwrap();
            assert_eq!(
                v,
                vec![
                    (3 * PAGE, 0x11, 1),
                    (4 * PAGE, 0x12, 0),
                    (5 * PAGE, 0x13, 0)
                ]
            );
            assert_eq!(*g.reads.lock(), vec![((9 << APV_PAGE_SHIFT) + 12, 12)]);
        }

        #[test]
        fn the_root_read_refuses_out_of_bounds_indices() {
            let mut g = Guest::default();
            g.table(9, &[]);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 1, false, g.read_fn(), no_log());
            // start index ENTRIES (logged "Out of bounds start index")
            thrown_at(
                &walk(&w, ENTRIES * PAGE, PAGE).unwrap_err(),
                RaiseKind::WalkRootUnread,
                "the root table could not be read",
            );
            // start + count > ENTRIES (logged "Out of bounds count")
            thrown_at(
                &walk(&w, (ENTRIES - 96) * PAGE, 97 * PAGE).unwrap_err(),
                RaiseKind::WalkRootUnread,
                "the root table could not be read",
            );
            assert!(g.reads.lock().is_empty());
        }

        #[test]
        fn depth_two_descends_climbs_and_rereads() {
            // VA pages LAST - 1 ..= LAST + 2: root entries 0 and 1, each a child table.
            let mut g = Guest::default();
            g.table(9, &[(0, 20), (1, 0x8000_0000 | 21)]);
            let last = LAST.to_usize();
            g.table(20, &[(last - 1, 0x100), (last, 0x8000_0101)]);
            g.table(21, &[(0, 0x200), (1, 0x201)]);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 2, false, g.read_fn(), no_log());
            let v = walk(&w, (LAST - 1) * PAGE, 4 * PAGE).unwrap();
            assert_eq!(
                v,
                vec![
                    ((LAST - 1) * PAGE, 0x100, 0),
                    (LAST * PAGE, 0x101, 1),
                    ((LAST + 1) * PAGE, 0x200, 0),
                    ((LAST + 2) * PAGE, 0x201, 0),
                ]
            );
            assert_eq!(
                *g.reads.lock(),
                vec![
                    (9 << APV_PAGE_SHIFT, 8),
                    (20 << APV_PAGE_SHIFT, PAGE.to_usize()),
                    (21 << APV_PAGE_SHIFT, PAGE.to_usize())
                ]
            );
            // Nothing is cached: a second walk reads every table again.
            walk(&w, (LAST - 1) * PAGE, PAGE).unwrap();
            assert_eq!(g.reads.lock().len(), 5);
        }

        #[test]
        fn depth_three() {
            let mut g = Guest::default();
            g.table(9, &[(1, 30)]);
            g.table(30, &[(2, 40)]);
            g.table(40, &[(3, 0x8000_0555)]);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 3, false, g.read_fn(), no_log());
            let page = (1 << (2 * INDEX_BITS)) | (2 << INDEX_BITS) | 3;
            let v = walk(&w, page * PAGE, PAGE).unwrap();
            assert_eq!(v, vec![(page * PAGE, 0x555, 1)]);
        }

        #[test]
        fn a_hole_throws_after_the_earlier_visits() {
            let mut g = Guest::default();
            g.table(9, &[(0, 0x10), (2, 0x12)]);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 1, false, g.read_fn(), no_log());
            let mut visits = Vec::new();
            let e = caught(|hold| {
                w.visit_pages_in_range(hold, OnFault::Return, 0, 3 * PAGE, &mut |va, ppn, _| {
                    visits.push((va, ppn));
                    Ok(true)
                })
            })
            .unwrap_err();
            thrown_at(&e, RaiseKind::WalkHole, "a hole in the page table");
            assert_eq!(visits, vec![(0, 0x10)]);
        }

        #[test]
        fn allow_holes_skips_a_leaf_hole() {
            let mut g = Guest::default();
            g.table(9, &[(0, 0x10), (2, 0x12)]);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 1, true, g.read_fn(), no_log());
            let v = walk(&w, 0, 3 * PAGE).unwrap();
            assert_eq!(v, vec![(0, 0x10, 0), (2 * PAGE, 0x12, 0)]);
        }

        #[test]
        fn allow_holes_at_an_intermediate_level_dies_on_the_stale_index() {
            // The stale index is outside the levels here; the walk dies.
            let mut g = Guest::default();
            g.table(9, &[(1, 20)]);
            g.table(20, &[(0, 0x10)]);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 2, true, g.read_fn(), no_log());
            let d = catch_death(|| walk(&w, 0, (ENTRIES + 1) * PAGE)).unwrap_err();
            assert_eq!(d, DeathKind::WalkEntryPastTable.site());
        }

        #[test]
        fn depth_zero_dies() {
            let g = Arc::new(Guest::default());
            let w = PageTableWalker::new(9, 0, false, g.read_fn(), no_log());
            let d = catch_death(|| walk(&w, 0, PAGE)).unwrap_err();
            assert_eq!(d, DeathKind::WalkNoLevels.site());
        }

        #[test]
        fn raises_from_the_read_block_and_the_visitor_propagate() {
            let mut g = Guest::default();
            g.table(9, &[(0, 20)]);
            g.table(20, &[(0, 0x10), (1, 0x11)]);
            g.raise_on = Some(20 << APV_PAGE_SHIFT);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 2, false, g.read_fn(), no_log());
            let e = walk(&w, 0, PAGE).unwrap_err();
            assert_eq!(e.detail().unwrap_or_default(), "Read past end of page");

            let mut g2 = Guest::default();
            g2.table(9, &[(0, 0x10), (1, 0x11)]);
            let g2 = Arc::new(g2);
            let w = PageTableWalker::new(9, 1, false, g2.read_fn(), no_log());
            let mut n = 0;
            let e = caught(|hold| {
                w.visit_pages_in_range(hold, OnFault::Return, 0, 2 * PAGE, &mut |_, _, _| {
                    n += 1;
                    Err(crate::testkit::capture::raise(
                        OnFault::Return,
                        "visitor",
                        "visitor",
                    ))
                })
            })
            .unwrap_err();
            assert_eq!(e.detail().unwrap_or_default(), "visitor");
            assert_eq!(n, 1);
        }

        #[test]
        fn level_indices_are_truncated_to_32_bits() {
            // With depth 1, VA page 2^32 + 3 reads root index 3.
            let mut g = Guest::default();
            g.table(9, &[(3, 0x77)]);
            let g = Arc::new(g);
            let w = PageTableWalker::new(9, 1, false, g.read_fn(), no_log());
            let va = ((1u64 << 32) + 3) * PAGE;
            let v = walk(&w, va, PAGE).unwrap();
            assert_eq!(v, vec![(3 * PAGE, 0x77, 0)]);
        }

        #[test]
        fn a_read_that_answers_no_leaves_the_level_as_it_was() {
            // The local task's block reads through a weak device link. Once the device's teardown
            // has begun the link answers nothing, and the read answers false and writes nothing.
            // readRange ignores that, and the walk goes on over the entries the level already held.
            use std::sync::atomic::{AtomicBool, Ordering};
            let mut g = Guest::default();
            g.table(9, &[(0, 20), (1, 21)]);
            let last = LAST.to_usize();
            g.table(
                20,
                &[
                    (0, 0x300),
                    (1, 0x8000_0301),
                    (last - 1, 0x100),
                    (last, 0x101),
                ],
            );
            g.table(21, &[(0, 0x200), (1, 0x201)]);
            let g = Arc::new(g);
            let gone = Arc::new(AtomicBool::new(false));
            let nil_reads = Arc::new(Mutex::new(Vec::new()));
            let read: PageTableRead = {
                let (inner, gone, nil_reads) = (g.read_fn(), gone.clone(), nil_reads.clone());
                Arc::new(
                    move |hold: Hold<'_>,
                          on: OnFault,
                          address: u64,
                          dst: &mut [u8]|
                          -> RaiseResult<bool> {
                        if gone.load(Ordering::SeqCst) {
                            nil_reads.lock().push(address);
                            return Ok(false);
                        }
                        inner(hold, on, address, dst)
                    },
                )
            };
            let w = PageTableWalker::new(9, 2, false, read, no_log());
            // The device goes while the walk is at VA page LAST. The descent into root entry 1
            // re-reads table 21 through nil, so the child level still holds table 20, and its
            // entries 0 and 1 are visited as table 21's.
            let mut visits = Vec::new();
            caught(|hold| {
                w.visit_pages_in_range(
                    hold,
                    OnFault::Return,
                    (LAST - 1) * PAGE,
                    4 * PAGE,
                    &mut |va, ppn, writable| {
                        visits.push((va, ppn, writable));
                        if va == LAST * PAGE {
                            gone.store(true, Ordering::SeqCst);
                        }
                        Ok(true)
                    },
                )
            })
            .unwrap();
            assert_eq!(
                visits,
                vec![
                    ((LAST - 1) * PAGE, 0x100, 0),
                    (LAST * PAGE, 0x101, 0),
                    ((LAST + 1) * PAGE, 0x300, 0),
                    ((LAST + 2) * PAGE, 0x301, 1),
                ]
            );
            assert_eq!(
                *g.reads.lock(),
                vec![
                    (9 << APV_PAGE_SHIFT, 8),
                    (20 << APV_PAGE_SHIFT, PAGE.to_usize())
                ]
            );
            assert_eq!(*nil_reads.lock(), vec![21 << APV_PAGE_SHIFT]);
            // A walk that starts with the device gone: the root read writes nothing, readRange
            // still succeeds (no "the root table could not be read"), and the unwritten root entry is a hole.
            // The levels start zeroed, so the entry the read left unwritten is 0.
            let e = walk(&w, (LAST - 1) * PAGE, PAGE).unwrap_err();
            thrown_at(&e, RaiseKind::WalkHole, "a hole in the page table");
            assert_eq!(
                *nil_reads.lock(),
                vec![21 << APV_PAGE_SHIFT, 9 << APV_PAGE_SHIFT]
            );
            assert_eq!(g.reads.lock().len(), 2);
        }
    }

    #[test]
    fn read_range_refuses_indices_past_the_table() {
        let mut g = Guest::default();
        g.table(5, &[(10, 0xabc)]);
        let g = Arc::new(g);
        let read = g.read_fn();
        let (log, lines) = recording_log();
        let mut level = Level {
            entries: vec![0; PAGE.to_usize()],
            index: 0,
            end: 0,
            base: 0,
        };
        assert!(
            caught(|hold| read_range(hold, OnFault::Return, &read, &log, 5, &mut level, 10, 2))
                .unwrap()
        );
        assert_eq!(level.entry(10), 0xabc);
        assert_eq!(*g.reads.lock(), vec![((5 << APV_PAGE_SHIFT) + 40, 8)]);
        assert!(
            !caught(|hold| read_range(
                hold,
                OnFault::Return,
                &read,
                &log,
                5,
                &mut level,
                ENTRIES,
                1
            ))
            .unwrap()
        );
        assert!(
            !caught(|hold| read_range(hold, OnFault::Return, &read, &log, 5, &mut level, LAST, 2))
                .unwrap()
        );
        assert!(
            caught(|hold| read_range(
                hold,
                OnFault::Return,
                &read,
                &log,
                5,
                &mut level,
                0,
                ENTRIES
            ))
            .unwrap()
        );
        // One host error line per refusal.
        assert_eq!(
            *lines.lock(),
            vec![
                (WalkLog::StartOutOfTable, ENTRIES),
                (WalkLog::CountOutOfTable, 2)
            ]
        );
        let d =
            catch_death(|| read_range(Hold::NONE, OnFault::Die, &read, &log, 0, &mut level, 0, 1))
                .unwrap_err();
        assert_eq!(d, DeathKind::WalkZeroTable.site());
    }
}
