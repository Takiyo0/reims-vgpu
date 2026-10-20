//! The process's backing table on Windows: which section object, from which offset, each byte of
//! the memory the library may make a view of shows, so that a view is made by mapping sections at
//! a fixed address (`MapViewOfFile3` with `MEM_REPLACE_PLACEHOLDER`) and never by remapping
//! addresses, which Windows has no call for either (Mach's `mach_vm_remap`).
//!
//! This mirrors the Linux backing table exactly -- same two kinds of entry (registrations of
//! memory the library does not map itself: the guest's RAM and IOSurface's stand-in memory; and
//! runs the library mapped) and the same resolution order (a byte is found in a run first, in a
//! registration only where no run holds it). The only platform difference is the handle, a
//! [`Section`] in place of an `OwnedFd`, and the protection, a Win32 `PAGE_*` value ([`Prot`]) in
//! place of a POSIX `PROT_*` bitmask.
//!
//! **Mappings.** Unlike Linux's VMAs, each `MapViewOfFile3` view is its own region the kernel
//! does not coalesce with a neighbour, so the table counts one mapping per run with no join
//! accounting. There is no `vm.max_map_count` analogue; a view that cannot be placed fails the
//! call (`crate::os::windows::vm`) rather than being budgeted ahead.
//!
//! The table is a leaf lock: nothing is called under it.

use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(test)]
use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::sync::Mutex;

use super::section::Section;
use super::vm::Prot;
use crate::os::GuestRamFile;

/// What backs a range the table resolves. Two kinds:
/// - [`Backing::Section`]: a section object the library made (a view's run, a twin, an IOSurface
///   stand-in, or a guest-RAM registration on a host that hands sections) and the offset of the
///   range's first byte within it. A view of these pages is made with `MapViewOfFile3`.
/// - [`Backing::HostVa`]: the host virtual address of guest RAM the HOST already allocated (WHPX
///   hands QEMU's own `VirtualAlloc` region, `RvgMemoryRange.host_address`); there is no section
///   to map. The runtime imports this pointer directly through `VK_EXT_external_memory_host`, so
///   it is only ever read-write guest RAM, which takes the plain import path and is never mapped
///   or twinned by the library (`import_view`, `vm::map_view`). `base` is the host VA of the
///   range's first byte.
#[derive(Clone, Debug)]
pub(crate) enum Backing {
    /// A library-made section and the offset of the range's first byte within it.
    Section {
        /// The section object, held while any entry or view names it.
        fd: Arc<Section>,
        /// The offset of the range's first byte within the section.
        offset: u64,
    },
    /// Guest RAM the host already mapped: the host VA of the range's first byte. No section.
    HostVa {
        /// The host virtual address of the range's first byte.
        base: u64,
    },
}

impl Backing {
    /// The same backing advanced by `delta` bytes into the range (a sub-piece's backing).
    fn advanced(&self, delta: u64) -> Backing {
        match *self {
            Backing::Section { ref fd, offset } => Backing::Section {
                fd: Arc::clone(fd),
                offset: offset + delta,
            },
            Backing::HostVa { base } => Backing::HostVa { base: base + delta },
        }
    }

    /// The host VA of this backing's first byte, where it is borrowed guest RAM; `None` for a
    /// section (the library maps those to get an address).
    pub(crate) fn host_va(&self) -> Option<u64> {
        match *self {
            Backing::HostVa { base } => Some(base),
            Backing::Section { .. } => None,
        }
    }

    /// The section and offset, where this is a section backing; `None` for borrowed guest RAM
    /// (which `MapViewOfFile3` cannot map — the library never maps or twins it).
    pub(crate) fn section(&self) -> Option<(&Arc<Section>, u64)> {
        match self {
            Backing::Section { fd, offset } => Some((fd, *offset)),
            Backing::HostVa { .. } => None,
        }
    }

    /// Whether two backings name the same object and offset (registration counting, run joining).
    fn same(&self, other: &Backing) -> bool {
        match (self, other) {
            (Backing::Section { fd: a, offset: ao }, Backing::Section { fd: b, offset: bo }) => {
                Arc::ptr_eq(a, b) && ao == bo
            }
            (Backing::HostVa { base: a }, Backing::HostVa { base: b }) => a == b,
            _ => false,
        }
    }
}

/// A run of the library's own mapping: its end, its section and the protection set on it
/// (`PAGE_*`).
#[derive(Clone, Debug)]
struct Run {
    end: u64,
    backing: Backing,
    prot: Prot,
}

/// A registration: `first..end` shows `backing`, read and write, `count` times registered.
#[derive(Debug)]
struct Registered {
    first: u64,
    end: u64,
    backing: Backing,
    count: usize,
}

/// One piece of a range as the table resolves it: `length` bytes, `at` bytes into the range,
/// showing `backing`, with protection `prot`.
#[derive(Clone, Debug)]
pub(crate) struct Piece {
    pub(crate) at: u64,
    pub(crate) length: u64,
    pub(crate) backing: Backing,
    pub(crate) prot: Prot,
}

/// The table (module docs).
#[derive(Debug, Default)]
struct Table {
    registered: Vec<Registered>,
    runs: BTreeMap<u64, Run>,
}

/// The process's table.
static TABLE: Mutex<Table> = Mutex::new(Table {
    registered: Vec::new(),
    runs: BTreeMap::new(),
});

impl Table {
    /// The run holding byte `at`: its first byte and itself.
    fn run_at(&self, at: u64) -> Option<(u64, &Run)> {
        (self.runs.range(..=at).next_back())
            .filter(|(_, run)| at < run.end)
            .map(|(&first, run)| (first, run))
    }

    /// The first run starting after `at`, before `end`: its first byte.
    fn next_run(&self, at: u64, end: u64) -> Option<u64> {
        self.runs.range(at..end).next().map(|(&first, _)| first)
    }

    /// The registration holding byte `at`.
    fn registered_at(&self, at: u64) -> Option<&Registered> {
        self.registered.iter().find(|r| r.first <= at && at < r.end)
    }

    /// Drops what of the runs lies in `first..end`, keeping the rest of each.
    fn forget(&mut self, first: u64, end: u64) {
        let meeting: Vec<(u64, Run)> = (self.runs.range(..end).rev())
            .take_while(|(_, run)| run.end > first)
            .map(|(&at, run)| (at, run.clone()))
            .collect();
        for (at, run) in meeting {
            self.runs.remove(&at);
            if at < first {
                let mut head = run.clone();
                head.end = first;
                self.runs.insert(at, head);
            }
            if end < run.end {
                let mut tail = run.clone();
                tail.backing = run.backing.advanced(end - at);
                self.runs.insert(end, tail);
            }
        }
    }

    /// `first..end` shows `backing` with protection `prot` from now on (`record`).
    fn record(&mut self, first: u64, end: u64, backing: Backing, prot: Prot) {
        self.forget(first, end);
        if first < end {
            self.runs.insert(first, Run { end, backing, prot });
        }
    }

    /// Sets the protection of what of the runs lies in `first..end` (`protect`).
    fn protect(&mut self, first: u64, end: u64, prot: Prot) {
        let meeting: Vec<(u64, Run)> = (self.runs.range(..end).rev())
            .take_while(|(_, run)| run.end > first)
            .map(|(&at, run)| (at, run.clone()))
            .collect();
        for (at, run) in meeting {
            let (from, to) = (at.max(first), run.end.min(end));
            self.forget(from, to);
            let mut changed = run.clone();
            changed.end = to;
            changed.backing = run.backing.advanced(from - at);
            changed.prot = prot;
            self.runs.insert(from, changed);
        }
    }
}

/// The views the table's runs hold: one mapping each, as Windows places each `MapViewOfFile3`
/// view in its own region (module docs).
#[cfg(test)]
pub(crate) fn mappings() -> u64 {
    TABLE.lock().runs.len().to_u64()
}

/// The views the table's runs in `first..end` hold (tests: a range's count to set beside the
/// kernel's).
#[cfg(test)]
pub(crate) fn mappings_between(first: u64, end: u64) -> u64 {
    TABLE.lock().runs.range(first..end).count().to_u64()
}

/// `length` bytes from `first` show `backing` from now on, with protection `prot`: what of
/// another run they cover goes.
pub(crate) fn record(first: u64, length: u64, backing: Backing, prot: Prot) {
    TABLE
        .lock()
        .record(first, first.saturating_add(length), backing, prot);
}

/// `length` bytes from `first` show no section from now on.
pub(crate) fn forget(first: u64, length: u64) {
    TABLE.lock().forget(first, first.saturating_add(length));
}

/// The pieces of `length` bytes from `first`, each the bytes of one run or registration in
/// order, those of a run cut where it ends; `None` where a byte has no section behind it.
pub(crate) fn pieces(first: u64, length: u64) -> Option<Vec<Piece>> {
    let end = first.checked_add(length)?;
    let table = TABLE.lock();
    let mut pieces = Vec::new();
    let mut at = first;
    while at < end {
        if let Some((run_first, run)) = table.run_at(at) {
            let to = run.end.min(end);
            let backing = run.backing.advanced(at - run_first);
            pieces.push(Piece {
                at: at - first,
                length: to - at,
                backing,
                prot: run.prot,
            });
            at = to;
            continue;
        }
        let registered = table.registered_at(at)?;
        let to = (registered.end.min(end)).min(table.next_run(at, end).unwrap_or(end));
        let backing = registered.backing.advanced(at - registered.first);
        pieces.push(Piece {
            at: at - first,
            length: to - at,
            backing,
            prot: Prot::READ_WRITE,
        });
        at = to;
    }
    Some(pieces)
}

/// Sets the protection of what of the runs lies in `length` bytes from `first` to `prot`,
/// cutting the runs it meets at its ends.
pub(crate) fn protect(first: u64, length: u64, prot: Prot) {
    TABLE
        .lock()
        .protect(first, first.saturating_add(length), prot);
}

/// Whether the table knows every byte of `length` bytes from `first`: `Some(readable)` where
/// each lies in a run or a registration, readable where each has read access; `None` where one
/// lies in neither.
pub(crate) fn readable(first: u64, length: u64) -> Option<bool> {
    let pieces = pieces(first, length)?;
    Some(pieces.iter().all(|p| p.prot.is_readable()))
}

/// Whether a byte of `length` bytes from `first` lies in a registration.
pub(crate) fn meets_registration(first: u64, length: u64) -> bool {
    let end = first.saturating_add(length);
    let table = TABLE.lock();
    table
        .registered
        .iter()
        .any(|r| r.first < end && first < r.end)
}

/// A registration of `length` bytes from `first` as showing `backing` (module docs), until it
/// drops.
#[derive(Debug)]
pub(crate) struct Registration {
    first: u64,
    end: u64,
    backing: Backing,
}

impl Registration {
    /// Registers `length` bytes from `first` as showing `backing`: counted again where the same
    /// bytes are registered over the same section and offset.
    pub(crate) fn new(first: u64, length: u64, backing: Backing) -> Registration {
        let end = first.saturating_add(length);
        let mut table = TABLE.lock();
        match table
            .registered
            .iter_mut()
            .find(|r| r.is(first, end, &backing))
        {
            Some(r) => r.count += 1,
            None => table.registered.push(Registered {
                first,
                end,
                backing: backing.clone(),
                count: 1,
            }),
        }
        Registration {
            first,
            end,
            backing,
        }
    }
}

/// The guest's RAM registered while a device made over it lives: the `guest_memory` lease its
/// device holds, made of the host's memory map, and dropped after core's teardown. Views of the
/// guest's pages are made of these sections.
#[derive(Debug)]
pub struct GuestMemoryLease {
    _registrations: Vec<Registration>,
}

/// Registers each range of the guest's RAM the host names a section behind: counted where the
/// same range was registered already over the same section (module docs).
pub fn lease_guest_memory(ranges: impl IntoIterator<Item = GuestRamFile>) -> GuestMemoryLease {
    let registrations: Vec<Registration> = ranges
        .into_iter()
        .map(|r| {
            let backing = match r.fd {
                // The host already mapped this guest RAM (WHPX): back it by its host VA, which the
                // runtime imports directly through VK_EXT_external_memory_host. No section. It
                // stays mapped while the host runs the guest, so the runtime may keep imports of
                // it and bind buffers over its pages into them (`register_persistent_host_range`).
                None => {
                    if let Some(start) = std::ptr::NonNull::new(r.host as *mut u8) {
                        metal2vulkan_runtime::register_persistent_host_range(start, r.length);
                    }
                    Backing::HostVa { base: r.host }
                }
                // A section-backed range (a host that hands sections): a view is mapped from it.
                Some(fd) => Backing::Section {
                    fd,
                    offset: r.offset,
                },
            };
            Registration::new(r.host, r.length, backing)
        })
        .collect();
    GuestMemoryLease {
        _registrations: registrations,
    }
}

impl Registered {
    /// Whether this registers `first..end` over `backing`.
    fn is(&self, first: u64, end: u64, backing: &Backing) -> bool {
        self.first == first && self.end == end && self.backing.same(backing)
    }
}

impl Drop for Registration {
    /// Counts the registration down, and drops it at the last.
    fn drop(&mut self) {
        let mut table = TABLE.lock();
        if let Some(at) =
            (table.registered.iter()).position(|r| r.is(self.first, self.end, &self.backing))
        {
            let r = &mut table.registered[at];
            r.count -= 1;
            if r.count == 0 {
                table.registered.remove(at);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A section for the tests' backings: one page of pagefile-backed memory.
    fn file() -> Arc<Section> {
        Arc::new(Section::new(0x1000).unwrap())
    }

    /// The pieces' (offset in the range, length, section offset or host VA).
    fn shape(pieces: &[Piece]) -> Vec<(u64, u64, u64)> {
        pieces
            .iter()
            .map(|p| {
                let backing_at = p
                    .backing
                    .section()
                    .map(|(_, offset)| offset)
                    .or_else(|| p.backing.host_va())
                    .unwrap_or(0);
                (p.at, p.length, backing_at)
            })
            .collect()
    }

    // The table is the process's: each test uses address ranges of its own, far apart.

    /// A range resolves through the runs over it, cut where they end, then the registrations;
    /// a byte of neither leaves it unresolved
    #[test]
    fn a_range_resolves_through_runs_then_registrations() {
        let f = file();
        let base = 0x7a00_0000_0000;
        let backing = |offset| Backing::Section {
            fd: Arc::clone(&f),
            offset,
        };
        let held = Registration::new(base, 0x10_000, backing(0x100_000));
        record(base + 0x20_000, 0x2000, backing(0x5000), Prot::READ);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x2000).unwrap()),
            [(0, 0x2000, 0x101_000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x20_000, 0x2000).unwrap()),
            [(0, 0x2000, 0x5000)]
        );
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
        assert!(pieces(base + 0x1f_000, 0x2000).is_none());
        assert_eq!(readable(base + 0x20_000, 0x2000), Some(true));
        assert_eq!(readable(base, 0x10_000), Some(true));
        protect(base + 0x21_000, 0x1000, Prot::NONE);
        assert_eq!(readable(base + 0x20_000, 0x1000), Some(true));
        assert_eq!(readable(base + 0x20_000, 0x2000), Some(false));
        assert_eq!(
            shape(&pieces(base + 0x20_000, 0x2000).unwrap()),
            [(0, 0x1000, 0x5000), (0x1000, 0x1000, 0x6000)]
        );
        assert_eq!(readable(base + 0x30_000, 1), None);
        drop(held);
        assert!(pieces(base, 0x1000).is_none());
        forget(base + 0x20_000, 0x2000);
    }

    /// A run replaces what of another it covers, and an unmap drops what it covers, keeping
    /// each run's other bytes at their section offsets
    #[test]
    fn runs_never_overlap() {
        let f = file();
        let base = 0x7b00_0000_0000;
        let backing = |offset| Backing::Section {
            fd: Arc::clone(&f),
            offset,
        };
        record(base, 0x4000, backing(0x10_000), Prot::READ_WRITE);
        record(base + 0x1000, 0x1000, backing(0x90_000), Prot::READ);
        assert_eq!(
            shape(&pieces(base, 0x4000).unwrap()),
            [
                (0, 0x1000, 0x10_000),
                (0x1000, 0x1000, 0x90_000),
                (0x2000, 0x2000, 0x12_000)
            ]
        );
        forget(base + 0x2000, 0x1000);
        assert!(pieces(base, 0x4000).is_none());
        assert_eq!(
            shape(&pieces(base + 0x3000, 0x1000).unwrap()),
            [(0, 0x1000, 0x13_000)]
        );
        forget(base, 0x4000);
        assert!(pieces(base, 1).is_none());
    }

    /// A lease registers each range of the guest's RAM at its section offset, until it drops
    #[test]
    fn a_lease_registers_the_guests_ram_while_held() {
        let f = file();
        let base = 0x7d00_0000_0000;
        let ram = |host, length, offset| GuestRamFile {
            host,
            length,
            fd: Some(Arc::clone(&f)),
            offset,
        };
        let lease =
            lease_guest_memory([ram(base, 0x4000, 0), ram(base + 0x10_000, 0x2000, 0x4000)]);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x1000).unwrap()),
            [(0, 0x1000, 0x1000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x11_000, 0x1000).unwrap()),
            [(0, 0x1000, 0x5000)]
        );
        assert!(pieces(base + 0x4000, 0x1000).is_none());
        drop(lease);
        assert!(pieces(base, 0x1000).is_none());
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
    }

    /// The WHPX bridge: a lease of guest RAM with no backing object (`fd: None`) registers it by
    /// its host VA. The table then resolves each guest address to that host VA, read and write, so
    /// `import_view`'s plain path imports the pointer directly (no section, no twin). The piece's
    /// backing is a `HostVa` carrying the resolved host address.
    #[test]
    fn a_host_va_lease_registers_borrowed_guest_ram() {
        let base = 0x7e00_0000_0000;
        let host_va = |host, length| GuestRamFile {
            host,
            length,
            fd: None,
            offset: 0,
        };
        let lease = lease_guest_memory([host_va(base, 0x4000), host_va(base + 0x10_000, 0x2000)]);
        // Every byte resolves to read-write memory at its own host VA (base + offset-in-range).
        let p = pieces(base + 0x1000, 0x1000).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].prot, Prot::READ_WRITE);
        assert_eq!(p[0].backing.host_va(), Some(base + 0x1000));
        assert!(
            p[0].backing.section().is_none(),
            "no section backs borrowed RAM"
        );
        // readable() answers true, so a gather/import of these pages is not refused.
        assert_eq!(readable(base, 0x4000), Some(true));
        assert_eq!(readable(base + 0x10_000, 0x2000), Some(true));
        // A byte outside the leased ranges is unknown.
        assert!(pieces(base + 0x4000, 0x1000).is_none());
        drop(lease);
        assert!(pieces(base, 0x1000).is_none());
    }

    /// The table counts one mapping per run (Windows does not coalesce views); a protect or an
    /// unmap inside a run cuts it into more runs, each its own mapping
    #[test]
    fn the_table_counts_one_mapping_per_run() {
        let f = file();
        let mut t = Table::default();
        let base = 0x1000_0000;
        let page = 0x1000;
        let at = |i: u64| base + i * page;
        let b = |fd: &Arc<Section>, offset| Backing::Section {
            fd: Arc::clone(fd),
            offset,
        };
        for i in 0..4 {
            t.record(at(i), at(i + 1), b(&f, i * page), Prot::READ_WRITE);
        }
        assert_eq!(t.runs.len(), 4);
        // The second page read-only cuts nothing new in the count sense; still four runs.
        t.protect(at(1), at(2), Prot::READ);
        assert_eq!(t.runs.len(), 4);
        // An unmap of the second page drops one run.
        t.forget(at(1), at(2));
        assert_eq!(t.runs.len(), 3);
        t.forget(base, at(4));
        assert_eq!(t.runs.len(), 0);
    }

    /// The same bytes registered again over the same section and offset are counted: they stay
    /// until the last registration drops
    #[test]
    fn registrations_are_counted() {
        let f = file();
        let base = 0x7c00_0000_0000;
        let backing = || Backing::Section {
            fd: Arc::clone(&f),
            offset: 0,
        };
        let first = Registration::new(base, 0x1000, backing());
        let second = Registration::new(base, 0x1000, backing());
        drop(first);
        assert!(pieces(base, 0x1000).is_some());
        drop(second);
        assert!(pieces(base, 0x1000).is_none());
    }
}
