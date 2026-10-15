//! The object lists: where a task's object list and placement list are, in the task and in the
//! host, shared by the task and its resource manager.
//!
//! Two kinds: the legacy lists (the blocks setup) reach the object list through the task's
//! lazily mapped address and have no placement list; the memory-map lists (the map setup and the
//! isolated tasks' service) map both lists through the task's own mappings.
//!
//! Callers hold the public `mutex` around mapping the lists ([`ObjectLists::ensure_mapped`]).
//! The lists' state sits under a leaf lock that is never held across a call into the task or
//! the drop of a mapping (a mapping's teardown can reach the device), so each such call happens
//! at its point in the operation, with the leaf lock released around it.
//!
//! The task is held unretained. A task that is gone answers as nil would: a NULL address, no
//! mapping. The task owns its lists, so that never happens while they are in use.

use std::mem::offset_of;
use std::num::NonZeroU32;
use std::sync::{Arc, Weak};

use parking_lot::{Mutex, MutexGuard};

use crate::guest::GuestPtr;
use crate::guest::records::{APV_PAGE_MASK, APV_PAGE_SIZE, ObjectEntry, TaskRoot2};
use crate::host::HostPtr;
use crate::memory::mapping::Mapping;
use crate::num::{self, ToUsize, Truncate};
use crate::platform::Platform;
use crate::resources::host::{MappedAddress, ResourceHost};
use crate::rt::Hold;
use crate::tasks::DeathKind;

/// The two kinds of object lists.
pub enum ObjectListKind {
    /// The legacy lists: the object list through the task's mapped address, no placement list.
    Legacy {
        /// Whether the object list is mapped.
        object_list_mapped: bool,
    },
    /// The memory-map lists: both lists through the task's own mappings.
    MemoryMap,
}

/// Where a list is in the task: its offset and its length in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ListRange {
    /// The list's offset in the task.
    pub offset: u64,
    /// Its length in bytes.
    pub length: u32,
}

/// One list: where it is in the task, and where it is mapped in the host.
struct List<P: Platform> {
    /// The host mapping (the memory-map kind's; the legacy object list's when the task's address
    /// has one behind it).
    mapping: Option<Arc<Mapping<P>>>,
    /// Where the list is in the host; NULL while unmapped.
    address: GuestPtr,
    /// Resolved physical pages when contiguous host mapping is unavailable (WHPX discontiguous).
    pages: Vec<GuestPtr>,
    /// Its offset in the task.
    offset: u64,
    /// Its length in bytes.
    length: u32,
    /// Its length rounded up to whole pages from its offset: what is mapped.
    mapped_length: u64,
}

impl<P: Platform> List<P> {
    fn new() -> Self {
        List {
            mapping: None,
            address: GuestPtr::default(),
            pages: Vec::new(),
            offset: 0,
            length: 0,
            mapped_length: 0,
        }
    }

    /// Forgets the host mapping, and answers it for the caller to drop outside the leaf lock.
    fn unmap(&mut self) -> Option<Arc<Mapping<P>>> {
        let released = self.mapping.take();
        self.address = GuestPtr::default();
        self.pages.clear();
        released
    }

    /// Takes `mapping` as the list's host mapping: its address, if there is one (a nil mapping
    /// leaves the address as it was). Answers the previous mapping, for the caller to drop
    /// outside the leaf lock, and whether there was a mapping.
    fn adopt(&mut self, mapping: Option<Arc<Mapping<P>>>, pages: Vec<GuestPtr>) -> (Option<Arc<Mapping<P>>>, bool) {
        let address = mapping.as_ref().map(|m| GuestPtr::new(m.virtual_address()));
        let previous = std::mem::replace(&mut self.mapping, mapping);
        let mapped = if !pages.is_empty() {
            self.address = GuestPtr::default();
            self.pages = pages;
            true
        } else if let Some(a) = address {
            self.address = a;
            self.pages.clear();
            true
        } else {
            false
        };
        (previous, mapped)
    }
}

/// The lists' state, under the leaf lock.
struct Lists<P: Platform> {
    objects: List<P>,
    /// The object list's entry count.
    object_count: u32,
    placements: List<P>,
    kind: ObjectListKind,
}

impl<P: Platform> Lists<P> {
    fn is_legacy(&self) -> bool {
        matches!(self.kind, ObjectListKind::Legacy { .. })
    }

    fn object_list_mapped(&self) -> bool {
        match self.kind {
            ObjectListKind::Legacy { object_list_mapped } => object_list_mapped,
            ObjectListKind::MemoryMap => !self.objects.address.is_null() || !self.objects.pages.is_empty(),
        }
    }

    fn placement_list_mapped(&self) -> bool {
        match self.kind {
            ObjectListKind::Legacy { .. } => false,
            ObjectListKind::MemoryMap => !self.placements.address.is_null() || !self.placements.pages.is_empty(),
        }
    }

    /// Unmaps the object list; answers the mapping it released, for the caller to drop once
    /// the leaf lock is gone.
    fn unmap_object_list(&mut self) -> Option<Arc<Mapping<P>>> {
        match &mut self.kind {
            ObjectListKind::Legacy { object_list_mapped } => {
                *object_list_mapped = false;
                None
            }
            ObjectListKind::MemoryMap => self.objects.unmap(),
        }
    }
}

/// An object-list entry's size: the object list's stride.
const OBJECT_ENTRY_SIZE: u32 = 12;
/// A placement entry's size: the placement list's stride.
const OBJECT_PLACEMENT_SIZE: u64 = 16;
/// The guest page mask both lists' offsets are tested against and their lengths rounded to.
const LIST_PAGE_MASK: u64 = APV_PAGE_MASK;

/// The task root's list fields, as byte offsets into the task root (packed).
const ROOT_OBJECT_LIST_OFFSET: u64 = num::const_u64(offset_of!(TaskRoot2, object_list_offset));
const ROOT_OBJECT_LIST_LENGTH: u64 = num::const_u64(offset_of!(TaskRoot2, object_list_length));
const ROOT_PLACEMENT_LIST_OFFSET: u64 =
    num::const_u64(offset_of!(TaskRoot2, placement_list_offset));
const ROOT_PLACEMENT_LIST_LENGTH: u64 =
    num::const_u64(offset_of!(TaskRoot2, placement_list_length));

/// The object-list entry at `entry`, read with two loads: the offset's high word, then the
/// type/length word with the offset's low word (one 8-byte load). Each is one access, so a
/// guest's 32-bit store to the type/length word lands wholly before or wholly after the second
/// load; that load is single-copy atomic wherever it stays inside a 16-byte granule (index 0, 2
/// and 3 modulo 4 of a 16-aligned list). A guest store to the offset between the two loads
/// leaves the old high word with the new low one.
fn load_object_entry(entry: GuestPtr) -> ObjectEntry {
    let high = entry.read_u32(8, DeathKind::UnmappedEntry);
    #[cfg(test)]
    tests::after_entry_load(entry);
    let low = entry.read_u64(0, DeathKind::UnmappedEntry);
    #[cfg(test)]
    tests::after_entry_load(entry);
    ObjectEntry {
        type_and_length: low.low_u32(),
        offset: u64::from(high).wrapping_shl(32) | low.wrapping_shr(32),
    }
}

fn load_object_entry_from_pages(pages: &[GuestPtr], at: u64) -> ObjectEntry {
    let page_idx = (at / APV_PAGE_SIZE) as usize;
    let page_off = at % APV_PAGE_SIZE;
    if page_off.wrapping_add(u64::from(OBJECT_ENTRY_SIZE)) <= APV_PAGE_SIZE {
        if let Some(&page_ptr) = pages.get(page_idx) {
            return load_object_entry(page_ptr.offset(page_off));
        }
    } else if let (Some(&page0), Some(&page1)) = (pages.get(page_idx), pages.get(page_idx + 1)) {
        let high = if page_off + 8 < APV_PAGE_SIZE {
            page0.read_u32(page_off + 8, DeathKind::UnmappedEntry)
        } else {
            page1.read_u32(page_off + 8 - APV_PAGE_SIZE, DeathKind::UnmappedEntry)
        };
        let low = if page_off + 8 <= APV_PAGE_SIZE {
            page0.read_u64(page_off, DeathKind::UnmappedEntry)
        } else {
            let lo = page0.read_u32(page_off, DeathKind::UnmappedEntry);
            let hi = page1.read_u32(0, DeathKind::UnmappedEntry);
            u64::from(hi).wrapping_shl(32) | u64::from(lo)
        };
        return ObjectEntry {
            type_and_length: low.low_u32(),
            offset: u64::from(high).wrapping_shl(32) | low.wrapping_shr(32),
        };
    }
    crate::fatal!(DeathKind::UnmappedEntry)
}

/// A list's mapped length: its end rounded up to a page, less its offset, in wrapping `u64`.
fn mapped_length(end: u64, offset: u64) -> u64 {
    (end.wrapping_add(LIST_PAGE_MASK) & !LIST_PAGE_MASK).wrapping_sub(offset)
}

/// The placement-list copy's chunk: one guest page (see `copy_placement_list`).
const COPY_CHUNK: u64 = APV_PAGE_SIZE;

/// Both getters' range error.
fn out_of_range(index: u32, count: u32) -> String {
    format!("reference {index} is past the object list's {count} entries")
}

/// A task's object lists, shared by the task and its resource manager, holding the task
/// unretained. Dropping them releases both mappings, the object list's first.
pub struct ObjectLists<P: Platform> {
    /// The lock callers hold around mapping the lists. Public for the GPU task, which maps
    /// them itself.
    pub mutex: Mutex<()>,
    state: Mutex<Lists<P>>,
    task: Weak<dyn ResourceHost<P>>,
    task_length: u64,
    supports_argument_buffers: bool,
}

impl<P: Platform> ObjectLists<P> {
    /// Lists with nothing set: both unmapped, every offset, length and count 0 (the legacy
    /// kind's caller passes it unmapped).
    pub fn new(
        kind: ObjectListKind,
        task_length: u64,
        supports_argument_buffers: bool,
        task: Weak<dyn ResourceHost<P>>,
    ) -> Arc<Self> {
        Arc::new(ObjectLists {
            mutex: Mutex::new(()),
            state: Mutex::new(Lists {
                objects: List::new(),
                object_count: 0,
                placements: List::new(),
                kind,
            }),
            task,
            task_length,
            supports_argument_buffers,
        })
    }

    /// The leaf lock over the lists' state.
    fn st(&self) -> MutexGuard<'_, Lists<P>> {
        self.state.lock()
    }

    /// A mapping of `length` bytes of the task at `offset` for a list: the memory-map kind's
    /// through the task, a raise swallowed as nil (no log); the legacy kind never has one.
    /// Called without the leaf lock.
    fn resolve_pages(&self, hold: Hold<'_>, offset: u64, length: u64) -> Vec<GuestPtr> {
        match self.task.upgrade() {
            Some(task) => match task.resolve_virtual_pages(hold, offset, length) {
                Ok(Some(pages)) => pages,
                Ok(None) => Vec::new(),
                Err(raised) => {
                    hold.land(raised);
                    Vec::new()
                }
            },
            None => Vec::new(),
        }
    }

    fn new_mapping(
        &self,
        hold: Hold<'_>,
        legacy: bool,
        offset: u64,
        length: u64,
        writable: bool,
    ) -> Option<Arc<Mapping<P>>> {
        if legacy {
            return None;
        }
        let mapped = match self.task.upgrade() {
            Some(task) => task.new_virtual_mapping(hold, offset, length, writable),
            None => Ok(None),
        };
        match mapped {
            Ok(mapping) => mapping,
            Err(raised) => {
                hold.land(raised);
                None
            }
        }
    }

    /// Copies `n` bytes of the old placement list into the new one.
    ///
    /// The two lists are different host mappings, but both are shared remaps of guest pages;
    /// when they overlap in guest memory the same page shows through both, and the copy is
    /// libc's forward copy through the aliases ([`num::forward_copy`]): a byte already written
    /// through the new list is read back through the old one, so the data repeats with the lists'
    /// distance as its period. It is not a snapshot copy. Both mappings start on a guest page,
    /// the alias granule, so the copy goes a page at a time (`alias_tests` compares it with
    /// `memcpy` over real aliases).
    fn copy_placement_list(dst: GuestPtr, src: GuestPtr, n: u64) {
        let mut chunk = vec![0u8; n.min(COPY_CHUNK).to_usize()];
        num::forward_copy(
            n,
            &mut chunk,
            |at, bytes| src.copy_to(at, bytes, DeathKind::PlacementCopy),
            |at, bytes| dst.copy_from(at, bytes, DeathKind::PlacementCopy),
        );
    }

    /// The object-list entry at `index`, read from guest memory now ([`load_object_entry`]).
    pub fn get_object_entry(&self, index: u32) -> Result<ObjectEntry, String> {
        let st = self.st();
        if st.object_list_mapped() {
            if index >= st.object_count {
                return Err(out_of_range(index, st.object_count));
            }
            let at = u64::from(index).wrapping_mul(u64::from(OBJECT_ENTRY_SIZE));
            if !st.objects.pages.is_empty() {
                return Ok(load_object_entry_from_pages(&st.objects.pages, at));
            }
            return Ok(load_object_entry(st.objects.address.offset(at)));
        }
        Err("the object list is not mapped".to_owned())
    }

    /// Where the placement entry at `index` is; `Ok(None)` when there is no placement list to
    /// map.
    pub fn get_placement_list_entry(&self, index: u32) -> Result<Option<GuestPtr>, String> {
        let st = self.st();
        if st.placement_list_mapped() {
            // Bounded by the object list's count, not the placement list's length.
            if index >= st.object_count {
                return Err(out_of_range(index, st.object_count));
            }
            let at = u64::from(index).wrapping_mul(OBJECT_PLACEMENT_SIZE);
            if !st.placements.pages.is_empty() {
                let page_idx = (at / APV_PAGE_SIZE) as usize;
                let page_off = at % APV_PAGE_SIZE;
                if let Some(&page_ptr) = st.placements.pages.get(page_idx) {
                    return Ok(Some(page_ptr.offset(page_off)));
                }
            }
            return Ok(Some(st.placements.address.offset(at)));
        }
        if self.supports_argument_buffers && st.placements.length != 0 {
            return Err("the placement list is not mapped".to_owned());
        }
        Ok(None)
    }

    /// Sets the lists, writing the task root's list fields through `task_root` (guest memory)
    /// as it goes; answers whether the lists were taken.
    ///
    /// The object list is unmapped first. The root takes the new fields, then one `||` refuses
    /// the lists (zeroing the members and the root). Without argument buffers the object list
    /// is taken and nothing more happens. Otherwise the placement list moves: the old one is
    /// mapped if it was not (read-only), the new one is mapped (writable), the old entries are
    /// copied into the new list, and a list that could not be mapped zeroes the root's four
    /// fields while the members keep the new lists. A mapping is always released outside the
    /// leaf lock; the old placement list's is held to the end.
    pub fn set_object_list(
        &self,
        hold: Hold<'_>,
        object_list_offset: u64,
        object_list_length: u32,
        placement_list_offset: u64,
        placement_list_length: u32,
        task_root: GuestPtr,
    ) -> bool {
        let (new_ol_off, new_ol_len) = (object_list_offset, object_list_length);
        let (new_pl_off, new_pl_len) = (placement_list_offset, placement_list_length);
        let root = DeathKind::NullTaskRoot;
        let mut st = self.st();
        // Another reference to the old placement mapping, released at the end
        let mut old_pl_mapping = st.placements.mapping.clone();
        let released = st.unmap_object_list();
        drop(st);
        drop(released);
        let mut st = self.st();
        let old_pl_off = st.placements.offset;
        let old_pl_len = st.placements.length;
        let old_pl_mapped_len = st.placements.mapped_length;
        // The entry count is `length / 12`, stored as u32.
        let count = new_ol_len / const { NonZeroU32::new(OBJECT_ENTRY_SIZE).unwrap() };
        let required_pl_len = u64::from(count).wrapping_mul(OBJECT_PLACEMENT_SIZE);
        st.object_count = count;
        // The root takes the new fields before any validation. A NULL task root (task root
        // page 0 in-process) faults here.
        task_root.write_u64(ROOT_OBJECT_LIST_OFFSET, new_ol_off, root);
        task_root.write_u32(ROOT_OBJECT_LIST_LENGTH, new_ol_len, root);
        if self.supports_argument_buffers {
            task_root.write_u64(ROOT_PLACEMENT_LIST_OFFSET, new_pl_off, root);
            task_root.write_u32(ROOT_PLACEMENT_LIST_LENGTH, new_pl_len, root);
        }
        let (ol_end, ol_overflow) = new_ol_off.overflowing_add(u64::from(new_ol_len));
        let (pl_end, pl_overflow) = new_pl_off.overflowing_add(u64::from(new_pl_len));
        // One `||`, in this order
        let refused = ol_overflow
            || pl_overflow
            // A list ending exactly at the task's end is refused (`>=`).
            || ol_end >= self.task_length
            || pl_end >= self.task_length
            // The alignment test is on the members, the previous offsets (testing.md 7.1).
            || st.objects.offset & LIST_PAGE_MASK != 0
            || st.placements.offset & LIST_PAGE_MASK != 0
            || (new_pl_off != 0 && required_pl_len > u64::from(new_pl_len));
        if refused {
            log::warn!(
                "object lists refused: {new_ol_len:#x} bytes of objects at {new_ol_off:#x} and \
                 {new_pl_len:#x} bytes of placements at {new_pl_off:#x}, in a task of {:#x} bytes",
                self.task_length
            );
            st.placements.offset = 0;
            st.placements.length = 0;
            st.objects.offset = 0;
            st.objects.length = 0;
            st.object_count = 0;
            // The other reference keeps the old placement mapping alive to the end
            let released = st.placements.unmap();
            drop(st);
            drop(released);
            task_root.write_u64(ROOT_OBJECT_LIST_OFFSET, 0, root);
            task_root.write_u32(ROOT_OBJECT_LIST_LENGTH, 0, root);
            if self.supports_argument_buffers {
                task_root.write_u64(ROOT_PLACEMENT_LIST_OFFSET, 0, root);
                task_root.write_u32(ROOT_PLACEMENT_LIST_LENGTH, 0, root);
            }
            drop(old_pl_mapping);
            return false;
        }
        st.objects.offset = new_ol_off;
        st.objects.length = new_ol_len;
        st.objects.mapped_length = mapped_length(ol_end, new_ol_off);
        if !self.supports_argument_buffers {
            drop(st);
            drop(old_pl_mapping);
            return true;
        }
        st.placements.offset = new_pl_off;
        st.placements.length = new_pl_len;
        st.placements.mapped_length = mapped_length(pl_end, new_pl_off);
        let mut old_pl_address = st.placements.address;
        let released = st.placements.unmap();
        let legacy = st.is_legacy();
        let (pl_off, pl_len, pl_mapped_len) = (
            st.placements.offset,
            st.placements.length,
            st.placements.mapped_length,
        );
        drop(st);
        drop(released);
        if old_pl_mapping.is_none() && old_pl_off != 0 && old_pl_len != 0 {
            old_pl_mapping = self.new_mapping(hold, legacy, old_pl_off, old_pl_mapped_len, false);
            if let Some(m) = &old_pl_mapping {
                old_pl_address = GuestPtr::new(m.virtual_address());
            }
        }
        if pl_off != 0 && pl_len != 0 {
            let pages = if pl_mapped_len != 0 {
                self.resolve_pages(hold, pl_off, pl_mapped_len)
            } else {
                Vec::new()
            };
            let mapping = if pages.is_empty() {
                self.new_mapping(hold, legacy, pl_off, pl_mapped_len, true)
            } else {
                None
            };
            let (previous, _) = self.st().placements.adopt(mapping, pages);
            drop(previous);
        }
        let new_pl_address = self.st().placements.address;
        let new_pl_has_pages = !self.st().placements.pages.is_empty();
        if (old_pl_len != 0 && old_pl_address.is_null())
            || (new_pl_len != 0 && old_pl_len != 0 && new_pl_address.is_null() && !new_pl_has_pages)
        {
            log::warn!("object lists refused: the placement list to copy is not mapped");
            // All four root fields are zeroed here, whatever the flag; the members keep the new
            // lists.
            task_root.write_u64(ROOT_OBJECT_LIST_OFFSET, 0, root);
            task_root.write_u32(ROOT_OBJECT_LIST_LENGTH, 0, root);
            task_root.write_u64(ROOT_PLACEMENT_LIST_OFFSET, 0, root);
            task_root.write_u32(ROOT_PLACEMENT_LIST_LENGTH, 0, root);
            drop(old_pl_mapping);
            return false;
        }
        if !old_pl_address.is_null() && !new_pl_address.is_null() {
            let n = u64::from(old_pl_len.min(new_pl_len));
            Self::copy_placement_list(new_pl_address, old_pl_address, n);
        }
        // The old placement mapping goes after the copy
        drop(old_pl_mapping);
        true
    }

    /// Whether the object list is mapped.
    pub fn is_object_list_mapped(&self) -> bool {
        self.st().object_list_mapped()
    }

    /// Whether the placement list is mapped (the legacy kind never maps one).
    pub fn is_placement_list_mapped(&self) -> bool {
        self.st().placement_list_mapped()
    }

    /// Maps the object list; answers whether it is mapped. The legacy kind asks the task for
    /// its mapped address (a raise is logged and leaves the list unmapped, a NULL address
    /// leaves it as it was); the memory-map kind makes a read-only mapping.
    pub fn map_object_list(&self, hold: Hold<'_>) -> bool {
        let st = self.st();
        if st.is_legacy() {
            let (offset, length) = (st.objects.offset, st.objects.length);
            drop(st);
            // The task's lazily mapped address; the list holds the mapping behind it
            let r = match self.task.upgrade() {
                Some(task) => {
                    task.mapped_address_for_offset(hold, offset, u64::from(length), false)
                }
                None => Ok(MappedAddress::unmapped(HostPtr::NULL)),
            };
            if let Err(exception) = &r {
                log::warn!(
                    "object list of {length:#x} bytes at {offset:#x} not mapped: {}",
                    exception.exception()
                );
            }
            let mut st = self.st();
            let Lists {
                kind: ObjectListKind::Legacy { object_list_mapped },
                objects,
                ..
            } = &mut *st
            else {
                drop(st);
                if let Err(exception) = r {
                    hold.land(exception);
                }
                return false;
            };
            let mut previous = None;
            // The raise ends last, after the lock and the previous mapping go.
            let mut raised = None;
            match r {
                Ok(mapped) => {
                    if !mapped.address.is_null() {
                        objects.address = GuestPtr::new(mapped.address);
                        previous = std::mem::replace(&mut objects.mapping, mapped.mapping);
                        *object_list_mapped = true;
                    }
                }
                // The previous address stays
                Err(exception) => {
                    *object_list_mapped = false;
                    raised = Some(exception);
                }
            }
            let mapped = *object_list_mapped;
            drop(st);
            drop(previous);
            if let Some(exception) = raised {
                hold.land(exception);
            }
            return mapped;
        }
        if !st.objects.address.is_null() || !st.objects.pages.is_empty() {
            return true;
        }
        let (offset, length) = (st.objects.offset, st.objects.mapped_length);
        drop(st);
        let pages = if length != 0 {
            self.resolve_pages(hold, offset, length)
        } else {
            Vec::new()
        };
        let mapping = if pages.is_empty() {
            self.new_mapping(hold, false, offset, length, false)
        } else {
            None
        };
        let mut st = self.st();
        let (previous, mapped) = st.objects.adopt(mapping, pages);
        drop(st);
        drop(previous);
        if !mapped {
            log::warn!("object list of {length:#x} bytes at {offset:#x} not mapped");
        } else {
            log::info!(
                "object list of {length:#x} bytes at {offset:#x} mapped (pages={})",
                self.st().objects.pages.len()
            );
        }
        mapped
    }

    /// Maps the placement list; answers whether it is mapped (an empty one counts). The legacy
    /// kind has none: mapped only when empty.
    pub fn map_placement_list(&self, hold: Hold<'_>) -> bool {
        let st = self.st();
        if st.is_legacy() {
            return st.placements.length == 0;
        }
        if !st.placements.address.is_null() || !st.placements.pages.is_empty() || st.placements.length == 0 {
            return true;
        }
        let (offset, length) = (st.placements.offset, st.placements.mapped_length);
        drop(st);
        let pages = if length != 0 {
            self.resolve_pages(hold, offset, length)
        } else {
            Vec::new()
        };
        let mapping = if pages.is_empty() {
            self.new_mapping(hold, false, offset, length, true)
        } else {
            None
        };
        let mut st = self.st();
        let (previous, mapped) = st.placements.adopt(mapping, pages);
        drop(st);
        drop(previous);
        if !mapped {
            log::warn!("placement list of {length:#x} bytes at {offset:#x} not mapped");
        } else {
            log::info!(
                "placement list of {length:#x} bytes at {offset:#x} mapped (pages={})",
                self.st().placements.pages.len()
            );
        }
        mapped
    }

    /// The object list mapped, under `mutex` unless it already is: the test that it is comes
    /// before the lock, racing with another mapper.
    pub fn ensure_object_list_mapped(&self, hold: Hold<'_>) -> bool {
        if self.is_object_list_mapped() {
            return true;
        }
        let _lock = self.mutex.lock();
        self.map_object_list(hold)
    }

    /// The placement list mapped, as [`ObjectLists::ensure_object_list_mapped`].
    pub fn ensure_placement_list_mapped(&self, hold: Hold<'_>) -> bool {
        if self.is_placement_list_mapped() {
            return true;
        }
        let _lock = self.mutex.lock();
        self.map_placement_list(hold)
    }

    /// Both lists mapped: the placement list only once the object list is.
    pub fn ensure_mapped(&self, hold: Hold<'_>) -> bool {
        self.ensure_object_list_mapped(hold) && self.ensure_placement_list_mapped(hold)
    }

    /// Unmaps the object list.
    pub fn unmap_object_list(&self) {
        let released = self.st().unmap_object_list();
        drop(released);
    }

    /// Unmaps the placement list (the legacy kind has none).
    pub fn unmap_placement_list(&self) {
        let mut st = self.st();
        if st.is_legacy() {
            return;
        }
        let released = st.placements.unmap();
        drop(st);
        drop(released);
    }

    /// The object list's entry count.
    pub fn object_list_count(&self) -> u32 {
        self.st().object_count
    }

    /// The placement list's length.
    pub fn placement_list_length(&self) -> u32 {
        self.st().placements.length
    }

    /// Where the object list is in the task, as the last `set_object_list` left it (zero after
    /// a refusal).
    pub fn object_list_range(&self) -> ListRange {
        let st = self.st();
        ListRange {
            offset: st.objects.offset,
            length: st.objects.length,
        }
    }

    /// Where the placement list is in the task, as [`ObjectLists::object_list_range`].
    pub fn placement_list_range(&self) -> ListRange {
        let st = self.st();
        ListRange {
            offset: st.placements.offset,
            length: st.placements.length,
        }
    }
}

impl<P: Platform> Drop for ObjectLists<P> {
    /// Releases the object list's mapping, then the placement list's (a mapping's release can
    /// unmap in the host).
    fn drop(&mut self) {
        let lists = self.state.get_mut();
        let objects = lists.objects.mapping.take();
        let placements = lists.placements.mapping.take();
        drop(objects);
        drop(placements);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::HostPtr;
    use crate::memory::mapping::Teardown;
    use crate::testkit::calllog::{Arg, Call};
    use crate::testkit::capture::catch_death;
    use crate::testkit::guest_ram::RawRegion;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::resource_host::FakeResourceHost;
    use crate::testkit::script::Step;

    type P = FakePlatform;
    type Mgr = ObjectLists<P>;
    type Map = Option<Arc<Mapping<P>>>;

    const LEGACY: fn() -> ObjectListKind = || ObjectListKind::Legacy {
        object_list_mapped: false,
    };
    const TASK_LEN: u64 = 0x10_0000;
    /// A guest page, which the lists' offsets are tested against and their lengths rounded to.
    const PAGE: u64 = APV_PAGE_SIZE;

    struct Fx {
        p: Arc<P>,
        rec: Arc<FakeResourceHost>,
        root: Arc<RawRegion>,
        mem: Arc<RawRegion>,
    }

    fn fx() -> Fx {
        let p = Arc::new(FakePlatform::new());
        let rec = FakeResourceHost::new(p.clone());
        Fx {
            p,
            rec,
            root: RawRegion::new(0x4000).unwrap(),
            mem: RawRegion::new(0x10000).unwrap(),
        }
    }

    impl Fx {
        fn mgr(&self, kind: ObjectListKind, sab: bool) -> Arc<Mgr> {
            let t: Arc<dyn ResourceHost<P>> = self.rec.clone();
            Mgr::new(kind, TASK_LEN, sab, Arc::downgrade(&t))
        }
        fn root(&self) -> GuestPtr {
            GuestPtr::new(self.root.addr())
        }
        /// The four list fields of the task root.
        fn root_fields(&self) -> (u64, u32, u64, u32) {
            (
                self.root.read_u64(8).unwrap(),
                self.root.read_u32(0x10).unwrap(),
                self.root.read_u64(0x1c).unwrap(),
                self.root.read_u32(0x24).unwrap(),
            )
        }
        fn fill_root(&self) {
            assert!(self.root.write(0, &[0xaa; 40]));
        }
        /// A mapping over `mem` at `off`.
        fn mapping(&self, off: u64, len: u64) -> Arc<Mapping<P>> {
            Mapping::new(self.mem.addr().wrapping_add(off), len, true, Teardown::None)
        }
        fn next_mapping(&self, m: Map) {
            self.p.script().push("new_virtual_mapping", Step::value(m));
        }
        fn host_calls(&self) -> Vec<Call> {
            self.p
                .call_log()
                .calls()
                .into_iter()
                .filter(|c| c.class == "ResourceHost")
                .collect()
        }
    }

    fn nvm(offset: u64, length: u64, writable: bool) -> Vec<Arg> {
        vec![Arg::U64(offset), Arg::U64(length), Arg::Bool(writable)]
    }

    /// A guest store a test plays against an object entry the host is loading.
    type EntryRace = Box<dyn FnMut(GuestPtr)>;

    thread_local! {
        /// The guest store played after each of `load_object_entry`'s loads on this thread.
        static AFTER_ENTRY_LOAD: std::cell::RefCell<Option<EntryRace>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Called by `load_object_entry` after each of its loads: where a guest store racing the
    /// host lands.
    pub(super) fn after_entry_load(entry: GuestPtr) {
        AFTER_ENTRY_LOAD.with(|h| {
            if let Some(f) = h.borrow_mut().as_mut() {
                f(entry);
            }
        });
    }

    /// Clears this thread's entry race when dropped, so a failed test leaves none behind.
    struct EntryRaceGuard;

    impl Drop for EntryRaceGuard {
        fn drop(&mut self) {
            AFTER_ENTRY_LOAD.with(|h| *h.borrow_mut() = None);
        }
    }

    /// Plays `store` against the entry after the host's `nth` load of it (1 or 2), and counts
    /// the loads into the returned cell.
    fn race_after_entry_load(
        nth: u32,
        mut store: impl FnMut(GuestPtr) + 'static,
    ) -> (EntryRaceGuard, std::rc::Rc<std::cell::Cell<u32>>) {
        let loads = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let seen = loads.clone();
        AFTER_ENTRY_LOAD.with(|h| {
            *h.borrow_mut() = Some(Box::new(move |entry| {
                seen.set(seen.get() + 1);
                if seen.get() == nth {
                    store(entry);
                }
            }));
        });
        (EntryRaceGuard, loads)
    }

    mod new {
        use super::*;

        #[test]
        fn starts_empty_and_unmapped() {
            let f = fx();
            for m in [
                f.mgr(LEGACY(), false),
                f.mgr(ObjectListKind::MemoryMap, true),
            ] {
                let st = m.st();
                assert!(st.objects.mapping.is_none() && st.placements.mapping.is_none());
                assert!(st.objects.address.is_null() && st.placements.address.is_null());
                assert_eq!(
                    (
                        st.objects.offset,
                        st.objects.length,
                        st.object_count,
                        st.objects.mapped_length,
                        st.placements.offset,
                        st.placements.length,
                        st.placements.mapped_length
                    ),
                    (0, 0, 0, 0, 0, 0, 0)
                );
                drop(st);
                assert!(!m.is_object_list_mapped() && !m.is_placement_list_mapped());
                assert_eq!((m.object_list_count(), m.placement_list_length()), (0, 0));
                assert_eq!(m.task_length, TASK_LEN);
            }
            assert!(f.host_calls().is_empty());
        }

        #[test]
        fn drop_releases_both_mappings() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            let (a, b) = (f.mapping(0, 0x4000), f.mapping(0x4000, 0x4000));
            m.st().objects.mapping = Some(a.clone());
            m.st().placements.mapping = Some(b.clone());
            drop(m);
            assert_eq!((Arc::strong_count(&a), Arc::strong_count(&b)), (1, 1));
        }
    }

    mod get_object_entry {
        use super::*;

        #[test]
        fn not_mapped() {
            let f = fx();
            for m in [
                f.mgr(LEGACY(), false),
                f.mgr(ObjectListKind::MemoryMap, false),
            ] {
                m.st().object_count = 5;
                assert_eq!(
                    m.get_object_entry(0),
                    Err("the object list is not mapped".to_owned())
                );
            }
        }

        /// Mapped: range check, then a live 12-byte read
        #[test]
        fn mapped_reads_live_and_checks_range() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, false);
            {
                let mut st = m.st();
                st.objects.address = GuestPtr::new(f.mem.addr());
                st.object_count = 2;
            }
            assert!(f.mem.write(12, &[0x05, 0x10, 0, 0]));
            assert!(f.mem.write(16, &0x1234_5678_9abc_u64.to_le_bytes()));
            let e = m.get_object_entry(1).unwrap();
            assert_eq!(
                (e.object_type(), e.length(), { e.offset }),
                (5, 0x10, 0x1234_5678_9abc)
            );
            assert!(f.mem.write(12, &[0x07]));
            assert_eq!(m.get_object_entry(1).unwrap().object_type(), 7);
            assert_eq!(
                m.get_object_entry(2),
                Err("reference 2 is past the object list's 2 entries".to_owned())
            );
            assert_eq!(
                m.get_object_entry(u32::MAX),
                Err("reference 4294967295 is past the object list's 2 entries".to_owned())
            );
        }

        /// A mapped memory-map manager over `f.mem` with `count` entries, entry 3 holding type 1,
        /// length 0x40 and offset 0x1_0000_0010.
        fn entry_three(f: &Fx) -> Arc<Mgr> {
            let m = f.mgr(ObjectListKind::MemoryMap, false);
            {
                let mut st = m.st();
                st.objects.address = GuestPtr::new(f.mem.addr());
                st.object_count = 4;
            }
            assert!(f.mem.write(36, &(1u32 | (0x40 << 8)).to_le_bytes()));
            assert!(f.mem.write(40, &0x1_0000_0010u64.to_le_bytes()));
            m
        }

        /// `ldr w9, [x8, #0x8]` then `ldr x8, [x8]`: a guest's one 32-bit store of entry 3's
        /// type/length word, type 1 length 0x40 to type 2 length 0x80, lands after the high word's
        /// load and the second load reads it whole: never the new type byte with the old length
        /// bytes, which the byte-wise copy could combine
        #[test]
        fn the_type_and_length_word_is_one_load_after_the_high_word() {
            let f = fx();
            let m = entry_three(&f);
            let (_race, loads) = race_after_entry_load(1, |entry| {
                entry.write_u32(0, 2 | (0x80 << 8), DeathKind::UnmappedEntry.site());
            });
            let e = m.get_object_entry(3).unwrap();
            assert_eq!(loads.get(), 2);
            assert_eq!(
                (e.object_type(), e.length(), { e.offset }),
                (2, 0x80, 0x1_0000_0010)
            );
            // After the second load the copy is done: the entry keeps the word it read.
            assert!(f.mem.write(36, &(1u32 | (0x40 << 8)).to_le_bytes()));
            let (_race, loads) = race_after_entry_load(2, |entry| {
                entry.write_u32(0, 2 | (0x80 << 8), DeathKind::UnmappedEntry.site());
            });
            let e = m.get_object_entry(3).unwrap();
            assert_eq!(loads.get(), 2);
            assert_eq!((e.object_type(), e.length()), (1, 0x40));
        }

        /// The offset's high word is loaded first, then its low word with the type/length word; a
        /// guest's 64-bit store of the offset between the two loads leaves the old high word with
        /// the new low one
        #[test]
        fn the_offset_high_word_is_loaded_first() {
            let f = fx();
            let m = entry_three(&f);
            let (_race, loads) = race_after_entry_load(1, |entry| {
                entry.write_u64(4, 0x2_0000_0020, DeathKind::UnmappedEntry.site());
            });
            let e = m.get_object_entry(3).unwrap();
            assert_eq!(loads.get(), 2);
            assert_eq!(
                (e.object_type(), e.length(), { e.offset }),
                (1, 0x40, 0x1_0000_0020)
            );
        }

        /// Every index's two loads, the odd ones' 8-byte load crossing a 16-byte granule included,
        /// give the entry as stored
        #[test]
        fn every_index_reads_the_packed_entry() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, false);
            {
                let mut st = m.st();
                st.objects.address = GuestPtr::new(f.mem.addr());
                st.object_count = 8;
            }
            for i in 0..8u32 {
                let at = u64::from(i) * 12;
                let word = (i + 1) | ((0x10 * (i + 1)) << 8);
                let offset = 0x0123_4567_89ab_cdefu64.rotate_left(i * 8);
                assert!(f.mem.write(at, &word.to_le_bytes()));
                assert!(f.mem.write(at + 4, &offset.to_le_bytes()));
            }
            for i in 0..8u32 {
                let e = m.get_object_entry(i).unwrap();
                assert_eq!(
                    (e.object_type(), e.length(), { e.offset }),
                    (
                        (i + 1).low_u8(),
                        0x10 * (i + 1),
                        0x0123_4567_89ab_cdefu64.rotate_left(i * 8)
                    )
                );
            }
        }

        /// Legacy: the flag decides
        #[test]
        fn legacy_uses_the_flag() {
            let f = fx();
            let m = f.mgr(LEGACY(), false);
            {
                let mut st = m.st();
                st.objects.address = GuestPtr::new(f.mem.addr());
                st.object_count = 1;
            }
            assert!(m.get_object_entry(0).is_err());
            if let ObjectListKind::Legacy { object_list_mapped } = &mut m.st().kind {
                *object_list_mapped = true;
            }
            assert!(m.get_object_entry(0).is_ok());
        }
    }

    mod get_placement_list_entry {
        use super::*;

        /// Mapped: the entry's address, bounded by the object list's count
        #[test]
        fn mapped_gives_address_bounded_by_object_count() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            {
                let mut st = m.st();
                st.placements.address = GuestPtr::new(f.mem.addr());
                st.object_count = 3;
                st.placements.length = 0x1000;
            }
            assert_eq!(
                m.get_placement_list_entry(2),
                Ok(Some(GuestPtr::new(f.mem.addr().wrapping_add(32))))
            );
            assert_eq!(
                m.get_placement_list_entry(3),
                Err("reference 3 is past the object list's 3 entries".to_owned())
            );
        }

        /// Not mapped: an error only with argument buffers and a list; else nullptr
        #[test]
        fn not_mapped_cases() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            assert_eq!(m.get_placement_list_entry(0), Ok(None));
            m.st().placements.length = 16;
            assert_eq!(
                m.get_placement_list_entry(0),
                Err("the placement list is not mapped".to_owned())
            );
            let m = f.mgr(ObjectListKind::MemoryMap, false);
            m.st().placements.length = 16;
            assert_eq!(m.get_placement_list_entry(0), Ok(None));
            // Legacy is never mapped, whatever its address.
            let m = f.mgr(LEGACY(), true);
            m.st().placements.address = GuestPtr::new(f.mem.addr());
            m.st().placements.length = 16;
            assert_eq!(
                m.get_placement_list_entry(0),
                Err("the placement list is not mapped".to_owned())
            );
        }
    }

    mod set_object_list {
        use super::*;

        /// Success without argument buffers
        #[test]
        fn success_without_argument_buffers() {
            let f = fx();
            f.fill_root();
            let m = f.mgr(LEGACY(), false);
            // 25 / 12 = 2 entries.
            assert!(m.set_object_list(Hold::NONE, 0x8000, 25, 0x4000, 0x20, f.root()));
            assert_eq!(
                f.root_fields(),
                (0x8000, 25, 0xaaaa_aaaa_aaaa_aaaa, 0xaaaa_aaaa)
            );
            let st = m.st();
            assert_eq!(
                (
                    st.objects.offset,
                    st.objects.length,
                    st.object_count,
                    st.objects.mapped_length
                ),
                (0x8000, 25, 2, PAGE)
            );
            // The placement members are untouched.
            assert_eq!((st.placements.offset, st.placements.length), (0, 0));
            // The other root bytes are untouched.
            assert_eq!(f.root.read_u64(0).unwrap(), 0xaaaa_aaaa_aaaa_aaaa);
            assert_eq!(f.root.read_u64(0x14).unwrap(), 0xaaaa_aaaa_aaaa_aaaa);
            drop(st);
            assert!(f.host_calls().is_empty());
        }

        /// The object list is unmapped first
        #[test]
        fn unmaps_the_object_list_first() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, false);
            let old = f.mapping(0, 0x4000);
            m.st().objects.mapping = Some(old.clone());
            m.st().objects.address = GuestPtr::new(old.virtual_address());
            assert!(m.set_object_list(Hold::NONE, 0, 12, 0, 0, f.root()));
            assert!(!m.is_object_list_mapped());
            assert_eq!(Arc::strong_count(&old), 1);
            let l = f.mgr(LEGACY(), false);
            if let ObjectListKind::Legacy { object_list_mapped } = &mut l.st().kind {
                *object_list_mapped = true;
            }
            // Refused, and unmapped all the same.
            assert!(!l.set_object_list(Hold::NONE, 0, 12, 0, 0x10_0000, f.root()));
            assert!(!l.is_object_list_mapped());
        }

        /// An end at the task's end is refused; the root is zeroed after its write
        #[test]
        fn end_at_task_length_refused_and_zeroed() {
            let f = fx();
            f.fill_root();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            let pl = f.mapping(0x4000, 0x4000);
            m.st().placements.mapping = Some(pl.clone());
            m.st().placements.address = GuestPtr::new(pl.virtual_address());
            m.st().objects.mapped_length = 0x777;
            m.st().placements.mapped_length = 0x888;
            assert!(!m.set_object_list(Hold::NONE, TASK_LEN - 12, 12, 0, 0, f.root()));
            assert_eq!(f.root_fields(), (0, 0, 0, 0));
            let st = m.st();
            assert_eq!(
                (
                    st.objects.offset,
                    st.objects.length,
                    st.object_count,
                    st.placements.offset,
                    st.placements.length
                ),
                (0, 0, 0, 0, 0)
            );
            assert!(st.placements.mapping.is_none() && st.placements.address.is_null());
            // The mapped lengths keep their old values.
            assert_eq!(
                (st.objects.mapped_length, st.placements.mapped_length),
                (0x777, 0x888)
            );
            drop(st);
            assert_eq!(Arc::strong_count(&pl), 1);
            // One short of the end is accepted.
            assert!(m.set_object_list(Hold::NONE, TASK_LEN - 0x4000, 12, 0, 0, f.root()));
            // The placement end is tested too, even when zero-length at the end.
            assert!(!m.set_object_list(Hold::NONE, 0, 12, TASK_LEN, 0, f.root()));
        }

        /// Without argument buffers the root's placement fields stay on refusal
        #[test]
        fn refusal_without_argument_buffers_keeps_root_placement() {
            let f = fx();
            f.fill_root();
            let m = f.mgr(LEGACY(), false);
            assert!(!m.set_object_list(Hold::NONE, u64::MAX, 1, 0, 0, f.root()));
            assert_eq!(f.root_fields(), (0, 0, 0xaaaa_aaaa_aaaa_aaaa, 0xaaaa_aaaa));
        }

        /// Overflow of either end is refused
        #[test]
        fn overflow_refused() {
            let f = fx();
            let m = f.mgr(LEGACY(), true);
            assert!(!m.set_object_list(Hold::NONE, u64::MAX - 3, 4, 0, 0, f.root()));
            assert!(!m.set_object_list(Hold::NONE, 0, 12, u64::MAX, 1, f.root()));
            // An exact fit to u64::MAX does not overflow, but is past the task.
            assert!(!m.set_object_list(Hold::NONE, u64::MAX - 4, 4, 0, 0, f.root()));
            assert!(m.set_object_list(Hold::NONE, 0, 12, 0, 0, f.root()));
        }

        /// The previous offsets' alignment is tested
        #[test]
        fn previous_offsets_alignment() {
            let f = fx();
            let m = f.mgr(LEGACY(), false);
            // The members are 0: an unaligned new offset is accepted.
            assert!(m.set_object_list(Hold::NONE, 0x4010, 12, 0, 0, f.root()));
            assert_eq!(m.st().objects.mapped_length, PAGE - 0x10);
            // Now the member is unaligned: an aligned new offset is refused.
            assert!(!m.set_object_list(Hold::NONE, 0x8000, 12, 0, 0, f.root()));
            // The refusal zeroed the member: accepted again.
            assert!(m.set_object_list(Hold::NONE, 0x8000, 12, 0, 0, f.root()));
            // The previous placement offset counts too.
            let m = f.mgr(LEGACY(), true);
            assert!(m.set_object_list(Hold::NONE, 0, 12, PAGE / 2, 16, f.root()));
            assert!(!m.set_object_list(Hold::NONE, 0, 12, 0, 0, f.root()));
        }

        /// A placement list shorter than 16 per entry is refused only at a nonzero offset
        #[test]
        fn required_placement_length() {
            let f = fx();
            let m = f.mgr(LEGACY(), true);
            assert!(!m.set_object_list(Hold::NONE, 0, 36, 0x4000, 47, f.root()));
            assert!(m.set_object_list(Hold::NONE, 0, 36, 0x4000, 48, f.root()));
            let m = f.mgr(LEGACY(), true);
            assert!(m.set_object_list(Hold::NONE, 0, 36, 0, 0, f.root()));
            // Without argument buffers the check still applies.
            let m = f.mgr(LEGACY(), false);
            assert!(!m.set_object_list(Hold::NONE, 0, 36, 0x4000, 16, f.root()));
        }

        /// With argument buffers: the root, the members, the new placement mapping
        /// The task's mapping, adopted
        #[test]
        fn argument_buffers_map_new_placement() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            f.next_mapping(Some(f.mapping(0x8000, 2 * PAGE)));
            // A page and a byte of placements, mapped as two pages.
            let length = (PAGE + 1) as u32;
            assert!(m.set_object_list(Hold::NONE, 0x4000, 24, 0x8000, length, f.root()));
            assert_eq!(f.root_fields(), (0x4000, 24, 0x8000, length));
            let calls = f.host_calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(
                (calls[0].sel, calls[0].args.clone()),
                ("new_virtual_mapping", nvm(0x8000, 2 * PAGE, true))
            );
            let st = m.st();
            assert_eq!(
                (
                    st.placements.offset,
                    st.placements.length,
                    st.placements.mapped_length
                ),
                (0x8000, length, 2 * PAGE)
            );
            assert_eq!(
                st.placements.address.addr(),
                f.mem.addr().wrapping_add(0x8000)
            );
            drop(st);
            assert!(m.is_placement_list_mapped());
        }

        /// The old list is mapped read-only and copied into the new, min of the lengths
        #[test]
        fn copies_old_placement_into_new() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            // The first list, at 0x4000 (0x20 bytes), never mapped.
            assert!(m.set_object_list(Hold::NONE, 0, 24, 0x4000, 0x20, f.root()));
            // (Its own mapping attempt got the recorder's nil.)
            assert_eq!(f.host_calls().len(), 1);
            let seq0 = f.host_calls().len();
            let old: Vec<u8> = (1..=0x20).collect();
            assert!(f.mem.write(0x100, &old));
            f.next_mapping(Some(f.mapping(0x100, PAGE)));
            f.next_mapping(Some(f.mapping(0x8000, PAGE)));
            assert!(m.set_object_list(Hold::NONE, 0, 12, 0xc000, 0x18, f.root()));
            let calls = f.host_calls().split_off(seq0);
            assert_eq!(
                calls.iter().map(|c| c.args.clone()).collect::<Vec<_>>(),
                vec![nvm(0x4000, PAGE, false), nvm(0xc000, PAGE, true)]
            );
            assert_eq!(f.mem.read_vec(0x8000, 0x18).unwrap(), old[..0x18].to_vec());
            assert_eq!(f.mem.read_vec(0x8018, 8).unwrap(), vec![0; 8]);
            // The read-only mapping of the old list was released at the end.
            let st = m.st();
            assert_eq!(
                Arc::strong_count(st.placements.mapping.as_ref().unwrap()),
                1
            );
        }

        /// A mapped old list is copied from its address, not remapped
        #[test]
        fn copies_from_existing_old_mapping() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            f.next_mapping(Some(f.mapping(0x100, PAGE)));
            assert!(m.set_object_list(Hold::NONE, 0, 12, 0x4000, 0x10, f.root()));
            let old_map = m.st().placements.mapping.clone().unwrap();
            assert!(f.mem.write(0x100, &[9; 0x10]));
            f.next_mapping(Some(f.mapping(0x8000, PAGE)));
            let seq0 = f.host_calls().len();
            assert!(m.set_object_list(Hold::NONE, 0, 12, 0x8000, 0x20, f.root()));
            let calls = f.host_calls().split_off(seq0);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].args, nvm(0x8000, PAGE, true));
            assert_eq!(
                f.mem.read_vec(0x8000, 0x20).unwrap(),
                [vec![9; 0x10], vec![0; 0x10]].concat()
            );
            assert_eq!(Arc::strong_count(&old_map), 1);
        }

        /// An old list that cannot be mapped zeroes all four root fields; the members keep the
        /// new lists
        #[test]
        fn old_list_unmappable_fails_after_setting_members() {
            let f = fx();
            f.fill_root();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            assert!(m.set_object_list(Hold::NONE, 0, 24, 0x4000, 0x20, f.root()));
            // The old list's mapping raises (swallowed: nil); the new one maps.
            f.p.script().push(
                "new_virtual_mapping",
                Step::raise("NSGenericException", "x"),
            );
            f.next_mapping(Some(f.mapping(0x8000, 0x4000)));
            assert!(!m.set_object_list(Hold::NONE, 0x4000, 12, 0x8000, 0x10, f.root()));
            assert_eq!(f.root_fields(), (0, 0, 0, 0));
            let st = m.st();
            assert_eq!(
                (
                    st.objects.offset,
                    st.objects.length,
                    st.placements.offset,
                    st.placements.length
                ),
                (0x4000, 12, 0x8000, 0x10)
            );
            assert!(!st.placements.address.is_null());
        }

        /// A new list that cannot be mapped fails only when an old list exists
        #[test]
        fn new_list_unmappable() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            // No old list: a nil new mapping is still success.
            assert!(m.set_object_list(Hold::NONE, 0, 24, 0x4000, 0x20, f.root()));
            assert!(!m.is_placement_list_mapped());
            // An old list mapped, the new one nil: failure, root zeroed.
            f.next_mapping(Some(f.mapping(0x100, 0x4000)));
            f.next_mapping(None);
            f.fill_root();
            assert!(!m.set_object_list(Hold::NONE, 0, 24, 0x8000, 0x20, f.root()));
            assert_eq!(f.root_fields(), (0, 0, 0, 0));
            // A zero-length new list with an old one: success, nothing copied.
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            assert!(m.set_object_list(Hold::NONE, 0, 0, 0x4000, 0x20, f.root()));
            f.next_mapping(Some(f.mapping(0x100, 0x4000)));
            assert!(m.set_object_list(Hold::NONE, 0, 0, 0, 0, f.root()));
        }

        /// Legacy with argument buffers: newMapping is nil
        #[test]
        fn legacy_new_mapping_is_nil() {
            let f = fx();
            let m = f.mgr(LEGACY(), true);
            assert!(m.set_object_list(Hold::NONE, 0, 24, 0x4000, 0x20, f.root()));
            assert!(!m.set_object_list(Hold::NONE, 0, 24, 0x8000, 0x20, f.root()));
            assert!(f.host_calls().is_empty());
        }

        /// A NULL task root faults at the first store
        #[test]
        fn null_root_faults() {
            let f = fx();
            let m = f.mgr(LEGACY(), false);
            let r = catch_death(|| {
                m.set_object_list(Hold::NONE, 0, 12, 0, 0, GuestPtr::new(HostPtr::NULL))
            });
            assert_eq!(r, Err(DeathKind::NullTaskRoot.site()));
            // The count was stored before the fault.
            assert_eq!(m.st().object_count, 1);
        }

        /// A gone task maps nothing
        #[test]
        fn gone_task_maps_nothing() {
            let f = fx();
            let m = {
                let rec = FakeResourceHost::new(f.p.clone());
                let t: Arc<dyn ResourceHost<P>> = rec;
                Mgr::new(
                    ObjectListKind::MemoryMap,
                    TASK_LEN,
                    true,
                    Arc::downgrade(&t),
                )
            };
            assert!(m.set_object_list(Hold::NONE, 0, 24, 0x4000, 0x20, f.root()));
            assert!(!m.is_placement_list_mapped());
            assert!(!m.map_object_list(Hold::NONE));
        }
    }

    mod legacy_map_object_list {
        use super::*;

        /// An address maps it
        #[test]
        fn address_maps() {
            let f = fx();
            let m = f.mgr(LEGACY(), false);
            assert!(m.set_object_list(Hold::NONE, 0x4000, 24, 0, 0, f.root()));
            f.p.script()
                .push("mapped_address_for_offset", Step::value(f.mem.addr()));
            assert!(m.map_object_list(Hold::NONE));
            assert!(m.is_object_list_mapped());
            assert_eq!(m.st().objects.address.addr(), f.mem.addr());
            let calls = f.host_calls();
            assert_eq!(
                (calls[0].sel, calls[0].args.clone()),
                ("mapped_address_for_offset", nvm(0x4000, 24, false))
            );
        }

        /// NULL leaves the flag as it was
        #[test]
        fn null_keeps_flag() {
            let f = fx();
            let m = f.mgr(LEGACY(), false);
            assert!(!m.map_object_list(Hold::NONE));
            f.p.script()
                .push("mapped_address_for_offset", Step::value(f.mem.addr()));
            assert!(m.map_object_list(Hold::NONE));
            assert!(m.map_object_list(Hold::NONE));
            assert_eq!(f.host_calls().len(), 3);
        }

        /// An exception is caught: unmapped, the old address kept
        #[test]
        fn exception_unmaps_keeps_address() {
            let f = fx();
            let m = f.mgr(LEGACY(), false);
            f.p.script()
                .push("mapped_address_for_offset", Step::value(f.mem.addr()));
            assert!(m.map_object_list(Hold::NONE));
            f.p.script().push(
                "mapped_address_for_offset",
                Step::raise("NSGenericException", "x"),
            );
            assert!(!m.map_object_list(Hold::NONE));
            assert!(!m.is_object_list_mapped());
            assert_eq!(m.st().objects.address.addr(), f.mem.addr());
        }
    }

    mod memory_map_map_object_list {
        use super::*;

        /// Maps the page-rounded length read-only
        #[test]
        fn maps_read_only() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, false);
            // A page and a byte of objects, mapped as two pages.
            let length = (PAGE + 1) as u32;
            assert!(m.set_object_list(Hold::NONE, 0x4000, length, 0, 0, f.root()));
            f.next_mapping(Some(f.mapping(0x400, 2 * PAGE)));
            assert!(m.map_object_list(Hold::NONE));
            assert_eq!(f.host_calls()[0].args, nvm(0x4000, 2 * PAGE, false));
            assert_eq!(
                m.st().objects.address.addr(),
                f.mem.addr().wrapping_add(0x400)
            );
            // Already mapped: no call.
            assert!(m.map_object_list(Hold::NONE));
            assert_eq!(f.host_calls().len(), 1);
        }

        /// Nil, or a swallowed exception, is false
        /// An exception is swallowed as nil
        #[test]
        fn nil_or_exception_fails() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, false);
            assert!(!m.map_object_list(Hold::NONE));
            f.p.script().push(
                "new_virtual_mapping",
                Step::raise("NSGenericException", "zero length"),
            );
            assert!(!m.map_object_list(Hold::NONE));
            assert!(!m.is_object_list_mapped());
            assert!(m.st().objects.mapping.is_none());
        }
    }

    mod map_placement_list {
        use super::*;

        /// True only without a list
        #[test]
        fn legacy_is_length_zero() {
            let f = fx();
            let m = f.mgr(LEGACY(), true);
            assert!(m.map_placement_list(Hold::NONE));
            m.st().placements.length = 1;
            assert!(!m.map_placement_list(Hold::NONE));
            assert!(f.host_calls().is_empty());
        }

        /// Maps writable; no list or already mapped is true with no call; nil is false
        #[test]
        fn memory_map_cases() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            assert!(m.map_placement_list(Hold::NONE));
            assert!(f.host_calls().is_empty());
            {
                let mut st = m.st();
                st.placements.offset = 0x4000;
                st.placements.length = 0x10;
                st.placements.mapped_length = 0x4000;
            }
            assert!(!m.map_placement_list(Hold::NONE));
            f.next_mapping(Some(f.mapping(0, 0x4000)));
            assert!(m.map_placement_list(Hold::NONE));
            assert!(m.map_placement_list(Hold::NONE));
            let calls = f.host_calls();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[1].args, nvm(0x4000, 0x4000, true));
            assert!(m.is_placement_list_mapped());
        }
    }

    mod unmap {
        use super::*;

        #[test]
        fn legacy() {
            let f = fx();
            let m = f.mgr(LEGACY(), false);
            f.p.script()
                .push("mapped_address_for_offset", Step::value(f.mem.addr()));
            assert!(m.map_object_list(Hold::NONE));
            m.unmap_object_list();
            assert!(!m.is_object_list_mapped());
            // The address stays; only the flag changes.
            assert_eq!(m.st().objects.address.addr(), f.mem.addr());
            m.st().placements.address = GuestPtr::new(f.mem.addr());
            m.unmap_placement_list();
            assert_eq!(m.st().placements.address.addr(), f.mem.addr());
        }

        #[test]
        fn memory_map_releases() {
            let f = fx();
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            let (a, b) = (f.mapping(0, 0x4000), f.mapping(0x4000, 0x4000));
            {
                let mut st = m.st();
                st.objects.mapping = Some(a.clone());
                st.objects.address = GuestPtr::new(a.virtual_address());
                st.placements.mapping = Some(b.clone());
                st.placements.address = GuestPtr::new(b.virtual_address());
            }
            assert!(m.is_object_list_mapped() && m.is_placement_list_mapped());
            m.unmap_object_list();
            assert!(!m.is_object_list_mapped() && m.is_placement_list_mapped());
            assert_eq!(Arc::strong_count(&a), 1);
            m.unmap_placement_list();
            assert!(!m.is_placement_list_mapped());
            assert_eq!(Arc::strong_count(&b), 1);
        }
    }

    mod is_mapped {
        use super::*;

        #[test]
        fn by_flag_or_address() {
            let f = fx();
            let l = f.mgr(LEGACY(), true);
            l.st().objects.address = GuestPtr::new(f.mem.addr());
            l.st().placements.address = GuestPtr::new(f.mem.addr());
            assert!(!l.is_object_list_mapped() && !l.is_placement_list_mapped());
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            m.st().objects.address = GuestPtr::new(f.mem.addr());
            assert!(m.is_object_list_mapped() && !m.is_placement_list_mapped());
            m.st().placements.address = GuestPtr::new(f.mem.addr());
            assert!(m.is_placement_list_mapped());
        }

        #[test]
        fn counters() {
            let f = fx();
            let m = f.mgr(LEGACY(), true);
            assert!(m.set_object_list(Hold::NONE, 0, 120, 0x4000, 0xa0, f.root()));
            assert_eq!(
                (m.object_list_count(), m.placement_list_length()),
                (10, 0xa0)
            );
        }
    }

    mod logs {
        use super::*;
        use crate::tasks::test_support::Lines;

        /// Runs `body` with the fixture's platform in scope; the tasks area's lines it wrote.
        fn logged(f: &Fx, lines: &Lines, body: impl FnOnce()) -> Vec<String> {
            drop(lines.take());
            crate::resources::resource::with_platform(&f.p, body);
            lines.take()
        }

        /// The refusal and the failed copy log
        #[test]
        fn set_object_list_logs() {
            let lines = Lines::capture();
            let f = fx();
            let m = f.mgr(LEGACY(), false);
            let got = logged(&f, &lines, || {
                assert!(!m.set_object_list(Hold::NONE, u64::MAX, 1, 0, 0, f.root()));
            });
            assert_eq!(
                got,
                [
                    "object lists refused: 0x1 bytes of objects at 0xffffffffffffffff and 0x0 \
                     bytes of placements at 0x0, in a task of 0x100000 bytes"
                ]
            );
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            assert!(m.set_object_list(Hold::NONE, 0, 24, 0x4000, 0x20, f.root()));
            f.next_mapping(Some(f.mapping(0x100, 0x4000)));
            f.next_mapping(None);
            let got = logged(&f, &lines, || {
                assert!(!m.set_object_list(Hold::NONE, 0, 24, 0x8000, 0x20, f.root()));
            });
            assert_eq!(
                got,
                ["object lists refused: the placement list to copy is not mapped"]
            );
            // Success logs nothing.
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            assert!(
                logged(&f, &lines, || assert!(m.set_object_list(
                    Hold::NONE,
                    0,
                    0,
                    0,
                    0,
                    f.root()
                )))
                .is_empty()
            );
        }

        /// An exception logs its reason
        /// A nil mapping logs
        /// A nil mapping logs
        #[test]
        fn map_failures_log() {
            let lines = Lines::capture();
            let f = fx();
            let l = f.mgr(LEGACY(), false);
            f.p.script().push(
                "mapped_address_for_offset",
                Step::raise("NSGenericException", "boom"),
            );
            let got = logged(&f, &lines, || assert!(!l.map_object_list(Hold::NONE)));
            assert_eq!(got, ["object list of 0x0 bytes at 0x0 not mapped: boom"]);
            // A NULL address is not a failure that logs.
            assert!(logged(&f, &lines, || assert!(!l.map_object_list(Hold::NONE))).is_empty());
            let m = f.mgr(ObjectListKind::MemoryMap, true);
            let got = logged(&f, &lines, || assert!(!m.map_object_list(Hold::NONE)));
            assert_eq!(got, ["object list of 0x0 bytes at 0x0 not mapped"]);
            m.st().placements.length = 0x10;
            let got = logged(&f, &lines, || assert!(!m.map_placement_list(Hold::NONE)));
            assert_eq!(got, ["placement list of 0x0 bytes at 0x0 not mapped"]);
        }
    }
}

/// Real guest-page aliasing (`mach_vm_remap`), which only the host has: the placement-list copy
/// against libc's `memcpy`.
#[cfg(all(test, target_os = "macos"))]
#[allow(
    unsafe_code,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)] // The Mach calls that make the aliases, and libc's memcpy to compare against
mod alias_tests {
    use super::*;
    use crate::host::HostPtr;
    use crate::memory::mapping::Teardown;
    use crate::raise::death::DeathKind;
    use crate::raise::death::DeathSite;
    use crate::testkit::guest_ram::RawRegion;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::resource_host::FakeResourceHost;
    use crate::testkit::script::Step;

    unsafe extern "C" {
        static mach_task_self_: u32;
        fn mach_vm_allocate(t: u32, a: *mut u64, s: u64, f: i32) -> i32;
        fn mach_vm_deallocate(t: u32, a: u64, s: u64) -> i32;
        fn mach_vm_remap(
            t: u32,
            a: *mut u64,
            s: u64,
            m: u64,
            f: i32,
            st: u32,
            src: u64,
            copy: i32,
            cur: *mut i32,
            max: *mut i32,
            inh: u32,
        ) -> i32;
    }

    const SITE: DeathSite = DeathSite {
        kind: DeathKind::NullDeref,
        file: "test",
        line: 0,
    };

    /// A shared alias of `src..src+len` anywhere (VM_FLAGS_ANYWHERE, VM_INHERIT_SHARE).
    fn alias(src: u64, len: u64) -> u64 {
        let mut a = 0u64;
        let (mut c, mut m) = (0, 0);
        let kr = unsafe {
            mach_vm_remap(
                mach_task_self_,
                &mut a,
                len,
                0,
                1,
                mach_task_self_,
                src,
                0,
                &mut c,
                &mut m,
                1,
            )
        };
        assert_eq!(kr, 0);
        a
    }

    fn g(addr: u64) -> GuestPtr {
        GuestPtr::new(HostPtr::from_addr(addr as usize))
    }

    /// Guest 0xC000 bytes; the old list is guest 0x0..0x8000 (task offset 0x4000), the new one
    /// guest 0x4000..0xC000 (task offset 0x8000). Returns the new list's bytes after the copy,
    /// made by libc's `memcpy` or by the lists.
    fn run(copy_like_r: bool) -> (Vec<u8>, Vec<u8>) {
        let mut guest = 0u64;
        assert_eq!(
            unsafe { mach_vm_allocate(mach_task_self_, &mut guest, 0xC000, 1) },
            0
        );
        let a = alias(guest, 0x8000);
        let b = alias(guest + 0x4000, 0x8000);
        let p = Arc::new(FakePlatform::new());
        let rec = FakeResourceHost::new(p.clone());
        let t: Arc<dyn ResourceHost<FakePlatform>> = rec.clone();
        let m = ObjectLists::new(
            ObjectListKind::MemoryMap,
            0x10_0000,
            true,
            Arc::downgrade(&t),
        );
        let root = RawRegion::new(0x4000).unwrap();
        p.script().push(
            "new_virtual_mapping",
            Step::value(Some(Mapping::<FakePlatform>::new(
                HostPtr::from_addr(a as usize),
                0x8000,
                true,
                Teardown::None,
            ))),
        );
        assert!(m.set_object_list(
            Hold::NONE,
            0,
            24,
            0x4000,
            0x8000,
            GuestPtr::new(root.addr())
        ));
        let pattern: Vec<u8> = (0..0x8000u64)
            .map(|i| (i / 0x100) as u8 ^ (i as u8))
            .collect();
        g(guest).copy_from(0, &pattern, SITE);
        if copy_like_r {
            // A runtime length: a call of libc's `memcpy`
            let n = std::hint::black_box(0x8000usize);
            unsafe { std::ptr::copy_nonoverlapping(a as *const u8, b as *mut u8, n) };
        } else {
            p.script().push(
                "new_virtual_mapping",
                Step::value(Some(Mapping::<FakePlatform>::new(
                    HostPtr::from_addr(b as usize),
                    0x8000,
                    true,
                    Teardown::None,
                ))),
            );
            assert!(m.set_object_list(
                Hold::NONE,
                0,
                24,
                0x8000,
                0x8000,
                GuestPtr::new(root.addr())
            ));
        }
        let mut out = vec![0u8; 0x8000];
        g(guest + 0x4000).copy_to(0, &mut out, SITE);
        drop(m);
        for (addr, len) in [(a, 0x8000), (b, 0x8000), (guest, 0xC000)] {
            unsafe { mach_vm_deallocate(mach_task_self_, addr, len) };
        }
        (out, pattern)
    }

    /// Lists overlapping in guest memory: libc's forward copy through the aliases, the data
    /// repeating
    #[test]
    fn overlapping_placement_move_matches_memcpy() {
        let ((r, pattern), (rust, _)) = (run(true), run(false));
        assert_eq!(r.iter().zip(&rust).position(|(x, y)| x != y), None);
        // The old list's first page, twice.
        assert_eq!(rust, [&pattern[..0x4000], &pattern[..0x4000]].concat());
    }
}
