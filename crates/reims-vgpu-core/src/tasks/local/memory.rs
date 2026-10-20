//! Task memory: the bounds checks, the host mappings of both setups, the no-copy buffers over
//! task memory, and the copies and cursors guest commands read through.

use std::sync::Arc;

use super::{
    Backend, LocalTask, PAGE, PAGE_MASK, RaiseKind, page_address, raise_carrying, raise_caught,
    raise_on,
};
use crate::device::tasks::MAX_LOCAL_TASK_LENGTH;
use crate::host::{HostPtr, PhysRange};
use crate::memory::cursor::Cursor;
use crate::memory::mapping::Mapping;
use crate::memory::tracker::RangeTracker;
use crate::num::{ToU64, ToUsize};
use crate::platform::device::GpuDevice;
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::host::MappedAddress;
use crate::rt::Hold;
use crate::tasks::{Completion, ExecStatus};

/// Only completed command snapshots are kept; live cursors own their bytes independently.
const CURSOR_POOL_BYTES: usize = 8 << 20;
const CURSOR_POOL_ENTRIES: usize = 16;
const CURSOR_POOL_BUFFER: usize = 2 << 20;

#[derive(Default)]
struct CursorPool {
    buffers: Vec<Vec<u8>>,
    bytes: usize,
}

impl CursorPool {
    fn take(&mut self, length: usize) -> Option<Vec<u8>> {
        let maximum = length.saturating_mul(2).max(4096);
        let index = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capacity() >= length && buffer.capacity() <= maximum)
            .min_by_key(|(_, buffer)| buffer.capacity())
            .map(|(index, _)| index)?;
        let buffer = self.buffers.swap_remove(index);
        self.bytes -= buffer.capacity();
        Some(buffer)
    }

    fn put(&mut self, buffer: Vec<u8>) {
        let capacity = buffer.capacity();
        if capacity == 0
            || capacity > CURSOR_POOL_BUFFER
            || self.buffers.len() >= CURSOR_POOL_ENTRIES
            || capacity > CURSOR_POOL_BYTES - self.bytes
        {
            return;
        }
        self.bytes += capacity;
        self.buffers.push(buffer);
    }
}

type SharedCursorPool = Arc<std::sync::Mutex<CursorPool>>;

/// Its Arc is the cursor's keep-alive: only the final cursor/child release returns these bytes.
struct CursorSnapshot {
    bytes: Vec<u8>,
    pool: Option<SharedCursorPool>,
}

impl CursorSnapshot {
    fn new(length: usize) -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static POOL: std::sync::OnceLock<SharedCursorPool> = std::sync::OnceLock::new();
        let enabled = *ON.get_or_init(|| {
            std::env::var("REIMS_VGPU_CURSOR_POOL").map_or(true, |value| value != "0")
        });
        let pool = (enabled && length != 0 && length <= CURSOR_POOL_BUFFER).then(|| {
            Arc::clone(POOL.get_or_init(|| Arc::new(std::sync::Mutex::new(CursorPool::default()))))
        });
        Self::from_pool(length, pool)
    }

    fn from_pool(length: usize, pool: Option<SharedCursorPool>) -> Self {
        let reused = pool.as_ref().and_then(|pool| {
            pool.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take(length)
        });
        let bytes = match reused {
            Some(mut bytes) => {
                // clear + resize writes every requested byte, including a reused buffer's old
                // contents. Failed physical reads must leave zeros, never a previous snapshot.
                bytes.clear();
                bytes.resize(length, 0);
                bytes
            }
            None => vec![0; length],
        };
        Self { bytes, pool }
    }
}

impl Drop for CursorSnapshot {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .put(std::mem::take(&mut self.bytes));
        }
    }
}

#[cfg(test)]
mod cursor_pool_tests {
    use super::*;

    fn pool() -> SharedCursorPool {
        Arc::new(std::sync::Mutex::new(CursorPool::default()))
    }

    #[test]
    fn production_pool_switch_and_large_snapshot_fallback() {
        let snapshot = CursorSnapshot::new(32);
        let enabled = std::env::var("REIMS_VGPU_CURSOR_POOL").map_or(true, |value| value != "0");
        assert_eq!(snapshot.pool.is_some(), enabled);
        assert!(snapshot.bytes.iter().all(|&byte| byte == 0));
        assert!(CursorSnapshot::new(0).pool.is_none());
        assert!(CursorSnapshot::new(CURSOR_POOL_BUFFER + 1).pool.is_none());
    }

    #[test]
    fn reused_snapshot_clears_bytes_before_partial_copy() {
        let pool = pool();
        let mut first = CursorSnapshot::from_pool(64, Some(Arc::clone(&pool)));
        first.bytes.fill(0xab);
        let pointer = first.bytes.as_ptr();
        drop(first);
        let mut next = CursorSnapshot::from_pool(32, Some(pool));
        assert_eq!(next.bytes.as_ptr(), pointer);
        assert_eq!(next.bytes, vec![0; 32]);
        // The unchanged copy_pages path may leave bytes unwritten after failed reads.
        next.bytes[..4].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(&next.bytes[4..], &[0; 28]);
    }

    #[test]
    fn retained_capacity_and_entry_limits_are_bounded() {
        let mut pool = CursorPool::default();
        for _ in 0..CURSOR_POOL_ENTRIES + 1 {
            pool.put(Vec::with_capacity(16));
        }
        assert_eq!(pool.buffers.len(), CURSOR_POOL_ENTRIES);
        let mut pool = CursorPool::default();
        pool.put(Vec::with_capacity(CURSOR_POOL_BUFFER + 1));
        assert!(pool.buffers.is_empty());
        for _ in 0..5 {
            pool.put(Vec::with_capacity(CURSOR_POOL_BUFFER));
        }
        assert_eq!(pool.bytes, CURSOR_POOL_BYTES);
        assert_eq!(pool.buffers.len(), 4);
        assert!(pool.take(CURSOR_POOL_BUFFER / 4).is_none());
        assert!(pool.take(CURSOR_POOL_BUFFER).is_some());
        assert_eq!(pool.bytes, CURSOR_POOL_BYTES - CURSOR_POOL_BUFFER);
    }

    #[test]
    fn final_cursor_child_returns_snapshot() {
        let pool = pool();
        let mut snapshot = CursorSnapshot::from_pool(16, Some(Arc::clone(&pool)));
        snapshot.bytes.fill(7);
        let pointer = HostPtr::from_ptr(snapshot.bytes.as_mut_ptr());
        let mut cursor = Cursor::with_pointer(Arc::new(snapshot), pointer, 16);
        let clone = cursor.clone();
        let mut child = cursor.sub_cursor(8).unwrap();
        drop(cursor);
        drop(clone);
        assert!(pool.lock().unwrap().buffers.is_empty());
        let mut other = CursorSnapshot::from_pool(16, Some(Arc::clone(&pool)));
        assert_ne!(other.bytes.as_mut_ptr(), pointer.as_ptr::<u8>());
        let mut read = [0; 8];
        child.read_bytes(&mut read).unwrap();
        assert_eq!(read, [7; 8]);
        drop(child);
        assert_eq!(pool.lock().unwrap().buffers.len(), 1);
    }

    #[test]
    fn concurrent_snapshots_have_independent_bytes() {
        let pool = pool();
        std::thread::scope(|scope| {
            for id in 1..=4 {
                let pool = Arc::clone(&pool);
                scope.spawn(move || {
                    for _ in 0..64 {
                        let mut snapshot = CursorSnapshot::from_pool(128, Some(Arc::clone(&pool)));
                        assert!(snapshot.bytes.iter().all(|&byte| byte == 0));
                        snapshot.bytes.fill(id);
                        std::thread::yield_now();
                        assert!(snapshot.bytes.iter().all(|&byte| byte == id));
                    }
                });
            }
        });
        let pool = pool.lock().unwrap();
        assert!(pool.buffers.len() <= CURSOR_POOL_ENTRIES);
        assert!(pool.bytes <= CURSOR_POOL_BYTES);
    }
}

/// The bounds check every task accessor opens with. A range ending exactly at the task's end is
/// accepted (a copy's is not: [`LocalTask::copy_range`]).
pub(super) fn check_task_range(offset: u64, length: u64, task_length: u64) -> RaiseResult<()> {
    let (end, overflow) = offset.overflowing_add(length);
    if overflow || end > task_length {
        return raise_caught(
            RaiseKind::Range,
            format_args!(
                "{length:#x} bytes at task offset {offset:#x} run past the task's {task_length:#x}"
            ),
        );
    }
    Ok(())
}

/// The checks both setups make before mapping `offset..offset + length` page by page, in order:
/// the range, a page table, the offset's and the length's alignment, a length.
fn check_mappable(offset: u64, length: u64, task_length: u64, has_root: bool) -> RaiseResult<()> {
    check_task_range(offset, length, task_length)?;
    if !has_root {
        return raise_caught(
            RaiseKind::NoPageTable,
            "no page table to map task memory through",
        );
    }
    if offset & PAGE_MASK != 0 {
        return raise_caught(
            RaiseKind::UnalignedOffset,
            format_args!("task offset {offset:#x} to map is not page-aligned"),
        );
    }
    if length & PAGE_MASK != 0 {
        return raise_caught(
            RaiseKind::UnalignedLength,
            format_args!("length {length:#x} to map is not page-aligned"),
        );
    }
    if length == 0 {
        return raise_caught(
            RaiseKind::ZeroLength,
            format_args!("map of 0 bytes at task offset {offset:#x}"),
        );
    }
    Ok(())
}

use crate::tasks::shared::TaskLock;

impl<P: Platform> LocalTask<P> {
    /// The guest's map command. In the blocks setup it maps the range into the host's task
    /// under the task's shared lock, before the completion; in the map setup it only completes.
    pub(super) fn map_memory(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        // With a memory map this succeeds without any check.
        if let Backend::Blocks { mapped } = &self.backend {
            let _lock = self.base.mutex.read();
            self.map_into_host(hold, mapped, offset, length)?;
        }
        done(ExecStatus::OK, None);
        Ok(())
    }

    /// Maps `offset..offset + length` (page-aligned) into the host's task, one run of pages at a
    /// time, then adds it to `mapped`. A refused or raising run leaves `mapped` as it was.
    pub(super) fn map_into_host(
        &self,
        hold: Hold<'_>,
        mapped: &RangeTracker,
        offset: u64,
        length: u64,
    ) -> RaiseResult<()> {
        check_mappable(offset, length, self.length, !self.task_root.is_null())?;
        let task = self.task_info.task;
        let device = self.device.upgrade();
        // With the device gone nothing maps
        let map = |start: u64, read_only: bool, range: PhysRange| -> bool {
            device
                .as_ref()
                .is_some_and(|d| d.map_memory_in_task(task, start, read_only, &[range]))
        };
        let mut range = PhysRange::default();
        let mut range_start: u64 = 0;
        let mut writable = false;
        if let Some(walker) = &self.walker {
            walker.visit_pages_in_range(
                hold,
                OnFault::Return,
                offset,
                length,
                &mut |address, page, page_writable| {
                    let physical = page_address(page);
                    // The run grows only from its start page (so at most 2 pages), and the `{0, 0}`
                    // start matches a read-only first leaf at guest page 1.
                    if physical == range.physical_address.wrapping_add(PAGE)
                        && page_writable == u32::from(writable)
                    {
                        range.physical_length = range.physical_length.wrapping_add(PAGE);
                    } else {
                        // The flushed run takes the next page's writability.
                        if range.physical_length != 0
                            && !map(range_start, page_writable == 0, range)
                        {
                            return raise_caught(
                                RaiseKind::HostMap,
                                format_args!("the host did not map task offset {range_start:#x}"),
                            );
                        }
                        range = PhysRange {
                            physical_address: physical,
                            physical_length: PAGE,
                        };
                        range_start = address;
                        writable = page_writable != 0;
                    }
                    Ok(true)
                },
            )?;
        }
        if range.physical_length != 0 && !map(range_start, !writable, range) {
            return raise_caught(
                RaiseKind::HostMap,
                format_args!("the host did not map task offset {range_start:#x}"),
            );
        }
        mapped.add_to_range(offset, length)
    }

    /// The guest's unmap command, under the task's unique lock: the range check (every setup),
    /// the object list's unmap where the range overlaps it, then in the blocks setup the
    /// tracker (which refuses a zero or unaligned range) and the host, before the completion.
    pub(super) fn unmap_memory(
        &self,
        offset: u64,
        length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        // An unmap over the object list takes the unique lock (the execs read the lists); any
        // other takes the shared one, as the map does: the tracker and the host's task lock their
        // own, and the unique lock made every unmap wait for the other FIFO channels' execs of
        // the task (decstat1). REIMS_VGPU_SHARED_DELETE=0 takes the unique lock for all.
        check_task_range(offset, length, self.length)?;
        let end = offset.wrapping_add(length);
        let overlaps_list = || {
            let (list_offset, list_length) = *self.object_list_range.lock();
            let list_end = list_offset.wrapping_add(u64::from(list_length));
            offset < list_end && list_offset < end
        };
        let t0 = std::time::Instant::now();
        let mut lock = if crate::tasks::shared::shared_delete() {
            TaskLock::Shared(self.base.mutex.read())
        } else {
            TaskLock::Unique(self.base.mutex.write())
        };
        if matches!(lock, TaskLock::Shared(_)) && overlaps_list() {
            drop(lock);
            lock = TaskLock::Unique(self.base.mutex.write());
        }
        let _lock = lock;
        crate::perfstats::UNMAP_LOCK.add(t0.elapsed());
        let t = std::time::Instant::now();
        let (_, _, lists) = self.parts();
        // Under the unique lock the range is stable; under the shared one it did not overlap.
        if matches!(_lock, TaskLock::Unique(_)) && overlaps_list() {
            lists.unmap_object_list();
        }
        crate::perfstats::UNMAP_LIST.add(t.elapsed());
        // The placement list of a local task is empty: `0..0` overlaps nothing
        if let Backend::Blocks { mapped } = &self.backend {
            let t = std::time::Instant::now();
            mapped.remove_from_range(offset, length)?;
            crate::perfstats::UNMAP_TRACKER.add(t.elapsed());
            let t = std::time::Instant::now();
            // With the device gone the host refuses
            let unmapped = self
                .device
                .upgrade()
                .is_some_and(|d| d.unmap_memory_in_task(self.task_info.task, offset, length));
            crate::perfstats::UNMAP_HOST.add(t.elapsed());
            if !unmapped {
                return raise_caught(
                    RaiseKind::HostUnmap,
                    format_args!(
                        "the host did not unmap {length:#x} bytes at task offset {offset:#x}"
                    ),
                );
            }
        }
        done(ExecStatus::OK, None);
        Ok(())
    }

    /// A mapping of the task's pages `offset..offset + length` from the device's memory map, one
    /// physical range per run of pages, read-only if any page is. `None` without a memory map,
    /// where the map refuses the ranges, or where `need_writable` meets a read-only page.
    pub(super) fn virtual_mapping(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<Option<Arc<Mapping<P>>>> {
        let (ranges, writable) = self.physical_ranges(hold, offset, length)?;
        if need_writable && !writable {
            return Ok(None);
        }
        Ok(self.mapping_over(&ranges, length, writable))
    }

    /// A mapping of `length` bytes over `ranges`, which a walk of the task's range found
    /// ([`Self::physical_ranges`]).
    fn mapping_over(
        &self,
        ranges: &[PhysRange],
        length: u64,
        writable: bool,
    ) -> Option<Arc<Mapping<P>>> {
        let memory_map = self.device.upgrade().and_then(|d| d.memory_map());
        memory_map.and_then(|m| m.new_range_mapping(ranges, length, u32::from(!writable)))
    }

    /// The physical runs the task's page table names for `offset..offset + length`, and whether
    /// every leaf is writable: a walk only, nothing mapped.
    pub(super) fn physical_ranges(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<(Vec<PhysRange>, bool)> {
        check_mappable(offset, length, self.length, !self.task_root.is_null())?;
        let mut ranges: Vec<PhysRange> = Vec::new();
        let mut writable = true;
        let mut range = PhysRange::default();
        if let Some(walker) = &self.walker {
            walker.visit_pages_in_range(
                hold,
                OnFault::Return,
                offset,
                length,
                &mut |_address, page, page_writable| {
                    let physical = page_address(page);
                    // One read-only leaf makes the whole mapping read-only.
                    if page_writable == 0 {
                        writable = false;
                    }
                    // Runs are cut at 2 pages; a first leaf at guest page 1 extends the empty run into
                    // one at physical 0.
                    if physical == range.physical_address.wrapping_add(PAGE) {
                        range.physical_length = range.physical_length.wrapping_add(PAGE);
                    } else {
                        if range.physical_length != 0 {
                            ranges.push(range);
                        }
                        range = PhysRange {
                            physical_address: physical,
                            physical_length: PAGE,
                        };
                    }
                    Ok(true)
                },
            )?;
        }
        if range.physical_length != 0 {
            ranges.push(range);
        }
        Ok((ranges, writable))
    }

    /// Resolves the host virtual address for each guest virtual page in `offset..offset + length`.
    pub(super) fn resolve_pages(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<crate::guest::GuestPtr>>> {
        check_mappable(offset, length, self.length, !self.task_root.is_null())?;
        let Some(walker) = &self.walker else {
            return Ok(None);
        };
        // DIAG(walker-root): the walker took the root and depth once, at the task's making; a
        // root page that names another since is written (each time it differs).
        if !self.task_root.is_null() {
            let depth_now = self.task_root.read_u32(
                super::ROOT_PAGE_TABLE_DEPTH,
                crate::tasks::DeathKind::NullTaskRoot,
            );
            let root_now = u64::from(self.task_root.read_u32(
                super::ROOT_PAGE_TABLE_ROOT,
                crate::tasks::DeathKind::NullTaskRoot,
            ))
            .wrapping_shl(crate::guest::records::APV_PAGE_SHIFT);
            let (root, depth) = walker.root_and_depth();
            if root_now != root || depth_now != depth {
                log::warn!(
                    "task {}: page table moved: walker root {root:#x} depth {depth}, task root now names {root_now:#x} depth {depth_now}",
                    self.task_id
                );
            }
        }
        let Some(device) = self.device.upgrade() else {
            return Ok(None);
        };
        let Some(memory_map) = device.memory_map() else {
            return Ok(None);
        };
        let mut pages = Vec::new();
        let mut ok = true;
        walker.visit_pages_in_range(
            hold,
            OnFault::Return,
            offset,
            length,
            &mut |_address, page, _writable| {
                let physical = page_address(page);
                let host_va = memory_map.virtual_address_for_physical(physical, PAGE);
                if host_va.is_null() {
                    log::warn!(
                        "task {}: page {page:#x} phys {physical:#x} host_va is NULL",
                        self.task_id
                    );
                    ok = false;
                    return Ok(false);
                }
                pages.push(crate::guest::GuestPtr::new(host_va));
                Ok(true)
            },
        )?;
        if !ok {
            return Ok(None);
        }
        Ok(Some(pages))
    }

    /// The host address of task memory at `offset`, mapped for `length` bytes. In the map setup
    /// the covering pages get a fresh mapping, which comes back with the address (the caller
    /// holds it while it uses the address); a writability check that then raises carries it
    /// on with the raise, so it is unmapped where the raise ends, after the handler.
    /// In the blocks setup the range is mapped into the host's task first if the tracker says
    /// it is not; the host's mapping is not checked for writability (a read-only leaf is written
    /// through).
    pub(super) fn mapped_address(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<MappedAddress<P>> {
        check_task_range(offset, length, self.length)?;
        let Backend::Blocks { mapped } = &self.backend else {
            let start = offset & !PAGE_MASK;
            let end = offset.wrapping_add(length).wrapping_add(PAGE_MASK) & !PAGE_MASK;
            let Some(mapping) =
                self.virtual_mapping(hold, start, end.wrapping_sub(start), need_writable)?
            else {
                return raise_caught(
                    RaiseKind::Mapping,
                    format_args!("no mapping of {length:#x} bytes at task offset {offset:#x}"),
                );
            };
            if need_writable && !mapping.writable() {
                // The raise carries the mapping: its unmap comes where the raise ends, after
                // the handler's fault.
                return raise_carrying(
                    RaiseKind::ReadOnlyMapping,
                    format_args!("the mapping at task offset {offset:#x} is read-only"),
                    mapping,
                );
            }
            return Ok(MappedAddress {
                address: mapping.virtual_address().wrapping_add(offset & PAGE_MASK),
                mapping: Some(mapping),
            });
        };
        // Once the tracker says mapped, the host keeps the old mapping.
        if !mapped.test_range(offset, length)? {
            mapped.visit_unmapped_ranges(offset, length, &mut |range_offset, range_length| {
                self.map_into_host(hold, mapped, range_offset, range_length)?;
                Ok(true)
            })?;
        }
        Ok(MappedAddress::unmapped(
            self.task_info.base_address.wrapping_add(offset),
        ))
    }

    /// [`Self::buffer_for_page`] for a transfer buffer, in the map setup kept by the physical
    /// pages it lies over ([`TransferCache`]): a new texture over pages an earlier one used (a
    /// surface pool's next frame, under a new task address) takes the buffer already made, whose
    /// import of guest memory costs milliseconds for a large frame. Keyed by the physical runs
    /// the page walk finds now, never by the task address, so a page the guest moved is a new
    /// key. `REIMS_VGPU_TRANSFER_CACHE=0` makes every one afresh.
    pub(super) fn transfer_buffer_for_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<P::Buffer> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON
            .get_or_init(|| std::env::var("REIMS_VGPU_TRANSFER_CACHE").map_or(true, |v| v != "0"));
        if !on || matches!(self.backend, Backend::Blocks { .. }) {
            return self.buffer_for_page(hold, page, length);
        }
        let (ranges, writable) = self.physical_ranges(hold, page_address(page), length)?;
        if let Some(buffer) = self.transfer_cache.lock().get(&ranges, length, writable) {
            crate::perfstats::TRANSFER_HIT.add(std::time::Duration::ZERO);
            return Ok(buffer);
        }
        let buffer = self.buffer_for_page(hold, page, length)?;
        self.transfer_cache
            .lock()
            .insert(ranges, length, writable, buffer.clone_ref());
        Ok(buffer)
    }

    /// A no-copy buffer over the task's pages from `page`, `length` bytes. In the blocks setup
    /// it wraps the host task's memory, with no deallocator; in the map setup a fresh mapping,
    /// which the buffer's deallocator owns (and which is kept for good when Metal refuses the
    /// buffer: its address space stays taken).
    pub(super) fn buffer_for_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<P::Buffer> {
        let device = self.base.mtl_device.as_deref();
        let buffer = if let Backend::Blocks { .. } = self.backend {
            // The blocks setup: the host's task maps it, and no mapping comes back.
            let pointer = self
                .mapped_address(hold, page_address(page), length, false)?
                .address;
            if pointer.is_null() {
                return raise_caught(
                    RaiseKind::NullAddress,
                    format_args!("no host address for task page {page:#x}"),
                );
            }
            #[allow(unsafe_code)] // architecture 17.3: the no-copy buffer over host memory
            // SAFETY: `pointer..pointer+length` is the host task's own mapping of the task range
            // (the host's `map_memory` callback's), which the host keeps until the guest unmaps
            // it; Metal gets the same memory with no deallocator.
            let bytes = unsafe { crate::platform::HostRange::new(pointer, length) };
            // Trace `tbuf`: the task page, the length, the host address, 0 (the blocks setup) and
            // the guest's task id.
            let task = u64::from(self.task_id);
            crate::trace::event(
                "tbuf",
                &[u64::from(page), length, pointer.addr().to_u64(), 0, task],
            );
            match device {
                Some(d) => d.new_buffer_over_host_memory(OnFault::Return, bytes, 0, None)?,
                None => None,
            }
        } else {
            // DIAG(bind-steps): see the log below.
            let t0 = std::time::Instant::now();
            let Some(mapping) = self.virtual_mapping(hold, page_address(page), length, false)?
            else {
                return raise_caught(
                    RaiseKind::NullAddress,
                    format_args!("no host address for task page {page:#x}"),
                );
            };
            let t_map = t0.elapsed();
            let pointer = mapping.virtual_address();
            let owned = Arc::clone(&mapping);
            let deallocator = Box::new(move || drop(owned));
            #[allow(unsafe_code)] // architecture 17.3: the no-copy buffer over host memory
            // SAFETY: `pointer..pointer+length` is `mapping`'s VM range (`length` is the
            // mapping's own, page-aligned: `virtual_mapping` refused anything else), and the
            // deallocator owns the mapping, so the range stays mapped until Metal runs it.
            let bytes = unsafe { crate::platform::HostRange::new(pointer, length) };
            // Trace `tbuf`: the task page, the length, the host address, 1 (the map setup) and
            // the guest's task id.
            let task = u64::from(self.task_id);
            crate::trace::event(
                "tbuf",
                &[u64::from(page), length, pointer.addr().to_u64(), 1, task],
            );
            let buffer = match device {
                Some(d) => {
                    d.new_buffer_over_host_memory(OnFault::Return, bytes, 0, Some(deallocator))
                }
                None => Ok(None),
            };
            crate::perfstats::BIND_MAP.add(t_map);
            crate::perfstats::BIND_BUFFER.add(t0.elapsed() - t_map);
            if length >= 1 << 20 && t0.elapsed().as_secs_f64() >= 0.0003 {
                log::debug!(
                    target: "reims_vgpu_core::tasks::local::exec",
                    "bind steps: mapping={:.2} buffer={:.2} ms for {length:#x} bytes",
                    t_map.as_secs_f64() * 1000.0,
                    (t0.elapsed() - t_map).as_secs_f64() * 1000.0
                );
            }
            match buffer {
                // The deallocator holds the mapping now
                Ok(Some(buffer)) => {
                    drop(mapping);
                    Some(buffer)
                }
                // Refused: the mapping's range stays taken for good, and (under pressure on the
                // host, measured while a host app exhausted RAM) the guest's own WindowServer
                // got no buffer. One retry over a private copy of the pages, which needs no
                // import of the guest's RAM, before the exec faults.
                // REIMS_VGPU_NOBUF_RETRY=0 turns the retry off.
                // keep_forever is NOT changed: a refused import may still hold the range in the
                // runtime's address bookkeeping (metal_space), so unmapping it could alias a
                // later buffer. A refused mapping is leaked (address space only, not RAM for
                // borrowed ranges); with the retry the leak happens once per refusal, not per exec.
                Ok(None) => {
                    keep_forever(mapping, Kept::AddressSpace);
                    self.retry_buffer_over_copy(hold, device, page, length)
                }
                Err(thrown) => {
                    keep_forever(mapping, Kept::AddressSpace);
                    return Err(thrown);
                }
            }
        };
        match buffer {
            Some(buffer) => Ok(buffer),
            None => raise_caught(
                RaiseKind::Buffer,
                format_args!("Metal made no buffer over task page {page:#x}"),
            ),
        }
    }

    /// The one retry of a refused no-copy buffer: the pages copied into a private host buffer
    /// ([`MemoryMap::new_range_mapping_copy`]) and a no-copy buffer over that. `None` when the
    /// retry is off, finds no pages, or Metal refuses again.
    fn retry_buffer_over_copy(
        &self,
        hold: Hold<'_>,
        device: Option<&P::Device>,
        page: u32,
        length: u64,
    ) -> Option<P::Buffer> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on =
            *ON.get_or_init(|| std::env::var("REIMS_VGPU_NOBUF_RETRY").map_or(true, |v| v != "0"));
        if !on {
            return None;
        }
        let device = device?;
        let (ranges, writable) = self
            .physical_ranges(hold, page_address(page), length)
            .ok()?;
        let memory_map = self.device.upgrade().and_then(|d| d.memory_map())?;
        let mapping = memory_map.new_range_mapping_copy(&ranges, length, u32::from(!writable))?;
        let pointer = mapping.virtual_address();
        let owned = Arc::clone(&mapping);
        let deallocator = Box::new(move || drop(owned));
        #[allow(unsafe_code)] // architecture 17.3: the no-copy buffer over host memory
        // SAFETY: `pointer..pointer+length` is `mapping`'s VM range, and the deallocator owns
        // the mapping, so the range stays mapped until Metal runs it.
        let bytes = unsafe { crate::platform::HostRange::new(pointer, length) };
        match device.new_buffer_over_host_memory(OnFault::Return, bytes, 0, Some(deallocator)) {
            Ok(Some(buffer)) => {
                drop(mapping);
                log::warn!(
                    "task {}: no-copy buffer over task page {page:#x} ({length:#x} bytes) was refused; the retry over a copy succeeded",
                    self.task_id
                );
                Some(buffer)
            }
            _ => {
                keep_forever(mapping, Kept::AddressSpace);
                None
            }
        }
    }

    /// The range check of a copy; returns `offset + length`. Its raise is as `on` says: a copy
    /// is reached both under a handler and with none (a display's colour-space read).
    ///
    /// `end >= length` is refused: a copy ending exactly at the task's end fails (testing.md
    /// 7.1).
    fn copy_range(&self, on: OnFault, offset: u64, length: u64) -> RaiseResult<u64> {
        let (end, overflow) = offset.overflowing_add(length);
        if overflow || end >= self.length {
            return raise_on(
                on,
                RaiseKind::CopyRange,
                format_args!(
                    "copy of {length:#x} bytes at task offset {offset:#x} reaches the task's end, \
                     {:#x}",
                    self.length
                ),
            );
        }
        Ok(end)
    }

    /// Reads the task's pages from `offset` to `end` into `dst` through the device, page by
    /// page, after the range check. Its raises are as `on` says.
    fn copy_pages(
        &self,
        hold: Hold<'_>,
        on: OnFault,
        offset: u64,
        end: u64,
        dst: &mut [u8],
    ) -> RaiseResult<()> {
        let page_offset = offset & PAGE_MASK;
        let start = offset & !PAGE_MASK;
        let size = (end.wrapping_add(PAGE_MASK) & !PAGE_MASK).wrapping_sub(start);
        let mut remaining = dst.len().to_u64();
        let mut current: usize = 0;
        let device = self.device.upgrade();
        let Some(walker) = &self.walker else {
            return Ok(());
        };
        walker.visit_pages_in_range(hold, on, start, size, &mut |address, page, _flags| {
            let mut physical = page_address(page);
            let page_end = physical.wrapping_add(PAGE);
            if address == start {
                physical = physical.wrapping_add(page_offset);
            }
            let current_length = page_end.wrapping_sub(physical).min(remaining);
            let to = current.wrapping_add(current_length.to_usize());
            // `current + remaining` is `dst`'s length throughout, so the part is inside it
            let into = dst.get_mut(current..to).unwrap_or_default();
            let read = match &device {
                Some(d) => d.read_from_physical_address(on, physical, into)?,
                None => false,
            };
            // A failed read is silent: the position does not advance, and the next page is read
            // into it.
            if read {
                current = to;
                remaining = remaining.wrapping_sub(current_length);
            }
            // Ignored by the walker
            Ok(read)
        })
    }

    /// Copies task memory at `offset` into `dst`. Its raises are as `on` says.
    pub(super) fn copy(
        &self,
        hold: Hold<'_>,
        on: OnFault,
        offset: u64,
        dst: &mut [u8],
    ) -> RaiseResult<()> {
        let end = self.copy_range(on, offset, dst.len().to_u64())?;
        self.copy_pages(hold, on, offset, end, dst)
    }

    /// A fresh buffer of task memory at `offset`, `length` bytes (the core's reads). Its raises
    /// are as `on` says.
    pub(super) fn data(
        &self,
        hold: Hold<'_>,
        on: OnFault,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<u8>>> {
        self.base
            .data_from_virtual_offset(on, offset, length, |offset, dst| {
                self.copy(hold, on, offset, dst)
            })
    }

    /// A read-only snapshot of task memory at `offset`, copied now; `None` where the copy
    /// refuses or raises (no log), or where the length is past the largest task (32 GiB, a log),
    /// as it was where the host could not allocate the buffer. The buffer is made only once the
    /// range check passes, so a range the copy refuses touches no memory.
    pub(super) fn cursor(&self, hold: Hold<'_>, offset: u64, length: u64) -> Option<Cursor> {
        if length > MAX_LOCAL_TASK_LENGTH {
            log::warn!(
                "task {}: no cursor of {length:#x} bytes, longer than any task",
                self.task_id
            );
            return None;
        }
        let n = length.to_usize();
        let copied = (|| -> RaiseResult<_> {
            let end = self.copy_range(OnFault::Return, offset, length)?;
            // The buffer starts zeroed, so where a read fails its bytes read 0 (testing.md 7.5).
            let mut buffer = CursorSnapshot::new(n);
            self.copy_pages(hold, OnFault::Return, offset, end, &mut buffer.bytes)?;
            Ok(buffer)
        })();
        let mut buffer = match copied {
            Ok(buffer) => buffer,
            Err(raised) => {
                hold.land(raised);
                return None;
            }
        };
        let pointer = HostPtr::from_ptr(buffer.bytes.as_mut_ptr());
        let keep: Arc<dyn std::any::Any + Send + Sync> = Arc::new(buffer);
        Some(Cursor::with_pointer(keep, pointer, length))
    }
}

/// A task's transfer buffers by the physical runs they lie over, most recently used last: at
/// most [`TransferCache::ENTRIES`] and [`TransferCache::BYTES`], the least recently used going
/// first. A buffer here keeps its import (and its mapping of the guest's pages) alive.
pub(crate) struct TransferCache<P: Platform> {
    entries: Vec<(Vec<PhysRange>, u64, bool, P::Buffer)>,
    bytes: u64,
}

impl<P: Platform> Default for TransferCache<P> {
    fn default() -> Self {
        TransferCache {
            entries: Vec::new(),
            bytes: 0,
        }
    }
}

impl<P: Platform> TransferCache<P> {
    const ENTRIES: usize = 48;
    const BYTES: u64 = 768 << 20;

    fn same(a: &[PhysRange], b: &[PhysRange]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| {
                x.physical_address == y.physical_address && x.physical_length == y.physical_length
            })
    }

    fn get(&mut self, ranges: &[PhysRange], length: u64, writable: bool) -> Option<P::Buffer> {
        let i = self
            .entries
            .iter()
            .position(|(r, l, w, _)| *l == length && *w == writable && Self::same(r, ranges))?;
        let entry = self.entries.remove(i);
        let buffer = entry.3.clone_ref();
        self.entries.push(entry);
        Some(buffer)
    }

    fn insert(&mut self, ranges: Vec<PhysRange>, length: u64, writable: bool, buffer: P::Buffer) {
        if length > Self::BYTES / 4 {
            return;
        }
        self.bytes = self.bytes.saturating_add(length);
        self.entries.push((ranges, length, writable, buffer));
        while self.entries.len() > Self::ENTRIES || self.bytes > Self::BYTES {
            let (_, l, _, _) = self.entries.remove(0);
            self.bytes = self.bytes.saturating_sub(l);
        }
    }
}
