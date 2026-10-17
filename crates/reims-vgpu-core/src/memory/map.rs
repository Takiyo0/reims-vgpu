//! The memory map: the guest's physical ranges, validated and sorted at init, and the
//! translations and mappings made from them.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::guest::records::APV_PAGE_SIZE;
use crate::host::{HostPtr, PhysRange};
use crate::memory::DeathKind;
use crate::memory::descriptor::MemoryMapDescriptor;
use crate::memory::mapping::{Mapping, Teardown};
use crate::num::{ToU64, ToUsize};
use crate::platform::Platform;
use crate::platform::os::vm::VmProt;

/// One range of the map: `{start, start + length, length, host pointer}`. The field names are
/// ours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MemoryMapRange {
    /// `physicalAddress`: the first guest physical address.
    pub physical_address: u64,
    /// `physicalEnd`: one past the last.
    pub physical_end: u64,
    /// `physicalLength`.
    pub physical_length: u64,
    /// `virtualAddress`: where `physical_address` is mapped here.
    pub virtual_address: HostPtr,
}

/// One chunk of a discontiguous mapping.
#[derive(Debug, Clone)]
pub struct DiscontiguousChunk {
    /// The guest physical address of this chunk.
    pub cur_phys: u64,
    /// Offset inside the contiguous host allocation.
    pub host_offset: u64,
    /// Number of bytes in this chunk.
    pub chunk_size: usize,
}

/// A contiguous host buffer backing scattered guest physical pages under WHPX.
#[derive(Debug)]
pub struct DiscontiguousBuffer {
    /// Host virtual address of the allocated buffer.
    pub host_addr: HostPtr,
    /// Total buffer length in bytes.
    pub total_length: u64,
    /// The length of the host range under it, `total_length` rounded up to its scratch size
    /// class ([`scratch_length`]): what the teardown gives back.
    pub alloc_length: u64,
    /// The physical chunks copied into this buffer.
    pub chunks: Vec<DiscontiguousChunk>,
}

thread_local! {
    /// Whether a copy fallback made on this thread leaves its buffer unfilled
    /// ([`with_lazy_fallback`]).
    static LAZY_FALLBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with the copy fallback leaving the buffers it makes unfilled: their chunks are
/// planned and listed, but no guest byte is copied. The caller fills every byte before anything
/// reads one: a texture's prepare, which gathers the pages into its transfer buffer on the GPU or
/// copies them with [`MemoryMap::sync_range_from_guest`].
pub fn with_lazy_fallback<T>(f: impl FnOnce() -> T) -> T {
    LAZY_FALLBACK.with(|lazy| {
        let before = lazy.replace(true);
        let made = f();
        lazy.set(before);
        made
    })
}

/// A deferred fill of a transfer buffer: the copy of its guest pages ([`defer_sync`]).
pub type DeferredSync = Box<dyn FnOnce() + Send>;

thread_local! {
    /// The fills deferred on this thread, where [`with_deferred_syncs`] collects them.
    static DEFERRED_SYNCS: std::cell::RefCell<Option<Vec<DeferredSync>>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs `f`, collecting the transfer-buffer fills its prepares defer ([`defer_sync`]) instead of
/// running them in place; the caller runs them ([`run_deferred_syncs`]) before anything reads the
/// buffers (the commit of the command buffer whose copies read them).
pub fn with_deferred_syncs<T>(f: impl FnOnce() -> T) -> (T, Vec<DeferredSync>) {
    // Opt-in (`REIMS_VGPU_PARALLEL_FILLS=1`) while a hang seen with it on (WindowServer stuck
    // after Setup Assistant's language page, 2026-10-16) is unexplained: off, each fill runs in
    // place as before.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("REIMS_VGPU_PARALLEL_FILLS").is_ok_and(|v| v == "1")) {
        return (f(), Vec::new());
    }
    let before = DEFERRED_SYNCS.with(|d| d.replace(Some(Vec::new())));
    let made = f();
    let jobs = DEFERRED_SYNCS.with(|d| d.replace(before)).unwrap_or_default();
    (made, jobs)
}

/// Defers `job` to the enclosing [`with_deferred_syncs`]; runs it now where there is none.
pub fn defer_sync(job: DeferredSync) {
    let job = DEFERRED_SYNCS.with(|d| match d.borrow_mut().as_mut() {
        Some(list) => {
            list.push(job);
            None
        }
        None => Some(job),
    });
    if let Some(job) = job {
        job();
    }
}

/// Runs the deferred fills, independent textures' guest pages, on parallel threads (one each, at
/// most 8), and returns when every one is done.
pub fn run_deferred_syncs(jobs: Vec<DeferredSync>) {
    match jobs.len() {
        0 => {}
        1 => jobs.into_iter().for_each(|job| job()),
        n => {
            let threads = n.min(8);
            let queue = crate::sync::Mutex::new(jobs);
            std::thread::scope(|scope| {
                for _ in 1..threads {
                    scope.spawn(|| while let Some(job) = queue.lock().pop() {
                        job();
                    });
                }
                while let Some(job) = queue.lock().pop() {
                    job();
                }
            });
        }
    }
}

/// The scratch size class of a copy-fallback buffer of `len` bytes: whole MiB from 1 MiB, whole
/// 64 KiB below, so that buffers of nearly one size (a display surface's allocations differ by a
/// few pages) reuse each other's host ranges ([`HostVm::allocate_scratch`]).
pub fn scratch_length(len: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    const GRANULE: u64 = 64 << 10;
    let class = if len >= MIB { MIB } else { GRANULE };
    len.checked_next_multiple_of(class).unwrap_or(len)
}

/// The guest's physical ranges, sorted at init by start and then length, and looked up first match
/// in stored order. Shared as `Arc`; [`MemoryMap::copy`] gives a new map with the same ranges.
/// Mappings it makes tear down through `P`'s `HostVm`.
pub struct MemoryMap<P: Platform> {
    platform: Arc<P>,
    ranges: Vec<MemoryMapRange>,
    pub(crate) this: crate::sync::Mutex<std::sync::Weak<MemoryMap<P>>>,
    pub(crate) discontiguous: Arc<crate::sync::Mutex<Vec<Arc<DiscontiguousBuffer>>>>,
    /// Whether the host refused a remap of the guest's pages: on Windows (WHPX) the guest's RAM
    /// is the host's own `VirtualAlloc` memory, with no section a view can be made of, so every
    /// remap fails after its allocation of a fresh range (a new section, zero-filled: 8.7 ms p50
    /// for 8 MB, measured). Once one is refused there, the remap path is skipped for the fallbacks.
    remap_refused: Arc<std::sync::atomic::AtomicBool>,
}

impl<P: Platform> std::fmt::Debug for MemoryMap<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryMap")
            .field("ranges", &self.ranges)
            .finish_non_exhaustive()
    }
}

/// An address as an integer.
fn addr_u64(p: HostPtr) -> u64 {
    p.addr().to_u64()
}

/// An integer as an address.
fn ptr_u64(x: u64) -> HostPtr {
    HostPtr::from_addr(x.to_usize())
}

impl<P: Platform> MemoryMap<P> {
    /// `mach_vm_remap`s `piece_length` bytes of `map_range`'s host memory at `piece_address` to
    /// `target` (shared, overwrite); only `KERN_SUCCESS` is tested.
    fn map_physical_range(
        &self,
        map_range: &MemoryMapRange,
        piece_address: u64,
        piece_length: u64,
        target: u64,
    ) -> bool {
        // u64 arithmetic, wrapping.
        let source = addr_u64(map_range.virtual_address)
            .wrapping_add(piece_address)
            .wrapping_sub(map_range.physical_address);
        self.platform
            .vm()
            .remap_shared_overwrite(ptr_u64(target), ptr_u64(source), piece_length)
            .is_ok()
    }

    /// The host's map: the checked adds, the sort, the validation and its logs. `None` refuses
    /// it ("Failed to create memory map" is the device's to log). No platform call.
    pub fn new(platform: Arc<P>, descriptor: &MemoryMapDescriptor) -> Option<Arc<Self>> {
        // The descriptor's ranges, in the order the host added them.
        let mut ranges = Vec::with_capacity(descriptor.ranges().len());
        for range in descriptor.ranges() {
            let Some(end) = range.physical_address.checked_add(range.physical_length) else {
                log::warn!(
                    "memory map refused: the range at {:#x} of {:#x} bytes overflows",
                    range.physical_address,
                    range.physical_length
                );
                return None;
            };
            ranges.push(MemoryMapRange {
                physical_address: range.physical_address,
                physical_end: end,
                physical_length: range.physical_length,
                virtual_address: range.host_address,
            });
        }

        // By start, then length: zero-length ranges come before the range with a length that
        // shares their start, whatever order the host listed them in, so the validation below
        // accepts them.
        ranges.sort_by_key(|r| (r.physical_address, r.physical_length));

        let mut end: u64 = 0;
        let mut valid = true;
        for range in &ranges {
            // Only the previous range's end is compared.
            if range.physical_address < end {
                log::warn!(
                    "memory map refused: the range at {:#x} starts before the previous one's end \
                     {end:#x}",
                    range.physical_address
                );
                valid = false;
                break;
            }
            match range.physical_address.checked_add(range.physical_length) {
                Some(e) => end = e,
                None => {
                    log::warn!(
                        "memory map refused: the range at {:#x} of {:#x} bytes overflows",
                        range.physical_address,
                        range.physical_length
                    );
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            return None;
        }

        let map = Arc::new(MemoryMap {
            platform,
            ranges,
            this: crate::sync::Mutex::new(std::sync::Weak::new()),
            discontiguous: Arc::new(crate::sync::Mutex::new(Vec::new())),
            remap_refused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        });
        *map.this.lock() = Arc::downgrade(&map);
        Some(map)
    }

    /// A plain init (no validation): the same ranges.
    pub fn copy(&self) -> Arc<Self> {
        let map = Arc::new(MemoryMap {
            platform: self.platform.clone(),
            ranges: self.ranges.clone(),
            this: crate::sync::Mutex::new(std::sync::Weak::new()),
            discontiguous: Arc::clone(&self.discontiguous),
            remap_refused: Arc::clone(&self.remap_refused),
        });
        *map.this.lock() = Arc::downgrade(&map);
        map
    }

    /// Copies each planned chunk from the guest's RAM into the buffer at `base`: `false` where a
    /// read fails, after logging which.
    #[allow(unsafe_code)]
    fn copy_chunks(&self, base: usize, chunks: &[DiscontiguousChunk], total_length: u64) -> bool {
        let copy = |part: &[DiscontiguousChunk]| -> bool {
            part.iter().all(|c| {
                // SAFETY: the chunk lies in the fallback's own buffer at `base`, which it just
                // allocated for `total_length` bytes; chunks never overlap.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut((base + c.host_offset as usize) as *mut u8, c.chunk_size)
                };
                let read = self.read_bulk(c.cur_phys, dst);
                if !read {
                    log::warn!(
                        "discontiguous copy fallback: failed to read {:#x} bytes from guest phys {:#x}",
                        c.chunk_size,
                        c.cur_phys
                    );
                }
                read
            })
        };
        // One thread: splitting it across 4 (measured) moved the p50 from 1.85 to 1.47 ms and the
        // p90 of a whole bind from 3.5 to 5.0 ms; the copy is bound by memory, not a core.
        let _ = total_length;
        copy(chunks)
    }

    /// Records that the host refused a remap of the guest's pages (Windows only: see
    /// `remap_refused`).
    fn note_remap_refused(&self) {
        if cfg!(windows) && !self.remap_refused.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::info!(
                "memory map: the host refused a remap of the guest's pages; later mappings take the fallbacks directly"
            );
        }
    }

    /// The ranges, in stored (sorted) order.
    pub fn ranges(&self) -> &[MemoryMapRange] {
        &self.ranges
    }

    /// Remaps each request range into the caller's region at `address`, in request order, then
    /// protects the region; pieces remapped before a failure stay.
    pub fn map_memory_at_virtual_address(
        &self,
        address: HostPtr,
        ranges: &[PhysRange],
        read_only: bool,
    ) -> bool {
        let p = &*self.platform;
        let mut mapped: u64 = 0;
        let mut target = addr_u64(address);
        for request in ranges {
            let mut range = *request;
            // Guest physical 0 is refused, even with length 0.
            if range.physical_address == 0 {
                return false;
            }
            let Some(m) = mapped.checked_add(range.physical_length) else {
                log::warn!(
                    "mapping refused: the ranges' total length overflows at the range at {:#x} of \
                     {:#x} bytes",
                    range.physical_address,
                    range.physical_length
                );
                return false;
            };
            mapped = m;
            if range.physical_length == 0 {
                continue;
            }
            let Some(end) = range.physical_address.checked_add(range.physical_length) else {
                log::warn!(
                    "mapping refused: the range at {:#x} of {:#x} bytes overflows",
                    range.physical_address,
                    range.physical_length
                );
                return false;
            };
            for map_range in &self.ranges {
                if range.physical_address < map_range.physical_address
                    || range.physical_address >= map_range.physical_end
                {
                    continue;
                }
                if map_range.physical_end >= end {
                    if !self.map_physical_range(
                        map_range,
                        range.physical_address,
                        range.physical_length,
                        target,
                    ) {
                        log::error!(
                            "mapping failed: {:#x} bytes of guest memory at {:#x} not remapped",
                            range.physical_length,
                            range.physical_address
                        );
                        return false;
                    }
                    target = target.wrapping_add(range.physical_length);
                    range.physical_length = 0;
                    break;
                }
                let length = map_range.physical_end.wrapping_sub(range.physical_address);
                if !self.map_physical_range(map_range, range.physical_address, length, target) {
                    log::error!(
                        "mapping failed: {length:#x} bytes of guest memory at {:#x} not remapped",
                        range.physical_address
                    );
                    return false;
                }
                target = target.wrapping_add(length);
                range.physical_length = range.physical_length.wrapping_sub(length);
                range.physical_address = map_range.physical_end;
            }
            if range.physical_length != 0 {
                log::warn!(
                    "mapping refused: {:#x} bytes from {:#x} are not in guest memory",
                    range.physical_length,
                    range.physical_address
                );
                return false;
            }
        }

        let prot = if read_only {
            VmProt::READ
        } else {
            VmProt::READ | VmProt::WRITE
        };
        if let Err(kr) = p.vm().protect_max(address, mapped, prot) {
            log::error!(
                "mapping failed: protecting {mapped:#x} bytes at {:#x} returned {:#x}",
                addr_u64(address),
                kr.0
            );
            return false;
        }
        true
    }

    /// A mapping of `length` bytes from `physical_address`, as [`MemoryMap::new_range_mapping`]
    /// of the one range. `read_only` is the host's `uint32_t`.
    pub fn new_contiguous_mapping(
        &self,
        physical_address: u64,
        length: u64,
        read_only: u32,
    ) -> Option<Arc<Mapping<P>>> {
        let range = PhysRange {
            physical_address,
            physical_length: length,
        };
        self.new_range_mapping(&[range], length, read_only)
    }

    /// A fresh VM allocation with each request range remapped into it, in request order; every
    /// refusal after the allocation deallocates it, except a failed protect. A length that is not
    /// whole pages dies.
    ///
    /// Public entry point. First tries the alloc+remap path (`remap_range_mapping`), which aliases
    /// each guest range's host pages (section-backed memory, as macOS/Linux give). When that path
    /// fails AND the request is a single contiguous span inside one stored range — as the GPU FIFO
    /// root page and the EFI framebuffer are — it falls back to a BORROWED mapping that points
    /// straight at the guest range's existing host address (`host_address`), with no alias and no
    /// teardown. This is the WHPX case: QEMU guest RAM is anonymous VirtualAlloc with no section to
    /// remap, but its host pointer is valid and shared, which is all the FIFO/framebuffer need.
    #[allow(unsafe_code)]
    pub fn new_range_mapping(
        &self,
        ranges: &[PhysRange],
        total_length: u64,
        read_only: u32,
    ) -> Option<Arc<Mapping<P>>> {
        if let Some(m) = self.remap_range_mapping(ranges, total_length, read_only) {
            return Some(m);
        }
        // Fallback 1: a single contiguous span within one stored range -> borrow its host address.
        if let Some(host_va) = self.contiguous_host_address(ranges, total_length) {
            log::debug!(
                "range mapping: no section to remap; borrowing guest host address {:#x} for \
                 {total_length:#x} bytes (WHPX direct host-VA)",
                host_va.addr().to_u64()
            );
            return Some(Mapping::new(
                host_va,
                total_length,
                read_only == 0,
                Teardown::None,
            ));
        }

        // Fallback 2 (WHPX discontiguous): copy guest pages into a contiguous host buffer.
        if total_length == 0 {
            return Some(Mapping::new(
                HostPtr::NULL,
                0,
                read_only == 0,
                Teardown::None,
            ));
        }
        let vm = self.platform.vm();
        // DIAG(pool-a): the fallback's allocation and copy times, written on its debug line.
        let alloc_start = std::time::Instant::now();
        // Every byte of the buffer is written from the guest below, so a reused scratch range,
        // whose old contents are not zeroed, serves as well as a fresh one.
        let alloc_length = scratch_length(total_length);
        let allocated = vm.allocate_scratch(alloc_length);
        let alloc_ms = alloc_start.elapsed().as_secs_f64() * 1000.0;
        let copy_start = std::time::Instant::now();
        if let Ok(host_addr) = allocated {
            let mut offset = 0u64;
            let mut ok = true;
            let mut chunks = Vec::new();
            for (i, r) in ranges.iter().enumerate() {
                if r.physical_length == 0 {
                    continue;
                }
                if r.physical_address == 0 {
                    log::warn!("discontiguous copy fallback: range[{i}] physical_address is 0");
                    ok = false;
                    break;
                }
                let mut copied = 0u64;
                while copied < r.physical_length {
                    let cur_phys = r.physical_address.wrapping_add(copied);
                    let rem = r.physical_length.wrapping_sub(copied);
                    let chunk = self
                        .ranges
                        .iter()
                        .find(|mr| cur_phys >= mr.physical_address && cur_phys < mr.physical_end)
                        .map(|mr| (mr.physical_end - cur_phys).min(rem))
                        .unwrap_or(0);
                    if chunk == 0 {
                        log::warn!(
                            "discontiguous copy fallback: no map range for guest phys {cur_phys:#x}"
                        );
                        ok = false;
                        break;
                    }
                    // Planned here, copied below, in parallel for a large buffer.
                    chunks.push(DiscontiguousChunk {
                        cur_phys,
                        host_offset: offset + copied,
                        chunk_size: chunk as usize,
                    });
                    copied = copied.wrapping_add(chunk);
                }
                if !ok {
                    break;
                }
                offset = offset.wrapping_add(r.physical_length);
            }
            let lazy = LAZY_FALLBACK.with(std::cell::Cell::get);
            if ok && offset == total_length && !lazy {
                ok = self.copy_chunks(host_addr.addr(), &chunks, total_length);
            }
            if ok && offset == total_length {
                crate::perfstats::COPY_FALLBACK.add(alloc_start.elapsed());
                crate::perfstats::COPY_FALLBACK_BYTES
                    .fetch_add(total_length, std::sync::atomic::Ordering::Relaxed);
                log::debug!(
                    "range mapping: discontiguous copy fallback: allocated {total_length:#x} bytes at host address {:#x} (chunks={}; alloc {alloc_ms:.2} ms, copy {:.2} ms; lazy {lazy})",
                    host_addr.addr().to_u64(),
                    chunks.len(),
                    copy_start.elapsed().as_secs_f64() * 1000.0
                );
                let discontig_buf = Arc::new(DiscontiguousBuffer {
                    host_addr: HostPtr::from_addr(host_addr.addr()),
                    total_length,
                    alloc_length,
                    chunks,
                });
                // DIAG(list-lock): a push that waits 1 ms or more is written, with the list's length.
                let push_start = std::time::Instant::now();
                let listed = {
                    let mut list = self.discontiguous.lock();
                    list.push(Arc::clone(&discontig_buf));
                    list.len()
                };
                let push_ms = push_start.elapsed().as_secs_f64() * 1000.0;
                if push_ms >= 1.0 {
                    log::debug!("discontiguous list push: {push_ms:.2} ms, {listed} buffers listed");
                }
                let discontiguous = Arc::downgrade(&self.discontiguous);
                return Some(Mapping::new(
                    HostPtr::from_addr(host_addr.addr()),
                    total_length,
                    read_only == 0,
                    Teardown::Discontiguous {
                        platform: Arc::clone(&self.platform),
                        buffer: discontig_buf,
                        discontiguous,
                    },
                ));
            } else {
                let _ = vm.free_scratch(host_addr, alloc_length);
            }
        }

        None
    }

    /// The guest's memory behind the copy-fallback buffer that starts at `ptr`, `len` bytes long:
    /// each chunk as (its host address in the guest's RAM, its offset in the buffer, its length).
    /// `None` where no fallback buffer starts there with that length, or a chunk has no host
    /// address.
    pub fn discontiguous_ranges(&self, ptr: HostPtr, len: u64) -> Option<Vec<(HostPtr, u64, u64)>> {
        let addr = ptr.addr().to_u64();
        let buffer = self
            .discontiguous
            .lock()
            .iter()
            .find(|b| b.host_addr.addr().to_u64() == addr && b.total_length == len)
            .cloned()?;
        let mut ranges = Vec::with_capacity(buffer.chunks.len());
        for c in &buffer.chunks {
            let host = self.virtual_address_for_physical(c.cur_phys, c.chunk_size as u64);
            if host.is_null() {
                return None;
            }
            ranges.push((host, c.host_offset, c.chunk_size as u64));
        }
        Some(ranges)
    }

    /// Synchronizes from guest physical RAM any discontiguous buffers that intersect `[ptr, ptr + len)`.
    #[allow(unsafe_code)]
    pub fn sync_range_from_guest(&self, ptr: HostPtr, len: u64) {
        if ptr.is_null() || len == 0 {
            return;
        }
        let addr = ptr.addr().to_u64();
        let end = addr.saturating_add(len);
        // DIAG(pool-a): a sync of 1 ms or more is written, with the buffers it scanned and the
        // bytes it read.
        let start = std::time::Instant::now();
        let mut read_bytes = 0usize;
        let buffers = self.discontiguous.lock().clone();
        for b in &buffers {
            let b_addr = b.host_addr.addr().to_u64();
            let b_end = b_addr.saturating_add(b.total_length);
            if addr < b_end && end > b_addr {
                for chunk in &b.chunks {
                    let chunk_addr = b_addr.wrapping_add(chunk.host_offset);
                    let chunk_end = chunk_addr.saturating_add(chunk.chunk_size as u64);
                    if addr < chunk_end && end > chunk_addr {
                        let dst_slice = unsafe {
                            let dst_ptr =
                                (b.host_addr.addr() + chunk.host_offset as usize) as *mut u8;
                            std::slice::from_raw_parts_mut(dst_ptr, chunk.chunk_size)
                        };
                        self.read_bulk(chunk.cur_phys, dst_slice);
                        read_bytes += chunk.chunk_size;
                    }
                }
            }
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if ms >= 1.0 {
            log::debug!(
                "sync_range_from_guest: {ms:.2} ms, {read_bytes:#x} bytes read, {} buffers scanned, for {len:#x} bytes",
                buffers.len()
            );
        }
    }

    /// Synchronizes to guest physical RAM any discontiguous buffers that intersect `[ptr, ptr + len)`.
    #[allow(unsafe_code)]
    pub fn sync_range_to_guest(&self, ptr: HostPtr, len: u64) {
        if ptr.is_null() || len == 0 {
            return;
        }
        let addr = ptr.addr().to_u64();
        let end = addr.saturating_add(len);
        let buffers = self.discontiguous.lock().clone();
        for b in &buffers {
            let b_addr = b.host_addr.addr().to_u64();
            let b_end = b_addr.saturating_add(b.total_length);
            if addr < b_end && end > b_addr {
                for chunk in &b.chunks {
                    let chunk_addr = b_addr.wrapping_add(chunk.host_offset);
                    let chunk_end = chunk_addr.saturating_add(chunk.chunk_size as u64);
                    if addr < chunk_end && end > chunk_addr {
                        let src_slice = unsafe {
                            let src_ptr =
                                (b.host_addr.addr() + chunk.host_offset as usize) as *const u8;
                            std::slice::from_raw_parts(src_ptr, chunk.chunk_size)
                        };
                        self.write(chunk.cur_phys, src_slice);
                    }
                }
            }
        }
    }

    /// The single contiguous host address covering exactly `[.., total_length)` of `ranges`, when
    /// the request is one contiguous guest span that lies wholly inside one stored map range;
    /// `None` otherwise (scattered ranges, a gap, or a span crossing stored ranges — those need a
    /// real alias and cannot be borrowed). `physical_address == 0` is refused, as elsewhere.
    fn contiguous_host_address(&self, ranges: &[PhysRange], total_length: u64) -> Option<HostPtr> {
        // Collapse the request to one [start, end) span; refuse gaps / out-of-order / zero start.
        let mut start: Option<u64> = None;
        let mut cursor: u64 = 0;
        let mut sum: u64 = 0;
        for (i, r) in ranges.iter().enumerate() {
            if r.physical_length == 0 {
                continue;
            }
            if r.physical_address == 0 {
                log::warn!(
                    "contiguous_host_address: range[{}] physical_address is 0",
                    i
                );
                return None;
            }
            match start {
                None => {
                    start = Some(r.physical_address);
                    cursor = r.physical_address.checked_add(r.physical_length)?;
                }
                Some(_) => {
                    // Must be exactly contiguous with what we have so far.
                    if r.physical_address != cursor {
                        // Debug, not warn: under WHPX nearly every multi-page mapping is
                        // scattered (about 200 a second during an animation), and a line each
                        // on the FIFO thread costs frame time.
                        log::debug!(
                            "contiguous_host_address: discontiguous at range[{}]: {:#x} != cursor {:#x}",
                            i,
                            r.physical_address,
                            cursor
                        );
                        return None;
                    }
                    cursor = cursor.checked_add(r.physical_length)?;
                }
            }
            sum = sum.checked_add(r.physical_length)?;
        }
        let start = start?;
        if sum != total_length {
            log::warn!(
                "contiguous_host_address: sum {:#x} != total_length {:#x}",
                sum,
                total_length
            );
            return None;
        }
        let end = start.checked_add(total_length)?;
        // The whole span must lie inside ONE stored range.
        for map_range in &self.ranges {
            if start >= map_range.physical_address && end <= map_range.physical_end {
                let off = start.wrapping_sub(map_range.physical_address);
                return Some(ptr_u64(
                    addr_u64(map_range.virtual_address).wrapping_add(off),
                ));
            }
        }
        log::warn!(
            "contiguous_host_address: span [{:#x}..{:#x}) not in any stored map range (ranges={})",
            start,
            end,
            self.ranges.len()
        );
        None
    }

    /// A fresh VM allocation with each request range remapped into it, in request order; every
    /// refusal after the allocation deallocates it, except a failed protect. A length that is not
    /// whole pages dies.
    fn remap_range_mapping(
        &self,
        ranges: &[PhysRange],
        total_length: u64,
        read_only: u32,
    ) -> Option<Arc<Mapping<P>>> {
        // Windows: a host that refused one remap of the guest's pages refuses them all.
        if cfg!(windows) && self.remap_refused.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let vm = self.platform.vm();
        // A total length of 0 passes the assert.
        crate::fatal_unless!(
            total_length % const { NonZeroU64::new(APV_PAGE_SIZE).unwrap() } == 0,
            DeathKind::MappingLengthUnaligned
        );
        let Ok(address) = vm.allocate_anywhere(total_length) else {
            return None;
        };
        // The whole allocation freed, the result ignored.
        let free_all = || {
            let _ = vm.deallocate(address, total_length);
        };

        let mut mapped: u64 = 0;
        let mut target = addr_u64(address);
        for request in ranges {
            let mut range = *request;
            // Guest physical 0 is refused, even with length 0.
            if range.physical_address == 0 {
                free_all();
                return None;
            }
            let Some(m) = mapped.checked_add(range.physical_length) else {
                free_all();
                return None;
            };
            mapped = m;
            if mapped > total_length {
                free_all();
                return None;
            }
            if range.physical_length == 0 {
                continue;
            }
            // `end` stays the request's original end
            let Some(end) = range.physical_address.checked_add(range.physical_length) else {
                free_all();
                return None;
            };
            // Pieces split across contiguous map ranges, in stored order.
            for map_range in &self.ranges {
                if range.physical_address < map_range.physical_address
                    || range.physical_address >= map_range.physical_end
                {
                    continue;
                }
                let length = map_range.physical_end.wrapping_sub(range.physical_address);
                if map_range.physical_end < end {
                    if !self.map_physical_range(map_range, range.physical_address, length, target) {
                        self.note_remap_refused();
                        free_all();
                        return None;
                    }
                    target = target.wrapping_add(length);
                    range.physical_length = range.physical_length.wrapping_sub(length);
                    range.physical_address = map_range.physical_end;
                    continue;
                }
                if !self.map_physical_range(
                    map_range,
                    range.physical_address,
                    range.physical_length,
                    target,
                ) {
                    self.note_remap_refused();
                    free_all();
                    return None;
                }
                target = target.wrapping_add(range.physical_length);
                range.physical_length = 0;
                break;
            }
            if range.physical_length != 0 {
                free_all();
                return None;
            }
        }

        if mapped != total_length {
            free_all();
            return None;
        }
        let prot = if read_only != 0 {
            VmProt::READ
        } else {
            VmProt::READ | VmProt::WRITE
        };
        if vm.protect_max(address, total_length, prot).is_err() {
            // Deallocated, then the mapping is made anyway.
            free_all();
        }
        // Writable unless read-only; making the mapping never fails.
        Some(Mapping::new(
            address,
            total_length,
            read_only == 0,
            Teardown::Vm {
                platform: self.platform.clone(),
            },
        ))
    }

    /// [`Self::read`] for a bulk copy (a texture's pages): on x86_64 one `rep movsb` (ERMSB)
    /// per call in place of a volatile load per word, which held a 16 MiB texture's copy of
    /// scattered 4 KiB pages to about 5 GB/s (3 ms). The copy is inline assembly, so the compiler
    /// assumes nothing of the guest's bytes, as with the volatile loads; a byte the guest writes
    /// meanwhile may be either value, as before.
    #[allow(unsafe_code)]
    pub fn read_bulk(&self, physical: u64, dst: &mut [u8]) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            let source = self.virtual_address_for_physical(physical, dst.len().to_u64());
            if source.is_null() {
                return false;
            }
            if !dst.is_empty() {
                // SAFETY: `source` maps `dst.len()` bytes of guest RAM (the map's range, checked
                // above), `dst` is writable for its length, and they do not overlap.
                unsafe {
                    core::arch::asm!(
                        "rep movsb",
                        inout("rsi") source.addr() => _,
                        inout("rdi") dst.as_mut_ptr() => _,
                        inout("rcx") dst.len() => _,
                        options(nostack, preserves_flags)
                    );
                }
            }
            true
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.read(physical, dst)
        }
    }

    /// Copies `dst.len()` bytes from guest physical `physical`; false when no range holds them.
    pub fn read(&self, physical: u64, dst: &mut [u8]) -> bool {
        let source = self.virtual_address_for_physical(physical, dst.len().to_u64());
        if !source.is_null() && !dst.is_empty() {
            // A copy of 0 bytes touches nothing.
            crate::guest::GuestPtr::new(source).copy_to(0, dst, DeathKind::MapReadFault);
        }
        !source.is_null()
    }

    /// Where `length` bytes from guest physical `physical` are mapped here; NULL when no range
    /// holds them.
    pub fn virtual_address_for_physical(&self, physical: u64, length: u64) -> HostPtr {
        let Some(end) = physical.checked_add(length) else {
            log::warn!("{length:#x} bytes from guest address {physical:#x} overflow");
            return HostPtr::NULL;
        };
        // The first containing range decides; no straddling.
        for range in &self.ranges {
            if physical >= range.physical_address && physical < range.physical_end {
                if end > range.physical_end {
                    log::warn!(
                        "{length:#x} bytes from guest address {physical:#x} run past the end of \
                         their memory map range"
                    );
                    return HostPtr::NULL;
                }
                // u64 arithmetic, wrapping.
                return ptr_u64(
                    physical
                        .wrapping_sub(range.physical_address)
                        .wrapping_add(addr_u64(range.virtual_address)),
                );
            }
        }
        HostPtr::NULL
    }

    /// Copies `src` to guest physical `physical`. `src` is never NULL here, so this always
    /// answers true.
    pub fn write(&self, physical: u64, src: &[u8]) -> bool {
        let destination = self.virtual_address_for_physical(physical, src.len().to_u64());
        // `src` is tested, not `destination`: a NULL translation is copied to, which is the
        // process's death when `src` is not empty.
        if !src.is_empty() {
            crate::guest::GuestPtr::new(destination).copy_from(0, src, DeathKind::MapWriteNull);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    /// Scratch classes: whole MiB from 1 MiB, whole 64 KiB below
    #[test]
    fn scratch_length_rounds_to_its_class() {
        assert_eq!(super::scratch_length(0x1000), 0x1_0000);
        assert_eq!(super::scratch_length(0x1_0000), 0x1_0000);
        assert_eq!(super::scratch_length(0xf_f000), 0x10_0000);
        assert_eq!(super::scratch_length(0x7e_9000), 0x80_0000);
        assert_eq!(super::scratch_length(0x7f_8000), 0x80_0000);
        assert_eq!(super::scratch_length(0x80_0000), 0x80_0000);
    }

    use super::*;
    use crate::memory::descriptor::GuestPhysicalRange;
    use crate::platform::os::vm::KernReturn;
    use crate::testkit::calllog::{Arg, Call};
    use crate::testkit::capture::catch_death;
    use crate::testkit::guest_ram::GuestRam;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Map = MemoryMap<FakePlatform>;

    const PAGE: u64 = 0x4000;
    const RW: i32 = 3;
    const RO: i32 = 1;

    fn plat() -> Arc<FakePlatform> {
        Arc::new(FakePlatform::new())
    }

    /// The memory area's lines `logs` took since its last take, in order.
    fn memory_lines(logs: &crate::testkit::logs::Capture) -> Vec<String> {
        logs.take()
            .into_iter()
            .filter(|(_, target, _)| {
                target.starts_with(concat!(env!("CARGO_CRATE_NAME"), "::memory"))
            })
            .map(|(_, _, message)| message)
            .collect()
    }

    fn gpr(pa: u64, len: u64, host: HostPtr) -> GuestPhysicalRange {
        GuestPhysicalRange {
            physical_address: pa,
            physical_length: len,
            host_address: host,
        }
    }

    fn desc(ranges: &[GuestPhysicalRange]) -> MemoryMapDescriptor {
        let mut d = MemoryMapDescriptor::new();
        for r in ranges {
            d.add_range(*r);
        }
        d
    }

    fn fake_host(n: usize) -> HostPtr {
        HostPtr::from_addr(0x1_0000_0000 + n * 0x10_0000)
    }

    /// A map over real fake RAM: one range per `(pa, len)`, adopted by the fake VM.
    fn ram_map(p: &Arc<FakePlatform>, ranges: &[(u64, u64)]) -> (GuestRam, Arc<Map>) {
        let ram = GuestRam::new();
        for (pa, len) in ranges {
            ram.add_range(*pa, *len).unwrap();
        }
        p.fake_vm().adopt(&ram);
        let d = desc(
            &ram.ranges()
                .iter()
                .map(|r| gpr(r.physical_address, r.region.len(), r.host()))
                .collect::<Vec<_>>(),
        );
        let m = Map::new(p.clone(), &d).unwrap();
        (ram, m)
    }

    fn pr(pa: u64, len: u64) -> PhysRange {
        PhysRange {
            physical_address: pa,
            physical_length: len,
        }
    }

    fn vm_calls(p: &FakePlatform) -> Vec<Call> {
        p.call_log().calls_of("mach_vm")
    }

    fn sels(p: &FakePlatform) -> Vec<&'static str> {
        vm_calls(p).iter().map(|c| c.sel).collect()
    }

    fn pattern(n: usize, seed: u8) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
            .collect()
    }

    mod new {
        use super::*;

        #[test]
        fn empty_descriptor_is_an_empty_map() {
            let p = plat();
            let m = Map::new(p.clone(), &MemoryMapDescriptor::new()).unwrap();
            assert!(m.ranges().is_empty());
            assert!(p.call_log().is_empty());
        }

        #[test]
        fn sorts_by_start_and_fills_end() {
            let p = plat();
            let d = desc(&[
                gpr(0x10_0000, 0x8000, fake_host(2)),
                gpr(0x4000, 0x4000, fake_host(0)),
                gpr(0x8000, 0x6000, fake_host(1)),
            ]);
            let m = Map::new(p.clone(), &d).unwrap();
            let got: Vec<_> = m
                .ranges()
                .iter()
                .map(|r| {
                    (
                        r.physical_address,
                        r.physical_end,
                        r.physical_length,
                        r.virtual_address,
                    )
                })
                .collect();
            assert_eq!(
                got,
                vec![
                    (0x4000, 0x8000, 0x4000, fake_host(0)),
                    (0x8000, 0xe000, 0x6000, fake_host(1)),
                    (0x10_0000, 0x10_8000, 0x8000, fake_host(2)),
                ]
            );
            // No platform call.
            assert!(p.call_log().is_empty());
        }

        /// The overflow is refused by the first loop, before the sort, so the validation's own
        /// overflow check is dead code.
        #[test]
        fn overflow_in_descriptor_order_before_the_sort() {
            let p = plat();
            let d = desc(&[
                gpr(0x8000, 0x8000, fake_host(0)),
                gpr(0x4000, 0x8000, fake_host(1)), // would be an intersection after the sort
                gpr(u64::MAX, 1, fake_host(2)),
            ]);
            let logs = crate::testkit::logs::capture();
            assert!(Map::new(p.clone(), &d).is_none());
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "memory map refused: the range at 0xffffffffffffffff of 0x1 bytes overflows"
                        .to_owned()
                ]
            );
        }

        #[test]
        fn overflow_to_exactly_zero_is_refused_but_the_top_byte_is_not() {
            let p = plat();
            let d = desc(&[gpr(u64::MAX - 0x3fff, 0x4000, fake_host(0))]);
            assert!(Map::new(p.clone(), &d).is_none());
            let d = desc(&[gpr(u64::MAX - 0x3fff, 0x3fff, fake_host(0))]);
            assert!(Map::new(p.clone(), &d).is_some());
        }

        #[test]
        fn an_intersection_is_refused_with_a_warning() {
            let p = plat();
            let d = desc(&[
                gpr(0x8000, 0x4000, fake_host(1)),
                gpr(0x4000, 0x8000, fake_host(0)),
            ]);
            let logs = crate::testkit::logs::capture();
            assert!(Map::new(p.clone(), &d).is_none());
            assert_eq!(memory_lines(&logs), vec![INTERSECTION.to_owned()]);
        }

        /// The warning for a range at 0x8000 that starts inside the range before it.
        const INTERSECTION: &str =
            "memory map refused: the range at 0x8000 starts before the previous one's end 0xc000";

        #[test]
        fn adjacent_and_duplicate_empty_ranges_pass() {
            let p = plat();
            let d = desc(&[
                gpr(0x4000, 0x4000, fake_host(0)),
                gpr(0x8000, 0x4000, fake_host(1)),
                gpr(0xc000, 0, HostPtr::NULL),
                gpr(0xc000, 0, HostPtr::NULL),
            ]);
            assert_eq!(Map::new(p.clone(), &d).unwrap().ranges().len(), 4);
        }

        /// Zero-length ranges sort before the range with a length that shares their start, in
        /// either descriptor order, so both pass.
        #[test]
        fn a_zero_length_twin_passes_before_or_after() {
            for d in [
                desc(&[
                    gpr(0x8000, 0, fake_host(0)),
                    gpr(0x8000, 0x4000, fake_host(1)),
                ]),
                desc(&[
                    gpr(0x8000, 0x4000, fake_host(1)),
                    gpr(0x8000, 0, fake_host(0)),
                ]),
            ] {
                let p = plat();
                let m = Map::new(p.clone(), &d).unwrap();
                assert_eq!(m.ranges()[0].physical_length, 0);
                assert_eq!(m.ranges()[1].physical_length, 0x4000);
            }
        }

        /// Two ranges with a length that share a start intersect, the shorter sorted first.
        #[test]
        fn two_ranges_with_a_length_sharing_a_start_intersect() {
            let p = plat();
            let d = desc(&[
                gpr(0x8000, 0x8000, fake_host(1)),
                gpr(0x8000, 0x4000, fake_host(0)),
            ]);
            let logs = crate::testkit::logs::capture();
            assert!(Map::new(p.clone(), &d).is_none());
            assert_eq!(memory_lines(&logs), vec![INTERSECTION.to_owned()]);
        }

        /// Any number of ranges: by start, then length, and ties (equal start and length) in
        /// descriptor order.
        #[test]
        fn many_ranges_sort_by_start_then_length() {
            let p = plat();
            let mut input = Vec::new();
            for i in 0..40u64 {
                // Every start twice, the zero-length twin at odd positions only.
                let start = 0x10_0000 + (39 - i) / 2 * 0x10_0000;
                let len = if i % 2 == 0 { 0 } else { 0x4000 };
                input.push(gpr(start, len, fake_host(i as usize)));
            }
            // Two zero-length ranges more at one start, after its range with a length.
            input.push(gpr(0x10_0000, 0, fake_host(40)));
            input.push(gpr(0x10_0000, 0, fake_host(41)));
            let m = Map::new(p.clone(), &desc(&input)).unwrap();
            let got: Vec<_> = m
                .ranges()
                .iter()
                .map(|r| (r.physical_address, r.physical_length, r.virtual_address))
                .collect();
            let mut want: Vec<_> = input
                .iter()
                .map(|r| (r.physical_address, r.physical_length, r.host_address))
                .collect();
            want.sort_by_key(|&(start, len, _)| (start, len));
            assert_eq!(got, want);
            assert_eq!(
                &got[..3],
                &[
                    (0x10_0000, 0, fake_host(38)),
                    (0x10_0000, 0, fake_host(40)),
                    (0x10_0000, 0, fake_host(41)),
                ]
            );
        }
    }

    #[test]
    fn copy_shares_the_ranges_without_validation() {
        let p = plat();
        let d = desc(&[
            gpr(0x8000, 0x8000, fake_host(0)),
            gpr(0x4000, 0x4000, fake_host(1)),
        ]);
        let m = Map::new(p.clone(), &d).unwrap();
        let c = m.copy();
        assert!(!Arc::ptr_eq(&m, &c));
        assert_eq!(m.ranges(), c.ranges());
        assert!(p.call_log().is_empty());
        assert!(format!("{c:?}").contains("MemoryMap"));
    }

    mod virtual_address_for_physical {
        use super::*;

        fn map(p: &Arc<FakePlatform>) -> Arc<Map> {
            let d = desc(&[
                gpr(0x4000, 0x4000, fake_host(0)),
                gpr(0x8000, 0x4000, fake_host(1)),
                gpr(0x10000, 0, fake_host(2)),
                gpr(0x20000, 0x4000, HostPtr::NULL),
            ]);
            Map::new(p.clone(), &d).unwrap()
        }

        #[test]
        fn translates_inside_a_range() {
            let p = plat();
            let m = map(&p);
            assert_eq!(m.virtual_address_for_physical(0x4000, 0x4000), fake_host(0));
            assert_eq!(
                m.virtual_address_for_physical(0x9234, 0x10),
                fake_host(1).wrapping_add(0x1234)
            );
            // Length 0 is fine, also at the last byte.
            assert_eq!(
                m.virtual_address_for_physical(0xbfff, 0),
                fake_host(1).wrapping_add(0x3fff)
            );
        }

        /// Contiguous ranges, still refused.
        #[test]
        fn a_straddle_is_refused_with_a_log() {
            let p = plat();
            let m = map(&p);
            let logs = crate::testkit::logs::capture();
            assert!(m.virtual_address_for_physical(0x7ff0, 0x20).is_null());
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "0x20 bytes from guest address 0x7ff0 run past the end of their memory map \
                     range"
                        .to_owned()
                ]
            );
        }

        #[test]
        fn overflow_is_refused_with_a_log() {
            let p = plat();
            let m = map(&p);
            let logs = crate::testkit::logs::capture();
            assert!(m.virtual_address_for_physical(0x4000, u64::MAX).is_null());
            assert_eq!(
                memory_lines(&logs),
                vec!["0xffffffffffffffff bytes from guest address 0x4000 overflow".to_owned()]
            );
        }

        #[test]
        fn outside_every_range_and_empty_ranges_are_null_silently() {
            let p = plat();
            let m = map(&p);
            let logs = crate::testkit::logs::capture();
            assert!(m.virtual_address_for_physical(0x0, 1).is_null());
            assert!(m.virtual_address_for_physical(0xc000, 1).is_null());
            assert!(m.virtual_address_for_physical(0x10000, 0).is_null());
            assert!(memory_lines(&logs).is_empty());
        }

        #[test]
        fn a_null_host_range_gives_near_null_pointers() {
            let p = plat();
            let m = map(&p);
            assert!(m.virtual_address_for_physical(0x20000, 4).is_null());
            assert_eq!(
                m.virtual_address_for_physical(0x20010, 4),
                HostPtr::from_addr(0x10)
            );
        }
    }

    mod read {
        use super::*;

        #[test]
        fn copies_guest_bytes() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let mark = p.call_log().mark();
            let bytes = pattern(0x40, 3);
            assert!(ram.write(0x5000, &bytes));
            let mut dst = vec![0u8; 0x40];
            assert!(m.read(0x5000, &mut dst));
            assert_eq!(dst, bytes);
            assert!(p.call_log().since(mark).is_empty());
        }

        #[test]
        fn refusal_leaves_dst_alone() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let mut dst = vec![0xaau8; 8];
            let logs = crate::testkit::logs::capture();
            assert!(!m.read(0x100_0000, &mut dst));
            assert!(!m.read(0x7ffc, &mut dst));
            assert_eq!(dst, vec![0xaa; 8]);
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "0x8 bytes from guest address 0x7ffc run past the end of their memory map \
                     range"
                        .to_owned()
                ]
            );
        }

        #[test]
        fn an_empty_read_inside_a_range_is_yes() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            assert!(m.read(0x4000, &mut []));
            assert!(!m.read(0x1_0000, &mut []));
        }

        #[test]
        fn a_near_null_source_faults_at_the_memcpy() {
            let p = plat();
            let m = Map::new(p, &desc(&[gpr(0x4000, 0x4000, HostPtr::NULL)])).unwrap();
            let e = catch_death(|| {
                let mut dst = [0u8; 4];
                m.read(0x4010, &mut dst)
            });
            assert_eq!(e.unwrap_err(), DeathKind::MapReadFault.site());
        }
    }

    mod write {
        use super::*;
        use crate::raise::death::DeathSite;

        const SITE: DeathSite = DeathKind::MapWriteNull.site();

        #[test]
        fn copies_into_guest_ram() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let bytes = pattern(0x21, 9);
            assert!(m.write(0x9ff0, &bytes));
            assert_eq!(ram.read_vec(0x9ff0, 0x21).unwrap(), bytes);
        }

        /// Outside every range, the copy goes to NULL.
        #[test]
        fn outside_every_range_copies_to_null() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let e = catch_death(|| m.write(0x100_0000, &[1, 2, 3]));
            assert_eq!(e.unwrap_err(), SITE);
        }

        /// A straddle logs, then copies to NULL.
        #[test]
        fn a_straddle_logs_then_copies_to_null() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
            let logs = crate::testkit::logs::capture();
            let e = catch_death(|| m.write(0x7ffe, &[1, 2, 3, 4]));
            assert_eq!(e.unwrap_err(), SITE);
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "0x4 bytes from guest address 0x7ffe run past the end of their memory map \
                     range"
                        .to_owned()
                ]
            );
        }

        /// With nothing to copy, true even outside every range.
        #[test]
        fn an_empty_write_is_yes_anywhere() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            assert!(m.write(0x100_0000, &[]));
            assert!(m.write(0x4000, &[]));
        }
    }

    mod new_range_mapping {
        use super::*;

        #[test]
        fn unaligned_total_asserts_before_any_vm_call() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            // Half a guest page.
            let half = APV_PAGE_SIZE / 2;
            let e = catch_death(|| m.new_range_mapping(&[pr(0x4000, half)], half, 0));
            assert_eq!(e.unwrap_err(), DeathKind::MappingLengthUnaligned.site());
            assert!(vm_calls(&p).is_empty());
        }

        /// Length 0 passes the assert and makes an empty mapping.
        #[test]
        fn zero_total_makes_an_empty_mapping() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let mp = m.new_range_mapping(&[], 0, 0).unwrap();
            assert_eq!(mp.length(), 0);
            assert!(mp.writable());
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_protect"]);
            let calls = vm_calls(&p);
            assert_eq!(
                calls[0].args,
                vec![Arg::U64(0), Arg::Str("ANYWHERE".into())]
            );
            assert_eq!(
                calls[1].args,
                vec![
                    Arg::Ptr(mp.virtual_address().addr()),
                    Arg::U64(0),
                    Arg::Bool(true),
                    Arg::I32(RW)
                ]
            );
        }

        #[test]
        fn allocation_failure_is_nil_and_nothing_else() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            p.script()
                .push("mach_vm_allocate", Step::value(KernReturn(3)));
            let logs = crate::testkit::logs::capture();
            assert!(m.new_range_mapping(&[pr(0x4000, PAGE)], PAGE, 0).is_none());
            assert_eq!(sels(&p), vec!["mach_vm_allocate"]);
            assert!(memory_lines(&logs).is_empty());
        }

        #[test]
        fn maps_shared_guest_ram_and_deallocates_at_last_release() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let bytes = pattern(0x4000, 1);
            assert!(ram.write(0x8000, &bytes));
            let mp = m.new_range_mapping(&[pr(0x8000, PAGE)], PAGE, 0).unwrap();
            let va = mp.virtual_address();
            assert_eq!(mp.length(), PAGE);
            assert!(mp.writable());
            assert_eq!(p.fake_vm().read(va, PAGE).unwrap(), bytes);
            let calls = vm_calls(&p);
            assert_eq!(
                calls.iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec!["mach_vm_allocate", "mach_vm_remap", "mach_vm_protect"]
            );
            let host = ram.host_addr(0x8000).unwrap();
            assert_eq!(
                calls[1].args,
                vec![Arg::Ptr(va.addr()), Arg::Ptr(host.addr()), Arg::U64(PAGE)]
            );
            assert_eq!(
                calls[2].args,
                vec![
                    Arg::Ptr(va.addr()),
                    Arg::U64(PAGE),
                    Arg::Bool(true),
                    Arg::I32(RW)
                ]
            );
            // Shared: a write through the mapping is guest RAM.
            assert!(p.fake_vm().write(va, &[0xee]));
            p.fake_vm().sync();
            assert_eq!(ram.read_vec(0x8000, 1).unwrap(), vec![0xee]);
            let mark = p.call_log().mark();
            drop(mp);
            let after = p.call_log().since(mark);
            assert_eq!(after.len(), 1);
            assert_eq!(after[0].sel, "mach_vm_deallocate");
            assert_eq!(after[0].args, vec![Arg::Ptr(va.addr()), Arg::U64(PAGE)]);
        }

        #[test]
        fn any_nonzero_read_only_is_read_only() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let mp = m.new_range_mapping(&[pr(0x4000, PAGE)], PAGE, 2).unwrap();
            assert!(!mp.writable());
            let c = &vm_calls(&p)[2];
            assert_eq!(c.sel, "mach_vm_protect");
            assert_eq!(c.args[3], Arg::I32(RO));
            assert!(!p.fake_vm().is_writable(mp.virtual_address(), PAGE));
        }

        #[test]
        fn request_ranges_concatenate_in_order_with_zero_lengths_skipped() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x10000)]);
            let a = pattern(0x4000, 1);
            let b = pattern(0x4000, 2);
            assert!(ram.write(0xc000, &a));
            assert!(ram.write(0x4000, &b));
            let mp = m
                .new_range_mapping(
                    &[pr(0xc000, PAGE), pr(0x8000, 0), pr(0x4000, PAGE)],
                    2 * PAGE,
                    0,
                )
                .unwrap();
            let va = mp.virtual_address();
            assert_eq!(p.fake_vm().read(va, PAGE).unwrap(), a);
            assert_eq!(p.fake_vm().read(va.wrapping_add(PAGE), PAGE).unwrap(), b);
            let remaps = p.call_log().calls_to("mach_vm_remap");
            assert_eq!(remaps.len(), 2);
            assert_eq!(remaps[1].args[0], Arg::Ptr(va.addr() + PAGE as usize));
        }

        /// A request over contiguous map ranges is split, not refused.
        #[test]
        fn a_request_over_contiguous_ranges_is_split() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
            let mp = m.new_range_mapping(&[pr(0x6000, 0x4000), pr(0xa000, 0x4000)], 0x8000, 0);
            // 0xa000 + 0x4000 runs past the last range: refused.
            assert!(mp.is_none());
            let mark = p.call_log().mark();
            let bytes = pattern(0x8000, 5);
            assert!(ram.write(0x4000, &bytes[..0x4000]));
            assert!(ram.write(0x8000, &bytes[0x4000..]));
            let mp = m
                .new_range_mapping(&[pr(0x4000, 0x8000)], 0x8000, 0)
                .unwrap();
            let va = mp.virtual_address();
            let remaps: Vec<_> = p
                .call_log()
                .since(mark)
                .into_iter()
                .filter(|c| c.sel == "mach_vm_remap")
                .collect();
            assert_eq!(remaps.len(), 2);
            assert_eq!(
                remaps[0].args,
                vec![
                    Arg::Ptr(va.addr()),
                    Arg::Ptr(ram.host_addr(0x4000).unwrap().addr()),
                    Arg::U64(0x4000)
                ]
            );
            assert_eq!(
                remaps[1].args,
                vec![
                    Arg::Ptr(va.addr() + 0x4000),
                    // `host_addr(0x8000)` would name the first range's end.
                    Arg::Ptr(ram.ranges()[1].host().addr()),
                    Arg::U64(0x4000)
                ]
            );
            assert_eq!(p.fake_vm().read(va, 0x8000).unwrap(), bytes);
        }

        #[test]
        fn a_remainder_with_no_next_range_is_refused_after_the_first_piece() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0xc000, 0x4000)]);
            assert!(
                m.new_range_mapping(&[pr(0x4000, 0x8000)], 0x8000, 0)
                    .is_none()
            );
            assert_eq!(
                sels(&p),
                vec!["mach_vm_allocate", "mach_vm_remap", "mach_vm_deallocate"]
            );
            let c = vm_calls(&p);
            assert_eq!(c[2].args, vec![c[1].args[0].clone(), Arg::U64(0x8000)]);
        }

        /// Guest physical 0, even with length 0.
        #[test]
        fn physical_zero_is_refused_even_empty() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x0, 0x8000)]);
            assert!(m.new_range_mapping(&[pr(0, 0)], 0, 0).is_none());
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_deallocate"]);
            // The same zero-length request elsewhere is skipped.
            assert!(m.new_range_mapping(&[pr(0x4000, 0)], 0, 0).is_some());
        }

        #[test]
        fn physical_zero_is_checked_before_the_total() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            // Both refusals deallocate; the first request range decides which check fires,
            // and neither remaps.
            assert!(m.new_range_mapping(&[pr(0, 0x10_0000)], PAGE, 0).is_none());
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_deallocate"]);
        }

        #[test]
        fn more_than_the_total_is_refused_before_mapping() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            assert!(
                m.new_range_mapping(&[pr(0x4000, PAGE), pr(0x8000, PAGE)], PAGE, 0)
                    .is_none()
            );
            assert_eq!(
                sels(&p),
                vec!["mach_vm_allocate", "mach_vm_remap", "mach_vm_deallocate"]
            );
        }

        #[test]
        fn less_than_the_total_is_refused_after_mapping() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            assert!(
                m.new_range_mapping(&[pr(0x4000, PAGE)], 2 * PAGE, 0)
                    .is_none()
            );
            assert_eq!(
                sels(&p),
                vec!["mach_vm_allocate", "mach_vm_remap", "mach_vm_deallocate"]
            );
            assert_eq!(vm_calls(&p)[2].args[1], Arg::U64(2 * PAGE));
        }

        #[test]
        fn an_end_overflow_is_refused() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            assert!(
                m.new_range_mapping(&[pr(u64::MAX - 0xfff, PAGE)], PAGE, 0)
                    .is_none()
            );
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_deallocate"]);
        }

        #[test]
        fn outside_every_range_is_refused() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            assert!(
                m.new_range_mapping(&[pr(0x10_0000, PAGE)], PAGE, 0)
                    .is_none()
            );
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_deallocate"]);
        }

        #[test]
        fn a_failed_remap_is_refused() {
            for (req, n_ok) in [
                (pr(0x4000, PAGE), 0usize),
                (pr(0x6000, PAGE), 0),
                (pr(0x6000, PAGE), 1),
            ] {
                let p = plat();
                let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
                if n_ok > 0 {
                    p.script()
                        .push_n("mach_vm_remap", n_ok as u64, Step::Default);
                }
                p.script().push("mach_vm_remap", Step::value(KernReturn(1)));
                assert!(m.new_range_mapping(&[req], PAGE, 0).is_none());
                let s = sels(&p);
                assert_eq!(s.first(), Some(&"mach_vm_allocate"));
                assert_eq!(s.last(), Some(&"mach_vm_deallocate"));
                assert_eq!(
                    s.iter().filter(|x| **x == "mach_vm_remap").count(),
                    n_ok + 1
                );
            }
        }

        #[test]
        fn a_null_host_range_fails_the_remap() {
            let p = plat();
            let m = Map::new(p.clone(), &desc(&[gpr(0x4000, 0x4000, HostPtr::NULL)])).unwrap();
            assert!(m.new_range_mapping(&[pr(0x4000, PAGE)], PAGE, 0).is_none());
            let c = vm_calls(&p);
            assert_eq!(c[1].sel, "mach_vm_remap");
            assert_eq!(c[1].args[1], Arg::Ptr(0));
            assert_eq!(c[2].sel, "mach_vm_deallocate");
        }

        /// A failed protect deallocates, then returns the mapping anyway.
        #[test]
        fn a_failed_protect_still_returns_the_mapping() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            p.script()
                .push("mach_vm_protect", Step::value(KernReturn(4)));
            let mp = m.new_range_mapping(&[pr(0x4000, PAGE)], PAGE, 1).unwrap();
            assert!(!mp.writable());
            assert_eq!(
                sels(&p),
                vec![
                    "mach_vm_allocate",
                    "mach_vm_remap",
                    "mach_vm_protect",
                    "mach_vm_deallocate"
                ]
            );
            drop(mp);
            assert_eq!(p.call_log().count("mach_vm_deallocate"), 2);
        }

        #[test]
        fn map_physical_range_computes_the_source_with_wrapping() {
            let p = plat();
            let m = Map::new(p.clone(), &MemoryMapDescriptor::new()).unwrap();
            let r = MemoryMapRange {
                physical_address: 0x8000,
                physical_end: 0xc000,
                physical_length: 0x4000,
                virtual_address: HostPtr::from_addr(0x1000),
            };
            assert!(!m.map_physical_range(&r, 0x4000, 0x10, 0x7000));
            let c = vm_calls(&p);
            assert_eq!(
                c[0].args,
                vec![
                    Arg::Ptr(0x7000),
                    Arg::Ptr(0x1000usize.wrapping_sub(0x4000)),
                    Arg::U64(0x10)
                ]
            );
        }
    }

    #[test]
    fn new_contiguous_mapping_is_one_range() {
        let p = plat();
        let (ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
        let mp = m.new_contiguous_mapping(0x8000, PAGE, 0).unwrap();
        assert!(mp.writable());
        let c = vm_calls(&p);
        assert_eq!(c[0].args[0], Arg::U64(PAGE));
        assert_eq!(
            c[1].args[1],
            Arg::Ptr(ram.host_addr(0x8000).unwrap().addr())
        );
        assert_eq!(c[2].args[3], Arg::I32(RW));
        let mp2 = m.new_contiguous_mapping(0x4000, PAGE, 1).unwrap();
        assert!(!mp2.writable());
        let e = catch_death(|| m.new_contiguous_mapping(0x4000, 1, 0));
        assert_eq!(e.unwrap_err(), DeathKind::MappingLengthUnaligned.site());
    }

    mod map_memory_at_virtual_address {
        use super::*;
        use crate::platform::os::vm::HostVm;

        fn region(p: &FakePlatform, len: u64) -> HostPtr {
            p.fake_vm().unlogged().allocate_anywhere(len).unwrap()
        }

        #[test]
        fn maps_in_order_then_protects_what_was_mapped() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
            let a = pattern(0x8000, 4);
            assert!(ram.write(0x4000, &a[..0x4000]));
            assert!(ram.write(0x8000, &a[0x4000..]));
            let dst = region(&p, 0xc000);
            assert!(m.map_memory_at_virtual_address(
                dst,
                &[pr(0x4000, 0x8000), pr(0x4000, 0)],
                false
            ));
            assert_eq!(p.fake_vm().read(dst, 0x8000).unwrap(), a);
            let c = vm_calls(&p);
            assert_eq!(
                c.iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec!["mach_vm_remap", "mach_vm_remap", "mach_vm_protect"]
            );
            assert_eq!(c[1].args[0], Arg::Ptr(dst.addr() + 0x4000));
            assert_eq!(
                c[2].args,
                vec![
                    Arg::Ptr(dst.addr()),
                    Arg::U64(0x8000),
                    Arg::Bool(true),
                    Arg::I32(RW)
                ]
            );
        }

        #[test]
        fn read_only_protects_read() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            assert!(m.map_memory_at_virtual_address(dst, &[pr(0x4000, PAGE)], true));
            assert_eq!(vm_calls(&p)[1].args[3], Arg::I32(RO));
        }

        /// `newRangeMapping` and `mapMemoryAtVirtualAddress` refuse any request range whose guest
        /// physical address is 0, even a zero-length one
        #[test]
        fn physical_zero_is_refused_silently() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0, 0)], false));
            assert!(vm_calls(&p).is_empty());
            assert!(memory_lines(&logs).is_empty());
        }

        #[test]
        fn an_accumulation_overflow_logs_after_earlier_pieces() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(
                dst,
                &[pr(0x4000, PAGE), pr(0x4000, u64::MAX)],
                false
            ));
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "mapping refused: the ranges' total length overflows at the range at 0x4000 \
                     of 0xffffffffffffffff bytes"
                        .to_owned()
                ]
            );
            // No cleanup of the piece already remapped.
            assert_eq!(sels(&p), vec!["mach_vm_remap"]);
        }

        #[test]
        fn an_end_overflow_logs() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(u64::MAX, 1)], false));
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "mapping refused: the range at 0xffffffffffffffff of 0x1 bytes overflows"
                        .to_owned()
                ]
            );
        }

        #[test]
        fn a_failed_whole_remap_logs() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            p.script().push("mach_vm_remap", Step::value(KernReturn(1)));
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x4000, PAGE)], false));
            assert_eq!(memory_lines(&logs), vec![NOT_REMAPPED.to_owned()]);
            assert_eq!(sels(&p), vec!["mach_vm_remap"]);
        }

        #[test]
        fn a_failed_piece_remap_logs() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
            let dst = region(&p, 2 * PAGE);
            p.script().push("mach_vm_remap", Step::value(KernReturn(1)));
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x4000, 2 * PAGE)], false));
            assert_eq!(memory_lines(&logs), vec![NOT_REMAPPED.to_owned()]);
        }

        /// The error for the first page of guest memory at 0x4000, not remapped.
        const NOT_REMAPPED: &str =
            "mapping failed: 0x4000 bytes of guest memory at 0x4000 not remapped";

        #[test]
        fn an_incomplete_range_logs_and_keeps_its_pieces() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, 2 * PAGE);
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x4000, 2 * PAGE)], false));
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "mapping refused: 0x4000 bytes from 0x8000 are not in guest memory".to_owned()
                ]
            );
            assert_eq!(sels(&p), vec!["mach_vm_remap"]);
            // Outside every range: nothing mapped at all.
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x10_0000, 0x10)], false));
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "mapping refused: 0x10 bytes from 0x100000 are not in guest memory".to_owned()
                ]
            );
        }

        #[test]
        fn a_failed_protect_logs_the_kern_return() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            p.script()
                .push("mach_vm_protect", Step::value(KernReturn(-1)));
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x4000, PAGE)], false));
            // `kr` is an `int`, printed in hex as its bits.
            assert_eq!(
                memory_lines(&logs),
                vec![format!(
                    "mapping failed: protecting 0x4000 bytes at {:#x} returned 0xffffffff",
                    dst.addr()
                )]
            );
        }

        #[test]
        fn no_ranges_protects_zero_bytes() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            assert!(m.map_memory_at_virtual_address(dst, &[], false));
            assert_eq!(vm_calls(&p)[0].args[1], Arg::U64(0));
        }
    }
}
