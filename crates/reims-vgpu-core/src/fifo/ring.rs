
//! The ring cursors: where the thread reads the next command, what it has published back to the
//! guest, and where a rejected command rewinds to.
//!
//! The root's ring is a range of the root page; the guest publishes `written` through the
//! device's register, and the FIFO publishes `read` there. A child's ring is its own pages,
//! mapped at each start from the page list its node names; the guest writes `written` into the
//! node, and the FIFO publishes its read count there.
//!
//! A command's bytes are counted as they are copied and published only by [`Fifo::advance`],
//! after the whole command has been read; until then [`Fifo::rewind`] returns the cursor to the
//! last published point.

#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::{Mutex, MutexGuard};

use crate::fifo::{DeathKind, Fifo, NO_FAULT_OFFSET};
use crate::guest::GuestPtr;
use crate::guest::records::APV_PAGE_SIZE;
use crate::host::PhysRange;
use crate::num::{self, ToU64, ToUsize, Truncate};
use crate::platform::Platform;

/// The child node's fields (the guest's FIFO node).
// name is ours: the node's layout is the guest's; the roles follow what the FIFO does with each.
pub(crate) const NODE_WRITTEN: u64 = 0x0;
pub(crate) const NODE_READ_COUNT: u64 = 0x4;
pub(crate) const NODE_FAULT_OFFSET: u64 = 0x8;
pub(crate) const NODE_STAMP_INDEX: u64 = 0xc;
pub(crate) const NODE_RUNNING: u64 = 0xe;
pub(crate) const NODE_PAGE_LIST_PAGE: u64 = 0x10;

/// The page list's entries: one guest page of `u32` page numbers (4096, 1024 on x86-64).
pub(crate) const PAGE_LIST_ENTRIES: u64 = APV_PAGE_SIZE.wrapping_shr(2);

fn copy_from_ring_pages(pages: &[GuestPtr], offset: u32, mut dst: &mut [u8]) {
    let mut current_offset = offset;
    while !dst.is_empty() {
        let page_idx = (current_offset as u64 / APV_PAGE_SIZE) as usize;
        let page_off = current_offset as u64 % APV_PAGE_SIZE;
        let Some(&page_ptr) = pages.get(page_idx) else {
            break;
        };
        let bytes_in_page = (APV_PAGE_SIZE - page_off) as usize;
        let to_copy = dst.len().min(bytes_in_page);
        let (chunk, rest) = std::mem::take(&mut dst).split_at_mut(to_copy);
        page_ptr.copy_to(page_off, chunk, DeathKind::RingNull);
        current_offset = current_offset.wrapping_add(to_copy as u32);
        dst = rest;
    }
}

/// A FIFO's ring, by kind.
#[derive(Debug)]
pub(crate) enum Ring {
    /// The root FIFO's.
    Root(RootRing),
    /// A child FIFO's.
    Child(ChildRing),
}

/// The root's ring: a range of the root page, and the two registers.
#[derive(Debug)]
pub(crate) struct RootRing {
    /// The offset in the root page the ring starts at, as the device gave it (kept for a
    /// refused root too).
    pub(super) start_offset: u32,
    /// The ring's first byte; NULL for a refused root.
    pub(super) base: GuestPtr,
    /// Where the next byte is read.
    pub(super) read_offset: AtomicU32,
    /// The read offset at the last advance: where a rewind returns.
    pub(super) last_read_offset: AtomicU32,
    /// The guest's `written` register.
    pub(super) written: AtomicU32,
    /// The published read count.
    pub(super) read: AtomicU32,
    /// Bytes copied since the last advance.
    pub(super) pending_advance: AtomicU32,
    pub(super) fault_offset: AtomicU32,
}

/// A child's ring: its node, and its pages while its thread runs.
#[derive(Debug)]
pub(crate) struct ChildRing {
    /// The guest's node.
    pub(super) node: GuestPtr,
    /// The stamp slot, read from the node at init.
    stamp_index: u32,
    /// The mapped ring; NULL while unmapped.
    pub(super) base: Mutex<GuestPtr>,
    /// The resolved physical pages of the ring (for discontiguous physical rings on WHPX).
    pub(super) pages: Mutex<Vec<GuestPtr>>,
    /// Where the next byte is read.
    pub(super) read_offset: AtomicU32,
    /// The read offset at the last advance.
    pub(super) last_read_offset: AtomicU32,
    /// Bytes read, counted as they are copied.
    pub(super) read_count: AtomicU32,
    /// The read count at the last advance.
    pub(super) last_read_count: AtomicU32,
    /// The node's `written` as last loaded.
    pub(super) cache_written: AtomicU32,
    /// The FIFO's own copy of the fault offset (the node has another).
    pub(super) fault_offset: AtomicU32,
}

impl RootRing {
    pub(super) fn new(start_offset: u32, base: GuestPtr, fault_offset: u32) -> RootRing {
        RootRing {
            start_offset,
            base,
            read_offset: AtomicU32::new(0),
            last_read_offset: AtomicU32::new(0),
            written: AtomicU32::new(0),
            read: AtomicU32::new(0),
            pending_advance: AtomicU32::new(0),
            fault_offset: AtomicU32::new(fault_offset),
        }
    }

    pub(super) fn written(&self) -> u32 {
        self.written.load(Ordering::SeqCst)
    }

    pub(super) fn set_written(&self, written: u32) {
        self.written.store(written, Ordering::SeqCst);
    }

    pub(super) fn read(&self) -> u32 {
        self.read.load(Ordering::SeqCst)
    }

    /// The read count, and the cursor at `read` modulo the ring's length (0 for an empty ring,
    /// which only a refused root has).
    pub(super) fn set_read(&self, read: u32, ring_length: u32) {
        self.read.store(read, Ordering::SeqCst);
        let offset = num::rem0(self.read.load(Ordering::SeqCst), ring_length);
        self.read_offset.store(offset, Ordering::Relaxed);
        self.last_read_offset.store(offset, Ordering::Relaxed);
    }
}

impl ChildRing {
    pub(super) fn new(node: GuestPtr, stamp_index: u32) -> ChildRing {
        ChildRing {
            node,
            stamp_index,
            base: Mutex::new(GuestPtr::default()),
            pages: Mutex::new(Vec::new()),
            read_offset: AtomicU32::new(0),
            last_read_offset: AtomicU32::new(0),
            read_count: AtomicU32::new(0),
            last_read_count: AtomicU32::new(0),
            cache_written: AtomicU32::new(0),
            fault_offset: AtomicU32::new(NO_FAULT_OFFSET),
        }
    }

    pub(super) fn stamp_index(&self) -> u32 {
        self.stamp_index
    }

    /// A new child clears the node's read count and fault offset; a restored one takes its
    /// cursors and fault from the node.
    pub(super) fn init_from_node(&self, restoring: bool) {
        let node = self.node;
        let at = DeathKind::NodeNull;
        if !restoring {
            node.store_seqcst_u32(NODE_READ_COUNT, 0, at);
            node.store_seqcst_u32(NODE_FAULT_OFFSET, NO_FAULT_OFFSET, at);
        } else {
            let read_count = node.load_seqcst_u32(NODE_READ_COUNT, at);
            self.read_count.store(read_count, Ordering::Relaxed);
            let written = node.load_seqcst_u32(NODE_WRITTEN, at);
            self.cache_written.store(written, Ordering::Relaxed);
            let fault = node.load_seqcst_u32(NODE_FAULT_OFFSET, at);
            self.fault_offset.store(fault, Ordering::SeqCst);
            self.last_read_count.store(read_count, Ordering::Relaxed);
        }
    }

    /// Marks the node running or not.
    pub(super) fn set_running(&self, running: bool) {
        self.node
            .write_u16(NODE_RUNNING, u16::from(running), DeathKind::NodeNull);
    }
}

impl Ring {
    pub(super) fn fault_offset(&self) -> u32 {
        match self {
            Ring::Root(r) => r.fault_offset.load(Ordering::SeqCst),
            Ring::Child(c) => c.fault_offset.load(Ordering::SeqCst),
        }
    }

    pub(super) fn set_fault_offset(&self, offset: u32) {
        match self {
            Ring::Root(r) => r.fault_offset.store(offset, Ordering::SeqCst),
            Ring::Child(c) => {
                c.fault_offset.store(offset, Ordering::SeqCst);
                // Guest-visible
                c.node
                    .store_seqcst_u32(NODE_FAULT_OFFSET, offset, DeathKind::NodeNull);
            }
        }
    }
}

/// How many bytes one copy takes: what is available, but no more than asked for or than is left
/// before the ring wraps.
fn copy_count(available: u32, wanted: usize, until_wrap: u32) -> u32 {
    let inner = wanted.to_u64().min(u64::from(until_wrap));
    // `inner <= until_wrap`, a u32
    available.min(inner.low_u32())
}

impl<P: Platform> Fifo<P> {
    /// Latches the current read offset as the command's: what a fault of it reports.
    pub(super) fn latch_offset(&self) {
        let offset = match &self.ring {
            Ring::Root(r) => r.read_offset.load(Ordering::Relaxed),
            Ring::Child(c) => c.read_offset.load(Ordering::Relaxed),
        };
        self.command_offset.store(offset, Ordering::Relaxed);
    }

    /// One wait for ring bytes, under `lock`'s guard `g`: a poke, a root's `written` or a
    /// drain wakes it.
    fn wait_for_bytes(&self, g: &mut MutexGuard<'_, ()>) {
        #[cfg(test)]
        self.waiting_for_bytes.fetch_add(1, Ordering::SeqCst);
        self.bytes.wait(g);
        #[cfg(test)]
        self.waiting_for_bytes.fetch_sub(1, Ordering::SeqCst);
    }

    /// Fills `dst` from the ring, waiting for bytes; `false` once the FIFO is quiesced, even
    /// after the last byte was copied.
    pub(super) fn read_bytes(&self, dst: &mut [u8]) -> bool {
        match &self.ring {
            Ring::Root(r) => self.read_root(r, dst),
            Ring::Child(c) => self.read_child(c, dst),
        }
    }

    fn read_root(&self, r: &RootRing, destination: &mut [u8]) -> bool {
        let mut dst = destination;
        while !self.quiesced.load(Ordering::SeqCst) {
            let until_wrap = self
                .ring_length
                .load(Ordering::Relaxed)
                .wrapping_sub(r.read_offset.load(Ordering::Relaxed));
            // The wrap precedes the return of a finished read
            if until_wrap == 0 {
                r.read_offset.store(0, Ordering::Relaxed);
                continue;
            }
            if dst.is_empty() {
                return true;
            }
            // Against the published `read`, not what this command consumed. A `written` moved
            // backwards wraps.
            let available = r
                .written
                .load(Ordering::SeqCst)
                .wrapping_sub(r.read.load(Ordering::SeqCst));
            if available == 0 {
                let mut g = self.lock.lock();
                if !self.quiesced.load(Ordering::SeqCst)
                    && r.written.load(Ordering::SeqCst) == r.read.load(Ordering::SeqCst)
                {
                    // One wait, then the loop runs again
                    self.wait_for_bytes(&mut g);
                }
                continue;
            }
            let count = copy_count(available, dst.len(), until_wrap);
            let offset = r.read_offset.load(Ordering::Relaxed);
            let (head, tail) = std::mem::take(&mut dst).split_at_mut(count.to_usize());
            r.base.copy_to(u64::from(offset), head, DeathKind::RingNull);
            r.pending_advance.store(
                r.pending_advance
                    .load(Ordering::Relaxed)
                    .wrapping_add(count),
                Ordering::Relaxed,
            );
            r.read_offset
                .store(offset.wrapping_add(count), Ordering::Relaxed);
            dst = tail;
        }
        false
    }

    fn read_child(&self, c: &ChildRing, destination: &mut [u8]) -> bool {
        let mut dst = destination;
        // What the node says is written past the read count; the node is reloaded only when the
        // cached value falls short of the request
        let available = |wanted: usize| -> u32 {
            let read_count = c.read_count.load(Ordering::Relaxed);
            let mut count = c
                .cache_written
                .load(Ordering::Relaxed)
                .wrapping_sub(read_count);
            if wanted.to_u64() > u64::from(count) {
                let written = c.node.load_acquire_u32(NODE_WRITTEN, DeathKind::NodeNull);
                c.cache_written.store(written, Ordering::Relaxed);
                count = written.wrapping_sub(read_count);
            }
            count
        };
        while !self.quiesced.load(Ordering::SeqCst) {
            let until_wrap = self
                .ring_length
                .load(Ordering::Relaxed)
                .wrapping_sub(c.read_offset.load(Ordering::Relaxed));
            if until_wrap == 0 {
                c.read_offset.store(0, Ordering::Relaxed);
                continue;
            }
            if dst.is_empty() {
                return true;
            }
            // The cache is reloaded only when too small; a lowered `written` wraps.
            let count = available(dst.len());
            if count == 0 {
                let mut g = self.lock.lock();
                if !self.quiesced.load(Ordering::SeqCst) && available(dst.len()) == 0 {
                    self.wait_for_bytes(&mut g);
                }
                continue;
            }
            let count = copy_count(count, dst.len(), until_wrap);
            let offset = c.read_offset.load(Ordering::Relaxed);
            let (head, tail) = std::mem::take(&mut dst).split_at_mut(count.to_usize());
            let pages = c.pages.lock();
            if !pages.is_empty() {
                copy_from_ring_pages(&pages, offset, head);
            } else {
                let base = *c.base.lock();
                base.copy_to(u64::from(offset), head, DeathKind::RingNull);
            }
            dst = tail;
            c.read_offset
                .store(offset.wrapping_add(count), Ordering::Relaxed);
            c.read_count.store(
                c.read_count.load(Ordering::Relaxed).wrapping_add(count),
                Ordering::Relaxed,
            );
        }
        false
    }

    /// Publishes the command's bytes to the guest: the root's read count, a child's node.
    pub(super) fn advance(&self) {
        match &self.ring {
            Ring::Root(r) => {
                // A load and a store, not an atomic add
                let read = r.read.load(Ordering::SeqCst);
                r.read.store(
                    read.wrapping_add(r.pending_advance.load(Ordering::Relaxed)),
                    Ordering::SeqCst,
                );
                r.pending_advance.store(0, Ordering::Relaxed);
                r.last_read_offset
                    .store(r.read_offset.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            Ring::Child(c) => {
                let read_count = c.read_count.load(Ordering::Relaxed);
                c.node
                    .store_seqcst_u32(NODE_READ_COUNT, read_count, DeathKind::NodeNull);
                c.last_read_offset
                    .store(c.read_offset.load(Ordering::Relaxed), Ordering::Relaxed);
                c.last_read_count.store(read_count, Ordering::Relaxed);
            }
        }
    }

    /// Returns the cursor to the last advance: the unconsumed command is read again.
    pub(super) fn rewind(&self) {
        match &self.ring {
            Ring::Root(r) => {
                r.read_offset.store(
                    r.last_read_offset.load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
                r.pending_advance.store(0, Ordering::Relaxed);
            }
            Ring::Child(c) => {
                c.read_count
                    .store(c.last_read_count.load(Ordering::Relaxed), Ordering::Relaxed);
                c.read_offset.store(
                    c.last_read_offset.load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
            }
        }
    }

    /// Maps the ring for a start. The root's is the device's, always there; a child maps the
    /// page list its node names, then the pages the list names, up to the first zero entry.
    pub(super) fn map_ring(&self) -> bool {
        match &self.ring {
            Ring::Root(_) => true,
            Ring::Child(c) => self.map_child_ring(c),
        }
    }

    fn map_child_ring(&self, c: &ChildRing) -> bool {
        // A gone device maps nothing
        let device = self.device.upgrade();
        // The node's page list is read again at every start.
        let list_page = c.node.read_u32(NODE_PAGE_LIST_PAGE, DeathKind::NodeNull);
        log::info!("FIFO channel {}: mapping child ring, list_page={:#x}", self.stamp_index(), list_page);
        let list_mapping = device
            .as_ref()
            .and_then(|d| d.new_contiguous_mapping(list_page, APV_PAGE_SIZE));
        let Some(list_mapping) = list_mapping else {
            log::warn!("FIFO channel {}: failed to map list_page {:#x}", self.stamp_index(), list_page);
            return false;
        };
        let list = GuestPtr::new(list_mapping.virtual_address());
        let entry =
            |index: u16| list.read_u32(u64::from(index).wrapping_mul(4), DeathKind::PageListNull);
        let mut page_count: u16 = 0;
        while u64::from(page_count) < PAGE_LIST_ENTRIES {
            if entry(page_count) == 0 {
                break;
            }
            page_count = page_count.wrapping_add(1);
        }
        let mut entries_str = String::new();
        for i in 0..page_count {
            use std::fmt::Write as _;
            let _ = write!(entries_str, " {:#x}", entry(i));
        }
        log::info!("FIFO channel {}: list_page={:#x} page_count={} entries:{}", self.stamp_index(), list_page, page_count, entries_str);
        if page_count == 0 {
            drop(list_mapping);
            return false;
        }
        // The page numbers are read from guest memory again
        let ranges: Vec<PhysRange> = (0..page_count)
            .map(|index| PhysRange {
                physical_address: u64::from(entry(index)).wrapping_mul(APV_PAGE_SIZE),
                physical_length: APV_PAGE_SIZE,
            })
            .collect();
        drop(list_mapping);
        let mut resolved_pages = Vec::with_capacity(page_count as usize);
        let mut all_pages_resolved = false;
        if let Some(dev) = device.as_ref() {
            if let Some(map) = dev.memory_map() {
                let mut ok = true;
                for i in 0..page_count {
                    let phys = u64::from(entry(i)).wrapping_mul(APV_PAGE_SIZE);
                    let host_va = map.virtual_address_for_physical(phys, APV_PAGE_SIZE);
                    if host_va.is_null() {
                        log::warn!("FIFO channel {}: page {} phys={:#x} host_va is NULL", self.stamp_index(), i, phys);
                        ok = false;
                        break;
                    }
                    resolved_pages.push(GuestPtr::new(host_va));
                }
                all_pages_resolved = ok;
            }
        }

        let length = u64::from(page_count).wrapping_mul(APV_PAGE_SIZE).low_u32();
        let mapping = device
            .as_ref()
            .and_then(|d| d.new_range_mapping(&ranges, u64::from(length), false));
        drop(ranges);
        let mut base = mapping
            .as_ref()
            .map_or_else(GuestPtr::default, |m| GuestPtr::new(m.virtual_address()));
        if base.is_null() && all_pages_resolved {
            base = resolved_pages.first().copied().unwrap_or_default();
        }
        let mapped = mapping.is_some() || all_pages_resolved;
        // A mapping already here (a second thread's start) is kept for good: the first ring's
        // unmap never comes.
        if let Some(old) = std::mem::replace(&mut *self.mapping.lock(), mapping) {
            crate::raise::keep_forever(old, crate::raise::Kept::UnmapNeverComes);
        }
        // The length is set even when the pages could not be mapped
        self.ring_length.store(length, Ordering::Relaxed);
        *c.base.lock() = base;
        *c.pages.lock() = resolved_pages;
        // From the node's read count, not the FIFO's
        let node_read_count = c.node.load_seqcst_u32(NODE_READ_COUNT, DeathKind::NodeNull);
        let read_offset = num::rem0(node_read_count, length);
        c.read_offset.store(read_offset, Ordering::Relaxed);
        c.last_read_offset.store(read_offset, Ordering::Relaxed);
        log::info!("FIFO channel {}: map_child_ring final mapped={} (all_resolved={})", self.stamp_index(), mapped, all_pages_resolved);
        mapped
    }

    /// Unmaps a child's ring at its thread's exit (the root's is the device's).
    pub(super) fn unmap_ring(&self) {
        if let Ring::Child(c) = &self.ring {
            let old = self.mapping.lock().take();
            drop(old);
            c.pages.lock().clear();
            self.ring_length.store(0, Ordering::Relaxed);
            *c.base.lock() = GuestPtr::default();
        }
    }
}
