//! Guest FIFOs: the root FIFO, whose ring lies in the root page the device's registers name, and
//! the child FIFOs the guest defines on it, each with a ring of its own pages and a node the
//! guest reads.
//!
//! A FIFO is its ring cursors ([`ring`]), its stamp queue ([`stamps`]), and one thread
//! ([`thread`]) that reads commands off the ring ([`wire`]) and runs each ([`commands`]).
//!
//! Ordering contracts, each with its test (`fifo/tests.rs` unless named otherwise):
//! - **Framing.** A header whose length is under 12 bytes or over the ring is skipped with no
//!   stamp, fault or advance (`process::bad_length_skipped`); a barrier count that underflows the
//!   length or passes 64 pushes and faults the header's stamp, having consumed the header only
//!   (`process::underflow_faults_header`, `process::too_many_barriers`).
//! - **Barriers.** The barrier wait's "not yet" (quiesced) leaves the command unconsumed
//!   (`process::barrier_no_leaves_command`); a raise out of it consumes, pushes and faults the
//!   command (`process::barrier_raise_consumes_and_faults`).
//! - **Advance before push.** A command's bytes are published to the guest before its stamp is
//!   pushed, and the push may block (`process::advance_before_blocking_push`).
//! - **Each handler's check order** is its own, and tested per opcode (`commands/tests.rs`).
//! - **One pool and one holder per command**, but none for the deprecated opcodes.
//! - **Drain** waits once for the thread's exit, with no predicate, then drains the stamps, and
//!   hangs on a stamp that never completes (`lifecycle::drain_waits_for_thread_exit`).
//! - **Two threads on one FIFO**: a start over a live thread spawns a second on the same state
//!   (`lifecycle::second_start_second_thread`).
//! - **No handler**: a raise that reaches the exec-indirect task calls or a FIFO's shared-event
//!   creation dies, as no handler is live there. The exec calls run outside the command's
//!   handler, and a FIFO is made where none catches, a child's in the
//!   FIFO queue's block even under a define's handler (`device::fifos`
//!   `tests::a_define_whose_event_raises_dies_under_its_handler`).

pub mod commands;
pub mod ring;
pub mod stamps;
pub mod thread;
pub mod wire;

#[cfg(test)]
mod testkit;
#[cfg(test)]
mod tests;

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::{Condvar, Mutex};

use crate::fifo::ring::{ChildRing, NODE_STAMP_INDEX, Ring, RootRing};
use crate::fifo::stamps::{FifoSink, StampQueue};
use crate::guest::GuestPtr;
use crate::memory::mapping::Mapping;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::raise::RaiseId;
use crate::raise::death::DeathId;
use crate::raise::death::DeathSite;
use crate::seams::links::FifoDeviceLink;

/// The root FIFO's stamp slot.
pub const ROOT_STAMP_INDEX: u32 = 0;
/// No fault: the fault offset of a FIFO that has none.
pub const NO_FAULT_OFFSET: u32 = u32::MAX;

/// What a FIFO command raises. The command's handler catches it (the classes of
/// [`commands`]); its detail is the text the handler's log prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// The payload is shorter than the command's record.
    CommandLength,
    /// The guest's feature level does not have the command.
    Unsupported,
    /// A display or backing command names a task other than the kernel task.
    KernelOnly,
    /// An exec or info command names the kernel task.
    NotKernel,
    /// A count or length in the payload overflows its 32-bit size.
    Overflow,
    /// The device refused to delete the task.
    DeleteTask,
    /// A child FIFO command on a child FIFO.
    NotRoot,
    /// The device refused to create the child FIFO.
    CreateChild,
    /// The device refused to delete the child FIFO.
    DeleteChild,
}

impl RaiseKind {
    /// The dotted id.
    pub const fn name(self) -> &'static str {
        match self {
            RaiseKind::CommandLength => "fifo.command.length",
            RaiseKind::Unsupported => "fifo.command.unsupported",
            RaiseKind::KernelOnly => "fifo.command.kernel-only",
            RaiseKind::NotKernel => "fifo.command.not-kernel",
            RaiseKind::Overflow => "fifo.command.overflow",
            RaiseKind::DeleteTask => "fifo.task.delete-failed",
            RaiseKind::NotRoot => "fifo.child.not-root",
            RaiseKind::CreateChild => "fifo.child.create-failed",
            RaiseKind::DeleteChild => "fifo.child.delete-failed",
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<RaiseKind> for RaiseId {
    fn from(k: RaiseKind) -> RaiseId {
        RaiseId::Fifo(k)
    }
}

/// How a FIFO ends the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// The FIFO's device is gone where a command, or a child's init, uses it.
    DeviceGone,
    /// A child's node is NULL.
    NodeNull,
    /// The ring is NULL where a command's bytes are copied out of it (a child's ring torn down
    /// under a second thread).
    RingNull,
    /// The page list's mapping is NULL.
    PageListNull,
    /// A drain of a FIFO that is not quiesced.
    DrainUnquiesced,
    /// A signal with nothing pending.
    SignalNothingPending,
    /// A second signal of a stamp.
    SignalTwice,
    /// A signal of a faulted stamp.
    SignalFaulted,
    /// A fault with nothing pending.
    FaultNothingPending,
    /// A fault offset of 2^31 or more.
    FaultOffsetRange,
    /// A fault of a signalled stamp.
    FaultSignalled,
    /// A second fault of a stamp.
    FaultTwice,
    /// A drain of a stamp queue that is not quiesced.
    QueueDrainUnquiesced,
}

impl DeathKind {
    /// The dotted id.
    pub const fn name(self) -> &'static str {
        match self {
            DeathKind::DeviceGone => "fifo.device.gone",
            DeathKind::NodeNull => "fifo.node.null",
            DeathKind::RingNull => "fifo.ring.null",
            DeathKind::PageListNull => "fifo.page-list.null",
            DeathKind::DrainUnquiesced => "fifo.drain.unquiesced",
            DeathKind::SignalNothingPending => "fifo.stamps.signal-nothing-pending",
            DeathKind::SignalTwice => "fifo.stamps.signal-twice",
            DeathKind::SignalFaulted => "fifo.stamps.signal-faulted",
            DeathKind::FaultNothingPending => "fifo.stamps.fault-nothing-pending",
            DeathKind::FaultOffsetRange => "fifo.stamps.fault-offset-range",
            DeathKind::FaultSignalled => "fifo.stamps.fault-signalled",
            DeathKind::FaultTwice => "fifo.stamps.fault-twice",
            DeathKind::QueueDrainUnquiesced => "fifo.stamps.drain-unquiesced",
        }
    }

    /// The kind of death: a NULL access, an abort or an assertion.
    const fn class(self) -> crate::raise::death::DeathKind {
        use crate::raise::death::DeathKind as Class;
        match self {
            DeathKind::DeviceGone
            | DeathKind::NodeNull
            | DeathKind::RingNull
            | DeathKind::PageListNull => Class::NullDeref,
            DeathKind::DrainUnquiesced
            | DeathKind::SignalNothingPending
            | DeathKind::SignalTwice
            | DeathKind::SignalFaulted
            | DeathKind::FaultNothingPending
            | DeathKind::FaultOffsetRange
            | DeathKind::FaultSignalled
            | DeathKind::FaultTwice
            | DeathKind::QueueDrainUnquiesced => Class::Assert,
        }
    }

    /// The site the death reports: its class, and its id in place of a file.
    pub const fn site(self) -> DeathSite {
        DeathSite {
            kind: self.class(),
            file: self.name(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> DeathId {
        DeathId::Fifo(k)
    }
}

/// One guest FIFO. The device owns it; its thread holds another reference while it runs.
///
/// Dropping it releases the shared event, then the ring's mapping (a child's, which its thread
/// normally tore down already), then the rest, whose drops have no effects.
pub struct Fifo<P: Platform> {
    /// The device, unretained: it drains its FIFOs before it goes.
    device: FifoDeviceLink<P>,
    ring: Ring,
    stamps: StampQueue,
    /// Guards no data: the waits for ring bytes and for the thread's exit.
    lock: Mutex<()>,
    /// Wakes a thread waiting for ring bytes.
    bytes: Condvar,
    /// Tests: how many threads wait on `bytes`, counted under `lock`. A thread is counted while it
    /// holds `lock`, and gives `lock` up only by starting its wait. Quiesce takes `lock`, so a
    /// quiesce taken once a thread is counted finds that thread waiting, never on its way in.
    #[cfg(test)]
    waiting_for_bytes: AtomicU32,
    /// Wakes a drain waiting for the thread's exit.
    exited: Condvar,
    /// A thread is in the loop.
    running: AtomicBool,
    quiesced: AtomicBool,
    /// The ring's size in bytes: the command buffer's size and the bound on a command's length.
    ring_length: AtomicU32,
    /// The ring offset of the command being run, latched before its header is read.
    command_offset: AtomicU32,
    /// The shared event the exec commands hand their task (made at init, never changed), and
    /// its value (a remote task shares the value's cell).
    event: Option<P::SharedEvent>,
    event_value: Arc<AtomicU64>,
    /// A child's ring mapping, from a start to its thread's exit.
    mapping: Mutex<Option<Arc<Mapping<P>>>>,
    platform: Arc<P>,
}

impl<P: Platform> Drop for Fifo<P> {
    fn drop(&mut self) {
        // The event before the mapping, whose drop may unmap through the host
        drop(self.event.take());
        drop(self.mapping.get_mut().take());
    }
}

impl<P: Platform> Fifo<P> {
    /// What every FIFO's init does: makes its shared event on the device's Metal device, and its
    /// stamp queue, whose sink holds the FIFO weakly with its stamp index as it is now.
    ///
    /// The event's creation has no handler: every FIFO is made where none catches (the host's MMIO
    /// write, a resume, the FIFO queue's block), and a raise in the creation dies at the send.
    fn build(
        platform: &Arc<P>,
        device: FifoDeviceLink<P>,
        last_pending: Arc<AtomicU32>,
        stamp_index: u32,
        ring: Ring,
        ring_length: u32,
    ) -> Arc<Self> {
        // A gone device makes no event
        let event = device
            .upgrade()
            .and_then(|d| d.mtl_device().new_shared_event());
        Arc::new_cyclic(|weak: &Weak<Fifo<P>>| Fifo {
            device,
            ring,
            stamps: StampQueue::new(
                last_pending,
                Box::new(FifoSink {
                    fifo: weak.clone(),
                    stamp_index,
                }),
            ),
            lock: Mutex::new(()),
            bytes: Condvar::new(),
            #[cfg(test)]
            waiting_for_bytes: AtomicU32::new(0),
            exited: Condvar::new(),
            running: AtomicBool::new(false),
            quiesced: AtomicBool::new(false),
            ring_length: AtomicU32::new(ring_length),
            command_offset: AtomicU32::new(0),
            event,
            event_value: Arc::new(AtomicU64::new(0)),
            mapping: Mutex::new(None),
            platform: Arc::clone(platform),
        })
    }

    /// The root FIFO over `length` bytes of the root page at `base`, its ring starting at
    /// `start_offset`; `None` when the ring would be empty.
    pub fn try_new_root_on(
        platform: &Arc<P>,
        device: FifoDeviceLink<P>,
        length: u32,
        base: GuestPtr,
        start_offset: u32,
        last_pending: Arc<AtomicU32>,
    ) -> Option<Arc<Self>> {
        match Self::init_root(platform, device, length, base, start_offset, last_pending) {
            (fifo, true) => Some(fifo),
            // The refused FIFO's last reference goes
            (_, false) => None,
        }
    }

    /// The root init; the flag is false where the ring would be empty, and the FIFO is as the
    /// init left it (no ring, length 0).
    fn init_root(
        platform: &Arc<P>,
        device: FifoDeviceLink<P>,
        length: u32,
        base: GuestPtr,
        start_offset: u32,
        last_pending: Arc<AtomicU32>,
    ) -> (Arc<Self>, bool) {
        let refused = length <= start_offset;
        let (ring_base, ring_length) = if refused {
            (GuestPtr::default(), 0)
        } else {
            (
                base.offset(u64::from(start_offset)),
                length.wrapping_sub(start_offset),
            )
        };
        let fifo = Self::build(
            platform,
            device,
            last_pending,
            ROOT_STAMP_INDEX,
            // An accepted root starts with no fault
            Ring::Root(RootRing::new(
                start_offset,
                ring_base,
                if refused { 0 } else { NO_FAULT_OFFSET },
            )),
            ring_length,
        );
        if refused {
            log::warn!(
                "root FIFO refused: start offset {start_offset:#x} is not below its length \
                 {length:#x}"
            );
            return (fifo, false);
        }
        (fifo, true)
    }

    /// A child FIFO over the node at `node`, which names its stamp index and its ring's page
    /// list. Unless `restoring`, the node's read count and fault offset are cleared; restoring,
    /// the FIFO takes them from the node. `None` for a stamp index at or past the device's FIFO
    /// count, after the node writes. `fifo_index` is unused.
    pub fn new_child_on(
        platform: &Arc<P>,
        device: FifoDeviceLink<P>,
        node: GuestPtr,
        fifo_index: u32,
        restoring: bool,
        last_pending: Arc<AtomicU32>,
    ) -> Option<Arc<Self>> {
        let _ = fifo_index;
        // `last_pending` is whatever cell the device passes.
        // The stamp index is the node's now; later changes are ignored.
        let stamp_index = u32::from(node.read_u16(NODE_STAMP_INDEX, DeathKind::NodeNull));
        let fifo = Self::build(
            platform,
            device.clone(),
            last_pending,
            stamp_index,
            Ring::Child(ChildRing::new(node, stamp_index)),
            0,
        );
        if let Ring::Child(c) = &fifo.ring {
            c.init_from_node(restoring);
        }
        // The device's features; a gone device is a NULL access
        let Some(host) = device.upgrade() else {
            crate::fatal!(DeathKind::DeviceGone)
        };
        if stamp_index >= host.features().load().max_fifo_count {
            // The node writes above stay.
            return None;
        }
        Some(fifo)
    }

    /// Whether this is the root FIFO.
    pub fn is_root_fifo(&self) -> bool {
        matches!(self.ring, Ring::Root(_))
    }

    /// The FIFO's stamp slot: 0 for the root, the node's for a child.
    pub fn stamp_index(&self) -> u32 {
        match &self.ring {
            Ring::Root(_) => ROOT_STAMP_INDEX,
            Ring::Child(c) => c.stamp_index(),
        }
    }

    /// The platform the FIFO logs, traces and starts its threads on.
    pub(crate) fn platform(&self) -> &P {
        &self.platform
    }

    /// Whether the FIFO is quiesced.
    pub fn is_quiesced(&self) -> bool {
        self.quiesced.load(Ordering::SeqCst)
    }

    /// Whether the FIFO's thread is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// The fault offset (a child's own copy, not its node's).
    pub fn fault_offset(&self) -> u32 {
        self.ring.fault_offset()
    }

    /// Sets the fault offset (and a child's node's).
    pub fn set_fault_offset(&self, offset: u32) {
        self.ring.set_fault_offset(offset);
    }

    /// Faults the pending stamp `stamp_value` at `offset`.
    pub fn fault_at_offset(&self, offset: u32, stamp_value: u32) {
        self.stamps.fault_stamp(stamp_value, offset);
    }

    /// Faults the pending stamp `stamp_value` at the current command's offset, read now.
    pub fn fault_at_stamp_value(&self, stamp_value: u32) {
        let offset = self.command_offset.load(Ordering::Relaxed);
        self.stamps.fault_stamp(stamp_value, offset);
    }

    /// Signals the pending stamp `stamp_value`.
    pub fn signal_stamp_value(&self, stamp_value: u32) {
        self.stamps.signal_stamp(stamp_value);
    }

    /// Pushes a pending stamp for command `command_id` (which the slot widens to 32 bits).
    pub fn push_stamp(&self, stamp_value: u32, command_id: u16) {
        self.stamps.push_stamp(stamp_value, u32::from(command_id));
    }

    /// Diagnostics: the pending stamps.
    pub fn describe_pending(&self) -> String {
        self.stamps.describe_pending()
    }

    /// Clears the fault and resumes the stamp queue's faulted head.
    pub fn resume(&self) {
        let _g = self.lock.lock();
        self.set_fault_offset(NO_FAULT_OFFSET);
        self.stamps.resume_from_fault();
    }

    /// The root's `written` register; `None` for a child.
    pub fn root_written(&self) -> Option<u32> {
        match &self.ring {
            Ring::Root(r) => Some(r.written()),
            Ring::Child(_) => None,
        }
    }

    /// Sets the root's `written` register and wakes its thread. Any value is accepted (a
    /// `written` below `read` wraps). A child has no such register: nothing is done.
    pub fn set_root_written(&self, written: u32) {
        if let Ring::Root(r) = &self.ring {
            let _g = self.lock.lock();
            r.set_written(written);
            self.bytes.notify_one();
        }
    }

    /// The root's published read count; `None` for a child.
    pub fn root_read(&self) -> Option<u32> {
        match &self.ring {
            Ring::Root(r) => Some(r.read()),
            Ring::Child(_) => None,
        }
    }

    /// The offset in the root page the root's ring starts at. A child: `None`.
    pub fn root_start_offset(&self) -> Option<u32> {
        match &self.ring {
            Ring::Root(r) => Some(r.start_offset),
            Ring::Child(_) => None,
        }
    }

    /// Positions the root's read count and cursor at `read` (a resume). A child: nothing.
    pub fn set_root_read(&self, read: u32) {
        if let Ring::Root(r) = &self.ring {
            r.set_read(read, self.ring_length.load(Ordering::Relaxed));
        }
    }
}
