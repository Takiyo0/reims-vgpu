//! The child FIFOs (create, delete, resume and wake), the stamps, the faults and the guest
//! interrupt.
//!
//! ## Contract
//!
//! - Create, delete, resume, wake and the root FIFO's resume run their FIFO work on the FIFO
//!   queue, synchronously and on the caller's thread (`tests::create_fifo_block_runs_on_callers_thread`).
//!   During the device's shutdown, when no strong reference is left, the block runs inline.
//! - A delete takes the child out on the FIFO queue, clears its fault bit and finishes its wait,
//!   then stops and drains it, clears a node word, and tells every task, waiting for all their
//!   completions under the task table's lock, holding every task until the wait ends
//!   (`tests::the_delete_holds_every_task_until_the_wait_ends`).
//! - A stamp, a fault and a resume raise the interrupt with the signal lock held across the
//!   host's `raiseInterrupt` (`tests::signal_stamp_bound_is_inclusive`,
//!   `tests::signal_fault_in_fifo_sets_stamp_bit`).
//! - A delete clears the word at `0x410 + id·0x14`, the last word of node `id + 1`: the next
//!   child's word, not its own (`tests::create_then_delete_child_clears_next_node_word`).
//! - A delete clears fault bit `id`, while a fault sets bit `stampIndex`: a child whose stamp
//!   index differs from its id leaves its fault bit set.
//! - A resume logs on every call, and an id at or past `maxFIFOCount` is logged and then used
//!   anyway (`tests::resume_child_fifo_non_faulted_logs_and_reports`).
//! - A resume's `1 << id` is the 64-bit hardware shift (the id mod 64), and an id of 64 or more
//!   names no child FIFO: it clears its aliased fault bit and resumes nothing, as a wake ignores
//!   it (`tests::resume_child_fifo_past_the_table_resumes_nothing`).
//! - A stamp's check is `stamp > maxFIFOCount`, so stamp `maxFIFOCount` passes
//!   (`tests::signal_stamp_bound_is_inclusive`).
//! - A stamp's interrupt bit is the 32-bit hardware shift (the stamp mod 32)
//!   (`tests::signal_stamp_bits_alias_mod_32`).
//! - A fault's bit is the FIFO's stamp index; a resume clears by child id.
//! - `raiseInterrupt` is not checked for nil: the first interrupt dies
//!   (`tests::interrupt_nil_block_dies`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::device::{DeviceCore, DeviceDeath, DeviceRaise, RunState};
use crate::fifo::Fifo;
use crate::guest::GuestPtr;
use crate::num::{ToUsize, Truncate};
use crate::platform::Platform;
use crate::platform::os::Clock;
use crate::raise::RaiseResult;
use crate::seams::links::FifoDeviceLink;
use crate::tasks::{Completion, Task};

/// A waiter's next index when it waits on nothing.
const NOT_WAITING: u32 = u32::MAX;
/// The child FIFO slots (slot 0 is never used).
const CHILD_FIFO_SLOTS: u32 = 64;
/// The child nodes' offset in the root node (node `id` is at `0x400 + (id - 1)·0x14`).
const CHILD_NODES: u64 = 0x400;
/// One child node's size.
const CHILD_NODE_SIZE: u64 = 0x14;
/// The word a delete clears: `0x410 + id·0x14`, the 5th word of node `id + 1`.
const DELETED_NODE_WORD: u64 = 0x410;

/// The task completions a FIFO delete still waits for. The count sits under a lock of the
/// delete's own, and the last completion notifies under it: the wait checks the count and parks
/// under that lock, so a completion from another thread can never land between the two and leave
/// the delete waiting (`tests::countdown_misses_no_completion_from_another_thread`).
struct Countdown {
    count: Mutex<u32>,
    done: Condvar,
}

impl Countdown {
    /// `count` completions to come.
    fn new(count: u32) -> Self {
        Countdown {
            count: Mutex::new(count),
            done: Condvar::new(),
        }
    }

    /// One completion; the last one wakes the wait.
    fn count_down(&self) {
        let mut count = self.count.lock();
        *count = count.wrapping_sub(1);
        if *count == 0 {
            self.done.notify_all();
        }
    }

    /// Whether every completion has come.
    fn is_done(&self) -> bool {
        *self.count.lock() == 0
    }

    /// Waits for every completion.
    fn wait(&self, clock: &impl Clock) {
        let mut count = self.count.lock();
        while *count != 0 {
            clock.wait(&self.done, &mut count);
        }
    }
}

impl<P: Platform> DeviceCore<P> {
    /// Runs `f` on the FIFO queue, synchronously, with the device.
    ///
    /// The queue's block must be `'static`, so it holds a strong reference to the core (the
    /// back-link upgrades through the shutdown, so the shutdown's own stop goes through the
    /// queue). Only a core whose fields are going has none left: the block then runs inline,
    /// as a synchronous dispatch runs it on the calling thread, with no handler live.
    ///
    /// Either way `f` runs with no handler and no holder ([`Hold::NONE`](crate::rt::Hold::NONE)),
    /// whatever the caller's: it is `Send`, so it cannot take the caller's hold.
    pub(crate) fn on_fifo_queue<R: Send + 'static>(
        &self,
        f: impl FnOnce(&DeviceCore<P>) -> R + Send + 'static,
    ) -> R {
        if let Some(device) = self.weak_self.upgrade() {
            return self.fifo_queue.sync(move || f(&device));
        }
        f(self)
    }

    /// Creates and starts child `fifo` on the FIFO queue. Raises for id 0 or at or past
    /// `maxFIFOCount`; refuses while the root is torn down.
    pub fn create_fifo(&self, fifo: u32) -> RaiseResult<bool> {
        let fifo_id = fifo;
        let max = self.features.load().max_fifo_count;
        if fifo_id == 0 || fifo_id >= max {
            crate::raise!(
                DeviceRaise::InvalidFifoId,
                format_args!("child FIFO id {fifo_id} is 0 or not below the FIFO count {max}")
            );
        }
        if self.tearing_down_root.load(Ordering::Relaxed) {
            return Ok(false);
        }
        // The block has no handler, even under a define's: a raise in it dies there.
        let created = self.on_fifo_queue(move |d| {
            let Some(info) = d.waiting_info.get(fifo_id.to_usize()) else {
                crate::fatal!(DeviceDeath::PastWaitingInfo)
            };
            crate::fatal_unless!(
                info.next_waiting_index.load(Ordering::Relaxed) == NOT_WAITING,
                DeviceDeath::CreateWhileWaiting
            );
            log::info!("FIFO create_fifo: requested child slot {}", fifo_id);
            if d.create_fifo_internal(fifo_id, false) {
                // Its result is ignored.
                if let Some(child) = d.child_slot(fifo_id) {
                    let started = child.start(&d.platform);
                    log::info!("FIFO create_fifo: child slot {} start result={}", fifo_id, started);
                }
                return true;
            } else {
                log::warn!("FIFO create_fifo: create_fifo_internal returned false for slot {}", fifo_id);
            }
            false
        });
        Ok(created)
    }

    /// Makes child `fifo_id` over its node in the root node, not started. Every refusal but
    /// the FIFO's own is logged here.
    ///
    /// Its callers have no handler: the FIFO queue's block (a define's, whose caller's handler does
    /// not reach the block) and a resume. A raise making the FIFO's shared event dies at the send.
    ///
    /// A `maxFIFOCount` above 64 would let ids of 64 and up past the check; the slot index then
    /// dies as out of bounds.
    pub fn create_fifo_internal(&self, fifo_id: u32, restoring: bool) -> bool {
        let max = self.features.load().max_fifo_count;
        if fifo_id == 0 || fifo_id >= max {
            log::warn!(
                "child FIFO {fifo_id} refused: its id is 0 or not below the FIFO count {max}"
            );
            return false;
        }
        let Some(root_node) = self.root_node.atomics().root_node() else {
            log::warn!("child FIFO {fifo_id} refused: no root node");
            return false;
        };
        if self.child_slot(fifo_id).is_some() {
            log::warn!("child FIFO {fifo_id} refused: it exists");
            return false;
        }
        let node_offset = CHILD_NODES
            .wrapping_add(u64::from(fifo_id.wrapping_sub(1)).wrapping_mul(CHILD_NODE_SIZE));
        let base_node = root_node.offset(node_offset);
        let Some(last_pending) = self
            .waiting_info
            .get(fifo_id.to_usize())
            .map(|w| w.last_pending_stamp.clone())
        else {
            crate::fatal!(DeviceDeath::PastWaitingInfo)
        };
        let host: FifoDeviceLink<P> = self.weak_self.clone();
        let Some(fifo) = Fifo::new_child_on(
            &self.platform,
            host,
            base_node,
            fifo_id,
            restoring,
            last_pending,
        ) else {
            return false;
        };
        let mut children = self.child_fifos.lock();
        let Some(slot) = children.get_mut(fifo_id.to_usize()) else {
            crate::fatal!(DeviceDeath::PastChildFifos)
        };
        *slot = Some(fifo);
        true
    }

    /// The child in slot `fifo_id`; an id past the slots dies.
    fn child_slot(&self, fifo_id: u32) -> Option<Arc<Fifo<P>>> {
        let children = self.child_fifos.lock();
        let Some(slot) = children.get(fifo_id.to_usize()) else {
            crate::fatal!(DeviceDeath::PastChildFifos)
        };
        slot.clone()
    }

    /// Takes the child out on the FIFO queue, stops and drains it, clears a node word in the
    /// root node, then tells every task and waits for all their completions. A task's raise
    /// goes to the caller.
    ///
    /// The cleared word is `0x410 + id·0x14`, the last word of node `id + 1` (node `id` is at
    /// `0x400 + (id - 1)·0x14`): the next child's word, not this one's, as the suite measures
    /// (testing.md 7.1). The fault bit cleared is `1 << id` and the wait finished is the id's,
    /// while a fault sets bit `stampIndex`: a child whose stamp index differs from its id leaves
    /// its fault bit set.
    pub fn delete_fifo(&self, fifo: u32) -> RaiseResult<bool> {
        let fifo_id = fifo;
        if fifo_id == 0 || fifo_id >= self.features.load().max_fifo_count {
            return Ok(false);
        }
        // The block has no handler, even under a delete's; it has only deaths.
        let taken = self.on_fifo_queue(move |d| {
            let taken = {
                let mut children = d.child_fifos.lock();
                let Some(slot) = children.get_mut(fifo_id.to_usize()) else {
                    crate::fatal!(DeviceDeath::PastChildFifos)
                };
                slot.take()
            };
            if taken.is_some() {
                // fifo_id < 64 here.
                d.interrupt_fault_mask
                    .fetch_and(!crate::num::bit64(fifo_id), Ordering::SeqCst);
                d.finish_wait(fifo_id);
            }
            taken
        });
        let Some(child) = taken else {
            return Ok(false);
        };
        let stamp_index = child.stamp_index();
        child.stop();
        self.signal_fifos();
        child.drain();
        drop(child);
        let root_node = self.root_node.atomics().root_node().unwrap_or_default();
        let word = DELETED_NODE_WORD.wrapping_add(u64::from(fifo_id).wrapping_mul(CHILD_NODE_SIZE));
        root_node.write_u32(word, 0, DeviceDeath::DeletedNodeWithoutRoot);
        let mut tasks = self.tasks.lock();
        let remaining = Arc::new(Countdown::new(tasks.len().low_u32()));
        // Every task, in the table's order, held across the wait below, which lets the table's
        // lock go: a task another FIFO's delete removes meanwhile only loses the table's
        // reference, and its last release is here, after the wait, on this thread, under the
        // table's lock, after every reply. A task's raise lets them go as it leaves, before the
        // lock.
        let all: Arc<[Task<P>]> = tasks.values().cloned().collect();
        for task in all.iter() {
            let pending = remaining.clone();
            // A remote task keeps the completion until the service's reply, and the completion
            // holds a retain of the device (the host's `Device`) as long; a local task calls it
            // in place and takes none. A device whose shutdown started has no retain left to
            // take.
            let copied_self = match task {
                Task::Remote(_) => self.retain_self(),
                Task::Local(_) => None,
            };
            let done: Completion = Arc::new(move |_, _| {
                let _copied_self = &copied_self;
                pending.count_down();
            });
            task.fifo_deleted(stamp_index, done)?;
        }
        // The table's lock goes only while there is a completion to wait for.
        if !remaining.is_done() {
            MutexGuard::unlocked(&mut tasks, || {
                remaining.wait(self.platform.clock());
            });
        }
        drop(all);
        drop(remaining);
        drop(tasks);
        Ok(true)
    }

    /// On the FIFO queue: clears the FIFO's fault bit, raises the interrupt for the faults
    /// left, reports the stamp waits when the FIFO was not faulted (the first nine times per
    /// process), and resumes the child.
    ///
    /// An id at or past `maxFIFOCount` is logged and then used anyway. `1 << fifo` is the
    /// hardware shift (the id mod 64), so an id of 64 or more clears the fault bit of the id
    /// mod 64. Past the 64 slots there is no child FIFO, so such an id resumes nothing, as
    /// [`DeviceCore::wake_fifo`] ignores it.
    pub fn resume_child_fifo(&self, fifo_id: u32) {
        let fifo = fifo_id;
        log::info!("guest resumes child FIFO {fifo}");
        let max = self.features.load().max_fifo_count;
        if fifo >= max {
            log::warn!("resumed FIFO {fifo} is not below the FIFO count {max}");
        }
        self.on_fifo_queue(move |d| {
            let report_timeout;
            {
                let _signal = d.signal_mutex.lock();
                let mask = crate::num::bit64(fifo);
                let fault_mask = d.interrupt_fault_mask.fetch_and(!mask, Ordering::SeqCst);
                if fault_mask & mask != 0 {
                    report_timeout = false;
                } else {
                    log::warn!(
                        "resumed FIFO {fifo} was not faulted (fault mask {fault_mask:#018x})"
                    );
                    report_timeout = d.take_report_budget();
                }
                if d.interrupt_fault_mask.load(Ordering::SeqCst) != 0 {
                    d.interrupt();
                }
            }
            if report_timeout {
                let device = d.weak_self.clone();
                d.reporting_queue.sync(move || {
                    if let Some(device) = device.upgrade() {
                        device.report_stamp_wait_timeout();
                    }
                });
            }
            if fifo < CHILD_FIFO_SLOTS
                && let Some(child) = d.child_slot(fifo)
            {
                child.resume();
            }
        });
    }

    /// Pokes child `fifo` on the FIFO queue, if there is one. The check is against the 64
    /// slots, not `maxFIFOCount`.
    pub fn wake_fifo(&self, fifo: u32) {
        log::info!("FIFO wake_fifo: waking child slot {}", fifo);
        if fifo == 0 || fifo >= CHILD_FIFO_SLOTS {
            log::warn!("wake of FIFO {fifo} dropped: no such child slot");
            return;
        }
        self.on_fifo_queue(move |d| {
            if let Some(child) = d.child_slot(fifo) {
                // poke: the child FIFO woken
                crate::trace::event("poke", &[u64::from(fifo)]);
                if !child.is_running() {
                    let started = child.start(&d.platform);
                    log::info!("FIFO wake_fifo: child slot {} was not running, start result={}", fifo, started);
                }
                child.poke();
            } else {
                log::warn!("wake of FIFO {fifo} dropped: slot is empty");
            }
        });
    }

    /// Writes the stamp's value into the stamp memory, sets its interrupt bit, wakes the
    /// waiters and raises the interrupt, under the signal lock.
    ///
    /// The check is `stamp > maxFIFOCount`: stamp `maxFIFOCount` passes. `1 << stamp` is the
    /// 32-bit hardware shift (stamp mod 32): stamps 32-63 set the bit of stamp - 32. The suite
    /// measures both (testing.md 7.1).
    pub fn signal_stamp(&self, stamp_index: u32, stamp_value: u32) {
        let stamp = stamp_index;
        let max = self.features.load().max_fifo_count;
        if stamp > max {
            log::warn!("stamp {stamp} signal dropped: past the FIFO count {max}");
            return;
        }
        let _signal = self.signal_mutex.lock();
        let memory: GuestPtr = self.stamp_memory.lock().unwrap_or_default();
        memory.write_u32(
            u64::from(stamp).wrapping_mul(4),
            stamp_value,
            DeviceDeath::StampWriteWithoutMemory,
        );
        self.interrupt_stamp
            .fetch_or(crate::num::bit32(stamp), Ordering::SeqCst);
        // stamp: slot, value: written for the guest
        crate::trace::event("stamp", &[u64::from(stamp), u64::from(stamp_value)]);
        self.signal_cond.notify_all();
        self.interrupt();
    }

    /// Asynchronously on the FIFO queue: sets the FIFO's fault bit and raises the interrupt,
    /// under the signal lock.
    ///
    /// The bit is the FIFO's stamp index (a resume clears by child id).
    pub fn signal_fault_in_fifo(&self, fifo: Arc<Fifo<P>>) {
        // The block holds a retain of the device (the host's `Device`), whose release at the
        // block's end can be the last. A device whose shutdown started has no retain left to
        // take, but the block is still queued, and the shutdown's later child delete runs it on
        // the live core, which the block holds until the core's fields go.
        let retained = self.retain_self();
        let Some(device) = self.weak_self.upgrade() else {
            return; // the fields are gone: no shutdown is left to order the block after
        };
        self.fifo_queue.exec_async(move || {
            {
                let _signal = device.signal_mutex.lock();
                device
                    .interrupt_fault_mask
                    .fetch_or(crate::num::bit64(fifo.stamp_index()), Ordering::SeqCst);
                device.interrupt();
            }
            // The block's release of the device, after the body.
            drop(retained);
        });
    }

    /// Wakes every stamp waiter.
    pub fn signal_fifos(&self) {
        let _signal = self.signal_mutex.lock();
        self.signal_cond.notify_all();
    }

    /// Raises the guest interrupt (`raiseInterrupt(0)`) unless the device is suspended or
    /// paused.
    ///
    /// The block is not checked for nil: a nil `raiseInterrupt` dies at the first interrupt.
    pub fn interrupt(&self) {
        if self.run_state.load() == RunState::Running {
            let Some(raise) = &self.blocks.raise_interrupt else {
                crate::fatal!(DeviceDeath::NilBlock("raiseInterrupt"))
            };
            // irq: the guest interrupt raised
            crate::trace::event("irq", &[]);
            raise(0);
        }
    }

    /// Whether any FIFO is faulted.
    pub fn read_interrupt_fault(&self) -> u32 {
        u32::from(self.interrupt_fault_mask.load(Ordering::SeqCst) != 0)
    }

    /// Sets the port's display interrupt bit (`1 << port`, the hardware shift of the port's low
    /// 32 bits) and raises the interrupt.
    pub fn display_poke_port(&self, port: u64) {
        self.interrupt_display
            .fetch_or(crate::num::bit32(port.low_u32()), Ordering::SeqCst);
        self.interrupt();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::device::root::rig::{Rig, rig, rig_with};
    use crate::device::test_support::Lines;
    use crate::guest::records::APV_PAGE_SIZE;
    use crate::raise::RaiseId;
    use crate::rt::Hold;
    use crate::testkit::capture::catch_death;
    use crate::testkit::capture::expect_death;

    fn set_max(r: &Rig, max: u32) {
        let mut f = r.device.features.load();
        f.max_fifo_count = max;
        r.device.features.store(&f);
    }

    fn mask(r: &Rig) -> u64 {
        r.device.interrupt_fault_mask.load(Ordering::SeqCst)
    }

    /// Root page 1 at guest page 1 (0x4000 on a 16 KiB guest page); child node `id` at 0x400 +
    /// (id - 1)·0x14 into it.
    fn child_node_pa(id: u64) -> u64 {
        APV_PAGE_SIZE + 0x400 + (id - 1) * 0x14
    }

    /// A completion that comes from another thread at any moment around the wait's check wakes
    /// it: each round races one against a fresh wait, and every wait ends
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real threads, real-time waits
    fn countdown_misses_no_completion_from_another_thread() {
        let clock = crate::testkit::clock::VirtualClock::default();
        for round in 0..2000u32 {
            let countdown = Arc::new(Countdown::new(1));
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::scope(|s| {
                let c = Arc::clone(&countdown);
                s.spawn(move || {
                    for _ in 0..round % 7 {
                        std::thread::yield_now();
                    }
                    c.count_down();
                });
                let c = Arc::clone(&countdown);
                let clock = &clock;
                s.spawn(move || {
                    c.wait(clock);
                    let _ = tx.send(());
                });
                assert!(
                    rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                    "round {round}: the wait missed the completion"
                );
            });
            assert!(countdown.is_done());
        }
    }

    #[test]
    fn create_fifo_invalid_id_raises() {
        let r = rig();
        set_max(&r, 8);
        for id in [0, 8, 64, u32::MAX] {
            let raised = r.device.create_fifo(id).expect_err("raises");
            assert_eq!(raised.id(), RaiseId::Device(DeviceRaise::InvalidFifoId));
        }
    }

    #[test]
    fn create_fifo_refused_while_tearing_down() {
        let r = rig();
        r.device.tearing_down_root.store(true, Ordering::SeqCst);
        assert!(!r.device.create_fifo(1).unwrap());
    }

    #[test]
    fn create_fifo_without_root_node_logs() {
        let r = rig();
        let lines = Lines::capture();
        assert!(!r.device.create_fifo(1).unwrap());
        assert_eq!(lines.all(), ["child FIFO 1 refused: no root node"]);
    }

    #[test]
    fn create_fifo_while_waiting_dies() {
        let r = rig();
        r.device.waiting_info[3]
            .next_waiting_index
            .store(1, Ordering::SeqCst);
        let d = r.device.clone();
        expect_death(DeviceDeath::CreateWhileWaiting.site(), move || {
            let _ = d.create_fifo(3);
        });
    }

    #[test]
    fn create_fifo_internal_refusals_log() {
        let r = rig();
        set_max(&r, 4);
        let lines = Lines::capture();
        assert!(!r.device.create_fifo_internal(0, false));
        assert!(!r.device.create_fifo_internal(4, false));
        // The id and `maxFIFOCount`.
        assert_eq!(
            lines.all(),
            [
                "child FIFO 0 refused: its id is 0 or not below the FIFO count 4",
                "child FIFO 4 refused: its id is 0 or not below the FIFO count 4"
            ]
        );
    }

    // A delete zeroes the next node's page-list word, id 63 included (testing.md 7.1).
    #[test]
    fn create_then_delete_child_clears_next_node_word() {
        let r = rig();
        // Child 2's node: stamp index 2; child 3's node: a word the delete of 2 clears.
        r.set_ram_u32(child_node_pa(2), 2);
        r.set_ram_u32(child_node_pa(3) + 0x10, 0x7777);
        r.set_ram_u32(child_node_pa(2) + 0x10, 0x5555);
        r.device.set_root_page(Hold::NONE, 1);
        assert!(r.device.create_fifo(2).unwrap());
        assert!(r.device.child_fifos.lock()[2].is_some());
        // A second create of the same id is refused.
        assert!(!r.device.create_fifo_internal(2, false));
        r.device.interrupt_fault_mask.store(0b101, Ordering::SeqCst);
        assert!(r.device.delete_fifo(2).unwrap());
        assert!(r.device.child_fifos.lock()[2].is_none());
        assert_eq!(mask(&r), 0b001);
        r.platform.fake_vm().sync();
        assert_eq!(r.ram_u32(child_node_pa(3) + 0x10), 0);
        assert_eq!(r.ram_u32(child_node_pa(2) + 0x10), 0x5555);
        // Gone: a second delete is refused.
        assert!(!r.device.delete_fifo(2).unwrap());
    }

    /// The block runs on the caller's thread, as a synchronous dispatch runs it: the child's
    /// ring is mapped there, and the caller publishes the define's stamp with no hand-back from
    /// the queue's worker in between, which let the new child's first interrupt overtake the
    /// root's (fifo_child_define_delete).
    #[test]
    fn create_fifo_block_runs_on_callers_thread() {
        let r = rig();
        r.set_ram_u32(child_node_pa(1), 1);
        r.set_ram_u32(child_node_pa(1) + 0x10, 0x20);
        r.set_ram_u32(0x20 * 0x1000, 0x21);
        r.device.set_root_page(Hold::NONE, 1);
        let m = r.mark();
        let me = std::thread::current().name().map(str::to_owned);
        assert!(me.is_some());
        assert!(r.device.create_fifo(1).unwrap());
        let maps: Vec<_> = r
            .host_calls(m)
            .into_iter()
            .filter(|c| c.sel == "mapMemory")
            .collect();
        assert!(!maps.is_empty(), "{:?}", r.host_sels(m));
        for c in &maps {
            assert_eq!(c.thread, me, "{c:?}");
        }
        assert!(r.device.delete_fifo(1).unwrap());
    }

    /// The define's block on the FIFO queue has no handler, whatever its caller's: a raise making
    /// the child's shared event dies at the send even under a handler, as a DefineChildFifo's F
    /// class gives the define, and never becomes the command's fault.
    #[test]
    fn a_define_whose_event_raises_dies_under_its_handler() {
        use crate::raise::death::{DeathKind, DeathSite};
        use crate::testkit::Step;
        let r = rig();
        r.set_ram_u32(child_node_pa(1), 1);
        r.set_ram_u32(child_node_pa(1) + 0x10, 0x20);
        r.device.set_root_page(Hold::NONE, 1);
        r.platform.script().push(
            "newSharedEvent",
            Step::raise("NSGenericException", "no event"),
        );
        let d = r.device.clone();
        expect_death(
            DeathSite {
                kind: DeathKind::UncaughtObjc,
                file: "newSharedEvent",
                line: 0,
            },
            move || {
                let _ = d.create_fifo(1);
            },
        );
    }

    /// A delete's completion holds a retain of the device while a remote task keeps it, until
    /// the service's reply. A host release while that reply is outstanding is then not the
    /// last: it returns at once, and the shutdown runs where the completion goes, after the
    /// reply.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: threads and real-time waits
    fn remote_task_completion_retains_the_device_until_the_reply() {
        use crate::device::config::DeviceConfig;
        use crate::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
        use crate::tasks::launcher::{HostLink, ServiceSide};
        use crate::tasks::wire::{Reply, ToService};
        use crate::testkit::FakePlatform;
        use crate::testkit::GuestRam;
        use crate::testkit::capture::current_slot;
        use crate::testkit::capture::with_slot;
        use crate::testkit::host::FakeHost;
        use crate::testkit::launcher::ThreadLauncher;
        use std::time::{Duration, Instant};

        fn wait_until(what: &str, f: impl Fn() -> bool) {
            let start = Instant::now();
            while !f() {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "timed out: {what}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        /// Accepts the handshake, holds the `fifoDeleted:` reply for the test, answers the
        /// rest with 1.
        struct Svc(Arc<parking_lot::Mutex<Option<Reply<u32>>>>);

        impl ServiceSide<FakePlatform> for Svc {
            fn on_message(&self, m: ToService<FakePlatform>) {
                match m {
                    ToService::Handshake { reply, .. } => reply.call((1, 4242)),
                    ToService::FifoDeleted { reply, .. } => *self.0.lock() = Some(reply),
                    ToService::FlushResources { reply } => reply.call(1),
                    _ => {}
                }
            }
        }

        let held = Arc::new(parking_lot::Mutex::new(None));
        let mut p = FakePlatform::new();
        let h2 = held.clone();
        p.set_launcher(ThreadLauncher::with_factory(Arc::new(
            move |_p, _host: Arc<dyn HostLink<FakePlatform>>| {
                let svc: Arc<dyn ServiceSide<FakePlatform>> = Arc::new(Svc(h2.clone()));
                svc
            },
        )));
        let p = Arc::new(p);
        let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).expect("ram"));
        // Process isolation needs a memory map, over the host's RAM.
        let mut map = MemoryMapDescriptor::new();
        for r in host.ram().ranges() {
            map.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.region.len(),
                host_address: r.host(),
            });
        }
        let mut c = DeviceConfig::new(Some(p.device()));
        c.display_port_count = 1;
        c.blocks.raise_interrupt = host.blocks().raise_interrupt;
        c.memory_map_descriptor = Some(Arc::new(map));
        c.enable_process_isolation = true;
        let device = DeviceCore::create(p.clone(), c).expect("device");
        let mut f = device.features.load();
        f.max_fifo_count = 64;
        device.features.store(&f);
        let handle = Arc::downgrade(&device);
        // Child 1 (as `create_fifo_block_runs_on_callers_thread` lays it out), and user task 1:
        // a `PGRemoteTask`.
        assert!(host.ram().write_u32(child_node_pa(1), 1));
        assert!(host.ram().write_u32(child_node_pa(1) + 0x10, 0x20));
        assert!(host.ram().write_u32(0x20 * 0x1000, 0x21));
        device.set_root_page(Hold::NONE, 1);
        assert!(device.create_fifo(1).unwrap());
        device
            .create_task_id(Hold::NONE, 1, 7, 0x10000, false)
            .unwrap();
        assert!(matches!(device.tasks.lock().get(1), Some(Task::Remote(_))));
        // The root FIFO thread's delete (it holds the device unretained: the core).
        let fields: Arc<DeviceCore<FakePlatform>> = Arc::clone(&device);
        let slot = current_slot();
        let deleter = std::thread::spawn(move || with_slot(slot, || fields.delete_fifo(1)));
        wait_until("the fifoDeleted: message", || held.lock().is_some());
        // The host's reference and the completion's.
        assert_eq!(handle.strong_count(), 2);
        // The host's release while the reply is outstanding: not the last, it returns at once.
        let dropper = std::thread::spawn(move || drop(device));
        wait_until("the host's release to return", || dropper.is_finished());
        dropper.join().unwrap();
        assert_eq!(handle.strong_count(), 1);
        assert!(!deleter.is_finished());
        // The service replies: the completion runs, the delete returns, and the completion's
        // release is the last: the shutdown runs there.
        let reply = held.lock().take().expect("reply");
        reply.call(1);
        drop(reply);
        assert!(deleter.join().unwrap().unwrap());
        wait_until("the shutdown", || handle.strong_count() == 0);
        drop(host);
    }

    /// A local task calls the completion in place, so it takes no reference to the device.
    #[test]
    fn local_task_completion_takes_no_reference() {
        let r = rig();
        r.set_ram_u32(child_node_pa(1), 1);
        r.set_ram_u32(child_node_pa(1) + 0x10, 0x20);
        r.set_ram_u32(0x20 * 0x1000, 0x21);
        // Task 1's root node at page 0x55: page-table depth 1, length 0x4000.
        r.set_ram_u32(0x55 * APV_PAGE_SIZE + 4, 1);
        assert!(r.host.ram().write_u64(0x55 * APV_PAGE_SIZE + 0x14, 0x4000));
        r.platform.fake_vm().sync();
        r.device.set_root_page(Hold::NONE, 1);
        assert!(r.device.create_fifo(1).unwrap());
        r.device
            .create_task_id(Hold::NONE, 1, 0x55, 0x4000, false)
            .unwrap();
        assert!(matches!(r.device.tasks.lock().get(1), Some(Task::Local(_))));
        let handle = Arc::downgrade(&r.device);
        assert!(r.device.delete_fifo(1).unwrap());
        assert_eq!(handle.strong_count(), 1);
    }

    /// A delete keeps every task until the wait for the completions ends, with the task table's
    /// lock held again. The wait lets the lock go, so another FIFO's task delete can remove a
    /// task meanwhile: a local task (task 0, whose completion ran in place) or a remote task
    /// whose reply has already come back. The table's release is then not the last: each task
    /// goes after the wait, on the deleting thread, under the table's lock, after the
    /// outstanding reply, never inside the other FIFO's delete.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: threads and real-time waits
    fn the_delete_holds_every_task_until_the_wait_ends() {
        use crate::device::config::DeviceConfig;
        use crate::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
        use crate::tasks::launcher::{HostLink, ServiceSide};
        use crate::tasks::wire::{Reply, ToService};
        use crate::testkit::FakePlatform;
        use crate::testkit::GuestRam;
        use crate::testkit::capture::current_slot;
        use crate::testkit::capture::with_slot;
        use crate::testkit::host::FakeHost;
        use crate::testkit::launcher::ThreadLauncher;
        use std::sync::atomic::{AtomicBool, AtomicUsize};
        use std::time::{Duration, Instant};

        fn wait_until(what: &str, f: impl Fn() -> bool) {
            let start = Instant::now();
            while !f() {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "timed out: {what}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        /// The first task's service holds its `fifoDeleted:` reply for the test; the others
        /// answer it at once and flag that the reply (and the task it retains) is gone.
        struct Svc {
            held: Option<Arc<parking_lot::Mutex<Option<Reply<u32>>>>>,
            answered: Arc<AtomicBool>,
        }

        impl ServiceSide<FakePlatform> for Svc {
            fn on_message(&self, m: ToService<FakePlatform>) {
                match m {
                    ToService::Handshake { reply, .. } => reply.call((1, 4242)),
                    ToService::FifoDeleted { reply, .. } => match &self.held {
                        Some(held) => *held.lock() = Some(reply),
                        None => {
                            reply.call(1);
                            drop(reply);
                            self.answered.store(true, Ordering::SeqCst);
                        }
                    },
                    ToService::FlushResources { reply } => reply.call(1),
                    _ => {}
                }
            }
        }

        let held = Arc::new(parking_lot::Mutex::new(None));
        let answered = Arc::new(AtomicBool::new(false));
        let launches = Arc::new(AtomicUsize::new(0));
        let mut p = FakePlatform::new();
        let (h2, a2) = (held.clone(), answered.clone());
        p.set_launcher(ThreadLauncher::with_factory(Arc::new(
            move |_p, _host: Arc<dyn HostLink<FakePlatform>>| {
                let first = launches.fetch_add(1, Ordering::SeqCst) == 0;
                let svc: Arc<dyn ServiceSide<FakePlatform>> = Arc::new(Svc {
                    held: first.then(|| h2.clone()),
                    answered: a2.clone(),
                });
                svc
            },
        )));
        let p = Arc::new(p);
        let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).expect("ram"));
        let mut map = MemoryMapDescriptor::new();
        for r in host.ram().ranges() {
            map.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.region.len(),
                host_address: r.host(),
            });
        }
        let mut c = DeviceConfig::new(Some(p.device()));
        c.display_port_count = 1;
        c.blocks.raise_interrupt = host.blocks().raise_interrupt;
        c.memory_map_descriptor = Some(Arc::new(map));
        c.enable_process_isolation = true;
        let device = DeviceCore::create(p.clone(), c).expect("device");
        let mut f = device.features.load();
        f.max_fifo_count = 64;
        device.features.store(&f);
        assert!(host.ram().write_u32(child_node_pa(1), 1));
        assert!(host.ram().write_u32(child_node_pa(1) + 0x10, 0x20));
        assert!(host.ram().write_u32(0x20 * 0x1000, 0x21));
        // Task 0's root node at page 0x55: page-table depth 1, length 0x4000.
        assert!(host.ram().write_u32(0x55 * APV_PAGE_SIZE + 4, 1));
        assert!(host.ram().write_u64(0x55 * APV_PAGE_SIZE + 0x14, 0x4000));
        device.set_root_page(Hold::NONE, 1);
        assert!(device.create_fifo(1).unwrap());
        // Task 0 is local; user tasks 1 (reply held) and 2 are remote.
        device
            .create_task_id(Hold::NONE, 0, 0x55, 0x4000, false)
            .unwrap();
        device
            .create_task_id(Hold::NONE, 1, 7, 0x10000, false)
            .unwrap();
        device
            .create_task_id(Hold::NONE, 2, 8, 0x10000, false)
            .unwrap();
        let (local, remote) = {
            let tasks = device.tasks.lock();
            let local = match tasks.get(0) {
                Some(Task::Local(t0)) => Some(Arc::downgrade(t0)),
                _ => None,
            };
            let remote = match tasks.get(2) {
                Some(Task::Remote(t2)) => Some(Arc::downgrade(t2)),
                _ => None,
            };
            (
                local.expect("task 0 is local"),
                remote.expect("task 2 is remote"),
            )
        };
        assert!(matches!(device.tasks.lock().get(1), Some(Task::Remote(_))));
        let fields: Arc<DeviceCore<FakePlatform>> = Arc::clone(&device);
        let slot = current_slot();
        let deleter = std::thread::spawn(move || with_slot(slot, || fields.delete_fifo(1)));
        // The delete waits for task 1's reply with the task table's lock free; task 0's
        // completion ran in place and task 2's reply is gone.
        wait_until("task 1's fifoDeleted: message", || held.lock().is_some());
        wait_until("task 2's reply", || answered.load(Ordering::SeqCst));
        // The reply's host side lets go of task 2 after the service's flag: wait for it, so
        // that only the table and the delete hold the task (the check flaked under load).
        wait_until("task 2's reply released", || remote.strong_count() == 2);
        // Another FIFO's task delete for tasks 0 and 2: only the table's reference goes.
        assert!(device.delete_task_id(0).unwrap());
        assert!(device.delete_task_id(2).unwrap());
        assert!(device.tasks.lock().get(0).is_none());
        assert_eq!(local.strong_count(), 1, "the delete holds task 0");
        assert_eq!(remote.strong_count(), 1, "the delete holds task 2");
        // Task 1's reply comes while the task table's lock is held elsewhere: the delete cannot
        // re-take it, so its wait has not ended and the tasks live on.
        {
            let _other = device.tasks.lock();
            let reply = held.lock().take().expect("reply");
            reply.call(1);
            drop(reply);
            std::thread::sleep(Duration::from_millis(20));
            assert!(!deleter.is_finished());
            assert_eq!(local.strong_count(), 1);
            assert_eq!(remote.strong_count(), 1);
        }
        // The wait ends on the deleter's thread before the delete returns: both tasks go
        // there.
        assert!(deleter.join().unwrap().unwrap());
        assert_eq!(local.strong_count(), 0);
        assert_eq!(remote.strong_count(), 0);
        drop(device);
        drop(host);
    }

    #[test]
    fn delete_fifo_refusals() {
        let r = rig();
        set_max(&r, 4);
        assert!(!r.device.delete_fifo(0).unwrap());
        assert!(!r.device.delete_fifo(4).unwrap());
        assert!(!r.device.delete_fifo(1).unwrap());
    }

    #[test]
    fn resume_child_fifo_faulted_clears_and_raises() {
        let r = rig();
        r.device
            .interrupt_fault_mask
            .store(0b1010, Ordering::SeqCst);
        r.device.resume_child_fifo(3);
        assert_eq!(mask(&r), 0b0010);
        assert_eq!(r.host.interrupts(), [0]);
        r.device.resume_child_fifo(1);
        assert_eq!(mask(&r), 0);
        assert_eq!(r.host.interrupts(), [0]);
    }

    #[test]
    fn resume_child_fifo_non_faulted_logs_and_reports() {
        let r = rig();
        set_max(&r, 4);
        let lines = Lines::capture();
        r.device.resume_child_fifo(5);
        assert!(r.host.interrupts().is_empty());
        // The id, the id past the count, then the mask before the clear.
        assert_eq!(
            lines.all(),
            [
                "guest resumes child FIFO 5",
                "resumed FIFO 5 is not below the FIFO count 4",
                "resumed FIFO 5 was not faulted (fault mask 0x0000000000000000)"
            ]
        );
    }

    // A resume's `1 << fifo` is mod 64, and an id past the 64 slots resumes nothing.
    #[test]
    fn resume_child_fifo_past_the_table_resumes_nothing() {
        let r = rig();
        // 64: bit 0, with another fault left: the interrupt is raised for it.
        r.device.interrupt_fault_mask.store(0b11, Ordering::SeqCst);
        r.device.resume_child_fifo(64);
        assert_eq!(mask(&r), 0b10);
        assert_eq!(r.host.interrupts().len(), 1);
        // 65: bit 1.
        r.device.resume_child_fifo(65);
        assert_eq!(mask(&r), 0);
        // Past every slot: bit 63, nothing else.
        r.device.resume_child_fifo(u32::MAX);
        assert_eq!(mask(&r), 0);
    }

    // A stamp's check is `stamp > maxFIFOCount`: with maxFIFOCount 0 (no version yet) stamp 0 is
    // still published (0 > 0 is false).
    #[test]
    fn signal_stamp_bound_is_inclusive() {
        let r = rig();
        r.with_root_fifo(1, 2);
        set_max(&r, 0);
        r.device.signal_stamp(0, 0x11);
        r.device.signal_stamp(1, 0x22);
        r.platform.fake_vm().sync();
        // The stamps are at the start of the root FIFO's page 2.
        let stamps = 2 * APV_PAGE_SIZE;
        assert_eq!(r.ram_u32(stamps), 0x11);
        assert_eq!(r.ram_u32(stamps + 4), 0);
        assert_eq!(r.device.interrupt_stamp.load(Ordering::SeqCst), 1);
        assert_eq!(r.host.interrupts(), [0]);
        set_max(&r, 64);
        r.device.teardown_fifos();
    }

    // The stamp interrupt bit is `1 << stamp` in 32 bits: stamp indices 32-63 alias bits 0-31.
    #[test]
    fn signal_stamp_bits_alias_mod_32() {
        let r = rig();
        r.with_root_fifo(1, 2);
        r.device.signal_stamp(33, 7);
        assert_eq!(r.device.interrupt_stamp.load(Ordering::SeqCst), 0b10);
        r.platform.fake_vm().sync();
        assert_eq!(r.ram_u32(2 * APV_PAGE_SIZE + 33 * 4), 7);
        r.device.signal_stamp(64, 9);
        assert_eq!(r.device.interrupt_stamp.load(Ordering::SeqCst), 0b11);
        r.device.teardown_fifos();
    }

    #[test]
    fn signal_stamp_without_stamp_memory_dies() {
        let r = rig();
        let d = r.device.clone();
        assert_eq!(
            catch_death(|| d.signal_stamp(0, 1)).err(),
            Some(DeviceDeath::StampWriteWithoutMemory.site())
        );
    }

    #[test]
    fn signal_fault_in_fifo_sets_stamp_bit() {
        let r = rig();
        r.with_root_fifo(1, 2);
        let root = r.device.root_fifo.lock().root_fifo.clone().expect("root");
        r.device.signal_fault_in_fifo(root);
        r.device.on_fifo_queue(|_| ());
        assert_eq!(mask(&r), 1);
        assert_eq!(r.device.read_interrupt_fault(), 1);
        assert_eq!(r.host.interrupts(), [0]);
        r.device.teardown_fifos();
        assert_eq!(r.device.read_interrupt_fault(), 0);
    }

    /// A fault signalled once the shutdown has started (the handle is gone, so nothing is
    /// retained) still queues its block on the FIFO queue, and a later sync on the queue runs
    /// it: the fault bit is set and `raiseInterrupt(0)` is called.
    #[test]
    fn signal_fault_in_fifo_during_shutdown_still_raises() {
        let r = rig();
        r.with_root_fifo(1, 2);
        let root = r.device.root_fifo.lock().root_fifo.clone().expect("root");
        let device: Arc<DeviceCore<_>> = Arc::clone(&**r.device);
        let Rig {
            host,
            device: handle,
            ..
        } = r;
        drop(handle);
        assert!(device.retain_self().is_none());
        let before = host.interrupts().len();
        assert_eq!(device.run_state.load(), RunState::Running);
        device.signal_fault_in_fifo(root);
        device.on_fifo_queue(|_| ()); // The later sync
        assert_eq!(device.interrupt_fault_mask.load(Ordering::SeqCst), 1);
        let after = host.interrupts();
        assert_eq!(after.len(), before + 1);
        assert_eq!(after.last(), Some(&0));
    }

    #[test]
    fn interrupt_only_while_running() {
        let r = rig();
        r.device.run_state.store(RunState::Paused);
        r.device.interrupt();
        assert!(r.host.interrupts().is_empty());
        r.device.run_state.store(RunState::Running);
        r.device.interrupt();
        assert_eq!(r.host.interrupts(), [0]);
    }

    // No nil check on `raiseInterrupt`, the task blocks, or the command queues (testing.md 7.3).
    #[test]
    fn interrupt_nil_block_dies() {
        let r = rig_with(|b| b.raise_interrupt = None);
        let d = r.device.clone();
        assert_eq!(
            catch_death(|| d.interrupt()).err(),
            Some(DeviceDeath::NilBlock("raiseInterrupt").site())
        );
    }

    #[test]
    fn display_poke_port_sets_bit_and_raises() {
        let r = rig();
        r.device.display_poke_port(33);
        r.device.display_poke_port(2);
        assert_eq!(r.device.interrupt_display.load(Ordering::SeqCst), 0b110);
        assert_eq!(r.host.interrupts(), [0, 0]);
    }

    #[test]
    fn wake_fifo_bounds() {
        let r = rig();
        let lines = Lines::capture();
        r.device.wake_fifo(0);
        r.device.wake_fifo(64);
        // No child: nothing to poke.
        r.device.wake_fifo(5);
        // The wake's own lines: no resume was logged.
        assert_eq!(
            lines.all(),
            [
                "wake of FIFO 0 dropped: no such child slot",
                "wake of FIFO 64 dropped: no such child slot"
            ]
        );
    }
}
