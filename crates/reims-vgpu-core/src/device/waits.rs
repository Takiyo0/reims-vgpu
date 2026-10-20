//! The stamp waits: registration and cycle detection, the wait with its 2 s report, and the
//! timeout report.
//!
//! The waiting word is a 64-bit atomic whose every access is sequentially consistent, each
//! waiter's last pending stamp a 32-bit atomic, and each waiter's next index and value plain
//! words read and written with no lock (relaxed here).
//!
//! ## Contract
//!
//! - A wait reads the stamp slot, registers the wait (cycle detection), then under the signal
//!   lock waits for the slot to reach the value or the FIFO to quiesce. Each 2 s timed slice
//!   (`Deadline::Mono`) that times out logs, asks the reporting queue for a report, then waits
//!   untimed for the next wake (`tests::wait_stamps::timeout_reports_then_waits`).
//! - Reaching is `(slot - value) as i32 >= 0` (`tests::edges::reached_boundary`).
//! - A wait with no stamp memory dies at its first read; the report's read of the stamp memory
//!   is a log argument, evaluated only when the line logs, and dies there
//!   (`tests::wait_stamps::null_stamp_memory_dies`).
//! - A cycle of waits back to the waiter finishes the waiter's wait and raises
//!   (`tests::add_wait::circular_wait_raises`).
//! - The report runs on the reporting queue under the root FIFO's lock, with no pool of its own
//!   (`tests::report::the_report_opens_no_pool`); each child slot is read as the loop reaches it
//!   (`tests::report::each_child_slot_is_read_when_the_loop_reaches_it`).
//! - The waiter bit is `1 << waiter` in 32 bits, so waiters 32 apart share a bit and a cycle
//!   among waiters of 32 and up is never found (`tests::add_wait::alias_32_apart_dies`,
//!   `tests::add_wait::high_waiters_cycle_undetected`).
//! - A finished wait's mask `!(1 << waiter)` is a 32-bit signed value sign-extended: for a
//!   waiter ≡ 31 mod 32 it clears bits 31-63 (`tests::finish_wait::waiter_31_clears_high_bits`).
//! - The report walks the waiting word by 32-bit bits and reads the waiters by stamp index
//!   (`tests::report::alias_above_31_walks_low_entry`).

use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::device::{DeviceCore, DeviceDeath, DeviceRaise, RootFifoLocked};
use crate::fifo::Fifo;
use crate::fifo::wire::Records;
use crate::fifo::wire::StampWait;
use crate::guest::GuestPtr;
use crate::guest::records::APV_MAX_FIFOS;
use crate::num::{self, ToUsize};
use crate::platform::Platform;
use crate::platform::os::{Clock, Deadline};
use crate::raise::RaiseResult;

/// A waiter's next index when it waits on nothing.
const NOT_WAITING: u32 = u32::MAX;

/// A stamp wait's timed slice.
const STAMP_WAIT_TIMEOUT: Duration = Duration::from_secs(2);

/// Whether the slot has reached `value`: `(slot - value) as i32 >= 0`.
fn reached(slot: u32, value: u32) -> bool {
    slot.wrapping_sub(value).cast_signed() >= 0
}

/// `!(1 << waiter)` as a 32-bit signed value, sign-extended to 64 bits.
fn finish_wait_mask(waiter: u32) -> u64 {
    // The shift is the hardware's: waiter mod 32.
    num::sign_extend_32(!num::bit32(waiter))
}

impl<P: Platform> DeviceCore<P> {
    /// Every waiter idle with its last pending stamp 0; the values and the waiting word are
    /// left as they are.
    pub fn init_waiting_info(&self) {
        for info in &self.waiting_info {
            info.next_waiting_index
                .store(NOT_WAITING, Ordering::Relaxed);
            info.last_pending_stamp.store(0, Ordering::SeqCst);
        }
    }

    /// Each barrier in turn: an index at or past `maxFIFOCount` raises, a wait that ends
    /// quiesced answers `false`. Barriers already passed stay passed.
    pub fn wait_stamps(
        &self,
        barriers: Records<'_, StampWait>,
        fifo: &Fifo<P>,
    ) -> RaiseResult<bool> {
        self.wait_stamps_as(barriers, fifo.stamp_index(), &|| fifo.is_quiesced())
    }

    /// [`DeviceCore::wait_stamps`] for the waiter `waiter` (the waiting FIFO's stamp index),
    /// whose wait ends once `quiesced` answers true.
    pub(crate) fn wait_stamps_as(
        &self,
        barriers: Records<'_, StampWait>,
        waiter: u32,
        quiesced: &dyn Fn() -> bool,
    ) -> RaiseResult<bool> {
        for barrier in barriers.iter() {
            // `maxFIFOCount`, read at each barrier.
            let max = self.features.load().max_fifo_count;
            if barrier.index >= max {
                crate::raise!(
                    DeviceRaise::InvalidStampIndex,
                    format_args!(
                        "stamp barrier index {} is not below the FIFO count {max}",
                        barrier.index
                    )
                );
            }
            // DIAG(stamp-wait): a barrier that holds its FIFO 2 ms or more is written at debug.
            let timed = log::log_enabled!(log::Level::Debug).then(std::time::Instant::now);
            let passed = self.wait_stamp(barrier, waiter, quiesced)?;
            if let Some(t) = timed {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if ms >= 2.0 {
                    log::debug!(
                        "stamp barrier: FIFO {waiter} waited {ms:.2} ms for stamp {} to reach {:#x}",
                        barrier.index,
                        barrier.value
                    );
                }
            }
            if !passed {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Blocks until stamp slot `stamp.index` reaches `stamp.value` or the FIFO is quiesced;
    /// answers `false` when it ended quiesced. Each 2 s without a wake it logs, asks the
    /// reporting queue for a report, then waits untimed for the next wake.
    /// DIAG(empty-entry): stamp slot `index`'s value now; `None` without stamp memory.
    pub(crate) fn current_stamp(&self, index: u32) -> Option<u32> {
        let stamp_memory: GuestPtr = (*self.stamp_memory.lock())?;
        Some(stamp_memory.read_u32(
            u64::from(index).wrapping_mul(4),
            DeviceDeath::StampReadWithoutMemory,
        ))
    }

    fn wait_stamp(
        &self,
        stamp: StampWait,
        waiter: u32,
        is_quiesced: &dyn Fn() -> bool,
    ) -> RaiseResult<bool> {
        let platform = self.platform.as_ref();
        // Read once; NULL dies at the first read below.
        let stamp_memory: GuestPtr = self.stamp_memory.lock().unwrap_or_default();
        let slot_offset = u64::from(stamp.index).wrapping_mul(4);
        let slot = || stamp_memory.read_u32(slot_offset, DeviceDeath::StampReadWithoutMemory);
        if reached(slot(), stamp.value) {
            return Ok(true);
        }
        // May die at a check or raise for a cycle.
        self.add_wait(waiter, stamp.index, stamp.value)?;
        let mut quiesced = false;
        {
            let mut lock = self.signal_mutex.lock();
            while !reached(slot(), stamp.value) {
                quiesced = is_quiesced();
                if quiesced {
                    break;
                }
                let clock = platform.clock();
                let deadline = Deadline::Mono(clock.now_mono().saturating_add(STAMP_WAIT_TIMEOUT));
                if clock
                    .wait_until(&self.signal_cond, &mut lock, deadline)
                    .timed_out()
                {
                    // Each line's slot is read again for it, whether or not the line is
                    // written: the read can die.
                    if reached(slot(), stamp.value) {
                        let current = slot();
                        log::debug!(
                            "waiter {waiter}'s wait timed out as stamp {} reached {:#x} \
                             ({current:#x})",
                            stamp.index,
                            stamp.value
                        );
                        break;
                    }
                    let current = slot();
                    log::warn!(
                        "waiter {waiter} waited {STAMP_WAIT_TIMEOUT:?} for stamp {} to reach \
                         {:#x}; it holds {current:#x}",
                        stamp.index,
                        stamp.value
                    );
                    // The report runs on the reporting queue, coalesced.
                    self.report_stamp_wait_timeout_source.merge(1);
                    // Untimed, until the next wake.
                    clock.wait(&self.signal_cond, &mut lock);
                }
            }
        }
        self.finish_wait(waiter);
        Ok(!quiesced)
    }

    /// Registers `waiter` as waiting on stamp `stamp_index` for `value`, then walks the chain
    /// of waits it closes: a chain back to `waiter` finishes the wait and raises. No lock.
    ///
    /// The waiter mask is `1 << waiter` in 32 bits, so waiters 32 apart share a bit (the second
    /// dies at the check) and a cycle among waiters of 32 and up is never found (its walk tests
    /// bits 32-63, which are never set).
    pub fn add_wait(&self, fifo_index: u32, stamp_index: u32, value: u32) -> RaiseResult<()> {
        let waiter = fifo_index;
        crate::fatal_unless!(
            waiter < self.features.load().max_fifo_count,
            DeviceDeath::WaiterPastMax
        );
        let mask = num::bit32(waiter);
        // The 64-bit word, the mask zero-extended.
        crate::fatal_unless!(
            u64::from(mask) & self.waiting_stamps.load(Ordering::SeqCst) == 0,
            DeviceDeath::WaiterBitTaken
        );
        let Some(mut curr) = self.waiting_info.get(waiter.to_usize()) else {
            crate::fatal!(DeviceDeath::PastWaitingInfo)
        };
        crate::fatal_unless!(
            curr.next_waiting_index.load(Ordering::Relaxed) == NOT_WAITING,
            DeviceDeath::WaiterBusy
        );
        curr.next_waiting_index
            .store(stamp_index, Ordering::Relaxed);
        curr.next_waiting_value.store(value, Ordering::Relaxed);
        self.waiting_stamps
            .fetch_or(u64::from(mask), Ordering::SeqCst);
        // The walk has no bound.
        loop {
            let next = curr.next_waiting_index.load(Ordering::Relaxed);
            let Some(next_info) = self.waiting_info.get(next.to_usize()) else {
                log::warn!(
                    "waiter {waiter}'s wait chain reaches stamp index {next}, past the waiters"
                );
                return Ok(());
            };
            // A 64-bit shift.
            if !num::bit64_set(self.waiting_stamps.load(Ordering::SeqCst), next) {
                return Ok(());
            }
            if reached(
                next_info.last_pending_stamp.load(Ordering::Relaxed),
                curr.next_waiting_value.load(Ordering::Relaxed),
            ) {
                return Ok(());
            }
            curr = next_info;
            if next == waiter {
                self.finish_wait(waiter);
                crate::raise!(
                    DeviceRaise::CircularWait,
                    format_args!("waiter {waiter}'s wait closes a cycle of waits")
                );
            }
        }
    }

    /// Clears the waiter's bit and idles its entry.
    ///
    /// The mask `!(1 << waiter)` is a 32-bit signed value sign-extended to 64 bits: for a waiter
    /// ≡ 31 mod 32 it clears bits 31-63, otherwise just the waiter's bit.
    pub fn finish_wait(&self, fifo_index: u32) {
        let waiter = fifo_index;
        self.waiting_stamps
            .fetch_and(finish_wait_mask(waiter), Ordering::SeqCst);
        let Some(info) = self.waiting_info.get(waiter.to_usize()) else {
            crate::fatal!(DeviceDeath::PastWaitingInfo)
        };
        info.next_waiting_index
            .store(NOT_WAITING, Ordering::Relaxed);
    }

    /// Whether a FIFO carries `stamp_index`, under the caller's root FIFO lock (`root`): stamp 0
    /// is the root FIFO, which must exist; any other is the child in that slot, which must
    /// carry that stamp index.
    fn fifo_for_stamp_index(&self, root: &RootFifoLocked<P>, stamp_index: u32) -> bool {
        crate::fatal_unless!(
            stamp_index < APV_MAX_FIFOS,
            DeviceDeath::ReportStampPastFifos
        );
        if stamp_index == 0 {
            crate::fatal_unless!(root.root_fifo.is_some(), DeviceDeath::ReportRootWithoutFifo);
            return true;
        }
        // Read without a lock of its own.
        let child = self
            .child_fifos
            .lock()
            .get(stamp_index.to_usize())
            .cloned()
            .flatten();
        let Some(fifo) = &child else {
            log::warn!("stamp wait report: no FIFO carries stamp index {stamp_index}");
            return false;
        };
        crate::fatal_unless!(
            fifo.stamp_index() == stamp_index,
            DeviceDeath::ReportStampMismatch
        );
        true
    }

    /// Runs on the reporting queue (it dies elsewhere) under the root FIFO's lock: logs each
    /// waiting stamp's wait, then each FIFO's pending stamps. Its only effects besides the logs
    /// are its deaths (the stamp-index checks, the stamp memory read for a wait's line, whether
    /// or not the line is written).
    ///
    /// The waiting word is walked with the 32-bit `1 << i`, and the waiters are read by stamp
    /// index while their last pending stamps are filled by FIFO id: a waiter above 31 reports
    /// through its alias's entry ("waiting on invalid next waitingIndex", then stop).
    pub fn report_stamp_wait_timeout(&self) {
        let root = self.root_fifo.lock();
        if !self.reporting_queue.is_current() {
            crate::fatal!(DeviceDeath::ReportOffQueue)
        }
        let mut waiting_stamps = self.waiting_stamps.load(Ordering::SeqCst);
        while waiting_stamps != 0 {
            let stamp_index = waiting_stamps.trailing_zeros();
            if self.fifo_for_stamp_index(&root, stamp_index) {
                // stamp_index < 64 here (the check above).
                let (waiting_index, waiting_value) = self
                    .waiting_info
                    .get(stamp_index.to_usize())
                    .map_or((NOT_WAITING, 0), |info| {
                        (
                            info.next_waiting_index.load(Ordering::Relaxed),
                            info.next_waiting_value.load(Ordering::Relaxed),
                        )
                    });
                let Some(waiting) = self.waiting_info.get(waiting_index.to_usize()) else {
                    log::warn!(
                        "stamp wait report: waiter {stamp_index} waits on stamp index \
                         {waiting_index}, past the waiters"
                    );
                    break;
                };
                let pending_stamp = waiting.last_pending_stamp.load(Ordering::Relaxed);
                // The stamp memory, read unlocked whether or not the line is written; NULL
                // dies here.
                let current = {
                    let stamp_memory = self.stamp_memory.lock().unwrap_or_default();
                    stamp_memory.read_u32(
                        u64::from(waiting_index).wrapping_mul(4),
                        DeviceDeath::ReportWithoutStampMemory,
                    )
                };
                log::warn!(
                    "stamp wait report: waiter {stamp_index} waits for stamp {waiting_index} \
                     to reach {waiting_value:#x}; it holds {current:#x}, and its FIFO pushed \
                     up to {pending_stamp:#x}"
                );
            }
            waiting_stamps &= finish_wait_mask(stamp_index);
        }
        if let Some(root_fifo) = &root.root_fifo {
            let pending = root_fifo.describe_pending();
            if !pending.is_empty() {
                log::warn!("stamp wait report: root FIFO pending:\n{pending}");
            } else {
                log::warn!("stamp wait report: root FIFO has nothing pending");
            }
        }
        for index in 1u32..64 {
            // Each slot is read, without a lock of its own, as the loop reaches it: a FIFO
            // the FIFO queue creates or deletes meanwhile is listed as it is then. The FIFO
            // is held for the listing alone.
            let fifo = self
                .child_fifos
                .lock()
                .get(index.to_usize())
                .cloned()
                .flatten();
            let Some(fifo) = fifo else {
                continue;
            };
            let pending = fifo.describe_pending();
            if !pending.is_empty() {
                log::warn!("stamp wait report: FIFO {index} pending:\n{pending}");
            } else {
                log::warn!("stamp wait report: FIFO {index} has nothing pending");
            }
        }
        drop(root);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::device::test_support::{Lines, Opts, TestDevice, device};
    use crate::raise::RaiseId;
    use crate::raise::death::DeathSite;
    use crate::testkit::capture::catch_death;
    use crate::testkit::capture::expect_death;
    use std::sync::Arc;

    /// A waiter that never quiesces.
    fn awake() -> bool {
        false
    }

    fn records(waits: &[StampWait]) -> Vec<u8> {
        use zerocopy::IntoBytes;
        waits.as_bytes().to_vec()
    }

    fn dev() -> TestDevice {
        device(Opts::default())
    }

    fn assert_death(r: Result<impl std::fmt::Debug, DeathSite>, death: DeviceDeath) {
        assert_eq!(r.expect_err("expected a death"), death.site());
    }

    /// The raise's id, under a handler.
    fn raised<T: std::fmt::Debug>(body: impl FnOnce() -> RaiseResult<T>) -> DeviceRaise {
        match body().expect_err("expected a raise").id() {
            RaiseId::Device(k) => k,
            other => panic!("not the device's raise: {other:?}"),
        }
    }

    mod init_waiting_info {
        use super::*;

        /// every entry idle with no pending stamp; the values and the waiting word untouched
        #[test]
        fn resets_indices_and_pending_only() {
            let t = dev();
            let d = &t.device;
            for (i, info) in d.waiting_info.iter().enumerate() {
                info.next_waiting_index.store(i as u32, Ordering::Relaxed);
                info.next_waiting_value
                    .store(0x100 + i as u32, Ordering::Relaxed);
                info.last_pending_stamp.store(9, Ordering::SeqCst);
            }
            d.waiting_stamps.store(0x5, Ordering::SeqCst);
            d.init_waiting_info();
            assert_eq!(d.waiting_info.len(), 64);
            for (i, info) in d.waiting_info.iter().enumerate() {
                assert_eq!(info.next_waiting_index.load(Ordering::Relaxed), u32::MAX);
                assert_eq!(info.last_pending_stamp.load(Ordering::SeqCst), 0);
                assert_eq!(
                    info.next_waiting_value.load(Ordering::Relaxed),
                    0x100 + i as u32
                );
            }
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 0x5);
        }
    }

    mod add_wait {
        use super::*;

        /// registers the wait and returns when the awaited stamp's FIFO is not waiting
        #[test]
        fn registers() {
            let t = dev();
            let d = &t.device;
            d.add_wait(3, 5, 0x77).unwrap();
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 1 << 3);
            assert_eq!(
                d.waiting_info[3].next_waiting_index.load(Ordering::Relaxed),
                5
            );
            assert_eq!(
                d.waiting_info[3].next_waiting_value.load(Ordering::Relaxed),
                0x77
            );
        }

        /// a waiter at or above maxFIFOCount dies
        #[test]
        fn waiter_above_count_dies() {
            let t = device(Opts {
                max_fifo_count: 32,
                ..Opts::default()
            });
            assert_death(
                catch_death(|| t.device.add_wait(32, 0, 1)),
                DeviceDeath::WaiterPastMax,
            );
        }

        /// a second wait under the same bit dies
        #[test]
        fn same_waiter_twice_dies() {
            let t = dev();
            t.device.add_wait(1, 2, 1).unwrap();
            assert_death(
                catch_death(|| t.device.add_wait(1, 2, 1)),
                DeviceDeath::WaiterBitTaken,
            );
        }

        /// waiters 32 apart share a bit: the second dies
        #[test]
        fn alias_32_apart_dies() {
            let t = dev();
            t.device.add_wait(1, 2, 1).unwrap();
            assert_death(
                catch_death(|| t.device.add_wait(33, 2, 1)),
                DeviceDeath::WaiterBitTaken,
            );
        }

        /// an entry still busy with its bit clear dies
        #[test]
        fn busy_entry_dies() {
            let t = dev();
            t.device.waiting_info[4]
                .next_waiting_index
                .store(7, Ordering::Relaxed);
            assert_death(
                catch_death(|| t.device.add_wait(4, 2, 1)),
                DeviceDeath::WaiterBusy,
            );
        }

        /// a next index of 64 or more logs and leaves the wait registered
        #[test]
        fn invalid_next_index_logs() {
            let t = dev();
            let lines = Lines::capture();
            t.device.add_wait(2, 64, 1).unwrap();
            assert_eq!(
                lines.all(),
                ["waiter 2's wait chain reaches stamp index 64, past the waiters"]
            );
            assert_eq!(t.device.waiting_stamps.load(Ordering::SeqCst), 1 << 2);
        }

        /// the walk stops where the awaited FIFO has already pushed the value
        #[test]
        fn stops_at_pending_stamp() {
            let t = dev();
            let d = &t.device;
            d.add_wait(1, 2, 10).unwrap(); // 1 waits on 2:10
            d.waiting_info[1]
                .last_pending_stamp
                .store(5, Ordering::Relaxed);
            // 2 waits on 1:5, and 1 has pushed 5: no cycle
            d.waiting_info[2]
                .last_pending_stamp
                .store(0, Ordering::Relaxed);
            d.add_wait(2, 1, 5).unwrap();
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 0b110);
        }

        /// a cycle back to the waiter finishes its wait, then raises
        #[test]
        fn circular_wait_raises() {
            let t = dev();
            let d = &t.device;
            d.add_wait(1, 2, 10).unwrap();
            assert_eq!(raised(|| d.add_wait(2, 1, 10)), DeviceRaise::CircularWait);
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 1 << 1);
            assert_eq!(
                d.waiting_info[2].next_waiting_index.load(Ordering::Relaxed),
                u32::MAX
            );
        }

        /// the raise names the circular wait's site, which its death would name
        #[test]
        fn circular_wait_raise_names_its_site() {
            let t = dev();
            t.device.add_wait(1, 2, 10).unwrap();
            let r = t.device.add_wait(2, 1, 10).expect_err("a raise");
            assert_eq!(r.id(), RaiseId::Device(DeviceRaise::CircularWait));
            let d = r.site();
            assert_eq!(d.kind, crate::raise::death::DeathKind::UnhandledRaise);
            assert_eq!(d.file, DeviceRaise::CircularWait.name());
        }

        /// a cycle among waiters 32 and up is never found
        #[test]
        fn high_waiters_cycle_undetected() {
            let t = dev();
            let d = &t.device;
            d.add_wait(33, 34, 1).unwrap();
            d.add_wait(34, 33, 1).unwrap();
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 0b110);
        }

        /// the walk reads entries by stamp index: a waiter whose slot is its own chain's target
        /// raises at once
        #[test]
        fn self_wait_raises() {
            let t = dev();
            assert_eq!(
                raised(|| t.device.add_wait(3, 3, 1)),
                DeviceRaise::CircularWait
            );
        }
    }

    mod finish_wait {
        use super::*;

        /// clears the waiter's bit and idles its entry
        #[test]
        fn clears_bit() {
            let t = dev();
            let d = &t.device;
            d.waiting_stamps.store(0b1010, Ordering::SeqCst);
            d.waiting_info[3]
                .next_waiting_index
                .store(1, Ordering::Relaxed);
            d.finish_wait(3);
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 0b0010);
            assert_eq!(
                d.waiting_info[3].next_waiting_index.load(Ordering::Relaxed),
                u32::MAX
            );
        }

        /// waiter 31's mask sign-extends: bits 31-63 are cleared
        #[test]
        fn waiter_31_clears_high_bits() {
            let t = dev();
            let d = &t.device;
            d.waiting_stamps
                .store((1 << 40) | (1 << 31) | 1, Ordering::SeqCst);
            d.finish_wait(31);
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 1);
        }

        /// waiter 35 clears its alias bit 3
        #[test]
        fn waiter_35_clears_alias() {
            let t = dev();
            let d = &t.device;
            d.waiting_stamps.store(0b1001, Ordering::SeqCst);
            d.finish_wait(35);
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 0b0001);
            assert_eq!(
                d.waiting_info[35]
                    .next_waiting_index
                    .load(Ordering::Relaxed),
                u32::MAX
            );
        }

        /// the mask arithmetic
        #[test]
        fn masks() {
            assert_eq!(finish_wait_mask(0), !1u64);
            assert_eq!(finish_wait_mask(31), 0x7fff_ffff);
            assert_eq!(finish_wait_mask(32), !1u64);
            assert_eq!(finish_wait_mask(63), 0x7fff_ffff);
        }
    }

    mod wait_stamps {
        use super::*;
        use std::time::Duration;

        const REAL: Duration = Duration::from_secs(10);

        /// an index at or past maxFIFOCount raises
        #[test]
        fn invalid_index_raises() {
            let t = device(Opts {
                max_fifo_count: 32,
                ..Opts::default()
            })
            .with_stamp_memory();
            let bytes = records(&[StampWait {
                index: 32,
                value: 0,
            }]);
            assert_eq!(
                raised(|| t
                    .device
                    .wait_stamps_as(Records::new(&bytes, 1).unwrap(), 1, &awake)),
                DeviceRaise::InvalidStampIndex
            );
        }

        /// a slot already at the value passes with no registration
        #[test]
        fn already_reached() {
            let t = dev().with_stamp_memory();
            t.set_stamp(4, 10, false);
            let bytes = records(&[
                StampWait {
                    index: 4,
                    value: 10,
                },
                StampWait { index: 4, value: 9 },
                // 2^31 - 1 behind the slot: signed, so it has passed
                StampWait {
                    index: 4,
                    value: 10u32.wrapping_sub(0x7fff_ffff),
                },
            ]);
            let r = t
                .device
                .wait_stamps_as(Records::new(&bytes, 3).unwrap(), 1, &awake);
            assert!(r.unwrap());
            assert_eq!(t.device.waiting_stamps.load(Ordering::SeqCst), 0);
        }

        /// a slot 2^31 short of the value has not reached it: the wait blocks
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time polling
        fn wrap_blocks_until_signalled() {
            let t = Arc::new(dev().with_stamp_memory());
            t.set_stamp(4, 0, false);
            let t2 = t.clone();
            let h = std::thread::spawn(move || {
                let bytes = records(&[StampWait {
                    index: 4,
                    value: 0x8000_0000,
                }]);
                t2.device
                    .wait_stamps_as(Records::new(&bytes, 1).unwrap(), 1, &awake)
                    .unwrap()
            });
            assert!(t.platform.virtual_clock().wait_for_waiters(1, REAL));
            assert_eq!(t.device.waiting_stamps.load(Ordering::SeqCst), 1 << 1);
            assert_eq!(
                t.device.waiting_info[1]
                    .next_waiting_index
                    .load(Ordering::Relaxed),
                4
            );
            t.set_stamp(4, 0x8000_0000, true);
            assert!(h.join().unwrap());
            assert_eq!(t.device.waiting_stamps.load(Ordering::SeqCst), 0);
        }

        /// a quiesced FIFO ends its wait with `false` and finishes it; later barriers are skipped
        #[test]
        fn quiesced_returns_false() {
            let t = dev().with_stamp_memory();
            let bytes = records(&[
                StampWait { index: 5, value: 1 },
                StampWait {
                    index: 64,
                    value: 1,
                },
            ]);
            let r = t
                .device
                .wait_stamps_as(Records::new(&bytes, 2).unwrap(), 2, &|| true);
            assert!(!r.unwrap());
            assert_eq!(t.device.waiting_stamps.load(Ordering::SeqCst), 0);
            assert_eq!(
                t.device.waiting_info[2]
                    .next_waiting_index
                    .load(Ordering::Relaxed),
                u32::MAX
            );
        }

        /// the registration's raise for a cycle leaves the wait
        #[test]
        fn circular_wait_propagates() {
            let t = dev().with_stamp_memory();
            let bytes = records(&[StampWait { index: 3, value: 1 }]);
            assert_eq!(
                raised(|| t
                    .device
                    .wait_stamps_as(Records::new(&bytes, 1).unwrap(), 3, &awake)),
                DeviceRaise::CircularWait
            );
        }

        /// no stamp memory: the first slot read faults
        #[test]
        fn null_stamp_memory_dies() {
            let t = dev();
            let bytes = records(&[StampWait { index: 1, value: 1 }]);
            assert_death(
                catch_death(|| {
                    t.device
                        .wait_stamps_as(Records::new(&bytes, 1).unwrap(), 1, &awake)
                }),
                DeviceDeath::StampReadWithoutMemory,
            );
        }

        /// after 2 s it logs, asks for the report, then waits untimed for the next wake
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time polling
        fn timeout_reports_then_waits() {
            let t = Arc::new(dev().with_stamp_memory());
            let lines = Lines::capture();
            let t2 = t.clone();
            let h = std::thread::spawn(move || {
                let bytes = records(&[StampWait { index: 6, value: 2 }]);
                t2.device
                    .wait_stamps_as(Records::new(&bytes, 1).unwrap(), 1, &awake)
                    .unwrap()
            });
            let clock = t.platform.virtual_clock();
            assert!(clock.wait_for_waiters(1, REAL));
            clock.advance(Duration::from_secs(2));
            // the report ran on the reporting queue: waiter 1 has no child FIFO
            let deadline = std::time::Instant::now() + REAL;
            while !lines.contains("no FIFO carries stamp index 1") {
                assert!(std::time::Instant::now() < deadline, "no report");
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(
                lines.all(),
                [
                    "waiter 1 waited 2s for stamp 6 to reach 0x2; it holds 0x0",
                    "stamp wait report: no FIFO carries stamp index 1"
                ]
            );
            // untimed now: more time passes and it stays blocked
            clock.advance(Duration::from_secs(10));
            std::thread::sleep(Duration::from_millis(20));
            assert!(!h.is_finished());
            t.set_stamp(6, 2, true);
            assert!(h.join().unwrap());
        }

        /// a timeout that finds the slot complete logs and ends the wait
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time polling
        fn timeout_finds_complete() {
            let t = Arc::new(dev().with_stamp_memory());
            let lines = Lines::capture();
            let t2 = t.clone();
            let h = std::thread::spawn(move || {
                let bytes = records(&[StampWait { index: 6, value: 2 }]);
                t2.device
                    .wait_stamps_as(Records::new(&bytes, 1).unwrap(), 1, &awake)
                    .unwrap()
            });
            let clock = t.platform.virtual_clock();
            assert!(clock.wait_for_waiters(1, REAL));
            t.set_stamp(6, 3, false); // written, no wake
            clock.advance(Duration::from_secs(2));
            assert!(h.join().unwrap());
            // No report was asked for.
            assert_eq!(
                lines.all(),
                ["waiter 1's wait timed out as stamp 6 reached 0x2 (0x3)"]
            );
        }
    }

    mod report {
        use super::*;

        fn report(t: &TestDevice) {
            let d = t.device.clone();
            t.device
                .reporting_queue
                .sync(move || d.report_stamp_wait_timeout());
        }

        /// off the reporting queue it dies
        #[test]
        fn off_queue_dies() {
            let t = dev();
            let d = t.device.clone();
            expect_death(DeviceDeath::ReportOffQueue.site(), move || {
                d.report_stamp_wait_timeout()
            });
        }

        /// nothing waiting and no FIFOs: nothing logged
        #[test]
        fn idle_logs_nothing() {
            let t = dev();
            let lines = Lines::capture();
            report(&t);
            assert!(lines.all().is_empty());
        }

        /// a waiter with no FIFO under its stamp index is logged as missing
        #[test]
        fn missing_child_logged() {
            let t = dev();
            t.device.waiting_stamps.store(1 << 5, Ordering::SeqCst);
            let lines = Lines::capture();
            report(&t);
            assert_eq!(
                lines.all(),
                ["stamp wait report: no FIFO carries stamp index 5"]
            );
        }

        /// waiter 33's bit is bit 1, so the report walks child 1's entry, which waits on nothing
        #[test]
        fn alias_above_31_walks_low_entry() {
            let t = dev().with_stamp_memory().with_root_page();
            // child node 1 (root + 0x400) names stamp index 1 (u16 at +0xc)
            t.set_guest_root_word((0x400 + 0xc) / 4, 1);
            assert!(t.device.create_fifo_internal(1, true));
            t.device.add_wait(33, 2, 5).unwrap();
            assert_eq!(t.device.waiting_stamps.load(Ordering::SeqCst), 1 << 1);
            let lines = Lines::capture();
            report(&t);
            assert_eq!(
                lines.all(),
                [
                    "stamp wait report: waiter 1 waits on stamp index 4294967295, past the \
                     waiters",
                    "stamp wait report: FIFO 1 has nothing pending"
                ]
            );
        }

        /// A waiter on a valid stamp: its line, with the awaited stamp's slot read for it, then
        /// each child's pending list.
        #[test]
        fn waiting_child_logs_its_lines() {
            let t = dev().with_stamp_memory().with_root_page();
            // child node 1 (root + 0x400) names stamp index 1 (u16 at +0xc)
            t.set_guest_root_word((0x400 + 0xc) / 4, 1);
            assert!(t.device.create_fifo_internal(1, true));
            t.set_stamp(2, 3, false);
            t.device.add_wait(1, 2, 5).unwrap();
            let pending = t.device.waiting_info[2]
                .last_pending_stamp
                .load(Ordering::Relaxed);
            let lines = Lines::capture();
            report(&t);
            assert_eq!(
                lines.all(),
                [
                    format!(
                        "stamp wait report: waiter 1 waits for stamp 2 to reach 0x5; it holds \
                         0x3, and its FIFO pushed up to {pending:#x}"
                    ),
                    "stamp wait report: FIFO 1 has nothing pending".to_owned()
                ]
            );
        }

        /// Without stamp memory, a waiter's report dies reading the awaited slot for its line,
        /// whether or not the line is written (here nothing listens)
        #[test]
        fn waiting_child_without_stamp_memory_dies_for_its_line() {
            let t = dev().with_root_page();
            t.set_guest_root_word((0x400 + 0xc) / 4, 1);
            assert!(t.device.create_fifo_internal(1, true));
            t.device.add_wait(1, 2, 5).unwrap();
            assert_death(
                catch_death(|| report(&t)),
                DeviceDeath::ReportWithoutStampMemory,
            );
        }

        /// The report opens no pool of its own: nothing in it is autoreleased.
        #[test]
        fn the_report_opens_no_pool() {
            use crate::rt::pool::test_native::{events, with_test_native};
            let t = dev().with_stamp_memory().with_root_page();
            t.set_guest_root_word((0x400 + 0xc) / 4, 1);
            assert!(t.device.create_fifo_internal(1, true));
            let mut got = Vec::new();
            with_test_native(|| {
                report(&t);
                got = events();
            });
            assert!(got.is_empty(), "{got:?}");
        }

        /// Each child slot is read as the loop reaches it. While the report waits in FIFO 2's
        /// pending list (its queue mutex held by a pump inside the stamp's signal), the FIFO
        /// queue creates FIFO 3 and deletes FIFO 4: the report lists 3 and not 4.
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: threads, real-time waits
        fn each_child_slot_is_read_when_the_loop_reaches_it() {
            use std::sync::mpsc;
            use std::time::{Duration, Instant};
            let t = Arc::new(dev().with_stamp_memory().with_root_page());
            let lines = Lines::capture();
            // Child node n (root + 0x400 + (n - 1)·0x14) names stamp index n (u16 at +0xc).
            for n in 1u32..=4 {
                t.set_guest_root_word(u64::from(0x400 + (n - 1) * 0x14 + 0xc) / 4, n);
                assert!(t.device.create_fifo_internal(n, true));
            }
            let slot = |n: usize| t.device.child_fifos.lock()[n].clone().unwrap();
            let (fifo2, fifo3, fifo4) = (slot(2), slot(3), slot(4));
            // FIFO 3 is made later (a child FIFO created mid-report).
            t.device.child_fifos.lock()[3] = None;
            // A pump of FIFO 2 holds its queue mutex, stopped at the stamp's signal lock.
            let signal = t.device.signal_mutex.lock();
            let pump = {
                let fifo2 = fifo2.clone();
                std::thread::spawn(move || {
                    fifo2.push_stamp(7, 0);
                    fifo2.signal_stamp_value(7);
                })
            };
            // Wait until a pending list of FIFO 2 no longer gets its mutex.
            let start = Instant::now();
            let mut probes = Vec::new();
            loop {
                assert!(start.elapsed() < Duration::from_secs(10), "no pump");
                let (tx, rx) = mpsc::channel();
                let f = fifo2.clone();
                probes.push(std::thread::spawn(move || {
                    let _ = f.describe_pending();
                    let _ = tx.send(());
                }));
                if rx.recv_timeout(Duration::from_millis(50)).is_err() {
                    break;
                }
            }
            let report_thread = {
                let t = t.clone();
                std::thread::spawn(move || report(&t))
            };
            // The loop has read slot 1 and now waits in FIFO 2's pending list.
            while !lines.contains("FIFO 1 has nothing pending") {
                assert!(start.elapsed() < Duration::from_secs(10), "no report");
                std::thread::sleep(Duration::from_millis(1));
            }
            std::thread::sleep(Duration::from_millis(20));
            assert!(!report_thread.is_finished());
            t.device.child_fifos.lock()[3] = Some(fifo3.clone());
            let deleted = t.device.child_fifos.lock()[4].take();
            drop(signal);
            pump.join().unwrap();
            report_thread.join().unwrap();
            for p in probes {
                p.join().unwrap();
            }
            assert!(lines.contains("FIFO 1 has nothing pending"));
            assert!(lines.contains("FIFO 3 has nothing pending"));
            assert!(!lines.contains("FIFO 4"));
            drop((deleted, fifo4));
        }

        /// stamp 0 waiting with no root FIFO dies
        #[test]
        fn root_waiter_without_root_dies() {
            let t = dev();
            t.device.waiting_stamps.store(1, Ordering::SeqCst);
            let d = t.device.clone();
            expect_death(DeviceDeath::ReportRootWithoutFifo.site(), move || {
                let d2 = d.clone();
                d.reporting_queue
                    .sync(move || d2.report_stamp_wait_timeout());
            });
        }
    }

    /// Width and order edges.
    mod edges {
        use super::*;

        /// `(int32_t)(slot - value) >= 0` flips at a distance of 2^31
        #[test]
        fn reached_boundary() {
            assert!(reached(0x7fff_ffff, 0));
            assert!(!reached(0x8000_0000, 0));
            assert!(reached(0, 0x8000_0001));
            assert!(!reached(0, 0x8000_0000));
        }

        /// a passed barrier, then an invalid index: the raise with nothing left waiting
        #[test]
        fn passed_then_invalid_raises() {
            let t = device(Opts {
                max_fifo_count: 32,
                ..Opts::default()
            })
            .with_stamp_memory();
            t.set_stamp(4, 10, false);
            let bytes = records(&[
                StampWait {
                    index: 4,
                    value: 10,
                },
                StampWait {
                    index: 32,
                    value: 0,
                },
                StampWait { index: 5, value: 1 },
            ]);
            assert_eq!(
                raised(|| t
                    .device
                    .wait_stamps_as(Records::new(&bytes, 3).unwrap(), 1, &awake)),
                DeviceRaise::InvalidStampIndex
            );
            assert_eq!(t.device.waiting_stamps.load(Ordering::SeqCst), 0);
        }

        /// waiter 63 (31 mod 32) clears bits 31-63 only
        #[test]
        fn waiter_63_clears_high() {
            let t = dev();
            let d = &t.device;
            d.waiting_stamps
                .store(0x8000_0000 | 0x4000_0000, Ordering::SeqCst);
            d.waiting_info[63]
                .next_waiting_index
                .store(2, Ordering::Relaxed);
            d.finish_wait(63);
            assert_eq!(d.waiting_stamps.load(Ordering::SeqCst), 0x4000_0000);
            assert_eq!(
                d.waiting_info[63]
                    .next_waiting_index
                    .load(Ordering::Relaxed),
                u32::MAX
            );
        }
    }
}
