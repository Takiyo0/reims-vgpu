//! A FIFO's thread: its start, its stop and drain, and the loop that reads each command off the
//! ring, waits for its barriers, publishes its bytes, pushes its stamp and runs it.
//!
//! The thread (512 KiB of stack) is named `reims-vgpu.fifo`; its target holds the FIFO until the
//! thread's end, and the loop holds another reference. What is autoreleased on the thread outside
//! a command's pool, or held under a hold with no holder (the deprecated opcodes', a signal's
//! FIFO), goes to the thread's last-resort pool and holder, released at the loop's end, before
//! the target is dropped (`rt::thread::spawn_named`).

#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::fatal_unless;
use crate::fifo::wire::{
    BARRIER_BYTES_MAX, COMMAND_HEADER_SIZE, FifoCommand, HEADER_LEN, MAX_BARRIERS, Records,
    STAMP_WAIT_LEN, StampWait,
};
use crate::fifo::{DeathKind, Fifo};
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::raise::RaiseResult;
use crate::rt::Hold;
use crate::rt::thread::{FIFO_THREAD_STACK, spawn_named};

/// What the loop does with a command whose header it has read.
enum Framing {
    /// Skipped: no stamp, no fault, no advance.
    Skip,
    /// Its stamp is pushed and faulted; only the header is consumed.
    Fault,
    /// Its barriers and payload follow.
    Read {
        barrier_count: u16,
        barrier_len: usize,
        payload_len: usize,
    },
}

impl<P: Platform> Fifo<P> {
    /// Starts the FIFO's thread. `false` when a child's ring cannot be mapped, before the thread
    /// and before the node's running flag. There is no guard against a second start.
    pub fn start(self: &Arc<Self>, platform: &Arc<P>) -> bool {
        let _ = platform;
        // A failed start leaves the quiesced flag as it is.
        if !self.map_ring() {
            return false;
        }
        self.quiesced.store(false, Ordering::SeqCst);
        // The fault latch stays as it is
        self.stamps.clear_quiesced();
        self.running.store(true, Ordering::Relaxed);
        // A start over a live thread spawns a second one on the same state. The old thread,
        // woken, sees the FIFO unquiesced and runs on. A child's new thread starts from the
        // node's read count while the FIFO's stays where the old thread left it. A root's new
        // thread reads on from the shared cursors.
        log::info!(
            "FIFO channel {}: thread spawning (is_root={})",
            self.stamp_index(),
            self.is_root_fifo()
        );
        let me = Arc::clone(self);
        spawn_named(
            self.platform.threads(),
            "reims-vgpu.fifo",
            FIFO_THREAD_STACK,
            Box::new(Arc::clone(self)),
            move || me.run(),
        );
        if let crate::fifo::ring::Ring::Child(c) = &self.ring {
            // After the thread started
            c.set_running(true);
        }
        true
    }

    /// Marks a child's node stopped, then quiesces.
    pub fn stop(&self) {
        if let crate::fifo::ring::Ring::Child(c) = &self.ring {
            c.set_running(false);
        }
        self.quiesce();
    }

    /// Quiesces the FIFO and its stamps. It wakes no one: a thread waiting for ring bytes waits
    /// on until a poke or a write.
    pub fn quiesce(&self) {
        let _g = self.lock.lock();
        self.quiesced.store(true, Ordering::SeqCst);
        self.stamps.quiesce();
    }

    /// Wakes the thread waiting for ring bytes.
    pub fn poke(&self) {
        let _g = self.lock.lock();
        self.bytes.notify_one();
    }

    /// Waits for the thread's exit, then for every pending stamp. One wait for the exit, with no
    /// predicate; the stamps' drain hangs while one never completes.
    pub fn drain(&self) {
        fatal_unless!(
            self.quiesced.load(Ordering::SeqCst),
            DeathKind::DrainUnquiesced
        );
        self.decode.drain();
        {
            let mut g = self.lock.lock();
            if self.running.load(Ordering::Relaxed) {
                self.bytes.notify_all();
                self.exited.wait(&mut g);
            }
        }
        self.stamps.drain();
    }

    /// The barrier wait, with the device's raise.
    pub(super) fn wait_barriers(&self, barriers: Records<'_, StampWait>) -> RaiseResult<bool> {
        let Some(device) = self.device.upgrade() else {
            return Ok(false);
        };
        // The quiesced flag is read only once the waits are done
        Ok(device.wait_stamps(barriers, self)? && !self.quiesced.load(Ordering::SeqCst))
    }

    /// How the loop frames a command with this header, the ring being `ring_length` bytes.
    fn frame(&self, command: &FifoCommand, ring_length: u64) -> Framing {
        if command.length < HEADER_LEN || u64::from(command.length) > ring_length {
            // Skipped without a stamp, a fault or an advance.
            log::warn!(
                "channel {}: command {} of length {}, outside {HEADER_LEN}..={ring_length}; \
                 skipped",
                self.stamp_index(),
                command.command_id,
                command.length
            );
            return Framing::Skip;
        }
        let barrier_count = command.barrier_count;
        let barrier_bytes = u64::from(barrier_count).wrapping_mul(STAMP_WAIT_LEN);
        let (payload_len, underflow) = u64::from(command.length)
            .wrapping_sub(u64::from(HEADER_LEN))
            .overflowing_sub(barrier_bytes);
        if underflow {
            // Push and fault the header's stamp; only the header is consumed.
            log::warn!(
                "channel {}: command {} of length {} is too short for its {barrier_count} barriers",
                self.stamp_index(),
                command.command_id,
                command.length
            );
            return Framing::Fault;
        }
        if barrier_count > MAX_BARRIERS {
            // As the underflow.
            log::warn!(
                "channel {}: command {} has {barrier_count} barriers, more than {MAX_BARRIERS}",
                self.stamp_index(),
                command.command_id
            );
            return Framing::Fault;
        }
        Framing::Read {
            barrier_count,
            barrier_len: barrier_bytes.to_usize(),
            payload_len: payload_len.to_usize(),
        }
    }

    /// The thread's loop, until the FIFO is quiesced; then the unconsumed command is rewound, a
    /// child's ring unmapped, and a drain woken.
    ///
    /// The thread's body has no handler and no holder ([`Hold::NONE`]). The barrier wait has a
    /// handler of its own, and each command a holder of its own ([`Fifo::dispatch`]).
    fn run(self: &Arc<Self>) {
        crate::fifo::decode_worker::raise_priority();
        let hold = Hold::NONE;
        // One read of the length serves the buffer and the bound (only a racing second thread
        // could make two reads differ). The buffer is zero-allocated, so the pages no command
        // reaches are never touched: a ring is at most 4 GiB.
        let ring_length = self.ring_length.load(Ordering::Relaxed);
        let mut payload: Vec<u8> = vec![0u8; ring_length.to_usize()];
        let ring_length = u64::from(ring_length);
        let mut barriers = [0u8; BARRIER_BYTES_MAX];
        while !self.quiesced.load(Ordering::SeqCst) {
            self.latch_offset();
            let mut header = [0u8; COMMAND_HEADER_SIZE];
            if !self.read_bytes(&mut header) {
                continue;
            }
            let command: FifoCommand = zerocopy::transmute!(header);
            let (barrier_count, barrier_len, payload_len) = match self.frame(&command, ring_length)
            {
                Framing::Skip => continue,
                Framing::Fault => {
                    self.push_stamp(command.stamp_value, command.command_id);
                    self.fault_at_stamp_value(command.stamp_value);
                    continue;
                }
                Framing::Read {
                    barrier_count,
                    barrier_len,
                    payload_len,
                } => (barrier_count, barrier_len, payload_len),
            };
            if barrier_count != 0 {
                // barrier_len <= 512: at most 64 barriers
                let Some(dst) = barriers.get_mut(..barrier_len) else {
                    break;
                };
                if !self.read_bytes(dst) {
                    break;
                }
            }
            // payload_len < ring_length == payload.len()
            let Some(body) = payload.get_mut(..payload_len) else {
                continue;
            };
            if !self.read_bytes(body) {
                continue;
            }
            // DIAG(empty-entry): a SetObjectList (51) or ExecIndirect2 (55) with its stamp, its
            // barriers and the stamp slots 1 and 0 as they are now.
            // Debug only: nothing is built or asked of the device unless it is written.
            if (command.command_id == 51 || command.command_id == 55)
                && log::log_enabled!(log::Level::Debug)
            {
                let list = barriers.get(..barrier_len).unwrap_or(&[]);
                let waits: Vec<(u32, u32)> = list
                    .chunks_exact(8)
                    .map(|c| {
                        (
                            u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                            u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                        )
                    })
                    .collect();
                let now = self
                    .device
                    .upgrade()
                    .map(|d| (d.current_stamp(0), d.current_stamp(1)));
                let msg = format!(
                    "channel {} command {} stamp {:#x} barriers {waits:x?} slots(0,1) now {now:x?}",
                    self.stamp_index(),
                    command.command_id,
                    command.stamp_value
                );
                log::debug!("{msg}");
            }
            let waited = (|| -> RaiseResult<_> {
                if barrier_count == 0 {
                    return Ok(true);
                }
                let list = barriers.get(..barrier_len).unwrap_or(&[]);
                let records =
                    Records::new(list, u32::from(barrier_count)).unwrap_or_else(Records::empty);
                self.wait_barriers(records)
            })();
            match waited {
                // The command is not consumed: the rewind returns to it
                Ok(false) => break,
                Ok(true) => {}
                Err(r) => {
                    // The command is consumed, pushed and faulted.
                    log::warn!(
                        "channel {}: command {}'s barrier wait failed: {}",
                        self.stamp_index(),
                        command.command_id,
                        crate::fifo::commands::exception(&r)
                    );
                    self.advance();
                    self.push_stamp(command.stamp_value, command.command_id);
                    self.fault_at_stamp_value(command.stamp_value);
                    hold.land(r);
                    continue;
                }
            }
            // The read count publishes before the push, which may block.
            self.advance();
            if barrier_count != 0 {
                // bar: FIFO, stamp value: its barriers were met
                crate::trace::event(
                    "bar",
                    &[
                        u64::from(self.stamp_index()),
                        u64::from(command.stamp_value),
                    ],
                );
            }
            self.push_stamp(command.stamp_value, command.command_id);
            let body = payload.get(..payload_len).unwrap_or(&[]);
            self.dispatch(&command, body);
            // ran: FIFO, opcode, stamp value: its handler returned
            crate::trace::event(
                "ran",
                &[
                    u64::from(self.stamp_index()),
                    u64::from(command.command_id),
                    u64::from(command.stamp_value),
                ],
            );
        }
        drop(payload);
        // Every exec handed on finishes before the thread's end
        self.decode.drain();
        // Outside the lock
        self.rewind();
        let _g = self.lock.lock();
        self.unmap_ring();
        self.running.store(false, Ordering::Relaxed);
        self.exited.notify_all();
    }
}
