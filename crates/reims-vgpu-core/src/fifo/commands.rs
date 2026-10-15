//! The commands: the opcode table, and what every handler shares.
//!
//! A handler runs in one of three shapes:
//! - **F class**: the body runs under a handler; a raise is logged and the stamp is faulted at
//!   the command's latched offset, never signalled ([`Command::f_class`]).
//! - **S class** (7, 8, 10, 12): a raise is logged with the payload dumped, and the stamp is
//!   signalled anyway ([`Command::s_caught`]).
//! - **Completion**: a task command hands the task a completion that faults at the offset latched
//!   before the call when the status is not OK, and signals otherwise ([`Command::completion`]).
//!
//! Every non-deprecated command runs in a real pool of its own, with a holder ([`Held`]) of its own
//! for its whole run. The holder keeps the tasks and guest displays its
//! lookups return alive until the command ends, takes what a deep lookup holds (a user task's
//! reference to the kernel task, a page-table read's device handle) and what a caught raise
//! carries. It releases them, most recent first, when the handler returns (after its fault or
//! signal, where it gives one), and then the pool pops. The deprecated opcodes and an unknown one
//! run with neither.
//!
//! A command is a unit of work: its handler runs with a hold of the command's holder
//! ([`Hold::of`]) and no handler, so a raise outside the F and S classes' frames (an exec's
//! task call) dies at the raise.
//!
//! The lookups hand back a task or a guest display, or raise: none answers nil, so no handler
//! checks for one.

pub mod display;
pub mod exec;
pub mod misc;
pub mod task;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::device::DeviceCore;
use crate::display::guest_display::GuestDisplay;
use crate::fatal;
use crate::fifo::wire::{FifoCommand, Payload};
use crate::fifo::{DeathKind, Fifo, RaiseKind};
use crate::guest::records::GuestFeatures;
use crate::num::ToU64;
use crate::platform::Platform;
use crate::raise::{OnFault, Raise, RaiseId, RaiseResult};
use crate::rt::Hold;
use crate::rt::held::Held;
use crate::rt::pool::Pool;
use crate::seams::links::KernelTaskRef;
use crate::tasks::{Completion, ExecStatus, Task};

/// How many payload bytes a dump shows.
const DUMP_MAX: usize = 64;

/// The text every feature gate raises with.
pub(crate) const UNSUPPORTED: &str = "the command is not supported at this binary version";
/// The text of commands 4, 6, 7, 8 and 54 naming another task.
pub(crate) const KERNEL_ONLY: &str = "the command must name a kernel task";
/// The text of commands 55, 67 and 69 naming the kernel task.
pub(crate) const NOT_KERNEL: &str = "the command must not name a kernel task";

/// A handler.
type Handler<P> = fn(&mut Command<'_, P>);

/// The opcodes the table has a case for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    /// 0.
    Debug,
    /// 1.
    DisplaySetSharedStatePage,
    /// 2.
    DisplayAck,
    /// 4.
    DisplayCursorGlyph,
    /// 5.
    DisplayCursorShow,
    /// 6.
    DisplayTransaction2,
    /// 7.
    DisplayTransaction3,
    /// 8.
    DisplaySwapMapping,
    /// 9.
    DisplaySleepState,
    /// 10.
    DisplaySetProperties,
    /// 11.
    DisplayCompositorParameters,
    /// 12.
    DisplaySetGuestIccProfile,
    /// 30.
    Nop,
    /// 3, 31, 33, 35, 36, 38, 39, 41-47 and 50: logged, dumped and faulted, with no pool or
    /// holder of their own.
    Deprecated,
    /// 32.
    DeleteTask,
    /// 34.
    UnmapMemory,
    /// 37.
    DeleteResource,
    /// 40.
    DeleteObject,
    /// 48.
    DefineChildFifo,
    /// 49.
    DeleteChildFifo,
    /// 51.
    SetObjectList,
    /// 52.
    InvalidateResources,
    /// 53.
    SynchronizeResources,
    /// 54.
    DeleteIoSurfaceBacking2,
    /// 55.
    ExecIndirect2,
    /// 56.
    DefineTask2,
    /// 57.
    MapMemory2,
    /// 58.
    GetDeviceInfo,
    /// 59.
    GetComputeInfo,
    /// 60.
    ReplacePhysical,
    /// 61.
    Delay,
    /// 62.
    SynchronizeAndDiscardResources,
    /// 63.
    DiscardResources,
    /// 64.
    HeapTextureSizeAndAlign,
    /// 65.
    DeleteSharedTextureBacking,
    /// 66.
    ResetRasterizationRateMap,
    /// 67.
    ExecIndirect3,
    /// 68.
    SetObjectAndPlacementList,
    /// 69.
    InfoIndirect,
}

impl Opcode {
    /// The opcode `command_id` names; `None` for one the table has no case for (13-29, 70 and
    /// up), which is faulted as unknown.
    pub fn from_id(command_id: u16) -> Option<Opcode> {
        use Opcode::*;
        Some(match command_id {
            0 => Debug,
            1 => DisplaySetSharedStatePage,
            2 => DisplayAck,
            4 => DisplayCursorGlyph,
            5 => DisplayCursorShow,
            6 => DisplayTransaction2,
            7 => DisplayTransaction3,
            8 => DisplaySwapMapping,
            9 => DisplaySleepState,
            10 => DisplaySetProperties,
            11 => DisplayCompositorParameters,
            12 => DisplaySetGuestIccProfile,
            30 => Nop,
            3 | 31 | 33 | 35 | 36 | 38 | 39 | 41..=47 | 50 => Deprecated,
            32 => DeleteTask,
            34 => UnmapMemory,
            37 => DeleteResource,
            40 => DeleteObject,
            48 => DefineChildFifo,
            49 => DeleteChildFifo,
            51 => SetObjectList,
            52 => InvalidateResources,
            53 => SynchronizeResources,
            54 => DeleteIoSurfaceBacking2,
            55 => ExecIndirect2,
            56 => DefineTask2,
            57 => MapMemory2,
            58 => GetDeviceInfo,
            59 => GetComputeInfo,
            60 => ReplacePhysical,
            61 => Delay,
            62 => SynchronizeAndDiscardResources,
            63 => DiscardResources,
            64 => HeapTextureSizeAndAlign,
            65 => DeleteSharedTextureBacking,
            66 => ResetRasterizationRateMap,
            67 => ExecIndirect3,
            68 => SetObjectAndPlacementList,
            69 => InfoIndirect,
            _ => return None,
        })
    }

    /// The handler of an opcode that runs in a pool and a holder of its own: every one but
    /// [`Opcode::Deprecated`].
    fn handler<P: Platform>(self) -> Option<Handler<P>> {
        use Opcode::*;
        let h: Handler<P> = match self {
            Deprecated => return None,
            Debug => misc::debug,
            DisplaySetSharedStatePage => display::set_shared_state_page,
            DisplayAck => display::ack,
            DisplayCursorGlyph => display::cursor_glyph,
            DisplayCursorShow => display::cursor_show,
            DisplayTransaction2 => display::transaction2,
            DisplayTransaction3 => display::transaction3,
            DisplaySwapMapping => display::swap_mapping,
            DisplaySleepState => display::sleep_state,
            DisplaySetProperties => display::set_properties,
            DisplayCompositorParameters => display::compositor_parameters,
            DisplaySetGuestIccProfile => display::set_guest_icc_profile,
            Nop => misc::nop,
            DeleteTask => task::delete_task,
            UnmapMemory => task::unmap_memory,
            DeleteResource => task::delete_resource,
            DeleteObject => task::delete_object,
            DefineChildFifo => misc::define_child_fifo,
            DeleteChildFifo => misc::delete_child_fifo,
            SetObjectList => task::set_object_list,
            InvalidateResources => task::invalidate_resources,
            SynchronizeResources => task::synchronize_resources,
            DeleteIoSurfaceBacking2 => task::delete_iosurface_backing,
            ExecIndirect2 => exec::exec_indirect2,
            DefineTask2 => task::define_task,
            MapMemory2 => task::map_memory,
            GetDeviceInfo => misc::get_device_info,
            GetComputeInfo => task::get_compute_info,
            ReplacePhysical => task::replace_physical,
            Delay => misc::delay,
            SynchronizeAndDiscardResources => task::synchronize_and_discard_resources,
            DiscardResources => task::discard_resources,
            HeapTextureSizeAndAlign => task::heap_texture_size_and_align,
            DeleteSharedTextureBacking => task::delete_shared_texture_backing,
            ResetRasterizationRateMap => task::reset_rasterization_rate_map,
            ExecIndirect3 => exec::exec_indirect3,
            SetObjectAndPlacementList => task::set_object_and_placement_list,
            InfoIndirect => exec::info_indirect,
        };
        Some(h)
    }
}

impl<P: Platform> Fifo<P> {
    /// Runs one command, whose stamp the thread has pushed.
    pub(crate) fn dispatch(self: &Arc<Self>, header: &FifoCommand, payload: &[u8]) {
        let payload = Payload::new(payload);
        let Some(op) = Opcode::from_id(header.command_id) else {
            log::warn!(
                "FIFO channel {}: unknown command_id={} len={} stamp={}",
                self.stamp_index(),
                header.command_id,
                header.length,
                header.stamp_value
            );
            return misc::invalid(self, header, payload);
        };
        log::info!(
            "FIFO channel {}: dispatch command_id={} ({:?}) len={} stamp={}",
            self.stamp_index(),
            header.command_id,
            op,
            header.length,
            header.stamp_value
        );
        match op.handler::<P>() {
            // One pool and one holder per command: the holder releases what it holds, then the
            // pool pops.
            Some(h) => {
                let pool = Pool::push(self.platform());
                let held = Held::new();
                h(&mut Command {
                    fifo: self,
                    held: &held,
                    hold: Hold::of(&held),
                    header,
                    payload,
                });
                drop(held);
                drop(pool);
            }
            None => misc::deprecated(self, header, payload),
        }
    }
}

/// A raise in a log line: its detail, or its name when it has none.
pub(crate) fn exception(r: &Raise) -> &str {
    let name = match r.id() {
        RaiseId::Fifo(k) => k.name(),
        _ => "raise",
    };
    r.detail().unwrap_or(name)
}

/// Diagnostics, at debug: the first `min(size, 64)` bytes of `bytes` in hex.
fn dump<P: Platform>(fifo: &Fifo<P>, bytes: Payload<'_>, size: usize) {
    use std::fmt::Write as _;
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }
    let count = size.min(DUMP_MAX);
    let mut hex = String::new();
    for b in bytes.bytes().iter().take(count) {
        let _ = write!(hex, " {b:02x}");
    }
    log::debug!(
        "channel {}: the payload's first {count} of {size} bytes:{hex}",
        fifo.stamp_index()
    );
}

/// A payload of `size` bytes shorter than `expected` raises; `bytes` is what is dumped (a
/// property record is dumped from its own start).
fn check_size<P: Platform>(
    fifo: &Fifo<P>,
    bytes: Payload<'_>,
    size: usize,
    expected: u32,
) -> RaiseResult<()> {
    if size.to_u64() < u64::from(expected) {
        log::warn!(
            "channel {}: a payload of {size} bytes, shorter than the command's {expected}",
            fifo.stamp_index()
        );
        dump(fifo, bytes, size);
        crate::raise!(
            RaiseKind::CommandLength,
            format!("a payload of {size} bytes, shorter than the command's {expected}")
        );
    }
    Ok(())
}

/// `a * b + c` in 32 bits, and whether the exact value does not fit.
pub(crate) fn mul_add_overflows(a: u32, b: u32, c: u32) -> (u32, bool) {
    let exact = u64::from(a)
        .wrapping_mul(u64::from(b))
        .wrapping_add(u64::from(c));
    (
        crate::num::Truncate::low_u32(exact),
        exact > u64::from(u32::MAX),
    )
}

/// `a + b + c` in 32 bits, and whether the exact sum does not fit.
pub(crate) fn add3_overflows(a: u32, b: u32, c: u32) -> (u32, bool) {
    let exact = u64::from(a)
        .wrapping_add(u64::from(b))
        .wrapping_add(u64::from(c));
    (
        crate::num::Truncate::low_u32(exact),
        exact > u64::from(u32::MAX),
    )
}

/// One command as its handler runs it: the FIFO, the command's holder and its hold, its header
/// and its payload.
pub(crate) struct Command<'a, P: Platform> {
    fifo: &'a Arc<Fifo<P>>,
    /// Released when the handler returns, before the command's pool pops (module doc).
    held: &'a Held,
    /// The command's unit of work: no handler, and `held` as its holder.
    hold: Hold<'a>,
    header: &'a FifoCommand,
    payload: Payload<'a>,
}

impl<P: Platform> Command<'_, P> {
    fn stamp(&self) -> u32 {
        self.header.stamp_value
    }

    /// Signals the command's stamp.
    fn signal(&self) {
        self.fifo.signal_stamp_value(self.stamp());
    }

    /// The device. The device outlives its FIFOs (it drains them before it goes), so a gone one
    /// is a use after free: a death at the handler's first use of it.
    fn device(&self) -> Arc<DeviceCore<P>> {
        match self.fifo.device.upgrade() {
            Some(d) => d,
            None => fatal!(DeathKind::DeviceGone),
        }
    }

    /// The device's features, read now.
    fn features(&self) -> GuestFeatures {
        self.device().features().load()
    }

    /// The whole payload must hold `expected` bytes.
    fn check_size(&self, expected: u32) -> RaiseResult<()> {
        check_size(self.fifo, self.payload, self.payload.len(), expected)
    }

    /// The task `task_id`, kept until the command ends.
    fn task(&mut self, task_id: u32) -> RaiseResult<Task<P>> {
        let task = self.device().get_task_id(OnFault::Return, task_id)?;
        self.held.keep(task.clone());
        Ok(task)
    }

    /// The user task `task_id` (0 raises), kept until the command ends.
    fn user_task(&mut self, task_id: u32) -> RaiseResult<Task<P>> {
        let task = self.device().get_user_task_id(task_id)?;
        self.held.keep(task.clone());
        Ok(task)
    }

    /// The kernel task, kept until the command ends.
    fn kernel_task(&mut self) -> RaiseResult<KernelTaskRef<P>> {
        let task = self.device().get_kernel_task(OnFault::Return)?;
        self.held.keep(Arc::clone(&task));
        Ok(task)
    }

    /// The guest display at `port`, kept until the command ends.
    fn guest_display(&mut self, port: u32) -> RaiseResult<Arc<GuestDisplay<P>>> {
        let guest_display = self.device().get_guest_display_at_port(u64::from(port))?;
        self.held.keep(Arc::clone(&guest_display));
        Ok(guest_display)
    }

    /// The completion a task command passes: fault at the offset latched now for a status other
    /// than OK, signal otherwise. It holds the FIFO; the error is ignored.
    fn completion(&self) -> Completion {
        let offset = self.fifo.command_offset.load(Ordering::Relaxed);
        let stamp = self.stamp();
        let fifo = Arc::clone(self.fifo);
        Arc::new(move |status: ExecStatus, _error| {
            if !status.is_ok() {
                fifo.fault_at_offset(offset, stamp);
            } else {
                fifo.signal_stamp_value(stamp);
            }
        })
    }

    /// The F class: `body` under a handler; a raise is logged and faults the stamp, and answers
    /// `None`. What the raise carries goes to the command's holder.
    fn f_class<T>(
        &mut self,
        body: impl FnOnce(&mut Self, Hold<'_>) -> RaiseResult<T>,
    ) -> Option<T> {
        let hold = self.hold;
        match body(self, hold) {
            Ok(v) => Some(v),
            Err(r) => {
                self.f_caught(&r);
                hold.land(r);
                None
            }
        }
    }

    /// The F class's catch.
    fn f_caught(&self, r: &Raise) {
        log::warn!(
            "channel {}: command {} faulted: {}",
            self.fifo.stamp_index(),
            self.header.command_id,
            exception(r)
        );
        self.fifo.fault_at_stamp_value(self.stamp());
    }

    /// The S class's catch: logged with the command's name, and the first `size` bytes of the
    /// payload dumped; what the raise carries goes to the command's holder, among its values at
    /// the points they were first held, released at the command's end. The caller signals the
    /// stamp whatever happened.
    fn s_caught(&self, name: &'static str, r: Raise, size: usize) {
        log::warn!(
            "channel {}: {name} failed: {}",
            self.fifo.stamp_index(),
            exception(&r)
        );
        dump(self.fifo, self.payload, size);
        self.hold.land(r);
    }
}

#[cfg(test)]
mod tests;
