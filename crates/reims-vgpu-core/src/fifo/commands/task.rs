//! The task commands: each looks a task up and hands it the command with a completion, or asks
//! the device to define or delete one. All are F class.

#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use crate::fifo::RaiseKind;
use crate::fifo::commands::{Command, KERNEL_ONLY, UNSUPPORTED, mul_add_overflows};
use crate::guest::records::APV_PAGE_SHIFT;
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::raise;
use crate::raise::RaiseResult;
use crate::tasks::{ResourceInvalidation, ResourceReference};

/// The records' sizes.
const TASK: u32 = 4;
const MEMORY_RANGE: u32 = 20;
const TASK_OBJECT: u32 = 8;
const DELETE_OBJECT: u32 = 12;
const SET_OBJECT_LIST: u32 = 12;
/// A resource list's header.
const RESOURCE_LIST: u32 = 8;
const RESOURCE: u32 = 4;
const INVALIDATED_RESOURCE: u32 = 8;
const DELETE_IOSURFACE_BACKING: u32 = 8;
const DEFINE_TASK: u32 = 16;
const GET_COMPUTE_INFO: u32 = 24;
const HEAP_TEXTURE_SIZE_AND_ALIGN: u32 = 24;
const SET_OBJECT_AND_PLACEMENT_LIST: u32 = 28;
/// The object list's page, as a byte offset: a guest page number.
const OBJECT_LIST_PAGE_SHIFT: u32 = APV_PAGE_SHIFT;

/// 32: deletes a task.
pub(super) fn delete_task<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(TASK)?;
        let task_id = c.payload.u32(0);
        // DIAG(empty-entry): rare.
        log::debug!("DeleteTask: id {task_id}");
        if !c.device().delete_task_id(task_id)? {
            log::warn!("task {task_id} not deleted: the device refused it");
            raise!(RaiseKind::DeleteTask, format!("task {task_id} not deleted"));
        }
        c.signal();
        Ok(())
    });
}

/// 34: unmaps a range of a task's memory.
pub(super) fn unmap_memory<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(MEMORY_RANGE)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        task.unmap_memory_at_offset(p.u64(4), p.u64(12), c.completion())
    });
}

/// 37: deletes a resource.
pub(super) fn delete_resource<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        c.check_size(TASK_OBJECT)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        task.delete_resource(hold, p.u32(4), c.completion())
    });
}

/// 40: deletes an object. The task gets the whole payload after the task ID, not `length`.
pub(super) fn delete_object<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        c.check_size(DELETE_OBJECT)?;
        let p = c.payload;
        let declared = p.u32(8);
        let (length, overflow) = declared.overflowing_add(4);
        if overflow {
            raise!(
                RaiseKind::Overflow,
                format!("object length {declared:#x} overflows the payload length")
            );
        }
        c.check_size(length)?;
        // DIAG(empty-entry): which task's objects go, and when.
        log::debug!("DeleteObject: task {} ({declared:#x} bytes)", p.u32(0));
        let task = c.task(p.u32(0))?;
        task.delete_object_with_serialized_data(hold, p.tail(4).bytes(), c.completion())
    });
}

/// 51: sets a task's object list (no placement list).
pub(super) fn set_object_list<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        c.check_size(SET_OBJECT_LIST)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        task.set_object_list_offset(
            hold,
            u64::from(p.u32(4)).wrapping_shl(OBJECT_LIST_PAGE_SHIFT),
            p.u32(8),
            0,
            0,
            c.completion(),
        )
    });
}

/// Which resource-list command raises an overflow, for its text.
#[derive(Clone, Copy)]
enum ListCommand {
    Invalidate,
    Synchronize,
    SynchronizeAndDiscard,
    Discard,
}

impl ListCommand {
    fn name(self) -> &'static str {
        match self {
            ListCommand::Invalidate => "CmdInvalidateResources",
            ListCommand::Synchronize => "CmdSynchronizeResources",
            ListCommand::SynchronizeAndDiscard => "CmdSynchronizeAndDiscardResources",
            ListCommand::Discard => "CmdDiscardResources",
        }
    }
}

/// The resource-list commands' size checks (52, 53, 62, 63): the header, then
/// `count * record + 8`. Answers the count.
fn check_resource_list<P: Platform>(
    c: &Command<'_, P>,
    record: u32,
    command: ListCommand,
) -> RaiseResult<u32> {
    c.check_size(RESOURCE_LIST)?;
    let count = c.payload.u32(4);
    let (length, overflow) = mul_add_overflows(count, record, RESOURCE_LIST);
    if overflow {
        raise!(
            RaiseKind::Overflow,
            format!(
                "{} of {count} entries overflows the payload length",
                command.name()
            )
        );
    }
    c.check_size(length)?;
    Ok(count)
}

/// 52: invalidates resources.
pub(super) fn invalidate_resources<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        let count = check_resource_list(c, INVALIDATED_RESOURCE, ListCommand::Invalidate)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        let records = p.records::<ResourceInvalidation>(8, count);
        task.invalidate_resources(records, c.completion())
    });
}

/// 53: synchronizes resources.
pub(super) fn synchronize_resources<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        let count = check_resource_list(c, RESOURCE, ListCommand::Synchronize)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        let records = p.records::<ResourceReference>(8, count);
        task.synchronize_resources(hold, records, false, c.completion())
    });
}

/// 54: deletes an IOSurface backing, on the kernel task.
pub(super) fn delete_iosurface_backing<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        c.check_size(DELETE_IOSURFACE_BACKING)?;
        let p = c.payload;
        if p.u32(4) != 0 {
            raise!(RaiseKind::KernelOnly, KERNEL_ONLY);
        }
        let task = c.kernel_task()?;
        task.delete_backing(hold, p.u32(0), c.completion())
    });
}

/// 56: defines a task (synchronously), then signals.
pub(super) fn define_task<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        c.check_size(DEFINE_TASK)?;
        let p = c.payload;
        // The task ID is bits 1-31
        c.device()
            .create_task_id(hold, p.u32(0) >> 1, p.u32(12), p.u64(4), false)?;
        c.signal();
        Ok(())
    });
}

/// 57: maps a range of a task's memory.
pub(super) fn map_memory<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        c.check_size(MEMORY_RANGE)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        task.map_memory_at_offset(hold, p.u64(4), p.u64(12), c.completion())
    });
}

/// 59: a compute pipeline's info, on a user task.
pub(super) fn get_compute_info<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        c.check_size(GET_COMPUTE_INFO)?;
        let p = c.payload;
        let task = c.user_task(p.u32(0))?;
        task.get_compute_info(
            hold,
            p.u32(4),
            p.u32(8),
            p.u32(12),
            p.u64(16),
            c.completion(),
        )
    });
}

/// 60: replaces a resource's physical backing.
pub(super) fn replace_physical<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(TASK_OBJECT)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        task.replace_physical(p.u32(4), c.completion())
    });
}

/// 62: synchronizes and discards resources.
pub(super) fn synchronize_and_discard_resources<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        if !c.features().synchronize_and_discard_resources {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        let count = check_resource_list(c, RESOURCE, ListCommand::SynchronizeAndDiscard)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        let records = p.records::<ResourceReference>(8, count);
        task.synchronize_resources(hold, records, true, c.completion())
    });
}

/// 63: discards resources.
pub(super) fn discard_resources<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        if !c.features().discard_resources {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        let count = check_resource_list(c, RESOURCE, ListCommand::Discard)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        let records = p.records::<ResourceReference>(8, count);
        task.discard_resources(hold, records, c.completion())
    });
}

/// 64: a heap texture's size and alignment, on a user task.
pub(super) fn heap_texture_size_and_align<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        if !c.features().metal_heaps {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        c.check_size(HEAP_TEXTURE_SIZE_AND_ALIGN)?;
        let p = c.payload;
        let payload_length = p.u32(20);
        let (length, overflow) = payload_length.overflowing_add(HEAP_TEXTURE_SIZE_AND_ALIGN);
        if overflow {
            raise!(
                RaiseKind::Overflow,
                format!("descriptor length {payload_length:#x} overflows the payload length")
            );
        }
        c.check_size(length)?;
        // The serialized descriptor after the record, `payloadLength` bytes
        let serialized = p
            .bytes()
            .get(HEAP_TEXTURE_SIZE_AND_ALIGN.to_usize()..length.to_usize())
            .unwrap_or(&[]);
        let task = c.user_task(p.u32(0))?;
        task.heap_texture_size_and_align(serialized, p.u64(4), p.u64(12), c.completion())
    });
}

/// 65: deletes a shared texture's backing, on the kernel task.
pub(super) fn delete_shared_texture_backing<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        c.check_size(TASK)?;
        let p = c.payload;
        let task = c.kernel_task()?;
        task.delete_resource(hold, p.u32(0), c.completion())
    });
}

/// 66: resets a rasterization rate map, on a user task. The only gated command that checks the
/// size first.
pub(super) fn reset_rasterization_rate_map<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(TASK_OBJECT)?;
        // Size before feature here.
        if !c.features().supports_variable_rasterization_rate_maps {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        let p = c.payload;
        let task = c.user_task(p.u32(0))?;
        task.reset_rasterization_rate_map(p.u32(4), c.completion())
    });
}

/// 68: sets a task's object and placement lists.
pub(super) fn set_object_and_placement_list<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, hold| {
        if !c.features().supports_argument_buffers {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        c.check_size(SET_OBJECT_AND_PLACEMENT_LIST)?;
        let p = c.payload;
        let task = c.task(p.u32(0))?;
        task.set_object_list_offset(
            hold,
            p.u64(4),
            p.u32(20),
            p.u64(12),
            p.u32(24),
            c.completion(),
        )
    });
}
