//! The info command, the replies the guest reads back (compute info, heap texture size and
//! alignment), and the kernel task's backing lookups.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{DeathKind, LocalTask, PAGE, RaiseKind, kernel_task, raise_caught};
use crate::guest::GuestPtr;
use crate::guest::records::CmdExecIndirectResource3;
use crate::num::Truncate;
use crate::platform::Platform;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::command::{CommandBuffer, CommandQueue};
use crate::platform::event::SharedEvent;
use crate::platform::resource::ComputePipeline;
use crate::raise::{OnFault, RaiseResult};
use crate::resources::manager::create::Resources;
use crate::resources::manager::states::ComputePipelineStates;
use crate::resources::resource::{Resource, ResourceKind};
use crate::rt::{Held, Hold};
use crate::tasks::ExecArgs;
use crate::tasks::resource_list::ResourceList;
use crate::tasks::{Completion, ExecStatus};

/// How long the info command waits for the previous exec's FIFO value, in milliseconds.
const INFO_WAIT_MS: u64 = 5000;

/// A key of the compute-info reply and the pipeline's answer for it.
type Answer<P> = (u32, fn(&<P as Platform>::ComputePipeline) -> u64);

/// The stride of the compute-info reply: `{u32 key; u32 value;}` pairs.
const COMPUTE_INFO_ENTRY_SIZE: u64 = 8;

/// One `{key, value}` pair of the compute-info reply at `entry`, as one store of both words
/// ([`GuestPtr::write_u32_pair`]): a guest's 64-bit load of the entry sees the old pair or the new
/// one, never the new key with the old value, wherever the pair stays inside one 16-byte granule,
/// and one access per word across a granule (a reply offset of 4 modulo 8 puts every other pair
/// there).
fn store_compute_info_pair(entry: GuestPtr, key: u32, value: u32) {
    entry.write_u32_pair(0, key, value, DeathKind::NullReply);
}


/// The SIMD width a nil compute pipeline answers (`threadExecutionWidth`): this device's, never
/// 0, which a guest divides by.
const NIL_THREAD_EXECUTION_WIDTH: u64 = 32;

impl<P: Platform> LocalTask<P> {
    /// The info command (opcode 69), with no task lock: waits up to 5 s for the previous exec's
    /// FIFO value (3 on a timeout), then with the object lists mapped (else 2) prepares, decodes
    /// into a buffer it never commits, and advances and signals the FIFO value on the CPU. A
    /// raise is logged and reports 2.
    pub(super) fn info(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource3>,
        done: Completion,
    ) -> RaiseResult<()> {
        // Waits for the previous exec's completion, 5 s at most. An exec running longer makes
        // 69 report 3 (testing.md 7.4).
        let signalled = match args.fifo_event {
            Some(event) => event.wait_until_signaled_value(
                args.fifo_event_value.load(Ordering::SeqCst),
                INFO_WAIT_MS,
            ),
            None => false,
        };
        if !signalled {
            done(ExecStatus::GPU_OR_TIMEOUT, None);
            return Ok(());
        }
        let (deserializer, _, lists) = self.parts();
        if !lists.ensure_mapped(hold) {
            done(ExecStatus::FAIL, None);
            return Ok(());
        }
        let resource_list = ResourceList::V3(args.resources);
        let decoded = (|| -> RaiseResult<_> {
            // The prepare's shared-texture invalidation runs.
            // Only the preparation's effects are kept
            drop(self.prepare_resources(hold, &resource_list, None, 0)?);
            let queue = self.device.upgrade().and_then(|d| d.exec_queue());
            let command_buffer = match &queue {
                Some(queue) => queue.command_buffer(OnFault::Return)?,
                None => None,
            };
            let mut cursors = self.cursors(hold, &args.cmd_buffers);
            // The buffer is never committed.
            let _decoders =
                deserializer.decode_segments(hold, &mut cursors, command_buffer.as_ref())?;
            // The FIFO's counter advances, signalled on the CPU.
            let value = args
                .fifo_event_value
                .fetch_add(1, Ordering::SeqCst)
                .wrapping_add(1);
            if let Some(event) = args.fifo_event {
                event.set_signaled_value(value)?;
            }
            done(ExecStatus::OK, None);
            Ok(())
        })();
        if let Err(raised) = decoded {
            log::warn!(
                "task {}: info faulted: {}",
                self.task_id,
                raised.exception()
            );
            done(ExecStatus::FAIL, None);
            hold.land(raised);
        }
        Ok(())
    }

    /// The compute-info reply (opcode 64), under the task's shared lock: up to `count` pairs at
    /// task offset `offset` for the keys below `max_key` the pipeline answers (1 its maximum
    /// threads per threadgroup, 3 its execution width, 4 its static threadgroup memory, each cut
    /// to 32 bits), then a `{0, 0}` terminator if room is left.
    pub(super) fn compute_info(
        &self,
        hold: Hold<'_>,
        pipeline_reference: u32,
        max_key: u32,
        count: u32,
        offset: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        let _lock = self.base.mutex.read();
        let (_, resource_manager, lists) = self.parts();
        self.map_object_lists(hold, &lists)?;
        let pipeline =
            match resource_manager.get_object::<ComputePipelineStates>(hold, pipeline_reference)? {
                // Ref 0 is a nil pipeline, and every value is 0.
                Ok(pipeline) => pipeline,
                Err(error) => {
                    return raise_caught(
                        RaiseKind::Pipeline,
                        format_args!("compute pipeline {pipeline_reference} not found: {error}"),
                    );
                }
            };
        // The blocks setup ignores writability. The mapping is held until the reply is done.
        let mapped = self.mapped_address(
            hold,
            offset,
            u64::from(count).wrapping_mul(COMPUTE_INFO_ENTRY_SIZE),
            true,
        )?;
        let entries = GuestPtr::new(mapped.address);
        let entry =
            |index: u32| entries.offset(u64::from(index).wrapping_mul(COMPUTE_INFO_ENTRY_SIZE));
        let answers: [Answer<P>; 3] = [
            (1, |p| p.max_total_threads_per_threadgroup()),
            (3, |p| p.thread_execution_width()),
            (4, |p| p.static_threadgroup_memory_length()),
        ];
        let mut index: u32 = 0;
        if pipeline.is_none() {
            log::warn!(
                "compute pipeline info for nil pipeline {pipeline_reference}: threadExecutionWidth answered {NIL_THREAD_EXECUTION_WIDTH}, the rest 0"
            );
        }
        for (key, answer) in answers {
            if max_key > key && index < count {
                // A nil pipeline answers 0, as Metal's nil does, but for its SIMD width: guests
                // divide by it unchecked (WindowServer's CA::OGL::MetalContext::
                // calculate_average_luma died of SIGFPE on a 0), and no dispatch of a nil
                // pipeline runs anyway.
                let value = match &pipeline {
                    Some(p) => answer(p),
                    None if key == 3 => NIL_THREAD_EXECUTION_WIDTH,
                    None => 0,
                };
                store_compute_info_pair(entry(index), key, value.low_u32());
                index = index.wrapping_add(1);
            }
        }
        if index < count {
            // One 8-byte store of `{0, 0}`
            entries.write_u64(
                u64::from(index).wrapping_mul(COMPUTE_INFO_ENTRY_SIZE),
                0,
                DeathKind::NullReply,
            );
        }
        done(ExecStatus::OK, None);
        drop(mapped);
        Ok(())
    }

    /// The heap texture size and alignment reply (opcode 22's), under the task's unique lock and
    /// in a holder of its own ([`Held::raising`]): the reply length is checked (16 bytes to a
    /// page) and the reply mapped before the computation; a refused computation is logged and
    /// reports 2 before the holder's end; otherwise the 16-byte reply is written and 1 reported
    /// after the holder's end. On a raise what it holds goes on with the raise.
    pub(super) fn heap_texture_reply(
        &self,
        payload: &[u8],
        reply_offset: u64,
        reply_length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        let _lock = self.base.mutex.write();
        let computed = Held::raising(|held| {
            let hold = Hold::of(held);
            if !(16..PAGE).contains(&reply_length) {
                return raise_caught(
                    RaiseKind::ReplyLength,
                    format_args!("reply of {reply_length:#x} bytes is not 0x10 to below a page"),
                );
            }
            // The mapping is held to the holder's end (on a raise, where the raise ends).
            let mapped = self.mapped_address(hold, reply_offset, reply_length, true)?;
            let reply = GuestPtr::new(mapped.address);
            held.keep(mapped.mapping);
            let (_, resource_manager, _) = self.parts();
            match resource_manager.heap_texture_size_and_align(payload)? {
                Err(error) => {
                    log::warn!("task {}: heap texture size refused: {error}", self.task_id);
                    done(ExecStatus::FAIL, None);
                    Ok(false)
                }
                Ok(size_and_align) => {
                    reply.write_u64(0, size_and_align.size, DeathKind::NullReply);
                    reply.write_u64(8, size_and_align.align, DeathKind::NullReply);
                    Ok(true)
                }
            }
        })?;
        if computed {
            done(ExecStatus::OK, None);
        }
        Ok(())
    }

    /// The backing `identifier` names in the kernel task (a user task asks the kernel task).
    /// Anything else, identifier 0 included, raises.
    pub(super) fn backing(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        if self.task_id != 0 {
            // Nothing once the device is gone
            return match self.device.upgrade() {
                Some(d) => {
                    let kernel = kernel_task(hold, &d)?;
                    kernel.on_platform(|| kernel.backing(hold, identifier))
                }
                None => Ok(None),
            };
        }
        let (_, resource_manager, lists) = self.parts();
        self.map_object_lists(hold, &lists)?;
        match resource_manager.get_object::<Resources>(hold, identifier)? {
            Ok(Some(backing)) if backing.is_backing() => Ok(Some(backing)),
            _ => raise_caught(
                RaiseKind::Backing,
                format_args!("resource {identifier} is not a backing"),
            ),
        }
    }

    /// The kernel task's backing `identifier`, paged in on the blit-up queue first (waiting until
    /// the buffer is scheduled) unless its host copy is valid, under the task's shared lock.
    pub(super) fn prepared_backing(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let _lock = self.base.mutex.read();
        crate::fatal_unless!(self.task_id == 0, DeathKind::NotKernelTask);
        let backing = self.backing(hold, identifier)?;
        if !backing.as_ref().is_some_and(|b| b.is_host_valid()) {
            let queue = self.device.upgrade().and_then(|d| d.blit_up_queue());
            let command_buffer = match &queue {
                Some(queue) => queue.command_buffer(OnFault::Return)?,
                None => None,
            };
            let encoder = match &command_buffer {
                Some(buffer) => buffer.blit_encoder(OnFault::Return)?,
                None => None,
            };
            if let Some(b) = &backing {
                b.prepare_in_encoder(hold, encoder.as_ref())?;
            }
            if let Some(encoder) = &encoder {
                encoder.encode(OnFault::Return, BlitCmd::EndEncoding)?;
            }
            if let Some(buffer) = &command_buffer {
                buffer.commit(OnFault::Return)?;
                buffer.wait_until_scheduled()?;
            }
        }
        Ok(backing)
    }

    /// A shared-texture handle for the kernel task's shared-texture backing `identifier`, under
    /// the task's unique lock and the object database's lock, preparing the backing first where
    /// it needs it. Anything else is logged and raises. A user task asks the kernel task.
    pub(super) fn shared_texture_handle(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<P::SharedTextureHandle>> {
        if self.task_id != 0 {
            return match self.device.upgrade() {
                Some(d) => {
                    let kernel = kernel_task(hold, &d)?;
                    kernel.on_platform(|| kernel.shared_texture_handle(hold, identifier))
                }
                None => Ok(None),
            };
        }
        let _lock = self.base.mutex.write();
        let _database = self.base.object_database_mutex.lock();
        let (_, resource_manager, lists) = self.parts();
        self.map_object_lists(hold, &lists)?;
        let backing = resource_manager.get_object::<Resources>(hold, identifier)?;
        let shared = match &backing {
            Ok(Some(resource)) => match &resource.kind {
                ResourceKind::Texture(texture) if texture.is_shared_texture_backing() => {
                    Some((resource, texture))
                }
                _ => None,
            },
            _ => None,
        };
        let Some((resource, texture)) = shared else {
            // The lookup's error, or why what it found is not a backing (diagnostics only).
            let why = match &backing {
                Err(e) => e.as_str(),
                Ok(None) => "reference 0 names none",
                Ok(Some(_)) => "the resource is another kind",
            };
            log::warn!("shared texture backing {identifier} not found: {why}");
            return raise_caught(
                RaiseKind::Backing,
                format_args!("no shared texture backing {identifier}"),
            );
        };
        if resource.needs_prepare() {
            texture
                .prepare_shared_texture_backing(&resource.state, self.base.paging_queue.as_ref())?;
        }
        texture.new_shared_texture_handle()
    }

    /// Deletes the kernel task's backing `identifier` with its placement entry, under the task's
    /// unique lock, then reports 1.
    pub(super) fn remove_backing(
        &self,
        hold: Hold<'_>,
        identifier: u32,
        done: Completion,
    ) -> RaiseResult<()> {
        let _lock = self.base.mutex.write();
        let (_, resource_manager, lists) = self.parts();
        self.map_object_lists(hold, &lists)?;
        resource_manager.delete_object::<Resources>(hold, identifier, true);
        done(ExecStatus::OK, None);
        Ok(())
    }
}
