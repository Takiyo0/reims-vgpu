//! What every task kind shares: the task core ([`TaskCore`]) that the local task and the GPU
//! task service each embed, and the operations on its parts that the guest's commands
//! reach (delete, discard, flush, invalidate, replace, synchronize, rebuild).
//!
//! The kinds complete their commands around these (a completion with 1 where the core returns),
//! except `synchronize_resources`, whose completion is the paging buffer's.
//!
//! Locks: `mutex` is the task's writer-preferring shared/unique lock (commands on several FIFOs
//! interleave under it), `object_database_mutex` the recursive lock over the object database;
//! the order is task, then resource manager, then the object lists' mutex. `teardown` releases
//! the three parts under the unique lock, and a later use of a part dies at the op
//! ([`DeathKind::TornDown`]), only where the part is reached (a reference 0 never reaches it).

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};

use parking_lot::{Mutex, ReentrantMutex};

use crate::deserializer::Deserializer;
use crate::device::tasks::MAX_LOCAL_TASK_LENGTH;
use crate::fatal;
use crate::fifo::wire::Records;
use crate::guest::records::ObjectEntry;
use crate::num::ToUsize;
use crate::platform::Handle;
use crate::platform::Platform;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::command::{CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo};
use crate::platform::compute::{ComputeCmd, ComputeEncoder};
use crate::platform::types::CommandBufferStatus;
use crate::raise::OnFault;
use crate::raise::{Raise, RaiseResult};
use crate::resources::host::{CallBorrow, ResourceHost};
use crate::resources::manager::ResourceManager;
use crate::resources::manager::create::Resources;
use crate::resources::resource::Resource;
use crate::rt::Hold;
use crate::sync::WriterPrefRwLock;
use crate::tasks::isolation::TaskLife;
use crate::tasks::object_lists::ObjectLists;
use crate::tasks::{
    Completion, DeathKind, ExecStatus, RaiseKind, Unsupported, raise_caught, raise_on,
};
use crate::tasks::{ResourceInvalidation, ResourceReference};

/// The task core both task kinds embed: the Metal device, the paging queue, event and counter,
/// the capabilities, the two locks, and the three parts (deserializer, resource manager, object
/// lists) until `teardown`.
///
/// The public fields are the ones the task kinds reach directly.
pub struct TaskCore<P: Platform> {
    /// The task's Metal device (retained). `None` is nil: the GPU task's handshake carries on
    /// without a device. An `Arc` so the resource manager holds the same reference without a
    /// retain of its own.
    pub mtl_device: Option<Arc<P::Device>>,
    /// The three parts; `None` after `teardown`.
    parts: Mutex<Option<TaskParts<P>>>,
    /// The task itself, kept by `teardown` from the manager it releases, so a torn-down task
    /// still reads its features in `rebuild_task`.
    torn_down_host: Mutex<Option<Weak<dyn ResourceHost<P>>>>,
    /// The task's length.
    pub length: u64,
    /// The task's shared/unique lock.
    pub mutex: WriterPrefRwLock,
    /// The object database's recursive lock.
    pub object_database_mutex: ReentrantMutex<()>,
    /// The paging queue: the device's blit-down queue, or the service's own queue, neither
    /// nil-checked (`None` is nil, and a send to it does nothing).
    pub paging_queue: Option<P::CommandQueue>,
    /// The paging event.
    pub paging_event: Option<P::Event>,
    /// The paging counter (a local task's starts at 1, an isolated task's service's at 0).
    pub paging_counter: AtomicU64,
    /// Whether buffers can be made over address ranges.
    pub supports_buffer_with_address_ranges: bool,
    /// Whether heaps can be made over address ranges.
    pub supports_heap_with_address_ranges: bool,
    /// Whether a resource's backing can be detached.
    pub supports_resource_detach_backing: bool,
}

/// The three parts `teardown` releases.
pub struct TaskParts<P: Platform> {
    /// The command deserializer.
    pub deserializer: Arc<Deserializer<P>>,
    /// The resource manager (the task is its only owner; others borrow it for a call).
    pub resource_manager: Arc<ResourceManager<P>>,
    /// The object lists, shared with the resource manager.
    pub object_list_manager: Arc<ObjectLists<P>>,
}

impl<P: Platform> TaskParts<P> {
    /// Releases the parts in their order: the deserializer, then the resource manager, then
    /// the object lists (at teardown and at the task's drop alike).
    fn release(self) {
        let TaskParts {
            deserializer,
            resource_manager,
            object_list_manager,
        } = self;
        drop(deserializer);
        drop(resource_manager);
        drop(object_list_manager);
    }
}

/// The resources a discard deleted, oldest first. Their last releases (a mapper resource's
/// surface going back to the task) are the caller's to place: it lets them go after its
/// completion or reply; on a raise they go on with the raise to where it ends, after the
/// handler's report. They go newest first.
#[must_use = "the deleted resources go where the caller lets them go"]
pub struct Deleted<P: Platform>(Vec<Arc<Resource<P>>>);

impl<P: Platform> Deleted<P> {
    /// `raised`, carrying the resources on, oldest first, each as a value of its own
    /// ([`Raise::defer`]): the holder the raise ends in releases them newest first.
    fn carried_by(mut self, mut raised: Raise) -> Raise {
        for resource in std::mem::take(&mut self.0) {
            raised = raised.defer(resource);
        }
        raised
    }
}

impl<P: Platform> std::fmt::Debug for Deleted<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Deleted({})", self.0.len())
    }
}

impl<P: Platform> Drop for Deleted<P> {
    fn drop(&mut self) {
        while let Some(resource) = self.0.pop() {
            drop(resource);
        }
    }
}

/// The existing resource `resource` names in `rm` (the caller's view of the manager), without
/// making one. Reference 0 is none, and does not reach the manager.
fn existing<P: Platform>(
    rm: Option<&ResourceManager<P>>,
    resource: u32,
) -> Option<Arc<Resource<P>>> {
    if resource == 0 {
        return None;
    }
    let Some(rm) = rm else {
        fatal!(DeathKind::TornDown)
    };
    rm.get_existing_object::<Resources>(resource).flatten()
}

/// Whether a serialized delete reaches the resource manager's state: only the ten deletion
/// commands of 12 bytes or more do (they take the manager's lock); a short or unknown command
/// is refused before.
fn serialized_delete_reaches_manager(command: &[u8]) -> bool {
    let (Some(op), Some(_)) = (command.first_chunk::<4>(), command.get(8..12)) else {
        return false;
    };
    matches!(
        u32::from_le_bytes(*op),
        1000 | 1001 | 1012 | 1002 | 1003 | 1005 | 1006 | 1007 | 1009 | 1014
    )
}

impl<P: Platform> TaskCore<P> {
    /// A task core with no parts yet: the kind builds them once the task exists (the manager
    /// and the object lists hold it unretained) and installs them with [`TaskCore::set_parts`].
    #[expect(
        clippy::too_many_arguments,
        reason = "the core's values, as each kind's constructor has them"
    )]
    pub fn new(
        mtl_device: Option<Arc<P::Device>>,
        length: u64,
        paging_queue: Option<P::CommandQueue>,
        paging_event: Option<P::Event>,
        paging_counter: u64,
        supports_buffer_with_address_ranges: bool,
        supports_heap_with_address_ranges: bool,
        supports_resource_detach_backing: bool,
    ) -> Self {
        TaskCore {
            mtl_device,
            parts: Mutex::new(None),
            torn_down_host: Mutex::new(None),
            length,
            mutex: WriterPrefRwLock::default(),
            object_database_mutex: ReentrantMutex::new(()),
            paging_queue,
            paging_event,
            paging_counter: AtomicU64::new(paging_counter),
            supports_buffer_with_address_ranges,
            supports_heap_with_address_ranges,
            supports_resource_detach_backing,
        }
    }

    /// The parts, or `None` after `teardown`, for the kind's own use within one call. Never
    /// store one past the call: the release of the parts at teardown is behaviour.
    #[expect(clippy::type_complexity, reason = "the three parts")]
    pub fn parts(
        &self,
    ) -> Option<(
        Arc<Deserializer<P>>,
        Arc<ResourceManager<P>>,
        Arc<ObjectLists<P>>,
    )> {
        self.parts.lock().as_ref().map(|p| {
            (
                Arc::clone(&p.deserializer),
                Arc::clone(&p.resource_manager),
                Arc::clone(&p.object_list_manager),
            )
        })
    }

    /// The resource manager for this call, or `None` after `teardown`.
    fn manager(&self) -> Option<Arc<ResourceManager<P>>> {
        self.parts
            .lock()
            .as_ref()
            .map(|p| Arc::clone(&p.resource_manager))
    }

    /// The resource manager for an op that reaches it, which dies after `teardown`.
    fn live_manager(&self) -> Arc<ResourceManager<P>> {
        match self.manager() {
            Some(rm) => rm,
            None => fatal!(DeathKind::TornDown),
        }
    }

    /// Installs the parts. A second call releases the first parts, outside the leaf lock.
    pub fn set_parts(&self, parts: TaskParts<P>) {
        let previous = self.parts.lock().replace(parts);
        if let Some(previous) = previous {
            previous.release();
        }
    }

    /// Releases the parts under the unique lock. The leaf lock is released before they go:
    /// their drops reach Metal and the resources, which can call back into the task.
    pub fn teardown(&self) {
        let _lock = self.mutex.write();
        let parts = self.parts.lock().take();
        if let Some(parts) = parts {
            if let Some(task) = parts.resource_manager.task() {
                *self.torn_down_host.lock() = Some(Arc::downgrade(&task));
            }
            parts.release();
        }
    }

    /// Deletes `resource` and clears its placement entry, under the unique lock.
    pub fn delete_resource(&self, hold: Hold<'_>, resource: u32) -> RaiseResult<()> {
        let _lock = self.mutex.write();
        self.live_manager()
            .delete_object::<Resources>(hold, resource, true);
        Ok(())
    }

    /// Deletes the object a serialized delete command names, under the unique lock. A
    /// refused command is ignored.
    pub fn delete_object_with_serialized_data(
        &self,
        hold: Hold<'_>,
        data: &[u8],
    ) -> RaiseResult<()> {
        let _lock = self.mutex.write();
        match self.manager() {
            Some(rm) => {
                let _ = rm.delete_serialized_object(hold, data);
            }
            // After teardown only a command that reaches the manager's state dies; the others
            // are refused first (their log goes nowhere without the manager's platform).
            None if serialized_delete_reaches_manager(data) => fatal!(DeathKind::TornDown),
            None => {}
        }
        Ok(())
    }

    /// Discards each resource the records name, in order, under the unique lock. The resources
    /// it deleted come back ([`Deleted`]): the caller lets them go after its completion.
    pub fn discard_resources(
        &self,
        hold: Hold<'_>,
        resources: Records<'_, ResourceReference>,
    ) -> RaiseResult<Deleted<P>> {
        let _lock = self.mutex.write();
        let rm = self.manager();
        Self::discard_loop(hold, rm.as_deref(), &resources)
    }

    /// The discard loop, shared by `discard_resources` and a discarding synchronize: a
    /// resource that supports discard is discarded (a raise ends the loop, and carries what was
    /// deleted on to where it ends), any other is deleted with its placement entry kept.
    fn discard_loop(
        hold: Hold<'_>,
        rm: Option<&ResourceManager<P>>,
        resources: &Records<'_, ResourceReference>,
    ) -> RaiseResult<Deleted<P>> {
        let mut deleted = Deleted(Vec::new());
        for record in resources.iter() {
            let reference = record.reference;
            let Some(resource) = existing(rm, reference) else {
                continue;
            };
            if resource.supports_discard() {
                if let Err(raised) = resource.discard() {
                    drop(resource);
                    return Err(deleted.carried_by(raised));
                }
            } else if let Some(rm) = rm {
                // `existing` found the resource through this manager, so it is there
                rm.delete_object::<Resources>(hold, reference, false);
                deleted.0.push(resource);
            }
        }
        Ok(deleted)
    }

    /// Flushes the resource manager's resources, under the unique lock.
    pub fn flush_resources(&self, hold: Hold<'_>) -> RaiseResult<()> {
        let _lock = self.mutex.write();
        self.live_manager().flush_resources(hold)
    }

    /// The existing resource `resource` names, without making one.
    pub fn get_existing_host_resource(&self, resource: u32) -> Option<Arc<Resource<P>>> {
        let rm = self.manager();
        existing(rm.as_deref(), resource)
    }

    /// The resource `resource` names, made on first use. Reference 0 is none; a lookup the
    /// manager refuses raises.
    pub fn get_host_resource(
        &self,
        hold: Hold<'_>,
        resource: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        if resource == 0 {
            return Ok(None);
        }
        match self
            .live_manager()
            .get_object::<Resources>(hold, resource)?
        {
            Ok(object) => Ok(object),
            Err(error) => raise_caught(
                RaiseKind::ResourceLookup,
                format_args!("resource {resource} not found: {error}"),
            ),
        }
    }

    /// The object-list entry through the task: unsupported in process.
    pub fn get_object(&self, object: u32) -> RaiseResult<ObjectEntry> {
        let _ = object;
        // In process, reading an object entry through the task ends the process.
        self.unsupported(Unsupported::GetObject)
    }

    /// Applies the records' validity flags to each resource they name, in order, under the
    /// unique lock.
    pub fn invalidate_resources(
        &self,
        resources: Records<'_, ResourceInvalidation>,
    ) -> RaiseResult<()> {
        let _lock = self.mutex.write();
        let rm = self.manager();
        for record in resources.iter() {
            let Some(resource) = existing(rm.as_deref(), record.reference) else {
                continue;
            };
            // Guest clear, guest set, host clear, host set; each flag is a whole byte tested for
            // nonzero, so clear+set leaves the bit set.
            if record.clear_guest_valid != 0 {
                resource.set_is_guest_valid(false);
            }
            if record.set_guest_valid != 0 {
                resource.set_is_guest_valid(true);
            }
            if record.clear_host_valid != 0 {
                resource.set_is_host_valid(false);
            }
            if record.set_host_valid != 0 {
                resource.set_is_host_valid(true);
            }
        }
        Ok(())
    }

    /// Reserves the objects of a mapped object list again: with argument buffers and a
    /// placement list only, under the object database's lock, mapping both lists first.
    ///
    /// The task's features are read first (through its manager, or after `teardown` through
    /// the task teardown kept): a torn-down task without argument buffers returns, and only one
    /// with them dies, at the object lists.
    pub fn rebuild_task(&self, hold: Hold<'_>) -> RaiseResult<()> {
        let parts = self.parts();
        let task = match &parts {
            Some((_, rm, _)) => rm.task(),
            None => self.torn_down_host.lock().as_ref().and_then(Weak::upgrade),
        };
        let Some(task) = task else {
            fatal!(DeathKind::TaskGone)
        };
        let supports_argument_buffers = task.features().load().supports_argument_buffers;
        drop(task);
        if !supports_argument_buffers {
            return Ok(());
        }
        let Some((_, rm, lists)) = parts else {
            fatal!(DeathKind::TornDown)
        };
        if lists.placement_list_length() == 0 {
            return Ok(());
        }
        let _lock = self.object_database_mutex.lock();
        if lists.ensure_mapped(hold) {
            // A Metal raise from the reservation goes on
            if let Err(error) = rm.reserve_objects()? {
                log::warn!("objects not reserved: {error}");
            }
        }
        Ok(())
    }

    /// Discards the resource `physical` names, whether or not it supports discard, under the
    /// unique lock.
    pub fn replace_physical(&self, physical: u32) -> RaiseResult<()> {
        let _lock = self.mutex.write();
        let rm = self.manager();
        if let Some(resource) = existing(rm.as_deref(), physical) {
            resource.discard()?;
        }
        Ok(())
    }

    /// Synchronizes the resources the records name into guest memory, then (with `discard`)
    /// discards them, under the unique lock throughout.
    ///
    /// CPU paging runs first, then one paging buffer (made even with nothing to encode) gets
    /// one blit encoder, then one compute encoder, for the resources that need them. A raise
    /// in one resource's paging is logged and the loop goes on. `done` runs on the paging
    /// buffer's completion (a Metal thread), possibly before the discards end. A raise after
    /// the handler was added (the commit, the discards) returns `Err` and the completion still
    /// arrives: the caller sees both.
    pub fn synchronize_resources(
        &self,
        hold: Hold<'_>,
        resources: Records<'_, ResourceReference>,
        discard: bool,
        done: Completion,
    ) -> RaiseResult<()> {
        // The unique lock is held to the end: across the commit, the scheduling wait and the
        // discard loop, which the completion may precede.
        let _lock = self.mutex.write();
        let rm = self.manager();
        let mut blit_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut compute_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut cpu_resources: Vec<Arc<Resource<P>>> = Vec::new();
        for record in resources.iter() {
            let Some(resource) = existing(rm.as_deref(), record.reference) else {
                continue;
            };
            if !resource.needs_sync() {
                continue;
            }
            if resource.needs_cpu_paging() {
                cpu_resources.push(resource);
            } else if resource.needs_compute_paging() {
                compute_resources.push(resource);
            } else {
                blit_resources.push(resource);
            }
        }
        for resource in &cpu_resources {
            if let Err(raised) = resource.synchronize_with_cpu() {
                log::warn!(
                    "a resource's CPU synchronize raised, going on with the rest: {}",
                    raised.exception()
                );
                hold.land(raised);
            } else {
                resource.flush_to_guest();
            }
        }
        let command_buffer = match &self.paging_queue {
            Some(queue) => queue.command_buffer(OnFault::Return)?,
            None => None,
        };
        if !blit_resources.is_empty() || !compute_resources.is_empty() {
            if !blit_resources.is_empty() {
                let encoder = match &command_buffer {
                    Some(buffer) => buffer.blit_encoder(OnFault::Return)?,
                    None => None,
                };
                for resource in &blit_resources {
                    if let Err(raised) = resource.synchronize_in_encoder(hold, encoder.as_ref()) {
                        log::warn!(
                            "a resource's blit synchronize raised, going on with the rest: {}",
                            raised.exception()
                        );
                        hold.land(raised);
                    }
                }
                if let Some(encoder) = &encoder {
                    encoder.encode(OnFault::Return, BlitCmd::EndEncoding)?;
                }
            }
            if !compute_resources.is_empty() {
                let encoder = match &command_buffer {
                    Some(buffer) => buffer.compute_encoder(OnFault::Return)?,
                    None => None,
                };
                for resource in &compute_resources {
                    if let Err(raised) =
                        resource.synchronize_in_compute_encoder(hold, encoder.as_ref())
                    {
                        log::warn!(
                            "a resource's compute synchronize raised, going on with the rest: {}",
                            raised.exception()
                        );
                        hold.land(raised);
                    }
                }
                if let Some(encoder) = &encoder {
                    encoder.encode(OnFault::Return, ComputeCmd::EndEncoding)?;
                }
            }
        }
        let to_flush: Vec<Arc<Resource<P>>> = blit_resources
            .iter()
            .chain(&compute_resources)
            .cloned()
            .collect();
        // A nil buffer never runs the handler: no completion. The handler keeps the buffer
        // alive until it has run.
        if let Some(buffer) = &command_buffer {
            let captured = buffer.clone_ref();
            let handler = move |info: CompletedInfo| {
                if info.status == CommandBufferStatus::COMPLETED {
                    for r in &to_flush {
                        r.flush_to_guest();
                    }
                }
                completed(&done, info);
                drop(captured);
            };
            // An isolated task's handler runs as the task's code, and not at all once it ended.
            let handler: CompletedHandler = match TaskLife::current() {
                Some(life) => Box::new(life.carry(handler)),
                None => Box::new(handler),
            };
            buffer.add_completed_handler(OnFault::Return, handler)?;
            buffer.commit(OnFault::Return)?;
            buffer.wait_until_scheduled()?;
        }
        drop(command_buffer);
        drop((blit_resources, compute_resources, cpu_resources));
        // The deleted resources go at the end, or with a raise.
        if discard {
            let deleted = Self::discard_loop(hold, rm.as_deref(), &resources)?;
            drop(deleted);
        }
        Ok(())
    }

    /// A fresh buffer of `length` bytes that `copy` (the kind's copy from task memory) fills.
    ///
    /// A length past the largest task (32 GiB) raises before anything is allocated: no local
    /// task holds such a range, so the copy would raise anyway, and the guest sees the fault it
    /// saw when the host could not allocate the buffer. The buffer is made zeroed (lazily mapped
    /// for a large one), so a length the copy then refuses touches no memory, and a raise in the
    /// copy frees it. The bytes a failed physical read inside the copy leaves are unlockable;
    /// here they are zero. Its raise is as `on` says: the task's reads are reached both under a
    /// handler and with none.
    pub fn data_from_virtual_offset(
        &self,
        on: OnFault,
        offset: u64,
        length: u64,
        copy: impl FnOnce(u64, &mut [u8]) -> RaiseResult<()>,
    ) -> RaiseResult<Option<Vec<u8>>> {
        if length > MAX_LOCAL_TASK_LENGTH {
            return raise_on(
                on,
                RaiseKind::BufferLength,
                format_args!("buffer of {length:#x} bytes is longer than any task"),
            );
        }
        let mut buffer = vec![0u8; length.to_usize()];
        copy(offset, &mut buffer)?;
        Ok(Some(buffer))
    }

    /// Ends the process: `op` is one the task's kind does not provide.
    pub fn unsupported(&self, op: Unsupported) -> ! {
        fatal!(DeathKind::Unsupported(op))
    }

    /// The paging queue, retained for the caller.
    pub fn paging_queue(&self) -> Option<P::CommandQueue> {
        self.paging_queue.as_ref().map(Handle::clone_ref)
    }

    /// The deserializer, lent for this call; `None` after `teardown`.
    pub fn deserializer(&self) -> Option<CallBorrow<'_, Deserializer<P>>> {
        self.parts
            .lock()
            .as_ref()
            .map(|p| CallBorrow::new(Arc::clone(&p.deserializer)))
    }

    /// The resource manager, lent for this call; `None` after `teardown`.
    pub fn resource_manager(&self) -> Option<CallBorrow<'_, ResourceManager<P>>> {
        self.manager().map(CallBorrow::new)
    }

    /// The Metal device, retained for the caller; `None` is nil.
    pub fn mtl_device(&self) -> Option<P::Device> {
        self.mtl_device.as_deref().map(Handle::clone_ref)
    }

    /// Whether buffers can be made over address ranges.
    pub fn supports_buffer_with_address_ranges(&self) -> bool {
        self.supports_buffer_with_address_ranges
    }

    /// Whether heaps can be made over address ranges.
    pub fn supports_heap_with_address_ranges(&self) -> bool {
        self.supports_heap_with_address_ranges
    }

    /// Whether a resource's backing can be detached.
    pub fn supports_resource_detach_backing(&self) -> bool {
        self.supports_resource_detach_backing
    }
}

/// The paging buffer's completion: 1 when the buffer completed, else 3 with its error. The
/// platform reads the buffer's status and error when the handler runs.
fn completed(done: &Completion, info: CompletedInfo) {
    if info.status == CommandBufferStatus::COMPLETED {
        done(ExecStatus::OK, None);
    } else {
        done(ExecStatus::GPU_OR_TIMEOUT, info.error);
    }
}

impl<P: Platform> Drop for TaskCore<P> {
    /// The parts go first, in their order (the deserializer, the resource manager, the object
    /// lists; a torn-down task has none left), then the Metal device, the paging queue and the
    /// paging event.
    fn drop(&mut self) {
        if let Some(parts) = self.parts.get_mut().take() {
            parts.release();
        }
        let device = self.mtl_device.take();
        let queue = self.paging_queue.take();
        let event = self.paging_event.take();
        drop(device);
        drop(queue);
        drop(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fifo::wire::Records;
    use crate::guest::records::{GuestFeatures, SharedFeatures};
    use crate::memory::mapping::{Mapping, Teardown};
    use crate::platform::device::GpuDevice;
    use crate::platform::types::NsErrorText;
    use crate::raise::death::DeathKind as Fault;
    use crate::raise::death::DeathSite;
    use crate::resources::manager::fixture::{Fx, P, PL, Placement, REC};
    use crate::resources::resource::with_platform;
    use crate::resources::texture::views::SerializerTextureResource;
    use crate::tasks::{ResourceInvalidation, ResourceReference};
    use crate::testkit::capture::catch_death;
    use crate::testkit::{CompleteOn, Step, kind};
    use zerocopy::IntoBytes;

    const RESERVE: &str = "reserveResourceIndicesForResourceType:indices:indexCount:";

    /// A task base over the resource manager fixture: its manager, object list manager and a
    /// deserializer over the manager's delegate, a fresh paging queue and event.
    struct Tx {
        fx: Fx,
        base: TaskCore<P>,
    }

    fn tx_with(entries: &[(u8, u32, u64)], placement: Placement, queue: bool) -> Tx {
        let fx = Fx::new(entries, placement);
        let device = fx.platform.device();
        let paging_queue = if queue {
            device.new_command_queue()
        } else {
            None
        };
        let paging_event = device.new_event(OnFault::Die).unwrap();
        let base = TaskCore::new(
            Some(Arc::new(device)),
            0x1234,
            paging_queue,
            paging_event,
            1,
            false,
            true,
            false,
        );
        let deserializer = Deserializer::new(
            fx.platform.clone(),
            Some(fx.platform.device()),
            fx.m.object_delegate(),
        );
        base.set_parts(TaskParts {
            deserializer,
            resource_manager: fx.m.clone(),
            object_list_manager: fx.olm.clone(),
        });
        Tx { fx, base }
    }

    fn tx() -> Tx {
        tx_with(&entries3(), Placement::Off, true)
    }

    /// Runs `f` inside the fixture's platform scope; the tasks area's lines it wrote.
    fn logging<T>(fx: &Fx, f: impl FnOnce() -> T) -> (T, Vec<String>) {
        let lines = crate::tasks::test_support::Lines::capture();
        let r = with_platform(&fx.platform, f);
        (r, lines.take())
    }

    fn entries3() -> Vec<(u8, u32, u64)> {
        vec![(0, 0, 0), (6, 0x20, REC), (6, 0x10, REC + 0x100)]
    }

    fn texture(fx: &Fx) -> Fk<kind::Texture> {
        fx.platform
            .world()
            .new_obj::<kind::Texture>(Default::default())
    }

    use crate::testkit::Fk;

    /// A texture view (a serializer texture, discardable or not).
    fn view(fx: &Fx, discardable: bool) -> Arc<Resource<P>> {
        SerializerTextureResource::new(texture(fx), discardable)
    }

    /// A view that needs a sync (guest-invalid, host-valid): a blit paging whose
    /// `synchronizeInEncoder:` raises.
    fn stale_view(fx: &Fx, discardable: bool) -> Arc<Resource<P>> {
        let r = view(fx, discardable);
        r.set_is_host_valid(true);
        r.set_is_guest_valid(false);
        r
    }

    fn register(fx: &Fx, reference: u32, r: &Arc<Resource<P>>) {
        assert!(fx.m.register_object::<Resources>(reference, Arc::clone(r)));
    }

    fn refs(v: &[u32]) -> Vec<u8> {
        v.iter()
            .map(|&reference| ResourceReference { reference })
            .collect::<Vec<_>>()
            .as_bytes()
            .to_vec()
    }

    fn records(bytes: &[u8]) -> Records<'_, ResourceReference> {
        Records::new(bytes, (bytes.len() / 4) as u32).unwrap()
    }

    type Seen = Arc<parking_lot::Mutex<Vec<(ExecStatus, Option<NsErrorText>)>>>;

    fn completion() -> (crate::tasks::Completion, Seen) {
        let seen: Seen = Arc::default();
        let s = seen.clone();
        (Arc::new(move |st, e| s.lock().push((st, e))), seen)
    }

    fn features(fx: &Fx, argument_buffers: bool) {
        let a = GuestFeatures {
            supports_argument_buffers: argument_buffers,
            ..Default::default()
        };
        let s = SharedFeatures::new();
        s.store(&a);
        fx.task.set_features(Arc::new(s));
    }

    const TORN_DOWN: DeathSite = DeathKind::TornDown.site();

    /// A dual-plane compute texture (`needsComputePaging` YES, `supportsDiscard` YES; its
    /// `discard` sends `setPurgeableState:`, which a test can script to raise).
    fn compute_texture(fx: &Fx) -> Arc<Resource<P>> {
        use crate::resources::texture::create_texture_resource;
        use crate::resources::texture::test_support::{desc, header, mip, paging, rec2};
        let tex = crate::testkit::impls::new_texture(
            fx.platform.world(),
            desc(550, 4, 4, 1, 1),
            None,
            0,
            None,
            false,
        );
        tex.state().device = Some(fx.platform.device().erase());
        let r = rec2(
            header(0, 4, 1),
            vec![mip(0, 0x40, 16, 4, 4, 1)],
            header(0, 4, 1),
            vec![mip(0x40, 0x40, 16, 4, 4, 1)],
        );
        with_platform(&fx.platform, || {
            create_texture_resource(
                Arc::downgrade(&fx.task_dyn),
                12,
                &paging(0, 0x80),
                &r,
                0x78,
                tex,
            )
        })
        .unwrap()
        .unwrap()
    }

    /// A resource whose `supportsDiscard` is YES and whose `discard` raises.
    fn raising_discard(fx: &Fx) -> Arc<Resource<P>> {
        let r = compute_texture(fx);
        assert!(r.supports_discard());
        fx.platform.script().push(
            "setPurgeableState:",
            Step::raise("NSInternalInconsistencyException", "x"),
        );
        r
    }

    mod construct_and_getters {
        use super::*;

        #[test]
        fn new_keeps_its_parts_and_the_getters_answer_them() {
            let t = tx();
            let b = &t.base;
            assert_eq!(b.length, 0x1234);
            assert_eq!(
                b.mtl_device().map(|d| d.id()),
                Some(t.fx.platform.device().id())
            );
            assert_eq!(
                b.paging_queue().map(|q| q.id()),
                b.paging_queue.as_ref().map(|q| q.id())
            );
            assert!(b.paging_queue().is_some());
            assert!(b.paging_event.is_some());
            assert_eq!(
                b.paging_counter.load(std::sync::atomic::Ordering::Relaxed),
                1
            );
            assert!(!b.supports_buffer_with_address_ranges());
            assert!(b.supports_heap_with_address_ranges());
            assert!(!b.supports_resource_detach_backing());
            let (d, m, o) = b.parts().unwrap();
            assert!(Arc::ptr_eq(&m, &t.fx.m));
            assert!(Arc::ptr_eq(&o, &t.fx.olm));
            assert!(std::ptr::eq(&*b.deserializer().unwrap(), &*d));
            assert!(std::ptr::eq(&*b.resource_manager().unwrap(), &*t.fx.m));
            // The locks start free.
            drop(b.mutex.try_write().unwrap());
            drop(b.object_database_mutex.try_lock().unwrap());
        }

        #[test]
        fn a_nil_paging_queue_is_none() {
            let t = tx_with(&entries3(), Placement::Off, false);
            assert!(t.base.paging_queue().is_none());
        }

        /// `set_parts` a second time releases the first parts.
        #[test]
        fn set_parts_replaces_and_releases() {
            let t = tx();
            let (d, _, _) = t.base.parts().unwrap();
            let weak = Arc::downgrade(&d);
            drop(d);
            let deserializer = Deserializer::new(
                t.fx.platform.clone(),
                Some(t.fx.platform.device()),
                t.fx.m.object_delegate(),
            );
            t.base.set_parts(TaskParts {
                deserializer: deserializer.clone(),
                resource_manager: t.fx.m.clone(),
                object_list_manager: t.fx.olm.clone(),
            });
            assert!(weak.upgrade().is_none());
            assert!(Arc::ptr_eq(&t.base.parts().unwrap().0, &deserializer));
        }
    }

    mod teardown {
        use super::*;

        /// The three parts go (the task was their only owner), under the unique lock.
        #[test]
        fn releases_the_parts_under_the_unique_lock() {
            let Tx { fx, base } = tx();
            let Fx { m, olm, .. } = fx;
            let (d, _, _) = base.parts().unwrap();
            let (wd, wm, wo) = (Arc::downgrade(&d), Arc::downgrade(&m), Arc::downgrade(&olm));
            drop((d, m, olm));
            let guard = base.mutex.read();
            std::thread::scope(|s| {
                let h = s.spawn(|| base.teardown());
                std::thread::sleep(std::time::Duration::from_millis(50));
                // Blocked behind the shared holder.
                assert!(base.parts().is_some());
                drop(guard);
                h.join().unwrap();
            });
            assert!(base.parts().is_none());
            assert!(base.deserializer().is_none());
            assert!(base.resource_manager().is_none());
            assert!(wd.upgrade().is_none());
            assert!(wm.upgrade().is_none());
            assert!(wo.upgrade().is_none());
            // A second teardown finds nothing to release.
            base.teardown();
        }

        /// Use after teardown: each op that reaches a part dies, and only such an op (reference
        /// 0, an empty record list, a short or unknown serialized delete do not).
        #[test]
        fn later_uses_die_at_the_op() {
            let t = tx();
            t.base.teardown();
            let b = &t.base;
            assert_eq!(
                catch_death(|| b.delete_resource(Hold::NONE, 1)).err(),
                Some(TORN_DOWN)
            );
            assert_eq!(
                catch_death(|| b.flush_resources(Hold::NONE)).err(),
                Some(TORN_DOWN)
            );
            assert_eq!(
                catch_death(|| b.get_host_resource(Hold::NONE, 1)).err(),
                Some(TORN_DOWN)
            );
            assert_eq!(
                b.get_host_resource(Hold::NONE, 0).unwrap().map(|_| ()),
                None
            );
            assert_eq!(
                catch_death(|| b.get_existing_host_resource(2)).err(),
                Some(TORN_DOWN)
            );
            assert!(b.get_existing_host_resource(0).is_none());
            let cmd = [1000u32, 0, 1].as_bytes().to_vec();
            assert_eq!(
                catch_death(|| b.delete_object_with_serialized_data(Hold::NONE, &cmd)).err(),
                Some(TORN_DOWN)
            );
            let bad = [999u32, 0, 1].as_bytes().to_vec();
            assert!(
                b.delete_object_with_serialized_data(Hold::NONE, &bad)
                    .is_ok()
            );
            assert!(
                b.delete_object_with_serialized_data(Hold::NONE, &cmd[..8])
                    .is_ok()
            );
            // Features first: only with argument buffers is the NULL `_objectListManager` reached
            assert!(b.rebuild_task(Hold::NONE).is_ok());
            features(&t.fx, true);
            assert_eq!(
                catch_death(|| b.rebuild_task(Hold::NONE)).err(),
                Some(TORN_DOWN)
            );
            let zero = refs(&[0, 0]);
            assert!(b.discard_resources(Hold::NONE, records(&zero)).is_ok());
            assert!(b.replace_physical(0).is_ok());
            let one = refs(&[0, 3]);
            assert_eq!(
                catch_death(|| b.discard_resources(Hold::NONE, records(&one))).err(),
                Some(TORN_DOWN)
            );
            assert_eq!(catch_death(|| b.replace_physical(3)).err(), Some(TORN_DOWN));
            let inv = ResourceInvalidation {
                reference: 0,
                ..Default::default()
            };
            let bytes = [inv].as_bytes().to_vec();
            assert!(
                b.invalidate_resources(Records::new(&bytes, 1).unwrap())
                    .is_ok()
            );
            let inv = ResourceInvalidation {
                reference: 7,
                ..Default::default()
            };
            let bytes = [inv].as_bytes().to_vec();
            assert_eq!(
                catch_death(|| b.invalidate_resources(Records::new(&bytes, 1).unwrap())).err(),
                Some(TORN_DOWN)
            );
            let (done, _) = completion();
            assert!(
                b.synchronize_resources(Hold::NONE, records(&zero), true, done.clone())
                    .is_ok()
            );
            assert_eq!(
                catch_death(|| b.synchronize_resources(Hold::NONE, records(&one), false, done))
                    .err(),
                Some(TORN_DOWN)
            );
            // Nothing is held after a death: the lock is free again.
            drop(b.mutex.try_write().unwrap());
        }
    }

    mod drop {
        use super::*;

        #[test]
        fn dropping_the_task_releases_its_parts() {
            let Tx { fx, base } = tx();
            let Fx { m, olm, .. } = fx;
            let (d, _, _) = base.parts().unwrap();
            let (wd, wm, wo) = (Arc::downgrade(&d), Arc::downgrade(&m), Arc::downgrade(&olm));
            drop((d, m, olm));
            drop(base);
            assert!(wd.upgrade().is_none());
            assert!(wm.upgrade().is_none());
            assert!(wo.upgrade().is_none());
        }
    }

    mod get_host_resource {
        use super::*;

        #[test]
        fn zero_is_nil_without_touching_the_manager() {
            let t = tx();
            let mark = t.fx.clear_log();
            assert!(t.base.get_host_resource(Hold::NONE, 0).unwrap().is_none());
            assert!(t.fx.sels_since(mark).is_empty());
        }

        #[test]
        fn a_cached_resource_is_returned() {
            let t = tx();
            let r = view(&t.fx, false);
            register(&t.fx, 2, &r);
            let got = t.base.get_host_resource(Hold::NONE, 2).unwrap().unwrap();
            assert!(Arc::ptr_eq(&got, &r));
        }

        /// The manager's error raises, with the manager's error in the detail.
        #[test]
        fn an_error_raises() {
            let t = tx();
            let hold = Hold::NONE;
            let e = t.base.get_host_resource(hold, 5).err().unwrap();
            assert!(matches!(e.origin(), crate::raise::Origin::Core(_)));
            assert_eq!(
                e.detail().unwrap_or_default(),
                "resource 5 not found: reference 5 is past the object list's 3 entries"
            );
            assert_eq!((e.site().file, e.site().line), ("tasks.resource.lookup", 0));
        }

        /// Under the handler every caller has, the raise names the lookup as the site it dies at
        /// with none.
        #[test]
        fn an_error_names_the_lookup_site() {
            let t = tx();
            assert_eq!(
                t.base
                    .get_host_resource(Hold::NONE, 5)
                    .err()
                    .map(|r| r.site()),
                Some(DeathSite {
                    kind: Fault::UnhandledRaise,
                    file: "tasks.resource.lookup",
                    line: 0
                })
            );
        }
    }

    mod get_existing_host_resource {
        use super::*;

        #[test]
        fn never_creates() {
            let t = tx();
            let mark = t.fx.clear_log();
            assert!(t.base.get_existing_host_resource(0).is_none());
            // Entry 1 exists in the list but was never created: nil, and no cursor asked for.
            assert!(t.base.get_existing_host_resource(1).is_none());
            assert!(t.base.get_existing_host_resource(99).is_none());
            assert!(t.fx.sels_since(mark).is_empty());
            let r = view(&t.fx, false);
            register(&t.fx, 1, &r);
            assert!(Arc::ptr_eq(
                &t.base.get_existing_host_resource(1).unwrap(),
                &r
            ));
        }
    }

    mod delete_resource {
        use super::*;

        /// clearPlacement = true: the placement entry is zeroed.
        #[test]
        fn deletes_and_clears_the_placement() {
            let t = tx_with(&entries3(), Placement::Mapped, true);
            register(&t.fx, 1, &view(&t.fx, false));
            t.fx.set_placement(1, 0x10, 5);
            t.base.delete_resource(Hold::NONE, 1).unwrap();
            assert!(t.base.get_existing_host_resource(1).is_none());
            assert_eq!(t.fx.placement(1), (0, 0));
        }
    }

    mod delete_object_with_serialized_data {
        use super::*;

        #[test]
        fn a_resource_delete_removes_it() {
            let t = tx();
            register(&t.fx, 2, &view(&t.fx, false));
            let cmd = [1001u32, 0, 2].as_bytes().to_vec();
            t.base
                .delete_object_with_serialized_data(Hold::NONE, &cmd)
                .unwrap();
            assert!(t.base.get_existing_host_resource(2).is_none());
        }

        /// An unknown command is ignored (the bool is dropped).
        #[test]
        fn an_unknown_command_is_ignored() {
            let t = tx();
            register(&t.fx, 2, &view(&t.fx, false));
            let cmd = [1004u32, 0, 2].as_bytes().to_vec();
            let r = t.base.delete_object_with_serialized_data(Hold::NONE, &cmd);
            assert!(r.is_ok());
            assert!(t.base.get_existing_host_resource(2).is_some());
        }

        #[test]
        fn which_commands_reach_the_manager() {
            for op in [
                1000u32, 1001, 1012, 1002, 1003, 1005, 1006, 1007, 1009, 1014,
            ] {
                assert!(serialized_delete_reaches_manager([op, 0, 1].as_bytes()));
                assert!(!serialized_delete_reaches_manager(&[op, 0].as_bytes()[..8]));
            }
            for op in [0u32, 999, 1004, 1008, 1010, 1011, 1013, 1015] {
                assert!(!serialized_delete_reaches_manager([op, 0, 1].as_bytes()));
            }
        }
    }

    mod flush_resources {
        use super::*;

        /// The manager's flush runs under the lock (here it evicts one stale view, whose
        /// eviction raises and is caught inside the manager).
        #[test]
        fn flushes_through_the_manager() {
            let t = tx();
            register(&t.fx, 1, &stale_view(&t.fx, false));
            t.fx.platform
                .world()
                .set_auto_complete(Some(CompleteOn::Caller));
            let (r, lines) = t.fx.logged(|| t.base.flush_resources(Hold::NONE));
            assert!(r.is_ok());
            assert_eq!(lines.len(), 1, "the eviction's raise is caught and logged");
            assert_eq!(t.fx.log().calls_to("waitUntilCompleted").len(), 1);
        }
    }

    mod invalidate_resources {
        use super::*;

        fn run(t: &Tx, recs: &[ResourceInvalidation]) {
            let bytes = recs.as_bytes().to_vec();
            t.base
                .invalidate_resources(Records::new(&bytes, recs.len() as u32).unwrap())
                .unwrap();
        }

        fn rec(reference: u32, ch: u8, sh: u8, cg: u8, sg: u8) -> ResourceInvalidation {
            ResourceInvalidation {
                reference,
                clear_host_valid: ch,
                set_host_valid: sh,
                clear_guest_valid: cg,
                set_guest_valid: sg,
            }
        }

        /// Guest clear, guest set, host clear, host set: clear+set leaves set; any nonzero byte
        /// counts.
        #[test]
        fn flags_apply_in_order() {
            let t = tx();
            let r = view(&t.fx, false);
            register(&t.fx, 1, &r);
            run(&t, &[rec(1, 1, 0x80, 2, 0xff)]);
            assert!(r.is_guest_valid());
            assert!(r.is_host_valid());
            run(&t, &[rec(1, 1, 0, 1, 0)]);
            assert!(!r.is_guest_valid());
            assert!(!r.is_host_valid());
            run(&t, &[rec(1, 0, 0, 0, 1)]);
            assert!(r.is_guest_valid());
            assert!(!r.is_host_valid());
            run(&t, &[rec(1, 0, 0, 0, 0)]);
            assert!(r.is_guest_valid());
            assert!(!r.is_host_valid());
        }

        /// Zero and unknown references are skipped; the loop goes on; nothing is created.
        #[test]
        fn zero_and_unknown_are_skipped() {
            let t = tx();
            let r = view(&t.fx, false);
            register(&t.fx, 2, &r);
            let mark = t.fx.clear_log();
            run(
                &t,
                &[rec(0, 0, 1, 0, 1), rec(1, 0, 1, 0, 1), rec(2, 0, 1, 0, 1)],
            );
            assert!(t.fx.sels_since(mark).is_empty());
            assert!(r.is_host_valid() && r.is_guest_valid());
            assert!(t.base.get_existing_host_resource(1).is_none());
        }
    }

    mod synchronize_resources {
        use super::*;

        /// Nothing needs a sync: the paging buffer is still made, gets the handler, is
        /// committed and waited on until scheduled; its completion reports 1.
        #[test]
        fn an_empty_sync_still_commits_a_buffer() {
            let t = tx();
            register(&t.fx, 1, &view(&t.fx, false)); // in sync: no paging
            let (done, seen) = completion();
            let b = refs(&[0, 1, 2]);
            let mark = t.fx.clear_log();
            t.base
                .synchronize_resources(Hold::NONE, records(&b), false, done)
                .unwrap();
            assert_eq!(
                t.fx.sels_since(mark),
                vec![
                    "commandBuffer",
                    "addCompletedHandler:",
                    "commit",
                    "waitUntilScheduled"
                ]
            );
            assert!(seen.lock().is_empty());
            assert_eq!(t.fx.platform.world().complete_all(CompleteOn::Caller), 1);
            assert_eq!(*seen.lock(), vec![(ExecStatus::OK, None)]);
        }

        /// A buffer that ends in error completes with 3 and its error.
        #[test]
        fn an_error_completes_with_three() {
            let t = tx();
            let (done, seen) = completion();
            let b = refs(&[]);
            t.base
                .synchronize_resources(Hold::NONE, records(&b), false, done)
                .unwrap();
            let w = t.fx.platform.world();
            let id = w.committed()[0];
            let err = NsErrorText {
                domain: "MTLCommandBufferErrorDomain".into(),
                code: 2,
                description: "gpu".into(),
            };
            assert!(w.complete(
                id,
                CommandBufferStatus::ERROR,
                Some(err.clone()),
                CompleteOn::Caller
            ));
            assert_eq!(*seen.lock(), vec![(ExecStatus::GPU_OR_TIMEOUT, Some(err))]);
        }

        /// Blit paging: one blit encoder for every blit resource; a raise is caught, logged, and
        /// the loop goes on; then `endEncoding`.
        #[test]
        fn blit_resources_share_an_encoder_and_raises_are_caught() {
            let t = tx();
            register(&t.fx, 1, &stale_view(&t.fx, false));
            register(&t.fx, 2, &stale_view(&t.fx, true));
            let (done, _) = completion();
            let b = refs(&[1, 2]);
            let mark = t.fx.clear_log();
            let (r, lines) = logging(&t.fx, || {
                t.base
                    .synchronize_resources(Hold::NONE, records(&b), false, done)
            });
            assert!(r.is_ok());
            assert_eq!(
                t.fx.sels_since(mark),
                vec![
                    "commandBuffer",
                    "blitCommandEncoder",
                    "endEncoding",
                    "addCompletedHandler:",
                    "commit",
                    "waitUntilScheduled"
                ]
            );
            assert_eq!(
                lines,
                ["a resource's blit synchronize raised, going on with the rest: cannot \
                     synchronize a child texture"; 2]
            );
        }

        /// CPU paging comes first, before the buffer: an ICB's write-back raise is caught and
        /// logged; no encoder is made for it.
        #[test]
        fn cpu_resources_sync_before_the_buffer() {
            let t = tx();
            let icb = crate::resources::icb::IcbResource::new(
                Arc::downgrade(&t.fx.task_dyn),
                &[0; 12],
                None,
                Default::default(),
                Default::default(),
            )
            .unwrap()
            .unwrap();
            icb.set_is_host_valid(true);
            icb.set_is_guest_valid(false);
            register(&t.fx, 1, &icb);
            t.fx.platform.script().push(
                "writable_cursor_from_virtual_offset",
                Step::raise("NSGenericException", "x"),
            );
            let (done, _) = completion();
            let b = refs(&[1]);
            let mark = t.fx.clear_log();
            let (r, lines) = logging(&t.fx, || {
                t.base
                    .synchronize_resources(Hold::NONE, records(&b), false, done)
            });
            assert!(r.is_ok());
            let sels = t.fx.sels_since(mark);
            let cursor = sels
                .iter()
                .position(|s| *s == "writable_cursor_from_virtual_offset")
                .unwrap();
            let cb = sels.iter().position(|s| *s == "commandBuffer").unwrap();
            assert!(cursor < cb);
            assert!(!sels.contains(&"blitCommandEncoder"));
            assert!(!sels.contains(&"computeCommandEncoder"));
            assert_eq!(
                lines,
                ["a resource's CPU synchronize raised, going on with the rest: x"]
            );
        }

        /// Compute paging (a dual-plane compute texture) gets a compute encoder, after the
        /// blit encoder when both are needed.
        #[test]
        fn compute_resources_get_a_compute_encoder_after_the_blit_one() {
            let t = tx();
            let compute = compute_texture(&t.fx);
            assert!(compute.needs_compute_paging());
            compute.set_is_host_valid(true);
            compute.set_is_guest_valid(false);
            register(&t.fx, 1, &compute);
            register(&t.fx, 2, &stale_view(&t.fx, false));
            let (done, _) = completion();
            let b = refs(&[1, 2]);
            let mark = t.fx.clear_log();
            let (r, lines) = logging(&t.fx, || {
                t.base
                    .synchronize_resources(Hold::NONE, records(&b), false, done)
            });
            assert!(r.is_ok());
            let sels = t.fx.sels_since(mark);
            let at = |s: &str| sels.iter().position(|x| *x == s).unwrap();
            assert!(at("blitCommandEncoder") < at("computeCommandEncoder"));
            assert!(at("computeCommandEncoder") < at("dispatchThreads:threadsPerThreadgroup:"));
            assert!(at("dispatchThreads:threadsPerThreadgroup:") < at("addCompletedHandler:"));
            assert_eq!(sels.iter().filter(|s| **s == "endEncoding").count(), 2);
            // Only the view's raise (the compute paging succeeded).
            assert_eq!(
                lines,
                [
                    "a resource's blit synchronize raised, going on with the rest: cannot \
                     synchronize a child texture"
                ]
            );
        }

        /// A compute paging's raise is caught, logged, and the encoder still ends.
        #[test]
        fn a_compute_raise_is_logged() {
            let t = tx();
            let compute = compute_texture(&t.fx);
            compute.set_is_host_valid(true);
            compute.set_is_guest_valid(false);
            register(&t.fx, 1, &compute);
            t.fx.platform.script().push(
                "setComputePipelineState:",
                Step::raise("NSInvalidArgumentException", "x"),
            );
            let (done, _) = completion();
            let b = refs(&[1]);
            let mark = t.fx.clear_log();
            let (r, lines) = logging(&t.fx, || {
                t.base
                    .synchronize_resources(Hold::NONE, records(&b), false, done)
            });
            assert!(r.is_ok());
            let sels = t.fx.sels_since(mark);
            assert!(!sels.contains(&"dispatchThreads:threadsPerThreadgroup:"));
            assert!(!sels.contains(&"blitCommandEncoder"));
            assert_eq!(sels.iter().filter(|s| **s == "endEncoding").count(), 1);
            assert_eq!(
                lines,
                ["a resource's compute synchronize raised, going on with the rest: x"]
            );
        }

        /// With `discard`, the loop runs after `waitUntilScheduled`, under the lock; the
        /// completion may come first (here it runs inside the commit). Discardable resources are
        /// discarded; the others deleted without clearing their placement.
        #[test]
        fn discard_runs_after_the_commit_and_the_completion_may_precede_it() {
            let t = tx_with(&entries3(), Placement::Mapped, true);
            t.fx.platform
                .world()
                .set_auto_complete(Some(CompleteOn::Caller));
            let keep = view(&t.fx, true);
            register(&t.fx, 1, &keep);
            register(&t.fx, 2, &view(&t.fx, false));
            t.fx.set_placement(2, 0x20, 9);
            let m = t.fx.m.clone();
            let present_at_completion = Arc::new(parking_lot::Mutex::new(None));
            let p = present_at_completion.clone();
            let done: crate::tasks::Completion = Arc::new(move |st, _| {
                assert_eq!(st, ExecStatus::OK);
                *p.lock() = Some(m.get_existing_object::<Resources>(2).is_some());
            });
            let b = refs(&[1, 2, 0, 7]);
            t.base
                .synchronize_resources(Hold::NONE, records(&b), true, done)
                .unwrap();
            assert_eq!(*present_at_completion.lock(), Some(true));
            assert!(t.base.get_existing_host_resource(1).is_some());
            assert!(t.base.get_existing_host_resource(2).is_none());
            assert_eq!(t.fx.placement(2), (0x20, 9));
        }

        /// A raise in the discard loop escapes after the commit: the caller sees the raise,
        /// and the completion still arrives.
        #[test]
        fn a_discard_raise_escapes_after_the_commit() {
            let t = tx();
            register(&t.fx, 1, &raising_discard(&t.fx));
            register(&t.fx, 2, &view(&t.fx, false));
            let (done, seen) = completion();
            let b = refs(&[1, 2]);
            let hold = Hold::NONE;
            let e = t
                .base
                .synchronize_resources(hold, records(&b), true, done)
                .err()
                .unwrap();
            assert_eq!(
                e.origin(),
                crate::raise::Origin::Boundary("setPurgeableState:")
            );
            // The later record was not reached.
            assert!(t.base.get_existing_host_resource(2).is_some());
            assert_eq!(t.fx.platform.world().complete_all(CompleteOn::Caller), 1);
            assert_eq!(*seen.lock(), vec![(ExecStatus::OK, None)]);
        }

        /// A nil paging queue: every send is to nil, so no completion ever comes; the
        /// discards still run.
        #[test]
        fn a_nil_paging_queue_never_completes() {
            let t = tx_with(&entries3(), Placement::Off, false);
            register(&t.fx, 1, &view(&t.fx, false));
            let (done, seen) = completion();
            let b = refs(&[1]);
            let mark = t.fx.clear_log();
            t.base
                .synchronize_resources(Hold::NONE, records(&b), true, done)
                .unwrap();
            assert!(t.fx.sels_since(mark).is_empty());
            assert!(seen.lock().is_empty());
            assert!(t.base.get_existing_host_resource(1).is_none());
        }

        /// A Metal raise outside the `@try`s (here `commandBuffer`) escapes before any handler
        /// is added.
        #[test]
        fn a_metal_raise_escapes() {
            let t = tx();
            t.fx.platform.script().push(
                "commandBuffer",
                Step::raise("NSInternalInconsistencyException", "x"),
            );
            let (done, seen) = completion();
            let b = refs(&[]);
            let hold = Hold::NONE;
            assert!(
                t.base
                    .synchronize_resources(hold, records(&b), false, done)
                    .is_err()
            );
            assert!(t.fx.log().calls_to("commit").is_empty());
            assert!(seen.lock().is_empty());
            drop(t.base.mutex.try_write().unwrap());
        }
    }

    mod discard_resources {
        use super::*;

        #[test]
        fn discards_or_deletes_in_record_order() {
            let t = tx_with(&entries3(), Placement::Mapped, true);
            let keep = view(&t.fx, true);
            register(&t.fx, 1, &keep);
            register(&t.fx, 2, &view(&t.fx, false));
            t.fx.set_placement(2, 0x20, 9);
            let b = refs(&[0, 1, 2, 5]);
            let mark = t.fx.clear_log();
            let deleted = t.base.discard_resources(Hold::NONE, records(&b)).unwrap();
            // Resource 2 was deleted, and comes back for the caller to let go.
            assert_eq!(deleted.0.len(), 1);
            drop(deleted);
            assert!(t.fx.sels_since(mark).is_empty());
            assert!(t.base.get_existing_host_resource(1).is_some());
            assert!(t.base.get_existing_host_resource(2).is_none());
            // clearPlacement = false.
            assert_eq!(t.fx.placement(2), (0x20, 9));
        }

        /// A discard's raise escapes; later records are not reached.
        #[test]
        fn a_discard_raise_escapes() {
            let t = tx();
            register(&t.fx, 1, &raising_discard(&t.fx));
            register(&t.fx, 2, &view(&t.fx, false));
            let b = refs(&[1, 2]);
            let hold = Hold::NONE;
            assert!(t.base.discard_resources(hold, records(&b)).is_err());
            assert!(t.base.get_existing_host_resource(2).is_some());
        }

        /// A raise carries what the loop deleted before it to where it ends: the catcher's
        /// holder, whose end follows the handler, takes the last release
        #[test]
        fn a_raise_carries_the_deleted_resources_to_the_catchers_holder() {
            let t = tx();
            register(&t.fx, 1, &view(&t.fx, false));
            register(&t.fx, 2, &raising_discard(&t.fx));
            let one = Arc::downgrade(&t.base.get_existing_host_resource(1).unwrap());
            let b = refs(&[1, 2]);
            let held = crate::rt::Held::new();
            {
                let hold = Hold::of(&held);
                let raised = t.base.discard_resources(hold, records(&b)).unwrap_err();
                assert!(t.base.get_existing_host_resource(1).is_none());
                assert!(one.upgrade().is_some(), "carried by the raise");
                hold.land(raised);
            }
            assert!(one.upgrade().is_some(), "held by the holder");
            drop(held);
            assert!(one.upgrade().is_none());
        }
    }

    mod replace_physical {
        use super::*;

        /// `discard` is sent whether or not the resource supports it: a plain view raises.
        #[test]
        fn discards_regardless_of_support() {
            let t = tx();
            register(&t.fx, 1, &view(&t.fx, false));
            register(&t.fx, 2, &view(&t.fx, true));
            let e = t.base.replace_physical(1).err().unwrap();
            assert_eq!(
                e.detail().unwrap_or_default(),
                "cannot discard a child texture that is not discardable"
            );
            assert!(t.base.replace_physical(2).is_ok());
            // Never deleted.
            assert!(t.base.get_existing_host_resource(1).is_some());
            assert!(t.base.get_existing_host_resource(2).is_some());
        }

        #[test]
        fn zero_and_unknown_do_nothing() {
            let t = tx();
            let mark = t.fx.clear_log();
            assert!(t.base.replace_physical(0).is_ok());
            assert!(t.base.replace_physical(2).is_ok());
            assert!(t.fx.sels_since(mark).is_empty());
        }
    }

    mod rebuild_task {
        use super::*;

        fn texture_list() -> Vec<(u8, u32, u64)> {
            vec![(2, 0, 0)]
        }

        #[test]
        fn reserves_when_argument_buffers_and_a_placement_list() {
            let t = tx_with(&texture_list(), Placement::Mapped, true);
            t.fx.set_placement(0, 0x10, 1);
            features(&t.fx, true);
            t.base.rebuild_task(Hold::NONE).unwrap();
            assert_eq!(t.fx.log().calls_to(RESERVE).len(), 1);
        }

        #[test]
        fn nothing_without_argument_buffers() {
            let t = tx_with(&texture_list(), Placement::Mapped, true);
            t.fx.set_placement(0, 0x10, 1);
            features(&t.fx, false);
            t.base.rebuild_task(Hold::NONE).unwrap();
            assert!(t.fx.log().calls_to(RESERVE).is_empty());
        }

        #[test]
        fn nothing_without_a_placement_list() {
            let t = tx_with(&texture_list(), Placement::Off, true);
            features(&t.fx, true);
            let mark = t.fx.clear_log();
            t.base.rebuild_task(Hold::NONE).unwrap();
            assert!(!t.fx.sels_since(mark).contains(&"new_virtual_mapping"));
            assert!(t.fx.log().calls_to(RESERVE).is_empty());
        }

        /// An unmapped placement list is mapped first (under the manager's mutex); a failed
        /// map reserves nothing, a later successful one does.
        #[test]
        fn maps_the_placement_list_first() {
            let t = tx_with(&texture_list(), Placement::Unmapped, true);
            features(&t.fx, true);
            assert!(t.fx.olm.is_object_list_mapped());
            assert!(!t.fx.olm.is_placement_list_mapped());
            let mark = t.fx.clear_log();
            t.base.rebuild_task(Hold::NONE).unwrap();
            assert_eq!(
                t.fx.sels_since(mark)
                    .iter()
                    .filter(|s| **s == "new_virtual_mapping")
                    .count(),
                1
            );
            assert!(t.fx.log().calls_to(RESERVE).is_empty());
            t.fx.platform.script().push(
                "new_virtual_mapping",
                Step::value(Some(Mapping::<P>::new(
                    t.fx.addr(PL),
                    0x4000,
                    true,
                    Teardown::None,
                ))),
            );
            t.fx.set_placement(0, 0x10, 1);
            t.base.rebuild_task(Hold::NONE).unwrap();
            assert!(t.fx.olm.is_placement_list_mapped());
            assert_eq!(t.fx.log().calls_to(RESERVE).len(), 1);
            // Both mapped: the fast paths take no mapping.
            let mark = t.fx.clear_log();
            t.base.rebuild_task(Hold::NONE).unwrap();
            assert!(!t.fx.sels_since(mark).contains(&"new_virtual_mapping"));
        }

        /// A Metal raise in the reservation escapes.
        #[test]
        fn a_reservation_raise_escapes() {
            let t = tx_with(&texture_list(), Placement::Mapped, true);
            t.fx.set_placement(0, 0x10, 1);
            features(&t.fx, true);
            t.fx.platform
                .script()
                .push(RESERVE, Step::raise("NSInvalidArgumentException", "x"));
            let hold = Hold::NONE;
            assert!(t.base.rebuild_task(hold).is_err());
        }

        /// The features of a task that is gone cannot be read: the process ends there.
        #[test]
        fn a_gone_task_dies_at_the_features_read() {
            let Tx { fx, base } = tx();
            let Fx { task, task_dyn, .. } = fx;
            drop((task, task_dyn));
            assert_eq!(
                catch_death(|| base.rebuild_task(Hold::NONE)).err(),
                Some(DeathKind::TaskGone.site())
            );
        }
    }

    mod data_from_virtual_offset {
        use super::*;

        #[test]
        fn copies_into_a_fresh_buffer() {
            let t = tx();
            let mut asked = None;
            let d = t
                .base
                .data_from_virtual_offset(OnFault::Die, 0x40, 4, |offset, dst| {
                    asked = Some((offset, dst.len()));
                    dst.copy_from_slice(&[1, 2, 3, 4]);
                    Ok(())
                })
                .unwrap();
            assert_eq!(asked, Some((0x40, 4)));
            assert_eq!(d, Some(vec![1, 2, 3, 4]));
            let d = t
                .base
                .data_from_virtual_offset(OnFault::Die, 0, 0, |_, dst| {
                    assert!(dst.is_empty());
                    Ok(())
                })
                .unwrap();
            assert_eq!(d, Some(vec![]));
        }

        /// A raise in the copy escapes.
        #[test]
        fn a_copy_raise_escapes() {
            let t = tx();
            let r = t
                .base
                .data_from_virtual_offset(OnFault::Return, 0, 8, |_, _| {
                    Err(crate::testkit::capture::raise(
                        OnFault::Return,
                        "test.copy",
                        "boom",
                    ))
                });
            assert_eq!(r.err().unwrap().detail().unwrap_or_default(), "boom");
        }

        /// A zero-length copy that raises: the raise escapes.
        #[test]
        fn a_zero_length_copy_raise_escapes() {
            let t = tx();
            let e = t
                .base
                .data_from_virtual_offset(OnFault::Return, 8, 0, |offset, dst| {
                    assert_eq!((offset, dst.len()), (8, 0));
                    Err(crate::testkit::capture::raise(
                        OnFault::Return,
                        "test.copy",
                        "copy of 0x0 bytes at task offset 0x8 reaches the task's end, 0x8",
                    ))
                })
                .unwrap_err();
            assert_eq!(
                e.detail().unwrap_or_default(),
                "copy of 0x0 bytes at task offset 0x8 reaches the task's end, 0x8"
            );
        }

        /// A length past the largest task raises before anything is allocated or copied; the
        /// largest task's length itself is allocated lazily and reaches the copy.
        #[test]
        fn a_length_past_any_task_raises() {
            let t = tx();
            for length in [MAX_LOCAL_TASK_LENGTH + 1, u64::MAX] {
                let e = t
                    .base
                    .data_from_virtual_offset(OnFault::Return, 0, length, |_, _| panic!("no copy"))
                    .err()
                    .unwrap();
                assert_eq!((e.site().file, e.site().line), ("tasks.buffer.length", 0));
            }
            let e = t
                .base
                .data_from_virtual_offset(OnFault::Return, 0, MAX_LOCAL_TASK_LENGTH, |_, dst| {
                    assert_eq!(dst.len() as u64, MAX_LOCAL_TASK_LENGTH);
                    Err(crate::testkit::capture::raise(
                        OnFault::Return,
                        "test.copy",
                        "out of range",
                    ))
                })
                .unwrap_err();
            assert_eq!((e.site().file, e.site().line), ("test.copy", 0));
        }
    }

    mod unsupported {
        use super::*;

        /// Each unsupported op ends the process with its own id, with no side effect.
        #[test]
        fn every_unsupported_op_aborts() {
            let t = tx();
            let mark = t.fx.clear_log();
            for op in [
                Unsupported::GetObject,
                Unsupported::WritableCursor,
                Unsupported::AddressRanges,
            ] {
                let site = catch_death(|| t.base.unsupported(op)).unwrap_err();
                assert_eq!(site, DeathKind::Unsupported(op).site());
                assert_eq!(site.kind, Fault::Abort);
            }
            assert_eq!(
                catch_death(|| t.base.get_object(3)).err(),
                Some(DeathKind::Unsupported(Unsupported::GetObject).site())
            );
            assert!(t.fx.sels_since(mark).is_empty());
        }
    }

    mod edges {
        use super::*;

        /// Features are read before `_objectListManager`.
        #[test]
        fn rebuild_after_teardown_without_argument_buffers_returns() {
            let t = tx();
            features(&t.fx, false);
            t.base.teardown();
            assert_eq!(catch_death(|| t.base.rebuild_task(Hold::NONE)).err(), None);
        }

        /// With argument buffers the torn-down task dies at the object lists.
        #[test]
        fn rebuild_after_teardown_with_argument_buffers_dies() {
            let t = tx();
            features(&t.fx, true);
            t.base.teardown();
            assert_eq!(
                catch_death(|| t.base.rebuild_task(Hold::NONE)).err(),
                Some(TORN_DOWN)
            );
        }

        /// `length >= 12`
        #[test]
        fn serialized_delete_boundary_after_teardown() {
            let t = tx();
            t.base.teardown();
            let cmd = [1000u32, 0, 1].as_bytes().to_vec();
            assert!(
                t.base
                    .delete_object_with_serialized_data(Hold::NONE, &cmd[..11])
                    .is_ok()
            );
            let mut long = cmd.clone();
            long.push(0xee);
            assert_eq!(
                catch_death(|| t.base.delete_object_with_serialized_data(Hold::NONE, &long)).err(),
                Some(TORN_DOWN)
            );
        }

        #[test]
        fn invalidation_duplicates_apply_in_record_order() {
            let t = tx();
            let r = view(&t.fx, false);
            register(&t.fx, 1, &r);
            let recs = [
                ResourceInvalidation {
                    reference: 1,
                    clear_host_valid: 0,
                    set_host_valid: 1,
                    clear_guest_valid: 0,
                    set_guest_valid: 1,
                },
                ResourceInvalidation {
                    reference: 1,
                    clear_host_valid: 1,
                    set_host_valid: 0,
                    clear_guest_valid: 1,
                    set_guest_valid: 0,
                },
            ];
            let bytes = recs.as_bytes().to_vec();
            t.base
                .invalidate_resources(Records::new(&bytes, 2).unwrap())
                .unwrap();
            assert!(!r.is_host_valid());
            assert!(!r.is_guest_valid());
        }

        #[test]
        fn sync_discard_duplicate_reference_deletes_once() {
            let t = tx();
            register(&t.fx, 1, &view(&t.fx, false));
            let (done, _seen) = completion();
            let b = refs(&[1, 1]);
            t.base
                .synchronize_resources(Hold::NONE, records(&b), true, done)
                .unwrap();
            assert!(t.base.get_existing_host_resource(1).is_none());
        }
    }
}
