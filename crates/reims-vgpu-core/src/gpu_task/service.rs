//! The service: [`GpuTask`], its message dispatch, the list, resource and FIFO messages, and
//! what it provides to its resources ([`ResourceHost`]).
//!
//! The handshake builds the task's parts once every value they need is known, so until it has
//! a lock of its own stands in for the task lock: every job that takes the task lock takes it
//! first ([`GpuTask::task_lock`]), and the handshake holds it throughout.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::{Mutex, MutexGuard};

use super::exec::FifoInfo;
use super::surfaces::CachedSurface;
use super::{DeathKind, Unsupported, blit, exception, unraised};
use crate::collections::LookupMap;
use crate::deserializer::Deserializer;
use crate::fatal;
use crate::fifo::wire::Records;
use crate::guest::GuestPtr;
use crate::guest::records::{FeaturesRef, SharedFeatures};
use crate::memory::cursor::{Cursor, WritableCursor};
use crate::memory::map::MemoryMap;
use crate::memory::mapping::Mapping;
use crate::memory::page_table::PageTableWalker;
use crate::platform::device::GpuDevice;
use crate::platform::types::NsErrorText;
use crate::platform::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::host::{CallBorrow, MappedAddress, ResourceHost};
use crate::resources::manager::ResourceManager;
use crate::resources::resource::{Resource, with_platform};
use crate::rt::once::ProcessOnce;
use crate::rt::queue::SerialQueue;
use crate::rt::{Held, Hold, Pool};
use crate::sync::{ReadGuard, WriteGuard};
use crate::tasks::isolation::TaskLife;
use crate::tasks::launcher::{HostLink, ServiceSide};
use crate::tasks::object_lists::ObjectLists;
use crate::tasks::shared::TaskCore;
use crate::tasks::wire::{Reply, ToHost, ToService};
use crate::tasks::{Completion, ExecStatus};

/// The service of one isolated guest task: the service's end of the host's remote task.
///
/// Its handshake sets it up in stages (the map, the length, the task root, the object lists,
/// then the parts), and a later message sees whatever stages a failed handshake reached.
pub struct GpuTask<P: Platform> {
    /// The host's end of the connection. `None` only in the drop.
    pub(super) host: Option<Arc<dyn HostLink<P>>>,
    /// The task's Metal device: `None` before the handshake, and after one that found none.
    pub(super) device: Mutex<Option<P::Device>>,
    /// The exec's command queue.
    pub(super) command_queue: Mutex<Option<P::CommandQueue>>,
    /// The shared task core, once the handshake built it.
    pub(super) core: OnceLock<TaskCore<P>>,
    /// The task root's mapping.
    pub(super) root_mapping: Mutex<Option<Arc<Mapping<P>>>>,
    /// The task root, in the host.
    pub(super) task_root: Mutex<Option<GuestPtr>>,
    /// The guest's page table (shared with the walks in progress).
    pub(super) page_table: Mutex<Option<Arc<PageTableWalker>>>,
    /// The guest's memory map, as the handshake decoded it.
    pub(super) memory_map: Mutex<Option<Arc<MemoryMap<P>>>>,
    /// The platform.
    pub(super) platform: Arc<P>,
    /// The task's life: its jobs and completions run as its code.
    pub(super) life: Arc<TaskLife>,
    /// The XR10 blit pipelines, made once per task.
    pub(super) blit: ProcessOnce<blit::BlitPipelines<P>>,
    /// Whether the task rebuilt its resource indices (on its first mapped object lists).
    pub(super) restored: AtomicBool,
    /// The task's id, as the handshake gave it.
    pub(super) task_id: AtomicU32,
    /// The service queue: one job at a time, in arrival order.
    pub(super) queue: SerialQueue,
    /// The task's features, as the handshake gave them.
    pub(super) features: Arc<SharedFeatures>,
    /// The IOSurfaces the host lent, by mapping id, with their retain counts (lookup only).
    pub(super) surfaces: Mutex<LookupMap<u64, CachedSurface<P>>>,
    /// Serializes the execs' and infos' paging-in.
    pub(super) prepare_lock: Mutex<()>,
    /// Serializes the object lists' mapping.
    pub(super) object_lists_lock: Mutex<()>,
    /// The FIFOs' events and signalled values, by stamp index (lookup only).
    pub(super) fifos: Mutex<LookupMap<u32, FifoInfo<P>>>,
    /// The task itself, which every job holds.
    pub(super) me: Weak<GpuTask<P>>,
    /// The task's length.
    pub(super) length: AtomicU64,
    /// The task lock until the handshake has built the core (module doc).
    pub(super) handshake_lock: Mutex<()>,
    /// The object lists, built by the handshake whether or not it found a device. The core's
    /// parts hold the same lists; with no core they are the only reference.
    pub(super) object_lists: Mutex<Option<Arc<ObjectLists<P>>>>,
}

/// The task lock as a job holds it.
#[expect(dead_code, reason = "the guards are held for their drop")]
pub(super) enum TaskGuard<'a> {
    /// Unique.
    Unique(WriteGuard<'a>),
    /// Shared.
    Shared(ReadGuard<'a>),
    /// The handshake's lock, before the core exists (module doc).
    Early(MutexGuard<'a, ()>),
}

/// How a job takes the task lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    /// Unique: it changes the task's lists or FIFOs.
    Unique,
    /// Shared: an exec, an info or a rate-map reset.
    Shared,
}

/// `Records` over a message's record bytes. The host always sends `count` records; fewer bytes
/// than that is a death, not a short read.
pub(super) fn records<T>(bytes: &[u8], count: u32) -> Records<'_, T>
where
    T: zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable + zerocopy::KnownLayout,
{
    match Records::new(bytes, count) {
        Some(r) => r,
        None => fatal!(DeathKind::ShortRecords),
    }
}

impl<P: Platform> GpuTask<P> {
    /// The service for one connection, whose host end is `host`, living `life`: its jobs and
    /// completions run as the task's code, so a death in them ends the task. The connection's
    /// own invalidation and interruption are the transport's.
    pub fn new(platform: Arc<P>, host: Arc<dyn HostLink<P>>, life: Arc<TaskLife>) -> Arc<Self> {
        let queue = SerialQueue::new(platform.threads(), "reims-vgpu.task-service");
        Arc::new_cyclic(|me| GpuTask {
            host: Some(host),
            device: Mutex::new(None),
            command_queue: Mutex::new(None),
            core: OnceLock::new(),
            root_mapping: Mutex::new(None),
            task_root: Mutex::new(None),
            page_table: Mutex::new(None),
            memory_map: Mutex::new(None),
            platform,
            life,
            blit: ProcessOnce::new(),
            restored: AtomicBool::new(false),
            task_id: AtomicU32::new(0),
            queue,
            features: Arc::new(SharedFeatures::new()),
            surfaces: Mutex::new(LookupMap::new()),
            prepare_lock: Mutex::new(()),
            object_lists_lock: Mutex::new(()),
            fifos: Mutex::new(LookupMap::new()),
            me: me.clone(),
            length: AtomicU64::new(0),
            handshake_lock: Mutex::new(()),
            object_lists: Mutex::new(None),
        })
    }

    /// The task's length.
    pub(super) fn task_length(&self) -> u64 {
        self.length.load(Ordering::SeqCst)
    }

    /// The task root, once the handshake mapped it.
    pub(super) fn task_root_ptr(&self) -> Option<GuestPtr> {
        *self.task_root.lock()
    }

    /// The guest's memory map.
    pub(super) fn map(&self) -> Option<Arc<MemoryMap<P>>> {
        self.memory_map.lock().clone()
    }

    /// The guest's page table.
    pub(super) fn walker(&self) -> Option<Arc<PageTableWalker>> {
        self.page_table.lock().clone()
    }

    /// The Metal device, another reference.
    pub(super) fn device_handle(&self) -> Option<P::Device> {
        self.device.lock().as_ref().map(Handle::clone_ref)
    }

    /// The task core; a message that needs it before the handshake built it dies here.
    pub(super) fn core(&self) -> &TaskCore<P> {
        match self.core.get() {
            Some(core) => core,
            None => fatal!(DeathKind::NotSetUp),
        }
    }

    /// The deserializer, the resource manager and the object lists, or death as
    /// [`GpuTask::core`].
    #[expect(clippy::type_complexity, reason = "the three parts")]
    pub(super) fn parts(
        &self,
    ) -> (
        Arc<Deserializer<P>>,
        Arc<ResourceManager<P>>,
        Arc<ObjectLists<P>>,
    ) {
        match self.core.get().and_then(TaskCore::parts) {
            Some(parts) => parts,
            None => fatal!(DeathKind::NotSetUp),
        }
    }

    /// The object lists, or death before a handshake built them.
    pub(super) fn lists(&self) -> Arc<ObjectLists<P>> {
        match self.object_lists.lock().clone() {
            Some(lists) => lists,
            None => fatal!(DeathKind::NotSetUp),
        }
    }

    /// Takes the task lock. The handshake's lock comes first, so a job queued during the
    /// handshake waits for it; once the core exists, its lock is the task lock.
    pub(super) fn task_lock(&self, access: Access) -> TaskGuard<'_> {
        let early = self.handshake_lock.lock();
        match self.core.get() {
            Some(core) => {
                drop(early);
                match access {
                    Access::Unique => TaskGuard::Unique(core.mutex.write()),
                    Access::Shared => TaskGuard::Shared(core.mutex.read()),
                }
            }
            None => TaskGuard::Early(early),
        }
    }

    /// Sends `m` to the host.
    pub(super) fn send_host(&self, m: ToHost<P>) {
        if let Some(host) = &self.host {
            host.send(m);
        }
    }

    /// Completes the host's async token `token_id` with `success`.
    pub(super) fn complete_token(&self, token_id: u64, success: u32, error: Option<NsErrorText>) {
        self.send_host(ToHost::CompleteAsyncToken {
            token_id,
            success,
            error,
        });
    }

    /// Queues `f` as a job: it holds the task, runs as the task's code (not at all once the
    /// task has ended) with no raise handler live, and makes resources with the task's platform.
    /// It is a unit of work with no handler, so a raise that none of its catches takes ends the
    /// task at the raise.
    ///
    /// The job runs inside a real pool, for Metal's `+0` results, and with a holder of its own,
    /// which `f` gets a hold of ([`Hold::of`]): what a caught raise carries
    /// ([`crate::raise::Raise::defer`]) lands there when the catch ends it ([`Hold::land`]), after
    /// the catch's reply, and the holder releases it, most recent first, as the job ends, before
    /// the pool pops. A death in the job releases none of it and leaves the pool pushed.
    pub(super) fn job(&self, f: impl for<'u> FnOnce(&Arc<GpuTask<P>>, Hold<'u>) + Send + 'static) {
        let Some(this) = self.me.upgrade() else {
            return;
        };
        self.queue.exec_async(move || {
            let life = Arc::clone(&this.life);
            let _ = life.run(|| {
                let platform = Arc::clone(&this.platform);
                let pool = Pool::push(&*platform);
                let held = Held::new();
                with_platform(&platform, || f(&this, Hold::of(&held)));
                drop(held);
                drop(pool);
            });
        });
    }

    /// `f` as a Metal completion handler of the task's: it runs as the task's code, and not at
    /// all once the task has ended.
    pub(super) fn carry<A: 'static>(
        &self,
        f: impl FnOnce(A) + Send + 'static,
    ) -> Box<dyn FnOnce(A) + Send> {
        Box::new(self.life.carry(f))
    }

    /// The task's id, as its lines name it.
    pub(super) fn id(&self) -> u32 {
        self.task_id.load(Ordering::Relaxed)
    }

    // ---- the list, resource and FIFO messages

    /// Flushes the resources; a raise ends the task. Replies 1.
    fn flush_resources(&self, reply: Reply<u32>) {
        self.job(move |this, hold| {
            unraised(this.core().flush_resources(hold));
            reply.call(1);
        });
    }

    /// Deletes the object a serialized delete command names: 1, even for a command that
    /// names nothing; 2 when it raises.
    fn delete_object(&self, data: Vec<u8>, reply: Reply<u32>) {
        self.job(move |this, hold| {
            let mut result = 1;
            if let Err(t) = this.core().delete_object_with_serialized_data(hold, &data) {
                log::warn!(
                    "task {}: object delete faulted: {}",
                    this.id(),
                    exception(&t)
                );
                result = 2;
                hold.land(t);
            }
            reply.call(result);
        });
    }

    /// Replaces the physical page of the task's buffers: 1, or 2 when it raises.
    fn replace_physical(&self, physical: u32, reply: Reply<u32>) {
        self.job(move |this, hold| {
            if let Err(t) = this.core().replace_physical(physical) {
                log::warn!(
                    "task {}: discard of resource {physical} faulted: {}",
                    this.id(),
                    exception(&t)
                );
                reply.call(2);
                hold.land(t);
                return;
            }
            reply.call(1);
        });
    }

    /// Sets the object lists under the unique lock, writing the task root as it goes: 1 when
    /// the lists are taken, 2 when refused.
    ///
    /// With no device the handshake still built the lists, so this runs on them as with one.
    pub(super) fn set_object_list(
        &self,
        object_list_offset: u64,
        object_list_length: u32,
        placement_list_offset: u64,
        placement_list_length: u32,
        reply: Reply<u32>,
    ) {
        self.job(move |this, hold| {
            let _lock = this.task_lock(Access::Unique);
            // The task root is read now; a NULL one faults in the lists
            let lists = this.lists();
            let root = this.task_root_ptr().unwrap_or_default();
            let taken = lists.set_object_list(
                hold,
                object_list_offset,
                object_list_length,
                placement_list_offset,
                placement_list_length,
                root,
            );
            reply.call(if taken { 1 } else { 2 });
        });
    }

    /// Invalidates the resources the records name: 1, or 2 when it raises.
    fn invalidate_resources(&self, resources: Vec<u8>, count: u32, reply: Reply<u32>) {
        self.job(move |this, hold| {
            let invalidated = {
                let records = records(&resources, count);
                this.core().invalidate_resources(records)
            };
            if let Err(t) = invalidated {
                log::warn!(
                    "task {}: resource invalidate faulted: {}",
                    this.id(),
                    exception(&t)
                );
                reply.call(2);
                hold.land(t);
                return;
            }
            reply.call(1);
        });
    }

    /// Synchronizes the resources the records name into guest memory (then discards them with
    /// `discard`). The reply normally comes from the paging buffer's completion, on a Metal
    /// thread; a raise replies 2 here, and a raise after the commit still gets the
    /// completion's reply too.
    fn synchronize_resources(
        &self,
        resources: Vec<u8>,
        count: u32,
        discard: bool,
        reply: Reply<u32>,
    ) {
        self.job(move |this, hold| {
            let id = this.id();
            let completion_reply = reply.clone();
            let completion: Completion =
                Arc::new(move |status: ExecStatus, error: Option<NsErrorText>| {
                    if status.raw() != 1 {
                        log::warn!(
                            "task {id}: resource synchronize finished with status {}: {}",
                            status.raw(),
                            super::error_text(error.as_ref())
                        );
                    }
                    completion_reply.call(status.raw());
                });
            let synchronized = {
                let records = records(&resources, count);
                this.core()
                    .synchronize_resources(hold, records, discard, completion)
            };
            if let Err(t) = synchronized {
                log::warn!(
                    "task {}: resource synchronize faulted: {}",
                    this.id(),
                    exception(&t)
                );
                reply.call(2);
                hold.land(t);
            }
        });
    }

    /// Discards the resources the records name: 1, or 2 when it raises.
    fn discard_resources(&self, resources: Vec<u8>, count: u32, reply: Reply<u32>) {
        self.job(move |this, hold| {
            let discarded = {
                let records = records(&resources, count);
                this.core().discard_resources(hold, records)
            };
            let deleted = match discarded {
                Ok(deleted) => deleted,
                Err(t) => {
                    log::warn!(
                        "task {}: resource discard faulted: {}",
                        this.id(),
                        exception(&t)
                    );
                    reply.call(2);
                    hold.land(t);
                    return;
                }
            };
            reply.call(1);
            drop(deleted);
        });
    }

    /// Unmaps whichever object list overlaps `offset..offset + length`, under the unique lock.
    ///
    /// Always replies 1 (testing.md 7.1): no range check; the ends wrap.
    fn unmap_memory(&self, offset: u64, length: u64, reply: Reply<u32>) {
        self.job(move |this, _| {
            let _lock = this.task_lock(Access::Unique);
            let lists = this.lists();
            let end = offset.wrapping_add(length);
            let objects = lists.object_list_range();
            let placements = lists.placement_list_range();
            let objects_end = objects.offset.wrapping_add(u64::from(objects.length));
            let placements_end = placements.offset.wrapping_add(u64::from(placements.length));
            if objects.offset < end && offset < objects_end {
                lists.unmap_object_list();
            }
            if offset < placements_end && placements.offset < end {
                lists.unmap_placement_list();
            }
            reply.call(1);
        });
    }

    /// Deletes `resource`: 1, or 2 when it raises.
    fn delete_resource(&self, resource: u32, reply: Reply<u32>) {
        self.job(move |this, hold| {
            if let Err(t) = this.core().delete_resource(hold, resource) {
                log::warn!(
                    "task {}: delete of resource {resource} faulted: {}",
                    this.id(),
                    exception(&t)
                );
                reply.call(2);
                hold.land(t);
                return;
            }
            reply.call(1);
        });
    }

    /// Sets FIFO `index`'s event, under the unique lock. The old entry is released before the
    /// new event is made; the device's event is taken unchecked (nil stays nil), and the
    /// signalled value starts at `u64::MAX`.
    fn set_fifo_event(&self, event: Option<P::SharedEventHandle>, index: u32) {
        self.job(move |this, _| {
            let _lock = this.task_lock(Access::Unique);
            let old = this.fifos.lock().remove(&index);
            drop(old);
            let device = this.device_handle();
            let shared_event = match &device {
                Some(d) => d.open_shared_event(event.as_ref()),
                None => None,
            };
            {
                let mut fifos = this.fifos.lock();
                if !fifos.contains_key(&index) {
                    fifos.insert(index, FifoInfo::new(shared_event));
                }
            }
            drop(device);
        });
    }

    /// Forgets FIFO `index`'s event, under the unique lock. Replies 1.
    fn fifo_deleted(&self, stamp_index: u32, reply: Reply<u32>) {
        self.job(move |this, _| {
            let _lock = this.task_lock(Access::Unique);
            let old = this.fifos.lock().remove(&stamp_index);
            drop(old);
            reply.call(1);
        });
    }
}

impl<P: Platform> ServiceSide<P> for GpuTask<P> {
    /// One message, on the connection's thread, in arrival order. The handshake runs here, in
    /// the connection queue's block, with no handler and no holder ([`Hold::NONE`]); every other
    /// message queues its job.
    fn on_message(&self, m: ToService<P>) {
        match m {
            ToService::Handshake {
                task_root,
                task_size,
                task_id,
                configuration,
                features,
                memory_map,
                reply,
            } => {
                let r = self.handshake(
                    task_root,
                    task_size,
                    task_id,
                    configuration,
                    features,
                    memory_map,
                    &reply,
                );
                // No handler is live: a raise in the handshake died at its own site
                unraised(r);
            }
            ToService::FlushResources { reply } => self.flush_resources(reply),
            ToService::UnmapMemory {
                offset,
                length,
                reply,
            } => self.unmap_memory(offset, length, reply),
            ToService::SetObjectList {
                object_list_offset,
                object_list_length,
                placement_list_offset,
                placement_list_length,
                reply,
            } => self.set_object_list(
                object_list_offset,
                object_list_length,
                placement_list_offset,
                placement_list_length,
                reply,
            ),
            ToService::GetComputeInfo {
                pipeline_state_reference,
                max_key,
                count,
                offset,
                reply,
            } => self.get_compute_info(pipeline_state_reference, max_key, count, offset, reply),
            ToService::ReplacePhysical { physical, reply } => {
                self.replace_physical(physical, reply)
            }
            ToService::InvalidateResources {
                resources,
                count,
                reply,
            } => self.invalidate_resources(resources, count, reply),
            ToService::SynchronizeResources {
                resources,
                count,
                discard,
                reply,
            } => self.synchronize_resources(resources, count, discard, reply),
            ToService::DiscardResources {
                resources,
                count,
                reply,
            } => self.discard_resources(resources, count, reply),
            ToService::HeapTextureSizeAndAlign {
                data,
                reply_virtual_offset,
                reply_length,
                reply,
            } => self.heap_texture_size_and_align(data, reply_virtual_offset, reply_length, reply),
            ToService::DeleteObject { data, reply } => self.delete_object(data, reply),
            ToService::DeleteResource { resource, reply } => self.delete_resource(resource, reply),
            ToService::ExecIndirect3 {
                data,
                cmd_buffer_count,
                resource_count,
                stamp_index,
                fifo_event_value_in,
                gpu_completion,
                reply,
            } => self.exec(
                data,
                cmd_buffer_count,
                resource_count,
                stamp_index,
                fifo_event_value_in,
                gpu_completion,
                reply,
            ),
            ToService::InfoIndirect {
                data,
                cmd_buffer_count,
                resource_count,
                stamp_index,
                fifo_event_value_in,
                reply,
            } => self.info(
                data,
                cmd_buffer_count,
                resource_count,
                stamp_index,
                fifo_event_value_in,
                reply,
            ),
            ToService::ResetRasterizationRateMap { map, reply } => {
                self.reset_rasterization_rate_map(map, reply)
            }
            ToService::SetFifoEvent { event, index } => self.set_fifo_event(event, index),
            ToService::FifoDeleted { stamp_index, reply } => self.fifo_deleted(stamp_index, reply),
        }
    }

    /// The task's end: what the drop releases goes now, in its order, whoever still holds the
    /// task (a job or a completion in flight, which then runs nothing). The resource manager and
    /// the object lists go with the core's parts; then the surfaces the host lent, the FIFOs'
    /// events and the lists the handshake built.
    fn end(&self) {
        drop(self.device.lock().take());
        drop(self.command_queue.lock().take());
        if let Some(core) = self.core.get() {
            core.teardown();
        }
        drop(self.root_mapping.lock().take());
        *self.task_root.lock() = None;
        drop(self.page_table.lock().take());
        drop(self.memory_map.lock().take());
        let surfaces = std::mem::take(&mut *self.surfaces.lock());
        drop(surfaces);
        let fifos = std::mem::take(&mut *self.fifos.lock());
        drop(fifos);
        drop(self.object_lists.lock().take());
    }
}

impl<P: Platform> Drop for GpuTask<P> {
    /// The host's end, the device and the command queue go first, then the core (its queues,
    /// its event and its parts), then the task root's mapping, the page table and the memory
    /// map; the rest follows in field order. A task that ended released most of it then.
    fn drop(&mut self) {
        drop(self.host.take());
        drop(self.device.get_mut().take());
        drop(self.command_queue.get_mut().take());
        drop(self.core.take());
        drop(self.root_mapping.get_mut().take());
        *self.task_root.get_mut() = None;
        drop(self.page_table.get_mut().take());
        drop(self.memory_map.get_mut().take());
    }
}

impl<P: Platform> ResourceHost<P> for GpuTask<P> {
    fn new_buffer_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<P::Buffer> {
        GpuTask::new_buffer_for_virtual_page(self, hold, page, length)
    }

    fn prepare_backing(
        &self,
        _hold: Hold<'_>,
        _backing: &Resource<P>,
        _encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        fatal!(DeathKind::Unsupported(Unsupported::PrepareBacking))
    }

    fn synchronize_backing(
        &self,
        _hold: Hold<'_>,
        _backing: &Resource<P>,
        _encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        fatal!(DeathKind::Unsupported(Unsupported::SynchronizeBacking))
    }

    fn retain_io_surface_with_mapping_id(
        &self,
        identifier: u64,
    ) -> RaiseResult<Option<P::Surface>> {
        Ok(self.retain_surface(identifier))
    }

    fn release_io_surface_with_mapping_id(&self, identifier: u64, surface: Option<P::Surface>) {
        self.release_surface(identifier, surface)
    }

    fn address_ranges_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
        read_only: &mut bool,
    ) -> RaiseResult<Option<P::AddressRangeArray>> {
        GpuTask::address_ranges_for_virtual_page(self, hold, page, length, read_only)
    }

    fn new_shared_texture_handle_for_id(
        &self,
        _hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<P::SharedTextureHandle>> {
        Ok(self.new_shared_texture_handle(identifier))
    }

    fn data_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<u8>>> {
        // The resource manager's reads, always under a handler.
        GpuTask::data_from_virtual_offset(self, hold, OnFault::Return, offset, length)
    }

    /// A reference texture ends the task.
    fn backing_for_id(
        &self,
        _hold: Hold<'_>,
        _identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        fatal!(DeathKind::Unsupported(Unsupported::BackingForId))
    }

    fn cursor_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Cursor>> {
        self.cursor(hold, offset, length, false)
    }

    fn writable_cursor_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<WritableCursor>> {
        Ok(self.cursor(hold, offset, length, true)?.map(WritableCursor))
    }

    fn deserializer(&self) -> Option<CallBorrow<'_, Deserializer<P>>> {
        let (deserializer, _, _) = self.core.get()?.parts()?;
        Some(CallBorrow::new(deserializer))
    }

    /// `None` before the handshake, and after one that found no device.
    fn mtl_device(&self) -> Option<P::Device> {
        self.device_handle()
    }

    fn features(&self) -> FeaturesRef {
        Arc::clone(&self.features)
    }

    fn resource_manager(&self) -> Option<CallBorrow<'_, ResourceManager<P>>> {
        let (_, resource_manager, _) = self.core.get()?.parts()?;
        Some(CallBorrow::new(resource_manager))
    }

    fn blit_in_rgb_2p_xr10_a8_pipeline(&self) -> Option<P::ComputePipeline> {
        Some(self.blit_pipelines().blit_in.clone_ref())
    }

    fn blit_out_rgb_2p_xr10_a8_pipeline(&self) -> Option<P::ComputePipeline> {
        Some(self.blit_pipelines().blit_out.clone_ref())
    }

    fn supports_buffer_with_address_ranges(&self) -> bool {
        self.core
            .get()
            .is_some_and(|c| c.supports_buffer_with_address_ranges)
    }

    fn supports_heap_with_address_ranges(&self) -> bool {
        self.core
            .get()
            .is_some_and(|c| c.supports_heap_with_address_ranges)
    }

    fn supports_resource_detach_backing(&self) -> bool {
        self.core
            .get()
            .is_some_and(|c| c.supports_resource_detach_backing)
    }

    fn paging_queue(&self) -> Option<P::CommandQueue> {
        self.core
            .get()?
            .paging_queue
            .as_ref()
            .map(Handle::clone_ref)
    }

    fn new_virtual_mapping(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<Option<Arc<Mapping<P>>>> {
        GpuTask::new_virtual_mapping(self, hold, offset, length, need_writable)
    }

    fn mapped_address_for_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<MappedAddress<P>> {
        GpuTask::mapped_address_for_offset(self, hold, offset, length, need_writable)
    }

    fn resolve_virtual_pages(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<crate::guest::GuestPtr>>> {
        self.resolve_pages(hold, offset, length)
    }

    fn memory_map(&self) -> Option<Arc<crate::memory::map::MemoryMap<P>>> {
        self.map()
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::raise::death::DeathId;
    use crate::testkit::capture::catch_death;

    /// Notes its name when released.
    struct Rel(&'static str, Arc<Mutex<Vec<&'static str>>>);
    impl Drop for Rel {
        fn drop(&mut self) {
            self.1.lock().push(self.0);
        }
    }

    /// A job's own holder takes a deep hold made in the job and what a raise the job caught
    /// carries: both go at the job's end, most recent first, after the job's work (its reply).
    /// A job that dies releases none of what it held
    #[test]
    fn a_jobs_holder_releases_its_values_at_the_jobs_end() {
        let tx = ready(&rw_pages(4));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        tx.task.job(move |_, hold| {
            hold.keep(Rel("deep", s.clone()));
            let r: RaiseResult<()> = Err(crate::raise::Raise::new(
                super::super::RaiseKind::ReplyLength,
            )
            .defer(Rel("carried", s.clone())));
            s.lock().push("reply");
            if let Err(r) = r {
                hold.land(r);
            }
            s.lock().push("work done");
        });
        tx.settle();
        assert_eq!(*seen.lock(), ["reply", "work done", "carried", "deep"]);
        let s = seen.clone();
        tx.task.job(move |_, hold| {
            hold.keep(Rel("dead", s.clone()));
            crate::fatal!(DeathKind::NotSetUp)
        });
        tx.settle();
        assert_eq!(
            tx.life.death().map(DeathId::site),
            Some(DeathKind::NotSetUp.site())
        );
        assert_eq!(seen.lock().len(), 4, "nothing more released");
    }

    /// A message the handshake never prepared for ends the task at the op, without a reply,
    /// and later jobs do not run
    #[test]
    fn unmap_before_the_handshake_dies() {
        let tx = fixture(1, &rw_pages(4));
        let (got, reply) = sink();
        tx.send(ToService::UnmapMemory {
            offset: 0,
            length: 1,
            reply,
        });
        assert_eq!(
            tx.life.death().map(DeathId::site),
            Some(DeathKind::NotSetUp.site())
        );
        let (later, reply) = sink();
        tx.send(ToService::FlushResources { reply });
        assert!(got.lock().is_empty() && later.lock().is_empty());
    }

    /// Unmap always replies 1, even for a wrapping range
    #[test]
    fn unmap_always_succeeds() {
        let tx = ready(&rw_pages(4));
        let (got, reply) = sink();
        tx.send(ToService::UnmapMemory {
            offset: u64::MAX,
            length: 2,
            reply,
        });
        assert_eq!(*got.lock(), vec![1]);
    }

    /// Unmap releases the object list it overlaps and keeps one it does not, by the bounds the
    /// last set-object-list left: a refused set leaves nothing to overlap
    #[test]
    fn unmap_follows_the_last_set_object_list() {
        let tx = ready(&rw_pages(4));
        let lists = tx.task.lists();
        let set = |off: u64, len: u32| {
            let (got, reply) = sink();
            tx.send(ToService::SetObjectList {
                object_list_offset: off,
                object_list_length: len,
                placement_list_offset: 0,
                placement_list_length: 0,
                reply,
            });
            got.lock().clone()
        };
        let unmap = |offset: u64, length: u64| {
            tx.send(ToService::UnmapMemory {
                offset,
                length,
                reply: Reply::new(|_| {}),
            });
        };
        assert_eq!(set(PAGE, 24), vec![1]);
        assert!(lists.ensure_mapped(Hold::NONE));
        unmap(0, PAGE);
        assert!(lists.is_object_list_mapped());
        unmap(PAGE + 8, 1);
        assert!(!lists.is_object_list_mapped());
        // A refused list zeroes the bounds: only a range over offset 0 overlaps
        assert_eq!(set(TASK_LEN, 24), vec![2]);
        assert_eq!(lists.object_list_range().length, 0);
    }

    /// A list past the task's end is refused: 2
    #[test]
    fn set_object_list_replies() {
        let tx = ready(&rw_pages(4));
        let set = |off: u64, len: u32| {
            let (got, reply) = sink();
            tx.send(ToService::SetObjectList {
                object_list_offset: off,
                object_list_length: len,
                placement_list_offset: 0,
                placement_list_length: 0,
                reply,
            });
            got.lock().clone()
        };
        assert_eq!(set(PAGE, 24), vec![1]);
        assert_eq!(tx.task.lists().object_list_range().offset, PAGE);
        assert_eq!(set(TASK_LEN, 24), vec![2]);
        assert_eq!(tx.task.lists().object_list_range().offset, 0);
    }

    /// Empty requests reply 1
    #[test]
    fn resource_messages_reply_1() {
        let tx = ready(&rw_pages(4));
        let (got, reply) = sink();
        tx.task.on_message(ToService::DiscardResources {
            resources: Vec::new(),
            count: 0,
            reply: reply.clone(),
        });
        tx.task.on_message(ToService::InvalidateResources {
            resources: Vec::new(),
            count: 0,
            reply: reply.clone(),
        });
        tx.task.on_message(ToService::DeleteResource {
            resource: 3,
            reply: reply.clone(),
        });
        tx.task.on_message(ToService::FlushResources { reply });
        tx.settle();
        assert_eq!(*got.lock(), vec![1, 1, 1, 1]);
    }

    /// Records shorter than their count end the task before any is read
    #[test]
    fn short_records_die() {
        let tx = ready(&rw_pages(4));
        tx.send(ToService::DiscardResources {
            resources: vec![0; 2],
            count: 1,
            reply: Reply::new(|_| {}),
        });
        assert_eq!(
            tx.life.death().map(DeathId::site),
            Some(DeathKind::ShortRecords.site())
        );
    }

    /// The task's end releases its Metal objects and its views of guest memory at once, while
    /// something still holds the task
    #[test]
    fn the_end_releases_the_tasks_state() {
        let tx = ready(&rw_pages(4));
        with_fifo(&tx, 2);
        let held = Arc::clone(&tx.task);
        tx.task.end();
        assert!(held.device_handle().is_none());
        assert!(held.command_queue.lock().is_none());
        assert!(held.core.get().and_then(TaskCore::parts).is_none());
        assert!(held.map().is_none() && held.walker().is_none());
        assert!(held.task_root_ptr().is_none());
        assert!(held.fifos.lock().is_empty() && held.object_lists.lock().is_none());
    }

    /// With no device the handshake still builds a resource manager over nil: a flush walks its
    /// empty map and a delete finds nothing, and neither dies
    #[test]
    fn resource_messages_without_a_device_survive() {
        let tx = fixture(1, &rw_pages(4));
        let r = handshake_with(&tx, ROOT_PAGE, TASK_LEN, 0x42, true);
        assert_eq!(r, vec![(2, pid()), (1, pid())]);
        let d = catch_death(|| {
            unraised(tx.task.core().flush_resources(Hold::NONE));
            unraised(tx.task.core().delete_resource(Hold::NONE, 7));
        });
        assert!(d.is_ok(), "died: {:?}", d.err());
        let (got, reply) = sink();
        tx.send(ToService::FlushResources { reply });
        assert_eq!(*got.lock(), vec![1]);
    }

    /// With no device the list messages run on the lists the handshake built: unmap replies 1
    /// and set-object-list replies as the lists decide
    #[test]
    fn list_messages_without_a_device() {
        let tx = fixture(1, &rw_pages(4));
        let r = handshake_with(&tx, ROOT_PAGE, TASK_LEN, 0x42, true);
        assert_eq!(r, vec![(2, pid()), (1, pid())]);
        let (got, reply) = sink();
        tx.send(ToService::UnmapMemory {
            offset: 0,
            length: PAGE,
            reply,
        });
        assert_eq!(*got.lock(), vec![1]);
        let (got, reply) = sink();
        tx.task.on_message(ToService::SetObjectList {
            object_list_offset: 0,
            object_list_length: 12,
            placement_list_offset: 0,
            placement_list_length: 0,
            reply,
        });
        let (refused, reply) = sink();
        tx.send(ToService::SetObjectList {
            object_list_offset: TASK_LEN,
            object_list_length: 12,
            placement_list_offset: 0,
            placement_list_length: 0,
            reply,
        });
        assert_eq!(*got.lock(), vec![1]);
        assert_eq!(*refused.lock(), vec![2]);
    }

    /// A FIFO's event is set with the signalled value at `u64::MAX`, and deleted with 1
    #[test]
    fn fifo_event_set_and_deleted() {
        let tx = ready(&rw_pages(4));
        with_fifo(&tx, 2);
        {
            let fifos = tx.task.fifos.lock();
            let f = fifos.get(&2).unwrap();
            assert!(f.event.is_some());
            assert_eq!(f.signaled.load(Ordering::SeqCst), u64::MAX);
        }
        let (got, reply) = sink();
        tx.send(ToService::FifoDeleted {
            stamp_index: 2,
            reply,
        });
        assert_eq!(*got.lock(), vec![1]);
        assert!(tx.task.fifos.lock().is_empty());
    }

    /// The stubs the service does not provide end the task
    #[test]
    fn unsupported_operations_die() {
        let tx = fixture(1, &rw_pages(4));
        assert_eq!(
            catch_death(|| tx.task.backing_for_id(Hold::NONE, 1)).unwrap_err(),
            DeathKind::Unsupported(Unsupported::BackingForId).site()
        );
    }
}
