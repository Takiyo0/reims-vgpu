//! Exec commands: resource preparation (paging in), the decode into an exec command buffer, its
//! commit and completion, the synchronize buffer (paging out), and the rate-map reset.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zerocopy::IntoBytes;

use super::{DeathKind, LocalTask, kernel_task};
use crate::fifo::wire::Records;
use crate::guest::records::CmdExecIndirectCmdBuffer;
use crate::memory::cursor::Cursor;
use crate::num::Truncate;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::command::{CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo};
use crate::platform::compute::{ComputeCmd, ComputeEncoder};
use crate::platform::event::EventRef;
use crate::platform::types::{CommandBufferStatus, NsErrorText};
use crate::platform::{Handle, Platform};
use crate::raise::OnFault;
use crate::raise::RaiseResult;
use crate::resources::manager::create::Resources;
use crate::resources::manager::states::RasterizationRateMaps;
use crate::resources::resource::Resource;
use crate::rt::{Held, Hold};
use crate::tasks::resource_list::ResourceList;
use crate::tasks::{Completion, ExecStatus, Unsupported};

/// What the preparation hands the exec.
pub(super) struct PrepareSync<P: Platform> {
    /// A paging buffer was committed: the exec buffer waits for it.
    pub(super) needs_sync: bool,
    /// A resource asked for synchronize: a synchronize buffer completes the exec. (name is
    /// ours: set by the synchronize flag)
    pub(super) needs_complete: bool,
    /// The paging event the exec buffer waits for, held by the exec; nil until a paging buffer
    /// is made.
    pub(super) event: Option<P::Event>,
    /// The paging event's value the paging buffer signals. (name is ours: the counter's value)
    pub(super) value: u64,
}

/// What the synchronize buffer waits for.
pub(super) struct CompleteSync<'a, P: Platform> {
    /// Whether any resource asked for synchronize.
    pub(super) needs_complete: bool,
    /// The FIFO's event.
    pub(super) event: Option<&'a P::SharedEvent>,
    /// The FIFO event's value the exec buffer signals.
    pub(super) value: u64,
}

/// A command buffer's error in a line: Metal's, or that Metal gave none.
fn error_text(error: Option<&NsErrorText>) -> &str {
    error.map_or("Metal gave no error", |e| e.description.as_str())
}

impl<P: Platform> LocalTask<P> {
    /// Prepares an exec's resources, under the preparation lock: each is made on first use and
    /// its flags applied in order (a raise abandons the rest); the ones that need paging in go to
    /// one paging buffer (blit, then compute) that waits for `event` at `value` and signals the
    /// paging event, committed even when a prepare raised; CPU paging follows; then the
    /// guest-invalidated resources are marked, and the shared-texture backings among them are
    /// invalidated in the kernel task.
    pub(super) fn prepare_resources(
        &self,
        hold: Hold<'_>,
        resources: &ResourceList<'_>,
        event: Option<&P::SharedEvent>,
        value: u64,
    ) -> RaiseResult<PrepareSync<P>> {
        // DIAG(prepare-steps): each step's time, written at debug where it takes 5 ms or more.
        let diag = log::log_enabled!(log::Level::Debug);
        let t0 = std::time::Instant::now();
        let mut steps: Vec<(&'static str, f64)> = Vec::new();
        let mut mark = |label: &'static str| {
            if diag {
                steps.push((label, t0.elapsed().as_secs_f64() * 1000.0));
            }
        };
        let _prepare = self.prepare_mutex.lock();
        crate::perfstats::PREPARE_MUTEX.add(t0.elapsed());
        mark("prepare_mutex");
        let mut sync = PrepareSync::<P> {
            needs_sync: false,
            needs_complete: false,
            event: None,
            value: 0,
        };
        let mut blit_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut compute_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut cpu_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut invalidated: Vec<(u32, Option<Arc<Resource<P>>>)> = Vec::new();
        let mut shared_texture_backings: Vec<u32> = Vec::new();
        for i in 0..resources.count() {
            let resource = resources.get(i)?;
            // Made on first use; a raise abandons the rest
            let t_get = std::time::Instant::now();
            let host_resource = self.base.get_host_resource(hold, resource.reference)?;
            let took = t_get.elapsed();
            if took >= std::time::Duration::from_micros(20) {
                crate::perfstats::LOOKUP_SLOW.add(took);
            }
            if resource.invalidate_host()
                && let Some(r) = &host_resource
            {
                r.set_is_guest_valid(true);
                r.set_is_host_valid(false);
            }
            if let Some(r) = &host_resource
                && r.needs_prepare()
            {
                if r.needs_cpu_paging() {
                    cpu_resources.push(Arc::clone(r));
                } else if r.needs_compute_paging() {
                    compute_resources.push(Arc::clone(r));
                } else {
                    blit_resources.push(Arc::clone(r));
                }
            }
            if resource.invalidate_guest() {
                invalidated.push((resource.reference, host_resource.clone()));
            }
            if resource.synchronize() {
                sync.needs_complete = true;
            }
        }
        mark("resources");
        let t_lookup = t0.elapsed();
        crate::perfstats::PREPARE_LOOKUP.add(t_lookup);
        crate::perfstats::PREPARE_RESOURCES
            .fetch_add(resources.count(), std::sync::atomic::Ordering::Relaxed);
        // A volatile resource whose host copy is valid only needs making non-volatile, a CPU call:
        // done here, it needs no paging buffer, and an exec with nothing else to page builds none
        // (a paging buffer, its wait and its commit cost ~1 ms an exec, on every video frame).
        let devolatilized = |list: Vec<Arc<Resource<P>>>| -> RaiseResult<Vec<Arc<Resource<P>>>> {
            let mut kept = Vec::with_capacity(list.len());
            for r in list {
                if r.is_host_valid() && r.state.is_volatile() {
                    r.devolatilize_now()?;
                    if !r.needs_prepare() {
                        continue;
                    }
                }
                kept.push(r);
            }
            Ok(kept)
        };
        let blit_resources = devolatilized(blit_resources)?;
        let compute_resources = devolatilized(compute_resources)?;
        let (n_blit, n_compute, n_cpu) =
            (blit_resources.len(), compute_resources.len(), cpu_resources.len());
        if !blit_resources.is_empty() || !compute_resources.is_empty() {
            let t_round = std::time::Instant::now();
            crate::perfstats::PAGING_PREPARES.fetch_add(
                (n_blit + n_compute) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            use crate::perfstats::PAGING_STEP as PS;
            let mut t_step = std::time::Instant::now();
            let mut step = |i: usize| {
                PS[i].add(t_step.elapsed());
                t_step = std::time::Instant::now();
            };
            let command_buffer = match &self.base.paging_queue {
                Some(queue) => queue.command_buffer(OnFault::Return)?,
                None => None,
            };
            mark("paging command_buffer");
            step(0);
            if let Some(event) = event
                && let Some(buffer) = &command_buffer
            {
                buffer.encode_wait(Some(EventRef::Shared(event)), value)?;
            }
            mark("paging encode_wait");
            step(1);
            sync.needs_sync = true;
            let mut blit_encoder: Option<P::BlitEncoder> = None;
            let mut compute_encoder: Option<P::ComputeEncoder> = None;
            let (mut prepared, fills) = crate::memory::map::with_deferred_syncs(|| -> RaiseResult<_> {
                if !blit_resources.is_empty() {
                    blit_encoder = match &command_buffer {
                        Some(buffer) => buffer.blit_encoder(OnFault::Return)?,
                        None => None,
                    };
                    mark("paging blit_encoder");
                    step(2);
                    for r in &blit_resources {
                        let t = std::time::Instant::now();
                        let done = r.prepare_in_encoder(hold, blit_encoder.as_ref());
                        crate::perfstats::PAGING_BLIT[r.kind_slot()].add(t.elapsed());
                        done?;
                    }
                    step(2);
                    mark("paging prepares");
                    if let Some(encoder) = &blit_encoder {
                        encoder.encode(OnFault::Return, BlitCmd::EndEncoding)?;
                    }
                    blit_encoder = None;
                    mark("paging end_encoding");
                    step(3);
                }
                if !compute_resources.is_empty() {
                    compute_encoder = match &command_buffer {
                        Some(buffer) => buffer.compute_encoder(OnFault::Return)?,
                        None => None,
                    };
                    step(4);
                    for r in &compute_resources {
                        let t = std::time::Instant::now();
                        let done = r.prepare_in_compute_encoder(hold, compute_encoder.as_ref());
                        crate::perfstats::PAGING_COMPUTE[r.kind_slot()].add(t.elapsed());
                        done?;
                    }
                    step(4);
                    if let Some(encoder) = &compute_encoder {
                        encoder.encode(OnFault::Return, ComputeCmd::EndEncoding)?;
                    }
                    compute_encoder = None;
                }
                Ok(())
            });
            // The transfer buffers' fills, one thread per texture, before the commit whose
            // copies read them.
            crate::memory::map::run_deferred_syncs(fills);
            mark("paging fills");
            step(5);
            // Whatever the prepares did: an encoder a raise left open is ended, the paging event
            // is signalled at the next counter value, and the buffer is committed; a raise here
            // goes on in place of the prepare's, which ends there
            if let Some(encoder) = &blit_encoder {
                hold.cleanup_step(
                    &mut prepared,
                    encoder.encode(OnFault::Return, BlitCmd::EndEncoding),
                )?;
            }
            if let Some(encoder) = &compute_encoder {
                hold.cleanup_step(
                    &mut prepared,
                    encoder.encode(OnFault::Return, ComputeCmd::EndEncoding),
                )?;
            }
            sync.event = self.base.paging_event.as_ref().map(Handle::clone_ref);
            sync.value = self
                .base
                .paging_counter
                .fetch_add(1, Ordering::SeqCst)
                .wrapping_add(1);
            if let Some(buffer) = &command_buffer {
                let signalled = buffer.encode_signal(
                    self.base.paging_event.as_ref().map(EventRef::Event),
                    sync.value,
                );
                hold.cleanup_step(&mut prepared, signalled)?;
                mark("paging encode_signal");
                hold.cleanup_step(&mut prepared, buffer.commit(OnFault::Return))?;
                mark("paging commit");
            }
            step(6);
            crate::perfstats::PAGING_ROUNDS.add(t_round.elapsed());
            // Every caller is under a handler (an exec's or an info's): the raise goes on
            prepared?;
        }
        mark("gpu_paging");
        let t_paging = t0.elapsed();
        crate::perfstats::PREPARE_PAGING.add(t_paging - t_lookup);
        (|| -> RaiseResult<_> {
            for r in &cpu_resources {
                r.prepare_with_cpu(hold)?;
            }
            Ok(())
        })()?;
        mark("cpu_paging");
        for (identifier, host_resource) in &invalidated {
            let Some(r) = host_resource else {
                // Nothing to mark, and no host copy either
                log::warn!(
                    "task {}: guest copy of reference {identifier} invalidated, but it names no \
                     resource",
                    self.task_id
                );
                continue;
            };
            r.set_is_guest_valid(false);
            let backing = r.get_shared_texture_backing_id()?;
            if backing != 0 {
                shared_texture_backings.push(backing);
            }
            if !r.is_host_valid() {
                log::warn!(
                    "task {}: guest copy of resource {identifier} invalidated while its host copy \
                     is not valid either",
                    self.task_id
                );
            }
        }
        let count = shared_texture_backings.len().low_u32();
        if count != 0
            && let Some(records) = Records::<u32>::new(shared_texture_backings.as_bytes(), count)
        {
            // A user task's go to the kernel task, under its lock
            self.invalidate_guest(hold, records)?;
        }
        mark("invalidate_guest");
        crate::perfstats::PREPARE_TAIL.add(t0.elapsed() - t_paging);
        crate::perfstats::PREPARE.add(t0.elapsed());
        if diag && t0.elapsed().as_secs_f64() >= 0.005 {
            let mut line = String::new();
            let mut last = 0.0;
            for (label, at) in &steps {
                line.push_str(&format!(" {label}={:.1}", at - last));
                last = *at;
            }
            log::debug!(
                "task {}: prepare steps (ms):{line}; blit {n_blit} compute {n_compute} cpu {n_cpu} of {}",
                self.task_id,
                resources.count()
            );
        }
        crate::fatal_unless!(
            !sync.needs_sync || sync.event.is_some(),
            DeathKind::NoPagingEvent
        );
        Ok(sync)
    }

    /// Marks the shared-texture backings `backings` names guest-invalid, under the task's unique
    /// lock, logging each one not found. A user task sends them to the kernel task.
    pub(super) fn invalidate_guest(
        &self,
        hold: Hold<'_>,
        backings: Records<'_, u32>,
    ) -> RaiseResult<()> {
        if backings.count() == 0 {
            return Ok(());
        }
        if self.task_id != 0 {
            if let Some(d) = self.device.upgrade() {
                let kernel = kernel_task(hold, &d)?;
                kernel.on_platform(|| kernel.invalidate_guest(hold, backings))?;
            }
            return Ok(());
        }
        let _lock = self.base.mutex.write();
        let (_, resource_manager, _) = self.parts();
        for identifier in backings.iter() {
            match resource_manager.get_existing_object::<Resources>(identifier) {
                // Id 0 is found as nothing, and nothing is marked
                Some(resource) => {
                    if let Some(r) = resource {
                        r.set_is_guest_valid(false);
                    }
                }
                None => log::warn!("shared texture backing {identifier} to invalidate not found"),
            }
        }
        Ok(())
    }

    /// Pages the resources flagged synchronize back out, in a buffer on the blit-down queue that
    /// waits for the exec's FIFO value: `handler` is registered before anything is encoded, so a
    /// raise in the encoding still completes; the buffer is committed whatever the encoding did.
    /// Nothing at all without `needs_complete` (the handler is never called).
    pub(super) fn complete_resources(
        &self,
        hold: Hold<'_>,
        resources: &ResourceList<'_>,
        sync: CompleteSync<'_, P>,
        handler: CompletedHandler,
    ) -> RaiseResult<()> {
        // A handler no buffer took goes with the call
        let handler = Some(handler);
        if !sync.needs_complete {
            return Ok(());
        }
        let mut blit_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut compute_resources: Vec<Arc<Resource<P>>> = Vec::new();
        for i in 0..resources.count() {
            let resource = resources.get(i)?;
            if !resource.synchronize() {
                continue;
            }
            // CPU-paging resources are not excluded: an ICB goes to the blit list, whose
            // `synchronizeInEncoder:` it does not recognize.
            if let Some(r) = self.base.get_existing_host_resource(resource.reference)
                && r.needs_sync()
            {
                if r.needs_compute_paging() {
                    compute_resources.push(r);
                } else {
                    blit_resources.push(r);
                }
            }
        }
        let queue = self.device.upgrade().and_then(|d| d.blit_down_queue());
        let command_buffer = match &queue {
            Some(queue) => queue.command_buffer(OnFault::Return)?,
            None => None,
        };
        crate::fatal_unless!(sync.event.is_some(), DeathKind::NoFifoEvent);
        let to_flush: Vec<Arc<Resource<P>>> = blit_resources
            .iter()
            .chain(&compute_resources)
            .cloned()
            .collect();
        let mut handler = handler.map(|h| {
            let wrapped: CompletedHandler = Box::new(move |info: CompletedInfo| {
                if info.status == CommandBufferStatus::COMPLETED {
                    for r in &to_flush {
                        r.flush_to_guest();
                    }
                }
                h(info);
            });
            wrapped
        });
        if let Some(buffer) = &command_buffer {
            buffer.encode_wait(sync.event.map(EventRef::Shared), sync.value)?;
            if let Some(handler) = handler.take() {
                buffer.add_completed_handler(OnFault::Return, handler)?;
            }
        }
        let mut blit_encoder: Option<P::BlitEncoder> = None;
        let mut compute_encoder: Option<P::ComputeEncoder> = None;
        let mut synchronized = (|| -> RaiseResult<_> {
            if !blit_resources.is_empty() {
                blit_encoder = match &command_buffer {
                    Some(buffer) => buffer.blit_encoder(OnFault::Return)?,
                    None => None,
                };
                for r in &blit_resources {
                    r.synchronize_in_encoder(hold, blit_encoder.as_ref())?;
                }
                if let Some(encoder) = &blit_encoder {
                    encoder.encode(OnFault::Return, BlitCmd::EndEncoding)?;
                }
                blit_encoder = None;
            }
            if !compute_resources.is_empty() {
                compute_encoder = match &command_buffer {
                    Some(buffer) => buffer.compute_encoder(OnFault::Return)?,
                    None => None,
                };
                for r in &compute_resources {
                    r.synchronize_in_compute_encoder(hold, compute_encoder.as_ref())?;
                }
                if let Some(encoder) = &compute_encoder {
                    encoder.encode(OnFault::Return, ComputeCmd::EndEncoding)?;
                }
                compute_encoder = None;
            }
            Ok(())
        })();
        // Whatever the encoding did: an open encoder is ended and the buffer committed; a raise
        // here goes on in place of the encoding's, which ends there
        if let Some(encoder) = &blit_encoder {
            hold.cleanup_step(
                &mut synchronized,
                encoder.encode(OnFault::Return, BlitCmd::EndEncoding),
            )?;
        }
        if let Some(encoder) = &compute_encoder {
            hold.cleanup_step(
                &mut synchronized,
                encoder.encode(OnFault::Return, ComputeCmd::EndEncoding),
            )?;
        }
        if let Some(buffer) = &command_buffer {
            hold.cleanup_step(&mut synchronized, buffer.commit(OnFault::Return))?;
        }
        // The one caller is under the exec's handler: the encoding's raise goes on
        synchronized
    }

    /// The cursors of an exec's command buffers, each copied now, in order; null ones are kept.
    pub(super) fn cursors(
        &self,
        hold: Hold<'_>,
        cmd_buffers: &Records<'_, CmdExecIndirectCmdBuffer>,
    ) -> Vec<Option<Cursor>> {
        cmd_buffers
            .iter()
            .map(|cmd_buffer| self.cursor(hold, cmd_buffer.offset, cmd_buffer.length))
            .collect()
    }

    /// An exec (opcodes 55 and 67), under the task's shared lock: the object lists mapped (else
    /// 2 at once), the preparation, then one buffer on the exec queue that waits for the paging
    /// buffer, decodes every command buffer, signals the FIFO value and commits, then the
    /// synchronize buffer. A raise is logged and reports 2.
    pub(super) fn exec_indirect(
        &self,
        hold: Hold<'_>,
        cmd_buffers: Records<'_, CmdExecIndirectCmdBuffer>,
        resource_list: ResourceList<'_>,
        fifo_event: Option<&P::SharedEvent>,
        fifo_event_value: &Arc<AtomicU64>,
        done: Completion,
    ) -> RaiseResult<()> {
        // DIAG(exec-steps): each step's time, written at debug where the exec takes 5 ms or more.
        let diag = log::log_enabled!(log::Level::Debug);
        let t0 = std::time::Instant::now();
        let steps = std::cell::RefCell::new(Vec::<(&'static str, f64)>::new());
        // Each step's time also goes to the per-step counters, always (`perfstats`).
        let last = std::cell::Cell::new(std::time::Duration::ZERO);
        let mark = |label: &'static str| {
            let now = t0.elapsed();
            crate::perfstats::exec_step(label, now.saturating_sub(last.replace(now)));
            if diag {
                steps.borrow_mut().push((label, now.as_secs_f64() * 1000.0));
            }
        };
        let _lock = self.base.mutex.read();
        mark("lock");
        let (deserializer, _, lists) = self.parts();
        // Whether the task's list was set within the last few seconds, as this exec starts: only
        // then can a missing object be a dead process's leftover (below).
        let young = lists.is_young();
        let mapped = lists.ensure_mapped(hold);
        mark("ensure_mapped");
        if !mapped {
            done(ExecStatus::FAIL, None);
            return Ok(());
        }
        let executed = (|| -> RaiseResult<_> {
            let prepare_sync = self.prepare_resources(
                hold,
                &resource_list,
                fifo_event,
                fifo_event_value.load(Ordering::SeqCst),
            )?;
            mark("prepare_resources");
            let queue = self.device.upgrade().and_then(|d| d.exec_queue());
            let command_buffer = match &queue {
                Some(queue) => queue.command_buffer(OnFault::Return)?,
                None => None,
            };
            if prepare_sync.needs_sync
                && let Some(buffer) = &command_buffer
            {
                buffer.encode_wait(
                    prepare_sync.event.as_ref().map(EventRef::Event),
                    prepare_sync.value,
                )?;
            }
            // Every copy now, before the decode
            let mut cursors = self.cursors(hold, &cmd_buffers);
            mark("cursors");
            // Held past the commit, to the end of the exec (`Decoders`).
            let _decoders =
                deserializer.decode_segments(hold, &mut cursors, command_buffer.as_ref())?;
            mark("decode");
            if let Some(buffer) = &command_buffer {
                let exec_buffer = buffer.clone_ref();
                let handler = Arc::clone(&done);
                let needs_complete = prepare_sync.needs_complete;
                buffer.add_completed_handler(
                    OnFault::Return,
                    Box::new(move |info: CompletedInfo| {
                        exec_completed::<P>(&exec_buffer, &handler, needs_complete, &info);
                    }),
                )?;
            }
            // The FIFO value advances before the commit
            let value = fifo_event_value
                .fetch_add(1, Ordering::SeqCst)
                .wrapping_add(1);
            if let Some(buffer) = &command_buffer {
                buffer.encode_signal(fifo_event.map(EventRef::Shared), value)?;
                buffer.commit(OnFault::Return)?;
            }
            mark("commit");
            let complete_sync = CompleteSync::<P> {
                needs_complete: prepare_sync.needs_complete,
                event: fifo_event,
                value,
            };
            let exec_buffer = command_buffer.as_ref().map(Handle::clone_ref);
            let handler = Arc::clone(&done);
            self.complete_resources(
                hold,
                &resource_list,
                complete_sync,
                Box::new(move |info: CompletedInfo| {
                    sync_completed::<P>(exec_buffer.as_ref(), &handler, &info);
                }),
            )
        })();
        mark("complete_resources");
        crate::perfstats::EXEC.add(t0.elapsed());
        crate::perfstats::task_exec(self.task_id, t0.elapsed());
        if diag && t0.elapsed().as_secs_f64() >= 0.005 {
            let mut line = String::new();
            let mut last = 0.0;
            for (label, at) in steps.borrow().iter() {
                line.push_str(&format!(" {label}={:.1}", at - last));
                last = *at;
            }
            log::debug!(
                "task {}: exec steps (ms):{line}; total {:.1}",
                self.task_id,
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        if let Err(raised) = executed {
            // An exec whose objects are missing from the task's list (an empty entry, a failed
            // lookup) is, as measured on WHPX, a dead process's leftover: the guest deletes a
            // task, defines its id again for a new process within milliseconds, and still drains
            // the old process's queued commands into the FIFOs. Run against the new process's
            // fresh list they fault, and the fault, reported against the id, gets the new process
            // killed, which repeats (blank icons, windows that never open, a restart that waits
            // on them). Such an exec is dropped as done instead, but only while the task is young
            // (its list set within the last seconds, when leftovers drain): a missing object in
            // an established task is a real fault and is reported as one.
            // REIMS_VGPU_EMPTY_ENTRY_FAULTS=1 reports every one as before.
            static REPORT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let report = *REPORT.get_or_init(|| {
                std::env::var("REIMS_VGPU_EMPTY_ENTRY_FAULTS").is_ok_and(|v| v == "1")
            });
            let what = raised.exception();
            let missing = what.contains("has an empty entry") || what.ends_with("lookup failed");
            if missing && young && !report {
                log::warn!(
                    "task {}: exec dropped (an object it names is not in the task's list): {what}",
                    self.task_id
                );
                done(ExecStatus::OK, None);
                drop(raised);
                return Ok(());
            }
            // A raise after `complete_resources` committed its buffer reports 2 here, and that
            // buffer's handler reports again later.
            log::warn!(
                "task {}: exec faulted: {}",
                self.task_id,
                raised.exception()
            );
            done(ExecStatus::FAIL, None);
            hold.land(raised);
        }
        Ok(())
    }

    /// Resets a rasterization rate map from its guest data, under the task's shared lock and in
    /// a holder of its own ([`Held::raising`]), where the object lists' page-table reads hold
    /// the device: released at its end, or, on a raise, carried on with the raise. Reference 0
    /// logs and reports 2, an absent map reports 2, and an existing one ends the process.
    pub(super) fn reset_rate_map(&self, map: u32, done: &Completion) -> RaiseResult<()> {
        let _lock = self.base.mutex.read();
        Held::raising(|held| {
            let hold = Hold::of(held);
            let (_, resource_manager, lists) = self.parts();
            self.map_object_lists(hold, &lists)?;
            // Ref 0 faults; an absent map faults (2); an existing one aborts.
            if map == 0 {
                log::warn!("task {}: rate map reset refused: reference 0", self.task_id);
                done(ExecStatus::FAIL, None);
                return Ok(());
            }
            if resource_manager
                .get_existing_object::<RasterizationRateMaps>(map)
                .is_none()
            {
                done(ExecStatus::FAIL, None);
                return Ok(());
            }
            // The map's data is read through its object-list entry, which a local task does not
            // provide: the process ends here, before any reset.
            self.base.unsupported(Unsupported::GetObject)
        })
    }
}

/// The exec buffer's completion: 3 (logged) on a GPU error, else 1, reported only when no
/// synchronize buffer will report.
fn exec_completed<P: Platform>(
    command_buffer: &P::CommandBuffer,
    handler: &Completion,
    needs_complete: bool,
    info: &CompletedInfo,
) {
    let result = if info.status == CommandBufferStatus::ERROR {
        let error = command_buffer.error();
        log::warn!(
            "an exec's command buffer failed: {}",
            error_text(error.as_ref())
        );
        ExecStatus::GPU_OR_TIMEOUT
    } else {
        ExecStatus::OK
    };
    if !needs_complete {
        handler(result, command_buffer.error());
    }
}

/// The synchronize buffer's completion: waits for the exec buffer, then reports 1 when both
/// completed, else 3 with the exec buffer's error, or this buffer's.
pub(super) fn sync_completed<P: Platform>(
    command_buffer: Option<&P::CommandBuffer>,
    handler: &Completion,
    info: &CompletedInfo,
) {
    // Metal's completion thread: no handler, so a raise dies at its site. The wait's `Die` dies
    // at the send, so this arm is never reached.
    if let Some(buffer) = command_buffer
        && let Err(raised) = buffer.wait_until_completed(OnFault::Die)
    {
        raised.fatal();
    }
    let status = || command_buffer.map_or(CommandBufferStatus::default(), |b| b.status());
    // The exec buffer's status is asked only when this buffer completed
    if info.status == CommandBufferStatus::COMPLETED && status() == CommandBufferStatus::COMPLETED {
        handler(ExecStatus::OK, None);
    } else if status() == CommandBufferStatus::ERROR {
        handler(
            ExecStatus::GPU_OR_TIMEOUT,
            command_buffer.and_then(|b| b.error()),
        );
    } else {
        handler(ExecStatus::GPU_OR_TIMEOUT, info.error.clone());
    }
}
