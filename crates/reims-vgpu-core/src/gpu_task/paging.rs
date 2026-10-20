//! Paging around an exec: the resources it pages in before it runs ([`GpuTask::prepare_resources`])
//! and the synchronized ones it pages out after it ([`GpuTask::complete_resources`]), each on the
//! task's paging queue.
//!
//! Each encodes inside a handler whose cleanup always runs: the open encoders end, the buffer is
//! signalled and committed, and only then does a raise go back to the caller as `Err` (the
//! caller's handler logs it and replies 2).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::DeathKind;
use super::service::GpuTask;
use crate::fatal_unless;
use crate::fifo::wire::Records;
use crate::guest::records::CmdExecIndirectResource3;
use crate::platform::Platform;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::command::{CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo};
use crate::platform::compute::{ComputeCmd, ComputeEncoder};
use crate::platform::event::EventRef;
use crate::platform::types::CommandBufferStatus;
use crate::raise::{OnFault, RaiseResult};
use crate::resources::resource::Resource;
use crate::rt::Hold;

/// What paging-in left for the exec.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PrepareSync {
    /// The exec waits for the paging buffer: one was committed.
    pub(super) needs_sync: bool,
    /// The exec pages synchronized resources out after it.
    pub(super) needs_complete: bool,
    /// The paging event's value the paging buffer signals.
    pub(super) value: u64,
}

/// What paging-out waits for: the FIFO's event at the exec's signal.
pub(super) struct CompleteSync<P: Platform> {
    /// Paging-out has resources to synchronize.
    pub(super) needs_complete: bool,
    /// The FIFO's event.
    pub(super) event: Option<P::SharedEvent>,
    /// The value the exec signals.
    pub(super) value: u64,
}

impl<P: Platform> GpuTask<P> {
    /// Pages in the resources the records name, under the prepare lock.
    ///
    /// Each record's resource is marked (host-invalid, guest-valid) as its flags say and sorted
    /// by how it pages: CPU, compute or blit. Blit and compute resources get one paging buffer,
    /// which waits on `event` at `value` when given, then one blit encoder and one compute
    /// encoder; the cleanup ends them, bumps the paging counter, signals the paging event at
    /// the new value and commits, before a raise goes on. CPU paging runs next, then the
    /// guest-invalidated resources are marked, and their shared-texture backings go to
    /// `shared_texture_backing_ids`.
    pub(super) fn prepare_resources(
        &self,
        hold: Hold<'_>,
        resources: Records<'_, CmdExecIndirectResource3>,
        event: Option<&P::SharedEvent>,
        value: u64,
        shared_texture_backing_ids: &mut Vec<u32>,
    ) -> RaiseResult<PrepareSync> {
        let _prepare = self.prepare_lock.lock();
        let core = self.core();
        let mut blit_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut compute_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut cpu_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut invalidated_resources: Vec<(u32, Option<Arc<Resource<P>>>)> = Vec::new();
        let mut sync = PrepareSync::default();
        for record in resources.iter() {
            // Reference 0 is none
            let resource = core.get_host_resource(hold, record.reference)?;
            if let Some(r) = resource.as_ref().filter(|_| record.invalidate_host()) {
                r.set_is_guest_valid(true);
                r.set_is_host_valid(false);
            }
            if let Some(r) = resource.as_ref().filter(|r| r.needs_prepare()) {
                if r.needs_cpu_paging() {
                    cpu_resources.push(Arc::clone(r));
                } else if r.needs_compute_paging() {
                    compute_resources.push(Arc::clone(r));
                } else {
                    blit_resources.push(Arc::clone(r));
                }
            }
            if record.invalidate_guest() {
                invalidated_resources.push((record.reference, resource.clone()));
            }
            if record.synchronize() {
                sync.needs_complete = true;
            }
        }
        if !blit_resources.is_empty() || !compute_resources.is_empty() {
            let command_buffer = match &core.paging_queue {
                Some(queue) => queue.command_buffer(OnFault::Return)?,
                None => None,
            };
            if let Some(e) = event
                && let Some(b) = &command_buffer
            {
                b.encode_wait(Some(EventRef::Shared(e)), value)?;
            }
            let mut blit_encoder: Option<P::BlitEncoder> = None;
            let mut compute_encoder: Option<P::ComputeEncoder> = None;
            let mut encoded = (|| -> RaiseResult<_> {
                if !blit_resources.is_empty() {
                    blit_encoder = match &command_buffer {
                        Some(b) => b.blit_encoder(OnFault::Return)?,
                        None => None,
                    };
                    for r in &blit_resources {
                        r.prepare_in_encoder(hold, blit_encoder.as_ref())?;
                    }
                    if let Some(e) = &blit_encoder {
                        e.encode(OnFault::Return, BlitCmd::EndEncoding)?;
                    }
                    blit_encoder = None;
                }
                if !compute_resources.is_empty() {
                    compute_encoder = match &command_buffer {
                        Some(b) => b.compute_encoder(OnFault::Return)?,
                        None => None,
                    };
                    for r in &compute_resources {
                        r.prepare_in_compute_encoder(hold, compute_encoder.as_ref())?;
                    }
                    if let Some(e) = &compute_encoder {
                        e.encode(OnFault::Return, ComputeCmd::EndEncoding)?;
                    }
                    compute_encoder = None;
                }
                Ok(())
            })();
            // The cleanup, on both paths: a raise here goes on in place of the body's
            if let Some(e) = &blit_encoder {
                hold.cleanup_step(
                    &mut encoded,
                    e.encode(OnFault::Return, BlitCmd::EndEncoding),
                )?;
            }
            if let Some(e) = &compute_encoder {
                hold.cleanup_step(
                    &mut encoded,
                    e.encode(OnFault::Return, ComputeCmd::EndEncoding),
                )?;
            }
            sync.value = core
                .paging_counter
                .fetch_add(1, Ordering::SeqCst)
                .wrapping_add(1);
            if let Some(b) = &command_buffer {
                let event = core.paging_event.as_ref().map(EventRef::Event);
                let signalled = b.encode_signal(event, sync.value);
                hold.cleanup_step(&mut encoded, signalled)?;
            }
            if let Some(b) = &command_buffer {
                hold.cleanup_step(&mut encoded, b.commit(OnFault::Return))?;
            }
            // Every caller is under a handler (an exec's or an info's): the raise goes on
            encoded?;
            // Only when nothing raised
            sync.needs_sync = true;
        }
        // A handler that re-raises what it catches
        (|| -> RaiseResult<_> {
            for r in &cpu_resources {
                r.prepare_with_cpu(hold)?;
            }
            Ok(())
        })()?;
        for (reference, resource) in &invalidated_resources {
            if let Some(r) = resource {
                r.set_is_guest_valid(false);
            }
            let backing = match resource {
                Some(r) => r.get_shared_texture_backing_id()?,
                None => 0,
            };
            // The host invalidates the kernel task's backings after the reply
            if backing != 0 {
                shared_texture_backing_ids.push(backing);
            }
            if !resource.as_ref().is_some_and(|r| r.is_host_valid()) {
                log::warn!(
                    "task {}: guest copy of reference {reference} invalidated while it has no \
                     valid host copy",
                    self.id()
                );
            }
        }
        Ok(sync)
    }

    /// Pages out, behind the exec, the resources the records mark to synchronize (not the CPU
    /// ones): one paging buffer waits on the FIFO's event at the exec's signal, takes
    /// `completion_handler`, and gets one blit and one compute encoder; the cleanup ends them and
    /// commits before a raise goes on. Nothing without `needs_complete`.
    pub(super) fn complete_resources(
        &self,
        hold: Hold<'_>,
        resources: Records<'_, CmdExecIndirectResource3>,
        sync: CompleteSync<P>,
        completion_handler: CompletedHandler,
    ) -> RaiseResult<()> {
        let mut blit_resources: Vec<Arc<Resource<P>>> = Vec::new();
        let mut compute_resources: Vec<Arc<Resource<P>>> = Vec::new();
        if !sync.needs_complete {
            return Ok(());
        }
        let core = self.core();
        for record in resources.iter() {
            if !record.synchronize() {
                continue;
            }
            let resource = core.get_existing_host_resource(record.reference);
            // The service leaves CPU-paging resources out.
            if let Some(r) = resource.filter(|r| r.needs_sync() && !r.needs_cpu_paging()) {
                if r.needs_compute_paging() {
                    compute_resources.push(r);
                } else {
                    blit_resources.push(r);
                }
            }
        }
        let command_buffer = match &core.paging_queue {
            Some(queue) => queue.command_buffer(OnFault::Return)?,
            None => None,
        };
        fatal_unless!(sync.event.is_some(), DeathKind::SyncWithoutFifoEvent);
        let to_flush: Vec<Arc<Resource<P>>> = blit_resources
            .iter()
            .chain(&compute_resources)
            .cloned()
            .collect();
        let completion_handler: CompletedHandler = Box::new(move |info: CompletedInfo| {
            if info.status == CommandBufferStatus::COMPLETED {
                for r in &to_flush {
                    r.flush_to_guest();
                }
            }
            completion_handler(info);
        });
        if let Some(b) = &command_buffer {
            b.encode_wait(sync.event.as_ref().map(EventRef::Shared), sync.value)?;
        }
        if let Some(b) = &command_buffer {
            b.add_completed_handler(OnFault::Return, completion_handler)?;
        }
        let mut blit_encoder: Option<P::BlitEncoder> = None;
        let mut compute_encoder: Option<P::ComputeEncoder> = None;
        let mut encoded = (|| -> RaiseResult<_> {
            if !blit_resources.is_empty() {
                blit_encoder = match &command_buffer {
                    Some(b) => b.blit_encoder(OnFault::Return)?,
                    None => None,
                };
                for r in &blit_resources {
                    r.synchronize_in_encoder(hold, blit_encoder.as_ref())?;
                }
                if let Some(e) = &blit_encoder {
                    e.encode(OnFault::Return, BlitCmd::EndEncoding)?;
                }
                blit_encoder = None;
            }
            if !compute_resources.is_empty() {
                compute_encoder = match &command_buffer {
                    Some(b) => b.compute_encoder(OnFault::Return)?,
                    None => None,
                };
                for r in &compute_resources {
                    r.synchronize_in_compute_encoder(hold, compute_encoder.as_ref())?;
                }
                if let Some(e) = &compute_encoder {
                    e.encode(OnFault::Return, ComputeCmd::EndEncoding)?;
                }
                compute_encoder = None;
            }
            Ok(())
        })();
        // The cleanup, on both paths: a raise here goes on in place of the body's
        if let Some(e) = &blit_encoder {
            hold.cleanup_step(
                &mut encoded,
                e.encode(OnFault::Return, BlitCmd::EndEncoding),
            )?;
        }
        if let Some(e) = &compute_encoder {
            hold.cleanup_step(
                &mut encoded,
                e.encode(OnFault::Return, ComputeCmd::EndEncoding),
            )?;
        }
        if let Some(b) = &command_buffer {
            hold.cleanup_step(&mut encoded, b.commit(OnFault::Return))?;
        }
        // The one caller is under the exec's handler: the encoding's raise goes on
        encoded
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::platform::device::GpuDevice;
    use crate::testkit::capture::catch_death;
    use crate::testkit::script::Step;

    /// Nothing to page: no buffer, the paging counter unmoved, nothing to wait for
    #[test]
    fn nothing_to_page_makes_no_buffer() {
        let tx = ready(&rw_pages(4));
        let mark = tx.p.call_log().mark();
        let mut ids = Vec::new();
        let sync = tx
            .task
            .prepare_resources(Hold::NONE, Records::new(&[], 0).unwrap(), None, 0, &mut ids)
            .unwrap();
        assert_eq!(sync, PrepareSync::default());
        assert!(tx.p.call_log().sels_since(mark).is_empty());
        let r = tx.task.complete_resources(
            Hold::NONE,
            Records::new(&[], 0).unwrap(),
            CompleteSync {
                needs_complete: false,
                event: None,
                value: 0,
            },
            Box::new(|_| {}),
        );
        assert_eq!(r, Ok(()));
        assert!(tx.p.call_log().sels_since(mark).is_empty());
    }

    /// Paging-out with resources to synchronize but no FIFO event dies before any send
    #[test]
    fn paging_out_without_a_fifo_event_dies() {
        let tx = ready(&rw_pages(4));
        let d = catch_death(|| {
            tx.task.complete_resources(
                Hold::NONE,
                Records::new(&[], 0).unwrap(),
                CompleteSync {
                    needs_complete: true,
                    event: None,
                    value: 0,
                },
                Box::new(|_| {}),
            )
        });
        assert_eq!(d.unwrap_err(), DeathKind::SyncWithoutFifoEvent.site());
    }

    /// Paging-out waits on the FIFO's event, adds the handler, then commits; a raise before
    /// the encoding (the handler's) goes on with no commit
    #[test]
    fn paging_out_order_and_an_early_raise() {
        let tx = ready(&rw_pages(4));
        let event = tx.p.device().new_shared_event();
        let mark = tx.p.call_log().mark();
        // The wait is the first send inside; the commit comes after the handler
        tx.p.script().push(
            "addCompletedHandler:",
            Step::raise("NSInternalInconsistencyException", "x"),
        );
        let r = tx.task.complete_resources(
            Hold::NONE,
            Records::new(&[], 0).unwrap(),
            CompleteSync {
                needs_complete: true,
                event,
                value: 1,
            },
            Box::new(|_| {}),
        );
        // The handler's raise is outside the encoding: it goes on before the commit
        assert!(r.is_err());
        let sels = tx.p.call_log().sels_since(mark);
        assert!(sels.contains(&"encodeWaitForEvent:value:"));
        assert!(!sels.contains(&"commit"));
        tx.p.script().clear();
        let mark = tx.p.call_log().mark();
        let r = tx.task.complete_resources(
            Hold::NONE,
            Records::new(&[], 0).unwrap(),
            CompleteSync {
                needs_complete: true,
                event: tx.p.device().new_shared_event(),
                value: 1,
            },
            Box::new(|_| {}),
        );
        assert_eq!(r, Ok(()));
        let sels = tx.p.call_log().sels_since(mark);
        let pos = |s: &str| sels.iter().position(|x| *x == s).unwrap();
        assert!(pos("encodeWaitForEvent:value:") < pos("addCompletedHandler:"));
        assert!(pos("addCompletedHandler:") < pos("commit"));
    }
}
