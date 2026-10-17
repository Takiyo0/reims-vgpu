//! The frame a display presents: the present itself, the wait for frames in flight, and the
//! guest present's signal into the newFrame source.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::Duration;

use parking_lot::MutexGuard;

use crate::display::{ColorMatrix, DeathKind, DisplayCore, DisplayLocked, GammaTableArgs};
use crate::host::HostQueue;
use crate::num::Truncate;
use crate::platform::os::{Clock, Deadline};
use crate::platform::resource::Texture;
use crate::platform::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};
use crate::rt::Hold;
use crate::rt::queue::Executor;
use crate::rt::signal::CoalescingSignal;
use crate::seams::links::KernelTaskRef;

/// How long a present waits for frames in flight: 2 s of wall clock.
pub const PENDING_FRAMES_TIMEOUT: Duration = Duration::from_secs(2);

/// The newFrame source: core's model of a `DATA_ADD` source on the display's queue
/// (`rt::CoalescingSignal`). `Clone` is a reference to the same source.
#[derive(Clone)]
pub struct NewFrameSource(CoalescingSignal);

impl NewFrameSource {
    /// `dispatch_source_merge_data(source, v)`.
    pub fn merge(&self, v: u64) {
        self.0.merge(v)
    }

    /// `dispatch_source_cancel(source)`.
    pub fn cancel(&self) {
        self.0.cancel()
    }

    /// Whether it is cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }

    /// A resumed `DATA_ADD` source on `queue` running `handler`.
    pub(crate) fn new_add(
        queue: &Arc<dyn HostQueue>,
        handler: impl Fn(u64) + Send + Sync + 'static,
    ) -> Self {
        let target: Arc<dyn Executor> = queue.clone();
        NewFrameSource(CoalescingSignal::new_add(target, handler))
    }
}

impl<P: Platform> DisplayCore<P> {
    /// Waits for frames in flight (up to 2 s), then the mode change, the gamma table, the frame,
    /// the colour matrix and the completion, all under the display mutex. A gamma raise leaves
    /// the frame and the matrix alone but the mode change done. `frame` is the caller's
    /// reference, which the display keeps. The matrix's raise is as `on` says: a guest
    /// display's present runs under its FIFO command's handler, and the EFI display's with none
    /// (and with no gamma table or matrix).
    #[allow(clippy::too_many_arguments)] // the present's own arguments
    pub fn present_frame(
        &self,
        hold: Hold<'_>,
        on: OnFault,
        frame: Option<P::Texture>,
        iosurface_pixel_format: u32,
        protection_requirements: u64,
        task: Option<KernelTaskRef<P>>,
        gamma: GammaTableArgs,
        color_matrix: Option<&dyn ColorMatrix>,
        completion: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<()> {
        // DIAG(present-wait): the display lock and the in-flight wait, written at debug where they
        // take 1 ms or more.
        let timed = log::log_enabled!(log::Level::Debug).then(std::time::Instant::now);
        let mut g = self.display_mutex.lock();
        let locked = timed.map(|t| t.elapsed());
        let in_flight = self.in_flight_frame_count.load(Ordering::SeqCst);
        self.wait_for_pending_frames(&mut g);
        if let (Some(t), Some(locked)) = (timed, locked) {
            let all = t.elapsed();
            if all.as_secs_f64() >= 0.001 {
                log::debug!(
                    "present wait: {:.2} ms ({:.2} ms for the display lock, {in_flight} frame(s) in flight at the lock)",
                    all.as_secs_f64() * 1000.0,
                    locked.as_secs_f64() * 1000.0
                );
            }
        }
        if self.guest_color_space_dirty() {
            // The guest display's colour space (nil without a guest display). The guest display
            // pointer is read without its lock held across the call; the cell is a leaf.
            let guest_display = self.guest_display.lock().clone();
            g.guest_color_space = guest_display
                .as_ref()
                .and_then(Weak::upgrade)
                .and_then(|n| n.guest_color_space());
            *self.guest_color_space_unlocked.lock() = g.guest_color_space.clone();
        }
        // An EFI present to a display without the handler dies here.
        crate::fatal_unless!(
            self.queue().is_some() && self.new_frame_event_handler().is_some(),
            DeathKind::PresentWithoutNewFrameHandler
        );
        // The frame's size truncates to 16 bits.
        let width = crate::nil_send!(frame, width()).low_u16();
        let height = crate::nil_send!(frame, height()).low_u16();
        if g.width != width
            || g.height != height
            || iosurface_pixel_format != g.iosurface_pixel_format
            || protection_requirements != g.protection_requirements
            || self.guest_color_space_dirty()
        {
            // Rare (a mode change): the guest's presented size and IOSurface format, e.g. 'RGhA'
            // (half float) or 'BGRA'.
            log::debug!(
                "present mode: {width}x{height} iosurface format {:?} ({iosurface_pixel_format:#x})",
                String::from_utf8_lossy(&iosurface_pixel_format.to_be_bytes())
            );
            self.set_guest_color_space_dirty(false);
            self.mode_change_locked(
                &mut g,
                u64::from(width),
                u64::from(height),
                iosurface_pixel_format,
                protection_requirements,
            );
        }
        if gamma.entry_count != 0 {
            // A raise here leaves the mode change dispatched and the frame and matrix
            // untouched.
            self.update_gamma_table_locked(hold, &mut g, task.as_deref(), gamma)?;
        }
        // The old frame released, the new one held (a view: never copied).
        if g.current_frame.as_ref().map(Handle::id) != frame.as_ref().map(Handle::id) {
            g.current_frame = frame;
            self.has_current_frame
                .store(g.current_frame.is_some(), Ordering::SeqCst);
        }
        if let Some(m) = color_matrix {
            self.update_color_matrix_locked(on, &g, m)?;
        }
        // Under the display mutex.
        completion();
        Ok(())
    }

    /// Waits on the in-flight condition with the display mutex held until no frame is in
    /// flight, or 2 s of wall clock have passed; then logs a fault and proceeds.
    pub fn wait_for_pending_frames(&self, guard: &mut MutexGuard<'_, DisplayLocked<P>>) {
        let clock = self.platform.clock();
        let deadline = Deadline::Wall(clock.now_wall().saturating_add(PENDING_FRAMES_TIMEOUT));
        // Spurious wakeups loop.
        while self.in_flight_frame_count.load(Ordering::SeqCst) != 0 {
            // An uncommitted held encode delays this by up to 2 s.
            if clock
                .wait_until(&self.in_flight_condition, guard, deadline)
                .timed_out()
            {
                log::error!(
                    "frames still in flight after {PENDING_FRAMES_TIMEOUT:?}; going on without them"
                );
                return;
            }
        }
    }

    /// Nothing without a current frame; else a guest present: the count, and one merge into the
    /// newFrame source (coalescing).
    pub fn signal_current_frame(&self) {
        // No lock: the guest display calls it without the display mutex, EFI's completion with it
        // held.
        if !self.has_current_frame.load(Ordering::SeqCst) {
            return;
        }
        self.current_frame_encoded.store(false, Ordering::SeqCst);
        // A load and a store of the count.
        let guest = self.guest_present_count().wrapping_add(1);
        self.props.set_guest_present_count(guest);
        // DATA_ADD coalesces: several signals, one handler call.
        match &*self.new_frame_event_source.lock() {
            Some(source) => source.merge(1),
            None => crate::fatal!(DeathKind::NoNewFrameSource),
        }
    }
}
