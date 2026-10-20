//! The cursor: the glyph the guest draws (copied into the host's image), the show byte, the
//! doorbell that hands the host the guest's cursor changes, and the position.
//!
//! Each hands the host its handler's call on the client's queue, read again for the dispatch;
//! the block reads the client and its handler again when it runs.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{DeathKind, GuestDisplay, GuestDisplayLocked, RaiseKind};
use crate::device::tasks::MAX_LOCAL_TASK_LENGTH;
use crate::display::host_calls::CursorGlyph;
use crate::display::page as sp;
use crate::display::{DisplayCoord, DisplayCore};
use crate::num::{ToUsize, Truncate};
use crate::platform::Platform;
use crate::raise::{OnFault, RaiseResult};
use crate::rt::{Held, Hold};
use crate::tasks::local::LocalTask;

/// A cursor glyph as the guest describes it (the glyph command's and the shared page's
/// records): the BGRA pixels' place in the kernel task, their row stride, the size, the hot
/// spot and the guest's sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CursorGlyphArgs {
    /// The pixels' offset in the kernel task.
    pub virtual_offset: u64,
    /// The bytes the task copies.
    pub mapped_length: u64,
    /// The guest's row stride in bytes.
    pub stride: u64,
    /// The width in pixels.
    pub width: u16,
    /// The height in pixels.
    pub height: u16,
    /// The hot spot's x.
    pub hot_spot_x: u16,
    /// The hot spot's y.
    pub hot_spot_y: u16,
    /// The guest's sum of the pixels; 0 skips the check.
    pub sum: u32,
}

/// The glyph limit in either dimension.
pub(super) const GLYPH_LIMIT: u16 = 256;
/// `cursorHandlers` bit 0: a show handler.
pub(super) const CURSOR_HANDLER_SHOW: u32 = 1;
/// `cursorHandlers` bit 1: a move handler.
pub(super) const CURSOR_HANDLER_MOVE: u32 = 2;

impl<P: Platform> GuestDisplay<P> {
    /// Copies the guest's BGRA glyph into `buffer` as RGBA (`width * height` words, packed):
    /// the task's copy of `mapped_length` bytes into a block, the rows with bytes 0 and 2
    /// swapped, then the sum check (a zero sum skips it). Raises for a mapped length past the
    /// largest local task, or when the sum differs; the task's raise frees the block.
    pub(super) fn copy_guest_glyph(
        &self,
        hold: Hold<'_>,
        task: Option<&LocalTask<P>>,
        args: CursorGlyphArgs,
        buffer: &mut [u32],
    ) -> RaiseResult<()> {
        // A mapped length past the largest local task faults here, before any block is made: no
        // task holds such a range, so the copy would fault it anyway
        // (`tests::cursor_glyph::glyph_mapped_length_past_any_task_faults`).
        if args.mapped_length > MAX_LOCAL_TASK_LENGTH {
            crate::raise!(
                RaiseKind::CursorBuffer,
                format_args!(
                    "cursor glyph mapped length {:#x} is past any task",
                    args.mapped_length
                )
            );
        }
        // Zeroed (lazily mapped); the bytes past the task's copy are zero. No task copies
        // nothing, and a raise out of the copy frees the block.
        let mut guest = vec![0u8; args.mapped_length.to_usize()];
        if let Some(t) = task {
            t.copy_from_virtual_offset(hold, args.virtual_offset, &mut guest)?;
        }
        // `int` loops, u32 sum wrap, packed output.
        let mut host_sum = 0u32;
        let mut out = buffer.iter_mut();
        let mut y = 0u16;
        while y < args.height {
            let row = u64::from(y).wrapping_mul(args.stride);
            let mut x = 0u16;
            while x < args.width {
                let at = row.wrapping_add(u64::from(x).wrapping_mul(4));
                // The callers checked `(height - 1) * stride + width * 4 <= mapped_length`.
                let pixel = guest
                    .get(at.to_usize()..)
                    .and_then(|s| s.get(..4))
                    .and_then(|s| <[u8; 4]>::try_from(s).ok())
                    .map_or(0, u32::from_le_bytes);
                host_sum = host_sum.wrapping_add(pixel);
                let v = (pixel & 0xff00_ff00)
                    | (pixel & 0xff).wrapping_shl(16)
                    | (pixel.wrapping_shr(16) & 0xff);
                if let Some(o) = out.next() {
                    *o = v;
                }
                x = x.wrapping_add(1);
            }
            y = y.wrapping_add(1);
        }
        drop(guest);
        // A zero sum skips the check.
        if args.sum != 0 && args.sum != host_sum {
            crate::raise!(
                RaiseKind::CursorCorrupt,
                format!(
                    "guest display on port {}: the guest's cursor sum {:#010x} is not the glyph's \
                     {host_sum:#010x}",
                    self.port, args.sum
                )
            );
        }
        Ok(())
    }

    /// The guest's cursor glyph: checked, copied into the host's image (or a bitmap) and handed
    /// to the client's glyph handler on its queue, then recorded in the page. A raise leaves the
    /// page as it was and dispatches nothing.
    pub fn cursor_glyph(&self, args: CursorGlyphArgs) -> RaiseResult<()> {
        let mut l = self.guest_display_mutex.lock();
        let keep = self.client_core();
        self.cursor_glyph_locked(&mut l, keep.as_ref(), args)
    }

    /// See [`GuestDisplay::cursor_glyph`]. The image is the host's: the handler gets the RGBA copy
    /// the image holds.
    ///
    /// The checks, the copy and the dispatch run in a holder: the kernel task, the host's image
    /// and what the copy holds with no handle to a holder of its own go at its end, most recent
    /// first, before the page records the glyph. On a raise they go on with it to where it ends.
    pub(super) fn cursor_glyph_locked(
        &self,
        l: &mut GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
        args: CursorGlyphArgs,
    ) -> RaiseResult<()> {
        Self::assert_running(l);
        let CursorGlyphArgs {
            virtual_offset,
            mapped_length,
            stride,
            width,
            height,
            hot_spot_x,
            hot_spot_y,
            sum,
        } = args;
        Held::raising(|held| {
            let hold = Hold::of(held);
            // The reason is built first.
            let reason = format!(
                "guest display on port {}: cursor of {width}x{height} (at most {GLYPH_LIMIT} a side), \
                 stride {stride:#x}, in {mapped_length:#x} mapped bytes is not valid",
                self.port
            );
            if width == 0 || width > GLYPH_LIMIT || height == 0 || height > GLYPH_LIMIT {
                crate::raise!(RaiseKind::CursorParameters, reason);
            }
            // `os_mul_and_add_overflow((height - 1), stride, width * 4)`.
            let (product, mul_over) = u64::from(height).wrapping_sub(1).overflowing_mul(stride);
            let (needed, add_over) = product.overflowing_add(u64::from(width).wrapping_mul(4));
            if mul_over || add_over {
                crate::raise!(RaiseKind::CursorParameters, reason);
            }
            if needed > mapped_length {
                crate::raise!(RaiseKind::CursorParameters, reason);
            }
            // Required even with no handler.
            let task = self.kernel_task(OnFault::Return, held)?;
            let hot_spot = DisplayCoord {
                x: hot_spot_x,
                y: hot_spot_y,
            };
            let pixels = u64::from(width).wrapping_mul(u64::from(height)).to_usize();
            if let Some(c) = client
                && c.queue().is_some()
                && c.cursor_glyph_handler().is_some()
            {
                if c.cursor_glyph_handler().is_some()
                    && let Some(make) = c.descriptor.cursor_glyph_image_maker.clone()
                {
                    // The host's image, made here on the device thread under the guest display's
                    // mutex, and held.
                    let Some((image, made)) = make(width, height) else {
                        crate::raise!(
                            RaiseKind::CursorImage,
                            "no cursor image made from the glyph"
                        );
                    };
                    held.keep(made);
                    // The copy writes the image's own buffer; a raise takes the image on with it,
                    // and nothing is dispatched.
                    image.write_bitmap_data(hold, &mut |hold, buffer| {
                        let Some(buffer) = buffer else {
                            crate::fatal!(DeathKind::GlyphBufferNull)
                        };
                        self.copy_guest_glyph(hold, task.as_deref(), args, buffer)
                    })?;
                    // The block retains the image (before the holder's end releases the maker's
                    // reference), and the handler receives it +0.
                    let retained = image.retain();
                    let glyph = CursorGlyph {
                        width: u32::from(width),
                        height: u32::from(height),
                        rgba: Arc::from(Vec::new()),
                        image: Some(image),
                    };
                    // The queue read again.
                    let Some(queue) = c.queue() else {
                        crate::fatal!(DeathKind::QueueGone)
                    };
                    self.dispatch(&queue, move |_, client| {
                        match client.and_then(|c| c.cursor_glyph_handler()) {
                            Some(h) => h(&glyph, hot_spot),
                            None => crate::fatal!(DeathKind::GlyphHandlerGone),
                        }
                        drop(glyph);
                        drop(retained);
                    });
                } else if c.cursor_glyph_handler().is_some() {
                    // A bitmap of the copy; its checks cannot fail here.
                    let mut buffer = vec![0u32; pixels];
                    // A raise leaves nothing dispatched.
                    self.copy_guest_glyph(hold, task.as_deref(), args, &mut buffer)?;
                    let glyph = CursorGlyph {
                        width: u32::from(width),
                        height: u32::from(height),
                        rgba: buffer.iter().flat_map(|w| w.to_le_bytes()).collect(),
                        image: None,
                    };
                    // The queue read again.
                    let Some(queue) = c.queue() else {
                        crate::fatal!(DeathKind::QueueGone)
                    };
                    self.dispatch(&queue, move |_, client| {
                        match client.and_then(|c| c.cursor_glyph_handler()) {
                            Some(h) => h(&glyph, hot_spot),
                            None => crate::fatal!(DeathKind::GlyphHandlerGone),
                        }
                    });
                } else {
                    // The handler was let go since it was read (a display's end racing the
                    // command): the second handler's path, whose check fails.
                    crate::fatal!(DeathKind::GlyphHandlerRaced)
                }
            }
            Ok(())
        })?;
        // Only when nothing raised.
        let p = self.page(l);
        p.w64(sp::CURSOR_GLYPH_VIRTUAL_OFFSET, virtual_offset);
        self.page(l)
            .w64(sp::CURSOR_GLYPH_MAPPED_LENGTH, mapped_length);
        self.page(l).w64(sp::CURSOR_GLYPH_STRIDE, stride);
        self.page(l).w32(sp::FIELD_F18, 0);
        self.page(l).w32(sp::CURSOR_GLYPH_SUM, sum);
        self.page(l).w16(sp::CURSOR_GLYPH_WIDTH, width);
        self.page(l).w16(sp::CURSOR_GLYPH_HEIGHT, height);
        self.page(l).w16(sp::CURSOR_GLYPH_HOT_SPOT_X, hot_spot_x);
        self.page(l).w16(sp::CURSOR_GLYPH_HOT_SPOT_Y, hot_spot_y);
        Ok(())
    }

    /// The guest's cursor show byte: handed to the client's show handler on its queue, then
    /// stored in the page. `show` is the guest's raw `BOOL` byte (FIFO opcode 5): stored at the
    /// page's cursor show, compared raw against the guest's, and handed on raw.
    pub fn cursor_show(&self, show: u8) {
        let mut l = self.guest_display_mutex.lock();
        let keep = self.client_core();
        self.cursor_show_locked(&mut l, keep.as_ref(), show);
    }

    /// See [`GuestDisplay::cursor_show`].
    pub(super) fn cursor_show_locked(
        &self,
        l: &mut GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
        show: u8,
    ) {
        Self::assert_running(l);
        if let Some(c) = client
            && c.queue().is_some()
            && c.cursor_show_handler().is_some()
        {
            // The queue read again.
            let Some(queue) = c.queue() else {
                crate::fatal!(DeathKind::QueueGone)
            };
            // The raw byte.
            self.dispatch(&queue, move |_, client| {
                match client.and_then(|c| c.cursor_show_handler()) {
                    Some(h) => h(show),
                    None => crate::fatal!(DeathKind::ShowHandlerGone),
                }
            });
        }
        // The raw byte.
        self.page(l).w8(sp::CURSOR_SHOW, show);
    }

    /// The guest rang the port's doorbell: a changed show byte and every move go to the client's
    /// handlers on its queue, and the guest's live-resize byte latches the client's
    /// responsiveness.
    pub fn doorbell(&self) {
        let l = self.guest_display_mutex.lock();
        // A doorbell while paused dies (testing.md 7.4).
        Self::assert_running(&l);
        let enabled = self.page(&l).load(sp::ENABLED_INTERRUPTS);
        let raised = self.page(&l).load(sp::RAISED_INTERRUPTS);
        if enabled & raised != 0 {
            self.poke();
        }
        let keep = self.client_core();
        let Some(c) = keep.as_ref() else {
            return;
        };
        if c.queue().is_none() {
            return;
        }
        if c.cursor_show_handler().is_some() {
            // Raw bytes.
            let show = self.page(&l).r8(sp::GUEST_CURSOR_SHOW);
            if show != self.page(&l).r8(sp::CURSOR_SHOW) {
                self.page(&l).w8(sp::CURSOR_SHOW, show);
                // The queue read again.
                let Some(queue) = c.queue() else {
                    crate::fatal!(DeathKind::QueueGone)
                };
                self.dispatch(&queue, move |_, client| {
                    match client.and_then(|c| c.cursor_show_handler()) {
                        Some(h) => h(show),
                        None => crate::fatal!(DeathKind::ShowHandlerGone),
                    }
                });
            }
        }
        // Every doorbell.
        if c.cursor_move_handler().is_some() {
            // The queue read again.
            let Some(queue) = c.queue() else {
                crate::fatal!(DeathKind::QueueGone)
            };
            self.dispatch(&queue, move |_, client| {
                match client.and_then(|c| c.cursor_move_handler()) {
                    Some(h) => h(),
                    None => crate::fatal!(DeathKind::MoveHandlerGone),
                }
            });
        }
        // A one-way latch; the guest byte tested nonzero.
        if !self.compositor_supports_live_resize()
            && self.page(&l).r8(sp::COMPOSITOR_SUPPORTS_LIVE_RESIZE) != 0
        {
            self.unlocked
                .compositor_supports_live_resize
                .store(true, Ordering::SeqCst);
            c.update_responsiveness();
        }
    }

    /// The guest's cursor position: one seq_cst load of the page's word, x in the low half.
    /// Lock-free: the display asks under its guest display pointer lock. Dies while paused.
    pub fn cursor_position(&self) -> DisplayCoord {
        let position = self.page_unlocked().load(sp::CURSOR_POSITION);
        DisplayCoord {
            x: position.low_u16(),
            y: position.wrapping_shr(16).low_u16(),
        }
    }
}
