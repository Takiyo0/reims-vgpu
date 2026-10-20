//! The presents, the gamma table, the colour matrix, the properties and sleep.
//!
//! A present runs under `guest_display_mutex` and hands the frame to the client's present, which
//! takes the display's mutex inside: guest display, then display. It counts the present in the page
//! only when nothing raised. The completion it hands on raises the present interrupt when the frame
//! is done.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{DeathKind, GuestDisplay, GuestDisplayLocked, INTERRUPT_PRESENT, Page, RaiseKind};
use crate::display::page as sp;
use crate::display::{ColorMatrix, DisplayCore, GammaTableArgs};
use crate::fifo::wire::Records;
use crate::num::{ToUsize, Truncate};
use crate::platform::descriptor::TextureDescriptor as _;
use crate::platform::device::GpuDevice as _;
use crate::platform::os::surface::Surface as _;
use crate::platform::resource::Texture as _;
use crate::platform::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::resource::ResourceKind;
use crate::rt::{Held, Hold, pool};
use crate::seams::links::KernelTaskRef;

/// `'BGRA'`.
pub(super) const FOURCC_BGRA: u32 = 0x4247_5241;
/// `'RGBA'`.
pub(super) const FOURCC_RGBA: u32 = 0x5247_4241;
/// `'RGhA'`.
pub(super) const FOURCC_RGHA: u32 = 0x5247_6841;
/// `MTLPixelFormatRGBA8Unorm`.
pub(super) const MTL_PIXEL_FORMAT_RGBA8_UNORM: u64 = 70;
/// `MTLPixelFormatBGRA8Unorm`.
pub(super) const MTL_PIXEL_FORMAT_BGRA8_UNORM: u64 = 80;
/// `MTLPixelFormatRGBA16Float`.
pub(super) const MTL_PIXEL_FORMAT_RGBA16_FLOAT: u64 = 115;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: u64 = 1;
/// The gamma table's entry limit.
const GAMMA_ENTRY_LIMIT: u32 = 1024;
/// Property 0's word count (a colour matrix, `float[3][4]`).
const COLOR_MATRIX_WORDS: u32 = 12;

/// Whether the RGhA log was written: once per process. It only gates a log line (diagnostics,
/// testing.md 4.9).
pub(super) static RGHA_LOGGED: AtomicBool = AtomicBool::new(false);

/// A colour matrix in the page, `&colorMatrices[index]`: each element is read from the page when
/// asked for, so the display's late read (after its wait for frames in flight) sees what the
/// guest's page holds then. The page stays mapped while `guest_display_mutex` is held, which the
/// presents hold across the display's present.
#[derive(Debug, Clone, Copy)]
pub(super) struct PageColorMatrix {
    pub(super) page: Page,
    pub(super) index: u8,
}

impl PageColorMatrix {
    /// The matrix the display's present takes.
    fn as_dyn(&self) -> &dyn ColorMatrix {
        self
    }
}

impl ColorMatrix for PageColorMatrix {
    fn at(&self, row: u8, col: u8) -> f32 {
        color_matrix_access!(self.index, row, col, 4);
        self.page.rf32(sp::color_matrix(self.index, row, col))
    }

    /// One 64-bit load of the page (`ldp s, s`: the pair is 8-byte aligned, the matrices 16-byte
    /// aligned at `0xa10 + 48 * index`), so a guest's 64-bit store to it is seen whole or not at
    /// all. Little-endian: `matrix[row][col]` is the low half.
    fn pair(&self, row: u8, col: u8) -> (f32, f32) {
        color_matrix_access!(self.index, row, col, 8);
        let v = self.page.r64(sp::color_matrix(self.index, row, col));
        (
            f32::from_bits(v.low_u32()),
            f32::from_bits(v.wrapping_shr(32).low_u32()),
        )
    }
}

/// The client, when it presents: it has its queue (which the unplug clears) and a newFrame
/// handler.
fn presenting<P: Platform>(client: Option<&Arc<DisplayCore<P>>>) -> Option<&Arc<DisplayCore<P>>> {
    client.filter(|c| c.queue().is_some() && c.new_frame_event_handler().is_some())
}

/// The Metal pixel format and bytes per pixel the mapped present uses for an IOSurface format,
/// `None` for any other than BGRA, RGBA and RGhA.
fn mapped_pixel_format<P: Platform>(
    guest_display: &GuestDisplay<P>,
    format: u32,
) -> Option<(u64, u64)> {
    match format {
        FOURCC_BGRA => Some((MTL_PIXEL_FORMAT_BGRA8_UNORM, 4)),
        FOURCC_RGBA => Some((MTL_PIXEL_FORMAT_RGBA8_UNORM, 4)),
        FOURCC_RGHA => {
            // The device's features; a gone device reads through NULL.
            let Some(device) = guest_display.device.upgrade() else {
                crate::fatal!(DeathKind::RghaWithoutDevice)
            };
            if !device.features().load().supports_rgha_presents {
                // Once per process, and RGhA presents anyway.
                if !RGHA_LOGGED.load(Ordering::Relaxed) {
                    log::warn!(
                        "the guest presents RGhA, which its binary version does not support; \
                         presenting it anyway"
                    );
                    RGHA_LOGGED.store(true, Ordering::Relaxed);
                }
            }
            Some((MTL_PIXEL_FORMAT_RGBA16_FLOAT, 8))
        }
        _ => None,
    }
}

impl<P: Platform> GuestDisplay<P> {
    /// The presents' completion: it holds the guest display and raises the present interrupt in the
    /// page when it runs, poking the device when the guest enabled it.
    fn completion(&self) -> Box<dyn FnOnce() + Send> {
        let this = self.unlocked.this.upgrade();
        Box::new(move || {
            if let Some(guest_display) = this {
                guest_display
                    .page_unlocked()
                    .fetch_or(sp::RAISED_INTERRUPTS, INTERRUPT_PRESENT);
                if guest_display.page_unlocked().load(sp::ENABLED_INTERRUPTS) & INTERRUPT_PRESENT
                    != 0
                {
                    guest_display.poke();
                }
            }
        })
    }

    /// Stores the gamma table's record in the page; raises for more than 1024 entries or a
    /// table longer than its mapping.
    pub(super) fn store_gamma_table(
        &self,
        l: &GuestDisplayLocked<P>,
        gamma: GammaTableArgs,
    ) -> RaiseResult<()> {
        Self::assert_running(l);
        // `size_t`: cannot overflow.
        let length = u64::from(gamma.entry_count).wrapping_mul(4);
        if gamma.entry_count > GAMMA_ENTRY_LIMIT || length > gamma.mapped_length {
            crate::raise!(
                RaiseKind::GammaTable,
                format!(
                    "guest display on port {}: a gamma table of {} entries (at most {}) in {:#x} \
                     mapped bytes",
                    self.port, gamma.entry_count, GAMMA_ENTRY_LIMIT, gamma.mapped_length
                )
            );
        }
        self.page(l)
            .w64(sp::GAMMA_TABLE_VIRTUAL_OFFSET, gamma.virtual_offset);
        self.page(l)
            .w64(sp::GAMMA_TABLE_MAPPED_LENGTH, gamma.mapped_length);
        self.page(l)
            .w32(sp::GAMMA_TABLE_ENTRY_COUNT, gamma.entry_count);
        self.page(l).w32(sp::GAMMA_TABLE_SUM, gamma.sum);
        Ok(())
    }

    /// The colour matrix to present, or `None` when it is neither dirty nor forced. A dirty
    /// matrix flips the indices: the guest's write index becomes the read index. The answer is
    /// the page and the index, not a copy: a guest write before the display's late read is
    /// uploaded.
    pub(super) fn take_dirty_color_matrix(
        &self,
        l: &GuestDisplayLocked<P>,
    ) -> Option<PageColorMatrix> {
        Self::assert_running(l);
        let p = self.page(l);
        if p.r8(sp::COLOR_MATRIX_FORCE_READ) == 0 {
            if p.r8(sp::COLOR_MATRIX_DIRTY) == 0 {
                return None;
            }
            // One load of the write index serves both stores, so a guest write between the two
            // is overwritten and the page is left with write == !read.
            let w = p.r8(sp::COLOR_MATRIX_WRITE_INDEX);
            p.w8(sp::COLOR_MATRIX_READ_INDEX, w);
            #[cfg(test)]
            super::tests::between_color_matrix_index_stores(p.0);
            // Logical not: 0 -> 1, any other byte -> 0.
            p.w8(sp::COLOR_MATRIX_WRITE_INDEX, u8::from(w == 0));
            p.w8(sp::COLOR_MATRIX_DIRTY, 0);
        }
        // The read index reloaded.
        let index = p.r8(sp::COLOR_MATRIX_READ_INDEX);
        p.w8(sp::COLOR_MATRIX_FORCE_READ, 0);
        Some(PageColorMatrix { page: p, index })
    }

    /// Presents the kernel task's backing `surface`, with the gamma table when it has entries.
    pub fn present_surface(&self, surface: u32, gamma: GammaTableArgs) -> RaiseResult<()> {
        let mut l = self.guest_display_mutex.lock();
        let keep = self.client_core();
        self.present_surface_locked(&mut l, keep.as_ref(), surface, gamma)
    }

    /// See [`GuestDisplay::present_surface`]: a view of the backing's first texture in the
    /// surface's format goes to the client, then the page counts the present.
    ///
    /// The present runs in a pool (the prepared backing's command buffer, Metal's `+0`) and a
    /// holder: the kernel task, and what the backing lookup holds with no handle to a holder of
    /// its own, go at the holder's end, most recent first, then the pool pops, before the page
    /// counts the present. A raise leaves the pool pushed, and what the holder holds goes on
    /// with it to where the raise ends.
    fn present_surface_locked(
        &self,
        l: &mut GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
        surface: u32,
        gamma: GammaTableArgs,
    ) -> RaiseResult<()> {
        Self::assert_running(l);
        pool::around(&*self.platform, || {
            Held::raising(|held| {
                self.present_surface_held(Hold::of(held), held, l, client, surface, gamma)
            })
        })?;
        let p = self.page(l);
        p.bump64(sp::PRESENT_COUNT);
        self.page(l).w32(sp::PRESENTED_SURFACE, surface);
        self.page(l).w32(sp::FIELD_FA4, 0);
        Ok(())
    }

    /// The present of [`GuestDisplay::present_surface_locked`], in its pool and `held`.
    fn present_surface_held(
        &self,
        hold: Hold<'_>,
        held: &Held,
        l: &mut GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
        surface: u32,
        gamma: GammaTableArgs,
    ) -> RaiseResult<()> {
        let task = self.kernel_task(OnFault::Return, held)?;
        // Only a gone device answers no task.
        let Some(task) = task else {
            crate::fatal!(DeathKind::PresentWithoutKernelTask)
        };
        let backing = task.prepared_backing_for_id(hold, surface)?;
        // The backing's IOSurface and first texture; nothing for no backing.
        let (io_surface, texture) = match backing.as_deref() {
            None => (None, None),
            Some(r) => match &r.kind {
                ResourceKind::Backing(b) => {
                    let io = b.io_surface();
                    let Some(t) = b.textures().into_iter().next() else {
                        // Unreachable: a backing has at least one texture.
                        crate::raise!(
                            RaiseKind::BackingWithoutTexture,
                            "a texture backing with no texture"
                        );
                    };
                    (io, Some(t))
                }
                _ => {
                    crate::raise!(
                        RaiseKind::NotABacking,
                        format!(
                            "guest display on port {}: surface {surface} is a {}, not a texture \
                             backing",
                            self.port,
                            r.kind_name()
                        )
                    );
                }
            },
        };
        // No IOSurface: its pixel format reads through NULL.
        let Some(io) = io_surface else {
            crate::fatal!(DeathKind::PresentWithoutSurface)
        };
        io.sync_from_guest();
        let pixel_format = io.pixel_format();
        let mut mtl_format = 0u64;
        if pixel_format == FOURCC_BGRA {
            mtl_format = MTL_PIXEL_FORMAT_BGRA8_UNORM;
        }
        if pixel_format == FOURCC_RGBA {
            mtl_format = MTL_PIXEL_FORMAT_RGBA8_UNORM;
        }
        if mtl_format == 0 {
            crate::raise!(
                RaiseKind::PixelFormat,
                format!(
                    "guest display on port {}: pixel format {pixel_format:#x} is not presentable",
                    self.port
                )
            );
        }
        let requirements = io.protection_options();
        // The page's value, read twice (a guest overwrite counts).
        let p = self.page(l);
        if (p.r64(sp::PROTECTION_OPTIONS) | requirements) != p.r64(sp::PROTECTION_OPTIONS) {
            crate::raise!(
                RaiseKind::ProtectionRequirements,
                format!(
                    "guest display on port {}: the surface's protection requirements \
                     {requirements:#x} are not in the page's options {:#x}",
                    self.port,
                    self.page(l).r64(sp::PROTECTION_OPTIONS)
                )
            );
        }
        // No texture gives none.
        let view = match texture.as_ref() {
            Some(t) => t.new_texture_view(mtl_format)?,
            None => None,
        };
        let completion = self.completion();
        let presented = (|| -> RaiseResult<()> {
            if gamma.entry_count != 0 {
                self.store_gamma_table(l, gamma)?;
            }
            match presenting(client) {
                Some(c) => {
                    // The matrix is taken before the call; its floats are read inside it.
                    let matrix = self.take_dirty_color_matrix(l);
                    c.present_frame(
                        hold,
                        OnFault::Return,
                        view.as_ref().map(Handle::clone_ref),
                        pixel_format,
                        requirements,
                        Some(task.clone()),
                        gamma,
                        matrix.as_ref().map(PageColorMatrix::as_dyn),
                        completion,
                    )?;
                    // Bit 0.
                    if l.asleep & 1 == 0 {
                        c.signal_current_frame();
                    }
                }
                None => completion(),
            }
            Ok(())
        })();
        if let Err(raised) = presented {
            // The view goes on with the raise, to where it ends.
            return Err(match view {
                Some(v) => raised.defer(v),
                None => raised,
            });
        }
        drop(view);
        Ok(())
    }

    /// Presents the IOSurface the mapper service has for the mapping id `surface` (the whole
    /// u64), with the gamma table when it has entries.
    pub fn present_mapped_surface(&self, surface: u64, gamma: GammaTableArgs) -> RaiseResult<()> {
        let mut l = self.guest_display_mutex.lock();
        let keep = self.client_core();
        self.present_mapped_surface_locked(&mut l, keep.as_ref(), surface, gamma)
    }

    /// See [`GuestDisplay::present_mapped_surface`]. It runs in a pool and a holder, as
    /// [`GuestDisplay::present_surface_locked`] does.
    pub(super) fn present_mapped_surface_locked(
        &self,
        l: &mut GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
        surface: u64,
        gamma: GammaTableArgs,
    ) -> RaiseResult<()> {
        Self::assert_running(l);
        pool::around(&*self.platform, || {
            Held::raising(|held| {
                let hold = Hold::of(held);
                let task = self.kernel_task(OnFault::Return, held)?;
                // Dead with a device: the lookup raises instead.
                let Some(task) = task else {
                    crate::raise!(RaiseKind::NoKernelTask, "no kernel task");
                };
                // Retained; no mapper answers nothing.
                let io_surface = self
                    .device
                    .upgrade()
                    .and_then(|d| d.mapper_service())
                    .and_then(|m| m.retain_surface_with_mapping_id(surface));
                self.present_mapped_retained(hold, l, client, task, io_surface, gamma)
            })
        })?;
        self.presented_mapped_surface(l, surface);
        Ok(())
    }

    /// The mapped present from the retained surface on: a texture over the surface in its format
    /// goes to the client.
    pub(super) fn present_mapped_retained(
        &self,
        hold: Hold<'_>,
        l: &mut GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
        task: KernelTaskRef<P>,
        io_surface: Option<P::Surface>,
        gamma: GammaTableArgs,
    ) -> RaiseResult<()> {
        let Some(io) = io_surface else {
            crate::raise!(RaiseKind::MappingId, "no surface under the mapping id");
        };
        io.sync_from_guest();
        let mut texture: Option<P::Texture> = None;
        let presented = (|| -> RaiseResult<()> {
            let pixel_format = io.pixel_format();
            let width = io.width();
            let height = io.height();
            let Some((mtl_format, bytes_per_element)) = mapped_pixel_format(self, pixel_format)
            else {
                crate::raise!(
                    RaiseKind::PixelFormat,
                    format!(
                        "guest display on port {}: pixel format {pixel_format:#x} is not presentable",
                        self.port
                    )
                );
            };
            let surface_bytes_per_element = io.bytes_per_element();
            if surface_bytes_per_element != bytes_per_element {
                crate::raise!(
                    RaiseKind::BytesPerPixel,
                    format!(
                        "guest display on port {}: the surface has {surface_bytes_per_element} bytes \
                         per pixel, not {bytes_per_element}",
                        self.port
                    )
                );
            }
            let descriptor = self.platform.texture_2d_descriptor(
                OnFault::Return,
                mtl_format,
                width,
                height,
                false,
            )?;
            descriptor.set_usage(MTL_TEXTURE_USAGE_SHADER_READ);
            // Metal's own validation aborts inside this call for a surface it refuses
            // (testing.md 7.4; the metal crate makes the same call).
            texture = match self.device.upgrade() {
                Some(d) => d
                    .mtl_device_ref()
                    .new_texture_on_surface(&descriptor, Some(&io), 0)?,
                None => None,
            };
            crate::fatal_unless!(texture.is_some(), DeathKind::MappedTextureRefused);
            let completion = self.completion();
            if gamma.entry_count != 0 {
                self.store_gamma_table(l, gamma)?;
            }
            match presenting(client) {
                Some(c) => {
                    if l.guest_color_space_dirty {
                        l.guest_color_space_dirty = false;
                        c.set_guest_color_space_dirty(true);
                    }
                    // The arguments in order, then the matrix.
                    let format = io.pixel_format();
                    let requirements = io.protection_options();
                    let matrix = self.take_dirty_color_matrix(l);
                    c.present_frame(
                        hold,
                        OnFault::Return,
                        texture.as_ref().map(Handle::clone_ref),
                        format,
                        requirements,
                        Some(task.clone()),
                        gamma,
                        matrix.as_ref().map(PageColorMatrix::as_dyn),
                        completion,
                    )?;
                    // Bit 0.
                    if l.asleep & 1 == 0 {
                        c.signal_current_frame();
                    }
                }
                None => completion(),
            }
            Ok(())
        })();
        if let Err(raised) = presented {
            // The surface, then the texture, go on with the raise, to where it ends: released
            // there most recent first. Until then the surface's count stays up, so a mapper unmap
            // of it meanwhile is Busy.
            let raised = raised.defer(io);
            return Err(match texture {
                Some(t) => raised.defer(t),
                None => raised,
            });
        }
        // Most recent first.
        drop(texture);
        drop(io);
        Ok(())
    }

    /// Counts a mapped present in the page, after its pool and holder have ended.
    pub(super) fn presented_mapped_surface(&self, l: &GuestDisplayLocked<P>, surface: u64) {
        let p = self.page(l);
        p.bump64(sp::PRESENT_COUNT);
        self.page(l).w64(sp::PRESENTED_MAPPING_ID, surface);
        self.page(l).w32(sp::FIELD_FA4, 0);
    }

    /// Property 0, the colour matrix: twelve words into the matrix at the guest's write index,
    /// then dirty. Raises for another count or another property.
    pub fn set_property(&self, property: i32, value: Records<'_, u32>) -> RaiseResult<()> {
        let l = self.guest_display_mutex.lock();
        Self::assert_running(&l);
        match property {
            0 => {
                let p = self.page(&l);
                if value.count() != COLOR_MATRIX_WORDS {
                    crate::raise!(
                        RaiseKind::PropertyWordCount,
                        format!(
                            "guest display on port {}: a colour matrix of {} words, not \
                             {COLOR_MATRIX_WORDS}",
                            self.port,
                            value.count()
                        )
                    );
                }
                // 48 bytes at the guest's write index; a matrix that leaves the page dies before
                // any of them is written.
                let w = p.r8(sp::COLOR_MATRIX_WRITE_INDEX);
                color_matrix_access!(w, 0, 0, sp::COLOR_MATRIX_SIZE);
                let bytes = value.as_bytes();
                let n = sp::COLOR_MATRIX_SIZE.to_usize();
                p.at().copy_from(
                    sp::color_matrix(w, 0, 0),
                    bytes.get(..n).unwrap_or(bytes),
                    DeathKind::NullPage,
                );
                p.w8(sp::COLOR_MATRIX_DIRTY, 1);
                Ok(())
            }
            _ => {
                // The property as the guest wrote it, unsigned.
                crate::raise!(
                    RaiseKind::UnknownProperty,
                    format!(
                        "guest display on port {}: unknown property {}",
                        self.port,
                        property.cast_unsigned()
                    )
                );
            }
        }
    }

    /// The guest's sleep: the raw byte, stored in the page and kept for the presents. With a
    /// queue, a block on it logs the sleep, or logs the wake and signals the current frame.
    /// `asleep` is the guest's raw `BOOL` byte (FIFO opcode 9), stored and compared unnormalised.
    pub fn sleep_state(&self, asleep: u8) {
        let mut l = self.guest_display_mutex.lock();
        let keep = self.client_core();
        self.sleep_state_locked(&mut l, keep.as_ref(), asleep);
    }

    /// See [`GuestDisplay::sleep_state`].
    pub(super) fn sleep_state_locked(
        &self,
        l: &mut GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
        asleep: u8,
    ) {
        Self::assert_running(l);
        if let Some(c) = client
            && c.queue().is_some()
        {
            // The queue read again.
            let Some(queue) = c.queue() else {
                crate::fatal!(DeathKind::QueueGone)
            };
            // The client read again when the block runs; each handler is tested first.
            self.dispatch(&queue, move |guest_display, client| {
                // `== 1`: any other byte wakes.
                if asleep == 1 {
                    // No display has a sleep handler (no front end sets one): the client is let
                    // go, and the sleep is logged unhandled.
                    drop(client);
                    log::info!(
                        "guest display on port {}: sleep, unhandled",
                        guest_display.port
                    );
                } else {
                    let handled = |client: &Option<Arc<DisplayCore<P>>>| {
                        client
                            .as_ref()
                            .is_some_and(|c| c.new_frame_event_handler().is_some())
                    };
                    // The handler is read for the line whether or not it is written.
                    let how = if handled(&client) {
                        "handled"
                    } else {
                        "unhandled"
                    };
                    log::info!("guest display on port {}: wake, {how}", guest_display.port);
                    // The handler read again.
                    if handled(&client)
                        && let Some(c) = client
                    {
                        c.signal_current_frame();
                    }
                }
            });
        }
        l.asleep = asleep;
        // The raw byte.
        self.page(l).w8(sp::ASLEEP, asleep);
    }
}
