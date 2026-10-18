//! Textures: the guest's textures, paged between its pages and a Metal texture, and the kinds that
//! only name a Metal texture (views, mapper and shared textures, IOSurface backings).
//!
//! A paged texture ([`TextureResource`]) is its Metal texture and the guest pages it is paged
//! from ([`PagedTexture`]), laid out one of five ways ([`TextureLayout`]):
//! - one plane paged by blits ([`blit::SinglePlane`], object types 2 and 3);
//! - two planes paged by blits, a depth-stencil format's halves ([`blit::DualPlaneBlit`], type 12);
//! - two planes paged by the XR10 compute kernels ([`compute::DualPlaneCompute`], type 12 in
//!   formats 550 and 551);
//! - a shared backing of one or two planes, which other tasks open by handle
//!   ([`shared::SharedSingle`], [`shared::SharedDual`], types 17 and 18).
//!
//! Each layout's dimensions are checked against the Metal texture at creation
//! ([`dimension::validate_texture_dimension`], with [`pixel_format`]'s descriptions). The other
//! kinds are [`backing`] (task 0's IOSurface-backed planes), [`views`] (a view of a backing, and
//! the textures the deserializer makes), [`mapper`] (a texture over a mapper IOSurface) and
//! [`shared`]'s [`SharedTextureResource`](shared::SharedTextureResource) (another task's texture
//! over a shared backing).
//!
//! # Contract
//!
//! What callers and the guest observe, each held by the tests named (under each module's
//! `tests`):
//! - **Validation** checks each plane in a fixed order and logs one line for the first check that
//!   fails. Its widths are part of it: the allocation span is computed in 32 bits
//!   (`dimension::plane::allocation_span_u32`), a cube counts six slices
//!   (`dimension::plane::cube_six_slices`), mip ends must increase from 0 and each halving clamps
//!   to 1 (`dimension::normal::non_increasing`, `dimension::normal::halving_clamps`).
//! - **Guest pages are bound once per life**: a held transfer buffer is never rebound; only a
//!   discard drops it (`paged::binds_once`).
//! - **A discard makes the texture volatile and keeps host-valid**; the next prepare makes it
//!   non-volatile again, and pages it again only if Metal emptied it
//!   (`devolatilize::volatile_emptied_goes_host_invalid`).
//! - **Validity is written after encoding**, and a protected texture is marked valid without
//!   paging (`blit::single::prepare::guest_invalid_logs_protected_not_paged`,
//!   `blit::single::synchronize::protected`).
//! - **Blit paging order** is plane, slice, face, level, with fixed widths: slices and rows wrap in
//!   32 bits, offsets in 64 (`blit::single::paging::download_cube_array_order`,
//!   `blit::page::widths_wrap`).
//! - **XR10 paging** is one compute dispatch per direction over two plane views of the transfer
//!   buffer; the views are made again after a refusal, and one overwritten without a release is
//!   kept for good (`compute::ensure_physical::plane_one_fails_then_retry_leaks`).
//! - **Shared backings** prepare on a command buffer of their own, committed and waited for, in a
//!   pool a raise leaves open (`shared::prepare_now::raise_leaves_the_real_pool_pushed`).
//! - **Backings' plane copies** use a bytes-per-image computed in 32 bits
//!   (`backing::bytes_per_image_wraps_in_32_bits`).
//! - **Mapper textures** hand their surface back through their task at the last release, and
//!   keep the retain for good when the task is gone
//!   (`mapper::a_gone_task_leaks_the_surface_retain`).
//! - **Values kept for good** ([`crate::raise::kept`]): a raise out of a creation the guest can
//!   reach keeps the caller's texture, `MetalNumbering`
//!   (`blit::dual::new::mipped_unsupported_raises`), and so does an XR10 view made again (above);
//!   a mapper surface's retain, `SurfaceUseCount` (above). A raise no path from the manager
//!   reaches releases what the creation made and the caller's texture
//!   (`blit::single::new::short_length_raises_and_releases`).

pub mod backing;
pub mod blit;
pub mod compute;
pub mod dimension;
pub mod mapper;
pub mod pixel_format;
pub mod shared;
pub mod views;

use std::fmt;
use std::sync::Arc;

use zerocopy::{FromBytes, IntoBytes};

use crate::num::ToUsize;
use crate::platform::resource::{GpuResource, Texture};
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::resources::resource::AnyResource;
use crate::resources::resource::{
    PURGEABLE_EMPTY, PURGEABLE_NON_VOLATILE, PURGEABLE_VOLATILE, Resource, ResourceClass,
    ResourceKind, ResourceState, TaskRef,
};
use crate::resources::texture::blit::{DualPlaneBlit, SinglePlane};
use crate::resources::texture::compute::DualPlaneCompute;
use crate::resources::texture::dimension::{
    TextureDimension, TextureDimensionRecord, TextureMipDimension, TexturePagingInfo,
};
use crate::resources::texture::pixel_format::is_xr10;
use crate::resources::texture::shared::{SharedDual, SharedSingle};
use crate::resources::{Op, raise_with};
use crate::rt::Hold;
use crate::sync::Mutex;

/// The size of a dimension record's fixed part.
pub(crate) const DIMENSION_SIZE: u64 = 0x18;
/// The size of one mip's record.
pub(crate) const MIP_DIMENSION_SIZE: u64 = 0x24;

/// What the texture kinds raise, carried as [`crate::resources::RaiseKind::Texture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// The factory was asked for an object type it does not make (the detail is the type).
    UnexpectedObjectType,
    /// A dimension record shorter than its layout needs.
    DimensionLength,
    /// A mipped two-plane texture the task cannot page (a blit layout without multi-plane mipped
    /// blits, or any XR10 texture).
    MippedMultiPlane,
    /// A compressed two-plane texture.
    CompressedMultiPlane,
    /// An XR10 texture that is a cube, an array, compressed or multisampled.
    ComputeShape,
    /// An XR10 layout over a format that is not XR10.
    ComputeFormat,
    /// Metal refused an XR10 plane view over the transfer buffer.
    PlaneViewRefused,
    /// A backing record of no planes, or of more than three.
    BackingPlaneCount,
    /// A backing plane whose rows are shorter than its width, or that overruns the allocation.
    BackingPlaneInvalid,
    /// Metal refused a backing plane's texture over the IOSurface.
    BackingTextureRefused,
    /// A deserializer-made texture was asked to prepare.
    ViewPrepare,
    /// A deserializer-made texture was asked to synchronize.
    ViewSynchronize,
    /// A non-discardable deserializer-made texture was asked to discard.
    ViewDiscard,
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RaiseKind::UnexpectedObjectType => "texture.object_type",
            RaiseKind::DimensionLength => "texture.dimension_length",
            RaiseKind::MippedMultiPlane => "texture.multi_plane_mipped",
            RaiseKind::CompressedMultiPlane => "texture.multi_plane_compressed",
            RaiseKind::ComputeShape => "texture.xr10_shape",
            RaiseKind::ComputeFormat => "texture.xr10_format",
            RaiseKind::PlaneViewRefused => "texture.xr10_view_refused",
            RaiseKind::BackingPlaneCount => "backing.plane_count",
            RaiseKind::BackingPlaneInvalid => "backing.plane_invalid",
            RaiseKind::BackingTextureRefused => "backing.texture_refused",
            RaiseKind::ViewPrepare => "view.prepare",
            RaiseKind::ViewSynchronize => "view.synchronize",
            RaiseKind::ViewDiscard => "view.discard",
        })
    }
}

impl From<RaiseKind> for crate::resources::RaiseKind {
    fn from(k: RaiseKind) -> Self {
        crate::resources::RaiseKind::Texture(k)
    }
}

/// How the texture kinds die, carried as [`crate::resources::DeathKind::Texture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// The task went away before the guest pages were bound (a send through a NULL task).
    GoneTaskPages,
    /// The task went away before an XR10 texture read its kernels.
    GoneTaskKernels,
    /// An upload row shorter than its mip's width (an assert validation makes unreachable).
    ShortUploadRow,
    /// A shared texture or backing over a Metal texture that is not shareable (an assert the
    /// factories make unreachable).
    NotShareable,
}

impl DeathKind {
    /// Its dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::GoneTaskPages => "resources.texture.pages_task_gone",
            DeathKind::GoneTaskKernels => "resources.texture.kernels_task_gone",
            DeathKind::ShortUploadRow => "resources.texture.short_upload_row",
            DeathKind::NotShareable => "resources.texture.not_shareable",
        }
    }

    /// The kind of death: a NULL dereference, or an assert.
    pub const fn kind(self) -> crate::raise::death::DeathKind {
        match self {
            DeathKind::GoneTaskPages | DeathKind::GoneTaskKernels => {
                crate::raise::death::DeathKind::NullDeref
            }
            DeathKind::ShortUploadRow | DeathKind::NotShareable => {
                crate::raise::death::DeathKind::Assert
            }
        }
    }
}

impl From<DeathKind> for crate::raise::death::DeathId {
    fn from(k: DeathKind) -> Self {
        crate::resources::DeathKind::Texture(k).into()
    }
}

/// Raises `kind` (see [`crate::resources::raise`]).
#[track_caller]
pub(crate) fn raise<T>(kind: RaiseKind) -> RaiseResult<T> {
    crate::resources::raise(kind.into())
}

/// DIAG(texture-fingerprint): whether a texture's guest pages changed since its last prepare,
/// from 64 bytes of each 4 KiB page (a sample: a change elsewhere in a page goes unseen). Counts
/// are written every 500 prepares at debug.
#[allow(unsafe_code)]
fn fingerprint_diag(texture: usize, ranges: &[(crate::host::HostPtr, u64, u64)]) {
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static LAST: StdMutex<Option<HashMap<usize, (u64, u64)>>> = StdMutex::new(None);
    static SAME: AtomicU64 = AtomicU64::new(0);
    static CHANGED: AtomicU64 = AtomicU64::new(0);
    static FIRST: AtomicU64 = AtomicU64::new(0);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut bytes = 0u64;
    for &(host, _, size) in ranges {
        let mut at = 0u64;
        while at < size {
            let n = 64.min(size - at) as usize;
            // SAFETY: `host` maps `size` bytes of guest RAM (discontiguous_ranges); a read only.
            let sample = unsafe { std::slice::from_raw_parts((host.addr() + at as usize) as *const u8, n) };
            for chunk in sample.chunks(8) {
                let mut w = [0u8; 8];
                w[..chunk.len()].copy_from_slice(chunk);
                hash = (hash ^ u64::from_le_bytes(w)).wrapping_mul(0x100_0000_01b3);
            }
            at += 4096;
        }
        bytes += size;
    }
    let previous = {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        last.get_or_insert_with(HashMap::new).insert(texture, (hash, bytes))
    };
    match previous {
        Some((h, b)) if h == hash && b == bytes => SAME.fetch_add(1, Relaxed),
        Some(_) => CHANGED.fetch_add(1, Relaxed),
        None => FIRST.fetch_add(1, Relaxed),
    };
    let (same, changed, first) = (SAME.load(Relaxed), CHANGED.load(Relaxed), FIRST.load(Relaxed));
    if (same + changed + first) % 500 == 0 {
        log::debug!(
            target: "reims_vgpu_core::tasks::local::exec",
            "texture fingerprint: {same} unchanged, {changed} changed, {first} first of {} prepares",
            same + changed + first
        );
    }
}

/// A paged texture's Metal texture and the guest pages it is paged from, with the process it
/// makes descriptors in.
pub struct PagedTexture<P: Platform> {
    /// The process (reached through the creator's platform scope, kept for later paging).
    pub(crate) platform: Arc<P>,
    /// The task, unretained.
    pub(crate) task: TaskRef<P>,
    /// The guest pages' length.
    pub(crate) length: u64,
    /// The guest pages' first virtual page.
    pub(crate) page: u32,
    /// A PVRTC format (validation sets it).
    pub(crate) is_pvrtc: bool,
    /// The Metal texture. Declared before `transfer_buffer`: the last release lets the texture go
    /// first, then the guest pages.
    pub(crate) texture: P::Texture,
    /// The guest pages, bound at the first paging of a life and dropped by a discard.
    pub(crate) transfer_buffer: Mutex<Option<P::Buffer>>,
}

/// The five layouts of a paged texture.
pub enum TextureLayout<P: Platform> {
    /// One plane, paged by blits (object types 2 and 3).
    SinglePlane(SinglePlane),
    /// Two planes paged by blits (type 12).
    DualPlaneBlit(DualPlaneBlit),
    /// Two planes paged by the XR10 kernels (type 12 in 550 and 551).
    DualPlaneCompute(DualPlaneCompute<P>),
    /// A one-plane shared backing (type 17).
    SharedSingle(SharedSingle),
    /// A two-plane shared backing (type 18).
    SharedDual(SharedDual),
}

/// A paged texture: its layout and its Metal texture over guest pages.
///
/// `layout` is declared first so that the last release drops the layout's own handles (the XR10
/// views, 0 then 1) before the Metal texture and the guest pages.
pub struct TextureResource<P: Platform> {
    /// The layout.
    pub(crate) layout: TextureLayout<P>,
    /// The Metal texture and its guest pages.
    pub(crate) paged: PagedTexture<P>,
}

/// Plane 1 of a two-plane record: the record read from byte `at` (header, then mips as bytes) as
/// a record of `mip_count` mips. The resource manager lays plane 1's header and mips after plane
/// 0's `mipLevels` mips, zero-padded to a whole mip. Bytes past the record read as zero.
pub(crate) fn record_at(
    dimension: &TextureDimensionRecord,
    at: u64,
    mip_count: u64,
) -> TextureDimensionRecord {
    let mut src = dimension.header.as_bytes().to_vec();
    src.extend_from_slice(dimension.mips.as_bytes());
    let tail = src.get(at.to_usize()..).unwrap_or(&[]);
    let len = DIMENSION_SIZE.wrapping_add(mip_count.wrapping_mul(MIP_DIMENSION_SIZE));
    let mut buf = vec![0u8; len.to_usize()];
    for (d, s) in buf.iter_mut().zip(tail) {
        *d = *s;
    }
    let (h, rest) = buf
        .split_at_checked(DIMENSION_SIZE.to_usize())
        .unwrap_or((&[], &[]));
    TextureDimensionRecord {
        header: TextureDimension::read_from_bytes(h).unwrap_or_default(),
        mips: rest
            .as_chunks::<{ size_of::<TextureMipDimension>() }>()
            .0
            .iter()
            .map(|c| TextureMipDimension::read_from_bytes(c).unwrap_or_default())
            .collect(),
    }
}

/// The mip a validated record holds at `level`. Validation copied every level it accepted, so
/// the default is never taken.
pub(crate) fn mip_at(dimension: &TextureDimensionRecord, level: u64) -> TextureMipDimension {
    dimension
        .mips
        .get(level.to_usize())
        .copied()
        .unwrap_or_default()
}

/// A paged texture as a resource: guest-valid, as every layout starts
/// ([`ResourceClass::initial_guest_valid`]).
pub(crate) fn into_resource<P: Platform>(
    paged: PagedTexture<P>,
    layout: TextureLayout<P>,
) -> Arc<Resource<P>> {
    Resource::new(ResourceKind::Texture(TextureResource { layout, paged }))
}

/// Makes a paged texture of object type 2, 3 or 12: one plane for 2 and 3, two planes for 12
/// (paged by the XR10 kernels in 550 and 551, by blits otherwise). `None` when validation
/// refuses the record, or with no platform scope open.
///
/// `texture` is the caller's reference: released here, or on a raise kept for good or released
/// by the layout that raised (the caller never reaches its release). The resource holds its own.
pub fn create_texture_resource<P: Platform>(
    task: TaskRef<P>,
    object_type: u32,
    paging_info: &TexturePagingInfo,
    dimension: &TextureDimensionRecord,
    dimension_length: u64,
    texture: P::Texture,
) -> RaiseResult<Option<Arc<Resource<P>>>> {
    match object_type {
        2 | 3 => SinglePlane::new(
            task,
            object_type,
            paging_info,
            dimension,
            dimension_length,
            texture,
        ),
        12 if is_xr10(texture.pixel_format()) => {
            DualPlaneCompute::new(task, paging_info, dimension, dimension_length, texture)
        }
        12 => DualPlaneBlit::new(task, paging_info, dimension, dimension_length, texture),
        _ => {
            // The caller's texture is released. Unreachable from the manager (2, 3, 12).
            drop(texture);
            raise_with(
                RaiseKind::UnexpectedObjectType.into(),
                object_type.cast_signed(),
            )
        }
    }
}

/// Makes a shared backing of object type 17 (one plane) or 18 (two planes; `None` in 550 and
/// 551); `None` for any other type. `texture` is the caller's reference, as for
/// [`create_texture_resource`].
pub fn create_shared_texture_resource<P: Platform>(
    task: TaskRef<P>,
    object_type: u32,
    paging_info: &TexturePagingInfo,
    dimension: &TextureDimensionRecord,
    dimension_length: u64,
    texture: P::Texture,
) -> RaiseResult<Option<Arc<Resource<P>>>> {
    match object_type {
        17 => SharedSingle::new(task, paging_info, dimension, dimension_length, texture),
        18 if is_xr10(texture.pixel_format()) => Ok(None),
        18 => SharedDual::new(task, paging_info, dimension, dimension_length, texture),
        _ => Ok(None),
    }
}

impl<P: Platform> PagedTexture<P> {
    /// A paged texture over `texture` (retained), with no guest pages yet.
    pub(crate) fn new(platform: &Arc<P>, task: TaskRef<P>, texture: &P::Texture) -> Self {
        // DIAG(boot-tile): the first 400 paged textures' sizes, for the grey boot frame whose
        // tile x>=1024, y>=1024 never lands.
        {
            use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
            static N: AtomicU32 = AtomicU32::new(0);
            if N.fetch_add(1, Relaxed) < 400 {
                log::debug!(
                    target: "reims_vgpu_core::boot_tile",
                    "paged texture {}x{} format {}",
                    texture.width(),
                    texture.height(),
                    texture.pixel_format()
                );
            }
        }
        PagedTexture {
            platform: Arc::clone(platform),
            task,
            length: 0,
            page: 0,
            is_pvrtc: false,
            texture: texture.clone_ref(),
            transfer_buffer: Mutex::new(None),
        }
    }

    /// A texture a discard made volatile becomes non-volatile again; if Metal emptied it
    /// meanwhile, it is host-invalid.
    pub(crate) fn devolatilize(&self, state: &ResourceState) -> RaiseResult<()> {
        if state.is_volatile() {
            let previous = self.texture.set_purgeable_state(PURGEABLE_NON_VOLATILE)?;
            state.set_is_volatile(false);
            // Only an emptied texture is paged again.
            if previous == PURGEABLE_EMPTY {
                state.set_host_valid(false);
            }
        }
        Ok(())
    }

    /// Binds the guest pages unless bound: nothing walks them again while the transfer buffer is
    /// held. The lock is a leaf; the task is asked outside it.
    pub(crate) fn ensure_physical(&self, hold: Hold<'_>) -> RaiseResult<()> {
        // A held transfer buffer is never rebound.
        if self.transfer_buffer.lock().is_some() {
            return Ok(());
        }
        let Some(task) = self.task.upgrade() else {
            crate::fatal!(DeathKind::GoneTaskPages)
        };
        // The task raises rather than answer nothing.
        let buffer = task.new_transfer_buffer_for_virtual_page(hold, self.page, self.length)?;
        *self.transfer_buffer.lock() = Some(buffer);
        Ok(())
    }

    /// Drops the guest pages and makes the texture volatile; host-valid is kept.
    pub(crate) fn discard(&self, state: &ResourceState) -> RaiseResult<()> {
        drop(self.transfer_buffer.lock().take());
        self.texture.set_purgeable_state(PURGEABLE_VOLATILE)?;
        state.set_is_volatile(true);
        Ok(())
    }

    /// Guest to host: devolatilize, then page up unless host-valid. A protected texture is never
    /// paged, yet marked host-valid. `ensure` binds the pages, `page` pages them. The transfer
    /// buffer is brought up to date from the guest on the CPU first.
    pub(crate) fn prepare(
        &self,
        state: &ResourceState,
        ensure: impl FnOnce() -> RaiseResult<()>,
        page: impl FnOnce() -> RaiseResult<()>,
    ) -> RaiseResult<()> {
        self.prepare_filled(None, state, ensure, page)
    }

    /// [`Self::prepare`] for a prepare through the blit `encoder`: a transfer buffer the WHPX copy
    /// fallback made, whose guest pages are scattered in host memory, is filled by the GPU, its
    /// pages gathered into it in `encoder` before `page`'s copies ([`BlitEncoder::gather_host_ranges`]),
    /// in place of a copy on the CPU; one the platform cannot gather into is filled on the CPU.
    pub(crate) fn prepare_with_blit(
        &self,
        encoder: Option<&P::BlitEncoder>,
        state: &ResourceState,
        ensure: impl FnOnce() -> RaiseResult<()>,
        page: impl FnOnce() -> RaiseResult<()>,
    ) -> RaiseResult<()> {
        self.prepare_filled(encoder, state, ensure, page)
    }

    fn prepare_filled(
        &self,
        encoder: Option<&P::BlitEncoder>,
        state: &ResourceState,
        ensure: impl FnOnce() -> RaiseResult<()>,
        page: impl FnOnce() -> RaiseResult<()>,
    ) -> RaiseResult<()> {
        use crate::platform::blit::BlitEncoder as _;
        // DIAG(texture-prepare): each step's time, written where the prepare takes 2 ms or more.
        let diag =
            log::log_enabled!(target: "reims_vgpu_core::tasks::local::exec", log::Level::Debug);
        let t0 = std::time::Instant::now();
        let was_volatile = state.is_volatile();
        self.devolatilize(state)?;
        let t_devol = t0.elapsed();
        if state.host_valid() {
            return Ok(());
        }
        if !state.guest_valid() {
            log::warn!("preparing a texture whose guest copy is not valid");
        }
        let (mut t_ensure, mut t_sync) = (t_devol, t_devol);
        let mut gathered = false;
        if self.texture.protection_options() == 0 {
            // A transfer buffer bound here is filled below, every byte, so its copy fallback
            // leaves it unfilled (`with_lazy_fallback`).
            crate::memory::map::with_lazy_fallback(ensure)?;
            t_ensure = t0.elapsed();
            let tb = self.transfer_buffer.lock().as_ref().map(Handle::clone_ref);
            if let Some(tb) = tb
                && let Some(t) = self.task.upgrade()
                && let Some(map) = t.memory_map()
            {
                use crate::platform::resource::Buffer as _;
                if diag && let Some(ranges) = map.discontiguous_ranges(tb.contents(), tb.length()) {
                    fingerprint_diag(self as *const Self as usize, &ranges);
                }
                if let (Some(e), Some(ranges)) =
                    (encoder, map.discontiguous_ranges(tb.contents(), tb.length()))
                {
                    gathered = e.gather_host_ranges(&tb, &ranges)?;
                }
                // EXPERIMENT(skip-fill): REIMS_VGPU_DIAG_SKIP_FILL=1 leaves transfer buffers of
                // 1 MiB or more unfilled (wrong pixels), to measure what the copy costs a frame.
                static SKIP_FILL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let skip = *SKIP_FILL.get_or_init(|| {
                    std::env::var("REIMS_VGPU_DIAG_SKIP_FILL").is_ok_and(|v| v == "1")
                }) && tb.length() >= 1 << 20;
                if !gathered && !skip {
                    // Filled before the paging buffer commits, on a worker beside the other
                    // textures' fills where the exec collects them.
                    let (at, length) = (tb.contents().addr(), tb.length());
                    crate::memory::map::defer_sync(Box::new(move || {
                        map.sync_range_from_guest(crate::host::HostPtr::from_addr(at), length);
                    }));
                }
            }
            t_sync = t0.elapsed();
            crate::perfstats::TEXTURE_ENSURE.add(t_ensure - t_devol);
            crate::perfstats::TEXTURE_SYNC.add(t_sync - t_ensure);
            page()?;
        }
        state.set_host_valid(true);
        if diag && t0.elapsed().as_secs_f64() >= 0.002 {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            log::debug!(
                target: "reims_vgpu_core::tasks::local::exec",
                "texture prepare: devolatilize={:.2} (was volatile {was_volatile}) ensure={:.2} sync={:.2} page={:.2} ms; gathered {gathered}; {}x{} length {:#x}",
                ms(t_devol),
                ms(t_ensure - t_devol),
                ms(t_sync - t_ensure),
                ms(t0.elapsed() - t_sync),
                self.texture.width(),
                self.texture.height(),
                self.length
            );
        }
        Ok(())
    }

    /// Host to guest: page down unless guest-valid. Never devolatilizes. A protected texture is
    /// never paged, yet marked guest-valid.
    pub(crate) fn synchronize(
        &self,
        state: &ResourceState,
        ensure: impl FnOnce() -> RaiseResult<()>,
        page: impl FnOnce() -> RaiseResult<()>,
    ) -> RaiseResult<()> {
        if state.guest_valid() {
            return Ok(());
        }
        if !state.host_valid() {
            log::warn!("synchronizing a texture whose host copy is not valid");
        }
        if self.texture.protection_options() == 0 {
            ensure()?;
            page()?;
        }
        state.set_guest_valid(true);
        Ok(())
    }

    pub(crate) fn flush_to_guest(&self) {
        if let Some(tb) = self.transfer_buffer.lock().as_ref() {
            if let Some(t) = self.task.upgrade() {
                if let Some(map) = t.memory_map() {
                    use crate::platform::resource::Buffer as _;
                    map.sync_range_to_guest(tb.contents(), tb.length());
                }
            }
        }
    }

    /// The Metal texture.
    pub(crate) fn texture(&self) -> P::Texture {
        self.texture.clone_ref()
    }
}

impl<P: Platform> ResourceClass<P> for TextureResource<P> {
    fn initial_guest_valid(&self) -> bool {
        true
    }

    fn devolatilize_now(&self, this: &Resource<P>) -> RaiseResult<()> {
        self.paged.devolatilize(&this.state)
    }

    fn needs_compute_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        matches!(self.layout, TextureLayout::DualPlaneCompute(_))
    }

    fn prepare_in_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let (paged, state) = (&self.paged, &this.state);
        match &self.layout {
            TextureLayout::SinglePlane(l) => l.prepare(hold, paged, state, encoder),
            TextureLayout::DualPlaneBlit(l) => l.prepare(hold, paged, state, encoder),
            // The XR10 layout is paged by compute only; a blit prepare does nothing.
            TextureLayout::DualPlaneCompute(_) => Ok(()),
            TextureLayout::SharedSingle(l) => l.prepare(hold, paged, state, encoder),
            TextureLayout::SharedDual(l) => l.prepare(hold, paged, state, encoder),
        }
    }

    fn synchronize_in_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let (paged, state) = (&self.paged, &this.state);
        match &self.layout {
            TextureLayout::SinglePlane(l) => l.synchronize(hold, paged, state, encoder),
            TextureLayout::DualPlaneBlit(l) => l.synchronize(hold, paged, state, encoder),
            TextureLayout::DualPlaneCompute(_) => Ok(()),
            TextureLayout::SharedSingle(l) => l.synchronize(hold, paged, state, encoder),
            TextureLayout::SharedDual(l) => l.synchronize(hold, paged, state, encoder),
        }
    }

    fn prepare_in_compute_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        match &self.layout {
            TextureLayout::DualPlaneCompute(l) => {
                l.prepare(hold, &self.paged, &this.state, encoder)
            }
            _ => this.unrecognized(Op::PrepareCompute),
        }
    }

    fn synchronize_in_compute_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        match &self.layout {
            TextureLayout::DualPlaneCompute(l) => {
                l.synchronize(hold, &self.paged, &this.state, encoder)
            }
            _ => this.unrecognized(Op::SynchronizeCompute),
        }
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        match &self.layout {
            TextureLayout::DualPlaneCompute(l) => l.discard(&self.paged, &this.state),
            _ => self.paged.discard(&this.state),
        }
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(Some(AnyResource::Texture(self.paged.texture())))
    }

    fn get_texture(&self, this: &Resource<P>) -> RaiseResult<Option<P::Texture>> {
        let _ = this;
        Ok(Some(self.paged.texture()))
    }

    fn flush_to_guest(&self, _this: &Resource<P>) {
        self.paged.flush_to_guest();
    }
}

impl<P: Platform> TextureResource<P> {
    /// Binds the guest pages (and, for the XR10 layout, makes the plane views) unless bound.
    pub fn ensure_physical(&self, hold: Hold<'_>) -> RaiseResult<()> {
        match &self.layout {
            TextureLayout::DualPlaneCompute(l) => l.ensure_physical(hold, &self.paged),
            _ => self.paged.ensure_physical(hold),
        }
    }

    /// Whether this is a shared backing, which other tasks open by handle.
    pub fn is_shared_texture_backing(&self) -> bool {
        matches!(
            self.layout,
            TextureLayout::SharedSingle(_) | TextureLayout::SharedDual(_)
        )
    }

    /// Prepares a shared backing now, on a command buffer of its own from `queue`, committed and
    /// waited for. `state` is the resource's. Another layout raises
    /// [`crate::resources::RaiseKind::Unrecognized`] (its callers ask
    /// [`Self::is_shared_texture_backing`] first).
    pub fn prepare_shared_texture_backing(
        &self,
        state: &ResourceState,
        queue: Option<&P::CommandQueue>,
    ) -> RaiseResult<()> {
        let paged = &self.paged;
        match &self.layout {
            TextureLayout::SharedSingle(l) => {
                shared::prepare_now::<P>(queue, |hold, e| l.prepare(hold, paged, state, e))
            }
            TextureLayout::SharedDual(l) => {
                shared::prepare_now::<P>(queue, |hold, e| l.prepare(hold, paged, state, e))
            }
            _ => raise_with(
                crate::resources::RaiseKind::Unrecognized(Op::PrepareSharedBacking),
                "texture",
            ),
        }
    }

    /// A new handle other tasks open the shared backing by (+1), or `None`. Another layout
    /// raises, as for [`Self::prepare_shared_texture_backing`].
    pub fn new_shared_texture_handle(&self) -> RaiseResult<Option<P::SharedTextureHandle>> {
        match &self.layout {
            TextureLayout::SharedSingle(_) | TextureLayout::SharedDual(_) => {
                self.paged.texture.new_shared_texture_handle()
            }
            _ => raise_with(
                crate::resources::RaiseKind::Unrecognized(Op::NewSharedTextureHandle),
                "texture",
            ),
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared scaffolding for the texture kinds' tests.

    use super::*;

    use crate::guest::records::{GuestFeatures, SharedFeatures};
    use crate::resources::host::ResourceHost;
    use crate::testkit::impls::new_texture;
    use crate::testkit::objects::Model;
    use crate::testkit::objects::TexDesc;
    use crate::testkit::resource_host::FakeResourceHost;
    use crate::testkit::{Call, FakePlatform, Fk, kind};

    pub use crate::resources::resource::test_support::logged;

    /// The fake texture handle.
    pub type Tex = Fk<kind::Texture>;

    /// The line of a prepare of a paged texture whose guest copy is not valid.
    pub const PREPARE_INVALID: &str = "preparing a texture whose guest copy is not valid";

    /// The line of a synchronize of a paged texture whose host copy is not valid.
    pub const SYNCHRONIZE_INVALID: &str = "synchronizing a texture whose host copy is not valid";

    /// A fake process and a recorder task over it.
    pub struct Env {
        pub p: Arc<FakePlatform>,
        pub rec: Arc<FakeResourceHost>,
    }

    impl Env {
        pub fn new() -> Env {
            let p = Arc::new(FakePlatform::new());
            let rec = FakeResourceHost::new(p.clone());
            Env { p, rec }
        }

        /// The task as a resource holds it.
        pub fn task(&self) -> TaskRef<FakePlatform> {
            let t: Arc<dyn ResourceHost<FakePlatform>> = self.rec.clone();
            Arc::downgrade(&t)
        }

        /// Runs `f` inside this process's platform scope.
        pub fn run<T>(&self, f: impl FnOnce() -> T) -> T {
            crate::resources::resource::with_platform(&self.p, f)
        }

        /// Sets the task's features.
        pub fn features(&self, f: &GuestFeatures) {
            let s = SharedFeatures::new();
            s.store(f);
            self.rec.set_features(Arc::new(s));
        }

        /// A texture of this process's device (no call logged).
        pub fn tex(&self, d: TexDesc, shareable: bool) -> Tex {
            let t = new_texture(self.p.world(), d, None, 0, None, shareable);
            t.state().device = Some(self.p.device().erase());
            t
        }

        /// A blit encoder (no call logged).
        pub fn blit(&self) -> Fk<kind::BlitEncoder> {
            self.p.world().new_obj(Model::None)
        }

        /// A compute encoder (no call logged).
        pub fn compute(&self) -> Fk<kind::ComputeEncoder> {
            self.p.world().new_obj(Model::None)
        }

        /// The calls since `mark` whose order is observable, as selectors: getters, queries and
        /// descriptor setters Metal does not check are left out.
        pub fn sels(&self, mark: u64) -> Vec<&'static str> {
            self.p
                .call_log()
                .since(mark)
                .into_iter()
                .map(|c| c.sel)
                .collect()
        }

        /// The calls since `mark` with the given selector.
        pub fn calls(&self, mark: u64, sel: &str) -> Vec<Call> {
            self.p
                .call_log()
                .since(mark)
                .into_iter()
                .filter(|c| c.sel == sel)
                .collect()
        }
    }

    /// A texture descriptor.
    pub fn desc(pixel_format: u64, w: u64, h: u64, d: u64, levels: u64) -> TexDesc {
        TexDesc {
            pixel_format,
            width: w,
            height: h,
            depth: d,
            mipmap_level_count: levels,
            sample_count: 1,
            array_length: 1,
            ..TexDesc::default()
        }
    }

    /// A mip.
    pub fn mip(offset: u64, size: u64, bpr: u64, w: u32, h: u32, d: u32) -> TextureMipDimension {
        TextureMipDimension {
            offset,
            size,
            bytes_per_row: bpr,
            width: w,
            height: h,
            depth: d,
        }
    }

    /// A normal (1x1x1-block) plane header of `bb` bytes per pixel and `levels` mips.
    pub fn header(bps: u64, bb: u8, levels: u32) -> TextureDimension {
        let mut h = TextureDimension {
            array_length: 1,
            bytes_per_slice: bps,
            block_width: 1,
            block_height: 1,
            block_depth: 1,
            block_bytes: bb,
            ..TextureDimension::default()
        };
        h.set_mip_levels(levels);
        h
    }

    /// A one-plane record.
    pub fn rec1(h: TextureDimension, mips: Vec<TextureMipDimension>) -> TextureDimensionRecord {
        TextureDimensionRecord { header: h, mips }
    }

    /// A two-plane record laid out as the resource manager reads it: plane 0's header and mips,
    /// then plane 1's header and mips, as mips (zero-padded).
    pub fn rec2(
        h0: TextureDimension,
        m0: Vec<TextureMipDimension>,
        h1: TextureDimension,
        m1: Vec<TextureMipDimension>,
    ) -> TextureDimensionRecord {
        let mut rest = m0.as_bytes().to_vec();
        rest.extend_from_slice(h1.as_bytes());
        rest.extend_from_slice(m1.as_bytes());
        while !rest.len().is_multiple_of(0x24) {
            rest.push(0);
        }
        let mips = rest
            .as_chunks::<0x24>()
            .0
            .iter()
            .map(|c| TextureMipDimension::read_from_bytes(c).unwrap())
            .collect();
        TextureDimensionRecord { header: h0, mips }
    }

    /// A paging info.
    pub fn paging(page: u32, length: u64) -> TexturePagingInfo {
        TexturePagingInfo {
            page,
            length,
            ..TexturePagingInfo::default()
        }
    }

    /// The texture resource inside `r`.
    pub fn texture_of(r: &Resource<FakePlatform>) -> &TextureResource<FakePlatform> {
        match &r.kind {
            ResourceKind::Texture(t) => t,
            _ => panic!("not a texture resource"),
        }
    }

    /// A `u64` argument.
    pub fn u(c: &Call, i: usize) -> u64 {
        c.args[i].as_u64().unwrap()
    }

    /// Where a texture death is recorded.
    pub fn died(k: DeathKind) -> crate::raise::death::DeathSite {
        crate::raise::death::DeathId::from(k).site()
    }

    /// The texture raise a `Raise` carries, if it carries one.
    pub fn raised(t: &crate::raise::Raise) -> Option<RaiseKind> {
        match crate::resources::raised_kind(t) {
            Some(crate::resources::RaiseKind::Texture(k)) => Some(k),
            _ => None,
        }
    }

    /// A raise of this area, as a creation or a prepare hands it on.
    pub fn a_raise() -> crate::raise::Raise {
        crate::raise::Raise::new(crate::resources::RaiseKind::from(RaiseKind::ViewPrepare))
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::platform::Handle;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{Arg, FakePlatform, Step};

    type P = FakePlatform;

    /// A paged texture over `t`, at page 3 with 0x400 bytes.
    fn base(e: &Env, t: &Tex) -> PagedTexture<P> {
        let mut b = PagedTexture::new(&e.p, e.task(), t);
        b.page = 3;
        b.length = 0x400;
        b
    }

    #[test]
    fn new_retains_the_texture_and_holds_the_task() {
        let e = Env::new();
        let t = e.tex(desc(70, 4, 4, 1, 1), false);
        let m = e.p.call_log().mark();
        let b = PagedTexture::new(&e.p, e.task(), &t);
        assert_eq!(t.retain_count(), 2);
        assert!(b.texture.id() == t.id());
        assert!(b.transfer_buffer.lock().is_none());
        assert!(!b.is_pvrtc);
        assert!(b.task.upgrade().is_some());
        assert!(e.sels(m).is_empty());
        drop(b);
        assert_eq!(t.retain_count(), 1);
    }

    mod devolatilize {
        use super::*;

        #[test]
        fn not_volatile_sends_nothing() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            let s = ResourceState::default();
            s.set_host_valid(true);
            let m = e.p.call_log().mark();
            b.devolatilize(&s).unwrap();
            assert!(e.sels(m).is_empty());
            assert!(s.host_valid());
        }

        /// Kept contents stay host-valid
        #[test]
        fn volatile_kept_stays_host_valid() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            let s = ResourceState::default();
            s.set_host_valid(true);
            s.set_is_volatile(true);
            e.p.script().push("setPurgeableState:", Step::value(3u64));
            let m = e.p.call_log().mark();
            b.devolatilize(&s).unwrap();
            let c = e.calls(m, "setPurgeableState:");
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].args, vec![Arg::U64(2)]);
            assert!(!s.is_volatile());
            assert!(s.host_valid());
        }

        /// Emptied contents: host-invalid
        #[test]
        fn volatile_emptied_goes_host_invalid() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            let s = ResourceState::default();
            s.set_host_valid(true);
            s.set_is_volatile(true);
            e.p.script().push("setPurgeableState:", Step::value(4u64));
            b.devolatilize(&s).unwrap();
            assert!(!s.is_volatile());
            assert!(!s.host_valid());
        }

        /// A raise leaves it volatile
        #[test]
        fn raise_leaves_volatile() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            let s = ResourceState::default();
            s.set_is_volatile(true);
            e.p.script()
                .push("setPurgeableState:", Step::raise("NSGenericException", "x"));
            assert!(b.devolatilize(&s).is_err());
            assert!(s.is_volatile());
        }
    }

    mod paged {
        use super::*;

        #[test]
        fn binds_once() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            let m = e.p.call_log().mark();
            b.ensure_physical(Hold::NONE).unwrap();
            let c = e.calls(m, "new_buffer_for_virtual_page");
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].args, vec![Arg::U32(3), Arg::U64(0x400)]);
            assert!(b.transfer_buffer.lock().is_some());
            // A held transfer buffer is never rebound.
            b.ensure_physical(Hold::NONE).unwrap();
            assert_eq!(e.calls(m, "new_buffer_for_virtual_page").len(), 1);
        }

        /// The task's raise propagates, nothing bound
        #[test]
        fn task_raise_propagates() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            e.p.script().push(
                "new_buffer_for_virtual_page",
                Step::raise("NSGenericException", "Failed to get guest pointer"),
            );
            let hold = Hold::NONE;
            assert!(b.ensure_physical(hold).is_err());
            assert!(b.transfer_buffer.lock().is_none());
        }

        /// A torn-down task is a dangling receiver
        #[test]
        fn dead_task_dies() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let mut b = base(&e, &t);
            b.task = {
                let dead: Arc<dyn crate::resources::host::ResourceHost<P>> =
                    crate::testkit::resource_host::FakeResourceHost::new(e.p.clone());
                Arc::downgrade(&dead)
            };
            let got = catch_death(|| b.ensure_physical(Hold::NONE));
            assert_eq!(got.err(), Some(died(DeathKind::GoneTaskPages)));
        }
    }

    mod discard {
        use super::*;

        /// Volatile, host validity kept, pages dropped
        #[test]
        fn drops_pages_and_goes_volatile() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            b.ensure_physical(Hold::NONE).unwrap();
            let s = ResourceState::default();
            s.set_host_valid(true);
            let m = e.p.call_log().mark();
            b.discard(&s).unwrap();
            assert!(b.transfer_buffer.lock().is_none());
            let c = e.calls(m, "setPurgeableState:");
            assert_eq!(c[0].args, vec![Arg::U64(3)]);
            assert!(s.is_volatile());
            assert!(s.host_valid());
            // A discard rebinds at the next ensurePhysical.
            b.ensure_physical(Hold::NONE).unwrap();
            assert_eq!(e.calls(m, "new_buffer_for_virtual_page").len(), 1);
        }

        /// A raise after the release: pages gone, not volatile
        #[test]
        fn raise_after_release() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            b.ensure_physical(Hold::NONE).unwrap();
            let s = ResourceState::default();
            e.p.script()
                .push("setPurgeableState:", Step::raise("NSGenericException", "x"));
            assert!(b.discard(&s).is_err());
            assert!(b.transfer_buffer.lock().is_none());
            assert!(!s.is_volatile());
        }
    }

    #[test]
    fn getters_and_last_release() {
        let e = Env::new();
        let t = e.tex(desc(70, 4, 4, 1, 1), false);
        let b = base(&e, &t);
        b.ensure_physical(Hold::NONE).unwrap();
        let buf = b.transfer_buffer.lock().as_ref().unwrap().clone_ref();
        assert_eq!(b.texture().id(), t.id());
        assert_eq!(buf.retain_count(), 2);
        drop(b);
        assert_eq!(t.retain_count(), 1);
        assert_eq!(buf.retain_count(), 1);
    }

    #[test]
    fn a_single_plane_texture_is_blit_paged_and_not_shared() {
        let e = Env::new();
        let t = e.tex(desc(70, 4, 4, 1, 1), false);
        let r = e
            .run(|| {
                create_texture_resource(
                    e.task(),
                    2,
                    &paging(0, 0x40),
                    &rec1(header(0x40, 4, 1), vec![mip(0, 0x40, 16, 4, 4, 1)]),
                    0x3c,
                    t,
                )
            })
            .unwrap()
            .unwrap();
        let tr = texture_of(&r);
        assert!(!r.needs_compute_paging());
        assert!(!tr.is_shared_texture_backing());
        let id = tr.paged.texture.id();
        assert_eq!(r.get_texture().unwrap().unwrap().id(), id);
        match r.get_resource().unwrap().unwrap() {
            AnyResource::Texture(x) => assert_eq!(x.id(), id),
            _ => panic!("not a texture"),
        }
    }

    mod create {
        use super::*;
        use crate::guest::records::GuestFeatures;

        #[test]
        fn types_two_and_three_are_single_plane() {
            for ty in [2u32, 3] {
                let e = Env::new();
                let t = e.tex(desc(70, 4, 4, 1, 1), false);
                let r = e
                    .run(|| {
                        create_texture_resource(
                            e.task(),
                            ty,
                            &paging(0, 0x40),
                            &rec1(header(0x40, 4, 1), vec![mip(0, 0x40, 16, 4, 4, 1)]),
                            0x3c,
                            t.clone_ref(),
                        )
                    })
                    .unwrap()
                    .unwrap();
                assert!(matches!(
                    texture_of(&r).layout,
                    TextureLayout::SinglePlane(_)
                ));
                // The caller's reference is gone; the resource holds one.
                assert_eq!(t.retain_count(), 2);
            }
        }

        fn d32s8_two_plane(e: &Env) -> (Tex, TextureDimensionRecord) {
            e.features(&GuestFeatures {
                dual_plane_textures: true,
                s8_byte_count_for_d32s8: 1,
                ..GuestFeatures::default()
            });
            let t = e.tex(desc(260, 4, 4, 1, 1), false);
            let r = rec2(
                header(0x40, 4, 1),
                vec![mip(0, 0x40, 16, 4, 4, 1)],
                header(0, 1, 1),
                vec![mip(0x40, 0x10, 4, 4, 4, 1)],
            );
            (t, r)
        }

        #[test]
        fn type_twelve_blit() {
            let e = Env::new();
            let (t, r) = d32s8_two_plane(&e);
            let res = e
                .run(|| create_texture_resource(e.task(), 12, &paging(0, 0x80), &r, 0x78, t))
                .unwrap()
                .unwrap();
            assert!(matches!(
                texture_of(&res).layout,
                TextureLayout::DualPlaneBlit(_)
            ));
        }

        #[test]
        fn type_twelve_550_551_compute() {
            for pf in [550u64, 551] {
                let e = Env::new();
                let t = e.tex(desc(pf, 4, 4, 1, 1), false);
                let r = rec2(
                    header(0, 4, 1),
                    vec![mip(0, 0x40, 16, 4, 4, 1)],
                    header(0, 4, 1),
                    vec![mip(0x40, 0x40, 16, 4, 4, 1)],
                );
                let res = e
                    .run(|| create_texture_resource(e.task(), 12, &paging(0, 0x80), &r, 0x78, t))
                    .unwrap()
                    .unwrap();
                assert!(matches!(
                    texture_of(&res).layout,
                    TextureLayout::DualPlaneCompute(_)
                ));
                assert!(res.needs_compute_paging());
            }
        }

        /// Other types raise, releasing the caller's texture
        #[test]
        fn other_type_raises_and_releases() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let m = e.p.call_log().mark();
            let err = create_texture_resource(
                e.task(),
                5,
                &paging(0, 0x40),
                &rec1(header(0x40, 4, 1), vec![]),
                0x18,
                t.clone_ref(),
            )
            .err()
            .unwrap();
            assert_eq!(raised(&err), Some(RaiseKind::UnexpectedObjectType));
            assert_eq!(err.detail().unwrap_or_default(), "5");
            assert!(e.sels(m).is_empty());
            assert_eq!(t.retain_count(), 1);
        }

        /// The raise names where it is raised
        #[test]
        fn other_type_raise_names_where_it_is_raised() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let task = e.task();
            let got = e.run(|| {
                create_texture_resource(
                    task,
                    0,
                    &paging(0, 0x40),
                    &rec1(header(0x40, 4, 1), vec![]),
                    0x18,
                    t.clone_ref(),
                )
            });
            let d = got.map(|_| ()).unwrap_err().site();
            assert_eq!(d.kind, crate::raise::death::DeathKind::UnhandledRaise);
            assert!(d.file.ends_with("texture.rs"), "{d:?}");
        }
    }

    mod create_shared {
        use super::*;
        use crate::guest::records::GuestFeatures;

        #[test]
        fn seventeen_single() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), true);
            let r = e
                .run(|| {
                    create_shared_texture_resource(
                        e.task(),
                        17,
                        &paging(0, 0x40),
                        &rec1(header(0x40, 4, 1), vec![mip(0, 0x40, 16, 4, 4, 1)]),
                        0x3c,
                        t,
                    )
                })
                .unwrap()
                .unwrap();
            let tr = texture_of(&r);
            assert!(matches!(tr.layout, TextureLayout::SharedSingle(_)));
            assert!(tr.is_shared_texture_backing());
        }

        #[test]
        fn eighteen_dual() {
            let e = Env::new();
            e.features(&GuestFeatures {
                dual_plane_textures: true,
                s8_byte_count_for_d32s8: 1,
                ..GuestFeatures::default()
            });
            let t = e.tex(desc(260, 4, 4, 1, 1), true);
            let r = rec2(
                header(0x40, 4, 1),
                vec![mip(0, 0x40, 16, 4, 4, 1)],
                header(0, 1, 1),
                vec![mip(0x40, 0x10, 4, 4, 4, 1)],
            );
            let res = e
                .run(|| create_shared_texture_resource(e.task(), 18, &paging(0, 0x80), &r, 0x78, t))
                .unwrap()
                .unwrap();
            assert!(matches!(
                texture_of(&res).layout,
                TextureLayout::SharedDual(_)
            ));
        }

        /// 18 with 550/551 is nil
        #[test]
        fn eighteen_550_551_nil() {
            for pf in [550u64, 551] {
                let e = Env::new();
                let t = e.tex(desc(pf, 4, 4, 1, 1), true);
                let m = e.p.call_log().mark();
                let r = e
                    .run(|| {
                        create_shared_texture_resource(
                            e.task(),
                            18,
                            &paging(0, 0x80),
                            &rec1(header(0x40, 4, 1), vec![]),
                            0x78,
                            t.clone_ref(),
                        )
                    })
                    .unwrap();
                assert!(r.is_none());
                assert!(e.p.call_log().since(m).is_empty());
                assert_eq!(t.retain_count(), 1);
            }
        }

        /// Other types are nil, nothing sent
        #[test]
        fn other_nil() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), true);
            let m = e.p.call_log().mark();
            for ty in [0u32, 2, 16, 19] {
                let r = e
                    .run(|| {
                        create_shared_texture_resource(
                            e.task(),
                            ty,
                            &paging(0, 0x40),
                            &rec1(header(0x40, 4, 1), vec![]),
                            0x3c,
                            t.clone_ref(),
                        )
                    })
                    .unwrap();
                assert!(r.is_none());
            }
            assert!(e.sels(m).is_empty());
            assert_eq!(t.retain_count(), 1);
        }
    }

    mod record_at_reads_plane_one {
        use super::*;

        /// Plane 1 sits after plane 0's mips; missing bytes read as zero
        #[test]
        fn reads_after_plane_zero() {
            let h1 = header(0x10, 1, 2);
            let r = rec2(
                header(0x40, 4, 2),
                vec![mip(0, 1, 2, 3, 4, 5), mip(6, 7, 8, 9, 10, 11)],
                h1,
                vec![mip(12, 13, 14, 15, 16, 17), mip(18, 19, 20, 21, 22, 23)],
            );
            let p1 = record_at(&r, 0x18 + 2 * 0x24, 2);
            assert_eq!(p1.header, h1);
            assert_eq!(
                p1.mips,
                vec![mip(12, 13, 14, 15, 16, 17), mip(18, 19, 20, 21, 22, 23)]
            );
            let past = record_at(&r, 0x1000, 1);
            assert_eq!(past.header, TextureDimension::default());
            assert_eq!(past.mips, vec![TextureMipDimension::default()]);
        }
    }

    mod dispatch {
        use super::*;

        /// A blit-paged layout does not answer the compute paging operations, nor the shared
        /// backing's
        #[test]
        fn compute_operations_are_unrecognized_on_blit_layouts() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let r = e
                .run(|| {
                    create_texture_resource(
                        e.task(),
                        2,
                        &paging(0, 0x40),
                        &rec1(header(0x40, 4, 1), vec![mip(0, 0x40, 16, 4, 4, 1)]),
                        0x3c,
                        t,
                    )
                })
                .unwrap()
                .unwrap();
            let tr = texture_of(&r);
            let hold = Hold::NONE;
            let unrecognized = |op| Some(crate::resources::RaiseKind::Unrecognized(op));
            let m = e.p.call_log().mark();
            let err = r.prepare_in_compute_encoder(hold, None).err().unwrap();
            assert_eq!(
                crate::resources::raised_kind(&err),
                unrecognized(Op::PrepareCompute)
            );
            let err = r.synchronize_in_compute_encoder(hold, None).err().unwrap();
            assert_eq!(
                crate::resources::raised_kind(&err),
                unrecognized(Op::SynchronizeCompute)
            );
            let err = tr
                .prepare_shared_texture_backing(&r.state, None)
                .err()
                .unwrap();
            assert_eq!(
                crate::resources::raised_kind(&err),
                unrecognized(Op::PrepareSharedBacking)
            );
            let err = tr.new_shared_texture_handle().err().unwrap();
            assert_eq!(
                crate::resources::raised_kind(&err),
                unrecognized(Op::NewSharedTextureHandle)
            );
            assert!(e.sels(m).is_empty());
            // The initial validity: guest-valid, host-invalid, not volatile.
            assert!(r.state.guest_valid());
            assert!(!r.state.host_valid());
        }
    }
}
