//! The textures' creators: each reads its object record and its serialized command, has Metal
//! make the texture (at the index the guest's placement slot asks for), checks where Metal put
//! it, and has the resources area make the resource over it. [`super::create`] dispatches to
//! them by object type:
//! - 2, 3, 12: a paged texture over a descriptor ([`create_normal_texture`]);
//! - 8, 9: a texture the deserializer makes: a memoryless texture, a view of another texture, a
//!   texture over a buffer or in a heap ([`create_serializer_texture`]);
//! - 11: a texture over a mapper's IOSurface ([`create_mapper_ref_texture`]);
//! - 16: another task's shared backing, by handle ([`create_shared_texture`]);
//! - 17, 18: a shared backing ([`create_shared_texture_backing`]);
//! - 5: a texture over task 0's IOSurface backing ([`create_backing_ref_texture`]);
//! - 4: that backing ([`create_backing`]).
//!
//! # Contract
//!
//! What the guest observes, each held by the tests named (under `tests`):
//! - **Placement.** With a placement slot whose first word (the index) is set, the descriptor,
//!   view or shared handle is made at that index; the texture's GPU resource ID is then checked
//!   against the slot's, and a mismatch is logged and refused with the slot left as it is. A
//!   made texture's index and ID are written back, index first (`normal_placement`). A view is
//!   made at the index it tested, one load (`views_place_at_the_index_they_tested`); the check
//!   loads the index again and the slot's ID once, after Metal's answer, and the line shows that
//!   ID (`misplaced_logs_the_resource_id_it_compared`, `every_maker_logs_a_misplaced_texture`). A
//!   shared texture's handle takes the index unchecked (`shared_texture_placement`). Mapper and
//!   backing-ref textures are not written back (`iosurface_refusals`).
//! - **Nil is not always refused**: a memoryless texture and a view are not checked for nil and
//!   reach their resource over nil; a placed nil texture writes back zeros
//!   (`serializer_memoryless`, `views_bases_and_nil_views`, `manager_a_nil_memoryless_texture`,
//!   `manager_a_nil_view`). A base of reference 0 is nil, and makes no child
//!   (`views_bases_and_nil_views`).
//! - **IOSurface validation** checks a non-planar surface only, in a fixed order, logging one
//!   line for the first check that fails; only a texture buffer is refused by type
//!   (`iosurface_validation_refusals`, `iosurface_validation_quirks`,
//!   `iosurface_validation_skips_planar_surfaces`).
//! - **Mapper surfaces** are retained before the texture is made and handed back on every
//!   refusal (`mapper_ref_refusals_release_the_surface`); a raise keeps the retain for good
//!   (`Kept::SurfaceUseCount`; `mapper_ref_raise_leaks_the_surface`).
//! - **A shared texture's handle** goes with the call, whatever the outcome
//!   (`shared_texture_handle_goes_with_the_call`).
//! - **A rotated IOSurface texture** whose rotation Metal did not keep is a death
//!   ([`DeathKind::RotationMismatch`]; `iosurface_rotation_mismatch_dies`).
//! - **Records.** The backing record is read as 0x14 bytes (`backing`); a dimension record's
//!   length is `(mips * 0x24 + 0x18) * planes` (`normal_dual_plane`).

#[cfg(test)]
mod tests;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::deserializer::texture_descriptor::{
    SerializedTextureDescriptor, SerializedTextureDescriptor2,
};
use crate::guest::GuestPtr;
use crate::guest::records::{GuestFeatures, ObjectEntry};
use crate::memory::cursor::Cursor;
use crate::num::{ToU64, ToUsize, Truncate};
use crate::platform::descriptor::TextureDescriptor;
use crate::platform::device::GpuDevice;
use crate::platform::os::surface::Surface;
use crate::platform::resource::{Buffer, GpuResource, Heap, Texture};
use crate::platform::types::{NsRange, Swizzle, TextureLevelQuery};
use crate::platform::values::TextureDesc;
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::manager::create::{CreateEnv, refuse};
use crate::resources::manager::records::{
    OperationHeader, placement_field0, placement_resource_id, set_placement_field0,
    set_placement_resource_id,
};
use crate::resources::manager::{DeathKind, ResourceManager};
use crate::resources::texture::backing::{BackingDescriptor, BackingPlane};
use crate::resources::texture::dimension::{
    TextureDimension, TextureDimensionRecord, TextureMipDimension, TexturePagingInfo,
};
use crate::resources::texture::{DIMENSION_SIZE, MIP_DIMENSION_SIZE};
use crate::rt::Hold;

/// `MTLStorageModeMemoryless`.
const STORAGE_MODE_MEMORYLESS: u64 = 3;
/// `MTLTextureType2DArray`.
const TEXTURE_TYPE_2D_ARRAY: u64 = 3;
/// `MTLTextureTypeTextureBuffer`.
const TEXTURE_TYPE_TEXTURE_BUFFER: u64 = 9;
/// `MTLTextureSwizzleOne`: what a channel Metal does not have reads as.
const SWIZZLE_ONE: u8 = 1;
/// The first swizzle value Metal does not have (`MTLTextureSwizzleAlpha` is 5).
const SWIZZLE_LIMIT: u8 = 6;
/// A usage whose low three bits are this is read-only (`MTLTextureUsageShaderRead`).
const USAGE_SHADER_READ: u64 = 1;

// ---- guest records

/// A paged texture's record (packed, 12 bytes): its guest pages' length and first page.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct ObjectTexture {
    /// Name is ours: it becomes the paging info's `length`.
    length: u64,
    /// Name is ours: it becomes the paging info's `page`.
    page: u32,
}

/// A mapper texture's record: the mapping whose surface it is made over.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct ObjectMapperRefTexture {
    mapping_id: u64,
}

/// A shared texture's record: the shared backing it opens.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct ObjectSharedTexture {
    shared_texture_backing_id: u32,
}

/// A shared backing's record (packed, 12 bytes): as a paged texture's.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct ObjectSharedTextureBacking {
    /// Name is ours: it becomes the paging info's `length`.
    length: u64,
    /// Name is ours: it becomes the paging info's `page`.
    page: u32,
}

/// A backing-ref texture's record: the backing's IOSurface id and its task (only 0 is taken).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct ObjectRefTexture {
    backing_id: u32,
    task_id: u32,
}

/// A backing's record (`pack(4)`: 0x14 bytes), then its planes.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct ObjectBacking {
    length: u64,
    page: u32,
    pixel_format: u32,
    plane_count: u8,
    field_11: [u8; 3],
}

/// A texture over a descriptor, first form (`pack(4)`: 0x24 bytes); a shared backing's too.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpWithDescriptor {
    field_00: u32,
    descriptor: SerializedTextureDescriptor,
}

/// A texture over a descriptor, second form (0x2c bytes); a shared backing's too.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpWithDescriptor2 {
    field_00: u32,
    descriptor: SerializedTextureDescriptor2,
    field_2b: u8,
}

/// An `NSRange` inside a `pack(4)` command.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RangeRecord {
    location: u64,
    length: u64,
}

impl RangeRecord {
    fn ns(self) -> NsRange {
        NsRange::new(self.location, self.length)
    }
}

/// A view in another pixel format (0xc bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpViewWithPixelFormat {
    field_00: u32,
    base_texture_reference: u32,
    pixel_format: u16,
    field_0a: [u8; 2],
}

/// A view of other levels and slices, in another type (0x2c bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpViewWithTextureType {
    field_00: u32,
    base_texture_reference: u32,
    pixel_format: u16,
    texture_type: u16,
    levels: RangeRecord,
    slices: RangeRecord,
}

/// As [`OpViewWithTextureType`], swizzled (0x30 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpViewWithSwizzle {
    field_00: u32,
    base_texture_reference: u32,
    pixel_format: u16,
    texture_type: u16,
    levels: RangeRecord,
    slices: RangeRecord,
    swizzle: [u8; 4],
}

/// A texture in a heap, first form (0x34 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpWithHeap {
    field_00: u32,
    heap_reference: u32,
    descriptor: SerializedTextureDescriptor,
    field_28: u32,
    offset: u64,
}

/// A texture in a heap, second form (0x3c bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpWithHeap2 {
    field_00: u32,
    heap_reference: u32,
    descriptor: SerializedTextureDescriptor2,
    field_2f: u8,
    field_30: u32,
    offset: u64,
}

/// A texture over an IOSurface plane (0x28 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpIoSurfaceTexture {
    field_00: u32,
    descriptor: SerializedTextureDescriptor,
    plane: u16,
    field_26: [u8; 2],
}

/// A rotated texture over an IOSurface plane, first form (0x28 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpIoSurfaceTextureWithRotation {
    field_00: u32,
    descriptor: SerializedTextureDescriptor,
    plane: u16,
    rotation: u16,
}

/// A rotated texture over an IOSurface plane, second form (0x30 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct OpIoSurfaceTextureWithRotation2 {
    field_00: u32,
    descriptor: SerializedTextureDescriptor2,
    field_2b: u8,
    plane: u16,
    rotation: u16,
}

/// A texture over a buffer, first form (natural alignment: 0x38 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct OpViewWithBuffer {
    field_00: u32,
    buffer_reference: u32,
    offset: u64,
    bytes_per_row: u64,
    descriptor: SerializedTextureDescriptor,
}

/// A texture over a buffer, second form (0x40 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct OpViewWithBuffer2 {
    field_00: u32,
    buffer_reference: u32,
    offset: u64,
    bytes_per_row: u64,
    descriptor: SerializedTextureDescriptor2,
    field_3f: u8,
}

const _: () = {
    assert!(size_of::<ObjectTexture>() == 12);
    assert!(size_of::<ObjectMapperRefTexture>() == 8);
    assert!(size_of::<ObjectSharedTexture>() == 4);
    assert!(size_of::<ObjectSharedTextureBacking>() == 12);
    assert!(size_of::<ObjectRefTexture>() == 8);
    assert!(size_of::<ObjectBacking>() == 0x14);
    assert!(size_of::<OpWithDescriptor>() == 0x24);
    assert!(size_of::<OpWithDescriptor2>() == 0x2c);
    assert!(size_of::<OpViewWithPixelFormat>() == 0xc);
    assert!(size_of::<OpViewWithTextureType>() == 0x2c);
    assert!(size_of::<OpViewWithSwizzle>() == 0x30);
    assert!(size_of::<OpWithHeap>() == 0x34);
    assert!(size_of::<OpWithHeap2>() == 0x3c);
    assert!(size_of::<OpIoSurfaceTexture>() == 0x28);
    assert!(size_of::<OpIoSurfaceTextureWithRotation>() == 0x28);
    assert!(size_of::<OpIoSurfaceTextureWithRotation2>() == 0x30);
    assert!(size_of::<OpViewWithBuffer>() == 0x38);
    assert!(size_of::<OpViewWithBuffer2>() == 0x40);
    assert!(crate::num::const_u64(size_of::<TextureMipDimension>()) == MIP_DIMENSION_SIZE);
    assert!(crate::num::const_u64(size_of::<TextureDimension>()) == DIMENSION_SIZE);
};

/// Which of a command's two forms: the first carries a [`SerializedTextureDescriptor`], the
/// second a [`SerializedTextureDescriptor2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    First,
    Second,
}

/// A command's serialized descriptor, either form.
#[derive(Debug, Clone, Copy)]
enum Serialized {
    First(SerializedTextureDescriptor),
    Second(SerializedTextureDescriptor2),
}

impl Serialized {
    /// Its value, forced to `resource_index` when that is not 0.
    fn desc(&self, resource_index: u64) -> TextureDesc {
        match self {
            Serialized::First(d) => texture_desc(d, resource_index),
            Serialized::Second(d) => texture_desc2(d, resource_index),
        }
    }

    /// A Metal descriptor made from its value ([`Serialized::desc`]).
    fn build<P: Platform>(&self, platform: &P, resource_index: u64) -> P::TextureDescriptor {
        platform.texture_descriptor(&self.desc(resource_index))
    }
}

// ---- descriptors

impl<P: Platform> ResourceManager<P> {
    /// A Metal descriptor from a first-form serialized descriptor, at no forced index.
    pub fn texture_descriptor_for_descriptor(
        &self,
        d: &SerializedTextureDescriptor,
    ) -> RaiseResult<Option<P::TextureDescriptor>> {
        Ok(Some(
            self.platform().texture_descriptor(&texture_desc(d, 0)),
        ))
    }

    /// A Metal descriptor from a second-form serialized descriptor, at no forced index.
    pub fn texture_descriptor_for_descriptor2(
        &self,
        d: &SerializedTextureDescriptor2,
    ) -> RaiseResult<Option<P::TextureDescriptor>> {
        Ok(Some(
            self.platform().texture_descriptor(&texture_desc2(d, 0)),
        ))
    }
}

/// The index a descriptor is forced to: `resource_index`, unless that is 0.
fn forced_index(resource_index: u64) -> Option<(u64, bool)> {
    (resource_index != 0).then_some((resource_index, true))
}

/// The value of a first-form serialized descriptor, forced to `resource_index` when that is not
/// 0. Every property is sent ([`TextureDesc`] gives the order).
fn texture_desc(s: &SerializedTextureDescriptor, resource_index: u64) -> TextureDesc {
    let s = *s;
    TextureDesc {
        texture_type: u64::from(s.texture_type()),
        pixel_format: u64::from(s.pixel_format()),
        width: u64::from(s.width),
        height: u64::from(s.height),
        depth: u64::from(s.depth),
        mipmap_level_count: u64::from(s.mipmap_level_count),
        sample_count: u64::from(s.sample_count),
        array_length: u64::from(s.array_length),
        usage: u64::from(s.usage()),
        resource_options: Some(u64::from(s.resource_options)),
        swizzle: None,
        resource_index: forced_index(resource_index),
        protection_options: Some(s.protection_options),
        framebuffer_only: Some(s.framebuffer_only()),
        is_drawable: Some(s.is_drawable()),
        write_swizzle_enabled: None,
        allow_gpu_optimized_contents: Some(s.allow_gpu_optimized_contents()),
        rotation: None,
    }
}

/// As [`texture_desc`] for the second form: its usage is 32 bits wide, and it has the
/// write-swizzle flag and a swizzle.
fn texture_desc2(s: &SerializedTextureDescriptor2, resource_index: u64) -> TextureDesc {
    let s = *s;
    TextureDesc {
        texture_type: u64::from(s.texture_type()),
        pixel_format: u64::from(s.pixel_format()),
        width: u64::from(s.width),
        height: u64::from(s.height),
        depth: u64::from(s.depth),
        mipmap_level_count: u64::from(s.mipmap_level_count),
        sample_count: u64::from(s.sample_count),
        array_length: u64::from(s.array_length),
        usage: u64::from(s.usage),
        resource_options: Some(u64::from(s.resource_options)),
        swizzle: Some(Swizzle {
            red: swizzle_channel(s.swizzle_red),
            green: swizzle_channel(s.swizzle_green),
            blue: swizzle_channel(s.swizzle_blue),
            alpha: swizzle_channel(s.swizzle_alpha),
        }),
        resource_index: forced_index(resource_index),
        protection_options: Some(s.protection_options),
        framebuffer_only: Some(s.framebuffer_only()),
        is_drawable: Some(s.is_drawable()),
        write_swizzle_enabled: Some(s.write_swizzle_enabled()),
        allow_gpu_optimized_contents: Some(s.allow_gpu_optimized_contents()),
        rotation: None,
    }
}

/// A swizzle channel of a descriptor: one Metal has, or One.
fn swizzle_channel(v: u8) -> u8 {
    if v < SWIZZLE_LIMIT { v } else { SWIZZLE_ONE }
}

/// A view's swizzle as the command carries it: raw, unchecked.
fn swizzle_of(raw: [u8; 4]) -> Swizzle {
    let [red, green, blue, alpha] = raw;
    Swizzle {
        red,
        green,
        blue,
        alpha,
    }
}

/// Whether `descriptor`, made from `desc`, fits plane `plane` of `surface`. A planar surface is
/// not checked. Every `false` has logged one line. A nil surface or device answers zero to every
/// question.
///
/// The surface's bytes per row must be aligned to the device's
/// `iosurfaceReadOnlyTextureAlignmentBytes` when the descriptor's usage (its low three bits) is
/// shader read only, and to its `iosurfaceTextureAlignmentBytes` otherwise.
///
/// The rotation and the pixel format are Metal's readings of the descriptor; the other
/// properties are `desc`'s, which Metal's descriptor keeps as set (a rotation set after the
/// build is Metal's only change to it).
fn validate_iosurface_descriptor<P: Platform>(
    device: Option<&P::Device>,
    desc: &TextureDesc,
    descriptor: &P::TextureDescriptor,
    surface: Option<&P::Surface>,
    plane: u16,
) -> bool {
    // Rotations 1, 3 and 4 trade the width and the height.
    let (width, height) = if matches!(descriptor.rotation(), 1 | 3 | 4) {
        (desc.height, desc.width)
    } else {
        (desc.width, desc.height)
    };
    // A planar surface is not checked at all.
    if surface.map_or(0, |s| s.plane_count()) != 0 {
        return true;
    }
    if plane != 0 {
        log::warn!("iosurface texture refused: plane {plane} of a surface with no planes");
        return false;
    }
    let query = TextureLevelQuery {
        pixel_format: descriptor.pixel_format(),
        width,
        height,
        depth: 1,
        array_length: 1,
        options: 0,
        flags: 0,
    };
    let info = device
        .map(|d| d.texture_level_info(&query))
        .unwrap_or_default();
    let row_bytes = info.bytes_per_row;
    let surface_row_bytes = surface.map_or(0, |s| s.bytes_per_row());
    if row_bytes > surface_row_bytes {
        log::warn!(
            "iosurface texture refused: {row_bytes} bytes per row, more than the surface's \
             {surface_row_bytes}"
        );
        return false;
    }
    let alignment = device.map_or(0, |d| {
        let limits = d.caps().limits;
        if desc.usage & 7 == USAGE_SHADER_READ {
            limits.iosurface_read_only_texture_alignment_bytes
        } else {
            limits.iosurface_texture_alignment_bytes
        }
    });
    if surface_row_bytes & alignment.wrapping_sub(1) != 0 {
        log::warn!(
            "iosurface texture refused: the surface's {surface_row_bytes:#x} bytes per row are \
             not aligned to {alignment:#x}"
        );
        return false;
    }
    let (rows, overflow) = info.aligned_height.overflowing_mul(desc.array_length);
    if overflow {
        log::warn!(
            "iosurface texture refused: {} rows times {} array slices overflows",
            info.aligned_height,
            desc.array_length
        );
        return false;
    }
    if rows > surface.map_or(0, |s| s.height()) {
        // The surface's height is asked a second time, for the line, whether or not the line is
        // written.
        let surface_height = surface.map_or(0, |s| s.height());
        log::warn!(
            "iosurface texture refused: {rows} rows, more than the surface's {surface_height}"
        );
        return false;
    }
    if !surface.is_some_and(|s| s.allows_pixel_size_casting())
        && info.bytes_per_pixel != surface.map_or(0, |s| s.bytes_per_element())
    {
        log::warn!(
            "iosurface texture refused: {} bytes per pixel, not the surface's",
            info.bytes_per_pixel
        );
        return false;
    }
    if desc.mipmap_level_count != 1 {
        log::warn!(
            "iosurface texture refused: {} mip levels, where only 1 is allowed",
            desc.mipmap_level_count
        );
        return false;
    }
    if desc.depth != 1 {
        log::warn!(
            "iosurface texture refused: depth {}, where only 1 is allowed",
            desc.depth
        );
        return false;
    }
    // Only a texture buffer is refused.
    if desc.texture_type == TEXTURE_TYPE_TEXTURE_BUFFER {
        log::warn!("iosurface texture refused: a texture buffer");
        return false;
    }
    true
}

// ---- placement

/// The index a placement slot asks for: its first word, loaded now; 0 without a slot.
fn placement_index(placement: Option<GuestPtr>) -> u64 {
    placement.map_or(0, placement_field0)
}

/// Whether a texture made for a placement slot whose index is set landed elsewhere: its GPU
/// resource ID (0 for nil) is not the slot's. A misplaced texture is logged.
///
/// The slot's ID is loaded once, after Metal's answer. A misplaced texture's ID is asked a
/// second time, for the line, whether or not the line is written.
fn misplaced<P: Platform>(texture: Option<&P::Texture>, placement: Option<GuestPtr>) -> bool {
    let Some(slot) = placement else {
        return false;
    };
    if placement_field0(slot) == 0 {
        return false;
    }
    let got = texture.map_or(0, |t| t.gpu_resource_id());
    let wanted = placement_resource_id(slot);
    if got == wanted {
        return false;
    }
    let asked_again = texture.map_or(0, |t| t.gpu_resource_id());
    log::warn!("texture placed at resource ID {asked_again:#x}, not the guest's {wanted:#x}");
    true
}

/// Writes a made texture's index, then its GPU resource ID (0 each for nil), into the slot.
fn write_back<P: Platform>(placement: Option<GuestPtr>, texture: Option<&P::Texture>) {
    if let Some(slot) = placement {
        set_placement_field0(slot, texture.map_or(0, |t| t.resource_index()));
        set_placement_resource_id(slot, texture.map_or(0, |t| t.gpu_resource_id()));
    }
}

/// The refusal of a misplaced texture.
fn misplaced_refusal<T>() -> Result<T, String> {
    Err("texture not placed where the guest asked".to_owned())
}

// ---- records

/// Reads a descriptor command of `form` (0x24 or 0x2c bytes).
fn read_descriptor_op(cursor: &mut Cursor, form: Form) -> Result<Serialized, String> {
    let read = match form {
        Form::First => cursor
            .read::<OpWithDescriptor>()
            .map(|op| Serialized::First(op.descriptor)),
        Form::Second => cursor
            .read::<OpWithDescriptor2>()
            .map(|op| Serialized::Second(op.descriptor)),
    };
    read.map_err(|e| format!("texture descriptor command not readable: {e}"))
}

/// The form a descriptor texture's command type names: 1 or 52.
fn descriptor_form(command_type: u32) -> Option<Form> {
    match command_type {
        1 => Some(Form::First),
        52 => Some(Form::Second),
        _ => None,
    }
}

/// Reads the rest of a dimension record after its fixed part: `(mips * 0x24 + 0x18) * planes -
/// 0x18` more bytes. A second plane's header starts at `mips[mips]`: the bytes are kept in
/// order in `mips` (the last one zero-padded), so the header's bytes then the mips', cut at the
/// returned length, are the guest's record.
fn read_dimension_rest(
    cursor: &mut Cursor,
    dimension: TextureDimension,
    plane_count: u64,
) -> (u64, Result<TextureDimensionRecord, String>) {
    let mip_levels = u64::from(dimension.mip_levels());
    // At most (0x3fff * 0x24 + 0x18) * 2: it never wraps.
    let length = mip_levels
        .wrapping_mul(MIP_DIMENSION_SIZE)
        .wrapping_add(DIMENSION_SIZE)
        .wrapping_mul(plane_count);
    let mut rest = vec![0u8; length.wrapping_sub(DIMENSION_SIZE).to_usize()];
    if let Err(e) = cursor.read_bytes(&mut rest) {
        return (
            length,
            Err(format!(
                "rest of the texture dimension record not readable: {e}"
            )),
        );
    }
    let mips = rest
        .chunks(size_of::<TextureMipDimension>())
        .map(|chunk| {
            let mut mip = TextureMipDimension::default();
            if let Some(dst) = mip.as_mut_bytes().get_mut(..chunk.len()) {
                dst.copy_from_slice(chunk);
            }
            mip
        })
        .collect();
    (
        length,
        Ok(TextureDimensionRecord {
            header: dimension,
            mips,
        }),
    )
}

/// The task's features, loaded.
fn features<P: Platform, E: CreateEnv<P>>(env: &E) -> GuestFeatures {
    env.host().features().load()
}

/// The planes a paged texture of `object_type` has: `two_plane_type`'s are two, which the task
/// must support.
fn plane_count<P: Platform, E: CreateEnv<P>>(
    env: &E,
    object_type: u8,
    two_plane_type: u8,
) -> Result<u64, String> {
    if object_type != two_plane_type {
        return Ok(1);
    }
    if !features(env).dual_plane_textures {
        return Err(
            "two-plane texture refused: the task's features have no dual-plane textures".to_owned(),
        );
    }
    Ok(2)
}

// ---- paged textures

/// A paged texture over a descriptor (object types 2, 3 and 12, the last two-plane): its record,
/// its dimension record, then a descriptor command.
pub(crate) fn create_normal_texture<P: Platform, E: CreateEnv<P>>(
    env: &E,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let plane_count = refuse!(plane_count(env, entry.object_type(), 12));
    let object = refuse!(
        cursor
            .read::<ObjectTexture>()
            .map_err(|e| format!("paged texture record not readable: {e}"))
    );
    let dimension = refuse!(
        cursor
            .read::<TextureDimension>()
            .map_err(|e| format!("texture dimension record not readable: {e}"))
    );
    let (dimension_length, record) = read_dimension_rest(&mut cursor, dimension, plane_count);
    let record = refuse!(record);
    let header = refuse!(read_header(&mut cursor));
    let Some(form) = descriptor_form(header.command_type) else {
        return Ok(Err(format!(
            "unknown command type {} for a paged texture",
            header.command_type
        )));
    };
    let texture = refuse!(new_texture(env, form, &mut cursor, placement)?);
    let paging_info = TexturePagingInfo {
        page: object.page,
        unknown04: [0; 4],
        length: object.length,
    };
    let kept = texture.clone_ref();
    let Some(resource) = env.create_texture_resource(
        u32::from(entry.object_type()),
        &paging_info,
        &record,
        dimension_length,
        texture,
    )?
    else {
        return Ok(Err("no host resource made for the paged texture".to_owned()));
    };
    write_back::<P>(placement, Some(&kept));
    Ok(Ok(resource))
}

/// A shared backing (object types 17 and 18, the second two-plane): as a paged texture, over a
/// shareable texture.
pub(crate) fn create_shared_texture_backing<P: Platform, E: CreateEnv<P>>(
    env: &E,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let object_type = entry.object_type();
    if object_type != 17 && object_type != 18 {
        return Ok(Err(format!(
            "object of type {object_type} is not a shared texture backing"
        )));
    }
    let plane_count = refuse!(plane_count(env, object_type, 18));
    let object = refuse!(
        cursor
            .read::<ObjectSharedTextureBacking>()
            .map_err(|e| format!("shared texture backing record not readable: {e}"))
    );
    let dimension = refuse!(
        cursor
            .read::<TextureDimension>()
            .map_err(|e| format!("texture dimension record not readable: {e}"))
    );
    if dimension.mip_levels() == 0 {
        return Ok(Err(
            "shared texture backing refused: 0 mip levels".to_owned()
        ));
    }
    let (dimension_length, record) = read_dimension_rest(&mut cursor, dimension, plane_count);
    let record = refuse!(record);
    let header = refuse!(read_header(&mut cursor));
    let texture = match header.command_type {
        48 => new_shared_texture(env, Form::First, &mut cursor, placement)?,
        53 => new_shared_texture(env, Form::Second, &mut cursor, placement)?,
        command_type => Err(format!(
            "unknown command type {command_type} for a shared texture backing"
        )),
    };
    let texture = refuse!(texture);
    let paging_info = TexturePagingInfo {
        page: object.page,
        unknown04: [0; 4],
        length: object.length,
    };
    let kept = texture.clone_ref();
    let Some(resource) = env.create_shared_texture_resource(
        u32::from(object_type),
        &paging_info,
        &record,
        dimension_length,
        texture,
    )?
    else {
        return Ok(Err(
            "no host resource made for the shared texture backing".to_owned()
        ));
    };
    write_back::<P>(placement, Some(&kept));
    Ok(Ok(resource))
}

/// The operation header a record's command opens with.
fn read_header(cursor: &mut Cursor) -> Result<OperationHeader, String> {
    cursor
        .read::<OperationHeader>()
        .map_err(|e| format!("command header not readable: {e}"))
}

/// A texture over a descriptor at the slot's index, refused if memoryless, nil or misplaced.
fn new_texture<P: Platform, E: CreateEnv<P>>(
    env: &E,
    form: Form,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::Texture, String>> {
    let serialized = refuse!(read_descriptor_op(cursor, form));
    let descriptor = serialized.build(env.platform().as_ref(), placement_index(placement));
    let storage_mode = descriptor.storage_mode();
    if storage_mode == STORAGE_MODE_MEMORYLESS {
        return Ok(Err(format!(
            "texture refused: storage mode {} is memoryless",
            storage_mode.low_u32()
        )));
    }
    let texture = env
        .device()
        .map_or(Ok(None), |d| d.new_texture(OnFault::Return, &descriptor))?;
    let Some(texture) = texture else {
        let desc = serialized.desc(placement_index(placement));
        log::warn!("{}", refused_line("texture", &desc, storage_mode));
        return Ok(Err("Metal made no texture".to_owned()));
    };
    if misplaced::<P>(Some(&texture), placement) {
        drop(texture);
        return Ok(misplaced_refusal());
    }
    Ok(Ok(texture))
}

/// The log line of a texture Metal made none of (`what`, a texture or a shareable texture),
/// over `desc`, the descriptor's value, whose storage mode the descriptor gives as
/// `storage_mode`: its pixel format, type, usage, storage mode, protection options and size.
/// Such a texture is over no IOSurface and no plane. A diagnostic: what the guest sees is the
/// refusal alone.
fn refused_line(what: &str, desc: &TextureDesc, storage_mode: u64) -> String {
    let protection = match desc.protection_options {
        Some(options) => format!("{options:#x}"),
        None => "unset".to_owned(),
    };
    format!(
        "Metal made no {what}: pixel format {}, texture type {}, usage {:#x}, storage mode {}, \
         protection options {protection}, size {}x{}x{}, mip levels {}, samples {}, \
         array length {}, over no IOSurface",
        desc.pixel_format,
        desc.texture_type,
        desc.usage,
        storage_mode,
        desc.width,
        desc.height,
        desc.depth,
        desc.mipmap_level_count,
        desc.sample_count,
        desc.array_length,
    )
}

/// A shareable texture over a descriptor, as [`new_texture`].
fn new_shared_texture<P: Platform, E: CreateEnv<P>>(
    env: &E,
    form: Form,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::Texture, String>> {
    let serialized = refuse!(read_descriptor_op(cursor, form));
    let descriptor = serialized.build(env.platform().as_ref(), placement_index(placement));
    let storage_mode = descriptor.storage_mode();
    if storage_mode == STORAGE_MODE_MEMORYLESS {
        return Ok(Err(format!(
            "shared texture refused: storage mode {} is memoryless",
            storage_mode.low_u32()
        )));
    }
    let texture = env
        .device()
        .map_or(Ok(None), |d| d.new_shared_texture(&descriptor))?;
    let Some(texture) = texture else {
        let desc = serialized.desc(placement_index(placement));
        log::warn!("{}", refused_line("shareable texture", &desc, storage_mode));
        return Ok(Err("Metal made no shareable texture".to_owned()));
    };
    if misplaced::<P>(Some(&texture), placement) {
        drop(texture);
        return Ok(misplaced_refusal());
    }
    Ok(Ok(texture))
}

// ---- the deserializer's textures

/// A texture the deserializer makes (object types 8 and 9), by its command: a memoryless
/// texture (1, 52), a view (7, 8, 27), a texture over a buffer (9, 55) or in a heap (21, 56). A
/// made texture with a slot is written back through its resource.
pub(crate) fn create_serializer_texture<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let header = refuse!(read_header(&mut cursor));
    let object = match header.command_type {
        1 => new_memoryless_texture(env, Form::First, &mut cursor, placement)?,
        52 => new_memoryless_texture(env, Form::Second, &mut cursor, placement)?,
        7 => new_view(hold, env, ViewKind::PixelFormat, &mut cursor, placement)?,
        8 => new_view(hold, env, ViewKind::TextureType, &mut cursor, placement)?,
        27 => new_view(hold, env, ViewKind::Swizzle, &mut cursor, placement)?,
        9 => new_texture_over_buffer(hold, env, Form::First, &mut cursor, placement)?,
        55 => new_texture_over_buffer(hold, env, Form::Second, &mut cursor, placement)?,
        21 => new_texture_in_heap(hold, env, Form::First, &mut cursor, placement)?,
        56 => new_texture_in_heap(hold, env, Form::Second, &mut cursor, placement)?,
        command_type => {
            return Ok(Err(format!(
                "unknown command type {command_type} for a child texture"
            )));
        }
    };
    let object = refuse!(object);
    if placement.is_some() {
        let texture = env.res_get_texture(&object)?;
        write_back::<P>(placement, texture.as_ref());
    }
    Ok(Ok(object))
}

/// A memoryless texture: any other storage mode is refused, and a nil texture reaches its
/// resource.
fn new_memoryless_texture<P: Platform, E: CreateEnv<P>>(
    env: &E,
    form: Form,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let serialized = refuse!(read_descriptor_op(cursor, form));
    let descriptor = serialized.build(env.platform().as_ref(), placement_index(placement));
    if descriptor.storage_mode() != STORAGE_MODE_MEMORYLESS {
        return Ok(Err(
            "memoryless texture refused: its storage mode is not memoryless".to_owned(),
        ));
    }
    let texture = env
        .device()
        .map_or(Ok(None), |d| d.new_texture(OnFault::Return, &descriptor))?;
    if misplaced::<P>(texture.as_ref(), placement) {
        drop(texture);
        return Ok(misplaced_refusal());
    }
    match env.new_discardable_serializer_texture_resource(texture)? {
        Some(r) => Ok(Ok(r)),
        None => Ok(Err(
            "no host resource made for the memoryless texture".to_owned()
        )),
    }
}

/// The three views of a base texture.
#[derive(Debug, Clone, Copy)]
enum ViewKind {
    /// Another pixel format (7).
    PixelFormat,
    /// Other levels and slices, in another type (8).
    TextureType,
    /// As `TextureType`, swizzled (27).
    Swizzle,
}

/// A view command as read.
#[derive(Debug, Clone, Copy)]
enum View {
    PixelFormat(OpViewWithPixelFormat),
    TextureType(OpViewWithTextureType),
    Swizzle(OpViewWithSwizzle),
}

impl View {
    fn read(cursor: &mut Cursor, kind: ViewKind) -> Result<View, String> {
        let read = match kind {
            ViewKind::PixelFormat => cursor
                .read::<OpViewWithPixelFormat>()
                .map(View::PixelFormat),
            ViewKind::TextureType => cursor
                .read::<OpViewWithTextureType>()
                .map(View::TextureType),
            ViewKind::Swizzle => cursor.read::<OpViewWithSwizzle>().map(View::Swizzle),
        };
        read.map_err(|e| format!("texture view command not readable: {e}"))
    }

    fn base_reference(self) -> u32 {
        match self {
            View::PixelFormat(op) => op.base_texture_reference,
            View::TextureType(op) => op.base_texture_reference,
            View::Swizzle(op) => op.base_texture_reference,
        }
    }

    /// Asks `base` for the view: at `index` when that is not 0.
    fn make<P: Platform>(self, base: &P::Texture, index: u64) -> RaiseResult<Option<P::Texture>> {
        match (self, index) {
            (View::PixelFormat(op), 0) => base.new_texture_view(u64::from(op.pixel_format)),
            (View::PixelFormat(op), index) => {
                base.new_texture_view_at_index(u64::from(op.pixel_format), index)
            }
            (View::TextureType(op), 0) => base.new_texture_subview(
                u64::from(op.pixel_format),
                u64::from(op.texture_type),
                op.levels.ns(),
                op.slices.ns(),
            ),
            (View::TextureType(op), index) => base.new_texture_subview_at_index(
                u64::from(op.pixel_format),
                u64::from(op.texture_type),
                op.levels.ns(),
                op.slices.ns(),
                index,
            ),
            (View::Swizzle(op), 0) => base.new_swizzled_texture_subview(
                u64::from(op.pixel_format),
                u64::from(op.texture_type),
                op.levels.ns(),
                op.slices.ns(),
                swizzle_of(op.swizzle),
            ),
            (View::Swizzle(op), index) => base.new_swizzled_texture_subview_at_index(
                u64::from(op.pixel_format),
                u64::from(op.texture_type),
                op.levels.ns(),
                op.slices.ns(),
                swizzle_of(op.swizzle),
                index,
            ),
        }
    }
}

/// A view of a base texture, made a child of the base's resource. The slot's index is loaded
/// once: the view is made at the index tested, whatever the guest stores after.
fn new_view<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    kind: ViewKind,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let view = refuse!(View::read(cursor, kind));
    let base = refuse!(get_base(hold, env, view.base_reference(), "texture")?);
    let base_texture = match &base {
        Some(b) => env.res_get_texture(b)?,
        None => None,
    };
    let index = placement_index(placement);
    // The view is not checked for nil.
    let texture = base_texture
        .as_ref()
        .map_or(Ok(None), |t| view.make::<P>(t, index))?;
    drop(base_texture);
    if misplaced::<P>(texture.as_ref(), placement) {
        drop(texture);
        return Ok(misplaced_refusal());
    }
    child_texture(env, base.as_ref(), texture)
}

/// A texture over a buffer, at the buffer's options: linear for a 2D array. A nil texture
/// reaches the child.
fn new_texture_over_buffer<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    form: Form,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let read = match form {
        Form::First => cursor.read::<OpViewWithBuffer>().map(|op| {
            (
                op.buffer_reference,
                op.offset,
                op.bytes_per_row,
                Serialized::First(op.descriptor),
            )
        }),
        Form::Second => cursor.read::<OpViewWithBuffer2>().map(|op| {
            (
                op.buffer_reference,
                op.offset,
                op.bytes_per_row,
                Serialized::Second(op.descriptor),
            )
        }),
    };
    let (buffer_reference, offset, bytes_per_row, serialized) =
        refuse!(read.map_err(|e| format!("buffer texture command not readable: {e}")));
    let base = refuse!(get_base(hold, env, buffer_reference, "buffer")?);
    let buffer = match &base {
        Some(b) => env.res_get_buffer(b)?,
        None => None,
    };
    let descriptor = serialized.build(env.platform().as_ref(), placement_index(placement));
    descriptor.set_resource_options(buffer.as_ref().map_or(0, |b| b.resource_options()));
    let linear = descriptor.texture_type() == TEXTURE_TYPE_2D_ARRAY;
    let texture = buffer.as_ref().map_or(Ok(None), |b| {
        if linear {
            b.new_linear_texture(&descriptor, offset, bytes_per_row, 0)
        } else {
            b.new_texture(OnFault::Return, &descriptor, offset, bytes_per_row)
        }
    })?;
    drop(buffer);
    if misplaced::<P>(texture.as_ref(), placement) {
        drop(texture);
        return Ok(misplaced_refusal());
    }
    child_texture(env, base.as_ref(), texture)
}

/// A texture in a heap, at the heap's options, made a child of the heap's resource.
fn new_texture_in_heap<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    form: Form,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let read = match form {
        Form::First => cursor.read::<OpWithHeap>().map(|op| {
            (
                op.heap_reference,
                op.offset,
                Serialized::First(op.descriptor),
            )
        }),
        Form::Second => cursor.read::<OpWithHeap2>().map(|op| {
            (
                op.heap_reference,
                op.offset,
                Serialized::Second(op.descriptor),
            )
        }),
    };
    let (heap_reference, offset, serialized) =
        refuse!(read.map_err(|e| format!("heap texture command not readable: {e}")));
    let base = refuse!(get_base(hold, env, heap_reference, "heap")?);
    let heap = match &base {
        Some(b) => env.res_get_heap(b)?,
        None => None,
    };
    let descriptor = serialized.build(env.platform().as_ref(), placement_index(placement));
    descriptor.set_resource_options(heap.as_ref().map_or(0, |h| h.resource_options()));
    let texture = heap
        .as_ref()
        .map_or(Ok(None), |h| h.new_texture(&descriptor, offset))?;
    drop(heap);
    let Some(texture) = texture else {
        return Ok(Err("Metal made no texture in the heap".to_owned()));
    };
    if misplaced::<P>(Some(&texture), placement) {
        drop(texture);
        return Ok(misplaced_refusal());
    }
    child_texture(env, base.as_ref(), Some(texture))
}

/// The resource at a base's reference, the failure named for `what`.
fn get_base<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    reference: u32,
    what: &str,
) -> RaiseResult<Result<Option<E::Res>, String>> {
    Ok(env
        .get_resource(hold, reference)?
        .map_err(|e| format!("no base {what} at reference {reference}: {e}")))
}

/// A child of `base` over `texture`; a nil base (reference 0) makes none.
fn child_texture<P: Platform, E: CreateEnv<P>>(
    env: &E,
    base: Option<&E::Res>,
    texture: Option<P::Texture>,
) -> RaiseResult<Result<E::Res, String>> {
    let child = match base {
        Some(b) => env.new_child_texture_resource(b, texture)?,
        None => None,
    };
    match child {
        Some(r) => Ok(Ok(r)),
        None => Ok(Err("no child resource made for the texture".to_owned())),
    }
}

// ---- IOSurface textures

/// The three IOSurface texture commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceCommand {
    /// 12: no rotation.
    Plain,
    /// 47: rotated, first form.
    Rotated,
    /// 57: rotated, second form.
    Rotated2,
}

impl SurfaceCommand {
    /// The command `command_type` names, if it is one of the three.
    fn of(command_type: u32) -> Option<SurfaceCommand> {
        match command_type {
            12 => Some(SurfaceCommand::Plain),
            47 => Some(SurfaceCommand::Rotated),
            57 => Some(SurfaceCommand::Rotated2),
            _ => None,
        }
    }

    /// Reads the command: its descriptor, its plane and (the rotated forms) its rotation.
    fn read(self, cursor: &mut Cursor) -> Result<(Serialized, u16, Option<u16>), String> {
        let read = match self {
            SurfaceCommand::Plain => cursor
                .read::<OpIoSurfaceTexture>()
                .map(|o| (Serialized::First(o.descriptor), o.plane, None)),
            SurfaceCommand::Rotated => cursor
                .read::<OpIoSurfaceTextureWithRotation>()
                .map(|o| (Serialized::First(o.descriptor), o.plane, Some(o.rotation))),
            SurfaceCommand::Rotated2 => cursor
                .read::<OpIoSurfaceTextureWithRotation2>()
                .map(|o| (Serialized::Second(o.descriptor), o.plane, Some(o.rotation))),
        };
        read.map_err(|e| format!("IOSurface texture command not readable: {e}"))
    }
}

/// A texture over plane `plane` of `surface`, validated first. The rotated forms set the
/// rotation, and die if Metal did not keep it.
fn new_iosurface_texture<P: Platform, E: CreateEnv<P>>(
    env: &E,
    command: SurfaceCommand,
    surface: Option<&P::Surface>,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<(P::Texture, u16), String>> {
    let (serialized, plane, rotation) = refuse!(command.read(cursor));
    let desc = serialized.desc(placement_index(placement));
    let descriptor = env.platform().texture_descriptor(&desc);
    let storage_mode = descriptor.storage_mode();
    match rotation {
        None => {
            if storage_mode == STORAGE_MODE_MEMORYLESS {
                return Ok(Err(format!(
                    "IOSurface texture refused: storage mode {} is memoryless",
                    storage_mode.low_u32()
                )));
            }
        }
        Some(rotation) => {
            if storage_mode == STORAGE_MODE_MEMORYLESS {
                return Ok(Err(
                    "IOSurface texture refused: its storage mode is memoryless".to_owned(),
                ));
            }
            descriptor.set_rotation(u64::from(rotation));
            crate::fatal_unless!(
                rotation == descriptor.rotation().low_u16(),
                DeathKind::RotationMismatch
            );
        }
    }
    if !validate_iosurface_descriptor::<P>(env.device(), &desc, &descriptor, surface, plane) {
        return Ok(Err("IOSurface texture descriptor refused".to_owned()));
    }
    let texture = env.device().map_or(Ok(None), |d| {
        d.new_texture_on_surface(&descriptor, surface, u64::from(plane))
    })?;
    let Some(texture) = texture else {
        return Ok(Err("Metal made no IOSurface texture".to_owned()));
    };
    if misplaced::<P>(Some(&texture), placement) {
        drop(texture);
        return Ok(misplaced_refusal());
    }
    Ok(Ok((texture, plane)))
}

/// A texture over a mapper's IOSurface (object type 11). The surface is retained through the
/// task before the texture is made and handed back on every refusal; a raise leaks the retain.
/// Not written back.
pub(crate) fn create_mapper_ref_texture<P: Platform, E: CreateEnv<P>>(
    env: &E,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let host = env.host();
    // The features are asked for, and not read.
    drop(host.features());
    let object = refuse!(
        cursor
            .read::<ObjectMapperRefTexture>()
            .map_err(|e| format!("mapper-ref texture record not readable: {e}"))
    );
    let mapping_id = object.mapping_id;
    let header = refuse!(read_header(&mut cursor));
    let Some(surface) = host.retain_io_surface_with_mapping_id(mapping_id)? else {
        return Ok(Err(format!("no IOSurface for mapping {mapping_id}")));
    };
    let Some(command) = SurfaceCommand::of(header.command_type) else {
        host.release_io_surface_with_mapping_id(mapping_id, Some(surface));
        return Ok(Err(format!(
            "unknown command type {} for a mapper-ref texture",
            header.command_type
        )));
    };
    let (texture, _) = match new_iosurface_texture(env, command, Some(&surface), &mut cursor, placement)
    {
        Ok(Ok((texture, plane))) => (texture, plane),
        Ok(Err(e)) => {
            host.release_io_surface_with_mapping_id(mapping_id, Some(surface));
            return Ok(Err(e));
        }
        Err(raised) => {
            keep_forever(surface, Kept::SurfaceUseCount);
            return Err(raised);
        }
    };
    use crate::platform::os::surface::Surface as _;
    let surface_base = surface.base_address().addr().to_u64();
    let surface_size = surface.alloc_size();
    let resource = match env.new_mapper_ref_texture_resource(
        mapping_id,
        texture,
        surface_base,
        surface_size,
    ) {
        Ok(resource) => resource,
        Err(raised) => {
            keep_forever(surface, Kept::SurfaceUseCount);
            return Err(raised);
        }
    };
    let Some(resource) = resource else {
        host.release_io_surface_with_mapping_id(mapping_id, Some(surface));
        return Ok(Err(
            "no host resource made for the mapper-ref texture".to_owned()
        ));
    };
    drop(surface);
    Ok(Ok(resource))
}

/// A texture over task 0's IOSurface backing (object type 5), holding the backing. Not written
/// back.
pub(crate) fn create_backing_ref_texture<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let object = refuse!(
        cursor
            .read::<ObjectRefTexture>()
            .map_err(|e| format!("ref texture record not readable: {e}"))
    );
    if object.task_id != 0 {
        return Ok(Err(format!(
            "ref texture refused: its backing is in task {}, not task 0",
            object.task_id
        )));
    }
    let header = refuse!(read_header(&mut cursor));
    let backing_id = object.backing_id;
    let Some(backing) = env.backing_for_id(hold, backing_id)? else {
        return Ok(Err(format!("no backing with IOSurface ID {backing_id}")));
    };
    let Some(command) = SurfaceCommand::of(header.command_type) else {
        return Ok(Err(format!(
            "unknown command type {} for a ref texture",
            header.command_type
        )));
    };
    let surface = env.res_io_surface(&backing)?;
    let texture = new_iosurface_texture(env, command, surface.as_ref(), &mut cursor, placement)?;
    drop(surface);
    let (texture, plane) = refuse!(texture);
    match env.new_ref_texture_resource(u32::from(entry.object_type()), texture, backing, plane)? {
        Some(r) => Ok(Ok(r)),
        None => Ok(Err("no host resource made for the ref texture".to_owned())),
    }
}

/// An IOSurface backing of up to three planes (object type 4): its record, then its planes.
pub(crate) fn create_backing<P: Platform, E: CreateEnv<P>>(
    env: &E,
    mut cursor: Cursor,
) -> RaiseResult<Result<E::Res, String>> {
    // The backing record is read as 0x14 bytes.
    let object = refuse!(
        cursor
            .read::<ObjectBacking>()
            .map_err(|e| format!("backing record not readable: {e}"))
    );
    let plane_count = object.plane_count;
    if plane_count >= 4 {
        return Ok(Err(format!(
            "backing refused: {plane_count} planes, more than 3"
        )));
    }
    let mut planes = vec![BackingPlane::default(); usize::from(plane_count)];
    refuse!(
        cursor
            .read_bytes(planes.as_mut_slice().as_mut_bytes())
            .map_err(|e| format!("backing planes not readable: {e}"))
    );
    // The descriptor is the record's four fields; its padding is zero.
    let descriptor = BackingDescriptor {
        length: object.length,
        page: object.page,
        pixel_format: object.pixel_format,
        plane_count,
        unknown11: [0; 7],
    };
    match env.new_backing_resource(&descriptor, &planes)? {
        Some(r) => Ok(Ok(r)),
        None => Ok(Err("no host resource made for the backing".to_owned())),
    }
}

// ---- shared textures

/// Another task's shared backing, opened by handle (object type 16, command 49). Written back.
pub(crate) fn create_shared_texture<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let host = env.host();
    if !host.features().load().supports_shared_textures {
        return Ok(Err(
            "shared texture refused: the task's features have no shared textures".to_owned(),
        ));
    }
    let object = refuse!(
        cursor
            .read::<ObjectSharedTexture>()
            .map_err(|e| format!("shared texture record not readable: {e}"))
    );
    let backing_id = object.shared_texture_backing_id;
    let header = refuse!(read_header(&mut cursor));
    let Some(handle) = host.new_shared_texture_handle_for_id(hold, backing_id)? else {
        return Ok(Err(format!(
            "no shared texture handle for backing {backing_id}"
        )));
    };
    if header.command_type != 49 {
        return Ok(Err(format!(
            "unknown command type {} for a shared texture",
            header.command_type
        )));
    }
    // The handle takes the slot's index as it is, and nothing checks the ID.
    let index = placement_index(placement);
    let texture = env
        .device()
        .map_or(Ok(None), |d| d.open_shared_texture(Some(&handle), index))?;
    let Some(texture) = texture else {
        return Ok(Err("Metal made no shared texture".to_owned()));
    };
    let kept = texture.clone_ref();
    let Some(resource) = env.new_shared_texture_resource(texture, backing_id)? else {
        return Ok(Err(
            "no host resource made for the shared texture".to_owned()
        ));
    };
    write_back::<P>(placement, Some(&kept));
    Ok(Ok(resource))
}
