//! Creation of the guest's resources: the resource kind ([`Resources`]) and its dispatch on the
//! object type, the environment every creator asks ([`CreateEnv`]), and the buffers' and heaps'
//! creators. Each buffer or heap creator reads its record, has the resources area make the
//! resource, and (where the task registers GPU addresses) registers the Metal buffer's address.
//! The textures' creators are [`super::texture`]'s and the ICB's the ICB manager's; the Metal
//! state kinds' creators are this module's children ([`sampler`], [`depth_stencil`], [`fence`],
//! [`function`], [`pipeline`], [`rate_map`]).
//!
//! The creators are generic over [`CreateEnv`], what they ask of the manager, its task and the
//! resources area, so tests stand in for each (`test_env`).

#[cfg(test)]
pub(crate) mod test_env;

use std::sync::Arc;

use zerocopy::{FromBytes, IntoBytes};

use crate::deserializer::texture_descriptor::SerializedTextureDescriptor;
use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::resource::{Buffer, Heap};
use crate::platform::types::SizeAndAlign;
use crate::raise::RaiseResult;
use crate::raise::{Kept, keep_forever};
use crate::resources::Op;
use crate::resources::host::ResourceHost;
use crate::resources::manager::cache::{Cache, Caches, ManagedKind};
use crate::resources::manager::records::{
    self, ObjectBuffer, ObjectHeap, ObjectHeapBuffer, ObjectMapperRefBuffer, word,
};
use crate::resources::manager::texture;
use crate::resources::manager::{DeathKind, ResourceManager};
use crate::resources::resource::{Resource, ResourceKind};
use crate::resources::texture::backing::{BackingDescriptor, BackingPlane, BackingResource};
use crate::resources::texture::dimension::{TextureDimensionRecord, TexturePagingInfo};
use crate::resources::texture::mapper::MapperRefTextureResource;
use crate::resources::texture::shared::SharedTextureResource;
use crate::resources::texture::views::{RefTextureResource, SerializerTextureResource};
use crate::rt::Hold;

/// `MTLResourceStorageModeShift`: a heap's storage mode moves up by this into the options of a
/// buffer made from it.
const RESOURCE_STORAGE_MODE_SHIFT: u32 = 4;

/// The heap texture size query's command type, and its command's least length.
const HEAP_TEXTURE_SIZE_COMMAND: u32 = 22;
const HEAP_TEXTURE_SIZE_LENGTH: usize = 0x28;

/// The value, or `Ok(Err(e))` from the enclosing creator.
macro_rules! refuse {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Ok(Err(e)),
        }
    };
}
pub(crate) use refuse;

/// What a creator asks of the manager, its task and the resources area. [`ResourceManager`] is
/// the product's implementation (`Res` is its `Arc<Resource<P>>`); the tests stand in for each
/// call.
pub(crate) trait CreateEnv<P: Platform> {
    /// A resource.
    type Res;

    /// The process's platform: what the creators make descriptors through.
    fn platform(&self) -> &Arc<P>;
    /// The task's device; `None` is nil, and every send to it answers zero.
    fn device(&self) -> Option<&P::Device>;
    /// The task, for what the creators ask of it.
    fn host(&self) -> Arc<dyn ResourceHost<P>>;
    /// Whether buffers and heaps register their GPU addresses.
    fn register_gpu_addresses(&self) -> bool;
    /// Registers `address` as `reference`'s.
    fn register_gpu_address(&self, address: u64, reference: u32) -> Result<(), String>;
    /// The resource at `reference`, made first if it is not cached; reference 0 is `None`.
    fn get_resource(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<Self::Res>, String>>;
    /// The task's IOSurface backing with this id.
    fn backing_for_id(&self, hold: Hold<'_>, identifier: u32) -> RaiseResult<Option<Self::Res>>;
    /// An indirect command buffer (object type 20).
    fn create_indirect_command_buffer(
        &self,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Self::Res, String>>;

    /// The resource's Metal buffer.
    fn res_get_buffer(&self, r: &Self::Res) -> RaiseResult<Option<P::Buffer>>;
    /// The resource's Metal texture.
    fn res_get_texture(&self, r: &Self::Res) -> RaiseResult<Option<P::Texture>>;
    /// The resource's Metal heap.
    fn res_get_heap(&self, r: &Self::Res) -> RaiseResult<Option<P::Heap>>;
    /// A backing's IOSurface.
    fn res_io_surface(&self, r: &Self::Res) -> RaiseResult<Option<P::Surface>>;
    /// A child buffer of `r` over `buffer`.
    fn new_child_buffer_resource(
        &self,
        r: &Self::Res,
        buffer: P::Buffer,
    ) -> RaiseResult<Option<Self::Res>>;
    /// A child texture of `r` over `texture`, which may be nil.
    fn new_child_texture_resource(
        &self,
        r: &Self::Res,
        texture: Option<P::Texture>,
    ) -> RaiseResult<Option<Self::Res>>;

    /// A buffer resource from the object's record.
    fn new_buffer_resource(
        &self,
        hold: Hold<'_>,
        object_type: i32,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Self::Res>>;
    /// A heap resource from the object's record.
    fn new_heap_resource(
        &self,
        object_type: i32,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Self::Res>>;
    /// A paged texture resource over `texture`.
    fn create_texture_resource(
        &self,
        object_type: u32,
        paging_info: &TexturePagingInfo,
        dimension: &TextureDimensionRecord,
        dimension_length: u64,
        texture: P::Texture,
    ) -> RaiseResult<Option<Self::Res>>;
    /// A shared texture backing resource over `texture`.
    fn create_shared_texture_resource(
        &self,
        object_type: u32,
        paging_info: &TexturePagingInfo,
        dimension: &TextureDimensionRecord,
        dimension_length: u64,
        texture: P::Texture,
    ) -> RaiseResult<Option<Self::Res>>;
    /// A discardable texture resource over `texture`, which may be nil.
    fn new_discardable_serializer_texture_resource(
        &self,
        texture: Option<P::Texture>,
    ) -> RaiseResult<Option<Self::Res>>;
    /// A texture resource over a mapper's surface.
    fn new_mapper_ref_texture_resource(
        &self,
        mapping_id: u64,
        texture: P::Texture,
        surface_base: u64,
        surface_size: u64,
    ) -> RaiseResult<Option<Self::Res>>;
    /// A texture resource over another task's shared backing.
    fn new_shared_texture_resource(
        &self,
        texture: P::Texture,
        shared_texture_backing_id: u32,
    ) -> RaiseResult<Option<Self::Res>>;
    /// A texture resource over a backing's surface, holding the backing.
    fn new_ref_texture_resource(
        &self,
        object_type: u32,
        texture: P::Texture,
        backing: Self::Res,
        plane: u16,
    ) -> RaiseResult<Option<Self::Res>>;
    /// An IOSurface backing of up to three planes.
    fn new_backing_resource(
        &self,
        descriptor: &BackingDescriptor,
        planes: &[BackingPlane],
    ) -> RaiseResult<Option<Self::Res>>;
}

impl<P: Platform> ResourceManager<P> {
    /// The task, alive for as long as a creator runs: the task owns its manager and lends it only
    /// for a call, so a creator never outlives it.
    fn live_task(&self) -> Arc<dyn ResourceHost<P>> {
        self.task()
            .unwrap_or_else(|| crate::fatal!(DeathKind::TaskGone))
    }
}

impl<P: Platform> CreateEnv<P> for ResourceManager<P> {
    type Res = Arc<Resource<P>>;

    fn platform(&self) -> &Arc<P> {
        ResourceManager::platform(self)
    }
    fn device(&self) -> Option<&P::Device> {
        ResourceManager::device(self)
    }
    fn host(&self) -> Arc<dyn ResourceHost<P>> {
        self.live_task()
    }
    fn register_gpu_addresses(&self) -> bool {
        ResourceManager::register_gpu_addresses(self)
    }
    fn register_gpu_address(&self, address: u64, reference: u32) -> Result<(), String> {
        ResourceManager::register_gpu_address(self, address, reference)
    }
    fn get_resource(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<Self::Res>, String>> {
        self.get_object::<Resources>(hold, reference)
    }
    fn backing_for_id(&self, hold: Hold<'_>, identifier: u32) -> RaiseResult<Option<Self::Res>> {
        self.live_task().backing_for_id(hold, identifier)
    }
    fn create_indirect_command_buffer(
        &self,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Self::Res, String>> {
        ResourceManager::create_indirect_command_buffer(self, reference, entry, cursor, placement)
    }
    fn res_get_buffer(&self, r: &Self::Res) -> RaiseResult<Option<P::Buffer>> {
        r.get_buffer()
    }
    fn res_get_texture(&self, r: &Self::Res) -> RaiseResult<Option<P::Texture>> {
        r.get_texture()
    }
    fn res_get_heap(&self, r: &Self::Res) -> RaiseResult<Option<P::Heap>> {
        r.get_heap()
    }
    fn res_io_surface(&self, r: &Self::Res) -> RaiseResult<Option<P::Surface>> {
        match &r.kind {
            ResourceKind::Backing(b) => Ok(b.io_surface()),
            // The task answers backings only.
            _ => r.unrecognized(Op::IoSurface),
        }
    }
    fn new_child_buffer_resource(
        &self,
        r: &Self::Res,
        buffer: P::Buffer,
    ) -> RaiseResult<Option<Self::Res>> {
        r.new_child_buffer_resource_with_buffer(buffer)
    }
    fn new_child_texture_resource(
        &self,
        r: &Self::Res,
        texture: Option<P::Texture>,
    ) -> RaiseResult<Option<Self::Res>> {
        // A nil texture (a refused view, or a texture over a nil buffer) still makes a child:
        // discardable when `r` supports a discard, over nil.
        let discardable = r.supports_discard();
        Ok(Some(SerializerTextureResource::new(texture, discardable)))
    }
    fn new_buffer_resource(
        &self,
        hold: Hold<'_>,
        object_type: i32,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Self::Res>> {
        let task = Arc::downgrade(&self.live_task());
        crate::resources::buffer::new_buffer_resource(
            hold,
            task,
            object_type,
            descriptor,
            placement,
        )
    }
    fn new_heap_resource(
        &self,
        object_type: i32,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Self::Res>> {
        let task = Arc::downgrade(&self.live_task());
        crate::resources::heap::new_heap_resource(task, object_type, descriptor, placement)
    }
    fn create_texture_resource(
        &self,
        object_type: u32,
        paging_info: &TexturePagingInfo,
        dimension: &TextureDimensionRecord,
        dimension_length: u64,
        texture: P::Texture,
    ) -> RaiseResult<Option<Self::Res>> {
        let task = Arc::downgrade(&self.live_task());
        crate::resources::texture::create_texture_resource(
            task,
            object_type,
            paging_info,
            dimension,
            dimension_length,
            texture,
        )
    }
    fn create_shared_texture_resource(
        &self,
        object_type: u32,
        paging_info: &TexturePagingInfo,
        dimension: &TextureDimensionRecord,
        dimension_length: u64,
        texture: P::Texture,
    ) -> RaiseResult<Option<Self::Res>> {
        let task = Arc::downgrade(&self.live_task());
        crate::resources::texture::create_shared_texture_resource(
            task,
            object_type,
            paging_info,
            dimension,
            dimension_length,
            texture,
        )
    }
    fn new_discardable_serializer_texture_resource(
        &self,
        texture: Option<P::Texture>,
    ) -> RaiseResult<Option<Self::Res>> {
        // A memoryless creation's nil texture is not tested, and the resource is made over nil.
        Ok(Some(SerializerTextureResource::new(texture, true)))
    }
    fn new_mapper_ref_texture_resource(
        &self,
        mapping_id: u64,
        texture: P::Texture,
        surface_base: u64,
        surface_size: u64,
    ) -> RaiseResult<Option<Self::Res>> {
        let task = Arc::downgrade(&self.live_task());
        MapperRefTextureResource::new(task, mapping_id, texture, surface_base, surface_size)
    }
    fn new_shared_texture_resource(
        &self,
        texture: P::Texture,
        shared_texture_backing_id: u32,
    ) -> RaiseResult<Option<Self::Res>> {
        let task = Arc::downgrade(&self.live_task());
        SharedTextureResource::new(task, texture, shared_texture_backing_id)
    }
    fn new_ref_texture_resource(
        &self,
        object_type: u32,
        texture: P::Texture,
        backing: Self::Res,
        plane: u16,
    ) -> RaiseResult<Option<Self::Res>> {
        let task = Arc::downgrade(&self.live_task());
        RefTextureResource::new(task, object_type, texture, backing, plane)
    }
    fn new_backing_resource(
        &self,
        descriptor: &BackingDescriptor,
        planes: &[BackingPlane],
    ) -> RaiseResult<Option<Self::Res>> {
        let task = Arc::downgrade(&self.live_task());
        BackingResource::new(self.platform().as_ref(), task, descriptor, planes)
    }
}

/// The resource kind: buffers, heaps, indirect command buffers and textures share one cache.
#[derive(Debug)]
pub struct Resources;

impl<P: Platform> ManagedKind<P> for Resources {
    type Obj = Arc<Resource<P>>;
    fn sub_manager(c: &Caches<P>) -> &Cache<Self::Obj> {
        &c.resources
    }
    fn sub_manager_mut(c: &mut Caches<P>) -> &mut Cache<Self::Obj> {
        &mut c.resources
    }
    fn retain(o: &Self::Obj) -> Self::Obj {
        Arc::clone(o)
    }
    fn create_object(
        hold: Hold<'_>,
        m: &ResourceManager<P>,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Self::Obj, String>> {
        create_resource(hold, m, reference, entry, cursor, placement)
    }
}

/// Makes the resource `entry` names, by its object type (the entry's low byte).
pub(crate) fn create_resource<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    reference: u32,
    entry: &ObjectEntry,
    cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    match entry.object_type() {
        1 => create_buffer(hold, env, reference, entry, cursor, placement),
        15 => create_mapper_ref_buffer(hold, env, reference, entry, cursor, placement),
        14 => create_heap_buffer(hold, env, reference, entry, cursor, placement),
        2 | 3 | 12 => texture::create_normal_texture(env, entry, cursor, placement),
        8 | 9 => texture::create_serializer_texture(hold, env, cursor, placement),
        13 => create_heap(env, reference, entry, cursor, placement),
        11 => texture::create_mapper_ref_texture(env, cursor, placement),
        16 => texture::create_shared_texture(hold, env, cursor, placement),
        5 => texture::create_backing_ref_texture(hold, env, entry, cursor, placement),
        17 | 18 => texture::create_shared_texture_backing(env, entry, cursor, placement),
        4 => texture::create_backing(env, cursor),
        20 => env.create_indirect_command_buffer(reference, entry, cursor, placement),
        object_type => Ok(Err(format!(
            "object of type {object_type} is not a resource"
        ))),
    }
}

// The Metal state kinds' creators (their kinds are `states`'). After `refuse!`, which they use.
pub(crate) mod depth_stencil;
pub(crate) mod fence;
pub(crate) mod function;
pub(crate) mod pipeline;
pub(crate) mod rate_map;
pub(crate) mod sampler;

/// The object type of every Metal state's entry (samplers, depth-stencil states, fences,
/// pipelines, rate maps); a function's is [`FUNCTION_OBJECT_TYPE`].
const STATE_OBJECT_TYPE: u8 = 7;
/// The object type of a function's entry.
const FUNCTION_OBJECT_TYPE: u8 = 6;

/// A creator's placement slot: the guest's 16-byte entry holding the resource index (`field0`)
/// and the resource ID, which a creator reads before Metal makes the object and writes Metal's
/// back to after. Each access is one load or store of guest memory, re-read at every use.
#[derive(Debug, Clone, Copy)]
struct Slot(GuestPtr);

impl Slot {
    /// The resource index the guest placed the object at (the entry's `field0`).
    fn resource_index(self) -> u64 {
        records::placement_field0(self.0)
    }

    /// The resource ID the guest expects the object to get.
    fn resource_id(self) -> u64 {
        records::placement_resource_id(self.0)
    }

    fn set_resource_index(self, index: u64) {
        records::set_placement_field0(self.0, index);
    }

    fn set_resource_id(self, id: u64) {
        records::set_placement_resource_id(self.0, id);
    }
}

impl<P: Platform> ResourceManager<P> {
    /// A buffer (object types 1 and 19).
    pub fn create_buffer(
        &self,
        hold: Hold<'_>,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Arc<Resource<P>>, String>> {
        create_buffer(hold, self, reference, entry, cursor, placement)
    }

    /// A buffer over a mapper's surface (object type 15).
    pub fn create_mapper_ref_buffer(
        &self,
        hold: Hold<'_>,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Arc<Resource<P>>, String>> {
        create_mapper_ref_buffer(hold, self, reference, entry, cursor, placement)
    }

    /// A buffer in a heap (object type 14).
    pub fn create_heap_buffer(
        &self,
        hold: Hold<'_>,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Arc<Resource<P>>, String>> {
        create_heap_buffer(hold, self, reference, entry, cursor, placement)
    }

    /// A heap (object type 13).
    pub fn create_heap(
        &self,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Arc<Resource<P>>, String>> {
        create_heap(self, reference, entry, cursor, placement)
    }

    /// The size and alignment a heap texture of the serialized descriptor in `command` (type
    /// 22, at least 0x28 bytes) takes on the device. A descriptor that builds nothing is a
    /// death (an assert).
    pub fn heap_texture_size_and_align(
        &self,
        command: &[u8],
    ) -> RaiseResult<Result<SizeAndAlign, String>> {
        if command.len() < 8 {
            return Ok(Err(format!(
                "heap texture size command of {} bytes, shorter than 8",
                command.len()
            )));
        }
        if command.len() < HEAP_TEXTURE_SIZE_LENGTH
            || word(command, 0) != Some(HEAP_TEXTURE_SIZE_COMMAND)
        {
            return Ok(Err(format!(
                "not a heap texture size command of at least {HEAP_TEXTURE_SIZE_LENGTH:#x} bytes"
            )));
        }
        let serialized = command
            .get(8..)
            .and_then(|b| SerializedTextureDescriptor::read_from_prefix(b).ok())
            .map(|(d, _)| d)
            .unwrap_or_default();
        let Some(descriptor) = self.texture_descriptor_for_descriptor(&serialized)? else {
            crate::fatal!(DeathKind::NoHeapTextureDescriptor)
        };
        Ok(Ok(self
            .device
            .as_deref()
            .map_or(Ok(SizeAndAlign::default()), |d| {
                d.heap_texture_size_and_align(&descriptor)
            })?))
    }
}

/// Registers the new resource's buffer address when the task registers addresses; on a failed
/// registration the resource is released and the error returned.
///
/// A raise from the buffer getter (a legacy heap, which has no buffer) leaves the resource
/// unreleased: it is kept for good, with what it made.
fn register<P: Platform, E: CreateEnv<P>>(
    env: &E,
    resource: E::Res,
    reference: u32,
) -> RaiseResult<Result<E::Res, String>> {
    if env.register_gpu_addresses() {
        let buffer = match env.res_get_buffer(&resource) {
            Ok(b) => b,
            Err(raised) => {
                // Kept for good, with what it made (a legacy heap and its GPU VA range).
                keep_forever(resource, Kept::MetalNumbering);
                return Err(raised);
            }
        };
        let address = buffer.as_ref().map_or(0, |b| b.gpu_address());
        drop(buffer);
        if let Err(e) = env.register_gpu_address(address, reference) {
            drop(resource);
            return Ok(Err(e));
        }
    }
    Ok(Ok(resource))
}

pub(crate) fn create_buffer<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    reference: u32,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    // The record is read as 16 bytes, flags and padding included.
    let buffer = refuse!(
        cursor
            .read::<ObjectBuffer>()
            .map_err(|e| format!("buffer record not readable: {e}"))
    );
    let resource = env.new_buffer_resource(
        hold,
        u32::from(entry.object_type()).cast_signed(),
        buffer.as_bytes(),
        placement,
    )?;
    let Some(resource) = resource else {
        return Ok(Err("no host resource made for the buffer".to_owned()));
    };
    // Only when the task registers addresses.
    register(env, resource, reference)
}

pub(crate) fn create_heap<P: Platform, E: CreateEnv<P>>(
    env: &E,
    reference: u32,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    // The heap record is read as 16 bytes.
    let heap = refuse!(
        cursor
            .read::<ObjectHeap>()
            .map_err(|e| format!("heap record not readable: {e}"))
    );
    let resource = env.new_heap_resource(
        u32::from(entry.object_type()).cast_signed(),
        heap.as_bytes(),
        placement,
    )?;
    // An in-process shared heap is nil here.
    let Some(resource) = resource else {
        return Ok(Err("no host resource made for the heap".to_owned()));
    };
    register(env, resource, reference)
}

pub(crate) fn create_heap_buffer<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    _reference: u32,
    _entry: &ObjectEntry,
    mut cursor: Cursor,
    _placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let heap_buffer = refuse!(
        cursor
            .read::<ObjectHeapBuffer>()
            .map_err(|e| format!("heap buffer record not readable: {e}"))
    );
    // A heap buffer naming itself recurses without bound.
    let heap_reference = heap_buffer.heap_reference;
    let heap = refuse!(
        env.get_resource(hold, heap_reference)?
            .map_err(|e| format!("no base heap at reference {heap_reference}: {e}"))
    );
    // Reference 0 is nil, and every send below answers nil.
    let mtl_heap = match &heap {
        Some(h) => env.res_get_heap(h)?,
        None => None,
    };
    let options = mtl_heap
        .as_ref()
        .map_or(0, |h| h.storage_mode())
        .wrapping_shl(RESOURCE_STORAGE_MODE_SHIFT);
    let buffer = mtl_heap.as_ref().map_or(Ok(None), |h| {
        h.new_buffer_of_length(heap_buffer.length, options, heap_buffer.offset)
    })?;
    drop(mtl_heap);
    let Some(buffer) = buffer else {
        return Ok(Err("Metal made no buffer in the heap".to_owned()));
    };
    let resource = match &heap {
        Some(h) => env.new_child_buffer_resource(h, buffer)?,
        None => None,
    };
    // No address is registered
    match resource {
        Some(r) => Ok(Ok(r)),
        None => Ok(Err("no host resource made for the heap buffer".to_owned())),
    }
}

pub(crate) fn create_mapper_ref_buffer<P: Platform, E: CreateEnv<P>>(
    hold: Hold<'_>,
    env: &E,
    reference: u32,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<E::Res, String>> {
    let buffer = refuse!(
        cursor
            .read::<ObjectMapperRefBuffer>()
            .map_err(|e| format!("mapper-ref buffer record not readable: {e}"))
    );
    let resource = env.new_buffer_resource(
        hold,
        u32::from(entry.object_type()).cast_signed(),
        buffer.as_bytes(),
        placement,
    )?;
    let Some(resource) = resource else {
        return Ok(Err(
            "no host resource made for the mapper-ref buffer".to_owned()
        ));
    };
    // Before its surface is mapped the buffer is nil, and address 0 is registered.
    register(env, resource, reference)
}

#[cfg(test)]
mod tests {
    use super::test_env::*;
    use super::*;
    use crate::guest::GuestPtr;
    use crate::guest::records::ObjectEntry;
    use crate::memory::cursor::Cursor;
    use crate::platform::Handle;
    use crate::testkit::{Arg, Step};

    fn ok<T: std::fmt::Debug>(r: RaiseResult<Result<T, String>>) -> T {
        r.expect("no raise").expect("created")
    }

    fn err<T: std::fmt::Debug>(r: RaiseResult<Result<T, String>>) -> String {
        r.expect("no raise").expect_err("refused")
    }

    fn buffer_record() -> Bytes {
        Bytes::new().u64(0x2000).u32(7).u8(1).raw(&[0xa, 0xb, 0xc])
    }

    /// The whole 16-byte record, padding included, is the descriptor.
    #[test]
    fn makes_buffer() {
        let e = TestEnv::new();
        let (_m, g) = placement(0, 0);
        let r = ok(create_buffer(
            Hold::NONE,
            &e,
            3,
            &entry(1),
            buffer_record().cursor(),
            Some(g),
        ));
        assert_eq!(r.class, "newBufferResourceWithTask:");
        assert_eq!(
            e.calls("newBufferResourceWithTask:")[0].args,
            vec![Arg::I32(1), Arg::Bytes(buffer_record().0), Arg::Bool(true)]
        );
        // In-process (no registration): no getBuffer.
        assert_eq!(e.calls("getBuffer").len(), 0);
        assert_eq!(e.calls("registerGPUAddress").len(), 0);
        assert!(
            err(create_buffer(
                Hold::NONE,
                &e,
                3,
                &entry(1),
                cur(&buffer_record().0[..15]),
                None
            ))
            .starts_with("buffer record not readable: ")
        );
        e.p.script().push("newBufferResourceWithTask:", Step::Nil);
        assert_eq!(
            err(create_buffer(
                Hold::NONE,
                &e,
                3,
                &entry(1),
                buffer_record().cursor(),
                None
            )),
            "no host resource made for the buffer"
        );
    }

    /// The service registers the buffer's address.
    #[test]
    fn makes_buffer_registers_its_address() {
        let mut e = TestEnv::new();
        e.register = true;
        let r = ok(create_buffer(
            Hold::NONE,
            &e,
            3,
            &entry(1),
            buffer_record().cursor(),
            None,
        ));
        let address = r.buffer.as_ref().unwrap().gpu_address();
        assert_eq!(
            e.calls("registerGPUAddress")[0].args,
            vec![Arg::U64(address), Arg::U32(3)]
        );
        assert!(e.p.call_log().contains_in_order(&[
            "newBufferResourceWithTask:",
            "getBuffer",
            "registerGPUAddress"
        ]));
        *e.register_result.lock() = Err("Address already registered".to_owned());
        assert_eq!(
            err(create_buffer(
                Hold::NONE,
                &e,
                3,
                &entry(1),
                buffer_record().cursor(),
                None
            )),
            "Address already registered"
        );
    }

    /// The heap record is 16 bytes.
    /// A nil heap resource is "no host resource made for the heap".
    #[test]
    fn makes_heap() {
        let mut e = TestEnv::new();
        let rec = Bytes::new().u64(0x1111).u64(0x2222);
        let r = ok(create_heap(&e, 2, &entry(13), rec.cursor(), None));
        assert_eq!(r.class, "newHeapResourceWithTask:");
        assert_eq!(
            e.calls("newHeapResourceWithTask:")[0].args,
            vec![Arg::I32(13), Arg::Bytes(rec.0.clone()), Arg::Bool(false)]
        );
        assert_eq!(e.calls("registerGPUAddress").len(), 0);
        assert!(
            err(create_heap(&e, 2, &entry(13), cur(&rec.0[..15]), None))
                .starts_with("heap record not readable: ")
        );
        e.p.script().push("newHeapResourceWithTask:", Step::Nil);
        assert_eq!(
            err(create_heap(&e, 2, &entry(13), rec.cursor(), None)),
            "no host resource made for the heap"
        );
        // The service registers `[[resource getBuffer] gpuAddress]`.
        e.register = true;
        let r = ok(create_heap(&e, 2, &entry(13), rec.cursor(), None));
        let address = r.buffer.as_ref().unwrap().gpu_address();
        assert_eq!(
            e.calls("registerGPUAddress")[0].args,
            vec![Arg::U64(address), Arg::U32(2)]
        );
        *e.register_result.lock() = Err("dup".to_owned());
        assert_eq!(
            err(create_heap(&e, 2, &entry(13), rec.cursor(), None)),
            "dup"
        );
    }

    /// A mapper-ref buffer not yet mapped registers address 0.
    #[test]
    fn makes_mapper_ref_buffer() {
        let mut e = TestEnv::new();
        let rec = Bytes::new().u64(0x99);
        let r = ok(create_mapper_ref_buffer(
            Hold::NONE,
            &e,
            4,
            &entry(15),
            rec.cursor(),
            None,
        ));
        assert_eq!(r.class, "newBufferResourceWithTask:");
        assert_eq!(
            e.calls("newBufferResourceWithTask:")[0].args,
            vec![Arg::I32(15), Arg::Bytes(rec.0.clone()), Arg::Bool(false)]
        );
        assert!(
            err(create_mapper_ref_buffer(
                Hold::NONE,
                &e,
                4,
                &entry(15),
                cur(&rec.0[..7]),
                None
            ))
            .starts_with("mapper-ref buffer record not readable: ")
        );
        e.p.script().push("newBufferResourceWithTask:", Step::Nil);
        assert_eq!(
            err(create_mapper_ref_buffer(
                Hold::NONE,
                &e,
                4,
                &entry(15),
                rec.cursor(),
                None
            )),
            "no host resource made for the mapper-ref buffer"
        );
        e.register = true;
        e.p.call_log().clear();
        ok(create_mapper_ref_buffer(
            Hold::NONE,
            &e,
            4,
            &entry(15),
            rec.cursor(),
            None,
        ));
        assert_eq!(e.calls("registerGPUAddress").len(), 1);
        // A nil buffer's address is 0.
        let r = TestEnv::res(|_| ());
        let got = register(&e, r, 4);
        assert!(got.unwrap().is_ok());
        assert_eq!(
            e.calls("registerGPUAddress")[1].args,
            vec![Arg::U64(0), Arg::U32(4)]
        );
        *e.register_result.lock() = Err("dup".to_owned());
        assert_eq!(
            err(create_mapper_ref_buffer(
                Hold::NONE,
                &e,
                4,
                &entry(15),
                rec.cursor(),
                None
            )),
            "dup"
        );
    }

    fn heap_buffer_record(heap: u32, offset: u64, length: u64) -> Bytes {
        Bytes::new().u32(heap).u64(offset).u64(length)
    }

    #[test]
    fn makes_heap_buffer() {
        let mut e = TestEnv::new();
        e.register = true;
        let heap = e.heap(0x20); // private storage
        let hh = heap.clone_ref();
        e.object(5, Ok(Some(TestEnv::res(|r| r.heap = Some(hh)))));
        let (_m, g) = placement(9, 9);
        let r = ok(create_heap_buffer(
            Hold::NONE,
            &e,
            1,
            &entry(14),
            heap_buffer_record(5, 0x100, 0x80).cursor(),
            Some(g),
        ));
        assert_eq!(r.class, "newChildBufferResourceWithBuffer:");
        let c = e.calls("newBufferWithLength:options:offset:");
        assert_eq!(c[0].obj, heap.id());
        // storageMode << MTLResourceStorageModeShift.
        assert_eq!(
            c[0].args,
            vec![Arg::U64(0x80), Arg::U64(0x20), Arg::U64(0x100)]
        );
        assert!(e.p.call_log().contains_in_order(&[
            "getObject",
            "getHeap",
            "newBufferWithLength:options:offset:",
            "newChildBufferResourceWithBuffer:"
        ]));
        // Neither the placement nor an address.
        assert_eq!(read_placement(g), (9, 9));
        assert_eq!(e.calls("registerGPUAddress").len(), 0);
    }

    /// A heap of reference 0 is nil: "Metal made no buffer in the heap".
    #[test]
    fn makes_heap_buffer_refusals() {
        let e = TestEnv::new();
        let run = |b: Bytes| {
            err(create_heap_buffer(
                Hold::NONE,
                &e,
                1,
                &entry(14),
                b.cursor(),
                None,
            ))
        };
        assert!(
            run(Bytes(heap_buffer_record(5, 0, 0).0[..0x13].to_vec()))
                .starts_with("heap buffer record not readable: ")
        );
        e.object(5, Err("no".to_owned()));
        assert_eq!(
            run(heap_buffer_record(5, 0, 0x10)),
            "no base heap at reference 5: no"
        );
        assert_eq!(
            run(heap_buffer_record(0, 0, 0x10)),
            "Metal made no buffer in the heap"
        );
        assert_eq!(e.calls("newChildBufferResourceWithBuffer:").len(), 0);
        let heap = e.heap(0x20);
        e.object(6, Ok(Some(TestEnv::res(|r| r.heap = Some(heap)))));
        e.p.script()
            .push("newBufferWithLength:options:offset:", Step::Nil);
        assert_eq!(
            run(heap_buffer_record(6, 0, 0x10)),
            "Metal made no buffer in the heap"
        );
        e.p.script()
            .push("newChildBufferResourceWithBuffer:", Step::Nil);
        assert_eq!(
            run(heap_buffer_record(6, 0, 0x10)),
            "no host resource made for the heap buffer"
        );
        // A base that is not a heap: getHeap answers nil.
        e.object(7, Ok(Some(TestEnv::res(|_| ()))));
        assert_eq!(
            run(heap_buffer_record(7, 0, 0x10)),
            "Metal made no buffer in the heap"
        );
        // A raise from the lookup propagates.
        e.p.script()
            .push("getObject", Step::raise("NSGenericException", "x"));
        let hold = Hold::NONE;
        assert!(
            create_heap_buffer(
                hold,
                &e,
                1,
                &entry(14),
                heap_buffer_record(6, 0, 0x10).cursor(),
                None
            )
            .is_err()
        );
    }

    /// `TestEnv`, with the buffer getter raising as a legacy heap's does (it has no buffer).
    struct RaisingGetBuffer(TestEnv);

    mod raising_env {
        use super::*;
        use crate::resources::host::ResourceHost;
        use crate::resources::texture::backing::{BackingDescriptor, BackingPlane};
        use crate::resources::texture::dimension::{TextureDimensionRecord, TexturePagingInfo};
        use crate::testkit::{Fk, kind};

        impl CreateEnv<Fp> for RaisingGetBuffer {
            type Res = Res;

            fn platform(&self) -> &Arc<Fp> {
                self.0.platform()
            }
            fn device(&self) -> Option<&Fk<kind::Device>> {
                self.0.device()
            }
            fn host(&self) -> Arc<dyn ResourceHost<Fp>> {
                self.0.host()
            }
            fn register_gpu_addresses(&self) -> bool {
                self.0.register_gpu_addresses()
            }
            fn register_gpu_address(&self, address: u64, reference: u32) -> Result<(), String> {
                self.0.register_gpu_address(address, reference)
            }
            fn get_resource(
                &self,
                hold: Hold<'_>,
                reference: u32,
            ) -> RaiseResult<Result<Option<Res>, String>> {
                self.0.get_resource(hold, reference)
            }
            fn backing_for_id(&self, hold: Hold<'_>, identifier: u32) -> RaiseResult<Option<Res>> {
                self.0.backing_for_id(hold, identifier)
            }
            fn create_indirect_command_buffer(
                &self,
                reference: u32,
                entry: &ObjectEntry,
                cursor: Cursor,
                placement: Option<GuestPtr>,
            ) -> RaiseResult<Result<Res, String>> {
                self.0
                    .create_indirect_command_buffer(reference, entry, cursor, placement)
            }
            fn res_get_buffer(&self, r: &Res) -> RaiseResult<Option<Fk<kind::Buffer>>> {
                self.0.res_get_buffer(r)?;
                crate::resources::raise(crate::resources::RaiseKind::Unrecognized(
                    crate::resources::Op::GetBuffer,
                ))
            }
            fn res_get_texture(&self, r: &Res) -> RaiseResult<Option<Fk<kind::Texture>>> {
                self.0.res_get_texture(r)
            }
            fn res_get_heap(&self, r: &Res) -> RaiseResult<Option<Fk<kind::Heap>>> {
                self.0.res_get_heap(r)
            }
            fn res_io_surface(&self, r: &Res) -> RaiseResult<Option<Fk<kind::Surface>>> {
                self.0.res_io_surface(r)
            }
            fn new_child_buffer_resource(
                &self,
                r: &Res,
                buffer: Fk<kind::Buffer>,
            ) -> RaiseResult<Option<Res>> {
                self.0.new_child_buffer_resource(r, buffer)
            }
            fn new_child_texture_resource(
                &self,
                r: &Res,
                texture: Option<Fk<kind::Texture>>,
            ) -> RaiseResult<Option<Res>> {
                self.0.new_child_texture_resource(r, texture)
            }
            fn new_buffer_resource(
                &self,
                hold: Hold<'_>,
                object_type: i32,
                descriptor: &[u8],
                placement: Option<GuestPtr>,
            ) -> RaiseResult<Option<Res>> {
                self.0
                    .new_buffer_resource(hold, object_type, descriptor, placement)
            }
            fn new_heap_resource(
                &self,
                object_type: i32,
                descriptor: &[u8],
                placement: Option<GuestPtr>,
            ) -> RaiseResult<Option<Res>> {
                self.0.new_heap_resource(object_type, descriptor, placement)
            }
            fn create_texture_resource(
                &self,
                object_type: u32,
                paging_info: &TexturePagingInfo,
                dimension: &TextureDimensionRecord,
                dimension_length: u64,
                texture: Fk<kind::Texture>,
            ) -> RaiseResult<Option<Res>> {
                self.0.create_texture_resource(
                    object_type,
                    paging_info,
                    dimension,
                    dimension_length,
                    texture,
                )
            }
            fn create_shared_texture_resource(
                &self,
                object_type: u32,
                paging_info: &TexturePagingInfo,
                dimension: &TextureDimensionRecord,
                dimension_length: u64,
                texture: Fk<kind::Texture>,
            ) -> RaiseResult<Option<Res>> {
                self.0.create_shared_texture_resource(
                    object_type,
                    paging_info,
                    dimension,
                    dimension_length,
                    texture,
                )
            }
            fn new_discardable_serializer_texture_resource(
                &self,
                texture: Option<Fk<kind::Texture>>,
            ) -> RaiseResult<Option<Res>> {
                self.0.new_discardable_serializer_texture_resource(texture)
            }
            fn new_mapper_ref_texture_resource(
                &self,
                mapping_id: u64,
                texture: Fk<kind::Texture>,
                surface_base: u64,
                surface_size: u64,
            ) -> RaiseResult<Option<Res>> {
                self.0
                    .new_mapper_ref_texture_resource(mapping_id, texture, surface_base, surface_size)
            }
            fn new_shared_texture_resource(
                &self,
                texture: Fk<kind::Texture>,
                shared_texture_backing_id: u32,
            ) -> RaiseResult<Option<Res>> {
                self.0
                    .new_shared_texture_resource(texture, shared_texture_backing_id)
            }
            fn new_ref_texture_resource(
                &self,
                object_type: u32,
                texture: Fk<kind::Texture>,
                backing: Res,
                plane: u16,
            ) -> RaiseResult<Option<Res>> {
                self.0
                    .new_ref_texture_resource(object_type, texture, backing, plane)
            }
            fn new_backing_resource(
                &self,
                descriptor: &BackingDescriptor,
                planes: &[BackingPlane],
            ) -> RaiseResult<Option<Res>> {
                self.0.new_backing_resource(descriptor, planes)
            }
        }
    }

    /// When the buffer getter raises (the service's legacy heap has none), the raise leaves the
    /// heap's creation with the new resource never released, so the resource and its heap leak.
    #[test]
    fn makes_heap_get_buffer_raise_leaks_the_resource() {
        let mut inner = TestEnv::new();
        inner.register = true;
        let e = RaisingGetBuffer(inner);
        let rec = Bytes::new().u64(0x1111).u64(0x2222);
        let hold = Hold::NONE;
        let got = create_heap(&e, 2, &entry(13), rec.cursor(), None);
        assert!(got.is_err(), "the getBuffer raise propagates");
        assert_eq!(e.0.calls("registerGPUAddress").len(), 0);
        let made = e.0.calls("newHeapResourceWithTask:");
        assert_eq!(made.len(), 1);
        assert_eq!(e.0.calls("getBuffer").len(), 1);
        // Same for createBuffer.
        let got = create_buffer(hold, &e, 3, &entry(1), buffer_record().cursor(), None);
        assert!(got.is_err());
        // A resource `register` is handed and that raises is forgotten, not dropped.
        let r = TestEnv::res(|_| ());
        let weak = Arc::downgrade(&r);
        assert!(register(&e, r, 4).is_err());
        assert!(weak.upgrade().is_some(), "the +1 leaks");
        assert_eq!(e.0.calls("registerGPUAddress").len(), 0);
    }
}
