//! A [`CreateEnv`] over the fake, for the creators' tests: the manager's, the task's and the
//! resources area's calls go into the platform's call log with class `"Env"` (the task's through
//! a [`FakeResourceHost`], class `"ResourceHost"`), and the script decides each. Also the guest
//! record builders and placement slots the tests use.

use std::sync::Arc;

use zerocopy::IntoBytes;

use super::CreateEnv;
use crate::deserializer::texture_descriptor::{
    SerializedTextureDescriptor, SerializedTextureDescriptor2,
};
use crate::guest::GuestPtr;
use crate::guest::records::{GuestFeatures, ObjectEntry, SharedFeatures};
use crate::host::HostPtr;
use crate::memory::cursor::Cursor;
use crate::platform::device::GpuDevice;
use crate::platform::values::{HeapDesc, OptionSet};
use crate::platform::{Handle, ObjId};
use crate::raise::RaiseResult;
use crate::resources::host::ResourceHost;
use crate::resources::manager::DeathKind;
use crate::resources::manager::records::{PLACEMENT_FIELD0, PLACEMENT_RESOURCE_ID};
use crate::resources::texture::backing::{BackingDescriptor, BackingPlane};
use crate::resources::texture::dimension::{TextureDimensionRecord, TexturePagingInfo};
use crate::rt::Hold;
use crate::testkit::guest_ram::RawRegion;
use crate::testkit::impls::{Backing, new_buffer, new_texture};
use crate::testkit::objects::TexDesc;
use crate::testkit::resource_host::FakeResourceHost;

pub(crate) use crate::resources::resource::test_support::logged;
use crate::testkit::{Arg, Call, FakePlatform, Fk, Policy, Step, kind};
use parking_lot::Mutex;

/// The fake platform.
pub type Fp = FakePlatform;

/// A resource the environment made: the class that made it and what it holds.
#[derive(Debug, Default)]
pub struct TRes {
    pub class: &'static str,
    pub buffer: Option<Fk<kind::Buffer>>,
    pub texture: Option<Fk<kind::Texture>>,
    pub heap: Option<Fk<kind::Heap>>,
    pub surface: Option<Fk<kind::Surface>>,
    pub object_type: u32,
    pub record: Option<(TextureDimensionRecord, u64, TexturePagingInfo)>,
    pub backing: Option<Res>,
    pub plane: u16,
    pub value: u64,
}

/// The environment's resource.
pub type Res = Arc<TRes>;

/// What the resource lookup answers for a reference.
pub type Lookup = (u32, Result<Option<Res>, String>);

/// The tests' environment.
pub struct TestEnv {
    pub p: Arc<Fp>,
    pub device: Fk<kind::Device>,
    pub rec: Arc<FakeResourceHost>,
    pub register: bool,
    pub objects: Mutex<Vec<Lookup>>,
    pub backings: Mutex<Vec<(u32, Res)>>,
    pub register_result: Mutex<Result<(), String>>,
}

impl TestEnv {
    /// A fresh process.
    pub fn new() -> TestEnv {
        let p = Arc::new(FakePlatform::new());
        let rec = FakeResourceHost::new(p.clone());
        TestEnv {
            device: p.device(),
            p,
            rec,
            register: false,
            objects: Mutex::new(Vec::new()),
            backings: Mutex::new(Vec::new()),
            register_result: Mutex::new(Ok(())),
        }
    }

    /// The task's features.
    pub fn features(&self, f: &GuestFeatures) {
        let s = SharedFeatures::new();
        s.store(f);
        self.rec.set_features(Arc::new(s));
    }

    /// What the resource lookup answers for `reference`.
    pub fn object(&self, reference: u32, r: Result<Option<Res>, String>) {
        self.objects.lock().push((reference, r));
    }

    /// A resource holding `f`'s objects.
    pub fn res(f: impl FnOnce(&mut TRes)) -> Res {
        let mut r = TRes {
            class: "test",
            ..TRes::default()
        };
        f(&mut r);
        Arc::new(r)
    }

    /// A texture of the fake (`MTLTextureType2D`, RGBA8).
    pub fn texture(&self) -> Fk<kind::Texture> {
        new_texture(self.p.world(), TexDesc::default(), None, 0, None, false)
    }

    /// A buffer of the fake.
    pub fn buffer(&self, options: u64) -> Fk<kind::Buffer> {
        new_buffer(self.p.world(), 0x1000, options, Backing::Fresh, None).unwrap()
    }

    /// A heap of the fake with `options`.
    pub fn heap(&self, options: u64) -> Fk<kind::Heap> {
        let made = self
            .p
            .device()
            .new_heap(&HeapDesc {
                size: 0x10000,
                heap_type: 0,
                options: vec![OptionSet::ResourceOptions(options)],
                address_ranges: None,
                pinned_gpu_address: None,
            })
            .unwrap();
        made.heap.unwrap()
    }

    /// The calls to `sel`.
    pub fn calls(&self, sel: &str) -> Vec<Call> {
        self.p.call_log().calls_to(sel)
    }

    fn step(&self, sel: &'static str, args: Vec<Arg>) -> Step {
        self.p.call_log().push(ObjId(0), "Env", sel, args);
        self.p.script().next(ObjId(0), sel)
    }

    fn make(
        &self,
        sel: &'static str,
        args: Vec<Arg>,
        r: impl FnOnce() -> TRes,
    ) -> RaiseResult<Option<Res>> {
        let s = self.step(sel, args);
        s.throws(Policy::Caught, sel)?;
        if s.is_nil() {
            return Ok(None);
        }
        let mut r = r();
        r.class = sel;
        Ok(Some(Arc::new(r)))
    }
}

/// A cursor over `bytes` (copied; the cursor keeps them).
pub fn cur(bytes: &[u8]) -> Cursor {
    let v = Arc::new(bytes.to_vec());
    let ptr = HostPtr::from_ptr(v.as_ptr().cast_mut());
    let len = v.len() as u64;
    Cursor::with_pointer(v, ptr, len)
}

/// A placement entry `{field0, resourceID}` in fake guest memory.
pub fn placement(field0: u64, resource_id: u64) -> (Arc<RawRegion>, GuestPtr) {
    let r = RawRegion::new(16).unwrap();
    let g = GuestPtr::new(r.addr());
    g.write_u64(PLACEMENT_FIELD0, field0, DeathKind::NullPlacement);
    g.write_u64(PLACEMENT_RESOURCE_ID, resource_id, DeathKind::NullPlacement);
    (r, g)
}

/// The placement entry's two words.
pub fn read_placement(g: GuestPtr) -> (u64, u64) {
    (
        g.read_u64(PLACEMENT_FIELD0, DeathKind::NullPlacement),
        g.read_u64(PLACEMENT_RESOURCE_ID, DeathKind::NullPlacement),
    )
}

/// The word `race_after_load` names for the slot's first word (the index).
pub const FIELD0: u64 = PLACEMENT_FIELD0;
/// The word `race_after_load` names for the slot's GPU resource ID.
pub const RESOURCE_ID: u64 = PLACEMENT_RESOURCE_ID;

std::thread_local! {
    /// The guest stores racing the host's placement loads on this thread, in order.
    static RACES: std::cell::RefCell<Vec<Race>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A guest store that lands right after one of the host's placement loads.
#[derive(Debug)]
struct Race {
    /// The placement word (`FIELD0` or `RESOURCE_ID`).
    offset: u64,
    /// Loads of that word still to come; the store follows the last of them.
    loads: u32,
    /// What the guest stores.
    value: u64,
}

/// Drops this thread's pending races, so a failed test leaves none for the next one.
#[must_use]
#[derive(Debug)]
pub struct Races;

impl Drop for Races {
    fn drop(&mut self) {
        RACES.with(|r| r.borrow_mut().clear());
    }
}

/// The guest stores `value` into the placement word at `offset` right after the host's
/// `nth` load of that word on this thread: a vCPU racing the host, pinned between two of
/// the host's loads. Races on one word fire in the order they were set, the count of each
/// starting after the one before it fired. Pending races go when the guard drops.
pub fn race_after_load(offset: u64, nth: u32, value: u64) -> Races {
    RACES.with(|r| {
        r.borrow_mut().push(Race {
            offset,
            loads: nth,
            value,
        });
    });
    Races
}

/// Races still waiting for their load.
pub fn races_pending() -> usize {
    RACES.with(|r| r.borrow().len())
}

/// After every placement load (`records::placement_load`): counts it against the first race on
/// that word, and makes the guest's store when that race is due.
pub(crate) fn after_placement_load(p: GuestPtr, offset: u64) {
    let due = RACES.with(|r| {
        let mut r = r.borrow_mut();
        let i = r.iter().position(|race| race.offset == offset)?;
        r[i].loads -= 1;
        (r[i].loads == 0).then(|| r.remove(i).value)
    });
    if let Some(v) = due {
        p.write_u64(offset, v, DeathKind::NullPlacement);
    }
}

/// An object entry of `object_type`.
pub fn entry(object_type: u8) -> ObjectEntry {
    ObjectEntry {
        type_and_length: u32::from(object_type),
        offset: 0,
    }
}

/// A little-endian byte builder for guest records.
#[derive(Debug, Default, Clone)]
pub struct Bytes(pub Vec<u8>);

impl Bytes {
    pub fn new() -> Bytes {
        Bytes(Vec::new())
    }
    pub fn u8(mut self, v: u8) -> Bytes {
        self.0.push(v);
        self
    }
    pub fn u16(mut self, v: u16) -> Bytes {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u32(mut self, v: u32) -> Bytes {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u64(mut self, v: u64) -> Bytes {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f32(mut self, v: f32) -> Bytes {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn raw(mut self, v: &[u8]) -> Bytes {
        self.0.extend_from_slice(v);
        self
    }
    pub fn zeros(mut self, n: usize) -> Bytes {
        self.0.resize(self.0.len() + n, 0);
        self
    }
    /// An operation header with `command_type`.
    pub fn header(self, command_type: u32) -> Bytes {
        self.u32(command_type).u32(0)
    }
    pub fn cursor(&self) -> Cursor {
        cur(&self.0)
    }
}

/// A v1 descriptor: 2D (or `texture_type`), `pixel_format`, `w`x`h`, one mip, one sample,
/// one slice, `ShaderRead`, `resource_options`.
pub fn desc1(
    texture_type: u8,
    pixel_format: u16,
    w: u32,
    h: u32,
    resource_options: u16,
) -> SerializedTextureDescriptor {
    SerializedTextureDescriptor {
        bits: u32::from(texture_type) | (1 << 8) | (u32::from(pixel_format) << 16),
        width: w,
        height: h,
        depth: 1,
        mipmap_level_count: 1,
        sample_count: 1,
        array_length: 1,
        resource_options,
        protection_options: 0,
    }
}

/// A v2 descriptor like [`desc1`], identity swizzle.
pub fn desc2(
    texture_type: u8,
    pixel_format: u16,
    w: u32,
    h: u32,
    resource_options: u16,
) -> SerializedTextureDescriptor2 {
    let pf = pixel_format.to_le_bytes();
    SerializedTextureDescriptor2 {
        bits: [texture_type, pf[0], pf[1]],
        usage: 1,
        width: w,
        height: h,
        depth: 1,
        mipmap_level_count: 1,
        sample_count: 1,
        array_length: 1,
        resource_options,
        protection_options: 0,
        swizzle_red: 2,
        swizzle_green: 3,
        swizzle_blue: 4,
        swizzle_alpha: 5,
    }
}

impl CreateEnv<Fp> for TestEnv {
    type Res = Res;

    fn platform(&self) -> &Arc<Fp> {
        &self.p
    }
    fn device(&self) -> Option<&Fk<kind::Device>> {
        Some(&self.device)
    }
    fn host(&self) -> Arc<dyn ResourceHost<Fp>> {
        self.rec.clone()
    }
    fn register_gpu_addresses(&self) -> bool {
        self.register
    }
    fn register_gpu_address(&self, address: u64, reference: u32) -> Result<(), String> {
        self.step(
            "registerGPUAddress",
            vec![Arg::U64(address), Arg::U32(reference)],
        );
        self.register_result.lock().clone()
    }
    fn get_resource(
        &self,
        _hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<Res>, String>> {
        let s = self.step("getObject", vec![Arg::U32(reference)]);
        s.throws(Policy::Caught, "getObject")?;
        let g = self.objects.lock();
        let found = g.iter().find(|(r, _)| *r == reference);
        Ok(match found {
            Some((_, Ok(o))) => Ok(o.clone()),
            Some((_, Err(e))) => Err(e.clone()),
            None => Ok(None),
        })
    }
    fn backing_for_id(&self, _hold: Hold<'_>, identifier: u32) -> RaiseResult<Option<Res>> {
        let s = self.step("backingForID:", vec![Arg::U32(identifier)]);
        s.throws(Policy::Caught, "backingForID:")?;
        let g = self.backings.lock();
        Ok(g.iter()
            .find(|(i, _)| *i == identifier)
            .map(|(_, r)| r.clone()))
    }
    fn create_indirect_command_buffer(
        &self,
        reference: u32,
        _entry: &ObjectEntry,
        _cursor: Cursor,
        _placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Res, String>> {
        self.step("createIndirectCommandBuffer", vec![Arg::U32(reference)]);
        Ok(Ok(TestEnv::res(|r| r.class = "icb")))
    }
    fn res_get_buffer(&self, r: &Res) -> RaiseResult<Option<Fk<kind::Buffer>>> {
        self.step("getBuffer", vec![Arg::Str(r.class.to_owned())]);
        Ok(r.buffer.as_ref().map(Handle::clone_ref))
    }
    fn res_get_texture(&self, r: &Res) -> RaiseResult<Option<Fk<kind::Texture>>> {
        self.step("getTexture", vec![Arg::Str(r.class.to_owned())]);
        Ok(r.texture.as_ref().map(Handle::clone_ref))
    }
    fn res_get_heap(&self, r: &Res) -> RaiseResult<Option<Fk<kind::Heap>>> {
        self.step("getHeap", vec![Arg::Str(r.class.to_owned())]);
        Ok(r.heap.as_ref().map(Handle::clone_ref))
    }
    fn res_io_surface(&self, r: &Res) -> RaiseResult<Option<Fk<kind::Surface>>> {
        self.step("ioSurface", vec![Arg::Str(r.class.to_owned())]);
        Ok(r.surface.as_ref().map(Handle::clone_ref))
    }
    fn new_child_buffer_resource(
        &self,
        _r: &Res,
        buffer: Fk<kind::Buffer>,
    ) -> RaiseResult<Option<Res>> {
        self.make(
            "newChildBufferResourceWithBuffer:",
            vec![Arg::obj(&buffer)],
            || TRes {
                buffer: Some(buffer),
                ..TRes::default()
            },
        )
    }
    fn new_child_texture_resource(
        &self,
        _r: &Res,
        texture: Option<Fk<kind::Texture>>,
    ) -> RaiseResult<Option<Res>> {
        self.make(
            "newChildTextureResourceWithTexture:",
            vec![Arg::opt(texture.as_ref())],
            || TRes {
                texture,
                ..TRes::default()
            },
        )
    }
    fn new_buffer_resource(
        &self,
        _hold: Hold<'_>,
        object_type: i32,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Res>> {
        let buffer = self.buffer(0);
        self.make(
            "newBufferResourceWithTask:",
            vec![
                Arg::I32(object_type),
                Arg::Bytes(descriptor.to_vec()),
                Arg::Bool(placement.is_some()),
            ],
            || TRes {
                buffer: Some(buffer),
                ..TRes::default()
            },
        )
    }
    fn new_heap_resource(
        &self,
        object_type: i32,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Res>> {
        let buffer = self.buffer(0);
        self.make(
            "newHeapResourceWithTask:",
            vec![
                Arg::I32(object_type),
                Arg::Bytes(descriptor.to_vec()),
                Arg::Bool(placement.is_some()),
            ],
            || TRes {
                buffer: Some(buffer),
                ..TRes::default()
            },
        )
    }
    fn create_texture_resource(
        &self,
        object_type: u32,
        paging_info: &TexturePagingInfo,
        dimension: &TextureDimensionRecord,
        dimension_length: u64,
        texture: Fk<kind::Texture>,
    ) -> RaiseResult<Option<Res>> {
        self.make(
            "createTextureResourceWithTask:",
            vec![
                Arg::U32(object_type),
                Arg::val(paging_info),
                Arg::U64(dimension_length),
                Arg::obj(&texture),
            ],
            || TRes {
                object_type,
                texture: Some(texture),
                record: Some((dimension.clone(), dimension_length, *paging_info)),
                ..TRes::default()
            },
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
        self.make(
            "createSharedTextureResourceWithTask:",
            vec![
                Arg::U32(object_type),
                Arg::val(paging_info),
                Arg::U64(dimension_length),
                Arg::obj(&texture),
            ],
            || TRes {
                object_type,
                texture: Some(texture),
                record: Some((dimension.clone(), dimension_length, *paging_info)),
                ..TRes::default()
            },
        )
    }
    fn new_discardable_serializer_texture_resource(
        &self,
        texture: Option<Fk<kind::Texture>>,
    ) -> RaiseResult<Option<Res>> {
        self.make(
            "new_discardable_serializer_texture_resource",
            vec![Arg::opt(texture.as_ref())],
            || TRes {
                texture,
                ..TRes::default()
            },
        )
    }
    fn new_mapper_ref_texture_resource(
        &self,
        mapping_id: u64,
        texture: Fk<kind::Texture>,
        surface_base: u64,
        surface_size: u64,
    ) -> RaiseResult<Option<Res>> {
        self.make(
            "new_mapper_ref_texture_resource",
            vec![
                Arg::U64(mapping_id),
                Arg::obj(&texture),
                Arg::U64(surface_base),
                Arg::U64(surface_size),
            ],
            || TRes {
                texture: Some(texture),
                value: mapping_id,
                ..TRes::default()
            },
        )
    }
    fn new_shared_texture_resource(
        &self,
        texture: Fk<kind::Texture>,
        shared_texture_backing_id: u32,
    ) -> RaiseResult<Option<Res>> {
        self.make(
            "new_shared_texture_resource",
            vec![Arg::obj(&texture), Arg::U32(shared_texture_backing_id)],
            || TRes {
                texture: Some(texture),
                value: u64::from(shared_texture_backing_id),
                ..TRes::default()
            },
        )
    }
    fn new_ref_texture_resource(
        &self,
        object_type: u32,
        texture: Fk<kind::Texture>,
        backing: Res,
        plane: u16,
    ) -> RaiseResult<Option<Res>> {
        self.make(
            "new_ref_texture_resource",
            vec![
                Arg::U32(object_type),
                Arg::obj(&texture),
                Arg::Str(backing.class.to_owned()),
                Arg::U16(plane),
            ],
            || TRes {
                object_type,
                texture: Some(texture),
                backing: Some(backing),
                plane,
                ..TRes::default()
            },
        )
    }
    fn new_backing_resource(
        &self,
        descriptor: &BackingDescriptor,
        planes: &[BackingPlane],
    ) -> RaiseResult<Option<Res>> {
        self.make(
            "new_backing_resource",
            vec![
                Arg::val(descriptor),
                Arg::Bytes(planes.as_bytes().to_vec()),
                Arg::U64(planes.len() as u64),
            ],
            TRes::default,
        )
    }
}
