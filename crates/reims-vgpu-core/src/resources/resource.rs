//! [`Resource`]: a guest object's validity flags and its kind, shared as `Arc<Resource<P>>` (the
//! manager's map, the uses a command keeps alive, children holding their base). Its last
//! release is the kind's `Drop`.
//!
//! [`ResourceKind`] is a closed enum; [`ResourceClass`] is what each kind answers. The trait's
//! default methods are the answers of a kind that does not override them: the derived validity
//! predicates, the flag accessors, and [`RaiseKind::Unrecognized`] for every operation a kind has
//! no body for. `Resource`'s methods dispatch through the kind and hand it the whole resource,
//! so a default that asks the resource itself (needs sync, needs prepare, evict) asks the
//! overriding kind too.
//!
//! A kind reaches the process's platform (the VM, Metal's descriptor classes) through
//! [`platform_for`]: the scope the caller opens with [`with_platform`] (the task's entry points,
//! the manager). A kind made outside a scope refuses what needs a descriptor or a VM allocation.
//! The kinds log through the `log` crate, scope or not.

use std::any::Any;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

use zerocopy::{FromBytes, IntoBytes};

use crate::guest::records::GuestFeatures;
use crate::platform::Platform;
use crate::raise::RaiseResult;
use crate::resources::buffer::child::HeapBufferResource;
use crate::resources::buffer::detach::DetachBufferResource;
use crate::resources::buffer::mapper_ref::MapperRefBufferResource;
use crate::resources::buffer::private::PrivateBufferResource;
use crate::resources::buffer::shared::SharedBufferResource;
use crate::resources::heap::detach::DetachHeapResource;
use crate::resources::heap::legacy::LegacyHeapResource;
use crate::resources::host::ResourceHost;
use crate::resources::icb::IcbResource;
use crate::resources::texture::TextureResource;
use crate::resources::texture::backing::BackingResource;
use crate::resources::texture::mapper::MapperRefTextureResource;
use crate::resources::texture::shared::SharedTextureResource;
use crate::resources::texture::views::{RefTextureResource, SerializerTextureResource};
use crate::resources::{DeathKind, Op, RaiseKind};
use crate::rt::Hold;

pub use crate::resources::validity::ResourceState;

/// Whichever Metal resource a resource is (`Resource::get_resource`), as the deserializer's
/// untyped lookup hands it out.
pub enum AnyResource<P: Platform> {
    /// An `MTLBuffer`.
    Buffer(P::Buffer),
    /// An `MTLTexture`.
    Texture(P::Texture),
    /// An `MTLHeap`.
    Heap(P::Heap),
    /// An `MTLIndirectCommandBuffer`.
    Icb(P::Icb),
}

impl<P: Platform> std::fmt::Debug for AnyResource<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::platform::Handle;
        match self {
            AnyResource::Buffer(b) => f.debug_tuple("Buffer").field(&b.id()).finish(),
            AnyResource::Texture(t) => f.debug_tuple("Texture").field(&t.id()).finish(),
            AnyResource::Heap(h) => f.debug_tuple("Heap").field(&h.id()).finish(),
            AnyResource::Icb(i) => f.debug_tuple("Icb").field(&i.id()).finish(),
        }
    }
}

/// The resource kinds, each holding its own state. Which operations each answers is
/// [`ResourceClass`].
pub enum ResourceKind<P: Platform> {
    /// A no-copy buffer over guest pages.
    SharedBuffer(SharedBufferResource<P>),
    /// A private buffer paged by blits.
    PrivateBuffer(PrivateBufferResource<P>),
    /// A buffer whose backing is detached and replaced by the guest's address ranges.
    DetachBuffer(DetachBufferResource<P>),
    /// A buffer made in a heap (discardable or not).
    HeapBuffer(HeapBufferResource<P>),
    /// A buffer over an IOSurface the mapper names.
    MapperRefBuffer(MapperRefBufferResource<P>),
    /// A private heap paged by blits.
    LegacyHeap(LegacyHeapResource<P>),
    /// A heap whose backing is detached and replaced by the guest's address ranges.
    DetachHeap(DetachHeapResource<P>),
    /// An indirect command buffer.
    Icb(IcbResource<P>),
    /// A texture paged from guest memory.
    Texture(TextureResource<P>),
    /// A child texture made over a base's texture (discardable or not).
    SerializerTexture(SerializerTextureResource<P>),
    /// A texture over a backing.
    RefTexture(RefTextureResource<P>),
    /// A texture over an IOSurface the mapper names.
    MapperRefTexture(MapperRefTextureResource<P>),
    /// A texture shared through a shared-texture handle.
    SharedTexture(SharedTextureResource<P>),
    /// An IOSurface backing.
    Backing(BackingResource<P>),
}

/// A guest object with host backing. The last release of its `Arc` is its kind's teardown.
pub struct Resource<P: Platform> {
    /// The validity flags.
    pub state: ResourceState,
    /// The kind and its state.
    pub kind: ResourceKind<P>,
}

impl<P: Platform> std::fmt::Debug for Resource<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resource")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

/// The task a resource holds, unretained: a failed upgrade is a torn-down task, which answers
/// as nil does (nothing made, zero, NO).
pub type TaskRef<P> = Weak<dyn ResourceHost<P>>;

/// What a resource kind answers. The defaults are the answers of a kind that has no body of its
/// own; each kind overrides exactly what it answers. `this` is the resource the method runs on:
/// its flags, and the receiver when a default asks the resource.
pub trait ResourceClass<P: Platform>: Send + Sync {
    /// DIAG: the class's type name, for timing lines.
    fn diag_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    /// The guest-valid flag a new resource of this kind starts with (host-valid and volatile
    /// always start false).
    fn initial_guest_valid(&self) -> bool {
        false
    }

    /// Whether the guest's copy is current.
    fn is_guest_valid(&self, this: &Resource<P>) -> bool {
        this.state.guest_valid()
    }
    /// Marks the guest's copy current or stale.
    fn set_is_guest_valid(&self, this: &Resource<P>, v: bool) {
        this.state.set_guest_valid(v)
    }
    /// Whether the host's copy is current.
    fn is_host_valid(&self, this: &Resource<P>) -> bool {
        this.state.host_valid()
    }
    /// Marks the host's copy current or stale.
    fn set_is_host_valid(&self, this: &Resource<P>, v: bool) {
        this.state.set_host_valid(v)
    }
    /// Whether opcode 63 may discard it.
    fn supports_discard(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }
    /// Whether it pages through a compute encoder. No kind here does.
    fn needs_compute_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }
    /// Whether the guest's copy needs writing back.
    fn needs_sync(&self, this: &Resource<P>) -> bool {
        // Two questions to the resource, guest first, short-circuit.
        !this.is_guest_valid() && this.is_host_valid()
    }
    /// Whether the host's copy needs making or refreshing.
    /// Makes a volatile resource non-volatile on the CPU, as its prepare does first, with no
    /// encoder ([`Resource::devolatilize_now`]). Nothing for a kind that is never volatile.
    fn devolatilize_now(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        Ok(())
    }
    fn needs_prepare(&self, this: &Resource<P>) -> bool {
        // A question to the resource, then the raw volatile flag.
        !this.is_host_valid() || this.state.is_volatile()
    }
    /// Whether it pages on the CPU rather than through an encoder.
    fn needs_cpu_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }
    /// Guest to host through a blit encoder. `encoder` is `None` when Metal gave no encoder,
    /// which nothing checks: the sends are skipped and the flag writes still happen.
    fn prepare_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = encoder;
        this.unrecognized(Op::Prepare)
    }
    /// Host to guest through a blit encoder (`None` as for a prepare).
    fn synchronize_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = encoder;
        this.unrecognized(Op::Synchronize)
    }
    /// Guest to host through a compute encoder (`None` as for a prepare).
    fn prepare_in_compute_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        let _ = encoder;
        this.unrecognized(Op::PrepareCompute)
    }
    /// Host to guest through a compute encoder (`None` as for a prepare).
    fn synchronize_in_compute_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        let _ = encoder;
        this.unrecognized(Op::SynchronizeCompute)
    }
    /// Opcode 63's discard.
    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        this.unrecognized(Op::Discard)
    }
    /// The Metal object an encoder binds.
    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        this.unrecognized(Op::GetResource)
    }
    /// The Metal buffer, retained.
    fn get_buffer(&self, this: &Resource<P>) -> RaiseResult<Option<P::Buffer>> {
        this.unrecognized(Op::GetBuffer)
    }
    /// The Metal texture, retained.
    fn get_texture(&self, this: &Resource<P>) -> RaiseResult<Option<P::Texture>> {
        this.unrecognized(Op::GetTexture)
    }
    /// The Metal heap, retained.
    fn get_heap(&self, this: &Resource<P>) -> RaiseResult<Option<P::Heap>> {
        this.unrecognized(Op::GetHeap)
    }
    /// The Metal indirect command buffer, retained.
    fn get_indirect_command_buffer(&self, this: &Resource<P>) -> RaiseResult<Option<P::Icb>> {
        this.unrecognized(Op::GetIcb)
    }
    /// Writes back through a blit encoder when the raw flags (not the kind's answers) say the
    /// guest's copy is stale and the host's current.
    fn evict_in_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        if !this.state.guest_valid() && this.state.host_valid() {
            this.synchronize_in_encoder(hold, encoder)?;
        }
        Ok(())
    }
    /// The same through a compute encoder.
    fn evict_in_compute_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        if !this.state.guest_valid() && this.state.host_valid() {
            this.synchronize_in_compute_encoder(hold, encoder)?;
        }
        Ok(())
    }
    /// The shared texture backing's id: 0 for every kind but a texture's.
    fn get_shared_texture_backing_id(&self, this: &Resource<P>) -> RaiseResult<u32> {
        let _ = this;
        Ok(0)
    }
    /// A child buffer over `buffer`, made from a heap: none for other kinds. `buffer` is the
    /// caller's reference, released when this returns.
    fn new_child_buffer_resource_with_buffer(
        &self,
        this: &Arc<Resource<P>>,
        buffer: P::Buffer,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let _ = this;
        drop(buffer);
        Ok(None)
    }
    /// Guest to host on the CPU.
    fn prepare_with_cpu(&self, _hold: Hold<'_>, this: &Resource<P>) -> RaiseResult<()> {
        this.unrecognized(Op::PrepareCpu)
    }
    /// Host to guest on the CPU.
    fn synchronize_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        this.unrecognized(Op::SynchronizeCpu)
    }
    /// Writes back on the CPU before an eviction.
    fn evict_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        this.unrecognized(Op::EvictCpu)
    }
    /// Flushes any modified host transfer buffer contents back to guest memory (e.g. for discontiguous WHPX allocations).
    fn flush_to_guest(&self, this: &Resource<P>) {
        let _ = this;
    }
}

/// The kind's answers.
fn class_of<P: Platform>(kind: &ResourceKind<P>) -> &dyn ResourceClass<P> {
    match kind {
        ResourceKind::SharedBuffer(k) => k,
        ResourceKind::PrivateBuffer(k) => k,
        ResourceKind::DetachBuffer(k) => k,
        ResourceKind::HeapBuffer(k) => k,
        ResourceKind::MapperRefBuffer(k) => k,
        ResourceKind::LegacyHeap(k) => k,
        ResourceKind::DetachHeap(k) => k,
        ResourceKind::Icb(k) => k,
        ResourceKind::Texture(k) => k,
        ResourceKind::SerializerTexture(k) => k,
        ResourceKind::RefTexture(k) => k,
        ResourceKind::MapperRefTexture(k) => k,
        ResourceKind::SharedTexture(k) => k,
        ResourceKind::Backing(k) => k,
    }
}

impl<P: Platform> Resource<P> {
    /// A resource of `kind`, guest-valid as the kind starts.
    pub fn new(kind: ResourceKind<P>) -> Arc<Self> {
        let state = ResourceState::default();
        state.set_guest_valid(class_of(&kind).initial_guest_valid());
        Arc::new(Resource { state, kind })
    }

    fn class(&self) -> &dyn ResourceClass<P> {
        class_of(&self.kind)
    }

    /// DIAG(paging-class): the kind's slot in [`crate::perfstats::PAGING_KINDS`].
    pub fn kind_slot(&self) -> usize {
        match &self.kind {
            ResourceKind::SharedBuffer(_) => 0,
            ResourceKind::PrivateBuffer(_) => 1,
            ResourceKind::DetachBuffer(_) => 2,
            ResourceKind::HeapBuffer(_) => 3,
            ResourceKind::MapperRefBuffer(_) => 4,
            ResourceKind::LegacyHeap(_) => 5,
            ResourceKind::DetachHeap(_) => 6,
            ResourceKind::Icb(_) => 7,
            ResourceKind::Texture(_) => 8,
            ResourceKind::SerializerTexture(_) => 9,
            ResourceKind::RefTexture(_) => 10,
            ResourceKind::MapperRefTexture(_) => 11,
            ResourceKind::SharedTexture(_) => 12,
            ResourceKind::Backing(_) => 13,
        }
    }

    /// Whether the guest's copy is current (the kind's answer).
    pub fn is_guest_valid(&self) -> bool {
        self.class().is_guest_valid(self)
    }
    /// Marks the guest's copy (the kind decides whether that writes anything).
    pub fn set_is_guest_valid(&self, v: bool) {
        self.class().set_is_guest_valid(self, v)
    }
    /// Whether the host's copy is current (the kind's answer).
    pub fn is_host_valid(&self) -> bool {
        self.class().is_host_valid(self)
    }
    /// Marks the host's copy.
    pub fn set_is_host_valid(&self, v: bool) {
        self.class().set_is_host_valid(self, v)
    }
    /// Whether opcode 63 may discard it.
    pub fn supports_discard(&self) -> bool {
        self.class().supports_discard(self)
    }
    /// Whether it pages through a compute encoder.
    pub fn needs_compute_paging(&self) -> bool {
        self.class().needs_compute_paging(self)
    }
    /// Whether the guest's copy needs writing back.
    pub fn needs_sync(&self) -> bool {
        self.class().needs_sync(self)
    }
    /// Makes it non-volatile now, on the CPU: the first step of its prepare, needing no encoder.
    /// An exec's prepare does it for a resource whose host copy is valid, so that one needs no
    /// paging buffer ([`crate::tasks`]'s `prepare_resources`). A purged resource comes out
    /// host-invalid, and is paged as before.
    pub fn devolatilize_now(&self) -> RaiseResult<()> {
        self.class().devolatilize_now(self)
    }
    /// Whether the host's copy needs making or refreshing.
    pub fn needs_prepare(&self) -> bool {
        self.class().needs_prepare(self)
    }
    /// Whether it pages on the CPU.
    pub fn needs_cpu_paging(&self) -> bool {
        self.class().needs_cpu_paging(self)
    }
    /// Guest to host through a blit encoder. `encoder` is `None` when Metal gave no encoder,
    /// which nothing checks: the sends are skipped, and the flag and placement writes still
    /// happen.
    pub fn prepare_in_encoder(
        &self,
        hold: Hold<'_>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        // DIAG(prepare-each): a prepare of 2 ms or more is written at debug with its class.
        let timed =
            log::log_enabled!(target: "reims_vgpu_core::tasks::local::exec", log::Level::Debug)
                .then(std::time::Instant::now);
        let done = self.class().prepare_in_encoder(hold, self, encoder);
        if let Some(t) = timed {
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if ms >= 2.0 {
                log::debug!(
                    target: "reims_vgpu_core::tasks::local::exec",
                    "prepare_in_encoder {}: {ms:.2} ms",
                    self.class().diag_name()
                );
            }
        }
        done
    }
    /// Host to guest through a blit encoder (`None` as for a prepare).
    pub fn synchronize_in_encoder(
        &self,
        hold: Hold<'_>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        self.class().synchronize_in_encoder(hold, self, encoder)
    }
    /// Guest to host through a compute encoder (`None` as for a prepare).
    pub fn prepare_in_compute_encoder(
        &self,
        hold: Hold<'_>,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        self.class().prepare_in_compute_encoder(hold, self, encoder)
    }
    /// Host to guest through a compute encoder (`None` as for a prepare).
    pub fn synchronize_in_compute_encoder(
        &self,
        hold: Hold<'_>,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        self.class()
            .synchronize_in_compute_encoder(hold, self, encoder)
    }
    /// Opcode 63's discard.
    pub fn discard(&self) -> RaiseResult<()> {
        self.class().discard(self)
    }
    /// The Metal object an encoder binds.
    pub fn get_resource(&self) -> RaiseResult<Option<AnyResource<P>>> {
        self.class().get_resource(self)
    }
    /// The Metal buffer, retained.
    pub fn get_buffer(&self) -> RaiseResult<Option<P::Buffer>> {
        self.class().get_buffer(self)
    }
    /// The Metal texture, retained.
    pub fn get_texture(&self) -> RaiseResult<Option<P::Texture>> {
        self.class().get_texture(self)
    }
    /// The Metal heap, retained.
    pub fn get_heap(&self) -> RaiseResult<Option<P::Heap>> {
        self.class().get_heap(self)
    }
    /// The Metal indirect command buffer, retained.
    pub fn get_indirect_command_buffer(&self) -> RaiseResult<Option<P::Icb>> {
        self.class().get_indirect_command_buffer(self)
    }
    /// Writes back through a blit encoder when the raw flags say so (`None` as for a
    /// prepare).
    pub fn evict_in_encoder(
        &self,
        hold: Hold<'_>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        self.class().evict_in_encoder(hold, self, encoder)
    }
    /// Writes back through a compute encoder when the raw flags say so.
    pub fn evict_in_compute_encoder(
        &self,
        hold: Hold<'_>,
        encoder: Option<&P::ComputeEncoder>,
    ) -> RaiseResult<()> {
        self.class().evict_in_compute_encoder(hold, self, encoder)
    }
    /// The shared texture backing's id (0 for every kind but a texture's).
    pub fn get_shared_texture_backing_id(&self) -> RaiseResult<u32> {
        self.class().get_shared_texture_backing_id(self)
    }
    /// A child buffer over `buffer`, made from a heap; `None` for other kinds. +1.
    pub fn new_child_buffer_resource_with_buffer(
        self: &Arc<Self>,
        buffer: P::Buffer,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        self.class()
            .new_child_buffer_resource_with_buffer(self, buffer)
    }
    /// A child texture over `texture`. +1.
    ///
    /// No kind overrides it: the child is discardable exactly when this resource supports a
    /// discard.
    pub fn new_child_texture_resource_with_texture(
        self: &Arc<Self>,
        texture: P::Texture,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        // A question to the resource itself.
        let discardable = self.supports_discard();
        Ok(Some(SerializerTextureResource::new(texture, discardable)))
    }
    /// Guest to host on the CPU.
    pub fn prepare_with_cpu(&self, hold: Hold<'_>) -> RaiseResult<()> {
        self.class().prepare_with_cpu(hold, self)
    }
    /// Host to guest on the CPU.
    pub fn synchronize_with_cpu(&self) -> RaiseResult<()> {
        self.class().synchronize_with_cpu(self)
    }
    /// Writes back on the CPU before an eviction.
    pub fn evict_with_cpu(&self) -> RaiseResult<()> {
        self.class().evict_with_cpu(self)
    }
    /// Flushes any modified host transfer buffer contents back to guest memory.
    pub fn flush_to_guest(&self) {
        self.class().flush_to_guest(self);
    }
    /// Whether this is an IOSurface backing (the tasks' test before a backing's operations).
    pub fn is_backing(&self) -> bool {
        matches!(self.kind, ResourceKind::Backing(_))
    }

    /// Raises [`RaiseKind::Unrecognized`] for `op` on this resource: what a kind answers for an
    /// operation it has no body for.
    #[track_caller]
    pub fn unrecognized<T>(&self, op: Op) -> RaiseResult<T> {
        crate::resources::raise_with(RaiseKind::Unrecognized(op), self.kind_name())
    }

    /// The kind's name (diagnostics).
    pub fn kind_name(&self) -> &'static str {
        match &self.kind {
            ResourceKind::SharedBuffer(_) => "shared buffer",
            ResourceKind::PrivateBuffer(_) => "private buffer",
            ResourceKind::DetachBuffer(_) => "detach buffer",
            ResourceKind::HeapBuffer(k) if k.discardable() => "discardable heap buffer",
            ResourceKind::HeapBuffer(_) => "heap buffer",
            ResourceKind::MapperRefBuffer(_) => "mapper-ref buffer",
            ResourceKind::LegacyHeap(_) => "legacy heap",
            ResourceKind::DetachHeap(_) => "detach heap",
            ResourceKind::Icb(_) => "indirect command buffer",
            ResourceKind::Texture(_) => "texture",
            ResourceKind::SerializerTexture(_) => "child texture",
            ResourceKind::RefTexture(_) => "ref texture",
            ResourceKind::MapperRefTexture(_) => "mapper-ref texture",
            ResourceKind::SharedTexture(_) => "shared texture",
            ResourceKind::Backing(_) => "backing",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers the kinds share.

/// The open [`with_platform`] scopes of a thread, innermost last: each platform's type and the
/// pointer of its `Arc`, borrowed for the scope. A fixed table with no destructor: a thread-local
/// with one runs a fence and atomics at each access on the Windows GNU target, and this one is
/// reached at every task command (measured hot in a profile of the decode threads).
struct Platforms {
    slots: [(Option<std::any::TypeId>, *const ()); 8],
    len: usize,
}

thread_local! {
    static PLATFORMS: RefCell<Platforms> =
        const { RefCell::new(Platforms { slots: [(None, std::ptr::null()); 8], len: 0 }) };
}

/// Runs `f` with `platform` as the one the resources made or paged inside it reach for the VM
/// and Metal's descriptor classes. The task's entry points and the manager open it; so do
/// unit tests.
pub fn with_platform<P: Platform, T>(platform: &Arc<P>, f: impl FnOnce() -> T) -> T {
    struct Pop(bool);
    impl Drop for Pop {
        fn drop(&mut self) {
            if self.0 {
                PLATFORMS.with(|s| s.borrow_mut().len -= 1);
            }
        }
    }
    let pushed = PLATFORMS.with(|s| {
        let mut s = s.borrow_mut();
        if s.len == s.slots.len() {
            return false;
        }
        let at = s.len;
        s.slots[at] = (
            Some(std::any::TypeId::of::<P>()),
            Arc::as_ptr(platform).cast::<()>(),
        );
        s.len += 1;
        true
    });
    let _pop = Pop(pushed);
    f()
}

/// The platform a resource of `task` reaches, or `None` outside every [`with_platform`] scope:
/// the kinds then refuse the creations that need a descriptor or a VM allocation.
#[allow(unsafe_code)] // an `Arc` remade from the pointer its scope borrows
pub(crate) fn platform_for<P: Platform>(task: &TaskRef<P>) -> Option<Arc<P>> {
    let _ = task;
    let want = std::any::TypeId::of::<P>();
    let ptr = PLATFORMS.with(|s| {
        let s = s.borrow();
        s.slots[..s.len]
            .iter()
            .rev()
            .find(|(id, _)| *id == Some(want))
            .map(|&(_, p)| p.cast::<P>())
    })?;
    // SAFETY: the pointer is an `Arc<P>`'s (its type checked), alive while its scope is open,
    // which it is while it is in the table; the count is raised for the new `Arc`.
    unsafe {
        Arc::increment_strong_count(ptr);
        Some(Arc::from_raw(ptr))
    }
}

/// A record's first `size_of::<T>()` bytes. The factories check only that the record is at
/// least 8 bytes before reading up to 13; every caller passes the whole record (16, or 8 for
/// the mapper's), so the bytes past a short slice read as zero.
pub(crate) fn read_descriptor<T: FromBytes + IntoBytes>(descriptor: &[u8]) -> T {
    let mut v = T::new_zeroed();
    let dst = v.as_mut_bytes();
    let n = dst.len().min(descriptor.len());
    if let (Some(d), Some(s)) = (dst.get_mut(..n), descriptor.get(..n)) {
        d.copy_from_slice(s);
    }
    v
}

/// The task's features, read at each use. A torn-down task has none to read, and the process
/// dies there.
pub(crate) fn features_of<P: Platform>(task: &TaskRef<P>) -> GuestFeatures {
    match task.upgrade() {
        Some(t) => t.features().load(),
        None => crate::fatal!(DeathKind::GoneTaskFeatures),
    }
}

/// `MTLPurgeableStateNonVolatile`.
pub(crate) const PURGEABLE_NON_VOLATILE: u64 = 2;
/// `MTLPurgeableStateVolatile`.
pub(crate) const PURGEABLE_VOLATILE: u64 = 3;
/// `MTLPurgeableStateEmpty`.
pub(crate) const PURGEABLE_EMPTY: u64 = 4;
/// `MTLResourceStorageModePrivate` (`MTLStorageModePrivate << 4`).
pub(crate) const RESOURCE_STORAGE_MODE_PRIVATE: u64 = 0x20;
/// `MTLStorageModeShared`.
pub(crate) const STORAGE_MODE_SHARED: u64 = 0;
/// `MTLStorageModePrivate`.
pub(crate) const STORAGE_MODE_PRIVATE: u64 = 2;
/// The private resource option a read-only resource ORs in (not a public
/// `MTLResourceOptions` bit).
pub(crate) const RESOURCE_OPTION_READ_ONLY: u64 = 0x100_0000;
/// `MTLHeapTypePlacement`.
pub(crate) const HEAP_TYPE_PLACEMENT: i64 = 1;
/// `MTLHazardTrackingModeDefault`.
pub(crate) const HAZARD_TRACKING_DEFAULT: u64 = 0;
/// `MTLHazardTrackingModeTracked`.
pub(crate) const HAZARD_TRACKING_TRACKED: u64 = 2;

#[cfg(test)]
pub(crate) mod test_support {
    //! What the kinds' tests share: a fake platform, a recorder task and a placement slot.

    use std::sync::Arc;

    use super::TaskRef;

    use crate::guest::GuestPtr;
    use crate::guest::records::GuestFeatures;

    use crate::raise::Raise;
    use crate::resources::RaiseKind;
    use crate::resources::host::ResourceHost;
    use crate::testkit::guest_ram::RawRegion;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::resource_host::FakeResourceHost;

    /// A fake process and a recorder task.
    pub struct Env {
        pub p: Arc<FakePlatform>,
        pub rec: Arc<FakeResourceHost>,
        pub host: Arc<dyn ResourceHost<FakePlatform>>,
    }

    impl Env {
        pub fn new() -> Env {
            let p = Arc::new(FakePlatform::new());
            let rec = FakeResourceHost::new(p.clone());
            let host: Arc<dyn ResourceHost<FakePlatform>> = rec.clone();
            Env { p, rec, host }
        }

        /// The weak task the resources hold.
        pub fn task(&self) -> TaskRef<FakePlatform> {
            Arc::downgrade(&self.host)
        }

        /// Sets the task's features.
        pub fn set_features(&self, f: impl FnOnce(&mut GuestFeatures)) {
            let mut a = GuestFeatures::default();
            f(&mut a);
            let s = crate::guest::records::SharedFeatures::new();
            s.store(&a);
            self.rec.set_features(Arc::new(s));
        }

        /// Runs `f` inside the platform scope.
        pub fn run<T>(&self, f: impl FnOnce() -> T) -> T {
            super::with_platform(&self.p, f)
        }
    }

    /// Runs `f`, and answers what it returned with the messages of the lines this area logged
    /// meanwhile, in order. It captures the `log` crate's lines (`testkit::logs`), so it relies on
    /// nextest's one process per test, and must not be nested.
    pub fn logged<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
        let logs = crate::testkit::logs::capture();
        let r = f();
        let lines = logs
            .take()
            .into_iter()
            .filter(|(_, target, _)| {
                target.starts_with(concat!(env!("CARGO_CRATE_NAME"), "::resources"))
            })
            .map(|(_, _, message)| message)
            .collect();
        (r, lines)
    }

    /// A task that is gone (a weak load of it fails).
    pub fn gone_task() -> TaskRef<FakePlatform> {
        let w: std::sync::Weak<FakeResourceHost> = std::sync::Weak::new();
        w
    }

    /// A 16-byte placement slot in real memory holding `field0`.
    pub fn placement(field0: u64) -> (Arc<RawRegion>, GuestPtr) {
        let r = RawRegion::new(0x4000).expect("region");
        assert!(r.write_u64(0, field0));
        let g = GuestPtr::new(r.addr());
        (r, g)
    }

    /// The area's raise a result carries: `None` for `Ok` or another raise.
    pub fn kind<T>(r: Result<T, Raise>) -> Option<RaiseKind> {
        r.err().as_ref().and_then(crate::resources::raised_kind)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{Env, kind};
    use super::*;
    use crate::platform::device::GpuDevice;
    use crate::raise::OnFault;
    use crate::resources::buffer::child::HeapBufferResource;
    use crate::resources::icb::{IndirectCommandBufferFlags, IndirectCommandLayout};
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::{Model, kind};
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Fp = FakePlatform;

    /// The `Debug` form of an `AnyResource` names the kind and the object (diagnostics only).
    #[test]
    fn any_resource_debug_names_the_kind_and_object() {
        use crate::platform::Handle;
        let p = FakePlatform::new();
        let w = p.world();
        let b = w.new_obj::<kind::Buffer>(Model::None).clone_ref();
        let t = w.new_obj::<kind::Texture>(Model::None).clone_ref();
        let h = w.new_obj::<kind::Heap>(Model::None).clone_ref();
        let i = w.new_obj::<kind::Icb>(Model::None).clone_ref();
        let ids = [b.id(), t.id(), h.id(), i.id()];
        let all: [AnyResource<Fp>; 4] = [
            AnyResource::Buffer(b),
            AnyResource::Texture(t),
            AnyResource::Heap(h),
            AnyResource::Icb(i),
        ];
        for ((r, name), id) in all
            .iter()
            .zip(["Buffer", "Texture", "Heap", "Icb"])
            .zip(ids)
        {
            assert_eq!(format!("{r:?}"), format!("{name}({id:?})"));
        }
    }

    /// A heap buffer: the kind that keeps most of the defaults.
    fn heap_buffer(env: &Env, discardable: bool) -> Arc<Resource<Fp>> {
        let b = env
            .p
            .device()
            .new_buffer_of_length(OnFault::Die, 16, 0)
            .unwrap()
            .unwrap();
        let heap = private_buffer(env);
        HeapBufferResource::new(heap, b, discardable)
    }

    /// A private buffer: it answers every blit paging operation.
    fn private_buffer(env: &Env) -> Arc<Resource<Fp>> {
        env.run(|| {
            crate::resources::buffer::private::PrivateBufferResource::new(
                env.task(),
                &[16, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                None,
            )
        })
        .unwrap()
        .unwrap()
    }

    /// A kind that answers nothing of its own: every operation is the default.
    struct Bare;
    impl ResourceClass<Fp> for Bare {}

    /// Every operation a kind has no body for raises `Unrecognized` with that operation
    #[test]
    fn defaults_raise_unrecognized() {
        let env = Env::new();
        let r = heap_buffer(&env, false);
        let enc: Option<&<Fp as Platform>::BlitEncoder> = None;
        let cenc: Option<&<Fp as Platform>::ComputeEncoder> = None;
        let hold = Hold::NONE;
        let b = Bare;
        let cases = [
            (Op::Prepare, kind(b.prepare_in_encoder(hold, &r, enc))),
            (
                Op::Synchronize,
                kind(b.synchronize_in_encoder(hold, &r, enc)),
            ),
            (
                Op::PrepareCompute,
                kind(b.prepare_in_compute_encoder(hold, &r, cenc)),
            ),
            (
                Op::SynchronizeCompute,
                kind(b.synchronize_in_compute_encoder(hold, &r, cenc)),
            ),
            (Op::Discard, kind(b.discard(&r))),
            (Op::GetResource, kind(b.get_resource(&r))),
            (Op::GetBuffer, kind(b.get_buffer(&r))),
            (Op::GetTexture, kind(b.get_texture(&r))),
            (Op::GetHeap, kind(b.get_heap(&r))),
            (Op::GetIcb, kind(b.get_indirect_command_buffer(&r))),
            (Op::PrepareCpu, kind(b.prepare_with_cpu(hold, &r))),
            (Op::SynchronizeCpu, kind(b.synchronize_with_cpu(&r))),
            (Op::EvictCpu, kind(b.evict_with_cpu(&r))),
        ];
        for (op, got) in cases {
            assert_eq!(got, Some(RaiseKind::Unrecognized(op)), "{op:?}");
        }
    }

    /// An unrecognized operation's raise names where it is raised
    #[test]
    fn unrecognized_names_where_it_is_raised() {
        let env = Env::new();
        let r = heap_buffer(&env, false);
        let d = Bare.discard(&r).unwrap_err().site();
        assert_eq!(d.kind, crate::raise::death::DeathKind::UnhandledRaise);
        assert!(d.file.ends_with("resource.rs"), "{d:?}");
    }

    /// Which kind answers which operation, and which raise `Unrecognized`: one row per kind of
    /// this area, the operations in [`OPS`]' order
    #[test]
    fn responds_table() {
        const OPS: [Op; 13] = [
            Op::Prepare,
            Op::Synchronize,
            Op::PrepareCompute,
            Op::SynchronizeCompute,
            Op::Discard,
            Op::GetResource,
            Op::GetBuffer,
            Op::GetTexture,
            Op::GetHeap,
            Op::GetIcb,
            Op::PrepareCpu,
            Op::SynchronizeCpu,
            Op::EvictCpu,
        ];
        // 1: answers; 0: raises `Unrecognized`.
        let table: [(&str, [u8; 13]); 9] = [
            ("shared buffer", [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]),
            ("private buffer", [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            ("detach buffer", [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]),
            ("heap buffer", [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            (
                "discardable heap buffer",
                [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ),
            ("mapper-ref buffer", [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]),
            ("legacy heap", [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
            ("detach heap", [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]),
            (
                "indirect command buffer",
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            ),
        ];
        let env = Env::new();
        let in_process = Env::new();
        in_process
            .p
            .script()
            .always("supports_resource_detach_backing", Step::value(false));
        let buffer = |e: &Env, ty: i32, d: &[u8]| {
            e.run(|| {
                crate::resources::buffer::new_buffer_resource(Hold::NONE, e.task(), ty, d, None)
            })
            .unwrap()
            .unwrap()
        };
        let heap = |e: &Env, flags: u8| {
            let mut d = [0u8; 16];
            d[4] = 0x40;
            d[12] = flags;
            e.run(|| crate::resources::heap::new_heap_resource(e.task(), 13, &d, None))
                .unwrap()
                .unwrap()
        };
        let buf16 = |flags: u8| {
            let mut d = [0u8; 16];
            d[0] = 16;
            d[8] = 1;
            d[12] = flags;
            d
        };
        let legacy = heap(&in_process, 1);
        let detach_heap = heap(&env, 0);
        let child = |h: &Arc<Resource<Fp>>| {
            let b = env
                .p
                .device()
                .new_buffer_of_length(OnFault::Die, 16, 0)
                .unwrap()
                .unwrap();
            h.new_child_buffer_resource_with_buffer(b).unwrap().unwrap()
        };
        let icb = IcbResource::new(
            env.task(),
            &[0; 12],
            None,
            IndirectCommandLayout::default(),
            IndirectCommandBufferFlags::default(),
        )
        .unwrap()
        .unwrap();
        let resources = [
            buffer(&in_process, 1, &buf16(0)),
            buffer(&in_process, 1, &buf16(1)),
            buffer(&env, 1, &buf16(0)),
            child(&legacy),
            child(&detach_heap),
            buffer(&env, 15, &7u64.to_le_bytes()),
            legacy.clone(),
            detach_heap.clone(),
            icb,
        ];
        for ((name, row), r) in table.iter().zip(&resources) {
            assert_eq!(r.kind_name(), *name);
            in_process.run(|| {
                let got = [
                    kind(r.prepare_in_encoder(Hold::NONE, None)),
                    kind(r.synchronize_in_encoder(Hold::NONE, None)),
                    kind(r.prepare_in_compute_encoder(Hold::NONE, None)),
                    kind(r.synchronize_in_compute_encoder(Hold::NONE, None)),
                    kind(r.discard()),
                    kind(r.get_resource()),
                    kind(r.get_buffer()),
                    kind(r.get_texture()),
                    kind(r.get_heap()),
                    kind(r.get_indirect_command_buffer()),
                    kind(r.prepare_with_cpu(Hold::NONE)),
                    kind(r.synchronize_with_cpu()),
                    kind(r.evict_with_cpu()),
                ];
                for ((op, answers), got) in OPS.iter().zip(row).zip(got) {
                    let raised = got == Some(RaiseKind::Unrecognized(*op));
                    assert_eq!(raised, *answers == 0, "{name} {op:?}: {got:?}");
                }
            });
        }
    }

    /// A raise's identity names the operation and the kind
    #[test]
    fn unrecognized_names_the_kind_in_its_detail() {
        let env = Env::new();
        let hold = Hold::NONE;
        let r = heap_buffer(&env, true);
        let e = r.get_heap().unwrap_err();
        assert_eq!(e.detail().unwrap_or_default(), "discardable heap buffer");
        assert_eq!(e.id().to_string(), "resources.unrecognized.get_heap");
        let r = private_buffer(&env);
        let e = r.prepare_with_cpu(hold).unwrap_err();
        assert_eq!(e.detail().unwrap_or_default(), "private buffer");
    }

    /// The flag accessors write and read the flags
    #[test]
    fn accessors_write_and_read_the_flags() {
        let env = Env::new();
        let r = heap_buffer(&env, false);
        assert!(!r.is_guest_valid() && !r.is_host_valid());
        r.set_is_guest_valid(true);
        assert!(r.state.guest_valid() && r.is_guest_valid());
        r.set_is_host_valid(true);
        assert!(r.state.host_valid() && r.is_host_valid());
        r.set_is_guest_valid(false);
        r.set_is_host_valid(false);
        assert!(!r.state.guest_valid() && !r.state.host_valid());
    }

    /// The defaults' constant answers
    #[test]
    fn default_constants() {
        let env = Env::new();
        let r = heap_buffer(&env, true);
        // A discardable heap buffer answers the default's true.
        assert!(r.supports_discard());
        assert!(!r.needs_compute_paging());
        assert!(!r.needs_cpu_paging());
        assert_eq!(r.get_shared_texture_backing_id().ok(), Some(0));
    }

    /// Needs sync: guest-invalid and host-valid
    #[test]
    fn needs_sync_is_guest_invalid_and_host_valid() {
        let env = Env::new();
        // A private buffer keeps the default; it starts guest-valid.
        let r = private_buffer(&env);
        for (g, h, want) in [
            (false, false, false),
            (false, true, true),
            (true, false, false),
            (true, true, false),
        ] {
            r.set_is_guest_valid(g);
            r.set_is_host_valid(h);
            assert_eq!(r.needs_sync(), want, "g={g} h={h}");
        }
    }

    /// Needs prepare: host-invalid, or volatile
    #[test]
    fn needs_prepare_is_host_invalid_or_volatile() {
        let env = Env::new();
        let r = private_buffer(&env);
        for (h, v, want) in [
            (false, false, true),
            (true, false, false),
            (true, true, true),
            (false, true, true),
        ] {
            r.set_is_host_valid(h);
            r.state.set_is_volatile(v);
            assert_eq!(r.needs_prepare(), want, "h={h} v={v}");
        }
    }

    /// Made non-volatile now, a valid resource whose contents were kept needs no prepare: the
    /// exec's prepare drops it and builds no paging buffer for it
    #[test]
    fn devolatilize_now_of_kept_contents_needs_no_prepare() {
        let env = Env::new();
        let r = private_buffer(&env);
        r.set_is_host_valid(true);
        r.state.set_is_volatile(true);
        assert!(r.needs_prepare());
        env.p.script().push("setPurgeableState:", Step::value(3u64));
        r.devolatilize_now().unwrap();
        assert!(!r.state.is_volatile());
        assert!(r.is_host_valid());
        assert!(!r.needs_prepare());
    }

    /// Made non-volatile now, a resource whose contents were purged is host-invalid and still
    /// needs its prepare: it is paged as before
    #[test]
    fn devolatilize_now_of_purged_contents_still_needs_a_prepare() {
        let env = Env::new();
        let r = private_buffer(&env);
        r.set_is_host_valid(true);
        r.state.set_is_volatile(true);
        env.p.script().push("setPurgeableState:", Step::value(4u64));
        r.devolatilize_now().unwrap();
        assert!(!r.state.is_volatile());
        assert!(!r.is_host_valid());
        assert!(r.needs_prepare());
    }

    /// An eviction writes back only when the raw flags say guest-invalid and host-valid
    #[test]
    fn evict_synchronizes_only_when_raw_guest_invalid_and_host_valid() {
        let env = Env::new();
        let r = private_buffer(&env);
        let enc = env.p.world().new_obj::<kind::BlitEncoder>(Model::None);
        // Guest-valid: nothing.
        r.state.set_host_valid(true);
        let m = env.p.call_log().mark();
        r.evict_in_encoder(Hold::NONE, Some(&enc)).unwrap();
        assert!(env.p.call_log().since(m).is_empty());
        // Guest-invalid, host-valid: the private buffer's synchronize copies down.
        r.state.set_guest_valid(false);
        let m = env.p.call_log().mark();
        env.run(|| r.evict_in_encoder(Hold::NONE, Some(&enc)))
            .unwrap();
        let sels = env.p.call_log().sels_since(m);
        assert!(
            sels.contains(&"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
            "{sels:?}"
        );
        assert!(r.state.guest_valid());
        // Host-invalid: nothing.
        r.state.set_guest_valid(false);
        r.state.set_host_valid(false);
        let m = env.p.call_log().mark();
        r.evict_in_encoder(Hold::NONE, Some(&enc)).unwrap();
        assert!(env.p.call_log().since(m).is_empty());
    }

    /// A compute eviction reaches the kind's compute synchronize
    #[test]
    fn evict_in_compute_encoder_sends_the_compute_synchronize() {
        let env = Env::new();
        let r = private_buffer(&env);
        r.state.set_guest_valid(false);
        r.state.set_host_valid(true);
        let hold = Hold::NONE;
        // A private buffer has no compute synchronize.
        assert_eq!(
            kind(r.evict_in_compute_encoder(hold, None)),
            Some(RaiseKind::Unrecognized(Op::SynchronizeCompute))
        );
        // Not reached when the raw flags say there is nothing to write back.
        r.state.set_guest_valid(true);
        assert!(r.evict_in_compute_encoder(hold, None).is_ok());
    }

    /// Only a heap makes a child buffer
    #[test]
    fn the_default_makes_no_child_buffer() {
        let env = Env::new();
        let r = private_buffer(&env);
        let b = env
            .p
            .device()
            .new_buffer_of_length(OnFault::Die, 4, 0)
            .unwrap()
            .unwrap();
        assert!(
            r.new_child_buffer_resource_with_buffer(b)
                .unwrap()
                .is_none()
        );
    }

    /// A child texture is discardable exactly when its base supports a discard
    #[test]
    fn child_texture_follows_supports_discard() {
        let env = Env::new();
        for discardable in [true, false] {
            let base = heap_buffer(&env, discardable);
            let t = crate::resources::texture::test_support::Env::new();
            let tex = t.tex(
                crate::resources::texture::test_support::desc(70, 4, 4, 1, 1),
                false,
            );
            let child = base
                .new_child_texture_resource_with_texture(tex)
                .unwrap()
                .unwrap();
            assert_eq!(child.supports_discard(), discardable);
        }
    }

    /// The platform scopes nest and pop
    #[test]
    fn platform_scope_nests_and_pops() {
        let env = Env::new();
        let t = env.task();
        assert!(platform_for::<Fp>(&t).is_none());
        env.run(|| {
            let p = platform_for::<Fp>(&t).expect("inside the run");
            assert!(Arc::ptr_eq(&p, &env.p));
            let other = Arc::new(FakePlatform::new());
            with_platform(&other, || {
                assert!(Arc::ptr_eq(&platform_for::<Fp>(&t).unwrap(), &other));
            });
            assert!(Arc::ptr_eq(&platform_for::<Fp>(&t).unwrap(), &env.p));
        });
        assert!(platform_for::<Fp>(&t).is_none());
    }

    /// A short record reads as zero past its end
    #[test]
    fn read_descriptor_zero_fills() {
        let v: [u8; 4] = read_descriptor(&[1, 2]);
        assert_eq!(v, [1, 2, 0, 0]);
        let v: [u8; 2] = read_descriptor(&[1, 2, 3]);
        assert_eq!(v, [1, 2]);
    }

    /// A torn-down task's features cannot be read: the process dies there
    #[test]
    fn features_of_a_gone_task_dies() {
        let t = super::test_support::gone_task();
        let d = catch_death(|| features_of::<Fp>(&t));
        assert_eq!(d.err(), Some(DeathKind::GoneTaskFeatures.site()));
    }
}
