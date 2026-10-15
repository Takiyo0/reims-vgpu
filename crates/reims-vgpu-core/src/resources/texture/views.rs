//! Textures that are views: a view of a task-0 backing ([`RefTextureResource`]), and the
//! textures the deserializer makes ([`SerializerTextureResource`]: views, textures on buffers or
//! in heaps, memoryless ones), which are never paged.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::resources::raise_with;
use crate::resources::resource::AnyResource;
use crate::resources::resource::{Resource, ResourceClass, ResourceKind, TaskRef};
use crate::resources::texture::RaiseKind;
use crate::rt::{Held, Hold};

/// A texture over a task-0 backing. It holds the backing resource itself, and forwards its
/// paging and validity to it: those are the backing's own operations, run on the calling task's
/// thread. It has no discard: opcode 60 on it is unrecognized.
pub struct RefTextureResource<P: Platform> {
    /// The view (released first at the last release: fields drop in order).
    texture: Option<P::Texture>,
    /// The backing, a backing resource of task 0.
    backing: Arc<Resource<P>>,
    /// Plane index in the multi-plane backing.
    plane: u16,
    /// Whether host texture contains up-to-date contents of the backing plane.
    host_valid: AtomicBool,
    /// Generation of the backing when this view was last prepared.
    last_generation: AtomicU64,
}

impl<P: Platform> RefTextureResource<P> {
    /// A view `texture` of `backing`. The task and the object type are not kept; its own flags
    /// are never read (the getters forward). Never `None`, never a raise.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        object_type: u32,
        texture: P::Texture,
        backing: Arc<Resource<P>>,
        plane: u16,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let _ = (task, object_type);
        Ok(Some(Resource::new(ResourceKind::RefTexture(
            RefTextureResource {
                texture: Some(texture),
                backing,
                plane,
                host_valid: AtomicBool::new(false),
                last_generation: AtomicU64::new(0),
            },
        ))))
    }

    /// The backing it holds.
    pub fn backing(&self) -> &Arc<Resource<P>> {
        &self.backing
    }

    /// Plane index in the backing.
    pub fn plane(&self) -> u16 {
        self.plane
    }
}

impl<P: Platform> ResourceClass<P> for RefTextureResource<P> {
    /// The backing's prepare, as a landing frame of its own ([`Held::raising`]): what the
    /// backing's guest pages hold as they are bound (the page-table read's device handle) is
    /// released when it returns, or goes on with its raise.
    fn prepare_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let was_valid = self.is_host_valid(this);
        Held::raising(|held| {
            let hold = Hold::of(held);
            self.backing.prepare_in_encoder(hold, encoder)?;
            let backing_gen = if let ResourceKind::Backing(backing) = &self.backing.kind {
                backing.generation()
            } else {
                0
            };
            if was_valid && self.last_generation.load(Ordering::SeqCst) == backing_gen {
                return Ok(());
            }
            // The view aliases the backing's IOSurface. When the GPU owns the current
            // contents, uploading the guest transfer buffer would overwrite those newer
            // pixels through the alias, even though backing preparation correctly skipped it.
            if !self.backing.is_guest_valid() {
                log::debug!("backing view preserves GPU contents: plane={} generation={backing_gen}", self.plane);
            } else if let Some(texture) = &self.texture {
                if let ResourceKind::Backing(backing) = &self.backing.kind {
                    backing.prepare_plane_in_encoder(
                        hold,
                        usize::from(self.plane),
                        texture,
                        encoder,
                    )?;
                }
            }
            self.last_generation.store(backing_gen, Ordering::SeqCst);
            self.host_valid.store(true, Ordering::SeqCst);
            Ok(())
        })
    }

    /// The backing's synchronize, as a landing frame of its own, as for the prepare.
    fn synchronize_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = this;
        Held::raising(|held| self.backing.synchronize_in_encoder(Hold::of(held), encoder))
    }

    fn set_is_host_valid(&self, this: &Resource<P>, v: bool) {
        let _ = this;
        self.host_valid.store(v, Ordering::SeqCst);
        if v {
            if let ResourceKind::Backing(backing) = &self.backing.kind {
                self.last_generation
                    .store(backing.generation(), Ordering::SeqCst);
            }
        } else {
            self.last_generation.store(0, Ordering::SeqCst);
        }
        self.backing.set_is_host_valid(v);
    }

    fn set_is_guest_valid(&self, this: &Resource<P>, v: bool) {
        let _ = this;
        self.backing.set_is_guest_valid(v);
    }

    fn is_host_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        let backing_valid = self.backing.is_host_valid();
        if !backing_valid || !self.host_valid.load(Ordering::SeqCst) {
            return false;
        }
        if let ResourceKind::Backing(backing) = &self.backing.kind {
            self.last_generation.load(Ordering::SeqCst) == backing.generation()
        } else {
            true
        }
    }

    fn is_guest_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        self.backing.is_guest_valid()
    }

    fn get_texture(&self, this: &Resource<P>) -> RaiseResult<Option<P::Texture>> {
        let _ = this;
        Ok(self.texture.as_ref().map(Handle::clone_ref))
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(self
            .texture
            .as_ref()
            .map(|t| AnyResource::Texture(t.clone_ref())))
    }

    fn evict_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = (this, encoder);
        // A view's eviction does nothing, not even the flags' test.
        Ok(())
    }

    /// Not discardable, so its own views are not discardable either.
    fn supports_discard(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }

    fn flush_to_guest(&self, _this: &Resource<P>) {
        self.backing.flush_to_guest();
    }
}

/// A texture the deserializer made; `discardable` for one made discardable (a view of a
/// discardable texture). It starts host- and guest-invalid, so an exec that lists one prepares
/// it, which raises. Its raises say what was attempted (the tasks' tests read the text).
pub struct SerializerTextureResource<P: Platform> {
    /// The texture; `None` when Metal made none (a memoryless texture or a view): the resource is
    /// made anyway.
    texture: Option<P::Texture>,
    discardable: bool,
}

impl<P: Platform> SerializerTextureResource<P> {
    /// A texture resource over `texture` (possibly none).
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(texture: impl Into<Option<P::Texture>>, discardable: bool) -> Arc<Resource<P>> {
        Resource::new(ResourceKind::SerializerTexture(SerializerTextureResource {
            texture: texture.into(),
            discardable,
        }))
    }

    /// Whether it was made discardable.
    pub fn discardable(&self) -> bool {
        self.discardable
    }
}

impl<P: Platform> ResourceClass<P> for SerializerTextureResource<P> {
    fn get_texture(&self, this: &Resource<P>) -> RaiseResult<Option<P::Texture>> {
        let _ = this;
        Ok(self.texture.as_ref().map(Handle::clone_ref))
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(self
            .texture
            .as_ref()
            .map(|t| AnyResource::Texture(t.clone_ref())))
    }

    /// Always raises.
    fn prepare_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = (this, encoder);
        // One in an exec's resource list always faults.
        raise_with(
            RaiseKind::ViewPrepare.into(),
            "cannot prepare a child texture",
        )
    }

    /// Always raises.
    fn synchronize_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = (this, encoder);
        raise_with(
            RaiseKind::ViewSynchronize.into(),
            "cannot synchronize a child texture",
        )
    }

    /// Raises, unless discardable (then nothing).
    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        if self.discardable {
            return Ok(());
        }
        raise_with(
            RaiseKind::ViewDiscard.into(),
            "cannot discard a child texture that is not discardable",
        )
    }

    /// Only when discardable (so a view of a view is discardable only under a discardable one).
    fn supports_discard(&self, this: &Resource<P>) -> bool {
        let _ = this;
        self.discardable
    }

    fn needs_compute_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }
}

#[cfg(test)]
mod tests {
    /// Views of a backing.
    mod backing_view {

        use super::super::*;

        use crate::resources::host::ResourceHost;
        use crate::resources::texture::backing::{
            BackingDescriptor, BackingPlane, BackingResource,
        };
        use crate::testkit::calllog::Arg;
        use crate::testkit::objects::Model;
        use crate::testkit::resource_host::FakeResourceHost;
        use crate::testkit::{FakePlatform, Fk, kind};

        type P = FakePlatform;

        struct Env {
            p: Arc<P>,
            host: Arc<dyn ResourceHost<P>>,
        }

        fn env() -> Env {
            let p = Arc::new(FakePlatform::new());
            let rec = FakeResourceHost::new(p.clone());
            let host: Arc<dyn ResourceHost<P>> = rec;
            Env { p, host }
        }

        /// A one-plane task-0 backing (4x4, 16 bytes per row).
        fn backing(e: &Env) -> Arc<Resource<P>> {
            crate::resources::resource::with_platform(&e.p, || {
                BackingResource::new(
                    &*e.p,
                    Arc::downgrade(&e.host),
                    &BackingDescriptor {
                        length: 0x1000,
                        page: 2,
                        plane_count: 1,
                        ..BackingDescriptor::default()
                    },
                    &[BackingPlane {
                        offset: 0,
                        width: 4,
                        height: 4,
                        bits: (4 << 24) | 16,
                    }],
                )
            })
            .unwrap()
            .unwrap()
        }

        fn make(e: &Env, b: &Arc<Resource<P>>) -> (Arc<Resource<P>>, crate::platform::ObjId) {
            let t: Fk<kind::Texture> = e.p.world().new_obj(Model::None);
            let id = t.id();
            let r = RefTextureResource::new(Arc::downgrade(&e.host), 5, t, b.clone(), 0)
                .unwrap()
                .unwrap();
            (r, id)
        }

        fn kind_of(r: &Resource<P>) -> &RefTextureResource<P> {
            match &r.kind {
                ResourceKind::RefTexture(x) => x,
                _ => panic!("not a ref texture"),
            }
        }

        /// Holds the texture and the backing; the task and type are not kept
        #[test]
        fn new_and_getters() {
            let e = env();
            let b = backing(&e);
            let (r, id) = make(&e, &b);
            let x = kind_of(&r);
            assert!(Arc::ptr_eq(x.backing(), &b));
            assert_eq!(x.plane(), 0);
            assert_eq!(Arc::strong_count(&b), 2);
            assert_eq!(r.get_texture().unwrap().unwrap().id(), id);
            match r.get_resource().unwrap().unwrap() {
                AnyResource::Texture(t) => assert_eq!(t.id(), id),
                other => panic!("{other:?}"),
            }
            assert!(!r.supports_discard());
            // No discard: opcode 60 on it is unrecognized.
            {
                let e = r.discard().err().unwrap();
                assert_eq!(
                    crate::resources::raised_kind(&e),
                    Some(crate::resources::RaiseKind::Unrecognized(
                        crate::resources::Op::Discard
                    ))
                );
                assert!(!r.state.host_valid());
                assert!(!r.state.guest_valid());
            };
        }

        /// Releases the texture, then the backing, which outlives task 0's delete
        #[test]
        fn the_last_release_releases_texture_and_backing() {
            let e = env();
            let b = backing(&e);
            let (r, id) = make(&e, &b);
            let weak = Arc::downgrade(&b);
            drop(b);
            // Task 0 deleted its backing: the ref texture keeps it.
            assert!(weak.upgrade().is_some());
            drop(r);
            assert!(weak.upgrade().is_none());
            assert!(e.p.world().find(id).is_none());
        }

        /// Nothing, even when the backing needs a synchronize
        #[test]
        fn evict_does_nothing() {
            let e = env();
            let b = backing(&e);
            let (r, _) = make(&e, &b);
            b.state.set_guest_valid(false);
            b.state.set_host_valid(true);
            let mark = e.p.call_log().mark();
            r.evict_in_encoder(Hold::NONE, None).unwrap();
            assert!(e.p.call_log().since(mark).is_empty());
            assert!(!b.state.guest_valid());
        }

        /// Forwarded to the backing and prepares the ref texture's plane
        #[test]
        fn prepare_and_synchronize_forward() {
            let e = env();
            let b = backing(&e);
            let (r, id) = make(&e, &b);
            let enc: Fk<kind::BlitEncoder> = e.p.world().new_obj(Model::None);
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(b.state.host_valid());
            assert!(r.is_host_valid());
            let calls = e.p.call_log().calls_on(enc.id());
            assert_eq!(calls.len(), 2);
            assert!(calls[0].sel.starts_with("copyFromBuffer:"));
            assert_eq!(
                calls[0].args[0],
                Arg::Obj(Some(e.p.world().live("MTLBuffer")[0]))
            );
            assert!(calls[1].sel.starts_with("copyFromBuffer:"));
            assert_eq!(
                calls[1].args[0],
                Arg::Obj(Some(e.p.world().live("MTLBuffer")[0]))
            );
            assert_eq!(calls[1].args[5], Arg::Obj(Some(id)));

            // Redundant prepare does nothing
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(e.p.call_log().calls_on(enc.id()).len(), 2);

            b.state.set_guest_valid(false);
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(b.state.guest_valid());
            let calls = e.p.call_log().calls_on(enc.id());
            assert_eq!(calls.len(), 3);
            assert!(calls[2].sel.starts_with("copyFromTexture:"));
        }

        /// A new alias must not upload stale guest pages over a GPU-owned backing.
        #[test]
        fn new_backing_view_preserves_gpu_owned_contents() {
            let e = env();
            let b = backing(&e);
            b.set_is_host_valid(true);
            b.set_is_guest_valid(false);
            let (r, _) = make(&e, &b);
            let enc: Fk<kind::BlitEncoder> = e.p.world().new_obj(Model::None);
            let mark = e.p.call_log().mark();
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(r.is_host_valid());
            assert!(!b.is_guest_valid());
            assert!(e.p.call_log().since(mark).iter().all(
                |call| !call.sel.starts_with("copyFromBuffer:")),
                "preparing a new alias reloaded stale guest bytes over GPU contents");
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(e.p.call_log().since(mark).iter().all(
                |call| !call.sel.starts_with("copyFromBuffer:")));
        }

        #[test]
        fn new_backing_view_still_uploads_current_guest_contents() {
            let e = env();
            let b = backing(&e);
            b.set_is_host_valid(true);
            assert!(b.is_guest_valid());
            let (r, _) = make(&e, &b);
            let enc: Fk<kind::BlitEncoder> = e.p.world().new_obj(Model::None);
            let mark = e.p.call_log().mark();
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(r.is_host_valid());
            assert_eq!(e.p.call_log().since(mark).iter().filter(
                |call| call.sel.starts_with("copyFromBuffer:")).count(), 1);
        }

        /// Invalidation of backing re-prepares sibling ref textures via generation tracking
        #[test]
        fn backing_invalidation_reprepares_sibling_ref_textures() {
            let e = env();
            let b = backing(&e);
            let (r1, _id1) = make(&e, &b);
            let (r2, _id2) = make(&e, &b);
            let enc: Fk<kind::BlitEncoder> = e.p.world().new_obj(Model::None);

            // Initial prepare of r1 copies backing + plane 0 to r1
            r1.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(e.p.call_log().calls_on(enc.id()).len(), 2);

            // Initial prepare of r2 copies plane 0 to r2 (backing was already valid)
            r2.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(e.p.call_log().calls_on(enc.id()).len(), 3);

            // Redundant prepare on both does nothing
            r1.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            r2.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(e.p.call_log().calls_on(enc.id()).len(), 3);

            // Backing is invalidated (e.g. guest UI interaction / dirty rect)
            b.set_is_host_valid(false);
            assert!(!r1.is_host_valid());
            assert!(!r2.is_host_valid());

            // r1 prepares: re-prepares backing and plane 0 for r1
            r1.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(e.p.call_log().calls_on(enc.id()).len(), 5);
            assert!(r1.is_host_valid());

            // Backing is now host-valid, but r2 was not yet updated to new generation
            assert!(!r2.is_host_valid());
            r2.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(e.p.call_log().calls_on(enc.id()).len(), 6);
            assert!(r2.is_host_valid());
        }

        /// Forwarded; its own flags stay false
        #[test]
        fn validity_forwards() {
            let e = env();
            let b = backing(&e);
            let (r, _) = make(&e, &b);
            let x = &*r;
            assert!(!x.is_host_valid());
            assert!(x.is_guest_valid());
            // needsPrepare follows the backing through the forwarded getter.
            assert!(x.needs_prepare());
            x.set_is_host_valid(true);
            x.set_is_guest_valid(false);
            assert!(!x.needs_prepare());
            assert!(x.needs_sync());
            assert!(b.state.host_valid());
            assert!(!b.state.guest_valid());
            assert!(x.is_host_valid());
            assert!(!x.is_guest_valid());
            assert!(!r.state.host_valid());
            assert!(!r.state.guest_valid());
        }
    }

    /// Textures the deserializer made.
    mod serializer {

        use super::super::*;
        use crate::resources::texture::RaiseKind;
        use crate::resources::texture::test_support::raised;
        use crate::testkit::objects::Model;
        use crate::testkit::{FakePlatform, Fk, kind};

        type P = FakePlatform;

        fn tex(p: &P) -> Fk<kind::Texture> {
            p.world().new_obj(Model::None)
        }

        fn kind_of(r: &Resource<P>) -> &SerializerTextureResource<P> {
            match &r.kind {
                ResourceKind::SerializerTexture(s) => s,
                _ => panic!("not a serializer texture"),
            }
        }

        /// Both validity flags start false; the getters hand out the texture
        #[test]
        fn new_and_getters() {
            let p = P::new();
            let t = tex(&p);
            let id = t.id();
            for discardable in [false, true] {
                let r = SerializerTextureResource::<P>::new(t.clone_ref(), discardable);
                assert!(!r.state.host_valid());
                assert!(!r.state.guest_valid());
                assert!(!r.state.is_volatile());
                assert_eq!(kind_of(&r).discardable(), discardable);
                assert_eq!(r.get_texture().unwrap().unwrap().id(), id);
                match r.get_resource().unwrap().unwrap() {
                    AnyResource::Texture(x) => assert_eq!(x.id(), id),
                    other => panic!("{other:?}"),
                }
                assert!(!r.needs_compute_paging());
                // Host-invalid from the start: an exec's prepare reaches it.
                assert!(r.needs_prepare());
                assert!(!r.needs_cpu_paging());
            }
        }

        /// The last drop releases the texture
        #[test]
        fn the_last_release_releases_the_texture() {
            let p = P::new();
            let t = tex(&p);
            let r = SerializerTextureResource::<P>::new(t.clone_ref(), false);
            assert_eq!(t.retain_count(), 2);
            drop(r);
            assert_eq!(t.retain_count(), 1);
        }

        /// Both raise, in both classes
        #[test]
        fn prepare_and_synchronize_raise() {
            let p = P::new();
            for discardable in [false, true] {
                let r = SerializerTextureResource::<P>::new(tex(&p), discardable);
                let e = r.prepare_in_encoder(Hold::NONE, None).err().unwrap();
                assert_eq!(raised(&e), Some(RaiseKind::ViewPrepare));
                let e = r.synchronize_in_encoder(Hold::NONE, None).err().unwrap();
                assert_eq!(raised(&e), Some(RaiseKind::ViewSynchronize));
            }
        }

        #[test]
        fn discard_by_class() {
            let p = P::new();
            let plain = SerializerTextureResource::<P>::new(tex(&p), false);
            assert!(!plain.supports_discard());
            let e = plain.discard().err().unwrap();
            assert_eq!(raised(&e), Some(RaiseKind::ViewDiscard));
            let disc = SerializerTextureResource::<P>::new(tex(&p), true);
            assert!(disc.supports_discard());
            assert!(disc.discard().is_ok());
            // Their children follow `supportsDiscard`.
            let child = plain
                .new_child_texture_resource_with_texture(tex(&p))
                .unwrap()
                .unwrap();
            assert!(!kind_of(&child).discardable());
            let child = disc
                .new_child_texture_resource_with_texture(tex(&p))
                .unwrap()
                .unwrap();
            assert!(kind_of(&child).discardable());
        }

        /// The raise names where it is raised
        #[test]
        fn prepare_raise_names_where_it_is_raised() {
            let p = P::new();
            let r = SerializerTextureResource::<P>::new(tex(&p), true);
            let d = r.prepare_in_encoder(Hold::NONE, None);
            let d = d.unwrap_err().site();
            assert_eq!(d.kind, crate::raise::death::DeathKind::UnhandledRaise);
            assert!(d.file.ends_with("views.rs"), "{d:?}");
        }

        /// A nil texture still makes the resource; the getters return nil
        #[test]
        fn a_nil_texture_still_makes_the_resource() {
            let r = SerializerTextureResource::<P>::new(None, true);
            assert!(r.get_texture().unwrap().is_none());
            assert!(r.get_resource().unwrap().is_none());
            assert!(r.supports_discard());
            r.discard().unwrap();
            let plain = SerializerTextureResource::<P>::new(None, false);
            assert!(plain.get_texture().unwrap().is_none());
            assert!(!plain.supports_discard());
        }
    }
}
