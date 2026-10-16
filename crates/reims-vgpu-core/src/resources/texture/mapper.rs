//! A texture over an IOSurface the task's mapper names ([`MapperRefTextureResource`]): never
//! paged, always valid, and handing its surface back through its task at the last release.

use std::sync::Arc;

use crate::platform::resource::Texture;
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::raise::{Kept, keep_forever};
use crate::resources::resource::AnyResource;
use crate::resources::resource::{Resource, ResourceClass, ResourceKind, TaskRef};
use crate::rt::Hold;

/// A texture over a mapper IOSurface. Its last release hands the surface back through its task,
/// which a task being torn down no longer takes. It has no discard: opcode 60 on it is
/// unrecognized.
pub struct MapperRefTextureResource<P: Platform> {
    /// The task, weak.
    task: TaskRef<P>,
    mapping_id: u64,
    /// Taken by the drop.
    texture: Option<P::Texture>,
    surface_base: u64,
    surface_size: u64,
}

impl<P: Platform> MapperRefTextureResource<P> {
    /// A texture over the mapping `mapping_id`'s surface. Its own flags stay false (dead state:
    /// the getters answer valid). Never `None`, never a raise.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        mapping_id: u64,
        texture: P::Texture,
        surface_base: u64,
        surface_size: u64,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        Ok(Some(Resource::new(ResourceKind::MapperRefTexture(
            MapperRefTextureResource {
                task,
                mapping_id,
                texture: Some(texture),
                surface_base,
                surface_size,
            },
        ))))
    }

    /// The mapping's id.
    pub fn mapping_id(&self) -> u64 {
        self.mapping_id
    }

    /// The surface base address in host memory.
    pub fn surface_base(&self) -> u64 {
        self.surface_base
    }

    /// The surface allocation size in bytes.
    pub fn surface_size(&self) -> u64 {
        self.surface_size
    }

    /// Synchronizes guest memory into the host surface.
    fn sync_from_guest(&self) {
        if let Some(surface) = self.texture.as_ref().and_then(Texture::iosurface) {
            use crate::platform::os::surface::Surface as _;
            surface.sync_from_guest();
        }
        if let Some(t) = self.task.upgrade() {
            if let Some(map) = t.memory_map() {
                if self.surface_base != 0 && self.surface_size != 0 {
                    use crate::host::HostPtr;
                    use crate::num::ToUsize;
                    let ptr = HostPtr::from_addr(self.surface_base.to_usize());
                    map.sync_range_from_guest(ptr, self.surface_size);
                }
            }
        }
    }

    /// Synchronizes host surface memory back into guest memory.
    fn sync_to_guest(&self) {
        if let Some(surface) = self.texture.as_ref().and_then(Texture::iosurface) {
            use crate::platform::os::surface::Surface as _;
            surface.sync_to_guest();
        }
        if let Some(t) = self.task.upgrade() {
            if let Some(map) = t.memory_map() {
                if self.surface_base != 0 && self.surface_size != 0 {
                    use crate::host::HostPtr;
                    use crate::num::ToUsize;
                    let ptr = HostPtr::from_addr(self.surface_base.to_usize());
                    map.sync_range_to_guest(ptr, self.surface_size);
                }
            }
        }
    }
}

impl<P: Platform> ResourceClass<P> for MapperRefTextureResource<P> {
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

    fn needs_prepare(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn prepare_with_cpu(&self, _hold: Hold<'_>, this: &Resource<P>) -> RaiseResult<()> {
        self.sync_from_guest();
        this.set_is_host_valid(true);
        Ok(())
    }

    fn synchronize_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        self.sync_to_guest();
        Ok(())
    }

    fn prepare_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = encoder;
        self.sync_from_guest();
        this.set_is_host_valid(true);
        Ok(())
    }

    fn synchronize_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = (this, encoder);
        self.sync_to_guest();
        Ok(())
    }

    /// Not discardable: 63 deletes it, and the last release hands the surface back.
    fn supports_discard(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }

    fn is_host_valid(&self, this: &Resource<P>) -> bool {
        this.state.host_valid()
    }

    fn is_guest_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }
}

impl<P: Platform> Drop for MapperRefTextureResource<P> {
    /// In a pool of its own: the texture's surface (retained), the texture released, then the
    /// surface handed back to the task. The weak task goes with the fields.
    fn drop(&mut self) {
        let texture = self.texture.take();
        let task = &self.task;
        let mapping_id = self.mapping_id;
        let pool = crate::rt::pool::Pool::here();
        // A nil texture has no surface.
        let surface = texture.as_ref().and_then(Texture::iosurface);
        drop(texture);
        match task.upgrade() {
            Some(t) => t.release_io_surface_with_mapping_id(mapping_id, surface),
            // The task is tearing down; the retain is kept and the surface is never handed back.
            None => keep_forever(surface, Kept::SurfaceUseCount),
        }
        drop(pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::host::ResourceHost;
    use crate::testkit::calllog::Arg;
    use crate::testkit::impls::new_texture;
    use crate::testkit::objects::{Model, TexDesc};
    use crate::testkit::resource_host::FakeResourceHost;
    use crate::testkit::{FakePlatform, Fk, kind};

    type P = FakePlatform;

    fn kind_of(r: &Resource<P>) -> &MapperRefTextureResource<P> {
        match &r.kind {
            ResourceKind::MapperRefTexture(m) => m,
            _ => panic!("not a mapper texture"),
        }
    }

    fn setup() -> (Arc<P>, Arc<FakeResourceHost>, Arc<dyn ResourceHost<P>>) {
        let p = Arc::new(FakePlatform::new());
        let rec = FakeResourceHost::new(p.clone());
        let host: Arc<dyn ResourceHost<P>> = rec.clone();
        (p, rec, host)
    }

    /// A texture over a fresh surface, the surface's creator reference dropped (the texture
    /// holds the only one, as after the factory's `[surface release]`).
    fn surface_texture(p: &P) -> (Fk<kind::Texture>, crate::platform::ObjId) {
        let s = p.fake_surfaces().make_simple(4, 4, 0, 4);
        let id = s.id();
        let t = new_texture(p.world(), TexDesc::default(), Some(s), 0, None, false);
        (t, id)
    }

    #[test]
    fn new_and_operations() {
        let (p, _rec, host) = setup();
        let t: Fk<kind::Texture> = p.world().new_obj(Model::None);
        let id = t.id();
        let r = MapperRefTextureResource::new(Arc::downgrade(&host), 7, t, 0x1000, 0x4000)
            .unwrap()
            .unwrap();
        let m = kind_of(&r);
        assert_eq!(m.mapping_id(), 7);
        assert_eq!(m.surface_base(), 0x1000);
        assert_eq!(m.surface_size(), 0x4000);
        assert_eq!(r.get_texture().unwrap().unwrap().id(), id);
        match r.get_resource().unwrap().unwrap() {
            AnyResource::Texture(x) => assert_eq!(x.id(), id),
            other => panic!("{other:?}"),
        }
        // Starts with host_valid false, guest_valid true.
        assert!(!r.is_host_valid() && r.is_guest_valid());
        assert!(!r.state.host_valid() && !r.state.guest_valid());
        assert!(r.needs_prepare() && !r.needs_sync());
        assert!(!r.needs_cpu_paging());
        assert!(!r.supports_discard());
        let mark = p.call_log().mark();
        r.prepare_with_cpu(Hold::NONE).unwrap();
        assert!(r.is_host_valid() && r.state.host_valid());
        // needs_prepare() always returns true so guest memory is reliably synced on every access.
        assert!(r.needs_prepare() && !r.needs_sync());
        // Second prepare (via encoder) also runs and keeps needs_prepare true.
        r.prepare_in_encoder(Hold::NONE, None).unwrap();
        assert!(r.is_host_valid() && r.state.host_valid());
        assert!(r.needs_prepare());
        r.synchronize_with_cpu().unwrap();
        r.synchronize_in_encoder(Hold::NONE, None).unwrap();
        let calls = p.call_log().since(mark);
        assert!(calls.iter().all(|c| c.sel == "iosurface"));
        assert!(!calls.is_empty());
        r.set_is_host_valid(false);
        assert!(!r.is_host_valid());
        assert!(r.needs_prepare());
        // No discard: opcode 60 on it is unrecognized.
        {
            let e = r.discard().err().unwrap();
            assert_eq!(
                crate::resources::raised_kind(&e),
                Some(crate::resources::RaiseKind::Unrecognized(
                    crate::resources::Op::Discard
                ))
            );
        };
    }

    /// The surface goes back to the live task, the texture released first
    #[test]
    fn the_last_release_hands_the_surface_back() {
        let (p, _rec, host) = setup();
        let (t, sid) = surface_texture(&p);
        let tid = t.id();
        let r = MapperRefTextureResource::new(Arc::downgrade(&host), 9, t, 0, 0)
            .unwrap()
            .unwrap();
        let mark = p.call_log().mark();
        drop(r);
        let calls = p.call_log().since(mark);
        let sels: Vec<_> = calls.iter().map(|c| c.sel).collect();
        let at_getter = sels.iter().position(|s| *s == "iosurface").unwrap();
        let at_release = sels
            .iter()
            .position(|s| *s == "release_io_surface_with_mapping_id")
            .unwrap();
        assert!(at_getter < at_release);
        assert_eq!(
            calls[at_release].args,
            vec![Arg::U64(9), Arg::Obj(Some(sid))]
        );
        // The recorder drops what it was handed: everything is gone.
        assert!(p.world().find(tid).is_none());
        assert!(p.world().find(sid).is_none());
    }

    /// With the task gone the retain leaks and nothing is handed back
    #[test]
    fn a_gone_task_leaks_the_surface_retain() {
        let (p, rec, _host) = setup();
        let gone: TaskRef<P> = {
            let other: Arc<dyn ResourceHost<P>> = rec.sibling();
            Arc::downgrade(&other)
        };
        let (t, sid) = surface_texture(&p);
        let r = MapperRefTextureResource::new(gone, 9, t, 0, 0).unwrap().unwrap();
        drop(r);
        assert_eq!(p.call_log().count("release_io_surface_with_mapping_id"), 0);
        // The texture is released; the surface lives on through the leaked retain.
        assert!(p.world().live("MTLTexture").is_empty());
        assert_eq!(p.world().live("IOSurface"), vec![sid]);
    }
}
