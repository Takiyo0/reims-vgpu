//! IOSurface's stand-in on Linux: what `IOSurfaceCreate` and `[[IOSurface alloc]
//! initWithProperties:]` answer on macOS, over memory a view can be made of.
//!
//! - **References.** A [`LinuxSurface`] is one reference: `clone_ref` is a retain, `Drop` a
//!   release, and the retain count, which the guest sees, is the number of references to the
//!   object. Its ID is the object's address, unique among live surfaces. A lookup
//!   (`lookup_by_port`) is another object over the same memory, with a count of its own, as
//!   IOSurface's lookup in the process that made the surface answers.
//! - **Memory.** A surface of its own memory is a file of its own, mapped and registered in the
//!   backing table ([`super::vm::OwnMemory`]), so a view can be made of it (an alias of its
//!   allocation, a buffer over it) and no `deallocate` frees it. One over the guest's pages
//!   (`kIOSurfaceClientAddressRanges`) of one range is that range, its base address its offset
//!   into it; one of several ranges is a view of them ([`super::vm::LinuxVm`]), read-only for
//!   the CPU where it is read-only for the GPU (`kIOSurfaceClientDirection` 2), as IOSurface maps
//!   it on macOS. A protected surface (protection options other than 0) has no memory the CPU
//!   reaches: its base address is NULL.
//!   The mapping goes with the surface's last object, or later with the last hold on it
//!   ([`surface_memory_hold`]): a device's import of the pages at the surface's base address
//!   holds one until the runtime has freed the import, so the pages stay mapped while it lives
//!   (`VK_EXT_external_memory_host`), though the surface's last reference went at Metal's
//!   moment. One over a single range maps nothing of its own. Core makes such a surface only
//!   over one page of the host's own RAM, which a memory map names (core's walk of the guest's
//!   page table gives each page by the memory map's host address), and which the host maps and
//!   unmaps, not the library ([`crate::remap`]'s docs, "Not reported"). Its hold is an unmap hold
//!   on the range ([`crate::remap::UnmapHold`]), which defers an unmap only of pages mapped
//!   through the platform's VM, so for such a surface it defers nothing. A hold is no
//!   reference: the retain count stays the objects'. Measured on an Iris Xe (Mesa 26.2.4's ANV,
//!   i915): ANV puts every live import in each execbuffer, and one whose pages are unmapped
//!   fails it with `EFAULT`, which ANV answers as the device lost.
//! - **What it answers** is the spec's: its planes, rows and sizes, its allocation size exactly
//!   as given. It holds no values (`IOSurfaceCopyValue`): nothing in the process sets one.
//!   `lock` and `unlock` do nothing.
//!
//! **What it refuses**, as `IOSurfaceCreate` does on macOS 26 (measured on the M3 Max, 16 KiB
//! pages, one `IOSurfaceCreate` per case): a width, height, bytes per element, bytes per row,
//! element width or height, or allocation size of 0; rows shorter than a row of elements
//! (`ceil(width / element width) × bytes per element`); an allocation shorter than its offset
//! and the bytes of its rows (`offset + (rows − 1) × bytes per row + a row of elements`, rows
//! `ceil(height / element height)`); a plane that ends past the allocation or starts inside the
//! plane before it, or whose rows are shorter than a row of its elements (measured with 1 × 1
//! elements; the stand-in applies the surface's rule to a plane's element size); a
//! `kIOSurfaceClientDirection` other than 2 or 3; protection options past 0xf, or any over the
//! guest's pages; address ranges that are none, or not whole pages, or that hold fewer bytes than
//! the allocation size. An allocation past the host's usable memory is refused: on the M3 Max,
//! of 64 GiB, every size up to 0xf_c846_0000 bytes was made and every size past it, by one byte,
//! refused, which is its `hw.memsize_usable`, the part of `hw.memsize` its kernel manages; so
//! for a surface of its own memory, one over the guest's pages and one made from plane
//! properties alike (measured 2026-10-10, macOS 26.7, 25G229). The stand-in refuses one past
//! Linux's count of the memory its kernel manages, `_SC_PHYS_PAGES` pages (`MemTotal`). The
//! protection options read back with the lower options each implies: 2 with 1, 4 with 2 and 1.
//!
//! A surface made from plane properties alone (`initWithProperties:`) is its planes' memory,
//! whole pages from its first byte through the end of the plane that ends last; it answers 1
//! byte per element and rows of its width rounded up to 128 bytes (measured on the M3 Max:
//! widths 16, 17, 32 and 65535).

use std::ptr::NonNull;
use std::sync::Arc;

use reims_vgpu_core::host::{HostPtr, PhysicalMemoryRange};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::surface::{
    BackingSurfaceSpec, IoSurfaces, Surface, SurfacePlaneSpec, SurfaceSpec,
};
use reims_vgpu_core::platform::os::vm::{HostVm as _, KernReturn, VmProt};
use reims_vgpu_core::platform::{Handle, ObjId};
use reims_vgpu_core::sync::Mutex;

use super::vm::{LinuxVm, OwnMemory};
use crate::os::{MadeOver, READ_ONLY_DIRECTION, page_runs};
use crate::platform::VulkanPlatform;

/// The `kIOSurfaceClientDirection` of a surface over the guest's pages whose GPU writes reach
/// them, which IOSurface takes beside [`READ_ONLY_DIRECTION`].
const WRITABLE_DIRECTION: u32 = 3;

/// The protection options IOSurface takes: the four lowest bits.
const PROTECTION_OPTIONS: u64 = 0xf;

/// The bytes a row of a surface made from plane properties is rounded up to.
const BACKING_ROW_ALIGNMENT: u64 = 128;

/// One reference to an IOSurface stand-in (module docs).
#[derive(Debug)]
pub struct LinuxSurface(Arc<Object>);

/// An object over a surface: the one made, or a lookup of it.
#[derive(Debug)]
struct Object(Arc<Data>);

/// One plane, as the spec gives it: its offset from the base address, its size, its bytes per
/// element and per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plane {
    offset: u64,
    width: u64,
    height: u64,
    bytes_per_element: u64,
    bytes_per_row: u64,
}

/// A surface: what it answers, and the memory it holds.
#[derive(Debug)]
struct Data {
    pixel_format: u32,
    width: u64,
    height: u64,
    bytes_per_element: u64,
    bytes_per_row: u64,
    planes: Vec<Plane>,
    protection_options: u64,
    alloc_size: u64,
    /// The base address; 0 for a protected surface.
    base: u64,
    /// The memory it mapped, freed with the last holder of it.
    memory: Arc<Memory>,
    /// What it was made over ([`crate::os::surface_memory`] and the helpers beside it).
    made: Mutex<MadeOver>,
}

/// The memory a surface mapped: its own, or its view of the guest's pages; none for one over a
/// single range of them, or a protected one.
#[derive(Debug)]
struct Memory {
    /// Its own memory.
    own: Option<OwnMemory>,
    /// The view of the guest's pages it made, its first byte and length.
    view: Option<(u64, u64)>,
}

impl Drop for Memory {
    /// Frees the view the surface made, and its own memory.
    fn drop(&mut self) {
        if let Some((first, length)) = self.view {
            let _ = LinuxVm::new().deallocate(HostPtr::from_addr(first.to_usize()), length);
        }
        drop(self.own.take());
    }
}

/// A hold on the memory `surface` mapped, which keeps it mapped while it lives, the surface gone
/// or not, and counts no reference of the surface; for one over a single range of the guest's
/// pages, which maps nothing, an unmap hold on that range, which defers nothing for a range of
/// the host's own RAM, the only one core makes such a surface over (module docs).
pub(crate) fn surface_memory_hold(surface: &LinuxSurface) -> impl Send + 'static {
    let data = surface.data();
    let memory = &data.memory;
    let cores_unmap = (memory.own.is_none() && memory.view.is_none())
        .then(|| {
            NonNull::new(std::ptr::with_exposed_provenance_mut::<u8>(
                data.base.to_usize(),
            ))
        })
        .flatten()
        .map(|first| crate::remap::UnmapHold::new(first, data.alloc_size));
    (Arc::clone(memory), cores_unmap)
}

impl LinuxSurface {
    /// A new surface: one reference to a new object over `data`.
    fn new(data: Data) -> LinuxSurface {
        LinuxSurface(Arc::new(Object(Arc::new(data))))
    }

    /// The surface the object is over.
    fn data(&self) -> &Data {
        &self.0.0
    }

    /// Marks the surface as written by the GPU.
    pub fn mark_gpu_written(&self) {}
}

impl Handle for LinuxSurface {
    fn clone_ref(&self) -> Self {
        LinuxSurface(Arc::clone(&self.0))
    }

    fn id(&self) -> ObjId {
        ObjId(Arc::as_ptr(&self.0).addr().to_u64())
    }
}

impl Surface for LinuxSurface {
    fn retain_count(&self) -> i64 {
        i64::try_from(Arc::strong_count(&self.0)).unwrap_or(i64::MAX)
    }

    fn pixel_format(&self) -> u32 {
        self.data().pixel_format
    }

    fn width(&self) -> u64 {
        self.data().width
    }

    fn height(&self) -> u64 {
        self.data().height
    }

    fn bytes_per_element(&self) -> u64 {
        self.data().bytes_per_element
    }

    fn bytes_per_row(&self) -> u64 {
        self.data().bytes_per_row
    }

    fn plane_count(&self) -> u64 {
        self.data().planes.len().to_u64()
    }

    /// True, as IOSurface answers for every surface measured (its own memory, the guest's
    /// pages, planes or none, made either way).
    fn allows_pixel_size_casting(&self) -> bool {
        true
    }

    fn protection_options(&self) -> u64 {
        self.data().protection_options
    }

    fn lock(&self, _options: u32) -> KernReturn {
        KernReturn::SUCCESS
    }

    fn unlock(&self, _options: u32) -> KernReturn {
        KernReturn::SUCCESS
    }

    fn base_address(&self) -> HostPtr {
        HostPtr::from_addr(self.data().base.to_usize())
    }

    fn alloc_size(&self) -> u64 {
        self.data().alloc_size
    }

    fn lookup_by_port(&self) -> Option<Self> {
        Some(LinuxSurface(Arc::new(Object(Arc::clone(&self.0.0)))))
    }
}

/// Plane `plane` of `surface`: its offset from the base address and its bytes per row, as the
/// spec gave them; `None` for a plane the surface lacks.
pub(crate) fn plane_at(surface: &LinuxSurface, plane: u64) -> Option<(u64, u64)> {
    let p = surface.data().planes.get(usize::try_from(plane).ok()?)?;
    Some((p.offset, p.bytes_per_row))
}

/// The width, height and bytes per element of plane `plane` of `surface`; `None` for a plane it
/// lacks.
pub(crate) fn plane_shape(surface: &LinuxSurface, plane: u64) -> Option<(u64, u64, u64)> {
    let p = surface.data().planes.get(usize::try_from(plane).ok()?)?;
    Some((p.width, p.height, p.bytes_per_element))
}

/// The YCbCr matrix `surface` names (`kIOSurfaceYCbCrMatrix`): none, as the stand-in holds no
/// values and nothing sets one.
pub(crate) fn ycbcr_matrix(
    _surface: &LinuxSurface,
) -> Option<metal2vulkan_runtime::SurfaceYCbCrMatrix> {
    None
}

/// What `surface` was made over, `change` given it: every stand-in holds its own record.
pub(crate) fn recorded_mut<T>(
    surface: &LinuxSurface,
    change: impl FnOnce(&mut MadeOver) -> T,
) -> Option<T> {
    Some(change(&mut surface.data().made.lock()))
}

/// The bytes a run of `height` rows of `width` elements, each `element` wide and high and
/// `bytes_per_element` long, rows `bytes_per_row` apart, spans: through the last row's last
/// element. `None` where IOSurface refuses the layout (module docs) or past `u64::MAX`.
fn extent(
    (width, height): (u64, u64),
    (element_width, element_height): (u64, u64),
    bytes_per_element: u64,
    bytes_per_row: u64,
) -> Option<u64> {
    if [
        width,
        height,
        element_width,
        element_height,
        bytes_per_element,
    ]
    .contains(&0)
    {
        return None;
    }
    let row = width
        .div_ceil(element_width)
        .checked_mul(bytes_per_element)?;
    if bytes_per_row < row {
        return None;
    }
    let rows = height.div_ceil(element_height);
    (rows - 1).checked_mul(bytes_per_row)?.checked_add(row)
}

/// The host's usable memory, in bytes, as Linux counts it (`MemTotal`): the largest allocation
/// the stand-in makes, as macOS's IOSurface makes none past `hw.memsize_usable` (module docs).
fn physical_memory() -> u64 {
    // SAFETY: sysconf reads a system value.
    let (pages, page) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    u64::try_from(pages)
        .unwrap_or(0)
        .saturating_mul(u64::try_from(page).unwrap_or(0))
}

/// The host's page size.
fn page_size() -> u64 {
    LinuxVm::new().page_size()
}

/// The planes of `spec`, checked against its allocation (module docs); `None` where IOSurface
/// refuses them. No plane info is no planes.
fn planes(spec: &SurfaceSpec) -> Option<Vec<Plane>> {
    let mut planes = Vec::new();
    let mut free_from = 0u64;
    for p in spec.planes.iter().flatten() {
        let SurfacePlaneSpec {
            offset,
            width,
            height,
            bytes_per_row,
            bytes_per_element,
            element_width,
            element_height,
            ..
        } = *p;
        let element = (u64::from(element_width), u64::from(element_height));
        let span = extent((width, height), element, bytes_per_element, bytes_per_row)?;
        let end = offset.checked_add(span)?;
        if offset < free_from || end > spec.alloc_size {
            return None;
        }
        free_from = end;
        planes.push(Plane {
            offset,
            width,
            height,
            bytes_per_element,
            bytes_per_row,
        });
    }
    Some(planes)
}

/// The protection options IOSurface answers for `options` (module docs); `None` past those it
/// takes.
fn protection_options(options: u64) -> Option<u64> {
    if options & !PROTECTION_OPTIONS != 0 {
        return None;
    }
    let mut answered = options;
    if answered & 4 != 0 {
        answered |= 2;
    }
    if answered & 2 != 0 {
        answered |= 1;
    }
    Some(answered)
}

/// `ranges`, checked (module docs): each whole pages, together holding `alloc_size` bytes or
/// more; `None` where IOSurface refuses them.
fn checked_ranges(ranges: &[PhysicalMemoryRange], alloc_size: u64) -> Option<()> {
    let page = page_size();
    let mut total = 0u64;
    for r in ranges {
        let whole = r.physical_length != 0
            && r.physical_address.is_multiple_of(page)
            && r.physical_length.is_multiple_of(page);
        if !whole {
            return None;
        }
        total = total.checked_add(r.physical_length)?;
    }
    (!ranges.is_empty() && alloc_size <= total).then_some(())
}

/// A view of the first `length` bytes of `ranges`, in their order, read-only when `read_only`:
/// its first byte, or `None` where a range has no file behind it or the host gives no address
/// space, nothing left mapped.
fn view(ranges: &[PhysicalMemoryRange], length: u64, read_only: bool) -> Option<u64> {
    let vm = LinuxVm::new();
    let at = vm.allocate_anywhere(length).ok()?;
    let mut offset = 0u64;
    for r in ranges {
        if offset >= length {
            break;
        }
        let now = r.physical_length.min(length - offset);
        let source = HostPtr::from_addr(r.physical_address.to_usize());
        if vm
            .remap_shared_overwrite(at.wrapping_add(offset), source, now)
            .is_err()
        {
            log::warn!(
                target: "reims_vgpu_vulkan::os",
                "IOSurface over the guest's pages: no view of {now:#x} bytes at {:#x}, which the \
                 library has no file of; refused",
                r.physical_address
            );
            let _ = vm.deallocate(at, length);
            return None;
        }
        offset += now;
    }
    if read_only && vm.protect_max(at, length, VmProt::READ).is_err() {
        let _ = vm.deallocate(at, length);
        return None;
    }
    Some(at.addr().to_u64())
}

/// IOSurface's creation on Linux (module docs).
#[derive(Debug, Default)]
pub struct LinuxSurfaces;

impl LinuxSurfaces {
    /// The process's IoSurfaces.
    pub fn new() -> LinuxSurfaces {
        LinuxSurfaces
    }
}

impl IoSurfaces<VulkanPlatform> for LinuxSurfaces {
    fn create(&self, spec: &SurfaceSpec) -> Option<LinuxSurface> {
        if !matches!(
            spec.client_direction,
            None | Some(READ_ONLY_DIRECTION | WRITABLE_DIRECTION)
        ) {
            return None;
        }
        let protection = protection_options(spec.protection_options)?;
        let element = (
            u64::from(spec.element_width),
            u64::from(spec.element_height),
        );
        let span = extent(
            (spec.width, spec.height),
            element,
            spec.bytes_per_element,
            spec.bytes_per_row,
        )?;
        if spec.alloc_size < spec.offset.checked_add(span)? || spec.alloc_size > physical_memory() {
            return None;
        }
        let planes = planes(spec)?;
        let mut made = MadeOver {
            offset: spec.offset,
            ..MadeOver::default()
        };
        let (base, own, view_made) = match &spec.client_address_ranges {
            Some(ranges) => {
                if protection != 0 {
                    return None;
                }
                checked_ranges(ranges, spec.alloc_size)?;
                let pages = ranges
                    .iter()
                    .map(|r| (r.physical_address, r.physical_length));
                made.pages = page_runs(pages);
                made.gpu_read_only = spec.client_direction == Some(READ_ONLY_DIRECTION);
                match ranges.as_slice() {
                    [one] => (one.physical_address, None, None),
                    _ => {
                        let length = spec.alloc_size.checked_next_multiple_of(page_size())?;
                        let at = view(ranges, length, made.gpu_read_only)?;
                        (at, None, Some((at, length)))
                    }
                }
            }
            None if protection != 0 => (0, None, None),
            None => {
                let own = OwnMemory::new(spec.alloc_size)?;
                (own.first(), Some(own), None)
            }
        };
        Some(LinuxSurface::new(Data {
            pixel_format: spec.pixel_format,
            width: spec.width,
            height: spec.height,
            bytes_per_element: spec.bytes_per_element,
            bytes_per_row: spec.bytes_per_row,
            planes,
            protection_options: protection,
            alloc_size: spec.alloc_size,
            base: if base == 0 { 0 } else { base + spec.offset },
            memory: Arc::new(Memory {
                own,
                view: view_made,
            }),
            made: Mutex::new(made),
        }))
    }

    fn create_backing(&self, spec: &BackingSurfaceSpec) -> Option<LinuxSurface> {
        let (width, height) = (u64::from(spec.width), u64::from(spec.height));
        if width == 0 || height == 0 {
            return None;
        }
        let mut planes = Vec::with_capacity(spec.planes.len());
        let mut end = 0u64;
        for p in &spec.planes {
            let bytes_per_element = u64::try_from(p.bytes_per_element).ok()?;
            let bytes_per_row = u64::try_from(p.bytes_per_row).ok()?;
            let (w, h) = (u64::from(p.width), u64::from(p.height));
            let span = extent((w, h), (1, 1), bytes_per_element, bytes_per_row)?;
            let offset = u64::from(p.offset);
            end = end.max(offset.checked_add(span)?);
            planes.push(Plane {
                offset,
                width: w,
                height: h,
                bytes_per_element,
                bytes_per_row,
            });
        }
        let alloc_size = end.checked_next_multiple_of(page_size())?;
        if alloc_size == 0 || alloc_size > physical_memory() {
            return None;
        }
        let own = OwnMemory::new(alloc_size)?;
        Some(LinuxSurface::new(Data {
            pixel_format: spec.pixel_format,
            width,
            height,
            bytes_per_element: 1,
            bytes_per_row: width.checked_next_multiple_of(BACKING_ROW_ALIGNMENT)?,
            planes,
            protection_options: 0,
            alloc_size,
            base: own.first(),
            memory: Arc::new(Memory {
                own: Some(own),
                view: None,
            }),
            made: Mutex::new(MadeOver::default()),
        }))
    }

    /// The spec's fields, as a dictionary's description names the keys it holds.
    fn describe(&self, spec: &SurfaceSpec) -> String {
        format!("{spec:?}")
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::os::surface::BackingPlaneSpec;

    use super::super::vm::allocate_aliasable;
    use super::super::vm::tests::mapped;
    use super::*;

    const BGRA: u32 = u32::from_be_bytes(*b"BGRA");

    fn host_only() -> SurfaceSpec {
        SurfaceSpec {
            pixel_format: BGRA,
            alloc_size: 1024,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            ..SurfaceSpec::default()
        }
    }

    /// `count` fresh pages a view can be made of, each byte its page's index.
    fn guest_pages(count: u64) -> (u64, Vec<PhysicalMemoryRange>) {
        let page = page_size();
        let at = allocate_aliasable(count * page).unwrap().addr().to_u64();
        for i in 0..count {
            // SAFETY: the pages were just mapped, readable and writable.
            unsafe {
                std::ptr::write_bytes(
                    std::ptr::with_exposed_provenance_mut::<u8>((at + i * page).to_usize()),
                    u8::try_from(i).unwrap(),
                    page.to_usize(),
                );
            }
        }
        let ranges = (0..count)
            .map(|i| PhysicalMemoryRange {
                physical_address: at + i * page,
                physical_length: page,
            })
            .collect();
        (at, ranges)
    }

    /// The range `s` mapped, its first byte and length: its own memory's whole pages, or its
    /// view.
    fn mapped_range(s: &LinuxSurface) -> Option<(u64, u64)> {
        let data = s.data();
        let own = data.memory.own.as_ref().map(|own| {
            let length = data.alloc_size.next_multiple_of(page_size());
            (own.first(), length)
        });
        own.or(data.memory.view)
    }

    fn free(at: u64, length: u64) {
        let vm = LinuxVm::new();
        assert!(
            vm.deallocate(HostPtr::from_addr(at.to_usize()), length)
                .is_success()
        );
    }

    fn read(at: u64) -> u8 {
        // SAFETY: the tests' mapped memory.
        unsafe { std::ptr::with_exposed_provenance::<u8>(at.to_usize()).read_volatile() }
    }

    /// A surface of its own memory is one reference, answers the spec, and its memory is
    /// freed with its last object
    #[test]
    fn a_surface_answers_its_spec_and_counts_its_references() {
        let s = LinuxSurfaces::new().create(&host_only()).unwrap();
        assert_eq!(s.retain_count(), 1);
        assert_eq!((s.pixel_format(), s.width(), s.height()), (BGRA, 16, 16));
        assert_eq!((s.bytes_per_element(), s.bytes_per_row()), (4, 64));
        assert_eq!((s.plane_count(), s.alloc_size()), (0, 1024));
        assert!(s.allows_pixel_size_casting());
        assert_eq!(s.protection_options(), 0);
        assert_eq!(s.lock(0), KernReturn::SUCCESS);
        assert_eq!(s.unlock(0), KernReturn::SUCCESS);
        let t = s.clone_ref();
        assert_eq!((t.id(), s.retain_count()), (s.id(), 2));
        drop(t);
        let base = s.base_address();
        assert!(mapped(base));
        drop(s);
        assert!(!mapped(base));
    }

    /// A lookup is another object over the same memory, whose count is its own, and the
    /// memory outlives the object made while the lookup lives
    #[test]
    fn a_lookup_is_another_object() {
        let s = LinuxSurfaces::new().create(&host_only()).unwrap();
        let t = s.lookup_by_port().unwrap();
        assert_ne!(s.id(), t.id());
        let held = t.clone_ref();
        assert_eq!((s.retain_count(), t.retain_count()), (1, 2));
        assert_eq!(
            (t.base_address(), t.alloc_size()),
            (s.base_address(), s.alloc_size())
        );
        let base = s.base_address();
        drop(s);
        assert!(mapped(base));
        drop((t, held));
        assert!(!mapped(base));
    }

    /// A hold on a surface's memory keeps it mapped past the surface's last object and counts no
    /// reference of it, its own memory and its view of the guest's pages alike; the memory goes
    /// with the hold
    #[test]
    fn a_hold_keeps_the_memory_past_the_last_reference() {
        let page = page_size();
        let (at, ranges) = guest_pages(2);
        let several = SurfaceSpec {
            alloc_size: 2 * page,
            client_address_ranges: Some(ranges),
            ..host_only()
        };
        for spec in [host_only(), several] {
            let s = LinuxSurfaces::new().create(&spec).unwrap();
            let base = s.base_address();
            let hold = surface_memory_hold(&s);
            assert_eq!(s.retain_count(), 1);
            drop(s);
            assert!(mapped(base), "{spec:?}");
            drop(hold);
            assert!(!mapped(base), "{spec:?}");
        }
        free(at, 2 * page);
    }

    /// The layouts IOSurface refuses are refused (module docs), and those at its limits made
    #[test]
    fn refused_layouts_are_none() {
        let surfaces = LinuxSurfaces::new();
        let refused = [
            SurfaceSpec {
                width: u64::MAX,
                height: u64::MAX,
                ..host_only()
            },
            SurfaceSpec {
                alloc_size: 0,
                ..host_only()
            },
            SurfaceSpec {
                bytes_per_row: 60,
                ..host_only()
            },
            SurfaceSpec {
                alloc_size: 1023,
                ..host_only()
            },
            SurfaceSpec {
                offset: 0x40,
                ..host_only()
            },
            SurfaceSpec {
                element_width: 0,
                ..host_only()
            },
            SurfaceSpec {
                client_direction: Some(1),
                ..host_only()
            },
            SurfaceSpec {
                protection_options: 0x10,
                ..host_only()
            },
            SurfaceSpec {
                alloc_size: physical_memory() + 1,
                ..host_only()
            },
        ];
        for spec in refused {
            assert!(surfaces.create(&spec).is_none(), "{spec:?}");
        }
        let made = [
            SurfaceSpec {
                offset: 0x40,
                alloc_size: 1088,
                ..host_only()
            },
            SurfaceSpec {
                bytes_per_row: 100,
                alloc_size: 1564,
                ..host_only()
            },
            // 4 × 4 blocks of 8 bytes: 5 blocks a row, 5 rows.
            SurfaceSpec {
                width: 17,
                height: 17,
                element_width: 4,
                element_height: 4,
                bytes_per_element: 8,
                bytes_per_row: 40,
                alloc_size: 200,
                ..host_only()
            },
            SurfaceSpec {
                client_direction: Some(3),
                ..host_only()
            },
        ];
        for spec in made {
            assert!(surfaces.create(&spec).is_some(), "{spec:?}");
        }
    }

    /// A host-only surface at an offset has its base address that far into its memory
    #[test]
    fn an_offset_is_into_the_allocation() {
        let spec = SurfaceSpec {
            offset: 0x40,
            alloc_size: 2048,
            ..host_only()
        };
        let s = LinuxSurfaces::new().create(&spec).unwrap();
        let first = mapped_range(&s).unwrap().0;
        assert_eq!(s.base_address().addr().to_u64(), first + 0x40);
        assert_eq!(recorded_mut(&s, |made| made.offset).unwrap(), 0x40);
    }

    /// Planes answer their offsets, rows and sizes; one that ends past the allocation, starts
    /// inside the one before or has short rows is refused
    #[test]
    fn planes_are_the_specs() {
        let plane = |offset, width, height, bytes_per_element, bytes_per_row| SurfacePlaneSpec {
            offset,
            base: offset,
            width,
            height,
            bytes_per_element,
            bytes_per_row,
            element_width: 1,
            element_height: 1,
            ..SurfacePlaneSpec::default()
        };
        let two = |p1_offset, p1_row, alloc_size| SurfaceSpec {
            planes: Some(vec![
                plane(0, 16, 16, 1, 16),
                plane(p1_offset, 8, 8, 2, p1_row),
            ]),
            alloc_size,
            bytes_per_element: 1,
            bytes_per_row: 16,
            ..host_only()
        };
        let surfaces = LinuxSurfaces::new();
        let s = surfaces.create(&two(0x101, 16, 0x1000)).unwrap();
        assert_eq!(s.plane_count(), 2);
        assert_eq!(plane_at(&s, 1), Some((0x101, 16)));
        assert_eq!(plane_shape(&s, 1), Some((8, 8, 2)));
        assert_eq!(plane_at(&s, 2), None);
        assert!(surfaces.create(&two(0x100, 16, 0x180)).is_some());
        assert!(surfaces.create(&two(0x100, 16, 0x17f)).is_none());
        assert!(surfaces.create(&two(0x80, 16, 0x1000)).is_none());
        assert!(surfaces.create(&two(0x100, 15, 0x1000)).is_none());
    }

    /// Protection options read back with what each implies, and leave the surface no memory
    #[test]
    fn protection_options_read_back_and_unmap() {
        let answered: Vec<u64> = (0..16).map(|o| protection_options(o).unwrap()).collect();
        assert_eq!(
            answered,
            [0, 1, 3, 3, 7, 7, 7, 7, 8, 9, 11, 11, 15, 15, 15, 15]
        );
        let spec = SurfaceSpec {
            protection_options: 4,
            ..host_only()
        };
        let s = LinuxSurfaces::new().create(&spec).unwrap();
        assert_eq!(s.protection_options(), 7);
        assert!(s.base_address().is_null());
    }

    /// A surface over one range of the guest's pages is that range, at its offset; one over
    /// several is a view of them, read-only where the GPU's writes never reach them
    #[test]
    fn surfaces_over_the_guests_pages() {
        let page = page_size();
        let (at, ranges) = guest_pages(4);
        let surfaces = LinuxSurfaces::new();
        let over = |ranges: &[PhysicalMemoryRange], alloc_size, offset, direction| SurfaceSpec {
            alloc_size,
            offset,
            client_direction: direction,
            client_address_ranges: Some(ranges.to_vec()),
            ..host_only()
        };
        let one = surfaces
            .create(&over(&ranges[2..3], page, 0x40, None))
            .unwrap();
        assert_eq!(one.base_address().addr().to_u64(), at + 2 * page + 0x40);
        assert!(mapped_range(&one).is_none());
        let backwards: Vec<_> = ranges.iter().rev().copied().collect();
        let several = surfaces
            .create(&over(&backwards, 3 * page, 0x40, Some(READ_ONLY_DIRECTION)))
            .unwrap();
        let base = several.base_address().addr().to_u64();
        assert_eq!(
            [read(base), read(base + page), read(base + 2 * page)],
            [3, 2, 1]
        );
        assert!(recorded_mut(&several, |made| made.gpu_read_only).unwrap());
        let (view, length) = mapped_range(&several).unwrap();
        assert_eq!((view, length), (base - 0x40, 3 * page));
        let pieces = super::super::files::pieces(view, length).unwrap();
        assert!(pieces.iter().all(|p| p.prot == libc::PROT_READ));
        assert_eq!(
            recorded_mut(&several, |made| made.pages.clone()).unwrap(),
            backwards
                .iter()
                .map(|r| r.physical_address..r.physical_address + page)
                .collect::<Vec<_>>()
        );
        let refused = [
            over(&ranges[..1], 2 * page, 0, None),
            over(&[], page, 0, None),
            over(
                &[PhysicalMemoryRange {
                    physical_address: at + 0x40,
                    physical_length: page,
                }],
                page,
                0,
                None,
            ),
            SurfaceSpec {
                protection_options: 1,
                ..over(&ranges[..1], page, 0, None)
            },
        ];
        for spec in refused {
            assert!(surfaces.create(&spec).is_none(), "{spec:?}");
        }
        drop((one, several));
        free(at, 4 * page);
    }

    /// A surface over pages no file is behind is refused, nothing left mapped
    #[test]
    fn pages_without_a_file_are_refused() {
        let vm = LinuxVm::new();
        let page = page_size();
        let at = vm.allocate_anywhere(2 * page).unwrap().addr().to_u64();
        let ranges: Vec<_> = (0..2)
            .map(|i| PhysicalMemoryRange {
                physical_address: at + i * page,
                physical_length: page,
            })
            .collect();
        let spec = SurfaceSpec {
            alloc_size: 2 * page,
            client_address_ranges: Some(ranges),
            ..host_only()
        };
        assert!(LinuxSurfaces::new().create(&spec).is_none());
        free(at, 2 * page);
    }

    /// A surface made from plane properties is its planes' whole pages, 1 byte per element and
    /// rows of its width rounded up to 128 bytes
    #[test]
    fn a_backing_surface_is_its_planes() {
        let spec = BackingSurfaceSpec {
            pixel_format: BGRA,
            planes: vec![
                BackingPlaneSpec {
                    width: 16,
                    height: 16,
                    bytes_per_element: 1,
                    bytes_per_row: 16,
                    base: 0,
                    offset: 0,
                },
                BackingPlaneSpec {
                    width: 8,
                    height: 8,
                    bytes_per_element: 2,
                    bytes_per_row: 16,
                    base: 0x5000,
                    offset: 0x5000,
                },
            ],
            width: 17,
            height: 16,
        };
        let surfaces = LinuxSurfaces::new();
        let s = surfaces.create_backing(&spec).unwrap();
        let end = 0x5000 + 7 * 16 + 16;
        assert_eq!(s.alloc_size(), u64::next_multiple_of(end, page_size()));
        assert_eq!((s.bytes_per_element(), s.bytes_per_row()), (1, 128));
        assert_eq!(plane_at(&s, 1), Some((0x5000, 16)));
        assert_eq!(s.retain_count(), 1);
        let mut short = spec.clone();
        short.planes[0].bytes_per_row = 15;
        assert!(surfaces.create_backing(&short).is_none());
        let huge = BackingSurfaceSpec {
            planes: vec![BackingPlaneSpec {
                width: u32::MAX,
                height: u32::MAX,
                bytes_per_element: 4,
                bytes_per_row: 64,
                base: 0,
                offset: 0,
            }],
            width: u32::MAX,
            height: u32::MAX,
            ..spec
        };
        assert!(surfaces.create_backing(&huge).is_none());
    }
}
