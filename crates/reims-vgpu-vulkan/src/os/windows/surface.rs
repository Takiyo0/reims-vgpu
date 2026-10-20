//! IOSurface's stand-in on Windows: what `IOSurfaceCreate` and `[[IOSurface alloc]
//! initWithProperties:]` answer on macOS, over memory a view can be made of.
//!
//! - **References.** A [`WindowsSurface`] is one reference: `clone_ref` is a retain, `Drop` a
//!   release, and the retain count, which the guest sees, is the number of references to the
//!   object. Its ID is the object's address, unique among live surfaces. A lookup
//!   (`lookup_by_port`) is another object over the same memory, with a count of its own, as
//!   IOSurface's lookup in the process that made the surface answers.
//! - **Memory.** A surface of its own memory is a section of its own, mapped and registered in
//!   the backing table ([`super::vm::OwnMemory`]), so a view can be made of it (an alias of its
//!   allocation, a buffer over it) and no `deallocate` frees it. One over the guest's pages
//!   (`kIOSurfaceClientAddressRanges`) of one range is that range, its base address its offset
//!   into it; one of several ranges is a view of them ([`super::vm::WindowsVm`]), read-only for
//!   the CPU where it is read-only for the GPU (`kIOSurfaceClientDirection` 2), as IOSurface maps
//!   it on macOS. A protected surface (protection options other than 0) has no memory the CPU
//!   reaches: its base address is NULL.
//! - **What it answers** is the spec's: its planes, rows and sizes, its allocation size exactly
//!   as given. It holds no values (`IOSurfaceCopyValue`): nothing in the process sets one.
//!   `lock` and `unlock` do nothing.
//!
//! This mirrors the Linux stand-in byte for byte, with Windows VM calls behind
//! [`super::vm::OwnMemory`] and the view, and the host's managed memory read from
//! `GlobalMemoryStatusEx` in place of `_SC_PHYS_PAGES`. The refusal rules (measured on the M3
//! Max; see the Linux module docs) are identical.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_core::host::{HostPtr, PhysicalMemoryRange};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::surface::{
    BackingSurfaceSpec, IoSurfaces, Surface, SurfacePlaneSpec, SurfaceSpec,
};
use reims_vgpu_core::platform::os::vm::{HostVm as _, KernReturn, VmProt};
use reims_vgpu_core::platform::{Handle, ObjId};
use reims_vgpu_core::sync::Mutex;

use super::vm::{OwnMemory, WindowsVm};
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
pub struct WindowsSurface(Arc<Object>);

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
    /// Its own memory, freed with it.
    own: Option<OwnMemory>,
    /// The view of the guest's pages it made, its first byte and length, freed with it.
    view: Option<(u64, u64)>,
    /// What it was made over ([`crate::os::surface_memory`] and the helpers beside it).
    made: Mutex<MadeOver>,
    /// The physical ranges in guest RAM backing this surface if using OwnMemory fallback under WHPX.
    source_ranges: Vec<PhysicalMemoryRange>,
    /// Lock serializing refresh and sync operations.
    sync_lock: Mutex<()>,
    /// Whether this surface was written by the GPU.
    gpu_written: AtomicBool,
}

impl Drop for Data {
    /// Frees the view the surface made, and its own memory.
    fn drop(&mut self) {
        if let Some((first, length)) = self.view {
            let _ = WindowsVm::new().deallocate(HostPtr::from_addr(first.to_usize()), length);
        }
        drop(self.own.take());
    }
}

impl WindowsSurface {
    /// A new surface: one reference to a new object over `data`.
    fn new(data: Data) -> WindowsSurface {
        WindowsSurface(Arc::new(Object(Arc::new(data))))
    }

    /// The surface the object is over.
    fn data(&self) -> &Data {
        &self.0.0
    }

    /// Refreshes `own` memory from `source_ranges` (guest memory) if this surface was created
    /// via the WHPX multi-range fallback.
    pub fn refresh(&self) {
        let data = self.data();
        let Some(own) = &data.own else { return };
        if data.source_ranges.is_empty() {
            return;
        }
        if data.gpu_written.load(Ordering::Acquire) {
            return;
        }
        let _guard = data.sync_lock.lock();
        if data.gpu_written.load(Ordering::Acquire) {
            return;
        }
        let length = own.length();
        let dst = own.first() as *mut u8;
        let mut offset = 0usize;
        for r in &data.source_ranges {
            if offset as u64 >= length {
                break;
            }
            let now = (r.physical_length.min(length.saturating_sub(offset as u64))) as usize;
            let src = r.physical_address as *const u8;
            // SAFETY: Under WHPX, physical_address is the direct host VA within QEMU's guest RAM allocation.
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.add(offset), now);
            }
            offset += now;
        }
    }

    /// Synchronizes changes from `own` back to `source_ranges` (guest memory).
    pub fn sync_to_guest(&self) {
        let data = self.data();
        let Some(own) = &data.own else { return };
        if data.source_ranges.is_empty() {
            return;
        }
        let _guard = data.sync_lock.lock();
        let length = own.length();
        let src = own.first() as *const u8;
        let mut offset = 0usize;
        for r in &data.source_ranges {
            if offset as u64 >= length {
                break;
            }
            let now = (r.physical_length.min(length.saturating_sub(offset as u64))) as usize;
            let dst = r.physical_address as *mut u8;
            // SAFETY: Under WHPX, physical_address is the direct host VA within QEMU's guest RAM allocation.
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(offset), dst, now);
            }
            offset += now;
        }
        data.gpu_written.store(false, Ordering::Release);
    }

    /// Synchronizes guest changes into `own` unless the GPU owns newer contents. A writeback
    /// makes both copies agree, after which subsequent guest writes may be refreshed.
    pub fn sync_from_guest(&self) {
        self.refresh();
    }

    /// Marks the surface as written by the GPU.
    pub fn mark_gpu_written(&self) {
        let data = self.data();
        let _guard = data.sync_lock.lock();
        data.gpu_written.store(true, Ordering::Release);
    }

    /// Clears the GPU-written flag, allowing subsequent `refresh()` calls to update from guest RAM.
    pub fn clear_gpu_written(&self) {
        let data = self.data();
        let _guard = data.sync_lock.lock();
        data.gpu_written.store(false, Ordering::Release);
    }

    /// Whether this surface was marked as written by the GPU.
    pub fn is_gpu_written(&self) -> bool {
        self.data().gpu_written.load(Ordering::Acquire)
    }
}

impl Handle for WindowsSurface {
    fn clone_ref(&self) -> Self {
        WindowsSurface(Arc::clone(&self.0))
    }

    fn id(&self) -> ObjId {
        ObjId(Arc::as_ptr(&self.0).addr().to_u64())
    }
}

impl Surface for WindowsSurface {
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
        Some(WindowsSurface(Arc::new(Object(Arc::clone(&self.0.0)))))
    }

    fn refresh(&self) {
        WindowsSurface::refresh(self);
    }

    fn sync_from_guest(&self) {
        WindowsSurface::sync_from_guest(self);
    }

    fn sync_to_guest(&self) {
        WindowsSurface::sync_to_guest(self);
    }

    fn mark_gpu_written(&self) {
        WindowsSurface::mark_gpu_written(self);
    }
}

/// Plane `plane` of `surface`: its offset from the base address and its bytes per row, as the
/// spec gave them; `None` for a plane the surface lacks.
pub(crate) fn plane_at(surface: &WindowsSurface, plane: u64) -> Option<(u64, u64)> {
    let p = surface.data().planes.get(usize::try_from(plane).ok()?)?;
    Some((p.offset, p.bytes_per_row))
}

/// The width, height and bytes per element of plane `plane` of `surface`; `None` for a plane it
/// lacks.
pub(crate) fn plane_shape(surface: &WindowsSurface, plane: u64) -> Option<(u64, u64, u64)> {
    let p = surface.data().planes.get(usize::try_from(plane).ok()?)?;
    Some((p.width, p.height, p.bytes_per_element))
}

/// The YCbCr matrix `surface` names (`kIOSurfaceYCbCrMatrix`): none, as the stand-in holds no
/// values and nothing sets one.
pub(crate) fn ycbcr_matrix(
    _surface: &WindowsSurface,
) -> Option<metal2vulkan_runtime::SurfaceYCbCrMatrix> {
    None
}

/// What `surface` was made over, `change` given it: every stand-in holds its own record.
pub(crate) fn recorded_mut<T>(
    surface: &WindowsSurface,
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

/// The host's physical memory, in bytes, as Windows reports it (`GlobalMemoryStatusEx`'s
/// `ullTotalPhys`): the largest allocation the stand-in makes, as macOS's IOSurface makes none
/// past `hw.memsize_usable` (module docs).
fn physical_memory() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = unsafe { std::mem::zeroed::<MEMORYSTATUSEX>() };
    status.dwLength = u32::try_from(std::mem::size_of::<MEMORYSTATUSEX>()).unwrap_or(0);
    // SAFETY: a live out-parameter the call fills; dwLength set as it requires.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return 0;
    }
    status.ullTotalPhys
}

/// The host's page size.
fn page_size() -> u64 {
    WindowsVm::new().page_size()
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
/// its first byte, or `None` where a range has no section behind it or the host gives no address
/// space, nothing left mapped.
fn view(ranges: &[PhysicalMemoryRange], length: u64, read_only: bool) -> Option<u64> {
    let vm = WindowsVm::new();
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
                 library has no section of; refused",
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

/// IOSurface's creation on Windows (module docs).
#[derive(Debug, Default)]
pub struct WindowsSurfaces;

impl WindowsSurfaces {
    /// The process's IoSurfaces.
    pub fn new() -> WindowsSurfaces {
        WindowsSurfaces
    }
}

impl IoSurfaces<VulkanPlatform> for WindowsSurfaces {
    fn create(&self, spec: &SurfaceSpec) -> Option<WindowsSurface> {
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
        let (base, own, view_made, source_ranges) = match &spec.client_address_ranges {
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
                    [one] => (one.physical_address, None, None, Vec::new()),
                    _ => {
                        let length = spec.alloc_size.checked_next_multiple_of(page_size())?;
                        match view(ranges, length, made.gpu_read_only) {
                            Some(at) => (at, None, Some((at, length)), Vec::new()),
                            None => {
                                log::info!(
                                    target: "reims_vgpu_vulkan::os",
                                    "WHPX multi-range surface fallback: copying {length:#x} bytes across {} ranges into OwnMemory",
                                    ranges.len()
                                );
                                let own = OwnMemory::new(length)?;
                                let dst = own.first() as *mut u8;
                                let mut offset = 0usize;
                                for r in ranges {
                                    if offset as u64 >= length {
                                        break;
                                    }
                                    let now = (r
                                        .physical_length
                                        .min(length.saturating_sub(offset as u64)))
                                        as usize;
                                    let src = r.physical_address as *const u8;
                                    // SAFETY: Under WHPX, physical_address is the direct host VA within QEMU's guest RAM allocation.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(src, dst.add(offset), now);
                                    }
                                    offset += now;
                                }
                                (own.first(), Some(own), None, ranges.clone())
                            }
                        }
                    }
                }
            }
            None if protection != 0 => (0, None, None, Vec::new()),
            None => {
                let own = OwnMemory::new(spec.alloc_size)?;
                (own.first(), Some(own), None, Vec::new())
            }
        };
        Some(WindowsSurface::new(Data {
            pixel_format: spec.pixel_format,
            width: spec.width,
            height: spec.height,
            bytes_per_element: spec.bytes_per_element,
            bytes_per_row: spec.bytes_per_row,
            planes,
            protection_options: protection,
            alloc_size: spec.alloc_size,
            base: if base == 0 { 0 } else { base + spec.offset },
            own,
            view: view_made,
            made: Mutex::new(made),
            source_ranges,
            sync_lock: Mutex::new(()),
            gpu_written: AtomicBool::new(false),
        }))
    }

    fn create_backing(&self, spec: &BackingSurfaceSpec) -> Option<WindowsSurface> {
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
        let (bytes_per_element, bytes_per_row) = match spec.planes.first() {
            Some(p) => (
                u64::try_from(p.bytes_per_element).unwrap_or(4),
                u64::try_from(p.bytes_per_row).unwrap_or_else(|_| width.saturating_mul(4)),
            ),
            None => (
                4,
                width
                    .saturating_mul(4)
                    .checked_next_multiple_of(BACKING_ROW_ALIGNMENT)
                    .unwrap_or(width.saturating_mul(4)),
            ),
        };
        let own = OwnMemory::new(alloc_size)?;
        Some(WindowsSurface::new(Data {
            pixel_format: spec.pixel_format,
            width,
            height,
            bytes_per_element,
            bytes_per_row,
            planes,
            protection_options: 0,
            alloc_size,
            base: own.first(),
            own: Some(own),
            view: None,
            made: Mutex::new(MadeOver::default()),
            source_ranges: Vec::new(),
            sync_lock: Mutex::new(()),
            gpu_written: AtomicBool::new(false),
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

    use super::super::vm::Prot;
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
    fn mapped_range(s: &WindowsSurface) -> Option<(u64, u64)> {
        let data = s.data();
        let own = data.own.as_ref().map(|own| {
            let length = data.alloc_size.next_multiple_of(page_size());
            (own.first(), length)
        });
        own.or(data.view)
    }

    fn free(at: u64, length: u64) {
        let vm = WindowsVm::new();
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
        let s = WindowsSurfaces::new().create(&host_only()).unwrap();
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
        let s = WindowsSurfaces::new().create(&host_only()).unwrap();
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

    /// The layouts IOSurface refuses are refused (module docs), and those at its limits made
    #[test]
    fn refused_layouts_are_none() {
        let surfaces = WindowsSurfaces::new();
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
        let s = WindowsSurfaces::new().create(&spec).unwrap();
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
        let surfaces = WindowsSurfaces::new();
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
        let s = WindowsSurfaces::new().create(&spec).unwrap();
        assert_eq!(s.protection_options(), 7);
        assert!(s.base_address().is_null());
    }

    /// A surface over one range of the guest's pages is that range, at its offset; one over
    /// several is a view of them, read-only where the GPU's writes never reach them
    #[test]
    fn surfaces_over_the_guests_pages() {
        let page = page_size();
        let (at, ranges) = guest_pages(4);
        let surfaces = WindowsSurfaces::new();
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
        assert!(pieces.iter().all(|p| p.prot == Prot::READ));
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

    /// A multi-range surface over pages with no section falls back to OwnMemory under WHPX;
    /// guest changes refresh it, GPU-owned contents survive guest refreshes, and writeback
    /// returns ownership to the guest.
    #[test]
    fn pages_without_a_section_fall_back_to_own_memory_and_refresh() {
        let vm = WindowsVm::new();
        let page = page_size();
        let at = vm.allocate_anywhere(2 * page).unwrap().addr().to_u64();
        // Write initial data to guest memory (all 0x11).
        unsafe {
            std::ptr::write_bytes(
                std::ptr::with_exposed_provenance_mut::<u8>(at.to_usize()),
                0x11,
                (2 * page).to_usize(),
            );
        }
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
        let surface = WindowsSurfaces::new()
            .create(&spec)
            .expect("fallback to OwnMemory");
        let base = surface.base_address().addr().to_u64();
        assert_ne!(base, 0);
        assert_ne!(base, at); // base is in OwnMemory, not pointing at `at`
        assert_eq!(read(base), 0x11);

        // Guest writes new data (0x42) after surface creation.
        unsafe {
            std::ptr::write_bytes(
                std::ptr::with_exposed_provenance_mut::<u8>(at.to_usize()),
                0x42,
                (2 * page).to_usize(),
            );
        }
        // Before refresh, OwnMemory still has old 0x11.
        assert_eq!(read(base), 0x11);

        // After refresh, OwnMemory has updated 0x42.
        surface.refresh();
        assert_eq!(read(base), 0x42);

        // A GPU write is newer than the guest's copy. Refresh must not replace it.
        surface.mark_gpu_written();
        unsafe {
            std::ptr::write_bytes(
                std::ptr::with_exposed_provenance_mut::<u8>(base.to_usize()),
                0x77,
                (2 * page).to_usize(),
            );
        }
        unsafe {
            std::ptr::write_bytes(
                std::ptr::with_exposed_provenance_mut::<u8>(at.to_usize()),
                0x55,
                (2 * page).to_usize(),
            );
        }
        surface.sync_from_guest();
        assert_eq!(read(base), 0x77);
        assert!(surface.is_gpu_written());

        assert_eq!(read(at), 0x55);

        // After sync_to_guest, guest memory has 0x77 and both copies agree.
        surface.sync_to_guest();
        assert_eq!(read(at), 0x77);
        assert!(!surface.is_gpu_written());

        // A subsequent guest write can now refresh the host copy.
        unsafe {
            std::ptr::write_bytes(
                std::ptr::with_exposed_provenance_mut::<u8>(at.to_usize()),
                0x33,
                (2 * page).to_usize(),
            );
        }
        surface.sync_from_guest();
        assert_eq!(read(base), 0x33);

        drop(surface);
        free(at, 2 * page);
    }

    /// A compute input and a texture-copy source are reads even when their allocation also
    /// permits GPU writes. Neither may freeze the surface's guest-memory staging.
    #[test]
    fn read_only_bindings_and_copy_sources_keep_guest_refresh_enabled() {
        use reims_vgpu_core::platform::Platform;
        use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
        use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
        use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
        use reims_vgpu_core::platform::device::GpuDevice;
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        use reims_vgpu_core::platform::values::{ColorAttachmentSet, TextureDesc};
        use reims_vgpu_core::raise::OnFault;
        let vm = WindowsVm::new();
        let page = page_size();
        let at = vm.allocate_anywhere(2 * page).unwrap().addr().to_u64();
        let ranges = (0..2)
            .map(|i| PhysicalMemoryRange {
                physical_address: at + i * page,
                physical_length: page,
            })
            .collect();
        let surface = WindowsSurfaces::new()
            .create(&SurfaceSpec {
                alloc_size: 2 * page,
                client_address_ranges: Some(ranges),
                ..host_only()
            })
            .unwrap();
        let p = crate::platform::tests::platform();
        let d = crate::gpu::device::tests::dev();
        let descriptor = p.texture_descriptor(&TextureDesc {
            width: 16,
            height: 16,
            pixel_format: 80,
            usage: 3,
            ..TextureDesc::default()
        });
        let t = d
            .new_texture_on_surface(&descriptor, Some(&surface), 0)
            .unwrap()
            .unwrap();
        let queue = d.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        e.encode(
            OnFault::Return,
            ComputeCmd::SetTexture {
                texture: Some(&t),
                index: 0,
            },
        )
        .unwrap();
        assert!(!surface.is_gpu_written(), "binding an input is not a write");
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Return,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&t)),
                load_action: Some(1),  // Load.
                store_action: Some(1), // Store.
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        assert!(
            !surface.is_gpu_written(),
            "assigning an attachment does not constitute a GPU write"
        );
        // SAFETY: allocated guest pages, no GPU work writes this source.
        unsafe {
            std::ptr::write_bytes(at.to_usize() as *mut u8, 0x42, (2 * page).to_usize());
        }
        t.copied("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:");
        assert!(
            !surface.is_gpu_written(),
            "a copy source is not a destination"
        );
        assert_eq!(read(surface.base_address().addr().to_u64()), 0x42);
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        let load_encoder = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        assert!(
            !surface.is_gpu_written(),
            "opening a load/store pass without drawing is not a write"
        );
        load_encoder
            .encode(OnFault::Return, RenderCmd::EndEncoding)
            .unwrap();

        let clear_pass = p.render_pass_descriptor();
        clear_pass
            .set_color_attachment(
                OnFault::Return,
                0,
                &ColorAttachmentSet {
                    texture: Some(Some(&t)),
                    load_action: Some(2),  // Clear.
                    store_action: Some(1), // Store.
                    ..ColorAttachmentSet::default()
                },
            )
            .unwrap();
        let clear_encoder = cb
            .render_encoder(OnFault::Return, &clear_pass)
            .unwrap()
            .unwrap();
        assert!(surface.is_gpu_written(), "a clear owns the target surface");
        clear_encoder
            .encode(OnFault::Return, RenderCmd::EndEncoding)
            .unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        drop((e, cb, queue, t, surface));
        free(at, 2 * page);
    }

    /// A surface made from plane properties is its planes' whole pages, its first plane's
    /// element and row bytes
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
        let surfaces = WindowsSurfaces::new();
        let s = surfaces.create_backing(&spec).unwrap();
        let end = 0x5000 + 7 * 16 + 16;
        assert_eq!(s.alloc_size(), u64::next_multiple_of(end, page_size()));
        assert_eq!((s.bytes_per_element(), s.bytes_per_row()), (1, 16));
        assert_eq!(plane_at(&s, 1), Some((0x5000, 16)));
        assert_eq!(s.retain_count(), 1);
        let spec_bgra = BackingSurfaceSpec {
            pixel_format: BGRA,
            planes: vec![BackingPlaneSpec {
                width: 16,
                height: 16,
                bytes_per_element: 4,
                bytes_per_row: 64,
                base: 0,
                offset: 0,
            }],
            width: 16,
            height: 16,
        };
        let s_bgra = surfaces.create_backing(&spec_bgra).unwrap();
        assert_eq!(
            (s_bgra.bytes_per_element(), s_bgra.bytes_per_row()),
            (4, 64)
        );
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
