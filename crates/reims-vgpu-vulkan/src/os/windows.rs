//! The OS services on Windows: the clocks ([`WindowsClock`]), threads ([`WindowsThreads`]), Mach
//! VM's calls over Win32 ([`WindowsVm`]), whose views of the guest's pages map the section
//! objects the backing table names (`files`), IOSurface's stand-in over that memory
//! ([`WindowsSurface`]), and what the window system's Vulkan extensions take here (`crate::os`'s
//! `wsi`: `VK_KHR_win32_surface` and the loader). No Objective-C runs here: core pushes no
//! autorelease pool where none is installed.
//!
//! Where Linux leans on `mmap`/`memfd_create`/`mprotect`/`CLOCK_BOOTTIME`, Windows leans on
//! placeholder reservations (`VirtualAlloc2` with `MEM_RESERVE_PLACEHOLDER`), section objects
//! (`CreateFileMapping` over the pagefile) mapped at a fixed address with
//! `MapViewOfFile3`/`MEM_REPLACE_PLACEHOLDER`, `VirtualProtect`, and the unbiased interrupt time.

mod clock;
mod files;
mod import;
mod section;
mod surface;
mod threads;
mod vm;

use std::ffi::CStr;

use ash::vk;
use metal2vulkan_runtime as rt;

pub use clock::WindowsClock;
pub use files::{GuestMemoryLease, lease_guest_memory};
pub use surface::{WindowsSurface, WindowsSurfaces};
pub use threads::WindowsThreads;
pub use vm::WindowsVm;

/// The backing object behind a guest-RAM range (`crate::os::GuestRamFile`): a section object on
/// Windows, a view of which maps the guest's pages at a fixed address.
pub use section::Section as BackingHandle;

use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, VIRTUAL_ALLOCATION_TYPE};

/// `MapViewOfFile3` with no extended parameters, as the VM backend calls it: a view of
/// `filemapping` from byte `offset`, `view_size` bytes, into the placeholder at `base`
/// (`allocation_type` carries `MEM_REPLACE_PLACEHOLDER`), at `protection` (`PAGE_*`). The
/// returned `MEMORY_MAPPED_VIEW_ADDRESS` has a null `Value` on failure.
///
/// # Safety
///
/// `base` must name a placeholder of exactly `view_size` bytes the caller owns, and the section
/// must have at least `offset + view_size` bytes. The returned view is unmapped by the caller.
pub(crate) unsafe fn map_view_of_file3(
    filemapping: HANDLE,
    base: *mut core::ffi::c_void,
    offset: u64,
    view_size: usize,
    allocation_type: VIRTUAL_ALLOCATION_TYPE,
    protection: u32,
) -> MEMORY_MAPPED_VIEW_ADDRESS {
    // SAFETY: forwarded under the caller's contract; current-process (null) target, no extended
    // parameters.
    unsafe {
        windows_sys::Win32::System::Memory::MapViewOfFile3(
            filemapping,
            std::ptr::null_mut(),
            base.cast_const(),
            offset,
            view_size,
            allocation_type,
            protection,
            std::ptr::null_mut(),
            0,
        )
    }
}

pub(crate) use import::{TWINS_READ_ONLY, allocate_zeros, import_view, shown_at};
pub(crate) use surface::{plane_at, plane_shape, recorded_mut, ycbcr_matrix};
#[cfg(test)]
pub(crate) use vm::allocate_aliasable;
#[cfg(test)]
pub(crate) use vm::tests::{PAGE, mapped, writable};

/// The instance's surface extension a display's window may present over, besides
/// `VK_KHR_surface`, asked for where the loader lists it (`crate::os`'s `wsi`):
/// `VK_KHR_win32_surface`, over the window's `HWND`.
pub(crate) const WINDOW_SURFACE_EXTENSIONS: [&CStr; 1] = [ash::khr::win32_surface::NAME];

/// The device extensions a window's swapchain needs besides `VK_KHR_swapchain`: none.
pub(crate) const WINDOW_DEVICE_EXTENSIONS: [&CStr; 0] = [];

/// The Vulkan loader, opened by its usual name on Windows (`vulkan-1.dll`), as the runtime opens
/// it.
pub(crate) fn open_loader() -> Result<ash::Entry, ash::LoadingError> {
    // SAFETY: opening the loader runs its initialization, and it is kept open while the entry
    // lives.
    unsafe { ash::Entry::load_from("vulkan-1.dll") }
}

/// What fetches a swapchain image's drawable ahead of its frame: nothing here, where an image's
/// memory is the driver's own once it is acquired (`crate::swapchain`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Drawables {}

impl Drawables {
    /// None on Windows.
    pub(crate) fn load(_raw: &rt::Device) -> Option<Drawables> {
        None
    }

    /// Never called: there is no value.
    ///
    /// # Safety
    ///
    /// None needed.
    pub(crate) unsafe fn fetch(&self, _image: vk::Image) -> bool {
        match *self {}
    }
}

/// The clocks.
pub type OsClock = WindowsClock;
/// The threads.
pub type OsThreads = WindowsThreads;
/// Mach VM's calls.
pub type OsVm = WindowsVm;
/// An IOSurface: the stand-in's, one reference.
pub type OsSurface = WindowsSurface;
/// IOSurface creation: the stand-in's.
pub type OsSurfaces = WindowsSurfaces;

/// Installs what the process's platform needs before it runs: nothing on Windows. Unlike Linux,
/// there is no open-file limit to raise -- a section object is a handle, not a file descriptor,
/// and Windows's handle table is not bounded by a soft `RLIMIT_NOFILE`. Unlike macOS, no
/// autorelease pool is pushed.
pub fn install_process_services() {}

/// The macOS release whose Metal the runtime's devices answer as where releases differ
/// (`DeviceOptions::metal_level`): macOS 26.7, as on Linux -- a Windows host has no Metal to ask,
/// so it answers the guest as the measured Mac does. See the Linux note.
pub const METAL_LEVEL: Option<rt::MetalLevel> = Some(rt::MetalLevel::MacOS26_7);

/// The registry ID the platform answers for the runtime's default device in place of its own:
/// none, as there is no Metal here to ask, so the runtime's own stands.
pub fn default_registry_id() -> Option<u64> {
    None
}

/// The GPU core count Metal answers for the device [`default_registry_id`] names: none, as there
/// is no Metal here to ask.
pub fn default_gpu_core_count() -> Option<u64> {
    None
}
