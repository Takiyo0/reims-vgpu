//! Stage 1 runtime milestone (manual, `#[ignore]`): prove the Windows window → Vulkan surface
//! chain end to end on real hardware, with zero guest.
//!
//! It opens a real Win32 window on the UI thread, builds a Vulkan instance with `VK_KHR_surface`
//! and `VK_KHR_win32_surface`, creates the window's surface through the backend's own
//! [`Target::create_surface`] (resolving `vkCreateWin32SurfaceKHR` via the instance's
//! `vkGetInstanceProcAddr`, exactly as the runtime does), then — rather than ASSUMING present
//! support — enumerates the physical device's queue families and REPORTS which index supports
//! present to this surface (`vkGetPhysicalDeviceSurfaceSupportKHR`), cross-checked against
//! `vkGetPhysicalDeviceWin32PresentationSupportKHR`. Finally it exercises the required teardown
//! order: the surface is destroyed before the window.
//!
//! Run it on the host with the GPU:
//! ```text
//! cargo test -p reims-vgpu-window --target x86_64-pc-windows-msvc --offline \
//!     -- --ignored --nocapture native::windows::stage1_test
//! ```
//! It needs a Vulkan loader (`vulkan-1.dll`) and a windowing session; it prints its findings and
//! only hard-fails if the surface cannot be created or no queue family presents to it.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ash::vk;

use super::render::{FrameSink, Target};
use super::ui;
use crate::config::WindowConfig;
use crate::handle::Window;
use crate::present::{Presenter, VulkanTarget};

/// A sink that makes no presenter: the render thread runs and ends on close without drawing. The
/// Stage 1 test creates the surface itself (through the backend `Target`), so the sink needs none.
struct NoPresenter;

impl FrameSink for NoPresenter {
    fn presenter(&self, _target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>> {
        None
    }
}

/// Opens a window, creates its Vulkan surface, reports the present-capable queue family, and tears
/// down surface-before-window. Ignored by default: it opens a real window and talks to the GPU.
#[test]
#[ignore = "opens a real window and uses the Vulkan loader: run manually on the GPU host"]
fn stage1_window_surface_and_present_queue() {
    // 1. A minimal window, opened through the real attach() path (starts the render thread and
    //    posts open_window to the UI thread), so close() later routes through the deferred
    //    destruction (close -> render thread releases -> ui posts released -> DestroyWindow).
    let window = Window::new(WindowConfig::default(), 0, crate::handle::title(0));
    // Force the UI thread up first and prove it runs by listing screens.
    let screens = crate::screens::host_screens();
    println!(
        "stage1: UI thread up; {} host screen(s): {screens:?}",
        screens.len()
    );
    window.attach(Arc::new(NoPresenter));
    let id = window.id();
    let hwnd = wait_for(|| ui::test_hwnd(id), Duration::from_secs(10))
        .expect("the window did not open within 10s (CreateWindowExW failed or UI thread stalled)");
    assert!(!hwnd.is_null(), "a null HWND");
    // SAFETY: the process module handle.
    let hinstance =
        unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null()) };
    let target = Target { hwnd, hinstance };

    // 2. A Vulkan instance with the window-surface extensions.
    // SAFETY: loading the Vulkan loader runs its initialization; it is kept while `entry` lives.
    let entry = unsafe { ash::Entry::load_from("vulkan-1.dll") }
        .expect("the Vulkan loader (vulkan-1.dll) did not open");
    let exts = [
        ash::khr::surface::NAME.as_ptr(),
        ash::khr::win32_surface::NAME.as_ptr(),
    ];
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_extension_names(&exts);
    // SAFETY: a valid create info, no layers, only loader-listed surface extensions.
    let instance = unsafe { entry.create_instance(&info, None) }
        .expect("could not create a Vulkan instance with VK_KHR_win32_surface");

    // 3. The window's surface, through the backend's own VulkanTarget (the real path).
    let gipa = entry.static_fn().get_instance_proc_addr;
    let surface = target
        .create_surface(gipa, instance.handle())
        .expect("the backend created no VK_KHR_win32_surface");
    assert_ne!(surface, vk::SurfaceKHR::null(), "a null surface");
    println!("stage1: created VK_KHR_win32_surface {surface:?} for HWND {hwnd:?}");

    // 4. Report the present-capable queue family on each physical device (do NOT assume it).
    let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
    let win32_fn = ash::khr::win32_surface::Instance::new(&entry, &instance);
    // SAFETY: `instance` is live.
    let devices = unsafe { instance.enumerate_physical_devices() }.expect("no physical devices");
    assert!(!devices.is_empty(), "no Vulkan physical devices");
    let mut any_present = false;
    for (di, &pd) in devices.iter().enumerate() {
        // SAFETY: `pd` is a device the instance listed.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = props
            .device_name_as_c_str()
            .unwrap_or(c"<unknown>")
            .to_string_lossy();
        // SAFETY: `pd` is live; the count is read then the slice filled.
        let fams = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        for qi in 0..fams.len() as u32 {
            // SAFETY: `pd`, `qi` and `surface` are all live and from this instance.
            let supports =
                unsafe { surface_fn.get_physical_device_surface_support(pd, qi, surface) }
                    .unwrap_or(false);
            // SAFETY: `pd` and `qi` are live; the Win32 query takes no surface.
            let win32_supports =
                unsafe { win32_fn.get_physical_device_win32_presentation_support(pd, qi) };
            if supports {
                any_present = true;
                println!(
                    "stage1: device {di} ({name}): queue family {qi} PRESENTS to the surface \
                     (win32-presentation-support={win32_supports})"
                );
            }
        }
    }
    assert!(
        any_present,
        "no queue family on any device presents to the window surface"
    );

    // 5. Teardown order: the surface is destroyed BEFORE the window.
    // SAFETY: `surface` was made on `surface_fn`'s instance and nothing uses it now.
    unsafe { surface_fn.destroy_surface(surface, None) };
    println!("stage1: destroyed the surface (before the window)");
    // SAFETY: nothing made of `instance` outlives it.
    unsafe { instance.destroy_instance(None) };

    // The window's close routes through the deferred-destruction path (close -> render thread
    // releases -> ui posts released -> DestroyWindow). The sink makes no presenter, so the render
    // thread ends promptly on close.
    window.close();
    // Give the UI thread a moment to perform the deferred DestroyWindow.
    let gone = wait_for(
        || ui::test_hwnd(id).is_none().then_some(()),
        Duration::from_secs(3),
    );
    assert!(gone.is_some(), "the window was not destroyed after close");
    println!("stage1: window destroyed after the surface — teardown order held");
}

/// Polls `f` until it is `Some` or `timeout` elapses (the UI thread runs the message pump);
/// returns the value or `None`.
fn wait_for<T>(mut f: impl FnMut() -> Option<T>, timeout: Duration) -> Option<T> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(v) = f() {
            return Some(v);
        }
        if Instant::now() >= deadline {
            return None;
        }
        thread::sleep(Duration::from_millis(20));
    }
}
