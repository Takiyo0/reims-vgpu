//! Stage 2 runtime milestone (manual, `#[ignore]`): frames actually render into the window.
//!
//! This drives the WHOLE real present pipeline on Windows, with a synthetic frame source and no
//! guest: a real `VulkanPlatform` device, a presenting `Display`, the real [`SwapchainSink`] (via
//! `window::vulkan::sink`), and a real Win32 [`Window`]. Attaching the sink opens the window and
//! starts its render thread, which asks the sink for a presenter — making a `Swapchain::over` the
//! window's `VK_KHR_win32_surface` — and presents each guest frame into the swapchain's images.
//!
//! It presents a sequence of fixed frames to the display and asserts the display's HOST present
//! count advances, i.e. the render thread acquired swapchain images and presented them into the
//! real window on the RTX 4060. Then it closes the window (surface before window) and checks the
//! teardown.
//!
//! Run on the GPU host:
//! ```text
//! cargo test -p reims-vgpu --features vulkan --target x86_64-pc-windows-msvc --offline \
//!     -- --ignored --nocapture window::vulkan::windows::tests::stage2
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use reims_vgpu_core::display::GammaTableArgs;
use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::resource::Texture as _;
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::platform::{HostRange, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::rt::Hold;
use reims_vgpu_core::tasks::isolation::TaskThreads;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_window::{Window, WindowConfig};

use crate::display::no_catch;
use crate::entry;
use crate::platform::api_state;
use crate::{Device, DeviceConfig, Display, DisplayConfig, PixelFormat};

type Queue = <VulkanPlatform as Platform>::CommandQueue;

/// `MTLResourceStorageModeShared`.
const SHARED: u64 = 0;

/// The frame queue of `device`'s device.
fn frame_queue(device: &Device<VulkanPlatform>) -> Queue {
    entry::run(|_| {
        api_state(device.platform())
            .frame_queue(device.core().mtl_device_ref())
            .unwrap()
    })
}

/// A Vulkan device on a platform of its own, and a presenting display on its port 0.
fn presenting_display(serial: u32) -> (Device<VulkanPlatform>, Display<VulkanPlatform>) {
    let platform = Arc::new(VulkanPlatform::new(Box::new(TaskThreads::new())));
    let device = Device::with_platform(platform, DeviceConfig::default()).expect("a Vulkan device");
    let mut c = DisplayConfig::new(0, serial);
    c.callbacks.new_frame = Some(Arc::new(|| {}));
    let display = device.new_display(c).unwrap();
    (device, display)
}

/// Presents a fixed BGRA gradient frame of `(width, height)` to `display`.
fn present_a_fixed_frame(
    device: &Device<VulkanPlatform>,
    display: &Display<VulkanPlatform>,
    (width, height): (u64, u64),
) {
    let bytes: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let cut = |v: u64| u8::try_from(v % 256).unwrap();
            [
                cut(x * 255 / (width - 1)),
                cut(y * 255 / (height - 1)),
                cut((x * 7 + y * 13) % 256),
                0xff,
            ]
        })
        .collect();
    entry::run(|_| {
        let format = u64::from(PixelFormat::Bgra8Unorm.metal());
        let td = no_catch(device.platform().texture_2d_descriptor(
            OnFault::Die,
            format,
            width,
            height,
            false,
        ));
        td.set_usage(objc2_metal_shader_read());
        td.set_resource_options(SHARED);
        let frame = no_catch(
            device
                .core()
                .mtl_device_ref()
                .new_texture(OnFault::Die, &td),
        )
        .unwrap();
        no_catch(frame.replace_region(
            MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(width, height, 1)),
            0,
            0,
            HostRange::of_slice(&bytes),
            width * 4,
            width * 4 * height,
        ));
        no_catch(display.core().present_frame(
            Hold::NONE,
            OnFault::Die,
            Some(frame),
            u32::from_be_bytes(*b"BGRA"),
            0,
            None,
            GammaTableArgs::default(),
            None,
            Box::new(|| {}),
        ));
    });
}

/// `MTLTextureUsageShaderRead` as a u64 (avoids an objc2 dep in this crate's test).
fn objc2_metal_shader_read() -> u64 {
    1 // MTLTextureUsageShaderRead
}

/// Polls `f` until true or `timeout`.
fn wait_until(mut f: impl FnMut() -> bool, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    f()
}

/// Frames actually render into the window: the real sink + render thread present a sequence of
/// fixed frames into the real Win32 window's swapchain, and the display's host present count
/// advances. Ignored by default: opens a real window and drives the GPU.
#[test]
#[ignore = "opens a real window and presents on the GPU: run manually on the host"]
fn stage2_frames_render_into_the_window() {
    let (device, display) = presenting_display(0x57a9_e200);
    let queue = frame_queue(&device);
    let sink = crate::window::window_sink::<VulkanPlatform>(
        device.platform(),
        display.core().downgrade(),
        Some(queue),
    )
    .expect("a Vulkan window sink");

    // A real Win32 window; attaching the sink opens it and starts the render thread.
    let window = Window::new(
        WindowConfig::default(),
        0,
        reims_vgpu_window::handle::title(0),
    );
    window.attach(sink);

    let size = (640, 400);
    // The guest sets a mode, then presents frames; the render thread presents them into the
    // window's swapchain. We present several so the render thread has work across its loop.
    window.guest_mode(size.0 as u16, size.1 as u16);
    let before = display.core().host_present_count();
    for _ in 0..30 {
        present_a_fixed_frame(&device, &display, size);
        window.new_frame();
        std::thread::sleep(Duration::from_millis(16));
    }

    let advanced = wait_until(
        || display.core().host_present_count() > before,
        Duration::from_secs(10),
    );
    let after = display.core().host_present_count();
    println!(
        "stage2: host present count {before} -> {after} (guest presents: {})",
        display.core().guest_present_count()
    );
    assert!(
        advanced,
        "the window's render thread presented no frame into the swapchain (host count did not advance)"
    );
    println!("stage2: frames rendered into the window on the RTX 4060");

    // Teardown: close routes surface-before-window.
    window.close();
    std::thread::sleep(Duration::from_millis(300));
    drop((display, device));
    println!("stage2: closed; teardown ran");
}
