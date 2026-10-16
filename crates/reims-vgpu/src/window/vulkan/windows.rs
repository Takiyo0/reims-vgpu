//! What Windows's window asks of the Vulkan platform's sink ([`FrameSink`]): a presenter over the
//! window's swapchain ([`SwapchainSink::make_presenter`]), which the Win32 window makes its
//! surface for (`VK_KHR_win32_surface`). Identical to Linux's: both draw into a Vulkan swapchain
//! over the window's surface; only how the surface is created differs, and that lives in the
//! window crate's `VulkanTarget` for each OS.

#[cfg(test)]
mod tests;

use reims_vgpu_window::{FrameSink, Presenter, VulkanTarget};

use super::swapchain::SwapchainSink;

impl FrameSink for SwapchainSink {
    /// The presenter ([`SwapchainSink::make_presenter`]).
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn Presenter>> {
        self.make_presenter(target)
    }
}
