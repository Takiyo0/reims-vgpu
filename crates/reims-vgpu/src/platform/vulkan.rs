//! `VulkanPlatform` as an `ApiPlatform` (feature `vulkan`): the runtime's devices, on macOS over
//! MoltenVK and on Linux natively.

use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::os::GuestRamFile;

use super::*;

/// The platform over metal2vulkan's runtime (architecture 2.2): the runtime's devices, and on
/// macOS windows whose frames Metal copies from the runtime's (`window::window_sink`).
impl ApiPlatform for VulkanPlatform {
    fn process() -> Arc<VulkanPlatform> {
        crate::singleton::vulkan_platform()
    }

    /// The runtime's system default device (`MTLCreateSystemDefaultDevice()`), or the device
    /// whose registry ID, as the platform answers it, is `id` (`Platform::open_device`): the
    /// default device answers Metal's `registryID` of the same GPU, which is what a host
    /// names it by, and any other device the runtime's own ID.
    fn metal_device(&self, registry_id: Option<u64>) -> Option<<Self as Platform>::Device> {
        match registry_id {
            None => VulkanPlatform::system_default_device(),
            Some(id) => self.open_device(id),
        }
    }

    /// The whole texture, through the runtime's `get_bytes`
    /// (`reims_vgpu_vulkan::api::read_pixels`).
    fn read_pixels(
        &self,
        texture: &<Self as Platform>::Texture,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) {
        reims_vgpu_vulkan::api::read_pixels(texture, dst, bytes_per_row);
    }

    /// The guest's RAM, registered while the device lives
    /// (`reims_vgpu_vulkan::os::lease_guest_memory`).
    ///
    /// On Linux the platform makes its views of the guest's pages from the files the host named,
    /// so a range is leased with its backing fd (a range without a file is not registered, and a
    /// view of its pages is refused as a failed remap). On macOS nothing is registered, since Mach
    /// remaps addresses. On Windows (WHPX) the host already mapped every range at its
    /// `host_address` (`VirtualAlloc`), which the runtime imports directly through
    /// `VK_EXT_external_memory_host`; there is no backing object, so every range is leased by its
    /// host VA with `fd: None`, regardless of `memory_files` (which the Windows C API does not use).
    fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync> {
        #[cfg(windows)]
        let files = map.ranges().iter().map(|range| GuestRamFile {
            host: range.host_address.addr().to_u64(),
            length: range.physical_length,
            fd: None,
            offset: 0,
        });
        #[cfg(not(windows))]
        let files = map
            .ranges()
            .iter()
            .zip(map.files())
            .filter_map(|(range, file)| {
                let file = file.as_ref()?;
                Some(GuestRamFile {
                    host: range.host_address.addr().to_u64(),
                    length: range.physical_length,
                    fd: Some(Arc::clone(&file.fd)),
                    offset: file.offset,
                })
            });
        Box::new(reims_vgpu_vulkan::os::lease_guest_memory(files))
    }

    /// On Windows (WHPX) the guest's RAM is `VirtualAlloc` memory the runtime imports through
    /// `VK_EXT_external_memory_host`: import it once now, in large chunks, instead of a chunk at
    /// the first buffer that needs it (a stall of tens of ms each, on the FIFO thread).
    fn prewarm_guest_memory(
        &self,
        device: Option<&<Self as Platform>::Device>,
        map: Option<&MemoryMap>,
    ) {
        #[cfg(windows)]
        if let (Some(device), Some(map)) = (device, map) {
            reims_vgpu_vulkan::api::prewarm_guest_ram(
                device,
                map.ranges()
                    .iter()
                    .map(|range| (range.host_address.addr().to_u64(), range.physical_length)),
            );
        }
        #[cfg(not(windows))]
        let _ = (device, map);
    }
}
