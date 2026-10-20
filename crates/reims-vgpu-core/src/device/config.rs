//! [`DeviceConfig`], the host's device descriptor as the device reads it. The device makes its
//! checks on it, in its order, as it builds itself ([`crate::device::Device::new`]).

use std::sync::Arc;

use crate::host::HostBlocks;
use crate::memory::descriptor::MemoryMapDescriptor;
use crate::platform::Platform;

/// The most display ports a device takes.
pub const MAX_DISPLAY_PORT_COUNT: u32 = 8;
/// The descriptor's default `mmioLength`.
pub const DEFAULT_MMIO_LENGTH: u64 = 0x4000;

/// The host's device descriptor, as the device's creation reads it: the public APIs build it from
/// their device configs. [`Default`] is the SDK's defaults: `mmioLength` 0x4000, one display port,
/// feature level 0, argument buffers on; everything else nil, 0 or false.
pub struct DeviceConfig<P: Platform> {
    /// `device`: nil refuses the device (no default device is looked up).
    pub device: Option<P::Device>,
    /// `mmioLength`.
    ///
    /// Never read (`device::tests::mmio_length_is_ignored`).
    pub mmio_length: u64,
    /// `displayPortCount` (0 or more than 8 refuses).
    pub display_port_count: u32,
    /// The eight host blocks.
    pub blocks: HostBlocks,
    /// `memoryMapDescriptor`.
    ///
    /// Replacing this field drops the old value.
    pub memory_map_descriptor: Option<Arc<MemoryMapDescriptor>>,
    /// `usingIOSurfaceMapper`.
    pub using_iosurface_mapper: bool,
    /// `enableArgumentBuffers`.
    pub enable_argument_buffers: bool,
    /// `enableProcessIsolation`.
    pub enable_process_isolation: bool,
    /// Isolate nonzero tasks only while the IOSurface device has a mapped descriptor table.
    pub process_isolation_follows_guest: bool,
    /// `enableProtectedContent` (ignored).
    pub enable_protected_content: bool,
    /// `deviceFeatureLevel`.
    pub device_feature_level: u64,
    /// `displayPortConnectionTypes`.
    pub display_port_connection_types: u32,
}

impl<P: Platform> Default for DeviceConfig<P> {
    fn default() -> DeviceConfig<P> {
        DeviceConfig {
            device: None,
            mmio_length: DEFAULT_MMIO_LENGTH,
            display_port_count: 1,
            blocks: HostBlocks::default(),
            memory_map_descriptor: None,
            using_iosurface_mapper: false,
            enable_argument_buffers: true,
            enable_process_isolation: false,
            process_isolation_follows_guest: false,
            enable_protected_content: false,
            device_feature_level: 0,
            display_port_connection_types: 0,
        }
    }
}

impl<P: Platform> DeviceConfig<P> {
    /// The SDK's defaults with `device` set.
    pub fn new(device: Option<P::Device>) -> DeviceConfig<P> {
        DeviceConfig {
            device,
            ..DeviceConfig::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::FakePlatform;

    /// Every host block, from the fake host.
    fn all_blocks() -> HostBlocks {
        let p = FakePlatform::new();
        crate::testkit::host::FakeHost::new(&p, crate::testkit::GuestRam::new()).blocks()
    }

    #[test]
    fn defaults() {
        let c = DeviceConfig::<FakePlatform>::default();
        assert!(c.device.is_none());
        assert_eq!(c.mmio_length, 0x4000);
        assert_eq!(c.display_port_count, 1);
        assert_eq!(c.device_feature_level, 0);
        assert!(c.enable_argument_buffers);
        assert!(!c.enable_process_isolation && !c.enable_protected_content);
        assert!(!c.using_iosurface_mapper);
        assert_eq!(c.display_port_connection_types, 0);
        assert!(c.memory_map_descriptor.is_none());
        assert!(c.blocks.raise_interrupt.is_none() && c.blocks.add_trace_range.is_none());
    }

    /// The last drop releases the device, the blocks and the map descriptor.
    #[test]
    fn drop_releases() {
        let p = FakePlatform::new();
        let mut c = DeviceConfig::<FakePlatform>::new(Some(p.device()));
        let map = Arc::new(MemoryMapDescriptor::new());
        c.memory_map_descriptor = Some(map.clone());
        c.blocks = all_blocks();
        let device_refs = p.device().retain_count();
        assert_eq!(Arc::strong_count(&map), 2);
        drop(c);
        assert_eq!(p.device().retain_count(), device_refs - 1);
        assert_eq!(Arc::strong_count(&map), 1);
    }
}
