//! What the Rust API (`reims-vgpu`, feature `vulkan`) asks of this platform beyond core's
//! `Platform`: a frame target's pixels ([`read_pixels`]).

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToU64;

use crate::answer::{get, plain};
use crate::gpu::resource::VulkanTexture;

/// Copies the whole of `texture`'s first level and slice into `dst`: `height` rows of `width * 4`
/// bytes, each `bytes_per_row` after the last, the bytes between rows left as they are
/// (`getBytes:bytesPerRow:fromRegion:mipmapLevel:`). The caller has checked that `dst` holds
/// them. A refusal of the runtime's dies at the selector, as a raise in Metal's `getBytes` dies
/// at its send.
pub fn read_pixels(texture: &VulkanTexture, dst: &mut [u8], bytes_per_row: usize) {
    let sel = "getBytes:bytesPerRow:fromRegion:mipmapLevel:";
    let t = texture.copied(sel);
    let region = rt::Region::new_2d(0, 0, plain(|| t.width()), plain(|| t.height()));
    get(sel, || {
        t.get_bytes(dst, bytes_per_row.to_u64(), 0, region, 0, 0)
    });
}

/// Imports the guest's RAM into Vulkan once, at startup, before the guest runs (roadmap #2):
/// each `(host address, length)` range in `ranges` is a persistent range
/// ([`crate::os::lease_guest_memory`] registered it) and the runtime imports it in aligned chunks
/// (`REIMS_VGPU_RAM_IMPORT_CHUNK_MB`, 256 MiB by default), so that no-copy buffers over it bind at an
/// offset of a kept import instead of importing their pages. A chunk the driver refuses is left
/// to the buffers' own imports. One warning line says what was done. `REIMS_VGPU_RAM_IMPORT=0`
/// turns it off. Never fails.
pub fn prewarm_guest_ram(
    device: &crate::gpu::device::VulkanDevice,
    ranges: impl IntoIterator<Item = (u64, u64)>,
) {
    let mut total = rt::PrewarmReport::default();
    let mut count = 0u32;
    for (host, length) in ranges {
        let Some(start) = std::ptr::NonNull::new(host as *mut u8) else {
            continue;
        };
        let report = device.raw.prewarm_persistent_host_range(start, length);
        count += 1;
        total.chunks += report.chunks;
        total.failed += report.failed;
        total.bytes += report.bytes;
        total.elapsed += report.elapsed;
    }
    if total.chunks == 0 && total.failed == 0 {
        return;
    }
    log::warn!(
        "guest RAM import: {} chunks imported, {} refused, {} MiB over {count} ranges in {:.1} ms",
        total.chunks,
        total.failed,
        total.bytes >> 20,
        total.elapsed.as_secs_f64() * 1000.0
    );
    if total.failed != 0 {
        log::warn!(
            "guest RAM import: {} chunks were refused by the driver; buffers over them import their own pages",
            total.failed
        );
    }
}

#[cfg(test)]
mod tests {
    use metal2vulkan::metal::PixelFormat;
    use reims_vgpu_core::platform::Handle as _;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::TextureObject;
    use crate::platform::Obj;

    /// A BGRA8 texture of `width` by `height` (shared storage, the descriptor's default), holding
    /// `px` (rows of `width * 4` bytes).
    fn texture(width: u64, height: u64, px: &[u8]) -> VulkanTexture {
        let d = dev();
        let desc = rt::TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        let raw = d.raw.new_texture(&desc).unwrap();
        let all = rt::Region::new_2d(0, 0, width, height);
        raw.replace_region(all, 0, 0, px, width * 4, 0).unwrap();
        Obj::new(TextureObject {
            raw,
            device: d.clone_ref(),
            surface: None,
            plane: None,
            under: None,
            private_usage: 0,
            space: std::sync::Arc::new(crate::gpu::metal_space::Hold::NONE),
        })
    }

    /// Every pixel of the texture, row by row; the bytes past each row's pixels are left alone
    #[test]
    fn read_pixels_reads_the_whole_texture() {
        let (w, h) = (3u64, 2u64);
        let px: Vec<u8> = (0u8..24).collect();
        let t = texture(w, h, &px);
        let mut tight = vec![0xaa; 24];
        read_pixels(&t, &mut tight, 12);
        assert_eq!(tight, px);
        let mut padded = vec![0xaa; 16 + 12];
        read_pixels(&t, &mut padded, 16);
        assert_eq!(&padded[..12], &px[..12]);
        assert_eq!(&padded[12..16], &[0xaa; 4]);
        assert_eq!(&padded[16..], &px[12..]);
    }

    #[test]
    fn backdrop_texture_descriptor_and_replace_region_succeeds() {
        use crate::gpu::desc::{TextureDescriptorObject, TextureFields};
        use reims_vgpu_core::platform::HostRange;
        use reims_vgpu_core::platform::device::GpuDevice;
        use reims_vgpu_core::platform::resource::Texture;
        use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
        use reims_vgpu_core::platform::values::TextureDesc;
        use reims_vgpu_core::raise::OnFault;

        let desc = TextureDesc {
            pixel_format: 80, // MTL_PIXEL_FORMAT_BGRA8_UNORM
            width: 256,
            height: 256,
            ..TextureDesc::default()
        };
        let d = Obj::new(TextureDescriptorObject::new(TextureFields::of(&desc)));
        let mtl_device = dev();
        let tex = mtl_device
            .new_texture(OnFault::Return, &d)
            .expect("new_texture failed")
            .expect("new_texture returned None");

        let bgra_bytes =
            reims_vgpu_core::display::wallpaper::generate_recovery_wallpaper_bgra(256, 256);
        assert!(!bgra_bytes.is_empty());
        assert!(
            bgra_bytes.iter().any(|&b| b > 0),
            "wallpaper bytes are all zero!"
        );
        let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(256, 256, 1));
        let pixels = HostRange::of_slice(&bgra_bytes);
        let res = tex.replace_region(region, 0, 0, pixels, 256 * 4, 256 * 256 * 4);
        assert!(res.is_ok(), "replace_region failed: {:?}", res);

        let mut readback = vec![0u8; 256 * 256 * 4];
        read_pixels(&tex, &mut readback, 256 * 4);
        assert_eq!(&readback[..], &bgra_bytes[..]);
    }
}
