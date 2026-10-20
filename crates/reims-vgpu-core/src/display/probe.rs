//! Bounded, opt-in source and target captures, copied on the presenting command buffer.
//! The completion handler reads only dedicated staging buffers, never a live guest texture.

use super::DeathKind;
use crate::guest::GuestPtr;
use crate::platform::Platform;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::command::CommandBuffer;
use crate::platform::device::GpuDevice;
use crate::platform::os::surface::Surface as _;
use crate::platform::resource::{Buffer, Texture};
use crate::platform::types::{CommandBufferStatus, MtlOrigin, MtlSize};
use crate::raise::{OnFault, RaiseResult};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) fn encode<P: Platform>(
    _platform: &P,
    source: &P::Texture,
    target: &P::Texture,
    command: &P::CommandBuffer,
    gamma: bool,
) -> RaiseResult<()> {
    let Some(dir) = std::env::var_os("REIMS_VGPU_CAPTURE_DIR") else {
        return Ok(());
    };
    if source.pixel_format() != 80 || target.pixel_format() != 80 {
        return Ok(());
    }
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n % 120 != 0 || n >= 2400 {
        return Ok(());
    }
    let Some(device) = command.device() else {
        return Ok(());
    };
    let Some(blit) = command.blit_encoder(OnFault::Die)? else {
        return Ok(());
    };
    let mut captures = Vec::new();
    for (label, texture) in [("source", source), ("target", target)] {
        let (w, h) = (texture.width(), texture.height());
        let Some(length) = w.checked_mul(h).and_then(|v| v.checked_mul(4)) else {
            continue;
        };
        if length > 128 * 1024 * 1024 {
            continue;
        }
        let Some(buffer) = device.new_buffer_of_length(OnFault::Die, length, 0)? else {
            continue;
        };
        blit.encode(
            OnFault::Die,
            BlitCmd::CopyTextureToBuffer {
                source_texture: texture,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(w, h, 1),
                destination_buffer: &buffer,
                destination_offset: 0,
                destination_bytes_per_row: w * 4,
                destination_bytes_per_image: length,
            },
        )?;
        captures.push((label, w, h, length, buffer));
    }
    // Compare the IOSurface's CPU-visible backing with the GPU texture readback. If they differ,
    // the first divergence is in the surface mapping/coherency path; if they match, the guest
    // already composed those pixels into the surface.
    let backing = source.iosurface().and_then(|surface| {
        let (w, h, row_bytes) = (surface.width(), surface.height(), surface.bytes_per_row());
        let Some(row_len) = w.checked_mul(4) else {
            return None;
        };
        let Some(length) = row_bytes.checked_mul(h) else {
            return None;
        };
        if surface.base_address().is_null()
            || row_bytes < row_len
            || length > 128 * 1024 * 1024
            || length > surface.alloc_size()
        {
            return None;
        }
        Some((surface, w, h, row_bytes, length))
    });
    blit.encode(OnFault::Die, BlitCmd::EndEncoding)?;
    let texture_id = source.unique_identifier();
    command.add_completed_handler(OnFault::Die, Box::new(move |info| {
        if info.status == CommandBufferStatus::ERROR { return }
        let dir = std::path::PathBuf::from(dir);
        if std::fs::create_dir_all(&dir).is_err() { return }
        for (label, w, h, length, buffer) in captures {
            let Ok(length) = usize::try_from(length) else { continue };
            let mut bytes = vec![0u8; length];
            GuestPtr::new(buffer.contents()).copy_to(0, &mut bytes, DeathKind::NullContents);
            for px in bytes.chunks_exact_mut(4) { px.swap(0, 2); }
            let header = format!("P7\n# source_texture={texture_id} gamma={gamma} capture={n}\nWIDTH {w}\nHEIGHT {h}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n");
            let mut pam = header.into_bytes();
            pam.extend_from_slice(&bytes);
            let path = dir.join(format!("{}-{n:04}-{label}.pam", std::process::id()));
            if let Err(error) = std::fs::write(&path, pam) {
                log::warn!("display capture {} failed: {error}", path.display());
            } else {
                log::info!("display capture {} texture={texture_id} gamma={gamma}", path.display());
            }
        }
        if let Some((surface, w, h, row_bytes, length)) = backing {
            let (Ok(w), Ok(h), Ok(row_bytes), Ok(length)) = (
                usize::try_from(w), usize::try_from(h), usize::try_from(row_bytes), usize::try_from(length),
            ) else { return };
            let mut rows = vec![0u8; length];
            GuestPtr::new(surface.base_address()).copy_to(0, &mut rows, DeathKind::NullContents);
            let Some(packed_len) = w.checked_mul(h).and_then(|v| v.checked_mul(4)) else { return };
            let mut bytes = vec![0u8; packed_len];
            for y in 0..h {
                let src = y * row_bytes;
                let dst = y * w * 4;
                bytes[dst..dst + w * 4].copy_from_slice(&rows[src..src + w * 4]);
            }
            for px in bytes.chunks_exact_mut(4) { px.swap(0, 2); }
            let header = format!("P7\n# source_texture={texture_id} gamma={gamma} capture={n} backing=iosurface\nWIDTH {w}\nHEIGHT {h}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n");
            let mut pam = header.into_bytes();
            pam.extend_from_slice(&bytes);
            let path = dir.join(format!("{}-{n:04}-source-backing.pam", std::process::id()));
            if let Err(error) = std::fs::write(&path, pam) {
                log::warn!("display capture {} failed: {error}", path.display());
            } else {
                log::info!("display capture {} texture={texture_id} gamma={gamma}", path.display());
            }
        }
    }))
}
