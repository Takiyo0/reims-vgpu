//! The blit decoder (encoder type 2): each record of a segment is one command on the blit
//! encoder the decoder makes from the command buffer.
//!
//! # Contract
//!
//! - **The encoder** is asked of the command buffer when the decoder is made, and never checked:
//!   with no command buffer, or a nil encoder, every send is a no-op while the lookups and checks
//!   still run (`tests::init::a_nil_encoder_is_kept_and_every_send_is_a_no_op`). The decoder
//!   keeps the encoder, not the command buffer
//!   (`tests::init::makes_one_encoder_and_keeps_no_command_buffer`).
//! - **The deserializer is held weakly**, so the task's release stays the deserializer's last
//!   while the caller still holds the decoder
//!   (`tests::init::keeps_the_deserializer_unretained`).
//! - **Per record**: the record's whole size is read, then its objects are looked up in record
//!   order, then its checks run, then the one send. Buffer-to-buffer copies (301) check nothing;
//!   the two buffer fills (306, 319) check the range against the buffer's `length`, read first;
//!   the texture fill from bytes (320) checks the format's block bytes, then shared storage, then
//!   the range; the ICB commands (305, 312, 313) look up through the required getter, so ref 0 is
//!   refused before any range check (`tests::decode_fill_texture_with_bytes`,
//!   `tests::decode_copy_indirect_command_buffer`).
//! - **Synchronize (314)** takes the nullable resource lookup: nil is sent as nil
//!   (`tests::decode_synchronize_resource`).
//! - **The end of a segment** ends the encoder unless the segment is continued by the next; the
//!   ended encoder stays held until the decoder drops
//!   (`tests::decode::an_empty_segment_ends_the_encoder_unless_continued`).
//! - **A raise** out of a record faults the decoder (ends the encoder and drops it) and goes on;
//!   a raise from that `endEncoding` goes on in its place and leaves the encoder held
//!   (`tests::fault`).

use std::fmt;
use std::sync::{Arc, Weak};

use crate::deserializer::records::{
    CmdCopyFromBufferToBuffer, CmdCopyFromBufferToTexture, CmdCopyFromTextureToBuffer,
    CmdCopyFromTextureToTexture, CmdCopyFromTextureToTextureWithNumSliceLevel,
    CmdCopyFromTextureToTextureWithOptions, CmdCopyIndirectCommandBuffer, CmdFenceOp,
    CmdFillBuffer, CmdFillBufferWithPattern, CmdFillTextureWithBytes, CmdFillTextureWithColor,
    CmdGenerateMipmaps, CmdInvalidateCompressedTexture, CmdInvalidateCompressedTextureImage,
    CmdOptimize, CmdOptimizeImage, CmdOptimizeResetIndirectBuffer, CmdSynchronize,
    CmdSynchronizeImage, SegmentHeader,
};
use crate::deserializer::walk::next_record;
use crate::deserializer::{Deserializer, RaiseKind, upgrade};
use crate::memory::cursor::Cursor;
use crate::platform::blit::{BlitCmd, BlitEncoder, ResourceRef};
use crate::platform::command::CommandBuffer;
use crate::platform::device::GpuDevice;
use crate::platform::icb::IndirectCommandBuffer;
use crate::platform::resource::{Buffer, GpuResource, Texture};
use crate::platform::types::{ClearColor, MtlOrigin, MtlRegion, MtlSize, NsRange, PixelFormatInfo};
use crate::platform::{HostRange, Platform};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::resource::AnyResource;
use crate::rt::Hold;

/// `MTLStorageModeShared`.
const MTL_STORAGE_MODE_SHARED: u64 = 0;

/// The blit decoder's own raises: its range and fill checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlitRaise {
    /// A buffer fill's range is past the buffer's length.
    BufferRange,
    /// An ICB command's range is past the ICB's size.
    IcbRange,
    /// An ICB copy's destination index is not below the destination's size.
    IcbDestinationIndex,
    /// A texture fill's byte count is below its format's block bytes.
    FillBytesLength,
    /// A texture fill's source buffer is not shared.
    FillBytesStorageMode,
    /// A texture fill's bytes run past its source buffer.
    FillBytesRange,
}

impl BlitRaise {
    /// The dotted id.
    pub fn name(self) -> &'static str {
        match self {
            BlitRaise::BufferRange => "deserializer.blit.buffer_range",
            BlitRaise::IcbRange => "deserializer.blit.icb_range",
            BlitRaise::IcbDestinationIndex => "deserializer.blit.icb_destination_index",
            BlitRaise::FillBytesLength => "deserializer.blit.fill_bytes_length",
            BlitRaise::FillBytesStorageMode => "deserializer.blit.fill_bytes_storage_mode",
            BlitRaise::FillBytesRange => "deserializer.blit.fill_bytes_range",
        }
    }
}

impl fmt::Display for BlitRaise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Refuses `location..location + length` unless it lies within `size`, a wrapping sum
/// included. The caller asks `size` before the check, so it is asked even when the sum wraps.
fn check_range(location: u64, length: u64, size: u64, kind: BlitRaise) -> RaiseResult<()> {
    let (end, wrapped) = location.overflowing_add(length);
    if wrapped || end > size {
        let what = match kind {
            BlitRaise::IcbRange => "indirect command buffer",
            _ => "buffer",
        };
        crate::raise!(
            RaiseKind::Blit(kind),
            format_args!("range {location:#x}+{length:#x} runs past the {what}'s size {size:#x}")
        );
    }
    Ok(())
}

/// The decoder of one blit segment, and of the segments it is carried into.
///
/// Its drop releases the encoder.
pub struct BlitDecoder<P: Platform> {
    /// The deserializer, weakly (the walk's contract).
    deserializer: Weak<Deserializer<P>>,
    /// The command buffer's blit encoder, as it answered: `None` sends nothing. Taken by a
    /// fault.
    encoder: Option<P::BlitEncoder>,
}

impl<P: Platform> BlitDecoder<P> {
    /// A decoder with the blit encoder of `command_buffer` (which it does not keep); with no
    /// command buffer, no encoder. A raise from `blitCommandEncoder` goes on.
    pub fn new(
        deserializer: Weak<Deserializer<P>>,
        command_buffer: Option<&P::CommandBuffer>,
    ) -> RaiseResult<Self> {
        let t_encoder = std::time::Instant::now();
        let encoder = match command_buffer {
            Some(buffer) => buffer.blit_encoder(OnFault::Return)?,
            None => None,
        };
        crate::perfstats::DECODE_ENCODER.add(t_encoder.elapsed());
        Ok(BlitDecoder {
            deserializer,
            encoder,
        })
    }

    /// The deserializer, for one record.
    fn deserializer(&self) -> Arc<Deserializer<P>> {
        upgrade(&self.deserializer)
    }

    /// Sends `cmd` to the encoder; nothing without one.
    fn send(&self, cmd: BlitCmd<'_, P>) -> RaiseResult<()> {
        match &self.encoder {
            Some(encoder) => Ok(encoder.encode(OnFault::Return, cmd)?),
            None => Ok(()),
        }
    }

    /// Decodes one segment. A raise out of a record faults the decoder and goes on; a raise from
    /// the fault goes on in its place, and the record's ends there. `continues_previous` is not
    /// read.
    pub fn decode(
        &mut self,
        hold: Hold<'_>,
        header: &SegmentHeader,
        mut segment: Cursor,
    ) -> RaiseResult<()> {
        let mut decoded = self.decode_records(hold, header, &mut segment);
        if decoded.is_err() {
            hold.cleanup_step(&mut decoded, self.fault())?;
        }
        decoded
    }

    /// Ends the encoder and drops it; a second fault has none to end. A raise from
    /// `endEncoding` goes on and leaves the encoder held.
    pub fn fault(&mut self) -> RaiseResult<()> {
        self.send(BlitCmd::EndEncoding)?;
        self.encoder = None;
        Ok(())
    }

    fn decode_records(
        &mut self,
        hold: Hold<'_>,
        header: &SegmentHeader,
        segment: &mut Cursor,
    ) -> RaiseResult<()> {
        while let Some((record_header, mut record)) = next_record(segment)? {
            // Each command reads its record's whole size; a longer record is fine, a shorter one
            // fails to read.
            let c = &mut record;
            #[cfg(feature = "record-timing")]
            let t_record = std::time::Instant::now();
            match record_header.command_id {
                300 => self.decode_copy_from_buffer_to_texture(hold, c)?,
                301 => self.decode_copy_from_buffer_to_buffer(hold, c)?,
                302 => self.decode_copy_from_texture_to_buffer(hold, c)?,
                303 => self.decode_copy_from_texture_to_texture(hold, c)?,
                304 => self.decode_copy_from_texture_to_texture_with_options(hold, c)?,
                305 => self.decode_copy_indirect_command_buffer(hold, c)?,
                306 => self.decode_fill_buffer(hold, c)?,
                307 => self.decode_generate_mipmaps(hold, c)?,
                308 => self.decode_optimize_for_cpu_access(hold, c)?,
                309 => self.decode_optimize_for_gpu_access(hold, c)?,
                310 => self.decode_optimize_image_for_cpu_access(hold, c)?,
                311 => self.decode_optimize_image_for_gpu_access(hold, c)?,
                312 => self.decode_optimize_indirect_command_buffer(hold, c)?,
                313 => self.decode_reset_commands_in_command_buffer(hold, c)?,
                314 => self.decode_synchronize_resource(hold, c)?,
                315 => self.decode_synchronize_texture_image(hold, c)?,
                316 => self.decode_update_fence(hold, c)?,
                317 => self.decode_wait_for_fence(hold, c)?,
                318 => self.decode_copy_from_texture_to_texture_with_num_slice_level(hold, c)?,
                319 => self.decode_fill_buffer_with_pattern(hold, c)?,
                320 => self.decode_fill_texture_with_bytes(hold, c)?,
                321 => self.decode_fill_texture_with_color(hold, c)?,
                322 => self.decode_invalidate_compressed_texture(hold, c)?,
                323 => self.decode_invalidate_compressed_texture_image(hold, c)?,
                id => crate::raise!(
                    RaiseKind::UnrecognizedCommand,
                    format_args!("unknown command {id}")
                ),
            }
            #[cfg(feature = "record-timing")]
            crate::perfstats::record(record_header.command_id, t_record.elapsed());
        }
        if header.continued_by_next == 0 {
            self.send(BlitCmd::EndEncoding)?;
        }
        Ok(())
    }

    fn decode_copy_from_buffer_to_texture(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdCopyFromBufferToTexture = cursor.read_or_raise(OnFault::Return)?;
        let buffer = self
            .deserializer()
            .require_buffer(hold, command.source_buffer_reference)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.destination_texture_reference)?;
        self.send(BlitCmd::CopyBufferToTextureOptions {
            source_buffer: Some(&buffer),
            source_offset: command.source_offset,
            source_bytes_per_row: command.source_bytes_per_row,
            source_bytes_per_image: command.source_bytes_per_image,
            source_size: MtlSize::from(command.source_size),
            destination_texture: Some(&texture),
            destination_slice: u64::from(command.destination_slice),
            destination_level: u64::from(command.destination_level),
            destination_origin: MtlOrigin::from(command.destination_origin),
            options: u64::from(command.options),
        })
    }

    fn decode_copy_from_buffer_to_buffer(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdCopyFromBufferToBuffer = cursor.read_or_raise(OnFault::Return)?;
        let source = self
            .deserializer()
            .require_buffer(hold, command.source_buffer_reference)?;
        let destination = self
            .deserializer()
            .require_buffer(hold, command.destination_buffer_reference)?;
        // No range check at all; Metal validates.
        self.send(BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&source),
            source_offset: command.source_offset,
            destination_buffer: Some(&destination),
            destination_offset: command.destination_offset,
            size: command.size,
        })
    }

    fn decode_copy_from_texture_to_buffer(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdCopyFromTextureToBuffer = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.source_texture_reference)?;
        let buffer = self
            .deserializer()
            .require_buffer(hold, command.destination_buffer_reference)?;
        self.send(BlitCmd::CopyTextureToBufferOptions {
            source_texture: Some(&texture),
            source_slice: u64::from(command.source_slice),
            source_level: u64::from(command.source_level),
            source_origin: MtlOrigin::from(command.source_origin),
            source_size: MtlSize::from(command.source_size),
            destination_buffer: Some(&buffer),
            destination_offset: command.destination_offset,
            destination_bytes_per_row: command.destination_bytes_per_row,
            destination_bytes_per_image: command.destination_bytes_per_image,
            options: u64::from(command.options),
        })
    }

    fn decode_copy_from_texture_to_texture(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdCopyFromTextureToTexture = cursor.read_or_raise(OnFault::Return)?;
        let source = self
            .deserializer()
            .require_texture(hold, command.source_texture_reference)?;
        let destination = self
            .deserializer()
            .require_texture(hold, command.destination_texture_reference)?;
        self.send(BlitCmd::CopyTextureRegion {
            source_texture: &source,
            source_slice: u64::from(command.source_slice),
            source_level: u64::from(command.source_level),
            source_origin: MtlOrigin::from(command.source_origin),
            source_size: MtlSize::from(command.source_size),
            destination_texture: &destination,
            destination_slice: u64::from(command.destination_slice),
            destination_level: u64::from(command.destination_level),
            destination_origin: MtlOrigin::from(command.destination_origin),
        })
    }

    fn decode_copy_from_texture_to_texture_with_num_slice_level(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdCopyFromTextureToTextureWithNumSliceLevel =
            cursor.read_or_raise(OnFault::Return)?;
        let source = self
            .deserializer()
            .require_texture(hold, command.source_texture_reference)?;
        let destination = self
            .deserializer()
            .require_texture(hold, command.destination_texture_reference)?;
        self.send(BlitCmd::CopyTextureSlices {
            source_texture: &source,
            source_slice: u64::from(command.source_slice),
            source_level: u64::from(command.source_level),
            destination_texture: &destination,
            destination_slice: u64::from(command.destination_slice),
            destination_level: u64::from(command.destination_level),
            slice_count: u64::from(command.slice_count),
            level_count: u64::from(command.level_count),
        })
    }

    fn decode_copy_from_texture_to_texture_with_options(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdCopyFromTextureToTextureWithOptions =
            cursor.read_or_raise(OnFault::Return)?;
        let source = self
            .deserializer()
            .require_texture(hold, command.source_texture_reference)?;
        let destination = self
            .deserializer()
            .require_texture(hold, command.destination_texture_reference)?;
        self.send(BlitCmd::CopyTextureRegionOptions {
            source_texture: &source,
            source_slice: u64::from(command.source_slice),
            source_level: u64::from(command.source_level),
            source_origin: MtlOrigin::from(command.source_origin),
            source_size: MtlSize::from(command.source_size),
            destination_texture: &destination,
            destination_slice: u64::from(command.destination_slice),
            destination_level: u64::from(command.destination_level),
            destination_origin: MtlOrigin::from(command.destination_origin),
            options: u64::from(command.options),
        })
    }

    fn decode_fill_buffer(&self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdFillBuffer = cursor.read_or_raise(OnFault::Return)?;
        let buffer = self
            .deserializer()
            .require_buffer(hold, command.buffer_reference)?;
        let range = NsRange::new(command.range_location, command.range_length);
        // Checked against `length` only, which is read first.
        let length = buffer.length();
        check_range(range.location, range.length, length, BlitRaise::BufferRange)?;
        self.send(BlitCmd::FillBuffer {
            buffer: &buffer,
            range,
            value: command.value,
        })
    }

    fn decode_generate_mipmaps(&self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdGenerateMipmaps = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        self.send(BlitCmd::GenerateMipmaps {
            texture: Some(&texture),
        })
    }

    fn decode_optimize_for_cpu_access(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdOptimize = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        self.send(BlitCmd::OptimizeForCpu { texture: &texture })
    }

    fn decode_optimize_for_gpu_access(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdOptimize = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        self.send(BlitCmd::OptimizeForGpu { texture: &texture })
    }

    fn decode_optimize_image_for_cpu_access(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdOptimizeImage = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        self.send(BlitCmd::OptimizeForCpuSlice {
            texture: &texture,
            slice: u64::from(command.slice),
            level: u64::from(command.level),
        })
    }

    fn decode_optimize_image_for_gpu_access(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdOptimizeImage = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        self.send(BlitCmd::OptimizeForGpuSlice {
            texture: &texture,
            slice: u64::from(command.slice),
            level: u64::from(command.level),
        })
    }

    fn decode_synchronize_resource(&self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSynchronize = cursor.read_or_raise(OnFault::Return)?;
        // The nullable lookup; nil is sent as nil.
        let resource = self
            .deserializer()
            .object_delegate()
            .get_resource(hold, command.resource_reference)?;
        let resource = match &resource {
            None => None,
            Some(AnyResource::Buffer(b)) => Some(ResourceRef::Buffer(b)),
            Some(AnyResource::Texture(t)) => Some(ResourceRef::Texture(t)),
            Some(AnyResource::Icb(i)) => Some(ResourceRef::Icb(i)),
            // Never answered: a heap resource raises in the lookup instead.
            Some(AnyResource::Heap(_)) => None,
        };
        self.send(BlitCmd::SynchronizeResource { resource })
    }

    fn decode_synchronize_texture_image(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSynchronizeImage = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        self.send(BlitCmd::SynchronizeTexture {
            texture: &texture,
            slice: u64::from(command.slice),
            level: u64::from(command.level),
        })
    }

    fn decode_update_fence(&self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdFenceOp = cursor.read_or_raise(OnFault::Return)?;
        let fence = self
            .deserializer()
            .require_fence(hold, command.fence_reference)?;
        self.send(BlitCmd::UpdateFence { fence: &fence })
    }

    fn decode_wait_for_fence(&self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdFenceOp = cursor.read_or_raise(OnFault::Return)?;
        let fence = self
            .deserializer()
            .require_fence(hold, command.fence_reference)?;
        self.send(BlitCmd::WaitForFence { fence: &fence })
    }

    fn decode_fill_buffer_with_pattern(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdFillBufferWithPattern = cursor.read_or_raise(OnFault::Return)?;
        let buffer = self
            .deserializer()
            .require_buffer(hold, command.buffer_reference)?;
        let range = NsRange::new(command.range_location, command.range_length);
        // Checked against `length` only, which is read first.
        let length = buffer.length();
        check_range(range.location, range.length, length, BlitRaise::BufferRange)?;
        self.send(BlitCmd::FillBufferPattern4 {
            buffer: &buffer,
            range,
            pattern: command.pattern,
        })
    }

    fn decode_fill_texture_with_bytes(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdFillTextureWithBytes = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        let buffer = self
            .deserializer()
            .require_buffer(hold, command.buffer_reference)?;
        let offset = command.buffer_offset;
        let buffer_length = buffer.length();
        let length = command.length;
        let device = texture.device();
        let pixel_format = texture.pixel_format();
        let info = match &device {
            Some(device) => device.pixel_format_info(pixel_format),
            // A Metal texture always has a device; the zeroed info otherwise.
            None => PixelFormatInfo::default(),
        };
        // Block bytes, then shared storage, then the range.
        if length < info.block_bytes {
            crate::raise!(
                RaiseKind::Blit(BlitRaise::FillBytesLength),
                format_args!(
                    "fill of {length} bytes is shorter than a block of format {} ({} bytes)",
                    info.name.as_deref().unwrap_or("unknown"),
                    info.block_bytes
                )
            );
        }
        if buffer.storage_mode() != MTL_STORAGE_MODE_SHARED {
            crate::raise!(
                RaiseKind::Blit(BlitRaise::FillBytesStorageMode),
                "fill of a buffer that is not in shared storage"
            );
        }
        let (end, wrapped) = offset.overflowing_add(length);
        if wrapped || end > buffer_length {
            crate::raise!(
                RaiseKind::Blit(BlitRaise::FillBytesRange),
                format_args!(
                    "fill of {length:#x} bytes at offset {offset:#x} runs past the buffer's \
                     {buffer_length:#x} bytes"
                )
            );
        }
        let bytes = buffer.contents().wrapping_add(offset);
        // The record's size comes before its origin.
        let region = MtlRegion::new(
            MtlOrigin::from(command.region_origin),
            MtlSize::from(command.region_size),
        );
        #[allow(unsafe_code)]
        // SAFETY: the buffer is shared and `offset + length <= length()` without wrapping, so
        // `bytes..bytes + length` lies in its contents; `buffer` is held until the send returns.
        let bytes = unsafe { HostRange::new(bytes, length) };
        self.send(BlitCmd::FillTexture {
            texture: &texture,
            level: u64::from(command.level),
            slice: u64::from(command.slice),
            region,
            bytes,
        })
    }

    fn decode_fill_texture_with_color(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdFillTextureWithColor = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        let region = MtlRegion::new(
            MtlOrigin::from(command.region_origin),
            MtlSize::from(command.region_size),
        );
        self.send(BlitCmd::FillTextureColor {
            texture: &texture,
            level: u64::from(command.level),
            slice: u64::from(command.slice),
            region,
            color: ClearColor::from(command.color),
            pixel_format: u64::from(command.pixel_format),
        })
    }

    fn decode_invalidate_compressed_texture(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdInvalidateCompressedTexture = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        self.send(BlitCmd::InvalidateCompressedTexture { texture: &texture })
    }

    fn decode_invalidate_compressed_texture_image(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdInvalidateCompressedTextureImage = cursor.read_or_raise(OnFault::Return)?;
        let texture = self
            .deserializer()
            .require_texture(hold, command.texture_reference)?;
        self.send(BlitCmd::InvalidateCompressedTextureSlice {
            texture: &texture,
            slice: u64::from(command.slice),
            level: u64::from(command.level),
        })
    }

    fn decode_copy_indirect_command_buffer(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdCopyIndirectCommandBuffer = cursor.read_or_raise(OnFault::Return)?;
        // The required lookups, so ref 0 is refused before any range check.
        let source = self.deserializer().require_indirect_command_buffer(
            hold,
            command.source_indirect_command_buffer_reference,
        )?;
        let destination = self.deserializer().require_indirect_command_buffer(
            hold,
            command.destination_indirect_command_buffer_reference,
        )?;
        let range = NsRange::new(command.source_range_location, command.source_range_length);
        let size = source.size();
        check_range(range.location, range.length, size, BlitRaise::IcbRange)?;
        let index = command.destination_index;
        if index >= destination.size() {
            crate::raise!(
                RaiseKind::Blit(BlitRaise::IcbDestinationIndex),
                format_args!("destination index {index} is past the indirect command buffer's end")
            );
        }
        self.send(BlitCmd::CopyIndirectCommandBuffer {
            source: &source,
            source_range: range,
            destination: &destination,
            destination_index: index,
        })
    }

    fn decode_reset_commands_in_command_buffer(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdOptimizeResetIndirectBuffer = cursor.read_or_raise(OnFault::Return)?;
        // The required lookup, before the range check.
        let icb = self
            .deserializer()
            .require_indirect_command_buffer(hold, command.indirect_command_buffer_reference)?;
        let range = NsRange::new(command.range_location, command.range_length);
        let size = icb.size();
        check_range(range.location, range.length, size, BlitRaise::IcbRange)?;
        self.send(BlitCmd::ResetCommandsInBuffer {
            buffer: &icb,
            range,
        })
    }

    fn decode_optimize_indirect_command_buffer(
        &self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdOptimizeResetIndirectBuffer = cursor.read_or_raise(OnFault::Return)?;
        // The required lookup, before the range check.
        let icb = self
            .deserializer()
            .require_indirect_command_buffer(hold, command.indirect_command_buffer_reference)?;
        let range = NsRange::new(command.range_location, command.range_length);
        let size = icb.size();
        check_range(range.location, range.length, size, BlitRaise::IcbRange)?;
        self.send(BlitCmd::OptimizeIndirectCommandBuffer {
            indirect_command_buffer: &icb,
            range,
        })
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use zerocopy::{Immutable, IntoBytes};

    use crate::deserializer::Deserializer;
    use crate::deserializer::RaiseKind;
    use crate::deserializer::records::{CmdHeader, SegmentHeader};
    use crate::deserializer::test_lookups::Lookups;
    use crate::host::HostPtr;
    use crate::memory::cursor::{Cursor, CursorRange};
    use crate::platform::ObjId;
    use crate::raise::{Origin, Raise, RaiseId};
    use crate::testkit::impls::{Backing, new_buffer, new_state};
    use crate::testkit::objects::{Model, TextureModel};
    use crate::testkit::{Call, FakePlatform, Fk, kind};

    /// A fake process with a deserializer over a recording delegate.
    pub(crate) struct Env {
        pub(crate) p: Arc<FakePlatform>,
        pub(crate) r: Arc<Lookups>,
        pub(crate) d: Arc<Deserializer<FakePlatform>>,
    }

    pub(crate) fn env() -> Env {
        let p = Arc::new(FakePlatform::new());
        let r = Arc::new(Lookups::new(&p));
        let d = Deserializer::new(p.clone(), Some(p.device()), r.lookup());
        p.call_log().clear();
        Env { p, r, d }
    }

    impl Env {
        /// A Shared (options 0) or Private (0x20) buffer of `len` bytes under `reference`.
        pub(crate) fn buffer(&self, reference: u32, len: u64, options: u64) -> Fk<kind::Buffer> {
            let b = new_buffer(self.p.world(), len, options, Backing::Fresh, None).unwrap();
            self.r.provide(reference, &b);
            b
        }

        /// A texture of the fake device under `reference`.
        pub(crate) fn texture(&self, reference: u32) -> Fk<kind::Texture> {
            let t: Fk<kind::Texture> = self
                .p
                .world()
                .new_obj(Model::Texture(TextureModel::default()));
            t.state().device = Some(self.p.device().erase());
            self.r.provide(reference, &t);
            t
        }

        /// A state object (ICB, pipeline, sampler, rate map, …) under `reference`.
        pub(crate) fn state<K: crate::testkit::objects::Kind>(&self, reference: u32) -> Fk<K> {
            let o: Fk<K> = new_state(self.p.world(), None);
            self.r.provide(reference, &o);
            o
        }

        /// An ICB of `size` commands under `reference`.
        pub(crate) fn icb(&self, reference: u32, size: u64) -> Fk<kind::Icb> {
            let i: Fk<kind::Icb> = self.state(reference);
            i.set_size(size);
            i
        }

        /// The calls on `obj`, as `(sel, args)`.
        pub(crate) fn on(&self, obj: ObjId) -> Vec<Call> {
            self.p.call_log().calls_on(obj)
        }

        /// The delegate calls, as `(sel, first argument)`.
        pub(crate) fn lookups(&self) -> Vec<(&'static str, u32)> {
            self.p
                .call_log()
                .calls_of("ManagerLookup")
                .into_iter()
                .map(|c| {
                    let r = match c.args.first() {
                        Some(crate::testkit::Arg::U32(v)) => *v,
                        _ => u32::MAX,
                    };
                    (c.sel, r)
                })
                .collect()
        }

        /// The ordered calls' selectors in the log, in order: lookups and the Metal calls whose
        /// position is observable.
        pub(crate) fn sels(&self) -> Vec<&'static str> {
            self.p.call_log().sels()
        }
    }

    /// Guest bytes a cursor reads: kept alive (and never grown) while the cursor is used.
    pub(crate) struct Stream(pub(crate) Vec<u8>);

    impl Stream {
        pub(crate) fn new() -> Stream {
            Stream(Vec::new())
        }

        /// A record: its header (length counts the header) and its bytes.
        pub(crate) fn rec<T: IntoBytes + Immutable>(self, id: u32, cmd: &T) -> Stream {
            let bytes = cmd.as_bytes().to_vec();
            self.raw(id, 8 + bytes.len() as u32, &bytes)
        }

        /// A record with an explicit header length.
        pub(crate) fn raw(mut self, id: u32, length: u32, payload: &[u8]) -> Stream {
            self.0.extend_from_slice(
                CmdHeader {
                    command_id: id,
                    length,
                }
                .as_bytes(),
            );
            self.0.extend_from_slice(payload);
            self
        }

        /// Raw trailing bytes.
        pub(crate) fn bytes(mut self, b: &[u8]) -> Stream {
            self.0.extend_from_slice(b);
            self
        }

        /// A cursor over the whole stream.
        pub(crate) fn cursor(&mut self) -> Cursor {
            let length = self.0.len() as u64;
            Cursor::with_ranges(vec![CursorRange {
                address: HostPtr::from_ptr(self.0.as_mut_ptr()),
                length,
            }])
        }
    }

    /// A segment header with `continuedByNext`.
    pub(crate) fn segment(encoder_type: u8, continued_by_next: u8) -> SegmentHeader {
        SegmentHeader {
            length: 0,
            encoder_type,
            continues_previous: 0,
            continued_by_next,
            unknown07: 0,
        }
    }

    /// Asserts `r` is the deserializer's `kind`, with `detail`.
    #[track_caller]
    pub(crate) fn assert_raised(r: &Raise, kind: RaiseKind, detail: &str) {
        assert_eq!(r.id(), RaiseId::from(kind), "{r}");
        assert_eq!(r.detail(), Some(detail), "{r}");
    }

    /// The send a platform raise came from; `None` for core's own raise.
    pub(crate) fn platform_site(r: &Raise) -> Option<&'static str> {
        match r.origin() {
            Origin::Platform(sel) => Some(sel),
            Origin::Boundary(site) => Some(site),
            Origin::Core(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::deserializer::ObjectKind;
    use crate::deserializer::records::{ClearColorRecord, MtlOriginRecord, MtlSizeRecord};
    use crate::platform::Handle;
    use crate::raise::death::DeathKind;
    use crate::raise::death::DeathSite;
    use crate::testkit::objects::{CommandBufferModel, Model};
    use crate::testkit::{Arg, FakePlatform, Step, kind};
    use zerocopy::IntoBytes;

    type Dec = BlitDecoder<FakePlatform>;

    fn decoder(e: &Env) -> Dec {
        let cb = e
            .p
            .world()
            .new_obj::<kind::CommandBuffer>(Model::CommandBuffer(CommandBufferModel::default()));
        let d = BlitDecoder::new(Arc::downgrade(&e.d), Some(&cb)).unwrap();
        e.p.call_log().clear();
        d
    }

    fn encoder_id(d: &Dec) -> crate::platform::ObjId {
        d.encoder.as_ref().unwrap().id()
    }

    /// Decodes `s` as one segment under a handler (the walk's).
    fn run(d: &mut Dec, s: &mut Stream, continued_by_next: u8) -> RaiseResult<()> {
        let c = s.cursor();
        d.decode(Hold::NONE, &segment(2, continued_by_next), c)
    }

    /// The encoder's selectors, `endEncoding` excluded.
    fn encoded(e: &Env, d_id: crate::platform::ObjId) -> Vec<(&'static str, Vec<Arg>)> {
        e.on(d_id)
            .into_iter()
            .filter(|c| c.sel != "endEncoding")
            .map(|c| (c.sel, c.args))
            .collect()
    }

    #[test]
    fn its_encoder_type_is_two() {
        let e = env();
        let d = crate::deserializer::decoder::Decoder::Blit(decoder(&e));
        assert_eq!(d.encoder_type(), 2);
    }

    mod init {
        use super::*;

        #[test]
        fn makes_one_encoder_and_keeps_no_command_buffer() {
            let e = env();
            let cb =
                e.p.world()
                    .new_obj::<kind::CommandBuffer>(Model::CommandBuffer(
                        CommandBufferModel::default(),
                    ));
            let cb_id = cb.id();
            let probe = cb.clone_ref();
            let d = BlitDecoder::new(Arc::downgrade(&e.d), Some(&cb)).unwrap();
            drop(cb);
            assert_eq!(e.sels(), vec!["blitCommandEncoder"]);
            assert_eq!(e.on(cb_id).len(), 1);
            // The command buffer is not kept: only the probe is left.
            assert_eq!(probe.retain_count(), 1);
            let enc = d.encoder.as_ref().unwrap().clone_ref();
            assert_eq!(enc.retain_count(), 2);
            drop(d);
            // The drop: the one release of the encoder, and nothing else sent.
            assert_eq!(enc.retain_count(), 1);
            assert_eq!(e.sels(), vec!["blitCommandEncoder"]);
        }

        #[test]
        fn keeps_the_deserializer_unretained() {
            // The deserializer is held weakly (the drop releases the encoder only). The task's
            // release after the decode is the deserializer's last while the decoder still waits
            // in the pool: the deserializer's drop runs there, not at the decoder's.
            let Env { p, r, d } = env();
            let dev = p.device();
            let device_refs = dev.retain_count();
            let cb = p
                .world()
                .new_obj::<kind::CommandBuffer>(
                    Model::CommandBuffer(CommandBufferModel::default()),
                );
            let mut dec = BlitDecoder::new(Arc::downgrade(&d), Some(&cb)).unwrap();
            let mut s = Stream::new();
            dec.decode(Hold::NONE, &segment(2, 1), s.cursor()).unwrap();
            assert_eq!(Arc::strong_count(&d), 1);
            let enc = dec.encoder.as_ref().unwrap().clone_ref();
            let gone = Arc::downgrade(&d);
            drop(d);
            assert!(gone.upgrade().is_none());
            assert_eq!(dev.retain_count(), device_refs - 1);
            assert_eq!(Arc::strong_count(&r), 1);
            assert!(p.world().live("MTLSharedEventListener").is_empty());
            assert_eq!(enc.retain_count(), 2);
            drop(dec);
            assert_eq!(enc.retain_count(), 1);
            assert_eq!(dev.retain_count(), device_refs - 1);
        }

        #[test]
        fn a_nil_encoder_is_kept_and_every_send_is_a_no_op() {
            let e = env();
            e.p.script().push("blitCommandEncoder", Step::Nil);
            let cb =
                e.p.world()
                    .new_obj::<kind::CommandBuffer>(Model::CommandBuffer(
                        CommandBufferModel::default(),
                    ));
            let mut d = BlitDecoder::new(Arc::downgrade(&e.d), Some(&cb)).unwrap();
            assert!(d.encoder.is_none());
            let t = e.texture(3);
            let mut s = Stream::new().rec(
                307,
                &CmdGenerateMipmaps {
                    texture_reference: 3,
                },
            );
            run(&mut d, &mut s, 0).unwrap();
            // The lookup still happens; no encoder send, no endEncoding.
            assert_eq!(e.lookups(), vec![("get_texture", 3)]);
            assert!(e.on(t.id()).is_empty());
            assert_eq!(e.p.call_log().calls_of("MTLBlitCommandEncoder").len(), 0);
            d.fault().unwrap();
            assert!(d.encoder.is_none());
        }

        #[test]
        fn a_nil_command_buffer_gives_a_nil_encoder() {
            // `[nil blitCommandEncoder]` is nil.
            let e = env();
            let d = BlitDecoder::new(Arc::downgrade(&e.d), None).unwrap();
            assert!(d.encoder.is_none());
            assert!(e.sels().is_empty());
        }

        #[test]
        fn a_raise_from_blit_command_encoder_propagates() {
            let e = env();
            e.p.script().push(
                "blitCommandEncoder",
                Step::raise("NSInternalInconsistencyException", "x"),
            );
            let cb =
                e.p.world()
                    .new_obj::<kind::CommandBuffer>(Model::CommandBuffer(
                        CommandBufferModel::default(),
                    ));
            let r = BlitDecoder::new(Arc::downgrade(&e.d), Some(&cb));
            let t = r.err().unwrap();
            assert_eq!(platform_site(&t), Some("blitCommandEncoder"));
        }
    }

    mod decode {
        use super::*;

        #[test]
        fn an_empty_segment_ends_the_encoder_unless_continued() {
            let e = env();
            let mut d = decoder(&e);
            let id = encoder_id(&d);
            run(&mut d, &mut Stream::new(), 0).unwrap();
            assert_eq!(e.sels(), vec!["endEncoding"]);
            e.p.call_log().clear();
            run(&mut d, &mut Stream::new(), 1).unwrap();
            assert!(e.on(id).is_empty());
            assert!(d.encoder.is_some());
        }

        #[test]
        fn trailing_bytes_under_a_header_are_ignored() {
            let e = env();
            let mut d = decoder(&e);
            let mut s = Stream::new().bytes(&[1, 2, 3, 4, 5, 6, 7]);
            run(&mut d, &mut s, 0).unwrap();
            assert_eq!(e.sels(), vec!["endEncoding"]);
        }

        #[test]
        fn records_run_in_stream_order() {
            let e = env();
            let mut d = decoder(&e);
            let t = e.texture(1);
            let mut s = Stream::new()
                .rec(
                    308,
                    &CmdOptimize {
                        texture_reference: 1,
                    },
                )
                .rec(
                    309,
                    &CmdOptimize {
                        texture_reference: 1,
                    },
                )
                .rec(
                    322,
                    &CmdInvalidateCompressedTexture {
                        texture_reference: 1,
                    },
                );
            run(&mut d, &mut s, 0).unwrap();
            let id = encoder_id_of(&e);
            assert_eq!(
                e.on(id).iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec![
                    "optimizeContentsForCPUAccess:",
                    "optimizeContentsForGPUAccess:",
                    "invalidateCompressedTexture:",
                    "endEncoding"
                ]
            );
            drop(t);
        }

        fn encoder_id_of(e: &Env) -> crate::platform::ObjId {
            e.p.call_log().calls_of("MTLBlitCommandEncoder")[0].obj
        }

        #[test]
        fn a_longer_record_is_fine_and_its_tail_skipped() {
            // Records longer than their struct are fine (sub-cursor), shorter fail
            let e = env();
            let mut d = decoder(&e);
            let _t = e.texture(1);
            let mut payload = CmdOptimize {
                texture_reference: 1,
            }
            .as_bytes()
            .to_vec();
            payload.extend_from_slice(&[0xff; 4]);
            let mut s = Stream::new().raw(308, 16, &payload).rec(
                309,
                &CmdOptimize {
                    texture_reference: 1,
                },
            );
            run(&mut d, &mut s, 0).unwrap();
            assert_eq!(e.lookups(), vec![("get_texture", 1), ("get_texture", 1)]);
        }

        #[test]
        fn a_short_record_fails_to_read() {
            // 301's 0x20 bytes, one short.
            let e = env();
            let mut d = decoder(&e);
            let mut s = Stream::new().raw(301, 8 + 0x1f, &[0; 0x1f]);
            let t = run(&mut d, &mut s, 0).unwrap_err();
            assert_eq!(
                t.detail(),
                Some(
                    "record not readable: read of 0x20 bytes at offset 0x0 runs past the cursor's 0x1f bytes"
                )
            );
            assert!(e.lookups().is_empty());
        }

        #[test]
        fn a_record_shorter_than_its_header_underflows_and_faults() {
            // A record length under 8 is an underflow in the main loops
            let e = env();
            let mut d = decoder(&e);
            let id = encoder_id(&d);
            let mut s = Stream::new().raw(308, 7, &[0; 8]);
            let t = run(&mut d, &mut s, 1).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::RecordUnderflow,
                "record length 7 is shorter than its header",
            );
            // fault: endEncoding, release, nil (even for a continued segment).
            assert_eq!(
                e.on(id).iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec!["endEncoding"]
            );
            assert!(d.encoder.is_none());
        }

        #[test]
        fn a_record_overrunning_the_segment_fails_before_dispatch() {
            // A record overrunning the segment fails `subCursor` before dispatch
            let e = env();
            let mut d = decoder(&e);
            let mut s = Stream::new().raw(308, 12, &[0; 2]);
            let t = run(&mut d, &mut s, 0).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::RecordOverrun,
                "record not readable: a sub-cursor of 0x4 bytes, with 0x2 left",
            );
            assert!(e.lookups().is_empty());
            assert!(d.encoder.is_none());
        }

        #[test]
        fn an_unknown_command_is_unrecognized() {
            let e = env();
            let mut d = decoder(&e);
            for id in [299u32, 324, 0, 450] {
                let mut d2 = decoder(&e);
                let mut s = Stream::new().raw(id, 8, &[]);
                let t = run(&mut d2, &mut s, 0).unwrap_err();
                assert_raised(
                    &t,
                    RaiseKind::UnrecognizedCommand,
                    &format!("unknown command {id}"),
                );
            }
            d.fault().unwrap();
        }

        /// Under the walk's handler the rethrow goes on as the raise thrown, after the fault
        #[test]
        fn the_rethrow_goes_on_as_the_raise_thrown() {
            let e = env();
            let mut d = decoder(&e);
            let mut s = Stream::new().raw(308, 4, &[]);
            let c = s.cursor();
            let r = d.decode(Hold::NONE, &segment(2, 0), c);
            assert_eq!(
                r.err().map(|r| r.site()),
                Some(DeathSite {
                    kind: DeathKind::UnhandledRaise,
                    file: RaiseKind::RecordUnderflow.name(),
                    line: 0
                })
            );
            // The fault ran first.
            assert_eq!(e.sels(), vec!["endEncoding"]);
        }

        #[test]
        fn a_raise_from_the_final_end_encoding_faults_and_rethrows() {
            let e = env();
            let mut d = decoder(&e);
            let id = encoder_id(&d);
            e.p.script().push(
                "endEncoding",
                Step::raise("NSInternalInconsistencyException", "e"),
            );
            let t = run(&mut d, &mut Stream::new(), 0).unwrap_err();
            assert_eq!(platform_site(&t), Some("endEncoding"));
            // endEncoding raised, then fault's endEncoding, release, nil.
            assert_eq!(
                e.on(id).iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec!["endEncoding", "endEncoding"]
            );
            assert!(d.encoder.is_none());
        }

        #[test]
        fn a_raise_inside_fault_replaces_the_exception_and_keeps_the_encoder() {
            let e = env();
            let mut d = decoder(&e);
            e.p.script().push(
                "endEncoding",
                Step::raise("NSInternalInconsistencyException", "f"),
            );
            let mut s = Stream::new().raw(308, 4, &[]);
            let t = run(&mut d, &mut s, 0).unwrap_err();
            assert_eq!(platform_site(&t), Some("endEncoding"));
            assert_eq!(t.detail(), Some("f"));
            assert!(d.encoder.is_some());
        }

        #[test]
        fn a_raise_from_an_encoder_send_faults_and_rethrows() {
            let e = env();
            let mut d = decoder(&e);
            let _t = e.texture(1);
            e.p.script().push(
                "invalidateCompressedTexture:",
                Step::unrecognized("invalidateCompressedTexture:"),
            );
            let mut s = Stream::new().rec(
                322,
                &CmdInvalidateCompressedTexture {
                    texture_reference: 1,
                },
            );
            let t = run(&mut d, &mut s, 1).unwrap_err();
            assert_eq!(platform_site(&t), Some("invalidateCompressedTexture:"));
            assert_eq!(e.sels().last(), Some(&"endEncoding"));
            assert!(d.encoder.is_none());
        }
    }

    mod fault {
        use super::*;

        #[test]
        fn ends_releases_and_nils_once() {
            let e = env();
            let mut d = decoder(&e);
            let enc = d.encoder.as_ref().unwrap().clone_ref();
            d.fault().unwrap();
            assert_eq!(
                e.on(enc.id()).iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec!["endEncoding"]
            );
            assert!(d.encoder.is_none());
            assert_eq!(enc.retain_count(), 1);
            // A second fault messages nil.
            d.fault().unwrap();
            assert_eq!(e.on(enc.id()).len(), 1);
        }

        #[test]
        fn a_raise_from_end_encoding_keeps_the_encoder() {
            let e = env();
            let mut d = decoder(&e);
            e.p.script().push(
                "endEncoding",
                Step::raise("NSInternalInconsistencyException", "f"),
            );
            d.fault().unwrap_err();
            assert!(d.encoder.is_some());
        }

        /// The raise from `endEncoding` is what leaves the fault, not the raise in flight.
        #[test]
        fn fault_surfaces_the_end_encoding_raise() {
            let e = env();
            let mut d = decoder(&e);
            e.p.script().push(
                "endEncoding",
                Step::raise("NSInternalInconsistencyException", "end"),
            );
            let t = d.fault().unwrap_err();
            assert_eq!(platform_site(&t), Some("endEncoding"));
            assert_eq!(t.detail(), Some("end"));
            assert!(d.encoder.is_some());
            // With no raise it ends, releases and nils.
            d.fault().unwrap();
            assert!(d.encoder.is_none());
        }
    }

    mod range_check {
        use super::*;

        #[test]
        fn bounds_and_overflow() {
            let buffer = BlitRaise::BufferRange;
            check_range(0, 16, 16, buffer).unwrap();
            check_range(16, 0, 16, buffer).unwrap();
            let t = check_range(1, 16, 16, buffer).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::Blit(buffer),
                "range 0x1+0x10 runs past the buffer's size 0x10",
            );
            let icb = BlitRaise::IcbRange;
            let t = check_range(u64::MAX, 1, u64::MAX, icb).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::Blit(icb),
                "range 0xffffffffffffffff+0x1 runs past the indirect command buffer's size \
                 0xffffffffffffffff",
            );
        }
    }

    mod required_icb_lookup {}

    #[test]
    fn decode_copy_from_buffer_to_texture() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let b = e.buffer(1, 64, 0);
        let t = e.texture(2);
        let cmd = CmdCopyFromBufferToTexture {
            source_buffer_reference: 1,
            destination_texture_reference: 2,
            source_offset: 3,
            source_bytes_per_row: 4,
            source_bytes_per_image: 5,
            source_size: MtlSizeRecord {
                width: 6,
                height: 7,
                depth: 8,
            },
            destination_origin: MtlOriginRecord { x: 9, y: 10, z: 11 },
            destination_slice: 0xfffe,
            destination_level: 13,
            options: 0xffff_fff0,
        };
        run(&mut d, &mut Stream::new().rec(300, &cmd), 0).unwrap();
        assert_eq!(e.lookups(), vec![("get_buffer", 1), ("get_texture", 2)]);
        assert_eq!(
            encoded(&e, id),
            vec![(
                "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:",
                vec![
                    Arg::obj(&b),
                    Arg::U64(3),
                    Arg::U64(4),
                    Arg::U64(5),
                    Arg::val(&MtlSize::new(6, 7, 8)),
                    Arg::obj(&t),
                    Arg::U64(0xfffe),
                    Arg::U64(13),
                    Arg::val(&MtlOrigin::new(9, 10, 11)),
                    Arg::U64(0xffff_fff0),
                ]
            )]
        );
    }

    #[test]
    fn decode_copy_from_buffer_to_buffer() {
        // No range check, even far past both buffers.
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let s = e.buffer(1, 16, 0);
        let t = e.buffer(2, 16, 0);
        let cmd = CmdCopyFromBufferToBuffer {
            source_buffer_reference: 1,
            destination_buffer_reference: 2,
            source_offset: u64::MAX,
            destination_offset: 1 << 40,
            size: u64::MAX,
        };
        run(&mut d, &mut Stream::new().rec(301, &cmd), 0).unwrap();
        assert_eq!(e.lookups(), vec![("get_buffer", 1), ("get_buffer", 2)]);
        assert_eq!(
            encoded(&e, id),
            vec![(
                "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
                vec![
                    Arg::Obj(Some(s.id())),
                    Arg::U64(u64::MAX),
                    Arg::Obj(Some(t.id())),
                    Arg::U64(1 << 40),
                    Arg::U64(u64::MAX)
                ]
            )]
        );
        // The source lookup failing stops before the destination's.
        let e = env();
        let mut d = decoder(&e);
        let _t = e.buffer(2, 16, 0);
        run(&mut d, &mut Stream::new().rec(301, &cmd), 0).unwrap_err();
        assert_eq!(e.lookups(), vec![("get_buffer", 1)]);
    }

    #[test]
    fn decode_copy_from_texture_to_buffer() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let t = e.texture(1);
        let b = e.buffer(2, 16, 0);
        let cmd = CmdCopyFromTextureToBuffer {
            source_texture_reference: 1,
            destination_buffer_reference: 2,
            source_origin: MtlOriginRecord { x: 1, y: 2, z: 3 },
            source_size: MtlSizeRecord {
                width: 4,
                height: 5,
                depth: 6,
            },
            destination_offset: 7,
            destination_bytes_per_row: 8,
            destination_bytes_per_image: 9,
            source_slice: 10,
            source_level: 11,
            options: 0x8001,
            unknown56: [0xaa; 2],
        };
        run(&mut d, &mut Stream::new().rec(302, &cmd), 0).unwrap();
        assert_eq!(e.lookups(), vec![("get_texture", 1), ("get_buffer", 2)]);
        assert_eq!(
            encoded(&e, id),
            vec![(
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:",
                vec![
                    Arg::obj(&t),
                    Arg::U64(10),
                    Arg::U64(11),
                    Arg::val(&MtlOrigin::new(1, 2, 3)),
                    Arg::val(&MtlSize::new(4, 5, 6)),
                    Arg::obj(&b),
                    Arg::U64(7),
                    Arg::U64(8),
                    Arg::U64(9),
                    Arg::U64(0x8001),
                ]
            )]
        );
    }

    fn t2t() -> CmdCopyFromTextureToTexture {
        CmdCopyFromTextureToTexture {
            source_texture_reference: 1,
            destination_texture_reference: 2,
            source_origin: MtlOriginRecord { x: 1, y: 2, z: 3 },
            source_size: MtlSizeRecord {
                width: 4,
                height: 5,
                depth: 6,
            },
            destination_origin: MtlOriginRecord { x: 7, y: 8, z: 9 },
            source_slice: 10,
            source_level: 11,
            destination_slice: 12,
            destination_level: 13,
        }
    }

    #[test]
    fn decode_copy_from_texture_to_texture() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let s = e.texture(1);
        let t = e.texture(2);
        run(&mut d, &mut Stream::new().rec(303, &t2t()), 0).unwrap();
        assert_eq!(e.lookups(), vec![("get_texture", 1), ("get_texture", 2)]);
        assert_eq!(
            encoded(&e, id),
            vec![(
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
                vec![
                    Arg::obj(&s),
                    Arg::U64(10),
                    Arg::U64(11),
                    Arg::val(&MtlOrigin::new(1, 2, 3)),
                    Arg::val(&MtlSize::new(4, 5, 6)),
                    Arg::obj(&t),
                    Arg::U64(12),
                    Arg::U64(13),
                    Arg::val(&MtlOrigin::new(7, 8, 9)),
                ]
            )]
        );
    }

    #[test]
    fn decode_copy_from_texture_to_texture_with_options() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let s = e.texture(1);
        let t = e.texture(2);
        let b = t2t();
        let cmd = CmdCopyFromTextureToTextureWithOptions {
            source_texture_reference: 1,
            destination_texture_reference: 2,
            source_origin: b.source_origin,
            source_size: b.source_size,
            destination_origin: b.destination_origin,
            source_slice: 10,
            source_level: 11,
            destination_slice: 12,
            destination_level: 13,
            options: 0x8000_0001,
        };
        run(&mut d, &mut Stream::new().rec(304, &cmd), 0).unwrap();
        assert_eq!(e.lookups(), vec![("get_texture", 1), ("get_texture", 2)]);
        assert_eq!(
            encoded(&e, id),
            vec![(
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:",
                vec![
                    Arg::obj(&s),
                    Arg::U64(10),
                    Arg::U64(11),
                    Arg::val(&MtlOrigin::new(1, 2, 3)),
                    Arg::val(&MtlSize::new(4, 5, 6)),
                    Arg::obj(&t),
                    Arg::U64(12),
                    Arg::U64(13),
                    Arg::val(&MtlOrigin::new(7, 8, 9)),
                    Arg::U64(0x8000_0001),
                ]
            )]
        );
    }

    #[test]
    fn decode_copy_from_texture_to_texture_with_num_slice_level() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let s = e.texture(1);
        let t = e.texture(2);
        let cmd = CmdCopyFromTextureToTextureWithNumSliceLevel {
            source_texture_reference: 1,
            destination_texture_reference: 2,
            source_slice: 3,
            source_level: 4,
            destination_slice: 5,
            destination_level: 6,
            slice_count: 7,
            level_count: 0xffff,
        };
        run(&mut d, &mut Stream::new().rec(318, &cmd), 0).unwrap();
        assert_eq!(e.lookups(), vec![("get_texture", 1), ("get_texture", 2)]);
        assert_eq!(
            encoded(&e, id),
            vec![(
                "copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:",
                vec![
                    Arg::obj(&s),
                    Arg::U64(3),
                    Arg::U64(4),
                    Arg::obj(&t),
                    Arg::U64(5),
                    Arg::U64(6),
                    Arg::U64(7),
                    Arg::U64(0xffff),
                ]
            )]
        );
    }

    #[test]
    fn decode_fill_buffer() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let b = e.buffer(1, 16, 0x20);
        let fill = |loc: u64, len: u64| CmdFillBuffer {
            buffer_reference: 1,
            range_location: loc,
            range_length: len,
            value: 0x5a,
            unknown15: [0; 3],
        };
        // Only the length bounds it (a Private buffer is fine).
        run(&mut d, &mut Stream::new().rec(306, &fill(4, 12)), 1).unwrap();
        assert_eq!(
            encoded(&e, id),
            vec![(
                "fillBuffer:range:value:",
                vec![Arg::obj(&b), Arg::val(&NsRange::new(4, 12)), Arg::U8(0x5a)]
            )]
        );
        for (loc, len) in [(4, 13), (u64::MAX, 2)] {
            e.p.call_log().clear();
            let t = run(&mut d, &mut Stream::new().rec(306, &fill(loc, len)), 1).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::Blit(BlitRaise::BufferRange),
                &format!("range {loc:#x}+{len:#x} runs past the buffer's size 0x10"),
            );
            // Nothing was encoded.
            assert!(encoded(&e, id).is_empty());
            d = decoder(&e);
        }
    }

    #[test]
    fn decode_fill_buffer_with_pattern() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let b = e.buffer(1, 16, 0);
        let fill = |loc: u64, len: u64| CmdFillBufferWithPattern {
            buffer_reference: 1,
            range_location: loc,
            range_length: len,
            pattern: 0xdead_beef,
        };
        run(&mut d, &mut Stream::new().rec(319, &fill(0, 16)), 1).unwrap();
        assert_eq!(
            encoded(&e, id),
            vec![(
                "fillBuffer:range:pattern4:",
                vec![
                    Arg::obj(&b),
                    Arg::val(&NsRange::new(0, 16)),
                    Arg::U32(0xdead_beef)
                ]
            )]
        );
        let t = run(&mut d, &mut Stream::new().rec(319, &fill(1, 16)), 1).unwrap_err();
        assert_raised(
            &t,
            RaiseKind::Blit(BlitRaise::BufferRange),
            "range 0x1+0x10 runs past the buffer's size 0x10",
        );
    }

    #[test]
    fn decode_texture_image_commands() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let t = e.texture(9);
        let img = CmdOptimizeImage {
            texture_reference: 9,
            slice: 0xfffe,
            level: 3,
        };
        let sync = CmdSynchronizeImage {
            texture_reference: 9,
            slice: 1,
            level: 2,
        };
        let inv = CmdInvalidateCompressedTextureImage {
            texture_reference: 9,
            slice: 4,
            level: 5,
        };
        let mut s = Stream::new()
            .rec(310, &img)
            .rec(311, &img)
            .rec(315, &sync)
            .rec(323, &inv);
        run(&mut d, &mut s, 1).unwrap();
        assert_eq!(e.lookups(), vec![("get_texture", 9); 4]);
        let a = |s: u64, l: u64| vec![Arg::obj(&t), Arg::U64(s), Arg::U64(l)];
        assert_eq!(
            encoded(&e, id),
            vec![
                ("optimizeContentsForCPUAccess:slice:level:", a(0xfffe, 3)),
                ("optimizeContentsForGPUAccess:slice:level:", a(0xfffe, 3)),
                ("synchronizeTexture:slice:level:", a(1, 2)),
                ("invalidateCompressedTexture:slice:level:", a(4, 5)),
            ]
        );
    }

    fn fill_bytes(offset: u64, length: u64) -> CmdFillTextureWithBytes {
        CmdFillTextureWithBytes {
            texture_reference: 1,
            level: 2,
            slice: 3,
            region_size: MtlSizeRecord {
                width: 4,
                height: 5,
                depth: 6,
            },
            region_origin: MtlOriginRecord { x: 7, y: 8, z: 9 },
            buffer_reference: 10,
            buffer_offset: offset,
            length,
        }
    }

    mod decode_fill_texture_with_bytes {
        use super::*;
        use crate::platform::types::PixelFormatInfo;

        #[test]
        fn passes_the_contents_at_the_offset() {
            let e = env();
            let mut d = decoder(&e);
            let id = encoder_id(&d);
            let t = e.texture(1);
            let b = e.buffer(10, 64, 0);
            run(&mut d, &mut Stream::new().rec(320, &fill_bytes(16, 48)), 1).unwrap();
            assert_eq!(e.lookups(), vec![("get_texture", 1), ("get_buffer", 10)]);
            let sels = e.sels();
            let contents = b.contents();
            assert_eq!(
                encoded(&e, id),
                vec![(
                    "fillTexture:level:slice:region:bytes:length:",
                    vec![
                        Arg::obj(&t),
                        Arg::U64(2),
                        Arg::U64(3),
                        Arg::val(&MtlRegion::new(
                            MtlOrigin::new(7, 8, 9),
                            MtlSize::new(4, 5, 6)
                        )),
                        Arg::Ptr(contents.wrapping_add(16).addr()),
                        Arg::U64(48),
                    ]
                )]
            );
            // The pixel format's info, asked once, before the fill.
            let at = |sel: &str| sels.iter().position(|s| *s == sel);
            assert_eq!(
                sels.iter()
                    .filter(|s| **s == "MTLPixelFormatGetInfoForDevice()")
                    .count(),
                1
            );
            assert!(
                at("MTLPixelFormatGetInfoForDevice()")
                    < at("fillTexture:level:slice:region:bytes:length:")
            );
        }

        #[test]
        fn checks_block_bytes_then_storage_then_range() {
            // Every check fails here; blockBytes wins.
            let e = env();
            let dev = e.p.device();
            e.p.script().always_on(
                dev.id(),
                "MTLPixelFormatGetInfoForDevice()",
                Step::value(PixelFormatInfo {
                    name: Some("MTLPixelFormatBGRA8Unorm".into()),
                    block_bytes: 8,
                    ..PixelFormatInfo::default()
                }),
            );
            let _t = e.texture(1);
            let _b = e.buffer(10, 4, 0x20);
            let mut d = decoder(&e);
            let t = run(
                &mut d,
                &mut Stream::new().rec(320, &fill_bytes(u64::MAX, 7)),
                1,
            )
            .unwrap_err();
            assert_raised(
                &t,
                RaiseKind::Blit(BlitRaise::FillBytesLength),
                "fill of 7 bytes is shorter than a block of format MTLPixelFormatBGRA8Unorm (8 bytes)",
            );
            // Storage next.
            let mut d = decoder(&e);
            let t = run(
                &mut d,
                &mut Stream::new().rec(320, &fill_bytes(u64::MAX, 8)),
                1,
            )
            .unwrap_err();
            assert_raised(
                &t,
                RaiseKind::Blit(BlitRaise::FillBytesStorageMode),
                "fill of a buffer that is not in shared storage",
            );
        }

        #[test]
        fn a_null_format_name_prints_unknown() {
            let e = env();
            e.p.script().always_on(
                e.p.device().id(),
                "MTLPixelFormatGetInfoForDevice()",
                Step::value(PixelFormatInfo {
                    name: None,
                    block_bytes: 16,
                    ..PixelFormatInfo::default()
                }),
            );
            let _t = e.texture(1);
            let _b = e.buffer(10, 64, 0);
            let mut d = decoder(&e);
            let t = run(&mut d, &mut Stream::new().rec(320, &fill_bytes(0, 0)), 1).unwrap_err();
            assert_eq!(
                t.detail(),
                Some("fill of 0 bytes is shorter than a block of format unknown (16 bytes)")
            );
        }

        #[test]
        fn the_range_is_checked_without_overflow() {
            let e = env();
            let _t = e.texture(1);
            let _b = e.buffer(10, 64, 0);
            for (offset, length) in [(1, 64), (u64::MAX, 4)] {
                let mut d = decoder(&e);
                let t = run(
                    &mut d,
                    &mut Stream::new().rec(320, &fill_bytes(offset, length)),
                    1,
                )
                .unwrap_err();
                assert_raised(
                    &t,
                    RaiseKind::Blit(BlitRaise::FillBytesRange),
                    &format!(
                        "fill of {length:#x} bytes at offset {offset:#x} runs past the buffer's \
                         0x40 bytes"
                    ),
                );
            }
            let mut d = decoder(&e);
            run(&mut d, &mut Stream::new().rec(320, &fill_bytes(60, 4)), 1).unwrap();
        }

        #[test]
        fn a_texture_without_a_device_reads_a_zeroed_format() {
            let e = env();
            let t = e.texture(1);
            t.state().device = None;
            let _b = e.buffer(10, 64, 0);
            let mut d = decoder(&e);
            // blockBytes 0: any length passes that check.
            run(&mut d, &mut Stream::new().rec(320, &fill_bytes(0, 0)), 1).unwrap();
            assert!(!e.sels().contains(&"MTLPixelFormatGetInfoForDevice()"));
        }
    }

    #[test]
    fn decode_fill_texture_with_color() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let t = e.texture(1);
        let cmd = CmdFillTextureWithColor {
            texture_reference: 1,
            level: 2,
            slice: 3,
            region_size: MtlSizeRecord {
                width: 4,
                height: 5,
                depth: 6,
            },
            region_origin: MtlOriginRecord { x: 7, y: 8, z: 9 },
            color: ClearColorRecord {
                red: 0.5,
                green: -0.0,
                blue: f64::MAX,
                alpha: 1.0,
            },
            pixel_format: 0xfffe,
            unknown5a: [0; 2],
        };
        run(&mut d, &mut Stream::new().rec(321, &cmd), 1).unwrap();
        assert_eq!(e.lookups(), vec![("get_texture", 1)]);
        assert_eq!(
            encoded(&e, id),
            vec![(
                "fillTexture:level:slice:region:color:pixelFormat:",
                vec![
                    Arg::obj(&t),
                    Arg::U64(2),
                    Arg::U64(3),
                    Arg::val(&MtlRegion::new(
                        MtlOrigin::new(7, 8, 9),
                        MtlSize::new(4, 5, 6)
                    )),
                    Arg::val(&ClearColor {
                        red: 0.5,
                        green: -0.0,
                        blue: f64::MAX,
                        alpha: 1.0
                    }),
                    Arg::U64(0xfffe),
                ]
            )]
        );
    }

    mod decode_copy_indirect_command_buffer {
        use super::*;

        fn cmd(src: u32, loc: u64, len: u64, index: u64) -> CmdCopyIndirectCommandBuffer {
            CmdCopyIndirectCommandBuffer {
                source_indirect_command_buffer_reference: src,
                destination_indirect_command_buffer_reference: 2,
                source_range_location: loc,
                source_range_length: len,
                destination_index: index,
            }
        }

        #[test]
        fn copies_within_both_bounds() {
            let e = env();
            let mut d = decoder(&e);
            let id = encoder_id(&d);
            let s = e.icb(1, 8);
            let t = e.icb(2, 4);
            run(&mut d, &mut Stream::new().rec(305, &cmd(1, 2, 6, 3)), 1).unwrap();
            assert_eq!(
                e.lookups(),
                vec![
                    ("get_indirect_command_buffer", 1),
                    ("get_indirect_command_buffer", 2)
                ]
            );
            assert_eq!(
                encoded(&e, id),
                vec![(
                    "copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:",
                    vec![
                        Arg::obj(&s),
                        Arg::val(&NsRange::new(2, 6)),
                        Arg::obj(&t),
                        Arg::U64(3)
                    ]
                )]
            );
        }

        #[test]
        fn ref_zero_is_refused_before_any_range_check() {
            // ICB ref 0 refused by the decoder's own non-null getter before any range check
            let e = env();
            let mut d = decoder(&e);
            let _t = e.icb(2, 4);
            let t = run(&mut d, &mut Stream::new().rec(305, &cmd(0, 0, 0, 0)), 1).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::MissingObject(ObjectKind::Icb),
                "no indirect command buffer for reference 0",
            );
            assert_eq!(e.lookups(), vec![("get_indirect_command_buffer", 0)]);
        }

        #[test]
        fn source_range_then_destination_index() {
            let e = env();
            let _s = e.icb(1, 8);
            let _t = e.icb(2, 4);
            let mut d = decoder(&e);
            let t = run(&mut d, &mut Stream::new().rec(305, &cmd(1, 7, 2, 9)), 1).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::Blit(BlitRaise::IcbRange),
                "range 0x7+0x2 runs past the indirect command buffer's size 0x8",
            );
            let mut d = decoder(&e);
            let t = run(&mut d, &mut Stream::new().rec(305, &cmd(1, 0, 8, 4)), 1).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::Blit(BlitRaise::IcbDestinationIndex),
                "destination index 4 is past the indirect command buffer's end",
            );
            let mut d = decoder(&e);
            run(&mut d, &mut Stream::new().rec(305, &cmd(1, 8, 0, 3)), 1).unwrap();
        }
    }

    #[test]
    fn decode_reset_and_optimize_indirect_command_buffer() {
        let e = env();
        let mut d = decoder(&e);
        let id = encoder_id(&d);
        let i = e.icb(1, 8);
        let c = |r: u32, loc: u64, len: u64| CmdOptimizeResetIndirectBuffer {
            indirect_command_buffer_reference: r,
            range_location: loc,
            range_length: len,
        };
        let mut s = Stream::new().rec(313, &c(1, 0, 8)).rec(312, &c(1, 7, 1));
        run(&mut d, &mut s, 1).unwrap();
        assert_eq!(
            encoded(&e, id),
            vec![
                (
                    "resetCommandsInBuffer:withRange:",
                    vec![Arg::obj(&i), Arg::val(&NsRange::new(0, 8))]
                ),
                (
                    "optimizeIndirectCommandBuffer:withRange:",
                    vec![Arg::obj(&i), Arg::val(&NsRange::new(7, 1))]
                ),
            ]
        );
        for op in [312u32, 313] {
            let mut d = decoder(&e);
            let t = run(&mut d, &mut Stream::new().rec(op, &c(1, 1, u64::MAX)), 1).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::Blit(BlitRaise::IcbRange),
                "range 0x1+0xffffffffffffffff runs past the indirect command buffer's size 0x8",
            );
            // The required lookup refuses ref 0.
            let mut d = decoder(&e);
            let t = run(&mut d, &mut Stream::new().rec(op, &c(0, 0, 0)), 1).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::MissingObject(ObjectKind::Icb),
                "no indirect command buffer for reference 0",
            );
        }
    }
}
