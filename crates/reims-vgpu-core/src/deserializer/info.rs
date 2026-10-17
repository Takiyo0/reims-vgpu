//! The info decoder (encoder type 4): answers the guest's queries about host objects by writing
//! replies into guest reply buffers. It has no encoder.
//!
//! Every reply is a CPU write into guest memory at decode time: it lands at once and no later
//! raise takes it back.
//!
//! A pipeline's counts and lengths (Metal's `NSUInteger`s: its thread counts, its threadgroup
//! memory and imageblock lengths) reach the guest as their low 32 bits: the compute info's three
//! are cut to 32 bits, then zero-extended into their 64-bit fields. Its GPU resource ID and
//! resource index are stored whole.

use std::sync::{Arc, Weak};

use zerocopy::{Immutable, IntoBytes};

use crate::deserializer::records::{
    Cmd2DSize16, CmdBufferHostResourceInfo, CmdComputePipelineStateInfo,
    CmdCopyRasterizationRateMapParameterBuffer, CmdGetRasterizationRateMapInfo,
    CmdHeapHostResourceInfo, CmdHeapTextureSizeAndAlignWithDescriptor,
    CmdHeapTextureSizeAndAlignWithDescriptor2, CmdICBHostResourceInfo, CmdMapCoordinates,
    CmdMapCoordinatesMultiple, CmdObjectHostInfo, CmdObjectUniqueIdentifier,
    CmdPipelineStateImageBlockMemoryLength, CmdRenderPipelineStateInfo,
    CmdSamplerStateHostResourceInfo, CmdTextureHostResourceInfo, Coordinate2DRecord,
    ReplyBufferHostResourceInfo, ReplyComputePipelineStateInfo, ReplyHeapTextureSizeAndAlign,
    ReplyHostObjectInfo, ReplyPipelineStateImageBlockMemoryLength,
    ReplyRasterizationRateMapCoordinate, ReplyRasterizationRateMapInfo,
    ReplyRenderPipelineStateInfo, ReplyUniqueIdentifierDeprecated, ResourceIdRecord, SegmentHeader,
};
use crate::deserializer::walk::next_record;
use crate::deserializer::{DeathKind, Deserializer, RaiseKind, upgrade};
use crate::memory::cursor::{Cursor, WritableCursor};
use crate::num::Truncate;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::icb::IndirectCommandBuffer;
use crate::platform::rate_map::RateMap;
use crate::platform::resource::{
    Buffer, ComputePipeline, DepthStencilState, GpuResource, Heap, RenderPipeline, SamplerState,
    Texture,
};
use crate::platform::types::{MtlCoordinate2D, MtlSize, SizeAndAlign};
use crate::raise::{OnFault, Raise, RaiseResult};
use crate::rt::Hold;

/// `MTLStorageModeShared`.
const MTL_STORAGE_MODE_SHARED: u64 = 0;

/// The object types a unique-identifier query may name.
const OBJECT_TYPE_TEXTURE: u32 = 1;
const OBJECT_TYPE_ICB: u32 = 5;
const OBJECT_TYPE_RENDER_PIPELINE: u32 = 9;
const OBJECT_TYPE_COMPUTE_PIPELINE: u32 = 10;

/// Writes `reply`'s whole size, padding included, and moves the cursor past it.
fn write_reply<T: IntoBytes + Immutable>(
    writer: &mut WritableCursor,
    reply: &T,
) -> RaiseResult<()> {
    // Past the end (a clamped advance), the write fails here.
    match writer.write(reply) {
        Ok(()) => Ok(()),
        Err(error) => Err(Raise::new(RaiseKind::ReplyWrite)
            .with_detail(format_args!("reply not writable: {error}"))),
    }
}

/// The reply cursor over `buffer_reference`, moved to `offset` (clamped at its end).
fn reply_cursor<P: Platform>(
    hold: Hold<'_>,
    d: &Deserializer<P>,
    buffer_reference: u32,
    offset: u64,
) -> RaiseResult<WritableCursor> {
    let Some(mut writer) = d
        .object_delegate()
        .new_memory_cursor_for_buffer(hold, buffer_reference)?
    else {
        crate::raise!(
            RaiseKind::ReplyCursor,
            format_args!("no reply cursor for buffer {buffer_reference}")
        );
    };
    writer.0.advance(offset);
    Ok(writer)
}

/// A heap texture's size and alignment reply, from the device's answer.
fn size_and_align_reply(s: SizeAndAlign) -> ReplyHeapTextureSizeAndAlign {
    ReplyHeapTextureSizeAndAlign {
        size: s.size,
        align: s.align,
    }
}

/// A host object's resource id and index reply.
fn host_object_reply(gpu_resource_id: u64, resource_index: u64) -> ReplyHostObjectInfo {
    ReplyHostObjectInfo {
        gpu_resource_id: ResourceIdRecord {
            impl_: gpu_resource_id,
        },
        resource_index,
    }
}

/// The info decoder: it holds its deserializer weakly (the caller holds a decoder past the
/// walk, and it must not keep the deserializer alive there).
pub struct InfoDecoder<P: Platform> {
    deserializer: Weak<Deserializer<P>>,
}

impl<P: Platform> InfoDecoder<P> {
    /// A decoder over `deserializer`.
    pub fn new(deserializer: Weak<Deserializer<P>>) -> Self {
        InfoDecoder { deserializer }
    }

    /// The deserializer, for one record.
    fn deserializer(&self) -> Arc<Deserializer<P>> {
        upgrade(&self.deserializer)
    }

    /// Decodes one segment's records. The header's flags are not read: there is no encoder to
    /// keep open or end, so a raise goes on as it is.
    pub fn decode(
        &mut self,
        hold: Hold<'_>,
        _header: &SegmentHeader,
        mut segment: Cursor,
    ) -> RaiseResult<()> {
        self.decode_records(hold, &mut segment)
    }

    fn decode_records(&mut self, hold: Hold<'_>, segment: &mut Cursor) -> RaiseResult<()> {
        while let Some((header, mut record)) = next_record(segment)? {
            // Each query reads its record's whole size.
            let c = &mut record;
            match header.command_id {
                450 => self.compute_pipeline_state_info(hold, c)?,
                451 => self.heap_texture_size_and_align(hold, c)?,
                452 => self.rate_map_info(hold, c)?,
                453 => self.copy_rate_map_parameter_buffer(hold, c)?,
                454 => self.map_screen_to_physical(hold, c)?,
                455 => self.map_physical_to_screen(hold, c)?,
                456 => self.map_physical_to_screen_multiple(hold, c)?,
                457 => self.render_pipeline_state_info(hold, c)?,
                458 => self.render_pipeline_image_block_memory_length(hold, c)?,
                459 => self.compute_pipeline_image_block_memory_length(hold, c)?,
                460 => self.unique_identifier(hold, c)?,
                461 => self.buffer_gpu_address(hold, c)?,
                462 => self.texture_host_info(hold, c)?,
                463 => self.heap_gpu_address(hold, c)?,
                464 => self.sampler_state_host_info(hold, c)?,
                465 => self.icb_host_info(hold, c)?,
                466 => self.render_pipeline_host_info(hold, c)?,
                467 => self.compute_pipeline_host_info(hold, c)?,
                468 => self.depth_stencil_host_info(hold, c)?,
                469 => self.heap_texture_size_and_align2(hold, c)?,
                id => crate::raise!(
                    RaiseKind::UnrecognizedCommand,
                    format_args!("unknown command {id}")
                ),
            }
        }
        Ok(())
    }

    /// Refuses a parameter buffer that is not shared or too short for `size` bytes at `offset`.
    /// `length` is asked in the check (unless the sum overflows) and again for the detail.
    fn check_buffer(buffer: &P::Buffer, offset: u64, size: u64) -> RaiseResult<()> {
        if buffer.storage_mode() != MTL_STORAGE_MODE_SHARED {
            crate::raise!(
                RaiseKind::BufferStorageMode,
                "parameter buffer is not in shared storage"
            );
        }
        let fits = offset
            .checked_add(size)
            .is_some_and(|end| end <= buffer.length());
        if !fits {
            crate::raise!(
                RaiseKind::BufferLength,
                format_args!(
                    "{size:#x} bytes at offset {offset:#x} run past the parameter buffer's {:#x} \
                     bytes",
                    buffer.length()
                )
            );
        }
        Ok(())
    }

    fn compute_pipeline_state_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdComputePipelineStateInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let pipeline = d.require_compute_pipeline_state(hold, command.pipeline_state_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        // Asked in this order.
        // Each count is cut to 32 bits, then zero-extended.
        // The 3 tail bytes are unspecified; zero here.
        let reply = ReplyComputePipelineStateInfo {
            max_total_threads_per_threadgroup: u64::from(
                pipeline.max_total_threads_per_threadgroup().low_u32(),
            ),
            // Never 0, which a guest divides by unchecked (WindowServer's luma kernel).
            thread_execution_width: match pipeline.thread_execution_width().low_u32() {
                0 => {
                    log::warn!(
                        "compute pipeline {} answered threadExecutionWidth 0: 32 given",
                        command.pipeline_state_reference
                    );
                    32
                }
                width => u64::from(width),
            },
            static_threadgroup_memory_length: u64::from(
                pipeline.static_threadgroup_memory_length().low_u32(),
            ),
            support_indirect_command_buffers: u8::from(pipeline.support_indirect_command_buffers()),
            unknown19: [0; 3],
        };
        write_reply(&mut writer, &reply)
    }

    fn heap_texture_size_and_align(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdHeapTextureSizeAndAlignWithDescriptor = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let serialized = command.descriptor;
        let Some(descriptor) = d
            .object_delegate()
            .texture_descriptor_for_descriptor(&serialized)?
        else {
            crate::fatal!(DeathKind::NilTextureDescriptor)
        };
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let size_and_align =
            crate::nil_send!(d.device(), heap_texture_size_and_align(&descriptor))?;
        write_reply(&mut writer, &size_and_align_reply(size_and_align))
    }

    fn heap_texture_size_and_align2(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdHeapTextureSizeAndAlignWithDescriptor2 =
            c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let serialized = command.descriptor;
        let Some(descriptor) = d
            .object_delegate()
            .texture_descriptor_for_descriptor2(&serialized)?
        else {
            crate::fatal!(DeathKind::NilTextureDescriptor)
        };
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let size_and_align =
            crate::nil_send!(d.device(), heap_texture_size_and_align(&descriptor))?;
        write_reply(&mut writer, &size_and_align_reply(size_and_align))
    }

    fn rate_map_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdGetRasterizationRateMapInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let rate_map =
            d.require_rasterization_rate_map(hold, command.rasterization_rate_map_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let size_and_align = rate_map.parameter_buffer_size_and_align();
        let granularity = rate_map.physical_granularity();
        // The granularity is cut to u16.
        write_reply(
            &mut writer,
            &ReplyRasterizationRateMapInfo {
                parameter_buffer_size: size_and_align.size,
                parameter_buffer_align: size_and_align.align,
                physical_granularity_width: granularity.width.low_u16(),
                physical_granularity_height: granularity.height.low_u16(),
            },
        )?;
        // `layerCount` is asked again before each layer.
        let mut layer: u64 = 0;
        while layer < rate_map.layer_count() {
            let size: MtlSize = rate_map.physical_size(layer)?;
            // Each layer's size is cut to u16.
            write_reply(
                &mut writer,
                &Cmd2DSize16 {
                    width: size.width.low_u16(),
                    height: size.height.low_u16(),
                },
            )?;
            layer = layer.wrapping_add(1);
        }
        Ok(())
    }

    fn copy_rate_map_parameter_buffer(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdCopyRasterizationRateMapParameterBuffer =
            c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let rate_map =
            d.require_rasterization_rate_map(hold, command.rasterization_rate_map_reference)?;
        let buffer = d.require_buffer(hold, command.buffer_reference)?;
        let size_and_align = rate_map.parameter_buffer_size_and_align();
        Self::check_buffer(&buffer, command.buffer_offset, size_and_align.size)?;
        // The mask is `align - 1`, wrapping (align 0: all ones).
        if command.buffer_offset & size_and_align.align.wrapping_sub(1) != 0 {
            crate::raise!(
                RaiseKind::BufferAlignment,
                format_args!(
                    "parameter buffer offset {:#x} is not aligned to {:#x}",
                    command.buffer_offset, size_and_align.align
                )
            );
        }
        // No reply: the copy is the answer.
        rate_map.copy_parameter_data(&buffer, command.buffer_offset)
    }

    fn map_screen_to_physical(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdMapCoordinates = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let rate_map =
            d.require_rasterization_rate_map(hold, command.rasterization_rate_map_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let physical = rate_map.screen_to_physical(
            MtlCoordinate2D::from(command.coordinates),
            u64::from(command.layer_index),
        )?;
        write_reply(
            &mut writer,
            &ReplyRasterizationRateMapCoordinate {
                x: physical.x,
                y: physical.y,
            },
        )
    }

    fn map_physical_to_screen(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdMapCoordinates = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let rate_map =
            d.require_rasterization_rate_map(hold, command.rasterization_rate_map_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let screen = rate_map.physical_to_screen(
            MtlCoordinate2D::from(command.coordinates),
            u64::from(command.layer_index),
        )?;
        write_reply(
            &mut writer,
            &ReplyRasterizationRateMapCoordinate {
                x: screen.x,
                y: screen.y,
            },
        )
    }

    /// The coordinates are read through a copy of the reply cursor, so each is replaced by its
    /// mapping in place, before the next is read.
    fn map_physical_to_screen_multiple(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdMapCoordinatesMultiple = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let rate_map =
            d.require_rasterization_rate_map(hold, command.rasterization_rate_map_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        // Shares the ranges, copies the position.
        let mut reader = writer.clone();
        for _ in 0..command.coordinate_count {
            let Ok(physical) = reader.0.read::<Coordinate2DRecord>() else {
                crate::raise!(RaiseKind::CoordinateRead, "guest coordinate not readable");
            };
            let screen = rate_map.physical_to_screen(
                MtlCoordinate2D::from(physical),
                u64::from(command.layer_index),
            )?;
            write_reply(&mut writer, &Coordinate2DRecord::from(screen))?;
        }
        Ok(())
    }

    fn render_pipeline_state_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdRenderPipelineStateInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let pipeline = d.require_render_pipeline_state(hold, command.pipeline_state_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        // Asked in this order.
        // The 2 padding bytes are unspecified; zero here.
        let reply = ReplyRenderPipelineStateInfo {
            max_total_threads_per_threadgroup: pipeline
                .max_total_threads_per_threadgroup()
                .low_u32(),
            imageblock_sample_length: pipeline.imageblock_sample_length().low_u32(),
            threadgroup_size_matches_tile_size: u8::from(
                pipeline.threadgroup_size_matches_tile_size(),
            ),
            support_indirect_command_buffers: u8::from(pipeline.support_indirect_command_buffers()),
            unknown0a: [0; 2],
        };
        write_reply(&mut writer, &reply)
    }

    fn render_pipeline_image_block_memory_length(
        &self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdPipelineStateImageBlockMemoryLength = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let pipeline = d.require_render_pipeline_state(hold, command.pipeline_state_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let dimensions = MtlSize::from(command.imageblock_dimensions);
        let length = pipeline.imageblock_memory_length(dimensions)?;
        // Cut to 32 bits.
        write_reply(
            &mut writer,
            &ReplyPipelineStateImageBlockMemoryLength {
                imageblock_memory_length: length.low_u32(),
            },
        )
    }

    fn compute_pipeline_image_block_memory_length(
        &self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdPipelineStateImageBlockMemoryLength = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let pipeline = d.require_compute_pipeline_state(hold, command.pipeline_state_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let dimensions = MtlSize::from(command.imageblock_dimensions);
        let length = pipeline.imageblock_memory_length(dimensions)?;
        // Cut to 32 bits.
        write_reply(
            &mut writer,
            &ReplyPipelineStateImageBlockMemoryLength {
                imageblock_memory_length: length.low_u32(),
            },
        )
    }

    fn unique_identifier(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdObjectUniqueIdentifier = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        // The reply cursor is asked for before the type is checked.
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let reference = command.object_reference;
        let identifier = match command.object_type {
            OBJECT_TYPE_TEXTURE => d.require_texture(hold, reference)?.unique_identifier(),
            OBJECT_TYPE_ICB => d
                .require_indirect_command_buffer(hold, reference)?
                .unique_identifier(),
            OBJECT_TYPE_RENDER_PIPELINE => d
                .require_render_pipeline_state(hold, reference)?
                .guest_unique_identifier(),
            OBJECT_TYPE_COMPUTE_PIPELINE => d
                .require_compute_pipeline_state(hold, reference)?
                .guest_unique_identifier(),
            other => crate::raise!(
                RaiseKind::UniqueIdentifierType,
                format_args!("object type {other} has no unique identifier")
            ),
        };
        write_reply(
            &mut writer,
            &ReplyUniqueIdentifierDeprecated {
                unique_identifier: identifier,
            },
        )
    }

    fn buffer_gpu_address(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdBufferHostResourceInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let buffer = d.require_buffer(hold, command.buffer_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        write_reply(
            &mut writer,
            &ReplyBufferHostResourceInfo {
                gpu_address: buffer.gpu_address(),
            },
        )
    }

    fn texture_host_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdTextureHostResourceInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let texture = d.require_texture(hold, command.texture_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        // The resource id, then the index.
        let gpu_resource_id = texture.gpu_resource_id();
        let reply = host_object_reply(gpu_resource_id, texture.resource_index());
        write_reply(&mut writer, &reply)
    }

    fn heap_gpu_address(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdHeapHostResourceInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let heap = d.require_heap(hold, command.heap_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        write_reply(
            &mut writer,
            &ReplyBufferHostResourceInfo {
                gpu_address: heap.gpu_address(),
            },
        )
    }

    fn sampler_state_host_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSamplerStateHostResourceInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let sampler_state = d.require_sampler_state(hold, command.sampler_state_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let gpu_resource_id = sampler_state.gpu_resource_id();
        let reply = host_object_reply(gpu_resource_id, sampler_state.resource_index());
        write_reply(&mut writer, &reply)
    }

    fn icb_host_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdICBHostResourceInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let icb =
            d.require_indirect_command_buffer(hold, command.indirect_command_buffer_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let gpu_resource_id = icb.gpu_resource_id();
        let reply = host_object_reply(gpu_resource_id, icb.resource_index());
        write_reply(&mut writer, &reply)
    }

    fn render_pipeline_host_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdObjectHostInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let pipeline = d.require_render_pipeline_state(hold, command.object_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let gpu_resource_id = pipeline.gpu_resource_id();
        let reply = host_object_reply(gpu_resource_id, pipeline.resource_index());
        write_reply(&mut writer, &reply)
    }

    fn compute_pipeline_host_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdObjectHostInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let pipeline = d.require_compute_pipeline_state(hold, command.object_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        let gpu_resource_id = pipeline.gpu_resource_id();
        let reply = host_object_reply(gpu_resource_id, pipeline.resource_index());
        write_reply(&mut writer, &reply)
    }

    fn depth_stencil_host_info(&self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdObjectHostInfo = c.read_or_raise(OnFault::Return)?;
        let d = self.deserializer();
        let state = d.require_depth_stencil_state(hold, command.object_reference)?;
        let mut writer = reply_cursor(
            hold,
            &d,
            command.reply_buffer_reference,
            command.reply_buffer_offset,
        )?;
        // Either getter may raise (Metal's stubs), and the raise goes on.
        let gpu_resource_id = state.gpu_resource_id()?;
        let reply = host_object_reply(gpu_resource_id, state.resource_index()?);
        write_reply(&mut writer, &reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deserializer::blit::test_support::{Env, Stream, env, segment};

    use crate::platform::Handle;
    use crate::raise::RaiseId;
    use crate::raise::death::DeathKind as DeathClass;

    use crate::testkit::{Arg, FakePlatform, kind};

    type Dec = InfoDecoder<FakePlatform>;

    fn decoder(e: &Env) -> Dec {
        InfoDecoder::new(Arc::downgrade(&e.d))
    }

    fn run(d: &mut Dec, s: &mut Stream) -> RaiseResult<()> {
        let c = s.cursor();
        d.decode(Hold::NONE, &segment(4, 0), c)
    }

    /// Asserts `r` is the deserializer's `kind`, with `detail` when one is given.
    #[track_caller]
    fn assert_raised(r: &Raise, kind: RaiseKind, detail: Option<&str>) {
        assert_eq!(r.id(), RaiseId::from(kind), "{r}");
        if let Some(detail) = detail {
            assert_eq!(r.detail(), Some(detail), "{r}");
        }
    }

    #[test]
    fn its_encoder_type_is_four_and_its_fault_sends_nothing() {
        let e = env();
        let d = decoder(&e);
        // Weak: the caller's is the only reference.
        assert_eq!(d.deserializer.as_ptr(), Arc::as_ptr(&e.d));
        assert_eq!(Arc::strong_count(&e.d), 1);
        let mut d = crate::deserializer::decoder::Decoder::Info(d);
        assert_eq!(d.encoder_type(), 4);
        d.fault().unwrap();
        d.fault().unwrap();
        drop(d);
        assert!(e.p.call_log().is_empty());
    }

    /// The decoder holds the deserializer weakly: the task's release after the decode is the
    /// deserializer's last while the caller still holds the decoder.
    #[test]
    fn it_holds_the_deserializer_weakly() {
        let Env { p, r, d } = env();
        let dev = p.device();
        let device_refs = dev.retain_count();
        let mut dec = InfoDecoder::new(Arc::downgrade(&d));
        let mut s = Stream::new();
        dec.decode(Hold::NONE, &segment(4, 0), s.cursor()).unwrap();
        assert_eq!(Arc::strong_count(&d), 1);
        let gone = Arc::downgrade(&d);
        drop(d);
        assert!(gone.upgrade().is_none());
        assert_eq!(dev.retain_count(), device_refs - 1);
        assert_eq!(Arc::strong_count(&r), 1);
        assert!(p.world().live("MTLSharedEventListener").is_empty());
        p.call_log().clear();
        drop(dec);
        assert!(p.call_log().is_empty());
        assert_eq!(dev.retain_count(), device_refs - 1);
    }

    mod decode {
        use super::*;

        #[test]
        fn an_empty_segment_does_nothing_whatever_its_flags() {
            let e = env();
            let mut d = decoder(&e);
            let c = Stream::new().cursor();
            d.decode(Hold::NONE, &segment(4, 1), c).unwrap();
            let mut s = Stream::new().bytes(&[0; 7]);
            run(&mut d, &mut s).unwrap();
            assert!(e.p.call_log().is_empty());
        }

        #[test]
        fn loop_errors() {
            let e = env();
            let mut d = decoder(&e);
            // A record length under 8 underflows
            let t = run(&mut d, &mut Stream::new().raw(450, 3, &[0; 8])).unwrap_err();
            assert_raised(&t, RaiseKind::RecordUnderflow, None);
            // A record overrunning the segment fails before dispatch
            let t = run(&mut d, &mut Stream::new().raw(450, 0x18, &[0; 4])).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::RecordOverrun,
                Some("record not readable: a sub-cursor of 0x10 bytes, with 0x4 left"),
            );
            // Blit's ids are not the info decoder's.
            for id in [300u32, 449, 470] {
                let t = run(&mut d, &mut Stream::new().raw(id, 8, &[])).unwrap_err();
                assert_raised(&t, RaiseKind::UnrecognizedCommand, None);
            }
            // A record shorter than its struct fails to read.
            let t = run(&mut d, &mut Stream::new().raw(450, 8 + 0xf, &[0; 0xf])).unwrap_err();
            assert_eq!(
                t.detail(),
                Some(
                    "record not readable: read of 0x10 bytes at offset 0x0 runs past the cursor's 0xf bytes"
                )
            );
            assert!(e.lookups().is_empty());
        }

        /// The raise goes on to the walk's handler as the core's own
        #[test]
        fn the_raise_goes_on_to_the_callers_handler() {
            let e = env();
            let mut d = decoder(&e);
            let mut s = Stream::new().raw(451, 8, &[]);
            let c = s.cursor();
            let r = d.decode(Hold::NONE, &segment(4, 0), c);
            assert_eq!(
                r.err().map(|r| r.site().kind),
                Some(DeathClass::UnhandledRaise)
            );
        }
    }

    mod heap_texture_size_and_align {}

    fn rate_map(e: &Env, reference: u32) -> crate::testkit::Fk<kind::RateMap> {
        let m = e.state::<kind::RateMap>(reference);
        m.set_parameter_buffer_size_and_align(SizeAndAlign {
            size: 0x40,
            align: 0x10,
        });
        m
    }

    mod copy_rate_map_parameter_buffer {
        use super::*;

        fn cmd(offset: u64) -> CmdCopyRasterizationRateMapParameterBuffer {
            CmdCopyRasterizationRateMapParameterBuffer {
                rasterization_rate_map_reference: 1,
                buffer_reference: 2,
                buffer_offset: offset,
            }
        }

        #[test]
        fn copies_into_the_buffer_with_no_reply_cursor() {
            let e = env();
            let m = rate_map(&e, 1);
            let b = e.buffer(2, 0x60, 0);
            run(&mut decoder(&e), &mut Stream::new().rec(453, &cmd(0x20))).unwrap();
            assert_eq!(
                e.lookups(),
                vec![("get_rasterization_rate_map", 1), ("get_buffer", 2)]
            );
            let calls = e.on(m.id());
            let last = calls.last().unwrap();
            assert_eq!(last.sel, "copyParameterDataToBuffer:offset:");
            assert_eq!(last.args, vec![Arg::obj(&b), Arg::U64(0x20)]);
        }

        #[test]
        fn storage_then_length_then_alignment() {
            let e = env();
            let _m = rate_map(&e, 1);
            let b = e.buffer(2, 0x60, 0x20);
            let t = run(&mut decoder(&e), &mut Stream::new().rec(453, &cmd(0))).unwrap_err();
            assert_raised(&t, RaiseKind::BufferStorageMode, None);
            drop(b);
            let e = env();
            let _m = rate_map(&e, 1);
            let _b = e.buffer(2, 0x60, 0);
            let t = run(&mut decoder(&e), &mut Stream::new().rec(453, &cmd(0x30))).unwrap_err();
            assert_raised(
                &t,
                RaiseKind::BufferLength,
                Some("0x40 bytes at offset 0x30 run past the parameter buffer's 0x60 bytes"),
            );
            // An overflowing sum is refused too.
            e.p.call_log().clear();
            let t = run(
                &mut decoder(&e),
                &mut Stream::new().rec(453, &cmd(u64::MAX)),
            )
            .unwrap_err();
            assert_raised(&t, RaiseKind::BufferLength, None);
            // Misaligned.
            let t = run(&mut decoder(&e), &mut Stream::new().rec(453, &cmd(0x18))).unwrap_err();
            assert_raised(&t, RaiseKind::BufferAlignment, None);
        }

        #[test]
        fn align_zero_refuses_every_nonzero_offset() {
            // 453 alignment mask `align - 1` (align 0 → every nonzero offset fails)
            let e = env();
            let m = e.state::<kind::RateMap>(1);
            m.set_parameter_buffer_size_and_align(SizeAndAlign { size: 1, align: 0 });
            let _b = e.buffer(2, 0x60, 0);
            run(&mut decoder(&e), &mut Stream::new().rec(453, &cmd(0))).unwrap();
            let t = run(&mut decoder(&e), &mut Stream::new().rec(453, &cmd(1))).unwrap_err();
            assert_raised(&t, RaiseKind::BufferAlignment, None);
        }
    }

    mod map_physical_to_screen_multiple {}
}
