//! The render decoder (encoder type 0): one render pass over a run of segments. The first
//! segment's pass prefix builds the pass descriptor ([`pass`]); the main loop's records are
//! [`draw`]'s (draws, ICB executes, tile dispatches) and [`state`]'s (the encoder's state,
//! bindings, fences and the tile-dimensions reply).
//!
//! # Contract
//!
//! - **A fresh segment** (one that does not continue the previous) reads the pass prefix into a
//!   new pass descriptor, then asks the command buffer for an encoder over it. The descriptor
//!   goes once the encoder is made; the decoder holds the encoder
//!   (`tests::a_fresh_segment_opens_encodes_and_ends`,
//!   `tests::the_deserializer_is_not_kept_while_the_decoder_waits`).
//! - **No encoder** (no command buffer, a nil answer from Metal, or a continuation with none
//!   open) raises after the prefix, before any record
//!   (`tests::no_command_buffer_raises_after_the_prefix`).
//! - **Records** go through the walk's record loop ([`next_record`]); an id outside the table is
//!   unrecognized (`tests::the_dispatch_table_covers_every_id`). 0x6a decodes as 0x69
//!   (`tests::id_0x6a_is_set_depth_store_action`).
//! - **The pass ends** with the segment that the next does not continue: `endEncoding`, then the
//!   pipeline state is released. The pipeline state persists across continuation segments
//!   (`tests::a_continuation_keeps_the_encoder_and_the_pipeline`); the highest colour attachment
//!   index a pass described is never reset (see [`pass`]).
//! - **A raise** out of the segment faults the decoder (`endEncoding`, then the encoder and the
//!   pipeline state are released) and goes on (`tests::an_underflow_faults_the_decoder`). A
//!   raise out of that `endEncoding` goes on in its place, with the encoder and the pipeline
//!   state still set (`tests::a_raise_in_the_fault_replaces_the_one_in_flight`). Every handler
//!   runs inside [`RenderDecoder::decode`]'s handler, so the handlers need none of their own.
//! - A carried decoder asked for a second encoder over its open one leaks the first (Metal asserts
//!   first on a real device) (`tests::a_second_encoder_leaks_the_first`).
//! - **Drop** releases the encoder, then the pipeline state, and sends nothing
//!   (`tests::drop_releases_the_encoder_then_the_pipeline`).

pub mod draw;
pub mod pass;
pub mod state;

use std::fmt;
use std::sync::{Arc, Weak};

use crate::deserializer::records::SegmentHeader;
use crate::deserializer::walk::next_record;
use crate::deserializer::{Deserializer, upgrade};
use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::platform::caps::{DeviceCaps, Flags};
use crate::platform::command::CommandBuffer as _;
use crate::platform::device::GpuDevice as _;
use crate::platform::render::{RenderCmd, RenderEncoder as _};
use crate::raise::{Kept, OnFault, RaiseId, RaiseResult, keep_forever};
use crate::rt::Hold;

/// The render decoder's raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// The pass has no encoder: no command buffer, a nil answer, or a continuation with none
    /// open.
    NoEncoder,
    /// A pass-prefix record that is neither 0x1a nor one of 0x1e-0x24.
    UnexpectedPassCommand,
    /// More than eight programmable sample positions.
    SamplePositionCount,
    /// A heap or resource list longer than the rest of its record.
    ListLength,
    /// A threadgroup memory index at or past the device's `maxComputeLocalMemorySizes`.
    ThreadgroupMemoryIndex,
    /// A primitive type of 10 or more.
    PrimitiveType,
    /// A draw with no pipeline state set.
    DrawWithoutPipeline,
    /// An indexed draw whose index buffer is not found.
    NoIndexBuffer,
    /// An indirect draw whose indirect buffer is not found.
    NoIndirectBuffer,
    /// An indexed patch draw whose control-point index buffer is not found.
    NoControlPointIndexBuffer,
    /// A tile dispatch with no pipeline state set.
    DispatchWithoutPipeline,
    /// An ICB execute whose indirect command buffer is not found.
    NoIcb,
    /// An ICB execute's range past the ICB's size, or overflowing.
    IcbRange,
    /// An ICB execute whose indirect range buffer is not found.
    NoIndirectRangeBuffer,
    /// 0x16, which the decoder does not support.
    BarrierResourcesUnsupported,
    /// A colour store action's attachment index past the pass's highest.
    RenderTargetIndex,
    /// A depth-stencil state reference of 0.
    DepthStencilStateReference,
    /// A render pipeline state reference of 0.
    RenderPipelineReference,
    /// A buffer range past the device's limit.
    BufferCount,
    /// A buffer index at or past the device's limit.
    BufferIndex,
    /// A sampler range past the device's limit.
    SamplerCount,
    /// A sampler index at or past the device's limit.
    SamplerIndex,
    /// A texture range past the device's limit.
    TextureCount,
    /// More viewports or scissor rects than the device's `maxViewportCount`.
    ViewportCount,
    /// A primitive-restart index at or past the device's `maxViewportCount`.
    ViewportIndex,
    /// A per-buffer attribute stride on a device without dynamic attribute strides.
    AttributeStrideUnsupported,
    /// A clip plane index of 6 or more.
    ClipPlaneIndex,
    /// A colour resolve texture's attachment index at or past the pass's highest.
    AttachmentIndex,
    /// More view mappings than the device's `maxVertexAmplificationCount`.
    VertexAmplificationCount,
    /// 0xa4 with no encoder.
    TileDimensionsWithoutEncoder,
    /// 0xa4's reply buffer has no cursor.
    ReplyCursor,
    /// 0xa4's reply does not fit its reply buffer.
    ReplyWrite,
}

impl RaiseKind {
    /// The dotted id.
    pub fn name(self) -> &'static str {
        match self {
            RaiseKind::NoEncoder => "deserializer.render.no_encoder",
            RaiseKind::UnexpectedPassCommand => "deserializer.render.pass.unexpected_command",
            RaiseKind::SamplePositionCount => "deserializer.render.pass.sample_position_count",
            RaiseKind::ListLength => "deserializer.render.list_length",
            RaiseKind::ThreadgroupMemoryIndex => "deserializer.render.threadgroup_memory_index",
            RaiseKind::PrimitiveType => "deserializer.render.draw.primitive_type",
            RaiseKind::DrawWithoutPipeline => "deserializer.render.draw.no_pipeline",
            RaiseKind::NoIndexBuffer => "deserializer.render.draw.no_index_buffer",
            RaiseKind::NoIndirectBuffer => "deserializer.render.draw.no_indirect_buffer",
            RaiseKind::NoControlPointIndexBuffer => {
                "deserializer.render.draw.no_control_point_index_buffer"
            }
            RaiseKind::DispatchWithoutPipeline => "deserializer.render.tile.no_pipeline",
            RaiseKind::NoIcb => "deserializer.render.icb.missing",
            RaiseKind::IcbRange => "deserializer.render.icb.range",
            RaiseKind::NoIndirectRangeBuffer => "deserializer.render.icb.no_range_buffer",
            RaiseKind::BarrierResourcesUnsupported => {
                "deserializer.render.barrier_resources_unsupported"
            }
            RaiseKind::RenderTargetIndex => "deserializer.render.render_target_index",
            RaiseKind::DepthStencilStateReference => {
                "deserializer.render.depth_stencil_state_reference"
            }
            RaiseKind::RenderPipelineReference => "deserializer.render.render_pipeline_reference",
            RaiseKind::BufferCount => "deserializer.render.buffer_count",
            RaiseKind::BufferIndex => "deserializer.render.buffer_index",
            RaiseKind::SamplerCount => "deserializer.render.sampler_count",
            RaiseKind::SamplerIndex => "deserializer.render.sampler_index",
            RaiseKind::TextureCount => "deserializer.render.texture_count",
            RaiseKind::ViewportCount => "deserializer.render.viewport_count",
            RaiseKind::ViewportIndex => "deserializer.render.viewport_index",
            RaiseKind::AttributeStrideUnsupported => {
                "deserializer.render.attribute_stride_unsupported"
            }
            RaiseKind::ClipPlaneIndex => "deserializer.render.clip_plane_index",
            RaiseKind::AttachmentIndex => "deserializer.render.attachment_index",
            RaiseKind::VertexAmplificationCount => "deserializer.render.vertex_amplification_count",
            RaiseKind::TileDimensionsWithoutEncoder => "deserializer.render.tile.no_encoder",
            RaiseKind::ReplyCursor => "deserializer.render.tile.reply_cursor",
            RaiseKind::ReplyWrite => "deserializer.render.tile.reply_write",
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<RaiseKind> for RaiseId {
    fn from(k: RaiseKind) -> RaiseId {
        RaiseId::Deserializer(super::RaiseKind::Render(k))
    }
}

/// The render decoder: the pass's encoder and pipeline state, across the segments it is carried
/// through.
pub struct RenderDecoder<P: Platform> {
    /// Held weakly: the caller holds the decoder past the walk, and it must not keep the
    /// deserializer alive there.
    deserializer: Weak<Deserializer<P>>,
    /// The pass's encoder, from the fresh segment on; kept (ended) after the pass ends, released
    /// by a fault.
    encoder: Option<P::RenderEncoder>,
    /// The exec's command buffer; `None` when the caller's queue answered none.
    command_buffer: Option<P::CommandBuffer>,
    /// The pipeline state the last 0x74 set, until the pass ends or faults.
    pipeline: Option<P::RenderPipeline>,
    /// The highest colour attachment index a pass described with a texture.
    max_color_attachment_index: u32,
}

impl<P: Platform> RenderDecoder<P> {
    /// A decoder over `deserializer` (held weakly) and the exec's command buffer.
    pub fn new(
        deserializer: Weak<Deserializer<P>>,
        command_buffer: Option<P::CommandBuffer>,
    ) -> Self {
        RenderDecoder {
            deserializer,
            encoder: None,
            command_buffer,
            pipeline: None,
            max_color_attachment_index: 0,
        }
    }

    /// The deserializer, for one record.
    fn deserializer(&self) -> Arc<Deserializer<P>> {
        upgrade(&self.deserializer)
    }

    /// Decodes one segment. A raise out of it has faulted the decoder.
    pub fn decode(
        &mut self,
        hold: Hold<'_>,
        header: &SegmentHeader,
        mut segment: Cursor,
    ) -> RaiseResult<()> {
        let mut decoded = self.decode_segment(hold, header, &mut segment);
        if decoded.is_err() {
            // Outside the handler: a raise from the fault goes on in place of this one, which
            // ends here.
            hold.cleanup_step(&mut decoded, self.fault())?;
        }
        decoded
    }

    fn decode_segment(
        &mut self,
        hold: Hold<'_>,
        header: &SegmentHeader,
        segment: &mut Cursor,
    ) -> RaiseResult<()> {
        if header.continues_previous == 0 {
            let t_encoder = std::time::Instant::now();
            let descriptor = self.pass_prefix(hold, segment)?;
            let encoder = match &self.command_buffer {
                Some(buffer) => buffer.render_encoder(OnFault::Return, &descriptor)?,
                None => None,
            };
            drop(descriptor);
            // A carried decoder's open encoder is replaced without being released: it is kept for
            // good, with its command buffer and what that holds.
            if let Some(open) = self.encoder.take() {
                keep_forever(open, Kept::MetalNumbering);
            }
            self.encoder = encoder;
            crate::perfstats::DECODE_ENCODER.add(t_encoder.elapsed());
        }
        if self.encoder.is_none() {
            crate::raise!(RaiseKind::NoEncoder, "no render encoder");
        }
        while let Some((record_header, mut record)) = next_record(segment)? {
            let t = std::time::Instant::now();
            let dispatched = self.dispatch(hold, record_header.command_id, &mut record);
            crate::perfstats::record(record_header.command_id, t.elapsed());
            dispatched?;
        }
        if header.continued_by_next == 0 {
            self.send(RenderCmd::EndEncoding)?;
            self.pipeline = None;
        }
        Ok(())
    }

    /// One main-loop record, by its command id.
    fn dispatch(&mut self, hold: Hold<'_>, command_id: u32, c: &mut Cursor) -> RaiseResult<()> {
        match command_id {
            0x0 => self.draw_primitives64(c),
            0x1 => self.draw_primitives16(c),
            0x2 => self.draw_instanced_primitives64(c),
            0x3 => self.draw_instanced_primitives16(c),
            0x4 => self.draw_instanced_base_primitives64(c),
            0x5 => self.draw_instanced_base_primitives16(c),
            0x6 => self.draw_indexed_primitives64(hold, c),
            0x7 => self.draw_indexed_primitives16(hold, c),
            0x8 => self.draw_indexed_instanced_primitives64(hold, c),
            0x9 => self.draw_indexed_instanced_primitives16(hold, c),
            0xa => self.draw_indexed_instanced_base_primitives64(hold, c),
            0xb => self.draw_indexed_instanced_base_primitives16(hold, c),
            0xc => self.draw_patches64(hold, c),
            0xd => self.draw_patches16(hold, c),
            0xe => self.draw_indexed_patches64(hold, c),
            0xf => self.draw_indexed_patches16(hold, c),
            0x10 => self.draw_primitives_indirect(hold, c),
            0x11 => self.draw_indexed_primitives_indirect(hold, c),
            0x12 => self.draw_patches_indirect(hold, c),
            0x13 => self.draw_indexed_patches_indirect(hold, c),
            0x14 => self.execute_commands_in_buffer(hold, c),
            0x15 => self.execute_commands_in_buffer_ranged(hold, c),
            0x16 => self.barrier_resources(c),
            0x17 => self.barrier_scope(c),
            0x18 => self.update_fence(hold, c),
            0x19 => self.wait_for_fence(hold, c),
            0x1b => self.use_heaps_with_stages(hold, c),
            0x1c => self.draw_indexed_instanced_base_primitives64_2(hold, c),
            0x1d => self.draw_indexed_instanced_base_primitives16_2(hold, c),
            0x65 => self.set_blend_color(c),
            0x66 => self.set_color_store_action(c),
            0x67 => self.set_color_store_action_options(c),
            0x68 => self.set_depth_stencil_state(hold, c),
            0x69 => self.set_depth_store_action(c),
            // 0x6a decodes as SetDepthStoreAction too.
            0x6a => self.set_depth_store_action(c),
            0x6b => self.set_cull_mode(c),
            0x6c => self.set_depth_bias(c),
            0x6d => self.set_depth_clip_mode(c),
            0x6e => self.set_fragment_buffers(hold, c),
            0x6f => self.set_fragment_buffer_offset(c),
            0x70 => self.set_fragment_sampler_states(hold, c),
            0x71 => self.set_fragment_sampler_states_lod_clamp(hold, c),
            0x72 => self.set_fragment_textures(hold, c),
            0x73 => self.set_front_facing_winding(c),
            0x74 => self.set_render_pipeline_state(hold, c),
            0x75 => self.set_scissor_rect(c),
            0x76 => self.set_scissor_rects(c),
            0x77 => self.set_stencil_ref(c),
            0x78 => self.set_stencil_store_action(c),
            0x79 => self.set_stencil_store_action_options(c),
            0x7a => self.set_tessellation_factor_buffer(hold, c),
            0x7b => self.set_tessellation_factor_scale(c),
            0x7c => self.set_triangle_fill_mode(c),
            0x7d => self.set_vertex_buffers(hold, c),
            0x7e => self.set_vertex_buffer_offset(c),
            0x7f => self.set_vertex_sampler_states(hold, c),
            0x80 => self.set_vertex_sampler_states_lod_clamp(hold, c),
            0x81 => self.set_vertex_textures(hold, c),
            0x82 => self.set_viewport(c),
            0x83 => self.set_viewports(c),
            0x84 => self.set_visibility_result_mode(c),
            0x85 => self.texture_barrier(c),
            0x86 => self.use_heaps(hold, c),
            0x87 => self.use_resources(hold, c),
            0x88 => self.set_line_width(c),
            0x89 => self.use_resources_with_stages(hold, c),
            0x8a => self.set_alpha_test_reference_value(c),
            0x8b => self.set_point_size(c),
            0x8c => self.set_clip_plane(c),
            0x8d => self.set_vertex_sampler_state(hold, c),
            0x8e => self.set_fragment_sampler_state(hold, c),
            0x8f => self.set_viewport_transform_enabled(c),
            0x90 => self.set_provoking_vertex_mode(c),
            0x91 => self.set_primitive_restart_index_enabled(c),
            0x92 => self.set_triangle_fill_mode_front_back(c),
            0x93 => self.set_transform_feedback_state(c),
            0x94 => self.set_depth_cleared(c),
            0x95 => self.set_stencil_cleared(c),
            0x96 => self.set_color_resolve_texture(hold, c),
            0x97 => self.set_depth_resolve_texture(hold, c),
            0x98 => self.set_stencil_resolve_texture(hold, c),
            0x99 => self.set_vertex_amplification_mode(c),
            0x9a => self.set_vertex_amplification_count(c),
            0x9b => self.dispatch_threads_per_tile(c),
            0x9c => self.set_threadgroup_memory_length(c),
            0x9d => self.set_tile_buffers(hold, c),
            0x9e => self.set_tile_buffer_offset(c),
            0x9f => self.set_tile_sampler_states(hold, c),
            0xa0 => self.set_tile_sampler_states_lod_clamp(hold, c),
            0xa1 => self.set_tile_textures(hold, c),
            0xa2 => self.dispatch_threads_per_tile_in_region(c),
            0xa3 => self.dispatch_threads_per_tile_in_region_with_index(c),
            0xa4 => self.get_tile_dimensions(hold, c),
            0xa5 => self.set_vertex_buffers_with_stride(hold, c),
            0xa6 => self.set_vertex_buffer_offset_with_stride(c),
            // 0x1a (the prefix's), 0x1e-0x64 and 0xa7 on
            id => crate::raise!(
                super::RaiseKind::UnrecognizedCommand,
                format_args!("unknown command {id}")
            ),
        }
    }

    /// Ends the encoder after a raise, then releases it and the pipeline state. A raise from
    /// `endEncoding` stops it before the releases; with no encoder it sends nothing.
    pub fn fault(&mut self) -> RaiseResult<()> {
        self.send(RenderCmd::EndEncoding)?;
        self.encoder = None;
        self.pipeline = None;
        Ok(())
    }

    /// Sends `cmd` to the encoder; with none, nothing is sent (a message to nil). The caller
    /// has evaluated every argument, lookups included.
    fn send(&self, cmd: RenderCmd<'_, P>) -> RaiseResult<()> {
        match &self.encoder {
            Some(encoder) => Ok(encoder.encode(OnFault::Return, cmd)?),
            None => Ok(()),
        }
    }

    /// A limit from the device's capabilities; a task without a device answers 0.
    fn device_limit(&self, limit: impl FnOnce(&DeviceCaps) -> u64) -> u64 {
        self.deserializer()
            .device()
            .map_or(0, |device| limit(&device.caps()))
    }

    /// A flag from the device's capabilities; a task without a device answers `false`.
    fn device_supports(&self, flag: impl FnOnce(&Flags) -> bool) -> bool {
        self.deserializer()
            .device()
            .is_some_and(|device| flag(&device.caps().flags))
    }

    /// The buffer `reference` names, if any (the delegate's nullable lookup).
    fn buffer(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<Option<P::Buffer>> {
        self.deserializer()
            .object_delegate()
            .get_buffer(hold, reference)
    }

    /// The texture `reference` names, if any.
    fn texture(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<Option<P::Texture>> {
        self.deserializer()
            .object_delegate()
            .get_texture(hold, reference)
    }

    /// The sampler state `reference` names, if any.
    fn sampler_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<P::SamplerState>> {
        self.deserializer()
            .object_delegate()
            .get_sampler_state(hold, reference)
    }
}

/// `Option<X>` elements as the `Option<&X>` slice a list send takes.
fn refs<T>(v: &[Option<T>]) -> Vec<Option<&T>> {
    v.iter().map(Option::as_ref).collect()
}

impl<P: Platform> Drop for RenderDecoder<P> {
    /// The encoder, then the pipeline state; then the command buffer with the other fields.
    fn drop(&mut self) {
        self.encoder = None;
        self.pipeline = None;
    }
}

/// Test scaffolding shared by this module's tests, `pass`'s and the walk's.
#[cfg(test)]
pub(super) mod test_rig {

    use std::fmt::Debug;
    use std::sync::Arc;

    use zerocopy::{Immutable, IntoBytes};

    use super::RenderDecoder;
    use crate::deserializer::Deserializer;
    use crate::deserializer::records::CmdHeader;
    use crate::deserializer::test_lookups::Lookups;
    use crate::deserializer::test_lookups::set_caps;
    use crate::host::HostPtr;
    use crate::memory::cursor::Cursor;
    use crate::platform::caps::DeviceCaps;
    use crate::platform::{Handle, ObjId};
    use crate::raise::{RaiseId, RaiseResult};
    use crate::testkit::calllog::Call;
    use crate::testkit::objects::{Fk, Kind, Model, kind};
    use crate::testkit::platform::FakePlatform;

    /// A render decoder over a fake platform; lookups answered by a resource manager.
    pub struct Rig {
        pub p: Arc<FakePlatform>,
        pub seams: Arc<Lookups>,
        pub dec: RenderDecoder<FakePlatform>,
        pub command_buffer: ObjId,
        /// The deserializer the decoder reaches weakly: the caller's reference.
        pub deserializer: Arc<Deserializer<FakePlatform>>,
    }

    impl Rig {
        /// A fresh decoder: no encoder, no pipeline; the call log cleared.
        pub fn new() -> Rig {
            let p = Arc::new(FakePlatform::new());
            let seams = Arc::new(Lookups::new(&p));
            let deserializer = Deserializer::new(p.clone(), Some(p.device()), seams.lookup());
            let cb = p.world().new_obj::<kind::CommandBuffer>(Model::None);
            let command_buffer = cb.id();
            let dec = RenderDecoder::new(Arc::downgrade(&deserializer), Some(cb));
            p.call_log().clear();
            Rig {
                p,
                seams,
                dec,
                command_buffer,
                deserializer,
            }
        }

        /// A decoder with an open encoder (a carried decoder).
        pub fn with_encoder() -> (Rig, Fk<kind::RenderEncoder>) {
            let mut r = Rig::new();
            let e = r.p.world().new_obj::<kind::RenderEncoder>(Model::None);
            r.dec.encoder = Some(e.clone_ref());
            (r, e)
        }

        /// A new object provided to the lookups under `reference`.
        pub fn provide<K: Kind>(&self, reference: u32) -> Fk<K> {
            let h = self.p.world().new_obj::<K>(Model::None);
            self.seams.provide(reference, &h);
            h
        }

        /// The ordered calls: lookups and the Metal calls whose position is observable.
        pub fn calls(&self) -> Vec<Call> {
            self.p.call_log().calls()
        }

        /// The ordered calls' selectors.
        pub fn sels(&self) -> Vec<&'static str> {
            self.p.call_log().sels()
        }

        /// Sets the device's capabilities: its defaults, changed by `f`.
        pub fn set_caps(&self, f: impl FnOnce(&mut DeviceCaps)) {
            set_caps(&self.p, f);
        }
    }

    /// A cursor over `bytes` (leaked host memory: the test's lifetime).
    pub fn cursor(bytes: &[u8]) -> Cursor {
        let mem: &'static mut [u8] = Box::leak(bytes.to_vec().into_boxed_slice());
        let len = mem.len() as u64;
        Cursor::with_pointer(Arc::new(()), HostPtr::from_ptr(mem.as_mut_ptr()), len)
    }

    /// A record: its header (the length counts the header) and `payload`.
    pub fn record(id: u32, payload: &[u8]) -> Vec<u8> {
        header(id, 8 + payload.len() as u32, payload)
    }

    /// A header with an explicit `length`, then `payload`.
    pub fn header(id: u32, length: u32, payload: &[u8]) -> Vec<u8> {
        let mut v = CmdHeader {
            command_id: id,
            length,
        }
        .as_bytes()
        .to_vec();
        v.extend_from_slice(payload);
        v
    }

    /// `t`'s bytes.
    pub fn bytes<T: IntoBytes + Immutable>(t: &T) -> Vec<u8> {
        t.as_bytes().to_vec()
    }

    /// A 0x1a record with every reference 0.
    pub fn describe_empty() -> Vec<u8> {
        record(0x1a, &[0u8; 0x248])
    }

    /// Asserts `r` raised `id` with `detail`.
    pub fn assert_raised<T: Debug>(r: RaiseResult<T>, id: impl Into<RaiseId>, detail: &str) {
        let e = r.expect_err("expected a raise");
        assert_eq!(e.id(), id.into(), "{e}");
        assert_eq!(e.detail(), Some(detail), "{e}");
    }

    /// Asserts `r` is the cursor's refused read (a short record).
    pub fn assert_failed_read<T: Debug>(r: RaiseResult<T>) {
        let e = r.expect_err("expected a raise");
        let detail = e.detail().unwrap_or_default();
        assert!(detail.starts_with("record not readable: "), "{e}");
        assert_eq!(
            e.id(),
            RaiseId::Memory(crate::memory::RaiseKind::CursorReadFailed),
            "{e}"
        );
    }
}

#[cfg(test)]
mod tests {

    use super::test_rig::*;
    use super::*;
    use crate::deserializer::RaiseKind as Shared;
    use crate::platform::Handle;
    use crate::raise::death::DeathKind;
    use crate::testkit::calllog::Arg;
    use crate::testkit::objects::{Model, kind};
    use crate::testkit::script::Step;

    fn segment(continues_previous: u8, continued_by_next: u8) -> SegmentHeader {
        SegmentHeader {
            length: 0,
            encoder_type: 0,
            continues_previous,
            continued_by_next,
            unknown07: 0,
        }
    }

    /// The decoder's encoder.
    fn encoder_id(r: &Rig) -> Option<crate::platform::ObjId> {
        r.dec.encoder.as_ref().map(|e| e.id())
    }

    #[test]
    fn a_new_decoder_holds_nothing_and_sends_nothing() {
        let r = Rig::new();
        assert!(r.dec.encoder.is_none());
        assert!(r.dec.pipeline.is_none());
        assert_eq!(r.dec.max_color_attachment_index, 0);
        assert_eq!(
            r.dec.command_buffer.as_ref().map(Handle::id),
            Some(r.command_buffer)
        );
        assert!(r.calls().is_empty(), "init sends nothing");
    }

    /// A fresh segment: the pass prefix, the encoder, the records, then `endEncoding` and the
    /// pipeline's release.
    #[test]
    fn a_fresh_segment_opens_encodes_and_ends() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        let pipeline = r.p.world().new_obj::<kind::RenderPipeline>(Model::None);
        r.dec.pipeline = Some(pipeline.clone_ref());
        let mut bytes = describe_empty();
        bytes.extend(record(0x85, &[]));
        // 7 trailing bytes: less than a header, so the loop ends.
        bytes.extend([0u8; 7]);
        r.dec.decode(hold, &segment(0, 0), cursor(&bytes)).unwrap();
        assert_eq!(
            r.sels(),
            vec![
                "renderCommandEncoderWithDescriptor:",
                "memoryBarrierWithScope:afterStages:beforeStages:",
                "endEncoding"
            ]
        );
        // The encoder is kept (ended), the pipeline released.
        assert!(r.dec.encoder.is_some());
        assert!(r.dec.pipeline.is_none());
        assert_eq!(pipeline.retain_count(), 1);
        let cb_calls: Vec<_> = r
            .calls()
            .into_iter()
            .filter(|c| c.obj == r.command_buffer)
            .collect();
        assert_eq!(cb_calls.len(), 1);
        assert_eq!(cb_calls[0].sel, "renderCommandEncoderWithDescriptor:");
    }

    /// A continuation reads no prefix; `continuedByNext` keeps the encoder open and the pipeline
    /// set.
    #[test]
    fn a_continuation_keeps_the_encoder_and_the_pipeline() {
        let hold = Hold::NONE;
        let (mut r, e) = Rig::with_encoder();
        let pipeline = r.p.world().new_obj::<kind::RenderPipeline>(Model::None);
        r.dec.pipeline = Some(pipeline.clone_ref());
        let bytes = record(0x85, &[]);
        r.dec.decode(hold, &segment(1, 1), cursor(&bytes)).unwrap();
        assert_eq!(
            r.sels(),
            vec!["memoryBarrierWithScope:afterStages:beforeStages:"]
        );
        assert_eq!(encoder_id(&r), Some(e.id()));
        assert!(r.dec.pipeline.is_some());
        // An empty continuation that ends: only endEncoding.
        r.p.call_log().clear();
        r.dec.decode(hold, &segment(1, 0), cursor(&[])).unwrap();
        assert_eq!(r.sels(), vec!["endEncoding"]);
        assert!(r.dec.pipeline.is_none());
    }

    /// No command buffer: the pass prefix is decoded, no encoder is asked for, and the raise
    /// follows.
    #[test]
    fn no_command_buffer_raises_after_the_prefix() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        r.dec.command_buffer = None;
        let res = r
            .dec
            .decode(hold, &segment(0, 0), cursor(&describe_empty()));
        assert_raised(res, RaiseKind::NoEncoder, "no render encoder");
        assert!(r.dec.encoder.is_none());
        assert!(!r.sels().contains(&"renderCommandEncoderWithDescriptor:"));
    }

    /// No encoder on a continuation: the raise, then the fault (no encoder: nothing sent), which
    /// releases the pipeline.
    #[test]
    fn a_continuation_without_an_encoder_raises() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        let pipeline = r.p.world().new_obj::<kind::RenderPipeline>(Model::None);
        r.dec.pipeline = Some(pipeline.clone_ref());
        let res = r
            .dec
            .decode(hold, &segment(1, 0), cursor(&record(0x85, &[])));
        assert_raised(res, RaiseKind::NoEncoder, "no render encoder");
        assert!(r.sels().is_empty());
        assert!(r.dec.pipeline.is_none(), "the fault releases the pipeline");
    }

    /// Metal answers no encoder: the same raise, after the prefix.
    #[test]
    fn a_nil_encoder_raises() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        r.p.script()
            .push("renderCommandEncoderWithDescriptor:", Step::Nil);
        let res = r
            .dec
            .decode(hold, &segment(0, 0), cursor(&describe_empty()));
        assert_raised(res, RaiseKind::NoEncoder, "no render encoder");
        assert!(r.dec.encoder.is_none());
    }

    /// A main-loop record under 8 bytes is an underflow; the fault ends the encoder and releases
    /// it.
    #[test]
    fn an_underflow_faults_the_decoder() {
        let hold = Hold::NONE;
        let (mut r, e) = Rig::with_encoder();
        let res = r
            .dec
            .decode(hold, &segment(1, 0), cursor(&header(0x85, 7, &[])));
        assert_raised(
            res,
            Shared::RecordUnderflow,
            "record length 7 is shorter than its header",
        );
        assert_eq!(r.sels(), vec!["endEncoding"]);
        assert!(r.dec.encoder.is_none());
        assert_eq!(e.retain_count(), 1, "the decoder's reference is released");
    }

    /// A record overrunning the segment fails its sub-cursor before dispatch.
    #[test]
    fn an_overrun_fails_before_dispatch() {
        let hold = Hold::NONE;
        let (mut r, _e) = Rig::with_encoder();
        let res = r
            .dec
            .decode(hold, &segment(1, 0), cursor(&header(0x85, 0x20, &[0; 4])));
        assert_raised(
            res,
            Shared::RecordOverrun,
            "record not readable: a sub-cursor of 0x18 bytes, with 0x4 left",
        );
        assert_eq!(r.sels(), vec!["endEncoding"]);
    }

    /// The pass prefix's own record and the ids outside the table are unrecognized in the main
    /// loop.
    #[test]
    fn unknown_ids_are_unrecognized() {
        for id in [0x1a, 0x1e, 0x24, 0x64, 0xa7, 0xffff_ffff] {
            let hold = Hold::NONE;
            let (mut r, _e) = Rig::with_encoder();
            let res = r.dec.decode(hold, &segment(1, 0), cursor(&record(id, &[])));
            assert_raised(
                res,
                Shared::UnrecognizedCommand,
                &format!("unknown command {id}"),
            );
            assert_eq!(r.sels(), vec!["endEncoding"], "id {id:#x}");
        }
    }

    /// The dispatch reaches `pass`'s handlers by id (0x1b, 0x85, 0x86, 0x87, 0x89, 0x9c).
    #[test]
    fn the_dispatch_reaches_the_pass_handlers() {
        let hold = Hold::NONE;
        let (mut r, _e) = Rig::with_encoder();
        let mut bytes = Vec::new();
        bytes.extend(record(0x1b, &[0, 0, 0, 0, 5, 0, 0, 0]));
        bytes.extend(record(0x85, &[]));
        bytes.extend(record(0x86, &[0, 0, 0, 0]));
        bytes.extend(record(0x87, &[0, 0, 0, 0, 1, 0, 0, 0]));
        bytes.extend(record(0x89, &[0, 0, 0, 0, 1, 0, 2, 0]));
        let mut tg = vec![0u8; 0x14];
        tg[0] = 16;
        bytes.extend(record(0x9c, &tg));
        r.dec.decode(hold, &segment(1, 1), cursor(&bytes)).unwrap();
        assert_eq!(
            r.sels(),
            vec![
                "useHeaps:count:stages:",
                "memoryBarrierWithScope:afterStages:beforeStages:",
                "useHeaps:count:",
                "useResources:count:usage:",
                "useResources:count:usage:stages:",
                "setThreadgroupMemoryLength:offset:atIndex:",
            ]
        );
    }

    /// 0x6a decodes as SetDepthStoreAction, as 0x69 does.
    #[test]
    fn id_0x6a_is_set_depth_store_action() {
        let hold = Hold::NONE;
        let (mut r, e) = Rig::with_encoder();
        let mut bytes = record(0x69, &3u64.to_le_bytes());
        bytes.extend(record(0x6a, &5u64.to_le_bytes()));
        r.dec.decode(hold, &segment(1, 1), cursor(&bytes)).unwrap();
        let calls: Vec<_> =
            r.p.call_log()
                .calls_on(e.id())
                .into_iter()
                .map(|c| (c.sel, c.args))
                .collect();
        assert_eq!(
            calls,
            vec![
                ("setDepthStoreAction:", vec![Arg::U64(3)]),
                ("setDepthStoreAction:", vec![Arg::U64(5)]),
            ]
        );
    }

    /// Every id of the table reaches a handler (an empty record fails its read there, or is
    /// decoded); every other id is unrecognized.
    #[test]
    fn the_dispatch_table_covers_every_id() {
        let known: Vec<u32> = (0x00..=0x19)
            .chain(0x1b..=0x1d)
            .chain(0x65..=0xa6)
            .collect();
        for id in 0u32..=0x1ff {
            {
                let hold = Hold::NONE;
                let (mut r, _e) = Rig::with_encoder();
                let res = r.dec.decode(hold, &segment(1, 1), cursor(&record(id, &[])));
                let unrecognized = matches!(
                    &res,
                    Err(e) if e.id() == RaiseId::from(Shared::UnrecognizedCommand)
                );
                assert_eq!(unrecognized, !known.contains(&id), "id {id:#x}: {res:?}");
            }
        }
    }

    /// A carried decoder opening a second encoder replaces the first without releasing it.
    #[test]
    fn a_second_encoder_leaks_the_first() {
        let hold = Hold::NONE;
        let (mut r, first) = Rig::with_encoder();
        assert_eq!(first.retain_count(), 2);
        r.dec
            .decode(hold, &segment(0, 1), cursor(&describe_empty()))
            .unwrap();
        assert_ne!(encoder_id(&r), Some(first.id()));
        assert_eq!(
            first.retain_count(),
            2,
            "the old reference is leaked, not released"
        );
        assert!(!r.sels().contains(&"endEncoding"));
    }

    /// A Metal raise at the segment's `endEncoding` faults the decoder (a second `endEncoding`)
    /// and goes on.
    #[test]
    fn a_raise_at_the_last_end_encoding_faults() {
        let hold = Hold::NONE;
        let (mut r, _e) = Rig::with_encoder();
        r.p.script().push(
            "endEncoding",
            Step::raise("NSInternalInconsistencyException", "x"),
        );
        let err = r.dec.decode(hold, &segment(1, 0), cursor(&[])).unwrap_err();
        assert_eq!(err.id(), RaiseId::Platform);
        assert_eq!(r.sels(), vec!["endEncoding", "endEncoding"]);
        assert!(r.dec.encoder.is_none());
    }

    /// Under the walk's handler, the raise goes on with its own site, after the fault ran.
    #[test]
    fn a_raise_goes_on_with_its_site_after_the_fault() {
        let (mut r, _e) = Rig::with_encoder();
        let log = r.p.call_log().clone();
        let raised = r
            .dec
            .decode(Hold::NONE, &segment(1, 0), cursor(&header(0x85, 3, &[])));
        assert_eq!(
            raised.map_err(|r| (r.site().kind, r.site().file)),
            Err((DeathKind::UnhandledRaise, "deserializer.record.underflow"))
        );
        assert_eq!(log.sels(), vec!["endEncoding"], "the fault ran first");
    }

    #[test]
    fn the_fault_ends_releases_and_repeats_as_nothing() {
        let (mut r, e) = Rig::with_encoder();
        let pipeline = r.p.world().new_obj::<kind::RenderPipeline>(Model::None);
        r.dec.pipeline = Some(pipeline.clone_ref());
        r.dec.fault().unwrap();
        assert_eq!(r.sels(), vec!["endEncoding"]);
        assert!(r.dec.encoder.is_none());
        assert!(r.dec.pipeline.is_none());
        assert_eq!(e.retain_count(), 1);
        assert_eq!(pipeline.retain_count(), 1);
        r.dec.fault().unwrap();
        assert_eq!(
            r.sels(),
            vec!["endEncoding"],
            "a second fault sends nothing"
        );
    }

    /// A raise from `endEncoding` in the fault stops it before the releases; the next fault ends
    /// and releases.
    #[test]
    fn a_raise_in_the_fault_stops_it_before_the_releases() {
        let (mut r, e) = Rig::with_encoder();
        let pipeline = r.p.world().new_obj::<kind::RenderPipeline>(Model::None);
        r.dec.pipeline = Some(pipeline.clone_ref());
        r.p.script().push(
            "endEncoding",
            Step::raise("NSInternalInconsistencyException", "x"),
        );
        assert!(r.dec.fault().is_err());
        assert!(r.dec.encoder.is_some());
        assert!(r.dec.pipeline.is_some());
        assert_eq!((e.retain_count(), pipeline.retain_count()), (2, 2));
        r.dec.fault().unwrap();
        assert_eq!(r.sels(), vec!["endEncoding", "endEncoding"]);
        assert!(r.dec.encoder.is_none() && r.dec.pipeline.is_none());
        assert_eq!((e.retain_count(), pipeline.retain_count()), (1, 1));
    }

    /// A raise from the fault's `endEncoding` replaces the raise in flight, and the fault's
    /// releases never run.
    #[test]
    fn a_raise_in_the_fault_replaces_the_one_in_flight() {
        let hold = Hold::NONE;
        let (mut r, e) = Rig::with_encoder();
        let pipeline = r.p.world().new_obj::<kind::RenderPipeline>(Model::None);
        r.dec.pipeline = Some(pipeline.clone_ref());
        r.p.script().push(
            "endEncoding",
            Step::raise("NSInternalInconsistencyException", "x"),
        );
        let err = r
            .dec
            .decode(hold, &segment(1, 0), cursor(&header(0x85, 7, &[])))
            .unwrap_err();
        assert_eq!(err.id(), RaiseId::Platform);
        assert_eq!(err.detail(), Some("x"));
        assert!(r.dec.encoder.is_some());
        assert!(r.dec.pipeline.is_some());
        assert_eq!((e.retain_count(), pipeline.retain_count()), (2, 2));
        assert_eq!(r.sels(), vec!["endEncoding"]);
    }

    #[test]
    fn drop_releases_the_encoder_then_the_pipeline() {
        let (mut r, e) = Rig::with_encoder();
        let pipeline = r.p.world().new_obj::<kind::RenderPipeline>(Model::None);
        r.dec.pipeline = Some(pipeline.clone_ref());
        assert_eq!((e.retain_count(), pipeline.retain_count()), (2, 2));
        let Rig { dec, p, .. } = r;
        drop(dec);
        assert_eq!((e.retain_count(), pipeline.retain_count()), (1, 1));
        assert!(
            p.call_log().sels().is_empty(),
            "the drop sends no endEncoding"
        );
    }

    /// The decoder holds the deserializer weakly: the task's release after the decode is the
    /// deserializer's last while the decoder still holds its open encoder, which goes with the
    /// decoder.
    #[test]
    fn the_deserializer_is_not_kept_while_the_decoder_waits() {
        let hold = Hold::NONE;
        let Rig {
            p,
            seams,
            mut dec,
            command_buffer: _,
            deserializer,
        } = Rig::new();
        let dev = p.device();
        let device_refs = dev.retain_count();
        dec.decode(hold, &segment(0, 1), cursor(&describe_empty()))
            .unwrap();
        assert_eq!(Arc::strong_count(&deserializer), 1);
        let enc = dec.encoder.as_ref().unwrap().clone_ref();
        let gone = Arc::downgrade(&deserializer);
        drop(deserializer);
        assert!(gone.upgrade().is_none());
        assert_eq!(dev.retain_count(), device_refs - 1);
        assert_eq!(Arc::strong_count(&seams), 1);
        assert!(p.world().live("MTLSharedEventListener").is_empty());
        // The decoder's reference and the probe.
        assert_eq!(enc.retain_count(), 2);
        drop(dec);
        assert_eq!(enc.retain_count(), 1);
        assert_eq!(dev.retain_count(), device_refs - 1);
    }
}
