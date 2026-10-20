//! The command-stream deserializer: one per task. It walks an exec's command buffers segment by
//! segment ([`walk`]), each segment through the decoder of its encoder type ([`decoder`]), and
//! resets rasterization rate maps from their serialized records ([`rate_map`]). The guest's
//! records are [`records`] and [`texture_descriptor`]; the info replies are [`info`]'s.
//!
//! # Contract
//!
//! - **The segment walk** takes the types 0 (render), 1 (compute), 2 (blit) and 4 (info), and
//!   the type-5 envelope that sets the command buffer's protection options and names the real
//!   header. A decoder may be carried from one segment into the next (`continued_by_next`), and a
//!   carried decoder is reused whatever the next segment's type says
//!   (`walk::tests::a_carried_decoder_is_reused_whatever_the_type`).
//! - **Two levels of catch.** A decoder that raises faults itself (ends its encoder) and raises
//!   on; the walk then faults the decoder carried into the failing segment, and a raise from that
//!   fault replaces the one in flight
//!   (`walk::tests::a_decoder_raise_faults_it_then_the_carried_one`,
//!   `walk::tests::a_raise_in_the_carried_decoders_fault_replaces_the_one_in_flight`).
//! - **Decoders outlive the walk**: the walk hands the decoders it made to its caller
//!   ([`walk::Decoders`]), which holds them past its commit, so an encoder left open stays open
//!   through it. A decoder holds the deserializer weakly, so the task's release stays the
//!   deserializer's last
//!   (`walk::tests::teardown::the_tasks_release_is_the_last_while_the_caller_holds_the_decoders`).
//! - **Info replies land at decode time**: each is written into guest memory when its record is
//!   decoded, and a later failure takes none back
//!   (`info::tests::decode::replies_before_a_failure_stay_written`).
//! - **Nullable and required lookups**: the delegate's getters answer `None`; the `require_*`
//!   getters here raise [`RaiseKind::MissingObject`] instead.
//! - **The rate-map reset** raises in the caller's frame: the in-process task catches it, the
//!   GPU service has no handler and dies
//!   (`rate_map::tests::a_raise_with_no_handler_is_the_service_death`).
//! - Every record keeps its guest size, padding included ([`records`]), and
//!   [`DESERIALIZER_VERSION`] is 8.
//! - The deserializer makes one `MTLSharedEventListener` at its creation and never uses it.

// Guest data flows through here, so arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::fmt;
use std::sync::{Arc, Weak};

use crate::platform::Platform;
use crate::raise::death::DeathKind as DeathClass;
use crate::raise::{Raise, RaiseResult};
use crate::rt::Hold;
use crate::seams::links::ObjectLookupRef;

pub mod blit;
pub mod compute;
pub mod decoder;
pub mod info;
pub mod rate_map;
pub mod records;
pub mod render;
pub mod texture_descriptor;
pub mod walk;

/// The deserializer's version, as the device info reports it (key 10).
pub const DESERIALIZER_VERSION: u32 = 8;

/// What a required lookup names when it finds nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    /// A compute pipeline state.
    ComputePipeline,
    /// A render pipeline state.
    RenderPipeline,
    /// A buffer.
    Buffer,
    /// A texture.
    Texture,
    /// A sampler state.
    SamplerState,
    /// A depth-stencil state.
    DepthStencilState,
    /// A fence.
    Fence,
    /// A heap.
    Heap,
    /// A rasterization rate map.
    RateMap,
    /// An indirect command buffer.
    Icb,
}

impl ObjectKind {
    /// The kind as the raise's detail names it.
    fn label(self) -> &'static str {
        match self {
            ObjectKind::ComputePipeline => "compute pipeline state",
            ObjectKind::RenderPipeline => "render pipeline state",
            ObjectKind::Buffer => "buffer",
            ObjectKind::Texture => "texture",
            ObjectKind::SamplerState => "sampler state",
            ObjectKind::DepthStencilState => "depth-stencil state",
            ObjectKind::Fence => "fence",
            ObjectKind::Heap => "heap",
            ObjectKind::RateMap => "rasterization rate map",
            ObjectKind::Icb => "indirect command buffer",
        }
    }
}

/// The deserializer's raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// A segment header's length is past its range, or below the header itself.
    SegmentLength,
    /// A continuation segment of another type than the carried decoder's (no carried decoder
    /// counts as type 0).
    ContinuationType,
    /// A type-0 continuation with no decoder carried into it.
    ContinuationMissing,
    /// A segment of no decoder's type.
    EncoderType,
    /// A required lookup found nothing.
    MissingObject(ObjectKind),
    /// A record's length is below its 8-byte header.
    RecordUnderflow,
    /// A record overruns its segment.
    RecordOverrun,
    /// A command id the decoder does not know.
    UnrecognizedCommand,
    /// The rate-map reset's record is shorter than its 8-byte header.
    RateMapRecordShort,
    /// The rate map to reset is not mutable (or absent).
    RateMapImmutable,
    /// Metal made no layer descriptor for a layer's sample counts.
    RateMapLayer,
    /// A layer's samples could not be read into Metal's storage.
    RateMapSamples,
    /// An info query's reply buffer has no cursor.
    ReplyCursor,
    /// An info reply does not fit its reply buffer.
    ReplyWrite,
    /// The parameter buffer of a rate-map copy is not shared.
    BufferStorageMode,
    /// The parameter buffer of a rate-map copy is too short for the offset and size.
    BufferLength,
    /// The parameter buffer offset of a rate-map copy is not aligned.
    BufferAlignment,
    /// A coordinate of a multiple mapping could not be read.
    CoordinateRead,
    /// A unique-identifier query names an object type that has none.
    UniqueIdentifierType,
    /// The blit decoder's checks.
    Blit(blit::BlitRaise),
    /// The compute decoder's checks.
    Compute(compute::ComputeRaise),
    /// The render decoder's.
    Render(render::RaiseKind),
    /// A scripted decoder's raise, by its dotted id (the walk's unit tests).
    #[cfg(test)]
    Mock(&'static str),
}

impl RaiseKind {
    /// The dotted id.
    pub fn name(self) -> &'static str {
        match self {
            RaiseKind::SegmentLength => "deserializer.segment.length",
            RaiseKind::ContinuationType => "deserializer.segment.continuation_type",
            RaiseKind::ContinuationMissing => "deserializer.segment.continuation_missing",
            RaiseKind::EncoderType => "deserializer.segment.encoder_type",
            RaiseKind::MissingObject(kind) => match kind {
                ObjectKind::ComputePipeline => "deserializer.lookup.compute_pipeline",
                ObjectKind::RenderPipeline => "deserializer.lookup.render_pipeline",
                ObjectKind::Buffer => "deserializer.lookup.buffer",
                ObjectKind::Texture => "deserializer.lookup.texture",
                ObjectKind::SamplerState => "deserializer.lookup.sampler_state",
                ObjectKind::DepthStencilState => "deserializer.lookup.depth_stencil_state",
                ObjectKind::Fence => "deserializer.lookup.fence",
                ObjectKind::Heap => "deserializer.lookup.heap",
                ObjectKind::RateMap => "deserializer.lookup.rate_map",
                ObjectKind::Icb => "deserializer.lookup.icb",
            },
            RaiseKind::RecordUnderflow => "deserializer.record.underflow",
            RaiseKind::RecordOverrun => "deserializer.record.overrun",
            RaiseKind::UnrecognizedCommand => "deserializer.record.unrecognized",
            RaiseKind::RateMapRecordShort => "deserializer.rate_map.record_short",
            RaiseKind::RateMapImmutable => "deserializer.rate_map.immutable",
            RaiseKind::RateMapLayer => "deserializer.rate_map.layer",
            RaiseKind::RateMapSamples => "deserializer.rate_map.samples",
            RaiseKind::ReplyCursor => "deserializer.info.reply_cursor",
            RaiseKind::ReplyWrite => "deserializer.info.reply_write",
            RaiseKind::BufferStorageMode => "deserializer.info.buffer_storage_mode",
            RaiseKind::BufferLength => "deserializer.info.buffer_length",
            RaiseKind::BufferAlignment => "deserializer.info.buffer_alignment",
            RaiseKind::CoordinateRead => "deserializer.info.coordinate_read",
            RaiseKind::UniqueIdentifierType => "deserializer.info.unique_identifier_type",
            RaiseKind::Blit(kind) => kind.name(),
            RaiseKind::Compute(kind) => kind.name(),
            RaiseKind::Render(k) => k.name(),
            #[cfg(test)]
            RaiseKind::Mock(name) => name,
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The deserializer's deaths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// An element of the segment list has no cursor: a null dereference.
    NullSegment,
    /// Metal's sample storage for a layer is a near-null address.
    SampleStorage,
    /// The rate-map descriptor built for a reset has another layer count than its record's.
    RateMapLayerCount,
    /// A decoder reached its deserializer after the deserializer was released.
    DeserializerGone,
    /// The delegate answered no Metal descriptor for a heap texture's size query.
    NilTextureDescriptor,
}

impl DeathKind {
    /// The dotted id.
    pub fn name(self) -> &'static str {
        match self {
            DeathKind::NullSegment => "deserializer.segment.null",
            DeathKind::SampleStorage => "deserializer.rate_map.sample_storage",
            DeathKind::RateMapLayerCount => "deserializer.rate_map.layer_count",
            DeathKind::DeserializerGone => "deserializer.decoder.deserializer_gone",
            DeathKind::NilTextureDescriptor => "deserializer.info.nil_texture_descriptor",
        }
    }

    /// How the process dies there.
    pub fn class(self) -> DeathClass {
        match self {
            DeathKind::NullSegment | DeathKind::SampleStorage | DeathKind::DeserializerGone => {
                DeathClass::NullDeref
            }
            DeathKind::RateMapLayerCount | DeathKind::NilTextureDescriptor => DeathClass::Assert,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A required lookup's answer: `found`, or a raise naming the kind and the reference.
fn required<T>(found: Option<T>, kind: ObjectKind, reference: u32) -> RaiseResult<T> {
    match found {
        Some(object) => Ok(object),
        None => Err(
            Raise::new(RaiseKind::MissingObject(kind)).with_detail(format_args!(
                "no {} for reference {reference}",
                kind.label()
            )),
        ),
    }
}

/// The deserializer of one task: its device, the delegate it looks objects up through, and the
/// platform its decoders make Metal descriptors with.
///
/// Its fields drop in declaration order: the device, the delegate, then the listener.
pub struct Deserializer<P: Platform> {
    /// The task's device, retained; `None` for a task made without one, whose sends answer zero.
    device: Option<P::Device>,
    /// The resource manager's lookup.
    object_delegate: ObjectLookupRef<P>,
    /// Made at creation and never used.
    #[allow(dead_code)] // made at creation and never read (the module's contract)
    event_listener: P::SharedEventListener,
    /// The platform (Metal descriptors, the decoders' `new` calls).
    platform: Arc<P>,
    /// This deserializer, for the decoders it makes: they hold it weakly, so a decoder the
    /// caller still holds keeps it from nothing.
    weak_self: Weak<Deserializer<P>>,
}

impl<P: Platform> Deserializer<P> {
    /// A deserializer over `device` (already retained; `None` is accepted) and the manager's
    /// lookup. Makes the unused listener.
    pub fn new(
        platform: Arc<P>,
        device: Option<P::Device>,
        object_delegate: ObjectLookupRef<P>,
    ) -> Arc<Self> {
        let event_listener = platform.new_shared_event_listener();
        Arc::new_cyclic(|weak_self| Deserializer {
            device,
            object_delegate,
            event_listener,
            platform,
            weak_self: weak_self.clone(),
        })
    }

    /// The task's device, if it has one.
    pub fn device(&self) -> Option<&P::Device> {
        self.device.as_ref()
    }

    /// The platform the deserializer was made with.
    pub fn platform(&self) -> &Arc<P> {
        &self.platform
    }

    /// The lookup the decoders go through: its getters are the nullable lookups.
    pub fn object_delegate(&self) -> &ObjectLookupRef<P> {
        &self.object_delegate
    }

    /// This deserializer, weakly, for a decoder to hold.
    pub(crate) fn weak(&self) -> Weak<Deserializer<P>> {
        Weak::clone(&self.weak_self)
    }

    /// The compute pipeline state `reference` names, or a raise.
    pub fn require_compute_pipeline_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<P::ComputePipeline> {
        let t_lookup = std::time::Instant::now();
        let found = self
            .object_delegate
            .get_compute_pipeline_state(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::ComputePipeline, reference)
    }

    /// The render pipeline state `reference` names, or a raise.
    pub fn require_render_pipeline_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<P::RenderPipeline> {
        let t_lookup = std::time::Instant::now();
        let found = self
            .object_delegate
            .get_render_pipeline_state(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::RenderPipeline, reference)
    }

    /// The buffer `reference` names, or a raise.
    pub fn require_buffer(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<P::Buffer> {
        let t_lookup = std::time::Instant::now();
        let found = self.object_delegate.get_buffer(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::Buffer, reference)
    }

    /// The texture `reference` names, or a raise.
    pub fn require_texture(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<P::Texture> {
        let t_lookup = std::time::Instant::now();
        let found = self.object_delegate.get_texture(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::Texture, reference)
    }

    /// The sampler state `reference` names, or a raise.
    pub fn require_sampler_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<P::SamplerState> {
        let t_lookup = std::time::Instant::now();
        let found = self.object_delegate.get_sampler_state(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::SamplerState, reference)
    }

    /// The depth-stencil state `reference` names, or a raise.
    pub fn require_depth_stencil_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<P::DepthStencilState> {
        let t_lookup = std::time::Instant::now();
        let found = self
            .object_delegate
            .get_depth_stencil_state(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::DepthStencilState, reference)
    }

    /// The fence `reference` names, or a raise.
    pub fn require_fence(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<P::Fence> {
        let t_lookup = std::time::Instant::now();
        let found = self.object_delegate.get_fence(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::Fence, reference)
    }

    /// The heap `reference` names, or a raise.
    pub fn require_heap(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<P::Heap> {
        let t_lookup = std::time::Instant::now();
        let found = self.object_delegate.get_heap(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::Heap, reference)
    }

    /// The rasterization rate map `reference` names, or a raise.
    pub fn require_rasterization_rate_map(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<P::RateMap> {
        let t_lookup = std::time::Instant::now();
        let found = self
            .object_delegate
            .get_rasterization_rate_map(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::RateMap, reference)
    }

    /// The indirect command buffer `reference` names, or a raise.
    pub fn require_indirect_command_buffer(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<P::Icb> {
        let t_lookup = std::time::Instant::now();
        let found = self
            .object_delegate
            .get_indirect_command_buffer(hold, reference)?;
        crate::perfstats::DECODE_LOOKUP.add(t_lookup.elapsed());
        required(found, ObjectKind::Icb, reference)
    }
}

/// The decoders' reach to their deserializer: upgrades the weak reference for one call. A
/// decoder runs only inside the walk, whose receiver the caller holds strongly across the call,
/// so the upgrade always succeeds and its reference is never the last.
pub(crate) fn upgrade<P: Platform>(deserializer: &Weak<Deserializer<P>>) -> Arc<Deserializer<P>> {
    match deserializer.upgrade() {
        Some(deserializer) => deserializer,
        None => crate::fatal!(DeathKind::DeserializerGone),
    }
}

#[cfg(test)]
pub(crate) mod test_lookups;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::platform::Handle;

    use crate::deserializer::test_lookups::Lookups;
    use crate::testkit::FakePlatform;

    type P = FakePlatform;

    fn setup() -> (Arc<P>, Arc<Lookups>, Arc<Deserializer<P>>) {
        let p = Arc::new(FakePlatform::new());
        let r = Arc::new(Lookups::new(&p));
        let d = Deserializer::new(p.clone(), Some(p.device()), r.lookup());
        p.call_log().clear();
        (p, r, d)
    }

    #[test]
    fn makes_an_unused_listener() {
        let p = Arc::new(FakePlatform::new());
        let r = Arc::new(Lookups::new(&p));
        let dev = p.device();
        let dev_id = dev.id();
        p.call_log().clear();
        let d = Deserializer::new(p.clone(), Some(dev), r.lookup());
        let calls = p.call_log().calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].class, "MTLSharedEventListener");
        assert_eq!(calls[0].sel, "new");
        assert_eq!(d.device().map(|d| d.id()), Some(dev_id));
        assert!(Arc::ptr_eq(d.platform(), &p));
        // A second deserializer: another listener.
        let _d2 = Deserializer::new(p.clone(), Some(p.device()), r.lookup());
        let listeners = p.call_log().calls_of("MTLSharedEventListener");
        assert_eq!(listeners.len(), 2);
    }

    #[test]
    fn holds_and_releases_its_delegate() {
        let (_p, r, d) = setup();
        let lookup = r.lookup();
        assert!(Arc::ptr_eq(d.object_delegate(), &lookup));
        let held = Arc::strong_count(&lookup);
        drop(d);
        assert_eq!(Arc::strong_count(&lookup), held - 1);
    }

    #[test]
    fn version_is_eight() {
        assert_eq!(DESERIALIZER_VERSION, 8);
    }
}
