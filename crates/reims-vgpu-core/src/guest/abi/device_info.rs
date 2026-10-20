//! The device info pairs a guest and its host exchange: [`write_device_info`] writes
//! the host's, [`check_capabilities`] checks a guest's against them, and [`get_device_info`] is
//! the guest's request for them (opcode 58).
//!
//! What the writer reads from the device is a [`DeviceInfoSnapshot`] (one call reads it once;
//! nothing writes those fields during a call) and the `MTLDevice`'s [`DeviceCaps`], read once
//! for the device. A query of the caps that raised raises again where the writer uses its
//! answer, in a fixed order and only where a pair needs it: a pair's value is computed only once
//! the pair fits (`PairWriter::put`, test `raises_where_asked`).
//!
//! What the guest sees: the pairs' order (keys 33 and 36 before 29, 41 before 35, 40 last before
//! the terminator), each key's gate and value, and the checker's order (every defined key, then
//! every value, then the flags and alignments). A key no reader takes (43, or 45 and up) dies in
//! the reader (`guest_key_43_dies`); a resume refuses a blob carrying one before it checks
//! (`suspend_blob`), so no guest reaches that death.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use crate::deserializer::DESERIALIZER_VERSION;
use crate::guest::GuestDeath;
use crate::guest::abi::suspend_blob::KeyValuePair;
use crate::guest::records::{APV_PAGE_SIZE, GuestFeatures};
use crate::num::{ToUsize, Truncate};
use crate::platform::caps::{self, DeviceCaps};
use crate::raise::{Kept, OnFault, RaiseResult, keep_forever};

/// `MTLGPUFamilyApple5`.
const MTL_GPU_FAMILY_APPLE5: i64 = 1005;
/// `MTLGPUFamilyApple6`.
const MTL_GPU_FAMILY_APPLE6: i64 = 1006;
/// `MTLGPUFamilyApple7`.
const MTL_GPU_FAMILY_APPLE7: i64 = 1007;
/// `MTLGPUFamilyApple8`.
const MTL_GPU_FAMILY_APPLE8: i64 = 1008;
/// `MTLGPUFamilyApple9`.
const MTL_GPU_FAMILY_APPLE9: i64 = 1009;
/// `MTLGPUFamilyApple10`.
const MTL_GPU_FAMILY_APPLE10: i64 = 1010;
/// The family the writer asks about that no `MTLGPUFamily` of the SDK is.
const MTL_GPU_FAMILY_MINUS_1009: i64 = -1009;

// The caps' answers are read by position: the queries they asked, in their order.
const _: () = assert!(matches!(caps::SAMPLE_COUNTS, [1, 2, 4, 8]));
const _: () = assert!(caps::PRIMITIVE_TYPE_COUNT == 10);
const _: () = assert!(matches!(
    caps::FAMILIES,
    [
        MTL_GPU_FAMILY_APPLE5,
        MTL_GPU_FAMILY_APPLE6,
        MTL_GPU_FAMILY_APPLE7,
        MTL_GPU_FAMILY_APPLE8,
        MTL_GPU_FAMILY_APPLE9,
        MTL_GPU_FAMILY_APPLE10,
        MTL_GPU_FAMILY_MINUS_1009
    ]
));
const _: () = assert!(matches!(caps::WRITE_ROUNDING_MODES, [1, 2]));

/// Whether the guest is told of the host's Apple GPU families. `REIMS_VGPU_APPLE_FAMILIES=0` says
/// none, as an Intel Mac's GPU does: an x86 guest told of Apple families takes the device for
/// Apple silicon, and RenderBox's binary archive then aborts in Metal (`+[MTLLoader
/// sliceIDForDevice:…]`, no slice for a paravirtual x86 device), crashing iconservicesagent,
/// Spotlight and the Dock's extras. Read once.
fn apple_families_reported() -> bool {
    static REPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REPORTED.get_or_init(|| !std::env::var("REIMS_VGPU_APPLE_FAMILIES").is_ok_and(|v| v == "0"))
}

/// A query's answer as the caps kept it: its value, or its raise thrown again here, where the
/// writer asks it, as `on` says.
fn asked<T: Clone>(on: OnFault, kept: &RaiseResult<T>) -> RaiseResult<T> {
    kept.clone().or_else(|r| on.throw(r))
}

/// The pairs [`check_capabilities`] asks of the host's own list.
pub const HOST_INFO_PAIRS: u32 = 46;
/// The key bound of the host's own list in [`check_capabilities`].
pub const HOST_INFO_MAX_KEY: u32 = 45;

/// One past the highest device info key (44); key 43 has no reader case.
const KEY_SLOTS: usize = 45;

/// What [`write_device_info`] reads from the device (a snapshot, so the writer is pure over it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceInfoSnapshot {
    /// The device's live features.
    pub features: GuestFeatures,
    /// The device's feature level (0: every feature).
    pub device_feature_level: u64,
    /// Whether the device supports argument buffers.
    pub supports_argument_buffers: bool,
    /// The max key the guest last asked device info for. Not read by the writer or the checker
    /// (the caller passes the max key it wants); carried for the suspend serializer's caller.
    pub guest_device_info_max_key: i32,
    /// Whether the device makes buffers from IOSurfaces.
    pub supports_buffer_with_iosurface: bool,
    /// Whether the device supports shared textures.
    pub supports_shared_textures: bool,
    /// Whether the device supports large user tasks.
    pub supports_large_user_tasks: bool,
    /// Whether the device supports large kernel tasks.
    pub supports_large_kernel_tasks: bool,
    /// Whether the device supports range buffers.
    pub supports_range_buffer: bool,
    /// Whether the device supports a shared memory heap.
    pub supports_shared_memory_heap: bool,
}

/// A device that states no level gets every feature.
fn feature_level_above(feature_level: u64, level: u64) -> bool {
    feature_level == 0 || feature_level > level
}

/// The writer's state: the count, the bounds and the destination.
struct PairWriter<F: FnMut(u32, KeyValuePair)> {
    max_key: u32,
    max_count: u32,
    count: u32,
    emit: F,
}

impl<F: FnMut(u32, KeyValuePair)> PairWriter<F> {
    fn fits(&self, key: u32) -> bool {
        key < self.max_key && self.count < self.max_count
    }

    /// One pair: `value` runs only when the pair fits, and its result is stored as a `u32`.
    fn put(&mut self, key: u32, value: impl FnOnce() -> RaiseResult<u32>) -> RaiseResult<()> {
        if self.fits(key) {
            let value = value()?;
            (self.emit)(self.count, KeyValuePair { key, value });
            self.count = self.count.wrapping_add(1);
        }
        Ok(())
    }
}

/// Writes the host's pairs for a guest that knows keys below `max_key`, at most `max_count` of them
/// including the terminator: `emit(i, pair)` stores pair `i`, in the fixed order. A pair that does
/// not fit is skipped and later pairs are still tried; the terminator `{0, 0}` is written iff
/// `max_key ≥ 1` and there is room. A raise the caps kept is thrown where its answer is asked
/// (none is caught here), after the pairs already written, as `on` says: the guest's request
/// asks under a FIFO command's handler, and a suspend and a resume with none.
pub fn write_device_info(
    on: OnFault,
    caps: &DeviceCaps,
    snapshot: &DeviceInfoSnapshot,
    max_key: u32,
    max_count: u32,
    emit: impl FnMut(u32, KeyValuePair),
) -> RaiseResult<()> {
    let features = &snapshot.features;
    let limits = &caps.limits;
    let flags = &caps.flags;
    // Computed once at entry.
    let level1 = feature_level_above(snapshot.device_feature_level, 1);
    let level2 = feature_level_above(snapshot.device_feature_level, 2);
    let level3 = feature_level_above(snapshot.device_feature_level, 3);
    // Every count 1, 2, 4, 8 is asked; the largest yes wins.
    let mut sample_count = 1u32;
    for (count, kept) in caps::SAMPLE_COUNTS.iter().zip(&caps.sample_counts) {
        if asked(on, kept)? {
            sample_count = count.low_u32();
        }
    }
    let mut w = PairWriter {
        max_key,
        max_count,
        count: 0,
        emit,
    };
    w.put(1, || Ok(sample_count))?;
    w.put(2, || {
        Ok(u32::from(flags.is_depth24_stencil8_pixel_format_supported))
    })?;
    let max_threads = caps.max_threads_per_threadgroup;
    w.put(3, || Ok(max_threads.width.low_u32()))?;
    w.put(4, || Ok(max_threads.height.low_u32()))?;
    w.put(5, || Ok(max_threads.depth.low_u32()))?;
    w.put(6, || Ok(limits.max_threadgroup_memory_length.low_u32()))?;
    w.put(7, || Ok(u32::from(flags.is_framebuffer_read_supported)))?;
    w.put(8, || Ok(u32::from(flags.is_rgb10a2_gamma_supported)))?;
    w.put(9, || Ok(u32::from(flags.supports_native_hardware_fp16)))?;
    w.put(10, || Ok(DESERIALIZER_VERSION))?;
    // Asked only when key 11 fits.
    w.put(11, || {
        let mut primitive_types = 0u32;
        for (t, kept) in (0u32..).zip(&caps.primitive_types) {
            if asked(on, kept)? {
                primitive_types |= 1u32.wrapping_shl(t);
            }
        }
        Ok(primitive_types)
    })?;
    w.put(12, || Ok(u32::from(features.dual_plane_textures)))?;
    w.put(13, || Ok(limits.linear_texture_alignment_bytes.low_u32()))?;
    if features.metal_heaps {
        w.put(14, || Ok(1))?;
        // Asked whether or not key 15 fits.
        let size_and_align = caps.heap_buffer_page_private(on)?;
        w.put(15, || Ok(size_and_align.align.low_u32()))?;
        w.put(16, || Ok(1))?;
    }
    if snapshot.supports_buffer_with_iosurface && features.buffer_from_io_surface {
        w.put(17, || Ok(1))?;
    }
    w.put(18, || {
        Ok(if level2 {
            0x20007
        } else if level1 {
            0x20006
        } else {
            0x20002
        })
    })?;
    let [apple5, apple6, apple7, apple8, apple9, apple10, minus_1009] = &caps.families;
    if level1 {
        if snapshot.supports_shared_textures && features.supports_shared_textures {
            w.put(19, || Ok(1))?;
        }
        w.put(20, || Ok(caps.max_vertex_amplification_count.low_u32()))?;
        if features.supports_programmable_sample_positions {
            w.put(21, || {
                Ok(u32::from(flags.supports_programmable_sample_positions))
            })?;
        }
        // The pair only for a nonzero count.
        let rate_layers = limits.max_rasterization_rate_layer_count;
        if rate_layers != 0 {
            w.put(22, || Ok(rate_layers.low_u32()))?;
        }
        if features.supports_tile_shaders_image_blocks {
            w.put(23, || {
                Ok(u32::from(
                    flags.supports_tile_shaders && flags.supports_non_square_tile_shaders,
                ))
            })?;
            w.put(24, || Ok(u32::from(flags.supports_image_blocks)))?;
            w.put(25, || {
                Ok(u32::from(flags.are_raster_order_groups_supported))
            })?;
            w.put(26, || Ok(u32::from(flags.supports_memory_order_atomics)))?;
            w.put(27, || Ok(u32::from(flags.is_large_mrt_supported)))?;
        }
        // The device's answer counts only when the feature is on.
        let dynamic_attribute_stride = if features.supports_dynamic_attribute_stride
            && flags.supports_dynamic_attribute_stride
        {
            2
        } else {
            0
        };
        let apple5_supported = asked(on, apple5)? && apple_families_reported();
        let multisample_array = flags.supports_texture_2d_multisample_array;
        w.put(28, || {
            Ok(dynamic_attribute_stride
                | u32::from(apple5_supported)
                | if multisample_array { 4 } else { 0 })
        })?;
    }
    if level2 {
        let bit = |b: bool, v: u32| if b { v } else { 0 };
        w.put(33, || {
            Ok(u32::from(snapshot.supports_large_user_tasks)
                | bit(snapshot.supports_large_kernel_tasks, 2)
                | bit(flags.supports_command_buffer_jump, 4)
                | bit(snapshot.supports_range_buffer, 8)
                | bit(snapshot.supports_shared_memory_heap, 0x10)
                | bit(snapshot.supports_argument_buffers, 0x20)
                | bit(flags.supports_simd_reduction, 0x40)
                | bit(flags.supports_float16_bcubic_filtering, 0x80)
                | bit(flags.supports_simd_shuffle_and_fill, 0x100)
                | bit(flags.supports_conditional_load_store, 0x200)
                | bit(flags.supports_compute_compressed_texture_write, 0x400)
                | bit(snapshot.supports_shared_textures, 0x800))
        })?;
        if features.supports_command_buffer_jump {
            w.put(36, || Ok(limits.max_predicated_nesting_depth.low_u32()))?;
        }
        w.put(29, || {
            Ok(limits.max_total_compute_threads_per_threadgroup.low_u32())
        })?;
        w.put(30, || Ok(limits.max_compute_local_memory_sizes.low_u32()))?;
        w.put(31, || Ok(limits.max_compute_threadgroup_memory.low_u32()))?;
        w.put(32, || {
            Ok(limits
                .max_compute_threadgroup_memory_alignment_bytes
                .low_u32())
        })?;
        // None for a device that does not respond to `gpuCoreCount`: no pair.
        if w.fits(34)
            && let Some(cores) = caps.gpu_core_count
        {
            w.put(34, || Ok(cores.low_u32()))?;
        }
        // Both asked always.
        let [mode1, mode2] = &caps.write_rounding_modes;
        let rounding_mode1 = asked(on, mode1)?;
        let rounding_mode2 = asked(on, mode2)?;
        let mut rounding_modes = 1u32;
        if rounding_mode1 {
            rounding_modes |= 2;
        }
        if rounding_mode2 {
            rounding_modes |= 4;
        }
        w.put(41, || Ok(rounding_modes))?;
        w.put(35, || Ok(limits.max_texture_layers.low_u32()))?;
        // The first family, highest first, that Metal supports.
        w.put(37, || {
            for (family, kept) in [
                (MTL_GPU_FAMILY_APPLE9, apple9),
                (MTL_GPU_FAMILY_APPLE8, apple8),
                (MTL_GPU_FAMILY_APPLE7, apple7),
                (MTL_GPU_FAMILY_APPLE6, apple6),
            ] {
                if asked(on, kept)? && apple_families_reported() {
                    return Ok(family.cast_unsigned().low_u32());
                }
            }
            Ok(MTL_GPU_FAMILY_APPLE5.cast_unsigned().low_u32())
        })?;
    }
    if level3 {
        // Every family asked, left to right.
        let mut families = 0u32;
        for (kept, bit) in [
            (apple5, 1u32),
            (apple6, 2),
            (apple7, 4),
            (apple8, 8),
            (apple9, 0x10),
            (minus_1009, 0x20),
            (apple10, 0x40),
        ] {
            // -1009 is no Apple family: it is told either way.
            if asked(on, kept)? && (apple_families_reported() || bit == 0x20) {
                families |= bit;
            }
        }
        // 0 or 3.
        let argument_buffers = snapshot.supports_argument_buffers;
        w.put(42, || {
            Ok(u32::from(argument_buffers) | if argument_buffers { 2 } else { 0 })
        })?;
        w.put(44, || Ok(families))?;
    }
    w.put(40, || {
        Ok(limits.device_linear_texture_alignment_bytes.low_u32())
    })?;
    w.put(0, || Ok(0))?;
    Ok(())
}

/// A list read back: each key's value and whether the list defined it (by key; 43 never is).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeviceInfo {
    values: [u32; KEY_SLOTS],
    present: [bool; KEY_SLOTS],
}

impl DeviceInfo {
    fn value(&self, key: u32) -> u32 {
        self.values.get(key.to_usize()).copied().unwrap_or(0)
    }
    fn present(&self, key: u32) -> bool {
        self.present.get(key.to_usize()).copied().unwrap_or(false)
    }
}

/// Reads a list up to its terminator (key 0) or the end of `pairs`. A key no reader takes (43, or
/// 45 and above) dies. Later duplicates win. The shader-version halves are read by nothing, so they
/// are not kept.
fn read_device_info(pairs: &[KeyValuePair]) -> DeviceInfo {
    let mut info = DeviceInfo {
        values: [0; KEY_SLOTS],
        present: [false; KEY_SLOTS],
    };
    for pair in pairs {
        // Key 0 ends the list.
        if pair.key == 0 {
            break;
        }
        let slot = pair.key.to_usize();
        match (
            pair.key != 43,
            info.values.get_mut(slot),
            info.present.get_mut(slot),
        ) {
            // Keys 1-42 and 44.
            (true, Some(v), Some(p)) => {
                *v = pair.value;
                *p = true;
            }
            // Key 43 (valid in a blob, below 45) has no reader and dies.
            _ => crate::fatal!(GuestDeath::UnknownInfoKey),
        }
    }
    info
}

/// The keys [`check_capabilities`] checks on their own terms, not by value.
const SPECIAL_KEYS: [u32; 9] = [11, 13, 15, 28, 32, 33, 40, 41, 42];
/// The key a host may leave undefined.
const OPTIONAL_KEYS: [u32; 1] = [40];

/// The keys whose definition is checked, in the checker's order.
const DEFINED_KEYS: [u32; 43] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44,
];
/// The keys whose value is checked, in the checker's order (28, 33, 42 and 44 are not listed).
const VALUE_KEYS: [u32; 39] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41,
];

/// The name each key's check logs, by key; 0 and 43 have none.
const KEY_NAMES: [&str; 45] = [
    "",
    "MSAASamples",
    "D24S8Supported",
    "MaxThreadsPerThreadgroupW",
    "MaxThreadsPerThreadgroupH",
    "MaxThreadsPerThreadgroupD",
    "MaxThreadgroupMemoryLength",
    "IsFramebufferReadSupported",
    "IsRGB10A2GammaSupported",
    "SupportsNativeHardwareFP16",
    "DeserializerVersion",
    "PrimitiveTypeSupport",
    "SupportsMultiplaneTextures",
    "LinearTextureAlignment",
    "HeapBuffers",
    "HeapBufferAlignment",
    "HeapTextures",
    "BufferFromIOSurface",
    "MaxMetalShaderVersion",
    "SupportsSharedTextures",
    "MaxVertexAmplificationCount",
    "SupportsProgrammableSamplePositions",
    "RasterizationRateLayerCount",
    "TileShaders",
    "ImageBlocks",
    "RasterOrderGroups",
    "MemoryOrderAtomics",
    "LargeMRT",
    "SupportFlags2023",
    "MaxTotalComputeThreadsPerThreadgroup",
    "MaxComputeLocalMemorySizes",
    "MaxComputeThreadgroupMemory",
    "MaxComputeThreadgroupMemoryAlignmentBytes",
    "SupportFlags2024",
    "GpuCoreCount",
    "MaxTextureLayers",
    "MaxPredicatedNestingDepth",
    "HostGPUFamilyClamped",
    "ArgumentBuffersTier",
    "ArgumentBuffersMaxSamplerCount",
    "MinimumLinearTextureAlignment",
    "SupportedTextureWriteRoundingModes",
    "SupportFlags2025",
    "",
    "HostGPUFamilies",
];

/// The name of `key` (empty for a key no check names): a literal.
fn key_name(key: u32) -> &'static str {
    KEY_NAMES.get(key.to_usize()).copied().unwrap_or("")
}

/// The line for a guest's flags under `key` that the host's lack.
fn lacks_flags_line(key: u32, host: u32, guest: u32) {
    log::warn!(
        "host lacks some of the guest's {}: the host has {host:#x}, the guest asks for {guest:#x}",
        key_name(key)
    );
}

/// The line for a host alignment under `key` above the guest's.
fn over_aligned_line(key: u32, host: u32, guest: u32) {
    log::warn!(
        "host's {} is above the guest's: the host has {host:#x}, the guest asks for {guest:#x}",
        key_name(key)
    );
}

/// Whether this host provides what a guest's list (`guest_pairs`, read to its terminator or its
/// end) asks for. The host's own list comes first, from [`write_device_info`] with max key 45
/// and 46 pairs, so a raise its caps kept is thrown before the guest's list is read; a guest key
/// with no reader case then dies, so the caller passes only keys a reader takes. Only a resume,
/// with no handler, asks it, so that raise dies where it is thrown.
pub fn check_capabilities(
    caps: &DeviceCaps,
    snapshot: &DeviceInfoSnapshot,
    guest_pairs: &[KeyValuePair],
) -> RaiseResult<bool> {
    let is_special = |k: u32| SPECIAL_KEYS.contains(&k);
    let is_optional = |k: u32| OPTIONAL_KEYS.contains(&k);
    // The host's list, written and read first.
    let mut host_pairs = [KeyValuePair::default(); 46];
    write_device_info(
        OnFault::Die,
        caps,
        snapshot,
        HOST_INFO_MAX_KEY,
        HOST_INFO_PAIRS,
        |i, pair| {
            if let Some(slot) = host_pairs.get_mut(i.to_usize()) {
                *slot = pair;
            }
        },
    )?;
    let host = read_device_info(&host_pairs);
    let guest = read_device_info(guest_pairs);
    // Every key the guest defines, the host defines (40 is optional).
    for k in DEFINED_KEYS {
        if guest.present(k) && !host.present(k) && !is_optional(k) {
            log::warn!(
                "host lacks the guest's device info key {k} ({}): the host does not define it",
                key_name(k)
            );
            return Ok(false);
        }
    }
    // Every value the guest asks for, the host reaches.
    for k in VALUE_KEYS {
        if guest.present(k) && !is_special(k) && host.value(k) < guest.value(k) {
            log::warn!(
                "host lacks the guest's {}: the host has {}, the guest asks for {}",
                key_name(k),
                host.value(k),
                guest.value(k)
            );
            return Ok(false);
        }
    }
    // A flag among `mask` that the guest asks for and the host lacks.
    let lacks_flags =
        |k: u32, mask: u32| guest.present(k) && guest.value(k) & !host.value(k) & mask != 0;
    // The host needs more alignment than the guest asks for.
    let over_aligned = |k: u32| guest.present(k) && host.value(k) > guest.value(k);
    // PrimitiveTypeSupport: its low 10 bits.
    if lacks_flags(11, 0x3ff) {
        lacks_flags_line(11, host.value(11), guest.value(11));
        return Ok(false);
    }
    if over_aligned(13) {
        over_aligned_line(13, host.value(13), guest.value(13));
        return Ok(false);
    }
    if over_aligned(15) {
        over_aligned_line(15, host.value(15), guest.value(15));
        return Ok(false);
    }
    if lacks_flags(28, u32::MAX) {
        lacks_flags_line(28, host.value(28), guest.value(28));
        return Ok(false);
    }
    if lacks_flags(33, u32::MAX) {
        lacks_flags_line(33, host.value(33), guest.value(33));
        return Ok(false);
    }
    if over_aligned(32) {
        over_aligned_line(32, host.value(32), guest.value(32));
        return Ok(false);
    }
    // MinimumLinearTextureAlignment, only when both define it.
    if guest.present(40) && host.present(40) && host.value(40) > guest.value(40) {
        over_aligned_line(40, host.value(40), guest.value(40));
        return Ok(false);
    }
    if guest.present(41) {
        if !host.present(41) {
            log::warn!(
                "host lacks the guest's device info key 41 ({}): the host does not define it",
                key_name(41)
            );
            return Ok(false);
        }
        if guest.value(41) & !host.value(41) != 0 {
            lacks_flags_line(41, host.value(41), guest.value(41));
            return Ok(false);
        }
    }
    if lacks_flags(42, u32::MAX) {
        lacks_flags_line(42, host.value(42), guest.value(42));
        return Ok(false);
    }
    Ok(true)
}

/// The length a device info request maps at the guest's page: one guest page.
const DEVICE_INFO_MAPPING_LENGTH: u64 = APV_PAGE_SIZE;

/// A guest's device info request (opcode 58): maps the guest's page, writes the host's pairs
/// into it, unmaps it, and stores the requested max key.
///
/// `new_contiguous_mapping(page, length)` maps the page (a new mapping, or none);
/// `write(&mapping, max_key, max_count)` is [`write_device_info`] into it; dropping the mapping
/// unmaps it; `store_max_key(max_key)` stores the requested key, whatever the guest wrote. No
/// mapping raises.
///
/// A raise out of the writer (a Metal exception, which is not caught here) skips the unmap:
/// the mapping is kept for good and the page stays mapped, and the key is not stored.
pub fn get_device_info<M>(
    max_key: u32,
    max_count: u32,
    destination_page: u32,
    new_contiguous_mapping: impl FnOnce(u32, u64) -> RaiseResult<Option<M>>,
    write: impl FnOnce(&M, u32, u32) -> RaiseResult<()>,
    store_max_key: impl FnOnce(u32),
) -> RaiseResult<()> {
    let Some(mapping) = new_contiguous_mapping(destination_page, DEVICE_INFO_MAPPING_LENGTH)?
    else {
        crate::raise!(
            crate::device::DeviceRaise::DeviceInfoNotMapped,
            format_args!("device info page {destination_page:#x} not mapped")
        );
    };
    if let Err(e) = write(&mapping, max_key, max_count) {
        keep_forever(mapping, Kept::UnmapNeverComes);
        return Err(e);
    }
    drop(mapping);
    store_max_key(max_key);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod request {
        use super::*;
        use crate::device::DeviceRaise;
        use crate::raise::RaiseId;
        use crate::raise::death::DeathKind;
        use crate::raise::death::DeathSite;
        use crate::sync::Mutex;
        use std::sync::Arc;

        /// A mapping whose unmap is recorded.
        struct Map(Arc<Mutex<Vec<String>>>);
        impl Drop for Map {
            fn drop(&mut self) {
                self.0.lock().push("release".into());
            }
        }

        /// maps, writes, unmaps, then stores the requested key
        #[test]
        fn maps_writes_releases_stores() {
            let log = Arc::new(Mutex::new(Vec::<String>::new()));
            let (l1, l2, l3) = (log.clone(), log.clone(), log.clone());
            get_device_info(
                0xffff_ffff,
                46,
                7,
                |page, len| {
                    l1.lock().push(format!("map {page} {len:#x}"));
                    Ok(Some(Map(l1.clone())))
                },
                |_, k, n| {
                    l2.lock().push(format!("write {k:#x} {n}"));
                    Ok(())
                },
                |k| l3.lock().push(format!("store {k:#x}")),
            )
            .unwrap();
            // One guest page: 0x4000 on a 16 KiB guest page.
            assert_eq!(
                *log.lock(),
                [
                    format!("map 7 {DEVICE_INFO_MAPPING_LENGTH:#x}"),
                    "write 0xffffffff 46".to_owned(),
                    "release".to_owned(),
                    "store 0xffffffff".to_owned()
                ]
            );
        }

        /// no mapping: a generic "Failed to map", nothing written
        #[test]
        fn nil_mapping_throws() {
            let log = Mutex::new(Vec::<&str>::new());
            let e = get_device_info::<Map>(
                45,
                46,
                0,
                |_, _| Ok(None),
                |_, _, _| {
                    log.lock().push("write");
                    Ok(())
                },
                |_| log.lock().push("store"),
            )
            .unwrap_err();
            assert!(log.lock().is_empty());
            assert!(matches!(e.origin(), crate::raise::Origin::Core(_)));
            assert_eq!(
                e.detail().unwrap_or_default(),
                "device info page 0x0 not mapped"
            );
            assert_eq!(e.id(), RaiseId::Device(DeviceRaise::DeviceInfoNotMapped));
        }

        /// under the handler the FIFO's opcode loop gives it, the raise names the site its death
        /// would
        #[test]
        fn nil_mapping_raise_names_the_throw() {
            let r = get_device_info::<Map>(45, 46, 0, |_, _| Ok(None), |_, _, _| Ok(()), |_| ())
                .unwrap_err();
            assert_eq!(r.id(), RaiseId::Device(DeviceRaise::DeviceInfoNotMapped));
            assert_eq!(
                r.site(),
                DeathSite {
                    kind: DeathKind::UnhandledRaise,
                    file: "device.info.not_mapped",
                    line: 0
                }
            );
        }

        /// A raise out of the writer leaks the mapping and stores no key
        #[test]
        fn write_raise_leaks_the_mapping() {
            let log = Arc::new(Mutex::new(Vec::<String>::new()));
            let (l1, l3) = (log.clone(), log.clone());
            let e = get_device_info(
                45,
                46,
                3,
                |_, _| Ok(Some(Map(l1.clone()))),
                |_, _, _| {
                    Err(crate::testkit::capture::foreign(
                        OnFault::Return,
                        "supportsFamily:",
                        "metal",
                    ))
                },
                |k| l3.lock().push(format!("store {k}")),
            )
            .unwrap_err();
            assert_eq!(e.detail().unwrap_or_default(), "metal");
            assert!(log.lock().is_empty(), "{:?}", log.lock());
        }

        /// the map raising: the raise propagates, nothing written or stored
        #[test]
        fn a_raising_map_propagates() {
            let log = Mutex::new(Vec::<&str>::new());
            let e = get_device_info::<Map>(
                45,
                46,
                3,
                |_, _| {
                    Err(crate::testkit::capture::foreign(
                        OnFault::Return,
                        "mapMemory",
                        "no",
                    ))
                },
                |_, _, _| {
                    log.lock().push("write");
                    Ok(())
                },
                |_| log.lock().push("store"),
            )
            .unwrap_err();
            assert_eq!(e.detail().unwrap_or_default(), "no");
            assert!(log.lock().is_empty());
        }
    }
    use crate::guest::abi::features::features_for_version;
    use crate::platform::device::GpuDevice as _;
    use crate::raise::Raise;
    use crate::testkit::FakePlatform;
    use crate::testkit::capture::catch_death;

    /// Every feature and device capability on, at a level.
    fn all_on(level: u64) -> DeviceInfoSnapshot {
        DeviceInfoSnapshot {
            features: features_for_version(245),
            device_feature_level: level,
            supports_argument_buffers: true,
            guest_device_info_max_key: 1,
            supports_buffer_with_iosurface: true,
            supports_shared_textures: true,
            supports_large_user_tasks: true,
            supports_large_kernel_tasks: true,
            supports_range_buffer: true,
            supports_shared_memory_heap: true,
        }
    }

    fn run(
        p: &FakePlatform,
        s: &DeviceInfoSnapshot,
        max_key: u32,
        max_count: u32,
    ) -> Vec<KeyValuePair> {
        run_with(&p.device().caps(), s, max_key, max_count)
    }

    fn run_with(
        caps: &DeviceCaps,
        s: &DeviceInfoSnapshot,
        max_key: u32,
        max_count: u32,
    ) -> Vec<KeyValuePair> {
        let mut out = Vec::new();
        write_device_info(OnFault::Die, caps, s, max_key, max_count, |i, kv| {
            assert_eq!(i as usize, out.len());
            out.push(kv);
        })
        .unwrap();
        out
    }

    /// The fake device's caps: its queries' answers, the platform's scripts applied.
    fn caps_of(p: &FakePlatform) -> DeviceCaps {
        DeviceCaps::clone(&p.device().caps())
    }

    fn keys(pairs: &[KeyValuePair]) -> Vec<u32> {
        pairs.iter().map(|p| p.key).collect()
    }

    fn value(pairs: &[KeyValuePair], key: u32) -> Option<u32> {
        pairs.iter().find(|p| p.key == key).map(|p| p.value)
    }

    fn pairs(kv: &[(u32, u32)]) -> Vec<KeyValuePair> {
        kv.iter()
            .map(|&(key, value)| KeyValuePair { key, value })
            .collect()
    }

    /// The host's own list at max key 45 (what `check_capabilities` compares against).
    fn host(p: &FakePlatform, s: &DeviceInfoSnapshot) -> Vec<KeyValuePair> {
        let mut v = run(p, s, HOST_INFO_MAX_KEY, HOST_INFO_PAIRS);
        v.pop(); // the terminator
        v
    }

    #[test]
    fn feature_level_above_zero_means_everything() {
        assert!(feature_level_above(0, 3));
        assert!(!feature_level_above(1, 1));
        assert!(feature_level_above(2, 1));
        assert!(!feature_level_above(3, 3));
        assert!(feature_level_above(4, 3));
    }

    mod write {
        use super::*;

        const LEVEL0_ORDER: [u32; 42] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 33, 36, 29, 30, 31, 32, 34, 41, 35, 37, 42, 44, 40, 0,
        ];

        /// The emission order and the fake's values with everything on.
        #[test]
        fn order_and_values_level0() {
            let p = FakePlatform::new();
            let out = run(&p, &all_on(0), 45, 46);
            assert_eq!(keys(&out), LEVEL0_ORDER);
            let expect = [
                (1, 4), // 1, 2, 4 supported
                (2, 0), // no D24S8
                (3, 1024),
                (6, 32768),
                (10, 8),     // PGDeserializerVersion
                (11, 0x3ff), // primitive types 0-9
                (12, 1),
                (13, 16),
                (14, 1),
                (15, 0x4000), // heap alignment
                (16, 1),
                (17, 1),
                (18, 0x20007),
                (19, 1),
                (20, 2),
                (22, 2),
                (23, 1),
                (28, 7), // stride 2 | Apple5 | multisample array 4
                (33, 0xfff),
                (36, 8),
                (34, 10),
                (41, 7),
                (35, 2048),
                (37, 1009),
                (42, 3),
                (44, 0x7f),
                (40, 16),
                (0, 0),
            ];
            for (k, v) in expect {
                assert_eq!(value(&out, k), Some(v), "key {k}");
            }
        }

        /// The keys written before `caps` raises, with a handler live; `None` when it does not.
        fn raised_after(
            caps: &DeviceCaps,
            s: &DeviceInfoSnapshot,
            max_key: u32,
        ) -> Option<Vec<u32>> {
            let mut out = Vec::new();
            let r = write_device_info(OnFault::Return, caps, s, max_key, 46, |_, kv| {
                out.push(kv.key)
            });
            r.is_err().then_some(out)
        }

        /// A raise the caps kept is thrown where its answer is asked, after the pairs before it,
        /// and only when it is asked: the sample counts always, the primitive types when key 11
        /// fits, the heap query with Metal heaps, Apple5 from level 1, the rounding modes and
        /// the key 37 chain (up to its first yes) from level 2, and every family from level 3.
        #[test]
        fn raises_where_asked() {
            fn raised<T>() -> RaiseResult<T> {
                Err(Raise::foreign("supportsFamily:"))
            }
            let base = caps_of(&FakePlatform::new());
            let before = |key: u32| {
                let at = LEVEL0_ORDER.iter().position(|k| *k == key).unwrap();
                Some(LEVEL0_ORDER[..at].to_vec())
            };
            let mut c = base.clone();
            c.sample_counts[3] = raised();
            assert_eq!(raised_after(&c, &all_on(0), 0), Some(vec![]));
            let mut c = base.clone();
            c.primitive_types[9] = raised();
            assert_eq!(raised_after(&c, &all_on(0), 11), None);
            assert_eq!(raised_after(&c, &all_on(0), 45), before(11));
            let mut c = base.clone();
            c.heap_buffer_page_private = raised();
            assert_eq!(raised_after(&c, &all_on(0), 45), before(15));
            let mut s = all_on(0);
            s.features = features_for_version(2);
            assert_eq!(raised_after(&c, &s, 45), None);
            let mut c = base.clone();
            c.families[0] = raised(); // Apple5
            assert_eq!(raised_after(&c, &all_on(1), 45), None);
            assert_eq!(raised_after(&c, &all_on(2), 45), before(28));
            let mut c = base.clone();
            c.write_rounding_modes[1] = raised();
            assert_eq!(raised_after(&c, &all_on(2), 45), None);
            assert_eq!(raised_after(&c, &all_on(3), 45), before(41));
            let mut c = base.clone();
            c.families[4] = raised(); // Apple9, the chain's first
            assert_eq!(raised_after(&c, &all_on(2), 45), None);
            assert_eq!(raised_after(&c, &all_on(3), 45), before(37));
            let mut c = base.clone();
            c.families[2] = raised(); // Apple7: the chain stops at Apple9's yes
            assert_eq!(raised_after(&c, &all_on(3), 45), None);
            assert_eq!(raised_after(&c, &all_on(0), 45), before(42));
            let mut c = base;
            c.families[6] = raised(); // -1009
            assert_eq!(raised_after(&c, &all_on(3), 45), None);
            assert_eq!(raised_after(&c, &all_on(0), 45), before(42));
        }

        /// Levels 1-4 cut the level blocks and the shader version.
        #[test]
        fn feature_levels() {
            let l1 = run(&FakePlatform::new(), &all_on(1), 45, 46);
            assert_eq!(
                keys(&l1),
                [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 40, 0
                ]
            );
            assert_eq!(value(&l1, 18), Some(0x20002));
            let l2 = run(&FakePlatform::new(), &all_on(2), 45, 46);
            assert!(keys(&l2).contains(&28) && !keys(&l2).contains(&33));
            assert_eq!(value(&l2, 18), Some(0x20006));
            let l3 = run(&FakePlatform::new(), &all_on(3), 45, 46);
            assert!(keys(&l3).contains(&37) && !keys(&l3).contains(&42));
            assert_eq!(value(&l3, 18), Some(0x20007));
            let l4 = run(&FakePlatform::new(), &all_on(4), 45, 46);
            assert_eq!(keys(&l4), LEVEL0_ORDER);
        }

        /// Fit rule: max key 0 or max count 0 writes nothing.
        #[test]
        fn nothing_fits() {
            for (k, c) in [(0, 46), (45, 0)] {
                assert!(run(&FakePlatform::new(), &all_on(0), k, c).is_empty());
            }
        }

        /// A key at or above max key is skipped, later lower keys still come, and the
        /// terminator is written when max key ≥ 1 and there is room.
        #[test]
        fn max_key_skips() {
            let out = run(&FakePlatform::new(), &all_on(0), 3, 46);
            assert_eq!(keys(&out), [1, 2, 0]);
            let out = run(&FakePlatform::new(), &all_on(0), 1, 46);
            assert_eq!(keys(&out), [0]);
            let out = run(&FakePlatform::new(), &all_on(0), 12, 46);
            assert_eq!(keys(&out), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]);
        }

        /// max count caps the list; the terminator only when it still fits.
        #[test]
        fn max_count_caps() {
            let out = run(&FakePlatform::new(), &all_on(0), 45, 3);
            assert_eq!(keys(&out), [1, 2, 3]);
            let out = run(&FakePlatform::new(), &all_on(1), 45, 20);
            assert_eq!(keys(&out).last(), Some(&0));
            let out = run(&FakePlatform::new(), &all_on(1), 45, 19);
            assert_eq!(keys(&out).last(), Some(&40));
        }

        /// Conditions: features and device capabilities gate keys 14-17, 19, 21, 23-27, 36.
        #[test]
        fn feature_gates() {
            let mut s = all_on(0);
            s.features = features_for_version(2);
            s.supports_buffer_with_iosurface = false;
            s.supports_shared_textures = false;
            let out = run(&FakePlatform::new(), &s, 45, 46);
            for k in [14, 15, 16, 17, 19, 21, 23, 24, 25, 26, 27, 36] {
                assert!(!keys(&out).contains(&k), "key {k}");
            }
            assert_eq!(value(&out, 12), Some(0));
            assert_eq!(value(&out, 28), Some(5)); // Apple5 | multisample array
            // 17 needs both the device and the feature.
            let mut s = all_on(0);
            s.supports_buffer_with_iosurface = false;
            assert!(!keys(&run(&FakePlatform::new(), &s, 45, 46)).contains(&17));
            let mut s = all_on(0);
            s.features.supports_shared_textures = false;
            assert!(!keys(&run(&FakePlatform::new(), &s, 45, 46)).contains(&19));
        }

        /// Metal-side values: sample counts, zero rate layers, tile shaders, families,
        /// rounding modes, no gpuCoreCount.
        #[test]
        fn metal_values() {
            let mut c = caps_of(&FakePlatform::new());
            c.sample_counts = std::array::from_fn(|_| Ok(false));
            c.limits.max_rasterization_rate_layer_count = 0;
            c.flags.supports_tile_shaders = false;
            c.gpu_core_count = None;
            // The key 37 chain: 1009 no, 1008 no, 1007 yes.
            c.families[4] = Ok(false);
            c.families[3] = Ok(false);
            c.write_rounding_modes[0] = Ok(false);
            c.max_threads_per_threadgroup =
                crate::platform::types::MtlSize::new(0x1_0000_0020, 2, 3);
            let out = run_with(&c, &all_on(0), 45, 46);
            assert_eq!(value(&out, 1), Some(1));
            assert_eq!(value(&out, 3), Some(0x20)); // truncated to 32 bits
            assert_eq!(value(&out, 22), None);
            assert_eq!(value(&out, 23), Some(0));
            assert_eq!(value(&out, 34), None);
            assert_eq!(value(&out, 37), Some(1007));
            assert_eq!(value(&out, 41), Some(5));
            // Key 20 is the public maxVertexAmplificationCount, truncated.
            c.max_vertex_amplification_count = 0x1_0000_0003;
            assert_eq!(value(&run_with(&c, &all_on(0), 45, 46), 20), Some(3));
        }

        /// Key 37 falls back to Apple5 when no family answers; key 42 is 0 without argument
        /// buffers; key 33's bits follow the device.
        #[test]
        fn family_fallback_and_flags() {
            let mut c = caps_of(&FakePlatform::new());
            c.families = std::array::from_fn(|_| Ok(false));
            let mut s = all_on(0);
            s.supports_argument_buffers = false;
            s.supports_large_kernel_tasks = false;
            let out = run_with(&c, &s, 45, 46);
            assert_eq!(value(&out, 37), Some(1005));
            assert_eq!(value(&out, 42), Some(0));
            assert_eq!(value(&out, 44), Some(0));
            assert_eq!(value(&out, 33), Some(0xfff & !0x20 & !0x2));
            assert_eq!(value(&out, 28), Some(6));
        }

        /// A Metal exception the caps kept propagates as it was raised, after the pairs already
        /// written (none is caught).
        #[test]
        fn raise_propagates() {
            let mut c = caps_of(&FakePlatform::new());
            let raised = Raise::foreign("supportsPrimitiveType:").with_detail("x");
            c.primitive_types[0] = Err(raised.clone());
            let mut out = Vec::new();
            let r = write_device_info(OnFault::Return, &c, &all_on(0), 45, 46, |_, kv| {
                out.push(kv)
            });
            assert_eq!(r, Err(raised));
            assert_eq!(keys(&out), (1..=10).collect::<Vec<_>>());
        }
    }

    mod read {
        use super::*;

        #[test]
        fn reads_to_the_terminator() {
            let i = read_device_info(&pairs(&[(1, 4), (44, 7), (1, 8), (0, 0), (43, 1)]));
            assert!(i.present(1) && i.present(44));
            assert_eq!((i.value(1), i.value(44)), (8, 7));
            assert!(!i.present(2));
        }

        /// Key 43, and any key ≥ 45, dies.
        #[test]
        fn unknown_keys_die() {
            for k in [43, 45, u32::MAX] {
                let r = catch_death(|| read_device_info(&pairs(&[(1, 1), (k, 0)])));
                assert_eq!(r.map(|_| ()), Err(GuestDeath::UnknownInfoKey.site()), "{k}");
            }
        }
    }

    mod check {
        use super::*;

        fn check(guest: &[KeyValuePair]) -> bool {
            let p = FakePlatform::new();
            check_capabilities(&p.device().caps(), &all_on(0), guest).unwrap()
        }

        fn with(base: &[KeyValuePair], key: u32, value: u32) -> Vec<KeyValuePair> {
            let mut v: Vec<KeyValuePair> = base.iter().copied().filter(|p| p.key != key).collect();
            v.push(KeyValuePair { key, value });
            v
        }

        #[test]
        fn own_list_and_empty_list_pass() {
            let h = host(&FakePlatform::new(), &all_on(0));
            assert!(check(&h));
            assert!(check(&[]));
        }

        /// A key the guest defines and the host does not fails (38 is never written).
        #[test]
        fn undefined_on_host() {
            assert!(!check(&pairs(&[(38, 0)])));
            let lower = host(&FakePlatform::new(), &all_on(1));
            let p = FakePlatform::new();
            let full = host(&p, &all_on(0));
            let q = FakePlatform::new();
            assert!(!check_capabilities(&q.device().caps(), &all_on(1), &full).unwrap());
            assert!(check_capabilities(&q.device().caps(), &all_on(0), &lower).unwrap());
        }

        /// Plain values: the host must have at least the guest's.
        #[test]
        fn values() {
            assert!(!check(&pairs(&[(1, 8)])));
            assert!(check(&pairs(&[(1, 4)])));
            assert!(!check(&pairs(&[(10, 9)])));
            assert!(check(&pairs(&[(10, 7)])));
        }

        /// The one line a refused guest list logs, as its level and message.
        fn refusal(s: &DeviceInfoSnapshot, guest: &[KeyValuePair]) -> (log::Level, String) {
            let p = FakePlatform::new();
            let logs = crate::testkit::logs::capture();
            assert!(!check_capabilities(&p.device().caps(), s, guest).unwrap());
            let lines = logs.take();
            assert_eq!(lines.len(), 1, "{lines:?}");
            let (level, _, message) = lines.into_iter().next().expect("one line");
            (level, message)
        }

        /// Each refusal is one warning that names the key.
        #[test]
        fn logs_name_the_key() {
            let logged = |guest: &[KeyValuePair]| {
                let (level, message) = refusal(&all_on(0), guest);
                assert_eq!(level, log::Level::Warn);
                message
            };
            let h = host(&FakePlatform::new(), &all_on(0));
            let value = |k: u32| h.iter().find(|p| p.key == k).unwrap().value;
            assert_eq!(
                logged(&pairs(&[(1, 8)])),
                format!(
                    "host lacks the guest's MSAASamples: the host has {}, the guest asks for 8",
                    value(1)
                )
            );
            assert_eq!(
                logged(&pairs(&[(38, 0)])),
                "host lacks the guest's device info key 38 (ArgumentBuffersTier): the host does \
                 not define it"
            );
            assert_eq!(
                logged(&pairs(&[(28, u32::MAX)])),
                format!(
                    "host lacks some of the guest's SupportFlags2023: the host has {:#x}, the \
                     guest asks for 0xffffffff",
                    value(28)
                )
            );
            assert_eq!(
                logged(&pairs(&[(13, 0)])),
                format!(
                    "host's LinearTextureAlignment is above the guest's: the host has {:#x}, the \
                     guest asks for 0x0",
                    value(13)
                )
            );
        }

        /// Each failure logs one line with its values: the key's name, then the host's value and
        /// the guest's.
        #[test]
        fn each_failure_logs_its_values() {
            let logged = |s: &DeviceInfoSnapshot, guest: &[KeyValuePair]| refusal(s, guest).1;
            let on = all_on(0);
            let h = host(&FakePlatform::new(), &on);
            let hv = |k: u32| u64::from(value(&h, k).unwrap());
            assert!(hv(32) > 0);
            let lacks = |key: &str, host: u64, guest: u64| {
                format!(
                    "host lacks some of the guest's {key}: the host has {host:#x}, the guest \
                     asks for {guest:#x}"
                )
            };
            let above = |key: &str, host: u64, guest: u64| {
                format!(
                    "host's {key} is above the guest's: the host has {host:#x}, the guest asks \
                     for {guest:#x}"
                )
            };
            let cases: Vec<(&[(u32, u32)], String)> =
                vec![
                (
                    &[(38, 0)],
                    "host lacks the guest's device info key 38 (ArgumentBuffersTier): the host \
                     does not define it"
                        .to_owned(),
                ),
                (
                    &[(1, 8)],
                    format!(
                        "host lacks the guest's MSAASamples: the host has {}, the guest asks for \
                         8",
                        hv(1)
                    ),
                ),
                (&[(13, 0)], above("LinearTextureAlignment", hv(13), 0)),
                (&[(15, 0x1000)], above("HeapBufferAlignment", hv(15), 0x1000)),
                (
                    &[(28, u32::MAX)],
                    lacks("SupportFlags2023", hv(28), u64::from(u32::MAX)),
                ),
                (
                    &[(32, 0)],
                    above("MaxComputeThreadgroupMemoryAlignmentBytes", hv(32), 0),
                ),
                (&[(40, 8)], above("MinimumLinearTextureAlignment", hv(40), 8)),
                (
                    &[(41, 8)],
                    lacks("SupportedTextureWriteRoundingModes", hv(41), 8),
                ),
            ];
            for (guest, want) in cases {
                assert_eq!(logged(&on, &pairs(guest)), want, "{guest:?}");
            }
            // SupportFlags2024 without range buffers, SupportFlags2025 without argument buffers.
            let mut s = all_on(0);
            s.supports_range_buffer = false;
            let h = host(&FakePlatform::new(), &s);
            assert_eq!(
                logged(&s, &pairs(&[(33, 8)])),
                lacks("SupportFlags2024", u64::from(value(&h, 33).unwrap_or(0)), 8)
            );
            let mut s = all_on(0);
            s.supports_argument_buffers = false;
            let h = host(&FakePlatform::new(), &s);
            assert_eq!(
                logged(&s, &pairs(&[(42, 1)])),
                lacks("SupportFlags2025", u64::from(value(&h, 42).unwrap_or(0)), 1)
            );
        }

        /// Special keys skip the value check and use their own: alignment (host ≤ guest) and
        /// flags (guest ⊆ host among the low bits).
        #[test]
        fn special_keys() {
            // LinearTextureAlignment: host 16.
            assert!(!check(&pairs(&[(13, 8)])));
            assert!(check(&pairs(&[(13, 64)])));
            // HeapBufferAlignment: host 0x4000.
            assert!(!check(&pairs(&[(15, 0x1000)])));
            // Primitive types: only 10 bits are checked.
            assert!(check(&pairs(&[(11, 0x3ff | 0x400)])));
            // SupportFlags2024: a bit the host lacks.
            let p = FakePlatform::new();
            let mut s = all_on(0);
            s.supports_range_buffer = false;
            assert!(!check_capabilities(&p.device().caps(), &s, &pairs(&[(33, 8)])).unwrap());
            // MinimumLinearTextureAlignment: host 16.
            assert!(!check(&pairs(&[(40, 8)])));
            assert!(check(&pairs(&[(40, 16)])));
            // Rounding modes: host 7; a guest bit outside fails.
            assert!(!check(&pairs(&[(41, 8)])));
            assert!(check(&pairs(&[(41, 3)])));
            // SupportFlags2025: host 0 without argument buffers.
            let p = FakePlatform::new();
            let mut s = all_on(0);
            s.supports_argument_buffers = false;
            assert!(!check_capabilities(&p.device().caps(), &s, &pairs(&[(42, 1)])).unwrap());
            assert!(check_capabilities(&p.device().caps(), &s, &pairs(&[(42, 0)])).unwrap());
        }

        /// The checks run in order: an undefined key fails before a value that would fail.
        #[test]
        fn order_defined_before_value() {
            let h = host(&FakePlatform::new(), &all_on(0));
            let bad = with(&with(&h, 1, 99), 38, 0);
            assert!(!check(&bad));
        }

        /// A guest key 43 dies, after the host's list was written: a raise the host's list
        /// throws comes first, and dies where it is thrown (only a resume, with no handler,
        /// checks).
        #[test]
        fn guest_key_43_dies() {
            let caps = caps_of(&FakePlatform::new());
            let r = catch_death(|| {
                check_capabilities(&caps, &all_on(0), &pairs(&[(43, 0)])).map(|_| ())
            });
            assert_eq!(r.map(|_| ()), Err(GuestDeath::UnknownInfoKey.site()));
            let mut raising = caps;
            raising.families[6] = Err(Raise::foreign("supportsFamily:"));
            let r = catch_death(|| {
                check_capabilities(&raising, &all_on(0), &pairs(&[(43, 0)])).map(|_| ())
            });
            assert_eq!(r, Err(Raise::foreign("supportsFamily:").site()));
        }

        /// The host's list is written with max key 45 and 46 pairs.
        #[test]
        fn host_list_bounds() {
            let p = FakePlatform::new();
            let h = host(&p, &all_on(0));
            assert_eq!(h.len(), 41);
            assert!(h.iter().all(|kv| kv.key < 45));
        }
    }
}
