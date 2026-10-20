//! `VulkanPlatform`: `impl Platform` over `metal2vulkan-runtime`, and [`Obj`], the handle every
//! associated type is.
//!
//! Every handle is an [`Obj<T>`]: one counted reference to a value of this crate, which holds the
//! runtime's object (one reference of the runtime's own) and what core's trait asks of the object
//! beside it. Not `Clone` (D6): `.clone_ref()` is the retain and `Drop` the release, so the
//! runtime's object goes when core's last handle to it does (leaked instead while an isolated task
//! that died lets go of it, as the Metal platform leaks its objects then). `id()` is the
//! reference's address: the same for every reference to one object, as a Metal object's pointer
//! is.
//!
//! One `VulkanPlatform` is one process's (D17), as `MetalPlatform` is.

use std::fmt;
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Weak};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::host::HostBlocks;
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::os::Threads;
use reims_vgpu_core::platform::os::surface::IoSurfaces;
use reims_vgpu_core::platform::os::vm::HostVm;
use reims_vgpu_core::platform::types::{MtlSize, ResourceAddressRange};
use reims_vgpu_core::platform::values::{RateMapDesc, TextureDesc};
use reims_vgpu_core::platform::{Handle, ObjId, Platform, ProcessState};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::tasks::launcher::TaskLauncher;

use crate::answer::{get, plain, refused};
use crate::gpu::command::{VulkanCommandBuffer, VulkanCommandQueue};
use crate::gpu::desc::{
    RenderPassDescriptorObject, TextureDescriptorObject, TextureFields,
    VulkanComputePipelineDescriptor, VulkanRenderPassDescriptor, VulkanRenderPipelineDescriptor,
    VulkanTextureDescriptor, VulkanTileRenderPipelineDescriptor,
};
use crate::gpu::device::{self, VulkanDevice};
use crate::gpu::encode_blit::VulkanBlitEncoder;
use crate::gpu::encode_compute::VulkanComputeEncoder;
use crate::gpu::encode_render::VulkanRenderEncoder;
use crate::gpu::event::{
    VulkanEvent, VulkanSharedEvent, VulkanSharedEventHandle, VulkanSharedEventListener,
};
use crate::gpu::icb::{VulkanIcb, VulkanIndirectComputeCommand, VulkanIndirectRenderCommand};
use crate::gpu::rate_map::{VulkanRateMap, VulkanRateMapDescriptor, VulkanRateMapLayerDescriptor};
use crate::gpu::resource::{
    VulkanBuffer, VulkanComputePipeline, VulkanDepthStencilState, VulkanFence, VulkanFunction,
    VulkanHeap, VulkanLibrary, VulkanRenderPipeline, VulkanSamplerState, VulkanSharedTextureHandle,
    VulkanTexture,
};
use crate::os::{OsClock, OsSurface, OsSurfaces, OsThreads};
use crate::remap::{self, VulkanVm};

/// One counted reference to an object of this platform (architecture 4.1).
///
/// Not `Clone` (D6): [`Handle::clone_ref`] is the only retain. `Drop` is the release, except
/// while the thread unwinds or releases an isolated task that died
/// ([`leaks_objects`](reims_vgpu_core::tasks::isolation::life::leaks_objects)): the reference
/// is then leaked, as a crashed process never released it.
pub struct Obj<T: Send + Sync + 'static>(ManuallyDrop<Arc<T>>);

impl<T: Send + Sync + 'static> Obj<T> {
    /// A new object holding `value`: its first reference.
    pub fn new(value: T) -> Obj<T> {
        Obj(ManuallyDrop::new(Arc::new(value)))
    }

    /// A reference that does not hold the object.
    pub fn downgrade(&self) -> Weak<T> {
        Arc::downgrade(&self.0)
    }

    /// Another reference to the object `weak` names, while one lives.
    pub fn upgrade(weak: &Weak<T>) -> Option<Obj<T>> {
        weak.upgrade().map(|arc| Obj(ManuallyDrop::new(arc)))
    }
}

impl<T: Send + Sync + 'static> Drop for Obj<T> {
    /// The release, or nothing while the thread unwinds or releases an isolated task that died.
    fn drop(&mut self) {
        if reims_vgpu_core::tasks::isolation::life::leaks_objects() {
            return;
        }
        // SAFETY: the reference is released once, here, and never touched after.
        unsafe { ManuallyDrop::drop(&mut self.0) }
    }
}

impl<T: Send + Sync + 'static> std::ops::Deref for Obj<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Send + Sync + fmt::Debug + 'static> fmt::Debug for Obj<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{:#x} {:?}>", self.id().0, **self.0)
    }
}

impl<T: Send + Sync + 'static> Handle for Obj<T> {
    fn clone_ref(&self) -> Self {
        Obj(ManuallyDrop::new(Arc::clone(&self.0)))
    }

    fn id(&self) -> ObjId {
        ObjId(Arc::as_ptr(&self.0).cast::<u8>().addr().to_u64())
    }
}

/// The process's platform: the runtime's GPU objects, and the OS's IOSurface, VM (reporting the
/// pages it changes, [`crate::remap`]), clocks and threads ([`crate::os`]), with the task
/// launcher.
pub struct VulkanPlatform {
    surfaces: OsSurfaces,
    vm: VulkanVm,
    clock: OsClock,
    threads: OsThreads,
    globals: ProcessState<VulkanPlatform>,
    launcher: Box<dyn TaskLauncher<VulkanPlatform>>,
}

impl fmt::Debug for VulkanPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanPlatform").finish_non_exhaustive()
    }
}

impl VulkanPlatform {
    /// The process's platform. `launcher` starts isolated GPU task services (each on threads of
    /// its own in the product, `tasks::isolation::TaskThreads`). Make one per process:
    /// `ProcessState` is the process-wide state (D17). It installs what the OS's services need
    /// first ([`crate::os::install_process_services`]): on macOS the Objective-C autorelease
    /// pool core's pools push, as `MetalPlatform::new` does, as IOSurface and Foundation still
    /// autorelease into it.
    pub fn new(launcher: Box<dyn TaskLauncher<VulkanPlatform>>) -> VulkanPlatform {
        // metal2vulkan's own work counters join the work line.
        let _ = reims_vgpu_core::perfstats::EXTRA.set(metal2vulkan_runtime::perfcount::take_line);
        crate::os::install_process_services();
        VulkanPlatform {
            surfaces: OsSurfaces::new(),
            vm: VulkanVm::new(),
            clock: OsClock::new(),
            threads: OsThreads::new(),
            globals: ProcessState::new(),
            launcher,
        }
    }

    /// The runtime's system default device (`MTLCreateSystemDefaultDevice`), for tests; `None`
    /// where the runtime finds no Vulkan device it can use, or no Vulkan loader. Any other
    /// refusal dies here, naming it ([`made_devices`]).
    pub fn system_default_device() -> Option<VulkanDevice> {
        made_devices("MTLCreateSystemDefaultDevice()", || {
            rt::Device::system_default_with(device_options()).map(|d| vec![d])
        })
        .pop()
        .map(|raw| device::adopt("MTLCreateSystemDefaultDevice()", raw))
    }
}

/// The devices `make` makes, a creator standing for `sel` (`MTLCreateSystemDefaultDevice()`,
/// `MTLCopyAllDevices()`): none where the runtime finds no Vulkan device it can use, or no Vulkan
/// loader. Any other refusal dies at `sel`, naming it: a macOS host below every release whose
/// Metal the runtime measured among them, or whose release does not read
/// (`Problem::HostReleaseUnmeasured`), where the runtime refuses every device, as it cannot answer
/// that release's Metal. On a host of a release the runtime has not measured above one it has,
/// the runtime makes the devices at the newest measured release below the host's and announces
/// it as it makes one, on this thread (`MetalLevelBelowHost`, [`crate::answer::HEARD`]): the
/// devices are made, and [`BELOW_HOST`] keeps it, so each answer that depends on the release dies
/// where it is asked ([`answering_level`]). A device that lists a window extension the options
/// name but lacks its feature (`Error::ExtensionFeature`, `swapchainMaintenance1`) dies at `sel`
/// too, naming the extension and the feature: the runtime refuses it rather than enable the
/// extension without the feature, and the window's probe names the extension only where every
/// device has the feature, asked as the runtime asks ([`crate::os::add_window_extensions`]). So
/// do options that name a device extension without an instance extension it requires
/// (`Error::InstanceExtensionRequired`), which the runtime checks before it makes the instance:
/// the probe names each window extension with those it requires.
fn made_devices(
    sel: &'static str,
    make: impl FnOnce() -> Result<Vec<rt::Device>, rt::Error>,
) -> Vec<rt::Device> {
    let (made, heard) = crate::answer::hearing(|| {
        get(sel, || match make() {
            Err(rt::Error::NoDevice | rt::Error::LoaderNotFound(_)) => Ok(Vec::new()),
            made => made,
        })
    });
    if heard.contains(&rt::Degradation::MetalLevelBelowHost) {
        // Relaxed: stored before the level is kept ([`answered_at`], a `OnceLock`, whose read
        // acquires it), and read after it.
        BELOW_HOST.store(true, Ordering::Relaxed);
    }
    made
}

/// Whether the runtime announced that its devices answer as a release below the host's
/// (`MetalLevelBelowHost`, [`made_devices`]): the host's release is not one it measured.
static BELOW_HOST: AtomicBool = AtomicBool::new(false);

/// The Metal level of an answer `sel` gives that depends on it ([`metal_level`]): `what`, which
/// the releases the runtime measured answer otherwise. On a host whose own release the runtime
/// has not measured ([`BELOW_HOST`]) the devices answer as the newest measured release below it,
/// and the host's Metal may answer otherwise, so `sel` relays it there, naming `what` and the
/// level, and dies in strict mode ([`crate::answer::Inexact::MetalLevel`]). The level either way.
pub(crate) fn answering_level(sel: &'static str, what: impl FnOnce() -> String) -> rt::MetalLevel {
    let level = metal_level(sel);
    if BELOW_HOST.load(Ordering::Relaxed) {
        crate::answer::inexact(
            sel,
            crate::answer::Inexact::MetalLevel,
            &format!(
                "{}, which Metal's releases answer differently: this host's macOS release is not \
                 one the Vulkan runtime measured, and its devices answer as {level}, the newest \
                 measured release below it",
                what()
            ),
        );
    }
    level
}

/// The Metal level every device of this process answers at (`rt::DeviceProperties::metal_level`):
/// the macOS release whose Metal the runtime answers as where releases differ, which pixel format
/// values a texture descriptor takes ([`rt::MetalLevel::knows`]) and which formats Metal describes
/// ([`rt::MetalLevel::describes`]). [`device_options`] names the OS's
/// ([`crate::os::METAL_LEVEL`]): on macOS none, so it is the host's release, or the newest
/// measured release below it; on Linux macOS 26.7, the contract machine's. The same for every
/// device. The first device the platform adopts sets it ([`answered_at`]); read before any is,
/// it is the runtime's system default device's, made for it, which dies as [`made_devices`]
/// does where the runtime makes none, at `sel`, the call that asks.
pub(crate) fn metal_level(sel: &'static str) -> rt::MetalLevel {
    if let Some(&level) = METAL_LEVEL.get() {
        return level;
    }
    let made = made_devices(sel, || {
        rt::Device::system_default_with(device_options()).map(|d| vec![d])
    });
    let Some(device) = made.first() else {
        refused(sel, &rt::Error::NoDevice)
    };
    let level = device.properties().metal_level;
    answered_at(sel, level);
    level
}

/// The level of [`metal_level`], once known.
static METAL_LEVEL: std::sync::OnceLock<rt::MetalLevel> = std::sync::OnceLock::new();

/// Keeps `level`, a device's, as the process's Metal level ([`metal_level`]), and logs it the
/// first time. A device of another level than the process's answers as its own, relayed at
/// `sel`, the call that made it, and dies there in strict mode
/// ([`crate::answer::Inexact::MetalLevel`]; unreachable: every device here is made with
/// [`device_options`]).
pub(crate) fn answered_at(sel: &'static str, level: rt::MetalLevel) {
    let kept = *METAL_LEVEL.get_or_init(|| {
        log::info!(
            target: "reims_vgpu_vulkan::platform",
            "the Vulkan runtime's devices answer Metal's pixel formats as {level}"
        );
        level
    });
    if kept != level {
        crate::answer::inexact(
            sel,
            crate::answer::Inexact::MetalLevel,
            &format!(
                "a device answering Metal as {level}, where this process's devices answer as {kept}"
            ),
        );
    }
}

/// The options every device is made with: the runtime's defaults, whose loader search finds
/// Homebrew's (`vulkan-loader`, beside its `molten-vk`), with the window system's extensions where
/// a display's window presents through Vulkan (on Linux, [`crate::os::add_window_extensions`]), the
/// OS's Metal level ([`crate::os::METAL_LEVEL`]: on Linux macOS 26.7, the contract machine's, where
/// the runtime's default off macOS is its newest measured release), the shader cache in
/// `~/.cache/reims-vgpu-vulkan` ([`SHADER_CACHE`]; the runtime's default directory when `HOME` is
/// not set), which every process shares, holding [`SHADER_CACHE_CAPACITY`], a command buffer's
/// GPU times answered from the host's clock (`gpu_times_from_host`), the runtime's device policy,
/// its fast native forms but those `REIMS_VGPU_EXACT` names ([`exact_policy`]), and the
/// degradations the
/// runtime announces sent to the answer layer ([`crate::answer::notice`]), where those that reach
/// what the guest sees are counted, and die in strict mode ([`crate::answer::strict`]), each
/// announced at every occurrence (`announce_each`, [`crate::answer::DIES`]), as are those a
/// creator learns of ([`crate::answer::HEARD`]). One set,
/// so the runtime keeps one device per physical device across `open_device` and every other open.
///
/// **GPU times from the host.** No command buffer writes GPU timestamps: its `GPUStartTime` is the
/// host time its submission ended (`kernelEndTime`, or its queue's latest GPU end before it,
/// where later, so a queue's command buffers do not overlap) and its `GPUEndTime` the host time
/// the runtime saw it complete. On MoltenVK 1.4.1 the runtime answers so whatever the option, as
/// its timestamps wait for every queue's (the runtime's `TimestampsWaitForEveryQueue`). Only this
/// platform's debug lines read them ([`crate::gpu::timing`]): core's traits answer no command
/// buffer's GPU times, so neither the guest nor the host sees them. The timestamps cost latency:
/// measured on the M3 Max (a release build, under a load average of 34), a command buffer of one
/// blit fill completed a median of 283 to 309 us after its commit, against 356 to 399 us with
/// them (`timing`'s `gpu_times_from_host_latency`).
pub(crate) fn device_options() -> &'static rt::DeviceOptions {
    static OPTIONS: LazyLock<rt::DeviceOptions> = LazyLock::new(|| {
        let mut options = rt::DeviceOptions::default();
        crate::os::add_window_extensions(&mut options);
        options.metal_level = crate::os::METAL_LEVEL;
        options.gpu_times_from_host = true;
        options.policy = exact_policy(std::env::var("REIMS_VGPU_EXACT").ok().as_deref());
        options.messages = Some(rt::MessageSink::new(crate::answer::notice));
        options.announce_each = crate::answer::DIES
            .into_iter()
            .chain(crate::answer::HEARD)
            .collect();
        if let Some(cache_dir) = std::env::var_os("REIMS_VGPU_CACHE_DIR") {
            options.shader_cache = rt::ShaderCacheOptions::in_directory(PathBuf::from(cache_dir));
        } else if let Some(home) =
            std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
        {
            options.shader_cache =
                rt::ShaderCacheOptions::in_directory(PathBuf::from(home).join(SHADER_CACHE));
        }
        options.shader_cache.capacity = SHADER_CACHE_CAPACITY;
        options
    });
    &OPTIONS
}

/// The fields of the runtime's device policy (`metal2vulkan::device::DevicePolicy`) whose exact
/// form `REIMS_VGPU_EXACT` can name, each by the field's own name: Metal's sampler emulated
/// (`sampling`, `SamplingPolicy::Exact`), the translation's routines for the maths functions,
/// divisions and fused multiply-adds Metal answers beyond Vulkan's precision table
/// (`float_maths`, `FloatMathsPolicy::Exact`), Metal's texel conversion (`texel_conversion`,
/// `TexelConversionPolicy::Exact`: every read's, and a sample's or gather's with `sampling`),
/// Metal's `half` rounding (`half_rounding`, `HalfRoundingPolicy::Exact`), Metal's NaN bits and
/// the operand its minimum and maximum functions pick at a NaN or a signed zero
/// (`nan_and_min_max`, `NanMinMaxPolicy::Exact`), Metal's rounding order of `dot`, `mix` and
/// `fast::sqrt` (`rounding_order`, `RoundingOrderPolicy::Exact`), Metal's `half` blend of a
/// safe-math `half4` output into a 16-bit float attachment, which the function computes
/// (`half_blend`, `HalfBlendPolicy::Exact`), and the runtime's shadow copy of a texture that only
/// reads over a buffer whose rows lie with bytes between them, so that no byte between the rows is
/// reached through an image (`padded_reads`, `PaddedReadsPolicy::Shadow`). [`EXACT_ALL`] names
/// every one but `sampling`.
pub const EXACT_FORMS: [&str; 8] = [
    "sampling",
    "float_maths",
    "texel_conversion",
    "half_rounding",
    "nan_and_min_max",
    "rounding_order",
    "half_blend",
    "padded_reads",
];

/// The name `REIMS_VGPU_EXACT` takes for the runtime's `DevicePolicy::exact_answers()`: every
/// field of [`EXACT_FORMS`] but `sampling` in its exact form, as the runtime's validation sets
/// them. `sampling` may be named beside it.
pub const EXACT_ALL: &str = "all";

/// The device policy every device is made with: the runtime's default, whose native forms are
/// fast to make and run and record their bounds off Apple's GPUs (docs/accuracy-ledger.md: the
/// fast method by default, the accurate one behind a switch), but for each field of
/// [`EXACT_FORMS`] that `exact`, `REIMS_VGPU_EXACT`'s comma-separated list, names, which takes
/// the runtime's exact form, and for [`EXACT_ALL`], which takes every such form but `sampling`'s:
/// larger modules and longer pipeline creation, and `padded_reads`' copy before each use. A
/// device that runs on Metal (MoltenVK) answers as Metal under either, but for the NaN a `half`
/// read, sample or gather hands back for a NaN texel, Metal's only under `nan_and_min_max`. A
/// name of neither changes nothing, and a line on stderr says so.
fn exact_policy(exact: Option<&str>) -> metal2vulkan::device::DevicePolicy {
    use metal2vulkan::device::{
        DevicePolicy, FloatMathsPolicy, HalfBlendPolicy, HalfRoundingPolicy, NanMinMaxPolicy,
        PaddedReadsPolicy, RoundingOrderPolicy, SamplingPolicy, TexelConversionPolicy,
    };
    let mut policy = DevicePolicy::default();
    for name in (exact.unwrap_or_default().split(',').map(str::trim)).filter(|n| !n.is_empty()) {
        match name {
            EXACT_ALL => policy = policy.exact_answers(),
            "sampling" => policy.sampling = SamplingPolicy::Exact,
            "float_maths" => policy.float_maths = FloatMathsPolicy::Exact,
            "texel_conversion" => policy.texel_conversion = TexelConversionPolicy::Exact,
            "half_rounding" => policy.half_rounding = HalfRoundingPolicy::Exact,
            "nan_and_min_max" => policy.nan_and_min_max = NanMinMaxPolicy::Exact,
            "rounding_order" => policy.rounding_order = RoundingOrderPolicy::Exact,
            "half_blend" => policy.half_blend = HalfBlendPolicy::Exact,
            "padded_reads" => policy.padded_reads = PaddedReadsPolicy::Shadow,
            other => {
                let line = format!(
                    "reims-vgpu: REIMS_VGPU_EXACT names {other:?}, none of {EXACT_ALL}, {}: \
                     ignored\n",
                    EXACT_FORMS.join(", ")
                );
                let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
            }
        }
    }
    policy
}

/// The shader cache's directory under `HOME`, which every process of the library shares. It
/// holds the runtime's translations of the guest's functions and each pipeline's driver record
/// (MoltenVK's included: about 1.7 MB for a macOS 26 guest's `Xgc` pipeline
/// on NVIDIA), within [`SHADER_CACHE_CAPACITY`], the least recently used going first.
#[cfg(not(test))]
const SHADER_CACHE: &str = ".cache/reims-vgpu-vulkan";

/// The most bytes [`SHADER_CACHE`] holds: the runtime's default, 25 GiB (a test keeps it so). A
/// macOS 26 guest's built-in apps, each launched once, need more than 1 GiB of translations and
/// driver records on an RTX 5080 laptop. The runtime seeds each pipeline's making from its own
/// record here; the driver's own disk cache is left as the driver keeps it, and may
/// hold the pipeline too. An entry this cache drops past its capacity is compiled again at the
/// next boot that draws with it, unless the driver's cache still holds it: a freeze of 13 to 26 s
/// there.
#[cfg(not(test))]
const SHADER_CACHE_CAPACITY: u64 = rt::DEFAULT_CAPACITY;

/// The unit tests' capacity, 1 GiB: their directory holds what the tests compile and nothing a
/// guest's boot reuses, so it is kept small on disk.
#[cfg(test)]
const SHADER_CACHE_CAPACITY: u64 = 1 << 30;

/// The unit tests' shader cache directory under `HOME`, their own: a test of what a device
/// compiles runs on an empty one as on a full one, and the tests write nothing into the one the
/// library's hosts share.
#[cfg(test)]
const SHADER_CACHE: &str = ".cache/reims-vgpu-vulkan-tests";

/// The runtime's registry ID of its system default device, read once: `None` where it has none.
pub(crate) fn default_device_registry_id() -> Option<u64> {
    static DEFAULT: LazyLock<Option<u64>> = LazyLock::new(|| {
        rt::Device::system_default_with(device_options())
            .ok()
            .map(|d| d.registry_id())
    });
    *DEFAULT
}

impl Platform for VulkanPlatform {
    type Device = VulkanDevice;
    type CommandQueue = VulkanCommandQueue;
    type CommandBuffer = VulkanCommandBuffer;
    type BlitEncoder = VulkanBlitEncoder;
    type ComputeEncoder = VulkanComputeEncoder;
    type RenderEncoder = VulkanRenderEncoder;
    type Buffer = VulkanBuffer;
    type Texture = VulkanTexture;
    type Heap = VulkanHeap;
    type SamplerState = VulkanSamplerState;
    type DepthStencilState = VulkanDepthStencilState;
    type Fence = VulkanFence;
    type Event = VulkanEvent;
    type SharedEvent = VulkanSharedEvent;
    type SharedEventHandle = VulkanSharedEventHandle;
    type SharedEventListener = VulkanSharedEventListener;
    type SharedTextureHandle = VulkanSharedTextureHandle;
    type Library = VulkanLibrary;
    type Function = VulkanFunction;
    type ComputePipeline = VulkanComputePipeline;
    type RenderPipeline = VulkanRenderPipeline;
    type RateMap = VulkanRateMap;
    type Icb = VulkanIcb;
    type IndirectRenderCommand = VulkanIndirectRenderCommand;
    type IndirectComputeCommand = VulkanIndirectComputeCommand;
    type TextureDescriptor = VulkanTextureDescriptor;
    type RenderPassDescriptor = VulkanRenderPassDescriptor;
    type RenderPipelineDescriptor = VulkanRenderPipelineDescriptor;
    type ComputePipelineDescriptor = VulkanComputePipelineDescriptor;
    type TileRenderPipelineDescriptor = VulkanTileRenderPipelineDescriptor;
    type RateMapDescriptor = VulkanRateMapDescriptor;
    type RateMapLayerDescriptor = VulkanRateMapLayerDescriptor;
    type AddressRangeArray = VulkanAddressRangeArray;
    type Surface = OsSurface;
    type Clock = OsClock;

    /// A descriptor of Metal's defaults.
    fn new_render_pass_descriptor(&self) -> VulkanRenderPassDescriptor {
        Obj::new(plain(RenderPassDescriptorObject::new))
    }

    /// A listener with a thread of its own, which core never posts to.
    fn new_shared_event_listener(&self) -> VulkanSharedEventListener {
        Obj::new(get("new", rt::SharedEventListener::new))
    }

    /// A descriptor of Metal's defaults, as `+new` makes.
    fn render_pass_descriptor(&self) -> VulkanRenderPassDescriptor {
        Obj::new(plain(RenderPassDescriptorObject::new))
    }

    /// A 2D descriptor with the full chain of levels when `mipmapped`, its pixel format taken as
    /// `setPixelFormat:` takes it: one the runtime lacks is kept raw, and a texture of it dies
    /// where it is made ([`crate::gpu::desc::lacked`]).
    fn texture_2d_descriptor(
        &self,
        _on: OnFault,
        pixel_format: u64,
        width: u64,
        height: u64,
        mipmapped: bool,
    ) -> RaiseResult<VulkanTextureDescriptor> {
        let sel = "texture2DDescriptorWithPixelFormat:width:height:mipmapped:";
        let (format, ycbcr_format, no_texture_format, lacked_format) =
            crate::gpu::desc::format_set(sel, pixel_format);
        let desc = plain(|| rt::TextureDescriptor::texture_2d(format, width, height, mipmapped));
        Ok(Obj::new(TextureDescriptorObject::new(TextureFields {
            desc,
            ycbcr_format,
            no_texture_format,
            lacked_format,
            ..TextureFields::default()
        })))
    }

    /// A layer of zeroed samples, nil for a count of 0 on either axis
    /// ([`crate::gpu::rate_map`]).
    fn new_rate_map_layer(
        &self,
        _on: OnFault,
        sample_count: MtlSize,
    ) -> RaiseResult<Option<VulkanRateMapLayerDescriptor>> {
        Ok(crate::gpu::rate_map::new_layer(sample_count))
    }

    /// A copy of `ranges`, as Metal's array copies them.
    fn new_address_range_array(
        &self,
        ranges: &[ResourceAddressRange],
    ) -> RaiseResult<VulkanAddressRangeArray> {
        Ok(Obj::new(AddressRanges {
            ranges: ranges.to_vec(),
        }))
    }

    /// A new descriptor with the value's properties, set in its order.
    fn texture_descriptor(&self, desc: &TextureDesc) -> VulkanTextureDescriptor {
        Obj::new(TextureDescriptorObject::new(TextureFields::of(desc)))
    }

    fn rate_map_descriptor(&self, desc: &RateMapDesc) -> VulkanRateMapDescriptor {
        crate::gpu::rate_map::new_descriptor(desc)
    }

    /// The device of the runtime's `Device::all_with` (`MTLCopyAllDevices`) whose registry ID, as
    /// this platform answers it ([`device::registry_id`]: Metal's for the default device), is
    /// `registry_id` (+1), or `None`, as where there is no Vulkan loader. In one process it is the
    /// object every other handle to that device is ([`device::adopt`]).
    fn open_device(&self, registry_id: u64) -> Option<VulkanDevice> {
        let all = made_devices("MTLCopyAllDevices()", || {
            rt::Device::all_with(device_options())
        });
        all.into_iter()
            .find(|d| device::registry_id(d) == registry_id)
            .map(|raw| device::adopt("MTLCopyAllDevices()", raw))
    }

    fn surfaces(&self) -> &dyn IoSurfaces<VulkanPlatform> {
        &self.surfaces
    }

    fn vm(&self) -> &dyn HostVm {
        &self.vm
    }

    fn clock(&self) -> &OsClock {
        &self.clock
    }

    fn threads(&self) -> &dyn Threads {
        &self.threads
    }

    fn globals(&self) -> &ProcessState<VulkanPlatform> {
        &self.globals
    }

    fn launcher(&self) -> &dyn TaskLauncher<VulkanPlatform> {
        &*self.launcher
    }

    /// The host's blocks, those that change pages reporting what they changed to every device
    /// ([`remap::reporting`]).
    fn host_blocks(&self, blocks: HostBlocks) -> HostBlocks {
        remap::reporting(blocks, Arc::new(device::host_range_remapped))
    }
}

/// The private `MTLResourceAddressRangeArray *`: the ranges, as the array copies them, which a
/// heap is made over and a detached buffer or heap is replaced by ([`crate::gpu::backing`]).
#[derive(Debug)]
pub struct AddressRanges {
    /// The ranges, in order.
    pub ranges: Vec<ResourceAddressRange>,
}

/// The private `MTLResourceAddressRangeArray *`.
pub type VulkanAddressRangeArray = Obj<AddressRanges>;

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::tasks::launcher::{HostSide, ServiceLink};

    use super::*;
    use crate::gpu::device::tests::dev;

    /// A launcher no test launches through.
    struct NoLauncher;

    impl TaskLauncher<VulkanPlatform> for NoLauncher {
        fn launch(
            &self,
            _platform: &Arc<VulkanPlatform>,
            _host: Arc<dyn HostSide<VulkanPlatform>>,
        ) -> Arc<dyn ServiceLink<VulkanPlatform>> {
            panic!("no task is launched in these tests")
        }
    }

    pub(crate) fn platform() -> VulkanPlatform {
        VulkanPlatform::new(Box::new(NoLauncher))
    }

    /// Without `REIMS_VGPU_EXACT` the device policy is the runtime's default, its native forms;
    /// each name of `EXACT_FORMS` takes that field's exact form alone, in any order, spaces
    /// around a name ignored; `all` takes the runtime's `exact_answers()`, every one but
    /// `sampling`; another name changes nothing
    #[test]
    fn reims_vgpu_exact_names_the_policys_exact_forms() {
        use metal2vulkan::device::{
            DevicePolicy, FloatMathsPolicy, HalfBlendPolicy, HalfRoundingPolicy, NanMinMaxPolicy,
            PaddedReadsPolicy, RoundingOrderPolicy, SamplingPolicy, TexelConversionPolicy,
        };
        let fast = DevicePolicy::default();
        assert_eq!(fast.sampling, SamplingPolicy::NativeDevice);
        assert_eq!(fast.float_maths, FloatMathsPolicy::Native);
        assert_eq!(fast.texel_conversion, TexelConversionPolicy::Device);
        assert_eq!(fast.half_rounding, HalfRoundingPolicy::Native);
        assert_eq!(fast.nan_and_min_max, NanMinMaxPolicy::Native);
        assert_eq!(fast.rounding_order, RoundingOrderPolicy::Native);
        assert_eq!(fast.half_blend, HalfBlendPolicy::Device);
        assert_eq!(fast.padded_reads, PaddedReadsPolicy::Alias);
        for unchanged in [None, Some(""), Some(","), Some("exact, bicubic, All")] {
            assert_eq!(exact_policy(unchanged), fast, "{unchanged:?}");
        }
        let one = |name| exact_policy(Some(name));
        type Set = fn(&mut DevicePolicy);
        let alone: [(&str, Set); 8] = [
            ("sampling", |p| p.sampling = SamplingPolicy::Exact),
            ("float_maths", |p| p.float_maths = FloatMathsPolicy::Exact),
            ("texel_conversion", |p| {
                p.texel_conversion = TexelConversionPolicy::Exact;
            }),
            ("half_rounding", |p| {
                p.half_rounding = HalfRoundingPolicy::Exact
            }),
            ("nan_and_min_max", |p| {
                p.nan_and_min_max = NanMinMaxPolicy::Exact
            }),
            ("rounding_order", |p| {
                p.rounding_order = RoundingOrderPolicy::Exact
            }),
            ("half_blend", |p| p.half_blend = HalfBlendPolicy::Exact),
            ("padded_reads", |p| {
                p.padded_reads = PaddedReadsPolicy::Shadow
            }),
        ];
        assert_eq!(alone.map(|(name, _)| name), EXACT_FORMS);
        let mut every = fast;
        for (name, set) in alone {
            let mut want = fast;
            set(&mut want);
            assert_ne!(want, fast, "{name}");
            assert_eq!(one(name), want, "{name}");
            set(&mut every);
        }
        let named = " half_blend,half_rounding,sampling , rounding_order,texel_conversion,\
                     nan_and_min_max,padded_reads,float_maths";
        assert_eq!(exact_policy(Some(named)), every);
        assert_eq!(exact_policy(Some(&EXACT_FORMS.join(","))), every);
        let answers = fast.exact_answers();
        assert_eq!(one(EXACT_ALL), answers);
        assert_eq!(answers.sampling, SamplingPolicy::NativeDevice);
        let mut but_sampling = every;
        but_sampling.sampling = SamplingPolicy::NativeDevice;
        assert_eq!(answers, but_sampling);
        assert_eq!(exact_policy(Some("sampling, all")), every);
        assert_eq!(exact_policy(Some("all,sampling")), every);
    }

    /// Obj: clone_ref() is another reference to the same object, and the last drop releases it
    #[test]
    fn obj_references_one_object() {
        let o = Obj::new(7u32);
        let weak = o.downgrade();
        let r = o.clone_ref();
        assert_eq!(r.id(), o.id());
        assert_eq!(*r, 7);
        assert_ne!(Obj::new(7u32).id(), o.id());
        drop(o);
        assert!(Obj::upgrade(&weak).is_some());
        drop(r);
        assert!(Obj::upgrade(&weak).is_none());
        assert!(format!("{:?}", Obj::new(3u8)).ends_with(" 3>"));
    }

    /// Obj: a handle an isolated task's death unwinds past is leaked, not released
    #[test]
    fn a_dead_tasks_unwind_leaks_its_handles() {
        use reims_vgpu_core::raise::death::{DeathKind, DeathSite};
        use reims_vgpu_core::tasks::isolation::TaskLife;

        let o = Obj::new(1u8);
        let weak = o.downgrade();
        let life = TaskLife::new();
        let r: Option<()> = life.run(|| {
            let _held = o.clone_ref();
            reims_vgpu_core::fatal!(DeathSite {
                kind: DeathKind::Abort,
                file: "platform.test",
                line: 0,
            })
        });
        assert_eq!(r, None);
        drop(o);
        assert!(
            Obj::upgrade(&weak).is_some(),
            "the task's reference was leaked"
        );
    }

    /// A device creator dies at its selector where the runtime refuses its devices otherwise than
    /// for having none, naming the refusal: a macOS host below every release whose Metal the
    /// runtime measured among them (`Problem::HostReleaseUnmeasured`); with no device or no
    /// loader it makes none
    #[test]
    fn a_host_whose_metal_the_runtime_cannot_answer_dies_at_the_creator() {
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;
        for sel in ["MTLCreateSystemDefaultDevice()", "MTLCopyAllDevices()"] {
            let unmeasured = rt::Error::Unproven(rt::Unproven::new(
                "all_with",
                rt::Problem::HostReleaseUnmeasured {
                    macos: Some([15, 6, 1]),
                },
            ));
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| made_devices(sel, || Err(unmeasured))).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(line.contains("HostReleaseUnmeasured"), "{line}");
            assert!(made_devices(sel, || Err(rt::Error::NoDevice)).is_empty());
        }
    }

    /// On a host whose macOS release the runtime has not measured, above one it has, the device
    /// is made, at the newest measured release below the host's (`MetalLevelBelowHost`, kept by
    /// `made_devices`, here set by hand, as this host's release is measured). An answer the
    /// measured releases give otherwise dies where it is asked, naming it: the setter's at a
    /// three-channel format (45, `RGB8Unorm`), a view's nil in 630, a shared texture's nil at 630
    /// and `Private` storage, a texture of 630 and its heap size, which macOS 26.7 asserts at and
    /// 27.0.1 answers, 630's description and a level's layout of it, which 27.0.1 has and 26.7
    /// does not, and 586's description, whose flags differ. Every other answer is the level's,
    /// with no death: the setter's at 80 and 630, a view in 81, a shared texture of 80, and 80's
    /// description, whose cast class differs, which core does not read, and a level's layout of
    /// it. The release is set back to a measured one as the test ends, whether it passes or not
    #[test]
    fn an_unmeasured_release_dies_only_where_an_answer_depends_on_it() {
        use crate::gpu::format_info::setter_takes;
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Texture as _;
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;
        /// Sets the release back to a measured one when dropped.
        struct Measured;
        impl Drop for Measured {
            fn drop(&mut self) {
                BELOW_HOST.store(false, Ordering::Relaxed);
            }
        }
        let d = dev();
        let _measured = Measured;
        BELOW_HOST.store(true, Ordering::Relaxed);
        let p = platform();
        let desc = |pixel_format, resource_options| {
            p.texture_descriptor(&TextureDesc {
                pixel_format,
                width: 4,
                height: 4,
                usage: 1,
                resource_options: Some(resource_options),
                ..TextureDesc::default()
            })
        };
        assert!(setter_takes("setPixelFormat:", 80));
        assert!(setter_takes("setPixelFormat:", 630));
        let t = d
            .new_texture(OnFault::Return, &desc(80, 0))
            .unwrap()
            .unwrap();
        assert!(t.new_texture_view(81).unwrap().is_some());
        assert!(d.new_shared_texture(&desc(80, 0x20)).unwrap().is_some());
        let level = |pixel_format| reims_vgpu_core::platform::types::TextureLevelQuery {
            pixel_format,
            width: 4,
            height: 4,
            depth: 1,
            array_length: 1,
            options: 0,
            flags: 0,
        };
        assert_eq!(d.pixel_format_info(80).block_bytes, 4);
        assert_eq!(d.texture_level_info(&level(80)).bytes_per_row, 16);
        let shared = desc(630, 0x20);
        let plain = desc(630, 0);
        let described = "MTLPixelFormatGetInfoForDevice()";
        let laid_out = "MTLGetTextureLevelInfoForDeviceWithOptions()";
        let cases: [(&str, &dyn Fn()); 8] = [
            ("setPixelFormat:", &|| {
                setter_takes("setPixelFormat:", 45);
            }),
            ("newTextureViewWithPixelFormat:", &|| {
                drop(t.new_texture_view(630));
            }),
            ("newSharedTextureWithDescriptor:", &|| {
                drop(d.new_shared_texture(&shared));
            }),
            ("newTextureWithDescriptor:", &|| {
                drop(d.new_texture(OnFault::Return, &plain));
            }),
            ("heapTextureSizeAndAlignWithDescriptor:", &|| {
                drop(d.heap_texture_size_and_align(&plain));
            }),
            (described, &|| {
                d.pixel_format_info(630);
            }),
            (described, &|| {
                d.pixel_format_info(586);
            }),
            (laid_out, &|| {
                d.texture_level_info(&level(630));
            }),
        ];
        for (sel, ask) in cases {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(ask).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(
                line.contains("not one the Vulkan runtime measured"),
                "{line}"
            );
        }
    }

    /// open_device finds the default device by the registry ID the platform answers for it
    /// (Metal's default device's on macOS, the runtime's own where no Metal stands for it), as
    /// the one object every handle to it is; None for an unknown ID, and, where Metal's stands
    /// for it, for the runtime's own ID, which no host sees
    #[test]
    fn open_device_finds_the_device() {
        let p = platform();
        let d = dev();
        let own = d.raw.registry_id();
        let id = crate::os::default_registry_id().unwrap_or(own);
        let found = p.open_device(id).unwrap();
        assert_eq!(found.id(), d.id());
        assert_eq!(d.registry_id, id);
        assert!(p.open_device(id ^ 0xdead_beef).is_none());
        assert_eq!(p.open_device(own).is_some(), id == own);
    }

    /// The library's shader cache holds the runtime's default of 25 GiB, sized for a macOS 26
    /// guest's apps on an RTX 5080 laptop: a pin whose default differs fails here,
    /// for review. The unit tests' own directory holds 1 GiB
    #[test]
    fn the_shader_cache_holds_25_gib() {
        assert_eq!(rt::DEFAULT_CAPACITY, 25 << 30);
        assert_eq!(device_options().shader_cache.capacity, 1 << 30);
    }

    /// The services are the ones the platform was made with
    #[test]
    fn services() {
        let p = platform();
        let _ = (p.surfaces(), p.vm(), p.clock(), p.threads());
        let _ = (p.globals(), p.launcher());
        assert!(format!("{p:?}").contains("VulkanPlatform"));
        let page = if cfg!(target_os = "macos") {
            16384
        } else {
            4096
        };
        assert_eq!(p.vm().page_size(), page);
    }
}
