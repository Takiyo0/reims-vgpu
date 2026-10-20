# Minimum Windows windowing for a Vulkan-presentable VMM window

Scope: find the *smallest* `reims-vgpu-window` Windows backend that lets the VMM
build and open a Vulkan-presentable window on this host (i7-13650HX / RTX 4060),
and give a staged plan. No broad rewrite. Nothing implemented here beyond what a
small reviewable prototype needs. Local changes preserved; no commit/push/remote.

## Headline

The crate is cleanly split into **OS-agnostic logic** (traits + geometry +
policy + scroll + keys, all unit-tested) and a thin **per-OS native backend**.
The Vulkan surface/swapchain/present machinery for Windows is already wired in
`reims-vgpu-vulkan` (the instance/device extension set and loader). What is
missing is the whole `native/windows.rs` backend: a Win32 window, a message
loop, input mapping, monitor/DPI handling, a `VulkanTarget` over an `HWND`, and a
safe shutdown ordering. That is the main *integration seam*, but it is still
**significant work** — a Win32 event loop, scancode→USB mapping, per-monitor DPI,
and a correct destruction order are each non-trivial. The value of the staged
plan is that Stage 0 and Stage 1 test those seams cheaply before committing to
the rest. There is **no architectural blocker** (unlike the WHPX/QEMU path), but
"no blocker" is not "small".

Recommended API: **raw Win32 via `windows-sys`** (already a workspace dep used by
the Vulkan backend), surface via **`VK_KHR_win32_surface`** (already wired). Do
**not** add SDL3 on Windows — there are no SDL3 dev libs on this box and the
native path removes the dependency the crate is trying to shed anyway.

## 1. What exists, and the exact seam

### 1a. The crate is logic + a thin native shim
`crates/reims-vgpu-window/src/`:
- OS-agnostic, already unit-tested, reused unchanged on Windows: `geometry.rs`,
  `policy.rs`, `scroll.rs`, `keys.rs`, `keymap.rs` (macOS-keycode→USB; a Windows
  backend maps its own scancodes→USB and reuses the USB side), `capture.rs`,
  `config.rs`, `env.rs`, `screens.rs`, `present.rs`, `handle.rs`.
- `present.rs` defines the whole present abstraction OS-agnostically:
  `trait Presenter`, `trait VulkanTarget { create_surface(gipa, instance) ->
  Option<vk::SurfaceKHR> }`, `Presenting`, `Drawn`. **A Windows backend writes no
  present logic** — it only implements `VulkanTarget` for its window handle.
- `handle.rs` is the public `Window` the VMM holds. It calls into the backend
  only through `native::{open, close, cursor_changed, guest_mode_changed}` and
  holds a `native::Surface` and `native::FrameSink` (`handle.rs:30,33,192,
  212,237,252`). It is OS-agnostic.

### 1b. The native dispatch — where Windows is absent
`crates/reims-vgpu-window/src/native.rs` has arms for `linux` and `macos` only.
The Linux arm (the one to parallel) re-exports exactly:
```
pub use linux::FrameSink;
pub(crate) use linux::{ Surface, close, cursor_changed, guest_mode_changed,
                        host_screens, open, sdl as sdl_os };
```
So a Windows backend must export the public `FrameSink` and the crate-internal
`Surface, open, close, cursor_changed, guest_mode_changed, host_screens`.
`sdl_os` is only used by the shared `sdl` module; a native-Win32 backend does
**not** compile `crate::sdl`, so the `windows` arm omits `sdl_os` (and the
`#[cfg(any(target_os="linux", feature="macos-sdl"))] mod sdl;` in `lib.rs` stays
false on Windows — nothing to change there).

### 1c. The Linux backend is the template, and it is small
`native/linux/` = `ui.rs` (SDL thread + event loop + screen enumeration),
`render.rs` (the `FrameSink`/`Surface` + render thread + `Target: VulkanTarget`
via `SDL_Vulkan_CreateSurface`), `input.rs`, `keymap.rs`, plus the shared
`src/sdl/` shell+cursor. The render-thread body (`render.rs::run`) and the
`Surface` state machine are **OS-agnostic SDL-free Rust** — they can be lifted
into Windows almost verbatim; only `Target::create_surface` and the UI/event/
screen functions are SDL-specific.

### 1d. The Vulkan surface side for Windows is ALREADY DONE
`crates/reims-vgpu-vulkan/src/os/windows.rs`:
- `WINDOW_SURFACE_EXTENSIONS = [ash::khr::win32_surface::NAME]` (`:83`)
- `WINDOW_DEVICE_EXTENSIONS = []` (`:86`)
- `open_loader()` → `ash::Entry::load_from("vulkan-1.dll")` (`:90-93`)
`crates/reims-vgpu-vulkan/src/os/wsi.rs::add_window_extensions` is OS-agnostic and
already consumes those. **The instance/device/swapchain path needs no change.**

### 1e. The VMM consumes the crate through a stable, OS-agnostic API
- `crates/reims-vgpu/src/display.rs:325` — `Window::new(wcfg, port, title)`,
  `window::tee(...)`, `window::host_screen::<P>(port)`, `window::host_matched(...)`.
- `crates/reims-vgpu/src/window.rs:72` — `window_sink::<P>(...) -> Option<Arc<dyn
  FrameSink>>`, and the comment at `:7,70` confirms the sink is
  `vulkan::SwapchainSink` **"on either OS"** — it draws into the window's
  swapchain via `VulkanTarget::create_surface`, host-independent. So once the
  Windows `VulkanTarget` exists, the existing sink presents to it unchanged.
- `crates/reims-vgpu/Cargo.toml` has `[target.'cfg(target_os="macos")']` and
  `[target.'cfg(target_os="linux")']` dep arms; it needs a `windows` arm that
  turns on `reims-vgpu-vulkan` (feature `vulkan`), mirroring Linux (where the
  library "builds only with it").

## 2. The SDL/AppKit/Linux assumptions to replace

| Assumption | Where | Windows replacement |
|---|---|---|
| `sdl3-sys` dep + feature gating | `window/Cargo.toml` (`cfg(linux)` dep; `macos-sdl`) | new `[target.'cfg(windows)']` dep on `windows-sys` + `ash` + `reims-vgpu-vulkan` |
| SDL window/video/event loop | `native/linux/ui.rs` | Win32: `RegisterClassExW`, `CreateWindowExW`, a `GetMessage`/`DispatchMessage` loop on a dedicated UI thread |
| `SDL_Vulkan_CreateSurface` | `native/linux/render.rs::Target::create_surface` | `vkCreateWin32SurfaceKHR` with `HINSTANCE`+`HWND` (ext already enabled) |
| SDL display enumeration | `native/linux/ui.rs::list_screens` | `EnumDisplayMonitors` + `GetMonitorInfoW` + `GetDpiForMonitor` → `HostScreen` |
| SDL input events | `native/linux/input.rs`, `keymap.rs` | `WM_MOUSEMOVE/WM_*BUTTON*/WM_MOUSEWHEEL/WM_INPUT`, `WM_KEYDOWN`→scancode→USB |
| shared `crate::sdl` shell/cursor | `src/sdl/`, re-exported as `sdl_os` | not compiled on Windows; a native window shell/cursor live in `native/windows/` |
| AppKit main-thread run loop | `native/macos/*` | irrelevant on Windows; the Win32 message pump is its own UI thread, like Linux |

Reused as-is on Windows (no replacement needed): `present.rs`, `geometry.rs`,
`policy.rs`, `scroll.rs`, `keys.rs`, `capture.rs`, `config.rs`, `env.rs`,
`screens.rs`, `handle.rs`, and the `Surface`/render-thread body copied from
`native/linux/render.rs`.

## 3. API / library choice

- **Win32 via `windows-sys`** (workspace dep already; `reims-vgpu-vulkan` uses it).
  Features to add for the window arm: `Win32_UI_WindowsAndMessaging`,
  `Win32_Graphics_Gdi` (monitors), `Win32_UI_HiDpi` (`GetDpiForMonitor`),
  `Win32_UI_Input_KeyboardAndMouse`, `Win32_Foundation`, `Win32_System_LibraryLoader`
  (`GetModuleHandleW` for the `HINSTANCE`).
- **Surface**: `ash::khr::win32_surface` (already in `WINDOW_SURFACE_EXTENSIONS`).
  **How the function is resolved (correction):** `VulkanTarget::create_surface`
  is handed a raw `vk::Instance` and the `vk::PFN_vkGetInstanceProcAddr` the
  instance was made with — NOT an `ash::Entry` or `ash::Instance`. The SDL path
  sidesteps this (SDL owns its loader), but the Win32 path must use that resolver.
  Resolve `vkCreateWin32SurfaceKHR` through it, one of two equivalent ways:
  - build an `ash::khr::win32_surface::Instance` from a reconstructed
    `ash::Instance::load(&StaticFn { get_instance_proc_addr }, instance)` (or
    `ash::Instance::from_parts`-style load) and call `.create_win32_surface(...)`; or
  - call the raw `PFN_vkCreateWin32SurfaceKHR` fetched directly via
    `get_instance_proc_addr(instance, c"vkCreateWin32SurfaceKHR")`.
  Either fills `vk::Win32SurfaceCreateInfoKHR { hinstance, hwnd }` (hinstance from
  `GetModuleHandleW(null)`, hwnd the window's) and returns the `vk::SurfaceKHR` the
  caller then owns and destroys. Do not assume an ambient ash loader exists.
- **No SDL3, no winit.** SDL3 has no dev libs here and is the dependency being
  shed; winit would add a large new dep and its own event model. Raw Win32 is
  ~1 window class + 1 message loop + 1 surface call — smaller than adopting winit.

## 4. Affected files & dependencies

New (all under `crates/reims-vgpu-window/src/native/windows/`):
- `native/windows.rs` — facade: re-export `FrameSink`, `Surface`, `open`, `close`,
  `cursor_changed`, `guest_mode_changed`, `host_screens`.
- `native/windows/ui.rs` — UI thread, window class, `CreateWindowExW`, message
  pump, `post()` (parallel to `linux/ui.rs`), `WM_*`→callbacks, monitor
  enumeration.
- `native/windows/render.rs` — `FrameSink`, `Surface` (copy the OS-agnostic state
  machine from `linux/render.rs`), and `Target(HWND, HINSTANCE): VulkanTarget`
  with the `vkCreateWin32SurfaceKHR` body.
  **Caveat (not a verbatim copy):** `linux/render.rs::run` ends by posting
  `crate::sdl::shell::released(id)` to let SDL's window go. `crate::sdl` is not
  compiled on Windows, so the Windows `run` must post its OWN release to the
  Windows UI thread (`ui::post(move || shell_release(id))`) that then performs the
  deferred `DestroyWindow` (see §5 Stage 1 and the destruction-order requirement).
  The `Surface` state machine and the draw loop body ARE reusable as-is.
- `native/windows/input.rs`, `native/windows/keymap.rs` — scancode→USB, pointer,
  wheel→`crate::scroll`.

Edited (small):
- `crates/reims-vgpu-window/src/native.rs` — add the `#[cfg(target_os="windows")]`
  arm with the re-export list in §1b.
- `crates/reims-vgpu-window/Cargo.toml` — add `[target.'cfg(target_os="windows")'
  .dependencies]`: `reims-vgpu-vulkan`, `ash`, `windows-sys` (features above).
  (`ash` is already a top-level dep; `reims-vgpu-vulkan` currently only under
  `cfg(linux)`.)
- `crates/reims-vgpu/Cargo.toml` — add a `cfg(windows)` dep arm enabling
  `reims-vgpu-vulkan` (feature `vulkan`), mirroring Linux.
- Possibly `crates/reims-vgpu/src/lib.rs` — if it has a `cfg(linux) => require
  vulkan` guard, extend it to `windows`.

No change needed: `reims-vgpu-vulkan` (WSI already done), `present.rs`,
`handle.rs`, and all OS-agnostic modules.

Risk: `keymap.rs` on Linux maps macOS keycodes→USB; the Windows backend maps
Win32 scancodes→USB and feeds the SAME USB-usage path the guest expects
(`docs/qemu.md` "Input": USB usages). This is table work, not architecture.

## 5. Staged implementation + validation

Each stage is independently reviewable; stop/park at any boundary. Build with
`CARGO_TARGET_DIR` off the source tree.

**Stage 0 — compile the empty arm (prototype gate).**
Add the `native.rs` windows arm + a `native/windows.rs` whose functions are
`unimplemented!()`/no-ops and a `Surface`/`FrameSink` lifted from Linux's
`render.rs` (SDL-free parts). Add the two Cargo arms.
*Validate:* `cargo check -p reims-vgpu-window --target x86_64-pc-windows-msvc` and
`cargo check -p reims-vgpu --features vulkan --target x86_64-pc-windows-msvc`
compile. Proves the seam and dep graph are right before any Win32 code. **This is
the small reviewable prototype.**

**Stage 1 — window lifetime + Vulkan surface (the first runtime milestone).**
Implement `ui.rs` enough to open one `CreateWindowExW` window on a dedicated UI
thread with a `GetMessage`/`DispatchMessage` pump, and `Target::create_surface`
via the gipa-resolved `vkCreateWin32SurfaceKHR` (§3).
**Destruction order is a hard requirement (correction):** the `HWND` must outlive
the Vulkan surface. The wndproc must NOT `DestroyWindow` on `WM_CLOSE`
immediately. Instead: `WM_CLOSE`/close request → mark the window closing and
signal the render thread (`Surface::close`); the render thread drops its presenter
(destroying swapchain + surface), then posts its release to the UI thread; only
that release handler calls `DestroyWindow`. This mirrors Linux's
"surface goes before the window under it" ordering (`linux/render.rs` doc), but
with a Win32 release path replacing `sdl::shell::released`.
*Validate (report, don't assume):* a manual `--ignored` test / example that opens
a window, builds a `reims-vgpu-vulkan` device, runs `add_window_extensions`,
creates the surface, and asserts a non-null `vk::SurfaceKHR`. Then — rather than
hard-asserting `vkGetPhysicalDeviceWin32PresentationSupportKHR` — **enumerate the
device's queue families and REPORT which family index supports present to this
surface** (`get_physical_device_surface_support` from
`ash::khr::surface::Instance`), plus the Win32 presentation-support query result
for cross-check, and log both for this RTX 4060 host. Then exercise the full
close path and confirm the surface is destroyed before `DestroyWindow` (log order
/ validation-layer clean). The milestone passes on: surface created + a present-
capable queue family found and reported + clean teardown in the required order.

**Stage 2 — present a test pattern.**
Wire `FrameSink`→`vulkan::SwapchainSink` (existing) so the render thread presents
through the window. Drive it with a synthetic frame source (no guest).
*Validate:* visible frames in the window at the screen's rate; resize produces a
new layout and keeps presenting; close ends the render thread then the window
(the Linux ordering in `render.rs::run`).

**Stage 3 — input + screens.**
`WM_*` → pointer/button/wheel/key callbacks (reusing `scroll.rs`/`keys.rs`), and
`host_screens()` via monitor enumeration + per-monitor DPI → `HostScreen`.
*Validate:* unit-test the scancode→USB table; manually confirm pointer/scroll/keys
reach `WindowCallbacks`; confirm `policy.rs` offers correct modes for the detected
screen.

**Stage 4 — build the VMM.**
With the window arm done, `cargo build -p reims-vgpu --features vulkan
--target x86_64-pc-windows-msvc` links. (The VMM *binary* that boots a guest still
needs the custom QEMU, which is the separate WHPX feasibility track — see
`docs/windows-whpx-feasibility.md`. Windowing does not depend on it; a window +
present can be proven headless of any guest via Stages 1-3.)

## 6. Bottom line

- **Feasibility: high.** No architectural blocker, unlike WHPX/QEMU. The Vulkan
  WSI/surface/swapchain path for Windows already exists; the sink is OS-agnostic;
  the crate's logic modules are reused unchanged.
- **Minimum to build + present:** Stages 0-2 (empty arm → window+surface →
  present). Stage 0 alone is the small reviewable prototype that proves the seam.
- **Library choice:** raw Win32 (`windows-sys`) + `VK_KHR_win32_surface`; no SDL3,
  no winit.
- **Decoupling:** a presentable window is provable **without** the guest/QEMU
  path — so this track can land and be validated independently of the WHPX work.
