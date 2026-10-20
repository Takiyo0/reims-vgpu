# Windows host

How to build and run an x86-64 macOS guest on Windows: QEMU with WHPX acceleration, the guest's
Metal work translated to Vulkan on the host GPU, and each guest display in a Win32 window of its
own. This page is the guide to the pieces, the launcher, the guest's setup, every environment
variable the Windows build reads, what the performance work changed, and what is known to be
wrong. [windows-whpx-feasibility.md](windows-whpx-feasibility.md) and
[windows-windowing-plan.md](windows-windowing-plan.md) are the analyses the port started from.

Where this page says "unverified", the claim comes from a note or a log and was not checked
against the code or a run when the page was written.

## What it is and how the pieces fit

```
macOS guest (x86-64, OpenCore, AppleParavirtGPU)
   |  PCI 106b:eeee, option ROM, MMIO + FIFOs in guest RAM
QEMU (WHPX)  hw/display/apple-gfx-pci.c  --C API (reims_vgpu.h)-->  libreims_vgpu.a (staticlib)
                                                                      |
                                       reims-vgpu-core: decodes the guest's command streams
                                                                      |
                       reims-vgpu-vulkan: the Platform answering as Metal does
                                                                      |
                       metal2vulkan-runtime: Metal-shaped runtime, AIR -> SPIR-V, Vulkan
                                                                      |
                                    NVIDIA (or other) Vulkan driver, VK_EXT_external_memory_host
```

| piece | where | what it does |
|---|---|---|
| QEMU | `vendor/qemu` of the main checkout (`dev\reims-vgpu`) | `q35` machine on `accel=whpx`; `apple-gfx-pci` is the device; links `libreims_vgpu.a` through pkg-config |
| `apple-gfx-pci` | `hw/display/apple-gfx-pci.c`, `apple-gfx.c` | the PCI device, its option ROM (`ReimsVgpu.rom`), MSI delivery, display hotplug over QMP, the pointer-agent socket |
| the staticlib | `crates/reims-vgpu-c` (`libreims_vgpu_c.a`, staged as `libreims_vgpu.a`) | the C API; mimalloc is its allocator on Windows |
| core | `crates/reims-vgpu-core` | the device, FIFOs, task and resource tables, the decoder, the decode workers |
| the Vulkan platform | `crates/reims-vgpu-vulkan`, `os/windows/` | the Platform over metal2vulkan-runtime; guest RAM as host-pointer imports; Win32 surfaces |
| metal2vulkan | `https://github.com/Takiyo0/metal2vulkan`, branch `master` | translates Metal API calls and AIR shaders to Vulkan |
| the window crate | `crates/reims-vgpu-window`, `src/native/windows/` | `reims-vgpu.ui`: a thread that owns every Win32 window and its message pump; `render.rs` draws and presents; `input.rs` and `keymap.rs` turn `WM_*` messages into pointer, wheel and key callbacks |
| the launcher | `qemu-vm\boot-perf.ps1` | builds QEMU's command line and environment, restarts QEMU on a guest restart |

Two properties of the Windows build shape everything below.

- **Guest RAM is anonymous `VirtualAlloc` memory** (WHPX maps it with `WHvMapGpaRange`), so no
  section object can be remapped. The library borrows the host address of a guest-contiguous
  range, imports it into Vulkan with `VK_EXT_external_memory_host`, and copies a guest-scattered
  range into a scratch range (the copy fallback) or gathers it on the GPU.
- **The Vulkan device is chosen by the runtime**, first discrete, then integrated; Windows's
  per-app GPU preference is ignored. `VK_DRIVER_FILES` pointing at one ICD's JSON is the only
  lever (unverified beyond the 2026-10-13 test that forced the Intel iGPU this way).

The window shows the guest's frame. A click into the image captures the keyboard (the title gains
a suffix saying so); Ctrl+Alt+Win releases it, and so does the window losing focus.

## Requirements

| need | detail |
|---|---|
| Windows 11 | the machine this was built and run on is Windows 11 Pro |
| WHPX | "Windows Hypervisor Platform" turned on in Windows Features; QEMU is built `--enable-whpx` |
| a Vulkan GPU | with `VK_EXT_external_memory_host` (every import of guest memory needs it). Run on an RTX 4060 Laptop GPU; Intel Raptor Lake graphics also presented frames in a test |
| host RAM | the launcher gives the guest 10 GiB (`-m 10240`); see "Host memory" under Known issues |
| MSYS2 UCRT64 | in `C:\dev\msys64`: gcc, ninja, meson, pkgconf, `mingw-w64-ucrt-x86_64-libslirp`, and `-libusb` for USB passthrough. The launcher puts `C:\dev\msys64\ucrt64\bin` on `PATH` |
| Rust | the target `x86_64-pc-windows-gnu` (`rustup target add`) |
| libusb, UsbDk | optional: USB passthrough (`-UsbPassthrough`, `usb.ps1`) needs a QEMU built `-Dlibusb=enabled` and the UsbDk driver installed; without UsbDk QEMU falls back to WinUSB, which needs a per-device driver swap |
| Python | on `PATH`; the launcher's display-persistence helper (`display-control.py`) and the report scripts use it |
| the guest | a macOS install booted through OpenCore. The disks in `qemu-vm\` are a converted VMware install |

## Using reims-ctl on Windows

`reims-ctl` builds as a native Windows executable and can manage and start x86-64 Q35 VMs with
WHPX. It expects a QEMU build that already includes this repository's `apple-gfx` device and ROM,
WHPX support, and `qemu-img.exe` beside `qemu-system-x86_64.exe`. Build the controller from the
checkout in PowerShell:

```powershell
cargo build --manifest-path .\crates\reims-ctl\Cargo.toml --release
$ctl = (Resolve-Path .\crates\reims-ctl\target\release\reims-ctl.exe).Path
```

Point it at the Windows QEMU build, its BIOS/data directory, and the Reims option ROM. Keep these
paths in the same PowerShell session when using the controller:

```powershell
$env:REIMS_CTL_QEMU = (Resolve-Path 'C:\path\to\qemu-system-x86_64.exe').Path
$env:REIMS_CTL_QEMU_DATA = (Resolve-Path 'C:\path\to\qemu-data').Path
$env:REIMS_CTL_QEMU_ROM = (Resolve-Path 'C:\path\to\ReimsVgpu.rom').Path
& $ctl doctor
```

`qemu-data` is the directory containing the firmware files QEMU normally receives through `-L`.
The option ROM path may also point to `ReimsVgpu.rom` inside that directory. `doctor` checks that
the selected QEMU lists WHPX; the QEMU binary and `qemu-img.exe` must both be present.

To register an existing macOS disk, give `--opencore` an OpenCore image with `OVMF_CODE.fd` and
`OVMF_VARS.fd` beside it. `--disk-mode in-place` makes the VM use the existing disk directly,
without copying it. For example:

```powershell
& $ctl create mac --disk 'C:\VMs\macOS.qcow2' --disk-mode in-place `
  --opencore 'C:\VMs\OpenCore.qcow2' --dir 'C:\VMs\reims-ctl\mac'
& $ctl list
& $ctl start mac
```

The controller also accepts a supplied installer image with `--installer`. Install the Windows
build of `reims-util` directly from your public `master` branch; no sibling checkout is needed:

```powershell
cargo install --locked --force --git https://github.com/Takiyo0/reims-util --branch master reims-util
$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
$env:REIMS_CTL_UTIL = Join-Path $cargoHome 'bin\reims-util.exe'
& $ctl doctor
```

With that set, `reims-ctl create mac --macos tahoe` can make OpenCore and download an online
installer; `--offline` downloads a full installer instead. Downloads and cached source archives
go under `%LOCALAPPDATA%\reims-util\cache` by default. `reims-ctl update` still uses the
Unix-oriented installer and is not Windows-ready. This port has Windows compile and CLI checks;
a full macOS guest boot through `reims-ctl` and a live Apple download still need end-to-end runs.

The guest's CPU is `Haswell` with AVX2 and XSAVE: the guest ships only the `x86_64h` dyld cache,
and a CPU without AVX2 makes the shared-region setup fail (`vm_shared_region_start_address()`
returns 1) and the guest panic in launchd. The machine presents as an iMac19,1 (set in the OpenCore
image's SMBIOS); a MacPro7,1 panics in `AppleIntelMCE` (both from the 2026-10-13 bring-up notes).

## Building

Three steps: the library, the staged copy QEMU links, QEMU.

### 1. The library

From the repository root, in PowerShell, install the whole Windows toolchain with:

```powershell
.\scripts\install-ctl.ps1
```

It checks for Rust and MSYS2 UCRT64, offers to install missing MSYS2 build packages, builds and
installs the Windows library, QEMU, option ROM, `reims-ctl`, and `reims-util`. Then run
`reims-ctl doctor`. The manual steps below are useful when changing one component at a time.

To build only the library, use Git Bash with MSYS2's compiler on `PATH`:

```sh
cd dev/reims-vgpu
export PATH="/c/dev/msys64/ucrt64/bin:$PATH"
cargo build --release --target x86_64-pc-windows-gnu -p reims-vgpu-c --lib --features reims-vgpu-c/vulkan
```

The result is `target/x86_64-pc-windows-gnu/release/libreims_vgpu_c.a`. Add `--features reims-vgpu-c/record-timing` for the per-record counters (`rec<id>` on the `work:` line). The build also
fetches metal2vulkan from `Takiyo0/metal2vulkan`'s `master` branch through Cargo.

### 2. The staged copy

QEMU's meson build finds the library through pkg-config, from the **reims-dist** tree (a
hand-written package; `cargo xtask dist` writes none for Windows):

```
reims-dist\include\reims_vgpu.h
reims-dist\lib\libreims_vgpu.a
reims-dist\lib\pkgconfig\reims-vgpu.pc      Libs: -L${libdir} -lreims_vgpu
                                            Libs.private: -lkernel32 -lntdll -luserenv -lws2_32
                                                          -ldbghelp -lonecore -lshcore -lmincore
```

`-lonecore -lshcore -lmincore` are not in what `cargo rustc --print native-static-libs` reports;
QEMU's static link needs them for `VirtualAlloc2`, `MapViewOfFile3`, `UnmapViewOfFile2`,
`QueryUnbiasedInterruptTimePrecise` and `GetDpiForMonitor`.

The header QEMU compiles against must be the one the library was built from. A QEMU object built
against a different `reims_vgpu.h` than the staticlib's links without error and shows a black
screen (see Troubleshooting).

### 3. QEMU

QEMU is built out of tree, in MSYS2, from `dev\reims-vgpu\vendor\qemu`. The build directory used
until 2026-10-19 was `qemu-build-patched`; the cleanup that day removed it and every `target`
directory, so a build starts from a configure:

```sh
export PATH=/ucrt64/bin:/usr/bin
export PKG_CONFIG_PATH=/c/Users/takiyo/Documents/macos/reims-dist/lib/pkgconfig
mkdir qemu-build && cd qemu-build
../dev/reims-vgpu/vendor/qemu/configure --target-list=x86_64-softmmu --enable-whpx --enable-pvg \
    --disable-fdt --disable-docs --disable-plugins --disable-werror
meson configure -Dlibusb=enabled          # USB passthrough
ninja qemu-system-x86_64.exe
```

(`configure` also needs `mingw-w64-ucrt-x86_64-libslirp` for the user-mode network. If
`PKG_CONFIG_PATH` does not reach `reims-vgpu.pc`, `pvg` is not enabled and the device is missing.)
After a library change, stage the new `libreims_vgpu.a` and run `ninja` again. Kill any running
`qemu-system-x86_64*.exe` first: the running exe is locked and the link fails with "Permission
denied".

The launcher runs `C:\Users\takiyo\Documents\macos\qemu\qemu-system-x86_64.exe` by default
(`-QemuPath`); that folder holds the executable, `qemu-img.exe` and the MSYS2 DLLs they load,
copied out of the build so the build directory can go. A new build is made without disturbing it:

1. stage the library (step 2) and run `ninja qemu-system-x86_64.exe`;
2. copy the exe to a name of its own, e.g. `qemu\qemu-system-x86_64-mybuild.exe` (the DLLs beside
   it are found by name);
3. write `<exe>.build.json` next to it: `{"qemu_sha256": "<SHA-256 of the exe>",
   "staticlib_sha256": "<SHA-256 of libreims_vgpu_c.a>", "built_utc": "<UTC time>"}`. The
   launcher stops if the exe's hash differs from the manifest's, and logs both hashes in
   `boot-gfx-<tag>.log` (`QEMU_SHA256`, `STATICLIB_SHA256`);
4. pass `-QemuPath` with it. Keep the staged library and the known-good exe unchanged until the
   new pair has booted.

`qemu-vm\relink-named.sh <name>` did these steps for the build directory that is now gone (it
swapped the staged files in and restored them, and wrote the manifest); the 2026-10-19 cleanup
list marks it deletable. It expects `qemu-build-patched` and two backup files, so it does not run
as it is.

The current API and QEMU sources must be built together. In particular, QEMU's
`memory_region_has_guest_memfd_private` API is the one this branch uses when checking whether its
EFI framebuffer can alias guest RAM. Build the library, stage its matching header and archive, then
build QEMU from this checkout's `vendor/qemu`; do not reuse an older QEMU object or static library
as a pair. The executable and manifest under the separate local `qemu\` folder are not repository
artifacts and are not evidence that this checkout's build succeeds.

Rebuilding the option ROM (`ReimsVgpu.rom`) is `make -C firmware` with clang and lld (needs
`MSYS2_ARG_CONV_EXCL=*` so `lld-link` keeps its `/flags`); the ROM in `qemu-vm\` is the one in use.

## Running

```powershell
cd C:\Users\takiyo\Documents\macos\qemu-vm
.\boot-perf.ps1                # or .\run-macos.ps1, which passes its arguments to it
```

`boot-perf.ps1` has no tracing, no Vulkan validation, no capture and no VNC by default; the Reims
window is the only display. It sets the environment (below), starts QEMU, and starts the display
persistence helper. A guest restart makes QEMU exit with 99; the launcher then starts a fresh
QEMU (logs of the next boot carry `-boot2`, `-boot3`, ...), up to 50 boots, unless `-NoRestart`.
It exits with QEMU's last exit code.

**Start a long run detached.** A VM started from a tool or app that is later restarted dies with
it (the launcher and QEMU are in that app's process tree). Detach it through WMI:

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'powershell -NoProfile -File C:\Users\takiyo\Documents\macos\qemu-vm\boot-perf.ps1 -RunTag mytest'
  CurrentDirectory = 'C:\Users\takiyo\Documents\macos\qemu-vm' }
```

Stop it with the QMP proxy: send `{"execute":"qmp_capabilities"}` then `{"execute":"quit"}` to
`127.0.0.1:4453`, or `quit` in the human monitor on 4452. Do not stop it with a debugger.

### Ports

| port | what |
|---|---|
| 4452 | QEMU's human monitor (telnet, `-MonitorPort`) |
| 4453 | the QMP proxy (`-QmpPort`), a local, unauthenticated control endpoint with the display commands (`query-reims-displays`, `reims-set-display-count`) |
| 4454 | QEMU's own QMP, private upstream of the proxy (`-QemuQmpPort`) |
| 4455 | the guest pointer agent's chardev (`-PointerAgentPort`), only with 2+ guest display ports |
| 5900+n | VNC, only with `-VncDisplay n` |
| 2222 | SSH into the guest: `hostfwd=tcp:127.0.0.1:2222-:22`, reachable from this machine alone. The guest reaches the host at 10.0.2.2 |

The three control ports must differ from each other; the pointer agent's from the monitor and QMP.

### Launcher parameters

| parameter | default | effect |
|---|---|---|
| `-RunTag` | `perf-<timestamp>` | names the run's log files |
| `-QemuPath` | `qemu\qemu-system-x86_64.exe` | the QEMU executable; checked against `<exe>.build.json` when present |
| `-MacosOverlay` | `macos-vmware-overlay-20261016.qcow2` | the guest's disk (a qcow2 over `macos.qcow2`); `macos-shaderpass-20261015.qcow2` is the fresh install (Setup Assistant) |
| `-EspImage`, `-VarsImage` | `esp-ocdebug-livev-20261016.img`, `OVMF_VARS-livev-20261016.fd` | the OpenCore EFI system partition and OVMF's variable store |
| `-SnapshotDisks` | off | `snapshot=on` on the ESP, variables and macOS disks: nothing is written back |
| `-NoRestart` | off | stop after the first QEMU exit |
| `-CpuTopology` | `sockets=1,cores=10,threads=1` | QEMU `-smp`. More than one vCPU needs CPUID leaf 4 in WHPX's exit list (it is) |
| `-KernelIrqchip` | `on` | `on` is Hyper-V's in-kernel APIC; `off` is QEMU's own |
| `-Usb` | `xhci` | `xhci` serves a device as soon as it has data; `ehci` is polled by a timer that slows when idle, so a 16-key keyboard queue overflows and drops key-ups: keys stick and repeat |
| `-UsbPassthrough` | none | host devices as `VID:PID` passed from boot; needs `-Usb xhci`, a libusb build and UsbDk. `usb.ps1 list/attach/detach` does it while running. Never pass the laptop's built-in keyboard (Windows loses it) |
| `-Audio` | `usb` | `usb` is a `usb-audio` device on xHCI played through DirectSound; `none` drops it. Without an output device, YouTube fails with a MSE `DecodeError` |
| `-AudioOutputFrequency` | 44100 | the audiodev's output rate (8000 to 192000) |
| `-AudioTuning` | `on` | `on`: audiodev `timer-period=10000` and `out.buffer-length=100000` (fewer host wakeups and xHCI events, about 100 ms more latency); `off`: the plain line |
| `-AudioBufferBytes` | 0 | the USB audio stream ring in bytes (0 is QEMU's default; 12288 is 64 ms at 48 kHz stereo S16) |
| `-AudioStats` | off | sets `QEMU_AUDIO_STATS=1` and the `usb_audio_*` / `dsound_*` trace events |
| `-Nic` | `virtio` | `virtio-net-pci,disable-legacy=on` at slot 5 (fewer VM exits than vmxnet3's register writes); if macOS does not bind it there is no network or SSH: relaunch with `vmxnet3` |
| `-DisplayCount` | saved count (1 at first) | connected displays, 1 to 8; saved next to the overlay as `<overlay>.displays.json` |
| `-DisplaySettingsPath` | `<overlay>.displays.json` | where the count is kept |
| `-HotplugDisplays`, `-GuestDisplayPorts` | off, 8 | tell the guest `-GuestDisplayPorts` ports (see Multiple displays); set automatically with 2+ displays |
| `-PointerAgentPort`, `-NoPointerAgent` | 4455, off | the guest pointer agent (below) |
| `-LegacyDisplay` | off | one display, no runtime controls, no persistence proxy; works with an exe lacking `display-ports` |
| `-GuestResolution` | QEMU's | `WIDTHxHEIGHT` cap (320x200 to 7680x4320) as the device's `resolution=` |
| `-WindowRefreshHz` | 0 (the screen's highest) | sets `REIMS_VGPU_WINDOW_REFRESH` |
| `-EventStats` | off | sets `REIMS_VGPU_EVENT_STATS=1`: QEMU prints `REIMS_INPUT_STATS` lines |
| `-Profile` | off | `REIMS_VGPU_LOG` debug filters, `REIMS_VGPU_LOG_TIME`, `M2V_TIMING`, `WHPX_MMIO_STATS`, `REIMS_VGPU_CMDHASH=1`, and four QEMU trace events (`apple_gfx_read/write/raise_irq/new_frame`) into `qemu-gfx-<tag>.log` |
| `-Sample` | off | `REIMS_VGPU_SAMPLE=1`, the sampling profiler |
| `-PowerTrace` | off | run-state, wakeup and shutdown traces; Reims lifecycle logs |
| `-FirmwareLog` | off | OVMF's debug console (port 0x402) to `ovmf-<tag>.log` |
| `-WarmReset` | off | a guest restart is QEMU's warm reset, which stalls in the firmware under WHPX (the APs never start); off, a restart exits 99 and a fresh QEMU boots |
| `-AllowS3` | off | offer ACPI S3; its wake is a warm reset and stalls too. Off, guest "Sleep" is a display sleep |
| `-ExtraLog` | empty | appended to `REIMS_VGPU_LOG`, e.g. `reims_vgpu_core::resources::manager=debug` |
| `-VncDisplay` | -1 (off) | VNC for scripted spot checks only; a connected client makes QEMU read frames back and stalls presents |
| `-MonitorPort`, `-QmpPort`, `-QemuQmpPort` | 4452, 4453, 4454 | see Ports |

Fixed in the launcher, not parameters: `-m 10240`; the Haswell CPU line
(`+invtsc,vmware-cpuid-freq=on,+avx,+avx2,+fma,+bmi1,+bmi2,+xsave,...`); the UUID; the machine
`q35,accel=whpx`; pflash OVMF; the OpenCore image and macOS disk on `ich9-ahci` (`cache=unsafe` on
the macOS disk); `-vga none -display none`; the serial port to a file; the xHCI controller with
`p2=8,p3=8` (high-speed root ports left for passthrough); `-global ICH9-LPC.disable_s3=1`;
`-action reboot=shutdown`. `qemu-vm\home` and `qemu-vm\tmp` must exist (the second may be emptied). The launcher also sets `HOME` to `qemu-vm\home` and the temp
directories to `qemu-vm\tmp`; the Vulkan shader cache lives under that `HOME`
(`.cache\reims-vgpu-vulkan`).

Environment it sets for every run: `REIMS_VGPU_STATS=1`, `REIMS_VGPU_LOG_ASYNC=1`,
`REIMS_VGPU_WINDOW=1`, `REIMS_VGPU_ASYNC_PREPARE=0`, `REIMS_VGPU_RAM_IMPORT=0`. It first removes a
list of the switches in the configuration reference below (and `RUST_LOG`, `VK_INSTANCE_LAYERS`,
`VK_LOADER_DEBUG` among others), so a stale value in the shell cannot disable a fix. To run with
one switch off, edit that list or set the variable inside the launched shell after the clear.

### Display disks

`macos.qcow2` (500 GiB virtual, the converted VMware install) is the backing file of
`macos-vmware-overlay-20261016.qcow2` (by the relative name `macos.qcow2`, so the two stay in one
directory), which takes every write; `macos.qcow2` stays untouched. The
fresh QEMU-native install is `macos-shaderpass-20261015.qcow2` over `macos-overlay.qcow2` over
`macos-fresh.raw`.

## Guest setup

The guest drivers were written against Apple's framework, so nothing is installed for the GPU.
What the guest needs beyond that:

How the guest disk, the OpenCore image, the firmware files, the SSH key and the guest agent were made, and how to
make them again: [guest-setup.md](guest-setup.md).

- **`FORCE_SOFTWARE_RENDERER=1`, for blank icons.** `iconservicesagent`, Spotlight and
  `com.apple.dock.extra` crash in IconRendering: it hands RenderBox its `binary.metallib` archive,
  and Metal asserts on the x86 paravirtual device (`+[MTLLoader sliceIDForDevice:]`, "Target
  device architecture is nil"; `AppleParavirtDevice` reports `supportsBinaryArchives=0` and no
  `targetDeviceArchitecture`). System Settings sidebar icons stay blank and the apps crash-loop.
  IconRendering reads `FORCE_SOFTWARE_RENDERER=1` and then uses no Metal device. `RB_USE_BINARY_ARCHIVE=0`
  does nothing for it and `REIMS_VGPU_APPLE_FAMILIES=0` changed nothing. Permanent setting, run by
  the user: `sudo launchctl config user setenv FORCE_SOFTWARE_RENDERER 1`, then reboot. The
  interim was a login agent, `~/Library/LaunchAgents/local.icon-software-renderer.plist`, that
  runs `launchctl setenv FORCE_SOFTWARE_RENDERER 1`. An agent that also restarts the Dock
  delayed the Dock and wallpaper by 30 to 60 s at login and was reduced to the `setenv`. Never
  kill the Dock in a degraded session. `com.apple.dock.extra` (Calendar's live Dock tile) did not
  receive the variable from either domain in the 2026-10-17 test; removing Calendar from the Dock
  was the only remedy then (current state unverified). A stale on-disk icon cache may need
  `sudo rm -rf /Library/Caches/com.apple.iconservices.store` and a reboot.
- **Liquid Glass shows white boxes** when Accessibility > Display > Reduce transparency is on.
  Turn it off.
- **The pointer agent**, only for 2+ displays (below).
- **Network.** The default `virtio-net-pci` is a QEMU user-mode NAT. SSH: `ssh -p 2222 -i
  qemu-vm\ssh\vm_ed25519 <user>@127.0.0.1`. If macOS does not bind the card, relaunch with `-Nic
  vmxnet3`.
- **USB audio.** macOS binds its own USB audio driver; no kext. The output is the host's default
  DirectSound device. "Audio Output - Disabled" in System Information is a cosmetic string of the
  device's alternate setting 0.
- **USB devices** attach to the xHCI controller's root ports with `usb.ps1` (below).
- **Memory.** The launcher gives 10 GiB; with 6 GiB the guest ran under memory pressure (a
  Jetsam event at 140 MB free was seen at login).

### Multiple displays

Each display is one Win32 window. The count is saved with the overlay and reloaded on every
launch; an explicit `-DisplayCount` saves a new one.

```powershell
.\boot-perf.ps1 -DisplayCount 2
.\display.ps1 list | add | remove | set-count 3         # on a running VM, through the QMP proxy
```

- Closing secondary window N disconnects it and every higher-numbered window. Closing the primary
  window shuts the VM down. At least one display stays connected; the maximum is eight. Arrangement,
  main display and mirroring are macOS System Settings.
- macOS's x86 driver negotiates binary version 6, at which the device answers the port-count
  register (0x122c) with one port, so a second display never gets a framebuffer. With
  `REIMS_VGPU_GUEST_DISPLAY_PORTS=<n>` the device tells the guest `min(n, display-ports)` ports.
  The launcher sets it to `-GuestDisplayPorts` (default 8) for a boot with 2+ displays or with
  `-HotplugDisplays`. The guest reads the count once, as its driver starts, so adding ports needs a
  fresh boot; `-HotplugDisplays` makes a one-display boot add displays live. If eight framebuffers,
  six empty, misbehave, lower `-GuestDisplayPorts`.
- Check in the guest: `ioreg -w0 -l | grep -c "o AppleParavirtFramebuffer "` and
  `system_profiler SPDisplaysDataType`.
- macOS maps the USB tablet's absolute axes onto its main display only, so window 2's pointer
  would land on display 1. The guest agent `qemu-vm\guest-agent\ReimsPointer.swift` (LaunchAgent
  `com.reims.pointer`, installed in the guest with `sh ~/reims-guest-agent/install.sh`, removed
  with `-u`) connects to `10.0.2.2:4455`. While it is connected, `apple-gfx` sends it
  `P <serial> <x> <y> <w> <h> <buttons>` per window; it finds the display by serial number and
  moves the cursor there. Buttons, wheels and keys stay on USB. With Accessibility permission
  (System Settings > Privacy & Security > Accessibility: add
  `~/Library/Application Support/ReimsPointer/ReimsPointer`) it posts moved and dragged events;
  without it, it warps the cursor and drags may not work. Rebuilding the agent invalidates the
  permission. Log: `~/Library/Logs/ReimsPointer.log`; QEMU logs `pointer agent connected` /
  `disconnected` on stderr. Needs an exe with the `pointer-chardev` property; `-NoPointerAgent`
  turns it off.
- QMP, on `127.0.0.1:4453` after `qmp_capabilities`:
  `{"execute":"query-reims-displays","arguments":{"device":"reims"}}` and
  `{"execute":"reims-set-display-count","arguments":{"device":"reims","count":2}}` return
  `{"device":"reims","count":2,"ports":8}`; the event `REIMS_DISPLAY_COUNT_CHANGED` carries
  `device` and `count`, including changes from closing a secondary window. A failed attach of
  several new monitors rolls the count back; query again to see the state. Persistence errors go to
  `display-persistence-<tag>.log`.
- `python test-display-control.py` (in `qemu-vm`; on the cleanup's deletable list) tests the settings and the QMP handling. Independent desktops,
  pointer placement, keyboard focus and reconnection after a restart need a guest to check
  (not done by that test).

### USB passthrough

```powershell
.\usb.ps1 list                    # host devices (VID:PID) and what the guest holds
.\usb.ps1 attach 05ac:12a8        # or by the Windows device name; waits if not plugged in yet
.\usb.ps1 detach 05ac:12a8        # 'detach all' for every passed device
```

A passed device leaves Windows while the VM holds it and returns on detach or when QEMU exits.
Devices match by VID:PID, so one that re-enumerates (an iPhone does) stays with the guest. The
QEMU device ids are `pt<vid><pid>`.

## Configuration reference

Every variable the Windows build reads, found by searching `crates/` of this branch, metal2vulkan's
`crates/`, and `vendor/qemu`. "Default" is the behaviour with the variable unset. Switches that are
"on unless `0`" are read once per process. The first column marks the kind: **S** a switch that
restores an older path, **T** a tuning value, **D** a diagnostic (no effect on what the guest or
the host shows beyond logs, files or cost), **E** an experiment.

Variables the tests read (`REIMS_VGPU_DEATH_*`, `*_CHILD`, `M2V_TEST_*`, `M2V_LANE_*`, `M2V_PARITY_CASE`,
`M2V_SEQUENCE_SEED`, `M2V_PRIVATE_CORPUS_ROOT`) and the Linux DRM shim's (`M2V_DRM_SHIM_*`) are
left out.

### Decoding and the FIFO

| | variable | default | effect |
|---|---|---|---|
| S | `REIMS_VGPU_ASYNC_DECODE` | on | `0` runs every exec's second half inline on the FIFO thread |
| T | `REIMS_VGPU_DECODE_WORKERS` | 4 | decode workers per FIFO, clamped 1 to 16 |
| S | `REIMS_VGPU_ASYNC_PREPARE` | on in the library, **0 in the launcher** | `0` keeps `prepare_resources` on the FIFO thread; on, workers run it in FIFO order before decoding. Needs the workers. Measured slower (see Performance) |
| S | `REIMS_VGPU_EARLY_COMMIT` | on | `0` keeps the completed handler, the signal and cleanup inside the commit turn |
| S | `REIMS_VGPU_DECODE_PRIORITY` | on | `0` leaves the decode workers' and FIFO threads' priority alone (default: one step above normal) |
| S | `REIMS_VGPU_PCORES` | on | `0` keeps the priority but lets the scheduler place the threads; default restricts them to the CPU sets of the highest efficiency class on a hybrid CPU |
| S | `REIMS_VGPU_SHARED_DELETE` | on | `0` makes DeleteResource and UnmapMemory take the task's unique lock; default takes the shared lock (an unmap overlapping the object list stays unique) |
| S | `REIMS_VGPU_EVENT_FIX` | on | `0` raises one interrupt per stamp signal; default raises none while stamp interrupt bits are already pending |
| S | `REIMS_VGPU_EMPTY_ENTRY_FAULTS` | off | `1` reports an exec naming an empty object-list entry (a dead process's leftover command) as a fault, and gives nil to a pipeline whose function entry is empty; default drops the exec while its task is young |
| S | `REIMS_VGPU_NOBUF_SOFT` | on | `0` reports "Metal made no buffer over task page" as exec failure (the guest sees a GPU hang and restarts the channel); default drops that command buffer as done |
| S | `REIMS_VGPU_NOBUF_RETRY` | on | `0` skips the one retry of a refused no-copy buffer over a private copy |
| S | `REIMS_VGPU_PIPELINE_WIDTH_CLAMP` | on | `0` reports a compute pipeline's zero `threadExecutionWidth` and `maxTotalThreadsPerThreadgroup` as they are; default answers 32 and 1024 (a zero divisor crashed DaVinci Resolve) |
| S | `REIMS_VGPU_FRAME_COPY` | on | `0` draws from the guest's frame directly; default copies it first, off the draw path (3 copies kept per display) |

### Memory and buffers

| | variable | default | effect |
|---|---|---|---|
| S | `REIMS_VGPU_RAM_IMPORT` | on in the library, **0 in the launcher** | `0` imports guest RAM lazily in 64 MiB stretches at the first buffer that needs one; on, the device imports all guest RAM at creation in aligned chunks. Read by metal2vulkan |
| T | `REIMS_VGPU_RAM_IMPORT_CHUNK_MB` | 256 | chunk size, rounded down to a power of two, clamped 64 to 1024 |
| S | `REIMS_VGPU_BUFFER_POOL` | on | `0` destroys a released no-copy buffer's `VkBuffer` instead of parking it for reuse (also `M2V_BUFFER_POOL=0`) |
| S | `REIMS_VGPU_GPU_GATHER` | on | `0` fills scattered guest ranges on the CPU; on, a texture or private-buffer transfer is gathered on the GPU from the imported chunks when it passes the two floors below |
| T | `REIMS_VGPU_GPU_GATHER_MIN_KB` | 1024 | smallest total transfer gathered on the GPU |
| T | `REIMS_VGPU_GPU_GATHER_MIN_RUN_KB` | 16 | smallest average run of merged pages; `0` removes the floor |
| S | `REIMS_VGPU_TRANSFER_CACHE` | on | `0` makes every transfer buffer afresh; default hands one out again by its physical page runs |
| S | `REIMS_VGPU_CURSOR_POOL` | on | `0` allocates every exec's command-stream copy (pool: up to 2 MiB each) |
| S | `REIMS_VGPU_SCRATCH_POOL` | on | `0` stops pooling copy-fallback ranges |
| S | `REIMS_VGPU_SCRATCH_CLASSES` | on | `0` makes scratch ranges exactly their length; default rounds up to an eighth-of-a-power-of-two class so nearby sizes share ranges and their kept imports |
| E | `REIMS_VGPU_PARALLEL_FILLS` | off | `1` runs texture page copies on a persistent thread pool before the paging commit. Off since a WindowServer hang seen with it on (2026-10-16) is unexplained |
| S | `REIMS_VGPU_BLOCK_ENV_TLS` | on | `0` restores the owning thread-local in `rt/block_env.rs` (see the thread-local note under Performance) |
| S | `REIMS_VGPU_LAST_RESORT_TLS` | on | `0` restores the old access path in `rt/held.rs` |

### Guest-visible behaviour

| | variable | default | effect |
|---|---|---|---|
| S | `REIMS_VGPU_APPLE_FAMILIES` | on | `0` tells the guest no Apple GPU families, as an Intel Mac's GPU would. It did not change RenderBox's archive abort and hides Metal 3 |
| T | `REIMS_VGPU_GUEST_DISPLAY_PORTS` | unset | port count 1 or more the guest reads at 0x122c; unset, the guest reads Apple's answer (1 below binary version 245) |
| E | `REIMS_VGPU_DISPLAY_NITS` | unset | `min,max,maxsdr` replaces the descriptor's nits, to see whether the guest composites in 8 bit |
| T | `REIMS_VGPU_STRICT` | off | `1`: the Vulkan platform ends the process at each difference from Metal's answer it would otherwise log |
| T | `REIMS_VGPU_EXACT` | unset | comma list of runtime fields run in their exact form: `sampling`, `float_maths`, `texel_conversion`, `half_rounding`, `nan_and_min_max`, `rounding_order`, `half_blend` ([accuracy-ledger.md](accuracy-ledger.md)) |
| S | `M2V_AIR_DEBUG_TOLERANT` | on | `0` makes metal2vulkan-air refuse debug metadata, debug module flags and `llvm.dbg.*` calls again (Resolve's debug-build metallibs) |
| E | `M2V_DONT_CARE_LOAD` | off | `1` loads every `DontCare` attachment; the default estimated reach cleared a tile the macOS boot logo screen kept |
| T | `REIMS_VGPU_CACHE_DIR` | `%HOME%\.cache\reims-vgpu-vulkan` (or `%USERPROFILE%`) | the shader cache directory |

### Window

| | variable | default | effect |
|---|---|---|---|
| T | `REIMS_VGPU_WINDOW` | on | read by QEMU: `0` keeps `apple-gfx` headless under `-display none` |
| T | `REIMS_VGPU_WINDOW_REFRESH` | the screen's highest | refresh rate in Hz (above 0, up to 1000) of the modes offered to the guest. The guest sees the host monitor's highest rate; a 60 Hz monitor needs this set to test a higher one |
| T | `REIMS_VGPU_WINDOW_SCREENS` | display i on screen i | comma list of host screen indices, display i on the list's i-th entry; one bad entry unsets the list. Read by the shared screen picker; behaviour on Windows unverified |
| T | `REIMS_VGPU_WINDOW_BORDERLESS` | off | borderless windows that fill their screen. Read by the SDL and AppKit windows only; the Win32 window does not read it |
| T | `REIMS_VGPU_WINDOW_IMPL` | AppKit | macOS only |

### Metal2vulkan fast paths

All default on; `0` turns one off. Counters appear on the `work:` line (below).

| | variable | effect |
|---|---|---|
| S | `M2V_FAST_DRAW` | reuses the last settled draw when nothing changed since (counter `m-fast-draw`) and skips refreshing a bind-state stage whose table stamp, pipeline interface and device binding generation are unchanged (`m-stage-skip`) |
| S | `M2V_FAST_INSTANTIATE` | reuses the instantiated `DrawCommands` of an adjacent, exactly equal draw (`m-draw-commands-hit`) |
| S | `M2V_DRAW_SETTINGS_CACHE` | reuses `DrawSettings` of adjacent same-generation quiet draws (`m-draw-settings-hit`) |
| S | `M2V_DESCRIPTOR_CACHE` | reuses the previous set-0 descriptor set when every write is equal (`m-dset-hit`) |
| S | `M2V_DRAW_RETENTION_CACHE` | skips `retain` of a buffer identical to the last retained one in the same role (`m-retain-hit`) |
| S | `M2V_DRAW_USE_SUMMARY` | merges an unordered write whose range equals a tracked one (`m-use-summary-hit`) |
| S | `M2V_FRAMEBUFFER_CACHE` | caches attachmentless framebuffers by kept pass, extent and layers (4096 entries; `m-framebuffer-hit`) |
| S | `M2V_PASS_CACHE` | caches final render passes by shape, ops and resolve ops (4096 entries, never evicted before device teardown) |
| S | `M2V_CONTEXTS_POOL_TLS` | `0` restores the owning thread-local pool in `contexts/mod.rs` |
| S | `M2V_BUFFER_POOL` | same switch as `REIMS_VGPU_BUFFER_POOL` |

### Logging and diagnostics

| | variable | default | effect |
|---|---|---|---|
| D | `REIMS_VGPU_LOG` | off | stderr logger and its filter: a level (`warn`, `debug`, `trace`) or per-target directives, `warn,reims_vgpu_vulkan::gpu::timing=debug`. Lines look like `reims-vgpu WARN <target>: <message>` |
| D | `REIMS_VGPU_LOG_TIME` | off | any value prefixes each line with `[<epoch seconds.microseconds> <thread>]` |
| D | `REIMS_VGPU_LOG_ASYNC` | off (**1 in the launcher**) | `1` writes stderr lines from a background thread through a 65536-line queue; a full queue drops lines ("log lines dropped"). Errors stay synchronous |
| D | `REIMS_VGPU_STATS` | off (**1 in the launcher**) | `1`, `true`, `yes` or `on`: once a second, written straight to stderr (not through the logger), a `reims-vgpu stats:` line per display and a `reims-vgpu work:` line |
| D | `REIMS_VGPU_SAMPLE` | off | `1` starts the sampling profiler (below) |
| D | `REIMS_VGPU_CMDHASH` | off | `1` prints one `reims-vgpu cmdhash:` line a second (below) |
| D | `M2V_TIMING` | off | any value: metal2vulkan writes `[<epoch> <thread>] m2v-timing <what>: <ms> ms, <bytes> bytes` for each wrapped call of 1 ms or more |
| D | `REIMS_VGPU_CAPTURE_DIR` | unset | enables bounded captures into the directory: the presenting command buffer's source and target textures (every 120th present, up to 2400, BGRA8 only), the wallpaper surface's bytes, and writes to 3840x2160 render targets |
| D | `REIMS_VGPU_CAPTURE_PIPELINES` | unset | with the capture directory: comma list of substrings; pipelines whose stages contain one are captured |
| D | `REIMS_VGPU_CLEAR_TEST` | off | `1` or `true`: the window clears each acquired image to a cycling colour instead of the guest's frame, to tell a present fault from a content fault |
| D | `REIMS_VGPU_DEBUG_PRESENT_COVERAGE` | unset | any value: a magenta clear with the synthetic backdrop bypassed, so any part of the target the present misses shows |
| D | `REIMS_VGPU_DIAG_SKIP_FILL` | off | `1` leaves transfer buffers of 1 MiB or more unfilled (wrong pixels), to measure what the copy costs |
| D | `REIMS_VGPU_CENSUS_DIR` | unset | the Vulkan platform's degradation census; **on Linux only** |
| D | `REIMS_VGPU_EVENT_STATS` | off | read by QEMU: `1` prints `REIMS_INPUT_STATS enqueued= coalesced= processed= bh= ...` lines (input event queue depth, age and bottom-half time) |
| D | `REIMS_VGPU_PROBE_TARGET` | unset | read by QEMU: forces the encode and read-back path and prints frame-target coverage and centre pixel |
| D | `QEMU_AUDIO_STATS` | off | read by QEMU's `usb-audio`: `1` enables its statistics trace events |

### QEMU

| | variable | default | effect |
|---|---|---|---|
| S | `QEMU_WHPX_MMIO_FAST` | on | `0` disables the WHPX MMIO fast path for common `mov` forms (64-bit mode, memory operand) and emulates every MMIO access |
| E | `QEMU_PCORES` | off | `1` restricts the whole QEMU process (vCPU threads included) to the highest efficiency class's CPU sets. No gain was measured |
| D | `WHPX_MMIO_STATS` | off | set by the launcher with `-Profile` for "fast-path counters every 50000 accesses"; **no read of it exists in the current `vendor/qemu`** (unverified: an older build may have had it) |

Names that appear only in the launcher's clear list and exist in no source: `REIMS_VGPU_REFRESH`.
Variables in `reims-ctl` (`REIMS_VGPU_WINDOW`, `_LOG`, `_WINDOW_BORDERLESS`, `_WINDOW_REFRESH`)
belong to the Linux and macOS launcher, not to `boot-perf.ps1`.

## Performance design

The guest's WindowServer is not CPU bound while dragging a window; it waits on the host. One frame
is about 50 exec command buffers (FIFO command 55, `ExecIndirect2`), 2,900 to 3,200 a second while
busy, and the guest often waits on them one after another, so the latency of a batch matters as
much as throughput. The pieces below are what the branch does about it. Each has a switch in the
reference above.

- **Decode workers with ordered turns** (`fifo/decode_worker.rs`). The FIFO thread keeps the
  barrier wait, the stamp, `ensure_mapped`, the exec's command buffer, the copy of the command
  stream and the reservation of the FIFO event's value. A pool of 4 workers per FIFO runs the
  rest: decode into the exec's command buffer, commit, and the synchronize buffer. Commits pass
  in FIFO order through a `Turn`; a job's turn passes when it ends, whatever it did. At most
  `MAX_AHEAD` = 16 execs are handed on and unfinished, under the queue's command-buffer limit.
  Every other FIFO command first drains the workers, so a command that changes the object lists
  or ends the task never overtakes an earlier exec. `SynchronizeResources` hands its read lock to
  the worker, so the task lock is never held while waiting on the GPU. With early commit, the
  completed handler and the signal are encoded before the wait and the turn passes right after
  the commit.
- **Prepare off the FIFO thread** (`REIMS_VGPU_ASYNC_PREPARE`) is implemented as a second ordered
  gate in front of the decode. It made things slower in the one run that measured it (a wait for
  the prepare turn) and the launcher turns it off.
- **Frame copy.** The frame the window draws is copied first, off the draw path.
- **Interrupt coalescing.** One MSI per stamp signal meant thousands of guest interrupts a second,
  each costing three MMIO reads (registers 0x1018, 0x1014, 0x102c) and VM exits. A signal that
  finds stamp bits already pending raises nothing; the pending handler's clear-on-read returns the
  new bit too. In a guest `spindump` the AppleParavirtEventMachine thread used 23% of a core
  before the change; a repeat spindump after it was not taken (unverified).
- **WHPX MMIO fast path** (QEMU `whpx_handle_mmio`): common `mov` forms to and from device
  registers are decoded and executed without the instruction emulator.
- **Guest RAM imported in chunks.** Importing a 64 MiB stretch at the first buffer that touched
  it stalled the FIFO thread for 40 to 90 ms (a `vkGetMemoryHostPointerPropertiesEXT` on the big
  `VirtualAlloc` region costs about 30 ms, the allocation 7 to 38 ms). The device now imports
  guest RAM at creation in 256 MiB chunks (12 GiB: 50 chunks in 1.8 s), and buffers bind at an
  offset in a kept import. The launcher turns this off (see Known issues, host memory).
- **No-copy buffer pool.** When the last hold on a no-copy buffer's pages goes and its `VkBuffer`
  is bound in a kept import, the `VkBuffer`, placement and device address are parked, keyed by
  (first byte, length, memory class), and reused for the same key. Caps: 4 per key, 8192 in all.
- **GPU gather.** A transfer of scattered guest pages of 1 MiB or more, averaging runs of 16 KiB
  or more once adjacent pages are merged, is assembled by one `vkCmdCopyBuffer` with many regions
  from the imported chunks, in the paging encoder. Shared buffers, heaps and IOSurface paths still
  copy on the CPU (their consumer is the exec being decoded).
- **Cursor pool, transfer cache, scratch classes.** Command-stream copies, transfer buffers (keyed
  by physical page runs) and copy-fallback ranges (size classes; 94% reuse in the run that added
  them) come from pools.
- **Thread-local destructors.** On the Windows GNU target every access to a `thread_local!` with a
  destructor goes through a registration guard (`std::sys::thread_local::guard::windows::enable`);
  it was 6.7 to 8.7% of busy samples. Hot-path thread-locals (`CURRENT` in `decode_worker.rs`,
  the lock read table, the platform scopes, `DEFERRED_SYNCS`, `block_env`, `held`, the runtime's
  contexts pool, the Vulkan crate's pending-answer flag) are `const` cells or raw pointers with no
  destructor.
- **mimalloc** is the global allocator of `reims-vgpu-c` on Windows: the heap and its locks were a
  large share of busy time (zero-filled allocations were 5.6% of samples).
- **Draw fast paths in metal2vulkan** (table above): most UI draws change state between draws, so
  these hit rarely; only `m-stage-skip` fired often (about 9,000 a second).
- **Thread placement.** The decode workers and FIFO threads run one priority step above normal and
  on the highest efficiency class's cores (E-cores ran the decode about 1.5 times slower); QEMU
  opts out of power throttling (EcoQoS) so that on battery Windows does not move it to E-cores.

### Measurements and what was tried

Figures are from the `reims-vgpu work:` and `stats:` lines of the dev runs (logs in
`qemu-vm\boot-stderr-*.log`). Runs are not matched A/B pairs: workloads differ (window drags,
Safari, Apple TV, 4K60 video, DaVinci Resolve) and several include idle or start-up seconds.
Treat the figures as orders of magnitude.

| build (change it added) | what was measured |
|---|---|
| start of the work | about 5 fps in a CPU-drawn animation test |
| AC power, texture-copy work | 24 to 30 fps in that test; 22 on battery with the power-throttling opt-out, 17 without |
| parallel decode (`pardec6`) | per exec 130 us on the FIFO thread plus about 380 us decode and commit on a worker; draws about 35 us each (about 4.2 per exec) |
| `fastdraw11` | fps median 41, p90 79; worst gap per second median 72 ms, p90 323 ms. Slow execs (over 5 ms) summed 16.6 s in the test; worst 2.4 s (180 textures paged in one exec) |
| `bufpool1` | exec decode 0.67 ms each; fps median 39 to 52, p90 62 to 76; max gap median 72 to 98 ms, p90 340 to 380 ms. Host GPU mostly idle: the limit was CPU translation and per-buffer memory import |
| `cmdhash1` | shape repeat p10/p50/p90 90/95/98%, full repeat 1 to 16%, 180 to 270 distinct shapes a second; decode of a repeated shape 331 us against 1103 us for a new one |
| channel-1 time per second while interacting | `pgsync1` 850 to 970 ms; `shdel1` 820 to 920; `bundle1` 800 to 850 (about 475 us per exec at 1,670 execs a second); goal was under 700 |

A summary of `reims-vgpu stats:` seconds with more than 1,000 execs a second (mixed workloads,
medians of the per-second guest present rate and max gap), computed from the logs on 2026-10-19:

| log (build) | seconds | guest fps median / p90 | max gap median / p90 (ms) |
|---|---|---|---|
| `perf-20261018-122534` (`dualfix1`, 2 displays) | 915 | 54 / 59 | 24 / 133 |
| `perf-20261018-161948` (`pardec7`) | 907 | 69 / 84 | 25 / 132 |
| `perf-20261018-195941` (`fastdraw11`) | 204 | 54 / 79 | 56 / 330 |
| `perf-20261018-202217` (`bufpool1`) | 138 | 53 / 91 | 75 / 343 |
| `perf-20261018-204320` (`cmdhash1`) | 276 | 64 / 110 | 83 / 375 |
| `perf-20261018-230937` (`bundle2`) | 292 | 70 / 120 | 85 / 242 |
| `perf-20261019-135047` (`resolve3`, Resolve running) | 534 | 59 / 123 | 64 / 299 |
| `perf-20261019-151401` (`flicker2`) | 545 | 37 / 74 | 100 / 406 |

The last row's drop against earlier builds has no explanation recorded; the run's workload was
not noted (unverified whether it is a regression).

**Tried, and did not help**

- **A small no-copy buffer pool** (`bufpool1`). About half of the creations hit the pool; the
  per-creation cost (`m-nc-import`, about 0.75 ms each) stayed, because the cost is the driver's
  host-pointer import and bind, not `vkCreateBuffer`. In the kept-import path there is no per-page
  loop in our code; the remainder is `vkBindBufferMemory` over host-pointer memory, which scales
  with the buffer's pages. The pool stays in, on.
- **Async prepare** (`bundle2`): slower, because of the wait for the prepare turn. Off in the
  launcher.
- **A conversion-plan or draw-plan cache.** Analysed from a busy second (977 execs, 4344 draws,
  `dec-render` 371 ms): the pieces that depend only on a command stream's shape (settle, plan,
  pipeline, targets, uses) add up to about 8 to 10 us a draw, and are already cached in the
  runtime's own memos (`settle_shape`, the settings cache, the pass cache). A plan cache would
  have saved about 35 ms on the heaviest second and about 2 ms typically, with a real risk of
  stale resource identities. Not built. The terms that dominate a busy second are per-resource
  (`m-nc-import` 241 ms, `tex-ensure` 236 ms, paging 262 ms) or ordering waits (`step-turn`
  294 ms). Whole-pass Vulkan secondary-command-buffer replay is the only variant with leverage
  and needs dynamic-offset plumbing and tracker feeds (not started).
- **Command-stream hashing as a cache key**: full-byte repeats are 1 to 16%, so only the shape
  repeats (95%); see above.
- **Folding the paging command buffer into the exec's**: no gain (an earlier session's A/B).
- **Skipping texture copies / GPU gather alone**: about +3 fps in the animation test; the GPU
  gather of 4,100 scattered 4 KiB regions was slower than the CPU copy (paging queue 417 ms a
  second against 103), which is why it has a run-length floor.
- **P-core pinning of QEMU** (`QEMU_PCORES`): no gain.
- **Per-draw `vkCmdDraw` bypass** (an early return that skips preparation for a repeated draw):
  not added, since it would bypass range validation, detach refusal, ordered hazard prefixes,
  retention and capture records.
- **Advertising 120 Hz**: nothing to change; the guest is already offered the host screen's
  highest refresh rate. There is no `REIMS_VGPU_REFRESH`.

**Still open** (from the notes): a full preparation bypass for changed draw arguments; a
texture-backed framebuffer cache; scheduler hand-off and submission batching (C3, C4); recycling
`VkImage`s by (format, size, usage) for the guest's roughly 500 resource creations and deletions
a second while scrubbing video; a CPU path for the first upload of small textures; and pipeline
cache persistence across runs (check whether metal2vulkan's shader cache already covers it:
unverified).

## Profiling tools

The helper scripts (`perf-report.py`, `sample-report.py`, `perf-cycle.py`) lived in `qemu-vm\`; the
2026-10-19 cleanup list marks them deletable. What each did is written out below, so they can be
rewritten.

### `stats` and `work` lines

With `REIMS_VGPU_STATS=1` (the launcher's default) stderr carries, once a second per display:

```
reims-vgpu stats: port 0: guest N/s, max gap N ms; window N/s, max gap N ms; window GPU N ms avg, N ms max;
  window CPU acquire N calls, N ms total, N ms max; ... encode ...; ... present-call ...;
  window outcomes N presented, N retry, N ended, N nothing
```

`guest` is presents the guest made; `window` is frames the window drew; "max gap" is the longest
time between two in the second. A `reims-vgpu work:` line follows. Each term is `name count/total-ms` for the second. **Terms nest, so do not add them.**

| term | meaning |
|---|---|
| `exec` | `ExecIndirect2` commands, whole (FIFO thread's part) |
| `decode-worker` | the worker jobs: decode, commit, synchronize |
| `step-lock`, `-ensure_mapped`, `-prepare_resources`, `-cursors`, `-decode`, `-commit`, `-complete_resources`, `-turn`, `-handoff` | the exec's steps; `turn` is the wait for the commit turn (ordering, not work) |
| `prepare`, `prep-lookup`, `prep-paging`, `prep-tail`, `prep-mutex` | resource preparation: the lookup loop (includes the preparer mutex wait), GPU paging, CPU paging and invalidation |
| `lookup-slow`, `ls-<kind>` | lookups of 20 us or more (first-use creation), by resource kind |
| `paging`, `pg-<step>`, `pb-<kind>`, `pc-<kind>` | GPU paging rounds, their steps (`cmdbuf wait blit-enc blit-end comp-enc fills commit`), and blit / compute prepares by kind (`sbuf pbuf dbuf hbuf mbuf lheap dheap icb tex stex rtex mtex shtex back`) |
| `bind-map`, `bind-buffer` | page binds: the guest-page mapping, and the buffer over it |
| `tex-ensure`, `tex-sync` | texture prepares: binding pages, and the page copy |
| `copy`, `copy-mb` | the copy fallback and its megabytes |
| `transfer-hit`, `scratch-fresh`, `scratch-reused` | the transfer cache and scratch pool |
| `gather-gpu`, `gather-cpu`, `gather-gpu-kb`, `gather-cpu-kb` | GPU gathers, and gathers asked for and left to the CPU |
| `prep-async`, `prep-turn` | the async-prepare path (zero while it is off) |
| `dec-render`, `dec-compute`, `dec-blit`, `dec-info`, `dec-encoder` | decode by encoder type; `dec-encoder` is a fresh segment's encoder creation |
| `rec<id>` | the 8 slowest record ids; only in a build with `--features reims-vgpu-c/record-timing` (per-record timing costs two clock reads per record, tens of thousands a second) |
| `task<id>`, `ch<N>` | execs by guest task id, and command time by FIFO channel |
| `del-lock`, `del-obj`, `unmap-lock`, `unmap-list`, `unmap-tracker`, `unmap-host` | DeleteResource and UnmapMemory, by lock wait and work |
| `cmd<id>` | the 6 slowest FIFO commands |
| `plane-fill`, `plane-copy` | reference-texture plane prepares |
| `m-*` | metal2vulkan's counters (`perfcount.rs`, feature `performance-counters`, on in the Vulkan crate): `m-draw-resident`, `m-draw-settle`, `m-enc-new`, `m-enc-end`, `m-commit`, `m-fast-draw`, `m-stage-skip`, `m-dset-hit`, `m-nc-*` (no-copy buffer creation: `total`, `import`, `ramhit`, `rammiss`, `kept-bind`, `kept-import`, `own-import`, `poolhit`, `pool-store`) |
| `segments`, `records`, `resources`, `paged` | totals |

`perf-report.py a.log b.log [--window START:END]` printed, per log, the average rate and time per
second of each term over the busy seconds (those with more than 1,000 execs), with optional
inclusive log-line windows (a run's logs carry no timestamps unless `REIMS_VGPU_LOG_TIME`); with two
logs it put them side by side. `perf-cycle.py` drove an animation benchmark in the guest. The same
summary in a shell, averaging each term over the interaction seconds (rate, ms a second, us per
unit):

```sh
grep "reims-vgpu work:" boot-stderr-<tag>.log | awk '{split($4,a,"/"); if (a[1]>1000) print}' \
  | grep -o " [a-z][a-z0-9-]* [0-9]*/[0-9.]*ms" \
  | awk '{split($2,a,"/"); sub("ms","",a[2]); n[$1]+=a[1]; t[$1]+=a[2]; c[$1]++}
         END{for(k in n) printf "%-18s %6d/s %7.1f ms/s %6.1f us\n",k,n[k]/c[k],t[k]/c[k],(n[k]?1000*t[k]/n[k]:0)}' \
  | sort -k3 -rn
```

The frame rate and gap of the same seconds:
`awk '/reims-vgpu stats/{s=$0} /reims-vgpu work:/{split($4,a,"/"); if (a[1]>1000) print s}'`.

### The sampling profiler

`-Sample` (`REIMS_VGPU_SAMPLE=1`, Windows only): the decode workers and FIFO threads register
themselves; a sampler thread wakes every 500 us, suspends each registered thread, reads RIP and
resumes it. Nothing is allocated or locked while a thread is suspended and no vCPU or host thread is
ever suspended. Every 10 s the samples are appended to `samples-<pid>.bin` in the working
directory:

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `RVGS` |
| 4 | 4 | version, little-endian `u32` (1) |
| 8 | 8 | the executable's load address, `u64` |
| 16 | 8 each | instruction pointers, `u64` little-endian |

`sample-report.py samples-<pid>.bin <qemu exe> [top]` removed the most common 64 KiB page outside
the exe (the kernel wait stubs, idle threads), subtracted the load address and added the image base
(`0x140000000`), mapped the rest to functions with `addr2line -f -C -i -a` (MSYS2 binutils on
`PATH`, DWARF from the release build), and printed the top functions by innermost frame and by
outermost non-inlined function. A busy profile of one build
(`pardec8`): 29% of busy samples outside the exe (the NVIDIA driver, ntdll heap and locks), the
thread-local guard 8.7%, mimalloc about 8%, CPU copy of scattered pages 4.9%, then a flat tail led
by `draw_resident` 1.7%.

### The `cmdhash` line

`-Profile` (or `REIMS_VGPU_CMDHASH=1`) hashes every exec's command stream twice: a full FNV-1a
hash of all bytes, and a shape hash (segment kinds and flags, each record's command id, the
pipeline state reference of a render pipeline set; no immediates). Once a second one warn line:

```
reims-vgpu cmdhash: execs N, full-repeat X%, shape-repeat Y%, distinct-shapes K,
  hit-decode A us avg, miss-decode B us avg, top: <hash32>:<count> ...
```

A repeat is a hash seen in the previous or the current one-second window. `hit-decode` and
`miss-decode` are the average decode times of shape hits and misses.

### Other

- `M2V_TIMING` lines and `-Profile`'s timestamped debug filters line up with QEMU's `-msg
  timestamp=on` trace in `qemu-gfx-<tag>.log` (both use epoch seconds).
- `-Profile` adds per-exit cost; do not compare its fps with an unprofiled run.
- Guest side: `spindump -notarget 10` (kernel symbols are stripped; attribute by thread name and
  kext), and `top` logs. Do not run several `log show` queries in a row; they swamp the guest.

## Known issues and limits

- **No hardware video decode.** `VTIsHardwareDecodeSupported` is false for `avc1`, `hvc1`, `vp09`
  and `av01`: VideoToolbox has no host path, so decode is software in the guest. VP9 1080p60 in
  Safari dropped 0 of 331 frames in a probe; 4K60 VP9 showed 8 dropped of 5,521 in a brief check
  and was not tested for long. MSE supports `avc1` and VP9, not AV1. The H.264 and HEVC encoders
  are software.
- **YouTube needs an audio device** (`-Audio usb`, the default). Safari's MSE player dies with a
  `DecodeError` when its audio queue cannot start.
- **Frame rate.** 60 fps is not reached in desktop use: medians sit between the 30s and 60s with
  a long tail of 100 to 400 ms gaps while apps start (first uses create resources and page
  textures in). The goal was 60 fps including on battery. Residual jitter and, at times, audio
  distortion (USB audio queue underfill, with packet drops when the ring was full) are open.
- **DaVinci Resolve.** Its metallibs are debug builds. metal2vulkan-air used to refuse them
  (`llvm.dbg.cu`, then `llvm.module.flags` entries), so `MSAAIMAverageResolveKernel` failed to
  build and `SetupMultiSampling` divided by a zero `threadExecutionWidth`, which killed the
  guest process with SIGFPE. Two fixes: metal2vulkan-air tolerates debug metadata, debug module
  flags and `llvm.dbg.*` intrinsics (`M2V_AIR_DEBUG_TOLERANT`), and the pipeline info reply never
  reports a zero width (`REIMS_VGPU_PIPELINE_WIDTH_CLAMP`; 0 becomes 32, and 1024 for the
  maximum). Other AIR gaps in Resolve's metallibs may remain (not checked offline). With the
  `resolve3` build and later, Resolve 21.1.1 starts, loads its pages and generates proxy media
  (confirmed on a VM boot on 2026-10-19); the debug-build MSAA kernel is the only shader the
  fixes newly accepted in the logs that were compared.
- **Vulkan device lost.** In four runs on 2026-10-18 (`perf-...-220946`, `-222412`, `-223217`,
  `-224314`), during YouTube video, the process ended with `fatal: platform call
  newTextureWithDescriptor:iosurface:plane:: ... the Vulkan device was lost` and exit code
  `-1073740791`. It followed the first build in which Resolve's MSAA kernel was accepted instead
  of refused; it persisted with RAM import, the buffer pool, early commit and the event fix each
  turned off in turn. No validation-layer run was made. The prime suspect is that kernel
  (compiled -O0 from a debug build); the bisect prepared was `M2V_AIR_DEBUG_TOLERANT=0`. Not seen
  in the 2026-10-19 logs; the cause is unconfirmed.
- **Flicker under host memory exhaustion** (Dock, wallpaper and window borders vanish for
  seconds). During a Resolve session with 12 GiB of guest RAM imported and Resolve's copy
  buffers, 0.57 GB of 32 GB was free. A refused no-copy buffer then made WindowServer's own
  exec fail; the guest saw `kIOAccelCommandBufferCallbackErrorHang`, restarted the channel and
  discarded the frame. The mitigations are the one retry over a private copy
  (`REIMS_VGPU_NOBUF_RETRY`) and dropping the exec as done (`REIMS_VGPU_NOBUF_SOFT`); the error
  text that was swallowed in `new_buffer_with_descriptor` is now logged (first 20, then a count
  a second). The fix was verified only by its build; the root cause (a transient host-memory
  failure in the no-copy import path) is not proven. Pages that were refused stay held
  (`keep_forever`), since unmapping could alias.
- **Host memory.** The guest has 10 GiB. Importing all guest RAM (`REIMS_VGPU_RAM_IMPORT` on)
  pins it in the driver at once: with the import on, the host paged and the SSD sat at 100% on
  the test laptop (2026-10-19), which is why the launcher sets it to 0. With it off the
  first-use import stalls of 40 to 90 ms return. Keep a few GB of host RAM free beyond the
  guest's 10 GiB and the Vulkan allocations; a smaller `REIMS_VGPU_RAM_IMPORT_CHUNK_MB` does not
  reduce the pinned total.
- **Restart and sleep.** A guest restart exits QEMU with 99 and a fresh QEMU boots (the warm reset
  stalls: the APs never start). After "Restart" macOS stalled 1 to 5 minutes before its
  confirmation dialog in the one test that measured it (a guest-side stall, with no device
  activity). Real S3 sleep does not work; "Sleep" is a display sleep. Shut Down works.
- **Empty-entry exec drops.** A dead process's queued commands run against a new process that
  reused its task id; they fault ("object has an empty entry") and used to get the new process
  killed (blank icons, crash loop). They are dropped with a warning `exec dropped` while the task
  is young. A few such warnings at start-up are expected.
- **Input.** USB keyboard and tablet only; multi-display pointer through the guest agent. A
  touchpad passthrough is a separate prototype, not part of this build.
- **The window and swapchain.** A first-present freeze was seen on 2 of about 12 launches
  ("no swapchain yet (window surface has no pixels)": window 0 fps while the guest renders).
  The device runs without `VK_KHR_swapchain_maintenance1`, so it waits for the presenting queue
  to idle.
- **Tests on Windows.** `cargo test -p reims-vgpu-core` has failures that are not regressions (10
  `new_range_mapping`, 2 `device::fifos`, 2 `gpu_task::exec` log tests, 2 tests that map 32 GiB);
  run it filtered and single-threaded (`-- <filter> --test-threads=1`). metal2vulkan's Windows
  tests do not compile (Unix fork tests); check it with `cargo check -p metal2vulkan-runtime
  --lib`.

### QEMU exit codes

| code | meaning |
|---|---|
| 0 | the guest shut down, or QEMU quit |
| 99 | a guest restart (`-action reboot=shutdown`); the launcher starts a fresh QEMU |
| -1073740791 (`0xC0000409`) | a fail-fast abort; the Reims library ends the process on a fatal error, and the log's last line says why (`reims-vgpu: fatal: ...`) |
| 1 | seen 8 times in the launcher logs, at start-up; reason not recorded (unverified) |
| -1, `-2147483645` (`0x80000003`) | seen in the launcher logs; meaning unverified |

## Troubleshooting

- **A black Reims window, guest running.** First suspect: the QEMU executable was built from
  an `apple-gfx-pci.c` and `reims_vgpu.h` that do not match the staticlib it links (the
  `flicker1` build did this: the object came from the merged upstream file, with process
  isolation and IOSurface-over-PCI that this library does not have). Compare `QEMU_SHA256` and
  `STATICLIB_SHA256` in `boot-gfx-<tag>.log` with what you built; the launcher stops on a
  manifest mismatch but not on a header mismatch. Rebuild the object against the header of the
  library you link. Other black-screen causes that were ruled out: alpha (opaque), cross-adapter
  presenting, the window itself (`REIMS_VGPU_CLEAR_TEST=1` shows the present path works).
- **Reading a run's logs** (all in `qemu-vm\`, named by tag; a restarted guest adds `-boot2`, ...):

  | file | content |
  |---|---|
  | `boot-gfx-<tag>.log` | the launcher's: settings, `QEMU_PID`, hashes, `=== QEMU exited rc=N ===`, `=== guest restart ===` |
  | `boot-stdout-<tag>.log`, `boot-stderr-<tag>.log` | QEMU's output; stderr holds Reims's log, the `stats:` and `work:` lines, `guest RAM import:` and `sampler:` lines, QEMU warnings |
  | `guest-serial-<tag>.log` | the guest's serial port (`serial=3` in boot-args freezes the on-screen verbose console at `kdp_core`; the ESP `livev` image avoids it) |
  | `qemu-log-<tag>.log` or `qemu-gfx-<tag>.log` | QEMU's `-D` log; the second with timestamps and trace events (`-Profile`, `-PowerTrace`) |
  | `display-persistence-<tag>.log` | the display-count helper's errors |
  | `ovmf-<tag>.log` | OVMF's debug console, with `-FirmwareLog` |
  | `samples-<pid>.bin` | the sampler's output |

  Do not tail `boot-gfx-<tag>.log` during restart tests (a reader holding it open made writes fail
  before the launcher retried them). A line like `exec dropped` or `exec faulted` names the
  task; `log lines dropped` means the async log queue overflowed.
- **Always give QEMU a `-D` log file.** Without it QEMU's trace and log output goes to stderr,
  which MinGW leaves unbuffered (one `WriteFile` per byte); a trace flood then starves the vCPUs
  and input, and the guest freezes while the Reims window keeps drawing. The launcher always
  passes `-D`.
- **A QEMU crash or silent exit.** Look for the `=== QEMU exited rc=` line first: a real crash
  writes one. The Windows Event Log may hold an Application Error or Windows Error Reporting
  entry for `qemu-system-x86_64*.exe`:
  `Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=(Get-Date).AddHours(-3)} | ? { $_.Message -match 'qemu-system' }`
  (whether an entry exists for every crash is unverified).
- **The VM vanished with no exit line.** It was probably started from an app or tool that
  restarted, which kills its process tree (the launcher and QEMU together). Check whether that
  app's processes restarted at that time, and start long runs detached (Running).
- **No network or SSH.** macOS did not bind `virtio-net-pci`: relaunch with `-Nic vmxnet3`. Check
  `ssh -p 2222 ...` after boot.
- **Keys stick or repeat.** An EHCI keyboard queue overflow; use `-Usb xhci` (the default).
- **A total freeze with the window still drawing**, input dead: see the `-D` note above.
- **"A QEMU is already running."** The launcher allows one `qemu-system-x86_64*` process.
- **"This QEMU lacks display hotplug."** The exe predates `display-ports`; use a newer one or
  `-LegacyDisplay`.
- **Everything slow on battery.** QEMU opts out of EcoQoS power throttling at start-up; if Windows
  still moves it, check the power mode. Verify with a profile run, not by feel.
- **Guest setup problems:** blank icons or crash-looping Spotlight/Dock extras (the
  `FORCE_SOFTWARE_RENDERER` section), white Liquid Glass (Reduce transparency), YouTube error
  (audio).
