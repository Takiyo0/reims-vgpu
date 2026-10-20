# WHPX / QEMU feasibility gate: `apple-gfx-pci` on a native Windows host

Scope: a *gate*, not a port. Question answered: what is the **smallest** set of
source/config changes to make QEMU even *build* `apple-gfx-pci` for a Windows +
WHPX `x86_64-softmmu` target, and does WHPX's memory + MMIO model allow the
device to function? Nothing here is implemented. No commit/push/remote.

Host for this assessment: i7-13650HX / RTX 4060 Laptop. Toolchain now present
(meson 1.12.1, ninja 1.12.1, MSVC cl 19.44, gcc 16.2 ucrt64, pkg-config).
`vendor/qemu` submodule initialized at `1a72df9`.

Verdict up front: **the build gate is small and mechanical (3 edits). The
runtime gate is NOT — it needs one genuinely new piece: a WHPX guest-RAM →
GPU-import bridge that replaces the Linux fd/`memfd` path with a host-pointer
path. The RTX 4060 probe already proved the keystone of that bridge works.**
Recommendation: do the minimal build + the `-accel qtest` MMIO experiment
(neither needs the bridge) to de-risk before any port.

---

## 1. Why the device is excluded from a Windows build today

Three independent gates, two in QEMU and one implied by the library's C API.

### Gate A — QEMU does not look up the library off Linux/macOS
`vendor/qemu/meson.build:882`
```meson
if host_os in ['darwin', 'linux']
  pvg = dependency('reims-vgpu', method: 'pkg-config', static: true, ...)
```
On Windows `pvg = not_found` (set at `meson.build:835`), so the device sources
are never added (`hw/display/meson.build:74`, `when: [pvg, 'CONFIG_MAC_PVG_PCI']`).

### Gate B — `CONFIG_MAC_PVG` only enabled for HVF/KVM targets
`vendor/qemu/meson.build:3422-3428`
```meson
# PVG is not cross-architecture ... HVF's on macOS, KVM's on Linux.
if pvg.found() and (target in accelerator_targets.get('CONFIG_HVF', []) or
                    target in accelerator_targets.get('CONFIG_KVM', []))
  target_kconfig += 'CONFIG_MAC_PVG=y'
endif
```
WHPX is deliberately not in the set. Note WHPX *is* a first-class accelerator in
this tree: `accel/whpx/`, `target/i386/whpx/`, and
`accelerator_targets['CONFIG_WHPX'] = ['x86_64-softmmu']` (`meson.build:314`),
enabled when `winhvplatform.h` is found (`meson.build:917-926`).

### Gate C — the Kconfig device node is NOT host-gated (good)
`hw/display/Kconfig:157-160`
```
config MAC_PVG_PCI
    bool
    depends on MAC_PVG && PCI
    default y if PCI_DEVICES
```
Only `MAC_PVG && PCI`. No `depends on HVF`. So once Gate B sets `MAC_PVG=y` for
the WHPX x86_64 target, `apple-gfx-pci.c` + `apple-gfx.c` compile. (By contrast
`hw/vmapple/Kconfig` hard-requires `depends on ARM` + `depends on HVF` — the ARM
`vmapple` path is macOS-only and irrelevant to WHPX. Windows is x86_64 PCI only.)

### Already-correct Windows seams
- `hw/display/meson.build:68-73`: activity backend already selects the
  Objective-C `apple-gfx-activity.m` only `if host_os == 'darwin'`, else the
  portable `apple-gfx-activity-stub.c`. Windows gets the stub for free.
- `hw/display/apple-gfx.c:729-731`: the `memory_files`-missing branch is
  `#ifdef CONFIG_LINUX … #else *memory_files = NULL; return true;`. On Windows
  the `#else` is taken, so realize does **not** hard-fail on missing fds — it
  passes `memory_files = NULL`, exactly as macOS does for anonymous RAM.

---

## 2. Smallest changes to BUILD the device for WHPX

Three edits, all mechanical. They make it compile and link; they do **not** make
guest RAM reach the GPU (that is §3).

1. **Library lookup on Windows** — `vendor/qemu/meson.build:882`
   ```
   -  if host_os in ['darwin', 'linux']
   +  if host_os in ['darwin', 'linux', 'windows']
   ```
   Requires the library to emit a Windows `reims-vgpu.pc` whose `Libs`/
   `Libs.private` are MSVC-link-correct (`.lib`, system libs, no `-lSystem`/
   `-lobjc`). `cargo xtask dist` currently writes a Unix-style `.pc`
   (`docs/qemu.md`, "The library"); this is a real sub-task, not a one-liner,
   but it is packaging, not device code.

2. **`CONFIG_MAC_PVG` for the WHPX target** — `vendor/qemu/meson.build:3426`
   ```
   -  if pvg.found() and (target in accelerator_targets.get('CONFIG_HVF', []) or
   -                      target in accelerator_targets.get('CONFIG_KVM', []))
   +  if pvg.found() and (target in accelerator_targets.get('CONFIG_HVF', []) or
   +                      target in accelerator_targets.get('CONFIG_KVM', []) or
   +                      target in accelerator_targets.get('CONFIG_WHPX', []))
   ```

3. **The option ROM install guard** — `vendor/qemu/meson.build:889-897` sits
   inside the Gate-A `if`, so widening Gate A already covers it; confirm the
   `optionrom` pkg-config variable resolves to a Windows-reachable path and that
   `ReimsVgpu.rom` (EFI x86-64 driver, host-agnostic bytes) installs. No code
   change expected beyond #1.

Nothing in `apple-gfx.c` / `apple-gfx-pci.c` needs touching to compile: they use
no Objective-C, Metal, IOSurface, dispatch, blocks, or POSIX syscalls
(`docs/qemu.md`, "apple-gfx on reims-vgpu"). They are plain C on `reims_vgpu.h`.

Build-only risk: QEMU on Windows is normally built in the MSYS2/ucrt64
environment (gcc), while the Rust static lib + `.pc` come from the MSVC
toolchain. Mixing MSVC-produced `libreims_vgpu.a`/`.lib` into a gcc/ucrt64 QEMU
link is the first thing that can break and must be settled in the experiment
(§5), e.g. by building QEMU itself with `--cc=cl`/clang-cl or by producing a
gcc-compatible static archive.

---

## 3. The runtime blocker: guest RAM → GPU import

This is the real gate, and it is independent of the build edits.

### The contract
`crates/reims-vgpu-c/include/reims_vgpu.h:389-399` — `RvgMemoryRangeFile` is
`{ int32_t fd; uint32_t flags; uint64_t offset; }`: "the range's host memory is
a shared mapping (MAP_SHARED) of `fd` from byte `offset`." It is a **POSIX file
descriptor**. On Linux the library builds GPU-visible views of guest pages from
these fds (`reims_vgpu.h:432-439`; `docs/qemu.md`, "The files behind guest RAM").

### What QEMU hands over per host
`hw/display/apple-gfx.c:642-736` (`apple_gfx_memory_range_file`,
`apple_gfx_memory_files`):
- Linux: each RAM range must be an fd-backed `MAP_SHARED` block in host-page
  size (`qemu_ram_is_shared && qemu_ram_pagesize == real_host_page_size`),
  typically `-object memory-backend-memfd,share=on`. A range without an fd fails
  realize.
- macOS/`#else`: `memory_files = NULL`; the library gets the flat guest ranges
  only and does its own thing (vmapple RAM is anonymous).
- Windows (would take `#else`): QEMU would pass `memory_files = NULL`.

### Where it breaks on WHPX
Under WHPX, QEMU allocates guest RAM as ordinary host virtual memory
(`VirtualAlloc`-class) and registers it with the hypervisor via `WHvMapGpuRange`.
There is **no fd and no section handle** for that memory. But the Windows Vulkan
backend's guest-RAM path is built entirely on **section objects it creates
itself**:
- `crates/reims-vgpu-vulkan/src/os/windows.rs:34` — `BackingHandle = Section`.
- `crates/reims-vgpu-vulkan/src/os/windows/section.rs:28-45` — `Section` is a
  pagefile-backed `CreateFileMappingW` object; views are `MapViewOfFile3`.
- `crates/reims-vgpu-vulkan/src/os/windows/files.rs` — the backing table stores
  `Backing { fd: Arc<Section>, offset }`.

So the current Windows backend expects to **own** the backing section, whereas
WHPX guest RAM is memory QEMU already allocated and the backend must **import**,
not create. The two models do not meet. This is the bridge that must be built,
and it is the whole of the runtime work.

### Why the keystone is already proven
The standalone probe `probes/windows-vulkan-host-import/` imported a
`VirtualAlloc` allocation into the RTX 4060 through `VK_EXT_external_memory_host`,
had the GPU write it, and read the value back through both mappings — PASS on
this hardware. `VK_EXT_external_memory_host` imports **by host pointer**, which
is exactly what WHPX guest RAM is. So the viable bridge is: QEMU passes the guest
RAM's **host base pointer + length** (not an fd) to the library, and the library
imports it with `VK_EXT_external_memory_host` instead of mapping an fd/section.
That is a C-API extension (`RvgMemoryRangeFile` gains a host-pointer form, or a
new field) plus a Windows import path in the Vulkan backend — bounded, and
de-risked by the probe, but new.

Secondary memory facts (favourable):
- No `vm.max_map_count` analogue on Windows; the backend already notes views are
  not budgeted (`files.rs` module docs). The Linux mapping-count ceiling
  (`docs/qemu.md`, "vm.max_map_count") does not apply.
- The library sizes guest pages for the host architecture; x86-64 host → 4 KiB
  pages (`reims_vgpu.h`, `APV_PAGE_SHIFT`), matching WHPX.

---

## 4. The MMIO model vs WHPX

`apple-gfx`'s MMIO contract (`docs/qemu.md`, "MMIO"; `reims_vgpu.h:775-786`):
- Reads/writes of the device's region call the library on the **vCPU thread**;
  the library never waits on QEMU. `raise_interrupt` only schedules a BH.
- The GPU region is **lockless** (`memory_region_enable_lockless_io`): under KVM
  a vCPU accesses it without the BQL; **under HVF the BQL is taken at every
  exit, TCG at every I/O access**, so under those the accesses serialize. WHPX,
  like HVF, exits to the VMM per MMIO and QEMU holds the BQL on that exit — i.e.
  WHPX lands in the **already-supported "serialized under the BQL" regime**, the
  same one HVF uses. Lockless is an optimization KVM gets, not a requirement.
- The one access that itself takes the BQL — the `apple-gfx-pci` 0x1210 write
  that moves BAR 2 (`docs/qemu.md`, "BAR 2") — runs after the library returns;
  WHPX already holds the BQL there, which is fine.
- The IOSurface registers are 32-bit MMIO (zero-extended/truncated); nothing
  host-specific.

Conclusion: the MMIO model is **WHPX-compatible on paper** — WHPX's
exit-with-BQL semantics match the HVF regime the device already supports. The
one thing to verify empirically is that WHPX actually delivers MMIO exits for
the device's BAR 0 region to QEMU's `MemoryRegion` ops (it does for normal
emulated PCI devices), and that MSI delivery (`apple-gfx-pci` uses `msi_notify`)
works under WHPX — WHPX MSI delivery has historically been a weak spot.

BAR 2 (`efi-framebuffer-bar`, the EFI framebuffer window onto guest RAM,
`docs/qemu.md` "BAR 2") is pure `MemoryRegion` alias plumbing, host-agnostic; it
was validated under KVM via `-accel qtest` and should behave identically.

---

## 5. Minimal experiments that prove/disprove the path

Two experiments, both small, neither needs the §3 bridge. Run in a worktree
with `CARGO_TARGET_DIR` off the source tree; build QEMU out-of-tree in
`build/qemu`.

### E1 — Build gate (proves §2, and the MSVC/gcc link question)
1. `cargo xtask dist` on Windows; inspect `target/dist/lib/pkgconfig/reims-vgpu.pc`.
   If it is Unix-shaped, hand-write a Windows `.pc` for the experiment only.
2. Apply the 3 edits in §2 (local, uncommitted).
3. Configure:
   ```
   PKG_CONFIG_PATH=<dist>/lib/pkgconfig vendor/qemu/configure \
     --target-list=x86_64-softmmu --enable-whpx --enable-pvg --disable-docs
   ```
   Pass if configure's summary shows `reims-vgpu support: true` AND `WHPX
   support: true` AND `MAC_PVG` is set for `x86_64-softmmu`.
4. `ninja -C build/qemu qemu-system-x86_64.exe`.
   **Pass:** it links with `apple-gfx.c`/`apple-gfx-pci.c` in. **Primary risk
   surfaced here:** MSVC static lib vs ucrt64/gcc QEMU link. If it fails to link,
   the whole path's cost is dominated by toolchain unification, and that is the
   finding.
5. `qemu-system-x86_64 -device apple-gfx-pci,help` → device is registered.

### E2 — MMIO/BAR under WHPX without the GPU bridge (proves §4)
Use `-accel qtest` (no guest code; the test drives PCI config + registers
itself), exactly as `docs/qemu.md` validated BAR 2 under KVM:
- Realize `apple-gfx-pci` on `q35` under qtest on Windows; confirm BAR 0
  (16 KiB), ROM BAR, and (default) BAR 2 appear, and `quit` exits 0.
- Drive the 0x1210 write and confirm BAR 2 shows the RAM window; confirm a
  32-bit read/write of the IOSurface registers round-trips.
  **Pass:** the device realizes and its MMIO plumbing works under a
  Windows/WHPX-configured QEMU binary, independent of GPU rendering.

Only if E1 and E2 both pass is the §3 bridge worth building. A tiny optional E3
(host-pointer import of a `WHvMapGpuRange` region, mirroring the existing probe
but on hypervisor-registered memory) would then de-risk the bridge before any
C-API change — but the existing probe already covers the `VK_EXT_external_memory_host`
mechanism, so E3 is confirmation, not discovery.

---

## 6. Summary

| Layer | Status | Smallest change |
|---|---|---|
| QEMU build gate (A/B) | Blocked by 2 meson `if`s | 2 one-line edits (`meson.build:882`, `:3426`) |
| Device Kconfig (C) | **Not blocked** | none |
| Activity backend | **Already portable** | none (stub via `:68-73`) |
| `memory_files` realize | **Already `#else`-safe** | none (`apple-gfx.c:729-731`) |
| Windows `.pc` for MSVC link | Blocked | `cargo xtask dist` must emit a Windows `.pc` |
| Guest-RAM → GPU import | **Blocked (new work)** | C-API host-pointer form + Vulkan import; keystone proven by the probe |
| MMIO / MSI under WHPX | Compatible on paper | verify empirically (E2) |

**Build feasibility: high.** **Runtime feasibility: plausible and keystone-proven,
but gated on a new fd→host-pointer memory bridge and on confirming MSI/MMIO
delivery under WHPX.** No broad port should start until E1 + E2 pass.

**Known hard blockers / unknowns**
- MSVC-produced static library linked into a ucrt64/gcc QEMU (toolchain
  unification) — unproven; E1 settles it.
- `RvgMemoryRangeFile` carries only `int32_t fd`; WHPX guest RAM has no fd — the
  C API must grow a host-pointer form. (New, bounded.)
- WHPX MSI delivery for `apple-gfx-pci` (`msi_notify`) — needs empirical check.
- WHPX cannot host the ARM `vmapple` machine at all (`hw/vmapple/Kconfig`:
  `depends on ARM && HVF`); Windows is strictly the x86-64 `q35` +
  `apple-gfx-pci` path, which the project has only ever run under Linux/KVM.
- x86-64 macOS guest quirks already documented in `docs/qemu.md` (METAL_ERROR_MODE,
  xHCI input, USB audio, installer BAR 2) are guest-side and host-agnostic; they
  will apply equally on Windows once rendering works.
