# Guest setup and rebuilding what the cleanup removed

How the macOS guest in `qemu-vm\` was made, and how to make each piece again. [windows.md](windows.md)
says how to build and run the QEMU side; this page covers the guest's disk, the OpenCore image, the
firmware files, the guest-side setup, and a table of what the 2026-10-19 cleanup deleted.

Where this page says "unverified", the claim comes from a file name, a note or an inspection of a file
and was not checked by repeating the step. No serial numbers, UUIDs, MLB or ROM values are written here:
they are "generated per VM" (see [The OpenCore image](#2-the-opencore-image)).

A step-by-step reinstall guide (resources, fresh install, post-install, ESP rebuild) is `qemu-vm\SETUP.md`.

## 1. The guest disk

**What exists.** The launcher boots `qemu-vm\macos-vmware-overlay-20261016.qcow2` over
`macos.qcow2` (500 GiB virtual, the converted VMware install). After the 2026-10-19 shrink the pair is
replaced by one compressed image, `macos-merged.qcow2` (see [section 8](#8-shrinking-the-guest-disk)),
and the launcher's `-MacosOverlay` default points at it once a boot of it is verified.

**How the converted disk was made (unverified).** The install was a VMware Workstation VM
(`intel-vm\macos.vmx`: `guestOS = "darwin24-64"`, firmware `efi`, a SATA disk `macos.vmdk`, OpenCore on
`opencore.iso` from oc4vm, the macOS installer ISO on the second CD drive). No note records the
conversion. The disk the VM wrote (`macos.vmdk`) is the source; the usual command is
`qemu-img convert -p -f vmdk -O qcow2 macos.vmdk macos.qcow2`. Neither `macos.vmdk` nor the installer
ISO exist any more. A 2026-10-13 note (`RESUME-NOTES.txt`) found that this converted disk did not boot
under an early QEMU/OpenCore combination (Penryn CPU, OSX-KVM's OpenCore), and that a QEMU-native
install did; the later Haswell CPU with oc4vm's OpenCore (below) boots it.

**Making a fresh guest from nothing** (the path the fresh-install chain `macos-fresh.raw` ->
`macos-overlay.qcow2` -> `macos-shaderpass-20261015.qcow2` took; exact commands for it were not kept,
so the steps are the standard ones, unverified):

1. Download a macOS recovery image with OpenCore's `macrecovery` (a copy of the tool is in oc4vm's
   `tools\windows\macrecovery.exe`; its source is `Utilities\macrecovery` in the public
   `acidanthera/OpenCorePkg`). It writes `BaseSystem.dmg` and `BaseSystem.chunklist`; convert the dmg to
   a raw image with `qemu-img convert -O raw BaseSystem.dmg BaseSystem.img`. The 2026-10-13 install used
   a `BaseSystem.img` this way. A full installer ISO (`macOS_Tahoe_26.7.1.iso`, once in the top-level
   folder, how it was made unverified) also works as a second drive.
2. Create the target disk: `qemu-img create -f qcow2 macos.qcow2 128G` (the fresh install was 80 GiB
   raw).
3. Boot with the OpenCore image (section 2) on `sata.0`, the target on `sata.2` as the launcher does,
   and the recovery image as one more `ide-hd` drive (add it to a copy of the launcher's `$qemuArgs`).
   Choose "Install macOS", erase the target as APFS, install, let it reboot into OpenCore's picker each
   time, and finish Setup Assistant.
4. Drop the recovery drive. Do the guest setup of sections 6 and 7.

A disk converted from VMware needs no install: boot it with the launcher.

## 2. The OpenCore image

`esp-ocdebug-livev-20261016.img` is a raw FAT32 image of 384 MiB, label `EFI`, with no partition table
(7-Zip lists it as `Type = FAT`). The launcher attaches it as a SATA disk (`ide-hd`, `sata.0`). OVMF finds
`EFI\BOOT\BOOTx64.efi` on it, which starts OpenCore.

**Provenance.** It is oc4vm 3.0.1 (David Parsons, MIT; <https://github.com/DrDonk/oc4vm>; it ships
OpenCore, Lilu, VirtualSMC, RestrictEvents, DebugEnhancer and Intel82574L, built 2025-10 to 2026-08;
OpenCore's own version number is unverified, the files are dated 2025-11-04) assembled from two of its
disk images:

- the tree is that of `oc4vm-3.0.1\disks\intel-verbose\opencore.iso` (`OPENCORE\EFI\...`), whose
  `config.plist` is `#OC4VM TYPE = macOS Intel Verbose OC`;
- `EFI\OC\OpenCore.efi` is the DEBUG build from `disks\intel-debug\opencore.iso` (SHA-256 starts
  `11cf57b4d907`; the release one differs). The edited `config.plist` has the intel-verbose base plus the
  11 changes in the table below, and one `Kernel\Patch` entry of the debug set (`OC4VM-DBG-01`, disable
  ACPI logging).

"livev" is this project's name for "live verbose": the console shows the verbose boot on the screen
(the `serial=3` boot-arg froze the on-screen `-v` console at `kdp_core`; this image leaves it out).
`ocdebug` is the DEBUG OpenCore. Nobody outside this project builds an image of that name.

**Contents** (as listed with `7z l`):

| part | what |
|---|---|
| `EFI\BOOT\BOOTx64.efi` | OpenCore's bootstrap |
| `EFI\OC\OpenCore.efi` | DEBUG OpenCore (above) |
| `EFI\OC\ACPI\` | empty: no SSDTs, no ACPI patches |
| `EFI\OC\Drivers\` | the full oc4vm set (about 60 `.efi`); `config.plist` loads only five: `OpenRuntime.efi`, `OpenPartitionDxe.efi`, `HfsPlus.efi`, `OpenCanopy.efi`, `AudioDxe.efi` |
| `EFI\OC\Kexts\` | `Lilu`, `VirtualSMC`, `DebugEnhancer`, `RestrictEvents`, `Intel82574L` (all enabled), and `AMFIPass` (present, not in `Kernel\Add`) |
| `EFI\OC\Tools\` | 15 tools; only `OpenShell.efi` is in `Misc\Tools` |
| `EFI\OC\config.plist`, `config.plist.bak-working` | the live config and an earlier working copy |
| `NvVars` | OpenCore's emulated NVRAM file, rewritten by the guest and OpenCore at run time |
| `SysReport\` | OpenCore's `SysReport` dump (ACPI tables, SMBIOS, CPU and PCI info) from a boot |

**Essential `config.plist` values** (the image's, non-secret; "oc4vm" is the unedited intel-verbose file):

| key | value | oc4vm default |
|---|---|---|
| `NVRAM\Add\7C436110-...\boot-args` | `keepsyms=1 -v debug=0x108 no_compat_check` | `keepsyms=1 -lilubetaall -v -no_compat_check -no_panic_dialog -liludbgall cwad` |
| `NVRAM\Add\7C436110-...\csr-active-config` | `F7060000` (SIP mostly off) | `00000000` |
| `Misc\Security\SecureBootModel` | `Disabled` | same |
| `Misc\Security\Vault`, `DmgLoading`, `ScanPolicy` | `Optional`, `Any`, `0` | same |
| `Misc\Boot\PickerMode` | `Builtin` (picker shown, `Timeout` 5 s, `ShowPicker` true) | `External` |
| `Misc\Boot\PickerVariant` | `Acidanthera\GoldenGateOC4VM` | same |
| `Misc\Debug` | `Target` 3, `DisplayLevel` 0x80000042, `AppleDebug`, `ApplePanic`, `SysReport` true | `Target` 0, `DisplayLevel` 0 |
| `PlatformInfo\Automatic`, `UpdateSMBIOS/DataHub/NVRAM` | all `true`, mode `Create` | `Automatic` false |
| `PlatformInfo\Generic\SystemProductName` | `iMac19,1` | empty |
| `PlatformInfo\Generic\SystemSerialNumber`, `MLB`, `ROM`, `SystemUUID` | generated per VM (`macserial` from oc4vm's `tools\`, or OpenCore's) | placeholders |
| `Kernel\Quirks` | `DisableLinkeditJettison`, `PanicNoKextDump`, `ProvideCurrentCpuInfo`, `SetApfsTrimTimeout` -1 | same |
| `Kernel\Emulate` | `DummyPowerManagement` true; no CPUID spoof | same |
| `Booter\Quirks` | `AvoidRuntimeDefrag`, `EnableSafeModeSlide`, `EnableWriteUnprotector`, `FixupAppleEfiImages`, `ProtectUefiServices`, `ProvideCustomSlide`, `ResizeAppleGpuBars` -1 | same |
| `UEFI\Output` | `Resolution` `1440x900@32`, `TextRenderer` `BuiltinGraphics`, `ProvideConsoleGop` true | same |
| `UEFI\Quirks` | `EnableVectorAcceleration`, `RequestBootVarRouting`, `ResizeGpuBars` -1; `ProtocolOverrides\FirmwareVolume` true | same |
| `UEFI\APFS` | `EnableJumpstart`, `HideVerbose` | same |
| `NVRAM\WriteFlash` | `true` | same |

Changing the Mac identity (a new `SystemSerialNumber`, `MLB`, `ROM`, `SystemUUID`) makes an installed
macOS see a different machine (iMessage, iCloud and App Store sign-in tie to it). The live image is the
only copy of the values in use; keep it, and copy it before an edit. The launcher's `-uuid` is a separate
value.

**Regenerating the image.**

1. Get oc4vm 3.0.1 (release zip from the repository above; MIT) and take
   `disks\intel-verbose\opencore.iso` and `disks\intel-debug\opencore.iso`. 7-Zip opens both; the tree is
   under `OPENCORE\`.
2. Make a 384 MiB FAT32 file with no partition table and label `EFI`. In MSYS2 with `dosfstools` and
   `mtools` (unverified that this is how the original was made): `truncate -s 384M esp.img;
   mkfs.fat -F32 -n EFI esp.img; mcopy -i esp.img -s OPENCORE/EFI ::/EFI`. (`oc-edit\1.primary.img` was
   the partition 7-Zip extracts from the oc4vm ISO, which is how the tree got on disk.)
3. Copy the DEBUG `OpenCore.efi` over `EFI\OC\OpenCore.efi`, and apply the table's values to
   `EFI\OC\config.plist` with OpenCore Configurator, ProperTree or any plist editor; generate a new Mac
   identity with `macserial -m iMac19,1 -g` and put it in `PlatformInfo\Generic`.
4. `qemu-vm\esp\` was a directory working copy and `esp-backup-full\` a backup of it; QEMU can also
   serve a directory as a FAT disk (`-drive file=fat:rw:<dir>,format=raw`, how the 2026-10-13 install ran),
   but that is read through QEMU's vvfat, not a file the launcher takes.

**Editing the image on Windows.** 7-Zip lists and extracts from it (`7z l`, `7z e`, `7z x`) but cannot
write to it. To change a file, extract the tree, edit, and write it back with `mcopy -i <img> -o
<file> ::/EFI/OC/` (MSYS2 `mtools`, run with the VM stopped). Windows's own Mount-DiskImage does not
take a raw partition-less FAT image (unverified). Never edit it while QEMU runs: OpenCore writes
`NvVars` into it.

## 3. OVMF code and variables

| file | what |
|---|---|
| `OVMF_CODE.fd` (3,653,632 bytes) | byte-identical to `bios\edk2-x86_64-code.fd`, the EDK2/OVMF image QEMU ships (Gerd Hoffmann's build, `roms/edk2`); mapped read-only as pflash |
| `OVMF_VARS-livev-20261016.fd` (3,653,632 bytes) | all zero bytes: an empty variable store of the same size as the code image, written by the guest's firmware (boot entries, OpenCore's runtime variables); mapped read-write. "livev" is the name of the pairing with the ESP above (written 2026-10-13 17:31, renamed 2026-10-16) |

To recreate: copy `edk2-x86_64-code.fd` from a QEMU `share` folder (or decompress
`dev\reims-vgpu\vendor\qemu\pc-bios\edk2-x86_64-code.fd.bz2`, which gives a different, newer build; the
one in use came with the QEMU binary-install `share` dir, unverified which) to `OVMF_CODE.fd`; make the
variable store with `fsutil file createnew OVMF_VARS-livev-20261016.fd 3653632` (or
`truncate -s 3653632` in MSYS2). OVMF initialises it on the first boot. A lost vars file loses
only boot variables; OpenCore's NVRAM is in the ESP's `NvVars`. If a boot hangs with the new vars file,
the size must match the code file's.

## 4. The option ROM, `ReimsVgpu.rom`

The ROM is the PCI device's UEFI driver (5,632 bytes; QEMU's `romfile=` loads it, so OVMF gets a
framebuffer before OpenCore starts). It is built from the `firmware\` directory of any checkout
(`dev\reims-vgpu\firmware` or `dev\reims-perf\firmware`; the sources are `src\*.c`, `src\OptionRom.S`):

```sh
cd dev/reims-perf
export MSYS2_ARG_CONV_EXCL='*'         # keeps lld-link's /flags from being turned into paths
make -C firmware                       # clang + lld (+ ld.lld); writes firmware/build/ReimsVgpu.rom
cp firmware/build/ReimsVgpu.rom ../../qemu-vm/ReimsVgpu.rom
```

It needs `clang` and `lld` on `PATH` (MSYS2: `pacman -S mingw-w64-ucrt-x86_64-clang
mingw-w64-ucrt-x86_64-lld`). `cargo xtask dist` also builds it (`make rom` with `BUILD` set to the
target's `firmware` directory). The Makefile warns that the bytes are the compiler's: a rebuild may not
match the file in use bit for bit (unverified; the ROM in use was built 2026-10-13).

## 5. `bios\`

`-L bios` is QEMU's data directory: a copy of the `share` folder of a QEMU for Windows install (310 MB;
files dated 2026-10-13: `bios-256k.bin`, `kvmvapic.bin`, `efi-*.rom`, `edk2-*-code.fd`, `keymaps\`,
`vgabios*`, ...). Only a few are read for `q35` (`bios-256k.bin`, `kvmvapic.bin`, `linuxboot_dma.bin`,
`efi-virtio.rom`, `efi-vmxnet3.rom`, `vgabios*`, `keymaps\`), so the whole directory can be a smaller set.
To recreate it, copy the `share` directory of any QEMU 10 install (the stock Windows installer puts it
beside `qemu-system-x86_64.exe`) or the `pc-bios` files of `dev\reims-vgpu\vendor\qemu` (the edk2
images there are `.bz2`: `bzip2 -d`). `C:\dev\qemu\` holds only an exe, no `share`. The custom QEMU build
in `qemu\` does not carry these files itself.

## 6. SSH key and the guest agent

**Key** (`qemu-vm\ssh\`; the private key is never written into docs):

```powershell
ssh-keygen -t ed25519 -N '""' -f qemu-vm\ssh\vm_ed25519
mkdir qemu-vm\keyserve; copy qemu-vm\ssh\vm_ed25519.pub qemu-vm\keyserve\k.pub
python -m http.server 8765 --bind 127.0.0.1 --directory qemu-vm\keyserve    # leave running
```

In the guest, with the VM up: turn on Remote Login (section 7), then in Terminal
`mkdir -p ~/.ssh && curl http://10.0.2.2:8765/k.pub >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh &&
chmod 600 ~/.ssh/authorized_keys` (10.0.2.2 is the host as QEMU's user network sees it). Then
`ssh -i qemu-vm\ssh\vm_ed25519 -p 2222 -o UserKnownHostsFile=qemu-vm\ssh\known_hosts takiyo@127.0.0.1`
(the guest user was `takiyo`). Stop the server afterwards; delete `keyserve\` if wanted.

**Pointer agent** (`qemu-vm\guest-agent\ReimsPointer.swift`, `install.sh`), only for 2+ displays:

```powershell
scp -P 2222 -i qemu-vm\ssh\vm_ed25519 -r qemu-vm\guest-agent takiyo@127.0.0.1:reims-guest-agent
ssh -p 2222 -i qemu-vm\ssh\vm_ed25519 takiyo@127.0.0.1 "sh ~/reims-guest-agent/install.sh"
```

It builds the agent in the guest with `swiftc` (needs Xcode Command Line Tools: `xcode-select --install`),
installs LaunchAgent `com.reims.pointer`, and connects to 10.0.2.2:4455. Uninstall: `sh install.sh -u`.
Allow it in System Settings > Privacy & Security > Accessibility; rebuilding invalidates the permission.
Log: `~/Library/Logs/ReimsPointer.log`. See windows.md, Multiple displays.

## 7. Remote Login and `FORCE_SOFTWARE_RENDERER`

- **Remote Login:** System Settings > General > Sharing > Remote Login on (or in Terminal,
  `sudo systemsetup -setremotelogin on`, which needs Full Disk Access for Terminal).
- **`FORCE_SOFTWARE_RENDERER=1`**, once, for blank icons and the IconRendering crash loop:
  `sudo launchctl config user setenv FORCE_SOFTWARE_RENDERER 1`, then reboot. The reason, the interim
  login agent and the cache reset are in windows.md, Guest setup.
- **Accessibility > Display > Reduce transparency off**, or Liquid Glass shows white boxes.
- A converted VMware disk keeps its guest tools (`vmtoolsd` etc.), which do nothing here.

## 8. Shrinking the guest disk

```powershell
qemu\qemu-img.exe convert -p -f qcow2 -O qcow2 -o compression_type=zstd -c -m 8 <src> <dst>
```

`<src>` is the overlay: `qemu-img` reads the whole backing chain and writes one flat image, dropping
unallocated and zero clusters. `-c` compresses every cluster with zstd, `-m 8` runs 8 coroutines.
Trade-offs:

- compressed clusters are decompressed on every read (CPU cost, a little slower reads, nothing at all
  for clusters the guest never reads), and `compression_type=zstd` needs QEMU 5.1 or later and a
  qcow2 v3 image, so only the custom QEMU or newer `qemu-img` reads it;
- a cluster the guest rewrites is stored uncompressed and the image grows again, so the shrink is a
  one-time saving that erodes with use; a later `convert` (with the VM off) shrinks it again;
- the compression cannot make a disk smaller than its data: free space inside APFS that was written
  once but not trimmed still counts. (`oc4vm-3.0.1\tools\guest\diskzero.zsh` and `diskshrink.zsh` zero
  free space in the guest first, for VMware; unverified here.)
- keep the originals (`macos.qcow2` and the overlay) until a boot from the new file is verified
  (login screen, desktop, a reboot). The overlay's backing-file name is relative, so the two stay in one
  directory. Never boot the merged image and the old overlay at the same time.

After verifying, set `-MacosOverlay` to the new file in the launcher (the default is a parameter
of `boot-perf.ps1`) and move `*.displays.json` to match: it sits next to the disk and is named
`<disk>.displays.json`.

## 9. Removed on 2026-10-19 and how to get it back

| removed | what it was | how to get it back |
|---|---|---|
| macOS installer ISO `macOS_Tahoe_26.7.1.iso` | installer image the VMware VM mounted | download a recovery image with `macrecovery` (section 1) or Apple's installer; how this ISO was built is unverified |
| the fresh-install chain `macos-fresh.raw`, `macos-overlay.qcow2`, `macos-shaderpass-20261015.qcow2` | an 80 GiB QEMU-native install that stopped at Setup Assistant | reinstall (section 1); not needed for the converted disk |
| `oc4vm-3.0.1\` | oc4vm 3.0.1, OpenCore-for-VMware tooling (MIT): configs for Intel and AMD, `opencore.iso` disks, `tools\` (macrecovery, macserial, vmxtool, cpuid), `packages\VoodooHDA-Tahoe.pkg` | download the 3.0.1 release from <https://github.com/DrDonk/oc4vm>; only the ESP image is needed at run time, and it contains what it took from here |
| `unlocker428\` | VMware Workstation macOS unlocker 4.2.8 (DrDonk, MIT) and `backup\25.0.1.25219725`, the VMware files it patched | irrelevant to QEMU. If VMware Workstation is still patched, restore it with the unlocker's uninstall (`windows\` scripts) before relying on VMware; unverified whether it is |
| `intel-vm\` | the VMware VM: `macos.vmx`, `macos.nvram`, `opencore.iso`(+`.ori`, `.pre-hcalls0`), scoreboards, VMware logs; the disk `macos.vmdk` was already gone | nothing needed; recreate with oc4vm's `vmware\intel` template if the VMware route is wanted |
| `runtime\` | `OSX-KVM-runtime-*` (git object stores only, no work tree; OSX-KVM is public, `kholia/OSX-KVM`) and `backup.ps1` (robocopy of the folder to `macos.bak`, excluding `target\`, `intel-vm\`, `macos-fresh.raw`, logs) | clone OSX-KVM; `backup.ps1` is a 40-line robocopy wrapper |
| `qemu-vm\esp\`, `esp-backup-full\`, `oc-edit\`, `keyserve\` | working copies of the OpenCore tree (older config: `boot-args` with `serial=3`), a backup, 7-Zip's extraction of an oc4vm ISO, and the SSH public key served to the guest | section 2 and section 6; the live image is `esp-ocdebug-livev-20261016.img` |
| `qemu-vm\esp-logo-20261016.img` | an alternate ESP (optional) | unverified what differs; not needed |
| `reims-dist\` | the staged library QEMU links | `cargo build ...` then copy the `.a` and header and write `reims-vgpu.pc` (windows.md, Building, steps 1 and 2). The `.pc` is `Libs: -L${libdir} -lreims_vgpu` and the `Libs.private` line of that section |
| every `target\` directory, the QEMU build tree `qemu-build-patched` and its executables | build output | windows.md, Building (library, then QEMU configure + `ninja`, then the `qemu\` copy and `.build.json` manifest) |
| logs (`boot-*`, `guest-serial-*`, `qemu-log-*`, `ovmf-*`, `display-persistence-*`, `samples-*.bin`, `captures\`, handoff and plan notes) | per-run output | the launcher writes new ones; see windows.md, Troubleshooting |
| `qvm\` build output | `ui\windows\bin`, `obj`, `tests\bin`, `tests\obj` | `qvm\REBUILD.md`, `qvm\PRUNED.txt` |

Not regenerable from public sources: the guest disk, the ESP image's Mac identity, the SSH key pair
(make a new pair), and `OVMF_VARS` contents (disposable).
