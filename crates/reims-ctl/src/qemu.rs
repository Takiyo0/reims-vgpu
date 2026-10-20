//! The QEMU command line a VM runs: one [`Invocation`], which `start` runs and `command`
//! prints as a shell script, so a printed script runs the VM exactly as reims-ctl does.
//!
//! Every profile gets:
//! - on reims-vgpu, the library's window for each display: `-display none` with
//!   `REIMS_VGPU_WINDOW` unset (`0` would keep apple-gfx headless), never a window of QEMU's own;
//! - a QMP socket (`-qmp`), through which reims-ctl stops the VM;
//! - QEMU's pid file;
//! - user networking with virtio-net, and an optional SSH port forward; with networking off,
//!   `-nic none`, without which QEMU gives the machine its default NIC on user networking (on
//!   q35 an e1000e in the first free slot, 1, where the root port's address is);
//! - a USB Audio Class device on the host's audio backend: x86-64 macOS 26 has no Intel HDA
//!   driver, and Safari's Media Source players fail without an output device;
//! - on `q35-x86_64`, also an ICH9 Intel HDA controller with a line-out codec on that backend,
//!   for OpenCore's boot chime: its AudioDxe drives HDA, not USB audio;
//! - with logs on: the library's log (`REIMS_VGPU_LOG`), the serial console, timestamps.
//!
//! `q35-x86_64` follows the command line the x86-64 guests were measured on (docs/qemu.md):
//! RAM as a shared memfd, which the library on Linux requires; the keyboard and tablet on a
//! qemu-xhci controller, since on ICH9's EHCI the guest stops re-arming a device's interrupt
//! transfers under GPU load and its input freezes; fixed PCI addresses (the AHCI controller at
//! slot 2, where OpenCore's boot entries name it); `isa-applesmc` with Apple's SMC key, [`OSK`];
//! VMX in the guest's CPU where the host's KVM offers it, for the guest's own VMs;
//! `dump-guest-core=off`, which leaves guest RAM out of a core dump of QEMU: with it in, a
//! crash's core takes the process tens of seconds to end (docs/qemu.md). QEMU refuses the
//! option where the OS cannot leave memory out of a core (macOS has no `MADV_DONTDUMP`); and no
//! ACPI S3 (`ICH9-LPC.disable_s3`): a guest in S3 never wakes again with its display. Only PS/2
//! input wakes QEMU from S3, not the USB keyboard and tablet, and QEMU's wake resets every
//! device, where the guest's GPU driver restores none of the device's state, so the display
//! stays dark and WindowServer waits on the GPU for good. Without S3, macOS's sleep stops at a
//! dark wake, from which input wakes the display (docs/qemu.md).
//!
//! An x86-64 VM may have QEMU's `vmware-svga` in apple-gfx's slot instead (`graphics =
//! "vmware-svga"`), with the device's id, `gfx`, which the terminal viewer's screendumps name.
//! The library then draws nothing, so the window's and the GPU's variables are left out, and
//! QEMU's own window, GTK's (`-display gtk`), shows the display. Where QEMU's environment names
//! no Wayland compositor or X server, GTK's window would end QEMU as it starts, so it is
//! `-display none` there, and the guest runs headless, as on the library's window.
//!
//! `vmapple-arm64` follows scripts/run_vmapple.sh.

use crate::host::{Host, Os, Profile};
use crate::vm::{Graphics, Vm};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Virtualization.framework's boot ROM for vmapple machines.
pub const AVPBOOTER: &str =
    "/System/Library/Frameworks/Virtualization.framework/Resources/AVPBooter.vmapple2.bin";

/// The SMC key `isa-applesmc` gives every x86-64 guest.
pub const OSK: &str = "ourhardworkbythesewordsguardedpleasedontsteal(c)AppleComputerInc";

#[derive(Clone, Debug, PartialEq)]
pub struct Invocation {
    pub program: PathBuf,
    pub args: Vec<String>,
    /// Variables set for QEMU.
    pub env: Vec<(String, String)>,
    /// Variables removed from QEMU's environment.
    pub unset: Vec<String>,
}

pub struct Ctx {
    pub host: Host,
    pub qemu: PathBuf,
    /// The audio backend for a VM that names none.
    pub audio_backend: String,
    /// Whether the host's KVM offers VMX to guests ([`crate::host::kvm_nested_vmx`]).
    pub nested_vmx: bool,
    /// Whether QEMU's own window can open ([`crate::host::has_screen`]).
    pub screen: bool,
}

/// A value inside a QEMU option list: commas doubled.
pub fn opt(s: &str) -> String {
    s.replace(',', ",,")
}

fn optp(p: &Path) -> String {
    opt(&p.to_string_lossy())
}

pub fn invocation(vm: &Vm, ctx: &Ctx) -> Result<Invocation> {
    let c = &vm.config;
    let mut a: Vec<String> = Vec::new();
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    push(&["-name", &c.name]);
    match c.profile {
        Profile::Q35X86_64 => q35(vm, ctx, &mut a)?,
        Profile::VmappleArm64 => vmapple(vm, ctx, &mut a)?,
    }
    common(vm, ctx, &mut a)?;
    a.extend(c.extra_args.iter().cloned());

    let mut env = Vec::new();
    let mut unset = vec![
        "REIMS_VGPU_WINDOW".to_string(),
        "REIMS_VGPU_GUEST_DISPLAY_PORTS".to_string(),
    ];
    if c.logs.enabled {
        env.push(("REIMS_VGPU_LOG".into(), c.logs.level.clone()));
    } else {
        unset.push("REIMS_VGPU_LOG".into());
    }
    let library = c.graphics == Graphics::ReimsVgpu;
    if library && (c.display.count > 1 || c.display.hotplug_displays) {
        env.push((
            "REIMS_VGPU_GUEST_DISPLAY_PORTS".into(),
            c.display.ports().to_string(),
        ));
    }
    if library && c.display.borderless {
        env.push(("REIMS_VGPU_WINDOW_BORDERLESS".into(), "1".into()));
    }
    if library && let Some(r) = c.display.refresh {
        env.push(("REIMS_VGPU_WINDOW_REFRESH".into(), r.to_string()));
    }
    if library
        && c.profile.gpu_choice(ctx.host)
        && let Some(f) = &c.gpu.driver_file
    {
        // The loader reads VK_DRIVER_FILES; loaders before 1.3.207 read VK_ICD_FILENAMES.
        let f = f.to_string_lossy().into_owned();
        env.push(("VK_DRIVER_FILES".into(), f.clone()));
        env.push(("VK_ICD_FILENAMES".into(), f));
        if let Some(id) = c.gpu.device_id {
            env.push(("VK_LOADER_DEVICE_ID_FILTER".into(), format!("{id:#06x}")));
        }
    }
    for (k, v) in &c.extra_env {
        env.retain(|(ek, _)| ek != k);
        unset.retain(|u| u != k);
        env.push((k.clone(), v.clone()));
    }
    Ok(Invocation {
        program: ctx.qemu.clone(),
        args: a,
        env,
        unset,
    })
}

fn q35(vm: &Vm, ctx: &Ctx, a: &mut Vec<String>) -> Result<()> {
    let c = &vm.config;
    let q = c.q35.as_ref().context("no [q35] settings")?;
    let mem = format!("{}M", c.memory_mib);
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    if c.profile.memfd_ram(ctx.host) {
        push(&["-machine", "q35,memory-backend=ram,dump-guest-core=off"]);
        push(&[
            "-object",
            &format!("memory-backend-memfd,id=ram,size={mem},share=on"),
        ]);
    } else {
        push(&["-machine", "q35"]);
    }
    push(&["-accel", c.profile.accel(ctx.host), "-m", &mem]);
    let mut cpu = format!(
        "{},kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on",
        q.cpu_model
    );
    // VMX lets the guest run VMs of its own (Hypervisor.framework, Docker). It is given only where
    // KVM offers it, so `check` never drops it with a warning. It is `vmx=on`, not `+vmx`, so a
    // VM's `-vmx` or `vmx=off` in `cpu_flags` takes it away: QEMU applies `name=value` flags in
    // order, and `+` and `-` ones after them, so a `+vmx` here would outlive a `vmx=off`.
    if ctx.nested_vmx {
        cpu.push_str(",vmx=on");
    }
    if !q.cpu_flags.trim().is_empty() {
        cpu.push(',');
        cpu.push_str(q.cpu_flags.trim());
    }
    let mut cpu = cpu;
    if ctx.host.os != Os::Linux {
        cpu = cpu.replace(",kvm=on", "");
    }
    push(&["-cpu", &cpu]);
    push(&[
        "-smp",
        &format!("{0},sockets=1,cores={0},threads=1", c.cpus),
    ]);
    push(&[
        "-global",
        "ICH9-LPC.acpi-pci-hotplug-with-bridge-support=off",
    ]);
    push(&["-global", "ICH9-LPC.disable_s3=1"]);
    push(&[
        "-device",
        "pcie-root-port,bus=pcie.0,addr=0x1,slot=1,x-speed=16,x-width=32",
    ]);
    push(&["-device", &format!("isa-applesmc,osk={OSK}")]);
    push(&[
        "-drive",
        &format!(
            "if=pflash,format=raw,readonly=on,file={}",
            optp(&vm.path(&q.firmware_code))
        ),
    ]);
    push(&[
        "-drive",
        &format!(
            "if=pflash,format=raw,file={}",
            optp(&vm.path(&q.firmware_vars))
        ),
    ]);
    push(&["-smbios", "type=2"]);
    push(&["-device", "ich9-ahci,id=sata,addr=0x2"]);
    push(&[
        "-drive",
        &format!(
            "id=opencore,if=none,format={},file={}",
            q.opencore_format.as_str(),
            optp(&vm.path(&q.opencore))
        ),
    ]);
    push(&["-device", "ide-hd,bus=sata.2,drive=opencore,bootindex=1"]);
    push(&[
        "-drive",
        &format!(
            "id=disk,if=none,format={},file={}",
            c.disk_format.as_str(),
            optp(&vm.disk())
        ),
    ]);
    push(&[
        "-device",
        &format!("ide-hd,bus=sata.{},drive=disk,rotation_rate=1", q.disk_port),
    ]);
    // The installer, on the next port after the disk's, past OpenCore's.
    if let Some(p) = q.installer.as_deref().map(|p| vm.path(p)) {
        let port = if q.disk_port == 4 { 5 } else { 4 };
        push(&[
            "-drive",
            &format!(
                "id=installer,if=none,format=raw,snapshot=on,file={}",
                optp(&p)
            ),
        ]);
        push(&[
            "-device",
            &format!("ide-hd,bus=sata.{port},drive=installer"),
        ]);
    }
    if c.network.enabled {
        push(&["-netdev", &netdev(vm)]);
        push(&[
            "-device",
            &format!(
                "virtio-net-pci,netdev=net0,id=net0,mac={},addr=0x3",
                c.network.mac
            ),
        ]);
    }
    push(&["-vga", "none"]);
    if let Some(port) = c.display.pointer_agent_port {
        push(&[
            "-chardev",
            &format!("socket,id=reimsptr,host=127.0.0.1,port={port},server=on,wait=off"),
        ]);
    }
    let gfx = match c.graphics {
        Graphics::ReimsVgpu => {
            let ports = if c.display.hotplug_displays {
                format!(",display-ports={}", c.display.ports())
            } else {
                String::new()
            };
            let pointer = if c.display.pointer_agent_port.is_some() {
                ",pointer-chardev=reimsptr"
            } else {
                ""
            };
            let rom = crate::install::qemu_rom(&ctx.qemu)
                .map(|p| format!(",romfile={}", optp(&p)))
                .unwrap_or_default();
            format!(
                "apple-gfx-pci,id=gfx,bus=pcie.0,addr=0x4,displays={}{ports},resolution={}{pointer}{rom}",
                c.display.count,
                c.display.qemu_resolution()?
            )
        }
        Graphics::VmwareSvga => "vmware-svga,id=gfx,bus=pcie.0,addr=0x4".into(),
    };
    push(&["-device", &gfx]);
    push(&[
        "-device",
        "qemu-xhci,id=xhci,addr=0x5,conditional-intr-mapping=on",
    ]);
    push(&[
        "-device",
        "usb-kbd,id=kbd,bus=xhci.0",
        "-device",
        "usb-tablet,id=tablet,bus=xhci.0",
    ]);
    if c.audio.enabled {
        push(&["-audiodev", &audiodev(vm, ctx)]);
        push(&["-device", "usb-audio,id=audio,audiodev=snd0,bus=xhci.0"]);
        // At ICH9's own slot for it.
        push(&["-device", "ich9-intel-hda,id=hda,addr=0x1b"]);
        push(&["-device", "hda-output,bus=hda.0,audiodev=snd0"]);
    }
    Ok(())
}

fn vmapple(vm: &Vm, ctx: &Ctx, a: &mut Vec<String>) -> Result<()> {
    let c = &vm.config;
    let v = c.vmapple.as_ref().context("no [vmapple] settings")?;
    let aux = vm.aux_trimmed().context("no auxiliary storage")?;
    let disk = vm.disk();
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    push(&["-accel", c.profile.accel(ctx.host)]);
    push(&["-M", &format!("vmapple,uuid={}", v.ecid)]);
    push(&[
        "-m",
        &format!("{}M", c.memory_mib),
        "-smp",
        &c.cpus.to_string(),
    ]);
    push(&["-bios", AVPBOOTER]);
    push(&[
        "-drive",
        &format!("file={},if=pflash,format=raw", optp(&aux)),
    ]);
    push(&[
        "-drive",
        &format!("file={},if=pflash,format=raw", optp(&disk)),
    ]);
    push(&[
        "-drive",
        &format!("file={},if=none,id=aux,format=raw", optp(&aux)),
    ]);
    push(&[
        "-drive",
        &format!("file={},if=none,id=root,format=raw", optp(&disk)),
    ]);
    push(&["-device", "vmapple-virtio-blk-pci,variant=aux,drive=aux"]);
    push(&["-device", "vmapple-virtio-blk-pci,variant=root,drive=root"]);
    if c.network.enabled {
        push(&["-netdev", &netdev(vm)]);
        push(&[
            "-device",
            &format!("virtio-net-pci,netdev=net0,mac={}", c.network.mac),
        ]);
    }
    push(&[
        "-global",
        &format!("apple-gfx-mmio.displays={}", c.display.count),
    ]);
    push(&[
        "-global",
        &format!("apple-gfx-mmio.resolution={}", c.display.qemu_resolution()?),
    ]);
    if c.audio.enabled {
        // On the USB bus of the machine's own xHCI controller, its only one.
        push(&["-audiodev", &audiodev(vm, ctx)]);
        push(&["-device", "usb-audio,id=audio,audiodev=snd0"]);
    }
    Ok(())
}

fn netdev(vm: &Vm) -> String {
    let mut n = "user,id=net0,ipv6=off".to_string();
    if let Some(p) = vm.config.network.ssh_port {
        n.push_str(&format!(",hostfwd=tcp:127.0.0.1:{p}-:22"));
    }
    n
}

fn audiodev(vm: &Vm, ctx: &Ctx) -> String {
    let backend = vm
        .config
        .audio
        .backend
        .as_deref()
        .unwrap_or(&ctx.audio_backend);
    format!("{},id=snd0", opt(backend))
}

fn common(vm: &Vm, ctx: &Ctx, a: &mut Vec<String>) -> Result<()> {
    let c = &vm.config;
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    if let Some(data) = crate::install::qemu_data_dir(&ctx.qemu) {
        push(&["-L", &optp(&data)]);
    }
    if !c.network.enabled {
        push(&["-nic", "none"]);
    }
    let qemu_window = c.graphics == Graphics::VmwareSvga && ctx.screen;
    let display = if qemu_window {
        if ctx.host.os == Os::Windows {
            "sdl"
        } else {
            "gtk"
        }
    } else {
        "none"
    };
    push(&["-display", display]);
    if c.logs.enabled {
        push(&["-serial", &format!("file:{}", vm.serial_log().display())]);
        push(&["-msg", "timestamp=on"]);
    } else {
        push(&["-serial", "none"]);
    }
    push(&["-monitor", "none"]);
    #[cfg(unix)]
    let qmp = format!("unix:{},server=on,wait=off", optp(&vm.qmp_socket()));
    #[cfg(windows)]
    let qmp = format!("{},server=on,wait=off", vm.qmp_socket().display());
    push(&["-qmp", &qmp]);
    push(&["-pidfile", &vm.pidfile().to_string_lossy()]);
    Ok(())
}

impl Invocation {
    /// A `std::process::Command` running it.
    pub fn command(&self) -> std::process::Command {
        let mut cmd = std::process::Command::new(&self.program);
        cmd.args(&self.args);
        for k in &self.unset {
            cmd.env_remove(k);
        }
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
        cmd
    }

    /// A POSIX shell script that runs it, an option and its value to a line.
    pub fn shell_script(&self, comment: &str) -> String {
        let mut s = String::from("#!/bin/sh\n");
        for line in comment.lines() {
            s.push_str("# ");
            s.push_str(line);
            s.push('\n');
        }
        for k in &self.unset {
            s.push_str(&format!("unset {k}\n"));
        }
        for (k, v) in &self.env {
            s.push_str(&format!("export {k}={}\n", quote(v)));
        }
        s.push_str("exec ");
        s.push_str(&quote(&self.program.to_string_lossy()));
        let mut i = 0;
        while i < self.args.len() {
            s.push_str(" \\\n  ");
            s.push_str(&quote(&self.args[i]));
            if self.args[i].starts_with('-')
                && i + 1 < self.args.len()
                && !self.args[i + 1].starts_with('-')
            {
                s.push(' ');
                s.push_str(&quote(&self.args[i + 1]));
                i += 1;
            }
            i += 1;
        }
        s.push('\n');
        s
    }

    /// A runnable script in the current host's shell.
    pub fn platform_script(&self, comment: &str) -> String {
        #[cfg(not(windows))]
        {
            self.shell_script(comment)
        }
        #[cfg(windows)]
        {
            let quote = |s: &str| format!("'{}'", s.replace('\'', "''"));
            let mut out = String::new();
            for line in comment.lines() {
                out.push_str("# ");
                out.push_str(line);
                out.push('\n');
            }
            for key in &self.unset {
                out.push_str(&format!(
                    "[Environment]::SetEnvironmentVariable({}, $null, 'Process')\n",
                    quote(key)
                ));
            }
            for (key, value) in &self.env {
                out.push_str(&format!(
                    "[Environment]::SetEnvironmentVariable({}, {}, 'Process')\n",
                    quote(key),
                    quote(value)
                ));
            }
            out.push_str("& ");
            out.push_str(&quote(&self.program.to_string_lossy()));
            for arg in &self.args {
                out.push(' ');
                out.push_str(&quote(arg));
            }
            out.push_str("\nexit $LASTEXITCODE\n");
            out
        }
    }
}

pub fn quote(s: &str) -> String {
    shlex::try_quote(s)
        .map(|c| c.into_owned())
        .unwrap_or_else(|_| format!("'{}'", s.replace('\'', "'\\''")))
}

/// The host OS's name, as a short word, for messages.
pub fn os_word(host: Host) -> &'static str {
    match host.os {
        Os::Mac => "macOS",
        Os::Linux => "Linux",
        Os::Windows => "Windows",
        Os::Other => "this OS",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Arch;
    use crate::vm::*;
    use std::collections::BTreeMap;

    fn linux() -> Host {
        Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        }
    }

    fn mac() -> Host {
        Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        }
    }

    fn q35_vm() -> Vm {
        Vm {
            dir: "/vms/t".into(),
            config: VmConfig {
                name: "t".into(),
                profile: Profile::Q35X86_64,
                cpus: 6,
                memory_mib: 12288,
                disk: "disk.qcow2".into(),
                disk_format: ImageFormat::Qcow2,
                stop_timeout_secs: 90,
                extra_args: vec![],
                graphics: Graphics::ReimsVgpu,
                display: Display::default(),
                gpu: Gpu::default(),
                audio: Audio::default(),
                network: Network {
                    enabled: true,
                    mac: "52:54:00:aa:bb:cc".into(),
                    ssh_port: Some(2326),
                },
                logs: Logs::default(),
                q35: Some(Q35 {
                    opencore: "opencore.qcow2".into(),
                    opencore_format: ImageFormat::Qcow2,
                    firmware_code: "OVMF_CODE.fd".into(),
                    firmware_vars: "OVMF_VARS.fd".into(),
                    installer: None,
                    cpu_model: default_cpu_model(),
                    cpu_flags: default_cpu_flags(),
                    disk_port: 3,
                }),
                vmapple: None,
                extra_env: BTreeMap::new(),
            },
        }
    }

    fn ctx(host: Host) -> Ctx {
        Ctx {
            host,
            qemu: "/q/bin/qemu".into(),
            audio_backend: "pipewire".into(),
            nested_vmx: false,
            screen: true,
        }
    }

    fn has(args: &[String], pair: &[&str]) -> bool {
        args.windows(pair.len()).any(|w| {
            w.iter()
                .zip(pair)
                .all(|(a, b)| a.replace('\\', "/") == b.replace('\\', "/"))
        })
    }

    #[test]
    fn q35_on_linux_has_the_measured_devices() {
        let inv = invocation(&q35_vm(), &ctx(linux())).unwrap();
        let a = &inv.args;
        for pair in [
            &["-machine", "q35,memory-backend=ram,dump-guest-core=off"][..],
            &[
                "-object",
                "memory-backend-memfd,id=ram,size=12288M,share=on",
            ],
            &["-accel", "kvm", "-m", "12288M"],
            &["-smp", "6,sockets=1,cores=6,threads=1"],
            &["-device", &format!("isa-applesmc,osk={OSK}")],
            &[
                "-drive",
                "if=pflash,format=raw,readonly=on,file=/vms/t/OVMF_CODE.fd",
            ],
            &["-device", "ich9-ahci,id=sata,addr=0x2"],
            &["-device", "ide-hd,bus=sata.2,drive=opencore,bootindex=1"],
            &["-device", "ide-hd,bus=sata.3,drive=disk,rotation_rate=1"],
            &[
                "-netdev",
                "user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:2326-:22",
            ],
            &[
                "-device",
                "apple-gfx-pci,id=gfx,bus=pcie.0,addr=0x4,displays=1,resolution=1920x1080",
            ],
            &[
                "-device",
                "qemu-xhci,id=xhci,addr=0x5,conditional-intr-mapping=on",
            ],
            &["-audiodev", "pipewire,id=snd0"],
            &["-device", "usb-audio,id=audio,audiodev=snd0,bus=xhci.0"],
            &["-device", "ich9-intel-hda,id=hda,addr=0x1b"],
            &["-device", "hda-output,bus=hda.0,audiodev=snd0"],
            &["-display", "none"],
            &["-pidfile", "/vms/t/qemu.pid"],
        ] {
            assert!(has(a, pair), "missing {pair:?} in {a:?}");
        }
        let qmp = a.iter().position(|arg| arg == "-qmp").unwrap();
        if cfg!(windows) {
            assert!(a[qmp + 1].starts_with("tcp:127.0.0.1:"));
            assert!(a[qmp + 1].ends_with(",server=on,wait=off"));
        } else {
            assert!(has(a, &["-qmp", "unix:/vms/t/qmp.sock,server=on,wait=off"]));
        }
        assert!(!a.iter().any(|x| x == "-usb"), "no EHCI");
        assert!(!a.iter().any(|x| x.contains("installer")), "no installer");
        assert!(inv.unset.contains(&"REIMS_VGPU_WINDOW".to_string()));
        assert!(
            inv.unset
                .contains(&"REIMS_VGPU_GUEST_DISPLAY_PORTS".to_string())
        );
        assert!(inv.unset.contains(&"REIMS_VGPU_LOG".to_string()));
    }

    #[test]
    fn multiple_displays_are_advertised_to_the_guest() {
        let mut vm = q35_vm();
        vm.config.display.count = 2;
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        assert!(has(
            &inv.args,
            &[
                "-device",
                "apple-gfx-pci,id=gfx,bus=pcie.0,addr=0x4,displays=2,resolution=1920x1080"
            ]
        ));
        assert!(
            inv.env
                .contains(&("REIMS_VGPU_GUEST_DISPLAY_PORTS".into(), "2".into()))
        );
    }

    #[test]
    fn hotplug_reserves_and_advertises_eight_display_ports() {
        let mut vm = q35_vm();
        vm.config.display.hotplug_displays = true;
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        assert!(has(
            &inv.args,
            &[
                "-device",
                "apple-gfx-pci,id=gfx,bus=pcie.0,addr=0x4,displays=1,display-ports=8,resolution=1920x1080"
            ]
        ));
        assert!(
            inv.env
                .contains(&("REIMS_VGPU_GUEST_DISPLAY_PORTS".into(), "8".into()))
        );
    }

    #[test]
    fn pointer_agent_is_wired_to_qemu_and_the_reims_device() {
        let mut vm = q35_vm();
        vm.config.display.count = 2;
        vm.config.display.pointer_agent_port = Some(4455);
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        assert!(has(
            &inv.args,
            &[
                "-chardev",
                "socket,id=reimsptr,host=127.0.0.1,port=4455,server=on,wait=off"
            ]
        ));
        assert!(has(
            &inv.args,
            &[
                "-device",
                "apple-gfx-pci,id=gfx,bus=pcie.0,addr=0x4,displays=2,resolution=1920x1080,pointer-chardev=reimsptr"
            ]
        ));
    }

    /// A guest that enters S3 keeps a dark display (the module doc), so q35 offers it none.
    #[test]
    fn q35_offers_the_guest_no_s3_sleep() {
        let inv = invocation(&q35_vm(), &ctx(linux())).unwrap();
        assert!(has(&inv.args, &["-global", "ICH9-LPC.disable_s3=1"]));
    }

    #[test]
    fn network_off_gives_the_machine_no_nic() {
        let mut q35 = q35_vm();
        q35.config.network.enabled = false;
        let mut vmapple = q35.clone();
        vmapple.config.profile = Profile::VmappleArm64;
        vmapple.config.q35 = None;
        vmapple.config.vmapple = Some(Vmapple {
            aux: "aux.img".into(),
            ecid: 1234,
        });
        for (vm, host) in [(&q35, linux()), (&vmapple, mac())] {
            let a = invocation(vm, &ctx(host)).unwrap().args;
            assert!(has(&a, &["-nic", "none"]), "{a:?}");
            assert!(!a.iter().any(|x| x == "-netdev" || x.contains("net-pci")));
        }
        let a = invocation(&q35_vm(), &ctx(linux())).unwrap().args;
        assert!(!a.iter().any(|x| x == "-nic"));
    }

    #[test]
    fn vmx_only_where_kvm_offers_it() {
        let cpu = |nested_vmx| {
            let inv = invocation(
                &q35_vm(),
                &Ctx {
                    nested_vmx,
                    ..ctx(linux())
                },
            )
            .unwrap();
            let i = inv.args.iter().position(|x| x == "-cpu").unwrap();
            inv.args[i + 1].clone()
        };
        assert_eq!(
            cpu(true),
            format!(
                "{},kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on,vmx=on,{}",
                default_cpu_model(),
                default_cpu_flags()
            )
        );
        assert!(!cpu(false).contains("vmx"));
    }

    #[test]
    fn q35_without_audio_has_no_sound_devices() {
        let mut vm = q35_vm();
        vm.config.audio.enabled = false;
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        for x in ["-audiodev", "usb-audio", "ich9-intel-hda", "hda-output"] {
            assert!(
                !inv.args.iter().any(|a| a.contains(x)),
                "{x} in {:?}",
                inv.args
            );
        }
    }

    #[test]
    fn installer_disk_goes_on_its_own_port_through_an_overlay() {
        let mut vm = q35_vm();
        let q = vm.config.q35.as_mut().unwrap();
        q.installer = Some("/isos/Base,System.img".into());
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=installer,if=none,format=raw,snapshot=on,file=/isos/Base,,System.img"
            ]
        ));
        assert!(has(&a, &["-device", "ide-hd,bus=sata.4,drive=installer"]));
        // The VM's copy of a DMG, in its directory.
        let q = vm.config.q35.as_mut().unwrap();
        q.installer = Some("installer.img".into());
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=installer,if=none,format=raw,snapshot=on,file=/vms/t/installer.img"
            ]
        ));
    }

    #[test]
    fn logs_gpu_and_window_settings_reach_the_environment() {
        let mut vm = q35_vm();
        vm.config.logs = Logs {
            enabled: true,
            level: "debug".into(),
        };
        vm.config.display.borderless = true;
        vm.config.display.refresh = Some(120);
        vm.config.gpu = Gpu {
            driver_file: Some("/icd/nvidia.json".into()),
            device_id: Some(0x2204),
            name: None,
        };
        vm.config
            .extra_env
            .insert("REIMS_VGPU_LOG".into(), "trace".into());
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        let env: BTreeMap<_, _> = inv.env.iter().cloned().collect();
        assert_eq!(env["REIMS_VGPU_LOG"], "trace");
        assert_eq!(env["REIMS_VGPU_WINDOW_BORDERLESS"], "1");
        assert_eq!(env["REIMS_VGPU_WINDOW_REFRESH"], "120");
        assert_eq!(env["VK_DRIVER_FILES"], "/icd/nvidia.json");
        assert_eq!(env["VK_LOADER_DEVICE_ID_FILTER"], "0x2204");
        assert!(has(&inv.args, &["-serial", "file:/vms/t/logs/serial.log"]));
        assert!(has(&inv.args, &["-msg", "timestamp=on"]));
        assert!(!inv.unset.contains(&"REIMS_VGPU_LOG".to_string()));
    }

    #[test]
    fn vmware_svga_takes_apple_gfx_slot_in_qemus_window() {
        let mut vm = q35_vm();
        vm.config.graphics = Graphics::VmwareSvga;
        vm.config.display.count = 2;
        vm.config.display.borderless = true;
        vm.config.display.refresh = Some(120);
        vm.config.gpu.driver_file = Some("/icd/nvidia.json".into());
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        let a = &inv.args;
        assert!(has(a, &["-vga", "none"]));
        assert!(has(
            a,
            &["-device", "vmware-svga,id=gfx,bus=pcie.0,addr=0x4"]
        ));
        assert!(has(a, &["-display", "gtk"]));
        assert!(!a.iter().any(|x| x.contains("apple-gfx")), "{a:?}");
        assert!(inv.env.is_empty(), "{:?}", inv.env);
        // Without a screen for the window, headless.
        let headless = Ctx {
            screen: false,
            ..ctx(linux())
        };
        let a = invocation(&vm, &headless).unwrap().args;
        assert!(has(&a, &["-display", "none"]));
        // On reims-vgpu, the library's window, never QEMU's.
        let a = invocation(&q35_vm(), &ctx(linux())).unwrap().args;
        assert!(has(&a, &["-display", "none"]));
    }

    #[test]
    fn vmapple_follows_run_vmapple_sh() {
        let mut vm = q35_vm();
        vm.config.profile = Profile::VmappleArm64;
        vm.config.q35 = None;
        vm.config.disk = "disk.img".into();
        vm.config.disk_format = ImageFormat::Raw;
        vm.config.vmapple = Some(Vmapple {
            aux: "aux.img".into(),
            ecid: 1234,
        });
        vm.config.display.count = 2;
        let mut c = ctx(mac());
        c.audio_backend = "coreaudio".into();
        let a = invocation(&vm, &c).unwrap().args;
        for pair in [
            &["-accel", "hvf,kernel-irqchip=on"][..],
            &["-M", "vmapple,uuid=1234"],
            &["-bios", AVPBOOTER],
            &["-drive", "file=/vms/t/aux.img.trimmed,if=pflash,format=raw"],
            &["-drive", "file=/vms/t/disk.img,if=none,id=root,format=raw"],
            &["-device", "vmapple-virtio-blk-pci,variant=root,drive=root"],
            &["-global", "apple-gfx-mmio.displays=2"],
            &["-global", "apple-gfx-mmio.resolution=1920x1080"],
            &["-audiodev", "coreaudio,id=snd0"],
        ] {
            assert!(has(&a, pair), "missing {pair:?} in {a:?}");
        }
        assert!(!a.iter().any(|x| x.contains("memfd")));
        assert!(!a.iter().any(|x| x.contains("dump-guest-core")));
        assert!(!a.iter().any(|x| x.contains("ICH9-LPC")));
    }

    #[test]
    fn shell_script_quotes_and_pairs() {
        let inv = Invocation {
            program: "/q/qemu".into(),
            args: vec![
                "-name".into(),
                "a b".into(),
                "-S".into(),
                "-m".into(),
                "8G".into(),
            ],
            env: vec![("X".into(), "1 2".into())],
            unset: vec!["Y".into()],
        };
        let s = inv.shell_script("hello");
        assert_eq!(
            s,
            "#!/bin/sh\n# hello\nunset Y\nexport X='1 2'\nexec /q/qemu \\\n  -name 'a b' \\\n  -S \\\n  -m 8G\n"
        );
    }
}
