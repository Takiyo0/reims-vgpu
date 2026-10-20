//! The host reims-ctl runs on, and the machine profiles it can run there.
//!
//! A profile is one QEMU machine for one guest architecture: its binary, accelerator, devices
//! and what it can attach. Each host supports the profiles [`Profile::supported_on`] lists:
//! - macOS on arm64: [`Profile::VmappleArm64`], QEMU's `vmapple` machine on HVF, the machine
//!   Virtualization.framework gives an arm64 macOS guest, with `apple-gfx-mmio`;
//! - Linux on x86-64: [`Profile::Q35X86_64`], `q35` on KVM, booted through OpenCore, with
//!   `apple-gfx-pci`.
//!
//! Another host or guest architecture is another profile (x86-64 guests on an Intel Mac would be
//! `q35` on HVF, without the memfd RAM only Linux has): add it here and in `qemu.rs`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    Mac,
    Linux,
    Windows,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Aarch64,
    X86_64,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Host {
    pub os: Os,
    pub arch: Arch,
}

impl Host {
    pub fn current() -> Host {
        let os = if cfg!(target_os = "macos") {
            Os::Mac
        } else if cfg!(target_os = "linux") {
            Os::Linux
        } else if cfg!(target_os = "windows") {
            Os::Windows
        } else {
            Os::Other
        };
        let arch = if cfg!(target_arch = "aarch64") {
            Arch::Aarch64
        } else if cfg!(target_arch = "x86_64") {
            Arch::X86_64
        } else {
            Arch::Other
        };
        Host { os, arch }
    }

    pub fn default_profile(self) -> Option<Profile> {
        Profile::supported_on(self).first().copied()
    }
}

/// Whether KVM here offers VMX to its guests: `kvm_intel` is loaded, which only a CPU with VMX
/// allows, and its `nested` is on, Linux's default since 4.20. AMD's KVM offers SVM instead,
/// which macOS's Hypervisor.framework does not drive, and HVF offers neither.
pub fn kvm_nested_vmx() -> bool {
    std::fs::read_to_string("/sys/module/kvm_intel/parameters/nested")
        .is_ok_and(|s| matches!(s.trim(), "Y" | "1"))
}

/// Whether a window can open for a QEMU run with `env` over this process's environment: it names
/// a Wayland compositor or an X server. Without one, QEMU's GTK window ends QEMU as it starts,
/// where the library's runs headless.
pub fn has_screen(env: &BTreeMap<String, String>) -> bool {
    if cfg!(target_os = "windows") {
        return true;
    }
    ["WAYLAND_DISPLAY", "DISPLAY"]
        .iter()
        .any(|v| match env.get(*v) {
            Some(s) => !s.is_empty(),
            None => std::env::var_os(v).is_some_and(|s| !s.is_empty()),
        })
}

impl fmt::Display for Host {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let os = match self.os {
            Os::Mac => "macOS",
            Os::Linux => "Linux",
            Os::Windows => "Windows",
            Os::Other => std::env::consts::OS,
        };
        let arch = match self.arch {
            Arch::Aarch64 => "arm64",
            Arch::X86_64 => "x86-64",
            Arch::Other => std::env::consts::ARCH,
        };
        write!(f, "{os} on {arch}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Profile {
    #[serde(rename = "vmapple-arm64")]
    VmappleArm64,
    #[serde(rename = "q35-x86_64")]
    Q35X86_64,
}

impl Profile {
    pub fn supported_on(host: Host) -> &'static [Profile] {
        match (host.os, host.arch) {
            (Os::Mac, Arch::Aarch64) => &[Profile::VmappleArm64],
            (Os::Linux | Os::Windows, Arch::X86_64) => &[Profile::Q35X86_64],
            _ => &[],
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Profile::VmappleArm64 => "vmapple-arm64",
            Profile::Q35X86_64 => "q35-x86_64",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Profile::VmappleArm64 => "arm64 macOS guest (vmapple, HVF)",
            Profile::Q35X86_64 => "x86-64 macOS guest (q35 and OpenCore, KVM or WHPX)",
        }
    }

    /// The QEMU binary's name in the install's `bin`.
    pub fn qemu_binary(self) -> &'static str {
        match self {
            Profile::VmappleArm64 => "qemu-system-aarch64",
            Profile::Q35X86_64 => "qemu-system-x86_64",
        }
    }

    /// The accelerator on `host`.
    pub fn accel(self, host: Host) -> &'static str {
        match (self, host.os) {
            // vendor/qemu's 4a2e2b3967 boots vmapple guests with the in-kernel GIC.
            (Profile::VmappleArm64, _) => "hvf,kernel-irqchip=on",
            (Profile::Q35X86_64, Os::Mac) => "hvf",
            (Profile::Q35X86_64, Os::Windows) => "whpx",
            (Profile::Q35X86_64, _) => "kvm",
        }
    }

    /// Whether the guest's RAM must be a shared memfd: the library on Linux builds its views of
    /// guest pages from the file behind the RAM, and refuses RAM without one.
    pub fn memfd_ram(self, host: Host) -> bool {
        host.os == Os::Linux
    }

    /// Whether a Vulkan GPU can be chosen: the library runs on Vulkan on Linux, and on Metal's
    /// default device on macOS.
    pub fn gpu_choice(self, host: Host) -> bool {
        host.os == Os::Linux
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_supported_host_has_one_profile() {
        let mac = Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        };
        let linux = Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        };
        assert_eq!(mac.default_profile(), Some(Profile::VmappleArm64));
        assert_eq!(linux.default_profile(), Some(Profile::Q35X86_64));
        assert_eq!(
            Host {
                os: Os::Linux,
                arch: Arch::Aarch64
            }
            .default_profile(),
            None
        );
        assert!(Profile::Q35X86_64.memfd_ram(linux));
        assert!(!Profile::VmappleArm64.memfd_ram(mac));
    }

    #[test]
    fn a_vms_environment_decides_its_screen_over_this_process() {
        let env = |v: &str| {
            BTreeMap::from(["WAYLAND_DISPLAY", "DISPLAY"].map(|k| (k.to_string(), v.to_string())))
        };
        if cfg!(target_os = "windows") {
            assert!(has_screen(&env("")));
        } else {
            assert!(has_screen(&env(":0")));
            assert!(!has_screen(&env("")));
        }
    }
}
