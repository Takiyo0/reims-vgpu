//! What scripts/install-ctl.sh installed: `install.toml` in the config directory names the
//! repository it built from, and the QEMU builds live in the data directory's `qemu/<build>`,
//! each a whole install prefix (QEMU finds its data files relative to its binary), with
//! `qemu/current` linking the newest. An update installs a new build beside the others and moves
//! the link, so a running VM's binary is never written.
//!
//! `REIMS_CTL_QEMU` names a QEMU binary to use instead (a build tree's, such as `make qemu`'s
//! `build/qemu/qemu-system-aarch64`); `qemu-img` is taken from beside it.

use crate::host::Profile;
use crate::paths;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Install {
    /// The reims-vgpu checkout the install built from, which `update` pulls.
    pub repo: PathBuf,
    /// Its commit at the install.
    #[serde(default)]
    pub commit: String,
    /// The QEMU build installed (a directory under the data directory's `qemu`).
    #[serde(default)]
    pub qemu: PathBuf,
    #[serde(default)]
    pub installed_at: String,
}

impl Install {
    pub fn load() -> Option<Install> {
        let text = std::fs::read_to_string(paths::install_file()).ok()?;
        toml::from_str(&text).ok()
    }

    pub fn save(&self) -> Result<()> {
        let file = paths::install_file();
        std::fs::create_dir_all(file.parent().unwrap())?;
        std::fs::write(&file, toml::to_string_pretty(self)?)
            .with_context(|| format!("write {}", file.display()))
    }
}

/// The QEMU binary for `profile`.
pub fn qemu_binary(profile: Profile) -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("REIMS_CTL_QEMU") {
        return Ok(PathBuf::from(p));
    }
    let bin = qemu_install_path(profile);
    if !bin.exists() {
        bail!(
            "QEMU is not installed ({} is missing): run scripts/install-ctl.sh (Unix) or \
             scripts/install-ctl.ps1 (Windows) in your reims-vgpu checkout, or set \
             REIMS_CTL_QEMU to a qemu-system binary",
            bin.display()
        );
    }
    Ok(bin)
}

/// The QEMU binary for `profile`, or where it would be, for printing a command.
pub fn qemu_binary_or_default(profile: Profile) -> PathBuf {
    qemu_binary(profile).unwrap_or_else(|_| qemu_install_path(profile))
}

fn qemu_install_path(profile: Profile) -> PathBuf {
    let name = profile.qemu_binary();
    #[cfg(windows)]
    let name = format!("{name}.exe");
    paths::qemu_root().join("current/bin").join(name)
}

/// `qemu-img`, beside the QEMU binary.
pub fn qemu_img(profile: Profile) -> Result<PathBuf> {
    let bin = qemu_binary(profile)?;
    #[cfg(windows)]
    let img = bin.with_file_name("qemu-img.exe");
    #[cfg(not(windows))]
    let img = bin.with_file_name("qemu-img");
    if !img.exists() {
        bail!(
            "{} is missing: reinstall with scripts/install-ctl.sh (Unix) or scripts/install-ctl.ps1 (Windows)",
            img.display()
        );
    }
    Ok(img)
}

/// The audio backends `qemu` was built with (`-audiodev help`), cached per binary.
pub fn audio_backends(qemu: &Path) -> Vec<String> {
    backends(qemu, "-audiodev")
}

/// The display backends `qemu` was built with (`-display help`), cached per binary.
pub fn display_backends(qemu: &Path) -> Vec<String> {
    backends(qemu, "-display")
}

/// What `qemu <option> help` listed, by binary and option.
type Lists = HashMap<(PathBuf, &'static str), Vec<String>>;

/// What `qemu <option> help` lists, a name to a line.
fn backends(qemu: &Path, option: &'static str) -> Vec<String> {
    static CACHE: OnceLock<Mutex<Lists>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (qemu.to_path_buf(), option);
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return v.clone();
    }
    let out = std::process::Command::new(qemu)
        .args([option, "help"])
        .output();
    let list: Vec<String> = match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.contains(' ') && !l.ends_with(':'))
            .map(String::from)
            .collect(),
        Err(_) => Vec::new(),
    };
    cache.lock().unwrap().insert(key, list.clone());
    list
}

/// The audio backend to use where the VM names none: the host's native one, among those built.
pub fn default_audio_backend(available: &[String]) -> String {
    let wanted: &[&str] = if cfg!(target_os = "macos") {
        &["coreaudio"]
    } else if cfg!(target_os = "windows") {
        &["dsound", "wasapi", "sdl"]
    } else {
        &["pipewire", "pa", "alsa", "sdl"]
    };
    for w in wanted {
        if available.iter().any(|a| a == w) {
            return (*w).into();
        }
    }
    if available.is_empty() {
        // QEMU could not be asked (not installed yet): its usual choice.
        return if cfg!(target_os = "macos") {
            "coreaudio"
        } else if cfg!(target_os = "windows") {
            "dsound"
        } else {
            "pipewire"
        }
        .into();
    }
    "none".into()
}

/// `qemu --version`'s first line.
pub fn qemu_version(qemu: &Path) -> Option<String> {
    let o = std::process::Command::new(qemu)
        .arg("--version")
        .output()
        .ok()?;
    String::from_utf8_lossy(&o.stdout)
        .lines()
        .next()
        .map(String::from)
}

/// Whether this QEMU build exposes the Windows Hypervisor Platform accelerator.
pub fn has_whpx(qemu: &Path) -> bool {
    let Ok(output) = std::process::Command::new(qemu)
        .args(["-accel", "help"])
        .output()
    else {
        return false;
    };
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output.status.success() && text.lines().any(|line| line.trim() == "whpx")
}

/// Whether `qemu -device help` lists a device this controller needs.
pub fn has_device(qemu: &Path, device: &str) -> bool {
    let Ok(output) = std::process::Command::new(qemu)
        .args(["-device", "help"])
        .output()
    else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout).contains(&format!("name \"{device}\""))
}

/// QEMU's firmware/data directory, either explicitly given or at the usual install-prefix path.
pub fn qemu_data_dir(qemu: &Path) -> Option<PathBuf> {
    std::env::var_os("REIMS_CTL_QEMU_DATA")
        .map(PathBuf::from)
        .or_else(|| {
            let bin = qemu.parent()?;
            let prefix = bin.parent()?;
            let data = prefix.join("share/qemu");
            data.is_dir().then_some(data)
        })
}

/// The Reims option ROM, if present at its explicit path or in QEMU's data directory.
pub fn qemu_rom(qemu: &Path) -> Option<PathBuf> {
    std::env::var_os("REIMS_CTL_QEMU_ROM")
        .map(PathBuf::from)
        .or_else(|| qemu_data_dir(qemu).map(|d| d.join("ReimsVgpu.rom")))
        .filter(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_backend_prefers_the_native_one() {
        let v = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        if cfg!(target_os = "macos") {
            assert_eq!(
                default_audio_backend(&v(&["none", "coreaudio"])),
                "coreaudio"
            );
        } else if cfg!(target_os = "linux") {
            assert_eq!(
                default_audio_backend(&v(&["none", "alsa", "pa", "pipewire"])),
                "pipewire"
            );
            assert_eq!(default_audio_backend(&v(&["none", "alsa"])), "alsa");
        } else {
            assert_eq!(default_audio_backend(&v(&["none", "wasapi"])), "wasapi");
        }
        assert_eq!(default_audio_backend(&v(&["none", "dbus"])), "none");
    }
}
