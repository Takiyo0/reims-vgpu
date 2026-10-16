//! The catch around a call of a host function ([`super::host_call`]), in one file per OS with the
//! same function: `catch(f)` runs `f` and answers its value, or `Err(())` for what the host threw.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
pub(super) use linux::catch;
#[cfg(target_os = "macos")]
pub(super) use macos::catch;
#[cfg(target_os = "windows")]
pub(super) use windows::catch;
