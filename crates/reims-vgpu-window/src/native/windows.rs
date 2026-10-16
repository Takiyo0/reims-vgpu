//! The window on Windows: raw Win32, drawn through Vulkan over `VK_KHR_win32_surface`. No SDL and
//! no AppKit; the window, its message loop, input and monitors are Win32's, on a UI thread of the
//! library's own (`ui`), and each window's render thread draws into its swapchain (`render`),
//! through the OS-agnostic presenter (`crate::present`) the display's sink makes.
//!
//! This parallels `native/linux.rs`, with two Windows-specific points the shared code does not
//! see:
//! - **No shared `crate::sdl`.** Linux re-exports `sdl` as `sdl_os` for the shared shell and
//!   cursor; the Windows backend compiles neither, so this facade exports no `sdl_os`, and the
//!   `#[cfg(any(target_os = "linux", feature = "macos-sdl"))] mod sdl` in `lib.rs` stays off here.
//! - **Deferred window destruction.** The `HWND` must outlive the Vulkan surface made of it. A
//!   close marks the window closing and wakes the render thread; the render thread drops its
//!   presenter (destroying the swapchain and surface) and only then posts its release to the UI
//!   thread (`ui::post`), which performs the `DestroyWindow`. So the surface always goes before
//!   the window under it, as it does on Linux (`render`, `ui`).
//!
//! This file is what the shared window code asks of the backend: what draws into the window
//! ([`FrameSink`]), the window's opening, changes and end posted to the UI thread, and the host's
//! screens as Win32 lists them.

mod input;
mod keymap;
mod render;
mod ui;

#[cfg(test)]
mod stage1_test;

use std::sync::Arc;

pub use render::FrameSink;
pub(crate) use render::Surface;

use crate::handle::Window;
use crate::screens::HostScreen;

/// Starts `window`'s render thread and posts its opening to the UI thread, starting that where it
/// has not started.
pub(crate) fn open(window: &Arc<Window>) {
    render::start(Arc::clone(window));
    let window = Arc::downgrade(window);
    ui::post(move || {
        if let Some(window) = window.upgrade() {
            ui::open_window(&window);
        }
    });
}

/// Posts to the UI thread that the guest of window `id` set a mode.
pub(crate) fn guest_mode_changed(id: u64) {
    ui::post(move || ui::guest_mode_changed(id));
}

/// Posts a cursor refresh of window `id` to the UI thread.
pub(crate) fn cursor_changed(id: u64) {
    ui::post(move || ui::cursor_changed(id));
}

/// Posts the close of window `id` to the UI thread: it signals the render thread, which releases
/// the surface and then posts the window's destruction back (the module doc).
pub(crate) fn close(id: u64) {
    ui::post(move || ui::close(id));
}

/// The host's screens, as Win32 lists them (`ui`).
pub(crate) fn host_screens() -> Vec<HostScreen> {
    ui::screens()
}
