//! A window's render thread on Windows, the [`Surface`] it shares with the window and the UI
//! thread, the display's sink ([`FrameSink`]), and the [`Target`] a window's surface is made of
//! (`VK_KHR_win32_surface` over its `HWND`).
//!
//! The `Surface` state machine and the draw loop are the same as Linux's (`native/linux/render.rs`):
//! one render thread per window (`reims-vgpu.window`), waiting until there is something to draw
//! and drawing only while the window is shown, through the OS-agnostic presenter
//! (`crate::present`). The one Windows difference is the end: the thread drops its presenter
//! (destroying the swapchain and the Vulkan surface) and only then posts the window's release to
//! the UI thread ([`crate::native::windows::ui::released`]), which performs the `DestroyWindow`,
//! so the surface always goes before the window under it. There is no shared `crate::sdl`, so the
//! release is posted the Windows way, not through `sdl::shell::released`.

use std::sync::Arc;

use ash::vk;
use reims_vgpu_core::rt::spawn_named;
use reims_vgpu_core::sync::{Condvar, Mutex};
use reims_vgpu_vulkan::os::OsThreads;
use windows_sys::Win32::Foundation::{HINSTANCE, HWND};

use crate::geometry::Layout;
use crate::handle::Window;
use crate::present::{Presenting, RETRY, VulkanTarget};

/// The render thread's stack.
const STACK: usize = 512 * 1024;

/// What draws a display's current frame into a window: `reims-vgpu`'s, over the display, which it
/// holds weakly.
pub trait FrameSink: Send + Sync {
    /// A presenter over `target`'s window: one that makes the window's surface on the sink's
    /// Vulkan instance and a swapchain over it. `None` where there is none (the display has
    /// ended, or this host has no swapchain): the window then stays empty.
    fn presenter(&self, target: &dyn VulkanTarget) -> Option<Box<dyn crate::present::Presenter>>;
}

/// A window's `HWND` and the module's `HINSTANCE`, as its render thread holds them: only to make
/// the window's Vulkan surface of it (`VK_KHR_win32_surface`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    /// The window's handle, valid until the UI thread destroys it, which it does only after this
    /// render thread has released the surface (the module doc).
    pub(super) hwnd: HWND,
    /// The process module handle the window class was registered with.
    pub(super) hinstance: HINSTANCE,
}

// SAFETY: the UI thread makes and destroys the window, and destroys it only once the render thread
// has posted its release (`ui::released`); the render thread, the only other one the target
// reaches, only reads the handles to make the window's Vulkan surface of them, while the window
// lives.
unsafe impl Send for Target {}

impl VulkanTarget for Target {
    /// The window's `VK_KHR_win32_surface`, resolved through the passed `get_instance_proc_addr`
    /// (the trait hands a raw instance and resolver, not an ash loader), as `crate::present`
    /// requires.
    fn create_surface(
        &self,
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        instance: vk::Instance,
    ) -> Option<vk::SurfaceKHR> {
        // Resolve vkCreateWin32SurfaceKHR through the instance's own resolver: no ambient ash
        // loader exists on this path.
        let name = c"vkCreateWin32SurfaceKHR";
        // SAFETY: `instance` is live and `get_instance_proc_addr` is the resolver it was made
        // with; the name is a NUL-terminated Vulkan command name.
        let create = unsafe { get_instance_proc_addr(instance, name.as_ptr()) };
        let create: vk::PFN_vkCreateWin32SurfaceKHR = match create {
            Some(f) => {
                // SAFETY: the loader returned a non-null pointer for this exact command name, so
                // it has that command's signature.
                unsafe {
                    std::mem::transmute::<vk::PFN_vkVoidFunction, vk::PFN_vkCreateWin32SurfaceKHR>(
                        Some(f),
                    )
                }
            }
            None => {
                log::warn!(
                    target: "reims_vgpu_window",
                    "the Vulkan instance resolves no vkCreateWin32SurfaceKHR: the window cannot present"
                );
                return None;
            }
        };
        let info = vk::Win32SurfaceCreateInfoKHR {
            s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::Win32SurfaceCreateFlagsKHR::empty(),
            // ash types the Win32 handles as `isize` (vk_platform HINSTANCE/HWND); windows-sys
            // types them as `*mut c_void`. Cast across.
            hinstance: self.hinstance as isize,
            hwnd: self.hwnd as isize,
            _marker: std::marker::PhantomData,
        };
        let mut surface = vk::SurfaceKHR::null();
        // SAFETY: `instance` is live; `info` names this window's live `HWND` and the module's
        // `HINSTANCE`; `create` is the resolved command; the out-pointer is a valid slot.
        let result = unsafe { create(instance, &info, std::ptr::null(), &mut surface) };
        if result == vk::Result::SUCCESS {
            Some(surface)
        } else {
            log::warn!(
                target: "reims_vgpu_window",
                "vkCreateWin32SurfaceKHR failed ({result:?}): the window cannot present"
            );
            None
        }
    }
}

/// What the render thread draws with, shared with the window and the UI thread.
#[derive(Debug, Default)]
pub(crate) struct Surface {
    state: Mutex<SurfaceState>,
    wake: Condvar,
}

/// The surface's state.
#[derive(Debug, Default)]
struct SurfaceState {
    /// The window's handles, once it has opened on the UI thread.
    target: Option<Target>,
    /// Where the drawable and the image are, once the window has laid out.
    layout: Option<Layout>,
    /// Whether the window is shown.
    visible: bool,
    /// Something new to show: a frame, a layout, the window showing again.
    dirty: bool,
    /// The window has ended: the render thread ends.
    closed: bool,
}

/// A draw the render thread takes: the window's handles and the layout.
struct Work {
    target: Target,
    layout: Layout,
}

impl SurfaceState {
    /// The draw to make now, if any: something new, the window shown, its handles and layout
    /// there.
    fn take(&mut self) -> Option<Work> {
        if !(self.dirty && self.visible) {
            return None;
        }
        let (Some(target), Some(layout)) = (self.target, self.layout) else {
            return None;
        };
        self.dirty = false;
        Some(Work { target, layout })
    }

    /// The draw to make again after one that had no image in time: the newest, else the same,
    /// while the window is shown.
    fn retake(&mut self, work: Work) -> Option<Work> {
        if !self.visible {
            return None;
        }
        Some(self.take().unwrap_or(work))
    }
}

impl Surface {
    /// An empty surface: no window, no layout, not shown.
    pub(crate) fn new() -> Surface {
        Surface::default()
    }

    /// Changes the state with `f` and wakes the render thread.
    fn update(&self, f: impl FnOnce(&mut SurfaceState)) {
        f(&mut self.state.lock());
        self.wake.notify_all();
    }

    /// The window's handles, from the UI thread as the window opens.
    pub(crate) fn set_target(&self, target: Target) {
        self.update(|s| {
            s.target = Some(target);
            s.dirty = true;
        });
    }

    /// The window's layout: a changed one is drawn.
    pub(crate) fn set_layout(&self, layout: Layout) {
        self.update(|s| {
            if s.layout != Some(layout) {
                s.layout = Some(layout);
                s.dirty = true;
            }
        });
    }

    /// Whether the window is shown: nothing is drawn while it is not, and it is drawn again when
    /// it shows.
    pub(crate) fn set_visible(&self, visible: bool) {
        self.update(|s| {
            if visible && !s.visible {
                s.dirty = true;
            }
            s.visible = visible;
        });
    }

    /// The guest presented a frame: it is drawn.
    pub(crate) fn request_draw(&self) {
        self.update(|s| s.dirty = true);
    }

    /// The window's end: the render thread ends.
    pub(crate) fn close(&self) {
        self.update(|s| s.closed = true);
    }

    /// Waits for the next draw; `None` at the window's end. `retry` asks for the last draw
    /// again after [`RETRY`], where nothing newer comes first.
    fn next(&self, retry: Option<Work>) -> Option<Work> {
        let mut state = self.state.lock();
        if let Some(work) = retry
            && !state.closed
        {
            self.wake.wait_for(&mut state, RETRY);
            if state.closed {
                return None;
            }
            if let Some(work) = state.retake(work) {
                return Some(work);
            }
        }
        loop {
            if state.closed {
                return None;
            }
            if let Some(work) = state.take() {
                return Some(work);
            }
            self.wake.wait(&mut state);
        }
    }
}

/// Starts `window`'s render thread.
pub(crate) fn start(window: Arc<Window>) {
    spawn_named(
        &OsThreads::new(),
        "reims-vgpu.window",
        STACK,
        Box::new(()),
        move || run(&window),
    );
}

/// The render thread's body: draws until the window's end, then lets its presenter go (destroying
/// the swapchain and surface) and posts the window's release to the UI thread, so the surface
/// goes before the window.
fn run(window: &Window) {
    let surface = window.surface();
    let mut presenting = Presenting::default();
    let mut retry = None;
    while let Some(work) = surface.next(retry.take()) {
        if presenting.needs_presenter() {
            // The sink is copied out of the window's lock: none is held while it answers.
            let Some(sink) = window.sink() else {
                continue;
            };
            presenting.make(|| sink.presenter(&work.target));
        }
        if presenting.draw(work.layout.drawable, work.layout.pixels) {
            retry = Some(work);
        }
    }
    drop(presenting);
    let id = window.id();
    super::ui::post(move || super::ui::released(id));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{self, Insets};

    /// A layout for a 100×50 point window at scale 2.
    fn a_layout() -> Layout {
        geometry::layout((100.0, 50.0), 2.0, Insets::default(), Some((1920, 1080)))
    }

    /// A target that names no window, for the state's logic.
    fn a_target() -> Target {
        Target {
            hwnd: std::ptr::null_mut(),
            hinstance: std::ptr::null_mut(),
        }
    }

    /// Nothing is drawn while the window is hidden, and the window is drawn again as it shows
    #[test]
    fn a_hidden_window_is_drawn_when_it_shows() {
        let mut s = SurfaceState {
            target: Some(a_target()),
            layout: Some(a_layout()),
            dirty: true,
            ..SurfaceState::default()
        };
        assert!(s.take().is_none());
        s.visible = true;
        assert!(s.take().is_some());
        assert!(s.take().is_none(), "one draw per request");
    }

    /// Nothing is drawn before the window and the layout
    #[test]
    fn nothing_is_drawn_before_the_window_and_layout() {
        let mut s = SurfaceState {
            visible: true,
            dirty: true,
            ..SurfaceState::default()
        };
        assert!(s.take().is_none());
        s.layout = Some(a_layout());
        assert!(s.take().is_none());
        s.target = Some(a_target());
        assert_eq!(s.take().map(|w| w.layout), Some(a_layout()));
    }

    /// Showing again, a new layout and a new frame each ask for a draw; an unchanged layout does
    /// not; a draw with no image comes again
    #[test]
    fn what_asks_for_a_draw() {
        let surface = Surface::new();
        surface.set_target(a_target());
        surface.set_layout(a_layout());
        surface.set_visible(true);
        assert!(surface.state.lock().take().is_some());
        surface.set_layout(a_layout());
        assert!(surface.state.lock().take().is_none());
        surface.request_draw();
        let work = surface.next(None).unwrap();
        let again = surface.next(Some(work)).unwrap();
        assert_eq!(
            again.layout,
            a_layout(),
            "the draw with no image comes again"
        );
        surface.set_visible(false);
        surface.set_visible(true);
        assert!(surface.state.lock().take().is_some());
        surface.close();
        assert!(surface.next(None).is_none());
    }

    /// A draw that had no image in time is not made again while the window is hidden
    #[test]
    fn a_hidden_window_takes_no_retry() {
        let mut s = SurfaceState {
            target: Some(a_target()),
            layout: Some(a_layout()),
            visible: true,
            dirty: true,
            ..SurfaceState::default()
        };
        let work = s.take().unwrap();
        assert!(s.retake(work).is_some(), "shown, the draw comes again");
        let work = Work {
            target: a_target(),
            layout: a_layout(),
        };
        s.visible = false;
        assert!(s.retake(work).is_none(), "hidden, it does not");
    }
}
