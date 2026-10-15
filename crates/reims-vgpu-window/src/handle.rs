//! [`Window`]: a display's window as the display holds it, its [`title`], and the [`FrameSink`]
//! it draws through.
//!
//! A `Window` is `Send` and `Sync`, and holds no AppKit or SDL object: the display's callbacks
//! tell it what the guest did from the display thread, and its last reference may go on any
//! thread. Its AppKit side (the `NSWindow`, its view and layer, the cursor) lives on the main
//! thread and is released there (`shell`); the render thread's reference to the layer, in the
//! window's surface, is taken back and released there too. On Linux SDL's window lives on the UI
//! thread (`sdl::shell`), and goes once the render thread has let its swapchain go. [`Window::close`] is its end: from then on no
//! callback of the host's is called, and the window closes on the UI thread (the main thread on
//! macOS, the library's `reims-vgpu.ui` on Linux). Nothing in it waits for the UI thread, for its
//! render thread or for the GPU.
//!
//! [`Window::attach`] has the OS's side (`native`) open the window: on macOS it starts the
//! window's render thread (`render`), starts the application once per process (`app`) and posts
//! the window's opening to the main thread; on Linux it starts the window's render thread and
//! posts the opening to the library's UI thread, which SDL runs on (`sdl`). The window opens sized
//! for the guest's mode, else for the first mode the display offers ([`Window::offered_mode`]),
//! which the display tells it before the attach and whenever its modes change. What the guest
//! does later reaches the window's two sides: a new frame wakes the render thread; a mode, a
//! cursor glyph or its visibility is stored here and posted to the UI thread, which resizes
//! the window and lays the view out again, or sets the cursor again. A cursor change posts at
//! most one refresh at a time, which reads the latest state.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use reims_vgpu_core::sync::Mutex;

pub use crate::native::FrameSink;

use crate::config::{WindowCallbacks, WindowConfig};
use crate::native::{self, Surface};
use crate::policy;

/// The next window's id.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// What a window's title starts with.
pub const TITLE: &str = "Reims vGPU";

/// The title of the window of the display on `port`: [`TITLE`], then " (port N)" for a port N
/// other than 0. The display's name is not in it.
pub fn title(port: u64) -> String {
    if port == 0 {
        TITLE.to_owned()
    } else {
        format!("{TITLE} (port {port})")
    }
}

/// A rectangle in a drawable's pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    /// Pixels from the left.
    pub x: u64,
    /// Pixels from the top.
    pub y: u64,
    /// Pixels wide.
    pub width: u64,
    /// Pixels high.
    pub height: u64,
}

/// The guest's side of the window, as the display's callbacks last reported it.
#[derive(Clone, Debug)]
pub(crate) struct GuestState {
    /// The guest's mode, once it has set one.
    pub(crate) mode: Option<(u16, u16)>,
    /// The guest's cursor glyph: width, height, hot spot and RGBA pixels.
    pub(crate) cursor: Option<Glyph>,
    /// Whether the guest shows its cursor.
    pub(crate) cursor_visible: bool,
}

impl Default for GuestState {
    fn default() -> Self {
        GuestState {
            mode: None,
            cursor: None,
            cursor_visible: true,
        }
    }
}

/// A cursor glyph, copied from the guest's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Glyph {
    /// Pixels wide.
    pub(crate) width: u32,
    /// Pixels high.
    pub(crate) height: u32,
    /// The hot spot, in pixels from the top left.
    pub(crate) hot_spot: (u16, u16),
    /// `width * 4` bytes a row, `height` rows, RGBA.
    pub(crate) rgba: Vec<u8>,
}

/// A display's window.
pub struct Window {
    /// The window's id among the process's, which the UI thread keeps its OS side by.
    id: u64,
    /// The display's port: which host screen the window opens on (`screens`).
    port: u64,
    /// The window's title.
    title: String,
    /// The config's flag.
    match_host_screen: bool,
    /// The config's chosen mode.
    chosen_mode: Option<(u16, u16)>,
    /// The config's density for the chosen mode.
    chosen_ppi: Option<f64>,
    /// The host's callbacks, until the window's end takes them out.
    callbacks: Mutex<Option<WindowCallbacks>>,
    /// What the guest last reported.
    guest: Mutex<GuestState>,
    /// The first mode the display offers the guest, once it offers one.
    offered: Mutex<Option<(u16, u16)>>,
    /// The size in millimetres the display tells the guest, once it is known: what the guest's
    /// scale follows ([`policy::guest_scale`]).
    told_size: Mutex<Option<(f64, f64)>>,
    /// What draws the display's frames, once the display exists.
    sink: Mutex<Option<Arc<dyn FrameSink>>>,
    /// Set at the window's end.
    closed: AtomicBool,
    /// Set at the attach: the window has a render thread, and opens on the UI thread.
    attached: AtomicBool,
    /// Set once the opening is posted to the UI thread: what is posted after it finds the
    /// window open there.
    open_posted: AtomicBool,
    /// A cursor refresh is posted to the UI thread and has not run yet.
    cursor_posted: AtomicBool,
    /// What the render thread draws with.
    surface: Surface,
}

impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Window")
            .field("port", &self.port)
            .field("title", &self.title)
            .field("closed", &self.closed.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl Window {
    /// The window of the display on `port`, titled `title`. It opens at [`Window::attach`];
    /// until then it only records what the guest reports.
    pub fn new(config: WindowConfig, port: u64, title: String) -> Arc<Window> {
        Arc::new(Window {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            port,
            title,
            match_host_screen: config.match_host_screen,
            chosen_mode: config.mode,
            chosen_ppi: config.mode_ppi,
            callbacks: Mutex::new(Some(config.callbacks)),
            guest: Mutex::new(GuestState::default()),
            offered: Mutex::new(None),
            told_size: Mutex::new(None),
            sink: Mutex::new(None),
            closed: AtomicBool::new(false),
            attached: AtomicBool::new(false),
            open_posted: AtomicBool::new(false),
            cursor_posted: AtomicBool::new(false),
            surface: Surface::new(),
        })
    }

    /// The display's port.
    pub fn port(&self) -> u64 {
        self.port
    }

    /// The window's title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Whether the config asked for modes and a density matching the host's screen.
    pub fn match_host_screen(&self) -> bool {
        self.match_host_screen
    }

    /// Opens the window over `sink`, the display's: its render thread starts, and the window
    /// opens on the UI thread (on macOS, when the main thread runs its run loop). Does nothing
    /// after the end; a second attach only replaces the sink.
    pub fn attach(self: &Arc<Self>, sink: Arc<dyn FrameSink>) {
        if self.is_closed() {
            return;
        }
        *self.sink.lock() = Some(sink);
        // An end that raced the store may have missed the sink.
        if self.is_closed() {
            drop(self.sink.lock().take());
            return;
        }
        if self.attached.swap(true, Ordering::AcqRel) {
            return;
        }
        native::open(self);
        self.open_posted.store(true, Ordering::Release);
    }

    /// The guest presented a frame: the window draws it.
    pub fn new_frame(&self) {
        self.surface.request_draw();
    }

    /// The display offers the guest `width` by `height` first: the window opens sized for it
    /// until the guest sets a mode. It sizes a window that has opened already for nothing.
    pub fn offered_mode(&self, width: u16, height: u16) {
        *self.offered.lock() = Some((width, height));
    }

    /// The guest set a mode `width` by `height`: the image takes its aspect ratio, and a window
    /// that follows the guest's mode is resized for it (`shell`).
    pub fn guest_mode(&self, width: u16, height: u16) {
        self.guest.lock().mode = Some((width, height));
        if self.is_open() {
            native::guest_mode_changed(self.id);
        }
    }

    /// The guest set a cursor glyph: `rgba` holds `height` rows of `width * 4` bytes.
    pub fn cursor_glyph(&self, width: u32, height: u32, hot_spot: (u16, u16), rgba: &[u8]) {
        self.guest.lock().cursor = Some(Glyph {
            width,
            height,
            hot_spot,
            rgba: rgba.to_vec(),
        });
        self.post_cursor_refresh();
    }

    /// The guest showed (`true`) or hid its cursor.
    pub fn cursor_show(&self, show: bool) {
        self.guest.lock().cursor_visible = show;
        self.post_cursor_refresh();
    }

    /// Posts a cursor refresh to the UI thread, unless one is posted and has not run yet: it
    /// reads the latest glyph and visibility when it runs.
    fn post_cursor_refresh(&self) {
        if self.is_open() && !self.cursor_posted.swap(true, Ordering::AcqRel) {
            native::cursor_changed(self.id);
        }
    }

    /// The window's end: no callback of the host's runs after it returns, other than one
    /// already running on the UI thread, and the window closes there. Its render thread ends
    /// by itself. Waits for nothing. Later calls do nothing.
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let callbacks = self.callbacks.lock().take();
        let sink = self.sink.lock().take();
        self.surface.close();
        if self.attached.load(Ordering::Acquire) {
            native::close(self.id);
        }
        drop((callbacks, sink));
    }

    /// Whether the window has ended.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Whether the window's opening is posted and the window has not ended: what is posted to
    /// the UI thread now runs after the opening, and finds the window open.
    fn is_open(&self) -> bool {
        self.open_posted.load(Ordering::Acquire) && !self.is_closed()
    }

    /// The first mode the display offers, once it offers one.
    pub fn offered(&self) -> Option<(u16, u16)> {
        *self.offered.lock()
    }

    /// The config's chosen mode (`WindowConfig::mode`).
    pub fn chosen_mode(&self) -> Option<(u16, u16)> {
        self.chosen_mode
    }

    /// The config's density for its chosen mode (`WindowConfig::mode_ppi`).
    pub fn chosen_ppi(&self) -> Option<f64> {
        self.chosen_ppi
    }

    /// The display tells the guest it is `size_in_millimeters`: the window opens and resizes at
    /// a guest point per host point, the guest's scale for each mode following it.
    pub fn told_size(&self, size_in_millimeters: (f64, f64)) {
        *self.told_size.lock() = Some(size_in_millimeters);
    }
}

/// What the window's backend reads of it.
impl Window {
    /// The window's id among the process's.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The host's callbacks, while the window lives.
    pub(crate) fn callbacks(&self) -> Option<WindowCallbacks> {
        self.callbacks.lock().clone()
    }

    /// What the guest last reported.
    pub(crate) fn guest(&self) -> GuestState {
        self.guest.lock().clone()
    }

    /// The guest's mode, once it has set one.
    pub(crate) fn mode(&self) -> Option<(u16, u16)> {
        self.guest.lock().mode
    }

    /// The scale the guest runs `mode` at: 2 or 1 by the size the display tells it
    /// ([`policy::guest_scale`]), 1 before the display says.
    #[allow(dead_code)]
    pub(crate) fn guest_scale(&self, mode: (u16, u16)) -> u8 {
        self.told_size
            .lock()
            .map_or(1, |size| policy::guest_scale(mode, size))
    }

    /// The sink, copied out of its lock, until the window's end.
    pub(crate) fn sink(&self) -> Option<Arc<dyn FrameSink>> {
        self.sink.lock().clone()
    }

    /// What the render thread draws with.
    pub(crate) fn surface(&self) -> &Surface {
        &self.surface
    }

    /// A posted cursor refresh runs: a later change posts another.
    pub(crate) fn take_cursor_refresh(&self) {
        self.cursor_posted.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "Reims vGPU", then the port when it is not 0
    #[test]
    fn a_window_s_title_is_reims_vgpu_and_its_port() {
        assert_eq!(title(0), "Reims vGPU");
        assert_eq!(title(1), "Reims vGPU (port 1)");
        assert_eq!(title(7), "Reims vGPU (port 7)");
    }
}
