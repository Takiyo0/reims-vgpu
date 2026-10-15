//! The UI thread on Windows, `reims-vgpu.ui`: a thread of the library's own that owns every
//! window and its Win32 message pump, the work posted to it, and the host's screens as Win32
//! lists them. It parallels `native/linux/ui.rs`, with Win32 in place of SDL.
//!
//! It starts at the first window's opening or the first look at the host's screens, and runs for
//! the process. Every window is created and destroyed here, and the message pump
//! (`GetMessage`/`DispatchMessage`) runs here; work is posted to it ([`post`]) and run between
//! messages, woken by a posted thread message.
//!
//! **Deferred destruction.** The `HWND` must outlive the Vulkan surface made of it. A close
//! ([`close`]) does NOT `DestroyWindow`: it marks the window closing and signals the render
//! thread (`Surface::close`), which drops its presenter (destroying the swapchain and surface)
//! and posts [`released`] back here; only [`released`] calls `DestroyWindow`. So the surface
//! always goes before the window under it, as on Linux.
//!
//! **Stage 1 scope.** This opens a window, lays it out for its mode, shows it (so the render
//! thread presents), tracks size through `WM_SIZE`, and tears down in the required order. Input
//! (`WM_*` to the host callbacks), per-monitor cursor and the guest-mode resize policy are Stage 3
//! and are not wired here yet; the hooks are marked.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use reims_vgpu_core::rt::spawn_named;
use reims_vgpu_core::sync::{Condvar, Mutex};
use reims_vgpu_vulkan::os::OsThreads;
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    CreateBitmap, CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject,
    EnumDisplayMonitors, EnumDisplaySettingsW, GetMonitorInfoW, DEVMODEW, ENUM_CURRENT_SETTINGS,
    HDC, HMONITOR, MONITORINFO, MONITORINFOEXW, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::HiDpi::{
    AdjustWindowRectExForDpi, GetDpiForMonitor, SetProcessDpiAwarenessContext,
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetCapture, MapVirtualKeyW, ReleaseCapture, SetCapture, SetFocus, TrackMouseEvent,
    TRACKMOUSEEVENT, TME_LEAVE,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateIconIndirect, CreateWindowExW, DefWindowProcW, DestroyIcon, DestroyWindow,
    DispatchMessageW, GetClientRect, GetCursorPos, GetMessageW, GetWindowLongPtrW, HCURSOR,
    ICONINFO, IDC_ARROW, LoadCursorW, PostThreadMessageW, RegisterClassExW, SetCursor, SetCursorPos,
    SetWindowLongPtrW, SetWindowPos, ShowWindow, CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT,
    GWLP_USERDATA, MSG, SW_SHOW, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, WM_ACTIVATE, WM_APP,
    WM_CLOSE, WM_DESTROY, WM_ERASEBKGND, WM_KEYDOWN, WM_KEYUP, WM_KILLFOCUS, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEHWHEEL, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_RBUTTONDOWN,
    WM_RBUTTONUP, WM_SETCURSOR, WM_SETFOCUS, WM_SIZE, WM_SYSKEYDOWN, WM_SYSKEYUP, WM_XBUTTONDOWN,
    WM_XBUTTONUP, WNDCLASSEXW, WS_OVERLAPPEDWINDOW,
};

use crate::geometry::{self, Insets, Layout};
use crate::handle::Window;
use crate::screens::HostScreen;

/// The Win32 message when the mouse cursor leaves the window client area.
const WM_MOUSELEAVE: u32 = 0x02a3;

/// The Win32 message when a window loses mouse capture.
const WM_CAPTURECHANGED: u32 = 0x0215;

/// The UI thread's stack.
const STACK: usize = 1024 * 1024;

/// How long a look at the host's screens waits for the UI thread's first snapshot.
const FIRST_SNAPSHOT: Duration = Duration::from_secs(1);

/// The thread message posted to wake the UI thread so it drains its work queue.
const WM_RUN_JOBS: u32 = WM_APP + 1;

/// Work for the UI thread.
type Job = Box<dyn FnOnce() + Send>;

/// Where the UI thread is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No UI thread yet.
    Idle,
    /// The UI thread is starting.
    Starting,
    /// It runs: work posted wakes it.
    Running,
}

/// The work posted and not run yet, with where the UI thread is and its id (for `PostThreadMessage`).
struct Queue {
    phase: Phase,
    thread_id: u32,
    jobs: VecDeque<Job>,
}

/// The queue.
static QUEUE: Mutex<Queue> = Mutex::new(Queue {
    phase: Phase::Idle,
    thread_id: 0,
    jobs: VecDeque::new(),
});

/// The last snapshot of the host's screens, once taken.
static SCREENS: Mutex<Option<Vec<HostScreen>>> = Mutex::new(None);
/// Signalled when a snapshot is taken.
static SCREENS_TAKEN: Condvar = Condvar::new();

/// Whether the window class is registered (once, on the UI thread).
static CLASS: AtomicU32 = AtomicU32::new(0);

thread_local! {
    /// The windows this UI thread owns, by id.
    static WINDOWS: RefCell<HashMap<u64, Owned>> = RefCell::new(HashMap::new());
    /// Whether this thread is the UI thread.
    static ON_UI: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A window the UI thread owns.
struct Owned {
    /// The display's window, so events reach its surface and callbacks.
    window: std::sync::Arc<Window>,
    /// Its `HWND`, once created.
    hwnd: HWND,
    /// Set when the render thread has released the surface: the UI thread may destroy the window.
    #[allow(dead_code)]
    released: bool,
    /// This window's input decisions (pointer, wheel, keys, capture).
    input: super::input::Input,
    /// The window's current layout, for mapping the pointer into the guest's image.
    layout: Option<Layout>,
    /// The currently active Win32 HCURSOR created from the guest cursor glyph.
    cursor: Option<HCURSOR>,
    /// Whether the guest cursor is currently shown.
    cursor_visible: bool,
    /// Whether mouse tracking (`WM_MOUSELEAVE`) is active.
    tracking_mouse: bool,
}

impl Drop for Owned {
    fn drop(&mut self) {
        if let Some(cur) = self.cursor.take() {
            unsafe { DestroyIcon(cur as _) };
        }
    }
}

/// Creates a 32-bit alpha-blended Win32 HCURSOR from a guest RGBA glyph.
unsafe fn make_win32_cursor(glyph: &crate::handle::Glyph) -> Option<HCURSOR> {
    unsafe {
        let width = glyph.width as i32;
        let height = glyph.height as i32;
        if width <= 0 || height <= 0 || glyph.rgba.is_empty() {
            return None;
        }

        let mut bmi: BITMAPINFO = std::mem::zeroed();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = -height; // top-down DIB
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        let hdc = CreateCompatibleDC(ptr::null_mut());
        let mut bits: *mut core::ffi::c_void = ptr::null_mut();
        let hbm_color = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &mut bits, ptr::null_mut(), 0);
        DeleteDC(hdc);

        if hbm_color.is_null() || bits.is_null() {
            return None;
        }

        let pixel_count = (width * height) as usize;
        let src = &glyph.rgba;
        let dst = std::slice::from_raw_parts_mut(bits.cast::<u8>(), pixel_count * 4);

        // Win32 32-bit color DIB with per-pixel alpha expects premultiplied BGRA.
        for y in 0..height as usize {
            for x in 0..width as usize {
                let i = y * width as usize + x;
                if i * 4 + 3 < src.len() && i * 4 + 3 < dst.len() {
                    let r = src[i * 4] as u32;
                    let g = src[i * 4 + 1] as u32;
                    let b = src[i * 4 + 2] as u32;
                    let a = src[i * 4 + 3] as u32;
                    // Premultiply RGB by alpha for DWM alpha composition
                    dst[i * 4] = ((b * a + 127) / 255) as u8;
                    dst[i * 4 + 1] = ((g * a + 127) / 255) as u8;
                    dst[i * 4 + 2] = ((r * a + 127) / 255) as u8;
                    dst[i * 4 + 3] = a as u8;
                }
            }
        }

        // For 32-bit ARGB alpha-blended cursors, CreateIconIndirect expects an all-zero
        // 1-bpp monochrome AND mask with 16-bit WORD aligned row stride.
        let mask_stride = ((width + 15) / 16) * 2;
        let mask_bytes = vec![0u8; (mask_stride * height) as usize];
        let hbm_mask = CreateBitmap(width, height, 1, 1, mask_bytes.as_ptr().cast());

        if hbm_mask.is_null() {
            DeleteObject(hbm_color as _);
            return None;
        }

        // Hotspot must remain within the bitmap bounds
        let x_hotspot = (glyph.hot_spot.0 as u32).min(width.saturating_sub(1) as u32);
        let y_hotspot = (glyph.hot_spot.1 as u32).min(height.saturating_sub(1) as u32);

        let mut icon_info = ICONINFO {
            fIcon: 0, // cursor
            xHotspot: x_hotspot,
            yHotspot: y_hotspot,
            hbmMask: hbm_mask,
            hbmColor: hbm_color,
        };

        let hcursor = CreateIconIndirect(&mut icon_info);
        DeleteObject(hbm_mask as _);
        DeleteObject(hbm_color as _);

        if hcursor.is_null() {
            None
        } else {
            Some(hcursor as _)
        }
    }
}

/// The window class name, UTF-16, NUL-terminated.
fn class_name() -> Vec<u16> {
    "ReimsVgpuWindow\0".encode_utf16().collect()
}

/// Starts the UI thread, once.
pub(crate) fn start() {
    {
        let mut queue = QUEUE.lock();
        if queue.phase != Phase::Idle {
            return;
        }
        queue.phase = Phase::Starting;
    }
    spawn_named(&OsThreads::new(), "reims-vgpu.ui", STACK, Box::new(()), run);
}

/// Posts `job` to the UI thread, starting it where it has not started.
pub(crate) fn post(job: impl FnOnce() + Send + 'static) {
    start();
    let mut queue = QUEUE.lock();
    queue.jobs.push_back(Box::new(job));
    if queue.phase == Phase::Running {
        let tid = queue.thread_id;
        drop(queue);
        // SAFETY: a thread message to the UI thread's id; the message has no pointer payload.
        unsafe { PostThreadMessageW(tid, WM_RUN_JOBS, 0, 0) };
    }
}

/// The host's screens: the snapshot, waited for up to [`FIRST_SNAPSHOT`] where none is taken yet,
/// off the UI thread. Empty where none comes.
pub(crate) fn screens() -> Vec<HostScreen> {
    start();
    let mut screens = SCREENS.lock();
    if screens.is_none() && !ON_UI.with(std::cell::Cell::get) {
        SCREENS_TAKEN.wait_for(&mut screens, FIRST_SNAPSHOT);
    }
    screens.clone().unwrap_or_default()
}

/// The UI thread's body.
fn run() {
    ON_UI.with(|on| on.set(true));
    // Set Per-Monitor V2 DPI awareness so client coordinates and window bounds are never virtualized.
    unsafe {
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
    register_class();
    snapshot_screens();
    {
        let mut queue = QUEUE.lock();
        // SAFETY: a Win32 call with no arguments, returning this thread's id.
        queue.thread_id = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
        queue.phase = Phase::Running;
    }
    run_jobs();
    // The message loop: run queued work on the wake message, dispatch window messages otherwise.
    let mut msg: MSG = unsafe { std::mem::zeroed() };
    // SAFETY: a message loop reading this thread's queue; `GetMessageW` fills `msg`.
    while unsafe { GetMessageW(&mut msg, ptr::null_mut(), 0, 0) } > 0 {
        if msg.hwnd.is_null() && msg.message == WM_RUN_JOBS {
            run_jobs();
            continue;
        }
        // SAFETY: `msg` was filled by `GetMessageW`.
        unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// Runs the work posted, in order, until there is none.
fn run_jobs() {
    loop {
        let Some(job) = QUEUE.lock().jobs.pop_front() else {
            return;
        };
        job();
    }
}

/// Registers the window class once, on the UI thread.
fn register_class() {
    if CLASS.swap(1, Ordering::AcqRel) != 0 {
        return;
    }
    let name = class_name();
    // SAFETY: a Win32 call with null arg, returning the process module handle.
    let hinstance = unsafe { GetModuleHandleW(ptr::null()) };
    let default_cursor = unsafe { LoadCursorW(ptr::null_mut(), IDC_ARROW) };
    let class = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wndproc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: hinstance,
        hIcon: ptr::null_mut(),
        hCursor: default_cursor,
        hbrBackground: ptr::null_mut(),
        lpszMenuName: ptr::null(),
        lpszClassName: name.as_ptr(),
        hIconSm: ptr::null_mut(),
    };
    // SAFETY: `class` is a valid class descriptor naming `wndproc`; the strings outlive the call.
    let atom = unsafe { RegisterClassExW(&class) };
    if atom == 0 {
        // SAFETY: a Win32 call with no arguments.
        let err = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        log::warn!(target: "reims_vgpu_window", "RegisterClassExW failed (GetLastError={err})");
    }
}

/// Opens `window`'s Win32 window, lays it out, shows it and hands its handles to the render
/// thread. On the UI thread.
pub(crate) fn open_window(window: &std::sync::Arc<Window>) {
    let id = window.id();
    let name = class_name();
    let title: Vec<u16> = window
        .title()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    // SAFETY: the process module handle.
    let hinstance = unsafe { GetModuleHandleW(ptr::null()) };
    // SAFETY: a standard overlapped window of the registered class; strings outlive the call.
    let hwnd = unsafe {
        CreateWindowExW(
            0,
            name.as_ptr(),
            title.as_ptr(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            ptr::null_mut(),
            ptr::null_mut(),
            hinstance,
            ptr::null(),
        )
    };
    if hwnd.is_null() {
        // SAFETY: a Win32 call with no arguments.
        let err = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        log::warn!(target: "reims_vgpu_window", "CreateWindowExW failed (GetLastError={err}): window {id} does not open");
        return;
    }
    // Associate the id with the HWND so wndproc can find the Owned entry.
    // SAFETY: `hwnd` is a window this thread owns; the value is the window id.
    unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, id as isize) };
    WINDOWS.with(|w| {
        w.borrow_mut().insert(
            id,
            Owned {
                window: std::sync::Arc::clone(window),
                hwnd,
                released: false,
                input: super::input::Input::new(),
                layout: None,
                cursor: None,
                cursor_visible: true,
                tracking_mouse: false,
            },
        );
    });
    window.surface().set_target(super::render::Target { hwnd, hinstance });
    lay_out(id, hwnd);
    // SAFETY: `hwnd` is this thread's window.
    unsafe { ShowWindow(hwnd, SW_SHOW) };
    window.surface().set_visible(true);
}

/// Lays the window out for its client size and tells its surface. Caches the layout for input
/// mapping. The backing scale is the window's per-monitor DPI (96 dpi = 1×).
fn lay_out(id: u64, hwnd: HWND) {
    let mut rect = unsafe { std::mem::zeroed() };
    // SAFETY: `hwnd` is this thread's window; `rect` is a valid out-slot.
    if unsafe { GetClientRect(hwnd, &mut rect) } == 0 {
        return;
    }
    // SAFETY: `hwnd` is this thread's window; GetDpiForWindow needs no out-param.
    let dpi = unsafe { windows_sys::Win32::UI::HiDpi::GetDpiForWindow(hwnd) };
    let scale = if dpi == 0 { 1.0 } else { f64::from(dpi) / 96.0 };
    // Client rect is in physical pixels; the layout wants points (pixels / scale).
    let width = (f64::from(rect.right - rect.left) / scale).max(1.0);
    let height = (f64::from(rect.bottom - rect.top) / scale).max(1.0);
    let guest = WINDOWS.with(|w| {
        w.borrow()
            .get(&id)
            .and_then(|o| o.window.mode().or_else(|| o.window.offered()))
    });
    let layout: Layout = geometry::layout((width, height), scale, Insets::default(), guest);
    WINDOWS.with(|w| {
        if let Some(o) = w.borrow_mut().get_mut(&id) {
            o.layout = Some(layout);
            o.window.surface().set_layout(layout);
        }
    });
}

/// Posts to the UI thread that the guest of window `id` set a mode. Resizes the Win32 window
/// to match the guest mode dimensions and updates the layout.
pub(crate) fn guest_mode_changed(id: u64) {
    let target = WINDOWS.with(|w| {
        w.borrow()
            .get(&id)
            .map(|o| (o.hwnd, o.window.mode().or_else(|| o.window.offered())))
    });
    if let Some((hwnd, mode)) = target {
        if let Some((gw, gh)) = mode {
            let dpi = unsafe { windows_sys::Win32::UI::HiDpi::GetDpiForWindow(hwnd) };
            let mut rect = windows_sys::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: gw as i32,
                bottom: gh as i32,
            };
            unsafe {
                AdjustWindowRectExForDpi(
                    &mut rect,
                    WS_OVERLAPPEDWINDOW,
                    0,
                    0,
                    if dpi == 0 { 96 } else { dpi },
                );
                let width = rect.right - rect.left;
                let height = rect.bottom - rect.top;
                SetWindowPos(
                    hwnd,
                    ptr::null_mut(),
                    0,
                    0,
                    width,
                    height,
                    SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
                );
            }
        }
        lay_out(id, hwnd);
    }
}

/// A cursor refresh of window `id`. Creates or updates the Win32 HCURSOR from the guest glyph.
pub(crate) fn cursor_changed(id: u64) {
    WINDOWS.with(|w| {
        let mut b = w.borrow_mut();
        if let Some(o) = b.get_mut(&id) {
            o.window.take_cursor_refresh();
            let guest = o.window.guest();
            let old = o.cursor.take();
            if let Some(old_cur) = old {
                unsafe { DestroyIcon(old_cur as _) };
            }
            o.cursor_visible = guest.cursor_visible;
            if let Some(ref glyph) = guest.cursor {
                o.cursor = unsafe { make_win32_cursor(glyph) };
            }
            // Nudge cursor position so Windows triggers WM_SETCURSOR immediately
            unsafe {
                let mut pt = std::mem::zeroed();
                if GetCursorPos(&mut pt) != 0 {
                    SetCursorPos(pt.x, pt.y);
                }
            }
        }
    });
}

/// The window's close: signal the render thread; it releases the surface and posts [`released`],
/// which destroys the window (the module doc). Does NOT destroy the window here.
pub(crate) fn close(id: u64) {
    WINDOWS.with(|w| {
        if let Some(o) = w.borrow().get(&id) {
            o.window.surface().close();
        }
    });
}

/// The render thread has released window `id`'s surface: the window may now be destroyed.
pub(crate) fn released(id: u64) {
    let hwnd = WINDOWS.with(|w| {
        let mut b = w.borrow_mut();
        if let Some(o) = b.get_mut(&id) {
            o.released = true;
            Some(o.hwnd)
        } else {
            None
        }
    });
    if let Some(hwnd) = hwnd
        && !hwnd.is_null()
    {
        // SAFETY: `hwnd` is this thread's window, and its surface has been released, so nothing
        // Vulkan references it any longer.
        unsafe { DestroyWindow(hwnd) };
    }
}

/// Reads back a created window's `HWND` for the Stage 1 runtime test. The `WINDOWS` map is a UI-
/// thread-local, so this posts a job to the UI thread to read it and returns the result over a
/// channel.
#[cfg(test)]
pub(crate) fn test_hwnd(id: u64) -> Option<HWND> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel::<usize>();
    post(move || {
        let hwnd = WINDOWS.with(|w| w.borrow().get(&id).map(|o| o.hwnd));
        // Send as usize (HWND isn't Send); null (0) means "not present".
        let _ = tx.send(hwnd.map_or(0usize, |h| h as usize));
    });
    match rx.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(0) => None,
        Ok(v) => Some(v as HWND),
        Err(_) => None,
    }
}

/// `At` for window `id` at client `point` in points (y down), from its cached layout and the
/// guest's mode. A short borrow.
fn at_for(id: u64, point: Option<(f64, f64)>) -> super::input::At {
    WINDOWS.with(|w| {
        let b = w.borrow();
        let Some(o) = b.get(&id) else {
            return super::input::At::default();
        };
        super::input::At {
            point,
            image: o.layout.map(|l| l.image).unwrap_or_default(),
            guest: o.window.mode().or_else(|| o.window.offered()),
        }
    })
}

/// The current wall clock in nanoseconds, for the focus-click window.
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// The client point of a mouse message's `lParam`, in points, by the window's DPI.
fn mouse_point(hwnd: HWND, lparam: LPARAM) -> (f64, f64) {
    let x = (lparam & 0xffff) as i16 as f64;
    let y = ((lparam >> 16) & 0xffff) as i16 as f64;
    // SAFETY: `hwnd` is this thread's window.
    let dpi = unsafe { windows_sys::Win32::UI::HiDpi::GetDpiForWindow(hwnd) };
    let scale = if dpi == 0 { 1.0 } else { f64::from(dpi) / 96.0 };
    (x / scale, y / scale)
}

/// Runs `f` on window `id`'s input (mutating it), then sends what it produced to the host and
/// applies any capture change (the title's suffix). Borrows `WINDOWS` only to mutate the input and
/// read the window's config/handle.
fn with_input(id: u64, f: impl FnOnce(&mut super::input::Input) -> super::input::Reaction) {
    let (events, reaction, callbacks, title, suffix, hwnd) = WINDOWS.with(|w| {
        let mut b = w.borrow_mut();
        let Some(o) = b.get_mut(&id) else {
            return (
                Vec::new(),
                super::input::Reaction::default(),
                None,
                String::new(),
                "",
                std::ptr::null_mut(),
            );
        };
        let reaction = f(&mut o.input);
        let events = o.input.take();
        (
            events,
            reaction,
            o.window.callbacks(),
            o.window.title().to_owned(),
            o.input.title_suffix(),
            o.hwnd,
        )
    });
    if let Some(callbacks) = callbacks {
        super::input::send(&callbacks, events);
    }
    if reaction.capture_changed && !hwnd.is_null() {
        let full: Vec<u16> = format!("{title}{suffix}")
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `hwnd` is this thread's window; the title is a NUL-terminated UTF-16 string.
        unsafe { windows_sys::Win32::UI::WindowsAndMessaging::SetWindowTextW(hwnd, full.as_ptr()) };
    }
}

/// The window class's wndproc, on the UI thread.
///
/// # Safety
/// A Win32 window-procedure callback: `hwnd` is a window of this class and the parameters are the
/// message's.
unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // SAFETY: the user data is the window id this thread stored at creation (0 before that).
    let id = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as u64;
    match msg {
        WM_SIZE => {
            if id != 0 {
                lay_out(id, hwnd);
            }
            0
        }
        WM_MOUSEMOVE => {
            if id != 0 {
                WINDOWS.with(|w| {
                    if let Some(o) = w.borrow_mut().get_mut(&id) {
                        if !o.tracking_mouse {
                            let mut tme = TRACKMOUSEEVENT {
                                cbSize: std::mem::size_of::<TRACKMOUSEEVENT>() as u32,
                                dwFlags: TME_LEAVE,
                                hwndTrack: hwnd,
                                dwHoverTime: 0,
                            };
                            unsafe { TrackMouseEvent(&mut tme) };
                            o.tracking_mouse = true;
                        }
                    }
                });
                let p = Some(mouse_point(hwnd, lparam));
                let at = at_for(id, p);
                with_input(id, |input| {
                    input.moved(&at);
                    super::input::Reaction::default()
                });
            }
            0
        }
        WM_MOUSELEAVE => {
            if id != 0 {
                WINDOWS.with(|w| {
                    if let Some(o) = w.borrow_mut().get_mut(&id) {
                        o.tracking_mouse = false;
                    }
                });
                let captured = unsafe { GetCapture() } == hwnd;
                if !captured {
                    let arrow = unsafe { LoadCursorW(ptr::null_mut(), IDC_ARROW) };
                    unsafe { SetCursor(arrow) };
                }
            }
            0
        }
        WM_ACTIVATE => {
            if (wparam & 0xffff) != 0 && id != 0 {
                unsafe { SetFocus(hwnd) };
            }
            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }
        WM_ERASEBKGND => {
            // An application presenting via Vulkan swapchain handles client drawing;
            // returning non-zero informs Windows not to erase the client background.
            1
        }
        WM_SETCURSOR => {
            let hit = (lparam as u32) & 0xffff;
            let captured = unsafe { GetCapture() } == hwnd;
            if (hit == 1 /* HTCLIENT */ || captured) && id != 0 {
                let (hcur, visible, has_guest_cursor) = WINDOWS.with(|w| {
                    w.borrow()
                        .get(&id)
                        .map(|o| (o.cursor, o.cursor_visible, o.window.guest().cursor.is_some()))
                        .unwrap_or((None, true, false))
                });
                if has_guest_cursor && visible {
                    if let Some(hcur) = hcur {
                        unsafe { SetCursor(hcur) };
                        return 1;
                    }
                }
                let arrow = unsafe { LoadCursorW(ptr::null_mut(), IDC_ARROW) };
                unsafe { SetCursor(arrow) };
                return 1;
            }
            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }
        WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN => {
            if id != 0 {
                unsafe {
                    SetFocus(hwnd);
                    SetCapture(hwnd);
                };
                let bit = match msg {
                    WM_LBUTTONDOWN => super::input::button_bit_left(),
                    WM_RBUTTONDOWN => super::input::button_bit_right(),
                    _ => super::input::button_bit_middle(),
                };
                let at = at_for(id, Some(mouse_point(hwnd, lparam)));
                let ts = now_ns();
                with_input(id, |input| input.button_down(bit, ts, &at));
            }
            0
        }
        WM_LBUTTONUP | WM_RBUTTONUP | WM_MBUTTONUP => {
            if id != 0 {
                let bit = match msg {
                    WM_LBUTTONUP => super::input::button_bit_left(),
                    WM_RBUTTONUP => super::input::button_bit_right(),
                    _ => super::input::button_bit_middle(),
                };
                let at = at_for(id, Some(mouse_point(hwnd, lparam)));
                with_input(id, |input| {
                    input.button_up(bit, &at);
                    super::input::Reaction::default()
                });
                let any_buttons = WINDOWS.with(|w| {
                    w.borrow()
                        .get(&id)
                        .map(|o| o.input.has_buttons())
                        .unwrap_or(false)
                });
                if !any_buttons {
                    unsafe { ReleaseCapture() };
                }
            }
            0
        }
        WM_XBUTTONDOWN | WM_XBUTTONUP => {
            if id != 0 {
                let down = msg == WM_XBUTTONDOWN;
                if down {
                    unsafe {
                        SetFocus(hwnd);
                        SetCapture(hwnd);
                    };
                }
                let which = ((wparam >> 16) & 0xffff) as u16;
                let bit = super::input::button_bit_x(which);
                let at = at_for(id, Some(mouse_point(hwnd, lparam)));
                let ts = now_ns();
                with_input(id, |input| {
                    if down {
                        input.button_down(bit, ts, &at)
                    } else {
                        input.button_up(bit, &at);
                        super::input::Reaction::default()
                    }
                });
                if !down {
                    let any_buttons = WINDOWS.with(|w| {
                        w.borrow()
                            .get(&id)
                            .map(|o| o.input.has_buttons())
                            .unwrap_or(false)
                    });
                    if !any_buttons {
                        unsafe { ReleaseCapture() };
                    }
                }
            }
            1 // XBUTTON messages expect TRUE
        }
        WM_CAPTURECHANGED => {
            if id != 0 {
                with_input(id, |input| {
                    input.capture_lost();
                    super::input::Reaction::default()
                });
            }
            0
        }
        WM_MOUSEWHEEL | WM_MOUSEHWHEEL => {
            if id != 0 {
                // wheel lParam is in SCREEN coords; convert to client first.
                let mut pt = windows_sys::Win32::Foundation::POINT {
                    x: (lparam & 0xffff) as i16 as i32,
                    y: ((lparam >> 16) & 0xffff) as i16 as i32,
                };
                // SAFETY: `hwnd` is this thread's window; `pt` is a valid in/out point.
                unsafe { windows_sys::Win32::Graphics::Gdi::ScreenToClient(hwnd, &mut pt) };
                let dpi = unsafe { windows_sys::Win32::UI::HiDpi::GetDpiForWindow(hwnd) };
                let scale = if dpi == 0 { 1.0 } else { f64::from(dpi) / 96.0 };
                let point = Some((f64::from(pt.x) / scale, f64::from(pt.y) / scale));
                let delta = ((wparam >> 16) & 0xffff) as i16 as f64 / 120.0; // WHEEL_DELTA
                let wheel = if msg == WM_MOUSEWHEEL {
                    super::input::Wheel {
                        notches: (0, delta.round() as i32),
                        lines: (0.0, delta as f32),
                    }
                } else {
                    super::input::Wheel {
                        notches: (delta.round() as i32, 0),
                        lines: (delta as f32, 0.0),
                    }
                };
                let at = at_for(id, point);
                with_input(id, |input| {
                    input.wheel(wheel, &at);
                    super::input::Reaction::default()
                });
            }
            0
        }
        WM_KEYDOWN | WM_SYSKEYDOWN | WM_KEYUP | WM_SYSKEYUP => {
            if id != 0 {
                let mut scancode = ((lparam >> 16) & 0xff) as u32;
                if scancode == 0 {
                    scancode = unsafe { MapVirtualKeyW(wparam as u32, 0) };
                }
                let extended = (lparam >> 24) & 1 != 0;
                let down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
                let repeat = down && (lparam >> 30) & 1 != 0; // previous key-state bit
                if let Some(usage) = super::keymap::usb_usage(scancode, extended) {
                    with_input(id, |input| input.key(usage, down, repeat));
                }
            }
            0
        }
        WM_SETFOCUS => {
            if id != 0 {
                let ts = now_ns();
                with_input(id, |input| {
                    input.focus_gained(ts);
                    super::input::Reaction::default()
                });
            }
            0
        }
        WM_KILLFOCUS => {
            if id != 0 {
                unsafe { ReleaseCapture() };
                with_input(id, |input| input.focus_lost());
            }
            0
        }
        WM_CLOSE => {
            // Closing a window powers the display off: lift any held input, call the host's close
            // callback, and signal the render thread; destruction is deferred to `released` (the
            // module doc). Do not DestroyWindow here.
            if id != 0 {
                with_input(id, |input| {
                    input.closing();
                    super::input::Reaction::default()
                });
                let close_cb = WINDOWS.with(|w| {
                    w.borrow()
                        .get(&id)
                        .and_then(|o| o.window.callbacks())
                        .and_then(|c| c.close)
                });
                if let Some(close_cb) = close_cb {
                    close_cb();
                }
                close(id);
            }
            0
        }
        WM_DESTROY => {
            // The window is gone: drop the owned entry.
            if id != 0 {
                WINDOWS.with(|w| {
                    w.borrow_mut().remove(&id);
                });
            }
            0
        }
        // SAFETY: default handling for every other message.
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

/// Takes a snapshot of the host's screens.
fn snapshot_screens() {
    let screens = list_screens();
    log::debug!(target: "reims_vgpu_window", "host screens: {screens:?}");
    *SCREENS.lock() = Some(screens);
    SCREENS_TAKEN.notify_all();
}

/// The host's screens as Win32 lists them, through `EnumDisplayMonitors`.
fn list_screens() -> Vec<HostScreen> {
    let mut out: Vec<HostScreen> = Vec::new();
    // SAFETY: `monitor_proc` matches the callback signature; `&mut out` is passed as the lparam
    // and only touched on this thread during the synchronous enumeration.
    unsafe {
        EnumDisplayMonitors(
            ptr::null_mut(),
            ptr::null(),
            Some(monitor_proc),
            &mut out as *mut Vec<HostScreen> as LPARAM,
        );
    }
    out
}

/// `EnumDisplayMonitors` callback: appends each monitor as a [`HostScreen`].
///
/// # Safety
/// A Win32 monitor-enum callback; `lparam` is the `&mut Vec<HostScreen>` passed to
/// `EnumDisplayMonitors`, live for the synchronous enumeration.
unsafe extern "system" fn monitor_proc(
    monitor: HMONITOR,
    _hdc: HDC,
    _rect: *mut windows_sys::Win32::Foundation::RECT,
    lparam: LPARAM,
) -> windows_sys::Win32::Foundation::BOOL {
    // SAFETY: the lparam is the out-vec from `list_screens`, live for this call.
    let out = unsafe { &mut *(lparam as *mut Vec<HostScreen>) };
    let mut info: MONITORINFOEXW = unsafe { std::mem::zeroed() };
    info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
    // SAFETY: `monitor` is from the enumeration; `info` is a sized MONITORINFOEXW.
    let ok = unsafe {
        GetMonitorInfoW(monitor, &mut info as *mut MONITORINFOEXW as *mut MONITORINFO)
    };
    if ok == 0 {
        return 1;
    }
    let r = info.monitorInfo.rcMonitor;
    let width = (r.right - r.left).max(0) as u32;
    let height = (r.bottom - r.top).max(0) as u32;
    // Per-monitor DPI → scale (96 dpi = 1.0). MDT_EFFECTIVE_DPI = 0.
    let mut dpi_x: u32 = 96;
    let mut dpi_y: u32 = 96;
    // SAFETY: `monitor` is valid; the out-slots are valid; 0 is MDT_EFFECTIVE_DPI.
    let _ = unsafe { GetDpiForMonitor(monitor, 0, &mut dpi_x, &mut dpi_y) };
    let scale = (f64::from(dpi_x) / 96.0).max(1.0);
    // The monitor's current refresh rate via EnumDisplaySettingsW on its device name.
    let mut dm: DEVMODEW = unsafe { std::mem::zeroed() };
    dm.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
    // SAFETY: szDevice is a NUL-terminated device name; `dm` is a sized DEVMODEW out-slot.
    let got = unsafe {
        EnumDisplaySettingsW(info.szDevice.as_ptr(), ENUM_CURRENT_SETTINGS, &mut dm)
    };
    let max_refresh_hz = if got != 0 && dm.dmDisplayFrequency > 1 {
        f64::from(dm.dmDisplayFrequency)
    } else {
        60.0
    };
    out.push(HostScreen {
        id: out.len() as u32,
        pixels: (width, height),
        scale,
        max_refresh_hz,
    });
    1
}
