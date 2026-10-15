//! A display's own window ([`DisplayConfig::window`](crate::DisplayConfig::window)): the tee that
//! tells the window what the guest does, the link that closes it with the display's last
//! reference, the modes and density a display matching its host screen offers, and the sink that
//! draws the display's current frame into the window's drawable: on macOS, `metal::MetalSink` on
//! the product's platform, and on the Vulkan platform (feature `vulkan`) `vulkan::VulkanSink`,
//! which has Metal copy each frame the runtime draws into the drawable; on Linux, on the Vulkan
//! platform, `vulkan::SwapchainSink`, which draws each frame into the window's swapchain.
//!
//! **What the guest sees.** A window display has all five display callbacks, whatever the host
//! gives ([`tee`]): the guest reads a cursor glyph size limit of 256 and both cursor handler
//! bits. Each callback tells the window first, then calls the host's where the host gave one.
//! Without a window, nothing here runs and the display is as the config says.
//!
//! **Lifetime: a window frame is a frame.** Nothing on the window's side holds the display: the
//! tee's closures hold the window, and the sink holds the display weakly. A draw upgrades that
//! reference for its encode and commit alone, and the encode's completion handler holds the
//! display until the frame completes, as it does for a [`Frame`](crate::Frame). So a release
//! that races a window frame ends the display on the window's render thread or on the platform's
//! completion thread (Metal's, or the runtime's), as a frame's would. The display's last
//! reference drops its [`WindowLink`] before its own reference to the display, which closes the
//! window: from then on no window callback of the host's runs, but one already running on the
//! window's UI thread (the main thread on macOS, the library's `reims-vgpu.ui` on Linux). It
//! waits for nothing: not for the UI thread, the render thread or the GPU.

use std::any::{Any, TypeId};
use std::sync::{Arc, Weak};

use reims_vgpu_core::display::Display as CoreDisplay;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_window::policy::{self, Density, OfferedMode};
use reims_vgpu_window::{FrameSink, HostScreen, PixelRect, Window};

use crate::display::{CursorGlyph, DisplayCallbacks, no_catch};
use crate::entry;
use crate::stats::{self, DisplayStats};

/// The host screen the window of the display on `port` opens on: on the product's platform and
/// on the Vulkan platform (feature `vulkan`), the one `reims_vgpu_window::screens` picks from
/// the host's screens; on the testkit's, `crate::platform::FAKE_HOST_SCREEN`; on any other,
/// none, and a display matching its host screen offers what its config says.
pub(crate) fn host_screen<P: Platform>(port: u64) -> Option<HostScreen> {
    if on_host_screens::<P>() {
        return reims_vgpu_window::screens::screen_for_port(port);
    }
    #[cfg(any(test, feature = "testkit"))]
    if TypeId::of::<P>() == TypeId::of::<reims_vgpu_core::testkit::FakePlatform>() {
        return Some(crate::platform::FAKE_HOST_SCREEN);
    }
    None
}

/// Whether a window display on `P` opens on the host's own screens ([`host_screen`]): on the
/// product's platform and on the Vulkan platform (feature `vulkan`).
fn on_host_screens<P: Platform>() -> bool {
    let platform = TypeId::of::<P>();
    #[cfg(feature = "vulkan")]
    if platform == TypeId::of::<reims_vgpu_vulkan::VulkanPlatform>() {
        return true;
    }
    platform == TypeId::of::<crate::DefaultPlatform>()
}

/// What draws a window display's current frame into its window, over `display`, held weakly, in
/// command buffers of `frame_queue`, the process's frame queue for the display's device (a Metal
/// device, or the runtime's on Vulkan): on macOS, on the product's platform, a
/// `metal::MetalSink`, and on the Vulkan platform (feature `vulkan`) the sink of `vulkan`, which
/// draws each frame into the window's swapchain, on either OS. `None` on any other platform (the
/// testkit's among them), or without a frame queue: nothing draws, and the window never opens.
pub(crate) fn window_sink<P: Platform>(
    platform: &Arc<P>,
    display: Weak<CoreDisplay<P>>,
    frame_queue: Option<P::CommandQueue>,
) -> Option<Arc<dyn FrameSink>> {
    let display: Box<dyn Any> = Box::new(display);
    let frame_queue: Box<dyn Any> = Box::new(frame_queue);
    #[cfg(target_os = "macos")]
    let (display, frame_queue) = match metal::sink(display, frame_queue) {
        Ok(sink) => return sink,
        Err(unmatched) => unmatched,
    };
    #[cfg(feature = "vulkan")]
    let display = match display.downcast::<Weak<CoreDisplay<reims_vgpu_vulkan::VulkanPlatform>>>() {
        Ok(display) => {
            use reims_vgpu_vulkan::VulkanPlatform;
            let platform: Arc<dyn Any + Send + Sync> = Arc::clone(platform) as _;
            let platform = platform.downcast::<VulkanPlatform>().ok()?;
            let frame_queue = frame_queue
                .downcast::<Option<<VulkanPlatform as Platform>::CommandQueue>>()
                .ok()?;
            return vulkan::sink(platform, *display, (*frame_queue)?);
        }
        Err(display) => display,
    };
    let _ = (platform, display, frame_queue);
    None
}

/// What a display matching its host screen offers the guest for `screen`: with a chosen `mode`,
/// that mode first and the standard sizes within it, at `ppi` (`WindowConfig::mode_ppi`) or else
/// a 1× density; without one, the policy's modes for the screen, the first the preferred, at the
/// screen's density. The density is that of the first mode. `None` when the policy offers no
/// mode: the display then offers what its config says. The modes run at `refresh_hz`
/// (`REIMS_VGPU_WINDOW_REFRESH`) where given, else at the screen's highest rate.
pub(crate) fn host_matched(
    screen: &HostScreen,
    mode: Option<(u16, u16)>,
    ppi: Option<f64>,
    refresh_hz: Option<f64>,
) -> Option<(Vec<OfferedMode>, Density)> {
    let modes = match mode {
        Some(mode) => policy::modes_up_to(mode, screen, refresh_hz),
        None => policy::offered_modes(screen, refresh_hz),
    };
    let first = modes.first()?;
    let first = (first.width, first.height);
    let density = match mode {
        Some(_) => policy::density_chosen(first, ppi),
        None => policy::density(screen, first),
    };
    Some((modes, density))
}

/// What a display's callbacks tell its window: [`Window`]'s calls, behind a trait so the tee is
/// tested over a recorder.
pub(crate) trait GuestEvents: Send + Sync + 'static {
    /// The guest set a mode.
    fn guest_mode(&self, width: u16, height: u16);
    /// The guest presented a frame.
    fn new_frame(&self);
    /// The guest set a cursor glyph.
    fn cursor_glyph(&self, glyph: &CursorGlyph<'_>);
    /// The guest showed (`true`) or hid its cursor.
    fn cursor_show(&self, show: bool);
}

impl GuestEvents for Window {
    fn guest_mode(&self, width: u16, height: u16) {
        Window::guest_mode(self, width, height);
    }

    fn new_frame(&self) {
        Window::new_frame(self);
    }

    fn cursor_glyph(&self, glyph: &CursorGlyph<'_>) {
        Window::cursor_glyph(self, glyph.width, glyph.height, glyph.hot_spot, glyph.rgba);
    }

    fn cursor_show(&self, show: bool) {
        Window::cursor_show(self, show);
    }
}

/// A window display's callbacks: each of the first four tells `window`, then calls the host's
/// in `host` where it gave one; `cursor_move` is the host's alone, and does nothing without it.
/// All five are present, which the guest reads (module doc). The closures hold the window and
/// the host's closures, never the display.
pub(crate) fn tee<W: GuestEvents>(window: &Arc<W>, host: DisplayCallbacks) -> DisplayCallbacks {
    let DisplayCallbacks {
        mode_change,
        new_frame,
        cursor_glyph,
        cursor_show,
        cursor_move,
    } = host;
    let w = window.clone();
    let mode_change: crate::ModeChangeFn = Arc::new(move |width, height, format| {
        w.guest_mode(width, height);
        if let Some(f) = &mode_change {
            f(width, height, format);
        }
    });
    let w = window.clone();
    let new_frame: crate::NewFrameFn = Arc::new(move || {
        w.new_frame();
        if let Some(f) = &new_frame {
            f();
        }
    });
    let w = window.clone();
    let cursor_glyph: crate::CursorGlyphFn = Arc::new(move |glyph: &CursorGlyph<'_>| {
        w.cursor_glyph(glyph);
        if let Some(f) = &cursor_glyph {
            f(glyph);
        }
    });
    let w = window.clone();
    let cursor_show: crate::CursorShowFn = Arc::new(move |show| {
        w.cursor_show(show != 0);
        if let Some(f) = &cursor_show {
            f(show);
        }
    });
    let cursor_move: crate::CursorMoveFn = cursor_move.unwrap_or_else(|| Arc::new(|| {}));
    DisplayCallbacks {
        mode_change: Some(mode_change),
        new_frame: Some(new_frame),
        cursor_glyph: Some(cursor_glyph),
        cursor_show: Some(cursor_show),
        cursor_move: Some(cursor_move),
    }
}

/// A display's window, as every reference to the display shares it: its drop, at the display's
/// last reference, is the window's end ([`Window::close`]).
pub(crate) struct WindowLink {
    window: Arc<Window>,
}

impl WindowLink {
    /// The link to `window`, which closes it when it drops.
    pub(crate) fn new(window: Arc<Window>) -> WindowLink {
        WindowLink { window }
    }

    /// The window.
    pub(crate) fn window(&self) -> &Arc<Window> {
        &self.window
    }
}

impl Drop for WindowLink {
    /// Closes the window, as an entry: it waits for nothing, and what the window lets go of here
    /// (the host's window callbacks, the sink) is released on this thread.
    fn drop(&mut self) {
        entry::run(|_| self.window.close());
    }
}

/// What a window sink draws from: the display, held weakly, and the process's frame queue for its
/// device, whose command buffers the sink's frames are encoded in. Neither keeps the display
/// alive (module doc). Every sink draws through it, each frame a [`WindowFrame`], and each frame
/// it commits counts as a window frame in the display's statistics, where they are on
/// ([`crate::stats`]).
#[cfg_attr(
    not(any(target_os = "macos", feature = "vulkan")),
    expect(dead_code, reason = "only the Metal and Vulkan platforms' sinks draw")
)]
pub(crate) struct SinkDisplay<P: Platform> {
    /// The display, held only for a draw.
    display: Weak<CoreDisplay<P>>,
    /// The frame queue.
    queue: P::CommandQueue,
    /// The display's statistics, where they are on.
    stats: Option<Arc<DisplayStats>>,
}

/// One window frame, as a sink draws it: the steps in which the sinks differ, which
/// [`SinkDisplay::encode`] runs in order. The frame is dropped after the commit, or where the
/// draw ends without one.
#[cfg_attr(
    not(any(target_os = "macos", feature = "vulkan")),
    expect(dead_code, reason = "only the Metal and Vulkan platforms' sinks draw")
)]
pub(crate) trait WindowFrame<P: Platform> {
    /// The texture the frame is encoded into, asked once the command buffer is made; `None`
    /// draws nothing, and the command buffer goes uncommitted.
    fn texture(&mut self) -> Option<&P::Texture>;

    /// The frame is encoded in `command_buffer`, which is committed next.
    fn encoded(&mut self, command_buffer: &P::CommandBuffer);
}

#[cfg_attr(
    not(any(target_os = "macos", feature = "vulkan")),
    expect(dead_code, reason = "only the Metal and Vulkan platforms' sinks draw")
)]
impl<P: Platform> SinkDisplay<P> {
    /// `display`, drawn in command buffers of `queue`, the frame queue of its device.
    pub(crate) fn new(display: Weak<CoreDisplay<P>>, queue: P::CommandQueue) -> SinkDisplay<P> {
        SinkDisplay {
            stats: stats::of_display(&display),
            display,
            queue,
        }
    }

    /// The display's statistics, where they are on: a sink that measures its frames' GPU time
    /// reports it there.
    #[cfg_attr(
        not(feature = "vulkan"),
        expect(
            dead_code,
            reason = "only the Vulkan platform's sink measures its frames' GPU time"
        )
    )]
    pub(crate) fn stats(&self) -> Option<&Arc<DisplayStats>> {
        self.stats.as_ref()
    }

    /// Draws `frame`: the display's current frame over `region` into its texture, in a new
    /// command buffer of the frame queue, which is committed after `frame` has seen it. In order:
    /// the display's upgrade, the command buffer, [`WindowFrame::texture`], the encode,
    /// [`WindowFrame::encoded`] and the commit: `true` once committed. `false`, with nothing
    /// committed, when the display has ended, when the frame queue gives no command buffer, when
    /// the frame has no texture, or when the display refuses the encode (a texture on another
    /// device, or one it cannot render into). A command buffer not committed goes with nothing
    /// encoded in it. Waits for nothing.
    ///
    /// The display is held from its upgrade to the commit, and the encode's completion handler
    /// holds it on until the frame completes. So when the host has released the display in the
    /// meantime, this draw's release, inside this entry, may be the display's end.
    pub(crate) fn encode(&self, region: PixelRect, frame: impl WindowFrame<P>) -> bool {
        entry::run(|_| {
            let Some(display) = self.display.upgrade() else {
                return false;
            };
            let committed = self.commit_frame(&display, region, frame);
            drop(display);
            committed
        })
    }

    /// [`SinkDisplay::encode`]'s work, with the display held.
    fn commit_frame(
        &self,
        display: &CoreDisplay<P>,
        region: PixelRect,
        mut frame: impl WindowFrame<P>,
    ) -> bool {
        let Some(command_buffer) = no_catch(self.queue.command_buffer(OnFault::Die)) else {
            return false;
        };
        let Some(texture) = frame.texture() else {
            return false;
        };
        let region = MtlRegion::new(
            MtlOrigin::new(region.x, region.y, 0),
            MtlSize::new(region.width, region.height, 1),
        );
        if !no_catch(display.encode_current_frame(&command_buffer, texture, region)) {
            return false;
        }
        frame.encoded(&command_buffer);
        no_catch(command_buffer.commit(OnFault::Die));
        if let Some(stats) = &self.stats {
            stats.window_present();
        }
        true
    }

    /// DIAG(clear-test): draws a single clear pass of `color` (no guest frame) into the window
    /// frame's texture, in a new command buffer, then signals and commits exactly as
    /// [`SinkDisplay::encode`] does — the window present bypass of advisor experiment 3. `true`
    /// once committed; `false` with nothing committed on the same conditions as `encode`.
    pub(crate) fn clear_test(
        &self,
        color: reims_vgpu_core::platform::types::ClearColor,
        mut frame: impl WindowFrame<P>,
    ) -> bool {
        entry::run(|_| {
            let Some(display) = self.display.upgrade() else {
                return false;
            };
            let Some(command_buffer) = no_catch(self.queue.command_buffer(OnFault::Die)) else {
                return false;
            };
            let Some(texture) = frame.texture() else {
                return false;
            };
            if !no_catch(display.clear_test_frame(&command_buffer, texture, color)) {
                return false;
            }
            frame.encoded(&command_buffer);
            no_catch(command_buffer.commit(OnFault::Die));
            drop(display);
            true
        })
    }
}

#[cfg(target_os = "macos")]
mod metal;
#[cfg(test)]
mod tests;
#[cfg(feature = "vulkan")]
mod vulkan;
