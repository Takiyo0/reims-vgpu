//! What a window's render thread presents through where the display's sink presents the frames
//! itself, over a Vulkan swapchain of the window's surface: on Linux always, on macOS on the
//! Vulkan platform (`FrameSink::presents`). The sink makes a [`Presenter`] over the window
//! ([`VulkanTarget`]: SDL's window on Linux, the window's `CAMetalLayer` on macOS), and the render
//! thread has it draw each frame ([`Presenting`]).
//!
//! **The presenter's life.** At its first draw the render thread asks the sink for a presenter;
//! without one (no swapchain on this host) it never draws, and the window stays empty. Each draw
//! has the presenter draw the current frame into its next image and present it
//! ([`Presenter::draw`]): taking the image is the pacing, at the screen's own rate. Where no image
//! came in time the render thread tries again after [`RETRY`]. A presenter that can draw no more
//! ([`Drawn::Ended`]: its surface is lost) is let go, and the next draw asks the sink for a new
//! one, up to [`RENEWALS`] times in a row without a frame presented; the render thread then waits
//! [`PAUSE`] before it starts again, so a window never stays frozen for good (a driver that needs
//! a while after a minimize, a surface that comes back).

use std::time::Duration;

use ash::vk;

use crate::handle::PixelRect;

/// How long the render thread waits before it draws again where no image came in time: about a
/// refresh at 120 Hz.
pub(crate) const RETRY: Duration = Duration::from_millis(8);

/// How many presenters in a row the render thread makes again after one ends, with no frame
/// presented between: past that, it waits [`PAUSE`] first.
const RENEWALS: u32 = 3;

/// How long the render thread waits before it makes presenters again, after [`RENEWALS`] ended in
/// a row or the sink made none.
const PAUSE: Duration = Duration::from_secs(2);

/// What a draw did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drawn {
    /// The frame was drawn and presented.
    Presented,
    /// Nothing was drawn: the display has ended, or refused the encode. The image taken is kept
    /// for the next draw.
    Nothing,
    /// No image came in time: the draw is tried again.
    Retry,
    /// The presenter can draw no more: the render thread asks the sink for a new one at its next
    /// draw (the module doc).
    Ended,
}

/// What a window's render thread presents through: the sink's, over the window's swapchain.
pub trait Presenter: Send {
    /// Draws the display's current frame into the next image of `drawable` pixels (width,
    /// height), over `region` of it, clearing the rest, and presents it. Blocks only for the next
    /// image, up to a few frames.
    fn draw(&mut self, drawable: (u64, u64), region: PixelRect) -> Drawn;
}

/// What makes a window's Vulkan surface, on its render thread: on Linux SDL's window, on macOS
/// the window's `CAMetalLayer`.
pub trait VulkanTarget {
    /// The window's surface, made on `instance` (whose commands `get_instance_proc_addr`, the
    /// loader's the instance was made with, looks up), which the caller owns from then on (and
    /// destroys before its instance), or `None` where none is made.
    fn create_surface(
        &self,
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        instance: vk::Instance,
    ) -> Option<vk::SurfaceKHR>;
}

/// A render thread's presenter, made of the sink at the first draw and made again after one ends,
/// up to [`RENEWALS`] times in a row (the module doc).
#[derive(Default)]
pub(crate) struct Presenting {
    /// The presenter, once made.
    presenter: Option<Box<dyn Presenter>>,
    /// No presenter is made before this: the sink made none, or the presenters ended too many
    /// times in a row.
    paused_until: Option<std::time::Instant>,
    /// A pause was logged since the last frame presented: the next is not.
    pause_logged: bool,
    /// The presenters made again since the last frame presented.
    renewals: Renewals,
}

impl Presenting {
    /// Whether the next draw needs a presenter made first ([`Presenting::make`]).
    pub(crate) fn needs_presenter(&self) -> bool {
        self.presenter.is_none() && !self.paused()
    }

    /// Whether presenters are not made yet ([`PAUSE`]).
    fn paused(&self) -> bool {
        self.paused_until
            .is_some_and(|until| std::time::Instant::now() < until)
    }

    /// Makes no presenter for [`PAUSE`], logging `why` once a streak.
    fn pause(&mut self, why: &str) {
        if !self.pause_logged {
            log::warn!(
                target: "reims_vgpu_window",
                "{why}: the window tries again every {} s",
                PAUSE.as_secs()
            );
            self.pause_logged = true;
        }
        self.presenter = None;
        self.renewals = Renewals::default();
        self.paused_until = Some(std::time::Instant::now() + PAUSE);
    }

    /// Makes the presenter with `make`, the sink's: where it makes none, the window stays as it
    /// is.
    pub(crate) fn make(&mut self, make: impl FnOnce() -> Option<Box<dyn Presenter>>) {
        self.presenter = make();
        if self.presenter.is_none() {
            self.pause("the window has no presenter");
        }
    }

    /// Has the presenter draw the current frame into an image of `drawable` pixels over
    /// `region`: whether to draw again after [`RETRY`] (no image came in time, or presenters are
    /// paused). Without a presenter, a draw again only while paused.
    pub(crate) fn draw(&mut self, drawable: (u64, u64), region: PixelRect) -> bool {
        let Some(presenter) = self.presenter.as_mut() else {
            return self.paused();
        };
        let drawn = presenter.draw(drawable, region);
        match drawn {
            Drawn::Ended => self.presenter = None,
            Drawn::Presented => {
                if self.pause_logged {
                    log::warn!(target: "reims_vgpu_window", "the window presents again");
                }
                self.pause_logged = false;
                self.paused_until = None;
            }
            Drawn::Nothing | Drawn::Retry => {}
        }
        if !self.renewals.after(drawn) {
            self.pause(&format!(
                "the window's presenter ended {RENEWALS} times in a row"
            ));
            return true;
        }
        // An ended presenter is made again at the next try, not at the next guest frame.
        drawn == Drawn::Retry || drawn == Drawn::Ended
    }
}

/// The presenters a render thread has made again since its last frame presented.
#[derive(Default)]
struct Renewals(u32);

impl Renewals {
    /// Counts what a draw did: whether the render thread may make a new presenter, where it
    /// needs one.
    fn after(&mut self, drawn: Drawn) -> bool {
        match drawn {
            Drawn::Presented => self.0 = 0,
            Drawn::Ended => self.0 = self.0.saturating_add(1),
            Drawn::Nothing | Drawn::Retry => {}
        }
        self.0 <= RENEWALS
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    /// A presenter that answers what it is told, in order, then `Presented`.
    struct Scripted(VecDeque<Drawn>);

    impl Presenter for Scripted {
        fn draw(&mut self, _drawable: (u64, u64), _region: PixelRect) -> Drawn {
            self.0.pop_front().unwrap_or(Drawn::Presented)
        }
    }

    /// A presenter answering `drawn` in order.
    fn scripted(drawn: &[Drawn]) -> Option<Box<dyn Presenter>> {
        Some(Box::new(Scripted(drawn.iter().copied().collect())))
    }

    /// A region.
    const REGION: PixelRect = PixelRect {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };

    /// A presenter that ends is made again, up to [`RENEWALS`] times in a row, and a frame
    /// presented starts the count again
    #[test]
    fn an_ended_presenter_is_made_again_a_few_times() {
        let mut r = Renewals::default();
        for _ in 0..RENEWALS {
            assert!(r.after(Drawn::Ended));
            assert!(r.after(Drawn::Retry) && r.after(Drawn::Nothing));
        }
        assert!(r.after(Drawn::Presented));
        for _ in 0..RENEWALS {
            assert!(r.after(Drawn::Ended));
        }
        assert!(!r.after(Drawn::Ended), "past the renewals: a pause");
    }

    /// The presenter is made at the first draw and kept; a draw with no image asks for a retry;
    /// one that ends is made again at the next try; a sink that makes none pauses presenters
    /// for [`PAUSE`], with retries asked meanwhile
    #[test]
    fn presenting_makes_keeps_and_renews_the_presenter() {
        let mut p = Presenting::default();
        assert!(p.needs_presenter());
        p.make(|| scripted(&[Drawn::Retry, Drawn::Presented, Drawn::Ended]));
        assert!(!p.needs_presenter());
        assert!(p.draw((4, 4), REGION), "no image: a retry");
        assert!(!p.draw((4, 4), REGION));
        assert!(p.draw((4, 4), REGION), "ended: tried again");
        assert!(p.needs_presenter(), "an ended presenter is made again");
        p.make(|| None);
        assert!(!p.needs_presenter(), "no presenter: paused");
        assert!(p.draw((4, 4), REGION), "paused: tried again later");
        p.paused_until = Some(std::time::Instant::now());
        assert!(
            p.needs_presenter(),
            "after the pause, a presenter is made again"
        );
    }

    /// Presenters that end too many times in a row pause the window, then it tries again
    #[test]
    fn presenters_ending_in_a_row_pause_then_try_again() {
        let mut p = Presenting::default();
        for _ in 0..=RENEWALS {
            assert!(p.needs_presenter());
            p.make(|| scripted(&[Drawn::Ended]));
            assert!(p.draw((4, 4), REGION));
        }
        assert!(!p.needs_presenter(), "paused");
        p.paused_until = Some(std::time::Instant::now());
        assert!(p.needs_presenter(), "tried again after the pause");
        p.make(|| scripted(&[Drawn::Presented]));
        assert!(!p.draw((4, 4), REGION));
        assert!(p.paused_until.is_none() && !p.pause_logged);
    }
}
