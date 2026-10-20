//! A window's input through Win32: its pointer, wheel and key messages to the host's callbacks
//! ([`crate::config::WindowCallbacks`]), on the UI thread.
//!
//! The decision logic ([`Input`], [`ToHost`], [`send`], [`At`]) is the same as the Linux
//! backend's (`native/linux/input.rs`) and rests on the same OS-agnostic modules
//! ([`crate::keys`], [`crate::capture`], [`crate::geometry`], [`crate::scroll`]); only the event
//! source differs — Win32 `WM_*` messages here, SDL events there. The guest contract is identical:
//! the pointer in guest pixels once the guest has a mode, held at the image's nearest edge; a
//! click in the image captures the keyboard (Ctrl+Alt+Win releases); the wheel in the device's
//! own direction, smooth where the host takes it; keys by position with host repeats dropped; and
//! no stuck keys on focus loss, release or close.

use crate::capture::Capture;
use crate::config::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, ScrollEvent,
    WindowCallbacks,
};
use crate::geometry::{self, Rect};
use crate::keys::{KeyEvent, KeyState};
use crate::scroll::POINTS_PER_LINE;

/// What the title ends with while captured.
const CAPTURED_SUFFIX: &str = " \u{2014} input captured (press Ctrl+Alt+Win to release)";

/// What the window knows when an event reaches it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct At {
    /// The event's point in the window, in points (y down), for a pointer or wheel event.
    pub(crate) point: Option<(f64, f64)>,
    /// The image's rectangle in the window, in points.
    pub(crate) image: Rect,
    /// The guest's mode, once it has set one.
    pub(crate) guest: Option<(u16, u16)>,
}

/// What the input sends the host, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToHost {
    /// A pointer position with the buttons held.
    Pointer {
        at: (u16, u16),
        mode: (u16, u16),
        buttons: u32,
    },
    /// Wheel notches, and the smooth distance in 1/256 point.
    Wheel {
        notches: (i32, i32),
        smooth: (i32, i32),
    },
    /// A key down or up.
    Key(KeyEvent),
}

/// One wheel message's values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Wheel {
    /// Whole notches, right then away from the person.
    pub(crate) notches: (i32, i32),
    /// Distance in lines, right then away.
    pub(crate) lines: (f32, f32),
}

/// The `BUTTON_*` bit for a Win32 message's button: left/right/middle, and the two X buttons.
pub(crate) fn button_bit_left() -> u32 {
    BUTTON_LEFT
}
/// Right button bit.
pub(crate) fn button_bit_right() -> u32 {
    BUTTON_RIGHT
}
/// Middle button bit.
pub(crate) fn button_bit_middle() -> u32 {
    BUTTON_MIDDLE
}
/// The X button bit for `which` (1 = XBUTTON1 "back", 2 = XBUTTON2 "forward").
pub(crate) fn button_bit_x(which: u16) -> u32 {
    match which {
        1 => BUTTON_SIDE,
        2 => BUTTON_EXTRA,
        _ => 0,
    }
}

/// The guest's mode in `at`, where it has one with pixels.
fn guest_mode(at: &At) -> Option<(u16, u16)> {
    at.guest.filter(|&(width, height)| width > 0 && height > 0)
}

/// Whether `point` is within `image`, with a small margin for DPI subpixel rounding at edges.
fn contains_point(image: Rect, point: (f64, f64)) -> bool {
    const EPS: f64 = 0.5;
    point.0 >= image.x - EPS
        && point.0 <= image.x + image.width + EPS
        && point.1 >= image.y - EPS
        && point.1 <= image.y + image.height + EPS
}

/// `points` in 1/256 point, rounded; 0 for what is not a number.
fn fixed(points: f64) -> i32 {
    if points.is_finite() {
        (points * 256.0)
            .round()
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
    } else {
        0
    }
}

/// What the window does after an event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reaction {
    /// Capture changed: the title and cursor follow it.
    pub(crate) capture_changed: bool,
}

/// One window's input decisions (the module doc). Mirrors the Linux backend's `Input`.
#[derive(Debug, Default)]
pub(crate) struct Input {
    keys: KeyState,
    capture: Capture,
    buttons: u32,
    last: Option<ToHost>,
    focused: bool,
    focused_at: Option<u64>,
    out: Vec<ToHost>,
}

impl Input {
    pub(crate) fn new() -> Input {
        Input::default()
    }

    pub(crate) fn is_captured(&self) -> bool {
        self.capture.is_captured()
    }

    pub(crate) fn has_buttons(&self) -> bool {
        self.buttons != 0
    }

    pub(crate) fn title_suffix(&self) -> &'static str {
        if self.is_captured() {
            CAPTURED_SUFFIX
        } else {
            ""
        }
    }

    pub(crate) fn take(&mut self) -> Vec<ToHost> {
        std::mem::take(&mut self.out)
    }

    fn pointer(&mut self, point: (f64, f64), image: Rect, mode: (u16, u16), always: bool) {
        let event = ToHost::Pointer {
            at: geometry::pointer_to_guest(point, image, mode),
            mode,
            buttons: self.buttons,
        };
        if always || self.last != Some(event) {
            self.last = Some(event);
            self.out.push(event);
        }
    }

    pub(crate) fn moved(&mut self, at: &At) {
        if let (Some(point), Some(mode)) = (at.point, guest_mode(at)) {
            self.pointer(point, at.image, mode, false);
        }
    }

    pub(crate) fn button_down(&mut self, bit: u32, _timestamp: u64, at: &At) -> Reaction {
        let (Some(point), Some(mode)) = (at.point, guest_mode(at)) else {
            return Reaction::default();
        };
        if bit == 0 || !contains_point(at.image, point) {
            return Reaction::default();
        }
        let was_captured = self.is_captured();
        self.capture.click_in_image(true);
        self.buttons |= bit;
        self.pointer(point, at.image, mode, true);
        Reaction {
            capture_changed: self.is_captured() != was_captured,
        }
    }

    pub(crate) fn button_up(&mut self, bit: u32, at: &At) {
        if self.buttons & bit == 0 {
            return;
        }
        self.buttons &= !bit;
        if let (Some(point), Some(mode)) = (at.point, guest_mode(at)) {
            self.pointer(point, at.image, mode, true);
        }
    }

    pub(crate) fn wheel(&mut self, wheel: Wheel, at: &At) {
        let (Some(point), Some(_)) = (at.point, guest_mode(at)) else {
            return;
        };
        if !contains_point(at.image, point) {
            return;
        }
        self.moved(at);
        // Win32 reports the device's own direction (no OS "natural scrolling" flip at this layer).
        let notches = (wheel.notches.1, wheel.notches.0.saturating_mul(-1));
        let lines = (f64::from(wheel.lines.1), f64::from(wheel.lines.0));
        let smooth = (
            fixed(POINTS_PER_LINE * lines.0),
            fixed(-POINTS_PER_LINE * lines.1),
        );
        if notches != (0, 0) || smooth != (0, 0) {
            self.out.push(ToHost::Wheel { notches, smooth });
        }
    }

    pub(crate) fn key(&mut self, usage: u16, down: bool, repeat: bool) -> Reaction {
        if !self.focused {
            return Reaction::default();
        }
        if !down {
            if let Some(event) = self.keys.key_up(usage) {
                self.out.push(ToHost::Key(event));
            }
            return Reaction::default();
        }
        if let Some(event) = self.keys.key_down(usage, repeat) {
            self.out.push(ToHost::Key(event));
        }
        if self.capture.release_chord(self.keys.held_modifiers()) {
            return self.release_capture();
        }
        Reaction::default()
    }

    pub(crate) fn focus_gained(&mut self, timestamp: u64) {
        self.focused = true;
        self.focused_at = Some(timestamp);
    }

    pub(crate) fn focus_lost(&mut self) -> Reaction {
        self.focused = false;
        self.lift_keys();
        self.lift_buttons();
        Reaction {
            capture_changed: self.capture.resign_key(),
        }
    }

    pub(crate) fn release_capture(&mut self) -> Reaction {
        self.lift_keys();
        Reaction {
            capture_changed: self.capture.release(),
        }
    }

    pub(crate) fn capture_lost(&mut self) {
        self.lift_buttons();
    }

    pub(crate) fn closing(&mut self) {
        self.lift_keys();
        self.lift_buttons();
        self.capture.release();
    }

    fn lift_keys(&mut self) {
        let ups = self.keys.release_all();
        self.out.extend(ups.into_iter().map(ToHost::Key));
    }

    fn lift_buttons(&mut self) {
        if self.buttons == 0 {
            return;
        }
        self.buttons = 0;
        if let Some(ToHost::Pointer { at, mode, .. }) = self.last {
            let event = ToHost::Pointer {
                at,
                mode,
                buttons: 0,
            };
            self.last = Some(event);
            self.out.push(event);
        }
    }
}

/// Hands `events` to the host's `callbacks`, in order. Smooth scrolling goes to `scroll` where the
/// host gives it, else notches to `wheel`.
pub(crate) fn send(callbacks: &WindowCallbacks, events: Vec<ToHost>) {
    for event in events {
        match event {
            ToHost::Pointer { at, mode, buttons } => {
                if let Some(f) = &callbacks.pointer {
                    f(at.0, at.1, mode.0, mode.1, buttons);
                }
            }
            ToHost::Wheel { notches, smooth } => {
                if let Some(f) = &callbacks.scroll {
                    f(ScrollEvent {
                        dy: smooth.0,
                        dx: smooth.1,
                        notches,
                        ..ScrollEvent::default()
                    });
                } else if let Some(f) = &callbacks.wheel
                    && notches != (0, 0)
                {
                    f(notches.0, notches.1);
                }
            }
            ToHost::Key(KeyEvent { usage, down }) => {
                if let Some(f) = &callbacks.key {
                    f(usage, down);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 100×50 image at the window origin, a 200×100 guest mode, the pointer at `point`.
    fn at(point: (f64, f64)) -> At {
        At {
            point: Some(point),
            image: Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 50.0,
            },
            guest: Some((200, 100)),
        }
    }

    fn focused() -> Input {
        let mut input = Input::new();
        input.focus_gained(0);
        input
    }

    fn pointer(x: u16, y: u16, buttons: u32) -> ToHost {
        ToHost::Pointer {
            at: (x, y),
            mode: (200, 100),
            buttons,
        }
    }

    /// A click in the image captures and is sent; one outside starts nothing
    #[test]
    fn a_click_in_the_image_captures_and_is_sent() {
        let mut input = focused();
        assert!(
            !input
                .button_down(BUTTON_RIGHT, 1_000_000_000, &at((120.0, 10.0)))
                .capture_changed
        );
        assert_eq!(input.take(), vec![]);
        let r = input.button_down(BUTTON_RIGHT, 1_000_000_000, &at((50.0, 25.0)));
        assert!(r.capture_changed && input.is_captured());
        assert_eq!(input.take(), vec![pointer(100, 50, BUTTON_RIGHT)]);
    }

    /// Host repeats are dropped; each key is one down and one up
    #[test]
    fn host_repeats_are_dropped() {
        let mut input = focused();
        input.key(0x04, true, false);
        input.key(0x04, true, true);
        input.key(0x04, false, false);
        assert_eq!(
            input.take(),
            vec![
                ToHost::Key(KeyEvent::down(0x04)),
                ToHost::Key(KeyEvent::up(0x04))
            ]
        );
    }

    /// Losing the keyboard lifts every key and button and releases capture
    #[test]
    fn losing_the_keyboard_leaves_nothing_stuck() {
        let mut input = focused();
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        input.key(0x04, true, false);
        input.take();
        let r = input.focus_lost();
        assert!(r.capture_changed);
        assert_eq!(
            input.take(),
            vec![ToHost::Key(KeyEvent::up(0x04)), pointer(20, 20, 0)]
        );
    }

    /// Losing mouse capture lifts held buttons
    #[test]
    fn losing_capture_lifts_held_buttons() {
        let mut input = focused();
        input.button_down(BUTTON_LEFT, 1_000_000_000, &at((10.0, 10.0)));
        assert!(input.has_buttons());
        input.take();
        input.capture_lost();
        assert!(!input.has_buttons());
        assert_eq!(input.take(), vec![pointer(20, 20, 0)]);
    }

    /// The wheel over the image sends notches up and left and lines as smooth scrolling
    #[test]
    fn the_wheel_sends_notches_and_smooth() {
        let mut input = focused();
        let up = Wheel {
            notches: (0, 1),
            lines: (0.0, 1.0),
        };
        input.wheel(up, &at((10.0, 10.0)));
        assert_eq!(
            input.take(),
            vec![
                pointer(20, 20, 0),
                ToHost::Wheel {
                    notches: (1, 0),
                    smooth: (2560, 0)
                },
            ]
        );
    }

    /// The X buttons map to the side and extra bits
    #[test]
    fn x_buttons_map_to_side_and_extra() {
        assert_eq!(button_bit_x(1), BUTTON_SIDE);
        assert_eq!(button_bit_x(2), BUTTON_EXTRA);
        assert_eq!(button_bit_x(3), 0);
    }
}
