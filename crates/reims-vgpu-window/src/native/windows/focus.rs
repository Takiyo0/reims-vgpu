//! Dormant focus-mode policy. No OS or guest calls occur here.
//!
//! The caller applies returned actions; a release keeps held inputs quarantined
//! until all-up. A lease timeout starts cleanup, not guaranteed host forwarding.

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Config {
    pub(crate) enabled: bool,
}

impl Config {
    /// Parse the master environment value without reading or mutating the environment.
    pub(crate) fn from_value(value: Option<&str>) -> Self {
        Self {
            enabled: value == Some("1"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Readiness {
    pub(crate) foreground: bool,
    pub(crate) touch_ready: bool,
    pub(crate) keyboard_ready: bool,
    pub(crate) guest_ready: bool,
}

impl Readiness {
    fn ready(self) -> bool {
        self.foreground && self.touch_ready && self.keyboard_ready && self.guest_ready
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum State {
    #[default]
    Windows,
    Arming,
    Captured,
    Releasing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReleaseReason {
    User,
    FocusLost,
    Minimized,
    SessionLocked,
    DeviceRemoved,
    QueueOverflow,
    Timeout,
    Closed,
    OwnerChanged,
    NotReady,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PhysicalInput {
    Key(u16),
    Contact(u32),
    Button(u8),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Actions {
    pub(crate) consume_click: bool,
    pub(crate) capture_started: bool,
    pub(crate) cleanup: Option<ReleaseReason>,
    pub(crate) guest_all_up: bool,
    pub(crate) forwarding_resumed: bool,
}

#[derive(Debug, Default)]
pub(crate) struct Coordinator {
    config: Config,
    state: State,
    owner: Option<u64>,
    held: BTreeSet<PhysicalInput>,
    physical_uncertain: bool,
    last_heartbeat_ms: Option<u64>,
}

impl Coordinator {
    pub(crate) const LEASE_MS: u64 = 1000;
    const MAX_HELD: usize = 288;

    pub(crate) fn new(config: Config) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    pub(crate) fn state(&self) -> State {
        self.state
    }

    /// Click-down itself must already be represented in `held` by the caller.
    /// Rejected requests leave the old input path responsible for the click.
    pub(crate) fn activate(&mut self, owner: u64, ready: Readiness, now_ms: u64) -> Actions {
        if !self.config.enabled || !ready.ready() || self.state != State::Windows {
            return Actions::default();
        }
        self.owner = Some(owner);
        self.state = State::Arming;
        let mut actions = self.advance(owner, ready, now_ms);
        actions.consume_click = true;
        actions
    }

    /// Set membership makes repeats and duplicate releases harmless.
    pub(crate) fn physical(&mut self, input: PhysicalInput, down: bool) -> Actions {
        if down {
            if !self.held.contains(&input) && self.held.len() >= Self::MAX_HELD {
                self.physical_uncertain = true;
                return self.release(ReleaseReason::QueueOverflow);
            }
            self.held.insert(input);
        } else {
            self.held.remove(&input);
        }
        if self.state == State::Captured && self.release_chord_held() {
            return self.release(ReleaseReason::User);
        }
        if self.state == State::Releasing && self.all_up() {
            self.reset();
            return Actions {
                forwarding_resumed: true,
                ..Actions::default()
            };
        }
        Actions::default()
    }

    /// A verified all-source all-up snapshot repairs an overflowed ledger.
    /// Never call this merely because a device disappeared or a timer expired.
    pub(crate) fn physical_all_up(&mut self) -> Actions {
        self.held.clear();
        self.physical_uncertain = false;
        if self.state == State::Releasing {
            self.reset();
            return Actions {
                forwarding_resumed: true,
                ..Actions::default()
            };
        }
        Actions::default()
    }

    fn all_up(&self) -> bool {
        self.held.is_empty() && !self.physical_uncertain
    }

    fn release_chord_held(&self) -> bool {
        let either = |left, right| {
            self.held.contains(&PhysicalInput::Key(left))
                || self.held.contains(&PhysicalInput::Key(right))
        };
        either(0xe0, 0xe4) && either(0xe2, 0xe6) && either(0xe3, 0xe7)
    }

    pub(crate) fn advance(&mut self, owner: u64, ready: Readiness, now_ms: u64) -> Actions {
        if self.state == State::Windows || self.state == State::Releasing {
            return Actions::default();
        }
        if self.owner != Some(owner) {
            return self.release(ReleaseReason::OwnerChanged);
        }
        if !ready.ready() {
            return self.release(if ready.foreground {
                ReleaseReason::NotReady
            } else {
                ReleaseReason::FocusLost
            });
        }
        if self.state == State::Arming && self.all_up() {
            self.state = State::Captured;
            self.last_heartbeat_ms = Some(now_ms);
            return Actions {
                capture_started: true,
                ..Actions::default()
            };
        }
        if let Some(last) = self.last_heartbeat_ms {
            if now_ms < last || now_ms - last >= Self::LEASE_MS {
                return self.release(ReleaseReason::Timeout);
            }
        }
        Actions::default()
    }

    /// An expired lease cannot be revived by a late heartbeat.
    pub(crate) fn heartbeat(&mut self, now_ms: u64) -> bool {
        if self.state != State::Captured {
            return false;
        }
        let Some(last) = self.last_heartbeat_ms else {
            return false;
        };
        if now_ms < last || now_ms - last >= Self::LEASE_MS {
            return false;
        }
        self.last_heartbeat_ms = Some(now_ms);
        true
    }

    pub(crate) fn release(&mut self, reason: ReleaseReason) -> Actions {
        if matches!(self.state, State::Windows | State::Releasing) {
            return Actions::default();
        }
        let captured = self.state == State::Captured;
        let resumed = self.all_up();
        if resumed {
            self.reset();
        } else {
            self.state = State::Releasing;
        }
        Actions {
            cleanup: Some(reason),
            guest_all_up: captured,
            forwarding_resumed: resumed,
            ..Actions::default()
        }
    }

    fn reset(&mut self) {
        self.state = State::Windows;
        self.owner = None;
        self.last_heartbeat_ms = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> Readiness {
        Readiness {
            foreground: true,
            touch_ready: true,
            keyboard_ready: true,
            guest_ready: true,
        }
    }
    fn enabled() -> Coordinator {
        Coordinator::new(Config::from_value(Some("1")))
    }

    #[test]
    fn explicit_opt_in_and_readiness_are_required() {
        for value in [None, Some("0"), Some("true"), Some(" 1"), Some("01")] {
            let mut c = Coordinator::new(Config::from_value(value));
            assert_eq!(c.activate(1, ready(), 0), Actions::default());
        }
        let mut c = enabled();
        assert_eq!(c.activate(1, Readiness::default(), 0), Actions::default());
        for field in 0..4 {
            let mut r = ready();
            match field {
                0 => r.foreground = false,
                1 => r.touch_ready = false,
                2 => r.keyboard_ready = false,
                _ => r.guest_ready = false,
            }
            assert!(!c.activate(1, r, 0).consume_click);
        }
    }

    #[test]
    fn activation_is_consumed_and_waits_for_every_input_up() {
        let mut c = enabled();
        for input in [
            PhysicalInput::Button(0),
            PhysicalInput::Key(0xe0),
            PhysicalInput::Contact(7),
        ] {
            c.physical(input, true);
        }
        assert!(c.activate(1, ready(), 0).consume_click);
        assert_eq!(c.state(), State::Arming);
        c.physical(PhysicalInput::Button(0), false);
        c.physical(PhysicalInput::Contact(7), false);
        assert!(!c.advance(1, ready(), 1).capture_started);
        c.physical(PhysicalInput::Key(0xe0), false);
        assert!(c.advance(1, ready(), 2).capture_started);
    }

    #[test]
    fn releases_are_idempotent_and_quarantine_held_input() {
        for reason in [
            ReleaseReason::User,
            ReleaseReason::FocusLost,
            ReleaseReason::Minimized,
            ReleaseReason::SessionLocked,
            ReleaseReason::DeviceRemoved,
            ReleaseReason::QueueOverflow,
            ReleaseReason::Timeout,
            ReleaseReason::Closed,
        ] {
            let mut c = enabled();
            c.activate(1, ready(), 0);
            c.physical(PhysicalInput::Key(0xe3), true);
            c.physical(PhysicalInput::Key(0xe3), true);
            let a = c.release(reason);
            assert_eq!(a.cleanup, Some(reason));
            assert!(a.guest_all_up);
            assert!(!a.forwarding_resumed);
            assert_eq!(c.state(), State::Releasing);
            assert_eq!(c.release(reason), Actions::default());
            assert!(
                c.physical(PhysicalInput::Key(0xe3), false)
                    .forwarding_resumed
            );
            assert_eq!(c.state(), State::Windows);
        }
    }

    #[test]
    fn owner_and_readiness_changes_release() {
        let mut c = enabled();
        c.activate(1, ready(), 0);
        assert_eq!(
            c.advance(2, ready(), 1).cleanup,
            Some(ReleaseReason::OwnerChanged)
        );
        c.activate(2, ready(), 2);
        assert_eq!(
            c.advance(2, Readiness::default(), 3).cleanup,
            Some(ReleaseReason::FocusLost)
        );
        c.activate(2, ready(), 4);
        let mut r = ready();
        r.guest_ready = false;
        assert_eq!(c.advance(2, r, 5).cleanup, Some(ReleaseReason::NotReady));
    }

    #[test]
    fn lease_boundary_and_clock_regression_fail_closed() {
        let mut c = enabled();
        c.activate(1, ready(), 100);
        assert!(c.heartbeat(300));
        assert!(!c.heartbeat(1300));
        assert_eq!(
            c.advance(1, ready(), 1300).cleanup,
            Some(ReleaseReason::Timeout)
        );
        c.activate(1, ready(), 1400);
        assert_eq!(
            c.advance(1, ready(), 1399).cleanup,
            Some(ReleaseReason::Timeout)
        );
    }

    #[test]
    fn arming_cancel_has_no_guest_all_up() {
        let mut c = enabled();
        c.physical(PhysicalInput::Button(0), true);
        c.activate(1, ready(), 0);
        let a = c.release(ReleaseReason::FocusLost);
        assert!(!a.guest_all_up);
        assert_eq!(c.state(), State::Releasing);
        assert!(!c.activate(2, ready(), 1).consume_click);
        assert!(
            c.physical(PhysicalInput::Button(0), false)
                .forwarding_resumed
        );
    }

    #[test]
    fn overflow_needs_a_verified_all_up_snapshot() {
        let mut c = enabled();
        c.activate(1, ready(), 0);
        for id in 0..Coordinator::MAX_HELD {
            c.physical(PhysicalInput::Contact(id as u32), true);
        }
        assert_eq!(
            c.physical(PhysicalInput::Contact(999), true).cleanup,
            Some(ReleaseReason::QueueOverflow)
        );
        for id in 0..Coordinator::MAX_HELD {
            c.physical(PhysicalInput::Contact(id as u32), false);
        }
        assert_eq!(c.state(), State::Releasing);
        assert!(c.physical_all_up().forwarding_resumed);
        assert_eq!(c.state(), State::Windows);
    }

    #[test]
    fn control_alt_win_releases_with_either_modifier_side() {
        for keys in [[0xe0, 0xe2, 0xe3], [0xe4, 0xe6, 0xe7]] {
            let mut c = enabled();
            c.activate(1, ready(), 0);
            assert_eq!(
                c.physical(PhysicalInput::Key(keys[0]), true),
                Actions::default()
            );
            assert_eq!(
                c.physical(PhysicalInput::Key(keys[1]), true),
                Actions::default()
            );
            assert_eq!(
                c.physical(PhysicalInput::Key(keys[2]), true).cleanup,
                Some(ReleaseReason::User)
            );
            assert_eq!(c.state(), State::Releasing);
        }
    }
}
