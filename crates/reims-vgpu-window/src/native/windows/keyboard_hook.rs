//! Pure keyboard normalization; no hook is installed and no host input is suppressed.
//!
//! The future hook adapter supplies the virtual key, scan code, E0 flag and
//! direction. LL hooks have no E1 flag: VK_PAUSE must be identified explicitly;
//! `e1` also supports replaying raw-input evidence. E1/Pause and unmapped keys
//! pass through rather than being guessed as Num Lock or Control.
//!
//! Right Alt maps to HID RAlt (Option). AltGr's accompanying Control events are
//! preserved; this layer cannot identify fabricated versus physical Control.
//! Ctrl+Alt+Delete remains host policy: a matching Delete event is marked for
//! pass-through. This does not restore modifiers already swallowed earlier in
//! the chord, nor claim that a user-mode hook receives or can block SAS. Full
//! SAS preservation must be decided at the source before suppressing any chord
//! member.
//! Ctrl+Alt+Win release belongs to the focus coordinator, not this mapper.

use super::keymap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScanEvent {
    pub(crate) scan_code: u32,
    pub(crate) virtual_key: u32,
    pub(crate) extended: bool,
    pub(crate) e1: bool,
    pub(crate) down: bool,
    /// A caller-supplied repeat indication; LL hooks themselves have no repeat bit.
    pub(crate) repeat: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KeyChange {
    pub(crate) usage: u16,
    pub(crate) down: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassReason {
    Unsupported,
    PauseOrE1,
    SecureAttention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Normalized {
    Key(KeyChange),
    /// No guest transition: duplicate make/break or repeat without a known make.
    Ignore,
    /// The later adapter must retain host handling for this event; earlier events
    /// already routed remain unchanged. This does not suppress anything here.
    PassThrough(PassReason),
}

#[derive(Debug, Default)]
pub(crate) struct Keyboard {
    // Arrival order permits releasing non-modifiers before earlier modifiers.
    held: Vec<u16>,
    host_delete_down: bool,
}

impl Keyboard {
    pub(crate) fn normalize(&mut self, event: ScanEvent) -> Normalized {
        const VK_PAUSE: u32 = 0x13;
        if event.e1 || event.virtual_key == VK_PAUSE {
            return Normalized::PassThrough(PassReason::PauseOrE1);
        }
        let Some(usage) = keymap::usb_usage(event.scan_code, event.extended) else {
            return Normalized::PassThrough(PassReason::Unsupported);
        };
        // Breaks never depend on the repeat flag or current modifier state.
        if !event.down {
            if usage == 0x4c && self.host_delete_down {
                self.host_delete_down = false;
                return Normalized::PassThrough(PassReason::SecureAttention);
            }
            return match self.held.iter().position(|&key| key == usage) {
                Some(index) => {
                    self.held.remove(index);
                    Normalized::Key(KeyChange { usage, down: false })
                }
                None => Normalized::Ignore,
            };
        }
        if usage == 0x4c && (self.host_delete_down || (self.control_held() && self.alt_held())) {
            self.host_delete_down = !self.held.contains(&usage);
            return Normalized::PassThrough(PassReason::SecureAttention);
        }
        if event.repeat || self.held.contains(&usage) {
            return Normalized::Ignore;
        }
        self.held.push(usage);
        Normalized::Key(KeyChange { usage, down: true })
    }

    pub(crate) fn release_all(&mut self) -> Vec<KeyChange> {
        self.held
            .drain(..)
            .rev()
            .map(|usage| KeyChange { usage, down: false })
            .collect()
    }

    fn control_held(&self) -> bool {
        self.held.contains(&0xe0) || self.held.contains(&0xe4)
    }
    fn alt_held(&self) -> bool {
        self.held.contains(&0xe2) || self.held.contains(&0xe6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(scan_code: u32, extended: bool, down: bool) -> ScanEvent {
        ScanEvent {
            scan_code,
            virtual_key: 0,
            extended,
            e1: false,
            down,
            repeat: false,
        }
    }
    fn key(usage: u16, down: bool) -> Normalized {
        Normalized::Key(KeyChange { usage, down })
    }

    #[test]
    fn e0_shared_scans_and_modifiers_preserve_sides() {
        for (scan_code, left, right) in [
            (0x1c, 0x28, 0x58),
            (0x1d, 0xe0, 0xe4),
            (0x38, 0xe2, 0xe6),
            (0x53, 0x63, 0x4c),
        ] {
            let mut k = Keyboard::default();
            assert_eq!(k.normalize(scan(scan_code, false, true)), key(left, true));
            assert_eq!(k.normalize(scan(scan_code, true, true)), key(right, true));
            assert_eq!(k.normalize(scan(scan_code, false, false)), key(left, false));
            assert_eq!(k.normalize(scan(scan_code, true, false)), key(right, false));
        }
        let mut k = Keyboard::default();
        for (sc, ext, usage) in [
            (0x2a, false, 0xe1),
            (0x36, false, 0xe5),
            (0x5b, true, 0xe3),
            (0x5c, true, 0xe7),
        ] {
            assert_eq!(k.normalize(scan(sc, ext, true)), key(usage, true));
        }
    }

    #[test]
    fn repeats_and_duplicate_ups_never_lose_the_real_release() {
        let mut k = Keyboard::default();
        let down = scan(0x1e, false, true);
        assert_eq!(
            k.normalize(ScanEvent {
                repeat: true,
                ..down
            }),
            Normalized::Ignore
        );
        assert_eq!(k.normalize(down), key(0x04, true));
        assert_eq!(k.normalize(down), Normalized::Ignore);
        assert_eq!(
            k.normalize(ScanEvent {
                repeat: true,
                ..down
            }),
            Normalized::Ignore
        );
        assert_eq!(
            k.normalize(ScanEvent {
                down: false,
                repeat: true,
                ..down
            }),
            key(0x04, false)
        );
        assert_eq!(k.normalize(scan(0x1e, false, false)), Normalized::Ignore);
    }

    #[test]
    fn pause_and_e1_are_not_misread_as_numlock_or_control() {
        let mut k = Keyboard::default();
        for event in [
            ScanEvent {
                virtual_key: 0x13,
                ..scan(0x45, false, true)
            },
            ScanEvent {
                e1: true,
                ..scan(0x1d, false, true)
            },
        ] {
            assert_eq!(
                k.normalize(event),
                Normalized::PassThrough(PassReason::PauseOrE1)
            );
        }
        assert!(k.release_all().is_empty());
        assert_eq!(k.normalize(scan(0x45, false, true)), key(0x53, true));
    }

    #[test]
    fn altgr_keeps_right_alt_and_control_without_character_translation() {
        let mut k = Keyboard::default();
        assert_eq!(k.normalize(scan(0x1d, false, true)), key(0xe0, true));
        assert_eq!(k.normalize(scan(0x38, true, true)), key(0xe6, true));
        assert_eq!(
            k.release_all(),
            vec![
                KeyChange {
                    usage: 0xe6,
                    down: false
                },
                KeyChange {
                    usage: 0xe0,
                    down: false
                }
            ]
        );
        assert!(k.release_all().is_empty());
    }

    #[test]
    fn sas_is_host_policy_but_an_already_sent_delete_gets_its_up() {
        let mut k = Keyboard::default();
        k.normalize(scan(0x1d, false, true));
        k.normalize(scan(0x38, false, true));
        assert_eq!(
            k.normalize(scan(0x53, true, true)),
            Normalized::PassThrough(PassReason::SecureAttention)
        );
        // Host routing survives modifier release and guest all-up cleanup.
        k.release_all();
        assert_eq!(
            k.normalize(scan(0x53, true, true)),
            Normalized::PassThrough(PassReason::SecureAttention)
        );
        assert_eq!(
            k.normalize(scan(0x53, true, false)),
            Normalized::PassThrough(PassReason::SecureAttention)
        );
        k.release_all();
        assert_eq!(k.normalize(scan(0x53, true, true)), key(0x4c, true));
        k.normalize(scan(0x1d, false, true));
        k.normalize(scan(0x38, false, true));
        assert_eq!(k.normalize(scan(0x53, true, false)), key(0x4c, false));
    }

    #[test]
    fn unsupported_keys_pass_through_and_all_up_clears_known_keys() {
        let mut k = Keyboard::default();
        assert_eq!(
            k.normalize(scan(0xffff, false, true)),
            Normalized::PassThrough(PassReason::Unsupported)
        );
        k.normalize(scan(0x1d, false, true));
        k.normalize(scan(0x1e, false, true));
        assert_eq!(
            k.release_all(),
            vec![
                KeyChange {
                    usage: 0x04,
                    down: false
                },
                KeyChange {
                    usage: 0xe0,
                    down: false
                }
            ]
        );
        assert_eq!(k.normalize(scan(0x1e, false, false)), Normalized::Ignore);
    }
}
