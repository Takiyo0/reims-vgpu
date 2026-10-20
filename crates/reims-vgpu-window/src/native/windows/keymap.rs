//! Win32 keyboard scancodes to USB HID keyboard usages (page 0x07), the usages the guest's
//! keyboard sends.
//!
//! A `WM_KEYDOWN`/`WM_KEYUP` carries, in `lParam`, the PS/2 set-1 scancode (bits 16..=23) and an
//! extended-key flag (bit 24). The guest reads keys BY POSITION (its own layout applies), so this
//! maps the physical scancode, not the character, to the USB usage of that position — the same
//! contract as Linux's keymap, which gets the usage straight from the evdev code.
//!
//! The table covers the keyboard page's keys (letters, digits, punctuation, function keys, the
//! navigation block, the keypad, the modifiers, and the ISO key left of 1 / right of left Shift).
//! A scancode with no usage on the keyboard page (media / power / browser keys) returns `None` and
//! is dropped, as Linux drops SDL's own high scancodes.

/// The USB usage of the key named by a Win32 set-1 `scancode`, with `extended` set for an E0-
/// prefixed key, or `None` for a key with no keyboard-page usage.
pub(crate) fn usb_usage(scancode: u32, extended: bool) -> Option<u16> {
    Some(match (scancode, extended) {
        // Row: Esc, digits, minus, equals, backspace
        (0x01, false) => 0x29, // Escape
        (0x02, false) => 0x1e, // 1
        (0x03, false) => 0x1f, // 2
        (0x04, false) => 0x20, // 3
        (0x05, false) => 0x21, // 4
        (0x06, false) => 0x22, // 5
        (0x07, false) => 0x23, // 6
        (0x08, false) => 0x24, // 7
        (0x09, false) => 0x25, // 8
        (0x0a, false) => 0x26, // 9
        (0x0b, false) => 0x27, // 0
        (0x0c, false) => 0x2d, // - _
        (0x0d, false) => 0x2e, // = +
        (0x0e, false) => 0x2a, // Backspace
        // Row: Tab, QWERTY, brackets, Enter
        (0x0f, false) => 0x2b, // Tab
        (0x10, false) => 0x14, // q
        (0x11, false) => 0x1a, // w
        (0x12, false) => 0x08, // e
        (0x13, false) => 0x15, // r
        (0x14, false) => 0x17, // t
        (0x15, false) => 0x1c, // y
        (0x16, false) => 0x18, // u
        (0x17, false) => 0x0c, // i
        (0x18, false) => 0x12, // o
        (0x19, false) => 0x13, // p
        (0x1a, false) => 0x2f, // [ {
        (0x1b, false) => 0x30, // ] }
        (0x1c, false) => 0x28, // Enter
        (0x1c, true) => 0x58,  // Keypad Enter
        // Row: Ctrl, ASDF, semicolon, quote, backtick
        (0x1d, false) => 0xe0, // Left Control
        (0x1d, true) => 0xe4,  // Right Control
        (0x1e, false) => 0x04, // a
        (0x1f, false) => 0x16, // s
        (0x20, false) => 0x07, // d
        (0x21, false) => 0x09, // f
        (0x22, false) => 0x0a, // g
        (0x23, false) => 0x0b, // h
        (0x24, false) => 0x0d, // j
        (0x25, false) => 0x0e, // k
        (0x26, false) => 0x0f, // l
        (0x27, false) => 0x33, // ; :
        (0x28, false) => 0x34, // ' "
        (0x29, false) => 0x35, // ` ~
        // Row: Shift, backslash, ZXCV, slash
        (0x2a, false) => 0xe1, // Left Shift
        (0x2b, false) => 0x31, // \ |
        (0x2c, false) => 0x1d, // z
        (0x2d, false) => 0x1b, // x
        (0x2e, false) => 0x06, // c
        (0x2f, false) => 0x19, // v
        (0x30, false) => 0x05, // b
        (0x31, false) => 0x11, // n
        (0x32, false) => 0x10, // m
        (0x33, false) => 0x36, // , <
        (0x34, false) => 0x37, // . >
        (0x35, false) => 0x38, // / ?
        (0x35, true) => 0x54,  // Keypad /
        (0x36, false) => 0xe5, // Right Shift
        // Bottom row and space
        (0x37, false) => 0x55, // Keypad *
        (0x37, true) => 0x46,  // Print Screen
        (0x38, false) => 0xe2, // Left Alt
        (0x38, true) => 0xe6,  // Right Alt
        (0x39, false) => 0x2c, // Space
        (0x3a, false) => 0x39, // Caps Lock
        // Function keys
        (0x3b, false) => 0x3a, // F1
        (0x3c, false) => 0x3b, // F2
        (0x3d, false) => 0x3c, // F3
        (0x3e, false) => 0x3d, // F4
        (0x3f, false) => 0x3e, // F5
        (0x40, false) => 0x3f, // F6
        (0x41, false) => 0x40, // F7
        (0x42, false) => 0x41, // F8
        (0x43, false) => 0x42, // F9
        (0x44, false) => 0x43, // F10
        (0x57, false) => 0x44, // F11
        (0x58, false) => 0x45, // F12
        // Locks and keypad
        (0x45, false) => 0x53, // Num Lock
        (0x46, false) => 0x47, // Scroll Lock
        (0x47, false) => 0x5f, // Keypad 7
        (0x48, false) => 0x60, // Keypad 8
        (0x49, false) => 0x61, // Keypad 9
        (0x4a, false) => 0x56, // Keypad -
        (0x4b, false) => 0x5c, // Keypad 4
        (0x4c, false) => 0x5d, // Keypad 5
        (0x4d, false) => 0x5e, // Keypad 6
        (0x4e, false) => 0x57, // Keypad +
        (0x4f, false) => 0x59, // Keypad 1
        (0x50, false) => 0x5a, // Keypad 2
        (0x51, false) => 0x5b, // Keypad 3
        (0x52, false) => 0x62, // Keypad 0
        (0x53, false) => 0x63, // Keypad .
        // The ISO key left of 1 (OEM_102) and right of left Shift
        (0x56, false) => 0x64, // ISO \ (between left Shift and Z on ISO layouts)
        // Extended (E0-prefixed) navigation + GUI
        (0x47, true) => 0x4a, // Home
        (0x48, true) => 0x52, // Up Arrow
        (0x49, true) => 0x4b, // Page Up
        (0x4b, true) => 0x50, // Left Arrow
        (0x4d, true) => 0x4f, // Right Arrow
        (0x4f, true) => 0x4d, // End
        (0x50, true) => 0x51, // Down Arrow
        (0x51, true) => 0x4e, // Page Down
        (0x52, true) => 0x49, // Insert
        (0x53, true) => 0x4c, // Delete
        (0x5b, true) => 0xe3, // Left GUI (guest's left Command)
        (0x5c, true) => 0xe7, // Right GUI
        (0x5d, true) => 0x65, // Application (Menu)
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Letters, digits and modifiers map to their USB usages by position
    #[test]
    fn common_keys_map_to_usb_usages() {
        assert_eq!(usb_usage(0x1e, false), Some(0x04), "a");
        assert_eq!(usb_usage(0x1c, false), Some(0x28), "Enter");
        assert_eq!(usb_usage(0x02, false), Some(0x1e), "1");
        assert_eq!(usb_usage(0x1d, false), Some(0xe0), "Left Control");
        assert_eq!(usb_usage(0x5b, true), Some(0xe3), "Left GUI = left Command");
        assert_eq!(usb_usage(0x56, false), Some(0x64), "ISO key");
    }

    /// The extended flag distinguishes keys that share a base scancode
    #[test]
    fn the_extended_flag_distinguishes_shared_scancodes() {
        assert_eq!(usb_usage(0x1c, false), Some(0x28), "Enter");
        assert_eq!(usb_usage(0x1c, true), Some(0x58), "Keypad Enter");
        assert_eq!(usb_usage(0x1d, false), Some(0xe0), "Left Control");
        assert_eq!(usb_usage(0x1d, true), Some(0xe4), "Right Control");
        assert_eq!(usb_usage(0x38, false), Some(0xe2), "Left Alt");
        assert_eq!(usb_usage(0x38, true), Some(0xe6), "Right Alt");
    }

    /// Unmapped scancodes (media / power keys, zero) are dropped
    #[test]
    fn unmapped_scancodes_are_dropped() {
        for (sc, ext) in [(0x00, false), (0x5e, true), (0x20, true), (0xff, false)] {
            assert_eq!(usb_usage(sc, ext), None, "{sc:#x} ext={ext}");
        }
    }
}
