//! What the guest display copies into the page for the guest: the display's settings, its modes,
//! the protection options and the guest's colour space.
//!
//! The syncs write each field only where the page differs, and answer whether any did; the state
//! machine raises its connect or update interrupt on that answer.

use std::sync::Arc;

use super::cursor::{CURSOR_HANDLER_MOVE, CURSOR_HANDLER_SHOW, GLYPH_LIMIT};
use super::{DeathKind, GuestDisplay, GuestDisplayLocked, Page};
use crate::display::page as sp;
use crate::display::{DisplayCoord, DisplayCore, DisplayName};
use crate::num::{self, ToUsize, Truncate};
use crate::platform::Platform;
use crate::raise::{OnFault, RaiseResult};
use crate::rt::{Held, Hold};

/// A string's UTF-8 bytes and a NUL in a zeroed buffer of `N` bytes when `len + 1 <= N`; a
/// longer string, or none, leaves the buffer all zero.
fn c_string<const N: usize>(s: Option<&str>) -> [u8; N] {
    let mut buf = [0u8; N];
    if let Some(s) = s {
        let bytes = s.as_bytes();
        if bytes.len() < N {
            for (d, b) in buf.iter_mut().zip(bytes) {
                *d = *b;
            }
        }
    }
    buf
}

/// `strncmp(page + off, s, n) != 0`: byte by byte, stopping at the first difference or at a NUL
/// both share.
fn strncmp_differs(page: Page, off: u64, s: &[u8], n: u64) -> bool {
    let mut i = 0u64;
    while i < n {
        let a = page.r8(off.wrapping_add(i));
        let b = s.get(i.to_usize()).copied().unwrap_or(0);
        if a != b {
            return true;
        }
        if a == 0 {
            return false;
        }
        i = i.wrapping_add(1);
    }
    false
}

/// `strncpy(page + off, s, n)`: up to the first NUL, then zeros to `n`.
fn strncpy(page: Page, off: u64, s: &[u8], n: u64) {
    let mut ended = false;
    let mut i = 0u64;
    while i < n {
        let b = if ended {
            0
        } else {
            s.get(i.to_usize()).copied().unwrap_or(0)
        };
        if b == 0 {
            ended = true;
        }
        page.w8(off.wrapping_add(i), b);
        i = i.wrapping_add(1);
    }
}

impl<P: Platform> GuestDisplay<P> {
    /// Stores the protection options in the page: `options` when the device enables protected
    /// content, else 0. Dies with the device gone.
    pub(super) fn store_protection_options(&self, page: Page, options: u64) {
        let device = self.device.upgrade();
        crate::fatal_unless!(device.is_some(), DeathKind::ProtectionWithoutDevice);
        let enabled = device.is_some_and(|d| d.enable_protected_content());
        page.w64(sp::PROTECTION_OPTIONS, if enabled { options } else { 0 });
    }

    /// The display's protection options, into the page. No lock and no paused check: the
    /// display calls it under its guest display pointer lock only.
    pub fn set_protection_options(&self, options: u64) {
        self.store_protection_options(self.raw_page(), options);
    }

    /// The page's protection options.
    pub fn protection_options(&self) -> u64 {
        self.raw_page().r64(sp::PROTECTION_OPTIONS)
    }

    /// The guest's compositor parameters. No display has a handler for them (no front end sets
    /// one): nothing is dispatched. The client is still taken, and let go, as when one is.
    pub fn compositor_parameters(&self, config_epoch: u64, origin: DisplayCoord, scale_factor: u8) {
        let _ = (config_epoch, origin, scale_factor);
        let l = self.guest_display_mutex.lock();
        Self::assert_running(&l);
        let _client = self.client_core();
    }

    /// Takes the guest's colour space again: marks it dirty, then reads the ICC profile the page
    /// names through the kernel task, or none for a zero length. The display keeps the bytes as
    /// they are, taken at its next present (`DisplayCore::guest_color_space`); no API hands them
    /// to the host. No paused check.
    ///
    /// The read runs in a holder: the kernel task, and what the read holds with no handle to a
    /// holder of its own, go at its end, most recent first; on a raise they go on with it to
    /// where it ends. Its raise is as `on` says: the state machine reaches it both under a
    /// handler and with none.
    pub(super) fn update_guest_color_space_locked(
        &self,
        on: OnFault,
        l: &mut GuestDisplayLocked<P>,
    ) -> RaiseResult<()> {
        l.guest_color_space_dirty = true;
        Held::raising(|held| {
            let hold = Hold::of(held);
            *self.unlocked.guest_color_space.lock() = None;
            let p = self.raw_page();
            if p.r32(sp::GUEST_ICC_PROFILE_LENGTH) != 0 {
                // The receiver first, then the arguments.
                let task = self.kernel_task(on, held)?;
                let data = match task {
                    Some(t) => t.data_from_virtual_offset(
                        hold,
                        on,
                        p.r64(sp::GUEST_ICC_PROFILE_VIRTUAL_OFFSET),
                        u64::from(p.r32(sp::GUEST_ICC_PROFILE_LENGTH)),
                    )?,
                    None => None,
                };
                *self.unlocked.guest_color_space.lock() = data.map(Arc::from);
            }
            Ok(())
        })
    }

    /// The guest's ICC profile record: stored in the page, then the colour space taken again
    /// (which may raise after the page writes). No paused check.
    pub fn set_guest_icc_profile_length(
        &self,
        length: u32,
        virtual_offset: u64,
    ) -> RaiseResult<()> {
        let mut l = self.guest_display_mutex.lock();
        let p = self.raw_page();
        p.w64(sp::GUEST_ICC_PROFILE_VIRTUAL_OFFSET, virtual_offset);
        p.w32(sp::GUEST_ICC_PROFILE_LENGTH, length);
        self.update_guest_color_space_locked(OnFault::Return, &mut l)
    }

    /// Syncs the page's modes to the display's list (at most the page's 128), each field
    /// written only where it differs, then consumes the list. Dies with no list, and for a guest
    /// count above 128.
    pub(super) fn sync_mode_list(&self, l: &mut GuestDisplayLocked<P>) -> bool {
        let Some(list) = l.mode_list.clone() else {
            crate::fatal!(DeathKind::NoModeList)
        };
        Self::assert_running(l);
        // A guest count above 128 dies (testing.md 7.4).
        crate::fatal_unless!(
            self.page(l).r16(sp::MODE_LIST_COUNT) <= sp::MAX_MODES,
            DeathKind::ModeCountTooLarge
        );
        let modes = list.get(..usize::from(sp::MAX_MODES)).unwrap_or(&*list);
        let count16 = u16::try_from(modes.len()).unwrap_or(sp::MAX_MODES);
        let mut changed = false;
        if self.page(l).r16(sp::MODE_LIST_COUNT) != count16 {
            self.page(l).w16(sp::MODE_LIST_COUNT, count16);
            changed = true;
        }
        for (i, mode) in (0u16..).zip(modes) {
            let e = sp::mode_entry(i);
            let p = self.page(l);
            let w = e.wrapping_add(sp::MODE_WIDTH);
            if p.r16(w) != mode.size_in_pixels.x {
                p.w16(w, mode.size_in_pixels.x);
                changed = true;
            }
            let h = e.wrapping_add(sp::MODE_HEIGHT);
            if p.r16(h) != mode.size_in_pixels.y {
                p.w16(h, mode.size_in_pixels.y);
                changed = true;
            }
            // 16.16 fixed point, `fcvtmu`.
            let rate = num::refresh_16_16(mode.refresh_rate_in_hz);
            let r = e.wrapping_add(sp::MODE_REFRESH_RATE);
            if p.r32(r) != rate {
                p.w32(r, rate);
                changed = true;
            }
            // Bit 1, a read-modify-write of the byte.
            let f = e.wrapping_add(sp::MODE_FLAGS);
            if (p.r8(f) & sp::MODE_FLAG_SUPPORTS_10BPC != 0) != mode.supports_10bpc {
                let b = p.r8(f);
                p.w8(
                    f,
                    if mode.supports_10bpc {
                        b | sp::MODE_FLAG_SUPPORTS_10BPC
                    } else {
                        b & !sp::MODE_FLAG_SUPPORTS_10BPC
                    },
                );
                changed = true;
            }
            // Bit 0.
            if (p.r8(f) & sp::MODE_FLAG_SUPPORTS_EDR != 0) != mode.supports_edr {
                let b = p.r8(f);
                p.w8(
                    f,
                    if mode.supports_edr {
                        b | sp::MODE_FLAG_SUPPORTS_EDR
                    } else {
                        b & !sp::MODE_FLAG_SUPPORTS_EDR
                    },
                );
                changed = true;
            }
        }
        // Consumed.
        l.mode_list = None;
        changed
    }

    /// Syncs the page's settings to the client's: the serial number, the size, the colour
    /// space, the origin and scale, the luminances, the epoch, the connection type, the name,
    /// the UUID and the cursor's limits and handlers; each field written only where it differs
    /// (the rounded sizes always). Dies with no client.
    pub(super) fn sync_settings(
        &self,
        l: &GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
    ) -> bool {
        let Some(c) = client else {
            crate::fatal!(DeathKind::SettingsWithoutClient)
        };
        Self::assert_running(l);
        let d = &c.descriptor;
        let mut changed = false;
        let p = self.page(l);
        // The serial number, read to compare and again to store.
        if p.r32(sp::SERIAL_NUM) != c.serial_num() {
            p.w32(sp::SERIAL_NUM, c.serial_num());
            changed = true;
        }
        // `(float)` of the `CGFloat`, IEEE `!=`; the getter read again to store.
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "f64 to f32 rounds to nearest"
        )]
        let to_f32 = |d: f64| d as f32;
        if p.rf32(sp::WIDTH_IN_MILLIMETERS) != to_f32(c.size_in_millimeters().0) {
            p.wf32(sp::WIDTH_IN_MILLIMETERS, to_f32(c.size_in_millimeters().0));
            changed = true;
        }
        if p.rf32(sp::HEIGHT_IN_MILLIMETERS) != to_f32(c.size_in_millimeters().1) {
            p.wf32(sp::HEIGHT_IN_MILLIMETERS, to_f32(c.size_in_millimeters().1));
            changed = true;
        }
        // Always stored, never counted; `(double)f + 0.5`, `fcvtzs w`, the low 16 bits.
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "f64 to i32 truncates toward zero, saturating, NaN to 0"
        )]
        let round = |f: f32| ((f64::from(f) + 0.5) as i32).cast_unsigned().low_u16();
        p.w16(
            sp::ROUNDED_WIDTH_IN_MILLIMETERS,
            round(p.rf32(sp::WIDTH_IN_MILLIMETERS)),
        );
        p.w16(
            sp::ROUNDED_HEIGHT_IN_MILLIMETERS,
            round(p.rf32(sp::HEIGHT_IN_MILLIMETERS)),
        );
        // Red, green, blue, white; x then y.
        let cs = d.color_space;
        let components = [
            cs.red_primary.x,
            cs.red_primary.y,
            cs.green_primary.x,
            cs.green_primary.y,
            cs.blue_primary.x,
            cs.blue_primary.y,
            cs.white_point.x,
            cs.white_point.y,
        ];
        for (i, v) in (0u64..).zip(components) {
            let off = sp::COLOR_SPACE.wrapping_add(i.wrapping_mul(4));
            if p.rf32(off) != v {
                p.wf32(off, v);
                changed = true;
            }
        }
        // The origin, the scale factor and the epoch, each getter read again to store.
        if p.r16(sp::ORIGIN) != c.origin().x {
            p.w16(sp::ORIGIN, c.origin().x);
            changed = true;
        }
        let origin_y = sp::ORIGIN.wrapping_add(2);
        if p.r16(origin_y) != c.origin().y {
            p.w16(origin_y, c.origin().y);
            changed = true;
        }
        if p.r8(sp::SCALE_FACTOR) != c.scale_factor() {
            p.w8(sp::SCALE_FACTOR, c.scale_factor());
            changed = true;
        }
        // EXPERIMENT(display-nits): REIMS_VGPU_DISPLAY_NITS="min,max,maxsdr" overrides the
        // descriptor's nits, to see whether the guest composites in 8-bit with an SDR display.
        static NITS: std::sync::OnceLock<Option<(f32, f32, f32)>> = std::sync::OnceLock::new();
        let nits = NITS.get_or_init(|| {
            let v = std::env::var("REIMS_VGPU_DISPLAY_NITS").ok()?;
            let n: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (n.len() == 3).then(|| (n[0], n[1], n[2]))
        });
        let (min_nits, max_nits, max_sdr_nits) =
            nits.unwrap_or((d.min_nits, d.max_nits, d.max_sdr_nits));
        for (off, v) in [
            (sp::MIN_NITS, min_nits),
            (sp::MAX_NITS, max_nits),
            (sp::MAX_SDR_NITS, max_sdr_nits),
        ] {
            if p.rf32(off) != v {
                p.wf32(off, v);
                changed = true;
            }
        }
        if p.r64(sp::CONFIG_EPOCH) != c.config_epoch() {
            p.w64(sp::CONFIG_EPOCH, c.config_epoch());
            changed = true;
        }
        let connection_type = d.connection_type.cast_unsigned();
        if p.r32(sp::CONNECTION_TYPE) != connection_type {
            p.w32(sp::CONNECTION_TYPE, connection_type);
            changed = true;
        }
        // 14 UTF-8 bytes or more fail and leave zeros.
        let name = d.name.as_ref().map(DisplayName::current);
        let name: [u8; 14] = c_string(name.as_deref());
        if strncmp_differs(p, sp::NAME, &name, sp::NAME_LEN) {
            strncpy(p, sp::NAME, &name, sp::NAME_LEN);
            changed = true;
        }
        // 40 bytes, compared and copied over 39 (a 39-byte UUID leaves no NUL in the page).
        let uuid: [u8; 40] = c_string(d.display_uuid.as_deref());
        if strncmp_differs(p, sp::DISPLAY_UUID, &uuid, sp::DISPLAY_UUID_LEN) {
            strncpy(p, sp::DISPLAY_UUID, &uuid, sp::DISPLAY_UUID_LEN);
            changed = true;
        }
        let glyph_limit = if c.queue().is_some() && c.cursor_glyph_handler().is_some() {
            GLYPH_LIMIT
        } else {
            0
        };
        if p.r16(sp::CURSOR_GLYPH_MAX_WIDTH) != glyph_limit {
            p.w16(sp::CURSOR_GLYPH_MAX_WIDTH, glyph_limit);
            changed = true;
        }
        if p.r16(sp::CURSOR_GLYPH_MAX_HEIGHT) != glyph_limit {
            p.w16(sp::CURSOR_GLYPH_MAX_HEIGHT, glyph_limit);
            changed = true;
        }
        let mut handlers = if c.queue().is_some() && c.cursor_show_handler().is_some() {
            CURSOR_HANDLER_SHOW
        } else {
            0
        };
        if c.queue().is_some() && c.cursor_move_handler().is_some() {
            handlers |= CURSOR_HANDLER_MOVE;
        }
        if p.r32(sp::CURSOR_HANDLERS) != handlers {
            p.w32(sp::CURSOR_HANDLERS, handlers);
            changed = true;
        }
        changed
    }
}
