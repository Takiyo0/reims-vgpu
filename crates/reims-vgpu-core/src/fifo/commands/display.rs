//! The display commands (1-12): each reaches a guest display by its port.
//!
//! The check order is each handler's own, and decides which failure the guest sees: 4 looks the
//! guest display up before its task check, 7 checks the task first; the gated ones (8-12) check the
//! feature before the size. 7, 8, 10 and 12 are S class, the rest F class.

#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use crate::display::guest_display::CursorGlyphArgs;
use crate::display::{DisplayCoord, GammaTableArgs};
use crate::fifo::RaiseKind;
use crate::fifo::commands::{Command, KERNEL_ONLY, UNSUPPORTED, check_size, mul_add_overflows};
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::raise;
use crate::raise::RaiseResult;

/// The records' sizes.
const SET_SHARED_STATE_PAGE: u32 = 8;
const ACK: u32 = 8;
const CURSOR_GLYPH: u32 = 44;
/// A BOOL padded to 8.
const CURSOR_SHOW: u32 = 8;
const TRANSACTION2: u32 = 12;
const TRANSACTION3: u32 = 36;
const SWAP_MAPPING: u32 = 40;
const SET_PROPERTIES: u32 = 8;
/// One property's header.
const PROPERTY: u32 = 8;
const SLEEP_STATE: u32 = 8;
const COMPOSITOR_PARAMETERS: u32 = 20;
const GUEST_ICC_PROFILE: u32 = 24;

/// 1: the display's shared state page, by port.
pub(super) fn set_shared_state_page<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(SET_SHARED_STATE_PAGE)?;
        let p = c.payload;
        c.device()
            .set_shared_display_state_page(p.u32(4), u64::from(p.u32(0)), false)?;
        c.signal();
        Ok(())
    });
}

/// 2: acknowledges a mode change.
pub(super) fn ack<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(ACK)?;
        let p = c.payload;
        log::info!("FIFO CmdDisplayAck: port={}, ack={}", p.u32(0), p.u32(4));
        let guest_display = c.guest_display(p.u32(0))?;
        guest_display.ack(p.u32(4))?;
        c.signal();
        Ok(())
    });
}

/// 4: the cursor's glyph. The guest display lookup, and its raise, precede the task check.
pub(super) fn cursor_glyph<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(CURSOR_GLYPH)?;
        let p = c.payload;
        // The lookup (and its raise) comes before the task check.
        let guest_display = c.guest_display(p.u32(0))?;
        if p.u32(4) != 0 {
            raise!(RaiseKind::KernelOnly, KERNEL_ONLY);
        }
        guest_display.cursor_glyph(CursorGlyphArgs {
            virtual_offset: p.u64(8),
            mapped_length: p.u64(16),
            stride: p.u64(24),
            width: p.u16(32),
            height: p.u16(34),
            hot_spot_x: p.u16(36),
            hot_spot_y: p.u16(38),
            sum: p.u32(40),
        })?;
        c.signal();
        Ok(())
    });
}

/// 5: shows or hides the cursor; the BOOL byte is passed raw.
pub(super) fn cursor_show<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(CURSOR_SHOW)?;
        let p = c.payload;
        let guest_display = c.guest_display(p.u32(0))?;
        // The byte, unmasked
        guest_display.cursor_show(p.u8(4));
        c.signal();
        Ok(())
    });
}

/// 6: presents a surface. The lookup precedes the task check.
pub(super) fn transaction2<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        c.check_size(TRANSACTION2)?;
        let p = c.payload;
        log::info!("FIFO CmdDisplayTransaction2: port={}, surface={}", p.u32(0), p.u32(4));
        let guest_display = c.guest_display(p.u32(0))?;
        if p.u32(8) != 0 {
            raise!(RaiseKind::KernelOnly, KERNEL_ONLY);
        }
        guest_display.present_surface(p.u32(4), GammaTableArgs::default())?;
        c.signal();
        Ok(())
    });
}

/// 7: presents a surface with a gamma table. S class: signalled even on failure; the task check
/// precedes the lookup.
pub(super) fn transaction3<P: Platform>(c: &mut Command<'_, P>) {
    let size = c.payload.len();
    let r = (|| -> RaiseResult<_> {
        c.check_size(TRANSACTION3)?;
        let p = c.payload;
        // The task check before the lookup (4 has them the other way).
        if p.u32(4) != 0 {
            raise!(RaiseKind::KernelOnly, KERNEL_ONLY);
        }
        let guest_display = c.guest_display(p.u32(0))?;
        guest_display.present_surface(
            p.u32(8),
            GammaTableArgs {
                virtual_offset: p.u64(12),
                mapped_length: p.u64(20),
                entry_count: p.u32(28),
                sum: p.u32(32),
            },
        )
    })();
    if let Err(r) = r {
        c.s_caught("CmdDisplayTransaction3", r, size);
    }
    // Signalled whatever happened.
    c.signal();
}

/// 8: presents a mapped surface. S class; the feature gate precedes the size check.
pub(super) fn swap_mapping<P: Platform>(c: &mut Command<'_, P>) {
    let size = c.payload.len();
    let r = (|| -> RaiseResult<_> {
        if !c.features().display_mapper_surface {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        c.check_size(SWAP_MAPPING)?;
        let p = c.payload;
        log::info!("FIFO CmdDisplaySwapMapping: port={}, mapping_id={:#x}", p.u32(0), p.u64(8));
        if p.u32(4) != 0 {
            raise!(RaiseKind::KernelOnly, KERNEL_ONLY);
        }
        let guest_display = c.guest_display(p.u32(0))?;
        guest_display.present_mapped_surface(
            p.u64(8),
            GammaTableArgs {
                virtual_offset: p.u64(16),
                mapped_length: p.u64(24),
                entry_count: p.u32(32),
                sum: p.u32(36),
            },
        )
    })();
    if let Err(r) = r {
        c.s_caught("CmdDisplaySwapMapping", r, size);
    }
    // Signalled whatever happened.
    c.signal();
}

/// 10: sets display properties, one record each. S class. Properties applied before a failing
/// one stay applied; the catch dumps the payload's start with the size left after the properties
/// consumed.
pub(super) fn set_properties<P: Platform>(c: &mut Command<'_, P>) {
    // The payload size, which the loop decrements
    let mut size = c.payload.len();
    let r = (|| -> RaiseResult<_> {
        if !c.features().display_set_properties {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        c.check_size(SET_PROPERTIES)?;
        let p = c.payload;
        let total_size = size;
        size = size.wrapping_sub(SET_PROPERTIES.to_usize());
        let guest_display = c.guest_display(p.u32(0))?;
        let mut cursor = SET_PROPERTIES.to_usize();
        // The count is read again at each test
        let mut index: u32 = 0;
        while index < p.u32(4) {
            let record = p.tail(cursor);
            // Dumps from the record
            check_size(c.fifo, record, size, PROPERTY)?;
            let property = record.u32(0);
            let word_count = record.u32(4);
            let (length, overflow) = mul_add_overflows(word_count, 4, PROPERTY);
            if overflow {
                raise!(
                    RaiseKind::Overflow,
                    format!(
                        "property {property} of {word_count} words overflows the payload of \
                         {total_size} bytes"
                    )
                );
            }
            check_size(c.fifo, record, size, length)?;
            guest_display
                .set_property(property.cast_signed(), record.records::<u32>(8, word_count))?;
            let consumed = length.to_usize();
            size = size.wrapping_sub(consumed);
            cursor = cursor.wrapping_add(consumed);
            index = index.wrapping_add(1);
        }
        Ok(())
    })();
    if let Err(r) = r {
        // The payload's start, with the decremented size
        c.s_caught("CmdSetProperty", r, size);
    }
    // Signalled whatever happened.
    c.signal();
}

/// 9: the display's sleep state; the feature gate precedes the size check, and the BOOL byte is
/// passed raw.
pub(super) fn sleep_state<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        if !c.features().display_sleep {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        c.check_size(SLEEP_STATE)?;
        let p = c.payload;
        let guest_display = c.guest_display(p.u32(0))?;
        guest_display.sleep_state(p.u8(4));
        c.signal();
        Ok(())
    });
}

/// 11: the compositor's parameters.
pub(super) fn compositor_parameters<P: Platform>(c: &mut Command<'_, P>) {
    c.f_class(|c, _| {
        if !c.features().supports_display_compositor_parameters {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        c.check_size(COMPOSITOR_PARAMETERS)?;
        let p = c.payload;
        let guest_display = c.guest_display(p.u32(0))?;
        guest_display.compositor_parameters(
            p.u64(8),
            DisplayCoord {
                x: p.u16(4),
                y: p.u16(6),
            },
            p.u8(16),
        );
        c.signal();
        Ok(())
    });
}

/// 12: the guest's ICC profile. S class; the length check precedes the lookup.
pub(super) fn set_guest_icc_profile<P: Platform>(c: &mut Command<'_, P>) {
    let size = c.payload.len();
    let r = (|| -> RaiseResult<_> {
        if !c.features().supports_display_set_guest_icc_profile {
            raise!(RaiseKind::Unsupported, UNSUPPORTED);
        }
        c.check_size(GUEST_ICC_PROFILE)?;
        let p = c.payload;
        let length = p.u32(4);
        let mapped_length = p.u64(16);
        if u64::from(length) > mapped_length {
            raise!(
                RaiseKind::Overflow,
                format!(
                    "ICC profile length {length:#x} is past its mapped length {mapped_length:#x}"
                )
            );
        }
        let guest_display = c.guest_display(p.u32(0))?;
        guest_display.set_guest_icc_profile_length(length, p.u64(8))
    })();
    if let Err(r) = r {
        c.s_caught("CmdDisplaySetGuestICCProfile", r, size);
    }
    // Signalled whatever happened.
    c.signal();
}
