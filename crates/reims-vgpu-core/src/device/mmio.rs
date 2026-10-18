//! The MMIO window: the register decode and the register reads and writes.
//!
//! ## Contract
//!
//! - The decode is `(offset - 0x1000) >> 2` truncated to 16 bits: offsets alias every 0x40000
//!   bytes, an unaligned offset rounds down, and an offset below 0x1000 wraps to indices
//!   0xfc00-0xffff, which no register answers (`tests::register_index_aliases_and_wraps`).
//! - No read or write takes a lock at this level. The registers the device also stores under a
//!   lock (the FIFO enable, the root FIFO's read offset, the root page, the EFI registers) are
//!   read from their staged view, so a register read from inside a host block the device calls
//!   under those locks answers at once (`tests::mmio_read_from_map_memory_block_does_not_block`,
//!   `tests::unlocked_registers_read_while_their_locks_are_held`).
//! - 0x1014 and 0x1018 clear on read; 0x102c is a boolean and is not cleared
//!   (`tests::interrupt_registers_clear_on_read_but_fault_does_not`).
//! - The EFI registers are stored with or without an EFI display
//!   (`tests::efi_registers_stored_without_efi_display`).

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::sync::atomic::Ordering;

use crate::device::{DeviceCore, DeviceDeath};
use crate::guest::abi::features::features_for_version;
use crate::num::Truncate;
use crate::platform::Platform;
use crate::raise::{RaiseResult, death::DeathKind};
use crate::rt::Hold;

/// `REIMS_VGPU_GUEST_DISPLAY_PORTS`, read once: a port count from 1 the guest reads at 0x122c
/// whatever its binary version. Unset (or not such a number), the guest reads Apple's answer: 1
/// below version 245.
///
/// x86-64 macOS's `AppleParavirtGPU` asks for binary version 6, the highest it knows (measured:
/// 0x1034 reads 6 on macOS 26.7, kext 15.0.0), and so reads one port; yet its control object
/// reads 0x122c as it starts and takes up to 8 ports (read from its code; docs/qemu.md, "Two
/// displays on x86-64"). Opt-in, so a guest's view of a device stays Apple's unless the host asks.
fn guest_display_ports_override() -> Option<u32> {
    static PORTS: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *PORTS.get_or_init(|| {
        std::env::var("REIMS_VGPU_GUEST_DISPLAY_PORTS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&ports| ports >= 1)
    })
}

/// Where the register window starts.
pub const MMIO_BASE: u64 = 0x1000;

/// The register index of an MMIO offset: `(offset - 0x1000) >> 2`, truncated to 16 bits.
///
/// Offsets alias every 0x40000 bytes, an unaligned offset rounds down, and an offset below 0x1000
/// wraps to indices 0xfc00-0xffff (none handled).
pub fn register_index(offset: u64) -> u16 {
    offset.wrapping_sub(MMIO_BASE).wrapping_shr(2).low_u16()
}

/// The registers, by index (offset `0x1000 + 4·index`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Register {
    /// 0x1000: the root FIFO's enable (read-write).
    FifoEnable = 0x00,
    /// 0x1004: the root FIFO's length, staged until the enable.
    FifoLength = 0x01,
    /// 0x1008: the root FIFO's written offset (its doorbell).
    FifoWritten = 0x02,
    /// 0x100c: the root FIFO's read offset.
    FifoRead = 0x03,
    /// 0x1010: the root FIFO's start offset, staged until the enable.
    FifoStart = 0x04,
    /// 0x1014: the display interrupt bits (read to clear).
    InterruptDisplay = 0x05,
    /// 0x1018: the stamp interrupt bits (read to clear).
    InterruptStamp = 0x06,
    /// 0x101c: the root page (read-write).
    RootPage = 0x07,
    /// 0x1020: wakes a child FIFO.
    WakeFifo = 0x08,
    /// 0x1024: resumes the root FIFO.
    ResumeFifo = 0x09,
    /// 0x1028: resumes a child FIFO.
    ResumeChildFifo = 0x0a,
    /// 0x102c: whether any FIFO is faulted.
    InterruptFault = 0x0b,
    /// 0x1030: the root FIFO's base page, staged until the enable.
    FifoBasePage = 0x0c,
    /// 0x1034: the guest's binary version (read-write).
    BinaryVersion = 0x0d,
    /// 0x1038: the guest's trace marker.
    TraceMarker = 0x0e,
    /// 0x103c: the root FIFO's fault offset.
    FifoFaultOffset = 0x0f,
    /// 0x1200: the EFI display's enable (read-write).
    EfiEnable = 0x80,
    /// 0x1204: the EFI display's mode count.
    EfiModeCount = 0x81,
    /// 0x1208: selects the EFI mode 0x120c reads.
    EfiModeSelect = 0x82,
    /// 0x120c: the selected EFI mode.
    EfiModeValue = 0x83,
    /// 0x1210: the EFI framebuffer's start (read-write).
    EfiStart = 0x84,
    /// 0x1214: the EFI framebuffer's length (read-write).
    EfiLength = 0x85,
    /// 0x1218: the EFI framebuffer's depth (read-write).
    EfiDepth = 0x86,
    /// 0x121c: the EFI framebuffer's mode (read-write).
    EfiMode = 0x87,
    /// 0x1220: rings a display port's doorbell.
    DisplayDoorbell = 0x88,
    /// 0x1224: the EFI framebuffer's stride alignment.
    EfiStrideAlignment = 0x89,
    /// 0x1228: the EFI framebuffer's stride (read-write).
    EfiStride = 0x8a,
    /// 0x122c: the display port count the guest sees.
    DisplayPortCount = 0x8b,
    /// 0x1234: the display ports' connection types the guest sees.
    DisplayPortConnectionTypes = 0x8d,
}

impl Register {
    /// The register at `index`, if one is there.
    pub fn decode(index: u16) -> Option<Register> {
        use Register::*;
        Some(match index {
            0x00 => FifoEnable,
            0x01 => FifoLength,
            0x02 => FifoWritten,
            0x03 => FifoRead,
            0x04 => FifoStart,
            0x05 => InterruptDisplay,
            0x06 => InterruptStamp,
            0x07 => RootPage,
            0x08 => WakeFifo,
            0x09 => ResumeFifo,
            0x0a => ResumeChildFifo,
            0x0b => InterruptFault,
            0x0c => FifoBasePage,
            0x0d => BinaryVersion,
            0x0e => TraceMarker,
            0x0f => FifoFaultOffset,
            0x80 => EfiEnable,
            0x81 => EfiModeCount,
            0x82 => EfiModeSelect,
            0x83 => EfiModeValue,
            0x84 => EfiStart,
            0x85 => EfiLength,
            0x86 => EfiDepth,
            0x87 => EfiMode,
            0x88 => DisplayDoorbell,
            0x89 => EfiStrideAlignment,
            0x8a => EfiStride,
            0x8b => DisplayPortCount,
            0x8d => DisplayPortConnectionTypes,
            _ => return None,
        })
    }
}

impl<P: Platform> DeviceCore<P> {
    /// A register read, after the decode. A write-only register
    /// or an index no register answers reads 0.
    pub(crate) fn mmio_read_register(&self, index: u16) -> u32 {
        let Some(register) = Register::decode(index) else {
            return 0;
        };
        let efi = self.displays.atomics();
        match register {
            Register::FifoEnable => self.root_fifo.atomics().enabled(),
            // 0 with no root FIFO.
            Register::FifoRead => {
                let fifo = self.root_fifo.atomics().root_fifo();
                fifo.and_then(|f| f.root_read()).unwrap_or(0)
            }
            Register::InterruptDisplay => self.interrupt_display.swap(0, Ordering::SeqCst),
            Register::InterruptStamp => self.interrupt_stamp.swap(0, Ordering::SeqCst),
            Register::RootPage => self
                .root_task
                .atomics()
                .root_page_number
                .load(Ordering::Relaxed),
            Register::InterruptFault => self.read_interrupt_fault(),
            // The signed version's bits.
            Register::BinaryVersion => self.binary_version.load(Ordering::Relaxed).cast_unsigned(),
            Register::FifoFaultOffset => self.fifo_fault_offset(),
            Register::EfiEnable => efi.enabled.load(Ordering::Relaxed),
            Register::EfiModeCount => self.get_efi_mode_count(),
            Register::EfiModeValue => self.get_efi_mode_value(),
            Register::EfiStart => efi.start.load(Ordering::Relaxed),
            Register::EfiLength => efi.length.load(Ordering::Relaxed),
            Register::EfiDepth => efi.depth.load(Ordering::Relaxed),
            Register::EfiMode => efi.mode.load(Ordering::Relaxed),
            Register::EfiStrideAlignment => self.get_efi_stride_alignment(),
            Register::EfiStride => efi.stride.load(Ordering::Relaxed),
            Register::DisplayPortCount => self.guest_display_port_count(),
            Register::DisplayPortConnectionTypes => self.guest_display_port_connection_types(),
            Register::FifoLength
            | Register::FifoWritten
            | Register::FifoStart
            | Register::WakeFifo
            | Register::ResumeFifo
            | Register::ResumeChildFifo
            | Register::FifoBasePage
            | Register::TraceMarker
            | Register::EfiModeSelect
            | Register::DisplayDoorbell => 0,
        }
    }

    /// A register write, after the decode, under the host's call's `hold`. A read-only register
    /// or an index no register answers ignores it.
    pub(crate) fn mmio_write_register(&self, hold: Hold<'_>, index: u16, value: u32) {
        let Some(register) = Register::decode(index) else {
            return;
        };
        match register {
            // A nonzero value other than the current nonzero one reaches the enable's assert; an
            // equal value does nothing.
            Register::FifoEnable => {
                let enabled = self.root_fifo.atomics().enabled();
                if value != enabled {
                    if value != 0 {
                        self.enable_fifo();
                    } else {
                        self.teardown_fifos();
                    }
                }
            }
            Register::FifoLength => self.set_fifo_length(value),
            Register::FifoWritten => self.set_fifo_written(value),
            Register::FifoStart => self.set_fifo_start(value),
            Register::RootPage => self.set_root_page(hold, value),
            Register::WakeFifo => self.wake_fifo(value),
            Register::ResumeFifo => self.resume_fifo(),
            Register::ResumeChildFifo => self.resume_child_fifo(value),
            Register::FifoBasePage => self.set_fifo_base_page(value),
            Register::BinaryVersion => self.set_binary_version(value),
            // The guest's trace marker: accepted, and nothing is traced.
            Register::TraceMarker => {}
            Register::EfiEnable => self.set_efi_display(value),
            Register::EfiModeSelect => self.set_efi_mode_select(value),
            Register::EfiStart => self.set_efi_framebuffer_start(value),
            Register::EfiLength => self.set_efi_framebuffer_length(value),
            Register::EfiDepth => self.set_efi_framebuffer_depth(value),
            Register::EfiMode => self.set_efi_framebuffer_mode(value),
            Register::DisplayDoorbell => self.ring_display_doorbell_at_port(hold, value),
            Register::EfiStride => self.set_efi_framebuffer_stride(value),
            Register::FifoRead
            | Register::InterruptDisplay
            | Register::InterruptStamp
            | Register::InterruptFault
            | Register::FifoFaultOffset
            | Register::EfiModeCount
            | Register::EfiModeValue
            | Register::EfiStrideAlignment
            | Register::DisplayPortCount
            | Register::DisplayPortConnectionTypes => {}
        }
    }

    /// The display port count the guest reads.
    ///
    /// Below version 245 every read answers `count != 0`, unless `REIMS_VGPU_GUEST_DISPLAY_PORTS`
    /// ([`guest_display_ports_override`]) is set: then every read answers that many, at most the
    /// device's count, whatever the version.
    pub fn guest_display_port_count(&self) -> u32 {
        let mut count = self.display_port_count.load(Ordering::Relaxed);
        if let Some(ports) = guest_display_ports_override() {
            count = count.min(ports);
            log::debug!("display port count reads {count}: REIMS_VGPU_GUEST_DISPLAY_PORTS");
        } else if !self.features.load().supports_multiple_displays {
            count = u32::from(count != 0);
            log::debug!("display port count reads {count}: the guest's version has one display");
        }
        count
    }

    /// The display ports' connection types the guest reads.
    ///
    /// Masked `& 3` below version 245.
    pub fn guest_display_port_connection_types(&self) -> u32 {
        let types = self.display_port_connection_types.load(Ordering::Relaxed);
        if self.features.load().supports_multiple_displays {
            types
        } else {
            types & 3
        }
    }

    /// Negotiates and stores the guest's binary version (`features::set_binary_version`), then
    /// the features for the version stored and, when those have no EFI display, turns EFI off.
    /// No lock: the features are replaced while FIFO threads read them, word by word.
    ///
    /// `features::step_binary_version` and `clamp_binary_version` decide the version stored. A
    /// device feature level of 5 or more dies here.
    pub fn set_binary_version(&self, binary_version: u32) {
        let stored = crate::guest::abi::features::set_binary_version(
            binary_version,
            self.caps.feature_level,
            |v| self.binary_version.store(v, Ordering::Relaxed),
        );
        let Some(stored) = stored else {
            return;
        };
        let features = features_for_version(stored);
        self.features.store(&features);
        if !features.efi_display {
            self.set_efi_display(0);
        }
    }

    /// Rings the doorbell of the port's guest display, under a handler: a port out of range or
    /// without a guest display is logged and dropped. The guest display is held for the doorbell
    /// alone.
    pub fn ring_display_doorbell_at_port(&self, hold: Hold<'_>, port: u32) {
        let rang = (|| -> RaiseResult<_> {
            self.get_guest_display_at_port(u64::from(port))?.doorbell();
            Ok(())
        })();
        if let Err(raised) = rang {
            log::warn!(
                "display doorbell dropped: {}",
                raised.detail().unwrap_or_default()
            );
            hold.land(raised);
        }
    }

    /// Forwards the mode select to the EFI display, if there is one, under the displays' lock.
    /// No register keeps it.
    pub fn set_efi_mode_select(&self, mode: u32) {
        let _held = self.displays.lock();
        if let Some(efi) = self.displays.atomics().efi_display() {
            efi.set_mode_select(mode);
        }
    }

    /// The EFI display's mode count, 0 without one.
    pub fn get_efi_mode_count(&self) -> u32 {
        let _held = self.displays.lock();
        self.displays
            .atomics()
            .efi_display()
            .map_or(0, |e| e.get_mode_count())
    }

    /// The EFI display's selected mode, 0 without one.
    pub fn get_efi_mode_value(&self) -> u32 {
        let _held = self.displays.lock();
        self.displays
            .atomics()
            .efi_display()
            .map_or(0, |e| e.get_mode_value())
    }

    /// The EFI display's stride alignment, 0 without one.
    pub fn get_efi_stride_alignment(&self) -> u32 {
        let _held = self.displays.lock();
        self.displays
            .atomics()
            .efi_display()
            .map_or(0, |e| e.get_stride_alignment())
    }

    /// Forwards the framebuffer start to the EFI display, if there is one, and stores it.
    pub fn set_efi_framebuffer_start(&self, start: u32) {
        let mut d = self.displays.lock();
        if let Some(efi) = self.displays.atomics().efi_display() {
            efi.set_framebuffer_start(start);
        }
        d.efi.start = start;
    }

    /// Forwards the framebuffer length to the EFI display, if there is one (it may map its
    /// framebuffer here), and stores it.
    pub fn set_efi_framebuffer_length(&self, length: u32) {
        let mut d = self.displays.lock();
        if let Some(efi) = self.displays.atomics().efi_display() {
            efi.set_framebuffer_length(length);
        }
        d.efi.length = length;
    }

    /// Forwards the framebuffer depth to the EFI display, if there is one, and stores it.
    pub fn set_efi_framebuffer_depth(&self, depth: u32) {
        let mut d = self.displays.lock();
        if let Some(efi) = self.displays.atomics().efi_display() {
            efi.set_framebuffer_depth(depth);
        }
        d.efi.depth = depth;
    }

    /// Forwards the framebuffer mode to the EFI display, if there is one, and stores it.
    pub fn set_efi_framebuffer_mode(&self, mode: u32) {
        let mut d = self.displays.lock();
        if let Some(efi) = self.displays.atomics().efi_display() {
            efi.set_framebuffer_mode(mode);
        }
        d.efi.mode = mode;
    }

    /// Forwards the framebuffer stride to the EFI display, if there is one, and stores it.
    pub fn set_efi_framebuffer_stride(&self, stride: u32) {
        let mut d = self.displays.lock();
        if let Some(efi) = self.displays.atomics().efi_display() {
            efi.set_framebuffer_stride(stride);
        }
        d.efi.stride = stride;
    }
}

impl DeviceDeath {
    /// The kind of a register-path death (`device/{mmio,root,fifos,waits}.rs`).
    pub(crate) fn register_kind(self) -> DeathKind {
        match self {
            DeviceDeath::PhysicalReadCopy
            | DeviceDeath::StampWriteWithoutMemory
            | DeviceDeath::StampReadWithoutMemory
            | DeviceDeath::ReportWithoutStampMemory
            | DeviceDeath::DeletedNodeWithoutRoot => DeathKind::NullDeref,
            DeviceDeath::PastChildFifos | DeviceDeath::PastWaitingInfo => DeathKind::OutOfBounds,
            DeviceDeath::ReportOffQueue => DeathKind::DispatchMisuse,
            DeviceDeath::NilBlock(_) => DeathKind::NilBlock,
            _ => DeathKind::Assert,
        }
    }

    /// The dotted name of a register-path death.
    pub(crate) fn register_name(self) -> &'static str {
        match self {
            DeviceDeath::FifoAlreadyEnabled => "device.fifo.already_enabled",
            DeviceDeath::ChildInSlotZero => "device.fifo.child_in_slot_zero",
            DeviceDeath::TeardownWithoutRootNode => "device.root.teardown_without_node",
            DeviceDeath::TeardownWithoutDestroyTask => "device.root.teardown_without_destroy",
            DeviceDeath::UnalignedRangeMapping => "device.mapping.unaligned",
            DeviceDeath::UnmapOutsideRootTask => "device.mapping.outside_root_task",
            DeviceDeath::PhysicalReadCopy => "device.read.copy",
            DeviceDeath::PastChildFifos => "device.fifo.past_slots",
            DeviceDeath::PastWaitingInfo => "device.wait.past_entries",
            DeviceDeath::CreateWhileWaiting => "device.fifo.create_while_waiting",
            DeviceDeath::StampWriteWithoutMemory => "device.stamp.write_without_memory",
            DeviceDeath::StampReadWithoutMemory => "device.stamp.read_without_memory",
            DeviceDeath::ReportWithoutStampMemory => "device.report.without_stamp_memory",
            DeviceDeath::DeletedNodeWithoutRoot => "device.fifo.deleted_node_without_root",
            DeviceDeath::WaiterPastMax => "device.wait.waiter_past_max",
            DeviceDeath::WaiterBitTaken => "device.wait.waiter_bit_taken",
            DeviceDeath::WaiterBusy => "device.wait.waiter_busy",
            DeviceDeath::ReportStampPastFifos => "device.report.stamp_past_fifos",
            DeviceDeath::ReportRootWithoutFifo => "device.report.root_without_fifo",
            DeviceDeath::ReportStampMismatch => "device.report.stamp_mismatch",
            DeviceDeath::ReportOffQueue => "device.report.off_queue",
            DeviceDeath::NilBlock(block) => block,
            DeviceDeath::MappingWhileLeavingSuspend => "device.mapping_while_leaving_suspend",
            DeviceDeath::ChildFifoInSlotZero => "device.child_fifo_in_slot_zero",
            DeviceDeath::SuspendNotBegun => "device.suspend_not_begun",
            DeviceDeath::ResumedRootFifoMissing => "device.resumed_root_fifo_missing",
            DeviceDeath::ResumeWalkLeftRootPage => "device.resume_walk_left_root_page",
            DeviceDeath::RootNodeFault => "device.root_node_fault",
            DeviceDeath::DeviceInfoWrite => "device.device_info_write",
            DeviceDeath::TaskRootWord => "device.task_root_word",
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::device::root::rig::{rig, rig_with};
    use crate::testkit::capture::expect_death;

    // The index `(offset - 0x1000) >> 2` in 16 bits aliases, wraps below 0x1000, and rounds
    // unaligned offsets down.
    #[test]
    fn register_index_aliases_and_wraps() {
        assert_eq!(register_index(0x1000), 0);
        assert_eq!(register_index(0x1003), 0);
        let root_page = Some(Register::RootPage);
        assert_eq!(Register::decode(register_index(0x101c)), root_page);
        assert_eq!(Register::decode(register_index(0x4_101c)), root_page);
        assert_eq!(
            Register::decode(register_index(0x1234)),
            Some(Register::DisplayPortConnectionTypes)
        );
        assert_eq!(register_index(0x0ffc), 0xffff);
        assert_eq!(register_index(0), 0xfc00);
        assert_eq!(Register::decode(0xffff), None);
        assert_eq!(Register::decode(0x8c), None);
    }

    /// Every register decodes from its own index.
    #[test]
    fn decode_is_the_discriminant() {
        for i in 0..=u16::MAX {
            if let Some(r) = Register::decode(i) {
                assert_eq!(r as u16, i);
            }
        }
    }

    #[test]
    fn root_page_register_round_trip_and_alias() {
        let r = rig();
        let m = r.mark();
        r.device.mmio_write(Hold::NONE, 0x4_101c, 1);
        assert_eq!(r.host_sels(m), ["createTask", "mapMemory"]);
        assert_eq!(r.device.mmio_read(0x101c), 1);
        assert_eq!(r.device.mmio_read(0x0ffc), 0);
    }

    // 0x1014 / 0x1018 clear on read.
    #[test]
    fn interrupt_registers_clear_on_read_but_fault_does_not() {
        let r = rig();
        r.device.interrupt_display.store(6, Ordering::SeqCst);
        r.device.interrupt_stamp.store(9, Ordering::SeqCst);
        r.device.interrupt_fault_mask.store(0x100, Ordering::SeqCst);
        assert_eq!(r.device.mmio_read(0x1014), 6);
        assert_eq!(r.device.mmio_read(0x1014), 0);
        assert_eq!(r.device.mmio_read(0x1018), 9);
        assert_eq!(r.device.mmio_read(0x1018), 0);
        assert_eq!(r.device.mmio_read(0x102c), 1);
        assert_eq!(r.device.mmio_read(0x102c), 1);
    }

    #[test]
    fn fifo_registers_through_mmio() {
        let r = rig();
        r.device.mmio_write(Hold::NONE, 0x101c, 1);
        r.device.mmio_write(Hold::NONE, 0x1030, 2);
        r.device.mmio_write(Hold::NONE, 0x1004, 0x4000);
        r.device.mmio_write(Hold::NONE, 0x1010, 0x400);
        r.device.mmio_write(Hold::NONE, 0x1000, 1);
        assert_eq!(r.device.mmio_read(0x1000), 1);
        // An equal value does nothing.
        r.device.mmio_write(Hold::NONE, 0x1000, 1);
        assert_eq!(r.device.mmio_read(0x103c), u32::MAX);
        r.device.mmio_write(Hold::NONE, 0x1000, 0);
        assert_eq!(r.device.mmio_read(0x1000), 0);
        // No root FIFO.
        assert_eq!(r.device.mmio_read(0x103c), 0);
        assert_eq!(r.device.mmio_read(0x100c), 0);
    }

    // 0x1000: a nonzero value other than the current nonzero value dies at the enable's assert
    // (testing.md 7.4).
    #[test]
    fn fifo_enable_other_nonzero_asserts() {
        let r = rig();
        r.with_root_fifo(1, 2);
        let d = r.device.clone();
        expect_death(DeviceDeath::FifoAlreadyEnabled.site(), move || {
            d.mmio_write(Hold::NONE, 0x1000, 2)
        });
    }

    // 0x122c reads `count != 0` below 245.
    #[test]
    fn display_port_registers_below_245() {
        let r = rig();
        r.device.display_port_count.store(3, Ordering::SeqCst);
        r.device
            .display_port_connection_types
            .store(0b1110, Ordering::SeqCst);
        let mut f = r.device.features.load();
        f.supports_multiple_displays = false;
        r.device.features.store(&f);
        assert_eq!(r.device.mmio_read(0x122c), 1);
        assert_eq!(r.device.mmio_read(0x122c), 1);
        assert_eq!(r.device.mmio_read(0x1234), 0b10);
        f.supports_multiple_displays = true;
        r.device.features.store(&f);
        assert_eq!(r.device.mmio_read(0x122c), 3);
        assert_eq!(r.device.mmio_read(0x1234), 0b1110);
    }

    // EFI: every EFI register is stored and read back without an EFI display.
    #[test]
    fn efi_registers_stored_without_efi_display() {
        let r = rig();
        r.device.mmio_write(Hold::NONE, 0x1210, 0x10);
        r.device.mmio_write(Hold::NONE, 0x1214, 0x20);
        r.device.mmio_write(Hold::NONE, 0x1218, 0x30);
        r.device.mmio_write(Hold::NONE, 0x121c, 0x40);
        r.device.mmio_write(Hold::NONE, 0x1228, 0x50);
        r.device.mmio_write(Hold::NONE, 0x1208, 3);
        assert_eq!(
            [0x1210, 0x1214, 0x1218, 0x121c, 0x1228].map(|o| r.device.mmio_read(o)),
            [0x10, 0x20, 0x30, 0x40, 0x50]
        );
        assert_eq!(r.device.mmio_read(0x1204), 0);
        assert_eq!(r.device.mmio_read(0x120c), 0);
        assert_eq!(r.device.mmio_read(0x1224), 0);
    }

    /// The doorbell's raise is caught, and its line says why
    #[test]
    fn a_doorbell_on_a_bad_port_logs_its_raise() {
        let r = rig();
        let lines = crate::device::test_support::Lines::capture();
        r.device.mmio_write(Hold::NONE, 0x1220, 7);
        assert_eq!(
            lines.all(),
            ["display doorbell dropped: display port 7 is past the last port, 0"]
        );
    }

    /// The unlocked register reads answer from inside a host block the device calls under the
    /// root node and root task locks (the root page's `mapMemory`).
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time wait
    fn mmio_read_from_map_memory_block_does_not_block() {
        use crate::device::DeviceCore;
        use crate::testkit::FakePlatform;
        use std::sync::{Arc, OnceLock, Weak};
        let slot: Arc<OnceLock<Weak<DeviceCore<FakePlatform>>>> = Arc::new(OnceLock::new());
        let seen: Arc<parking_lot::Mutex<Vec<(u64, u32)>>> = Arc::default();
        let (s2, seen2) = (slot.clone(), seen.clone());
        let r = rig_with(move |b| {
            let inner = b.map_memory.clone().unwrap();
            b.map_memory = Some(Arc::new(move |t, n, o, ro, ranges| {
                if let Some(d) = s2.get().and_then(Weak::upgrade) {
                    for off in [0x101c_u64, 0x1000] {
                        let v = d.mmio_read(off);
                        seen2.lock().push((off, v));
                    }
                }
                inner(t, n, o, ro, ranges)
            }));
        });
        slot.set(Arc::downgrade(&r.device)).ok().unwrap();
        let d = r.device.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            d.mmio_write(Hold::NONE, 0x101c, 1);
            tx.send(()).unwrap();
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_secs(3)).is_ok(),
            "an MMIO read of 0x101c from the host's mapMemory block deadlocked"
        );
        assert_eq!(seen.lock().as_slice(), &[(0x101c, 0), (0x1000, 0)]);
    }

    /// 0x1000 and 0x100c read from the mapMemory block the root FIFO's creation calls under the
    /// root FIFO's lock.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time wait
    fn mmio_read_fifo_regs_from_map_memory_during_enable() {
        use crate::device::DeviceCore;
        use crate::testkit::FakePlatform;
        use std::sync::atomic::AtomicBool;
        use std::sync::{Arc, OnceLock, Weak};
        let slot: Arc<OnceLock<Weak<DeviceCore<FakePlatform>>>> = Arc::new(OnceLock::new());
        let armed = Arc::new(AtomicBool::new(false));
        let (s2, a2) = (slot.clone(), armed.clone());
        let r = rig_with(move |b| {
            let inner = b.map_memory.clone().unwrap();
            b.map_memory = Some(Arc::new(move |t, n, o, ro, ranges| {
                if a2.load(Ordering::SeqCst)
                    && let Some(d) = s2.get().and_then(Weak::upgrade)
                {
                    let _ = d.mmio_read(0x1000);
                    let _ = d.mmio_read(0x100c);
                }
                inner(t, n, o, ro, ranges)
            }));
        });
        slot.set(Arc::downgrade(&r.device)).ok().unwrap();
        r.device.set_root_page(Hold::NONE, 1);
        r.device.set_fifo_base_page(2);
        r.device.set_fifo_length(0x4000);
        r.device.set_fifo_start(0x400);
        armed.store(true, Ordering::SeqCst);
        let d = r.device.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            d.mmio_write(Hold::NONE, 0x1000, 1);
            tx.send(()).unwrap();
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_secs(3)).is_ok(),
            "an MMIO read of 0x1000 from the host's mapMemory block deadlocked"
        );
    }

    /// The root page's write completes when its mapMemory block reads 0x101c after mapping.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time wait
    fn map_memory_block_reading_root_page_register_returns() {
        use crate::device::DeviceCore;
        use crate::testkit::FakePlatform;
        use parking_lot::Mutex as StdMutex;
        use std::sync::{Arc, Weak, mpsc};
        use std::time::Duration;
        type Slot = Arc<StdMutex<Option<Weak<DeviceCore<FakePlatform>>>>>;
        let slot: Slot = Arc::new(StdMutex::new(None));
        let s2 = slot.clone();
        let r = rig_with(move |b| {
            let inner = b.map_memory.clone().unwrap();
            b.map_memory = Some(Arc::new(move |t, n, o, ro, ranges| {
                let ok = inner(t, n, o, ro, ranges);
                if let Some(d) = s2.lock().as_ref().and_then(Weak::upgrade) {
                    let _ = d.mmio_read(0x101c);
                }
                ok
            }));
        });
        *slot.lock() = Some(Arc::downgrade(&r.device));
        let dev = r.device.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            dev.set_root_page(Hold::NONE, 1);
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(Duration::from_secs(3)).is_ok(),
            "set_root_page hung"
        );
    }

    /// The enable completes when its mapMemory block reads 0x1000, which answers 0 (the enable
    /// is stored after the mapping).
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time wait
    fn map_memory_block_reading_fifo_enable_during_enable_returns() {
        use crate::device::DeviceCore;
        use crate::testkit::FakePlatform;
        use parking_lot::Mutex as StdMutex;
        use std::sync::{Arc, Weak, mpsc};
        use std::time::Duration;
        type Slot = Arc<StdMutex<Option<Weak<DeviceCore<FakePlatform>>>>>;
        let slot: Slot = Arc::new(StdMutex::new(None));
        let s2 = slot.clone();
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let seen2 = seen.clone();
        let r = rig_with(move |b| {
            let inner = b.map_memory.clone().unwrap();
            b.map_memory = Some(Arc::new(move |t, n, o, ro, ranges| {
                let ok = inner(t, n, o, ro, ranges);
                if let Some(d) = s2.lock().as_ref().and_then(Weak::upgrade) {
                    seen2.lock().push(d.mmio_read(0x1000));
                }
                ok
            }));
        });
        r.device.set_root_page(Hold::NONE, 1);
        r.device.set_fifo_base_page(2);
        r.device.set_fifo_length(0x4000);
        r.device.set_fifo_start(0x400);
        *slot.lock() = Some(Arc::downgrade(&r.device));
        let dev = r.device.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            dev.mmio_write(Hold::NONE, 0x1000, 1);
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(Duration::from_secs(3)).is_ok(),
            "enable hung"
        );
        assert_eq!(*seen.lock(), vec![0]);
        assert_eq!(r.device.mmio_read(0x1000), 1);
    }

    /// The EFI registers and 0x1000 read without the displays' or the root FIFO's lock while
    /// another thread holds them.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time wait
    fn unlocked_registers_read_while_their_locks_are_held() {
        let r = rig();
        r.device.displays.lock().efi.start = 0x1234;
        let dev = r.device.clone();
        let _d = r.device.displays.lock();
        let _f = r.device.root_fifo.lock();
        let _t = r.device.root_task.lock();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let v = [
                dev.mmio_read(0x1210),
                dev.mmio_read(0x1000),
                dev.mmio_read(0x100c),
                dev.mmio_read(0x101c),
                dev.mmio_read(0x1200),
            ];
            let _ = tx.send(v);
        });
        let v = rx
            .recv_timeout(std::time::Duration::from_secs(3))
            .expect("an unlocked register read waited on a lock");
        assert_eq!(v, [0x1234, 0, 0, 0, 0]);
    }

    /// The doorbell opens no pool of its own (nothing in it is autoreleased), and the guest
    /// display it looks up is let go when it returns; a lookup's raise is caught and logged.
    #[test]
    fn ring_display_doorbell_opens_no_pool() {
        use crate::rt::pool::test_native::{events, with_test_native};
        let r = rig();
        r.device.set_root_page(Hold::NONE, 1);
        (|| -> RaiseResult<_> {
            r.device.set_shared_display_state_page(0x10, 0, false)?;
            Ok(())
        })()
        .unwrap();
        let guest_display = r.device.get_guest_display_at_port(0).unwrap();
        let refs = std::sync::Arc::strong_count(&guest_display);
        let mut got = (Vec::new(), Vec::new());
        with_test_native(|| {
            r.device.ring_display_doorbell_at_port(Hold::NONE, 0);
            got.0 = events();
            r.device.ring_display_doorbell_at_port(Hold::NONE, 5);
            got.1 = events();
        });
        assert!(got.0.is_empty(), "{:?}", got.0);
        // The guest display's lookup was let go.
        assert_eq!(std::sync::Arc::strong_count(&guest_display), refs);
        // Port 5 is out of range: the lookup's raise is caught, with no pool either.
        assert!(got.1.is_empty(), "{:?}", got.1);
    }
}
