//! The guest display: the guest's side of a display port. One per port, owned by the device, over
//! the guest page the guest gave for it ([`page`](crate::display::page) has its offsets). Its
//! client is the host's display on the port, reached unretained. (docs/testing.md calls it the
//! guest display too; a few of the suite's scenario labels call it the nub.)
//!
//! By file: the connect/ack machine, the client, pause and restore (here), the presents, the
//! gamma table, the colour matrix, the properties and sleep ([`present`]), the cursor and the
//! doorbell ([`cursor`]), and what the guest display copies into the page from the display, its
//! modes and the guest's colour space ([`settings`]).
//!
//! **The page is guest memory.** Every field is read or written at the moment the guest display
//! touches it, one access of the field's width, and the guest's bytes are never normalised: the ack
//! pending byte is tested with bit 0 by [`GuestDisplay::ack`] and `== 1` by the state machine,
//! `asleep` with bit 0 by the presents and `== 1` by the sleep block, and the cursor show bytes
//! are stored, compared and handed on raw.
//!
//! **Locks.** `guest_display_mutex` guards [`GuestDisplayLocked`]; a method that runs with it held
//! takes its contents. What is read without it while the same thread may hold it lives in
//! [`GuestDisplayUnlocked`], written under the lock: the display reads the guest colour space and
//! the live-resize latch inside the guest display's own present and doorbell, the cursor position
//! and the protection options come from the display under its guest display pointer lock only, and
//! a dispatched block reads the client when it runs.
//!
//! **Ordering contracts** (testing.md 7):
//! - The state machine writes the state word after the interrupt it raises
//!   (`state_machine::state0_connects`).
//! - A present runs under the guest display's mutex, then the display's; the display's
//!   `set_mode_list` takes the guest display's only after it has let the display's go
//!   (`tests::set_mode_list_while_a_present_holds_the_guest_display`).
//! - The device sets the client under its displays lock, then the guest display's; the display's
//!   end detaches on the thread of the host's last release, waiting there for the guest display's
//!   mutex (`device::displays`'s `shared_state_page_unretained_client::
//!   host_release_during_opcode_1s_set_guest_display_ends_it_on_the_host_thread`).
//! - A block the guest display dispatches reads the client's queue again, and the client and its
//!   handler when it runs (`cursor_show::detached_before_the_block_dies`); the cursor show byte
//!   goes to the handler raw (`cursor_show::raw_byte`).
//! - The client is +0: no guest display operation keeps the display alive, and a block drops the
//!   core before it calls a host handler, so a handler's last release runs the display's end inside
//!   the call (`handler_releases_the_display::cursor_show_handler`).

/// Dies unless the `width` bytes at `colorMatrices[matrix][row][col]` lie inside the page
/// ([`page::color_matrix_in_page`](crate::display::page::color_matrix_in_page)): on x86-64 an
/// access past the 4 KiB page ends the process at that access, before it is made.
#[cfg(target_arch = "x86_64")]
macro_rules! color_matrix_access {
    ($matrix:expr, $row:expr, $col:expr, $width:expr) => {
        $crate::fatal_unless!(
            $crate::display::page::color_matrix_in_page($matrix, $row, $col, $width),
            $crate::display::guest_display::DeathKind::ColorMatrixOutsidePage
        )
    };
}

/// Nothing on arm64: the 16 KiB page holds every matrix, so no access leaves it.
#[cfg(not(target_arch = "x86_64"))]
macro_rules! color_matrix_access {
    ($matrix:expr, $row:expr, $col:expr, $width:expr) => {};
}

mod cursor;
mod present;
mod settings;
#[cfg(test)]
mod tests;

pub use cursor::CursorGlyphArgs;

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::device::DeviceCore;
use crate::display::mode::DisplayMode;
use crate::display::page as sp;
use crate::display::{Display, DisplayCore, GammaTableArgs, ModeList};
use crate::guest::{GuestPage, GuestPtr};
use crate::host::{HostPtr, HostQueue};
use crate::memory::mapping::Mapping;
use crate::num::Truncate;
use crate::platform::Platform;
use crate::raise::{DeathId, Kept, OnFault, Raise, RaiseId, RaiseResult, keep_forever};
use crate::rt::{Held, Hold};
use crate::seams::links::KernelTaskRef;

/// `field1c`'s value at init. The name is ours: the field has no known meaning.
const FIELD_1C_INIT: u32 = 1024;

/// The page's interrupt bits (names are ours, after what each bit signals: present complete,
/// connect, disconnect, update).
const INTERRUPT_PRESENT: u32 = 2;
/// Connect.
const INTERRUPT_CONNECT: u32 = 4;
/// Disconnect.
const INTERRUPT_DISCONNECT: u32 = 8;
/// Update.
const INTERRUPT_UPDATE: u32 = 12;

/// What a guest display raises. The FIFO's display commands catch it and fault the command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// The device could not map the guest's page.
    MapPage,
    /// An ack with none pending.
    AckNothingPending,
    /// An ack of another number than the one expected.
    AckUnexpected,
    /// A gamma table over the entry limit or longer than its mapping.
    GammaTable,
    /// A present of a resource that is not a backing (it answers no IOSurface).
    NotABacking,
    /// A present of a backing with no texture.
    BackingWithoutTexture,
    /// A present of a surface whose pixel format the guest display does not present.
    PixelFormat,
    /// A present whose surface asks for protection the page does not allow.
    ProtectionRequirements,
    /// A mapped present with no kernel task.
    NoKernelTask,
    /// A mapped present of a mapping id the mapper has no surface for.
    MappingId,
    /// A mapped present whose surface's bytes per pixel are not its format's.
    BytesPerPixel,
    /// A colour matrix property of another word count than twelve.
    PropertyWordCount,
    /// A property the guest display does not know.
    UnknownProperty,
    /// A cursor glyph whose mapped length is past the largest local task.
    CursorBuffer,
    /// A cursor glyph whose sum is not the one the guest sent.
    CursorCorrupt,
    /// A cursor glyph of bad dimensions, stride or mapped length.
    CursorParameters,
    /// The host made no image for the cursor glyph.
    CursorImage,
}

impl RaiseKind {
    /// The dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            RaiseKind::MapPage => "guest_display.init.map_page",
            RaiseKind::AckNothingPending => "guest_display.ack.nothing_pending",
            RaiseKind::AckUnexpected => "guest_display.ack.unexpected",
            RaiseKind::GammaTable => "guest_display.gamma.invalid",
            RaiseKind::NotABacking => "guest_display.present.not_a_backing",
            RaiseKind::BackingWithoutTexture => "guest_display.present.no_texture",
            RaiseKind::PixelFormat => "guest_display.present.pixel_format",
            RaiseKind::ProtectionRequirements => "guest_display.present.protection",
            RaiseKind::NoKernelTask => "guest_display.present.no_kernel_task",
            RaiseKind::MappingId => "guest_display.present.mapping_id",
            RaiseKind::BytesPerPixel => "guest_display.present.bytes_per_pixel",
            RaiseKind::PropertyWordCount => "guest_display.property.word_count",
            RaiseKind::UnknownProperty => "guest_display.property.unknown",
            RaiseKind::CursorBuffer => "guest_display.cursor.buffer",
            RaiseKind::CursorCorrupt => "guest_display.cursor.corrupt",
            RaiseKind::CursorParameters => "guest_display.cursor.parameters",
            RaiseKind::CursorImage => "guest_display.cursor.image",
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<RaiseKind> for RaiseId {
    fn from(k: RaiseKind) -> RaiseId {
        RaiseId::GuestDisplay(k)
    }
}

/// Where a guest display ends the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A page access through a NULL page (unreachable: `new` raises before a guest display has
    /// none).
    NullPage,
    /// The page reached while the guest display is paused.
    PageWhilePaused,
    /// A display command, restore or settings update while the guest display is paused.
    CommandWhilePaused,
    /// A client set while another is.
    ClientAlreadySet,
    /// The protection options set with the device gone.
    ProtectionWithoutDevice,
    /// A present with no kernel task (the device gone).
    PresentWithoutKernelTask,
    /// A present of a backing with no IOSurface.
    PresentWithoutSurface,
    /// A mapped present whose texture Metal refused.
    MappedTextureRefused,
    /// An RGhA present with the device gone (its features read through NULL).
    RghaWithoutDevice,
    /// A dispatch to the client's queue after it was let go (`dispatch_async(NULL)`).
    QueueGone,
    /// The cursor show block found no handler when it ran (a call through nil).
    ShowHandlerGone,
    /// The cursor move block found no handler when it ran.
    MoveHandlerGone,
    /// The cursor glyph block found no handler when it ran.
    GlyphHandlerGone,
    /// The host's cursor image has no bitmap buffer.
    GlyphBufferNull,
    /// The glyph handler was let go between the two reads of it.
    GlyphHandlerRaced,
    /// The modes synced with no mode list set.
    NoModeList,
    /// The guest's mode count is above the page's 128.
    ModeCountTooLarge,
    /// The settings synced with no client.
    SettingsWithoutClient,
    /// A pause of a paused guest display.
    PauseTwice,
    /// An unpause of a running guest display.
    UnpauseRunning,
    /// A restore begun twice.
    WillRestoreTwice,
    /// A restore finished that was not begun.
    FinishRestoreUnexpected,
    /// A colour-matrix access past the page: the guest's matrix index is past the matrices a
    /// 4 KiB page holds. x86-64 only: a 16 KiB page holds every index.
    #[cfg(target_arch = "x86_64")]
    ColorMatrixOutsidePage,
}

impl DeathKind {
    /// The dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::NullPage => "guest_display.page.null",
            DeathKind::PageWhilePaused => "guest_display.page.paused",
            DeathKind::CommandWhilePaused => "guest_display.command.paused",
            DeathKind::ClientAlreadySet => "guest_display.client.already_set",
            DeathKind::ProtectionWithoutDevice => "guest_display.protection.no_device",
            DeathKind::PresentWithoutKernelTask => "guest_display.present.no_kernel_task",
            DeathKind::PresentWithoutSurface => "guest_display.present.no_surface",
            DeathKind::MappedTextureRefused => "guest_display.present.texture_refused",
            DeathKind::RghaWithoutDevice => "guest_display.present.rgha_no_device",
            DeathKind::QueueGone => "guest_display.dispatch.queue_gone",
            DeathKind::ShowHandlerGone => "guest_display.cursor.show_handler_gone",
            DeathKind::MoveHandlerGone => "guest_display.cursor.move_handler_gone",
            DeathKind::GlyphHandlerGone => "guest_display.cursor.glyph_handler_gone",
            DeathKind::GlyphBufferNull => "guest_display.cursor.glyph_buffer_null",
            DeathKind::GlyphHandlerRaced => "guest_display.cursor.glyph_handler_raced",
            DeathKind::NoModeList => "guest_display.modes.no_list",
            DeathKind::ModeCountTooLarge => "guest_display.modes.count_too_large",
            DeathKind::SettingsWithoutClient => "guest_display.settings.no_client",
            DeathKind::PauseTwice => "guest_display.pause.twice",
            DeathKind::UnpauseRunning => "guest_display.unpause.running",
            DeathKind::WillRestoreTwice => "guest_display.restore.twice",
            DeathKind::FinishRestoreUnexpected => "guest_display.restore.unexpected",
            #[cfg(target_arch = "x86_64")]
            DeathKind::ColorMatrixOutsidePage => "guest_display.color_matrix.outside_page",
        }
    }

    /// The kind of death.
    pub(crate) const fn class(self) -> crate::raise::death::DeathKind {
        use crate::raise::death::DeathKind as K;
        match self {
            DeathKind::NullPage
            | DeathKind::PresentWithoutSurface
            | DeathKind::RghaWithoutDevice
            | DeathKind::QueueGone => K::NullDeref,
            DeathKind::ShowHandlerGone
            | DeathKind::MoveHandlerGone
            | DeathKind::GlyphHandlerGone => K::NilBlock,
            #[cfg(target_arch = "x86_64")]
            DeathKind::ColorMatrixOutsidePage => K::OutOfBounds,
            _ => K::Assert,
        }
    }

    /// The site a death of this kind names (tests).
    #[cfg(test)]
    pub(crate) fn site(self) -> crate::raise::death::DeathSite {
        crate::raise::death::DeathId::from(self).site()
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> DeathId {
        DeathId::GuestDisplay(k)
    }
}

/// What `guest_display_mutex` guards: the state machine's, the presents', the cursor's and the
/// restore's state.
pub(crate) struct GuestDisplayLocked<P: Platform> {
    /// The page's mapping, released at the guest display's end.
    pub(crate) mapping: Option<Arc<Mapping<P>>>,
    /// The display's modes, which the state machine consumes; `None` once consumed.
    pub(crate) mode_list: Option<ModeList>,
    /// The guest's raw sleep byte: the presents test bit 0.
    pub(crate) asleep: u8,
    /// Whether the guest's colour space changed since a present handed it on.
    pub(crate) guest_color_space_dirty: bool,
    /// Paused (also in [`GuestDisplayUnlocked::paused`], for the lock-free readers).
    pub(crate) paused: bool,
    /// A restore is under way: the state machine waits for its end.
    pub(crate) will_restore: bool,
    /// A restore finished and the state machine has not run since.
    pub(crate) did_restore: bool,
}

/// The guest display's client: the host's display on the port, unretained, and its core as the
/// guest display reaches it.
struct Client<P: Platform> {
    /// The display, by identity (`None` while no display is attached).
    display: Option<Weak<Display<P>>>,
    /// The display's core as the device found it when the client was set
    /// ([`DeviceCore::display_core`]): it keeps the memory, not the display, so no guest display
    /// operation keeps the display alive, and a display whose last release has begun is still
    /// reached until its end detaches it.
    core: Option<Weak<DisplayCore<P>>>,
}

/// What is read without `guest_display_mutex` (module doc, "Locks"); every write is under the lock.
pub(crate) struct GuestDisplayUnlocked<P: Platform> {
    /// Paused, as the cursor position and the present completion read it.
    pub(crate) paused: AtomicBool,
    /// The guest's live-resize latch.
    pub(crate) compositor_supports_live_resize: AtomicBool,
    /// The guest's colour space (its ICC profile bytes); a leaf lock.
    pub(crate) guest_color_space: Mutex<Option<Arc<[u8]>>>,
    /// The client; a leaf lock.
    client: Mutex<Client<P>>,
    /// This guest display, for the blocks it dispatches, which hold it.
    this: Weak<GuestDisplay<P>>,
}

/// A display port's guest display (module doc).
pub struct GuestDisplay<P: Platform> {
    /// The platform (logs, Metal's descriptors).
    pub(crate) platform: Arc<P>,
    /// The device, unretained.
    pub(crate) device: Weak<DeviceCore<P>>,
    /// The port.
    pub(crate) port: u64,
    /// The page, mapped for the guest display's life.
    page: GuestPage,
    /// The state `guest_display_mutex` guards.
    pub(crate) guest_display_mutex: Mutex<GuestDisplayLocked<P>>,
    /// What is read without it.
    pub(crate) unlocked: GuestDisplayUnlocked<P>,
}

/// The shared page, each access one volatile or atomic access of the field's width (D18).
#[derive(Debug, Clone, Copy)]
struct Page(GuestPage);

impl Page {
    fn at(self) -> GuestPtr {
        self.0.base()
    }
    fn r8(self, off: u64) -> u8 {
        self.at().read_u8(off, DeathKind::NullPage)
    }
    fn w8(self, off: u64, v: u8) {
        self.at().write_u8(off, v, DeathKind::NullPage)
    }
    fn r16(self, off: u64) -> u16 {
        self.at().read_u16(off, DeathKind::NullPage)
    }
    fn w16(self, off: u64, v: u16) {
        self.at().write_u16(off, v, DeathKind::NullPage)
    }
    fn r32(self, off: u64) -> u32 {
        self.at().read_u32(off, DeathKind::NullPage)
    }
    fn w32(self, off: u64, v: u32) {
        self.at().write_u32(off, v, DeathKind::NullPage)
    }
    fn r64(self, off: u64) -> u64 {
        self.at().read_u64(off, DeathKind::NullPage)
    }
    fn w64(self, off: u64, v: u64) {
        self.at().write_u64(off, v, DeathKind::NullPage)
    }
    fn rf32(self, off: u64) -> f32 {
        f32::from_bits(self.r32(off))
    }
    fn wf32(self, off: u64, v: f32) {
        self.w32(off, v.to_bits())
    }
    /// A seq_cst load (`ldar`) of an atomic word.
    fn load(self, off: u64) -> u32 {
        self.at().load_seqcst_u32(off, DeathKind::NullPage)
    }
    /// A seq_cst `|=` (`ldsetal`) of an atomic word.
    fn fetch_or(self, off: u64, v: u32) {
        let _ = self.at().fetch_or_u32(off, v, DeathKind::NullPage);
    }
    /// Adds one to the u32 at `off`, wrapping.
    fn bump32(self, off: u64) {
        self.w32(off, self.r32(off).wrapping_add(1));
    }
    /// Adds one to the u64 at `off`, wrapping.
    fn bump64(self, off: u64) {
        self.w64(off, self.r64(off).wrapping_add(1));
    }
}

/// The text of a raise a catch logs: its detail, or its id.
fn raise_text(r: &Raise) -> String {
    r.detail().map_or_else(|| r.id().to_string(), str::to_owned)
}

impl<P: Platform> GuestDisplay<P> {
    /// Maps the guest's page `shared_state_page` through the device and makes the guest display
    /// over it: it writes the port, `field1c` and the identity diagonal of the matrix the guest
    /// writes next, and marks it dirty. Raises when the page does not map, after a warning; the
    /// host's `mapMemory` raise leaves before anything of the guest display is seen.
    pub fn new(
        platform: Arc<P>,
        device: Weak<DeviceCore<P>>,
        port: u64,
        shared_state_page: u32,
    ) -> RaiseResult<Arc<Self>> {
        // A gone device, or no mapping, gives a NULL address.
        let mapping = match device.upgrade() {
            Some(d) => d.new_contiguous_mapping(shared_state_page, sp::PAGE_LENGTH),
            None => None,
        };
        let va = mapping
            .as_ref()
            .map_or(HostPtr::NULL, |m| m.virtual_address());
        if va.is_null() {
            // The partial end releases the mapping only with a page: one whose address is NULL is
            // kept for good, and its unmap never comes.
            if let Some(m) = mapping {
                keep_forever(m, Kept::UnmapNeverComes);
            }
            log::warn!(
                "guest display on port {port} refused: its shared state page \
                 {shared_state_page:#x} could not be mapped"
            );
            crate::raise!(
                RaiseKind::MapPage,
                format!(
                    "guest display on port {port}: shared state page {shared_state_page:#x} \
                     could not be mapped"
                )
            );
        }
        let page = GuestPage::new(GuestPtr::new(va));
        let guest_display = Arc::new_cyclic(|this| GuestDisplay {
            platform: platform.clone(),
            device,
            port,
            page,
            guest_display_mutex: Mutex::new(GuestDisplayLocked {
                mapping,
                mode_list: None,
                asleep: 0,
                guest_color_space_dirty: false,
                paused: false,
                will_restore: false,
                did_restore: false,
            }),
            unlocked: GuestDisplayUnlocked {
                paused: AtomicBool::new(false),
                compositor_supports_live_resize: AtomicBool::new(false),
                guest_color_space: Mutex::new(None),
                client: Mutex::new(Client {
                    display: None,
                    core: None,
                }),
                this: this.clone(),
            },
        });
        log::info!("guest display on port {port} created");
        {
            let l = guest_display.guest_display_mutex.lock();
            let p = guest_display.page(&l);
            // The port truncated to the u16 field.
            p.w16(sp::PORT, port.low_u16());
            p.w32(sp::FIELD_1C, FIELD_1C_INIT);
            // Only the diagonal of the matrix at the guest's write index (read once, any byte),
            // then dirty. Each store past the page dies there, after the ones before it.
            let w = p.r8(sp::COLOR_MATRIX_WRITE_INDEX);
            color_matrix_access!(w, 0, 0, 4);
            p.wf32(sp::color_matrix(w, 0, 0), 1.0);
            color_matrix_access!(w, 1, 1, 4);
            p.wf32(sp::color_matrix(w, 1, 1), 1.0);
            color_matrix_access!(w, 2, 2, 4);
            p.wf32(sp::color_matrix(w, 2, 2), 1.0);
            p.w8(sp::COLOR_MATRIX_DIRTY, 1);
        }
        Ok(guest_display)
    }

    /// The page, from a method that holds `guest_display_mutex`: dies while paused.
    fn page(&self, l: &GuestDisplayLocked<P>) -> Page {
        crate::fatal_unless!(!l.paused, DeathKind::PageWhilePaused);
        Page(self.page)
    }

    /// The page with no paused check (the protection options, the colour space, the ICC
    /// profile).
    fn raw_page(&self) -> Page {
        Page(self.page)
    }

    /// The page from where no lock is held: dies while paused.
    fn page_unlocked(&self) -> Page {
        crate::fatal_unless!(
            !self.unlocked.paused.load(Ordering::SeqCst),
            DeathKind::PageWhilePaused
        );
        Page(self.page)
    }

    /// Dies unless the guest display is running: a display command's first check.
    fn assert_running(l: &GuestDisplayLocked<P>) {
        crate::fatal_unless!(!l.paused, DeathKind::CommandWhilePaused);
    }

    /// The client's core, for a method that holds `guest_display_mutex`. It keeps the memory, not
    /// the display: a display whose last reference went is still the client until its end detaches
    /// it, which waits for `guest_display_mutex`. The callers drop it before the guard, so the
    /// detach finds the core the display's own.
    fn client_core(&self) -> Option<Arc<DisplayCore<P>>> {
        self.unlocked
            .client
            .lock()
            .core
            .as_ref()
            .and_then(Weak::upgrade)
    }

    /// The device's kernel task, kept in `held` (a gone device answers `None`).
    ///
    /// Kept in the calling method's holder, so when a task delete on another FIFO leaves the
    /// guest display's the last reference, the kernel task's end (the task root's unmap, the host's
    /// `destroyTask`) runs at that holder's end, after what the method held after the lookup, or,
    /// after a raise, where the raise ends, after the FIFO's catch has logged and faulted; never
    /// at the raise, under `guest_display_mutex`. The lookup's raise is as `on` says: the
    /// colour-space read reaches it with no handler from API entries.
    fn kernel_task(&self, on: OnFault, held: &Held) -> RaiseResult<Option<KernelTaskRef<P>>> {
        match self.device.upgrade() {
            Some(d) => {
                let task = d.get_kernel_task(on)?;
                held.keep(Arc::clone(&task));
                Ok(Some(task))
            }
            None => Ok(None),
        }
    }

    /// Pokes the device's interrupt for the port (a gone device ignores it).
    fn poke(&self) {
        if let Some(d) = self.device.upgrade() {
            d.display_poke_port(self.port);
        }
    }

    /// Runs `block` on `queue`, holding the guest display; the block gets the client's core as it
    /// is when it runs.
    ///
    /// The client is +0, so a block reaches its core without keeping the display: a `block`
    /// drops the `Arc` it receives before it calls a host handler. A handler that sends the
    /// display its last release then runs the display's end inside the call, which releases its
    /// Metal objects (the current frame over the guest's IOSurface, the buffers, the pipelines)
    /// before its device and its destroyed line; a held `Arc` would leave them to the block's
    /// end.
    fn dispatch(
        &self,
        queue: &Arc<dyn HostQueue>,
        block: impl FnOnce(&GuestDisplay<P>, Option<Arc<DisplayCore<P>>>) + Send + 'static,
    ) {
        let Some(this) = self.unlocked.this.upgrade() else {
            return;
        };
        queue.exec_async(Box::new(move || {
            let client = this.client_core();
            block(&this, client);
        }));
    }

    /// The page, for tests and the display's EFI checks: dies while paused.
    pub fn shared_state(&self) -> Option<GuestPage> {
        let _ = self.page_unlocked();
        Some(self.page)
    }

    /// Sets the client (unretained), then runs the state machine. Dies when both the old and
    /// the new client are set. The state machine's raise is as `on` says: a FIFO command's page
    /// change sets the client under a handler, and a display's attach and detach with none.
    pub fn set_client(&self, on: OnFault, client: Option<Weak<Display<P>>>) -> RaiseResult<()> {
        // The core through the display's pointer, with no reference to it: an upgrade here
        // could be the last reference, whose drop would run the display's end on this thread
        // under the caller's locks (the device's displays lock, which that end's detach takes),
        // and it fails for a display whose last release has begun, which is still connected:
        // its end waits in the detach, and the pointer reaches it until then. The device that
        // sets the client holds a reference for the call.
        let core = client
            .as_ref()
            .and_then(|c| self.device.upgrade()?.display_core(c))
            .map(|c| Arc::downgrade(&c));
        let mut l = self.guest_display_mutex.lock();
        {
            let mut slot = self.unlocked.client.lock();
            crate::fatal_unless!(
                slot.display.is_none() || client.is_none(),
                DeathKind::ClientAlreadySet
            );
            *slot = Client {
                display: client,
                core,
            };
        }
        let keep = self.client_core();
        self.state_machine(on, &mut l, keep.as_ref())
    }

    /// The client, by identity. Read without `guest_display_mutex` (a leaf lock), so the device,
    /// which calls it under its displays lock, never waits on a guest display's state machine.
    pub fn client(&self) -> Option<Weak<Display<P>>> {
        self.unlocked.client.lock().display.clone()
    }

    /// The guest's live-resize latch, lock-free: the display reads it inside the guest display's
    /// own doorbell.
    pub fn compositor_supports_live_resize(&self) -> bool {
        self.unlocked
            .compositor_supports_live_resize
            .load(Ordering::SeqCst)
    }

    /// The modes not yet consumed by the state machine; empty once consumed.
    pub fn mode_list(&self) -> Vec<DisplayMode> {
        let list = self.guest_display_mutex.lock().mode_list.clone();
        list.as_deref()
            .map(<[DisplayMode]>::to_vec)
            .unwrap_or_default()
    }

    /// Takes the display's modes, then runs the state machine; the list replaced is released
    /// after the lock. The state machine's raise is as `on` says: the host's mode list comes
    /// with no handler, and a FIFO command's page change hands it over under one.
    pub fn set_mode_list(&self, on: OnFault, list: ModeList) -> RaiseResult<()> {
        let mut l = self.guest_display_mutex.lock();
        let old = l.mode_list.replace(list);
        let keep = self.client_core();
        let r = self.state_machine(on, &mut l, keep.as_ref());
        drop(l);
        drop(old);
        r
    }

    /// The guest's colour space taken last, lock-free: the display reads it inside the guest
    /// display's own present.
    pub fn guest_color_space(&self) -> Option<Arc<[u8]>> {
        self.unlocked.guest_color_space.lock().clone()
    }

    /// The guest's ack of the connect or update the guest display raised: raises with none pending
    /// (bit 0 of the pending byte) or for another number than expected; else stores it, clears the
    /// pending byte and runs the state machine.
    pub fn ack(&self, ack: u32) -> RaiseResult<()> {
        log::info!("guest display on port {}: received ack {}", self.port, ack);
        let mut l = self.guest_display_mutex.lock();
        // A display command while paused dies (testing.md 7.4).
        Self::assert_running(&l);
        // Bit 0.
        if self.page(&l).r8(sp::ACK_PENDING) & 1 == 0 {
            crate::raise!(
                RaiseKind::AckNothingPending,
                format!(
                    "guest display on port {}: ack {ack} with none pending",
                    self.port
                )
            );
        }
        if ack != self.page(&l).r32(sp::EXPECTED_ACK) {
            crate::raise!(
                RaiseKind::AckUnexpected,
                format!(
                    "guest display on port {}: ack {ack}, where {} was expected",
                    self.port,
                    self.page(&l).r32(sp::EXPECTED_ACK)
                )
            );
        }
        self.page(&l).w32(sp::LAST_ACK, ack);
        self.page(&l).w8(sp::ACK_PENDING, 0);
        let keep = self.client_core();
        self.state_machine(OnFault::Return, &mut l, keep.as_ref())
    }

    /// The connect/ack machine, under `guest_display_mutex`. Nothing while paused or restoring. A
    /// raise from the colour space leaves mid-way: settings and modes rewritten, no ack increment,
    /// no interrupt, the state not written. That raise is as `on` says: the machine runs under a
    /// FIFO command's or a restore's handler, and from API entries and a display's end with none.
    fn state_machine(
        &self,
        on: OnFault,
        l: &mut GuestDisplayLocked<P>,
        client: Option<&Arc<DisplayCore<P>>>,
    ) -> RaiseResult<()> {
        if l.paused || l.will_restore {
            return Ok(());
        }
        let mut bogus = false;
        let (state, interrupt) = loop {
            let state = self.page(l).r32(sp::STATE);
            match state {
                0 => {
                    let out = match client {
                        Some(c) if l.mode_list.is_some() => {
                            self.sync_settings(l, Some(c));
                            self.sync_mode_list(l);
                            let options = c.protection_options();
                            self.store_protection_options(self.raw_page(), options);
                            self.update_guest_color_space_locked(on, l)?;
                            self.page(l).bump32(sp::EXPECTED_ACK);
                            self.page(l).w8(sp::ACK_PENDING, 1);
                            (1, INTERRUPT_CONNECT)
                        }
                        _ => (0, 0),
                    };
                    l.did_restore = false;
                    break out;
                }
                1 => {
                    // `== 1`.
                    if self.page(l).r8(sp::ACK_PENDING) == 1 {
                        log::debug!(
                            "guest display on port {}: state {state} with an ack still pending, \
                             left as it is",
                            self.port
                        );
                        // No interrupt; the restore flag kept.
                        self.page(l).w32(sp::STATE, state);
                        return Ok(());
                    }
                    // And read the state again.
                    self.page(l).w32(sp::STATE, 2);
                    continue;
                }
                _ => {
                    if state != 2 {
                        log::warn!(
                            "guest display on port {}: state {state} is not a state of the \
                             protocol",
                            self.port
                        );
                        bogus = true;
                    }
                    // The state kept is the one read.
                    let mut out = (state, 0);
                    let restored = l.did_restore && self.page(l).r16(sp::MODE_LIST_COUNT) != 0;
                    let resend = bogus && self.page(l).r16(sp::MODE_LIST_COUNT) != 0;
                    if let Some(c) = client
                        && (l.mode_list.is_some() || restored || resend)
                    {
                        let settings_changed = self.sync_settings(l, Some(c));
                        let modes_changed = l.mode_list.is_some() && self.sync_mode_list(l);
                        if settings_changed || modes_changed || resend {
                            self.page(l).bump32(sp::EXPECTED_ACK);
                            self.page(l).w8(sp::ACK_PENDING, 1);
                            out = (1, INTERRUPT_UPDATE);
                        }
                    }
                    // The page's count after the updates.
                    if client.is_none() || self.page(l).r16(sp::MODE_LIST_COUNT) == 0 {
                        self.page(l).w16(sp::MODE_LIST_COUNT, 0);
                        out = (0, INTERRUPT_DISCONNECT);
                    }
                    l.did_restore = false;
                    break out;
                }
            }
        };
        if interrupt != 0 {
            let enabled = self.page(l).load(sp::ENABLED_INTERRUPTS);
            self.page(l).fetch_or(sp::RAISED_INTERRUPTS, interrupt);
            log::info!(
                "guest display on port {}: raising interrupt {:#x} (enabled={:#x}, state={}), poking port",
                self.port,
                interrupt,
                enabled,
                state
            );
            self.poke();
        }
        // Written after the interrupt.
        self.page(l).w32(sp::STATE, state);
        Ok(())
    }

    /// Pauses the guest display: the page is off limits until the unpause.
    pub fn pause(&self) {
        let mut l = self.guest_display_mutex.lock();
        // A second pause dies, only with a guest display (testing.md 7.3).
        crate::fatal_unless!(!l.paused, DeathKind::PauseTwice);
        l.paused = true;
        self.unlocked.paused.store(true, Ordering::SeqCst);
    }

    /// Unpauses the guest display and runs the state machine. Only the device's unpause, an API
    /// entry with no handler, reaches it, so the machine's raise dies there.
    pub fn unpause(&self) -> RaiseResult<()> {
        let mut l = self.guest_display_mutex.lock();
        // An unpause of a running guest display dies, only with a guest display (testing.md 7.3).
        crate::fatal_unless!(l.paused, DeathKind::UnpauseRunning);
        l.paused = false;
        self.unlocked.paused.store(false, Ordering::SeqCst);
        let keep = self.client_core();
        self.state_machine(OnFault::Die, &mut l, keep.as_ref())
    }

    /// A resume is restoring the guest display: the state machine waits for [`finish_restore`].
    ///
    /// [`finish_restore`]: GuestDisplay::finish_restore
    pub fn will_restore(&self) {
        let mut l = self.guest_display_mutex.lock();
        crate::fatal_unless!(!l.will_restore, DeathKind::WillRestoreTwice);
        l.will_restore = true;
    }

    /// Ends a restore: hands the host the page's cursor glyph and show byte again, forces the
    /// colour matrix, presents the last mapped surface again (or the sleep), then runs the state
    /// machine. A raise on the way is logged and the rest skipped, under a handler inside the
    /// resume's `hold`.
    pub fn finish_restore(&self, hold: Hold<'_>) {
        let mut l = self.guest_display_mutex.lock();
        crate::fatal_unless!(l.will_restore, DeathKind::FinishRestoreUnexpected);
        l.will_restore = false;
        l.did_restore = true;
        Self::assert_running(&l);
        let keep = self.client_core();
        let client = keep.as_ref();
        let l = &mut *l;
        let restored = (|| -> RaiseResult<_> {
            if self.page(l).r64(sp::CURSOR_GLYPH_MAPPED_LENGTH) != 0 {
                // The page's glyph record.
                let p = self.page(l);
                let args = CursorGlyphArgs {
                    virtual_offset: p.r64(sp::CURSOR_GLYPH_VIRTUAL_OFFSET),
                    mapped_length: p.r64(sp::CURSOR_GLYPH_MAPPED_LENGTH),
                    stride: p.r64(sp::CURSOR_GLYPH_STRIDE),
                    width: p.r16(sp::CURSOR_GLYPH_WIDTH),
                    height: p.r16(sp::CURSOR_GLYPH_HEIGHT),
                    hot_spot_x: p.r16(sp::CURSOR_GLYPH_HOT_SPOT_X),
                    hot_spot_y: p.r16(sp::CURSOR_GLYPH_HOT_SPOT_Y),
                    sum: p.r32(sp::CURSOR_GLYPH_SUM),
                };
                self.cursor_glyph_locked(l, client, args)?;
                // The raw byte.
                let show = self.page(l).r8(sp::CURSOR_SHOW);
                self.cursor_show_locked(l, client, show);
            }
            self.page(l).w8(sp::COLOR_MATRIX_FORCE_READ, 1);
            // `asleep` bit 0.
            if self.page(l).r64(sp::PRESENT_COUNT) != 0 && self.page(l).r8(sp::ASLEEP) & 1 == 0 {
                // Always the mapped path, with the page's last gamma table.
                let p = self.page(l);
                let surface = p.r64(sp::PRESENTED_MAPPING_ID);
                let gamma = GammaTableArgs {
                    virtual_offset: p.r64(sp::GAMMA_TABLE_VIRTUAL_OFFSET),
                    mapped_length: p.r64(sp::GAMMA_TABLE_MAPPED_LENGTH),
                    entry_count: p.r32(sp::GAMMA_TABLE_ENTRY_COUNT),
                    sum: p.r32(sp::GAMMA_TABLE_SUM),
                };
                self.present_mapped_surface_locked(l, client, surface, gamma)?;
            } else if self.page(l).r8(sp::ASLEEP) == 1 {
                // `== 1`, then the byte read again.
                let asleep = self.page(l).r8(sp::ASLEEP);
                self.sleep_state_locked(l, client, asleep);
            }
            self.state_machine(OnFault::Return, l, client)
        })();
        if let Err(raised) = restored {
            log::warn!(
                "guest display on port {}: restore stopped by a raise: {}",
                self.port,
                raise_text(&raised)
            );
            hold.land(raised);
        }
    }
}

impl<P: Platform> Drop for GuestDisplay<P> {
    /// The guest display's end: the page's mapping (its unmap), the mode list, then the log. The
    /// guest colour space goes with the fields.
    fn drop(&mut self) {
        let l = self.guest_display_mutex.get_mut();
        drop(l.mapping.take());
        drop(l.mode_list.take());
        log::info!("guest display on port {} destroyed", self.port);
    }
}
