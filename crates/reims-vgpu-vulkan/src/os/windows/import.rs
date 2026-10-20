//! What the runtime imports for host memory on Windows: the memory itself where the process may
//! write all of it, and otherwise a writable twin of it.
//!
//! Vulkan drivers pin imported pages for writing (`VK_EXT_external_memory_host`, the probe this
//! port rests on), so the runtime refuses read-only memory (typed, `HostMemory`), where Metal
//! reads it: the guest's ranges core maps read-only (the EFI framebuffer among them), aliases of
//! a read-only replace or heap, and surfaces the GPU only reads. For those the runtime is given a
//! **twin** instead: a second view of the same section pages, read and write, at an address of
//! its own ([`import_view`]). The GPU then reads what Metal's reads, but its writes reach those
//! pages, where Metal drops them ([`ImportView::writes_kept`]): every command that may write
//! through an object over such an import dies where it is encoded, naming it
//! (`crate::gpu::writes::kept_write`).
//!
//! The block of zeros a read-only alias maps past its ranges' pages is a section of zeros all
//! such aliases share ([`allocate_zeros`]). Its twin is private memory of zeros instead, so the
//! GPU's writes there stay in that twin and never reach the block, though they are kept for the
//! alias's life, where Metal drops them, which the same deaths stop. A detached object lies in no
//! memory (the runtime's detach, `crate::gpu::backing`), so nothing of it is imported.
//!
//! A buffer made over a twin answers the address it was made over as its contents ([`shown_at`]):
//! the runtime's contents are the twin's. The twin goes once the runtime frees the import (its
//! keep-alive, [`ImportView::release`]).
//!
//! **No map-count budget.** Unlike Linux there is no `vm.max_map_count`, so a twin is not
//! admitted against a budget ahead of mapping; a view the kernel cannot place simply fails the
//! call, where the twin creation gives up and the import is left plain (module `vm`).

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::Arc;

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::os::vm::{HostAddr, KernReturn};
use reims_vgpu_core::sync::Mutex;

use super::files;
use super::section::Section;
use super::vm::{self, Prot};

/// Whether the runtime imports read-only memory through a writable twin of its sections: on
/// Windows, so read-only memory the library makes has a section behind every byte (module docs).
pub(crate) const TWINS_READ_ONLY: bool = true;

/// Every live twin, by its first byte: its length and the first byte of the memory it twins.
static TWINS: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());

/// The sections of zeros ([`allocate_zeros`]), held for the process.
static ZERO_FILES: Mutex<Vec<Arc<Section>>> = Mutex::new(Vec::new());

/// A fresh range of `len` bytes of zeros that a view can be made of, as
/// [`vm::allocate_aliasable`] makes, whose section is kept for the process: an import's twin of
/// it is private zeros, never the section (module docs).
pub(crate) fn allocate_zeros(len: u64) -> Result<HostAddr, KernReturn> {
    let at = vm::allocate_aliasable(len)?;
    if let Some(piece) = files::pieces(at.addr().to_u64(), 1).and_then(|p| p.into_iter().next())
        && let Some((section, _)) = piece.backing.section()
    {
        ZERO_FILES.lock().push(Arc::clone(section));
    }
    Ok(at)
}

/// Whether `backing` is a section of zeros ([`allocate_zeros`]). Borrowed guest-RAM host VAs are
/// never zeros (they are the guest's own pages).
fn zeros(backing: &files::Backing) -> bool {
    let Some((section, _)) = backing.section() else {
        return false;
    };
    ZERO_FILES.lock().iter().any(|z| Arc::ptr_eq(z, section))
}

/// The memory the runtime imports for `length` bytes at `at` (module docs).
#[derive(Debug)]
pub(crate) struct ImportView {
    at: NonNull<u8>,
    twin: Option<Twin>,
}

/// A writable twin's address range, unmapped when it drops.
#[derive(Debug)]
struct Twin {
    first: u64,
    length: u64,
}

impl Drop for Twin {
    fn drop(&mut self) {
        TWINS.lock().remove(&self.first);
        // The twin's own reservation, which the runtime let go of (it freed the import and
        // dropped its keep-alive, or the call that would import it was refused): unmap each of
        // its views to placeholders, then release the whole reservation.
        vm::release_twin(self.first, self.length);
    }
}

impl ImportView {
    /// The first byte the runtime imports.
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }

    /// Whether the GPU's writes through the import are kept where Metal drops them: it is a twin
    /// of memory mapped read-only, the guest's pages or zeros (module docs).
    pub(crate) fn writes_kept(&self) -> bool {
        self.twin.is_some()
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves. With a twin: `inner`, given `original`, the address the bytes were
    /// given at, and the twin, which the runtime drops once it freed the import, so the twin is
    /// unmapped after it. Without one, `inner` as it is, and nothing.
    pub(crate) fn release(
        self,
        original: NonNull<u8>,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        let Some(twin) = self.twin else {
            return (inner, None);
        };
        let original = SendPtr(original);
        let inner = inner.map(|inner| -> rt::Deallocator {
            Box::new(move |_, length| inner(original.get(), length))
        });
        (inner, Some(Box::new(twin)))
    }
}

/// An address moved into a deallocator, which may run on any thread.
struct SendPtr(NonNull<u8>);

// SAFETY: an address only, handed back to the deallocator it was given for.
unsafe impl Send for SendPtr {}

impl SendPtr {
    /// The address.
    fn get(self) -> NonNull<u8> {
        self.0
    }
}

/// The memory the runtime imports for `length` bytes at `at`: those bytes where every page of
/// them is mapped writable or one has no section behind it (which the runtime then refuses as it
/// refuses any import there), and otherwise a writable twin of their pages, at the same offset in
/// its first page (module docs).
pub(crate) fn import_view(at: NonNull<u8>, length: u64) -> ImportView {
    let plain = ImportView { at, twin: None };
    let start = at.as_ptr().addr().to_u64();
    let page = vm::page_size();
    let first = start & !(page - 1);
    let Some(end) = start
        .checked_add(length.max(1))
        .and_then(|e| e.checked_next_multiple_of(page))
    else {
        return plain;
    };
    let Some(pieces) = files::pieces(first, end - first) else {
        return plain;
    };
    if pieces.iter().all(|p| p.prot == Prot::READ_WRITE) {
        return plain;
    }
    let Some(twin) = twin(first, end - first, &pieces) else {
        return plain;
    };
    let Some(twin_at) = address(twin.first + (start - first)) else {
        return plain;
    };
    ImportView {
        at: twin_at,
        twin: Some(twin),
    }
}

/// A writable twin of the `length` bytes from `first` that `pieces` show: each piece's section
/// pages, shared, read and write, but zeros' (private zeros); `None` where the host maps none.
fn twin(first: u64, length: u64, pieces: &[files::Piece]) -> Option<Twin> {
    let base = vm::reserve_twin(length).ok()?;
    let twin = Twin {
        first: base,
        length,
    };
    for piece in pieces {
        let at = base + piece.at;
        if zeros(&piece.backing) {
            // The twin of zeros is private zeros: committed over its placeholder, so no earlier
            // bytes stay and the GPU's writes never reach the shared section.
            vm::twin_private(at, piece.length).ok()?;
        } else {
            vm::twin_view(at, piece.length, &piece.backing).ok()?;
        }
    }
    TWINS.lock().insert(base, (length, first));
    Some(twin)
}

/// The address whose bytes the runtime's memory at `at` shows: `at`, but for an address in a
/// writable twin, the twinned memory's at the same offset (module docs).
pub(crate) fn shown_at(at: NonNull<u8>) -> NonNull<u8> {
    let addr = at.as_ptr().addr().to_u64();
    let twins = TWINS.lock();
    let Some((&twin, &(length, first))) = twins.range(..=addr).next_back() else {
        return at;
    };
    if addr >= twin + length {
        return at;
    }
    address(first + (addr - twin)).unwrap_or(at)
}

/// A pointer to `at` in the process's address space; `None` for 0.
fn address(at: u64) -> Option<NonNull<u8>> {
    NonNull::new(vm::ptr(at).cast::<u8>())
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::os::vm::{HostVm as _, VmProt};

    use super::super::WindowsVm;
    use super::*;

    fn read(at: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe { address(at).unwrap().as_ptr().read_volatile() }
    }

    fn write(at: u64, v: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { address(at).unwrap().as_ptr().write_volatile(v) }
    }

    fn addr(p: NonNull<u8>) -> u64 {
        p.as_ptr().addr().to_u64()
    }

    fn page() -> u64 {
        vm::page_size()
    }

    /// Writable memory is imported as it is; read-only memory through a writable twin of its
    /// pages, at the same offset, whose writes reach them, so the import keeps writes, and whose
    /// addresses answer the memory's own until the runtime drops its keep-alive; the deallocator
    /// is given the memory's own address, and the twin outlives it
    #[test]
    fn read_only_memory_is_imported_through_a_writable_twin() {
        let p0 = page();
        let vm = WindowsVm::new();
        let at = vm::allocate_aliasable(2 * p0).unwrap();
        let first = at.addr().to_u64();
        let p = address(first + 0x40).unwrap();
        let view = import_view(p, p0);
        assert_eq!(view.at(), p, "writable: the memory itself");
        assert!(!view.writes_kept());
        let (run, kept) = view.release(p, None);
        assert!(run.is_none() && kept.is_none());
        assert!(vm.protect_max(at, 2 * p0, VmProt::READ).is_ok());
        let view = import_view(p, p0);
        assert!(view.writes_kept());
        let twin = addr(view.at());
        assert_ne!(twin, addr(p));
        assert_eq!(twin & (p0 - 1), 0x40, "the offset in its page");
        write(twin + p0, 0x5a);
        assert_eq!(
            read(first + 0x40 + p0),
            0x5a,
            "the twin's writes reach the pages"
        );
        assert_eq!(shown_at(view.at()), p);
        let inside = address(twin + 0x10).unwrap();
        assert_eq!(addr(shown_at(inside)), addr(p) + 0x10);
        let given = Arc::new(Mutex::new(None));
        let seen = Arc::clone(&given);
        let inner: rt::Deallocator = Box::new(move |at, length| {
            *seen.lock() = Some((at.as_ptr().addr().to_u64(), length));
        });
        let (free, kept) = view.release(p, Some(inner));
        free.unwrap()(address(twin).unwrap(), p0);
        assert_eq!(*given.lock(), Some((addr(p), p0)));
        assert_eq!(
            addr(shown_at(inside)),
            addr(p) + 0x10,
            "the twin outlives it"
        );
        drop(kept);
        assert_eq!(addr(shown_at(inside)), twin + 0x10, "the twin is gone");
        assert!(vm.deallocate(at, 2 * p0).is_success());
    }

    /// A twin of zeros is private zeros: its writes never reach the shared section
    #[test]
    fn a_twin_of_zeros_keeps_its_writes_to_itself() {
        let p0 = page();
        let vm = WindowsVm::new();
        let at = allocate_zeros(p0).unwrap();
        let first = at.addr().to_u64();
        assert!(zeros(&files::pieces(first, 1).unwrap()[0].backing));
        assert!(vm.protect_max(at, p0, VmProt::READ).is_ok());
        let p = address(first).unwrap();
        let view = import_view(p, p0);
        let twin = addr(view.at());
        assert_ne!(twin, first);
        assert_eq!(read(twin), 0);
        write(twin, 7);
        assert_eq!((read(twin), read(first)), (7, 0));
        drop(view);
        assert!(vm.deallocate(at, p0).is_success());
    }

    /// Memory with no section behind it is given as it is, for the runtime to refuse
    #[test]
    fn memory_without_a_section_is_imported_as_it_is() {
        let p0 = page();
        let vm = WindowsVm::new();
        let at = vm.allocate_anywhere(p0).unwrap();
        assert!(vm.protect_max(at, p0, VmProt::READ).is_ok());
        let p = NonNull::new(at.as_ptr::<u8>()).unwrap();
        assert_eq!(import_view(p, p0).at(), p);
        assert!(vm.deallocate(at, p0).is_success());
    }
}
