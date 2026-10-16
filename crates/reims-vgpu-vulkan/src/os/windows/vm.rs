//! `HostVm` on Windows: Mach VM's calls, as core makes them, over Win32 placeholders and section
//! views.
//!
//! Windows cannot remap an address range's pages to another address as `mach_vm_remap` does, so a
//! view of guest pages ("virtual memory layouts") maps their section objects instead -- the exact
//! shape of the Linux backend, with Win32 calls in place of `mmap`:
//!
//! - The whole guest-RAM region and every range a view can be made of is a **reservation** split
//!   into **placeholders** (`VirtualAlloc2` with `MEM_RESERVE | MEM_RESERVE_PLACEHOLDER`). A
//!   placeholder is an owned, empty slot at a fixed address that `MapViewOfFile3` or
//!   `VirtualAlloc2` can fill with `MEM_REPLACE_PLACEHOLDER`, Windows's only way to place a view
//!   or private memory at an address of the caller's choosing. This is the stand-in for Linux's
//!   `MAP_FIXED`.
//! - **`allocate_anywhere`** reserves address space and commits private, zero-filled, read-write
//!   memory in it, in no section, so no view can be made of it: it serves core's reservations,
//!   its detach temporaries and the address space views are mapped over. A range a view must be
//!   made of comes from [`allocate_aliasable`], a view of a section of its own.
//! - **`allocate_fixed_overwrite`** turns the range back into placeholders and commits fresh
//!   private memory over them (`MEM_REPLACE_PLACEHOLDER`): the range then shows no section.
//! - **`remap_shared_overwrite`** fails with `KERN_INVALID_ADDRESS` where a byte of the source
//!   has no section behind it, as Mach fails for an unmapped source, before it maps anything. A
//!   piece the kernel refuses fails the call with nothing changed, as Mach's remap is all or
//!   nothing: the pieces mapped before it are put back as the target was ([`put_back`]). Each
//!   piece is mapped with `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` from the section and offset
//!   the backing table names for it ([`super::files`]).
//! - **`protect_max`** is `VirtualProtect`: Windows has no maximum protection, and the current
//!   one stands in for it, so a lowered protection can be raised again, which Mach refuses. A
//!   forbidden write is an access-violation exception, where Mach's is a `SIGBUS`: a death either
//!   way.
//! - Each call takes whole host pages (`GetSystemInfo`'s `dwPageSize`, 4 KiB on x86-64), from the
//!   page of its first byte to the page past its last, as Mach's do.
//!
//! **Mappings.** Unlike Linux's VMAs there is no `vm.max_map_count` and the kernel does not
//! coalesce neighbouring views, so there is no mapping budget: a view the kernel refuses for want
//! of address space or memory is `KERN_NO_SPACE`, as Mach's is.
//!
//! **Granularity.** Section views must begin on the allocation granularity (`dwAllocationGranularity`,
//! 64 KiB), not merely a page, so a placeholder a view fills is split on that boundary. Core maps
//! the guest's RAM and surfaces on granule boundaries already (the Linux backend's 16 KiB granule
//! note); where a sub-page remap is asked for, the view still begins on a granule and the offset
//! within it is honoured by the section offset.

use std::sync::Arc;

use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use windows_sys::Win32::Foundation::GetLastError;
use windows_sys::Win32::System::Memory::{
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, PAGE_NOACCESS, PAGE_READONLY,
    PAGE_READWRITE, VirtualAlloc2, VirtualFree, VirtualProtect,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

use super::files::{self, Backing};
use super::section::Section;

/// `KERN_INVALID_ADDRESS`.
const INVALID_ADDRESS: KernReturn = KernReturn(1);
/// `KERN_PROTECTION_FAILURE`.
const PROTECTION_FAILURE: KernReturn = KernReturn(2);
/// `KERN_NO_SPACE`.
const NO_SPACE: KernReturn = KernReturn(3);
/// `KERN_INVALID_ARGUMENT`.
const INVALID_ARGUMENT: KernReturn = KernReturn(4);
/// `KERN_FAILURE`.
const FAILURE: KernReturn = KernReturn(5);

/// `ERROR_INVALID_ADDRESS`.
const ERROR_INVALID_ADDRESS: u32 = 487;
/// `ERROR_INVALID_PARAMETER`.
const ERROR_INVALID_PARAMETER: u32 = 87;
/// `ERROR_ACCESS_DENIED`.
const ERROR_ACCESS_DENIED: u32 = 5;
/// `ERROR_NOT_ENOUGH_MEMORY` / `ERROR_OUTOFMEMORY`.
const ERROR_NOT_ENOUGH_MEMORY: u32 = 8;
const ERROR_OUTOFMEMORY: u32 = 14;
/// `ERROR_COMMITMENT_LIMIT`.
const ERROR_COMMITMENT_LIMIT: u32 = 1455;

/// A page protection as Windows names it (`PAGE_*`), the Windows stand-in for Linux's `PROT_*`
/// bitmask. Only the three protections core asks for are represented: no access, read-only, and
/// read-write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Prot(u32);

impl Prot {
    /// No access (`PAGE_NOACCESS`).
    pub(crate) const NONE: Prot = Prot(PAGE_NOACCESS);
    /// Read-only (`PAGE_READONLY`).
    pub(crate) const READ: Prot = Prot(PAGE_READONLY);
    /// Read and write (`PAGE_READWRITE`).
    pub(crate) const READ_WRITE: Prot = Prot(PAGE_READWRITE);

    /// `VmProt` as the nearest `PAGE_*`: write implies read (Windows has no write-only page), any
    /// read is `PAGE_READONLY`, nothing is `PAGE_NOACCESS`. Execute is dropped: the guest's pages
    /// are never run as code here.
    fn from_vm(prot: VmProt) -> Prot {
        if prot.raw() & VmProt::WRITE.raw() != 0 {
            Prot::READ_WRITE
        } else if prot.raw() & VmProt::READ.raw() != 0 {
            Prot::READ
        } else {
            Prot::NONE
        }
    }

    /// Whether a page at this protection may be read.
    pub(crate) fn is_readable(self) -> bool {
        self.0 == PAGE_READONLY || self.0 == PAGE_READWRITE
    }

    /// The raw `PAGE_*` value.
    fn raw(self) -> u32 {
        self.0
    }
}

/// The null process handle targets the calling process for `VirtualAlloc2`. Note that
/// `UnmapViewOfFile2` does NOT accept the null handle (it fails `ERROR_INVALID_HANDLE`); it is
/// given [`current_process`]'s real pseudo-handle instead.
const CURRENT_PROCESS: windows_sys::Win32::Foundation::HANDLE = std::ptr::null_mut();

/// The calling process's pseudo-handle (`GetCurrentProcess`, `(HANDLE)-1`), for the Win32 calls
/// that reject the null handle: `UnmapViewOfFile2` is one, where `VirtualAlloc2` is not.
fn current_process() -> windows_sys::Win32::Foundation::HANDLE {
    // SAFETY: a parameterless Win32 call returning a constant pseudo-handle.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentProcess() }
}

/// The host's page size and allocation granularity, read once (`GetSystemInfo`).
fn system_info() -> (u64, u64) {
    use std::sync::OnceLock;
    static INFO: OnceLock<(u64, u64)> = OnceLock::new();
    *INFO.get_or_init(|| {
        let mut si = unsafe { std::mem::zeroed::<SYSTEM_INFO>() };
        // SAFETY: a live out-parameter the call fills.
        unsafe { GetSystemInfo(&mut si) };
        (
            u64::from(si.dwPageSize).max(4096),
            u64::from(si.dwAllocationGranularity).max(65536),
        )
    })
}

/// The host's page size.
pub(super) fn page_size() -> u64 {
    system_info().0
}

/// The host's allocation granularity (the boundary a section view must begin on).
#[cfg(test)]
fn granularity() -> u64 {
    system_info().1
}

/// The whole pages `len` bytes at `addr` lie in: the first byte of the first, and their length;
/// `None` where they pass the end of the address space.
fn pages(addr: HostAddr, len: u64) -> Option<(u64, u64)> {
    let mask = page_size() - 1;
    let start = addr.addr().to_u64();
    let first = start & !mask;
    let end = start.checked_add(len)?.checked_add(mask)? & !mask;
    Some((first, end - first))
}

/// A Mach answer for the last-error of a refused call: `ERROR_INVALID_ADDRESS` is
/// `KERN_INVALID_ADDRESS` for a range not mapped, out-of-memory and the commitment limit are
/// `KERN_NO_SPACE`, access denied is `KERN_PROTECTION_FAILURE`.
fn refused() -> KernReturn {
    // SAFETY: reads the calling thread's last-error, set by the failed call just made.
    match unsafe { GetLastError() } {
        ERROR_INVALID_ADDRESS => INVALID_ADDRESS,
        ERROR_INVALID_PARAMETER => INVALID_ARGUMENT,
        ERROR_ACCESS_DENIED => PROTECTION_FAILURE,
        ERROR_NOT_ENOUGH_MEMORY | ERROR_OUTOFMEMORY | ERROR_COMMITMENT_LIMIT => NO_SPACE,
        _ => FAILURE,
    }
}

/// A pointer to `at` in the process's address space.
pub(super) fn ptr(at: u64) -> *mut core::ffi::c_void {
    std::ptr::with_exposed_provenance_mut(at.to_usize())
}

/// A `MEMORY_MAPPED_VIEW_ADDRESS` naming `at`.
fn view_addr(at: u64) -> MEMORY_MAPPED_VIEW_ADDRESS {
    MEMORY_MAPPED_VIEW_ADDRESS { Value: ptr(at) }
}

/// Reserves `length` bytes of address space at `at` (or anywhere, when `at` is `None`) as a
/// placeholder a view or private memory can replace: `VirtualAlloc2` with
/// `MEM_RESERVE | MEM_RESERVE_PLACEHOLDER`. Returns the first byte, or the last-error's answer.
fn reserve_placeholder(at: Option<u64>, length: u64) -> Result<u64, KernReturn> {
    // SAFETY: a fresh reservation; at a fixed address, replacing that range is the caller's
    // decision (core's reservation of its own).
    let got = unsafe {
        VirtualAlloc2(
            CURRENT_PROCESS,
            at.map_or(std::ptr::null(), |a| ptr(a).cast_const()),
            length.to_usize(),
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            std::ptr::null_mut(),
            0,
        )
    };
    if got.is_null() {
        return Err(refused());
    }
    Ok(got.addr().to_u64())
}

/// Reserves `length` bytes anywhere (or restores the placeholders at a fixed `at`) and fills it
/// with private, read-write zeros that sit in a placeholder: its first byte, or the last-error's
/// answer. The anonymous allocation of the Linux backend. The zeros are a view of a fresh zero
/// section ([`map_zeros`]), not a `VirtualAlloc2` commit, so the range is placeholder-backed and
/// a later remap can turn it back into a placeholder and alias a section over it.
pub(super) fn anonymous(at: Option<u64>, length: u64) -> Result<u64, KernReturn> {
    match at {
        None => {
            let first = reserve_placeholder(None, length)?;
            match map_zeros(first, length) {
                Ok(()) => Ok(first),
                Err(kr) => {
                    // SAFETY: releasing the placeholder just reserved, which nothing else holds.
                    unsafe {
                        let _ = VirtualFree(ptr(first), 0, MEM_RELEASE);
                    }
                    Err(kr)
                }
            }
        }
        Some(at) => {
            // At a fixed address: the range is placeholders of the library's. Turn whatever is
            // there back into placeholders, then map fresh zeros over them.
            restore_placeholders(at, length);
            map_zeros(at, length).map(|()| at)
        }
    }
}

/// Splits the placeholder `[at, at+length)` into one placeholder per page, so each page can be
/// filled and later restored independently. The range must be a single placeholder on entry; a
/// `VirtualFree(MEM_PRESERVE_PLACEHOLDER | MEM_RELEASE)` of each page but the last carves it off.
fn split_placeholder_pages(at: u64, length: u64) {
    let page = page_size();
    let mut from = at;
    let end = at + length;
    // Each split carves [from, from+page) off the front of the remaining placeholder; the final
    // page needs no split (it is what remains).
    while from + page < end {
        // SAFETY: carving one page off the front of a placeholder of the library's.
        let _ = unsafe {
            VirtualFree(
                ptr(from),
                page.to_usize(),
                MEM_PRESERVE_PLACEHOLDER | MEM_RELEASE,
            )
        };
        from += page;
    }
}

/// Maps `length` bytes of `backing`'s section, with protection `prot`, into the placeholder at
/// `at`, one view per page (`MapViewOfFile3` with `MEM_REPLACE_PLACEHOLDER`): `Ok` or the
/// last-error's answer. The placeholder is split per page first, so any page can later be
/// unmapped and replaced on its own ([`restore_placeholders`]), which a scatter-gather surface
/// view and a view-of-a-view both need. The placeholder must be the exact range on entry.
fn map_view(at: u64, length: u64, backing: &Backing, prot: Prot) -> Result<(), KernReturn> {
    // Only a section can be mapped. Borrowed guest-RAM host VAs (Backing::HostVa) are never
    // mapped by the library: guest RAM is read-write and the runtime imports its pointer directly
    // (import_view's plain path), so a HostVa backing never reaches map_view.
    let Some((section, base_offset)) = backing.section() else {
        return Err(refused());
    };
    let page = page_size();
    split_placeholder_pages(at, length);
    let mut off = 0u64;
    while off < length {
        let now = page.min(length - off);
        // SAFETY: replacing a one-page placeholder this owns with a view of an open section; the
        // offset and length lie within the section (core maps whole pages of registered RAM).
        let got = unsafe {
            super::map_view_of_file3(
                section.handle(),
                ptr(at + off),
                base_offset + off,
                now.to_usize(),
                MEM_REPLACE_PLACEHOLDER,
                prot.raw(),
            )
        };
        if got.Value.is_null() {
            return Err(refused());
        }
        off += now;
    }
    Ok(())
}

/// Maps `length` bytes of a fresh, private zero section into the placeholder at `at`, read and
/// write: the placeholder-backed stand-in for private zero memory. Win32 forbids committing
/// private memory into a placeholder (`MEM_REPLACE_PLACEHOLDER` serves only section views), so
/// "anonymous" memory that must sit in a placeholder — so it can be turned back into a
/// placeholder ([`restore_placeholders`]) and have a view or more zeros replace it in place — is a
/// view of a section of zeros of its own, not a `VirtualAlloc2` commit. The section is held by
/// the view until it is unmapped; the handle is closed once the view exists (the view keeps the
/// section's pages alive), so nothing need hold it. The range is not recorded in the backing
/// table, so no view can be made *of* it as a source.
fn map_zeros(at: u64, length: u64) -> Result<(), KernReturn> {
    let section = Arc::new(Section::new(length).ok_or(NO_SPACE)?);
    let backing = Backing::Section { fd: section, offset: 0 };
    map_view(at, length, &backing, Prot::READ_WRITE)
}
/// section view is unmapped preserving its placeholder, a private committed region is freed
/// preserving its placeholder, a free or reserved placeholder is left as is. The actual state is
/// read from the kernel (`VirtualQuery`), not the backing table, because a range the table has
/// forgotten (an `allocate_anywhere` block, a put-back's zeros) may still hold a view or private
/// memory the kernel knows about. After this the whole range is placeholders, ready for
/// `MEM_REPLACE_PLACEHOLDER`.
fn restore_placeholders(at: u64, length: u64) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT as STATE_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let end = at + length;
    let mut from = at;
    while from < end {
        let mut info = unsafe { std::mem::zeroed::<MEMORY_BASIC_INFORMATION>() };
        // SAFETY: a live out-parameter the call fills for the region containing `from`.
        let got = unsafe {
            VirtualQuery(
                ptr(from).cast_const(),
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if got == 0 {
            break;
        }
        let region_base = info.BaseAddress.addr().to_u64();
        let region_end = region_base + info.RegionSize.to_u64();
        let to = region_end.min(end);
        if info.Type == MEM_MAPPED {
            // A section view: unmap it at its own base, preserving the placeholder slot. A view
            // must be unmapped whole, so unmap at the region's base (its mapping base).
            // SAFETY: unmapping a view this mapped, preserving its placeholder slot.
            let _ = unsafe {
                windows_sys::Win32::System::Memory::UnmapViewOfFile2(
                    current_process(),
                    view_addr(region_base),
                    MEM_PRESERVE_PLACEHOLDER,
                )
            };
        } else if info.Type == MEM_PRIVATE && info.State == STATE_COMMIT {
            // Private committed memory inside a placeholder: free it back to a placeholder.
            // SAFETY: freeing private memory this owns back to its placeholder slot.
            let _ = unsafe {
                VirtualFree(
                    ptr(from),
                    (to - from).to_usize(),
                    MEM_PRESERVE_PLACEHOLDER | MEM_RELEASE,
                )
            };
        }
        // Else a free or reserved placeholder: already a placeholder, nothing to restore.
        from = to;
    }
}

/// `MEM_COALESCE_PLACEHOLDERS` (`0x1`), the `VirtualFree` flag that rejoins the adjacent
/// placeholders in a range into one. windows-sys 0.59 does not export it, so it is named here
/// from the Win32 headers. Required before a reservation that was split into placeholders (by
/// mapping views or committing private memory into pieces of it) can be released whole: without
/// coalescing, `VirtualFree(base, 0, MEM_RELEASE)` frees only the first placeholder and leaves
/// the rest mapped.
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;

/// Releases the whole reservation of `length` bytes at `first`, however it was split: its pieces
/// are first turned back into placeholders ([`restore_placeholders`]), then coalesced into one
/// placeholder and released. Returns whether the final release succeeded.
///
/// A reservation `allocate_anywhere`/`remap_shared_overwrite`/a twin split into several
/// placeholders (one per view or private commit) is several placeholder regions to the kernel; a
/// bare `VirtualFree(first, 0, MEM_RELEASE)` frees only the first, so the coalesce is what lets
/// the range go in one call, matching Mach's deallocate of a whole range.
fn release_reservation(first: u64, length: u64) -> bool {
    restore_placeholders(first, length);
    // Rejoin the adjacent placeholders into one, then release it. The coalesce needs the exact
    // span; a single-placeholder reservation coalesces to itself.
    // SAFETY: `first..first+length` is a reservation of the library's, now all placeholders, that
    // nothing else holds.
    unsafe {
        let _ = VirtualFree(
            ptr(first),
            length.to_usize(),
            MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
        );
        VirtualFree(ptr(first), 0, MEM_RELEASE) != 0
    }
}

/// Reserves `length` bytes of address space anywhere as placeholders for a twin's views
/// ([`super::import`]): the first byte, or the last-error's answer. A twin is filled piece by
/// piece with [`twin_view`] (shared section pages) and [`twin_private`] (private zeros), then
/// released whole with [`release_twin`].
pub(super) fn reserve_twin(length: u64) -> Result<u64, KernReturn> {
    reserve_placeholder(None, length)
}

/// Maps `length` bytes of `backing`'s section, read and write, into the twin's placeholder at
/// `at`: a shared view, so the GPU's writes reach the section's pages (module `import`).
pub(super) fn twin_view(at: u64, length: u64, backing: &Backing) -> Result<(), KernReturn> {
    restore_placeholders(at, length);
    map_view(at, length, backing, Prot::READ_WRITE)
}

/// Maps `length` bytes of private, read-write zeros into the twin's placeholder at `at`: the
/// twin of a section of zeros, so the GPU's writes stay private (module `import`). The zeros are
/// a view of a fresh zero section ([`map_zeros`]), placeholder-backed like every other range.
pub(super) fn twin_private(at: u64, length: u64) -> Result<(), KernReturn> {
    restore_placeholders(at, length);
    map_zeros(at, length)
}

/// Releases a twin's whole reservation of `length` bytes at `first`: each of its views is
/// unmapped back to a placeholder, the placeholders are coalesced, then the reservation is
/// freed. The twin's drop.
pub(super) fn release_twin(first: u64, length: u64) {
    let _ = release_reservation(first, length);
}

/// A fresh range of `len` bytes, rounded up to whole pages, of zeros, that a view can be made of:
/// a view of a section of its own, recorded in the backing table. Freed by `deallocate`, as any
/// other range. The block of zeros a detached object reads, and the tests' stand-in for guest
/// memory, are such ranges.
pub(crate) fn allocate_aliasable(len: u64) -> Result<HostAddr, KernReturn> {
    let Some((_, length)) = pages(HostAddr::NULL, len) else {
        return Err(NO_SPACE);
    };
    if length == 0 {
        return Ok(HostAddr::NULL);
    }
    let section = Arc::new(Section::new(length).ok_or(NO_SPACE)?);
    let at = reserve_placeholder(None, length)?;
    let backing = Backing::Section { fd: section, offset: 0 };
    map_view(at, length, &backing, Prot::READ_WRITE)?;
    files::record(at, length, backing, Prot::READ_WRITE);
    Ok(HostAddr::from_addr(at.to_usize()))
}

/// Memory of the library's own that a view can be made of and that only its holder frees: a view
/// of a new section, read and write, registered in the backing table ([`files::Registration`]),
/// so that `deallocate` refuses it. An IOSurface stand-in's memory.
#[derive(Debug)]
pub(crate) struct OwnMemory {
    first: u64,
    length: u64,
    registration: Option<files::Registration>,
}

impl OwnMemory {
    /// `len` bytes, rounded up to whole pages, of zeros; `None` for none, or where the host
    /// gives no section or no address space.
    pub(crate) fn new(len: u64) -> Option<OwnMemory> {
        let (_, length) = pages(HostAddr::NULL, len)?;
        if length == 0 {
            return None;
        }
        let section = Arc::new(Section::new(length)?);
        let first = reserve_placeholder(None, length).ok()?;
        let backing = Backing::Section { fd: section, offset: 0 };
        map_view(first, length, &backing, Prot::READ_WRITE).ok()?;
        Some(OwnMemory {
            first,
            length,
            registration: Some(files::Registration::new(first, length, backing)),
        })
    }

    /// The first byte.
    pub(crate) fn first(&self) -> u64 {
        self.first
    }

    /// The length.
    pub(crate) fn length(&self) -> u64 {
        self.length
    }
}

impl Drop for OwnMemory {
    /// Drops the registration, then the view and its reservation.
    fn drop(&mut self) {
        drop(self.registration.take());
        // The view this made is unmapped back to its placeholder, coalesced and released;
        // nothing uses it once its holder lets go.
        let _ = release_reservation(self.first, self.length);
    }
}

/// Maps `length` bytes from `at`, a piece of a remap that failed after mapping it, back to what
/// they showed before it, so the failed remap changes nothing: the sections of the table's runs
/// and registrations there, at the protection each was set to, and fresh zeros with no access
/// over bytes of no section. See the Linux backend's `put_back` for why bytes of no section are a
/// reservation whose protection never shows.
fn put_back(at: u64, length: u64) {
    let page = page_size();
    let has_section = |first: u64| files::pieces(first, page).is_some();
    let end = at + length;
    let mut from = at;
    while from < end {
        let filed = has_section(from);
        let mut to = from + page;
        while to < end && has_section(to) == filed {
            to += page;
        }
        let restored = if filed {
            files::pieces(from, to - from).is_some_and(|pieces| {
                pieces.iter().all(|p| {
                    restore_placeholders(from + p.at, p.length);
                    map_view(from + p.at, p.length, &p.backing, p.prot).is_ok()
                })
            })
        } else {
            restore_placeholders(from, to - from);
            if map_zeros(from, to - from).is_ok() {
                // Set the zeros to no access, as the Linux put-back does with mprotect: bytes of
                // no section show no access, never their protection. Per page, since each page is
                // its own view region.
                let page = page_size();
                let mut off = 0u64;
                let mut ok = true;
                while off < to - from {
                    let now = page.min((to - from) - off);
                    let mut old = 0u32;
                    // SAFETY: setting one page's protection within its own view region.
                    if unsafe {
                        VirtualProtect(ptr(from + off), now.to_usize(), PAGE_NOACCESS, &mut old)
                    } == 0
                    {
                        ok = false;
                        break;
                    }
                    off += now;
                }
                ok
            } else {
                false
            }
        };
        if !restored {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "a failed remap's put-back of {:#x} bytes at {from:#x} was refused; the range is \
                 forgotten",
                to - from,
            );
            files::forget(from, to - from);
        }
        from = to;
    }
}

/// Mach VM's calls on Windows (module docs).
#[derive(Debug, Default)]
pub struct WindowsVm;

impl WindowsVm {
    /// The process's HostVm.
    pub fn new() -> WindowsVm {
        WindowsVm
    }

    /// Whether every page of the `len` bytes at `addr` is mapped with read access: by the
    /// backing table's protections where it knows every byte, else by querying the pages
    /// (`VirtualQuery`), which reports an unmapped or no-access page. A `len` of 0 is `true`.
    pub fn readable(&self, addr: HostAddr, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let Some((first, length)) = pages(addr, len) else {
            return false;
        };
        files::readable(first, length).unwrap_or_else(|| query_readable(first, length))
    }
}

/// Whether every page of `length` bytes from `first` is committed with read access, by
/// `VirtualQuery`.
fn query_readable(first: u64, length: u64) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEMORY_BASIC_INFORMATION, MEM_COMMIT as STATE_COMMIT, VirtualQuery,
    };
    let end = first + length;
    let mut at = first;
    while at < end {
        let mut info = unsafe { std::mem::zeroed::<MEMORY_BASIC_INFORMATION>() };
        // SAFETY: a live out-parameter the call fills for the region containing `at`.
        let got = unsafe {
            VirtualQuery(
                ptr(at).cast_const(),
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if got == 0 || info.State != STATE_COMMIT {
            return false;
        }
        let readable = matches!(info.Protect, PAGE_READONLY | PAGE_READWRITE);
        if !readable {
            return false;
        }
        at = info.BaseAddress.addr().to_u64() + info.RegionSize.to_u64();
    }
    true
}

impl HostVm for WindowsVm {
    fn page_size(&self) -> u64 {
        page_size()
    }

    fn page_shift(&self) -> u32 {
        page_size().trailing_zeros()
    }

    /// Private memory anywhere; a length of 0 answers address 0, as Mach's does.
    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let Some((_, length)) = pages(HostAddr::NULL, len) else {
            return Err(NO_SPACE);
        };
        if length == 0 {
            return Ok(HostAddr::NULL);
        }
        let at = anonymous(None, length)?;
        files::forget(at, length);
        Ok(HostAddr::from_addr(at.to_usize()))
    }

    /// Fresh private memory over the pages of the range, which then shows no section.
    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let (first, length) = pages(addr, len).ok_or(NO_SPACE)?;
        if length == 0 {
            return Ok(());
        }
        anonymous(Some(first), length)?;
        files::forget(first, length);
        Ok(())
    }

    /// The source's pages, from the page of its first byte, mapped from their sections over
    /// `dst`'s pages, each with the protection the source has there (module docs).
    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn> {
        let mask = page_size() - 1;
        let into = src.addr().to_u64() & mask;
        let (target, length) = pages(dst, len.saturating_add(into)).ok_or(INVALID_ARGUMENT)?;
        let (source, _) = pages(src, len).ok_or(INVALID_ADDRESS)?;
        if length == 0 {
            return Ok(dst);
        }
        let Some(pieces) = files::pieces(source, length) else {
            log::debug!(
                target: "reims_vgpu_vulkan::os",
                "remap of {length:#x} bytes at {source:#x}: no section behind them"
            );
            return Err(INVALID_ADDRESS);
        };
        for (done, piece) in pieces.iter().enumerate() {
            restore_placeholders(target + piece.at, piece.length);
            if let Err(kr) = map_view(target + piece.at, piece.length, &piece.backing, piece.prot) {
                for mapped in pieces.iter().take(done) {
                    put_back(target + mapped.at, mapped.length);
                }
                // The piece that failed left a bare placeholder; put it back too.
                put_back(target + piece.at, piece.length);
                return Err(kr);
            }
        }
        for piece in pieces {
            files::record(target + piece.at, piece.length, piece.backing, piece.prot);
        }
        Ok(dst)
    }

    /// `VirtualProtect` of the range's pages (module docs).
    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        let (first, length) = pages(addr, len).ok_or(INVALID_ARGUMENT)?;
        if length == 0 {
            return Ok(());
        }
        let p = Prot::from_vm(prot);
        // Each page is its own view region now (views are mapped per page), and `VirtualProtect`
        // cannot span two regions, so the protection is set a page at a time. A later access a
        // lowered protection forbids faults (a process death), the caller's decision.
        let page = page_size();
        let mut off = 0u64;
        while off < length {
            let now = page.min(length - off);
            let mut old = 0u32;
            // SAFETY: setting one page's protection within its own view region.
            if unsafe {
                VirtualProtect(
                    ptr(first + off).cast_const(),
                    now.to_usize(),
                    p.raw(),
                    &mut old,
                )
            } == 0
            {
                return Err(refused());
            }
            off += now;
        }
        files::protect(first, length, p);
        Ok(())
    }

    /// Unmaps the range's views and frees its reservations, which show no section from then on. A
    /// range of memory the library registered and does not own (the guest's RAM) is refused,
    /// unmapped by nothing.
    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        let Some((first, length)) = pages(addr, len) else {
            return INVALID_ARGUMENT;
        };
        if length == 0 {
            return KernReturn::SUCCESS;
        }
        if files::meets_registration(first, length) {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "deallocate of {length:#x} bytes at {first:#x}: registered memory, not unmapped"
            );
            return INVALID_ARGUMENT;
        }
        files::forget(first, length);
        // Unmap any views back to placeholders, coalesce them, and release the whole reservation.
        if !release_reservation(first, length) {
            return refused();
        }
        KernReturn::SUCCESS
    }
}

/// The death at `call` naming a Win32 last-error (shared by the submodules).
#[allow(dead_code, reason = "used by section.rs via a shared helper in a later step")]
pub(super) fn fatal_win32(call: &'static str, detail: &str) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call,
        },
        detail,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn vm() -> WindowsVm {
        WindowsVm::new()
    }

    fn page() -> u64 {
        page_size()
    }

    fn read(a: HostAddr, off: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().read_volatile() }
    }

    fn write(a: HostAddr, off: u64, v: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().write_volatile(v) }
    }

    /// The host's page, which the tests lay their pages out by: x86-64's 4 KiB.
    pub(crate) const PAGE: u64 = 0x1000;

    /// Whether every page of `len` bytes from `at` is committed and writable now
    /// (`VirtualQuery`).
    pub(crate) fn writable(at: HostAddr, len: u64) -> bool {
        use windows_sys::Win32::System::Memory::{
            MEMORY_BASIC_INFORMATION, MEM_COMMIT as STATE_COMMIT, VirtualQuery,
        };
        let first = at.addr().to_u64();
        let end = first.saturating_add(len);
        let mut cur = first;
        while cur < end {
            let mut info = unsafe { std::mem::zeroed::<MEMORY_BASIC_INFORMATION>() };
            let got = unsafe {
                VirtualQuery(
                    ptr(cur).cast_const(),
                    &mut info,
                    std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            if got == 0 || info.State != STATE_COMMIT || info.Protect != PAGE_READWRITE {
                return false;
            }
            cur = info.BaseAddress.addr().to_u64() + info.RegionSize.to_u64();
        }
        true
    }

    /// Whether the page at `at` is committed at all (`VirtualQuery`).
    pub(crate) fn mapped(at: HostAddr) -> bool {
        use windows_sys::Win32::System::Memory::{
            MEMORY_BASIC_INFORMATION, MEM_COMMIT as STATE_COMMIT, VirtualQuery,
        };
        let mut info = unsafe { std::mem::zeroed::<MEMORY_BASIC_INFORMATION>() };
        let got = unsafe {
            VirtualQuery(
                ptr(at.addr().to_u64()).cast_const(),
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        got != 0 && info.State == STATE_COMMIT
    }

    /// The host page is 4 KiB on x86-64 Windows, its shift 12
    #[test]
    fn the_page_is_the_hosts() {
        let v = vm();
        assert_eq!(v.page_size(), 1 << v.page_shift());
        assert_eq!((v.page_size(), v.page_shift()), (4096, 12));
    }

    /// An allocation anywhere is page-aligned and zero-filled, and no view can be made of it;
    /// one of 0 bytes is address 0, as Mach's is
    #[test]
    fn an_allocation_is_zeros_no_view_is_made_of() {
        let v = vm();
        let len = 2 * page();
        let a = v.allocate_anywhere(len).unwrap();
        assert_eq!(a.addr().to_u64() % page(), 0);
        assert_eq!((read(a, 0), read(a, len - 1)), (0, 0));
        write(a, 7, 0xa5);
        let dst = v.allocate_anywhere(len).unwrap();
        assert_eq!(v.remap_shared_overwrite(dst, a, len), Err(INVALID_ADDRESS));
        assert_eq!(v.allocate_anywhere(0), Ok(HostAddr::NULL));
        assert!(v.deallocate(a, len).is_success());
        assert!(v.deallocate(dst, len).is_success());
    }

    /// A view shows its source's pages both ways; a view of the view shows them too, and keeps
    /// showing them after the first view is unmapped (aliasing through the shared section)
    #[test]
    fn a_view_of_a_view_outlasts_the_first() {
        let v = vm();
        let len = 4 * granularity();
        let src = allocate_aliasable(len).unwrap();
        let first = v.allocate_anywhere(len).unwrap();
        let second = v.allocate_anywhere(2 * granularity()).unwrap();
        write(src, granularity() + 5, 0x11);
        assert_eq!(v.remap_shared_overwrite(first, src, len), Ok(first));
        assert_eq!(read(first, granularity() + 5), 0x11);
        write(first, 9, 0x33);
        assert_eq!(read(src, 9), 0x33);
        v.remap_shared_overwrite(second, first.wrapping_add(granularity()), 2 * granularity())
            .unwrap();
        assert_eq!(read(second, 5), 0x11);
        assert!(v.deallocate(first, len).is_success());
        write(src, granularity() + 6, 0x22);
        assert_eq!(read(second, 6), 0x22);
        assert!(v.deallocate(second, 2 * granularity()).is_success());
        assert!(v.deallocate(src, len).is_success());
    }

    /// Deallocating nothing succeeds, and registered memory is never unmapped
    #[test]
    fn deallocate_keeps_registered_memory() {
        let v = vm();
        let a = v.allocate_anywhere(page()).unwrap();
        assert_eq!(v.deallocate(a, 0), KernReturn::SUCCESS);
        let own = OwnMemory::new(page()).unwrap();
        let ram = HostAddr::from_addr(own.first().to_usize());
        assert_eq!(v.deallocate(ram, page()), INVALID_ARGUMENT);
        assert!(mapped(ram));
        drop(own);
        assert!(v.deallocate(a, page()).is_success());
    }
}
