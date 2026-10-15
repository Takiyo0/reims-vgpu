//! A section object: Windows's stand-in for Linux's `memfd`. A pagefile-backed section
//! (`CreateFileMappingW` with `hFile == INVALID_HANDLE_VALUE`) of a fixed byte length, whose
//! views (`MapViewOfFile3`) all show the same pages, so a view of guest pages and an IOSurface
//! stand-in's memory are both made of one.
//!
//! The handle owns the kernel object: it is closed on drop (`CloseHandle`), and the section's
//! pages live until the last view of it is unmapped *and* the handle is closed, as a memfd's do
//! until the last mapping and descriptor go. Cloning the backing shares the one handle through an
//! `Arc`, as Linux shares the `Arc<OwnedFd>`.

use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{CreateFileMappingW, PAGE_READWRITE, SEC_COMMIT};

/// A pagefile-backed section object of a fixed byte length.
#[derive(Debug)]
pub struct Section {
    handle: HANDLE,
    length: u64,
}

// SAFETY: the handle is a kernel object usable from any thread; the type owns it and only reads
// it (to map views) and closes it on drop. Nothing in it is thread-affine.
unsafe impl Send for Section {}
// SAFETY: as above -- shared references only read the handle, which the kernel serializes.
unsafe impl Sync for Section {}

impl Section {
    /// A new pagefile-backed section of `length` bytes, committed and read-write, whose views
    /// show zeros. Where the kernel gives none, the process ends there, naming the error
    /// (as Linux's `memfd` death does).
    pub(crate) fn new(length: u64) -> Option<Section> {
        if length == 0 {
            return None;
        }
        let high = (length >> 32) as u32;
        let low = (length & 0xffff_ffff) as u32;
        // SAFETY: a backed-by-pagefile section (INVALID_HANDLE_VALUE) with no name and default
        // security; the size words describe `length`.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                high,
                low,
                std::ptr::null(),
            )
        };
        if handle.is_null() {
            no_section(length);
        }
        Some(Section { handle, length })
    }

    /// The section's handle, for `MapViewOfFile3`.
    pub(crate) fn handle(&self) -> HANDLE {
        self.handle
    }

    /// The section's length in bytes.
    #[allow(dead_code, reason = "symmetry with memfd's ftruncate length; read in tests")]
    pub(crate) fn length(&self) -> u64 {
        self.length
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: the handle this made, closed once, after the last view is unmapped by its
        // holder.
        let _ = unsafe { CloseHandle(self.handle) };
    }
}

/// The death of a section of `length` bytes the kernel gave none of ([`Section::new`]), with the
/// error just set.
fn no_section(length: u64) -> ! {
    let error = std::io::Error::last_os_error();
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: "CreateFileMappingW",
        },
        &format!(
            "the kernel gave no section of {length:#x} bytes for the library's memory ({error}); \
             IOSurface's and Mach's calls need no backing object on macOS"
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A section is made of the length asked for; zero is none
    #[test]
    fn a_section_has_its_length() {
        let s = Section::new(0x4000).unwrap();
        assert_eq!(s.length(), 0x4000);
        assert!(!s.handle().is_null());
        assert!(Section::new(0).is_none());
    }
}
