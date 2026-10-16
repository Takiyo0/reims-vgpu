//! Windows's side of the death harness, in the same names the Unix files use, over raw Win32
//! FFI (core links no `windows-sys` or `libc`: architecture 2.2).
//!
//! Windows has no POSIX signals, no `fork`/re-exec that inherits a `MAP_SHARED` mapping, and no
//! `_exit` with a signal; a process that faults ends with its exception code as its exit code,
//! and `std::process::abort` ends through `__fastfail`, which no in-process handler sees. The
//! harness's Unix model maps onto Windows like this:
//!
//! - **Fault capture** is a vectored exception handler ([`AddVectoredExceptionHandler`], armed
//!   last so it runs before std's). It is the analog of `sigaction`: it marks the death on the
//!   faulting thread and returns `EXCEPTION_CONTINUE_SEARCH`, so the fault goes on to end the
//!   process exactly as it would have (`SA_SIGINFO`'s chaining to the default action). A stack
//!   overflow is left to std's own guard-page handler, which prints its line and aborts; the
//!   handler marks it there as the Unix path marks the `SIGABRT` std raises.
//! - **`std::process::abort`** (every `fatal`, and a bare `abort()`) is unobservable from inside
//!   the process, so there is no signal handler to mark it. Instead the one `fatal` path marks
//!   the death itself before it aborts ([`crate::raise::death::die`]'s `cfg(windows)` call to
//!   [`super::super::mark_abort_here`]), and the parent synthesises a body-thread abort marker
//!   for a bare `abort()` that left none ([`super::super::synthesise_abort`]).
//! - **The shared scratch file** is a file-mapping ([`CreateFileMappingW`] + [`MapViewOfFile`]);
//!   a view stays coherent with the file and is flushed on process end, so what the child wrote
//!   survives its death, as the Unix `MAP_SHARED` mapping does. A span of live guest RAM is
//!   shared in place through placeholder reservations (the same Win32 dance the Vulkan backend's
//!   `vm.rs` uses, since Windows has no `mmap(MAP_FIXED)`).
//! - **Signal numbers** stay the Unix values the harness and its tests pin (`SIGABRT` 6,
//!   `SIGTRAP` 5, …): the exception codes translate to them ([`signal_of_exception`]), and the
//!   parent translates a faulted child's exit code back the same way
//!   ([`super::super::signal_of_exit`]).

use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------------------------
// The signal numbers the harness and its tests pin (Linux x86-64's; the harness never hands them
// to the OS on Windows, it only records and reports them).

pub const SIGILL: c_int = 4;
pub const SIGTRAP: c_int = 5;
pub const SIGABRT: c_int = 6;
pub const SIGBUS: c_int = 7;
pub const SIGSEGV: c_int = 11;
/// One past the highest signal the harness stores an action for. Part of the shared contract's
/// names; the Windows path marks faults through one vectored handler, so it indexes no array.
#[allow(dead_code)]
pub const NSIG: usize = 32;

// The flags and sentinels the shared `death.rs` names only on Unix; Windows arms no OS action,
// so the sigaction-shaped types are not needed here. The Windows `arm` installs the vectored
// handler instead (`arm_vectored`).

// ---------------------------------------------------------------------------------------------
// Win32 FFI (raw, as the Unix files declare their C library's calls).

type Handle = *mut c_void;
type Bool = i32;
type Dword = u32;
type Word = u16;

const PAGE_READWRITE: Dword = 0x04;
const FILE_MAP_ALL_ACCESS: Dword = 0x000F_001F;

/// `EXCEPTION_CONTINUE_SEARCH`: let the next handler (and ultimately the OS) deal with it.
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

// The exception codes a fault ends with. `windows.rs`'s probe measured these on this host.
const STATUS_BREAKPOINT: Dword = 0x8000_0003;
const STATUS_SINGLE_STEP: Dword = 0x8000_0004;
const STATUS_ACCESS_VIOLATION: Dword = 0xC000_0005;
const STATUS_IN_PAGE_ERROR: Dword = 0xC000_0006;
const STATUS_ILLEGAL_INSTRUCTION: Dword = 0xC000_001D;
const STATUS_PRIVILEGED_INSTRUCTION: Dword = 0xC000_0096;
const STATUS_STACK_OVERFLOW: Dword = 0xC000_00FD;

/// An `EXCEPTION_RECORD`'s head: the code is all the handler reads.
#[repr(C)]
struct ExceptionRecord {
    exception_code: Dword,
    exception_flags: Dword,
    exception_record: *mut ExceptionRecord,
    exception_address: *mut c_void,
    number_parameters: Dword,
    exception_information: [usize; 15],
}

/// An `EXCEPTION_POINTERS`.
#[repr(C)]
struct ExceptionPointers {
    exception_record: *mut ExceptionRecord,
    context_record: *mut c_void,
}

unsafe extern "system" {
    fn GetCurrentThreadId() -> Dword;
    fn GetThreadDescription(thread: Handle, out: *mut *mut u16) -> i32;
    fn GetCurrentThread() -> Handle;
    fn LocalFree(mem: *mut c_void) -> *mut c_void;
    fn GetSystemInfo(info: *mut SystemInfo);
    fn CreateFileMappingW(
        file: Handle,
        attrs: *mut c_void,
        protect: Dword,
        max_high: Dword,
        max_low: Dword,
        name: *const u16,
    ) -> Handle;
    fn MapViewOfFile(
        mapping: Handle,
        access: Dword,
        off_high: Dword,
        off_low: Dword,
        len: usize,
    ) -> *mut c_void;
    fn CloseHandle(h: Handle) -> Bool;
    fn AddVectoredExceptionHandler(
        first: Dword,
        handler: extern "system" fn(*mut ExceptionPointers) -> i32,
    ) -> *mut c_void;
    fn ExitProcess(code: Dword) -> !;
    fn RaiseException(code: Dword, flags: Dword, n: Dword, args: *const usize) -> !;
}

/// A `SYSTEM_INFO`, enough of it for the page and allocation granularity.
#[repr(C)]
struct SystemInfo {
    processor_architecture: Word,
    reserved: Word,
    page_size: Dword,
    min_app_address: *mut c_void,
    max_app_address: *mut c_void,
    active_processor_mask: usize,
    number_of_processors: Dword,
    processor_type: Dword,
    allocation_granularity: Dword,
    processor_level: Word,
    processor_revision: Word,
}

// ---------------------------------------------------------------------------------------------
// The calls `death.rs` makes, in the Unix names.

/// The calling thread's id (`GetCurrentThreadId`), as the Unix `current_thread_id`.
pub fn current_thread_id() -> u64 {
    // SAFETY: a parameterless Win32 call.
    u64::from(unsafe { GetCurrentThreadId() })
}

/// The thread handle `pthread_getname_np` takes on Unix; on Windows the name is read from the
/// current thread, so the argument is ignored.
pub fn pthread_self() -> *mut c_void {
    std::ptr::null_mut()
}

/// Writes the calling thread's name into `buf` (`len` bytes, NUL-terminated), as
/// `pthread_getname_np`. Reads `GetThreadDescription`, which holds the name std set with
/// `SetThreadDescription`. Async-signal-safe is not required on Windows (no signal handler calls
/// it); the vectored handler reads the id, not the name.
///
/// # Safety
/// `buf` points to `len` writable bytes.
pub unsafe fn pthread_getname_np(_thread: *mut c_void, buf: *mut c_char, len: usize) -> c_int {
    if buf.is_null() || len == 0 {
        return 0;
    }
    // SAFETY: `GetThreadDescription` allocates a wide string and writes its pointer to `wide`.
    let mut wide: *mut u16 = std::ptr::null_mut();
    let hr = unsafe { GetThreadDescription(GetCurrentThread(), &raw mut wide) };
    if hr < 0 || wide.is_null() {
        // SAFETY: writing one NUL at the start of a buffer of at least one byte.
        unsafe { *buf = 0 };
        return 0;
    }
    // Count the wide chars, then transcode to UTF-8 into the buffer, keeping room for the NUL.
    let mut n = 0usize;
    // SAFETY: `wide` is a NUL-terminated wide string the OS wrote.
    while unsafe { *wide.add(n) } != 0 {
        n = n.wrapping_add(1);
    }
    // SAFETY: `wide[..n]` are `n` valid `u16`s.
    let units = unsafe { std::slice::from_raw_parts(wide, n) };
    let name = String::from_utf16_lossy(units);
    let bytes = name.as_bytes();
    let keep = bytes.len().min(len.saturating_sub(1));
    // SAFETY: `buf` holds `len` bytes; `keep < len`, and the NUL at `keep` is in range.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.cast::<u8>(), keep);
        *buf.add(keep) = 0;
    }
    // SAFETY: the string the OS allocated is freed with `LocalFree`, as `GetThreadDescription`
    // documents.
    unsafe { LocalFree(wide.cast::<c_void>()) };
    0
}

/// The host page size (`GetSystemInfo`), as `getpagesize`.
pub fn getpagesize() -> c_int {
    // SAFETY: `info` is written whole by the call.
    let mut info: SystemInfo = unsafe { std::mem::zeroed() };
    unsafe { GetSystemInfo(&raw mut info) };
    c_int::try_from(info.page_size).unwrap_or(0x1000)
}

/// Ends the process now with `status`, as `_exit`: no `Drop`, no flush, as `ExitProcess`.
pub fn _exit(status: c_int) -> ! {
    // SAFETY: a parameterless-effect Win32 call that never returns.
    unsafe { ExitProcess(status.cast_unsigned()) }
}

/// Ends the process as the Unix `raise(sig)` would for the signals the harness raises itself: on
/// Windows the only one is `SIGTRAP` (the `trap()` test), which `RaiseException` with the
/// breakpoint code makes, so the vectored handler marks it (`signal_of_exception` → 5) and the
/// process ends with the breakpoint exit code the parent reads back as 5. Any other signal is
/// turned into an abort, which the parent reads back as 6.
pub fn raise(sig: c_int) -> c_int {
    if sig == SIGTRAP {
        // SAFETY: raises a breakpoint exception; the vectored handler marks it and lets it end
        // the process. Never returns.
        unsafe { RaiseException(STATUS_BREAKPOINT, 0, 0, std::ptr::null()) }
    }
    // Mark an abort on this thread and end as `std::process::abort` would.
    super::super::mark_abort_here();
    #[allow(clippy::disallowed_methods)]
    std::process::abort()
}

/// Unix widens the alternate signal stack so nested handlers fit; Windows uses no alternate
/// stack, so this is a no-op.
pub fn widen_alternate_stack() {}

// ---------------------------------------------------------------------------------------------
// Fault capture: the vectored exception handler, armed in place of `sigaction`.

/// Whether the vectored handler is installed (armed once, for every fault).
static ARMED: AtomicU32 = AtomicU32::new(0);

/// The signal number a fault's exception code maps to, keeping the harness's pinned values.
/// Returns 0 for a code that is not a fault the harness marks.
fn signal_of_exception(code: Dword) -> c_int {
    match code {
        STATUS_ACCESS_VIOLATION | STATUS_IN_PAGE_ERROR => SIGSEGV,
        STATUS_BREAKPOINT | STATUS_SINGLE_STEP => SIGTRAP,
        STATUS_ILLEGAL_INSTRUCTION | STATUS_PRIVILEGED_INSTRUCTION => SIGILL,
        // A stack overflow dies as it does without the harness: std's guard-page handler prints
        // its line and aborts, so it is marked as the `SIGABRT` that abort is.
        STATUS_STACK_OVERFLOW => SIGABRT,
        _ => 0,
    }
}

/// The vectored exception handler: marks the death on the faulting thread and lets the fault go
/// on to end the process (`EXCEPTION_CONTINUE_SEARCH`), the analog of the Unix handler that
/// re-raises on the default action. A stack overflow is left for std's own handler to print and
/// abort; this marks it and continues the search so std's handler still runs.
extern "system" fn on_vectored(info: *mut ExceptionPointers) -> i32 {
    // SAFETY: the OS passes a valid `EXCEPTION_POINTERS` with a valid `EXCEPTION_RECORD`.
    let code = unsafe {
        let rec = (*info).exception_record;
        if rec.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        (*rec).exception_code
    };
    let sig = signal_of_exception(code);
    if sig != 0 {
        super::super::mark_signal_here(sig);
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// Installs [`on_vectored`] once. `death.rs` calls this through `arm` for each signal it would
/// have armed on Unix; the first call installs the one handler and the rest are no-ops. The
/// handler is armed as the *last* vectored handler (`first` = 0) so std's guard-page handler,
/// armed first, still runs for a stack overflow.
pub fn arm_vectored() {
    if ARMED.swap(1, Ordering::AcqRel) == 0 {
        // SAFETY: installs a well-formed handler; the pointer it returns is not needed.
        unsafe { AddVectoredExceptionHandler(0, on_vectored) };
    }
}

// ---------------------------------------------------------------------------------------------
// Shared mappings: the Unix `mmap(MAP_SHARED[|MAP_FIXED])` the harness uses, over file mappings.

/// A mapped view's length rounded up to the page, and its file offset split into high/low dwords.
fn split_off(off: u64) -> (Dword, Dword) {
    ((off >> 32).low_u32(), (off & 0xFFFF_FFFF).low_u32())
}

use crate::num::Truncate as _;

/// Creates a file mapping over `fd` (a borrowed OS file handle) of at least `max` bytes and
/// returns it, or null. The mapping is kept open for the life of its views, so the caller leaks
/// it (the process is a short-lived test child).
///
/// # Safety
/// `fd` is a valid, open, writable file handle.
unsafe fn file_mapping(fd: isize, max: u64) -> Handle {
    let (hi, lo) = split_off(max);
    // SAFETY: `fd` is a valid file handle; a NULL name makes an unnamed mapping.
    unsafe {
        CreateFileMappingW(
            fd as Handle,
            std::ptr::null_mut(),
            PAGE_READWRITE,
            hi,
            lo,
            std::ptr::null(),
        )
    }
}

/// Maps `len` bytes of `fd` from `off` at an address the OS picks, shared and writable, as
/// `mmap(.., MAP_SHARED, fd, off)`. Returns the base, or a null-equivalent on failure (the
/// shared code checks `addr == usize::MAX || is_null`). Leaks the mapping handle (see
/// [`file_mapping`]).
///
/// # Safety
/// `fd` is a valid, open, writable file handle, and `off + len` is within it.
pub unsafe fn map_shared(fd: isize, len: usize, off: u64) -> *mut c_void {
    // SAFETY: `fd` is valid per the contract.
    let mapping = unsafe { file_mapping(fd, off.saturating_add(len as u64)) };
    if mapping.is_null() {
        return std::ptr::null_mut();
    }
    let (hi, lo) = split_off(off);
    // SAFETY: `mapping` is a live file mapping; the view is within the file.
    let p = unsafe { MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, hi, lo, len) };
    // The mapping handle may be closed once a view exists; the view keeps the section alive.
    // SAFETY: closing a handle whose view holds the section open.
    unsafe { CloseHandle(mapping) };
    p
}
