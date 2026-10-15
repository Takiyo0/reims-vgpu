//! Windows: std's `catch_unwind`, as on Linux. A Rust panic that unwinds this far (one the
//! library's panic hook did not already end) stops at this frame; a foreign exception through the
//! host's `extern "C-unwind"` function cannot be caught by Rust and ends the process. Either way
//! the process ends at the call, matching the Linux behaviour (6.7). Windows has no equivalent of
//! macOS's Objective-C `@try`/`@catch`, and the host here is C, so `catch_unwind` is the whole
//! guard.

use std::panic::AssertUnwindSafe;

/// Runs `f` inside `catch_unwind`: its value, or `Err(())` for a Rust unwind out of it.
#[expect(
    clippy::disallowed_methods,
    reason = "the C API's catch around a host call on Windows, which ends the process (6.7)"
)]
pub(crate) fn catch<R>(f: impl FnOnce() -> R) -> Result<R, ()> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).map_err(drop)
}
