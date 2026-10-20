//! `Threads` on Windows: `std::thread`s, detached, with the stack and name core asks for.
//!
//! As on Linux, no Objective-C runs here, so the reason macOS starts plain pthreads (D35: no
//! unwind handler at a thread's base for a foreign exception's search phase to find) does not
//! apply. A panic on one of these threads is the library's, and its panic hook aborts the process
//! before anything unwinds. A thread's name is a diagnostic; `std::thread` publishes it to the OS
//! (`SetThreadDescription` on a recent Windows), which keeps the whole name, so none is cut.

use std::sync::Arc;

use reims_vgpu_core::platform::os::{Threads, TimerClock};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};

use super::clock::WindowsClock;

/// A detached thread named `name` with `stack_size` bytes of stack that runs `body`. A refused
/// thread ends the process.
fn spawn_thread(name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
    let made = std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(stack_size)
        .spawn(body);
    if made.is_err() {
        fatal(DeathId::Platform {
            kind: DeathKind::Abort,
            call: "CreateThread",
        })
    }
}

/// The process's threads, and its clock for timers.
#[derive(Debug, Default)]
pub struct WindowsThreads;

impl WindowsThreads {
    /// The process's Threads.
    pub fn new() -> WindowsThreads {
        WindowsThreads
    }
}

impl Threads for WindowsThreads {
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
        spawn_thread(name, stack_size, body);
    }

    /// A worker named after its queue, with a GCD worker's 512 KiB of stack, as on macOS.
    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>) {
        spawn_thread(label, reims_vgpu_core::rt::queue::WORKER_STACK, body);
    }

    fn handle(&self) -> Arc<dyn Threads> {
        Arc::new(WindowsThreads)
    }

    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        Some(Arc::new(WindowsClock::new()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    const WAIT: Duration = Duration::from_secs(10);

    /// A spawned thread is named with the whole name Windows keeps, and runs its body
    #[test]
    fn a_thread_is_named_and_runs() {
        let (tx, rx) = mpsc::channel();
        WindowsThreads::new().spawn(
            "reims-vgpu.fifo",
            512 * 1024,
            Box::new(move || {
                tx.send(std::thread::current().name().map(str::to_owned))
                    .unwrap();
            }),
        );
        let name = rx.recv_timeout(WAIT).unwrap();
        assert_eq!(name.as_deref(), Some("reims-vgpu.fifo"));
    }

    /// A long name is kept whole on Windows, unlike Linux's 15-byte cut
    #[test]
    fn a_long_name_is_kept_whole() {
        let (tx, rx) = mpsc::channel();
        WindowsThreads::new().spawn(
            "reims-vgpu.display-present",
            512 * 1024,
            Box::new(move || {
                tx.send(std::thread::current().name().map(str::to_owned))
                    .unwrap();
            }),
        );
        assert_eq!(
            rx.recv_timeout(WAIT).unwrap().as_deref(),
            Some("reims-vgpu.display-present")
        );
    }

    /// A queue's worker is named after the queue
    #[test]
    fn a_worker_is_named_after_its_queue() {
        let (tx, rx) = mpsc::channel();
        WindowsThreads::new().spawn_worker(
            "reims-vgpu.ui",
            Box::new(move || {
                tx.send(std::thread::current().name().map(str::to_owned))
                    .unwrap();
            }),
        );
        assert_eq!(
            rx.recv_timeout(WAIT).unwrap().as_deref(),
            Some("reims-vgpu.ui")
        );
    }
}
