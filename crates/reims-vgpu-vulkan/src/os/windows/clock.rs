//! `Clock` over Windows's clocks, as macOS's answers them.
//!
//! The monotonic clock is `QueryUnbiasedInterruptTimePrecise`: like `CLOCK_BOOTTIME` on Linux
//! and `CLOCK_MONOTONIC_RAW` on macOS (libc++'s `steady_clock` there), it counts the time the
//! host sleeps, which Windows's *biased* interrupt time does not. It is reported in 100-ns
//! units. The wall clock is `GetSystemTimePreciseAsFileTime`, a FILETIME in 100-ns units since
//! 1601-01-01 UTC, which we re-base to nanoseconds since the Unix epoch (1970-01-01). A condvar's
//! timed wait blocks for its timeout in awake time (`parking_lot`'s `wait_for` hands the OS a
//! relative timeout), and answers as libc++'s does on macOS: after the wait, `now < deadline ?
//! no_timeout : timeout`, read on the deadline's own clock, where a host sleep counts.

use std::time::Duration;

use reims_vgpu_core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
use reims_vgpu_core::sync::{Condvar, MutexGuard};
use windows_sys::Win32::System::SystemInformation::GetSystemTimePreciseAsFileTime;
use windows_sys::Win32::System::WindowsProgramming::QueryUnbiasedInterruptTimePrecise;

/// A FILETIME tick is 100 nanoseconds.
const HUNDRED_NS: u64 = 100;

/// 100-ns ticks between the FILETIME epoch (1601-01-01) and the Unix epoch (1970-01-01):
/// 11,644,473,600 seconds.
const FILETIME_UNIX_EPOCH_TICKS: u64 = 11_644_473_600 * 10_000_000;

/// Unbiased interrupt time (the host's awake-plus-asleep monotonic clock) in nanoseconds.
fn mono_nanos() -> u64 {
    let mut ticks: u64 = 0;
    // SAFETY: a live out-parameter; the call cannot fail on a supported Windows.
    unsafe { QueryUnbiasedInterruptTimePrecise(&mut ticks) };
    ticks.saturating_mul(HUNDRED_NS)
}

/// Wall-clock time in nanoseconds since the Unix epoch.
fn wall_nanos() -> u64 {
    let mut ft = windows_sys::Win32::Foundation::FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    // SAFETY: a live out-parameter; the call cannot fail.
    unsafe { GetSystemTimePreciseAsFileTime(&mut ft) };
    let ticks = (u64::from(ft.dwHighDateTime) << 32) | u64::from(ft.dwLowDateTime);
    ticks
        .saturating_sub(FILETIME_UNIX_EPOCH_TICKS)
        .saturating_mul(HUNDRED_NS)
}

/// The host's clocks and timed waits.
#[derive(Debug, Default)]
pub struct WindowsClock;

impl WindowsClock {
    /// The process's Clock.
    pub fn new() -> WindowsClock {
        WindowsClock
    }

    /// How long until `deadline`, on its own clock; zero once reached.
    fn remaining(&self, deadline: Deadline) -> Duration {
        match deadline {
            Deadline::Mono(t) => t.saturating_duration_since(self.now_mono()),
            Deadline::Wall(t) => t.saturating_duration_since(self.now_wall()),
        }
    }
}

impl Clock for WindowsClock {
    /// Unbiased interrupt time, which counts a host sleep as macOS's monotonic clock does.
    fn now_mono(&self) -> MonoTime {
        MonoTime::from_nanos(mono_nanos())
    }

    /// System time, re-based to the Unix epoch.
    fn now_wall(&self) -> WallTime {
        WallTime::from_nanos(wall_nanos())
    }

    /// One wait, as libc++'s is on macOS: a notify or a spurious wakeup ends it; the answer is
    /// whether the deadline has passed afterwards. A deadline already reached answers timeout
    /// without releasing the lock.
    fn wait_until<T>(
        &self,
        cv: &Condvar,
        g: &mut MutexGuard<'_, T>,
        deadline: Deadline,
    ) -> TimedOut {
        let d = self.remaining(deadline);
        if d.is_zero() {
            return TimedOut(true);
        }
        let _ = cv.wait_for(g, d);
        TimedOut(deadline.reached(self))
    }

    /// One untimed wait; the caller loops.
    fn wait<T>(&self, cv: &Condvar, g: &mut MutexGuard<'_, T>) {
        cv.wait(g);
    }

    /// Sleeps until `deadline` is reached on its own clock.
    fn sleep_until(&self, deadline: Deadline) {
        loop {
            let d = self.remaining(deadline);
            if d.is_zero() {
                return;
            }
            std::thread::sleep(d);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::sync::Mutex;

    use super::*;

    const MS: Duration = Duration::from_millis(1);

    /// The monotonic clock is the unbiased interrupt time, and moves forward at the rate Instant
    /// counts while awake
    #[test]
    fn mono_is_unbiased_and_moves_forward() {
        let c = WindowsClock::new();
        let before = mono_nanos();
        let ours = c.now_mono().as_nanos();
        let after = mono_nanos();
        assert!(before <= ours && ours <= after);
        #[expect(
            clippy::disallowed_methods,
            reason = "the test compares with std's clock"
        )]
        let i0 = std::time::Instant::now();
        let a = c.now_mono();
        std::thread::sleep(5 * MS);
        let b = c.now_mono();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test compares with std's clock"
        )]
        let i1 = std::time::Instant::now();
        let elapsed = b.saturating_duration_since(a);
        assert!(elapsed >= 5 * MS);
        // The unbiased interrupt timer has ~15ms resolution by default, so allow slack over
        // Instant's finer count.
        assert!(elapsed <= i1.duration_since(i0) + 20 * MS);
    }

    /// The wall clock is nanoseconds since the Unix epoch
    #[test]
    fn wall_is_unix_epoch_nanos() {
        let c = WindowsClock::new();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test compares with std's clock"
        )]
        let sys = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let ours = Duration::from_nanos(c.now_wall().as_nanos());
        assert!(ours.abs_diff(sys) < Duration::from_secs(1));
    }

    /// A wait with nobody notifying times out at its deadline, on either clock; a past deadline
    /// answers timeout at once
    #[test]
    fn deadlines_time_out() {
        let c = WindowsClock::new();
        let m = Mutex::new(());
        let cv = Condvar::new();
        let mut g = m.lock();
        for deadline in [
            Deadline::Mono(c.now_mono().saturating_add(40 * MS)),
            Deadline::Wall(c.now_wall().saturating_add(40 * MS)),
        ] {
            while !c.wait_until(&cv, &mut g, deadline).timed_out() {}
            assert!(deadline.reached(&c));
        }
        let past = Deadline::Mono(MonoTime::from_nanos(0));
        assert_eq!(c.wait_until(&cv, &mut g, past), TimedOut(true));
        let past = Deadline::Wall(WallTime::from_nanos(0));
        assert_eq!(c.wait_until(&cv, &mut g, past), TimedOut(true));
    }

    /// A notify before the deadline answers no_timeout, and an untimed wait returns on it
    #[test]
    fn a_notify_ends_a_wait() {
        let c = Arc::new(WindowsClock::new());
        for timed in [true, false] {
            let state = Arc::new((Mutex::new(false), Condvar::new()));
            let s2 = Arc::clone(&state);
            let mut g = state.0.lock();
            let h = std::thread::Builder::new()
                .spawn(move || {
                    *s2.0.lock() = true;
                    s2.1.notify_all();
                })
                .unwrap();
            let deadline = Deadline::Mono(c.now_mono().saturating_add(Duration::from_secs(10)));
            while !*g {
                if timed {
                    assert_eq!(c.wait_until(&state.1, &mut g, deadline), TimedOut(false));
                } else {
                    c.wait(&state.1, &mut g);
                }
            }
            drop(g);
            h.join().unwrap();
        }
    }

    /// sleep_until returns once its own clock reaches the deadline
    #[test]
    fn sleeps_to_the_deadline_on_each_clock() {
        let c = WindowsClock::new();
        let d = Deadline::Mono(c.now_mono().saturating_add(20 * MS));
        c.sleep_until(d);
        assert!(d.reached(&c));
        let d = Deadline::Wall(c.now_wall().saturating_add(20 * MS));
        c.sleep_until(d);
        assert!(d.reached(&c));
        c.sleep_until(Deadline::Mono(MonoTime::from_nanos(0)));
    }
}
