use std::sync::atomic::AtomicUsize;

use reims_vgpu_core::sync::MutexGuard;
use reims_vgpu_core::testkit::FakePlatform;

use super::*;

/// A clock the test sets: `now_on` answers its time, and a wait returns at once.
#[derive(Default)]
struct SetClock(AtomicU64);

impl SetClock {
    fn at(&self, ms: u64) {
        self.0.store(ms * 1_000_000, Relaxed);
    }
}

impl TimerClock for SetClock {
    fn now_on(&self, _: Deadline) -> u64 {
        self.0.load(Relaxed)
    }

    fn wait_until(&self, _: &Condvar, _: &mut MutexGuard<'_, bool>, _: Deadline) {}
}

/// Stats of a display on `port` over a clock at `start_ms`, and the clock.
fn stats(port: u64, window: bool, start_ms: u64) -> (Arc<SetClock>, DisplayStats) {
    let clock = Arc::new(SetClock::default());
    clock.at(start_ms);
    let stats = DisplayStats::new(clock.clone(), port, window);
    (clock, stats)
}

/// The usual words turn the switch on, around spaces and in any case; anything else is off
#[test]
fn the_switch_takes_the_usual_words() {
    for on in ["1", "true", "YES", " on "] {
        assert!(switch(on), "{on}");
    }
    for off in ["", "0", "off", "2", "enable"] {
        assert!(!switch(off), "{off}");
    }
}

/// A line gives the guest's rate over the period and its longest gap; before the first frame
/// the gap is `-`, and a display without a window has no window part
#[test]
fn a_line_without_a_window() {
    let (clock, s) = stats(3, false, 5);
    clock.at(1005);
    assert_eq!(
        s.report(),
        "reims-vgpu stats: port 3: guest 0.0/s, max gap -"
    );
    for ms in [1010, 1020, 1050, 1060] {
        clock.at(ms);
        s.guest_present();
    }
    clock.at(1505);
    assert_eq!(
        s.report(),
        "reims-vgpu stats: port 3: guest 8.0/s, max gap 445.0 ms"
    );
}

/// The longest gap is the longest between two frames of the period, or the one still open
/// when it is longer: a guest that stops presenting shows a gap that grows each line
#[test]
fn the_open_gap_counts() {
    let (clock, s) = stats(0, false, 0);
    for ms in [100, 130, 140] {
        clock.at(ms);
        s.guest_present();
    }
    clock.at(150);
    assert!(s.report().ends_with("max gap 30.0 ms"));
    clock.at(1150);
    assert!(s.report().ends_with("guest 0.0/s, max gap 1010.0 ms"));
    clock.at(2150);
    assert!(s.report().ends_with("max gap 2010.0 ms"));
    clock.at(2160);
    s.guest_present();
    clock.at(2170);
    assert!(s.report().ends_with("max gap 2020.0 ms"));
    clock.at(2180);
    assert!(s.report().ends_with("max gap 20.0 ms"));
}

/// A window display's line adds the sink's draws and their GPU times, each period's own
#[test]
fn a_line_with_a_window() {
    let (clock, s) = stats(1, true, 0);
    for ms in [10, 26, 42] {
        clock.at(ms);
        s.guest_present();
        s.window_present();
    }
    s.window_gpu_time(Duration::from_micros(400));
    s.window_gpu_time(Duration::from_micros(1_000));
    s.window_gpu_time(Duration::from_micros(700));
    s.window_stage_time(WindowStage::Acquire, Duration::from_millis(1));
    s.window_stage_time(WindowStage::Acquire, Duration::from_millis(2));
    s.window_stage_time(WindowStage::Encode, Duration::from_micros(500));
    s.window_stage_time(WindowStage::PresentCall, Duration::from_millis(4));
    s.window_outcome(WindowOutcome::Presented);
    s.window_outcome(WindowOutcome::Retry);
    s.window_outcome(WindowOutcome::Retry);
    clock.at(500);
    assert_eq!(
        s.report(),
        "reims-vgpu stats: port 1: guest 6.0/s, max gap 458.0 ms; window 6.0/s, max gap 458.0 ms; \
         window GPU 0.70 ms avg, 1.00 ms max; window CPU acquire 2 calls, 3.00 ms total, 2.00 ms max; \
         window CPU encode 1 calls, 0.50 ms total, 0.50 ms max; window CPU present-call 1 calls, \
         4.00 ms total, 4.00 ms max; window outcomes 1 presented, 2 retry, 0 ended, 0 nothing"
    );
    clock.at(1000);
    assert_eq!(
        s.report(),
        "reims-vgpu stats: port 1: guest 0.0/s, max gap 958.0 ms; window 0.0/s, max gap 958.0 ms; \
         window GPU -; window CPU acquire 0 calls, 0.00 ms total, 0.00 ms max; window CPU encode \
         0 calls, 0.00 ms total, 0.00 ms max; window CPU present-call 0 calls, 0.00 ms total, \
         0.00 ms max; window outcomes 0 presented, 0 retry, 0 ended, 0 nothing"
    );
}

/// Wrapping counts each new frame before the callback runs; without a clock or a callback
/// there are no stats and the callback is as it was
#[test]
fn wrapping_counts_each_new_frame() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let inner: NewFrameFn = Arc::new(move || {
        c.fetch_add(1, Relaxed);
    });
    let clock: Arc<dyn TimerClock> = Arc::new(SetClock::default());

    let mut none: Option<NewFrameFn> = None;
    assert!(wrap(Some(clock.clone()), 0, false, &mut none).is_none());
    assert!(none.is_none());
    let mut kept = Some(inner.clone());
    assert!(wrap(None, 0, false, &mut kept).is_none());
    assert!(Arc::ptr_eq(kept.as_ref().unwrap(), &inner));

    let mut new_frame = Some(inner.clone());
    let s = wrap(Some(clock), 0, false, &mut new_frame).unwrap();
    assert!(!Arc::ptr_eq(new_frame.as_ref().unwrap(), &inner));
    new_frame.as_ref().unwrap()();
    new_frame.as_ref().unwrap()();
    assert_eq!(calls.load(Relaxed), 2);
    assert_eq!(s.guest.count.load(Relaxed), 2);
    // The wrapper holds the stats and the callback, and drops both with it.
    drop(new_frame);
    assert_eq!((Arc::strong_count(&s), Arc::strong_count(&inner)), (1, 2));
}

/// The registry finds a display's stats while they live, keeps one entry per address, and
/// gives the live ones by port
#[test]
fn the_registry_finds_live_stats_by_address() {
    let mut r = Registry {
        displays: Vec::new(),
        reporting: false,
    };
    let (_c2, two) = stats(2, false, 0);
    let (_c0, zero) = stats(0, false, 0);
    let (_c1, one) = stats(1, false, 0);
    let (two, zero, one) = (Arc::new(two), Arc::new(zero), Arc::new(one));
    r.add(0x20, &two);
    r.add(0x10, &zero);
    r.add(0x30, &one);
    assert!(Arc::ptr_eq(&r.find(0x10).unwrap(), &zero));
    assert!(r.find(0x40).is_none());
    let ports: Vec<u64> = r.live().iter().map(|s| s.port).collect();
    assert_eq!(ports, [0, 1, 2]);
    // An ended display's stats are not found, and its entry goes at the next live().
    drop(zero);
    assert!(r.find(0x10).is_none());
    assert_eq!(r.live().len(), 2);
    assert_eq!(r.displays.len(), 2);
    // A display at an address another left takes its entry.
    let (_c3, three) = stats(3, false, 0);
    let three = Arc::new(three);
    r.add(0x20, &three);
    assert!(Arc::ptr_eq(&r.find(0x20).unwrap(), &three));
    assert_eq!(r.displays.len(), 2);
    let text = lines(&r.live());
    assert_eq!(text.lines().count(), 2);
    assert!(text.starts_with("reims-vgpu stats: port 1: ") && text.ends_with('\n'));
}

/// A sink finds its display's stats by its reference to the display, while they live
#[test]
fn a_sink_finds_its_display_s_stats() {
    let display = Weak::<CoreDisplay<FakePlatform>>::new();
    assert!(of_display(&display).is_none());
    let (_clock, s) = stats(5, true, 0);
    let s = Arc::new(s);
    registry().lock().add(key(&display), &s);
    assert!(Arc::ptr_eq(&of_display(&display).unwrap(), &s));
    drop(s);
    assert!(of_display(&display).is_none());
}
