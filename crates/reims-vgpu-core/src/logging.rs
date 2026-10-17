//! The library's `log` logger: silent by default, stderr with `REIMS_VGPU_LOG`, or the host's
//! hook.
//!
//! The library writes its lines through the `log` crate's macros, each under its module's path as
//! its target (`reims_vgpu_core::fifo::thread`). Where they go:
//!
//! - a host that installed its own `log` logger gets them there, filtered as it filters;
//! - otherwise nowhere, unless `REIMS_VGPU_LOG` is set or the host gave a hook ([`set_hook`]).
//!   Then [`init`] or [`set_hook`] installs this module's logger, once per process, and each line
//!   goes to the hook if there is one, and to stderr otherwise, as
//!   `reims-vgpu <LEVEL> <target>: <message>`.
//!
//! `REIMS_VGPU_LOG` is a level (`off`, `error`, `warn`, `info`, `debug` or `trace`), optionally
//! followed by `,target=level` pairs: `warn,reims_vgpu_core::fifo=debug` ([`Filter::parse`]).
//! When it is set, its filter decides which lines are written, to the hook too; when it isn't,
//! the hook's own level does.
//!
//! With `REIMS_VGPU_LOG_ASYNC=1` the stderr lines below error are handed, formatted and
//! timestamped, to a writer thread over a bounded queue ([`async_writer`]), so the thread that
//! logs (the FIFO's, on the GPU path) never waits on the disk: a synchronous write to a growing
//! log file was measured stalling it 203 ms. A full queue drops the line and counts it; error
//! lines are still written in place, so a fatal's last words are never queued.
//!
//! Which lines are written, and whether a line's arguments are evaluated, are diagnostics: a line
//! evaluates its arguments only when a logger takes it, so no argument has an effect. A value a
//! line shows that has one (a read of guest memory, a question to Metal) is read before the line,
//! whether or not it is written.

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::{Arc, OnceLock};

use log::{Level, LevelFilter, Log, Metadata, Record};

use crate::sync::Mutex;

/// The environment variable that turns the stderr logger on, and holds its filter.
pub const ENV_VAR: &str = "REIMS_VGPU_LOG";

/// A host's receiver of lines: a line's level, target and message, on the thread that writes
/// the line.
pub type Hook = Box<dyn Fn(Level, &str, &str) + Send + Sync>;

/// A hook as the logger keeps it: its level, and the hook, shared with the lines being written.
type HookEntry = (LevelFilter, Arc<dyn Fn(Level, &str, &str) + Send + Sync>);

/// The level each target is written up to: a level per target prefix, and one for the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    rest: LevelFilter,
    targets: Vec<(String, LevelFilter)>,
}

impl Filter {
    /// Every target up to `level`.
    pub fn all(level: LevelFilter) -> Filter {
        Filter {
            rest: level,
            targets: Vec::new(),
        }
    }

    /// `spec`, read as `REIMS_VGPU_LOG` is: comma-separated entries, each a level (case aside)
    /// or `target=level`.
    ///
    /// - A lone level covers every target no pair names; without one, those are `off`.
    /// - A pair covers its target and the targets below it at a `::`: `reims_vgpu` covers
    ///   `reims_vgpu::fifo`. The longest covering target decides, and among equal ones the
    ///   last.
    /// - An entry that is neither is ignored.
    pub fn parse(spec: &str) -> Filter {
        let mut filter = Filter::all(LevelFilter::Off);
        for entry in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            match entry.split_once('=') {
                Some((target, level)) => {
                    if let Ok(level) = level.trim().parse() {
                        filter.targets.push((target.trim().to_owned(), level));
                    }
                }
                None => {
                    if let Ok(level) = entry.parse() {
                        filter.rest = level;
                    }
                }
            }
        }
        filter
    }

    /// The level lines of `target` are written up to.
    pub fn level(&self, target: &str) -> LevelFilter {
        self.targets
            .iter()
            .filter(|(prefix, _)| covers(prefix, target))
            .max_by_key(|(prefix, _)| prefix.len())
            .map_or(self.rest, |(_, level)| *level)
    }

    /// The most verbose level any target is written up to.
    pub fn max(&self) -> LevelFilter {
        self.targets
            .iter()
            .map(|(_, level)| *level)
            .fold(self.rest, Ord::max)
    }
}

/// Whether the filter's `prefix` covers `target`: it is `target`, or `target`'s path above a `::`.
fn covers(prefix: &str, target: &str) -> bool {
    target
        .strip_prefix(prefix)
        .is_some_and(|below| below.is_empty() || below.starts_with("::"))
}

/// A line as stderr shows it, with its newline.
pub fn stderr_line(level: Level, target: &str, message: &str) -> String {
    format!("reims-vgpu {level} {target}: {message}\n")
}

/// The environment variable that prefixes each stderr line with its wall-clock time and thread, a
/// diagnostic for lining lines up with a host's own timestamped trace.
pub const TIME_ENV_VAR: &str = "REIMS_VGPU_LOG_TIME";

/// Whether `REIMS_VGPU_LOG_TIME` is set, read on first use.
fn timestamps() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(TIME_ENV_VAR).is_some())
}

/// `line` after `[<UTC seconds since the epoch, to the microsecond> <thread>] `.
fn timestamped(line: String) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let thread = std::thread::current();
    let name = thread.name().unwrap_or("?");
    format!("[{}.{:06} {name}] {line}", now.as_secs(), now.subsec_micros())
}

/// The logger: the environment's filter, the host's hook, and stderr.
struct Logger {
    /// `REIMS_VGPU_LOG`'s filter, read on first use; `None` if it is not set.
    env: OnceLock<Option<Filter>>,
    hook: Mutex<Option<HookEntry>>,
}

impl Logger {
    const fn new() -> Logger {
        Logger {
            env: OnceLock::new(),
            hook: Mutex::new(None),
        }
    }

    fn env(&self) -> Option<&Filter> {
        self.env
            .get_or_init(|| std::env::var(ENV_VAR).ok().map(|spec| Filter::parse(&spec)))
            .as_ref()
    }

    /// The level lines of `target` are written up to, given the hook's level if there is a
    /// hook.
    fn level(&self, target: &str, hook: Option<LevelFilter>) -> LevelFilter {
        match (self.env(), hook) {
            (Some(env), _) => env.level(target),
            (None, Some(level)) => level,
            (None, None) => LevelFilter::Off,
        }
    }

    /// The most verbose level any line is written up to, given the hook's level.
    fn max_level(&self, hook: Option<LevelFilter>) -> LevelFilter {
        match (self.env(), hook) {
            (Some(env), _) => env.max(),
            (None, Some(level)) => level,
            (None, None) => LevelFilter::Off,
        }
    }
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        let hook = self.hook.lock().as_ref().map(|(level, _)| *level);
        metadata.level() <= self.level(metadata.target(), hook)
    }

    fn log(&self, record: &Record<'_>) {
        // The hook is called without the lock: it may write lines of its own, or be replaced.
        let hook = self.hook.lock().clone();
        if record.level() > self.level(record.target(), hook.as_ref().map(|(level, _)| *level)) {
            return;
        }
        let mut message = String::new();
        // A `Display` impl that fails leaves what it wrote.
        let _ = write!(message, "{}", record.args());
        match hook {
            Some((_, hook)) => hook(record.level(), record.target(), &message),
            None => {
                let mut line = stderr_line(record.level(), record.target(), &message);
                if timestamps() {
                    line = timestamped(line);
                }
                if record.level() > Level::Error
                    && let Some(tx) = async_writer()
                {
                    let dropped = DROPPED.swap(0, std::sync::atomic::Ordering::Relaxed);
                    if dropped != 0 {
                        let note = stderr_line(
                            Level::Warn,
                            "reims_vgpu_core::logging",
                            &format!("{dropped} log lines dropped: the writer fell behind"),
                        );
                        let _ = tx.try_send(note);
                    }
                    if tx.try_send(line).is_err() {
                        DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    return;
                }
                // One write per line, so lines from different threads do not interleave.
                let _ = std::io::stderr().write_all(line.as_bytes());
            }
        }
    }

    fn flush(&self) {}
}

static LOGGER: Logger = Logger::new();

/// Lines the asynchronous writer's full queue dropped since the last written.
static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The queue of the thread that writes stderr lines, when `REIMS_VGPU_LOG_ASYNC=1` (the module
/// doc); `None` otherwise, or where the thread could not start. Made on first use.
fn async_writer() -> Option<&'static std::sync::mpsc::SyncSender<String>> {
    static TX: OnceLock<Option<std::sync::mpsc::SyncSender<String>>> = OnceLock::new();
    TX.get_or_init(|| {
        if !std::env::var("REIMS_VGPU_LOG_ASYNC").is_ok_and(|v| v == "1") {
            return None;
        }
        let (tx, rx) = std::sync::mpsc::sync_channel::<String>(1 << 16);
        std::thread::Builder::new()
            .name("reims-vgpu.log".to_owned())
            .spawn(move || {
                let mut err = std::io::stderr();
                while let Ok(line) = rx.recv() {
                    let _ = err.write_all(line.as_bytes());
                }
            })
            .ok()?;
        Some(tx)
    })
    .as_ref()
}

/// Whether this module's logger is the process's: `Some(true)` once installed, `Some(false)` if
/// the host's was there first, `None` before anything asked.
static OURS: OnceLock<bool> = OnceLock::new();

/// Installs this module's logger unless the process has one: once per process.
fn install() -> bool {
    *OURS.get_or_init(|| log::set_logger(&LOGGER).is_ok())
}

/// Sets `log`'s global level to what the logger writes now. `hook` is the hook's level, read
/// under the hook's lock so a concurrent [`set_hook`] cannot leave a stale level.
fn refresh_max_level(hook: Option<LevelFilter>) {
    if OURS.get() == Some(&true) {
        log::set_max_level(LOGGER.max_level(hook));
    }
}

/// Installs the stderr logger if `REIMS_VGPU_LOG` is set, unless the process already has a
/// `log` logger (the host's). The product calls it when it makes its platform; a later call
/// changes nothing.
pub fn init() {
    if LOGGER.env().is_some() && install() {
        let hook = LOGGER.hook.lock();
        refresh_max_level(hook.as_ref().map(|(level, _)| *level));
    }
}

/// Sends every line up to `level` to `hook`, in place of stderr, from now on: the hook is called
/// synchronously on the thread that writes the line. When `REIMS_VGPU_LOG` is set, its filter
/// decides instead of `level`.
///
/// Installs this module's logger if it is not yet the process's. Answers false, and keeps no
/// hook, if the host installed its own `log` logger first: the lines go there.
pub fn set_hook(level: LevelFilter, hook: Hook) -> bool {
    if !install() {
        return false;
    }
    let mut slot = LOGGER.hook.lock();
    *slot = Some((level, Arc::from(hook)));
    refresh_max_level(Some(level));
    true
}

/// Removes the hook: lines go to stderr again if `REIMS_VGPU_LOG` is set, and nowhere otherwise.
/// A line being written on another thread may still reach the old hook.
pub fn clear_hook() {
    let mut slot = LOGGER.hook.lock();
    *slot = None;
    refresh_max_level(None);
}

#[cfg(test)]
mod tests {
    use super::*;

    type Seen = Arc<Mutex<Vec<(Level, String, String)>>>;

    /// A hook that keeps what it is given.
    fn recording_hook() -> (Seen, Hook) {
        let seen: Seen = Arc::default();
        let kept = Arc::clone(&seen);
        let hook: Hook = Box::new(move |level, target, message| {
            kept.lock()
                .push((level, target.to_owned(), message.to_owned()));
        });
        (seen, hook)
    }

    /// A logger that read `env` from the environment, with no hook yet.
    fn logger(env: Option<&str>) -> Logger {
        Logger {
            env: OnceLock::from(env.map(Filter::parse)),
            hook: Mutex::new(None),
        }
    }

    fn write(l: &Logger, level: Level, target: &str, text: &str) {
        l.log(
            &Record::builder()
                .args(format_args!("{text}"))
                .level(level)
                .target(target)
                .build(),
        );
    }

    fn enabled(l: &Logger, level: Level, target: &str) -> bool {
        l.enabled(&Metadata::builder().level(level).target(target).build())
    }

    /// A lone level covers every target; pairs cover their target and those below it
    #[test]
    fn filters_read_a_level_and_target_pairs() {
        let f = Filter::parse("warn,reims_vgpu::fifo=debug");
        assert_eq!(f.level("reims_vgpu::fifo"), LevelFilter::Debug);
        assert_eq!(f.level("reims_vgpu::fifo::thread"), LevelFilter::Debug);
        assert_eq!(f.level("reims_vgpu::fifos"), LevelFilter::Warn);
        assert_eq!(f.level("reims_vgpu::device"), LevelFilter::Warn);
        assert_eq!(f.max(), LevelFilter::Debug);

        let f = Filter::parse(" INFO , reims_vgpu = error , reims_vgpu::tasks=trace ");
        assert_eq!(f.level("other"), LevelFilter::Info);
        assert_eq!(f.level("reims_vgpu::fifo"), LevelFilter::Error);
        assert_eq!(f.level("reims_vgpu::tasks"), LevelFilter::Trace);
        assert_eq!(f.max(), LevelFilter::Trace);
    }

    /// Without a lone level the rest is off; unreadable entries are ignored; the last of equal
    /// pairs decides
    #[test]
    fn filters_default_to_off_and_skip_what_they_cannot_read() {
        let f = Filter::parse("reims_vgpu::fifo=info");
        assert_eq!(f.level("reims_vgpu::fifo"), LevelFilter::Info);
        assert_eq!(f.level("reims_vgpu::device"), LevelFilter::Off);
        assert_eq!(Filter::parse(""), Filter::all(LevelFilter::Off));
        assert_eq!(
            Filter::parse("loud,debug,x=loud,,=,y"),
            Filter::all(LevelFilter::Debug)
        );
        let f = Filter::parse("a=debug,a=error");
        assert_eq!(f.level("a"), LevelFilter::Error);
        assert_eq!(f.max(), LevelFilter::Debug);
    }

    /// A stderr line names the library, the level and the target
    #[test]
    fn stderr_lines_name_level_and_target() {
        assert_eq!(
            stderr_line(Level::Warn, "reims_vgpu::fifo", "stalled 3"),
            "reims-vgpu WARN reims_vgpu::fifo: stalled 3\n"
        );
    }

    /// With neither the variable nor a hook, nothing is enabled
    #[test]
    fn silent_without_the_variable_or_a_hook() {
        let l = logger(None);
        assert!(!enabled(&l, Level::Error, "reims_vgpu::device"));
        assert_eq!(l.max_level(None), LevelFilter::Off);
    }

    /// A hook takes every line up to its level; the variable's filter, when set, decides for it
    #[test]
    fn a_hook_takes_the_lines_its_filter_passes() {
        let l = logger(None);
        let (seen, hook) = recording_hook();
        *l.hook.lock() = Some((LevelFilter::Info, Arc::from(hook)));
        assert!(enabled(&l, Level::Info, "reims_vgpu::fifo"));
        assert!(!enabled(&l, Level::Debug, "reims_vgpu::fifo"));
        write(&l, Level::Warn, "reims_vgpu::fifo", "kept");
        write(&l, Level::Debug, "reims_vgpu::fifo", "dropped");
        assert_eq!(
            *seen.lock(),
            [(
                Level::Warn,
                "reims_vgpu::fifo".to_owned(),
                "kept".to_owned()
            )]
        );

        let l = logger(Some("error,reims_vgpu::tasks=debug"));
        let (seen, hook) = recording_hook();
        *l.hook.lock() = Some((LevelFilter::Trace, Arc::from(hook)));
        assert_eq!(l.max_level(Some(LevelFilter::Trace)), LevelFilter::Debug);
        write(&l, Level::Warn, "reims_vgpu::fifo", "dropped");
        write(&l, Level::Debug, "reims_vgpu::tasks", "kept");
        assert_eq!(
            *seen.lock(),
            [(
                Level::Debug,
                "reims_vgpu::tasks".to_owned(),
                "kept".to_owned()
            )]
        );
    }

    /// The marker that makes a test the child of `the_process_logger_*`.
    const CHILD: &str = "REIMS_VGPU_LOGGING_CHILD";

    /// Re-runs this binary on the test `name`, as a child, with `REIMS_VGPU_LOG` set to `env`.
    fn child(name: &str, env: Option<&str>) -> std::process::Output {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", name, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .env_remove(ENV_VAR);
        if let Some(env) = env {
            cmd.env(ENV_VAR, env);
        }
        cmd.output().unwrap()
    }

    /// With the variable set, init installs the stderr logger; a hook then takes the lines in
    /// its place, and stderr has them back once it is cleared
    #[test]
    fn the_process_logger_writes_stderr_until_a_hook_takes_over() {
        if std::env::var_os(CHILD).is_some() {
            init();
            init();
            assert_eq!(log::max_level(), LevelFilter::Debug);
            log::debug!(target: "reims_vgpu::fifo", "to stderr {}", 1);
            log::debug!(target: "reims_vgpu::device", "filtered");
            log::info!(target: "reims_vgpu::device", "to stderr {}", 2);
            let (seen, hook) = recording_hook();
            assert!(set_hook(LevelFilter::Error, hook));
            log::info!(target: "reims_vgpu::device", "to the hook");
            clear_hook();
            log::error!(target: "reims_vgpu::tasks", "to stderr {}", 3);
            assert_eq!(
                *seen.lock(),
                [(
                    Level::Info,
                    "reims_vgpu::device".to_owned(),
                    "to the hook".to_owned()
                )]
            );
            return;
        }
        let out = child(
            "logging::tests::the_process_logger_writes_stderr_until_a_hook_takes_over",
            Some("info,reims_vgpu::fifo=debug"),
        );
        assert!(out.status.success(), "{out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        let lines: Vec<&str> = err
            .lines()
            .filter(|l| l.starts_with("reims-vgpu "))
            .collect();
        assert_eq!(
            lines,
            [
                "reims-vgpu DEBUG reims_vgpu::fifo: to stderr 1",
                "reims-vgpu INFO reims_vgpu::device: to stderr 2",
                "reims-vgpu ERROR reims_vgpu::tasks: to stderr 3",
            ],
            "{err}"
        );
    }

    /// Without the variable, init installs nothing and the process stays silent; a hook still
    /// gets the lines up to its own level
    #[test]
    fn the_process_logger_is_silent_without_the_variable() {
        if std::env::var_os(CHILD).is_some() {
            init();
            assert_eq!(log::max_level(), LevelFilter::Off);
            log::error!(target: "reims_vgpu::device", "nowhere");
            let (seen, hook) = recording_hook();
            assert!(set_hook(LevelFilter::Warn, hook));
            assert_eq!(log::max_level(), LevelFilter::Warn);
            log::warn!(target: "reims_vgpu::device", "hooked");
            log::info!(target: "reims_vgpu::device", "too fine");
            clear_hook();
            assert_eq!(log::max_level(), LevelFilter::Off);
            log::error!(target: "reims_vgpu::device", "nowhere again");
            assert_eq!(
                *seen.lock(),
                [(
                    Level::Warn,
                    "reims_vgpu::device".to_owned(),
                    "hooked".to_owned()
                )]
            );
            return;
        }
        let out = child(
            "logging::tests::the_process_logger_is_silent_without_the_variable",
            None,
        );
        assert!(out.status.success(), "{out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!err.contains("reims-vgpu "), "{err}");
    }

    /// A logger the host installed first keeps the lines: init leaves it, and set_hook refuses
    #[test]
    fn the_process_logger_yields_to_the_hosts() {
        struct Host;
        impl Log for Host {
            fn enabled(&self, _: &Metadata<'_>) -> bool {
                true
            }
            fn log(&self, record: &Record<'_>) {
                eprintln!("host {}", record.args());
            }
            fn flush(&self) {}
        }
        static HOST: Host = Host;

        if std::env::var_os(CHILD).is_some() {
            log::set_logger(&HOST).unwrap();
            log::set_max_level(LevelFilter::Info);
            init();
            let (_, hook) = recording_hook();
            assert!(!set_hook(LevelFilter::Trace, hook));
            clear_hook();
            assert_eq!(log::max_level(), LevelFilter::Info);
            log::warn!(target: "reims_vgpu::device", "to the host");
            return;
        }
        let out = child(
            "logging::tests::the_process_logger_yields_to_the_hosts",
            Some("trace"),
        );
        assert!(out.status.success(), "{out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("host to the host"), "{err}");
        assert!(!err.contains("reims-vgpu "), "{err}");
    }
}
