//! An opt-in sampling profiler for the library's own hot threads (`REIMS_VGPU_SAMPLE=1`, Windows
//! only): the decode workers and the FIFO threads register themselves, and a sampler thread wakes
//! about once a millisecond to suspend each one, read its instruction pointer and resume it. Only
//! registered threads are ever suspended (never a vCPU or a host thread), and nothing is
//! allocated or locked while one is suspended: the samples go into a preallocated buffer.
//!
//! Every 10 s the samples taken so far are appended to `samples-<pid>.bin` in the working
//! directory: a 16-byte header (`b"RVGS"`, a version, the executable's load address), then each
//! sample as a little-endian `u64` instruction pointer. `qemu-vm/sample-report.py` turns them into
//! a flat profile by function with `addr2line`.

/// Registers the calling thread for sampling, where sampling is on.
#[cfg(windows)]
pub fn register_current_thread() {
    imp::register();
}

/// Nothing off Windows.
#[cfg(not(windows))]
pub fn register_current_thread() {}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 thread handles, a CONTEXT record and the module base
mod imp {
    use std::io::Write as _;
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetCurrentThread() -> isize;
        fn DuplicateHandle(
            source_process: isize,
            source: isize,
            target_process: isize,
            target: *mut isize,
            access: u32,
            inherit: i32,
            options: u32,
        ) -> i32;
        fn SuspendThread(thread: isize) -> u32;
        fn ResumeThread(thread: isize) -> u32;
        fn GetThreadContext(thread: isize, context: *mut u8) -> i32;
        fn GetModuleHandleW(name: *const u16) -> isize;
    }

    /// `THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT`.
    const ACCESS: u32 = 0x0002 | 0x0008;
    /// `CONTEXT_AMD64 | CONTEXT_CONTROL`.
    const CONTEXT_CONTROL: u32 = 0x0010_0001;
    /// The x64 `CONTEXT`'s size, and the offsets of `ContextFlags` and `Rip`.
    const CONTEXT_SIZE: usize = 1232;
    const FLAGS_AT: usize = 0x30;
    const RIP_AT: usize = 0xF8;

    /// A 16-byte aligned `CONTEXT`.
    #[repr(C, align(16))]
    struct Context([u8; CONTEXT_SIZE]);

    fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("REIMS_VGPU_SAMPLE").is_ok_and(|v| v == "1"))
    }

    /// The registered threads' handles (as `isize`).
    static THREADS: Mutex<Vec<isize>> = Mutex::new(Vec::new());

    pub(super) fn register() {
        if !enabled() {
            return;
        }
        let mut handle: isize = 0;
        // SAFETY: duplicates the calling thread's pseudo-handle into a real one we own.
        let ok = unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                GetCurrentThread(),
                GetCurrentProcess(),
                &mut handle,
                ACCESS,
                0,
                0,
            )
        };
        if ok == 0 {
            return;
        }
        THREADS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(handle);
        static STARTED: OnceLock<()> = OnceLock::new();
        STARTED.get_or_init(|| {
            let _ = std::thread::Builder::new()
                .name("reims-vgpu.sampler".into())
                .spawn(run);
        });
    }

    fn run() {
        // SAFETY: a null name asks for the executable's own module.
        let base = unsafe { GetModuleHandleW(std::ptr::null()) } as u64;
        let path = format!("samples-{}.bin", std::process::id());
        if let Ok(mut f) = std::fs::File::create(&path) {
            let mut header = Vec::with_capacity(16);
            header.extend_from_slice(b"RVGS");
            header.extend_from_slice(&1u32.to_le_bytes());
            header.extend_from_slice(&base.to_le_bytes());
            let _ = f.write_all(&header);
        }
        log::warn!("sampler: writing samples of the decode and FIFO threads to {path}");
        let mut samples: Vec<u64> = Vec::with_capacity(1 << 20);
        let mut threads: Vec<isize> = Vec::new();
        let mut context = Box::new(Context([0; CONTEXT_SIZE]));
        let mut flushed = Instant::now();
        loop {
            std::thread::sleep(Duration::from_micros(500));
            // The list is copied with no thread suspended
            {
                let list = THREADS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if list.len() != threads.len() {
                    threads.clear();
                    threads.extend_from_slice(&list);
                }
            }
            for &thread in &threads {
                if samples.len() == samples.capacity() {
                    break;
                }
                // SAFETY: `thread` is a handle we own with suspend and context access; the
                // context record is aligned and sized for x64; nothing between the suspend and
                // the resume allocates or locks.
                unsafe {
                    if SuspendThread(thread) == u32::MAX {
                        continue;
                    }
                    context.0[FLAGS_AT..FLAGS_AT + 4].copy_from_slice(&CONTEXT_CONTROL.to_le_bytes());
                    let ok = GetThreadContext(thread, context.0.as_mut_ptr());
                    ResumeThread(thread);
                    if ok != 0 {
                        let mut rip = [0u8; 8];
                        rip.copy_from_slice(&context.0[RIP_AT..RIP_AT + 8]);
                        samples.push(u64::from_le_bytes(rip));
                    }
                }
            }
            if flushed.elapsed() >= Duration::from_secs(10) && !samples.is_empty() {
                if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&path) {
                    let mut bytes = Vec::with_capacity(samples.len() * 8);
                    for s in &samples {
                        bytes.extend_from_slice(&s.to_le_bytes());
                    }
                    let _ = f.write_all(&bytes);
                }
                samples.clear();
                flushed = Instant::now();
            }
        }
    }
}
