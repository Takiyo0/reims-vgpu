//! A FIFO's decode workers: the second half of its exec commands (the decode into the exec
//! command buffer, its commit and the synchronize buffer), run on threads of their own while the
//! FIFO thread reads on.
//!
//! The FIFO thread keeps everything the next command can observe: the barrier wait, the stamp, the
//! preparation (paging in) and the copy of the command buffers' bytes, and it reserves the FIFO
//! event's value before it hands the rest on, so the next exec's paging buffer waits for this
//! exec's value as it did inline.
//!
//! Several workers decode at once (Metal lets different command buffers encode on different
//! threads), each into its exec's own command buffer, but every commit waits its turn
//! ([`Turn::wait`]): exec command buffers commit in the FIFO's order. A job's turn passes when it
//! ends, whatever it did ([`Turn`]'s drop).
//!
//! Every other command on the FIFO first waits until the workers are idle
//! ([`DecodeWorker::drain`]): a command that changes the task's object lists or ends the task
//! never overtakes an exec submitted before it. The FIFO's quiesce and its thread's end drain them
//! too.
//!
//! `REIMS_VGPU_ASYNC_DECODE=0` runs every exec inline, as before; `REIMS_VGPU_DECODE_WORKERS=n`
//! sets the workers per FIFO (default 4).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};

/// One exec's second half; it commits once its turn comes.
pub type DecodeJob = Box<dyn FnOnce(&Turn) + Send>;

/// How many execs may be handed on and not finished (`DecodeWorker::wait_below`): fewer than the
/// command buffers a queue holds made and not committed.
pub const MAX_AHEAD: usize = 16;

/// Whether a job moves the work that needs no place in the commit order out of its turn
/// (`REIMS_VGPU_EARLY_COMMIT`, on unless `0`): the completed handler and the signal are encoded
/// before the wait, and the turn passes right after the commit (after the synchronize buffer's
/// commit when there is one), before the job's cleanup. Commits stay in the FIFO's order.
pub fn early_release() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_EARLY_COMMIT").map_or(true, |v| v != "0"))
}

/// Whether an exec's preparation (`prepare_resources`) runs on the workers, in the FIFO's order
/// before the decode (`REIMS_VGPU_ASYNC_PREPARE`, on unless `0`; needs the workers).
pub fn async_prepare() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    enabled() && *ON.get_or_init(|| std::env::var("REIMS_VGPU_ASYNC_PREPARE").map_or(true, |v| v != "0"))
}

/// Whether execs go to the workers (`REIMS_VGPU_ASYNC_DECODE`, on unless `0`).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_ASYNC_DECODE").map_or(true, |v| v != "0"))
}

/// The workers per FIFO (`REIMS_VGPU_DECODE_WORKERS`, 1 to 16, default 4).
fn worker_count() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("REIMS_VGPU_DECODE_WORKERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
            .clamp(1, 16)
    })
}

#[derive(Default)]
struct State {
    /// Jobs not yet started, with their sequence numbers.
    queue: VecDeque<(u64, DecodeJob)>,
    /// Jobs started and not finished.
    running: usize,
    /// Threads started.
    threads: usize,
    /// The next job's sequence number.
    next_seq: u64,
    /// The sequence number whose turn it is to commit.
    turn: u64,
    /// The sequence number whose turn it is to prepare (`Turn::wait_prepare`).
    prepare_turn: u64,
}

/// A FIFO's workers: started with the first jobs, running until the process ends.
#[derive(Default)]
pub struct DecodeWorker {
    state: Mutex<State>,
    /// Wakes a thread for a job.
    work: Condvar,
    /// Wakes drains and the FIFO thread's bound once a job finished.
    idle: Condvar,
    /// Wakes the jobs waiting for their turn.
    turns: Condvar,
}

/// A job's place in the commit order.
pub struct Turn<'a> {
    worker: &'a DecodeWorker,
    seq: u64,
    /// The turn was passed on by [`Turn::pass`]; the drop then does nothing.
    passed: std::cell::Cell<bool>,
    /// The job took part in the preparation order ([`Turn::wait_prepare`]).
    prep_used: std::cell::Cell<bool>,
    /// The preparation turn was passed on by [`Turn::pass_prepare`].
    prep_passed: std::cell::Cell<bool>,
}

impl Turn<'_> {
    /// Waits until every earlier job has passed its preparation turn: preparations run one at a
    /// time in the FIFO's order, alongside the other jobs' decodes. Must be paired with
    /// [`Turn::pass_prepare`] (the drop passes it if the job ends first). The caller holds no
    /// lock but its task's shared one (handed to it, so the wait never queues for the lock).
    pub fn wait_prepare(&self) {
        self.prep_used.set(true);
        let mut state = self.worker.lock();
        while state.prepare_turn != self.seq {
            state = self.worker.turns.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Passes the preparation turn on (after [`Turn::wait_prepare`]). Idempotent.
    pub fn pass_prepare(&self) {
        if !self.prep_used.get() || self.prep_passed.replace(true) {
            return;
        }
        let mut state = self.worker.lock();
        debug_assert_eq!(state.prepare_turn, self.seq);
        state.prepare_turn = state.prepare_turn.wrapping_add(1);
        drop(state);
        self.worker.turns.notify_all();
    }

    /// Waits until every earlier job has passed its turn.
    pub fn wait(&self) {
        if self.passed.get() {
            return;
        }
        let mut state = self.worker.lock();
        while state.turn != self.seq {
            state = self.worker.turns.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl Turn<'_> {
    /// Passes the turn on now, once it came (it waits for it as the drop does): what the job
    /// does after it needs no place in the commit order. Idempotent; the drop then does nothing.
    pub fn pass(&self) {
        if self.passed.replace(true) {
            return;
        }
        let mut state = self.worker.lock();
        while state.turn != self.seq {
            state = self.worker.turns.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        state.turn = state.turn.wrapping_add(1);
        drop(state);
        self.worker.turns.notify_all();
    }
}

impl Drop for Turn<'_> {
    /// The turn passes on, once it came: a job that ended before its commit still waits for the
    /// earlier ones, so the order holds for the later ones.
    fn drop(&mut self) {
        if self.prep_used.get() && !self.prep_passed.get() {
            // A job that ended inside its preparation still waits for the earlier ones.
            let mut state = self.worker.lock();
            while state.prepare_turn != self.seq {
                state = self.worker.turns.wait(state).unwrap_or_else(PoisonError::into_inner);
            }
            state.prepare_turn = state.prepare_turn.wrapping_add(1);
            drop(state);
            self.worker.turns.notify_all();
        }
        if self.passed.get() {
            return;
        }
        let mut state = self.worker.lock();
        while state.turn != self.seq {
            state = self.worker.turns.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        state.turn = state.turn.wrapping_add(1);
        drop(state);
        self.worker.turns.notify_all();
    }
}

impl DecodeWorker {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Queues `job` behind the earlier ones, starting a thread while fewer run than the jobs
    /// waiting, up to the worker count.
    pub fn submit(self: &Arc<Self>, job: DecodeJob) {
        let mut state = self.lock();
        let seq = state.next_seq;
        state.next_seq = seq.wrapping_add(1);
        state.queue.push_back((seq, job));
        let idle_threads = state.threads.saturating_sub(state.running);
        if idle_threads < state.queue.len() && state.threads < worker_count() {
            state.threads += 1;
            let me = Arc::clone(self);
            std::thread::Builder::new()
                .name("reims-vgpu.decode".into())
                .stack_size(8 << 20)
                .spawn(move || me.run())
                .expect("a decode worker's thread");
        }
        drop(state);
        self.work.notify_one();
    }

    /// Waits until fewer than `limit` jobs are queued or running. The FIFO thread calls it before
    /// an exec takes the task's lock and makes its command buffer: a queue holds a bounded number
    /// of command buffers made and not yet committed, and a FIFO thread that made one past the
    /// bound would wait, holding the lock, for the workers (measured: a hang at 64 behind).
    pub fn wait_below(&self, limit: usize) {
        let mut state = self.lock();
        while state.queue.len() + state.running >= limit {
            state = self.idle.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Waits until every submitted job has finished.
    pub fn drain(&self) {
        let mut state = self.lock();
        while state.running != 0 || !state.queue.is_empty() {
            state = self.idle.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// A thread's loop.
    fn run(&self) {
        raise_priority();
        loop {
            let (seq, job) = {
                let mut state = self.lock();
                loop {
                    if let Some(next) = state.queue.pop_front() {
                        state.running += 1;
                        break next;
                    }
                    state = self.work.wait(state).unwrap_or_else(PoisonError::into_inner);
                }
            };
            let started = std::time::Instant::now();
            {
                let turn = Turn {
                    worker: self,
                    seq,
                    passed: std::cell::Cell::new(false),
                    prep_used: std::cell::Cell::new(false),
                    prep_passed: std::cell::Cell::new(false),
                };
                job(&turn);
            }
            crate::perfstats::DECODE_WORKER.add(started.elapsed());
            let mut state = self.lock();
            state.running -= 1;
            drop(state);
            self.idle.notify_all();
        }
    }
}

/// Raises the calling thread's priority one step above normal and limits it to the CPU's
/// performance cores, where the CPU has cores of several efficiency classes (a hybrid CPU's
/// E-cores ran the decode about 1.5 times slower). For the decode workers and the FIFO threads.
/// `REIMS_VGPU_DECODE_PRIORITY=0` leaves the thread as it is; `REIMS_VGPU_PCORES=0` keeps the
/// priority and leaves the cores to the scheduler.
#[cfg(windows)]
#[allow(unsafe_code)] // Win32 calls on the current thread's pseudo-handle and a local buffer
pub fn raise_priority() {
    crate::sampler::register_current_thread();
    if std::env::var("REIMS_VGPU_DECODE_PRIORITY").is_ok_and(|v| v == "0") {
        return;
    }
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(thread: isize, priority: i32) -> i32;
        fn GetSystemCpuSetInformation(
            information: *mut u8,
            buffer_length: u32,
            returned_length: *mut u32,
            process: isize,
            flags: u32,
        ) -> i32;
        fn SetThreadSelectedCpuSets(thread: isize, ids: *const u32, count: u32) -> i32;
    }
    /// `THREAD_PRIORITY_ABOVE_NORMAL`.
    const ABOVE_NORMAL: i32 = 1;
    // SAFETY: the pseudo-handle names the calling thread.
    unsafe {
        SetThreadPriority(GetCurrentThread(), ABOVE_NORMAL);
    }
    if std::env::var("REIMS_VGPU_PCORES").is_ok_and(|v| v == "0") {
        return;
    }
    static PCORES: OnceLock<Vec<u32>> = OnceLock::new();
    let ids = PCORES.get_or_init(|| {
        let mut length = 0u32;
        // SAFETY: a null buffer of length 0 asks for the length only.
        unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut length, 0, 0) };
        if length == 0 {
            return Vec::new();
        }
        let mut buffer = vec![0u8; length as usize];
        // SAFETY: the buffer is `length` bytes, as asked.
        let ok = unsafe {
            GetSystemCpuSetInformation(buffer.as_mut_ptr(), length, &mut length, 0, 0)
        };
        if ok == 0 {
            return Vec::new();
        }
        // SYSTEM_CPU_SET_INFORMATION: Size u32 at 0, Type u32 at 4 (0: CpuSetInformation), Id
        // u32 at 8, EfficiencyClass u8 at 18.
        let mut sets: Vec<(u32, u8)> = Vec::new();
        let mut at = 0usize;
        while at + 20 <= length as usize {
            let size = u32::from_le_bytes([buffer[at], buffer[at + 1], buffer[at + 2], buffer[at + 3]]) as usize;
            if size == 0 {
                break;
            }
            let kind = u32::from_le_bytes([buffer[at + 4], buffer[at + 5], buffer[at + 6], buffer[at + 7]]);
            if kind == 0 {
                let id = u32::from_le_bytes([buffer[at + 8], buffer[at + 9], buffer[at + 10], buffer[at + 11]]);
                sets.push((id, buffer[at + 18]));
            }
            at += size;
        }
        let best = sets.iter().map(|&(_, class)| class).max().unwrap_or(0);
        let worst = sets.iter().map(|&(_, class)| class).min().unwrap_or(0);
        if best == worst {
            // One class: nothing to choose
            return Vec::new();
        }
        let ids: Vec<u32> = sets.iter().filter(|&&(_, c)| c == best).map(|&(id, _)| id).collect();
        log::info!("decode and FIFO threads limited to the {} performance-core CPU sets", ids.len());
        ids
    });
    if !ids.is_empty() {
        // SAFETY: `ids` holds `ids.len()` CPU set ids the system listed.
        unsafe {
            SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), ids.len() as u32);
        }
    }
}

#[cfg(not(windows))]
pub fn raise_priority() {
    crate::sampler::register_current_thread();
}

thread_local! {
    /// The workers of the FIFO whose exec command this thread is running, while it runs: a
    /// borrowed pointer (no destructor, so no destructor registration at each access on the
    /// Windows GNU target), valid inside [`with_current`].
    static CURRENT: std::cell::Cell<*const DecodeWorker> = const { std::cell::Cell::new(std::ptr::null()) };
}

/// Runs `f` with `worker` as the thread's current worker.
pub fn with_current<T>(worker: &Arc<DecodeWorker>, f: impl FnOnce() -> T) -> T {
    struct Reset(*const DecodeWorker);
    impl Drop for Reset {
        fn drop(&mut self) {
            CURRENT.with(|c| c.set(self.0));
        }
    }
    let previous = CURRENT.with(|c| c.replace(Arc::as_ptr(worker)));
    let _reset = Reset(previous);
    f()
}

/// The thread's current worker: some only inside an exec command's dispatch, with async decode
/// on.
pub fn current() -> Option<Arc<DecodeWorker>> {
    if !enabled() {
        return None;
    }
    let ptr = CURRENT.with(std::cell::Cell::get);
    if ptr.is_null() {
        return None;
    }
    // SAFETY: set by `with_current` from a live `Arc`, which its caller holds while the pointer
    // is current; the count is raised before a new `Arc` is made of it.
    #[allow(unsafe_code)]
    unsafe {
        Arc::increment_strong_count(ptr);
        Some(Arc::from_raw(ptr))
    }
}
