//! Cheap work counters for performance measurement: how many times, and how long, the FIFO thread
//! spent in the costly steps (execs, their resource prepares, page binds, texture prepares, the
//! copy fallback), summed with relaxed atomic adds and nothing else on the measured path. The
//! stats reporter takes them once a second ([`take_line`]) beside its frame line, so a normal run
//! carries its own measurement: per-event debug lines cost the very thread they measure.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

/// A step's count and total time since the last [`take_line`].
pub struct Counter {
    count: AtomicU64,
    nanos: AtomicU64,
}

impl Counter {
    const fn new() -> Counter {
        Counter {
            count: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        }
    }

    /// One more, of `took`.
    pub fn add(&self, took: Duration) {
        self.count.fetch_add(1, Relaxed);
        self.nanos
            .fetch_add(u64::try_from(took.as_nanos()).unwrap_or(u64::MAX), Relaxed);
    }

    /// The count and milliseconds since the last take, both reset.
    fn take(&self) -> (u64, f64) {
        let count = self.count.swap(0, Relaxed);
        let nanos = self.nanos.swap(0, Relaxed);
        #[allow(clippy::cast_precision_loss)]
        (count, nanos as f64 / 1e6)
    }
}

/// Execs (`ExecIndirect2`), whole.
pub static EXEC: Counter = Counter::new();
/// The decode worker's jobs (`fifo::decode_worker`): the second halves of the execs it ran.
pub static DECODE_WORKER: Counter = Counter::new();
/// A guest present's wait for the display lock and the frames in flight (`display::frame`).
pub static PRESENT_WAIT: Counter = Counter::new();
/// Transfer buffers handed out again by their physical pages (`TransferCache`).
pub static TRANSFER_HIT: Counter = Counter::new();
/// Copy-fallback scratch ranges made fresh (a pool miss), with their making's time.
pub static SCRATCH_FRESH: Counter = Counter::new();
/// Copy-fallback scratch ranges taken from the pool.
pub static SCRATCH_REUSED: Counter = Counter::new();
/// Their resource prepares.
pub static PREPARE: Counter = Counter::new();
/// Scattered guest ranges gathered on the GPU (`REIMS_VGPU_GPU_GATHER`): the encoding's time.
pub static GATHER_GPU: Counter = Counter::new();
/// The KiB they gathered.
pub static GATHER_GPU_KB: AtomicU64 = AtomicU64::new(0);
/// Scattered guest ranges a gather was asked for and left to the CPU copy (count, KiB).
pub static GATHER_CPU: Counter = Counter::new();
/// The KiB left to the CPU copy.
pub static GATHER_CPU_KB: AtomicU64 = AtomicU64::new(0);
/// Preparations run on the decode workers (`REIMS_VGPU_ASYNC_PREPARE`): the call's time.
pub static PREP_ASYNC: Counter = Counter::new();
/// A worker's wait for its preparation turn.
pub static PREP_TURN: Counter = Counter::new();
/// Resource prepares: the loop that looks up each resource the exec names.
pub static PREPARE_LOOKUP: Counter = Counter::new();
/// Resource prepares: GPU paging (the paging buffer, its texture and buffer prepares, commit).
pub static PREPARE_PAGING: Counter = Counter::new();
/// Resource prepares: CPU paging and guest invalidation.
pub static PREPARE_TAIL: Counter = Counter::new();
/// Resource prepares that paged on the GPU: each one's paging section (the paging buffer, its
/// encoders and prepares, its commit).
pub static PAGING_ROUNDS: Counter = Counter::new();
/// Resources the GPU paging rounds prepared (blit and compute).
pub static PAGING_PREPARES: AtomicU64 = AtomicU64::new(0);
/// Resources the execs named.
pub static PREPARE_RESOURCES: AtomicU64 = AtomicU64::new(0);
/// The resource kinds, by [`crate::resources::resource::Resource::kind_slot`].
pub const PAGING_KINDS: [&str; 14] = [
    "sbuf", "pbuf", "dbuf", "hbuf", "mbuf", "lheap", "dheap", "icb", "tex", "stex", "rtex", "mtex",
    "shtex", "back",
];
/// GPU paging rounds: each blit prepare, by kind.
pub static PAGING_BLIT: [Counter; 14] = [const { Counter::new() }; 14];
/// Slow resource lookups (20 us or more: mostly a first use's creation), by resource kind.
pub static LOOKUP_SLOW_KIND: [Counter; 14] = [const { Counter::new() }; 14];
/// GPU paging rounds: each compute prepare, by kind.
pub static PAGING_COMPUTE: [Counter; 14] = [const { Counter::new() }; 14];
/// GPU paging round steps, in order.
pub const PAGING_STEPS: [&str; 7] = [
    "cmdbuf", "wait", "blit-enc", "blit-end", "comp-enc", "fills", "commit",
];
pub static PAGING_STEP: [Counter; 7] = [const { Counter::new() }; 7];

/// Page binds: the mapping of the guest's pages.
pub static BIND_MAP: Counter = Counter::new();
/// Page binds: the buffer made over the mapping.
pub static BIND_BUFFER: Counter = Counter::new();
/// Texture prepares: binding their pages (`ensure`).
pub static TEXTURE_ENSURE: Counter = Counter::new();
/// Texture prepares: filling the transfer buffer from the guest.
pub static TEXTURE_SYNC: Counter = Counter::new();
/// Copy-fallback mappings: allocation and copy.
pub static COPY_FALLBACK: Counter = Counter::new();
/// Bytes the copy-fallback mappings covered.
pub static COPY_FALLBACK_BYTES: AtomicU64 = AtomicU64::new(0);

/// The exec steps counted ([`exec_step`]), in order.
const EXEC_STEPS: [&str; 9] = [
    "lock",
    "ensure_mapped",
    "prepare_resources",
    "cursors",
    "decode",
    "commit",
    "complete_resources",
    "turn",
    "handoff",
];
static EXEC_STEP: [Counter; 9] = [const { Counter::new() }; 9];

/// One exec step of `took`, named as `exec_indirect` marks it; an unknown name is not counted.
pub fn exec_step(label: &str, took: Duration) {
    if let Some(i) = EXEC_STEPS.iter().position(|s| *s == label) {
        EXEC_STEP[i].add(took);
    }
}

/// Every FIFO command's handling time, by command id (ids past the table share its last slot).
static COMMAND: [Counter; 128] = [const { Counter::new() }; 128];

/// One FIFO command `id`, handled in `took` (its releases included).
pub fn command(id: u16, took: Duration) {
    COMMAND[usize::from(id).min(COMMAND.len() - 1)].add(took);
}

/// Decode: each segment's decode, by encoder type (render, compute, blit, -, info).
pub static DECODE_SEGMENT: [Counter; 5] = [const { Counter::new() }; 5];
const DECODE_TYPES: [&str; 5] = ["render", "compute", "blit", "t3", "info"];
/// Decode: a fresh segment's encoder (render pass prefix and encoder, compute pass, blit).
pub static DECODE_ENCODER: Counter = Counter::new();
/// Decode: each record's dispatch, by command id (ids past the table share its last slot).
static RECORD: [Counter; 512] = [const { Counter::new() }; 512];

/// One decoded record `id`, of `took`.
pub fn record(id: u32, took: Duration) {
    RECORDS.fetch_add(1, Relaxed);
    RECORD[usize::try_from(id)
        .unwrap_or(usize::MAX)
        .min(RECORD.len() - 1)]
    .add(took);
}

/// Decode: the decoders' object lookups (`require_*`).
pub static DECODE_LOOKUP: Counter = Counter::new();
/// Decode: segments and records walked.
pub static SEGMENTS: AtomicU64 = AtomicU64::new(0);
pub static RECORDS: AtomicU64 = AtomicU64::new(0);
/// Delete resource (cmd 37): the task's write lock, the object's delete.
pub static DELETE_LOCK: Counter = Counter::new();
pub static DELETE_OBJECT: Counter = Counter::new();
/// Unmap memory (cmd 34): the task's write lock, the object list's unmap, the tracker, the host.
pub static UNMAP_LOCK: Counter = Counter::new();
pub static UNMAP_LIST: Counter = Counter::new();
pub static UNMAP_TRACKER: Counter = Counter::new();
pub static UNMAP_HOST: Counter = Counter::new();

/// Execs by task id (folded into the table).
static TASK_EXEC: [Counter; 64] = [const { Counter::new() }; 64];

/// One exec of task `id`, of `took`.
pub fn task_exec(id: u32, took: Duration) {
    TASK_EXEC[(id as usize) % TASK_EXEC.len()].add(took);
}

/// Resource prepares: the wait for the task's prepare mutex.
pub static PREPARE_MUTEX: Counter = Counter::new();
/// Resource prepares: lookups of 20 us or more (a resource made on first use, mostly).
pub static LOOKUP_SLOW: Counter = Counter::new();
/// A view's plane prepare (RefTexture): the fill from the guest, and the copy's encode.
pub static PLANE_FILL: Counter = Counter::new();
pub static PLANE_COPY: Counter = Counter::new();

/// More parts for the line, from the platform (its own counters), set once.
pub static EXTRA: std::sync::OnceLock<fn() -> String> = std::sync::OnceLock::new();

/// FIFO channels' handling time, by channel index (indexes past the table share its last slot).
static CHANNEL: [Counter; 16] = [const { Counter::new() }; 16];

/// One FIFO command on `channel`, handled in `took`.
pub fn channel(channel: usize, took: Duration) {
    CHANNEL[channel.min(CHANNEL.len() - 1)].add(took);
}

/// The `n` costliest of `table`, as ` {prefix}{id} n/ms`.
fn top(line: &mut String, prefix: &str, table: &[Counter], n: usize) {
    let mut all: Vec<(usize, u64, f64)> = table
        .iter()
        .enumerate()
        .map(|(id, c)| {
            let (n, ms) = c.take();
            (id, n, ms)
        })
        .filter(|(_, n, _)| *n != 0)
        .collect();
    all.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (id, k, ms) in all.iter().take(n) {
        line.push_str(&format!(" {prefix}{id} {k}/{ms:.1}ms"));
    }
}

/// The line for the counters since the last call, reset: `None` where nothing was counted.
pub fn take_line() -> Option<String> {
    let parts = [
        ("exec", EXEC.take()),
        ("decode-worker", DECODE_WORKER.take()),
        ("present-wait", PRESENT_WAIT.take()),
        ("transfer-hit", TRANSFER_HIT.take()),
        ("scratch-fresh", SCRATCH_FRESH.take()),
        ("scratch-reused", SCRATCH_REUSED.take()),
        ("prepare", PREPARE.take()),
        ("gather-gpu", GATHER_GPU.take()),
        ("gather-cpu", GATHER_CPU.take()),
        ("prep-async", PREP_ASYNC.take()),
        ("prep-turn", PREP_TURN.take()),
        ("prep-lookup", PREPARE_LOOKUP.take()),
        ("prep-paging", PREPARE_PAGING.take()),
        ("prep-tail", PREPARE_TAIL.take()),
        ("paging", PAGING_ROUNDS.take()),
        ("bind-map", BIND_MAP.take()),
        ("bind-buffer", BIND_BUFFER.take()),
        ("tex-ensure", TEXTURE_ENSURE.take()),
        ("tex-sync", TEXTURE_SYNC.take()),
        ("copy", COPY_FALLBACK.take()),
    ];
    let bytes = COPY_FALLBACK_BYTES.swap(0, Relaxed);
    let gather_kb = (
        GATHER_GPU_KB.swap(0, Relaxed),
        GATHER_CPU_KB.swap(0, Relaxed),
    );
    let resources = PREPARE_RESOURCES.swap(0, Relaxed);
    let paged = PAGING_PREPARES.swap(0, Relaxed);
    if parts.iter().all(|(_, (n, _))| *n == 0) && COMMAND.iter().all(|c| c.count.load(Relaxed) == 0)
    {
        return None;
    }
    let mut line = String::from("reims-vgpu work:");
    for (name, (n, ms)) in parts {
        line.push_str(&format!(" {name} {n}/{ms:.1}ms"));
    }
    for (name, counter) in EXEC_STEPS.iter().zip(&EXEC_STEP) {
        let (n, ms) = counter.take();
        if n != 0 {
            line.push_str(&format!(" step-{name} {n}/{ms:.1}ms"));
        }
    }
    for (name, counter) in PAGING_STEPS.iter().zip(&PAGING_STEP) {
        let (n, ms) = counter.take();
        if n != 0 {
            line.push_str(&format!(" pg-{name} {n}/{ms:.1}ms"));
        }
    }
    for (prefix, table) in [
        ("pb", &PAGING_BLIT),
        ("pc", &PAGING_COMPUTE),
        ("ls", &LOOKUP_SLOW_KIND),
    ] {
        for (name, counter) in PAGING_KINDS.iter().zip(table.iter()) {
            let (n, ms) = counter.take();
            if n != 0 {
                line.push_str(&format!(" {prefix}-{name} {n}/{ms:.1}ms"));
            }
        }
    }
    for (name, counter) in DECODE_TYPES.iter().zip(&DECODE_SEGMENT) {
        let (n, ms) = counter.take();
        if n != 0 {
            line.push_str(&format!(" dec-{name} {n}/{ms:.1}ms"));
        }
    }
    let (n, ms) = DECODE_ENCODER.take();
    if n != 0 {
        line.push_str(&format!(" dec-encoder {n}/{ms:.1}ms"));
    }
    for (name, counter) in [
        ("dec-lookup", &DECODE_LOOKUP),
        ("prep-mutex", &PREPARE_MUTEX),
        ("lookup-slow", &LOOKUP_SLOW),
        ("plane-fill", &PLANE_FILL),
        ("plane-copy", &PLANE_COPY),
        ("del-lock", &DELETE_LOCK),
        ("del-obj", &DELETE_OBJECT),
        ("unmap-lock", &UNMAP_LOCK),
        ("unmap-list", &UNMAP_LIST),
        ("unmap-tracker", &UNMAP_TRACKER),
        ("unmap-host", &UNMAP_HOST),
    ] {
        let (n, ms) = counter.take();
        if n != 0 {
            line.push_str(&format!(" {name} {n}/{ms:.1}ms"));
        }
    }
    line.push_str(&format!(
        " segments {} records {}",
        SEGMENTS.swap(0, Relaxed),
        RECORDS.swap(0, Relaxed)
    ));
    top(&mut line, "rec", &RECORD, 8);
    top(&mut line, "task", &TASK_EXEC, 4);
    top(&mut line, "ch", &CHANNEL, 4);
    if let Some(extra) = EXTRA.get() {
        line.push_str(&extra());
    }
    let mut commands: Vec<(usize, u64, f64)> = COMMAND
        .iter()
        .enumerate()
        .map(|(id, c)| {
            let (n, ms) = c.take();
            (id, n, ms)
        })
        .filter(|(_, n, _)| *n != 0)
        .collect();
    commands.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (id, n, ms) in commands.iter().take(6) {
        line.push_str(&format!(" cmd{id} {n}/{ms:.1}ms"));
    }
    #[allow(clippy::cast_precision_loss)]
    line.push_str(&format!(
        " copy-mb {:.1} gather-gpu-kb {} gather-cpu-kb {} resources {resources} paged {paged}\n",
        bytes as f64 / f64::from(1u32 << 20),
        gather_kb.0,
        gather_kb.1
    ));
    Some(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A take reports the counts and times added since the last, then starts again from 0
    #[test]
    fn take_reports_and_resets() {
        // The statics are shared: drain whatever other tests added first.
        let _ = take_line();
        EXEC.add(Duration::from_millis(3));
        EXEC.add(Duration::from_millis(2));
        COPY_FALLBACK_BYTES.fetch_add(3 << 20, Relaxed);
        let line = take_line().unwrap_or_default();
        assert!(line.contains(" exec 2/5.0ms"), "{line}");
        assert!(line.contains(" copy-mb 3.0"), "{line}");
        assert!(take_line().is_none(), "reset after the take");
    }
}
