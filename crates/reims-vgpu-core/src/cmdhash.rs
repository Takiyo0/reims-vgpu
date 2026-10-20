//! Measurement only (`REIMS_VGPU_CMDHASH=1`): how much an exec's command stream repeats from one
//! exec to the next. Each exec gets a full hash (every byte of its segments) and a shape hash (the
//! segment kinds and flags, each record's command id, and the pipeline state reference of a render
//! pipeline set); immediate values, offsets, sizes and inline bytes stay out of the shape. Once a
//! second one `warn` line gives the repeat rates against the previous and the current window.
//! Off, nothing runs beyond one cached flag check.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use crate::deserializer::records::{CmdHeader, SegmentHeader};
use crate::memory::cursor::Cursor;

/// Whether the measurement runs (`REIMS_VGPU_CMDHASH=1`).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_CMDHASH").is_ok_and(|v| v == "1"))
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// One exec's hashes.
#[derive(Debug, Clone, Copy)]
pub struct Hasher {
    /// FNV-1a over every byte of the segments.
    pub full: u64,
    /// FNV-1a over segment kinds, command ids and render pipeline references.
    pub shape: u64,
    /// The records walked.
    pub commands: u32,
}

impl Hasher {
    /// Walks a copy of `segments` as the decoder will, without touching them.
    pub fn of(segments: &[Option<Cursor>]) -> Hasher {
        let mut h = Hasher {
            full: FNV_OFFSET,
            shape: FNV_OFFSET,
            commands: 0,
        };
        for segment in segments.iter().flatten() {
            h.segment(segment.clone());
        }
        h
    }

    fn segment(&mut self, mut segment: Cursor) {
        let Ok(mut header) = segment.read::<SegmentHeader>() else {
            return;
        };
        if segment.truncate_to(u64::from(header.length)).is_err() {
            return;
        }
        self.full = fnv(self.full, zerocopy::IntoBytes::as_bytes(&header));
        if header.encoder_type == 5 {
            // The envelope: protection options, then the real header
            let Ok(options) = segment.read::<u64>() else {
                return;
            };
            self.full = fnv(self.full, &options.to_le_bytes());
            let Ok(inner) = segment.read::<SegmentHeader>() else {
                return;
            };
            self.full = fnv(self.full, zerocopy::IntoBytes::as_bytes(&inner));
            header = inner;
        }
        self.shape = fnv(
            self.shape,
            &[
                header.encoder_type,
                header.continues_previous,
                header.continued_by_next,
            ],
        );
        while segment.remaining() >= 8 {
            let Ok(command) = segment.read::<CmdHeader>() else {
                return;
            };
            self.full = fnv(self.full, zerocopy::IntoBytes::as_bytes(&command));
            let Some(length) = u64::from(command.length).checked_sub(8) else {
                return;
            };
            let Ok(mut record) = segment.sub_cursor(length) else {
                return;
            };
            self.commands = self.commands.wrapping_add(1);
            self.shape = fnv(self.shape, &command.command_id.to_le_bytes());
            // Render encoder, set render pipeline state: its reference is part of the shape
            if header.encoder_type == 0
                && command.command_id == 0x74
                && let Ok(reference) = record.read_at::<u32>(0)
            {
                self.shape = fnv(self.shape, &reference.to_le_bytes());
            }
            let mut chunk = [0u8; 256];
            while record.remaining() > 0 {
                let n = record.remaining().min(chunk.len() as u64) as usize;
                if record.read_bytes(&mut chunk[..n]).is_err() {
                    return;
                }
                self.full = fnv(self.full, &chunk[..n]);
            }
        }
    }
}

#[derive(Default)]
struct Stats {
    window_start: Option<Instant>,
    prev_full: HashSet<u64>,
    cur_full: HashSet<u64>,
    prev_shape: HashSet<u64>,
    cur_shape: HashSet<u64>,
    shape_counts: HashMap<u64, u32>,
    execs: u64,
    full_repeat: u64,
    shape_repeat: u64,
    hit_time: Duration,
    hits: u64,
    miss_time: Duration,
    misses: u64,
}

fn stats() -> &'static Mutex<Stats> {
    static S: OnceLock<Mutex<Stats>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Stats::default()))
}

/// One finished exec's hashes and decode time; logs and rotates once a second.
pub fn record(hasher: Hasher, decode_time: Duration) {
    let mut s = stats().lock().unwrap_or_else(PoisonError::into_inner);
    let now = Instant::now();
    let start = *s.window_start.get_or_insert(now);
    s.execs += 1;
    if s.prev_full.contains(&hasher.full) || s.cur_full.contains(&hasher.full) {
        s.full_repeat += 1;
    }
    let shape_hit = s.prev_shape.contains(&hasher.shape) || s.cur_shape.contains(&hasher.shape);
    if shape_hit {
        s.shape_repeat += 1;
        s.hit_time += decode_time;
        s.hits += 1;
    } else {
        s.miss_time += decode_time;
        s.misses += 1;
    }
    s.cur_full.insert(hasher.full);
    s.cur_shape.insert(hasher.shape);
    *s.shape_counts.entry(hasher.shape).or_insert(0) += 1;
    if now.duration_since(start) < Duration::from_secs(1) {
        return;
    }
    let pct = |n: u64| n * 100 / s.execs.max(1);
    let avg = |t: Duration, n: u64| t.as_micros() / u128::from(n.max(1));
    let mut top: Vec<(u64, u32)> = s.shape_counts.iter().map(|(&h, &c)| (h, c)).collect();
    top.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top_text = top
        .iter()
        .take(5)
        .map(|(h, c)| format!("{:08x}:{c}", h >> 32))
        .collect::<Vec<_>>()
        .join(" ");
    log::warn!(
        "reims-vgpu cmdhash: execs {}, full-repeat {}%, shape-repeat {}%, distinct-shapes {}, \
         hit-decode {} us avg, miss-decode {} us avg, top: {}",
        s.execs,
        pct(s.full_repeat),
        pct(s.shape_repeat),
        s.shape_counts.len(),
        avg(s.hit_time, s.hits),
        avg(s.miss_time, s.misses),
        top_text
    );
    s.prev_full = std::mem::take(&mut s.cur_full);
    s.prev_shape = std::mem::take(&mut s.cur_shape);
    s.shape_counts.clear();
    s.execs = 0;
    s.full_repeat = 0;
    s.shape_repeat = 0;
    s.hit_time = Duration::ZERO;
    s.hits = 0;
    s.miss_time = Duration::ZERO;
    s.misses = 0;
    s.window_start = Some(now);
}
