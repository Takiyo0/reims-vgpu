//! All of the library's behaviour, generic over one `P: Platform` (D3). Links no Apple framework.

// The crate root denies `unsafe_code` (D43): `forbid` cannot be lowered by the per-module
// `#![allow(unsafe_code)]` that 17.3 grants `guest/ptr.rs` and `testkit/guest_ram.rs` (E0453).
#![deny(unsafe_code)]
// The crate's lint set. Tests are exempt: clippy.toml lets them unwrap, index and panic,
// and the conversion and division lints apply outside cfg(test) only, as clippy has no in-tests
// switch for them. `arithmetic_side_effects` is denied only in the modules guest data flows
// through (their own attributes).
#![deny(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    clippy::mod_module_files
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::integer_division_remainder_used,
        clippy::undocumented_unsafe_blocks
    )
)]
// Architecture 3 names `device/device.rs`, `fifo/fifo.rs`, `display/display.rs` and
// `deserializer/deserializer.rs`.
#![allow(clippy::module_inception)]

pub mod collections;
pub mod deserializer;
pub mod device;
pub mod display;
pub mod fifo;
pub mod gpu_task;
pub mod guest;
pub mod host;
pub mod iosurface;
pub mod logging;
pub mod memory;
pub mod num;
pub mod perfstats;
pub mod cmdhash;
pub mod sampler;
pub mod platform;
pub mod process;
pub mod raise;
pub mod resources;
pub mod rt;
pub mod seams;
pub mod sync;
pub mod tasks;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;
