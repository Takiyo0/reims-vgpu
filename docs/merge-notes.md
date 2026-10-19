# Merging `steelbrain/async-decode` into `master`

Notes for whoever merges the Windows performance branch later. The recommendation is not a
textual merge: replay the branch's commits on `master`, one at a time, skipping the Windows port
(`master` has the same port), and settle the QEMU and metal2vulkan pairings first. The figures
below were computed on 2026-10-19; recompute them before acting.

## The two sides

| | `master` | `steelbrain/async-decode` |
|---|---|---|
| merge base | `9b2c5452` (2026-10-12) | the same |
| commits since the base | 48 upstream commits, plus the owner's Windows commits `b1f87889` (`feat(windows)...`, 2026-10-16) and `2cca7e22` (`fix display composition and guest backing sync`, 2026-10-17), then the two QEMU submodule moves `f48e582e` and `d5fc9712` | 8 commits (below) |
| `vendor/qemu` | the merged `qemu-sanctum-vgpu` master (`cd5269f3ee`) | the pre-merge tree |
| metal2vulkan pin | `720ef1ce9ae2a2068999a8675717418b9416283e` | `13b279df1bf87893d389f955a90c55b6ff3d0f9b`, redirected by a `[patch]` block in `Cargo.toml` to `..\metal2vulkan-local` |
| line endings of the files both Windows ports touched | CRLF | CRLF (the base has LF) |

The branch's commits, oldest first:

| commit | what it does |
|---|---|
| `e9d53134` feat(windows): Working macOS boot with Reims vGPU, Sonoma backdrop and Vulkan display | the Windows port: `os/windows/*`, the Win32 window crate, `reims-vgpu` window and swapchain code, the feasibility and windowing docs. **The same patch as `b1f87889` on `master`**: 104 and 103 files, 10014 and 10013 insertions ignoring CR, and the same file lists except the `vendor/qemu` gitlink |
| `0871cb5a` fix(texture): preserve GPU-owned backing contents when preparing views | `resources/texture/views.rs` only |
| `64154c40` checkpoint: preserve preexisting display work | a checkpoint of uncommitted earlier work: 83 files, +4041 / -484 ignoring CR. Per-second counters (`perfstats.rs`), logging (`REIMS_VGPU_LOG_TIME`, async stderr), copy-fallback and scratch-pool work in `memory/map.rs`, display probe and wallpaper changes, the guest display-port override, dual-display changes in `reims-vgpu` and the window crate, the swapchain rework |
| `14770a99` measure Vulkan window present stages | `stats.rs`, `window/vulkan/swapchain.rs`: acquire, encode and present-call timing on the `stats:` line |
| `9aedb671` avoid draw buffer tracking allocations | `gpu/encode_render.rs` |
| `7dfae199` Make per-record timing opt-in | the `record-timing` feature (core and `reims-vgpu-c`) around the per-record counters |
| `ea5d2399` Decode execs on ordered workers and trim per-exec costs | see below |
| `c0551c1f` Prewarm guest RAM imports and reuse no-copy buffers on Vulkan | see below |

**`ea5d2399`** (27 files, +2247 / -133). The decode workers (`fifo/decode_worker.rs`: workers per
FIFO, ordered `Turn`, `MAX_AHEAD` 16, drain before other commands, early commit, async prepare);
the exec split in `tasks/local/exec.rs`; cursor and transfer-buffer pools and the transfer cache
(`tasks/local/memory.rs`); destructor-free thread-locals (`sync/rwlock.rs`, `rt/block_env.rs`,
`rt/held.rs`, `decode_worker.rs`, `resources/resource.rs`); the frame copy for the window
(`display/render.rs`); interrupt coalescing (`device/fifos.rs`); the no-buffer retry and the
"no buffer" exec drop (`tasks/local/memory.rs`, `tasks/local/exec.rs`); the pipeline-width clamp
(`gpu_task/exec.rs`); the sampler (`sampler.rs`) and `cmdhash.rs`; more counters.

**`c0551c1f`** (13 files, +232 / -25). The guest RAM prewarm (`reims-vgpu-vulkan/src/api.rs`,
`reims-vgpu/src/platform.rs` and `platform/vulkan.rs` call it at device creation); the GPU gather
(`gpu/encode_blit.rs`, `memory/map.rs`'s `gpu_gather_wanted`) default on; mimalloc in
`reims-vgpu-c`; the logged error of a failed buffer creation (`gpu/device.rs`); the pending-answer
flag without a `RefCell` (`answer.rs`); scratch-range pool changes (`remap.rs`). The runtime half
(host chunks, the no-copy buffer pool) is in metal2vulkan.

## The trial merge

```sh
git -c core.commitGraph=false merge-tree --write-tree --name-only --no-messages master steelbrain/async-decode
```

lists **62** paths (`Cargo.lock`, `Cargo.toml` and the `vendor/qemu` gitlink among them; the
owner's earlier count was 60). It exits 1 when there are conflicts and writes a tree anyway.
Almost half are not content conflicts but both sides rewriting whole files to CRLF. With
`-Xignore-cr-at-eol` (or `-Xignore-space-at-eol`) added, **50** remain; `-Xrenormalize` changes
nothing (62), since there is no `.gitattributes`. The 12 that go away are
`gpu_task/memory.rs`, `resources/buffer/shared.rs`, `resources/manager.rs`,
`resources/manager/resolver.rs`, `resources/resource.rs`, `tasks/local/exec.rs`,
`tasks/shared.rs`, `gpu/encode_compute.rs`, `os/linux/surface.rs`, `window/native.rs`,
`reims-vgpu/src/device.rs` and `window/vulkan.rs`.

The 50, by area:

| area | files |
|---|---|
| build | `Cargo.toml`, `Cargo.lock`, `crates/reims-vgpu-c/Cargo.toml`, `vendor/qemu` (gitlink) |
| C API | `crates/reims-vgpu-c/src/device.rs` |
| core display | `display.rs`, `display/frame.rs`, `display/render.rs`, `display/tests.rs`, `display/wallpaper.rs`, `fifo/commands/display.rs` |
| core FIFO | `device/fifos.rs`, `fifo/ring.rs`, `fifo/thread.rs` |
| core memory, resources, tasks | `memory/map.rs`, `memory/mapping.rs`, `gpu_task/paging.rs`, `resources/buffer/mapper_ref.rs`, `resources/buffer/private.rs`, `resources/manager/create.rs`, `resources/manager/texture.rs`, `resources/texture.rs`, `resources/texture/backing.rs`, `resources/texture/mapper.rs`, `resources/texture/views.rs`, `tasks/object_lists.rs` |
| core death handling | `raise/death.rs`, `testkit/death.rs` |
| Vulkan platform | `platform.rs`, `api.rs`, `gpu/desc.rs`, `gpu/device.rs`, `gpu/encode_blit.rs`, `gpu/encode_render.rs`, `gpu/resource.rs`, `os/linux.rs`, `os/windows/{files,import,section,surface,threads,vm}.rs` |
| window crate | `native/windows/{input,render,stage1_test,ui}.rs` |
| `reims-vgpu` | `lib.rs`, `platform/vulkan.rs`, `window/vulkan/swapchain.rs`, `window/vulkan/windows/tests.rs` |

(Paths are under `crates/<crate>/src/`.) Of the 50, 16 are files the two performance commits
change (`Cargo.lock`, `reims-vgpu-c/Cargo.toml`, `device/fifos.rs`, `display.rs`,
`display/frame.rs`, `display/render.rs`, `fifo/thread.rs`, `memory/map.rs`,
`resources/buffer/private.rs`, `resources/texture.rs`, `resources/texture/backing.rs`,
`api.rs`, `gpu/device.rs`, `gpu/encode_blit.rs`, `gpu/resource.rs`, `platform/vulkan.rs`); the
other 34 are the Windows port and the checkpoint colliding with `master`'s copy of the port.
Textual resolution would mean choosing, 50 times, between two copies of the same port with
different line endings and different later edits, which is the work to avoid.

## Recommended approach: replay the commits on `master`

1. **Make the repository readable.** The commit graph file is corrupt in this repository; every
   command above ran with `git -c core.commitGraph=false`. Either keep passing it or rewrite the
   graph: `git commit-graph write --reachable` (or `git commit-graph clear` and let git rebuild).
2. **Work in a worktree of its own**, from `master`, on a new branch
   (`steelbrain/async-decode-on-master`); `AGENTS.md` keeps the main checkout the owner's. Never
   initialise `vendor/qemu` inside a worktree that shares the main checkout's module repository.
3. **Settle the two pairings first** (next two sections), because the performance commits build
   against both.
4. **Replay**, skipping `e9d53134` (`master` has the port as `b1f87889`):
   `git rebase --onto master e9d53134 steelbrain/async-decode -Xignore-cr-at-eol` on the new
   branch, or `git cherry-pick -Xignore-cr-at-eol` of the seven commits in order. Simulated
   with `git merge-tree --merge-base=<commit>^ -Xignore-cr-at-eol <tree> <commit>`, chaining each
   result into the next, the replay stops for these (a conflicted tree was carried forward, so
   later counts are approximate):

   | commit | conflicts |
   |---|---|
   | `0871cb5a` | none |
   | `64154c40` | 7: `display/frame.rs`, `display/render.rs`, `display/tests.rs`, `fifo/thread.rs`, `gpu/device.rs`, `gpu/resource.rs`, `os/windows/surface.rs` |
   | `14770a99`, `9aedb671`, `7dfae199` | none |
   | `ea5d2399` | 1: `display/render.rs` |
   | `c0551c1f` | 3: `reims-vgpu-c/Cargo.toml`, `gpu/device.rs`, `gpu/resource.rs` |

   That is 11 conflicts in 9 files, against 50. Most are in files `master`'s `2cca7e22` (display
   composition and backing sync: `display/render.rs`, `display/tests.rs`,
   `texture/mapper.rs`, `gpu/resource.rs`, `gpu/states.rs`, `os/windows/surface.rs`) also edited;
   whether the branch already carries equivalent edits is unverified, so read each hunk against
   `2cca7e22` and keep `master`'s where they overlap.
5. **Resolve by area, not by file**, in this order, building and running the area's checks between
   them (below): (a) display and window (`64154c40`'s part), (b) memory and resources, (c) the
   FIFO and decode workers, (d) the Vulkan platform and RAM import, (e) the diagnostics.
6. **Keep each file's line endings.** Run `git ls-files --eol` over the changed files after each
   pick: the Windows port files are CRLF on `master`; a file edited by a tool that normalises
   endings shows up as a whole-file diff. Do not renormalise in the same commit as a behaviour
   change (one commit per task).

## Pairing 1: QEMU and the C API

`master`'s `vendor/qemu` (`cd5269f3ee`, via `d5fc9712`) has an `apple-gfx-pci.c` that fills
`RvgDeviceConfig.process_isolation_follows_guest` (line 165) and the IOSurface-over-PCI device
(`bd95509aa3`). On 2026-10-19 neither `reims_vgpu.h` on `master` (`crates/reims-vgpu-c/include`)
nor this branch's has `process_isolation_follows_guest`; both have `bool process_isolation`
(`master`'s line 421). So the C API and the submodule on `master` do not agree here either
(unverified that `master` builds `make qemu`; check before anything else). Decide, with the
owner:

- extend the C API additively (`docs/api.md` section 1: no rename, renumbering or removal) with
  `process_isolation_follows_guest` and whatever `bd95509aa3`'s device needs, and build the new
  QEMU against it; or
- pin `vendor/qemu` back to `a8a3e526c8`, which is what this branch's library was built with
  (plus the hand-made changes described in [windows.md](windows.md), "Building").

A QEMU object built against a header that differs from the linked staticlib's produces a black
screen without a build error (the `flicker1` build). After the decision, rebuild with `ninja`,
not a hand link, and relink under a new name (`relink-named.sh`); the launcher's build manifest
then records both hashes.

## Pairing 2: metal2vulkan

The branch's `Cargo.toml` redirects metal2vulkan to `..\metal2vulkan-local` (the block is marked
temporary). That checkout has 55 commits on the branch's pinned `13b279df`; the first 40 are
upstream's, up to `720ef1ce`, **which is `master`'s pin and an ancestor of the local `HEAD`**
(`8f665f2`), so the local fork is `master`'s pin plus 15 commits:

- host-chunk imports, timing diagnostics and blit staging (`1c3d1a0`);
- debug-build AIR tolerance (`9587d0c`: module flags, `llvm.dbg.*`), the Resolve fix;
- `performance-counters` feature and the finer draw, commit and no-copy counters
  (`4d6f44b`, `34024a1`, `d940e89`, `ced177b`, `b68e318`);
- the no-copy `VkBuffer` pool (`e5b485d`), descriptor reuse (`44e5c3e`), table-stamp skipping
  (`42e7fdb`), settled-variant and retention caches (`7c9b355`), allocation-free bind refresh
  (`a2e20fd`, `c82771c`);
- the census borrowed through a thread-local scope (`0bd39cd`);
- the foreign-acquirer stub off Unix (`8f665f2`).

These are all 15 (`git log 720ef1ce..HEAD`). The runtime half of the guest-RAM import (host
chunks, `REIMS_VGPU_RAM_IMPORT*`, the pool switch) is in this fork. Before the library's merge:
push the fork's commits to the public repository (its remote is `Takiyo0/metal2vulkan`), move
the pin in `Cargo.toml` to the pushed commit in one batch (`AGENTS.md`: the pin moves in
batches, with one merge's checks), and delete the `[patch]` block and its `Cargo.lock` entries.
Until then the library does not build off this machine.

## What to keep

Every switch below restores the older path and the launcher's clear list names them; keep them,
with their counters, until a boot of each guest has been clean on the new tree.

| switch | counters it explains |
|---|---|
| `REIMS_VGPU_ASYNC_DECODE`, `_DECODE_WORKERS`, `_EARLY_COMMIT`, `_ASYNC_PREPARE` | `decode-worker`, `step-turn`, `step-handoff`, `prep-async`, `prep-turn` |
| `REIMS_VGPU_RAM_IMPORT`, `_RAM_IMPORT_CHUNK_MB`, `REIMS_VGPU_BUFFER_POOL`, `_GPU_GATHER*` | `m-nc-*`, `gather-*` |
| `REIMS_VGPU_FRAME_COPY`, `_EVENT_FIX`, `_SHARED_DELETE` | `del-lock`, `unmap-lock` |
| `REIMS_VGPU_NOBUF_RETRY`, `_NOBUF_SOFT`, `_PIPELINE_WIDTH_CLAMP`, `M2V_AIR_DEBUG_TOLERANT` | guest-visible fixes (flicker, Resolve) |
| `REIMS_VGPU_CURSOR_POOL`, `_TRANSFER_CACHE`, `_SCRATCH_POOL`, `_SCRATCH_CLASSES`, the TLS switches, the `M2V_*` fast paths | `transfer-hit`, `scratch-*`, `m-fast-draw`, `m-stage-skip`, `m-dset-hit`, ... |
| `REIMS_VGPU_SAMPLE`, `_CMDHASH`, `_STATS`, `_LOG_TIME`, `_LOG_ASYNC`, `M2V_TIMING`, the `record-timing` feature | the profiling tools in [windows.md](windows.md) |

All are listed with defaults in the configuration reference of [windows.md](windows.md). Three
things are guest-visible and need a decision rather than a switch: the drop of an exec that
names an empty object-list entry (`REIMS_VGPU_EMPTY_ENTRY_FAULTS`; it first appears in the
`64154c40` checkpoint and `master` has no trace of it), the drop of an exec that lacks a buffer
(`REIMS_VGPU_NOBUF_SOFT`), and the pipeline-width clamp. The `FORCE_SOFTWARE_RENDERER` guest
setting is not code, but the guest needs it.

## Checks

From `AGENTS.md`, "Checking a change", for what this merge touches:

| the merge touches | check |
|---|---|
| the library (core, the Vulkan platform, the window crate, the C API) | `cargo xtask check` once, on the merged tree, and `make test` where the change reaches core, the platform or the C API. On Windows the tests are run filtered and single-threaded; the known failures (10 `new_range_mapping`, 2 `device::fifos`, 2 `gpu_task::exec` log tests, 2 that map 32 GiB) must be the only ones, and metal2vulkan's own tests do not compile on Windows (Unix fork tests), so `cargo check -p metal2vulkan-runtime --lib` stands in |
| what the Vulkan platform ends the process on, or what device info and feature queries answer (the pipeline-width clamp, AIR tolerance) | the library's checks, and a cold boot of each guest the change reaches, failing on any `reims-vgpu: fatal` line |
| `vendor/qemu` | build it and realize the devices paused (`-S`); `apple-gfx*.c` changes need a VM boot |
| the C API or header | `cargo xtask header --check`, and re-recorded pins if a value changed (`RVG_PINS_RECORD=1 cargo test -p reims-vgpu-c --test pins`) |
| a behaviour change (the drops, the interrupt coalescing) | its own commit with the expected files recorded again and the reason in the message |
| docs only | nothing |

A boot of this tree means the launcher in `qemu-vm\`: boot, log in, open Safari and System
Settings, play a video, drag windows for 30 s with `-Profile -Sample`, and compare the `work:` and
`stats:` lines with the table in [windows.md](windows.md). The branch's own acceptance numbers
were never met (60 fps was the goal), so a regression means "worse than the figures there", not
"below 60".

## Do not

- merge with `git merge` and resolve the 50 files by hand;
- take either side's copy of `os/windows/*`, the window crate or the swapchain wholesale before
  comparing it with the other side and with `2cca7e22`;
- renormalise line endings in the same commit as code changes;
- push, commit or rewrite `master` without the owner; the owner signs commits.
