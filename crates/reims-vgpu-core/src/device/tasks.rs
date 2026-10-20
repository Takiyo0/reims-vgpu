//! The task table: task lookup, creation and deletion, and the root teardown's fan-out.
//!
//! The table keeps ascending task ID order
//! ([`BucketOrderMap`](crate::collections::BucketOrderMap)), which orders every
//! fan-out across tasks and the releases when the table empties.
//!
//! A task's root word is written at creation only when the root node exists then, and never when
//! restoring (`tests::create::local_task_and_root_word`). Deletion and the root teardown tear
//! tasks down without flushing their resources (`tests::delete::deletes_and_clears_word`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::device::{DeviceCore, DeviceDeath, DeviceRaise};
use crate::guest::GuestPtr;
use crate::platform::Platform;
use crate::raise::{OnFault, RaiseResult};
use crate::rt::Hold;
use crate::seams::links::{KernelTaskRef, TaskDeviceLink};
use crate::tasks::Task;
use crate::tasks::local::LocalTask;
use crate::tasks::remote::RemoteTask;

/// The task table's size: task ids are below 256.
pub const MAX_TASKS: u32 = 256;

/// The largest local task (32 GiB).
pub const MAX_LOCAL_TASK_LENGTH: u64 = 0x8_0000_0000;

/// A task's root word written through a missing root node (never reached: the root node is
/// checked first).
const ROOT_WORD: DeviceDeath = DeviceDeath::TaskRootWord;

/// Task `task_id`'s word in the root node.
fn write_root_word(root_node: GuestPtr, task_id: u32, word: u32) {
    root_node.write_u32(u64::from(task_id).wrapping_mul(4), word, ROOT_WORD);
}

impl<P: Platform> DeviceCore<P> {
    /// The task, retained, or a raise for a task the table does not hold, as `on` says: its
    /// callers reach it both under a handler and with none.
    pub fn get_task_id(&self, on: OnFault, task_id: u32) -> RaiseResult<Task<P>> {
        let tasks = self.tasks.lock();
        let task = tasks.get(task_id).cloned();
        let Some(task) = task else {
            crate::raise!(on => DeviceRaise::InvalidTask, format_args!("no task {task_id}"));
        };
        Ok(task)
    }

    /// A user task: task 0 raises before the lookup.
    pub fn get_user_task_id(&self, task_id: u32) -> RaiseResult<Task<P>> {
        if task_id == 0 {
            crate::raise!(DeviceRaise::UserTaskZero, "task 0 is not a user task");
        }
        self.get_task_id(OnFault::Return, task_id)
    }

    /// Task 0 as a kernel task, retained. Task 0 is always local (tasks are remote only for
    /// non-zero ids), so the remote arm is dead; there the send would not be recognized. Its
    /// raises are as `on` says: its callers reach it both under a handler and with none (a
    /// display's colour-space read from an API entry).
    pub fn get_kernel_task(&self, on: OnFault) -> RaiseResult<KernelTaskRef<P>> {
        let task = self.get_task_id(on, 0)?;
        match &task {
            Task::Local(local) => Ok(Arc::clone(local)),
            Task::Remote(_) => crate::raise!(on => DeviceRaise::KernelTaskNotLocal,
                "task 0 is remote, so not the kernel task"
            ),
        }
    }

    /// Makes the task under the task lock (a remote task for a user task under isolation, a
    /// local one otherwise) and, unless restoring, writes its root word when a root node
    /// exists.
    pub fn create_task_id(
        &self,
        hold: Hold<'_>,
        task_id: u32,
        task_root: u32,
        length: u64,
        restoring: bool,
    ) -> RaiseResult<()> {
        // DIAG(empty-entry): rare; a task id defined again would show here twice.
        log::debug!(
            "DefineTask: id {task_id} root page {task_root:#x} length {length:#x} restoring {restoring}"
        );
        if task_id >= MAX_TASKS {
            crate::raise!(
                DeviceRaise::InvalidTaskId,
                format_args!("task id {task_id} is not below {MAX_TASKS}")
            );
        }
        {
            let mut tasks = self.tasks.lock();
            if tasks.get(task_id).is_some() {
                crate::raise!(
                    DeviceRaise::TaskExists,
                    format_args!("task {task_id} exists")
                );
            }
            let host: TaskDeviceLink<P> = self.weak_self.clone();
            let task = if task_id != 0 && self.isolates_guest_tasks() {
                RemoteTask::launch(
                    &self.platform,
                    host,
                    self.platform.launcher(),
                    task_root,
                    length,
                    task_id,
                )
                .map(Task::Remote)
            } else {
                if length > MAX_LOCAL_TASK_LENGTH {
                    crate::raise!(
                        DeviceRaise::TaskTooLarge,
                        format_args!(
                            "local task {task_id} of {length:#x} bytes is larger than \
                             {MAX_LOCAL_TASK_LENGTH:#x}"
                        )
                    );
                }
                LocalTask::new(
                    hold,
                    self.platform.clone(),
                    host,
                    task_root,
                    length,
                    task_id,
                )?
                .map(Task::Local)
            };
            let Some(task) = task else {
                crate::raise!(
                    DeviceRaise::TaskNotCreated,
                    format_args!("task {task_id} not created")
                );
            };
            tasks.insert(task_id, task);
        }
        if !restoring {
            let root = self.root_node.lock();
            if let Some(root_node) = root.root_node {
                write_root_word(root_node, task_id, task_root);
            }
        }
        Ok(())
    }

    /// Tears the task down and releases it under the task lock (no flush), then clears its root
    /// word. `Ok(false)` when there is no such task.
    pub fn delete_task_id(&self, task_id: u32) -> RaiseResult<bool> {
        if task_id >= MAX_TASKS {
            crate::raise!(
                DeviceRaise::InvalidTaskId,
                format_args!("task id {task_id} is not below {MAX_TASKS}")
            );
        }
        {
            let mut tasks = self.tasks.lock();
            let Some(task) = tasks.get(task_id) else {
                return Ok(false);
            };
            task.teardown();
            // The table's release, under the lock (a host block that raises in it ends the
            // process there).
            drop(tasks.remove(task_id));
        }
        let mut cleared = false;
        match self.root_node.try_lock() {
            Some(root) => {
                if let Some(root_node) = root.root_node {
                    write_root_word(root_node, task_id, 0);
                    cleared = true;
                }
            }
            // Contended: whether the root is being torn down, read without a lock.
            None => cleared = self.tearing_down_root.load(Ordering::Relaxed),
        }
        if !cleared {
            let root = self.root_node.lock();
            if let Some(root_node) = root.root_node {
                write_root_word(root_node, task_id, 0);
            }
        }
        Ok(true)
    }

    /// Tears every task down in the table's order, then empties the table (its releases in
    /// ascending task ID order). A raise from a teardown leaves the table as it is and goes on.
    ///
    /// The task lock is not held across the teardowns. It is taken only to copy the tasks and again
    /// to empty the table, so a thread holding it (a delete tearing its task down, a remote task's
    /// creation across its handshake) is waited for, and the table is seen as that thread left it.
    /// Reading and emptying it with no lock at all would be a race with no defined outcome; the
    /// sequential outcomes are kept.
    pub fn teardown_tasks(&self) {
        // A real pool: the host's synchronous callbacks reached inside it (the tasks' teardowns,
        // and their releases) autorelease into it.
        let pool = crate::rt::Pool::push(self.platform.as_ref());
        let all: Arc<[Task<P>]> = self.tasks.lock().values().cloned().collect();
        for task in all.iter() {
            task.teardown();
        }
        let removed = self.tasks.lock().drain();
        drop(removed);
        // The last references: the tasks go in the table's copy order.
        drop(all);
        drop(pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_support::{Opts, device};
    use crate::raise::RaiseId;
    use crate::raise::death::{DeathKind, DeathSite};
    use crate::testkit::capture::catch_death;

    /// `(raise, detail)` of a device raise.
    fn raised<T: std::fmt::Debug>(r: RaiseResult<T>) -> (DeviceRaise, String) {
        let r = r.expect_err("expected a raise");
        match r.id() {
            RaiseId::Device(k) => (k, r.detail().unwrap_or_default().to_owned()),
            other => panic!("not the device's raise: {other:?}"),
        }
    }

    mod lookup {
        use super::*;

        /// a missing task throws, naming it, for every lookup
        #[test]
        fn missing_task_throws() {
            let t = device(Opts::default());
            assert_eq!(
                raised(t.device.get_task_id(OnFault::Return, 9)),
                (DeviceRaise::InvalidTask, "no task 9".into())
            );
            assert_eq!(
                raised(t.device.get_kernel_task(OnFault::Return).map(drop)),
                (DeviceRaise::InvalidTask, "no task 0".into())
            );
            assert_eq!(
                raised(t.device.get_user_task_id(9)),
                (DeviceRaise::InvalidTask, "no task 9".into())
            );
        }

        /// the user lookup refuses task 0 before looking
        #[test]
        fn user_task_zero_throws() {
            let t = device(Opts::default());
            assert_eq!(
                raised(t.device.get_user_task_id(0)),
                (
                    DeviceRaise::UserTaskZero,
                    "task 0 is not a user task".into()
                )
            );
        }

        /// with no handler live the throw ends the process at the throw
        #[test]
        fn uncaught_dies() {
            let t = device(Opts::default());
            let site = catch_death(|| t.device.get_task_id(OnFault::Die, 1)).expect_err("death");
            assert_eq!(
                site,
                DeathSite {
                    kind: DeathKind::UnhandledRaise,
                    file: DeviceRaise::InvalidTask.name(),
                    line: 0
                }
            );
        }
    }

    mod create {
        use super::*;

        /// task IDs from 256 up throw
        #[test]
        fn id_out_of_range_throws() {
            let t = device(Opts::default()).with_root_node();
            let hold = Hold::NONE;
            assert_eq!(
                raised(t.device.create_task_id(hold, 256, 5, 0x4000, false)),
                (
                    DeviceRaise::InvalidTaskId,
                    "task id 256 is not below 256".into()
                )
            );
            assert!(t.device.tasks.lock().is_empty());
        }

        /// a local task above 32 GiB throws and writes no word
        #[test]
        fn local_too_large_throws() {
            let t = device(Opts::default()).with_root_node();
            let hold = Hold::NONE;
            assert_eq!(
                raised(
                    t.device
                        .create_task_id(hold, 0, 5, MAX_LOCAL_TASK_LENGTH + 1, false)
                ),
                (
                    DeviceRaise::TaskTooLarge,
                    "local task 0 of 0x800000001 bytes is larger than 0x800000000".into()
                )
            );
            assert!(t.device.tasks.lock().is_empty());
            assert_eq!(t.root_word(0), 0);
        }

        /// a define writes its root word only with a root node, and not when restoring; a second
        /// define of the same ID throws
        #[test]
        fn local_task_and_root_word() {
            let t = device(Opts::default()).with_root_page();
            let hold = Hold::NONE;
            t.task_root_node(0x55, 0x4000);
            t.task_root_node(0x66, 0x4000);
            t.device
                .create_task_id(hold, 3, 0x55, 0x4000, false)
                .unwrap();
            assert_eq!(t.guest_root_word(3), 0x55);
            assert!(matches!(
                t.device.get_task_id(OnFault::Return, 3).unwrap(),
                Task::Local(_)
            ));
            assert!(matches!(
                t.device.get_user_task_id(3).unwrap(),
                Task::Local(_)
            ));
            t.device
                .create_task_id(hold, 4, 0x66, 0x4000, true)
                .unwrap();
            assert_eq!(t.guest_root_word(4), 0);
            assert_eq!(
                raised(t.device.create_task_id(hold, 3, 0x55, 0x4000, false)),
                (DeviceRaise::TaskExists, "task 3 exists".into())
            );
            // the root task stays (the mapping needs it); only the root node is gone
            let no_root = device(Opts::default()).with_root_page();
            no_root.device.root_node.lock().root_node = None;
            no_root.task_root_node(0x55, 0x4000);
            no_root
                .device
                .create_task_id(hold, 3, 0x55, 0x4000, false)
                .unwrap();
            assert!(no_root.device.root_node.lock().root_node.is_none());
            assert_eq!(no_root.device.tasks.lock().len(), 1);
        }

        /// a task that fails to initialise throws
        #[test]
        fn nil_task_throws() {
            let t = device(Opts::default()).with_root_page();
            let hold = Hold::NONE;
            // length 0: the local task logs that its length is 0 and refuses
            assert_eq!(
                raised(t.device.create_task_id(hold, 5, 0x55, 0, false)),
                (DeviceRaise::TaskNotCreated, "task 5 not created".into())
            );
            assert!(t.device.tasks.lock().is_empty());
            assert_eq!(t.guest_root_word(5), 0);
        }
    }

    mod delete {
        use super::*;

        /// task IDs from 256 up throw
        #[test]
        fn id_out_of_range_throws() {
            let t = device(Opts::default());
            assert_eq!(
                raised(t.device.delete_task_id(300)),
                (
                    DeviceRaise::InvalidTaskId,
                    "task id 300 is not below 256".into()
                )
            );
        }

        /// no such task: `false`, and the root word is left alone
        #[test]
        fn missing_returns_false() {
            let t = device(Opts::default()).with_root_node();
            t.set_root_word(7, 0x99);
            assert!(!t.device.delete_task_id(7).unwrap());
            assert_eq!(t.root_word(7), 0x99);
        }

        /// tears down without a flush, drops the task, clears its word
        #[test]
        fn deletes_and_clears_word() {
            let t = device(Opts::default()).with_root_page();
            t.task_root_node(0x99, 0x4000);
            t.device
                .create_task_id(Hold::NONE, 7, 0x99, 0x4000, false)
                .unwrap();
            assert_eq!(t.guest_root_word(7), 0x99);
            assert!(t.device.delete_task_id(7).unwrap());
            assert_eq!(t.guest_root_word(7), 0);
            assert!(t.device.tasks.lock().is_empty());
            assert!(!t.device.delete_task_id(7).unwrap());
        }

        /// the task's release raising (the host's `destroyTask` block) ends the process
        /// there, before the root word is cleared
        #[test]
        fn a_raising_destroy_task_dies_before_the_word() {
            let t = device(Opts::default()).with_root_page();
            t.task_root_node(0x99, 0x4000);
            t.device
                .create_task_id(Hold::NONE, 7, 0x99, 0x4000, false)
                .unwrap();
            t.platform.script().push(
                "destroyTask",
                crate::testkit::Step::raise("HostRefused", "no destroy"),
            );
            let d = crate::testkit::capture::catch_death(|| {
                let _ = t.device.delete_task_id(7);
            })
            .unwrap_err();
            assert_eq!(
                (d.kind, d.file),
                (crate::raise::death::DeathKind::HostRaised, "destroyTask")
            );
            assert_eq!(t.guest_root_word(7), 0x99);
        }

        /// contended root-node lock during a root teardown: the word is left (`cleared =
        /// _tearingDownRoot`); not tearing down, the delete waits for the lock and clears it
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time polling
        fn contended_lock() {
            let t = std::sync::Arc::new(device(Opts::default()).with_root_page());
            t.task_root_node(0x99, 0x4000);
            t.task_root_node(0x98, 0x4000);
            t.device
                .create_task_id(Hold::NONE, 7, 0x99, 0x4000, false)
                .unwrap();
            t.device
                .create_task_id(Hold::NONE, 8, 0x98, 0x4000, false)
                .unwrap();
            {
                let _held = t.device.root_node.lock();
                t.device.tearing_down_root.store(true, Ordering::Relaxed);
                assert!(t.device.delete_task_id(7).unwrap());
                t.device.tearing_down_root.store(false, Ordering::Relaxed);
            }
            assert_eq!(t.guest_root_word(7), 0x99);
            let held = t.device.root_node.lock();
            let t2 = t.clone();
            let h = std::thread::spawn(move || t2.device.delete_task_id(8).unwrap());
            std::thread::sleep(std::time::Duration::from_millis(20));
            assert!(!h.is_finished());
            drop(held);
            assert!(h.join().unwrap());
            assert_eq!(t.guest_root_word(8), 0);
        }
    }

    mod edges {
        use super::*;

        /// No root node with its lock free: `true`, no word written, no death
        #[test]
        fn delete_without_root_node() {
            let t = device(Opts::default()).with_root_page();
            t.task_root_node(0x99, 0x4000);
            t.device
                .create_task_id(Hold::NONE, 7, 0x99, 0x4000, false)
                .unwrap();
            assert_eq!(t.guest_root_word(7), 0x99);
            t.device.root_node.lock().root_node = None;
            assert!(t.device.delete_task_id(7).unwrap());
            assert_eq!(t.guest_root_word(7), 0x99);
            assert!(t.device.tasks.lock().is_empty());
        }

        /// the size check is a strict `>`: exactly the limit passes it
        #[test]
        fn length_at_limit_passes_size_check() {
            let t = device(Opts::default()).with_root_page();
            t.task_root_node(0x55, MAX_LOCAL_TASK_LENGTH);
            let hold = Hold::NONE;
            let r = t
                .device
                .create_task_id(hold, 0, 0x55, MAX_LOCAL_TASK_LENGTH, false);
            if let Err(e) = r {
                assert_ne!(e.id(), RaiseId::Device(DeviceRaise::TaskTooLarge));
            }
        }
    }

    mod teardown {
        use super::*;

        /// an empty table: nothing to do
        #[test]
        fn empty_table() {
            let t = device(Opts::default());
            t.device.teardown_tasks();
            assert!(t.device.tasks.lock().is_empty());
        }

        /// every task torn down, then the table emptied
        #[test]
        fn tears_down_all() {
            let t = device(Opts::default()).with_root_page();
            t.task_root_node(0x40, 0x4000);
            for id in 0..9 {
                t.device
                    .create_task_id(Hold::NONE, id, 0x40, 0x4000, true)
                    .unwrap();
            }
            assert_eq!(t.device.tasks.lock().len(), 9);
            t.device.teardown_tasks();
            assert!(t.device.tasks.lock().is_empty());
        }

        /// The teardown's pool is a real one. The tasks' teardowns and their releases run
        /// inside it, so what the host's `destroyTask` and `unmapMemory` blocks autorelease there
        /// is released at its pop, not in the host's pool.
        #[test]
        fn the_host_blocks_run_inside_the_pool() {
            use crate::device::root::rig::rig_with;
            use crate::rt::pool::test_native::{events, note, with_test_native};
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, Ordering};
            let armed = Arc::new(AtomicBool::new(true));
            let (a, b) = (armed.clone(), armed.clone());
            let r = rig_with(|blocks| {
                let destroy = blocks.destroy_task.clone().expect("destroyTask");
                blocks.destroy_task = Some(Arc::new(move |t| {
                    if a.load(Ordering::SeqCst) {
                        note("host block");
                    }
                    destroy(t)
                }));
                let unmap = blocks.unmap_memory.clone().expect("unmapMemory");
                blocks.unmap_memory = Some(Arc::new(move |t, off, len| {
                    if b.load(Ordering::SeqCst) {
                        note("host block");
                    }
                    unmap(t, off, len)
                }));
            });
            r.device.set_root_page(Hold::NONE, 1);
            let pa = 0x40u64 * crate::guest::records::APV_PAGE_SIZE;
            r.platform.fake_vm().sync();
            assert!(r.host.ram().write_u32(pa + 4, 1));
            assert!(r.host.ram().write_u64(pa + 0x14, 0x4000));
            r.platform.fake_vm().sync();
            let hold = Hold::NONE;
            for id in 1..3 {
                r.device
                    .create_task_id(hold, id, 0x40, 0x4000, true)
                    .unwrap();
            }
            let mut got = Vec::new();
            with_test_native(|| {
                let _ = events();
                r.device.teardown_tasks();
                got = events();
                armed.store(false, Ordering::SeqCst);
            });
            assert!(r.device.tasks.lock().is_empty());
            assert_eq!(
                got.first().map(String::as_str),
                Some("push 0x100"),
                "{got:?}"
            );
            assert_eq!(got.last().map(String::as_str), Some("pop 0x100"), "{got:?}");
            assert!(
                got.iter().any(|e| e == "host block"),
                "no host block ran: {got:?}"
            );
        }

        /// An empty table still pushes and pops the pool.
        #[test]
        fn an_empty_table_pushes_and_pops_the_pool() {
            use crate::rt::pool::test_native::{events, with_test_native};
            let t = device(Opts::default());
            let mut got = Vec::new();
            with_test_native(|| {
                t.device.teardown_tasks();
                got = events();
            });
            assert_eq!(got, vec!["push 0x100", "pop 0x100"]);
        }
    }
}
