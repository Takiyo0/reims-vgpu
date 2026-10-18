//! Making a local task: its parts, its host task (blocks setup), the task root and the walker
//! over the guest's page table.

use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use super::{
    Backend, DeathKind, LocalTask, PAGE, ROOT_OBJECT_LIST_LENGTH, ROOT_OBJECT_LIST_OFFSET,
    ROOT_PAGE_TABLE_DEPTH, ROOT_PAGE_TABLE_ROOT, ROOT_TASK_LENGTH,
};
use crate::deserializer::Deserializer;
use crate::device::DeviceCore;
use crate::guest::GuestPtr;
use crate::guest::records::APV_PAGE_SHIFT;
use crate::memory::page_table::{INDEX_BITS, PageTableRead, PageTableWalker, host_error_log};
use crate::memory::tracker::RangeTracker;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::raise::RaiseResult;
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, Raise};
use crate::resources::host::ResourceHost;
use crate::resources::manager::ResourceManager;
use crate::resources::resource::with_platform;
use crate::rt::Hold;
use crate::seams::links::TaskDeviceLink;
use crate::tasks::object_lists::{ObjectListKind, ObjectLists};
use crate::tasks::shared::{TaskCore, TaskParts};
use crate::tasks::{DeathKind as TasksDeath, TaskInfo};

/// How the build of a task ended.
enum Built {
    /// The task is live.
    Live,
    /// The build refused: the partial task is dropped.
    Refused,
    /// A raise left the build: the half-built task is kept for good.
    Raised(Raise),
}

/// The guest memory a page table of `depth` levels covers: [`ENTRIES`] entries a level, of guest
/// pages (4096 of 16 KiB, 1024 of 4 KiB on x86-64).
///
/// [`ENTRIES`]: crate::memory::page_table::ENTRIES
pub(super) fn page_table_coverage(depth: u32) -> u64 {
    let mut pages: u64 = 1;
    for _ in 0..depth {
        pages = pages.wrapping_shl(INDEX_BITS);
    }
    pages.wrapping_shl(APV_PAGE_SHIFT)
}

impl<P: Platform> LocalTask<P> {
    /// A local task over the task root page `task_root` (0: none) of `length` bytes, or `None`
    /// where the build refuses (a zero length, a refused host task, a nil root mapping, a bad
    /// page table, a gone device): the partial task is dropped then, which destroys its host
    /// task. A raise out of the build (a host block's, or Metal's at the paging event) goes on,
    /// and the half-built task is kept for good with its host task and the parts it has.
    ///
    /// The device passes its platform: the resource manager and the deserializer need it.
    pub fn new(
        hold: Hold<'_>,
        platform: Arc<P>,
        device: TaskDeviceLink<P>,
        task_root: u32,
        length: u64,
        task_id: u32,
    ) -> RaiseResult<Option<Arc<Self>>> {
        with_platform(&platform.clone(), || {
            // With the device gone there is no Metal device: the build ends at the host task's
            // refusal, or at the length test before it
            let Some(dev) = device.upgrade() else {
                if length == 0 {
                    log::warn!("task {task_id} refused: its length is 0");
                }
                return Ok(None);
            };
            let mut built = Built::Live;
            let task = Arc::new_cyclic(|weak: &Weak<LocalTask<P>>| {
                let host: Weak<dyn ResourceHost<P>> = weak.clone();
                Self::build(
                    hold, platform, device, &*dev, host, task_root, length, task_id, &mut built,
                )
            });
            drop(dev);
            match built {
                Built::Live => Ok(Some(task)),
                Built::Refused => {
                    drop(task);
                    Ok(None)
                }
                Built::Raised(thrown) => {
                    // Its host task's `destroyTask` and its root page's unmap never come.
                    keep_forever(task, Kept::HostReleasesNeverCome);
                    Err(thrown)
                }
            }
        })
    }

    /// The build, inside `Arc::new_cyclic` so the resource manager and the object lists can hold
    /// the task unretained. Returns the task as far as it got; `built` says how it ended.
    #[expect(
        clippy::too_many_arguments,
        reason = "the hold, the task's values and the cyclic plumbing"
    )]
    fn build(
        hold: Hold<'_>,
        platform: Arc<P>,
        device: TaskDeviceLink<P>,
        dev: &DeviceCore<P>,
        host: Weak<dyn ResourceHost<P>>,
        task_root: u32,
        length: u64,
        task_id: u32,
        built: &mut Built,
    ) -> Self {
        // The Metal device, held for the task's life; the resource manager shares the reference
        let mtl_device = dev.mtl_device();
        // Until the map setup is found, the blocks setup's: a refused build still destroys the
        // (zero) host task
        let mut me = LocalTask {
            base: TaskCore::new(
                Some(Arc::new(mtl_device)),
                0,
                None,
                None,
                0,
                false,
                false,
                false,
            ),
            platform,
            device,
            length,
            task_id: 0,
            task_info: TaskInfo::default(),
            backend: Backend::Blocks {
                mapped: RangeTracker::new(),
            },
            task_root_mapping: None,
            task_root: GuestPtr::default(),
            walker: None,
            io_surface_buffers_allowed: false,
            prepare_mutex: Mutex::new(()),
            object_list_range: Mutex::new((0, 0)),
            transfer_cache: Mutex::new(super::memory::TransferCache::default()),
        };
        if length == 0 {
            log::warn!("task {task_id} refused: its length is 0");
            *built = Built::Refused;
            return me;
        }
        let lists = if dev.memory_map().is_none() {
            let lists = ObjectLists::new(
                ObjectListKind::Legacy {
                    object_list_mapped: false,
                },
                me.length,
                false,
                host.clone(),
            );
            if !dev.create_task(length, &mut me.task_info) {
                *built = Built::Refused;
                return me;
            }
            lists
        } else {
            me.backend = Backend::MemoryMap;
            ObjectLists::new(ObjectListKind::MemoryMap, me.length, false, host.clone())
        };
        // No GPU addresses registered
        let resource_manager = ResourceManager::new(
            me.platform.clone(),
            me.base.mtl_device.clone(),
            host,
            lists.clone(),
            false,
        );
        // The manager and the lists hold nothing yet (no resource, no mapping, and the Metal
        // device is the task's too): on a refusal they go before the task.
        if task_root != 0 && !me.setup_task_root(dev, task_root) {
            *built = Built::Refused;
            return me;
        }
        let deserializer = Deserializer::new(
            me.platform.clone(),
            me.base.mtl_device(),
            resource_manager.object_delegate(),
        );
        // Installed now, so a raise at the paging event below keeps them with the task
        me.base.set_parts(TaskParts {
            deserializer,
            resource_manager,
            object_list_manager: lists.clone(),
        });
        me.base.paging_queue = dev.blit_down_queue();
        let event = match &me.base.mtl_device {
            Some(d) => d.new_event(OnFault::Return),
            None => Ok(None),
        };
        me.base.paging_event = match event {
            Ok(event) => event,
            Err(thrown) => {
                *built = Built::Raised(thrown);
                return me;
            }
        };
        *me.base.paging_counter.get_mut() = 1;
        // With task root 0 the root is NULL and this read faults.
        let offset = me
            .task_root
            .read_u64(ROOT_OBJECT_LIST_OFFSET, TasksDeath::NullTaskRoot);
        let list_length = me
            .task_root
            .read_u32(ROOT_OBJECT_LIST_LENGTH, TasksDeath::NullTaskRoot);
        // The answer is only mirrored; the root's list fields are written back, zeros on a
        // refusal
        let stored = lists.set_object_list(hold, offset, list_length, 0, 0, me.task_root);
        *me.object_list_range.get_mut() = if stored {
            (offset, list_length)
        } else {
            (0, 0)
        };
        me.task_id = task_id;
        let device_supports = match &me.base.mtl_device {
            Some(d) => d.caps().flags.supports_buffer_with_iosurface,
            None => false,
        };
        me.io_surface_buffers_allowed =
            device_supports && dev.features().load().buffer_from_io_surface;
        me
    }

    /// Maps the task root page and sets up the walker over the page table it names, writing the
    /// task's length into the root. `false` where it refuses: a nil mapping (no log), or a page
    /// table whose depth is not 1 to 3 or which covers less than the task (a log).
    fn setup_task_root(&mut self, dev: &DeviceCore<P>, task_root: u32) -> bool {
        let Some(mapping) = dev.new_contiguous_mapping(task_root, PAGE) else {
            return false;
        };
        self.task_root = GuestPtr::new(mapping.virtual_address());
        self.task_root_mapping = Some(mapping);
        // The walker's read holds the device weakly. Forming that weak reference once the
        // device's teardown has begun is the runtime's abort: after the root is mapped, before
        // its depth is read
        if dev.is_deallocating_now() {
            crate::fatal!(DeathKind::DeviceDeallocating)
        }
        // Read once, now
        let depth = self
            .task_root
            .read_u32(ROOT_PAGE_TABLE_DEPTH, TasksDeath::NullTaskRoot);
        if depth == 0 || depth > 3 {
            log::warn!("task refused: page table depth {depth} is not 1 to 3");
            return false;
        }
        let root = self
            .task_root
            .read_u32(ROOT_PAGE_TABLE_ROOT, TasksDeath::NullTaskRoot);
        if self.length > page_table_coverage(depth) {
            log::warn!(
                "task refused: {:#x} bytes are more than a page table of depth {depth} covers",
                self.length
            );
            return false;
        }
        let device = self.device.clone();
        let read: PageTableRead = Arc::new(
            move |hold: Hold<'_>, on: OnFault, address: u64, dst: &mut [u8]| {
                // Each read loads the device weakly: nothing once its teardown has begun, else a
                // strong reference held to the end of the read's hold's holder (`Hold::keep`), so
                // a last host release mid-walk tears the device down at that holder's end, after the
                // walking work, never inside the walk (lessons.md: weak loads)
                let Some((d, handle)) = device
                    .upgrade()
                    .and_then(|d| d.retain_self().map(|handle| (d, handle)))
                else {
                    // Nothing read and `false`, which the walker ignores.
                    return Ok(false);
                };
                hold.keep(handle);
                d.read_from_physical_address(on, address, dst)
            },
        );
        // No holes allowed; the walker's construction never fails
        self.walker = Some(PageTableWalker::new(
            root,
            depth,
            false,
            read,
            host_error_log(self.platform.clone()),
        ));
        // Guest-visible: the root's task length
        self.task_root
            .write_u64(ROOT_TASK_LENGTH, self.length, TasksDeath::NullTaskRoot);
        true
    }
}
