//! The local task's own checks, and the raises its paging buffers' cleanups replace. Its other
//! behaviour against a device, the host's blocks and guest memory is the suite's (the blocks and
//! map setups).

use zerocopy::IntoBytes;

use super::exec::CompleteSync;
use super::memory::check_task_range;
use super::new::page_table_coverage;
use super::*;
use crate::device::test_support::{Opts, TestDevice, device};
use crate::guest::records::APV_PAGE_SIZE;
use crate::platform::device::GpuDevice;
use crate::raise::{Origin, Raise};
use crate::resources::buffer::BufferResourceDescriptor;
use crate::resources::buffer::private::PrivateBufferResource;
use crate::resources::manager::create::Resources;
use crate::resources::texture::views::RefTextureResource;
use crate::rt::Held;
use crate::testkit::FakePlatform;
use crate::testkit::faults::Fault;
use crate::testkit::objects::{Model, kind};

/// `t` is a raise of the local task's `kind`.
fn raised(t: &Raise, kind: RaiseKind) {
    assert_eq!(
        (t.site().file, t.site().line),
        (crate::tasks::RaiseKind::Local(kind).id(), 0),
        "{t:?}"
    );
}

/// Each level multiplies what the level below covers by a page of 4-byte entries: 64 MiB,
/// 256 GiB and 1 PiB on a 16 KiB guest page.
#[test]
fn coverage_per_depth() {
    let entries = APV_PAGE_SIZE / 4;
    assert_eq!(page_table_coverage(1), entries * APV_PAGE_SIZE);
    assert_eq!(page_table_coverage(2), entries * entries * APV_PAGE_SIZE);
    assert_eq!(
        page_table_coverage(3),
        entries * entries * entries * APV_PAGE_SIZE
    );
}

#[test]
fn accepts_a_range_ending_at_the_task_end_and_throws_past_it() {
    assert!(check_task_range(0, 0x10, 0x10).is_ok());
    raised(
        &check_task_range(1, 0x10, 0x10).unwrap_err(),
        RaiseKind::Range,
    );
    let e = check_task_range(u64::MAX, 2, u64::MAX).unwrap_err();
    raised(&e, RaiseKind::Range);
    assert_eq!(
        e.detail().unwrap_or_default(),
        format!(
            "0x2 bytes at task offset {:#x} run past the task's {:#x}",
            u64::MAX,
            u64::MAX
        )
    );
}

/// The resource the paging tests name.
const VIEW: u32 = 7;

/// A device's kernel task (the blocks setup) whose page table has a hole at task page 0, with a
/// view over a private buffer of that page cached under [`VIEW`]. The view's paging is a landing
/// frame: the buffer binds the page through the page table, whose read holds the device to the
/// frame's end, then the hole raises, and the raise carries that hold. The device goes last.
struct Paged {
    view: Arc<Resource<FakePlatform>>,
    task: Arc<LocalTask<FakePlatform>>,
    td: TestDevice,
}

fn paged() -> Paged {
    let td = device(Opts::default()).with_root_page();
    td.task_root_node(0x55, 0x4000);
    // The page table's root: page 0x56, all holes
    td.platform.fake_vm().sync();
    assert!(td.host.ram().write_u32(0x55 * APV_PAGE_SIZE, 0x56));
    td.platform.fake_vm().sync();
    let task = {
        let hold = Hold::NONE;
        td.device
            .create_task_id(hold, 0, 0x55, 0x4000, false)
            .unwrap();
        match td.device.get_task_id(OnFault::Return, 0).unwrap() {
            crate::tasks::Task::Local(t) => t,
            _ => panic!("no local kernel task"),
        }
    };
    let host: Arc<dyn ResourceHost<FakePlatform>> = task.clone();
    let descriptor = BufferResourceDescriptor {
        length: 0x4000,
        page: 0,
        flags: 1,
        unknown0d: [0; 3],
    };
    let view = task.on_platform(|| {
        let buffer = PrivateBufferResource::new(Arc::downgrade(&host), descriptor.as_bytes(), None)
            .unwrap()
            .unwrap();
        let texture = td.platform.world().new_obj::<kind::Texture>(Model::None);
        RefTextureResource::new(Arc::downgrade(&host), 0, texture, buffer, 0)
            .unwrap()
            .unwrap()
    });
    let (_, resource_manager, _) = task.parts();
    assert!(resource_manager.register_object::<Resources>(VIEW, Arc::clone(&view)));
    // The paging buffer's blit encoder raises at its `endEncoding`
    td.platform.faults().at(
        "MTLBlitCommandEncoder",
        "endEncoding",
        0,
        Fault::Raise {
            name: "NSInternalInconsistencyException".into(),
            reason: "endEncoding raised".into(),
        },
    );
    Paged { view, task, td }
}

/// A raise from the paging buffer's cleanup goes on in place of the prepare's, which ends in
/// the preparation: what it carries (the device the page-table read held) is released at the
/// catcher's holder's end, not as the preparation returns
#[test]
fn a_paging_buffer_cleanup_raise_ends_the_prepares_raise() {
    let p = paged();
    let records = [CmdExecIndirectResource3 {
        reference: VIEW,
        flags: 0,
    }];
    let resources = ResourceList::V3(Records::new(records.as_bytes(), 1).unwrap());
    let mark = p.td.platform.call_log().mark();
    let device = Arc::strong_count(&p.td.device);
    let held = Held::new();
    let hold = Hold::of(&held);
    let prepared = p
        .task
        .on_platform(|| p.task.prepare_resources(hold, &resources, None, 0));
    let raised = prepared.err().expect("the cleanup's raise");
    assert_eq!(raised.origin(), Origin::Boundary("endEncoding"));
    // The paging buffer is never committed
    assert!(
        !p.td
            .platform
            .call_log()
            .sels_since(mark)
            .contains(&"commit")
    );
    assert_eq!(Arc::strong_count(&p.td.device), device + 1);
    hold.land(raised);
    drop(held);
    assert_eq!(Arc::strong_count(&p.td.device), device);
}

/// A raise from the synchronize buffer's cleanup goes on in place of the synchronize's, which
/// ends there: what it carries is released at the catcher's holder's end
#[test]
fn a_synchronize_buffer_cleanup_raise_ends_the_synchronizes_raise() {
    let p = paged();
    // The view's host copy is current and its guest copy stale: it is written back
    p.view.set_is_host_valid(true);
    p.view.set_is_guest_valid(false);
    let records = [CmdExecIndirectResource3 {
        reference: VIEW,
        flags: 4,
    }];
    let resources = ResourceList::V3(Records::new(records.as_bytes(), 1).unwrap());
    let event = p.td.platform.device().new_shared_event().unwrap();
    let mark = p.td.platform.call_log().mark();
    let device = Arc::strong_count(&p.td.device);
    let held = Held::new();
    let hold = Hold::of(&held);
    let synchronized = p.task.on_platform(|| {
        p.task.complete_resources(
            hold,
            &resources,
            CompleteSync {
                needs_complete: true,
                event: Some(&event),
                value: 1,
            },
            Box::new(|_| {}),
        )
    });
    let raised = synchronized.unwrap_err();
    assert_eq!(raised.origin(), Origin::Boundary("endEncoding"));
    assert!(
        !p.td
            .platform
            .call_log()
            .sels_since(mark)
            .contains(&"commit")
    );
    assert_eq!(Arc::strong_count(&p.td.device), device + 1);
    hold.land(raised);
    drop(held);
    assert_eq!(Arc::strong_count(&p.td.device), device);
}
