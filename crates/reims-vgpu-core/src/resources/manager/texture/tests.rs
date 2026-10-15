//! The textures' creators over the test environment (`create::test_env`), and through a
//! manager.

use std::sync::Arc;

use super::*;
use crate::platform::ObjId;
use crate::platform::caps::DeviceCaps;
use crate::resources::host::ResourceHost;
use crate::resources::manager::cache::ManagedKind;
use crate::resources::manager::create::test_env::*;
use crate::resources::manager::create::{Resources, create_resource};
use crate::resources::resource::{Resource, ResourceKind};
use crate::testkit::Fk;
use crate::testkit::capture::catch_death;
use crate::testkit::objects::{Model, TexDesc};
use crate::testkit::{Arg, Step, kind};

fn ok<T: std::fmt::Debug>(r: RaiseResult<Result<T, String>>) -> T {
    r.expect("no raise").expect("created")
}

fn err<T: std::fmt::Debug>(r: RaiseResult<Result<T, String>>) -> String {
    r.expect("no raise").expect_err("refused")
}

/// Makes the next texture Metal makes answer `gpuResourceID` with `id`.
fn next_texture_id(e: &TestEnv, id: u64) {
    e.p.world()
        .on_new::<kind::Texture>(move |t| t.set_gpu_resource_id(id));
}

/// The misplaced texture's line.
fn misplaced(placed: u64, got: u64) -> String {
    format!("texture placed at resource ID {got:#x}, not the guest's {placed:#x}")
}

fn dual_plane(e: &TestEnv, on: bool) {
    e.features(&GuestFeatures {
        dual_plane_textures: on,
        supports_shared_textures: on,
        ..GuestFeatures::default()
    });
}

/// The descriptor values a fake texture was made with, as its model keeps them.
fn made_with(t: &Fk<kind::Texture>) -> TexDesc {
    match &t.state().model {
        Model::Texture(m) => m.desc,
        _ => panic!("not a texture"),
    }
}

/// The ordered calls `d` received (its pixel format and resource options), in order.
fn ordered_on(e: &TestEnv, id: ObjId) -> Vec<(&'static str, Vec<Arg>)> {
    e.p.call_log()
        .calls()
        .into_iter()
        .filter(|c| c.obj == id)
        .map(|c| (c.sel, c.args))
        .collect()
}

/// A dimension header with `mips` levels.
fn dimension(mips: u16) -> TextureDimension {
    TextureDimension {
        bits: mips,
        array_length: 1,
        base_offset: 0,
        bytes_per_slice: 0x100,
        block_width: 1,
        block_height: 1,
        block_depth: 1,
        block_bytes: 4,
    }
}

/// A normal texture's object record, dimension and `mips` mips per plane.
fn normal_prefix(mips: u16, planes: u64) -> Bytes {
    let mut b = Bytes::new()
        .u64(0x400)
        .u32(9)
        .raw(dimension(mips).as_bytes());
    let rest = (u64::from(mips) * 0x24 + 0x18) * planes - 0x18;
    for i in 0..rest {
        b = b.u8(i.to_le_bytes()[0]);
    }
    b
}

// ---- descriptors

/// The first form's value: every property as the guest wrote it, no index for 0; made, its
/// pixel format then its resource options are the descriptor's ordered sets
#[test]
fn descriptor_first_form_value() {
    let e = TestEnv::new();
    let mut s = desc1(3, 80, 16, 8, 0x20);
    s.bits |= 0x10 | 0x40; // framebufferOnly, allowGPUOptimizedContents
    s.depth = 2;
    s.mipmap_level_count = 4;
    s.sample_count = 1;
    s.array_length = 6;
    s.protection_options = 0x77;
    assert_eq!(
        texture_desc(&s, 0),
        TextureDesc {
            texture_type: 3,
            pixel_format: 80,
            width: 16,
            height: 8,
            depth: 2,
            mipmap_level_count: 4,
            sample_count: 1,
            array_length: 6,
            usage: 1,
            resource_options: Some(0x20),
            swizzle: None,
            resource_index: None,
            protection_options: Some(0x77),
            framebuffer_only: Some(true),
            is_drawable: Some(false),
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: Some(true),
            rotation: None,
        }
    );
    let d = Serialized::First(s).build(e.p.as_ref(), 0);
    assert_eq!(
        ordered_on(&e, d.id()),
        [
            ("setPixelFormat:", vec![Arg::U64(80)]),
            ("setResourceOptions:", vec![Arg::U64(0x20)]),
        ]
    );
}

/// A texture Metal made none of is logged with its descriptor's pixel format, type, usage,
/// storage mode, protection options and size
#[test]
fn refused_texture_line_names_the_descriptor() {
    let mut s = desc1(2, 80, 1920, 1080, 0x20);
    s.protection_options = 0x10;
    assert_eq!(
        refused_line("texture", &texture_desc(&s, 0), 2),
        "Metal made no texture: pixel format 80, texture type 2, usage 0x1, storage mode 2, \
         protection options 0x10, size 1920x1080x1, mip levels 1, samples 1, array length 1, \
         over no IOSurface"
    );
    let unset = TextureDesc {
        protection_options: None,
        ..TextureDesc::default()
    };
    assert!(
        refused_line("shareable texture", &unset, 0)
            .starts_with("Metal made no shareable texture: pixel format 70, texture type 2,")
    );
    assert!(refused_line("texture", &unset, 0).contains("protection options unset,"));
}

#[test]
fn descriptor_forces_a_nonzero_index() {
    let s = desc1(2, 70, 1, 1, 0);
    assert_eq!(texture_desc(&s, 7).resource_index, Some((7, true)));
    assert_eq!(texture_desc(&s, 0).resource_index, None);
    // Made: the texture is at the index.
    let e = TestEnv::new();
    let d = Serialized::First(s).build(e.p.as_ref(), 7);
    let t = e.device.new_texture(OnFault::Die, &d).unwrap().unwrap();
    assert_eq!(t.resource_index(), 7);
}

#[test]
fn descriptor_second_form_swizzle_and_index() {
    let mut s = desc2(2, 70, 4, 4, 0);
    s.bits[0] |= 0x80; // writeSwizzleEnabled
    s.usage = 0x13;
    s.swizzle_red = 5;
    s.swizzle_green = 6;
    s.swizzle_blue = 0xff;
    s.swizzle_alpha = 0;
    let v = texture_desc2(&s, 9);
    assert_eq!(v.usage, 0x13, "usage is 32-bit in v2");
    assert_eq!(v.write_swizzle_enabled, Some(true));
    // A channel Metal does not have reads as One.
    assert_eq!(
        v.swizzle,
        Some(Swizzle {
            red: 5,
            green: 1,
            blue: 1,
            alpha: 0
        })
    );
    assert_eq!(v.resource_index, Some((9, true)));
    assert_eq!(texture_desc2(&s, 0).resource_index, None);
    // The rest as the first form's.
    let first = texture_desc(&desc1(2, 70, 4, 4, 0), 9);
    assert_eq!(
        TextureDesc {
            usage: 1,
            write_swizzle_enabled: None,
            swizzle: None,
            ..v
        },
        first
    );
}

// ---- IOSurface validation

fn surface(e: &TestEnv) -> Fk<kind::Surface> {
    // 16x16, 4 bytes per element: 64 bytes per row.
    e.p.fake_surfaces().make_simple(16, 16, 0x42475241, 4)
}

fn validate(
    e: &TestEnv,
    s: &SerializedTextureDescriptor,
    sf: &Fk<kind::Surface>,
    plane: u16,
) -> bool {
    let desc = texture_desc(s, 0);
    let d = e.p.texture_descriptor(&desc);
    validate_iosurface_descriptor::<crate::testkit::FakePlatform>(
        Some(&e.device),
        &desc,
        &d,
        Some(sf),
        plane,
    )
}

#[test]
fn iosurface_validation_passes_and_queries_in_order() {
    let e = TestEnv::new();
    let sf = surface(&e);
    e.p.script()
        .push("IOSurfaceAllowsPixelSizeCasting", Step::value(false));
    e.p.call_log().clear();
    assert!(validate(&e, &desc1(2, 70, 16, 16, 0), &sf, 0));
    assert_eq!(
        e.p.call_log().sels(),
        [
            "setPixelFormat:",
            "setResourceOptions:",
            "IOSurfaceGetPlaneCount",
            "MTLGetTextureLevelInfoForDeviceWithOptions()",
            "IOSurfaceGetBytesPerRow",
            "IOSurfaceGetHeight",
            "IOSurfaceAllowsPixelSizeCasting",
            "IOSurfaceGetBytesPerElement",
        ]
    );
    let q = e.calls("MTLGetTextureLevelInfoForDeviceWithOptions()");
    assert_eq!(
        q[0].args,
        vec![Arg::val(&TextureLevelQuery {
            pixel_format: 70,
            width: 16,
            height: 16,
            depth: 1,
            array_length: 1,
            options: 0,
            flags: 0
        })]
    );
}

/// A planar surface is not checked at all.
#[test]
fn iosurface_validation_skips_planar_surfaces() {
    let e = TestEnv::new();
    let sf = surface(&e);
    e.p.script()
        .push("IOSurfaceGetPlaneCount", Step::value(2u64));
    // Plane 5, depth 3, TextureBuffer: nothing is looked at.
    let mut s = desc1(9, 70, 4096, 4096, 0);
    s.depth = 3;
    e.p.call_log().clear();
    assert!(validate(&e, &s, &sf, 5));
    assert_eq!(
        e.calls("MTLGetTextureLevelInfoForDeviceWithOptions()")
            .len(),
        0
    );
}

#[test]
fn iosurface_validation_rotation_swaps_the_query() {
    let e = TestEnv::new();
    let sf = e.p.fake_surfaces().make_simple(16, 16, 0, 4);
    for (rotation, swapped) in [(0u64, false), (1, true), (2, false), (3, true), (4, true)] {
        let desc = texture_desc(&desc1(2, 70, 8, 2, 0), 0);
        let d = e.p.texture_descriptor(&desc);
        d.set_rotation(rotation);
        validate_iosurface_descriptor::<crate::testkit::FakePlatform>(
            Some(&e.device),
            &desc,
            &d,
            Some(&sf),
            0,
        );
        let q = e.calls("MTLGetTextureLevelInfoForDeviceWithOptions()");
        let (w, h) = if swapped { (2, 8) } else { (8, 2) };
        assert!(
            q.last().unwrap().args[0]
                == Arg::val(&TextureLevelQuery {
                    pixel_format: 70,
                    width: w,
                    height: h,
                    depth: 1,
                    array_length: 1,
                    options: 0,
                    flags: 0
                }),
            "rotation {rotation}"
        );
    }
}

#[test]
fn iosurface_validation_refusals() {
    let e = TestEnv::new();
    let sf = surface(&e);
    let ((), lines) = logged(|| {
        // Non-zero plane.
        assert!(!validate(&e, &desc1(2, 70, 16, 16, 0), &sf, 1));
        // Row bytes: 32 * 4 > 64.
        assert!(!validate(&e, &desc1(2, 70, 32, 16, 0), &sf, 0));
        // Alignment: 64 & (128 - 1).
        let caps = DeviceCaps::clone(&e.device.caps());
        let mut aligned = caps.clone();
        aligned.limits.iosurface_read_only_texture_alignment_bytes = 128;
        e.device.set_caps(aligned);
        assert!(!validate(&e, &desc1(2, 70, 16, 16, 0), &sf, 0));
        e.device.set_caps(caps);
        // Rows: 32 > 16.
        assert!(!validate(&e, &desc1(2, 70, 16, 32, 0), &sf, 0));
        // Overflow of alignedHeight * arrayLength.
        e.p.script().push(
            "MTLGetTextureLevelInfoForDeviceWithOptions()",
            Step::value(crate::platform::types::TextureLevelInfo {
                aligned_height: u64::MAX,
                bytes_per_pixel: 4,
                bytes_per_row: 64,
                ..Default::default()
            }),
        );
        let mut s = desc1(2, 70, 16, 16, 0);
        s.array_length = 2;
        assert!(!validate(&e, &s, &sf, 0));
        // Pixel size.
        e.p.script()
            .push("IOSurfaceAllowsPixelSizeCasting", Step::value(false));
        e.p.script()
            .push("IOSurfaceGetBytesPerElement", Step::value(2u64));
        assert!(!validate(&e, &desc1(2, 70, 16, 16, 0), &sf, 0));
        // Mips.
        let mut s = desc1(2, 70, 16, 16, 0);
        s.mipmap_level_count = 2;
        assert!(!validate(&e, &s, &sf, 0));
        // Depth.
        let mut s = desc1(2, 70, 16, 16, 0);
        s.depth = 2;
        assert!(!validate(&e, &s, &sf, 0));
        // TextureBuffer.
        assert!(!validate(&e, &desc1(9, 70, 16, 16, 0), &sf, 0));
    });
    const R: &str = "iosurface texture refused: ";
    assert_eq!(
        lines,
        vec![
            format!("{R}plane 1 of a surface with no planes"),
            format!("{R}128 bytes per row, more than the surface's 64"),
            format!("{R}the surface's 0x40 bytes per row are not aligned to 0x80"),
            format!("{R}32 rows, more than the surface's 16"),
            format!("{R}18446744073709551615 rows times 2 array slices overflows"),
            format!("{R}4 bytes per pixel, not the surface's"),
            format!("{R}2 mip levels, where only 1 is allowed"),
            format!("{R}depth 2, where only 1 is allowed"),
            format!("{R}a texture buffer"),
        ]
    );
}

/// Only TextureBuffer is refused (a 3D or cube passes), and read-only usage picks its alignment.
#[test]
fn iosurface_validation_quirks() {
    let e = TestEnv::new();
    let sf = surface(&e);
    for t in [3u8, 5, 7] {
        assert!(validate(&e, &desc1(t, 70, 16, 16, 0), &sf, 0), "type {t}");
    }
    // usage & 7 == 1 (ShaderRead): the read-only alignment; 3 (read|write): the other. The
    // surface's 64 bytes per row are aligned to 16, not to 128.
    let read_write = {
        let mut s = desc1(2, 70, 16, 16, 0);
        s.bits |= 3 << 8;
        s
    };
    for (read_only_alignment, other_alignment) in [(16, 128), (128, 16)] {
        let mut caps = DeviceCaps::clone(&e.device.caps());
        caps.limits.iosurface_read_only_texture_alignment_bytes = read_only_alignment;
        caps.limits.iosurface_texture_alignment_bytes = other_alignment;
        e.device.set_caps(caps);
        let ((read_only, other), _) = logged(|| {
            (
                validate(&e, &desc1(2, 70, 16, 16, 0), &sf, 0),
                validate(&e, &read_write, &sf, 0),
            )
        });
        assert_eq!(
            (read_only, other),
            (read_only_alignment == 16, other_alignment == 16)
        );
    }
    // A refused height asks IOSurfaceGetHeight a second time, for the line, whether or not the
    // line is written.
    let e = TestEnv::new();
    let sf = surface(&e);
    e.p.call_log().clear();
    assert!(!validate(&e, &desc1(2, 70, 16, 32, 0), &sf, 0));
    assert_eq!(e.calls("IOSurfaceGetHeight").len(), 2);
}

// ---- paged textures

fn normal_op_v1(s: &SerializedTextureDescriptor) -> Bytes {
    Bytes::new().header(1).u32(0).raw(s.as_bytes())
}

#[test]
fn normal_single_plane() {
    let e = TestEnv::new();
    let b = normal_prefix(2, 1).raw(&normal_op_v1(&desc1(2, 70, 8, 8, 0)).0);
    let r = ok(create_normal_texture(&e, &entry(2), b.cursor(), None));
    assert_eq!(r.class, "createTextureResourceWithTask:");
    assert_eq!(r.object_type, 2);
    let (rec, len, paging) = r.record.clone().unwrap();
    assert_eq!(len, 2 * 0x24 + 0x18);
    assert_eq!(paging.page, 9);
    assert_eq!(paging.length, 0x400);
    assert_eq!(rec.header, dimension(2));
    assert_eq!(rec.mips.len(), 2);
    let mut raw = rec.header.as_bytes().to_vec();
    raw.extend_from_slice(rec.mips.as_bytes());
    assert_eq!(&raw[0x18..0x18 + 0x48], &normal_prefix(2, 1).0[0x24..]);
    assert!(e.p.call_log().contains_in_order(&[
        "newTextureWithDescriptor:",
        "createTextureResourceWithTask:"
    ]));
}

#[test]
fn normal_dual_plane() {
    let e = TestEnv::new();
    dual_plane(&e, false);
    let b = normal_prefix(1, 2).raw(&normal_op_v1(&desc1(2, 70, 8, 8, 0)).0);
    assert_eq!(
        err(create_normal_texture(&e, &entry(12), b.cursor(), None)),
        "two-plane texture refused: the task's features have no dual-plane textures"
    );
    assert_eq!(e.calls("features").len(), 1, "the features are read first");
    dual_plane(&e, true);
    let r = ok(create_normal_texture(&e, &entry(12), b.cursor(), None));
    let (rec, len, _) = r.record.clone().unwrap();
    // (1 * 0x24 + 0x18) * 2 bytes; plane 1's header is at mips[1].
    assert_eq!(len, 0x78);
    let mut raw = rec.header.as_bytes().to_vec();
    raw.extend_from_slice(rec.mips.as_bytes());
    assert_eq!(&raw[..0x78], &normal_prefix(1, 2).0[0xc..]);
    assert_eq!(
        rec.mips.len(),
        3,
        "0x60 bytes in 0x24-byte mips, the last padded"
    );
}

#[test]
fn normal_read_failures() {
    let e = TestEnv::new();
    let full = normal_prefix(1, 1).raw(&normal_op_v1(&desc1(2, 70, 8, 8, 0)).0);
    let cut = |n: usize| cur(&full.0[..n]);
    let msg = |n: usize| err(create_normal_texture(&e, &entry(3), cut(n), None));
    assert!(msg(11).starts_with("paged texture record not readable: "));
    assert!(msg(12 + 0x17).starts_with("texture dimension record not readable: "));
    assert!(
        msg(12 + 0x18 + 0x23).starts_with("rest of the texture dimension record not readable: ")
    );
    assert!(msg(12 + 0x3c + 7).starts_with("command header not readable: "));
    assert!(msg(12 + 0x3c + 8 + 0x23).starts_with("texture descriptor command not readable: "));
    assert_eq!(e.calls("newTextureWithDescriptor:").len(), 0);
}

#[test]
fn normal_refusals() {
    let e = TestEnv::new();
    let with = |b: Bytes| normal_prefix(0, 1).raw(&b.0).cursor();
    // Unexpected command type.
    let b = Bytes::new()
        .header(2)
        .u32(0)
        .raw(desc1(2, 70, 8, 8, 0).as_bytes());
    assert_eq!(
        err(create_normal_texture(&e, &entry(2), with(b), None)),
        "unknown command type 2 for a paged texture"
    );
    // Memoryless.
    let b = normal_op_v1(&desc1(2, 70, 8, 8, 0x30));
    assert_eq!(
        err(create_normal_texture(&e, &entry(2), with(b.clone()), None)),
        "texture refused: storage mode 3 is memoryless"
    );
    // Metal's nil.
    e.p.script().push("newTextureWithDescriptor:", Step::Nil);
    let b = normal_op_v1(&desc1(2, 70, 8, 8, 0));
    assert_eq!(
        err(create_normal_texture(&e, &entry(2), with(b.clone()), None)),
        "Metal made no texture"
    );
    // The resource class's nil.
    e.p.script()
        .push("createTextureResourceWithTask:", Step::Nil);
    assert_eq!(
        err(create_normal_texture(&e, &entry(2), with(b), None)),
        "no host resource made for the paged texture"
    );
}

/// Forced at field0, verified against the resourceID, written back index then ID.
#[test]
fn normal_placement() {
    let e = TestEnv::new();
    let b = normal_prefix(0, 1).raw(
        &Bytes::new()
            .header(52)
            .u32(0)
            .raw(desc2(2, 70, 8, 8, 0).as_bytes())
            .u8(0)
            .0,
    );
    // Placed where the guest asked: forced, verified, written back.
    let (_m, g) = placement(12, 0x5000);
    next_texture_id(&e, 0x5000);
    let r = ok(create_normal_texture(&e, &entry(3), b.cursor(), Some(g)));
    let t = r.texture.as_ref().unwrap();
    assert_eq!(t.resource_index(), 12);
    assert_eq!(read_placement(g), (12, t.gpu_resource_id()));
    // Placed elsewhere: logged, released, refused; the entry is left alone.
    let (_m, g) = placement(12, 0x5000);
    next_texture_id(&e, 0x6000);
    let (r, lines) = logged(|| create_normal_texture(&e, &entry(3), b.cursor(), Some(g)));
    assert_eq!(err(r), "texture not placed where the guest asked");
    assert_eq!(read_placement(g), (12, 0x5000));
    assert_eq!(lines, vec![misplaced(0x5000, 0x6000)]);
    // field0 0: not forced, not verified, still written back.
    let (_m, g) = placement(0, 0x5000);
    let r = ok(create_normal_texture(&e, &entry(3), b.cursor(), Some(g)));
    let t = r.texture.as_ref().unwrap();
    assert!(!made_with(t).force_resource_index);
    assert_eq!(read_placement(g), (t.resource_index(), t.gpu_resource_id()));
}

#[test]
fn normal_metal_raise_propagates() {
    let e = TestEnv::new();
    let b = normal_prefix(0, 1).raw(&normal_op_v1(&desc1(2, 70, 8, 8, 0)).0);
    e.p.script().push(
        "newTextureWithDescriptor:",
        Step::raise("NSInvalidArgumentException", "bad"),
    );
    assert!(create_normal_texture(&e, &entry(2), b.cursor(), None).is_err());
    assert_eq!(e.calls("createTextureResourceWithTask:").len(), 0);
}

// ---- the deserializer's textures

#[test]
fn serializer_header_and_type() {
    let e = TestEnv::new();
    assert!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            cur(&[0; 7]),
            None
        ))
        .starts_with("command header not readable: ")
    );
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            Bytes::new().header(12).cursor(),
            None
        )),
        "unknown command type 12 for a child texture"
    );
}

/// The memoryless texture is not checked for nil.
#[test]
fn serializer_memoryless() {
    let e = TestEnv::new();
    let op = |s: &SerializedTextureDescriptor| Bytes::new().header(1).u32(0).raw(s.as_bytes());
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op(&desc1(2, 70, 8, 8, 0x20)).cursor(),
            None
        )),
        "memoryless texture refused: its storage mode is not memoryless"
    );
    assert!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            cur(&op(&desc1(2, 70, 8, 8, 0x30)).0[..20]),
            None
        ))
        .starts_with("texture descriptor command not readable: ")
    );
    let r = ok(create_serializer_texture(
        Hold::NONE,
        &e,
        op(&desc1(2, 70, 8, 8, 0x30)).cursor(),
        None,
    ));
    assert_eq!(r.class, "new_discardable_serializer_texture_resource");
    assert!(r.texture.is_some());
    // A nil texture reaches the resource.
    e.p.script().push("newTextureWithDescriptor:", Step::Nil);
    let r = ok(create_serializer_texture(
        Hold::NONE,
        &e,
        op(&desc1(2, 70, 8, 8, 0x30)).cursor(),
        None,
    ));
    assert!(r.texture.is_none());
    let c = e.calls("new_discardable_serializer_texture_resource");
    assert_eq!(c.last().unwrap().args, vec![Arg::Obj(None)]);
    // The resource's nil.
    e.p.script()
        .push("new_discardable_serializer_texture_resource", Step::Nil);
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op(&desc1(2, 70, 8, 8, 0x30)).cursor(),
            None
        )),
        "no host resource made for the memoryless texture"
    );
    // v2, placed: written back through getTexture (index, then ID).
    let b = Bytes::new()
        .header(52)
        .u32(0)
        .raw(desc2(2, 70, 8, 8, 0x30).as_bytes())
        .u8(0);
    let (_m, g) = placement(0, 0);
    let r = ok(create_serializer_texture(
        Hold::NONE,
        &e,
        b.cursor(),
        Some(g),
    ));
    let t = r.texture.as_ref().unwrap();
    assert_eq!(read_placement(g), (t.resource_index(), t.gpu_resource_id()));
    assert_eq!(e.calls("getTexture").len(), 1);
    // A placed nil texture writes back zeros.
    e.p.script().push("newTextureWithDescriptor:", Step::Nil);
    let (_m, g) = placement(0, 0x77);
    ok(create_serializer_texture(
        Hold::NONE,
        &e,
        b.cursor(),
        Some(g),
    ));
    assert_eq!(read_placement(g), (0, 0));
}

fn base(e: &TestEnv, reference: u32) -> Fk<kind::Texture> {
    let t = e.texture();
    let tt = t.clone_ref();
    e.object(reference, Ok(Some(TestEnv::res(|r| r.texture = Some(tt)))));
    t
}

#[test]
fn views_pixel_format() {
    let e = TestEnv::new();
    let bt = base(&e, 4);
    let op = Bytes::new().header(7).u32(0).u32(4).u16(80).zeros(2);
    let r = ok(create_serializer_texture(Hold::NONE, &e, op.cursor(), None));
    assert_eq!(r.class, "newChildTextureResourceWithTexture:");
    let view = e.calls("newTextureViewWithPixelFormat:");
    assert_eq!(view[0].obj, bt.id());
    assert_eq!(view[0].args, vec![Arg::U64(80)]);
    // Placed: the index variant, then verification and write-back.
    let (_m, g) = placement(6, 0x900);
    next_texture_id(&e, 0x900);
    let r = ok(create_serializer_texture(
        Hold::NONE,
        &e,
        op.cursor(),
        Some(g),
    ));
    let v = e.calls("newTextureViewWithPixelFormat:resourceIndex:");
    assert_eq!(v[0].args, vec![Arg::U64(80), Arg::U64(6)]);
    let t = r.texture.as_ref().unwrap();
    assert_eq!(read_placement(g), (t.resource_index(), t.gpu_resource_id()));
    // Short command.
    assert!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            cur(&op.0[..8 + 11]),
            None
        ))
        .starts_with("texture view command not readable: ")
    );
}

/// A base of reference 0 is nil: no view, no child.
/// A nil view is not checked, and reaches the child resource.
#[test]
fn views_bases_and_nil_views() {
    let e = TestEnv::new();
    e.object(4, Err("boom".to_owned()));
    let op = |r: u32| Bytes::new().header(7).u32(0).u32(r).u16(80).zeros(2);
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op(4).cursor(),
            None
        )),
        "no base texture at reference 4: boom"
    );
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op(0).cursor(),
            None
        )),
        "no child resource made for the texture"
    );
    assert_eq!(e.calls("newChildTextureResourceWithTexture:").len(), 0);
    base(&e, 5);
    e.p.script()
        .push("newTextureViewWithPixelFormat:", Step::Nil);
    let r = ok(create_serializer_texture(
        Hold::NONE,
        &e,
        op(5).cursor(),
        None,
    ));
    assert!(r.texture.is_none());
    e.p.script()
        .push("newChildTextureResourceWithTexture:", Step::Nil);
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op(5).cursor(),
            None
        )),
        "no child resource made for the texture"
    );
    // A nil view with a placement: gpuResourceID is 0 against the entry's ID.
    e.p.script()
        .push("newTextureViewWithPixelFormat:resourceIndex:", Step::Nil);
    let (_m, g) = placement(3, 0x10);
    let (r, lines) = logged(|| create_serializer_texture(Hold::NONE, &e, op(5).cursor(), Some(g)));
    assert_eq!(err(r), "texture not placed where the guest asked");
    assert_eq!(lines, vec![misplaced(0x10, 0)]);
}

#[test]
fn views_texture_type_and_swizzle() {
    let e = TestEnv::new();
    base(&e, 4);
    let tt = Bytes::new()
        .u32(0)
        .u32(4)
        .u16(80)
        .u16(3)
        .u64(1)
        .u64(2)
        .u64(0)
        .u64(4);
    let r = ok(create_serializer_texture(
        Hold::NONE,
        &e,
        Bytes::new().header(8).raw(&tt.0).cursor(),
        None,
    ));
    assert!(r.texture.is_some());
    let c = e.calls("newTextureViewWithPixelFormat:textureType:levels:slices:");
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].args[..2], [Arg::U64(80), Arg::U64(3)]);
    let (_m, g) = placement(2, 0);
    next_texture_id(&e, 0);
    ok(create_serializer_texture(
        Hold::NONE,
        &e,
        Bytes::new().header(8).raw(&tt.0).cursor(),
        Some(g),
    ));
    assert_eq!(
        e.calls("newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:")
            .len(),
        1
    );
    // Swizzle: raw channels, unvalidated.
    let sw = tt.clone().raw(&[9, 1, 2, 3]);
    ok(create_serializer_texture(
        Hold::NONE,
        &e,
        Bytes::new().header(27).raw(&sw.0).cursor(),
        None,
    ));
    let c = e.calls("newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:");
    assert_eq!(c.len(), 1);
    assert_eq!(
        c[0].args[4],
        Arg::val(&Swizzle {
            red: 9,
            green: 1,
            blue: 2,
            alpha: 3
        })
    );
    let (_m, g) = placement(2, 0);
    next_texture_id(&e, 0);
    ok(create_serializer_texture(
        Hold::NONE,
        &e,
        Bytes::new().header(27).raw(&sw.0).cursor(),
        Some(g),
    ));
    assert_eq!(
        e.calls("newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:")
            .len(),
        1
    );
    // Short commands.
    assert!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            Bytes::new().header(27).raw(&tt.0).cursor(),
            None
        ))
        .starts_with("texture view command not readable: ")
    );
}

/// A view loads the slot's index once, and Metal is asked for the view at the index tested. A
/// guest rewriting the index after that load does not move the view; only the placement check,
/// which loads the index again, sees the new value.
#[test]
fn views_place_at_the_index_they_tested() {
    let e = TestEnv::new();
    base(&e, 4);
    let tt = Bytes::new()
        .u32(0)
        .u32(4)
        .u16(80)
        .u16(3)
        .u64(1)
        .u64(2)
        .u64(0)
        .u64(4);
    let views = [
        (
            Bytes::new().header(7).u32(0).u32(4).u16(80).zeros(2),
            "newTextureViewWithPixelFormat:resourceIndex:",
            "newTextureViewWithPixelFormat:",
        ),
        (
            Bytes::new().header(8).raw(&tt.0),
            "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:",
            "newTextureViewWithPixelFormat:textureType:levels:slices:",
        ),
        (
            Bytes::new()
                .header(27)
                .raw(&tt.clone().raw(&[9, 1, 2, 3]).0),
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:",
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:",
        ),
    ];
    for (op, placed, unplaced) in views {
        // 6 to 9 after the test: the view is asked for 6; the check loads 9 (nonzero), so
        // it verifies the resourceID; the write-back stores the view's index, 6.
        let (_m, g) = placement(6, 0x900);
        next_texture_id(&e, 0x900);
        let races = race_after_load(FIELD0, 1, 9);
        let r = ok(create_serializer_texture(
            Hold::NONE,
            &e,
            op.cursor(),
            Some(g),
        ));
        assert_eq!(
            races_pending(),
            0,
            "{placed}: the guest stored after the test"
        );
        drop(races);
        let c = e.calls(placed);
        assert_eq!(
            c.last().unwrap().args.last(),
            Some(&Arg::U64(6)),
            "{placed}"
        );
        let t = r.texture.as_ref().unwrap();
        assert_eq!(t.resource_index(), 6, "{placed}");
        assert_eq!(read_placement(g), (6, t.gpu_resource_id()), "{placed}");
        drop(r);
        // 6 to 0 after the test: still the index variant at 6, never the plain one; the
        // check loads 0 and verifies nothing; the write-back stores 6.
        let (_m, g) = placement(6, 0x900);
        let (n_placed, n_unplaced) = (e.calls(placed).len(), e.calls(unplaced).len());
        let races = race_after_load(FIELD0, 1, 0);
        let r = ok(create_serializer_texture(
            Hold::NONE,
            &e,
            op.cursor(),
            Some(g),
        ));
        assert_eq!(
            races_pending(),
            0,
            "{placed}: the guest stored after the test"
        );
        drop(races);
        assert_eq!(e.calls(unplaced).len(), n_unplaced, "{unplaced}");
        let c = e.calls(placed);
        assert_eq!(c.len(), n_placed + 1, "{placed}");
        assert_eq!(
            c.last().unwrap().args.last(),
            Some(&Arg::U64(6)),
            "{placed}"
        );
        let t = r.texture.as_ref().unwrap();
        assert_eq!(read_placement(g), (6, t.gpu_resource_id()), "{placed}");
    }
}

/// The misplaced line shows the slot's ID the check compared: the ID is loaded once, after
/// Metal's answer, and only the texture's ID is asked again. A guest rewriting the slot after
/// the compare does not reach the line, and the refusal leaves the guest's value in the slot.
#[test]
fn misplaced_logs_the_resource_id_it_compared() {
    let e = TestEnv::new();
    // A paged texture's descriptor texture.
    let b = normal_prefix(0, 1).raw(
        &Bytes::new()
            .header(52)
            .u32(0)
            .raw(desc2(2, 70, 8, 8, 0).as_bytes())
            .u8(0)
            .0,
    );
    let (_m, g) = placement(12, 0x5000);
    next_texture_id(&e, 0x6000);
    let races = race_after_load(RESOURCE_ID, 1, 0x7777);
    let (r, lines) = logged(|| create_normal_texture(&e, &entry(3), b.cursor(), Some(g)));
    assert_eq!(err(r), "texture not placed where the guest asked");
    assert_eq!(races_pending(), 0, "the guest stored after the compare");
    drop(races);
    assert_eq!(read_placement(g), (12, 0x7777));
    assert_eq!(lines, vec![misplaced(0x5000, 0x6000)]);
    // A pixel-format view.
    base(&e, 4);
    let op = Bytes::new().header(7).u32(0).u32(4).u16(80).zeros(2);
    let (_m, g) = placement(6, 0x900);
    next_texture_id(&e, 0x901);
    let races = race_after_load(RESOURCE_ID, 1, 0x7777);
    let (r, lines) = logged(|| create_serializer_texture(Hold::NONE, &e, op.cursor(), Some(g)));
    assert_eq!(err(r), "texture not placed where the guest asked");
    assert_eq!(races_pending(), 0, "the guest stored after the compare");
    drop(races);
    assert_eq!(read_placement(g), (6, 0x7777));
    assert_eq!(lines, vec![misplaced(0x900, 0x901)]);
}

#[test]
fn serializer_over_buffer() {
    let e = TestEnv::new();
    let buf = e.buffer(0x10);
    let bb = buf.clone_ref();
    e.object(6, Ok(Some(TestEnv::res(|r| r.buffer = Some(bb)))));
    let op = |t: u8| {
        Bytes::new()
            .header(9)
            .u32(0)
            .u32(6)
            .u64(0x40)
            .u64(0x100)
            .raw(desc1(t, 70, 8, 8, 0).as_bytes())
    };
    // 2D: newTextureWithDescriptor:offset:bytesPerRow:, the buffer's options.
    let r = ok(create_serializer_texture(
        Hold::NONE,
        &e,
        op(2).cursor(),
        None,
    ));
    assert!(r.texture.is_some());
    let c = e.calls("newTextureWithDescriptor:offset:bytesPerRow:");
    assert_eq!(c[0].obj, buf.id());
    assert_eq!(c[0].args[1..], [Arg::U64(0x40), Arg::U64(0x100)]);
    let so = e.calls("setResourceOptions:");
    assert_eq!(so.last().unwrap().args, vec![Arg::U64(0x10)]);
    // 2D array: linear, bytesPerImage 0.
    ok(create_serializer_texture(
        Hold::NONE,
        &e,
        op(3).cursor(),
        None,
    ));
    let c = e.calls("newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:");
    assert_eq!(
        c[0].args[1..],
        [Arg::U64(0x40), Arg::U64(0x100), Arg::U64(0)]
    );
    // v2 (55), natural alignment: 0x40 bytes.
    let b2 = Bytes::new()
        .header(55)
        .u32(0)
        .u32(6)
        .u64(0x40)
        .u64(0x100)
        .raw(desc2(2, 70, 8, 8, 0).as_bytes())
        .u8(0);
    ok(create_serializer_texture(Hold::NONE, &e, b2.cursor(), None));
    assert!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            cur(&b2.0[..0x47]),
            None
        ))
        .starts_with("buffer texture command not readable: ")
    );
    // Base errors and nil buffers.
    e.object(7, Err("gone".to_owned()));
    let op7 = Bytes::new()
        .header(9)
        .u32(0)
        .u32(7)
        .u64(0)
        .u64(0)
        .raw(desc1(2, 70, 8, 8, 0).as_bytes());
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op7.cursor(),
            None
        )),
        "no base buffer at reference 7: gone"
    );
    let op0 = Bytes::new()
        .header(9)
        .u32(0)
        .u32(0)
        .u64(0)
        .u64(0)
        .raw(desc1(2, 70, 8, 8, 0).as_bytes());
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op0.cursor(),
            None
        )),
        "no child resource made for the texture"
    );
    // A misplaced texture.
    let (_m, g) = placement(1, 5);
    next_texture_id(&e, 6);
    let (r, lines) = logged(|| create_serializer_texture(Hold::NONE, &e, op(2).cursor(), Some(g)));
    assert_eq!(err(r), "texture not placed where the guest asked");
    assert_eq!(lines, vec![misplaced(5, 6)]);
}

#[test]
fn serializer_in_heap() {
    let e = TestEnv::new();
    let heap = e.heap(0x20);
    let hh = heap.clone_ref();
    e.object(8, Ok(Some(TestEnv::res(|r| r.heap = Some(hh)))));
    let op = Bytes::new()
        .header(21)
        .u32(0)
        .u32(8)
        .raw(desc1(2, 70, 8, 8, 0).as_bytes())
        .u32(0xdead)
        .u64(0x2000);
    let r = ok(create_serializer_texture(Hold::NONE, &e, op.cursor(), None));
    assert!(r.texture.is_some());
    let c = e.calls("newTextureWithDescriptor:offset:");
    assert_eq!(c[0].obj, heap.id());
    assert_eq!(c[0].args[1], Arg::U64(0x2000));
    assert_eq!(
        e.calls("setResourceOptions:").last().unwrap().args,
        vec![Arg::U64(0x20)]
    );
    // v2 (56): the offset at 0x34.
    let op2 = Bytes::new()
        .header(56)
        .u32(0)
        .u32(8)
        .raw(desc2(2, 70, 8, 8, 0).as_bytes())
        .u8(0)
        .u32(0)
        .u64(0x3000);
    ok(create_serializer_texture(
        Hold::NONE,
        &e,
        op2.cursor(),
        None,
    ));
    assert_eq!(
        e.calls("newTextureWithDescriptor:offset:")[1].args[1],
        Arg::U64(0x3000)
    );
    assert!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            cur(&op2.0[..0x43]),
            None
        ))
        .starts_with("heap texture command not readable: ")
    );
    // A nil texture is refused here.
    e.p.script()
        .push("newTextureWithDescriptor:offset:", Step::Nil);
    assert_eq!(
        err(create_serializer_texture(Hold::NONE, &e, op.cursor(), None)),
        "Metal made no texture in the heap"
    );
    // Reference 0: nil heap, nil texture.
    let op0 = Bytes::new()
        .header(21)
        .u32(0)
        .u32(0)
        .raw(desc1(2, 70, 8, 8, 0).as_bytes())
        .u32(0)
        .u64(0);
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op0.cursor(),
            None
        )),
        "Metal made no texture in the heap"
    );
    e.object(9, Err("x".to_owned()));
    let op9 = Bytes::new()
        .header(21)
        .u32(0)
        .u32(9)
        .raw(desc1(2, 70, 8, 8, 0).as_bytes())
        .u32(0)
        .u64(0);
    assert_eq!(
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            op9.cursor(),
            None
        )),
        "no base heap at reference 9: x"
    );
    // Misplaced.
    let (_m, g) = placement(1, 5);
    next_texture_id(&e, 6);
    let (r, lines) = logged(|| create_serializer_texture(Hold::NONE, &e, op.cursor(), Some(g)));
    assert_eq!(err(r), "texture not placed where the guest asked");
    assert_eq!(lines, vec![misplaced(5, 6)]);
    e.p.script()
        .push("newChildTextureResourceWithTexture:", Step::Nil);
    assert_eq!(
        err(create_serializer_texture(Hold::NONE, &e, op.cursor(), None)),
        "no child resource made for the texture"
    );
}

// ---- IOSurface textures, mapper refs

fn iosurface_op(ct: u32, s: &SerializedTextureDescriptor, plane: u16, rotation: u16) -> Bytes {
    let b = Bytes::new().header(ct).u32(0).raw(s.as_bytes()).u16(plane);
    if ct == 12 {
        b.zeros(2)
    } else {
        b.u16(rotation)
    }
}

fn mapper(e: &TestEnv, id: u64) -> Fk<kind::Surface> {
    let sf = surface(e);
    e.rec.provide(id, &sf);
    sf
}

#[test]
fn mapper_ref() {
    let e = TestEnv::new();
    let sf = mapper(&e, 0x33);
    let b = Bytes::new()
        .u64(0x33)
        .raw(&iosurface_op(12, &desc1(2, 70, 16, 16, 0), 0, 0).0);
    e.p.call_log().clear();
    let r = ok(create_mapper_ref_texture(&e, b.cursor(), None));
    assert_eq!(r.class, "new_mapper_ref_texture_resource");
    assert_eq!(r.value, 0x33);
    assert!(e.p.call_log().contains_in_order(&[
        "features",
        "retain_io_surface_with_mapping_id",
        "IOSurfaceGetPlaneCount",
        "newTextureWithDescriptor:iosurface:plane:",
        "new_mapper_ref_texture_resource",
    ]));
    let c = e.calls("newTextureWithDescriptor:iosurface:plane:");
    assert_eq!(c[0].args[1..], [Arg::Obj(Some(sf.id())), Arg::U64(0)]);
    let c = e.calls("new_mapper_ref_texture_resource");
    use crate::num::ToU64 as _;
    assert_eq!(
        c[0].args,
        vec![
            Arg::U64(0x33),
            Arg::Obj(Some(r.texture.as_ref().unwrap().id())),
            Arg::U64(sf.base_address().addr().to_u64()),
            Arg::U64(sf.alloc_size()),
        ]
    );
    assert_eq!(e.calls("release_io_surface_with_mapping_id").len(), 0);
}

#[test]
fn mapper_ref_refusals_release_the_surface() {
    let e = TestEnv::new();
    mapper(&e, 0x33);
    let good = iosurface_op(12, &desc1(2, 70, 16, 16, 0), 0, 0);
    let with = |b: &Bytes| Bytes::new().u64(0x33).raw(&b.0).cursor();
    assert!(
        err(create_mapper_ref_texture(&e, cur(&[0; 7]), None))
            .starts_with("mapper-ref texture record not readable: ")
    );
    assert!(
        err(create_mapper_ref_texture(
            &e,
            cur(&[0x33, 0, 0, 0, 0, 0, 0, 0, 1]),
            None
        ))
        .starts_with("command header not readable: ")
    );
    // No surface for the ID: nothing to release.
    let b = Bytes::new().u64(0x34).raw(&good.0);
    assert_eq!(
        err(create_mapper_ref_texture(&e, b.cursor(), None)),
        "no IOSurface for mapping 52"
    );
    assert_eq!(e.calls("release_io_surface_with_mapping_id").len(), 0);
    // Unexpected command, a failed maker, a failed resource: each releases.
    assert_eq!(
        err(create_mapper_ref_texture(
            &e,
            with(&Bytes::new().header(1)),
            None
        )),
        "unknown command type 1 for a mapper-ref texture"
    );
    assert_eq!(e.calls("release_io_surface_with_mapping_id").len(), 1);
    e.p.script()
        .push("newTextureWithDescriptor:iosurface:plane:", Step::Nil);
    assert_eq!(
        err(create_mapper_ref_texture(&e, with(&good), None)),
        "Metal made no IOSurface texture"
    );
    assert_eq!(e.calls("release_io_surface_with_mapping_id").len(), 2);
    e.p.script()
        .push("new_mapper_ref_texture_resource", Step::Nil);
    assert_eq!(
        err(create_mapper_ref_texture(&e, with(&good), None)),
        "no host resource made for the mapper-ref texture"
    );
    let rel = e.calls("release_io_surface_with_mapping_id");
    assert_eq!(rel.len(), 3);
    assert_eq!(rel[2].args[0], Arg::U64(0x33));
    assert!(matches!(rel[2].args[1], Arg::Obj(Some(_))));
}

/// (`retainIOSurfaceWithMappingID:` gives +1), (a raise in
/// `newTextureWithDescriptor:iosurface:plane:` propagates), (a raise in the initializer
/// propagates): only the non-raising exits release the surface, so a raise leaks its +1 and
/// sends no `releaseIOSurfaceWithMappingID:surface:`.
#[test]
fn mapper_ref_raise_leaks_the_surface() {
    let e = TestEnv::new();
    let sf = mapper(&e, 0x33);
    let good = iosurface_op(12, &desc1(2, 70, 16, 16, 0), 0, 0);
    let rotated = iosurface_op(47, &desc1(2, 70, 16, 16, 0), 0, 0);
    let base = sf.retain_count();
    for (n, op) in [(1i64, &good), (2, &rotated)] {
        e.p.script().push(
            "newTextureWithDescriptor:iosurface:plane:",
            Step::raise("NSInvalidArgumentException", "bad"),
        );
        let b = Bytes::new().u64(0x33).raw(&op.0);
        assert!(create_mapper_ref_texture(&e, b.cursor(), None).is_err());
        assert_eq!(sf.retain_count(), base + n, "Metal raise leaks the +1");
    }
    e.p.script().push(
        "new_mapper_ref_texture_resource",
        Step::raise("NSInvalidArgumentException", "bad"),
    );
    let b = Bytes::new().u64(0x33).raw(&good.0);
    assert!(create_mapper_ref_texture(&e, b.cursor(), None).is_err());
    assert_eq!(
        sf.retain_count(),
        base + 3,
        "initializer raise leaks the +1"
    );
    assert_eq!(e.calls("release_io_surface_with_mapping_id").len(), 0);
}

#[test]
fn iosurface_refusals() {
    let e = TestEnv::new();
    mapper(&e, 1);
    let run = |b: Bytes, g: Option<GuestPtr>| {
        err(create_mapper_ref_texture(
            &e,
            Bytes::new().u64(1).raw(&b.0).cursor(),
            g,
        ))
    };
    assert!(
        run(Bytes::new().header(12).u32(0), None)
            .starts_with("IOSurface texture command not readable: ")
    );
    assert_eq!(
        run(iosurface_op(12, &desc1(2, 70, 16, 16, 0x30), 0, 0), None),
        "IOSurface texture refused: storage mode 3 is memoryless"
    );
    assert_eq!(
        run(iosurface_op(12, &desc1(2, 70, 16, 16, 0), 1, 0), None),
        "IOSurface texture descriptor refused"
    );
    let (_m, g) = placement(4, 0x44);
    next_texture_id(&e, 0x45);
    let (r, lines) = logged(|| run(iosurface_op(12, &desc1(2, 70, 16, 16, 0), 0, 0), Some(g)));
    assert_eq!(r, "texture not placed where the guest asked");
    assert_eq!(lines, vec![misplaced(0x44, 0x45)]);
    // Placed right: forced at field0; no write-back for a mapper ref.
    let (_m, g) = placement(4, 0x44);
    next_texture_id(&e, 0x44);
    ok(create_mapper_ref_texture(
        &e,
        Bytes::new()
            .u64(1)
            .raw(&iosurface_op(12, &desc1(2, 70, 16, 16, 0), 0, 0).0)
            .cursor(),
        Some(g),
    ));
    assert_eq!(read_placement(g), (4, 0x44));
}

#[test]
fn iosurface_with_rotation() {
    let e = TestEnv::new();
    mapper(&e, 1);
    let with = |b: Bytes| Bytes::new().u64(1).raw(&b.0).cursor();
    // 47: rotation 1 swaps the validated size (16x8 against a 16x16 surface).
    let r = ok(create_mapper_ref_texture(
        &e,
        with(iosurface_op(47, &desc1(2, 70, 8, 16, 0), 0, 1)),
        None,
    ));
    assert_eq!(made_with(r.texture.as_ref().unwrap()).rotation, 1);
    drop(r);
    // 57: v2, plane at 0x2c, rotation at 0x2e.
    let op57 = Bytes::new()
        .header(57)
        .u32(0)
        .raw(desc2(2, 70, 16, 16, 0).as_bytes())
        .u8(0)
        .u16(0)
        .u16(2);
    let r = ok(create_mapper_ref_texture(&e, with(op57.clone()), None));
    assert_eq!(made_with(r.texture.as_ref().unwrap()).rotation, 2);
    drop(r);
    assert!(
        err(create_mapper_ref_texture(
            &e,
            with(Bytes(op57.0[..0x37].to_vec())),
            None
        ))
        .starts_with("IOSurface texture command not readable: ")
    );
    // Memoryless: the rotated forms give no storage mode.
    assert_eq!(
        err(create_mapper_ref_texture(
            &e,
            with(iosurface_op(47, &desc1(2, 70, 16, 16, 0x30), 0, 0)),
            None
        )),
        "IOSurface texture refused: its storage mode is memoryless"
    );
    assert_eq!(
        err(create_mapper_ref_texture(
            &e,
            with(iosurface_op(47, &desc1(2, 70, 16, 16, 0), 2, 0)),
            None
        )),
        "IOSurface texture descriptor refused"
    );
}

/// A rotation the descriptor did not keep is a death.
#[test]
fn iosurface_rotation_mismatch_dies() {
    let e = TestEnv::new();
    mapper(&e, 1);
    e.p.world()
        .on_new::<kind::TextureDescriptor>(|d| d.set_reported_rotation(7));
    let b = Bytes::new()
        .u64(1)
        .raw(&iosurface_op(47, &desc1(2, 70, 16, 16, 0), 0, 3).0);
    let d = catch_death(|| create_mapper_ref_texture(&e, b.cursor(), None));
    assert_eq!(d.err(), Some(DeathKind::RotationMismatch.site()));
}

// ---- shared textures

fn handle(e: &TestEnv, id: u32) {
    let t = new_texture_shareable(e);
    let h = t.new_shared_texture_handle().unwrap().unwrap();
    e.rec.provide(u64::from(id), &h);
}

fn new_texture_shareable(e: &TestEnv) -> Fk<kind::Texture> {
    crate::testkit::impls::new_texture(
        e.p.world(),
        crate::testkit::objects::TexDesc::default(),
        None,
        0,
        None,
        true,
    )
}

#[test]
fn shared_texture() {
    let e = TestEnv::new();
    dual_plane(&e, false);
    let b = Bytes::new().u32(5).header(49);
    assert_eq!(
        err(create_shared_texture(Hold::NONE, &e, b.cursor(), None)),
        "shared texture refused: the task's features have no shared textures"
    );
    dual_plane(&e, true);
    assert!(
        err(create_shared_texture(Hold::NONE, &e, cur(&[5, 0, 0]), None))
            .starts_with("shared texture record not readable: ")
    );
    assert!(
        err(create_shared_texture(
            Hold::NONE,
            &e,
            cur(&[5, 0, 0, 0, 49]),
            None
        ))
        .starts_with("command header not readable: ")
    );
    assert_eq!(
        err(create_shared_texture(Hold::NONE, &e, b.cursor(), None)),
        "no shared texture handle for backing 5"
    );
    handle(&e, 5);
    assert_eq!(
        err(create_shared_texture(
            Hold::NONE,
            &e,
            Bytes::new().u32(5).header(48).cursor(),
            None
        )),
        "unknown command type 48 for a shared texture"
    );
    e.p.script()
        .push("newSharedTextureWithHandle:withResourceIndex:", Step::Nil);
    assert_eq!(
        err(create_shared_texture(Hold::NONE, &e, b.cursor(), None)),
        "Metal made no shared texture"
    );
    e.p.script().push("new_shared_texture_resource", Step::Nil);
    assert_eq!(
        err(create_shared_texture(Hold::NONE, &e, b.cursor(), None)),
        "no host resource made for the shared texture"
    );
    let r = ok(create_shared_texture(Hold::NONE, &e, b.cursor(), None));
    assert_eq!(r.value, 5);
    assert_eq!(
        e.calls("newSharedTextureWithHandle:withResourceIndex:")
            .last()
            .unwrap()
            .args[1],
        Arg::U64(0)
    );
}

/// The task's handle goes with the call, on success and on a refusal.
#[test]
fn shared_texture_handle_goes_with_the_call() {
    let e = TestEnv::new();
    dual_plane(&e, true);
    let t = new_texture_shareable(&e);
    let h = t.new_shared_texture_handle().unwrap().unwrap();
    e.rec.provide(5, &h);
    let base = h.retain_count();
    for (command, succeeds) in [(49u32, true), (48, false)] {
        let r = create_shared_texture(
            Hold::NONE,
            &e,
            Bytes::new().u32(5).header(command).cursor(),
            None,
        )
        .expect("no raise");
        assert_eq!(r.is_ok(), succeeds);
        drop(r);
        assert_eq!(h.retain_count(), base, "command {command}");
    }
}

/// The handle's index is field0, and nothing checks the ID.
#[test]
fn shared_texture_placement() {
    let e = TestEnv::new();
    dual_plane(&e, true);
    handle(&e, 5);
    let (_m, g) = placement(11, 0xabc);
    let r = ok(create_shared_texture(
        Hold::NONE,
        &e,
        Bytes::new().u32(5).header(49).cursor(),
        Some(g),
    ));
    assert_eq!(
        e.calls("newSharedTextureWithHandle:withResourceIndex:")[0].args[1],
        Arg::U64(11)
    );
    let t = r.texture.as_ref().unwrap();
    assert_eq!(read_placement(g), (t.resource_index(), t.gpu_resource_id()));
}

fn backing_prefix(mips: u16, planes: u64) -> Bytes {
    let mut b = Bytes::new()
        .u64(0x800)
        .u32(3)
        .raw(dimension(mips).as_bytes());
    let rest = (u64::from(mips) * 0x24 + 0x18) * planes - 0x18;
    for i in 0..rest {
        b = b.u8(i.to_le_bytes()[0]);
    }
    b
}

#[test]
fn shared_backing() {
    let e = TestEnv::new();
    dual_plane(&e, true);
    let v1 = Bytes::new()
        .header(48)
        .u32(0)
        .raw(desc1(2, 70, 8, 8, 0).as_bytes());
    let b = backing_prefix(1, 1).raw(&v1.0);
    let r = ok(create_shared_texture_backing(
        &e,
        &entry(17),
        b.cursor(),
        None,
    ));
    assert_eq!(r.class, "createSharedTextureResourceWithTask:");
    assert_eq!(r.object_type, 17);
    let (_, len, paging) = r.record.clone().unwrap();
    assert_eq!((len, paging.page, paging.length), (0x3c, 3, 0x800));
    assert_eq!(e.calls("newSharedTextureWithDescriptor:").len(), 1);
    // v2, two planes, placed.
    let v2 = Bytes::new()
        .header(53)
        .u32(0)
        .raw(desc2(2, 70, 8, 8, 0).as_bytes())
        .u8(0);
    let (_m, g) = placement(0, 0);
    let r = ok(create_shared_texture_backing(
        &e,
        &entry(18),
        backing_prefix(1, 2).raw(&v2.0).cursor(),
        Some(g),
    ));
    assert_eq!(r.record.as_ref().unwrap().1, 0x78);
    let t = r.texture.as_ref().unwrap();
    assert_eq!(read_placement(g), (t.resource_index(), t.gpu_resource_id()));
}

#[test]
fn shared_backing_refusals() {
    let e = TestEnv::new();
    dual_plane(&e, false);
    let v1 = |o: u16| {
        Bytes::new()
            .header(48)
            .u32(0)
            .raw(desc1(2, 70, 8, 8, o).as_bytes())
    };
    let run = |t: u8, b: &[u8]| err(create_shared_texture_backing(&e, &entry(t), cur(b), None));
    assert_eq!(
        run(16, &[]),
        "object of type 16 is not a shared texture backing"
    );
    assert_eq!(
        run(18, &[]),
        "two-plane texture refused: the task's features have no dual-plane textures"
    );
    assert!(run(17, &[0; 11]).starts_with("shared texture backing record not readable: "));
    assert!(run(17, &[0; 12 + 0x17]).starts_with("texture dimension record not readable: "));
    assert_eq!(
        run(17, &backing_prefix(0, 1).0),
        "shared texture backing refused: 0 mip levels"
    );
    let full = backing_prefix(1, 1).raw(&v1(0).0);
    assert!(
        run(17, &full.0[..0x24 + 0x23])
            .starts_with("rest of the texture dimension record not readable: ")
    );
    assert!(run(17, &full.0[..0x48 + 7]).starts_with("command header not readable: "));
    assert_eq!(
        run(17, &backing_prefix(1, 1).header(1).0),
        "unknown command type 1 for a shared texture backing"
    );
    assert!(
        run(17, &full.0[..0x50 + 0x23]).starts_with("texture descriptor command not readable: ")
    );
    assert_eq!(
        run(17, &backing_prefix(1, 1).raw(&v1(0x30).0).0),
        "shared texture refused: storage mode 3 is memoryless"
    );
    e.p.script()
        .push("newSharedTextureWithDescriptor:", Step::Nil);
    assert_eq!(run(17, &full.0), "Metal made no shareable texture");
    let (_m, g) = placement(1, 2);
    next_texture_id(&e, 3);
    let (r, lines) =
        logged(|| create_shared_texture_backing(&e, &entry(17), full.cursor(), Some(g)));
    assert_eq!(err(r), "texture not placed where the guest asked");
    assert_eq!(read_placement(g), (1, 2));
    assert_eq!(lines, vec![misplaced(2, 3)]);
    e.p.script()
        .push("createSharedTextureResourceWithTask:", Step::Nil);
    assert_eq!(
        run(17, &full.0),
        "no host resource made for the shared texture backing"
    );
}

/// Every maker's placement check logs a misplaced texture, with the slot's ID and the texture's
/// (asked a second time).
#[test]
fn every_maker_logs_a_misplaced_texture() {
    let e = TestEnv::new();
    base(&e, 4);
    mapper(&e, 1);
    // Placed at index 1, ID `id`; Metal answers `id + 1` to both sends.
    let misplace = |id: u64| {
        next_texture_id(&e, id + 1);
        placement(1, id)
    };
    let tt = Bytes::new()
        .u32(0)
        .u32(4)
        .u16(80)
        .u16(3)
        .u64(1)
        .u64(2)
        .u64(0)
        .u64(4);
    let serializer = |b: Bytes, g: GuestPtr| {
        err(create_serializer_texture(
            Hold::NONE,
            &e,
            b.cursor(),
            Some(g),
        ))
    };
    let placed = "texture not placed where the guest asked";
    let ((), lines) = logged(|| {
        // A texture-type view.
        let (_m, g) = misplace(0x10);
        assert_eq!(serializer(Bytes::new().header(8).raw(&tt.0), g), placed);
        // A swizzle view.
        let (_m, g) = misplace(0x20);
        assert_eq!(
            serializer(Bytes::new().header(27).raw(&tt.0).raw(&[0, 1, 2, 3]), g),
            placed
        );
        // A memoryless descriptor texture.
        let (_m, g) = misplace(0x30);
        assert_eq!(
            serializer(
                Bytes::new()
                    .header(1)
                    .u32(0)
                    .raw(desc1(2, 70, 8, 8, 0x30).as_bytes()),
                g
            ),
            placed
        );
        // A rotated IOSurface texture: the copy.
        let (_m, g) = misplace(0x40);
        assert_eq!(
            err(create_mapper_ref_texture(
                &e,
                Bytes::new()
                    .u64(1)
                    .raw(&iosurface_op(47, &desc1(2, 70, 16, 16, 0), 0, 0).0)
                    .cursor(),
                Some(g),
            )),
            placed
        );
        // A second-form shared texture backing.
        dual_plane(&e, false);
        let (_m, g) = misplace(0x50);
        let v2 = Bytes::new()
            .header(53)
            .u32(0)
            .raw(desc2(2, 70, 8, 8, 0).as_bytes())
            .u8(0);
        assert_eq!(
            err(create_shared_texture_backing(
                &e,
                &entry(17),
                backing_prefix(1, 1).raw(&v2.0).cursor(),
                Some(g),
            )),
            placed
        );
    });
    assert_eq!(
        lines,
        vec![
            misplaced(0x10, 0x11),
            misplaced(0x20, 0x21),
            misplaced(0x30, 0x31),
            misplaced(0x40, 0x41),
            misplaced(0x50, 0x51),
        ]
    );
}

// ---- backings

#[test]
fn backing_ref() {
    let e = TestEnv::new();
    let sf = surface(&e);
    let s2 = sf.clone_ref();
    e.backings.lock().push((
        7,
        TestEnv::res(|r| {
            r.class = "backing";
            r.surface = Some(s2);
        }),
    ));
    let op = iosurface_op(12, &desc1(2, 70, 16, 16, 0), 0, 0);
    let b = Bytes::new().u32(7).u32(0).raw(&op.0);
    let r = ok(create_backing_ref_texture(
        Hold::NONE,
        &e,
        &entry(5),
        b.cursor(),
        None,
    ));
    assert_eq!(r.class, "new_ref_texture_resource");
    assert_eq!(r.object_type, 5);
    assert_eq!(r.plane, 0);
    assert_eq!(r.backing.as_ref().unwrap().class, "backing");
    assert!(e.p.call_log().contains_in_order(&[
        "backingForID:",
        "ioSurface",
        "newTextureWithDescriptor:iosurface:plane:"
    ]));
    assert_eq!(
        e.calls("newTextureWithDescriptor:iosurface:plane:")[0].args[1],
        Arg::Obj(Some(sf.id()))
    );
    // 47 and 57 take the surface the same way.
    ok(create_backing_ref_texture(
        Hold::NONE,
        &e,
        &entry(5),
        Bytes::new()
            .u32(7)
            .u32(0)
            .raw(&iosurface_op(47, &desc1(2, 70, 16, 16, 0), 0, 0).0)
            .cursor(),
        None,
    ));
    let op57 = Bytes::new()
        .header(57)
        .u32(0)
        .raw(desc2(2, 70, 16, 16, 0).as_bytes())
        .u8(0)
        .u16(0)
        .u16(0);
    ok(create_backing_ref_texture(
        Hold::NONE,
        &e,
        &entry(5),
        Bytes::new().u32(7).u32(0).raw(&op57.0).cursor(),
        None,
    ));
    assert_eq!(e.calls("ioSurface").len(), 3);
}

#[test]
fn backing_ref_refusals() {
    let e = TestEnv::new();
    let run = |b: Bytes| {
        err(create_backing_ref_texture(
            Hold::NONE,
            &e,
            &entry(5),
            b.cursor(),
            None,
        ))
    };
    assert!(run(Bytes::new().u32(7)).starts_with("ref texture record not readable: "));
    assert_eq!(
        run(Bytes::new().u32(7).u32(1).header(12)),
        "ref texture refused: its backing is in task 1, not task 0"
    );
    assert!(run(Bytes::new().u32(7).u32(0).u32(12)).starts_with("command header not readable: "));
    assert_eq!(
        run(Bytes::new().u32(7).u32(0).header(12)),
        "no backing with IOSurface ID 7"
    );
    let sf = surface(&e);
    e.backings
        .lock()
        .push((7, TestEnv::res(|r| r.surface = Some(sf))));
    assert_eq!(
        run(Bytes::new().u32(7).u32(0).header(1)),
        "unknown command type 1 for a ref texture"
    );
    assert_eq!(e.calls("ioSurface").len(), 0);
    assert!(
        run(Bytes::new().u32(7).u32(0).header(12))
            .starts_with("IOSurface texture command not readable: ")
    );
    let op = iosurface_op(12, &desc1(2, 70, 16, 16, 0), 0, 0);
    e.p.script().push("new_ref_texture_resource", Step::Nil);
    assert_eq!(
        run(Bytes::new().u32(7).u32(0).raw(&op.0)),
        "no host resource made for the ref texture"
    );
}

/// The backing record is 0x14 bytes.
#[test]
fn backing() {
    let e = TestEnv::new();
    let rec = |planes: u8| {
        Bytes::new()
            .u64(0x1000)
            .u32(4)
            .u32(0x34323076)
            .u8(planes)
            .zeros(3)
    };
    let plane = |i: u32| Bytes::new().u32(i).u32(16).u32(16).u32(64 | (4 << 24));
    let b = rec(2).raw(&plane(0).0).raw(&plane(0x400).0);
    let r = ok(create_backing(&e, b.cursor()));
    assert_eq!(r.class, "new_backing_resource");
    let c = e.calls("new_backing_resource");
    assert_eq!(
        c[0].args,
        vec![
            Arg::val(&BackingDescriptor {
                length: 0x1000,
                page: 4,
                pixel_format: 0x34323076,
                plane_count: 2,
                unknown11: [0; 7]
            }),
            Arg::Bytes([plane(0).0, plane(0x400).0].concat()),
            Arg::U64(2),
        ]
    );
    // No planes.
    ok(create_backing(&e, rec(0).cursor()));
    let run = |b: Bytes| err(create_backing(&e, b.cursor()));
    assert!(run(Bytes(rec(1).0[..0x13].to_vec())).starts_with("backing record not readable: "));
    assert_eq!(run(rec(4)), "backing refused: 4 planes, more than 3");
    assert!(
        run(rec(3).raw(&plane(0).0).raw(&plane(0).0)).starts_with("backing planes not readable: ")
    );
    e.p.script().push("new_backing_resource", Step::Nil);
    assert_eq!(run(rec(0)), "no host resource made for the backing");
}

// ---- the dispatch

#[test]
fn dispatch_on_the_object_type() {
    let e = TestEnv::new();
    dual_plane(&e, false);
    // Each creator's first observable step on an empty cursor.
    let cases: [(u8, &str); 19] = [
        (1, "buffer record not readable: "),
        (15, "mapper-ref buffer record not readable: "),
        (14, "heap buffer record not readable: "),
        (2, "paged texture record not readable: "),
        (3, "paged texture record not readable: "),
        (
            12,
            "two-plane texture refused: the task's features have no dual-plane textures",
        ),
        (8, "command header not readable: "),
        (9, "command header not readable: "),
        (13, "heap record not readable: "),
        (11, "mapper-ref texture record not readable: "),
        (
            16,
            "shared texture refused: the task's features have no shared textures",
        ),
        (5, "ref texture record not readable: "),
        (17, "shared texture backing record not readable: "),
        (
            18,
            "two-plane texture refused: the task's features have no dual-plane textures",
        ),
        (4, "backing record not readable: "),
        (0, "object of type 0 is not a resource"),
        (6, "object of type 6 is not a resource"),
        (19, "object of type 19 is not a resource"),
        (255, "object of type 255 is not a resource"),
    ];
    for (t, want) in cases {
        let got = err(create_resource(
            Hold::NONE,
            &e,
            1,
            &entry(t),
            cur(&[]),
            None,
        ));
        assert!(got.starts_with(want), "type {t}: {got}");
    }
    // 20: the ICB creator (2.6's).
    let r = ok(create_resource(
        Hold::NONE,
        &e,
        9,
        &entry(20),
        cur(&[]),
        None,
    ));
    assert_eq!(r.class, "icb");
    assert_eq!(
        e.calls("createIndirectCommandBuffer")[0].args,
        vec![Arg::U32(9)]
    );
    // The length bits above the type are ignored.
    let ent = ObjectEntry {
        type_and_length: 0x1234_5600 | 13,
        offset: 0,
    };
    assert!(
        err(create_resource(Hold::NONE, &e, 1, &ent, cur(&[]), None))
            .starts_with("heap record not readable: ")
    );
}

// ---- through the manager

fn manager(e: &TestEnv) -> Arc<ResourceManager<Fp>> {
    use crate::tasks::object_lists::{ObjectListKind, ObjectLists};
    let task: Arc<dyn ResourceHost<Fp>> = e.rec.clone();
    let olm = ObjectLists::new(
        ObjectListKind::MemoryMap,
        0x10000,
        true,
        Arc::downgrade(&task),
    );
    ResourceManager::new(
        e.p.clone(),
        Some(e.device.clone_ref()),
        Arc::downgrade(&task),
        olm,
        false,
    )
}

#[test]
fn manager_texture_descriptor_for_descriptor() {
    let e = TestEnv::new();
    let m = manager(&e);
    let d = m
        .texture_descriptor_for_descriptor(&desc1(2, 70, 4, 4, 0))
        .unwrap()
        .unwrap();
    let pixel_format_then_options = [
        ("setPixelFormat:", vec![Arg::U64(70)]),
        ("setResourceOptions:", vec![Arg::U64(0)]),
    ];
    assert_eq!(ordered_on(&e, d.id()), pixel_format_then_options);
    let made = |d: &<Fp as Platform>::TextureDescriptor| {
        made_with(&e.device.new_texture(OnFault::Die, d).unwrap().unwrap())
    };
    let t = made(&d);
    assert_eq!((t.width, t.height, t.usage), (4, 4, 1));
    assert!(!t.force_resource_index, "index 0: none forced");
    let d = m
        .texture_descriptor_for_descriptor2(&desc2(2, 70, 4, 4, 0))
        .unwrap()
        .unwrap();
    assert_eq!(ordered_on(&e, d.id()), pixel_format_then_options);
    let t = made(&d);
    assert_eq!(
        (t.swizzle, t.write_swizzle_enabled, t.force_resource_index),
        (
            Swizzle {
                red: 2,
                green: 3,
                blue: 4,
                alpha: 5
            },
            false,
            false
        )
    );
}

#[test]
fn manager_create_object() {
    let e = TestEnv::new();
    let m = manager(&e);
    let got =
        <Resources as ManagedKind<Fp>>::create_object(Hold::NONE, &m, 1, &entry(0), cur(&[]), None);
    assert_eq!(err(got), "object of type 0 is not a resource");
    let hdr = |ct: u32| Bytes::new().header(ct);
    let f = <crate::resources::manager::states::Fences as ManagedKind<Fp>>::create_object(
        Hold::NONE,
        &m,
        1,
        &entry(7),
        hdr(13).cursor(),
        None,
    );
    ok(f);
    let s = <crate::resources::manager::states::SamplerStates as ManagedKind<Fp>>::create_object(
        Hold::NONE,
        &m,
        1,
        &entry(7),
        hdr(3).raw(&[0; 0x1c]).cursor(),
        None,
    );
    ok(s);
    let d =
        <crate::resources::manager::states::DepthStencilStates as ManagedKind<Fp>>::create_object(
            Hold::NONE,
            &m,
            1,
            &entry(7),
            hdr(4).raw(&[0; 0x20]).cursor(),
            None,
        );
    ok(d);
    assert!(e.p.call_log().contains_in_order(&[
        "newFence",
        "newSamplerStateWithDescriptor:",
        "newDepthStencilStateWithDescriptor:"
    ]));
}

/// A nil memoryless texture makes a resource over nil.
#[test]
fn manager_a_nil_memoryless_texture() {
    let e = TestEnv::new();
    let m = manager(&e);
    let op = |s: &SerializedTextureDescriptor| Bytes::new().header(1).u32(0).raw(s.as_bytes());
    e.p.script().push("newTextureWithDescriptor:", Step::Nil);
    let got = create_serializer_texture(
        Hold::NONE,
        &*m,
        op(&desc1(2, 70, 8, 8, 0x30)).cursor(),
        None,
    );
    // Builds the resource over nil.
    let r = got.expect("no raise").expect("a resource over nil");
    assert!(r.get_texture().expect("no raise").is_none());
    match &r.kind {
        ResourceKind::SerializerTexture(s) => assert!(s.discardable()),
        _ => panic!("not a serializer texture"),
    }
}

/// A nil view makes a child over nil, discardable as its base.
#[test]
fn manager_a_nil_view() {
    use crate::resources::texture::views::SerializerTextureResource;
    let e = TestEnv::new();
    let m = manager(&e);
    for discardable in [false, true] {
        let base: Arc<Resource<Fp>> = SerializerTextureResource::new(None, discardable);
        let child = CreateEnv::new_child_texture_resource(&*m, &base, None)
            .expect("no raise")
            .expect("a child over nil");
        assert!(child.get_texture().expect("no raise").is_none());
        match &child.kind {
            ResourceKind::SerializerTexture(s) => assert_eq!(s.discardable(), discardable),
            _ => panic!("not a serializer texture"),
        }
    }
}
