//! An x86-64 VM's OpenCore image: the Mac it presents, and making the VM a new one.
//!
//! reims-util makes a VM's image when the VM is made (reims_util.rs). A newer reims-util's
//! image can carry what an older one's lacks, such as Lilu and ReimsVgpu.kext, which on macOS
//! 26 let WindowServer run a display wider than 2048 pixels and Chrome draw on the GPU
//! (docs/qemu.md, "x86-64 macOS guests: displays wider than 2048 pixels"). [`renew`] replaces a
//! VM's image with one the installed reims-util makes, presenting the same Mac, so macOS sees
//! no new machine. reims-ctl keeps no copy of that identity: it is read from the old image's
//! config.plist.
//!
//! An image is read as reims-util writes it: qcow2 (or raw), a GPT, and a FAT32 EFI system
//! partition holding EFI/OC/config.plist.

use crate::vm::{ImageFormat, Vm};
use crate::{paths, reims_util, runtime};
use anyhow::{Context, Result, bail, ensure};
use std::fs::File;
use std::path::Path;

/// The Mac an OpenCore image presents: its config.plist's PlatformInfo > Generic, which
/// reims-util takes as `--model`, `--serial`, `--mlb`, `--uuid` and `--rom`.
#[derive(Clone, Debug, PartialEq)]
pub struct Identity {
    pub model: String,
    pub serial: String,
    pub mlb: String,
    pub uuid: String,
    /// Six bytes, as 12 hex digits.
    pub rom: String,
}

/// An OpenCore image's config.plist.
pub struct Config(plist::Dictionary);

impl Config {
    pub fn read(path: &Path, format: ImageFormat) -> Result<Config> {
        let disk = Disk::open(path, format)?;
        let bytes = Fat32::open(&disk, esp_start(&disk)?)?
            .read_file("EFI/OC/config.plist")?
            .context("its EFI system partition has no EFI/OC/config.plist")?;
        Config::parse(&bytes)
    }

    fn parse(bytes: &[u8]) -> Result<Config> {
        match plist::Value::from_reader(std::io::Cursor::new(bytes)).context("read config.plist")? {
            plist::Value::Dictionary(d) => Ok(Config(d)),
            _ => bail!("config.plist holds no dictionary"),
        }
    }

    pub fn identity(&self) -> Result<Identity> {
        let generic = self
            .0
            .get("PlatformInfo")
            .and_then(plist::Value::as_dictionary)
            .and_then(|p| p.get("Generic"))
            .and_then(plist::Value::as_dictionary)
            .context("config.plist has no PlatformInfo > Generic")?;
        let text = |key: &str| {
            generic
                .get(key)
                .and_then(plist::Value::as_string)
                .filter(|s| !s.trim().is_empty())
                .map(String::from)
                .with_context(|| format!("config.plist's PlatformInfo > Generic has no {key}"))
        };
        let rom = generic
            .get("ROM")
            .and_then(plist::Value::as_data)
            .filter(|d| d.len() == 6)
            .context("config.plist's PlatformInfo > Generic has no six-byte ROM")?;
        Ok(Identity {
            model: text("SystemProductName")?,
            serial: text("SystemSerialNumber")?,
            mlb: text("MLB")?,
            uuid: text("SystemUUID")?,
            rom: rom.iter().map(|b| format!("{b:02X}")).collect(),
        })
    }

    /// Whether OpenCore loads ReimsVgpu.kext: an enabled entry of Kernel > Add names it.
    pub fn loads_reims_vgpu(&self) -> bool {
        self.0
            .get("Kernel")
            .and_then(plist::Value::as_dictionary)
            .and_then(|k| k.get("Add"))
            .and_then(plist::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(plist::Value::as_dictionary)
            .any(|e| {
                e.get("BundlePath").and_then(plist::Value::as_string) == Some("ReimsVgpu.kext")
                    && e.get("Enabled").and_then(plist::Value::as_boolean) == Some(true)
            })
    }
}

/// What to tell of an x86-64 VM whose OpenCore image does not load ReimsVgpu.kext, made by an
/// older reims-util; None where it does, or where the image cannot be read.
pub fn missing_kext(vm: &Vm) -> Option<String> {
    let q = vm.config.q35.as_ref()?;
    let config = Config::read(&vm.path(&q.opencore), q.opencore_format).ok()?;
    (!config.loads_reims_vgpu()).then(|| {
        format!(
            "Its OpenCore image has no ReimsVgpu.kext, which macOS 26 needs for a display wider \
             than 2048 pixels (WindowServer crashes on one without it) and for GPU-accelerated \
             Chrome. To make it a new image that has it, presenting the same Mac, press U in \
             reims-ctl or run reims-ctl update-opencore {}.",
            vm.name()
        )
    })
}

/// Replaces a stopped VM's OpenCore image with one the installed reims-util makes, presenting
/// the same Mac. The new image is made beside the old one, which stays as it was if that fails.
/// The VM keeps its OVMF: its variables hold its boot entries. Returns what happened.
pub fn renew(vm: &mut Vm, progress: &mut dyn FnMut(&str)) -> Result<String> {
    let name = vm.name().to_string();
    if runtime::running(vm) {
        bail!("{name} is running: stop it first");
    }
    let q = vm
        .config
        .q35
        .clone()
        .context("only x86-64 VMs boot OpenCore")?;
    let old = vm.path(&q.opencore);
    let id = Config::read(&old, q.opencore_format)
        .and_then(|c| c.identity())
        .with_context(|| format!("read the Mac {} presents", paths::tilde(&old)))?;
    let image = vm.dir.join("opencore.qcow2");
    let new = vm.dir.join("opencore.new.qcow2");
    if let Err(e) = reims_util::opencore_as(&new, &id, progress) {
        let _ = std::fs::remove_file(&new);
        return Err(e);
    }
    std::fs::rename(&new, &image).with_context(|| format!("replace {}", image.display()))?;
    if let Some(q) = vm.config.q35.as_mut() {
        q.opencore = "opencore.qcow2".into();
        q.opencore_format = ImageFormat::Qcow2;
    }
    vm.save()?;
    // An image the VM was given under another name, in its directory, is replaced too.
    if old != image && old.starts_with(&vm.dir) {
        let _ = std::fs::remove_file(&old);
    }
    let mut s = format!(
        "{name}: a new OpenCore image, presenting the same {} (serial {})",
        id.model, id.serial
    );
    if Config::read(&image, ImageFormat::Qcow2).is_ok_and(|c| !c.loads_reims_vgpu()) {
        s.push_str(
            "; the installed reims-util put no ReimsVgpu.kext in it: run reims-ctl update, \
             which installs the reims-util it needs, and make it again",
        );
    }
    Ok(s)
}

const SECTOR: u64 = 512;

#[cfg(unix)]
fn read_exact_at(file: &File, mut buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        offset += n as u64;
        buf = &mut buf[n..];
    }
    Ok(())
}

/// The EFI system partition's type GUID, C12A7328-F81F-11D2-BA4B-00A0C93EC93B, as GPT stores
/// it.
const ESP_TYPE: [u8; 16] = [
    0x28, 0x73, 0x2a, 0xc1, 0x1f, 0xf8, 0xd2, 0x11, 0xba, 0x4b, 0x00, 0xa0, 0xc9, 0x3e, 0xc9, 0x3b,
];

/// Where the GPT's EFI system partition starts, in bytes.
fn esp_start(disk: &Disk) -> Result<u64> {
    let mut header = [0u8; SECTOR as usize];
    disk.read(SECTOR, &mut header)?;
    ensure!(&header[..8] == b"EFI PART", "it has no GPT");
    let size = u64::from(le32(&header, 84));
    ensure!((128..=4096).contains(&size), "GPT entries of {size} bytes");
    let count = u64::from(le32(&header, 80).min(128));
    let mut entries = vec![0u8; (count * size) as usize];
    disk.read(le64(&header, 72) * SECTOR, &mut entries)?;
    entries
        .chunks_exact(size as usize)
        .find(|e| e[..16] == ESP_TYPE)
        .map(|e| le64(e, 32) * SECTOR)
        .context("its GPT has no EFI system partition")
}

/// An image's bytes as the guest sees them.
enum Disk {
    Raw { file: File, size: u64 },
    Qcow2(Qcow2),
}

impl Disk {
    fn open(path: &Path, format: ImageFormat) -> Result<Disk> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(match format {
            ImageFormat::Raw => Disk::Raw {
                size: file.metadata()?.len(),
                file,
            },
            ImageFormat::Qcow2 => Disk::Qcow2(Qcow2::open(file)?),
        })
    }

    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let size = match self {
            Disk::Raw { size, .. } => *size,
            Disk::Qcow2(q) => q.size,
        };
        ensure!(
            offset.saturating_add(buf.len() as u64) <= size,
            "a read at {offset:#x}, past the image's end"
        );
        match self {
            Disk::Raw { file, .. } => read_exact_at(file, buf, offset)?,
            Disk::Qcow2(q) => q.read(offset, buf)?,
        }
        Ok(())
    }
}

/// Bits 9 to 55 of a qcow2 L1 or L2 entry: where the L2 table or the cluster is in the file.
const QCOW2_OFFSET: u64 = 0x00ff_ffff_ffff_fe00;

/// A qcow2 image (QEMU's docs/interop/qcow2.txt) with no backing file: each guest cluster
/// found through the L1 and L2 tables, unallocated and zero clusters reading as zeros.
struct Qcow2 {
    file: File,
    cluster_bits: u32,
    size: u64,
    l1: Vec<u64>,
}

impl Qcow2 {
    fn open(file: File) -> Result<Qcow2> {
        let mut h = [0u8; 48];
        read_exact_at(&file, &mut h, 0)?;
        ensure!(&h[..4] == b"QFI\xfb", "not a qcow2 image");
        ensure!(be64(&h, 8) == 0, "a qcow2 image with a backing file");
        let cluster_bits = be32(&h, 20);
        ensure!(
            (9..=21).contains(&cluster_bits),
            "qcow2 clusters of 2^{cluster_bits} bytes"
        );
        let mut l1 = vec![0u8; be32(&h, 36).min(1 << 20) as usize * 8];
        read_exact_at(&file, &mut l1, be64(&h, 40))?;
        Ok(Qcow2 {
            file,
            cluster_bits,
            size: be64(&h, 24),
            l1: l1.as_chunks::<8>().0.iter().map(|e| be64(e, 0)).collect(),
        })
    }

    /// Where guest cluster `index` is in the file; None where it reads as zeros.
    fn cluster(&self, index: u64) -> Result<Option<u64>> {
        let per_table = 1u64 << (self.cluster_bits - 3);
        let l1 = self
            .l1
            .get((index / per_table) as usize)
            .copied()
            .unwrap_or(0);
        if l1 & QCOW2_OFFSET == 0 {
            return Ok(None);
        }
        let mut e = [0u8; 8];
        read_exact_at(
            &self.file,
            &mut e,
            (l1 & QCOW2_OFFSET) + index % per_table * 8,
        )?;
        let e = u64::from_be_bytes(e);
        ensure!(e & 1 << 62 == 0, "a qcow2 image with compressed clusters");
        // Bit 0: a zero cluster.
        Ok((e & 1 == 0 && e & QCOW2_OFFSET != 0).then_some(e & QCOW2_OFFSET))
    }

    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let cluster = 1u64 << self.cluster_bits;
        let mut done = 0;
        while done < buf.len() {
            let at = offset + done as u64;
            let within = at & (cluster - 1);
            let n = ((cluster - within) as usize).min(buf.len() - done);
            let part = &mut buf[done..done + n];
            match self.cluster(at >> self.cluster_bits)? {
                Some(c) => read_exact_at(&self.file, part, c + within)?,
                None => part.fill(0),
            }
            done += n;
        }
        Ok(())
    }
}

/// The most of a file or directory read: a config.plist is tens of KiB.
const MOST: u64 = 16 << 20;

/// A FAT32 volume (Microsoft's FAT specification, 1.03).
struct Fat32<'a> {
    disk: &'a Disk,
    /// Where its first FAT and its cluster 2 are in the image.
    fat: u64,
    data: u64,
    cluster_size: u64,
    root: u32,
}

impl<'a> Fat32<'a> {
    fn open(disk: &'a Disk, start: u64) -> Result<Fat32<'a>> {
        let mut b = [0u8; SECTOR as usize];
        disk.read(start, &mut b)?;
        let sector = u64::from(le16(&b, 11));
        let per_cluster = u64::from(b[13]);
        let reserved = u64::from(le16(&b, 14));
        let fat_sectors = u64::from(le32(&b, 36));
        ensure!(
            b[510..] == [0x55, 0xaa] && le16(&b, 22) == 0 && fat_sectors > 0,
            "its EFI system partition is not FAT32"
        );
        ensure!(
            sector >= 512 && per_cluster > 0,
            "a FAT32 volume of no clusters"
        );
        Ok(Fat32 {
            disk,
            fat: start + reserved * sector,
            data: start + (reserved + u64::from(b[16]) * fat_sectors) * sector,
            cluster_size: per_cluster * sector,
            root: le32(&b, 44),
        })
    }

    /// The file at `path`, `/` between its names, matched without regard to case as FAT
    /// matches them; None where there is none.
    fn read_file(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let mut dir = self.root;
        let mut names = path.split('/').peekable();
        while let Some(name) = names.next() {
            let bytes = self.read_chain(dir, MOST)?;
            let Some(e) = entries(&bytes).into_iter().find(|e| e.is(name)) else {
                return Ok(None);
            };
            if names.peek().is_some() {
                dir = e.cluster;
                continue;
            }
            let bytes = self.read_chain(e.cluster, u64::from(e.size).min(MOST))?;
            ensure!(
                bytes.len() as u64 == u64::from(e.size),
                "{path} is cut short"
            );
            return Ok(Some(bytes));
        }
        Ok(None)
    }

    /// The bytes of the clusters from `first`, at most `most`.
    fn read_chain(&self, first: u32, most: u64) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut c = first;
        while (2..0x0fff_fff8).contains(&c) && (out.len() as u64) < most {
            let at = out.len();
            out.resize(at + self.cluster_size as usize, 0);
            let offset = self.data + u64::from(c - 2) * self.cluster_size;
            self.disk.read(offset, &mut out[at..])?;
            let mut next = [0u8; 4];
            self.disk.read(self.fat + u64::from(c) * 4, &mut next)?;
            c = u32::from_le_bytes(next) & 0x0fff_ffff;
        }
        out.truncate(most as usize);
        Ok(out)
    }
}

/// A directory entry: its long name where one precedes it, and its 8.3 name.
struct Entry {
    long: Option<String>,
    short: String,
    cluster: u32,
    size: u32,
}

impl Entry {
    fn is(&self, name: &str) -> bool {
        self.long
            .as_deref()
            .is_some_and(|l| l.eq_ignore_ascii_case(name))
            || self.short.eq_ignore_ascii_case(name)
    }
}

/// A directory's entries, deleted ones and the volume label left out.
fn entries(bytes: &[u8]) -> Vec<Entry> {
    // The long name's parts, in the order they are stored: its last part first.
    let mut parts: Vec<[u16; 13]> = Vec::new();
    let mut out = Vec::new();
    for e in bytes.as_chunks::<32>().0 {
        let attr = e[11];
        match e[0] {
            0 => break,
            0xe5 => parts.clear(),
            _ if attr & 0x3f == 0x0f => {
                if e[0] & 0x40 != 0 {
                    parts.clear();
                }
                let offsets = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30];
                parts.push(offsets.map(|o| le16(e, o)));
            }
            _ if attr & 0x08 != 0 => parts.clear(),
            _ => {
                let long = (!parts.is_empty()).then(|| {
                    let units: Vec<u16> = parts
                        .iter()
                        .rev()
                        .flatten()
                        .copied()
                        .take_while(|&u| u != 0)
                        .collect();
                    String::from_utf16_lossy(&units)
                });
                parts.clear();
                let base = String::from_utf8_lossy(&e[..8]).trim_end().to_string();
                let ext = String::from_utf8_lossy(&e[8..11]).trim_end().to_string();
                out.push(Entry {
                    long,
                    short: format!("{base}.{ext}").trim_end_matches('.').to_string(),
                    cluster: u32::from(le16(e, 20)) << 16 | u32::from(le16(e, 26)),
                    size: le32(e, 28),
                });
            }
        }
    }
    out
}

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn le64(b: &[u8], o: usize) -> u64 {
    u64::from(le32(b, o)) | u64::from(le32(b, o + 4)) << 32
}

fn be32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn be64(b: &[u8], o: usize) -> u64 {
    u64::from(be32(b, o)) << 32 | u64::from(be32(b, o + 4))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory entry of `name`, an 8.3 name, with no long name before it.
    fn short(name: &[u8; 11], attr: u8, cluster: u32, size: u32) -> [u8; 32] {
        let mut e = [0u8; 32];
        e[..11].copy_from_slice(name);
        e[11] = attr;
        e[20..22].copy_from_slice(&((cluster >> 16) as u16).to_le_bytes());
        e[26..28].copy_from_slice(&(cluster as u16).to_le_bytes());
        e[28..32].copy_from_slice(&size.to_le_bytes());
        e
    }

    /// The long-name entries of `name`, last part first, as FAT stores them.
    fn long(name: &str) -> Vec<[u8; 32]> {
        let mut units: Vec<u16> = name.encode_utf16().collect();
        units.push(0);
        while !units.len().is_multiple_of(13) {
            units.push(0xffff);
        }
        let n = units.len() / 13;
        (0..n)
            .rev()
            .map(|i| {
                let mut e = [0u8; 32];
                e[0] = (i + 1) as u8 | if i == n - 1 { 0x40 } else { 0 };
                e[11] = 0x0f;
                let offsets = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30];
                for (o, u) in offsets.into_iter().zip(&units[i * 13..]) {
                    e[o..o + 2].copy_from_slice(&u.to_le_bytes());
                }
                e
            })
            .collect()
    }

    #[test]
    fn entries_have_their_long_names() {
        let mut dir: Vec<[u8; 32]> = vec![short(b"EFI        ", 0x08, 0, 0)];
        dir.push(short(b"OC         ", 0x10, 0x1_0005, 0));
        dir.extend(long("config.plist"));
        dir.push(short(b"CONFIG~1PLI", 0x20, 7, 12345));
        let mut deleted = short(b"OLD     TXT", 0x20, 9, 1);
        deleted[0] = 0xe5;
        dir.push(deleted);
        dir.push([0; 32]);
        dir.push(short(b"AFTER      ", 0x20, 10, 1));
        let e = entries(&dir.concat());
        // The volume label, the deleted entry and what follows the end are not entries.
        assert_eq!(e.len(), 2);
        assert!(e[0].is("oc") && e[0].long.is_none() && e[0].cluster == 0x1_0005);
        assert!(e[1].is("config.plist") && e[1].is("CONFIG~1.PLI"));
        assert_eq!((e[1].cluster, e[1].size), (7, 12345));
    }

    fn config(kexts: &str) -> Config {
        Config::parse(
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Kernel</key><dict><key>Add</key><array>{kexts}</array></dict>
  <key>PlatformInfo</key><dict>
    <key>Automatic</key><true/>
    <key>Generic</key><dict>
      <key>MLB</key><string>C02029100GU0000JA</string>
      <key>ROM</key><data>UlQAAQID</data>
      <key>SystemProductName</key><string>iMac20,1</string>
      <key>SystemSerialNumber</key><string>C02D40VEPN5T</string>
      <key>SystemUUID</key><string>4A2B8C1D-0E3F-4A5B-8C6D-7E8F9A0B1C2D</string>
    </dict>
  </dict>
</dict></plist>"#
            )
            .as_bytes(),
        )
        .unwrap()
    }

    #[test]
    fn identity_is_platform_infos_generic() {
        assert_eq!(
            config("").identity().unwrap(),
            Identity {
                model: "iMac20,1".into(),
                serial: "C02D40VEPN5T".into(),
                mlb: "C02029100GU0000JA".into(),
                uuid: "4A2B8C1D-0E3F-4A5B-8C6D-7E8F9A0B1C2D".into(),
                rom: "525400010203".into(),
            }
        );
    }

    #[test]
    fn reims_vgpu_loads_where_an_enabled_entry_names_it() {
        let entry = |bundle: &str, enabled: bool| {
            format!(
                "<dict><key>BundlePath</key><string>{bundle}</string>\
                 <key>Enabled</key><{enabled}/></dict>"
            )
        };
        assert!(!config("").loads_reims_vgpu());
        assert!(!config(&entry("ReimsVgpu.kext", false)).loads_reims_vgpu());
        assert!(!config(&entry("Lilu.kext", true)).loads_reims_vgpu());
        let both = entry("Lilu.kext", true) + &entry("ReimsVgpu.kext", true);
        assert!(config(&both).loads_reims_vgpu());
    }
}
