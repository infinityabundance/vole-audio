//! Hardware identity and PCI topology evidence.
//!
//! D2 (peer-device) analysis and every topology-sensitive claim needs the real
//! PCI tree: BDFs, classes, drivers, NUMA nodes, IOMMU groups. Everything here
//! is read from sysfs with no external tool dependency. GPU/audio identity
//! fields are populated by the CUDA/ROCm/ALSA probe modules when they run.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// One PCI function discovered via sysfs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PciDevice {
    /// Domain:bus:device.function, e.g. "0000:01:00.0".
    pub bdf: String,
    /// PCI class code, e.g. "0x040300" (audio) / "0x030000" (VGA).
    pub class: Option<String>,
    pub vendor: Option<String>,
    pub device: Option<String>,
    /// Kernel driver bound (readlink of `driver`), if any.
    pub driver: Option<String>,
    pub numa_node: Option<i32>,
    pub iommu_group: Option<String>,
}

/// Read a small sysfs file fully as a trimmed string.
fn sysfs_read(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Read a sysfs symlink target's final component.
fn sysfs_readlink_tail(path: &Path) -> Option<String> {
    std::fs::read_link(path)
        .ok()
        .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
}

impl PciDevice {
    pub fn from_sysfs_dir(dir: &Path) -> Option<Self> {
        let bdf = dir.file_name()?.to_string_lossy().into_owned();
        let class = sysfs_read(&dir.join("class"));
        let vendor = sysfs_read(&dir.join("vendor"));
        let device = sysfs_read(&dir.join("device"));
        let driver = sysfs_readlink_tail(&dir.join("driver"));
        let numa_node = sysfs_read(&dir.join("numa_node")).and_then(|v| v.parse().ok());
        let iommu_group = sysfs_readlink_tail(&dir.join("iommu_group"));
        Some(Self {
            bdf,
            class,
            vendor,
            device,
            driver,
            numa_node,
            iommu_group,
        })
    }
}

/// Enumerate the PCI tree from `/sys/bus/pci/devices`.
pub fn pci_scan() -> Vec<PciDevice> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
        for e in entries.flatten() {
            if let Some(d) = PciDevice::from_sysfs_dir(&e.path()) {
                out.push(d);
            }
        }
    }
    out.sort_by(|a, b| a.bdf.cmp(&b.bdf));
    out
}

/// Class-code based helpers.
impl PciDevice {
    pub fn is_audio(&self) -> bool {
        self.class.as_deref().is_some_and(|c| c.starts_with("0x04"))
    }

    pub fn is_display(&self) -> bool {
        self.class.as_deref().is_some_and(|c| c.starts_with("0x03"))
    }

    /// PCI processing-accelerator class (0x12): headless compute devices
    /// (e.g. Instinct-class accelerators without a display function) are
    /// legitimate ROCm candidates and must not disappear from the probe
    /// merely because they are not VGA/display class.
    pub fn is_accelerator(&self) -> bool {
        self.class.as_deref().is_some_and(|c| c.starts_with("0x12"))
    }

    /// Vendor id 0x10de = NVIDIA, 0x1002/0x1022 = AMD.
    pub fn is_nvidia(&self) -> bool {
        self.vendor.as_deref() == Some("0x10de")
    }

    pub fn is_amd(&self) -> bool {
        matches!(self.vendor.as_deref(), Some("0x1002") | Some("0x1022"))
    }
}

/// GPU identity captured by a backend probe (CUDA/ROCm), if present.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: Option<String>,
    pub name: Option<String>,
    pub bdf: Option<String>,
    pub compute_capability: Option<String>,
    pub memory_bytes: Option<u64>,
    pub driver_version: Option<String>,
    pub runtime_version: Option<String>,
}

/// Audio endpoint identity captured by an ALSA probe, if present.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioDeviceInfo {
    pub card: Option<String>,
    pub device: Option<String>,
    pub name: Option<String>,
    pub subdevice: Option<String>,
}

/// Full hardware evidence bundle carried by receipts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardware {
    pub pci_devices: Vec<PciDevice>,
    pub gpu: GpuInfo,
    pub audio: AudioDeviceInfo,
}

impl Hardware {
    pub fn capture() -> Self {
        Self {
            pci_devices: pci_scan(),
            gpu: GpuInfo::default(),
            audio: AudioDeviceInfo::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_scan_is_deterministic_shape() {
        // On any Linux box the scan should succeed and be sorted by BDF.
        let devs = pci_scan();
        let mut sorted = devs.clone();
        sorted.sort_by(|a, b| a.bdf.cmp(&b.bdf));
        assert_eq!(devs, sorted);
        // Every entry has a non-empty bdf.
        assert!(devs.iter().all(|d| !d.bdf.is_empty()));
    }
}
