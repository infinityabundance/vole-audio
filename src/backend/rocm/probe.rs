//! ROCm/HIP presence probe (Phase I) — evidence for receipts, never a claim.
//!
//! Phase I has no ROCm hardware or userspace on the development host, so
//! this probe's job is to record *why* the ROCm surface is unavailable with
//! typed, machine-readable causes — the same honesty rule as every other
//! backend probe. When ROCm hardware + userspace exist, `court rocm` still
//! classifies the device-execution battery as Phase-J evidence; this module
//! never executes kernels.
//!
//! Detection is filesystem/sysfs only (no dlopen of absent libraries):
//!
//! * AMD display GPUs from the sysfs PCI walk (`hardware::Hardware`);
//! * the kernel graphics driver node `/sys/class/kfd` (+ `/dev/kfd`);
//! * the standard ROCm userspace sonames (`libhsa-runtime64`, `libamdhip64`,
//!   `librocm_smi64`) under the usual search paths (`/opt/rocm*/lib*`,
//!   `/usr/lib`, `/usr/lib64`).

use crate::error::Result;
use crate::evidence::hardware::Hardware;
use crate::status::Verdict;

/// ROCm userspace sonames probed (name -> found).
pub const ROCM_SONAMES: &[&str] = &[
    "libhsa-runtime64.so.1",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
    "librocm_smi64.so.1",
];

/// Search roots for the ROCm userspace libraries.
const LIB_ROOTS: &[&str] = &["/opt/rocm/lib", "/opt/rocm/lib64", "/usr/lib", "/usr/lib64"];

/// One detected AMD display GPU (sysfs evidence).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AmdGpu {
    pub bdf: String,
    pub vendor: Option<String>,
    pub device: Option<String>,
    pub driver: Option<String>,
}

/// ROCm presence evidence. All fields are measurements; `classify` turns
/// them into a verdict without inventing anything.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RocmProbe {
    /// AMD display GPUs found in the sysfs PCI walk.
    pub amd_gpus: Vec<AmdGpu>,
    /// `/sys/class/kfd` present (KFD userspace interface).
    pub kfd_class_present: bool,
    /// `/dev/kfd` present.
    pub kfd_dev_present: bool,
    /// ROCm userspace soname search results (soname, found).
    pub libs: Vec<(String, bool)>,
}

impl RocmProbe {
    /// Run the probe on this host.
    pub fn capture() -> Result<Self> {
        let hw = Hardware::capture();
        let amd_gpus = hw
            .pci_devices
            .iter()
            .filter(|d| d.is_display() && d.is_amd())
            .map(|d| AmdGpu {
                bdf: d.bdf.clone(),
                vendor: d.vendor.clone(),
                device: d.device.clone(),
                driver: d.driver.clone(),
            })
            .collect::<Vec<_>>();
        let kfd_class_present = std::path::Path::new("/sys/class/kfd").exists();
        let kfd_dev_present = std::path::Path::new("/dev/kfd").exists();
        let libs = ROCM_SONAMES
            .iter()
            .map(|soname| ((*soname).to_string(), find_soname(soname)))
            .collect::<Vec<_>>();
        Ok(RocmProbe {
            amd_gpus,
            kfd_class_present,
            kfd_dev_present,
            libs,
        })
    }

    /// Classify the probe into the evidence vocabulary. Pure: same probe,
    /// same verdict — never a manufactured outcome.
    pub fn classify(&self) -> (Verdict, String) {
        if self.amd_gpus.is_empty() {
            return (
                Verdict::UnsupportedByHardware,
                "no AMD display GPU in the sysfs PCI walk — the ROCm device surface cannot \
                 exist on this host"
                    .into(),
            );
        }
        if !self.kfd_class_present && !self.kfd_dev_present {
            return (
                Verdict::UnsupportedByHardware,
                "AMD GPU present but no /sys/class/kfd or /dev/kfd — the amdgpu KFD driver is \
                 not exposing the device to userspace"
                    .into(),
            );
        }
        if !self.libs.iter().any(|(_, found)| *found) {
            return (
                Verdict::UnsupportedByHardware,
                "AMD GPU + KFD present but no ROCm userspace soname found \
                 (hsa/hip/rocmsmi) — the runtime libraries are not installed"
                    .into(),
            );
        }
        (
            Verdict::Inconclusive,
            "ROCm device + userspace present; the scalar == ROCm differential battery and D1 \
             surface require the Phase J runtime and are not executed by this court"
                .into(),
        )
    }
}

fn find_soname(soname: &str) -> bool {
    LIB_ROOTS.iter().any(|root| {
        let p = std::path::Path::new(root).join(soname);
        p.exists()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(gpus: usize, kfd_class: bool, kfd_dev: bool, libs: &[(&str, bool)]) -> RocmProbe {
        RocmProbe {
            amd_gpus: (0..gpus)
                .map(|i| AmdGpu {
                    bdf: format!("0000:0{i}:00.0"),
                    vendor: Some("0x1002".into()),
                    device: Some("0x0000".into()),
                    driver: Some("amdgpu".into()),
                })
                .collect(),
            kfd_class_present: kfd_class,
            kfd_dev_present: kfd_dev,
            libs: libs.iter().map(|(n, f)| (n.to_string(), *f)).collect(),
        }
    }

    #[test]
    fn classify_no_amd_gpu_is_hardware_unsupported() {
        let p = probe(0, false, false, &[]);
        assert_eq!(p.classify().0, Verdict::UnsupportedByHardware);
        assert!(p.classify().1.contains("no AMD display GPU"));
    }

    #[test]
    fn classify_gpu_without_kfd_is_hardware_unsupported() {
        let p = probe(1, false, false, &[("libamdhip64.so.6", true)]);
        assert_eq!(p.classify().0, Verdict::UnsupportedByHardware);
        assert!(p.classify().1.contains("KFD"));
    }

    #[test]
    fn classify_gpu_kfd_without_userspace_is_hardware_unsupported() {
        let p = probe(1, true, true, &[("libhsa-runtime64.so.1", false)]);
        assert_eq!(p.classify().0, Verdict::UnsupportedByHardware);
        assert!(p.classify().1.contains("runtime libraries"));
    }

    #[test]
    fn classify_full_stack_is_inconclusive_not_supported() {
        // Phase I must never manufacture SUPPORTED for a device battery it
        // does not run: full stack -> Inconclusive with the Phase J reason.
        let p = probe(1, true, true, &[("libhsa-runtime64.so.1", true)]);
        assert_eq!(p.classify().0, Verdict::Inconclusive);
        assert!(p.classify().1.contains("Phase J"));
    }
}
