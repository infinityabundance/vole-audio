//! ROCm/HIP presence probe (Phase I) — evidence for receipts, never a claim.
//!
//! The probe walks the same chain the Phase-J runtime will need, in order,
//! and reports exactly where the chain breaks:
//!
//! ```text
//! AMD GPU candidate (sysfs PCI)
//!   -> bound to the amdgpu driver
//!   -> /dev/kfd exists and is accessible
//!   -> the compute runtime (HSA and/or HIP) actually dlopens
//!   -> the Phase-J-required runtime symbols resolve
//!   -> INCONCLUSIVE_PENDING_EXECUTION (Phase J battery, never fabricated)
//! ```
//!
//! Classification honors the evidence vocabulary (`status::Verdict`):
//!
//! * no AMD GPU / not amdgpu-bound / no KFD  -> `UNSUPPORTED_BY_HARDWARE`
//!   (the device or its kernel driver is absent);
//! * KFD present but not accessible          -> `INCONCLUSIVE` (device
//!   permissions/cgroup — an environment fact, not a hardware fact);
//! * compute runtime (HSA/HIP) not loadable  -> `UNSUPPORTED_BY_API`
//!   (runtime/library level — the ROCm *userspace* is absent, which is not
//!   a hardware deficiency);
//! * full chain resolves                     -> `INCONCLUSIVE` pending the
//!   Phase-J differential battery.
//!
//! `librocm_smi64` is auxiliary telemetry only (energy/clock source for
//! Phase M), never evidence that the compute runtime exists.

use crate::backend::rocm::loader::{Lib, probe_soname_symbol};
use crate::error::Result;
use crate::evidence::hardware::Hardware;
use crate::status::Verdict;

/// Compute-runtime sonames probed in order (name, required entry symbol).
pub const COMPUTE_SONAMES: &[(&str, &str)] = &[
    ("libamdhip64.so.6", "hipInit"),
    ("libamdhip64.so.5", "hipInit"),
    ("libhsa-runtime64.so.1", "hsa_init"),
];

/// Auxiliary telemetry soname (energy/clock source only; not evidence of a
/// compute runtime).
pub const TELEMETRY_SONAMES: &[&str] = &["librocm_smi64.so.1"];

/// One detected AMD display GPU (sysfs evidence).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AmdGpu {
    pub bdf: String,
    pub vendor: Option<String>,
    pub device: Option<String>,
    /// Kernel driver bound (`amdgpu` is required for the compute surface).
    pub driver: Option<String>,
}

/// KFD device-interface state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KfdState {
    /// No `/sys/class/kfd` or `/dev/kfd`.
    #[default]
    Absent,
    /// A node exists but is not openable (device permissions / cgroup).
    PresentNotAccessible,
    /// `/dev/kfd` exists and opens for read.
    Accessible,
}

/// One compute-runtime probe attempt outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAttempt {
    pub soname: String,
    /// Ok(()) when the soname dlopens and its required symbol resolves.
    pub loaded: std::result::Result<(), String>,
}

/// ROCm presence evidence. All fields are measurements; `classify` turns
/// them into a verdict without inventing anything.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RocmProbe {
    /// AMD display GPUs found in the sysfs PCI walk.
    pub amd_gpus: Vec<AmdGpu>,
    pub kfd: KfdState,
    /// Compute-runtime load attempts (HIP first, HSA second).
    pub compute: Vec<RuntimeAttempt>,
    /// Auxiliary telemetry libraries (rocm-smi); not part of the chain.
    pub telemetry: Vec<(String, bool)>,
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
        let kfd = kfd_state();
        let compute = COMPUTE_SONAMES
            .iter()
            .map(|(soname, symbol)| RuntimeAttempt {
                soname: (*soname).to_string(),
                loaded: probe_soname_symbol(soname, symbol).map(|_| ()),
            })
            .collect::<Vec<_>>();
        let telemetry = TELEMETRY_SONAMES
            .iter()
            .map(|soname| ((*soname).to_string(), Lib::open(soname).is_ok()))
            .collect::<Vec<_>>();
        Ok(RocmProbe {
            amd_gpus,
            kfd,
            compute,
            telemetry,
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
        if !self
            .amd_gpus
            .iter()
            .any(|g| g.driver.as_deref() == Some("amdgpu"))
        {
            return (
                Verdict::UnsupportedByHardware,
                "AMD GPU present but not bound to the amdgpu driver — the KFD compute \
                 interface is not available"
                    .into(),
            );
        }
        match self.kfd {
            KfdState::Absent => {
                return (
                    Verdict::UnsupportedByHardware,
                    "AMD GPU bound to amdgpu but no /sys/class/kfd or /dev/kfd — the KFD \
                     driver is not exposing the device to userspace"
                        .into(),
                );
            }
            KfdState::PresentNotAccessible => {
                return (
                    Verdict::Inconclusive,
                    "AMD GPU + KFD present but /dev/kfd is not accessible (device \
                     permissions / cgroup) — an environment fact, not a hardware fact"
                        .into(),
                );
            }
            KfdState::Accessible => {}
        }
        // The compute runtime (HIP or HSA) must actually load with the
        // Phase-J-required symbols. Missing userspace is a runtime/library
        // condition: UNSUPPORTED_BY_API, never a hardware verdict.
        let loadable = self.compute.iter().find(|a| a.loaded.is_ok());
        match loadable {
            Some(ok) => (
                Verdict::Inconclusive,
                format!(
                    "ROCm device + compute runtime present ({}); the scalar == ROCm \
                     differential battery and D1 surface require the Phase J runtime and \
                     are not executed by this court (INCONCLUSIVE_PENDING_EXECUTION)",
                    ok.soname
                ),
            ),
            None => {
                let why = self
                    .compute
                    .iter()
                    .map(|a| match &a.loaded {
                        Ok(()) => format!("{}: loaded", a.soname),
                        Err(e) => format!("{}: {e}", a.soname),
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                (
                    Verdict::UnsupportedByApi,
                    format!(
                        "AMD GPU + KFD present but no ROCm compute runtime is loadable \
                         (HIP/HSA): {why}"
                    ),
                )
            }
        }
    }
}

fn kfd_state() -> KfdState {
    let class = std::path::Path::new("/sys/class/kfd").exists();
    let dev = std::path::Path::new("/dev/kfd").exists();
    if !class && !dev {
        return KfdState::Absent;
    }
    if !dev {
        // A sysfs node without a device node: not usable.
        return KfdState::Absent;
    }
    match std::fs::OpenOptions::new().read(true).open("/dev/kfd") {
        Ok(_) => KfdState::Accessible,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            KfdState::PresentNotAccessible
        }
        Err(_) => KfdState::PresentNotAccessible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(driver: Option<&str>) -> AmdGpu {
        AmdGpu {
            bdf: "0000:01:00.0".into(),
            vendor: Some("0x1002".into()),
            device: Some("0x0000".into()),
            driver: driver.map(str::to_string),
        }
    }

    fn attempt(soname: &str, ok: bool) -> RuntimeAttempt {
        RuntimeAttempt {
            soname: soname.into(),
            loaded: if ok {
                Ok(())
            } else {
                Err(format!("dlopen {soname} failed"))
            },
        }
    }

    #[test]
    fn no_amd_gpu_is_hardware_unsupported() {
        let p = RocmProbe {
            amd_gpus: vec![],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByHardware);
        assert!(p.classify().1.contains("no AMD display GPU"));
    }

    #[test]
    fn gpu_not_amdgpu_bound_is_hardware_unsupported() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("radeon"))],
            kfd: KfdState::Accessible,
            compute: vec![attempt("libamdhip64.so.6", true)],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByHardware);
        assert!(p.classify().1.contains("amdgpu driver"));
    }

    #[test]
    fn gpu_without_kfd_is_hardware_unsupported() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"))],
            kfd: KfdState::Absent,
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByHardware);
        assert!(p.classify().1.contains("KFD"));
    }

    #[test]
    fn kfd_not_accessible_is_inconclusive() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"))],
            kfd: KfdState::PresentNotAccessible,
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::Inconclusive);
        assert!(p.classify().1.contains("permissions"));
    }

    #[test]
    fn missing_userspace_is_api_unsupported_not_hardware() {
        // The taxonomy bug from review: absent HIP/HSA userspace must NOT be
        // UNSUPPORTED_BY_HARDWARE — it is a runtime/library condition.
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"))],
            kfd: KfdState::Accessible,
            compute: vec![
                attempt("libamdhip64.so.6", false),
                attempt("libhsa-runtime64.so.1", false),
            ],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByApi);
        assert!(p.classify().1.contains("compute runtime"));
    }

    #[test]
    fn full_chain_is_inconclusive_pending_execution() {
        // Phase I must never manufacture SUPPORTED for a device battery it
        // does not run: full chain -> Inconclusive with the Phase J reason.
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"))],
            kfd: KfdState::Accessible,
            compute: vec![attempt("libamdhip64.so.6", true)],
            telemetry: vec![("librocm_smi64.so.1".into(), true)],
        };
        let (v, d) = p.classify();
        assert_eq!(v, Verdict::Inconclusive);
        assert!(d.contains("Phase J"));
    }

    #[test]
    fn telemetry_alone_is_not_compute_runtime() {
        // rocm-smi without HIP/HSA must not make the chain loadable.
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"))],
            kfd: KfdState::Accessible,
            compute: vec![
                attempt("libamdhip64.so.6", false),
                attempt("libhsa-runtime64.so.1", false),
            ],
            telemetry: vec![("librocm_smi64.so.1".into(), true)],
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByApi);
    }
}
