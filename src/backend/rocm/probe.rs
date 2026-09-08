//! ROCm/HIP presence probe (Phase I) — evidence for receipts, never a claim.
//!
//! The probe walks the same chain the Phase-J runtime will need, in order,
//! and reports exactly where the chain breaks:
//!
//! ```text
//! AMD compute candidate (sysfs PCI: display or processing-accelerator
//!   class, or amdgpu-bound)
//!   -> bound to the amdgpu driver
//!   -> /dev/kfd exists and opens read-write (the mode Phase J needs)
//!   -> the compute runtime (HIP and/or HSA) actually dlopens
//!   -> every Phase-J-required ABI symbol resolves
//!   -> INCONCLUSIVE_PENDING_EXECUTION (Phase J battery, never fabricated)
//! ```
//!
//! The required-symbol tables are frozen to the concrete ABI surface the
//! native ROCm runtime will call (HIP module API mirroring the CUDA driver
//! path; the direct-HSA set is the fallback surface). Probing only
//! `hipInit`/`hsa_init` would prove the runtime *exists* — not that the
//! symbols Phase J will call actually resolve, so every required symbol is
//! resolved here.
//!
//! Classification honors the evidence vocabulary (`status::Verdict`):
//!
//! * no AMD compute candidate / not amdgpu-bound / no KFD
//!   -> `UNSUPPORTED_BY_HARDWARE` (device or kernel driver absent);
//! * KFD present but not accessible          -> `INCONCLUSIVE` (device
//!   permissions/cgroup — an environment fact, not a hardware fact);
//! * compute runtime not loadable, or missing required symbols
//!   -> `UNSUPPORTED_BY_API` (runtime/library level — the ROCm *userspace*
//!   is absent or incomplete, which is not a hardware deficiency);
//! * full chain resolves                     -> `INCONCLUSIVE` pending the
//!   Phase-J differential battery.
//!
//! `librocm_smi64` is auxiliary telemetry only (energy/clock source for
//! Phase M), never evidence that the compute runtime exists.

use crate::backend::rocm::loader::{Lib, probe_soname_symbols};
use crate::error::Result;
use crate::evidence::hardware::Hardware;
use crate::status::Verdict;

/// Frozen Phase-J HIP ABI surface (the module/launch path mirroring the CUDA
/// driver API this repository already runs). Every symbol must resolve.
pub const HIP_REQUIRED_SYMBOLS: &[&str] = &[
    "hipInit",
    "hipGetDeviceCount",
    "hipSetDevice",
    "hipModuleLoadData",
    "hipModuleGetFunction",
    "hipModuleLaunchKernel",
    "hipMalloc",
    "hipFree",
    "hipMemcpy",
    "hipDeviceSynchronize",
    "hipStreamSynchronize",
    "hipHostRegister",
    "hipHostGetDevicePointer",
    "hipHostUnregister",
];

/// Frozen direct-HSA fallback surface (used only if Phase J goes HSA-native
/// rather than HIP).
pub const HSA_REQUIRED_SYMBOLS: &[&str] = &[
    "hsa_init",
    "hsa_shut_down",
    "hsa_agent_iterate_agents",
    "hsa_agent_get_info",
    "hsa_queue_create",
    "hsa_memory_allocate",
    "hsa_memory_free",
    "hsa_memory_copy",
    "hsa_signal_create",
    "hsa_signal_store_relaxed",
    "hsa_signal_wait_acquire",
    "hsa_executable_create",
    "hsa_executable_load_agent_code_object",
    "hsa_executable_freeze",
    "hsa_executable_get_symbol",
];

/// Compute-runtime sonames probed in order (name, required symbol table).
/// Both versioned and unversioned/current sonames are probed so a future
/// ROCm install is not mislabeled absent.
pub const COMPUTE_SONAMES: &[(&str, &[&str])] = &[
    ("libamdhip64.so.6", HIP_REQUIRED_SYMBOLS),
    ("libamdhip64.so.5", HIP_REQUIRED_SYMBOLS),
    ("libamdhip64.so", HIP_REQUIRED_SYMBOLS),
    ("libhsa-runtime64.so.1", HSA_REQUIRED_SYMBOLS),
    ("libhsa-runtime64.so", HSA_REQUIRED_SYMBOLS),
];

/// Auxiliary telemetry soname (energy/clock source only; not evidence of a
/// compute runtime).
pub const TELEMETRY_SONAMES: &[&str] = &["librocm_smi64.so.1"];

/// One detected AMD compute candidate (sysfs evidence).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AmdGpu {
    pub bdf: String,
    pub vendor: Option<String>,
    pub device: Option<String>,
    /// Kernel driver bound (`amdgpu` is required for the compute surface).
    pub driver: Option<String>,
    /// PCI class (display 0x03 / accelerator 0x12 / ...) — evidence of how
    /// the device presents itself.
    pub class: Option<String>,
}

/// KFD device-interface state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KfdState {
    /// No `/sys/class/kfd` or `/dev/kfd`.
    #[default]
    Absent,
    /// A node exists but is not openable read-write (device permissions /
    /// cgroup).
    PresentNotAccessible,
    /// `/dev/kfd` exists and opens read-write (the Phase-J access mode).
    Accessible,
}

/// One compute-runtime probe attempt outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAttempt {
    pub soname: String,
    /// Ok(required_symbols_still_missing) — Ok(empty) means the full ABI
    /// surface resolved.
    pub loaded: std::result::Result<Vec<String>, String>,
}

impl RuntimeAttempt {
    /// The full required ABI surface resolved.
    pub fn ready(&self) -> bool {
        matches!(&self.loaded, Ok(missing) if missing.is_empty())
    }
}

/// ROCm presence evidence. All fields are measurements; `classify` turns
/// them into a verdict without inventing anything.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RocmProbe {
    /// AMD compute candidates found in the sysfs PCI walk (display class,
    /// processing-accelerator class, or amdgpu-bound).
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
        // Compute candidacy is derived from AMD vendor + presentation
        // (display OR processing-accelerator class) OR an amdgpu-bound
        // device — a headless accelerator must not disappear just because
        // it has no VGA class.
        let amd_gpus = hw
            .pci_devices
            .iter()
            .filter(|d| {
                d.is_amd()
                    && (d.is_display()
                        || d.is_accelerator()
                        || d.driver.as_deref() == Some("amdgpu"))
            })
            .map(|d| AmdGpu {
                bdf: d.bdf.clone(),
                vendor: d.vendor.clone(),
                device: d.device.clone(),
                driver: d.driver.clone(),
                class: d.class.clone(),
            })
            .collect::<Vec<_>>();
        let kfd = kfd_state();
        let compute = COMPUTE_SONAMES
            .iter()
            .map(|(soname, required)| RuntimeAttempt {
                soname: (*soname).to_string(),
                loaded: probe_soname_symbols(soname, required)
                    .map(|(_name, missing)| missing.iter().map(|s| (*s).to_string()).collect()),
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
                "no AMD compute candidate in the sysfs PCI walk (display, accelerator, or \
                 amdgpu-bound) — the ROCm device surface cannot exist on this host"
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
                "AMD compute candidate present but not bound to the amdgpu driver — the KFD \
                 compute interface is not available"
                    .into(),
            );
        }
        match self.kfd {
            KfdState::Absent => {
                return (
                    Verdict::UnsupportedByHardware,
                    "AMD candidate bound to amdgpu but no /sys/class/kfd or /dev/kfd — the KFD \
                     driver is not exposing the device to userspace"
                        .into(),
                );
            }
            KfdState::PresentNotAccessible => {
                return (
                    Verdict::Inconclusive,
                    "AMD candidate + KFD present but /dev/kfd is not openable read-write \
                     (device permissions / cgroup) — an environment fact, not a hardware fact"
                        .into(),
                );
            }
            KfdState::Accessible => {}
        }
        // A compute runtime must load AND resolve its full frozen Phase-J
        // ABI surface. Missing/incomplete userspace is a runtime/library
        // condition: UNSUPPORTED_BY_API, never a hardware verdict.
        let ready = self.compute.iter().find(|a| a.ready());
        match ready {
            Some(ok) => (
                Verdict::Inconclusive,
                format!(
                    "ROCm device + compute runtime present ({}; full required ABI surface \
                     resolves); the scalar == ROCm differential battery and D1 surface \
                     require the Phase J runtime and are not executed by this court \
                     (INCONCLUSIVE_PENDING_EXECUTION)",
                    ok.soname
                ),
            ),
            None => {
                let why = self
                    .compute
                    .iter()
                    .map(|a| match &a.loaded {
                        Ok(missing) => format!(
                            "{}: loaded, missing {}",
                            a.soname,
                            if missing.is_empty() {
                                "nothing".to_string()
                            } else {
                                missing.join(",")
                            }
                        ),
                        Err(e) => format!("{}: {e}", a.soname),
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                (
                    Verdict::UnsupportedByApi,
                    format!(
                        "AMD compute candidate + KFD present but no ROCm compute runtime \
                         resolves its full required ABI surface (HIP/HSA): {why}"
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
    // Phase J needs read-write access (the mode libhsakmt uses), so probe
    // that mode rather than proving only a read-only open succeeds.
    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kfd")
    {
        Ok(_) => KfdState::Accessible,
        Err(_) => KfdState::PresentNotAccessible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(driver: Option<&str>, class: &str) -> AmdGpu {
        AmdGpu {
            bdf: "0000:01:00.0".into(),
            vendor: Some("0x1002".into()),
            device: Some("0x0000".into()),
            driver: driver.map(str::to_string),
            class: Some(class.into()),
        }
    }

    fn attempt(soname: &str, ready: bool) -> RuntimeAttempt {
        RuntimeAttempt {
            soname: soname.into(),
            loaded: if ready {
                Ok(vec![])
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
        assert!(p.classify().1.contains("no AMD compute candidate"));
    }

    #[test]
    fn accelerator_class_device_is_a_candidate() {
        // A headless Instinct-class accelerator (0x12) must not disappear
        // from candidacy just because it is not display class.
        let g = gpu(Some("amdgpu"), "0x120000");
        assert!(g.class.as_deref().unwrap_or("").starts_with("0x12"));
        let p = RocmProbe {
            amd_gpus: vec![g],
            kfd: KfdState::Accessible,
            compute: vec![attempt("libamdhip64.so.6", true)],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::Inconclusive); // full chain
    }

    #[test]
    fn gpu_not_amdgpu_bound_is_hardware_unsupported() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("radeon"), "0x030000")],
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
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
            kfd: KfdState::Absent,
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByHardware);
        assert!(p.classify().1.contains("KFD"));
    }

    #[test]
    fn kfd_not_accessible_is_inconclusive() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
            kfd: KfdState::PresentNotAccessible,
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::Inconclusive);
        assert!(p.classify().1.contains("permissions"));
    }

    #[test]
    fn missing_userspace_is_api_unsupported_not_hardware() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
            kfd: KfdState::Accessible,
            compute: vec![
                attempt("libamdhip64.so.6", false),
                attempt("libhsa-runtime64.so.1", false),
            ],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByApi);
        assert!(p.classify().1.contains("required ABI surface"));
    }

    #[test]
    fn partial_symbol_surface_is_not_ready() {
        // A runtime that loads but is missing a required symbol must not be
        // treated as ready.
        let a = RuntimeAttempt {
            soname: "libamdhip64.so.6".into(),
            loaded: Ok(vec!["hipModuleLaunchKernel".into()]),
        };
        assert!(!a.ready());
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
            kfd: KfdState::Accessible,
            compute: vec![a],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByApi);
    }

    #[test]
    fn full_chain_is_inconclusive_pending_execution() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
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
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
            kfd: KfdState::Accessible,
            compute: vec![attempt("libamdhip64.so.6", false)],
            telemetry: vec![("librocm_smi64.so.1".into(), true)],
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByApi);
    }
}
