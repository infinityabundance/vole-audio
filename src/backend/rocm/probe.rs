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

use crate::backend::rocm::loader::{Lib, probe_soname_surfaces};
use crate::error::Result;
use crate::evidence::hardware::Hardware;
use crate::status::Verdict;

/// Frozen Phase-J HIP **D0** ABI surface (the module/launch path mirroring
/// the CUDA driver API this repository already runs — scalar == ROCm
/// differential battery). Every symbol must resolve for D0 readiness.
pub const HIP_D0_REQUIRED: &[&str] = &[
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
];

/// Frozen Phase-J HIP **D1 additional** ABI surface (direct endpoint
/// mapping). D1 readiness = D0 readiness AND every symbol here. A stack that
/// runs the D0 battery but refuses host registration must report
/// "ROCm D0 READY; D1 UNSUPPORTED_BY_API" — never "runtime unavailable".
pub const HIP_D1_ADDITIONAL: &[&str] = &[
    "hipHostRegister",
    "hipHostGetDevicePointer",
    "hipHostUnregister",
];

/// Frozen direct-HSA **D0** ABI surface (used only if Phase J goes
/// HSA-native rather than HIP). Symbol names follow the current ROCR ABI
/// (AMD's tracing examples: hsa_executable_create_alt ->
/// hsa_code_object_reader_create_from_memory ->
/// hsa_executable_load_agent_code_object -> hsa_executable_freeze ->
/// hsa_executable_get_symbol_by_name -> hsa_executable_symbol_get_info).
pub const HSA_D0_REQUIRED: &[&str] = &[
    "hsa_init",
    "hsa_shut_down",
    "hsa_iterate_agents",
    "hsa_agent_get_info",
    "hsa_queue_create",
    "hsa_signal_create",
    "hsa_signal_store_relaxed",
    "hsa_signal_store_release",
    "hsa_signal_wait_acquire",
    "hsa_queue_store_write_index_relaxed",
    "hsa_memory_allocate",
    "hsa_memory_free",
    "hsa_memory_copy",
    "hsa_executable_create_alt",
    "hsa_code_object_reader_create_from_memory",
    "hsa_code_object_reader_destroy",
    "hsa_executable_load_agent_code_object",
    "hsa_executable_freeze",
    "hsa_executable_get_symbol_by_name",
    "hsa_executable_symbol_get_info",
];

/// Frozen direct-HSA **D1 additional** ABI surface (host-memory mapping for
/// the endpoint region).
pub const HSA_D1_ADDITIONAL: &[&str] = &[
    "hsa_host_malloc",
    "hsa_host_free",
    "hsa_amd_agent_memory_pool_get_info",
];

/// Compute-runtime sonames probed in order: (name, D0 table, D1 table).
/// Versioned and unversioned sonames are probed, and /opt/rocm* installs
/// are additionally scanned for `libamdhip64.so*` / `libhsa-runtime64.so*`
/// by prefix, so a future ROCm layout with a newer versioned SONAME is
/// discovered rather than mislabeled absent.
pub const COMPUTE_SONAMES: &[(&str, &[&str], &[&str])] = &[
    ("libamdhip64.so.6", HIP_D0_REQUIRED, HIP_D1_ADDITIONAL),
    ("libamdhip64.so.5", HIP_D0_REQUIRED, HIP_D1_ADDITIONAL),
    ("libamdhip64.so", HIP_D0_REQUIRED, HIP_D1_ADDITIONAL),
    ("libhsa-runtime64.so.1", HSA_D0_REQUIRED, HSA_D1_ADDITIONAL),
    ("libhsa-runtime64.so", HSA_D0_REQUIRED, HSA_D1_ADDITIONAL),
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

/// One compute-runtime probe attempt outcome, split by readiness surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAttempt {
    pub soname: String,
    /// D0-required symbols still missing (empty => D0 ready).
    pub d0_missing: std::result::Result<Vec<String>, String>,
    /// D1-additional symbols still missing (empty => D1 ready given D0).
    pub d1_missing: std::result::Result<Vec<String>, String>,
}

impl RuntimeAttempt {
    /// The full D0 ABI surface resolved (scalar == ROCm battery readiness).
    pub fn d0_ready(&self) -> bool {
        matches!(&self.d0_missing, Ok(m) if m.is_empty())
    }

    /// D0 + D1 additional resolved (direct endpoint mapping readiness).
    pub fn d1_ready(&self) -> bool {
        self.d0_ready() && matches!(&self.d1_missing, Ok(m) if m.is_empty())
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
            .map(|(soname, d0, d1)| {
                let r = probe_soname_surfaces(soname, d0, d1);
                let (name, m0, m1) = match r {
                    Ok(t) => t,
                    Err(e) => {
                        return RuntimeAttempt {
                            soname: (*soname).to_string(),
                            d0_missing: Err(e.clone()),
                            d1_missing: Err(e),
                        };
                    }
                };
                RuntimeAttempt {
                    soname: name,
                    d0_missing: Ok(m0.iter().map(|s| (*s).to_string()).collect()),
                    d1_missing: Ok(m1.iter().map(|s| (*s).to_string()).collect()),
                }
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
        // D0/D1 readiness split: the scalar == ROCm D0 battery and the D1
        // direct-endpoint path have separate ABI requirements. A stack that
        // runs D0 but cannot register host memory is "D0 READY; D1
        // UNSUPPORTED_BY_API", never "runtime unavailable".
        let d0_ready = self.compute.iter().find(|a| a.d0_ready());
        match d0_ready {
            Some(ok) => {
                let d1_state = if ok.d1_ready() {
                    "D1 READY".to_string()
                } else {
                    format!(
                        "D1 not ready (missing: {})",
                        ok.d1_missing
                            .as_ref()
                            .map(|m| m.join(","))
                            .unwrap_or_default()
                    )
                };
                (
                    Verdict::Inconclusive,
                    format!(
                        "ROCm D0 runtime READY ({}; full D0 ABI surface resolves; {}); the \
                         scalar == ROCm differential battery is the Phase J evidence target \
                         and is not executed by this court (INCONCLUSIVE_PENDING_EXECUTION)",
                        ok.soname, d1_state
                    ),
                )
            }
            None => {
                let why = self
                    .compute
                    .iter()
                    .map(|a| match (&a.d0_missing, &a.d1_missing) {
                        (Ok(m0), Ok(m1)) => format!(
                            "{}: loaded; D0 missing {}, D1 missing {}",
                            a.soname,
                            if m0.is_empty() {
                                "nothing".into()
                            } else {
                                m0.join(",")
                            },
                            if m1.is_empty() {
                                "nothing".into()
                            } else {
                                m1.join(",")
                            },
                        ),
                        (Err(e), _) | (_, Err(e)) => format!("{}: {e}", a.soname),
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                (
                    Verdict::UnsupportedByApi,
                    format!(
                        "AMD compute candidate + KFD present but no ROCm compute runtime \
                         resolves its full D0 ABI surface (HIP/HSA): {why}"
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

    fn attempt(soname: &str, d0_ready: bool, d1_ready: bool) -> RuntimeAttempt {
        RuntimeAttempt {
            soname: soname.into(),
            d0_missing: if d0_ready {
                Ok(vec![])
            } else {
                Err(format!("dlopen {soname} failed"))
            },
            d1_missing: if d1_ready {
                Ok(vec![])
            } else if d0_ready {
                Ok(vec!["hipHostRegister".into()])
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
            compute: vec![attempt("libamdhip64.so.6", true, true)],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::Inconclusive); // full chain
    }

    #[test]
    fn gpu_not_amdgpu_bound_is_hardware_unsupported() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("radeon"), "0x030000")],
            kfd: KfdState::Accessible,
            compute: vec![attempt("libamdhip64.so.6", true, true)],
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
                attempt("libamdhip64.so.6", false, false),
                attempt("libhsa-runtime64.so.1", false, false),
            ],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByApi);
        assert!(p.classify().1.contains("D0 ABI surface"));
    }

    #[test]
    fn partial_symbol_surface_is_not_d0_ready() {
        // A runtime that loads but is missing a D0 symbol must not be D0
        // ready — and a D1-only gap must not hide D0 readiness.
        let a = RuntimeAttempt {
            soname: "libamdhip64.so.6".into(),
            d0_missing: Ok(vec!["hipModuleLaunchKernel".into()]),
            d1_missing: Ok(vec![]),
        };
        assert!(!a.d0_ready());
        assert!(!a.d1_ready());
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
            kfd: KfdState::Accessible,
            compute: vec![a],
            ..Default::default()
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByApi);
    }

    #[test]
    fn d1_gap_does_not_hide_d0_readiness() {
        // D0 ready but host-registration symbols missing: the runtime is NOT
        // "unavailable" — it is D0 READY with D1 blocked (the exact case
        // Phase J must distinguish).
        let a = RuntimeAttempt {
            soname: "libamdhip64.so.6".into(),
            d0_missing: Ok(vec![]),
            d1_missing: Ok(vec!["hipHostRegister".into(), "hipHostUnregister".into()]),
        };
        assert!(a.d0_ready());
        assert!(!a.d1_ready());
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
            kfd: KfdState::Accessible,
            compute: vec![a],
            ..Default::default()
        };
        let (v, d) = p.classify();
        assert_eq!(v, Verdict::Inconclusive); // D0 ready -> pending battery
        assert!(d.contains("D0 runtime READY"));
        assert!(d.contains("D1 not ready"));
    }

    #[test]
    fn full_chain_is_inconclusive_pending_execution() {
        let p = RocmProbe {
            amd_gpus: vec![gpu(Some("amdgpu"), "0x030000")],
            kfd: KfdState::Accessible,
            compute: vec![attempt("libamdhip64.so.6", true, true)],
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
            compute: vec![attempt("libamdhip64.so.6", false, false)],
            telemetry: vec![("librocm_smi64.so.1".into(), true)],
        };
        assert_eq!(p.classify().0, Verdict::UnsupportedByApi);
    }
}
