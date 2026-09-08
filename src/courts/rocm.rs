//! `court rocm` — Phase I ROCm evidence (hardware-unavailable evidence +
//! clean amdgcn build).
//!
//! Phase I delivers the AMD device surface without ROCm hardware or
//! userspace on the development host. This court records, with typed causes:
//!
//! 1. the AMD/ROCm presence probe (`backend::rocm::probe`): AMD display
//!    GPUs from the sysfs PCI walk, the KFD nodes, and the ROCm userspace
//!    sonames;
//! 2. the code-object artifact state (`scripts/build-rocm-device.sh`
//!    output, env `VOLE_ROCM_ARTIFACT` override), with its SHA-256 when
//!    present;
//! 3. the honest verdict: no AMD GPU -> `UNSUPPORTED_BY_HARDWARE`; GPU
//!    without KFD/userspace -> `UNSUPPORTED_BY_HARDWARE` with the exact
//!    cause; full stack -> `INCONCLUSIVE` (the differential device battery —
//!    scalar == ROCm, facts F01–F14 on the device surface, D0/D1 worlds —
//!    is Phase J scope on ROCm hardware and is never manufactured here).
//!
//! This court never executes a ROCm kernel and never claims device
//! execution: the code object is compile evidence, and its loadability on a
//! specific gfx target is Phase-J evidence.

use crate::backend::rocm::probe::RocmProbe;
use crate::error::Result;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::hash::sha256::{Sha256, hex};
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Default code-object artifact produced by scripts/build-rocm-device.sh.
const DEFAULT_ARTIFACT: &str = "scripts/out/vole_audio.amdgcn.elf";

pub fn run(receipts_root: &Path) -> Result<Verdict> {
    // 1. Code-object artifact evidence.
    let path = std::env::var("VOLE_ROCM_ARTIFACT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_ARTIFACT));
    let (artifact, artifact_sha) = match std::fs::read(&path) {
        Ok(bytes) => {
            let sha = hex(&Sha256::digest(&bytes));
            (
                serde_json::json!({
                    "path": path.display().to_string(),
                    "present": true,
                    "bytes": bytes.len(),
                    "sha256": sha,
                }),
                Some(sha),
            )
        }
        Err(_) => (
            serde_json::json!({
                "path": path.display().to_string(),
                "present": false,
                "build_step": "sh scripts/build-rocm-device.sh (requires the pinned nightly + rust-src; see docs/PHASE_I.md)",
            }),
            None,
        ),
    };

    // 2. Presence probe + honest verdict.
    let probe = RocmProbe::capture()?;
    let (verdict, detail) = probe.classify();
    let probe_json = serde_json::json!({
        "amd_gpus": probe.amd_gpus.iter().map(|g| serde_json::json!({
            "bdf": g.bdf, "vendor": g.vendor, "device": g.device, "driver": g.driver,
        })).collect::<Vec<_>>(),
        "kfd_class_present": probe.kfd_class_present,
        "kfd_dev_present": probe.kfd_dev_present,
        "libs": probe.libs.iter().map(|(n, f)| serde_json::json!({ "soname": n, "found": f })).collect::<Vec<_>>(),
        "classification": detail,
    });

    // 3. Receipt.
    let mut b = ReceiptBuilder::new("rocm");
    b.result(verdict)
        .result_detail(format!("rocm: {detail}"))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1".into()),
            backend: Some("rocm".into()),
            content_kind: Some("phase-i rocm evidence".into()),
            ..Default::default()
        })
        .provenance(Provenance {
            gpu_artifact_hash: artifact_sha,
            ..Default::default()
        })
        .extra("rocm_probe", probe_json)
        .extra("artifact", artifact);
    b.limitation(
        "Phase I is hardware-unavailable evidence + clean amdgcn build: this court never \
         executes a ROCm kernel. The differential device battery (scalar == ROCm, facts \
         F01-F14 on the device surface, D0/D1 worlds) is Phase J scope on ROCm hardware and \
         is never manufactured here.",
    );
    b.limitation(
        "AMDGPU code objects are per-ISA (built for the VOLE_ROCM_GFX baseline, default \
         gfx906); loadability on a specific device is Phase-J evidence and is never assumed.",
    );
    let (_, path) = b.finish_write(receipts_root)?;
    println!("court rocm: {verdict}");
    println!("  {detail}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_default_path_is_the_build_script_output() {
        // Guards the court/script contract: the receipt reads what
        // scripts/build-rocm-device.sh writes.
        assert_eq!(DEFAULT_ARTIFACT, "scripts/out/vole_audio.amdgcn.elf");
    }
}
