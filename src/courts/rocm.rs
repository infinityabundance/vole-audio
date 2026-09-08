//! `court rocm` — Phase I ROCm evidence (hardware-unavailable evidence +
//! clean amdgcn build).
//!
//! The receipt is explicitly two-dimensional and fails closed:
//!
//! ```text
//! compile_surface:  artifact present + ELF AMDGPU + required kernel entries
//!                   + provenance sidecar + artifact/source-tree correspondence
//! runtime_surface:  AMD GPU -> amdgpu driver -> KFD -> HIP/HSA dlopen +
//!                   symbols (backend::rocm::probe)
//! ```
//!
//! * A missing or malformed compile artifact is **incapable of satisfying
//!   Phase I**: whatever the runtime says, the verdict is `INCONCLUSIVE`
//!   with the exact compile-surface reason (Phase I is defined as
//!   hardware-unavailable evidence **plus** a clean amdgcn build — the
//!   build is not optional).
//! * With the compile surface satisfied, the verdict comes from the runtime
//!   chain with the corrected taxonomy: no AMD GPU / not amdgpu-bound / no
//!   KFD -> `UNSUPPORTED_BY_HARDWARE`; KFD present but inaccessible ->
//!   `INCONCLUSIVE`; compute runtime (HIP/HSA) not loadable ->
//!   `UNSUPPORTED_BY_API` (userspace absence is not a hardware deficiency);
//!   full chain -> `INCONCLUSIVE` pending the Phase-J differential battery.
//! * The artifact is self-defending: it is validated from its bytes
//!   (`backend::rocm::elf`) — an arbitrary file named `VOLE_ROCM_ARTIFACT`
//!   is never merely hashed and called a GPU artifact — and its provenance
//!   sidecar must agree with the attested source tree.
//!
//! This court never executes a ROCm kernel and never claims device
//! execution: the code object is compile evidence, and its loadability on a
//! specific gfx target is Phase-J evidence.

use crate::backend::rocm::elf::inspect_amdgcn_code_object;
use crate::backend::rocm::probe::{KfdState, RocmProbe};
use crate::error::Result;
use crate::evidence::environment::Environment;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::hash::sha256::{Sha256, hex};
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Default code-object artifact produced by scripts/build-rocm-device.sh.
const DEFAULT_ARTIFACT: &str = "scripts/out/vole_audio.amdgcn.elf";
/// Determinism evidence written by `build-rocm-device.sh --verify-deterministic`.
const DEFAULT_DETERMINISM: &str = "scripts/out/vole_audio.amdgcn.determinism.json";

/// Compile-surface assessment. Satisfied only when the artifact is a valid
/// AMDGPU code object carrying every required kernel entry AND its
/// provenance sidecar matches the attested source tree.
fn compile_surface(path: &Path, sidecar: &Path, env: &Environment) -> (serde_json::Value, bool) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            return (
                serde_json::json!({
                    "satisfied": false,
                    "artifact_present": false,
                    "build_step": "sh scripts/build-rocm-device.sh (requires the pinned nightly + rust-src; see docs/PHASE_I.md)",
                }),
                false,
            );
        }
    };
    let sha = hex(&Sha256::digest(&bytes));
    let elf = match inspect_amdgcn_code_object(&bytes) {
        Ok(info) => info,
        Err(e) => {
            return (
                serde_json::json!({
                    "satisfied": false,
                    "artifact_present": true,
                    "bytes": bytes.len(),
                    "sha256": sha,
                    "elf_error": e,
                }),
                false,
            );
        }
    };
    let sidecar_json = std::fs::read_to_string(sidecar)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
    let (source_matches, build_dirty, sidecar_summary) = match &sidecar_json {
        Some(sj) => {
            let sidecar_tree = sj.get("source_tree_sha").and_then(|v| v.as_str());
            let dirty = sj.get("source_dirty").and_then(|v| v.as_bool());
            // None when one side has no tree identity: unverifiable.
            let matches: Option<bool> = match (&env.git_tree_sha, sidecar_tree) {
                (Some(runtime), Some(built)) => Some(runtime == built),
                _ => None,
            };
            (
                matches,
                dirty,
                serde_json::json!({
                    "sidecar_present": true,
                    "matches_runtime_tree": matches,
                    "build_dirty": dirty,
                    "target_cpu": sj.get("target_cpu"),
                    "toolchain": sj.get("rustc"),
                    "std": sj.get("std"),
                }),
            )
        }
        None => (None, None, serde_json::json!({ "sidecar_present": false })),
    };
    let determinism = std::fs::read_to_string(DEFAULT_DETERMINISM)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
    let satisfied = elf.valid() && source_matches == Some(true) && build_dirty == Some(false);
    let value = serde_json::json!({
        "satisfied": satisfied,
        "artifact_present": true,
        "bytes": bytes.len(),
        "sha256": sha,
        "elf_machine_amdgpu": elf.machine_amdgpu,
        "amdgpu_arch_version": elf.amdgpu_arch_version,
        "entries_found": elf.entries_found,
        "entries_missing": elf.entries_missing,
        "provenance": sidecar_summary,
        "determinism": determinism,
    });
    (value, satisfied)
}

pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let artifact_path = std::env::var("VOLE_ROCM_ARTIFACT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_ARTIFACT));
    // The build script writes the provenance sidecar as "<artifact>.json"
    // (vole_audio.amdgcn.elf -> vole_audio.amdgcn.json).
    let sidecar_path = artifact_path.with_extension("json");
    let env = Environment::capture();

    // 1. Compile surface (fail closed).
    let (compile, compile_satisfied) = compile_surface(&artifact_path, &sidecar_path, &env);
    let artifact_sha = compile
        .get("sha256")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    // 2. Runtime surface.
    let probe = RocmProbe::capture()?;
    let (runtime_verdict, runtime_detail) = probe.classify();
    let runtime = serde_json::json!({
        "amd_gpus": probe.amd_gpus.iter().map(|g| serde_json::json!({
            "bdf": g.bdf, "vendor": g.vendor, "device": g.device, "driver": g.driver,
        })).collect::<Vec<_>>(),
        "kfd": match probe.kfd {
            KfdState::Absent => "absent",
            KfdState::PresentNotAccessible => "present_not_accessible",
            KfdState::Accessible => "accessible",
        },
        "compute_runtime": probe.compute.iter().map(|a| serde_json::json!({
            "soname": a.soname,
            "loaded": a.loaded.is_ok(),
            "detail": a.loaded.as_ref().err().cloned(),
        })).collect::<Vec<_>>(),
        "telemetry_rocmsmi": probe.telemetry.iter().any(|(_, f)| *f),
        "classification": runtime_detail,
    });

    // 3. Verdict: compile surface first (Phase I evidence is incomplete
    // without its clean amdgcn build), then the runtime chain.
    let (verdict, detail) = if !compile_satisfied {
        (
            Verdict::Inconclusive,
            format!(
                "Phase I compile surface unsatisfied (run scripts/build-rocm-device.sh on \
                 this tree): {} — runtime classification recorded separately: {}",
                compile
                    .get("elf_error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("artifact missing or provenance unverifiable"),
                runtime_detail
            ),
        )
    } else {
        (runtime_verdict, runtime_detail)
    };

    // 4. Receipt.
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
        .extra("compile_surface", compile)
        .extra("runtime_surface", runtime);
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
    b.limitation(
        "The runtime_surface chain (GPU -> amdgpu -> KFD -> HIP/HSA dlopen + symbols) is \
         measured; a compute-runtime absence classifies UNSUPPORTED_BY_API (userspace), not \
         UNSUPPORTED_BY_HARDWARE.",
    );
    let (_, path) = b.finish_write(receipts_root)?;
    println!("court rocm: {verdict}");
    println!(
        "  compile_surface.satisfied={compile_satisfied} runtime={}",
        runtime_verdict.label()
    );
    println!("  {detail}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_default_path_is_the_build_script_output() {
        assert_eq!(DEFAULT_ARTIFACT, "scripts/out/vole_audio.amdgcn.elf");
    }
}
