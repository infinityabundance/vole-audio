//! `court rocm-d0` — Phase J differential battery: scalar == ROCm.
//!
//! Runs the frozen fixture worlds (semantic, authored, mixed), entropy
//! decode jobs (literal + exact residual), and the mono->stereo upmix
//! transform on the AMD device through the Phase-J HIP runtime, asserting
//! byte-exact equality with the scalar oracle on every window — the AMD
//! mirror of `court cuda`'s parity core (D0 only; no endpoint claim here).
//!
//! Claim boundary (unchanged from Phase I): the differential executes only
//! when the probe chain is D0-ready AND an AMD device is present. On hosts
//! without an AMD compute device the court records the typed runtime chain
//! and returns `UNSUPPORTED_BY_HARDWARE` — the ROCm runtime is never
//! pretended to have executed. The code object itself is compile evidence
//! (ELF AMDGPU, bound sidecar/determinism), exactly as `court rocm`.

use crate::backend::entropy_flat::{flatten_literal_range, flatten_residual_range};
use crate::backend::flatten::flatten;
use crate::backend::rocm::kernel::{EntropyWorldRocm, RocmWorld, upmix_mono_dup};
use crate::backend::rocm::probe::RocmProbe;
use crate::backend::rocm::runtime::{DeviceBuffer, Rocm};
use crate::entropy::represent::{ModelMode, RepresentedLiteral, RepresentedResidual};
use crate::entropy::symbol::Symbolization;
use crate::error::{Error, Kind, Result};
use crate::eval::ScalarOracle;
use crate::evidence::counters::Counters;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder, RunTiming};
use crate::hash::sha256::{Sha256, hex};
use crate::object::ObjectStore;
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::residual::{Residual, ResidualModel};
use crate::sampler::world::World;
use crate::status::Verdict;
use crate::universe::arithmetic::sat_i32;
use crate::universe::layout::Layout;
use crate::universe::observation::observation_sha256;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const RATE_HZ: u32 = 48_000;
/// Default code object produced by scripts/build-rocm-device.sh.
const DEFAULT_ARTIFACT: &str = "scripts/out/vole_audio.amdgcn.elf";

fn artifact_bytes() -> Result<Option<(Vec<u8>, String)>> {
    let path = std::env::var("VOLE_ROCM_ARTIFACT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_ARTIFACT));
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    let sha = hex(&Sha256::digest(&bytes));
    Ok(Some((bytes, sha)))
}

/// One parity row: (label, passed, detail).
type ParityRow = (String, bool, Option<String>);

/// Flatten a fixture world, record its scalar-oracle window hashes, and
/// return the flat form.
fn fixture_flat(
    label: &str,
    store: &ObjectStore,
    world: &World,
    windows: &[(i64, usize)],
    hashes: &mut Vec<(String, String)>,
) -> Result<crate::backend::flatten::FlattenedWorld> {
    let oracle = ScalarOracle::new(world.clone());
    for &(start, frames) in windows {
        let out = oracle.observe(store, start, frames)?;
        let h = hex(&observation_sha256(&out));
        hashes.push((format!("{label}/scalar[{start},+{frames})"), h));
    }
    flatten(store, world)
}

/// Render parity over one flattened fixture: every window must be
/// byte-identical to the scalar oracle on the device.
#[allow(clippy::too_many_arguments)]
fn device_parity(
    label: &str,
    artifact: &[u8],
    store: &ObjectStore,
    world: &World,
    flat: &crate::backend::flatten::FlattenedWorld,
    windows: &[(i64, usize)],
    hashes: &mut Vec<(String, String)>,
    counters: &mut Counters,
) -> Result<Vec<ParityRow>> {
    let max = windows.iter().map(|(_, f)| *f).max().unwrap_or(0);
    let mut kw = RocmWorld::open(0, artifact, flat.clone(), max)?;
    let oracle = ScalarOracle::new(world.clone());
    let mut rows = Vec::new();
    for &(start, frames) in windows {
        let want = oracle.observe(store, start, frames)?;
        let mut out = vec![0i32; frames * usize::from(flat.output_channels)];
        match kw.render(start, frames, &mut out) {
            Ok(()) => {
                let h = hex(&observation_sha256(&out));
                hashes.push((format!("{label}/rocm[{start},+{frames})"), h));
                // Direct byte equality with the scalar oracle (the hashes
                // agree iff the bytes do; compare the bytes themselves).
                rows.push((
                    format!("{label}/[{start},+{frames})"),
                    out == want,
                    if out == want {
                        None
                    } else {
                        let nbad = out
                            .iter()
                            .zip(want.iter())
                            .filter(|(a, b)| a != b)
                            .take(3)
                            .map(|(a, b)| format!("{a} != {b}"))
                            .collect::<Vec<_>>()
                            .join("; ");
                        Some(format!("device != scalar: {nbad} (+ more)"))
                    },
                ));
            }
            Err(e) => rows.push((
                format!("{label}/[{start},+{frames})"),
                false,
                Some(format!("{e}")),
            )),
        }
    }
    counters.add_from(&kw.counters);
    Ok(rows)
}

/// Entropy decode parity: one literal job and one exact-residual job; the
/// host flat decode is the authority, the device decode must match it
/// byte-for-byte.
fn entropy_parity(artifact: &[u8], hashes: &mut Vec<(String, String)>) -> Result<Vec<ParityRow>> {
    let mut rows = Vec::new();

    // Literal job (delta_lane4 stereo tone; RAW fallback exercised by the
    // noise control in the entropy courts — here the parity target is the
    // exact decode of a coded job).
    let frames = 4096usize;
    let samples: Vec<i32> = (0..frames * 2)
        .map(|k| {
            let f = k / 2;
            let mut v: i64 = (((f as i64) * 7) % 2047) - 1023;
            v *= 8192;
            if k % 2 == 1 {
                v = -v;
            }
            sat_i32(v)
        })
        .collect();
    let d_lit = ObjectDescriptor::new(Representation::Literal, frames as u64, Layout::Stereo, None)
        .ok_or_else(|| Error::new(Kind::Internal, "literal descriptor rejected (static-valid)"))?;
    let rl = RepresentedLiteral::encode(
        d_lit,
        &samples,
        512,
        Symbolization::DeltaLane4,
        ModelMode::Inline,
        false,
    )?;
    let lit_job = flatten_literal_range(&rl, 0, frames as u32)?;
    let mut host = vec![0i32; lit_job.arena_samples];
    let mut scratch = vec![0u8; lit_job.max_page_scratch.max(1)];
    if !lit_job.decode_pages_host(&mut host, &mut scratch, 0..lit_job.pages.len()) {
        return Err(Error::new(
            crate::error::Kind::Internal,
            "host decode of the literal job failed",
        ));
    }
    if host != samples {
        return Err(Error::new(
            crate::error::Kind::Internal,
            "host flat decode != scalar representation (literal)",
        ));
    }
    let world = EntropyWorldRocm::open(0, artifact, &lit_job)?;
    match world.decode() {
        Ok(dev) if dev == host => {
            hashes.push((
                "entropy/literal/host".into(),
                hex(&observation_sha256(&host)),
            ));
            hashes.push((
                "entropy/literal/rocm".into(),
                hex(&observation_sha256(&dev)),
            ));
            rows.push(("entropy/literal".into(), true, None));
        }
        Ok(dev) => {
            let nbad = dev
                .iter()
                .zip(host.iter())
                .filter(|(a, b)| a != b)
                .take(3)
                .map(|(a, b)| format!("{a} != {b}"))
                .collect::<Vec<_>>()
                .join("; ");
            rows.push((
                "entropy/literal".into(),
                false,
                Some(format!("device != host decode: {nbad} (+ more)")),
            ));
        }
        Err(e) => rows.push(("entropy/literal".into(), false, Some(format!("{e}")))),
    }

    // Exact-residual job (periodic hypothesis + sparse exact corrections).
    let rframes = 4096usize;
    let cycle: Vec<i32> = (0..64).map(|i| i << 20).collect();
    let model = ResidualModel::Periodic { cycle };
    let mut intrinsic = vec![0i32; rframes];
    for (f, slot) in intrinsic.iter_mut().enumerate() {
        *slot = model.model_sample(f as u64);
    }
    let mut rs = 0xabcdef12u64;
    for _ in 0..300 {
        rs = rs
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = (rs % rframes as u64) as usize;
        let delta = (((rs >> 33) as i64) % 20000) as i32 - 10000;
        intrinsic[f] = sat_i32(i64::from(intrinsic[f]) + i64::from(delta));
    }
    let d_res = ObjectDescriptor::new(
        Representation::PredictorResidual,
        rframes as u64,
        Layout::Mono,
        None,
    )
    .ok_or_else(|| {
        Error::new(
            Kind::Internal,
            "residual descriptor rejected (static-valid)",
        )
    })?;
    let records = Residual::closing_residual(&intrinsic, 1, &model)
        .ok_or_else(|| Error::new(Kind::Internal, "closing_residual failed (static-valid)"))?;
    let residual = Residual::new(&d_res, model, records)
        .ok_or_else(|| Error::new(Kind::Internal, "Residual::new failed (static-valid)"))?;
    let rr = RepresentedResidual::encode(d_res, &residual, 512, ModelMode::Inline, false)?;
    let res_job = flatten_residual_range(&rr, 0, rframes as u32)?;
    let mut host_res = vec![0i32; res_job.arena_samples];
    let mut scratch_r = vec![0u8; res_job.max_page_scratch.max(1)];
    if !res_job.decode_pages_host(&mut host_res, &mut scratch_r, 0..res_job.pages.len()) {
        return Err(Error::new(
            crate::error::Kind::Internal,
            "host decode of the residual job failed",
        ));
    }
    if host_res != intrinsic {
        return Err(Error::new(
            crate::error::Kind::Internal,
            "host flat decode != scalar representation (residual)",
        ));
    }
    let world_res = EntropyWorldRocm::open(0, artifact, &res_job)?;
    match world_res.decode() {
        Ok(dev) if dev == host_res => {
            hashes.push((
                "entropy/residual/host".into(),
                hex(&observation_sha256(&host_res)),
            ));
            hashes.push((
                "entropy/residual/rocm".into(),
                hex(&observation_sha256(&dev)),
            ));
            rows.push(("entropy/residual".into(), true, None));
        }
        Ok(dev) => {
            let nbad = dev
                .iter()
                .zip(host_res.iter())
                .filter(|(a, b)| a != b)
                .take(3)
                .map(|(a, b)| format!("{a} != {b}"))
                .collect::<Vec<_>>()
                .join("; ");
            rows.push((
                "entropy/residual".into(),
                false,
                Some(format!("device != host decode: {nbad} (+ more)")),
            ));
        }
        Err(e) => rows.push(("entropy/residual".into(), false, Some(format!("{e}")))),
    }
    Ok(rows)
}

/// Upmix parity: the device mono->stereo duplication transform must equal
/// the host expansion exactly.
fn upmix_parity(artifact: &[u8]) -> Result<Vec<ParityRow>> {
    let session = Rocm::open(0)?;
    let module = session.load_module(artifact)?;
    let frames: u64 = 512;
    let channels: u64 = 2;
    // Deterministic mono source (exact-procedural ramp values).
    let mono: Vec<i32> = (0..frames)
        .map(|f| sat_i32(((f as i64) - 256) * 4_000 * 2))
        .collect();
    let expect: Vec<i32> = mono.iter().flat_map(|&s| [s, s]).collect();
    let src = DeviceBuffer::alloc(&session.device, (frames as usize) * 4)?;
    src.upload(bytemuck(&mono))?;
    let dst = DeviceBuffer::alloc(&session.device, (frames as usize * channels as usize) * 4)?;
    upmix_mono_dup(
        &session,
        &module,
        src.device_ptr(),
        dst.device_ptr(),
        frames,
        channels,
    )?;
    let mut bytes = vec![0u8; expect.len() * 4];
    dst.download(&mut bytes)?;
    // SAFETY: bytes holds `expect.len()` i32 codes.
    let got: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
        .collect();
    Ok(vec![(
        "upmix/mono-dup[0,+512)@2ch".into(),
        got == expect,
        if got == expect {
            None
        } else {
            Some("device upmix != host duplication".into())
        },
    )])
}

fn bytemuck<T: Sized>(v: &[T]) -> &[u8] {
    // SAFETY: plain-data i32 slices have no padding secrets.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub(crate) fn phase_j_d0_gate(probe: &RocmProbe) -> (Verdict, String) {
    use crate::backend::rocm::probe::KfdState as K;
    if probe.amd_gpus.is_empty() {
        return (
            Verdict::UnsupportedByHardware,
            "no AMD compute candidate in the sysfs PCI walk — the scalar == ROCm battery \
             cannot execute on this host"
                .into(),
        );
    }
    if !probe
        .amd_gpus
        .iter()
        .any(|g| g.driver.as_deref() == Some("amdgpu"))
    {
        return (
            Verdict::UnsupportedByHardware,
            "AMD candidate not bound to amdgpu — the KFD compute interface is unavailable".into(),
        );
    }
    match probe.kfd {
        K::Absent => {
            return (
                Verdict::UnsupportedByHardware,
                "AMD candidate bound to amdgpu but no KFD device interface".into(),
            );
        }
        K::PresentNotAccessible => {
            return (
                Verdict::Inconclusive,
                "KFD present but /dev/kfd not openable read-write (permissions/cgroup)".into(),
            );
        }
        K::Accessible => {}
    }
    // The Phase-J executor is HIP-only: readiness must be proven by a HIP
    // D0-ready row. An HSA-ready/HIP-unavailable system is UNSUPPORTED_BY_API
    // (review finding: classify() over both families must not authorize the
    // HIP executor).
    let hip_d0 = probe
        .compute
        .iter()
        .find(|a| a.soname.contains("libamdhip64") && a.d0_ready());
    match hip_d0 {
        Some(_) => (Verdict::Supported, String::new()),
        None => {
            let hsa_ready = probe
                .compute
                .iter()
                .any(|a| a.soname.contains("libhsa-runtime64") && a.d0_ready());
            let why = probe
                .compute
                .iter()
                .filter(|a| a.soname.contains("libamdhip64"))
                .map(|a| match &a.d0_missing {
                    Ok(m) if m.is_empty() => {
                        format!("{}: loaded, D0 complete (D1 {})", a.soname, a.d1_ready())
                    }
                    Ok(m) => format!("{}: D0 missing {}", a.soname, m.join(",")),
                    Err(e) => format!("{}: {e}", a.soname),
                })
                .collect::<Vec<_>>()
                .join("; ");
            if hsa_ready {
                (
                    Verdict::UnsupportedByApi,
                    format!(
                        "HIP runtime unavailable (only HSA rows are D0-ready; the Phase-J \
                         executor is HIP-only): {why}"
                    ),
                )
            } else {
                (
                    Verdict::UnsupportedByApi,
                    format!("no HIP runtime resolves its full D0 ABI surface: {why}"),
                )
            }
        }
    }
}

pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let t0 = Instant::now();
    let mut counters = Counters::new();
    let mut hashes: Vec<(String, String)> = Vec::new();
    let mut extras: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    // 0. Compile surface + runtime surface, captured together so every gate
    // receipt records what was checked (fail closed, never bare).
    let (compile, compile_satisfied, artifact_sha) = crate::courts::rocm::artifact_chain()?;
    extras.insert("compile_surface".into(), compile);
    let probe = RocmProbe::capture()?;
    let runtime = crate::courts::rocm::runtime_surface(&probe);
    extras.insert("runtime_surface".into(), runtime);
    // Gate receipt: full surfaces + typed detail for every non-executing
    // outcome (compile unsatisfied / artifact unreadable / no D0-ready
    // device). `gate_extras` is a snapshot so the closure never borrows
    // `extras` (the battery keeps extending it).
    let gate_extras = extras.clone();
    let emit_gate =
        |verdict: Verdict, detail: &str, artifact_sha: Option<String>| -> Result<Verdict> {
            let mut b = ReceiptBuilder::new("rocm-d0");
            b.result(verdict)
                .result_detail(format!("rocm-d0: {detail}"))
                .params(CourtParams {
                    universe: Some("vole.audio.u1".into()),
                    profile: Some("u1/v1".into()),
                    backend: Some("rocm-d0".into()),
                    content_kind: Some("phase-j scalar==rocm differential battery".into()),
                    ..Default::default()
                })
                .provenance(Provenance {
                    gpu_artifact_hash: artifact_sha,
                    ..Default::default()
                })
                .timing(RunTiming {
                    total_ns: Some(t0.elapsed().as_nanos() as i64),
                    ..Default::default()
                });
            for (k, v) in &gate_extras {
                b.extra(k, v.clone());
            }
            b.limitation(
                "The scalar == ROCm differential battery requires a D0-ready ROCm runtime AND an \
             AMD compute device; neither is pretended. This receipt records the typed runtime \
             chain and the bound compile surface only.",
            );
            let (_, path) = b.finish_write(receipts_root)?;
            println!("court rocm-d0: {verdict}");
            println!("  {detail}");
            println!("  receipt: {}", path.display());
            Ok(verdict)
        };
    if !compile_satisfied {
        return emit_gate(
            Verdict::Inconclusive,
            "compile surface unsatisfied (run scripts/build-rocm-device.sh on this tree); \
             the differential battery never executes against an unbound artifact",
            artifact_sha,
        );
    }
    let artifact = match artifact_bytes()? {
        Some(a) => a,
        None => {
            return emit_gate(
                Verdict::Inconclusive,
                "artifact bytes unreadable (VOLE_ROCM_ARTIFACT path?)",
                artifact_sha,
            );
        }
    };
    // HIP-specific D0 gate (the executor is HIP-only; an HSA-ready row does
    // not authorize it).
    let (gate_verdict, gate_detail) = phase_j_d0_gate(&probe);
    if gate_verdict != Verdict::Supported {
        return emit_gate(gate_verdict, &gate_detail, artifact_sha);
    }

    // 2. Differential battery (scalar == ROCm). The whole battery is one
    // fallible unit: any operational failure (e.g. HIP open/launch) becomes
    // a typed INCONCLUSIVE gate receipt, never a propagated error that
    // skips evidence (review finding: `?` must not escape the court).
    let battery_run = (|| -> Result<()> {
        let mut failed_rows: Vec<String> = Vec::new();
        let record = |rows: Vec<ParityRow>,
                      extras: &mut BTreeMap<String, serde_json::Value>,
                      failed: &mut Vec<String>| {
            for (label, ok, detail) in rows {
                extras.insert(format!("row/{label}"), json!(ok));
                if !ok {
                    failed.push(format!("{label}: {}", detail.unwrap_or_default()));
                }
            }
        };
        // 2a. Frozen procedural fixtures.
        {
            let (store, events) = crate::courts::semantic::semantic_court_fixture();
            let world = World::new(RATE_HZ, crate::courts::semantic::CHANNELS, events)?;
            let windows = [(0i64, 2400usize), (700, 900), (1600, 800)];
            let flat = fixture_flat("semantic", &store, &world, &windows, &mut hashes)?;
            let rows = device_parity(
                "semantic",
                &artifact.0,
                &store,
                &world,
                &flat,
                &windows,
                &mut hashes,
                &mut counters,
            )?;
            record(rows, &mut extras, &mut failed_rows);
        }
        {
            let (store, events) = crate::courts::authored::authored_court_fixture();
            let world = World::new(RATE_HZ, 1, events)?;
            let windows = [(0i64, 4000usize), (400, 3600), (1600, 2400)];
            let flat = fixture_flat("authored", &store, &world, &windows, &mut hashes)?;
            let rows = device_parity(
                "authored",
                &artifact.0,
                &store,
                &world,
                &flat,
                &windows,
                &mut hashes,
                &mut counters,
            )?;
            record(rows, &mut extras, &mut failed_rows);
        }
        {
            let (store, events) = crate::courts::simd::mixed_world();
            let world = World::new(RATE_HZ, 2, events)?;
            let windows = [(0i64, 8192usize), (1234, 700), (7000, 1192)];
            let flat = fixture_flat("mixed", &store, &world, &windows, &mut hashes)?;
            let rows = device_parity(
                "mixed",
                &artifact.0,
                &store,
                &world,
                &flat,
                &windows,
                &mut hashes,
                &mut counters,
            )?;
            record(rows, &mut extras, &mut failed_rows);
        }
        // 2b. Entropy decode parity.
        {
            let rows = entropy_parity(&artifact.0, &mut hashes)?;
            record(rows, &mut extras, &mut failed_rows);
        }
        // 2c. Upmix transform parity.
        {
            let rows = upmix_parity(&artifact.0)?;
            record(rows, &mut extras, &mut failed_rows);
        }
        if failed_rows.is_empty() {
            extras.insert("battery_passed".into(), json!(true));
        } else {
            extras.insert("battery_passed".into(), json!(false));
            extras.insert(
                "failed_rows".into(),
                json!(failed_rows.iter().collect::<Vec<_>>()),
            );
        }
        Ok(())
    })();
    if let Err(e) = battery_run {
        return emit_gate(
            Verdict::Inconclusive,
            &format!("differential battery could not execute: {e}"),
            artifact_sha,
        );
    }

    let passed = extras
        .get("battery_passed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let verdict = if passed {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    let failed_rows: Vec<String> = extras
        .get("failed_rows")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let detail = if passed {
        format!(
            "scalar == ROCm byte-exact on the frozen fixture windows ({} hashes), the entropy \
             decode jobs, and the mono->stereo upmix transform; {} quanta, {} launches, {} \
             gpu->host bytes (D0)",
            hashes.len(),
            counters.quanta_submitted,
            counters.kernel_launches,
            counters.gpu_to_host_pcm_bytes
        )
    } else {
        format!(
            "{} row(s) failed: {}",
            failed_rows.len(),
            failed_rows.join("; ")
        )
    };

    let mut b = ReceiptBuilder::new("rocm-d0");
    b.result(verdict)
        .result_detail(format!("rocm-d0: {detail}"))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1".into()),
            backend: Some("rocm-d0".into()),
            sample_rate_hz: Some(RATE_HZ),
            channels: Some(2),
            quantum_frames: Some(8192),
            content_kind: Some("phase-j scalar==rocm differential battery".into()),
            ..Default::default()
        })
        .counters(counters)
        .timing(RunTiming {
            total_ns: Some(t0.elapsed().as_nanos() as i64),
            ..Default::default()
        })
        .provenance(Provenance {
            gpu_artifact_hash: artifact_sha,
            exact_equality: Some(passed),
            ..Default::default()
        });
    for (k, v) in extras {
        b.extra(k, v);
    }
    b.limitation(
        "Phase-J D0 differential: frozen fixtures + entropy decode + upmix only (no D1 \
         endpoint claim here; that is court rocm-d1). Device timing on this host is not \
         claimed — host wall is recorded when a device executes.",
    );
    let (_, path) = b.finish_write(receipts_root)?;
    println!("court rocm-d0: {verdict}");
    println!("  compile_surface.satisfied=true runtime=D0-ready");
    for (k, h) in &hashes {
        println!("  {k}: {h}");
    }
    if !passed {
        for f in &failed_rows {
            eprintln!("    failed: {f}");
        }
    }
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::rocm::probe::{AmdGpu, KfdState, RuntimeAttempt};

    fn gpu() -> AmdGpu {
        AmdGpu {
            bdf: "0000:01:00.0".into(),
            vendor: Some("1002".into()),
            device: Some("73bf".into()),
            driver: Some("amdgpu".into()),
            class: Some("0x030000".into()),
        }
    }

    fn attempt(soname: &str, d0_ok: bool) -> RuntimeAttempt {
        RuntimeAttempt {
            soname: soname.to_string(),
            d0_missing: if d0_ok {
                Ok(vec![])
            } else {
                Ok(vec!["hipInit".into()])
            },
            d1_missing: Ok(vec![]),
        }
    }

    fn probe_with(amd: Vec<AmdGpu>, compute: Vec<RuntimeAttempt>) -> RocmProbe {
        RocmProbe {
            amd_gpus: amd,
            kfd: KfdState::Accessible,
            compute,
            telemetry: vec![],
        }
    }

    #[test]
    fn hsa_readiness_does_not_authorize_the_hip_executor() {
        // The generic probe classify() treats any D0-ready row (HIP or HSA)
        // as pending-execution; the Phase-J gate must NOT let an HSA-ready /
        // HIP-unavailable system authorize the HIP-only executor.
        let probe = probe_with(vec![gpu()], vec![attempt("libhsa-runtime64.so.1", true)]);
        let (classify_verdict, _) = probe.classify();
        assert_eq!(classify_verdict, Verdict::Inconclusive);
        let (v, d) = phase_j_d0_gate(&probe);
        assert_eq!(v, Verdict::UnsupportedByApi);
        assert!(d.contains("HIP runtime unavailable"), "{d}");
    }

    #[test]
    fn hip_d0_readiness_authorizes_the_battery() {
        let probe = probe_with(
            vec![gpu()],
            vec![
                attempt("libhsa-runtime64.so.1", true),
                attempt("libamdhip64.so.6", true),
            ],
        );
        let (v, _) = phase_j_d0_gate(&probe);
        assert_eq!(v, Verdict::Supported);
    }

    #[test]
    fn missing_device_is_hardware_unsupported() {
        let probe = probe_with(vec![], vec![]);
        let (v, _) = phase_j_d0_gate(&probe);
        assert_eq!(v, Verdict::UnsupportedByHardware);
    }
}
