//! `court rocm-d1` — Phase J ROCm D1 endpoint experiment.
//!
//! The Phase-J analogue of `court d1` / `court entropy-d1` on the AMD
//! surface: register the **actual ALSA mmap endpoint region** with
//! `hipHostRegister(hipHostRegisterMapped)` (the HIP analogue of
//! `cuMemHostRegister(DEVICEMAP)`), and have the AMD kernels write the final
//! S32 endpoint codes **directly into the registered region** — no D0 block,
//! no device->host sample transfer, no host PCM materialization copy — while
//! an equal-work D0 baseline (full device render -> one DtoH -> per-chunk
//! host copy into the ring) runs beside it so the bytes D1 removes are
//! measured on equal work.
//!
//! Sessions (silence-safe; quiet fixture amplitudes; opt-in audible content
//! via `VOLE_ROCM_D1_EMIT_AUDIO=1`):
//!
//! * `d0-baseline` — stereo object, whole-object device render + one DtoH +
//!   per-chunk host copies into the ring;
//! * `d1-stereo-direct` — `vole_render_d0` writes stereo codes directly into
//!   the registered region per chunk;
//! * `d1-mono-upmix` — a mono procedural object renders into a bounded
//!   2048-byte GPU arena per chunk and `vole_upmix_mono_dup` expands
//!   mono -> stereo on the device directly into the registered region (the
//!   observation-transform case; the arena is exposed as the bounded
//!   GPU-global sample intermediate).
//!
//! Claim boundary: the experiment executes only when the probe chain is
//! D1-ready (device + KFD + HIP with host registration) AND a playback
//! endpoint exists. Absent any link, the court records the typed cause and
//! returns the corresponding verdict — the ROCm runtime is never pretended
//! to have executed. The code object itself is compile evidence, exactly as
//! `court rocm`.
//!
//! Session policy is the frozen Phase-H falsification policy (see
//! `courts::d1`): zero xruns is the success criterion; any xrun/short
//! commit/suspend terminates the session as FAILED_DEADLINE with the exact
//! reason; recovery is deliberately not attempted mid-session.

use crate::audio::alsa::{
    AlsaPcm, EndpointInfo, EndpointRequest, classify_failure, list_playback_endpoints,
};
use crate::backend::cuda::direct::{RegisterRange, host_page_size};
use crate::backend::flatten::flatten;
use crate::backend::rocm::kernel::RocmWorld;
use crate::backend::rocm::probe::{KfdState, RocmProbe};
use crate::backend::rocm::runtime::DeviceBuffer;
use crate::backend::rocm::runtime::{HostRegistration, RegistrationAttempt, attempt_register};
use crate::error::Result;
use crate::eval::ScalarOracle;
use crate::evidence::receipt::{CourtParams, EndpointEvidence, Provenance, ReceiptBuilder};
use crate::hash::sha256::{Sha256, hex};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::{Literal, ObjectData, ObjectStore};
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::pan::Route;
use crate::sampler::scheduler::TimelineEvent;
use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
use crate::sampler::world::World;
use crate::status::Verdict;
use crate::universe::arithmetic::sat_i32;
use crate::universe::layout::Layout;
use crate::universe::observation::observation_sha256;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

const RATE_HZ: u32 = 48_000;
const CHANNELS: u32 = 2;
const PERIOD_FRAMES: u32 = 512;
const BUFFER_FRAMES: u32 = 4096;
const WAIT_TIMEOUT_MS: i32 = 2000;
const MAX_STALLS: u32 = 8;
const DEFAULT_ARTIFACT: &str = "scripts/out/vole_audio.amdgcn.elf";
/// Session length (frames per media object).
const OBJECT_FRAMES: usize = 4096;
/// Bytes of one mono chunk window (the bounded GPU arena).
const MONO_ARENA_BYTES: u64 = PERIOD_FRAMES as u64 * 4;

fn artifact_bytes() -> Result<Option<(Vec<u8>, String)>> {
    let path = std::env::var("VOLE_ROCM_ARTIFACT").unwrap_or_else(|_| DEFAULT_ARTIFACT.to_string());
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    let sha = hex(&Sha256::digest(&bytes));
    Ok(Some((bytes, sha)))
}

fn audible_requested() -> bool {
    std::env::var("VOLE_ROCM_D1_EMIT_AUDIO").as_deref() == Ok("1")
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

// ---------------------------------------------------------------------------
// Fixtures (deterministic, quiet by default)
// ---------------------------------------------------------------------------

fn voice_spec(object: crate::object::ObjectId, trigger: i64) -> VoiceSpec {
    let instant = EnvelopeParams::new(0, 0, crate::sampler::envelope::ENV_UNITY, 0).unwrap();
    VoiceSpec {
        object,
        trigger_frame: trigger,
        note_off: None,
        start_pos_q24: 0,
        rate_q24: 1 << 24,
        object_channel: 0,
        route: Route::Mono(0),
        gain_q16: (1 << 16) / 2,
        pan_q16: 0,
        envelope: instant,
        loop_mode: LoopMode::Off,
        interp: Interp::Linear,
    }
}

fn literal_world(layout: Layout, seed: u64, channels: u8) -> Result<(ObjectStore, World)> {
    let mut store = ObjectStore::new();
    let ch = usize::from(channels.max(1));
    let mut s = seed;
    let samples: Vec<i32> = (0..OBJECT_FRAMES * ch)
        .map(|k| {
            let f = k / ch;
            let mut v: i64 = (((f as i64) * 7) % 2047) - 1023;
            v *= 8192;
            let wobble = (((f as i64) * 31) % 127) - 63;
            v += wobble * 33;
            if k % ch == 1 {
                v = -v;
            }
            lcg(&mut s);
            sat_i32(v + ((s >> 40) as i64 % 65) - 32)
        })
        .collect();
    let d = ObjectDescriptor::new(Representation::Literal, OBJECT_FRAMES as u64, layout, None)
        .expect("static-valid descriptor");
    let id = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, samples).expect("static-valid literal")),
        )
        .expect("insert");
    let world = World::new(
        RATE_HZ,
        channels.max(1),
        vec![TimelineEvent::VoiceOn(voice_spec(id, 0))],
    )?;
    Ok((store, world))
}

fn device_request(dev: &EndpointInfo) -> EndpointRequest {
    EndpointRequest {
        pcm_name: dev.pcm_name.clone(),
        rate_hz: RATE_HZ,
        channels: CHANNELS,
        period_frames: PERIOD_FRAMES,
        buffer_frames: BUFFER_FRAMES,
    }
}

/// Endpoint region length in bytes (whole mapped ring).
fn region_bytes(pcm: &AlsaPcm) -> u64 {
    pcm.buffer_frames * pcm.request.frame_bytes()
}

/// Deterministic default preference (mirrors `courts::d1::candidate_rank`).
fn candidate_rank(dev: &EndpointInfo) -> (u8, String) {
    let kind = match (dev.driver.as_deref(), dev.pcm_label.as_deref()) {
        (Some("snd_hda_intel"), Some(l)) if l.contains("Analog") => 0u8,
        (Some("snd_hda_intel"), _) => 1,
        (Some("snd-usb-audio"), _) => 2,
        _ => 3,
    };
    (kind, dev.pcm_name.clone())
}

// ---------------------------------------------------------------------------
// Session loop (frozen Phase-H falsification policy; mirrors courts::d1)
// ---------------------------------------------------------------------------

struct SessionOut {
    frames_committed: u64,
    chunks: u64,
    xruns: u64,
    wall: Vec<f64>,
    shadow_exact: bool,
    drain_state: Option<String>,
    detail: Option<String>,
}

fn wall_stats(wall: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let mut wall = wall.to_vec();
    wall.sort_by(f64::total_cmp);
    let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
    (
        (!wall.is_empty()).then(|| r3(wall.iter().sum::<f64>() / wall.len() as f64)),
        wall.get(wall.len() / 2).copied().map(r3),
        wall.last().copied().map(r3),
    )
}

/// Drive one endpoint session: per chunk the `writer` produces the codes in
/// the ring region at the mmap offset; every commit is exact-transfer
/// checked and any xrun terminates the session.
fn run_session(
    pcm: &AlsaPcm,
    expected: &[i32],
    mut writer: impl FnMut(i64, u64, u64, &[i32]) -> std::result::Result<(), String>,
) -> SessionOut {
    let period = pcm.period_frames;
    let channels = u64::from(pcm.request.channels);
    let total = expected.len() as u64 / channels;
    let mut frames_committed: u64 = 0;
    let mut chunks = 0u64;
    let mut xruns = 0u64;
    let mut wall: Vec<f64> = Vec::new();
    let mut shadow_exact = true;
    let mut detail: Option<String> = None;
    let mut drain_state: Option<String> = None;
    let mut stalls = 0u32;
    let mut started = false;
    let xrun_terminate =
        |xruns: &mut u64, detail: &mut Option<String>, stalls: &mut u32, what: String| {
            *xruns += 1;
            *stalls = 0;
            *detail = Some(what);
        };
    while frames_committed < total {
        let want = (total - frames_committed).min(period);
        let ready = match pcm.wait(WAIT_TIMEOUT_MS) {
            Ok(true) => true,
            Ok(false) => match pcm.avail() {
                Ok(avail) if avail >= want => true,
                Ok(_) => {
                    stalls += 1;
                    if stalls >= MAX_STALLS {
                        detail = Some(format!(
                            "endpoint stall: no progress after {stalls} waits (state {})",
                            pcm.state_label()
                        ));
                        break;
                    } else {
                        false
                    }
                }
                Err(f) if f.is_xrun_class() => {
                    xrun_terminate(
                        &mut xruns,
                        &mut detail,
                        &mut stalls,
                        format!("xrun while waiting for avail: {f}"),
                    );
                    break;
                }
                Err(f) => {
                    detail = Some(format!("avail failed: {f}"));
                    break;
                }
            },
            Err(f) => {
                if f.is_xrun_class() {
                    xrun_terminate(
                        &mut xruns,
                        &mut detail,
                        &mut stalls,
                        format!("xrun during wait: {f}"),
                    );
                } else {
                    detail = Some(format!("wait failed: {f}"));
                }
                break;
            }
        };
        if !ready {
            continue;
        }
        let (offset, frames) = match pcm.mmap_begin(want) {
            Ok(r) => r,
            Err(f) if f.is_xrun_class() => {
                xrun_terminate(
                    &mut xruns,
                    &mut detail,
                    &mut stalls,
                    format!("xrun during mmap_begin: {f}"),
                );
                break;
            }
            Err(f) => {
                detail = Some(format!("mmap_begin failed: {f}"));
                break;
            }
        };
        if frames == 0 {
            detail = Some("mmap_begin returned 0 frames".into());
            break;
        }
        let chunk_codes = (frames * channels) as usize;
        let base = (frames_committed * channels) as usize;
        let expect = &expected[base..base + chunk_codes];
        let t0 = Instant::now();
        if let Err(e) = writer(frames_committed as i64, offset, frames, expect) {
            shadow_exact = false;
            detail = Some(e);
            break;
        }
        match pcm.mmap_commit(offset, frames) {
            Ok(()) => wall.push(t0.elapsed().as_secs_f64() * 1e3),
            Err(f) if f.is_xrun_class() => {
                xrun_terminate(
                    &mut xruns,
                    &mut detail,
                    &mut stalls,
                    format!("xrun-class mmap_commit failure: {f}"),
                );
                break;
            }
            Err(f) => {
                detail = Some(format!("mmap_commit failed: {f}"));
                break;
            }
        }
        frames_committed += frames;
        chunks += 1;
        stalls = 0;
        if !started {
            started = true;
            if let Err(f) = pcm.start() {
                if f.is_xrun_class() {
                    xrun_terminate(
                        &mut xruns,
                        &mut detail,
                        &mut stalls,
                        format!("xrun during snd_pcm_start: {f}"),
                    );
                } else {
                    detail = Some(format!("snd_pcm_start failed: {f}"));
                }
                break;
            }
        }
    }
    if frames_committed > 0 {
        match pcm.drain() {
            Ok(_) => drain_state = Some(pcm.state_label().to_string()),
            Err(f) => {
                detail = Some(format!("drain failed: {f}"));
                drain_state = Some(pcm.state_label().to_string());
            }
        }
    }
    SessionOut {
        frames_committed,
        chunks,
        xruns,
        wall,
        shadow_exact,
        drain_state,
        detail,
    }
}

/// In-place verification of the codes the GPU wrote at `offset`: compare the
/// mapped region against `expected` without a shadow sample buffer.
fn region_matches(
    pcm: &AlsaPcm,
    region_base: usize,
    offset: u64,
    frames: u64,
    expected: &[i32],
) -> std::result::Result<(), String> {
    let frame_bytes = pcm.request.frame_bytes();
    let addr = region_base + (offset * frame_bytes) as usize;
    let len = (frames * u64::from(pcm.request.channels)) as usize;
    if expected.len() != len {
        return Err("expected length mismatch".into());
    }
    // SAFETY: the registered ALSA mapping is live and readable; the GPU
    // writes are visible after the full device synchronization performed by
    // the render/upmix calls (mapped-host-memory coherency). This reads the
    // endpoint region in place — verification instrumentation, not
    // materialization traffic.
    let got: &[i32] = unsafe { std::slice::from_raw_parts(addr as *const i32, len) };
    if got != expected {
        let nbad = got
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a != b)
            .take(3)
            .map(|(a, b)| format!("{a} != {b}"))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!(
            "region != oracle at offset {offset}: {nbad} (+ more)"
        ));
    }
    Ok(())
}

fn copy_into_region(
    pcm: &AlsaPcm,
    offset: u64,
    frames: u64,
    codes: &[i32],
) -> std::result::Result<(), String> {
    let frame_bytes = pcm.request.frame_bytes();
    let addr = pcm.area_base + (offset * frame_bytes) as usize;
    let len = (frames * u64::from(pcm.request.channels)) as usize;
    if codes.len() != len {
        return Err("copy length mismatch".into());
    }
    // SAFETY: the region is the live ALSA mapping; `len` i32 slots are
    // writable at `addr` (mmap_begin returned offset..offset+frames).
    unsafe {
        std::ptr::copy_nonoverlapping(codes.as_ptr(), addr as *mut i32, len);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Session cell + verdict
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Cell {
    label: String,
    verdict: Verdict,
    detail: String,
    frames_committed: u64,
    chunks: u64,
    xruns: u64,
    wall_mean_ms: Option<f64>,
    wall_median_ms: Option<f64>,
    wall_max_ms: Option<f64>,
    mat_gpu_to_host: u64,
    mat_host_copy: u64,
    mat_host_resident: u64,
    device_sample_block: u64,
    endpoint_obs: u64,
    verification_read: u64,
    gpu_global_sample_intermediate_peak: u64,
    launches: u64,
    drain_state: Option<String>,
    exact_equality: bool,
    endpoint_hash: Option<String>,
}

fn verdict_for(out: &SessionOut, completed: bool, what: &str) -> (Verdict, String) {
    if !out.shadow_exact {
        (
            Verdict::FailedCorrectness,
            format!(
                "written codes differ from the scalar oracle: {}",
                out.detail.clone().unwrap_or_default()
            ),
        )
    } else if out.xruns > 0 {
        (
            Verdict::FailedDeadline,
            format!(
                "shadow equality held but {} xrun(s) occurred ({what})",
                out.xruns
            ),
        )
    } else if completed {
        (
            Verdict::Supported,
            format!(
                "registered the actual ALSA endpoint region ({what}); every chunk was written \
                 byte-exact vs the scalar oracle, committed with an exact transfer check, and \
                 drained without xrun"
            ),
        )
    } else {
        (
            Verdict::FailedDeadline,
            format!(
                "session incomplete: {}",
                out.detail.clone().unwrap_or_default()
            ),
        )
    }
}

fn verdict_rank(v: Verdict) -> u8 {
    use Verdict as V;
    match v {
        V::FailedCorrectness => 5,
        V::FailedDeadline => 4,
        V::Inconclusive => 3,
        V::UnsupportedByApi => 2,
        V::UnsupportedByHardware => 2,
        _ => 1,
    }
}

/// Aggregate session cells into the court verdict. CUDA-D1-aligned: once an
/// endpoint produced a completed D1 session (`completed` = its pcm name),
/// ONLY that endpoint's cells are verdict-bearing; every other candidate's
/// open/registration failures remain trial evidence (they are preserved in
/// the receipt but never sink the global result). With no completed D1
/// session, every cell counts.
fn aggregate_verdict(cells: &[Cell], completed: Option<&str>) -> Verdict {
    let scope: Vec<&Cell> = match completed {
        Some(name) => cells
            .iter()
            .filter(|c| c.label.starts_with(&format!("{name}/")))
            .collect(),
        None => cells.iter().collect(),
    };
    scope
        .iter()
        .map(|c| c.verdict)
        .max_by_key(|v| verdict_rank(*v))
        .unwrap_or(Verdict::Inconclusive)
}

// ---------------------------------------------------------------------------
// Court
// ---------------------------------------------------------------------------

pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let t0 = Instant::now();
    let audible = audible_requested();
    let mut extras: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    extras.insert(
        "mode".into(),
        json!(if audible { "audible" } else { "quiet" }),
    );

    // 0. Capture the full evidence picture first (compile surface, runtime
    // chain, endpoints) so every gate receipt records what was checked.
    let (compile, compile_satisfied, artifact_sha) = crate::courts::rocm::artifact_chain()?;
    extras.insert("compile_surface".into(), compile);
    let probe = RocmProbe::capture()?;
    let runtime = crate::courts::rocm::runtime_surface(&probe);
    extras.insert("runtime_surface".into(), runtime.clone());
    let endpoints = list_playback_endpoints();
    extras.insert(
        "endpoints".into(),
        json!(
            endpoints
                .iter()
                .map(|e| json!({
                    "pcm_name": e.pcm_name,
                    "driver": e.driver,
                    "pcm_label": e.pcm_label,
                    "card_description": e.card_description,
                }))
                .collect::<Vec<_>>()
        ),
    );

    // Gate receipt: full surfaces + typed detail for every non-executing
    // outcome (fail closed, never bare). `gate_extras` is a snapshot so the
    // closure does not borrow `extras` (which later paths keep extending).
    let gate_extras = extras.clone();
    let emit_gate = |verdict: Verdict, detail: &str, sha: Option<String>| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("rocm-d1");
        b.result(verdict)
            .result_detail(format!("rocm-d1: {detail}"))
            .params(CourtParams {
                universe: Some("vole.audio.u1".into()),
                profile: Some("u1/v1".into()),
                backend: Some("rocm-d1".into()),
                sample_rate_hz: Some(RATE_HZ),
                channels: Some(CHANNELS),
                quantum_frames: Some(PERIOD_FRAMES),
                content_kind: Some("phase-j rocm-d1 endpoint experiment".into()),
                ..Default::default()
            })
            .provenance(Provenance {
                gpu_artifact_hash: sha,
                ..Default::default()
            })
            .timing(crate::evidence::receipt::RunTiming {
                total_ns: Some(t0.elapsed().as_nanos() as i64),
                ..Default::default()
            });
        for (k, v) in &gate_extras {
            b.extra(k, v.clone());
        }
        b.limitation(
            "The ROCm D1 endpoint experiment requires a D1-ready HIP runtime (module/launch + \
                 hipHostRegister surface), an AMD compute device, and a hw: playback endpoint; \
                 none is pretended. This receipt records the typed cause chain and the bound \
                 compile surface only.",
        );
        let (_, path) = b.finish_write(receipts_root)?;
        println!("court rocm-d1: {verdict}");
        println!("  {detail}");
        println!("  receipt: {}", path.display());
        Ok(verdict)
    };
    if !compile_satisfied {
        return emit_gate(
            Verdict::Inconclusive,
            "compile surface unsatisfied (run scripts/build-rocm-device.sh on this tree)",
            artifact_sha,
        );
    }
    let artifact = match artifact_bytes()? {
        Some(a) => a,
        None => {
            return emit_gate(
                Verdict::Inconclusive,
                "artifact bytes unreadable",
                artifact_sha,
            );
        }
    };
    extras.insert("artifact_sha256".into(), json!(artifact.1));

    // 1. Runtime surface D1 readiness (host registration required).
    let hip_d1_ready = probe
        .compute
        .iter()
        .any(|a| a.soname.contains("libamdhip64") && a.d1_ready());

    // 2. Typed capability gate.
    let (gate_verdict, gate_detail): (Verdict, String) = {
        if probe.amd_gpus.is_empty() {
            (
                Verdict::UnsupportedByHardware,
                "no AMD compute candidate in the sysfs PCI walk — the ROCm D1 endpoint \
                 experiment cannot execute on this host"
                    .into(),
            )
        } else if !probe
            .amd_gpus
            .iter()
            .any(|g| g.driver.as_deref() == Some("amdgpu"))
        {
            (
                Verdict::UnsupportedByHardware,
                "AMD candidate not bound to amdgpu — the KFD compute interface is unavailable"
                    .into(),
            )
        } else if probe.kfd == KfdState::Absent {
            (
                Verdict::UnsupportedByHardware,
                "AMD candidate bound to amdgpu but no KFD device interface".into(),
            )
        } else if probe.kfd == KfdState::PresentNotAccessible {
            (
                Verdict::Inconclusive,
                "KFD present but /dev/kfd not openable read-write (permissions/cgroup)".into(),
            )
        } else if !hip_d1_ready {
            // A D0-ready stack without host registration is exactly the
            // reviewer-required split: "ROCm D0 READY; D1 UNSUPPORTED_BY_API".
            let d0_ready = probe.compute.iter().any(|a| a.d0_ready());
            let why = runtime
                .get("compute_runtime")
                .and_then(|v| v.as_array())
                .and_then(|rows| {
                    rows.iter().find(|r| {
                        r.get("soname")
                            .and_then(|s| s.as_str())
                            .is_some_and(|s| s.contains("libamdhip64"))
                    })
                })
                .and_then(|r| r.get("detail"))
                .and_then(|d| d.as_str())
                .map(str::to_string);
            (
                Verdict::UnsupportedByApi,
                format!(
                    "ROCm D1 endpoint path not available (D0 ready: {d0_ready}): {}",
                    why.unwrap_or_else(|| "HIP D1 additional symbols missing".into())
                ),
            )
        } else if endpoints.is_empty() {
            (
                Verdict::Inconclusive,
                "ROCm D1-ready device present but no hw: playback endpoint found".into(),
            )
        } else {
            (Verdict::Supported, String::new()) // eligible
        }
    };
    if gate_verdict != Verdict::Supported {
        return emit_gate(gate_verdict, &gate_detail, artifact_sha);
    }

    // 4. The experiment (D1-ready device present). Candidates: env override
    // or the deterministic rank order.
    let override_name = std::env::var("VOLE_ROCM_D1_DEVICE").ok();
    let mut candidates = endpoints.clone();
    if let Some(name) = &override_name {
        candidates.retain(|e| &e.pcm_name == name);
    }
    candidates.sort_by_key(candidate_rank);

    // Fixtures + scalar oracles.
    let (stereo_store, stereo_world) =
        literal_world(Layout::Stereo, 0x51de_cafe_00dd_ba11, CHANNELS as u8)?;
    let (mono_store, mono_world) = literal_world(Layout::Mono, 0xabcd_1234_5678_9ef0, 1)?;
    let stereo_oracle = ScalarOracle::new(stereo_world.clone());
    let mono_oracle = ScalarOracle::new(mono_world.clone());
    let stereo_media: Vec<i32> = stereo_oracle.observe(&stereo_store, 0, OBJECT_FRAMES)?;
    let mono_media: Vec<i32> = mono_oracle.observe(&mono_store, 0, OBJECT_FRAMES)?;
    // Endpoint-format expectation for the mono session: L = R = sample.
    let mono_stereo: Vec<i32> = mono_media.iter().flat_map(|&s| [s, s]).collect();
    let stereo_flat = flatten(&stereo_store, &stereo_world)?;
    let mono_flat = flatten(&mono_store, &mono_world)?;
    let endpoint_hash = hex(&observation_sha256(&stereo_media));
    let mono_endpoint_hash = hex(&observation_sha256(&mono_stereo));

    let mut cells: Vec<Cell> = Vec::new();
    // Endpoint whose D1 sessions are verdict-bearing (set once a D1
    // session ran there). Other candidates' open/registration failures stay
    // in `cells` as trial evidence but never sink the global result — the
    // D1 question is "is there a real endpoint on which this path works?"
    // (review finding: alignment with CUDA D1).
    let mut verdict_endpoint: Option<String> = None;
    let mut endpoint_cell = |label: &str,
                             verdict: Verdict,
                             detail: String,
                             out: Option<&SessionOut>,
                             traffic: (u64, u64, u64, u64, u64, u64, u64, u64),
                             hash: Option<String>| {
        // traffic = (gpu_to_host, host_copy, host_resident, device_block,
        // endpoint_obs, verification_read, intermediate_peak, launches)
        let (wm, wmed, wmax) = out
            .map(|o| wall_stats(&o.wall))
            .unwrap_or((None, None, None));
        cells.push(Cell {
            label: label.to_string(),
            verdict,
            detail,
            frames_committed: out.map(|o| o.frames_committed).unwrap_or(0),
            chunks: out.map(|o| o.chunks).unwrap_or(0),
            xruns: out.map(|o| o.xruns).unwrap_or(0),
            wall_mean_ms: wm,
            wall_median_ms: wmed,
            wall_max_ms: wmax,
            mat_gpu_to_host: traffic.0,
            mat_host_copy: traffic.1,
            mat_host_resident: traffic.2,
            device_sample_block: traffic.3,
            endpoint_obs: traffic.4,
            verification_read: traffic.5,
            gpu_global_sample_intermediate_peak: traffic.6,
            launches: traffic.7,
            drain_state: out.and_then(|o| o.drain_state.clone()),
            exact_equality: verdict == Verdict::Supported,
            endpoint_hash: hash,
        });
    };

    for dev in &candidates {
        let pcm = match AlsaPcm::open(&device_request(dev)) {
            Ok(p) => p,
            Err(f) => {
                let (v, d) = classify_failure(&f);
                endpoint_cell(
                    &format!("{}/open", dev.pcm_name),
                    v,
                    format!("{d}: {f}"),
                    None,
                    (0, 0, 0, 0, 0, 0, 0, 0),
                    None,
                );
                continue;
            }
        };
        let pcm_name = pcm.name();
        let base = pcm.area_base;
        let len = region_bytes(&pcm) as usize;
        let page = host_page_size();
        let range = RegisterRange::analyze(base, len, page);
        let frame_bytes = pcm.request.frame_bytes();

        // --- D0 baseline session (no registration needed). ---
        {
            let mut kw = match RocmWorld::open(0, &artifact.0, stereo_flat.clone(), OBJECT_FRAMES) {
                Ok(k) => k,
                Err(e) => {
                    endpoint_cell(
                        &format!("{pcm_name}/d0-baseline"),
                        Verdict::Inconclusive,
                        format!("RocmWorld::open: {e}"),
                        None,
                        (0, 0, 0, 0, 0, 0, 0, 0),
                        None,
                    );
                    drop(pcm);
                    continue;
                }
            };
            let mut media = vec![0i32; OBJECT_FRAMES * CHANNELS as usize];
            if let Err(e) = kw.render(0, OBJECT_FRAMES, &mut media) {
                endpoint_cell(
                    &format!("{pcm_name}/d0-baseline"),
                    Verdict::FailedCorrectness,
                    format!("full-object device render failed: {e}"),
                    None,
                    (0, 0, 0, 0, 0, 0, 0, kw.counters.kernel_launches),
                    None,
                );
                drop(pcm);
                continue;
            }
            if media != stereo_media {
                endpoint_cell(
                    &format!("{pcm_name}/d0-baseline"),
                    Verdict::FailedCorrectness,
                    "device D0 render != scalar oracle".into(),
                    None,
                    (
                        kw.counters.gpu_to_host_pcm_bytes,
                        kw.counters.host_pcm_copy_bytes,
                        kw.counters.host_pcm_resident_peak_bytes,
                        kw.counters.device_sample_block_bytes,
                        0,
                        0,
                        0,
                        kw.counters.kernel_launches,
                    ),
                    None,
                );
                drop(pcm);
                continue;
            }
            let traffic = (
                kw.counters.gpu_to_host_pcm_bytes,
                kw.counters.host_pcm_copy_bytes,
                kw.counters.host_pcm_resident_peak_bytes,
                kw.counters.device_sample_block_bytes,
                0u64,
                0u64,
                0u64,
                kw.counters.kernel_launches,
            );
            drop(kw);
            let mut verification_read = 0u64;
            let out = run_session(&pcm, &stereo_media, |_, offset, frames, expect| {
                copy_into_region(&pcm, offset, frames, expect)?;
                verification_read += frames * u64::from(CHANNELS) * 4;
                region_matches(&pcm, base, offset, frames, expect)
            });
            let (v, d) = verdict_for(
                &out,
                out.frames_committed as usize == OBJECT_FRAMES,
                "D0 baseline",
            );
            endpoint_cell(
                &format!("{pcm_name}/d0-baseline"),
                v,
                d,
                Some(&out),
                (
                    traffic.0,
                    traffic.1,
                    traffic.2,
                    traffic.3,
                    out.frames_committed * u64::from(CHANNELS) * 4,
                    verification_read,
                    0,
                    traffic.7,
                ),
                Some(endpoint_hash.clone()),
            );
        }

        // --- D1 sessions: register the exact endpoint region. ---
        // The registration (drop = hipHostUnregister) is bound with the
        // world (drop = ROCm session teardown) in one struct so teardown
        // order cannot regress: registration first, world last.
        struct Direct {
            registration: HostRegistration,
            world: RocmWorld,
        }
        let mk_direct = |artifact: &[u8],
                         flat: &crate::backend::flatten::FlattenedWorld,
                         max_frames: usize|
         -> std::result::Result<Direct, (Verdict, String)> {
            let world = match RocmWorld::open(0, artifact, flat.clone(), max_frames) {
                Ok(w) => w,
                Err(e) => return Err((Verdict::Inconclusive, format!("RocmWorld::open: {e}"))),
            };
            let base64 = base as u64;
            // SAFETY: the ALSA mapping is live for the whole trial (pcm
            // holds it); the registration dies before pcm in this scope.
            let attempt = unsafe { attempt_register(&world.session().device, base64, len) };
            match attempt {
                RegistrationAttempt::Registered(reg) => Ok(Direct {
                    registration: reg,
                    world,
                }),
                RegistrationAttempt::MissingSymbol(m) => Err((
                    Verdict::UnsupportedByApi,
                    format!("HIP D1 surface missing symbols: {m}"),
                )),
                RegistrationAttempt::Failed { rc, message } => {
                    let (v, d) = HostRegistration::classify(rc, true);
                    Err((v, format!("{d}: {message}")))
                }
            }
        };

        // D1 stereo-direct.
        let mut direct = match mk_direct(&artifact.0, &stereo_flat, PERIOD_FRAMES as usize) {
            Ok(d) => d,
            Err((v, e)) => {
                endpoint_cell(
                    &format!("{pcm_name}/registration"),
                    v,
                    e,
                    None,
                    (0, 0, 0, 0, 0, 0, 0, 0),
                    None,
                );
                drop(pcm);
                continue;
            }
        };
        let dev_ptr = direct.registration.device_ptr.unwrap_or(0);
        let region_base = direct.registration.host_ptr as usize;
        let mut verification_read = 0u64;
        let out = {
            let w = &mut direct.world;
            run_session(&pcm, &stereo_media, |start, offset, frames, expect| {
                let chunk_dev = dev_ptr + offset * frame_bytes;
                w.render_direct(start, frames as usize, chunk_dev)
                    .map_err(|e| e.to_string())?;
                verification_read += frames * u64::from(CHANNELS) * 4;
                region_matches(&pcm, region_base, offset, frames, expect)
            })
        };
        let (v, d) = verdict_for(
            &out,
            out.frames_committed as usize == OBJECT_FRAMES,
            "D1 stereo direct render",
        );
        let w = &direct.world;
        endpoint_cell(
            &format!("{pcm_name}/d1-stereo-direct"),
            v,
            d,
            Some(&out),
            (
                w.counters.gpu_to_host_pcm_bytes,
                w.counters.host_pcm_copy_bytes,
                0,
                0,
                w.counters.endpoint_observation_bytes,
                verification_read,
                0,
                w.counters.kernel_launches,
            ),
            Some(endpoint_hash.clone()),
        );
        // A D1 session ran on this endpoint: from here on, its cells are
        // verdict-bearing; other candidates' rows stay as trial evidence.
        verdict_endpoint = Some(pcm_name.clone());
        drop(direct);
        println!(
            "court rocm-d1: D1 session on {pcm_name} (registered 0x{base:x}+{len}; \
             range {}; device ptr 0x{dev_ptr:x})",
            range.label()
        );
        let _ = range;

        // D1 mono-upmix: mono world -> bounded 2048 B GPU arena per chunk ->
        // device upmix -> registered region.
        let mut mono_direct = match mk_direct(&artifact.0, &mono_flat, PERIOD_FRAMES as usize) {
            Ok(d) => d,
            Err((v, e)) => {
                endpoint_cell(
                    &format!("{pcm_name}/d1-mono-upmix"),
                    v,
                    e,
                    None,
                    (0, 0, 0, 0, 0, 0, 0, 0),
                    None,
                );
                drop(pcm);
                continue;
            }
        };
        let mono_dev = mono_direct.registration.device_ptr.unwrap_or(0);
        // Bound mono arena (one 512-frame chunk); declared peak exposure.
        let arena = match DeviceBuffer::alloc(
            &mono_direct.world.session().device,
            MONO_ARENA_BYTES as usize,
        ) {
            Ok(a) => a,
            Err(e) => {
                endpoint_cell(
                    &format!("{pcm_name}/d1-mono-upmix"),
                    Verdict::Inconclusive,
                    format!("mono arena alloc failed: {e}"),
                    None,
                    (0, 0, 0, 0, 0, 0, 0, 0),
                    None,
                );
                drop(pcm);
                continue;
            }
        };
        let arena_dev = arena.device_ptr();
        let mut verification_read_m = 0u64;
        let out = {
            let w = &mut mono_direct.world;
            run_session(&pcm, &mono_stereo, |start, offset, frames, expect| {
                // 1) mono render into the bounded arena (window scratch, not
                // endpoint observation).
                w.render_to(start, frames as usize, arena_dev)
                    .map_err(|e| e.to_string())?;
                // 2) device mono->stereo expansion into the region chunk
                //    (this write IS endpoint observation).
                w.upmix(
                    arena_dev,
                    mono_dev + offset * frame_bytes,
                    frames,
                    u64::from(CHANNELS),
                )
                .map_err(|e| e.to_string())?;
                verification_read_m += frames * u64::from(CHANNELS) * 4;
                region_matches(&pcm, base, offset, frames, expect)
            })
        };
        let (v, d) = verdict_for(
            &out,
            out.frames_committed as usize == OBJECT_FRAMES,
            "D1 mono render + device upmix",
        );
        let w = &mono_direct.world;
        endpoint_cell(
            &format!("{pcm_name}/d1-mono-upmix"),
            v,
            d,
            Some(&out),
            (
                w.counters.gpu_to_host_pcm_bytes,
                w.counters.host_pcm_copy_bytes,
                0,
                0,
                w.counters.endpoint_observation_bytes,
                verification_read_m,
                MONO_ARENA_BYTES,
                w.counters.kernel_launches,
            ),
            Some(mono_endpoint_hash.clone()),
        );
        drop(mono_direct);
        drop(pcm);
        break; // first successfully opened endpoint runs the experiment
    }

    // Aggregate: once an endpoint produced a D1 session, only that
    // endpoint's cells are verdict-bearing; otherwise every cell counts
    // (CUDA-D1-aligned semantics — review finding).
    let worst = aggregate_verdict(&cells, verdict_endpoint.as_deref());
    let detail = if worst == Verdict::Supported {
        format!(
            "ROCm D1: registered the actual ALSA endpoint region with \
             hipHostRegister(hipHostRegisterMapped); final codes written directly by the AMD \
             kernels; {} session cells, all SUPPORTED, zero xruns",
            cells.len()
        )
    } else {
        format!(
            "ROCm D1 experiment: worst session cell is {worst} ({} cells)",
            cells.len()
        )
    };

    let mut b = ReceiptBuilder::new("rocm-d1");
    b.result(worst)
        .result_detail(format!("rocm-d1: {detail}"))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1".into()),
            backend: Some("rocm-d1".into()),
            sample_rate_hz: Some(RATE_HZ),
            channels: Some(CHANNELS),
            quantum_frames: Some(PERIOD_FRAMES),
            content_kind: Some("phase-j rocm-d1 endpoint experiment".into()),
            ..Default::default()
        })
        .provenance(Provenance {
            gpu_artifact_hash: artifact_sha,
            endpoint_hash: Some(endpoint_hash.clone()),
            exact_equality: Some(worst == Verdict::Supported),
            ..Default::default()
        })
        .endpoint(EndpointEvidence {
            directness: Some(
                if worst == Verdict::Supported {
                    "D1"
                } else {
                    "not-executed"
                }
                .into(),
            ),
            ..Default::default()
        })
        .timing(crate::evidence::receipt::RunTiming {
            total_ns: Some(t0.elapsed().as_nanos() as i64),
            ..Default::default()
        });
    for (k, v) in extras {
        b.extra(k, v);
    }
    b.extra(
        "sessions",
        json!(cells
            .iter()
            .map(|c| json!({
                "label": c.label,
                "verdict": c.verdict.label(),
                "detail": c.detail,
                "frames_committed": c.frames_committed,
                "chunks": c.chunks,
                "xruns": c.xruns,
                "wall_mean_ms": c.wall_mean_ms,
                "wall_median_ms": c.wall_median_ms,
                "wall_max_ms": c.wall_max_ms,
                "materialization_gpu_to_host_bytes": c.mat_gpu_to_host,
                "materialization_host_copy_bytes": c.mat_host_copy,
                "host_pcm_resident_peak_bytes": c.mat_host_resident,
                "device_sample_block_bytes": c.device_sample_block,
                "endpoint_observation_bytes": c.endpoint_obs,
                "verification_read_bytes": c.verification_read,
                "gpu_global_sample_intermediate_peak_bytes": c.gpu_global_sample_intermediate_peak,
                "kernel_launches": c.launches,
                "drain_state": c.drain_state,
                "exact_equality": c.exact_equality,
                "endpoint_hash": c.endpoint_hash,
            }))
            .collect::<Vec<_>>()),
    );
    b.limitation(
        "Phase-J D1 experiment on the ROCm surface. On this host the typed gate records why \
         nothing executed (compile surface + runtime chain + endpoints are all evidenced).",
    );
    let (_, path) = b.finish_write(receipts_root)?;
    println!("court rocm-d1: {worst}");
    println!("  receipt: {}", path.display());
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(label: &str, v: Verdict) -> Cell {
        Cell {
            label: label.to_string(),
            verdict: v,
            detail: String::new(),
            frames_committed: 0,
            chunks: 0,
            xruns: 0,
            wall_mean_ms: None,
            wall_median_ms: None,
            wall_max_ms: None,
            mat_gpu_to_host: 0,
            mat_host_copy: 0,
            mat_host_resident: 0,
            device_sample_block: 0,
            endpoint_obs: 0,
            verification_read: 0,
            gpu_global_sample_intermediate_peak: 0,
            launches: 0,
            drain_state: None,
            exact_equality: v == Verdict::Supported,
            endpoint_hash: None,
        }
    }

    #[test]
    fn candidate_a_failure_does_not_sink_endpoint_b_success() {
        // CUDA-D1-aligned semantics: endpoint A fails registration (trial
        // evidence only); endpoint B completes an exact D1 session. The
        // top-level verdict must be B's SUPPORTED, with A preserved as a
        // row — not the worst cell across candidates.
        let cells = vec![
            cell("hw:0,0/open", Verdict::UnsupportedByHardware),
            cell("hw:0,1/registration", Verdict::UnsupportedByApi),
            cell("hw:2,0/d0-baseline", Verdict::Supported),
            cell("hw:2,0/d1-stereo-direct", Verdict::Supported),
            cell("hw:2,0/d1-mono-upmix", Verdict::Supported),
        ];
        assert_eq!(
            aggregate_verdict(&cells, Some("hw:2,0")),
            Verdict::Supported
        );
        // Without a completed D1 session every row counts (strict).
        assert_eq!(aggregate_verdict(&cells, None), Verdict::UnsupportedByApi);
    }

    #[test]
    fn completed_but_failed_d1_session_is_the_verdict() {
        let cells = vec![
            cell("hw:0,0/registration", Verdict::UnsupportedByApi),
            cell("hw:2,0/d0-baseline", Verdict::Supported),
            cell("hw:2,0/d1-stereo-direct", Verdict::FailedDeadline),
        ];
        assert_eq!(
            aggregate_verdict(&cells, Some("hw:2,0")),
            Verdict::FailedDeadline
        );
    }
}
