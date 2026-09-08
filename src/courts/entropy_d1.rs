//! `court entropy-d1` — fused entropy -> CUDA -> D1 endpoint (H.2.18–H.2.20,
//! H.2.53).
//!
//! The flagship systems test of H.2: entropy-coded SampleObject state (a
//! literal tone, a procedural+residual object, and a noise control) stays
//! GPU-resident, and per bounded 512-frame observation window the
//! `vole_entropy_decode` kernel decodes **only the pages intersecting the
//! window** (two 256-frame pages, one thread per page) and writes the exact
//! final sample codes straight into the actual registered ALSA mmap endpoint
//! region (`cuMemHostRegister(DEVICEMAP)`), which is then committed to the
//! DAC — the Phase-H D1 mechanism carrying the H.2 entropy representation.
//!
//! The D1 sessions perform:
//!
//! * no D0 global sample block,
//! * no full-object decoded waveform (only the window's pages decode; the
//!   page-local decode is transient device scratch),
//! * no GPU->host sample materialization,
//! * no host PCM materialization copy.
//!
//! A D0 entropy baseline for the same object/frames/endpoint (whole-object
//! device decode -> one DtoH -> per-chunk host copy into the ring) runs beside
//! the D1 sessions so the bytes D1 removes are measured on equal work
//! (H.2.20). Verification instrumentation reads the committed ring region in
//! place and compares it with the scalar slice; it is separately accounted and
//! never conflated with materialization.
//!
//! The residual object is mono (Phase-E `Periodic` hypotheses are mono-only);
//! the endpoint is stereo, so the fused D1 path expands mono -> stereo by
//! duplication (L = R) **on the device** with a tiny second kernel
//! (`vole_upmix_mono_dup`) — a sampler/mix transform at the observation
//! boundary, never a host materialization. The D0 residual baseline performs
//! the same expansion host-side as part of its materialization copy.
//!
//! Silence-safe by default (quiet fixture amplitudes); opt-in audible content
//! via `VOLE_ENTROPY_D1_EMIT_AUDIO=1` (full-range noise is only emitted
//! then). GPU/ALSA absent -> honest receipts.

use crate::audio::alsa::{
    AlsaPcm, EndpointInfo, EndpointRequest, classify_failure, list_playback_endpoints,
};
use crate::backend::cuda::direct::{HostRegistration, RegistrationAttempt, attempt_register};
use crate::backend::cuda::driver::{Cuda, DeviceBuffer, Function};
use crate::backend::cuda::entropy::EntropyWorld;
use crate::backend::entropy_flat::{flatten_literal_range, flatten_residual_range};
use crate::entropy::represent::PageKind;
use crate::entropy::represent::{ModelMode, RepresentedLiteral, RepresentedResidual};
use crate::entropy::symbol::Symbolization;
use crate::error::Result;
use crate::evidence::receipt::{CourtParams, EndpointEvidence, ReceiptBuilder};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::residual::{Residual, ResidualModel};
use crate::status::Verdict;
use crate::universe::arithmetic::sat_i32;
use crate::universe::layout::Layout;
use std::path::Path;
use std::time::Instant;

const RATE_HZ: u32 = 48_000;
const CHANNELS: u32 = 2;
const PERIOD_FRAMES: u32 = 512;
const BUFFER_FRAMES: u32 = 4096;
const WAIT_TIMEOUT_MS: i32 = 2000;
const MAX_STALLS: u8 = 8;
const DEFAULT_PTX: &str = "scripts/out/vole_audio.ptx";
/// Entropy page frame size of every fixture representation: half the period,
/// so each 512-frame observation window decodes **two** independent pages in
/// parallel (one thread per page). The single-thread serial rANS decode chain
/// on an idle-clock GPU is latency-bound (~5 ms per 512-frame window here;
/// more pages per window does not help once the decode is latency-chain
/// limited), so the court gives the DMA deep buffering (8 x 512-frame
/// periods) and the per-chunk decode jitter is absorbed by queued periods.
/// The endpoint access geometry (S32_LE interleaved, exact 48 kHz) stays
/// identical to the sealed Phase-H D1 court.
const PAGE_FRAMES: u32 = 256;
/// Session length: a multiple of the page size, so every chunk is exactly one
/// page-aligned window.
const OBJECT_FRAMES: usize = 8 * PERIOD_FRAMES as usize; // 4096 frames

/// Number of pages covering the whole object (per representation).
const OBJECT_PAGES: usize = OBJECT_FRAMES / PAGE_FRAMES as usize; // 16

fn ptx_bytes() -> Result<Option<(Vec<u8>, String)>> {
    use crate::error::{Error, Kind};
    let path = std::env::var("VOLE_CUDA_PTX").unwrap_or_else(|_| DEFAULT_PTX.to_string());
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::new(Kind::Io, format!("read {path}: {e}"))),
    };
    let sha = crate::hash::sha256::hex(&crate::hash::sha256::Sha256::digest(&bytes));
    Ok(Some((bytes, sha)))
}

fn audible_requested() -> bool {
    std::env::var("VOLE_ENTROPY_D1_EMIT_AUDIO").as_deref() == Ok("1")
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

// ---------------------------------------------------------------------------
// Fixtures (deterministic; quiet by default; content-rich in both modes)
// ---------------------------------------------------------------------------

/// Stereo literal tone: a saw-like ramp with a wobble and LCG jitter.
/// Quiet amplitude `<= 2^12`; audible amplitude `<= ~2^24`.
fn literal_tone(audible: bool) -> Vec<i32> {
    let scale: i64 = if audible { 1 } else { 1 << 12 };
    let ch = CHANNELS as usize;
    let mut s = 0x51de_cafe_00dd_ba11u64;
    (0..OBJECT_FRAMES * ch)
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
            v + ((s >> 40) as i64 % 65) - 32
        })
        .map(|v| (v / scale) as i32)
        .collect()
}

/// Stereo high-entropy control. Full 32-bit range when audible (the RAW
/// negative control; page coding recorded per page). Quiet default keeps the
/// peak `<= 2^20` (~ -66 dBFS) by discarding low-order entropy bits.
fn noise_stereo(audible: bool) -> Vec<i32> {
    let ch = CHANNELS as usize;
    let mut s = 0x1234_5678_9abc_def0u64;
    (0..OBJECT_FRAMES * ch)
        .map(|_| {
            lcg(&mut s);
            let full = (s as u32) as i32;
            if audible { full } else { full >> 11 }
        })
        .collect()
}

/// Mono procedural fixture: a periodic hypothesis plus sparse exact
/// corrections (Phase-E residual semantics untouched). Unit amplitude is
/// `(f % 64 - 32) * 128` plus `+-1024` corrections; audible mode multiplies
/// every code by `AUDIBLE_GAIN` (exact: closure is recomputed on the scaled
/// intrinsic under the scaled model).
const AUDIBLE_GAIN: i64 = 1 << 12;

fn residual_fixture(audible: bool) -> (RepresentedResidual, Vec<i32>, usize) {
    let gain: i64 = if audible { AUDIBLE_GAIN } else { 1 };
    let mut intrinsic = vec![0i32; OBJECT_FRAMES];
    let mut rs = 0x1234_5678_9abc_def0u64;
    // Unit-domain intrinsic: deterministic periodic hypothesis + sparse
    // exact corrections (Phase-E semantics untouched), then scaled exactly.
    let mut unit = vec![0i32; OBJECT_FRAMES];
    for (f, slot) in unit.iter_mut().enumerate() {
        *slot = (((f as i64 % 64) - 32) * 128) as i32;
    }
    // ~256 sparse corrections over 4096 frames (~16 per 256-frame page).
    for _ in 0..(OBJECT_FRAMES / 16) {
        lcg(&mut rs);
        let f = (rs % OBJECT_FRAMES as u64) as usize;
        let delta = (((rs >> 33) as i64) % 2048) - 1024;
        unit[f] = sat_i32(i64::from(unit[f]) + delta);
    }
    for (f, slot) in intrinsic.iter_mut().enumerate() {
        *slot = sat_i32(i64::from(unit[f]) * gain);
    }
    let cycle_unit: Vec<i32> = (0..64).map(|i| ((i as i64 - 32) * 128) as i32).collect();
    let cycle: Vec<i32> = cycle_unit
        .iter()
        .map(|&v| sat_i32(i64::from(v) * gain))
        .collect();
    let model = ResidualModel::Periodic { cycle };
    let descriptor = ObjectDescriptor::new(
        Representation::PredictorResidual,
        OBJECT_FRAMES as u64,
        Layout::Mono,
        None,
    )
    .unwrap();
    let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
    let n_corrections = records.len();
    let residual = Residual::new(&descriptor, model, records).unwrap();
    let rr =
        RepresentedResidual::encode(descriptor, &residual, PAGE_FRAMES, ModelMode::Inline, false)
            .expect("residual encode");
    (rr, intrinsic, n_corrections)
}

/// Mono codes expanded to stereo by duplication (L = R): the observation
/// policy applied at the sampler boundary for the D1 residual sessions.
fn expand_mono_to_stereo(mono: &[i32]) -> Vec<i32> {
    mono.iter().flat_map(|&v| [v, v]).collect()
}

// ---------------------------------------------------------------------------
// Endpoint plumbing
// ---------------------------------------------------------------------------

fn candidate_rank(dev: &EndpointInfo) -> (u8, String) {
    let kind = match (dev.driver.as_deref(), dev.pcm_label.as_deref()) {
        (Some("snd_hda_intel"), Some(l)) if l.contains("Analog") => 0u8,
        (Some("snd_hda_intel"), _) => 1,
        (Some("snd-usb-audio"), _) => 2,
        _ => 3,
    };
    (kind, dev.pcm_name.clone())
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

fn region_bytes(pcm: &AlsaPcm) -> u64 {
    pcm.buffer_frames * pcm.request.frame_bytes()
}

fn open_validated(request: &EndpointRequest) -> std::result::Result<AlsaPcm, (Verdict, String)> {
    let pcm = AlsaPcm::open(request).map_err(|f| classify_failure(&f))?;
    // Blocking handle: the run loop paces with snd_pcm_wait and the final
    // snd_pcm_drain must actually block until the ring empties (a nonblock
    // drain returns -EAGAIN, which would misreport a clean session).
    pcm.set_blocking()
        .map_err(|e| (Verdict::Inconclusive, format!("set_blocking: {e}")))?;
    let snap = pcm.snapshot();
    if !snap.area_layout_validated {
        return Err((
            Verdict::UnsupportedByHardware,
            "interleaved S32_LE channel-area geometry not validated".into(),
        ));
    }
    Ok(pcm)
}

/// Register the live ALSA ring with CUDA (DEVICEMAP) and obtain the device
/// pointer. Returns an evidence-classified failure otherwise.
fn register_ring(
    cuda: &Cuda,
    pcm: &AlsaPcm,
) -> std::result::Result<HostRegistration, (Verdict, String)> {
    let bytes = region_bytes(pcm) as usize;
    // SAFETY: pcm keeps the ALSA mapping alive for the session.
    match unsafe { attempt_register(&cuda.fns, pcm.area_base, bytes) } {
        RegistrationAttempt::Registered(r) => Ok(r),
        RegistrationAttempt::MissingSymbol(m) => Err((
            Verdict::UnsupportedByApi,
            format!("missing driver symbols: {m}"),
        )),
        RegistrationAttempt::Failed { rc, message } => {
            let (v, base) = HostRegistration::classify(
                rc,
                cuda.fns.d1_surface_complete(),
                cuda.device.host_register_supported != 0,
            );
            Err((v, format!("{base} (rc {rc}: {message})")))
        }
    }
}

/// In-place verification of the codes written at `offset`: compare the mapped
/// region against `expected` without building a shadow sample buffer.
fn region_matches(
    pcm: &AlsaPcm,
    offset: u64,
    frames: u64,
    expected: &[i32],
) -> std::result::Result<(), String> {
    let frame_bytes = pcm.request.frame_bytes();
    let addr = pcm.area_base + (offset * frame_bytes) as usize;
    let len = (frames * u64::from(pcm.request.channels)) as usize;
    if expected.len() != len {
        return Err(format!(
            "expected length mismatch ({} != {len})",
            expected.len()
        ));
    }
    // SAFETY: the registered ALSA mapping is live and readable; GPU writes are
    // visible after cuStreamSynchronize (recorded coherency contract). This
    // reads the endpoint region in place — verification instrumentation, not
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

// ---------------------------------------------------------------------------
// Session pacing (mirrors the Phase-H court policy)
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct SessionTraffic {
    mat_gpu_to_host: u64,
    mat_host_copy: u64,
    mat_host_resident_peak: u64,
    endpoint_obs: u64,
    verification_read: u64,
    verification_shadow_copy: u64,
    /// Verdict-bearing (measurement) kernel launches only.
    launches: u64,
    /// Sustained-clock warm-up launches (court methodology; never verdict
    /// work — reported separately from `launches`, see kernel-launch
    /// accounting in the receipt).
    warmup_launches: u64,
    pages_decoded: u64,
    /// Peak GPU-global sample-domain intermediate residency (bounded window
    /// scratch arenas / the D0 device sample block), in bytes.
    gpu_global_sample_intermediate_peak: u64,
}

struct SessionOut {
    frames_committed: u64,
    chunks: u64,
    xruns: u64,
    wall: Vec<f64>,
    depth_min: u64,
    depth_max: u64,
    shadow_exact: bool,
    drain_state: Option<String>,
    detail: Option<String>,
}

impl SessionOut {
    fn verdict(&self, total_frames: u64) -> (Verdict, String) {
        if !self.shadow_exact {
            (
                Verdict::FailedCorrectness,
                format!(
                    "written codes differ from the scalar oracle: {}",
                    self.detail.clone().unwrap_or_default()
                ),
            )
        } else if self.xruns > 0 {
            (
                Verdict::FailedDeadline,
                format!(
                    "shadow equality held but {} xrun(s) occurred (explicit discontinuities \
                     recorded)",
                    self.xruns
                ),
            )
        } else if self.detail.is_none() && self.frames_committed >= total_frames {
            (
                Verdict::Supported,
                "entropy state evaluated on the GPU; exact final codes written into the \
                 registered ALSA ring and committed without discontinuity"
                    .to_string(),
            )
        } else {
            (
                Verdict::FailedDeadline,
                format!(
                    "session incomplete: {}",
                    self.detail.clone().unwrap_or_default()
                ),
            )
        }
    }
}

fn run_session(
    pcm: &AlsaPcm,
    expected: &[i32],
    mut writer: impl FnMut(i64, u64, u64, &[i32]) -> std::result::Result<(), String>,
) -> SessionOut {
    let period = pcm.period_frames;
    let buffer = pcm.buffer_frames;
    let channels = u64::from(pcm.request.channels);
    let total = expected.len() as u64 / channels;
    let mut frames_committed: u64 = 0;
    let mut chunks = 0u64;
    let mut xruns = 0u64;
    let mut wall: Vec<f64> = Vec::new();
    let mut depth_min = u64::MAX;
    let mut depth_max = 0u64;
    let mut shadow_exact = true;
    let mut detail: Option<String> = None;
    let mut drain_state: Option<String> = None;
    let mut stalls = 0u32;
    let mut started = false;
    // Falsification-court policy (frozen): zero xruns is the success
    // criterion, so ANY xrun/short-commit/suspend event TERMINATES the
    // session as FAILED_DEADLINE with the exact reason recorded. Recovery is
    // deliberately not attempted mid-session.
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
                    if stalls >= u32::from(MAX_STALLS) {
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
        // The expected slice derives from the media position actually written
        // (`frames_committed`); a failed commit that does not advance the
        // position cannot desync later chunks.
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
        // Actual progress resets the consecutive-stall counter.
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
        if let Ok(avail) = pcm.avail() {
            let queued = buffer.saturating_sub(avail.min(buffer));
            depth_min = depth_min.min(queued);
            depth_max = depth_max.max(queued);
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
        depth_min: if depth_min == u64::MAX { 0 } else { depth_min },
        depth_max,
        shadow_exact,
        drain_state,
        detail,
    }
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

struct SessionCtx<'a> {
    cuda: &'a Cuda,
    request: EndpointRequest,
    decode: Function,
    upmix: Function,
}

struct SessionCell {
    label: &'static str,
    /// SHA-256 (hex) over the session's expected endpoint codes (i32 LE) —
    /// the scalar-expected window digest. Every session verifies the ring
    /// region against these codes chunk-by-chunk in place, so this digest is
    /// simultaneously the reference (scalar) hash, the backend (CUDA) hash,
    /// and the endpoint-region hash of the verdict-bearing content.
    window_sha256: String,
    out: SessionOut,
    traffic: SessionTraffic,
}

/// SHA-256 hex over interleaved codes as canonical i32 LE bytes.
fn codes_sha256(codes: &[i32]) -> String {
    let mut hasher = crate::hash::sha256::Sha256::new();
    for c in codes {
        hasher.update(&c.to_le_bytes());
    }
    crate::hash::sha256::hex(&hasher.finalize())
}

fn session_json(cell: &SessionCell, verdict: &Verdict, total_frames: u64) -> serde_json::Value {
    let mut w = cell.out.wall.clone();
    w.sort_by(f64::total_cmp);
    let mean = if w.is_empty() {
        0.0
    } else {
        w.iter().sum::<f64>() / w.len() as f64
    };
    serde_json::json!({
        "label": cell.label,
        "verdict": verdict.label(),
        "frames_committed": cell.out.frames_committed,
        "chunks": cell.out.chunks,
        "xruns": cell.out.xruns,
        "completed": cell.out.detail.is_none() && cell.out.frames_committed >= total_frames,
        "shadow_exact": cell.out.shadow_exact,
        "wall_ms": {
            "mean": mean,
            "median": w.get(w.len() / 2).copied().unwrap_or(0.0),
            "max": w.last().copied().unwrap_or(0.0),
        },
        "depth_frames": { "min": cell.out.depth_min, "max": cell.out.depth_max },
        "materialization_gpu_to_host_bytes": cell.traffic.mat_gpu_to_host,
        "materialization_host_copy_bytes": cell.traffic.mat_host_copy,
        "materialization_host_resident_peak_bytes": cell.traffic.mat_host_resident_peak,
        "endpoint_observation_bytes": cell.traffic.endpoint_obs,
        "kernel_launches": cell.traffic.launches,
        "kernel_launches_total": cell.traffic.launches + cell.traffic.warmup_launches,
        "warmup_kernel_launches": cell.traffic.warmup_launches,
        "entropy_pages_decoded": cell.traffic.pages_decoded,
        "gpu_global_sample_intermediate_peak_bytes": cell
            .traffic
            .gpu_global_sample_intermediate_peak,
        "verification_host_read_bytes": cell.traffic.verification_read,
        "verification_shadow_copy_bytes": cell.traffic.verification_shadow_copy,
        "window_sha256": cell.window_sha256,
        "drain_state": cell.out.drain_state,
        "detail": cell.out.detail,
    })
}

/// D0 entropy baseline for a stereo literal object: whole-object device
/// decode -> one DtoH -> per-chunk host copy into the ring.
fn session_d0_literal(
    ctx: &SessionCtx<'_>,
    rl: &RepresentedLiteral,
    expected: &[i32],
) -> std::result::Result<SessionCell, (Verdict, String)> {
    let pcm = open_validated(&ctx.request)?;
    let full = flatten_literal_range(rl, 0, OBJECT_FRAMES as u32)
        .map_err(|e| (Verdict::Inconclusive, format!("flatten: {e}")))?;
    let stream = ctx
        .cuda
        .create_stream()
        .map_err(|e| (Verdict::Inconclusive, format!("stream: {e}")))?;
    let world = EntropyWorld::open_preloaded(&ctx.cuda.fns, ctx.decode, stream, &full)
        .map_err(|e| (Verdict::Inconclusive, format!("world: {e}")))?;
    let codes = world
        .decode()
        .map_err(|e| (Verdict::FailedCorrectness, format!("decode: {e}")))?;
    if codes != expected {
        return Err((Verdict::FailedCorrectness, "device decode != scalar".into()));
    }
    let mut traffic = SessionTraffic {
        mat_gpu_to_host: (codes.len() * 4) as u64,
        mat_host_resident_peak: (codes.len() * 4) as u64,
        launches: 1,
        pages_decoded: full.pages.len() as u64,
        // The whole-object device decode arena is the D0 sample block.
        gpu_global_sample_intermediate_peak: (codes.len() * 4) as u64,
        ..Default::default()
    };
    let frame_bytes = pcm.request.frame_bytes();
    let channels = u64::from(pcm.request.channels);
    let out = run_session(&pcm, expected, |pos, offset, frames, expect| {
        if u64::try_from(pos).unwrap_or(u64::MAX) + frames > OBJECT_FRAMES as u64 {
            return Err("d0 position outside object".into());
        }
        let base = (pos as u64 * channels) as usize;
        let codes = &codes[base..base + expect.len()];
        let addr = pcm.area_base + (offset * frame_bytes) as usize;
        // SAFETY: mapped region live; range within the buffer (mmap_begin
        // returned offset..offset+frames; geometry validated at open).
        unsafe {
            std::ptr::copy_nonoverlapping(codes.as_ptr(), addr as *mut i32, expect.len());
        }
        let bytes = (expect.len() * 4) as u64;
        traffic.mat_host_copy += bytes;
        traffic.endpoint_obs += bytes;
        region_matches(&pcm, offset, frames, expect)?;
        traffic.verification_read += bytes;
        Ok(())
    });
    Ok(SessionCell {
        label: "d0-literal",
        window_sha256: codes_sha256(expected),
        out,
        traffic,
    })
}

/// D0 entropy baseline for a mono residual object: whole-object device decode
/// -> one DtoH -> per-chunk host mono->stereo expansion copy into the ring.
fn session_d0_residual(
    ctx: &SessionCtx<'_>,
    rr: &RepresentedResidual,
    mono: &[i32],
    expected: &[i32],
) -> std::result::Result<SessionCell, (Verdict, String)> {
    let pcm = open_validated(&ctx.request)?;
    let full = flatten_residual_range(rr, 0, OBJECT_FRAMES as u32)
        .map_err(|e| (Verdict::Inconclusive, format!("flatten: {e}")))?;
    let stream = ctx
        .cuda
        .create_stream()
        .map_err(|e| (Verdict::Inconclusive, format!("stream: {e}")))?;
    let world = EntropyWorld::open_preloaded(&ctx.cuda.fns, ctx.decode, stream, &full)
        .map_err(|e| (Verdict::Inconclusive, format!("world: {e}")))?;
    let codes = world
        .decode()
        .map_err(|e| (Verdict::FailedCorrectness, format!("decode: {e}")))?;
    if codes != mono {
        return Err((
            Verdict::FailedCorrectness,
            "device residual decode != scalar intrinsic".into(),
        ));
    }
    let mut traffic = SessionTraffic {
        mat_gpu_to_host: (codes.len() * 4) as u64,
        mat_host_resident_peak: (codes.len() * 4) as u64,
        launches: 1,
        pages_decoded: full.pages.len() as u64,
        // The whole-object device decode arena is the D0 sample block.
        gpu_global_sample_intermediate_peak: (codes.len() * 4) as u64,
        ..Default::default()
    };
    let frame_bytes = pcm.request.frame_bytes();
    let channels = u64::from(pcm.request.channels);
    let out = run_session(&pcm, expected, |pos, offset, frames, expect| {
        if u64::try_from(pos).unwrap_or(u64::MAX) + frames > OBJECT_FRAMES as u64 {
            return Err("d0 position outside object".into());
        }
        let base = pos as usize;
        let mono_codes = &codes[base..base + frames as usize];
        // Expand mono -> stereo (L = R) while copying into the ring.
        let addr = pcm.area_base + (offset * frame_bytes) as usize;
        let stride = channels as usize;
        let out_slice: &mut [i32] = unsafe {
            std::slice::from_raw_parts_mut(addr as *mut i32, (frames * channels) as usize)
        };
        for (i, &v) in mono_codes.iter().enumerate() {
            for c in 0..stride {
                out_slice[i * stride + c] = v;
            }
        }
        let bytes = (expect.len() * 4) as u64;
        traffic.mat_host_copy += bytes;
        traffic.endpoint_obs += bytes;
        region_matches(&pcm, offset, frames, expect)?;
        traffic.verification_read += bytes;
        Ok(())
    });
    Ok(SessionCell {
        label: "d0-residual",
        window_sha256: codes_sha256(expected),
        out,
        traffic,
    })
}

/// D1 fused session for a stereo entropy-literal object: per window, decode
/// only the pages intersecting the window directly into the registered ring.
/// Window jobs (one per 512-frame chunk = two 256-frame pages) are flattened
/// once at session setup; the real-time chunk path is only `decode_into` +
/// in-place verification.
fn session_d1_literal(
    ctx: &SessionCtx<'_>,
    rl: &RepresentedLiteral,
    expected: &[i32],
) -> std::result::Result<SessionCell, (Verdict, String)> {
    let pcm = open_validated(&ctx.request)?;
    let reg = register_ring(ctx.cuda, &pcm)?;
    let Some(ring_dev) = reg.device_ptr else {
        return Err((Verdict::UnsupportedByApi, "no device pointer".into()));
    };
    let chunks = OBJECT_FRAMES / PERIOD_FRAMES as usize;
    let mut worlds = Vec::with_capacity(chunks);
    for w in 0..chunks {
        let job = flatten_literal_range(rl, (w * PERIOD_FRAMES as usize) as u64, PERIOD_FRAMES)
            .map_err(|e| (Verdict::Inconclusive, format!("flatten: {e}")))?;
        let stream = ctx
            .cuda
            .create_stream()
            .map_err(|e| (Verdict::Inconclusive, format!("stream: {e}")))?;
        let world = EntropyWorld::open_preloaded(&ctx.cuda.fns, ctx.decode, stream, &job)
            .map_err(|e| (Verdict::Inconclusive, format!("world: {e}")))?;
        worlds.push((job, world));
    }
    let mut traffic = SessionTraffic {
        // Warm-up launches below are methodology, not verdict work: 10 rounds
        // over the 8 window worlds. Reported separately from `launches`.
        warmup_launches: 10 * worlds.len() as u64,
        ..Default::default()
    };
    let frame_bytes = pcm.request.frame_bytes();
    // Sustained-clock warm-up (mandatory court methodology, not a pre-decode):
    // the rANS decode kernel is latency-chain bound and its wall time depends
    // on the GPU clock regime: ~20 ms per 512-frame window on an idle-first
    // launch, ~2-3 ms after ~7 back-to-back launches (court-warmup regime),
    // down to ~0.4-0.5 ms when the GPU is aggregate-hot. The warm-up launches
    // decode into scratch arenas
    // (nothing is pre-decoded for the session; every window is still decoded
    // at its chunk boundary through `decode_into(ring)`); without it the
    // first paced chunks would pay the cold-launch cost and underrun the
    // endpoint. Receipted as a limitation.
    {
        let warm = DeviceBuffer::alloc(&ctx.cuda.fns, 1024 * 4)
            .map_err(|e| (Verdict::Inconclusive, format!("warmup arena: {e}")))?;
        for _ in 0..10 {
            for (_, w) in &worlds {
                w.decode_into(warm.device_ptr())
                    .map_err(|e| (Verdict::Inconclusive, format!("warmup: {e}")))?;
            }
        }
    }

    let out = run_session(&pcm, expected, |pos, offset, frames, expect| {
        if frames != u64::from(PERIOD_FRAMES) {
            return Err(format!(
                "mmap_begin granted {frames} frames; the D1 fused path requires \
                 page-aligned chunks of {PERIOD_FRAMES} (2 x {PAGE_FRAMES} pages)"
            ));
        }
        let idx = (pos as u64 / u64::from(PERIOD_FRAMES)) as usize;
        let (job, world) = &worlds[idx];
        let target = ring_dev + offset * frame_bytes;
        world
            .decode_into(target)
            .map_err(|e| format!("decode_into ring: {e}"))?;
        traffic.launches += 1;
        traffic.pages_decoded += job.pages.len() as u64;
        let bytes = (expect.len() * 4) as u64;
        traffic.endpoint_obs += bytes;
        region_matches(&pcm, offset, frames, expect)?;
        traffic.verification_read += bytes;
        Ok(())
    });
    Ok(SessionCell {
        label: "d1-literal",
        window_sha256: codes_sha256(expected),
        out,
        traffic,
    })
}

/// D1 fused session for a mono procedural+residual object: per window, decode
/// the intersecting pages into a page-local transient device arena, then
/// expand mono -> stereo (L = R) on the device directly into the ring.
fn session_d1_residual(
    ctx: &SessionCtx<'_>,
    rr: &RepresentedResidual,
    expected: &[i32],
) -> std::result::Result<SessionCell, (Verdict, String)> {
    let pcm = open_validated(&ctx.request)?;
    let reg = register_ring(ctx.cuda, &pcm)?;
    let Some(ring_dev) = reg.device_ptr else {
        return Err((Verdict::UnsupportedByApi, "no device pointer".into()));
    };
    let up_stream = ctx
        .cuda
        .create_stream()
        .map_err(|e| (Verdict::Inconclusive, format!("upmix stream: {e}")))?;
    // Window-local transient decode arena (mono samples of one window =
    // two 256-frame pages), re-used across windows of the session.
    let arena = DeviceBuffer::alloc(&ctx.cuda.fns, PERIOD_FRAMES as usize * 4)
        .map_err(|e| (Verdict::Inconclusive, format!("arena: {e}")))?;
    let chunks = OBJECT_FRAMES / PERIOD_FRAMES as usize;
    let mut worlds = Vec::with_capacity(chunks);
    for w in 0..chunks {
        let job = flatten_residual_range(rr, (w * PERIOD_FRAMES as usize) as u64, PERIOD_FRAMES)
            .map_err(|e| (Verdict::Inconclusive, format!("flatten: {e}")))?;
        let stream = ctx
            .cuda
            .create_stream()
            .map_err(|e| (Verdict::Inconclusive, format!("stream: {e}")))?;
        let world = EntropyWorld::open_preloaded(&ctx.cuda.fns, ctx.decode, stream, &job)
            .map_err(|e| (Verdict::Inconclusive, format!("world: {e}")))?;
        worlds.push((job, world));
    }
    // Sustained-clock warm-up (see the literal session for the rationale).
    {
        let warm = DeviceBuffer::alloc(&ctx.cuda.fns, PERIOD_FRAMES as usize * 4)
            .map_err(|e| (Verdict::Inconclusive, format!("warmup arena: {e}")))?;
        for _ in 0..10 {
            for (_, w) in &worlds {
                w.decode_into(warm.device_ptr())
                    .map_err(|e| (Verdict::Inconclusive, format!("warmup: {e}")))?;
            }
        }
    }

    let mut traffic = SessionTraffic {
        // Warm-up launches below are methodology, not verdict work: 10 rounds
        // over the 8 window worlds. Reported separately from `launches`.
        warmup_launches: 10 * worlds.len() as u64,
        // Window-local GPU arena holding the decoded mono samples of one
        // 512-frame window (two 256-frame pages) before the on-device
        // mono->stereo expansion: 512 i32 = 2048 bytes of bounded
        // GPU-global sample intermediate.
        gpu_global_sample_intermediate_peak: (PERIOD_FRAMES as u64) * 4,
        ..Default::default()
    };
    let frame_bytes = pcm.request.frame_bytes();
    let upmix = ctx.upmix;
    let out = run_session(&pcm, expected, |pos, offset, frames, expect| {
        if frames != u64::from(PERIOD_FRAMES) {
            return Err(format!(
                "mmap_begin granted {frames} frames; the D1 fused path requires \
                 page-aligned chunks of {PERIOD_FRAMES} (2 x {PAGE_FRAMES} pages)"
            ));
        }
        let idx = (pos as u64 / u64::from(PERIOD_FRAMES)) as usize;
        let (job, world) = &worlds[idx];
        // 1) decode the window's pages into the transient arena
        // (device-resident; two 256-frame pages decode in parallel);
        world
            .decode_into(arena.device_ptr())
            .map_err(|e| format!("decode_into arena: {e}"))?;
        // 2) sampler transform: mono -> stereo duplication into the ring.
        let target = ring_dev + offset * frame_bytes;
        let params = [arena.device_ptr(), target, frames, u64::from(CHANNELS)];
        let blocks = u32::try_from(frames.div_ceil(128)).unwrap_or(1).max(1);
        upmix
            .launch((blocks, 1, 1), (128, 1, 1), up_stream.handle, &params)
            .map_err(|e| format!("upmix launch: {e}"))?;
        up_stream
            .synchronize()
            .map_err(|e| format!("upmix sync: {e}"))?;
        traffic.launches += 2;
        traffic.pages_decoded += job.pages.len() as u64;
        let bytes = (expect.len() * 4) as u64;
        traffic.endpoint_obs += bytes;
        region_matches(&pcm, offset, frames, expect)?;
        traffic.verification_read += bytes;
        Ok(())
    });
    Ok(SessionCell {
        label: "d1-residual",
        window_sha256: codes_sha256(expected),
        out,
        traffic,
    })
}

/// D1 fused session for the high-entropy literal control (RAW or RANS pages,
/// whichever the complete-cost encoder chose; kinds recorded).
fn session_d1_noise(
    ctx: &SessionCtx<'_>,
    rn: &RepresentedLiteral,
    expected: &[i32],
) -> std::result::Result<SessionCell, (Verdict, String)> {
    let pcm = open_validated(&ctx.request)?;
    let reg = register_ring(ctx.cuda, &pcm)?;
    let Some(ring_dev) = reg.device_ptr else {
        return Err((Verdict::UnsupportedByApi, "no device pointer".into()));
    };
    let chunks = OBJECT_FRAMES / PERIOD_FRAMES as usize;
    let mut worlds = Vec::with_capacity(chunks);
    for w in 0..chunks {
        let job = flatten_literal_range(rn, (w * PERIOD_FRAMES as usize) as u64, PERIOD_FRAMES)
            .map_err(|e| (Verdict::Inconclusive, format!("flatten: {e}")))?;
        let stream = ctx
            .cuda
            .create_stream()
            .map_err(|e| (Verdict::Inconclusive, format!("stream: {e}")))?;
        let world = EntropyWorld::open_preloaded(&ctx.cuda.fns, ctx.decode, stream, &job)
            .map_err(|e| (Verdict::Inconclusive, format!("world: {e}")))?;
        worlds.push((job, world));
    }
    // Sustained-clock warm-up (see the literal session for the rationale).
    {
        let warm = DeviceBuffer::alloc(&ctx.cuda.fns, PERIOD_FRAMES as usize * 4)
            .map_err(|e| (Verdict::Inconclusive, format!("warmup arena: {e}")))?;
        for _ in 0..10 {
            for (_, w) in &worlds {
                w.decode_into(warm.device_ptr())
                    .map_err(|e| (Verdict::Inconclusive, format!("warmup: {e}")))?;
            }
        }
    }

    let mut traffic = SessionTraffic {
        // Warm-up launches below are methodology, not verdict work: 10 rounds
        // over the 8 window worlds. Reported separately from `launches`.
        warmup_launches: 10 * worlds.len() as u64,
        ..Default::default()
    };
    let frame_bytes = pcm.request.frame_bytes();
    let out = run_session(&pcm, expected, |pos, offset, frames, expect| {
        if frames != u64::from(PERIOD_FRAMES) {
            return Err(format!(
                "mmap_begin granted {frames} frames; the D1 fused path requires \
                 page-aligned chunks of {PERIOD_FRAMES} (2 x {PAGE_FRAMES} pages)"
            ));
        }
        let idx = (pos as u64 / u64::from(PERIOD_FRAMES)) as usize;
        let (job, world) = &worlds[idx];
        let target = ring_dev + offset * frame_bytes;
        world
            .decode_into(target)
            .map_err(|e| format!("decode_into ring: {e}"))?;
        traffic.launches += 1;
        traffic.pages_decoded += job.pages.len() as u64;
        let bytes = (expect.len() * 4) as u64;
        traffic.endpoint_obs += bytes;
        region_matches(&pcm, offset, frames, expect)?;
        traffic.verification_read += bytes;
        Ok(())
    });
    Ok(SessionCell {
        label: "d1-noise",
        window_sha256: codes_sha256(expected),
        out,
        traffic,
    })
}

// ---------------------------------------------------------------------------
// Court
// ---------------------------------------------------------------------------

/// Run the court; writes an immutable receipt under `receipts/entropy-d1/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |verdict: Verdict, why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("entropy-d1");
        b.result(verdict)
            .result_detail(format!("entropy-d1: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court entropy-d1: {verdict} ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(verdict)
    };

    let audible = audible_requested();
    let samples = literal_tone(audible);
    let noise = noise_stereo(audible);
    let (rr, res_mono, n_corrections) = residual_fixture(audible);
    let res_stereo = expand_mono_to_stereo(&res_mono);

    let lit_desc = ObjectDescriptor::new(
        Representation::Literal,
        OBJECT_FRAMES as u64,
        Layout::Stereo,
        None,
    )
    .unwrap();
    let rl = RepresentedLiteral::encode(
        lit_desc,
        &samples,
        PAGE_FRAMES,
        Symbolization::DeltaLane4,
        ModelMode::Inline,
        false,
    )
    .map_err(|e| crate::error::Error::malformed(format!("literal encode: {e}")))?;
    let noise_desc = ObjectDescriptor::new(
        Representation::Literal,
        OBJECT_FRAMES as u64,
        Layout::Stereo,
        None,
    )
    .unwrap();
    let rn = RepresentedLiteral::encode(
        noise_desc,
        &noise,
        PAGE_FRAMES,
        Symbolization::DeltaLane4,
        ModelMode::Inline,
        false,
    )
    .map_err(|e| crate::error::Error::malformed(format!("noise encode: {e}")))?;

    let lit_pages = rl.pages.len();
    let noise_pages = rn.pages.len();
    let res_pages = rr.pages.len();
    let lit_rans = rl.pages.iter().filter(|p| p.kind == PageKind::Rans).count();
    let noise_rans = rn.pages.iter().filter(|p| p.kind == PageKind::Rans).count();
    let res_rans = rr.pages.iter().filter(|p| p.kind == PageKind::Rans).count();

    // Scalar authority: the entropy representations decode to the exact
    // fixtures before any GPU/endpoint work (host flat decoder == scalar).
    {
        let mut scratch = vec![0u8; 1];
        let job = flatten_literal_range(&rl, 0, OBJECT_FRAMES as u32)?;
        let mut host = vec![0i32; job.arena_samples];
        scratch.resize(job.max_page_scratch.max(1), 0);
        if !job.decode_pages_host(&mut host, &mut scratch, 0..job.pages.len()) || host != samples {
            return fail(
                Verdict::FailedCorrectness,
                "host literal decode != scalar fixture",
            );
        }
        let jn = flatten_literal_range(&rn, 0, OBJECT_FRAMES as u32)?;
        let mut hn = vec![0i32; jn.arena_samples];
        scratch.resize(jn.max_page_scratch.max(1), 0);
        if !jn.decode_pages_host(&mut hn, &mut scratch, 0..jn.pages.len()) || hn != noise {
            return fail(
                Verdict::FailedCorrectness,
                "host noise decode != scalar fixture",
            );
        }
        let jr = flatten_residual_range(&rr, 0, OBJECT_FRAMES as u32)?;
        let mut hr = vec![0i32; jr.arena_samples];
        scratch.resize(jr.max_page_scratch.max(1), 0);
        if !jr.decode_pages_host(&mut hr, &mut scratch, 0..jr.pages.len()) || hr != res_mono {
            return fail(
                Verdict::FailedCorrectness,
                "host residual decode != scalar intrinsic",
            );
        }
    }

    let ptx = match ptx_bytes()? {
        Some(p) => p,
        None => return fail(Verdict::Inconclusive, "PTX artifact absent"),
    };

    let mut endpoints = list_playback_endpoints();
    if let Ok(want) = std::env::var("VOLE_ENTROPY_D1_DEVICE") {
        endpoints.retain(|e| e.pcm_name == want);
        if endpoints.is_empty() {
            return fail(
                Verdict::NotApplicable,
                &format!("VOLE_ENTROPY_D1_DEVICE={want} matched no playback endpoint"),
            );
        }
    }
    endpoints.sort_by_key(candidate_rank);
    if endpoints.is_empty() {
        return fail(Verdict::UnsupportedByHardware, "no ALSA playback endpoints");
    }

    let mut trials: Vec<serde_json::Value> = Vec::new();
    let mut total_frames: u64 = 0;

    for dev in &endpoints {
        // Stage 1: CUDA/module/registration readiness on this endpoint.
        let cuda = match Cuda::open(0) {
            Ok(c) => c,
            Err(e) => {
                trials.push(serde_json::json!({
                    "device": dev.pcm_name, "driver": dev.driver, "stage": "cuda",
                    "result": "UNSUPPORTED_BY_HARDWARE",
                    "detail": format!("CUDA unavailable: {e}"),
                }));
                continue;
            }
        };
        let module = match cuda.load_module(&ptx.0) {
            Ok(m) => m,
            Err(e) => {
                trials.push(serde_json::json!({
                    "device": dev.pcm_name, "driver": dev.driver, "stage": "module",
                    "result": "INCONCLUSIVE", "detail": format!("PTX module load: {e}"),
                }));
                continue;
            }
        };
        let decode = match module.function("vole_entropy_decode") {
            Ok(f) => f,
            Err(e) => {
                trials.push(serde_json::json!({
                    "device": dev.pcm_name, "driver": dev.driver, "stage": "function",
                    "result": "INCONCLUSIVE", "detail": format!("kernel lookup: {e}"),
                }));
                continue;
            }
        };
        let upmix = match module.function("vole_upmix_mono_dup") {
            Ok(f) => f,
            Err(e) => {
                trials.push(serde_json::json!({
                    "device": dev.pcm_name, "driver": dev.driver, "stage": "function",
                    "result": "INCONCLUSIVE", "detail": format!("upmix kernel lookup: {e}"),
                }));
                continue;
            }
        };

        // Registration probe on the real ring of this endpoint.
        let request = device_request(dev);
        let probe = (|| -> std::result::Result<(), (Verdict, String)> {
            let pcm = open_validated(&request)?;
            let reg = register_ring(&cuda, &pcm)?;
            if reg.device_ptr.is_none() {
                return Err((Verdict::UnsupportedByApi, "no device pointer".into()));
            }
            drop(reg);
            drop(pcm);
            Ok(())
        })();
        if let Err((v, why)) = probe {
            trials.push(serde_json::json!({
                "device": dev.pcm_name, "driver": dev.driver, "stage": "register",
                "result": v.label(), "detail": why,
            }));
            continue;
        }

        // Stage 2: the five sessions on this endpoint (each opens and
        // registers its own pcm; the shared CUDA context stays current).
        let ctx = SessionCtx {
            cuda: &cuda,
            request: request.clone(),
            decode,
            upmix,
        };
        let mut cells: Vec<SessionCell> = Vec::new();
        let mut first_error: Option<(Verdict, String, &'static str)> = None;
        {
            let mut push =
                |name: &'static str, r: std::result::Result<SessionCell, (Verdict, String)>| match r
                {
                    Ok(c) => cells.push(c),
                    Err((v, why)) => {
                        if first_error.is_none() {
                            first_error = Some((v, why, name));
                        }
                    }
                };
            push("d0-literal", session_d0_literal(&ctx, &rl, &samples));
            push("d1-literal", session_d1_literal(&ctx, &rl, &samples));
            push(
                "d0-residual",
                session_d0_residual(&ctx, &rr, &res_mono, &res_stereo),
            );
            push("d1-residual", session_d1_residual(&ctx, &rr, &res_stereo));
            push("d1-noise", session_d1_noise(&ctx, &rn, &noise));
        }

        if let Some((v, why, name)) = first_error {
            // Session setup failed (endpoint or API): record the trial with
            // what ran and move to the next candidate.
            let snap_json = serde_json::Value::Null;
            trials.push(serde_json::json!({
                "device": dev.pcm_name, "driver": dev.driver,
                "stage": "session", "result": v.label(),
                "detail": format!("{name}: {why}"),
                "snapshot": snap_json,
                "sessions": cells.iter().map(|c| {
                    let (cv, _) = c.out.verdict(total_frames);
                    session_json(c, &cv, total_frames)
                }).collect::<Vec<_>>(),
            }));
            continue;
        }

        // All five sessions ran; assess their verdicts.
        total_frames = (samples.len() as u64) / u64::from(CHANNELS);
        let mut verdicts = Vec::new();
        let mut all_supported = true;
        for cell in &cells {
            let (v, detail) = cell.out.verdict(total_frames);
            if v != Verdict::Supported {
                all_supported = false;
            }
            verdicts.push(serde_json::json!({
                "label": cell.label,
                "verdict": v.label(),
                "detail": detail,
            }));
        }

        // Reopen to capture the endpoint snapshot + the exact registered
        // range for the receipt evidence (register, read, unregister).
        let (snap, reg_evidence) = match open_validated(&request) {
            Ok(p) => {
                let s = p.snapshot();
                let range = match register_ring(&cuda, &p) {
                    Ok(reg) => {
                        let r = format!("{:#x}+{}", reg.host_ptr as usize, reg.bytes);
                        let dp = reg.device_ptr.is_some();
                        drop(reg);
                        (Some(r), dp)
                    }
                    Err((_, why)) => (Some(format!("register-for-evidence failed: {why}")), false),
                };
                drop(p);
                (
                    serde_json::json!({
                        "pcm_name": s.pcm_name, "access": s.access, "format": s.format,
                        "rate_hz": s.rate_hz, "channels": s.channels,
                        "period_frames": s.period_frames, "buffer_frames": s.buffer_frames,
                        "area_layout": s.area_layout,
                        "area_layout_validated": s.area_layout_validated,
                    }),
                    range,
                )
            }
            Err((_, why)) => (serde_json::json!({ "detail": why }), (None, false)),
        };
        let _ = &reg_evidence;

        let cell = serde_json::json!({
            "device": dev.pcm_name,
            "driver": dev.driver,
            "stage": "session",
            "result": if all_supported { "SUPPORTED" } else { "FAILED" },
            "snapshot": snap,
            "sessions": cells.iter().map(|c| {
                let (cv, _) = c.out.verdict(total_frames);
                session_json(c, &cv, total_frames)
            }).collect::<Vec<_>>(),
            "session_verdicts": verdicts,
        });
        trials.push(cell.clone());

        if !all_supported {
            // Sessions ran but failed on this endpoint: keep its row and try
            // the next candidate.
            continue;
        }

        // Success: build the sealed receipt and stop probing.
        let lit = cells
            .iter()
            .find(|c| c.label == "d0-literal")
            .map(|c| &c.traffic)
            .copied()
            .unwrap_or_default();
        let d1 = cells
            .iter()
            .find(|c| c.label == "d1-literal")
            .map(|c| &c.traffic)
            .copied()
            .unwrap_or_default();
        let r0 = cells
            .iter()
            .find(|c| c.label == "d0-residual")
            .map(|c| &c.traffic)
            .copied()
            .unwrap_or_default();
        let r1 = cells
            .iter()
            .find(|c| c.label == "d1-residual")
            .map(|c| &c.traffic)
            .copied()
            .unwrap_or_default();
        let removed = serde_json::json!({
            "literal": {
                "gpu_to_host": lit.mat_gpu_to_host.saturating_sub(d1.mat_gpu_to_host),
                "host_copies": lit.mat_host_copy.saturating_sub(d1.mat_host_copy),
            },
            "residual": {
                "gpu_to_host": r0.mat_gpu_to_host.saturating_sub(r1.mat_gpu_to_host),
                "host_copies": r0.mat_host_copy.saturating_sub(r1.mat_host_copy),
            },
        });
        let object = serde_json::json!({
            "frames": OBJECT_FRAMES,
            "channels": CHANNELS,
            "page_frames": PAGE_FRAMES,
            "pages_per_object": OBJECT_PAGES,
            "literal": { "pages": lit_pages, "rans": lit_rans, "raw": lit_pages - lit_rans },
            "noise": { "pages": noise_pages, "rans": noise_rans, "raw": noise_pages - noise_rans },
            "residual": {
                "pages": res_pages, "rans": res_rans, "raw": res_pages - res_rans,
                "corrections": n_corrections,
                "mono_to_stereo": "duplicate (L=R) at the sampler boundary",
            },
        });

        let mut counters = crate::evidence::counters::Counters::new();
        counters.gpu_to_host_pcm_bytes = d1.mat_gpu_to_host;
        counters.host_pcm_copy_bytes = d1.mat_host_copy;
        counters.endpoint_observation_bytes = d1.endpoint_obs;
        counters.kernel_launches = d1.launches;
        let dl = cells
            .iter()
            .find(|c| c.label == "d1-literal")
            .map(|c| &c.out)
            .unwrap();
        if dl.chunks > 0 {
            counters.quanta_submitted = dl.chunks;
            counters.observe_endpoint_depth(dl.depth_min);
            counters.observe_endpoint_depth(dl.depth_max);
        }

        // Every candidate still gets its own row: append NOT_ATTEMPTED rows
        // for the remaining endpoints before sealing.
        for rest in endpoints.iter().filter(|d| d.pcm_name != dev.pcm_name) {
            trials.push(serde_json::json!({
                "device": rest.pcm_name,
                "driver": rest.driver,
                "stage": "session",
                "result": "NOT_ATTEMPTED_AFTER_SUCCESS",
                "detail": format!(
                    "a prior candidate ({}) completed a successful entropy-D1 session set",
                    dev.pcm_name
                ),
            }));
        }

        let mut builder = ReceiptBuilder::new("entropy-d1");
        builder
            .result(Verdict::Supported)
            .result_detail(format!(
                "entropy D1 on {}: per-window bounded page decode wrote exact final S32 codes \
                 directly into the registered ALSA ring for literal ({} pages), procedural \
                 mono+residual ({} pages, {} corrections, device mono->stereo expansion) and \
                 high-entropy control ({} pages); scalar == CUDA == endpoint ring codes; D0 \
                 equal-work baseline on the same object/frames/endpoint",
                dev.pcm_name, lit_pages, res_pages, n_corrections, noise_pages
            ))
            .params(CourtParams {
                universe: Some("vole.audio.u1".into()),
                profile: Some("u1/v1 + vole.entropy.p1/p1/v1".into()),
                backend: Some("cuda".into()),
                sample_rate_hz: Some(RATE_HZ),
                channels: Some(CHANNELS),
                quantum_frames: Some(PERIOD_FRAMES),
                duration_secs: Some(OBJECT_FRAMES as f64 / RATE_HZ as f64),
                content_kind: Some("entropy-corpus-v1 d1".into()),
                ..Default::default()
            })
            .counters(counters)
            .endpoint(EndpointEvidence {
                directness: Some("D1_ENDPOINT_MAPPED".into()),
                registered_range: reg_evidence.0,
                registration_result: Some("cuMemHostRegister(DEVICEMAP) succeeded".into()),
                device_pointer: Some(if reg_evidence.1 {
                    "present (registered host memory, DEVICEMAP)".into()
                } else {
                    "absent".into()
                }),
                ..Default::default()
            })
            .provenance({
                // Every session verified its committed ring region against the
                // scalar-expected window chunk-by-chunk in place, so the
                // window digest is simultaneously the reference (scalar)
                // hash, the backend (CUDA) hash, and the endpoint-region hash
                // of the verdict-bearing content. A direct post-session
                // re-hash of the ring is impossible (the DMA consumes the
                // region during drain); the per-chunk in-place equality is
                // the mechanism that makes the three digests identical, and
                // each session's window_sha256 is recorded in its row.
                let verdict_sha = cells
                    .iter()
                    .find(|c| c.label == "d1-literal")
                    .map(|c| c.window_sha256.clone())
                    .unwrap_or_default();
                crate::evidence::receipt::Provenance {
                    reference_hash: Some(verdict_sha.clone()),
                    backend_hash: Some(verdict_sha.clone()),
                    gpu_artifact_hash: Some(ptx.1.clone()),
                    benchmark_order: vec![
                        "d0-literal".into(),
                        "d1-literal".into(),
                        "d0-residual".into(),
                        "d1-residual".into(),
                        "d1-noise".into(),
                    ],
                    // Exact equality of the verdict-bearing path: the CUDA
                    // written ring codes equal the scalar oracle for every
                    // committed chunk (in-place verification), which makes
                    // reference_hash == backend_hash == endpoint content.
                    exact_equality: Some(all_supported),
                    ..Default::default()
                }
            })
            .extra(
                "endpoint_region_sha256",
                serde_json::json!({
                    "d1_literal_window": cells
                        .iter()
                        .find(|c| c.label == "d1-literal")
                        .map(|c| c.window_sha256.clone()),
                    "derivation": "the registered ring is verified chunk-by-chunk in place \
                        against the scalar window before every commit; the ring is then \
                        consumed by the DMA during drain, so the endpoint-region digest is \
                        the verified window digest (per-session window_sha256 in each row)",
                }),
            )
            .extra(
                "kernel_launch_accounting",
                serde_json::json!({
                    "definition": "counters.kernel_launches and each session's \
                        kernel_launches count verdict-bearing measurement launches only \
                        (consistent with every earlier seal); warm-up launches are reported \
                        separately as warmup_kernel_launches per session and in \
                        methodology.sustained_clock_warmup_launches_per_d1_session",
                    "top_level_counters_kernel_launches": d1.launches,
                    "top_level_includes_warmup": false,
                }),
            )
            .extra("trials", serde_json::Value::Array(trials))
            .extra("winning_trial", cell)
            .extra("object", object)
            .extra("bytes_d1_removes", removed)
            .extra(
                "verification_accounting",
                serde_json::json!(
                    "verification reads the committed ring in place; it is \
                    separate from materialization counters (never conflated)"
                ),
            )
            .extra(
                "methodology",
                serde_json::json!({
                    "buffer_frames": BUFFER_FRAMES,
                    "period_frames": PERIOD_FRAMES,
                    "page_frames": PAGE_FRAMES,
                    "windows_per_session": OBJECT_FRAMES / PERIOD_FRAMES as usize,
                    "pages_per_window": (PERIOD_FRAMES / PAGE_FRAMES) as usize,
                    "sustained_clock_warmup_launches_per_d1_session": 10
                        * (OBJECT_FRAMES / PERIOD_FRAMES as usize),
                    "warmup_target": "device scratch arenas; no window is pre-decoded for \
                        the session",
                }),
            )
            .extra(
                "artifact",
                serde_json::json!({ "path": DEFAULT_PTX, "sha256": ptx.1 }),
            );
        builder.limitation(if audible {
            "audible content emitted (opt-in VOLE_ENTROPY_D1_EMIT_AUDIO=1)"
        } else {
            "silence-safe content (quiet fixture amplitudes); endpoint consumption is proven \
             by commit/avail/drain evidence, not by listening"
        });
        builder.limitation(
            "D1 residual session uses a window-local transient device arena (the mono \
             samples of one 512-frame window = two 256-frame pages, 2048 bytes) for the \
             on-device mono->stereo expansion; it is bounded transient decode scratch, not \
             a full-object waveform, and is receipted as \
             gpu_global_sample_intermediate_peak_bytes (2048) per session",
        );
        builder.limitation(
            "EntropyWorld::launch allocates the per-decode device scratch/status buffers, \
             the parameter vector, and the host status buffer on every decode invocation; \
             the D1 court is therefore not yet an allocation-free production real-time \
             path — zero xruns here is court evidence, not the final RT architecture \
             (Phase M owns the preallocated persistent-resource path)",
        );
        builder.limitation(
            "the single-thread serial rANS decode is latency-chain bound on idle GPU clocks: \
             each D1 session first runs back-to-back warm-up decode launches into scratch \
             arenas (80 launches/session) so the paced chunk decodes run at sustained \
             clocks (wall regimes: ~20 ms idle-first-launch, ~2-3 ms court-warmup, \
             ~0.4-0.5 ms aggregate-hot — see PERFORMANCE.md); the warm-up decodes \
             nothing used by the session — every window is decoded at its own chunk \
             boundary directly into the registered ring",
        );
        builder.limitation(
            "high-entropy control: the full-range 32-bit RAW-fallback negative control is \
             audible-only (quiet mode truncates low-order bits to keep peak <= 2^20); RAW \
             fallback on full-range noise is separately sealed by court entropy-cuda and the \
             entropy-literal corpus cells",
        );
        let (_, path) = builder.finish_write(receipts_root)?;
        println!("court entropy-d1: SUPPORTED");
        println!("  device: {}", dev.pcm_name);
        println!("  receipt: {}", path.display());
        return Ok(Verdict::Supported);
    }

    let mut b = ReceiptBuilder::new("entropy-d1");
    b.result(Verdict::UnsupportedByHardware)
        .result_detail(format!(
            "no entropy-D1 session succeeded across {} endpoint candidates",
            endpoints.len()
        ))
        .extra("trials", serde_json::Value::Array(trials));
    let (_, path) = b.finish_write(receipts_root)?;
    println!("court entropy-d1: UNSUPPORTED_BY_HARDWARE");
    println!("  receipt: {}", path.display());
    Ok(Verdict::UnsupportedByHardware)
}
