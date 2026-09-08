//! `court d1` — Phase H CUDA D1 falsification against the real ALSA endpoint.
//!
//! The question this court answers is deliberately narrow: can the GPU write
//! final sample-domain observations *directly* into the actual ALSA-mapped
//! endpoint region (the memory the endpoint DMA reads), with no D0 VRAM
//! block, no device→host transfer, and no host PCM copy?
//!
//! Method (never a substitute-buffer shortcut):
//!
//! 1. Discover `hw:` playback PCMs (non-`hw:` names are refused: plug
//!    conversion would answer a different question). Configure the frozen
//!    shape `MMAP_INTERLEAVED + S32_LE + 48 kHz + stereo + 512-frame period /
//!    1024-frame buffer` on each candidate.
//! 2. Register the **exact mapped region** (`cuMemHostRegister` +
//!    `CU_MEMHOSTREGISTER_DEVICEMAP` + `cuMemHostGetDevicePointer`),
//!    recording base/length/page alignment/flags/rc/pointer attributes. A
//!    failed registration is recorded exactly and classified — never "fixed"
//!    by allocating a new pinned buffer.
//! 3. When registered: a paced real-time session where the kernel writes each
//!    contiguous mmap chunk directly into the registered region's device
//!    pointer, the stream synchronizes, the written codes are shadow-verified
//!    in place against the scalar oracle, and only then the chunk is
//!    committed. xruns are counted and recorded with an explicit
//!    discontinuity policy (never a silent reset).
//! 4. A controlled **D0-mmap baseline** runs on the same endpoint shape:
//!    `KernelWorld::render` (DtoH + host copy) plus a CPU copy into the
//!    region — the exact bytes D1 removes are measured, not asserted.
//!
//! Default content is silence-safe (peak |code| ≤ 2^16 ≈ −80 dBFS, but
//! nonzero and content-rich). Audible content requires the opt-in flag (the
//! CLI sets `VOLE_D1_EMIT_AUDIO=1` for `court d1 --emit-audio`).
//!
//! Verdicts: `SUPPORTED` only when a device registered, played the full
//! window byte-exact with no xrun and a clean drain; `FELL_BACK_TO_D0` when
//! registration failed everywhere but the D0 baseline ran; explicit per-device
//! `UNSUPPORTED_BY_HARDWARE` / `UNSUPPORTED_BY_API` rows are recorded
//! otherwise. Negative results are results.

use crate::evidence::receipt::{
    CourtParams, EndpointEvidence, Provenance, ReceiptBuilder, RunTiming,
};
use crate::status::Verdict;
use std::path::Path;

pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    #[cfg(target_os = "linux")]
    {
        linux::run(receipts_root)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let mut b = ReceiptBuilder::new("d1");
        b.result(Verdict::UnsupportedByApi)
            .result_detail("court d1 requires Linux + ALSA; this host is not Linux")
            .limitation("the ALSA endpoint module is Linux-only");
        let (_, path) = b.finish_write(receipts_root)?;
        println!("court d1: UNSUPPORTED_BY_API (not on Linux)");
        println!("  receipt: {}", path.display());
        Ok(Verdict::UnsupportedByApi)
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use crate::audio::alsa::{
        AlsaFailure, AlsaPcm, EndpointInfo, EndpointRequest, EndpointSnapshot, classify_failure,
        list_playback_endpoints,
    };
    use crate::backend::cuda::direct::{
        HostRegistration, PointerEvidence, RegisterRange, RegistrationAttempt, host_page_size,
    };
    use crate::backend::cuda::driver::Cuda;
    use crate::backend::cuda::kernel::{KernelWorld, Strategy};
    use crate::backend::cuda::probe::CudaProbe;
    use crate::backend::flatten::flatten;
    use crate::eval::ScalarOracle;
    use crate::evidence::counters::Counters;
    use crate::hash::sha256::{Sha256, hex};
    use crate::object::descriptor::{ObjectDescriptor, Representation};
    use crate::object::residual::{Residual, ResidualModel};
    use crate::object::{
        Constant, Cycle, Literal, LoopRegion, Noise, ObjectData, ObjectId, ObjectStore, Oscillator,
        PartialBank,
    };
    use crate::sampler::envelope::EnvelopeParams;
    use crate::sampler::gain::MAX_GAIN_Q16;
    use crate::sampler::pan::Route;
    use crate::sampler::procedural::Partial;
    use crate::sampler::scheduler::TimelineEvent;
    use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
    use crate::sampler::world::World;
    use crate::universe::layout::Layout;
    use crate::universe::observation::observation_sha256;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const RATE_HZ: u32 = 48_000;
    const CHANNELS: u32 = 2;
    const PERIOD_FRAMES: u32 = 512;
    const BUFFER_FRAMES: u32 = 1024;
    const DEFAULT_PTX: &str = "scripts/out/vole_audio.ptx";
    const DEFAULT_SECS: f64 = 1.0;
    const WAIT_TIMEOUT_MS: i32 = 300;
    const MAX_STALLS: u32 = 8;

    fn ptx_bytes() -> crate::error::Result<Option<(Vec<u8>, String)>> {
        let path = std::env::var("VOLE_CUDA_PTX")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_PTX));
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };
        let sha = hex(&Sha256::digest(&bytes));
        Ok(Some((bytes, sha)))
    }

    fn audible_requested() -> bool {
        std::env::var("VOLE_D1_EMIT_AUDIO")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    fn session_secs() -> f64 {
        std::env::var("VOLE_D1_SECS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| (0.05..=10.0).contains(v))
            .unwrap_or(DEFAULT_SECS)
    }

    fn device_override() -> Option<String> {
        std::env::var("VOLE_D1_DEVICE")
            .ok()
            .filter(|s| !s.is_empty())
    }

    fn media_frames(secs_factor: f64) -> usize {
        ((session_secs() * secs_factor * f64::from(RATE_HZ)).round() as usize).clamp(2400, 480_000)
    }

    // ---------------------------------------------------------------------
    // Fixture: a sustained stereo world covering every device-native class.
    // ---------------------------------------------------------------------

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Quiet,
        Audible,
    }

    struct Gains {
        literal: i32,
        wavetable: i32,
        osc: i32,
        noise: i32,
        bank: i32,
        constant: i32,
        stereo: i32,
        residual: i32,
    }

    fn gains(mode: Mode) -> Gains {
        match mode {
            Mode::Quiet => Gains {
                // Gains 1: code ≈ obs/2^16 — peak codes stay ≈ −80 dBFS or
                // below (silence-safe) while remaining nonzero and
                // content-rich (exact equality is still a full i32-domain
                // statement).
                literal: 1,
                wavetable: 1,
                osc: 1,
                noise: 1,
                bank: 1,
                constant: 1,
                stereo: 1,
                residual: 1,
            },
            Mode::Audible => Gains {
                literal: 1 << 12,
                wavetable: 1 << 13,
                osc: 1 << 14,
                noise: 1 << 12,
                bank: 1 << 13,
                constant: 1 << 10,
                stereo: 1 << 13,
                residual: 1 << 13,
            },
        }
    }

    /// Build the frozen D1 fixture: every voice sustains for the whole
    /// session (looped content + endless classes), stereo, deterministic.
    fn d1_fixture(mode: Mode) -> (ObjectStore, World) {
        let g = gains(mode);
        let mut store = ObjectStore::new();
        let instant = EnvelopeParams::new(0, 0, crate::sampler::envelope::ENV_UNITY, 0).unwrap();
        let mut mk_data = |rep: Representation, extent: u64, layout: Layout, data: ObjectData| {
            let d = ObjectDescriptor::new(rep, extent, layout, None).unwrap();
            store.insert(d.clone(), data).unwrap()
        };
        let lit = {
            let ramp: Vec<i32> = (0..2048)
                .map(|i| ((i as i64 - 1024) * (1 << 12)) as i32)
                .collect();
            mk_data(
                Representation::Literal,
                2048,
                Layout::Mono,
                ObjectData::Literal(
                    Literal::new(
                        &ObjectDescriptor::new(Representation::Literal, 2048, Layout::Mono, None)
                            .unwrap(),
                        ramp,
                    )
                    .unwrap(),
                ),
            )
        };
        let stereo = {
            let st: Vec<i32> = (0..1024 * 2)
                .map(|i| {
                    let l = ((i / 2) as i64 - 512) * (1 << 13);
                    let r = -l;
                    if i % 2 == 0 { l as i32 } else { r as i32 }
                })
                .collect();
            mk_data(
                Representation::Literal,
                1024,
                Layout::Stereo,
                ObjectData::Literal(
                    Literal::new(
                        &ObjectDescriptor::new(Representation::Literal, 1024, Layout::Stereo, None)
                            .unwrap(),
                        st,
                    )
                    .unwrap(),
                ),
            )
        };
        let wt = {
            let tri: Vec<i32> = (0..128)
                .map(|i| {
                    let m = (i % 128) as i64;
                    let v = if m < 64 { m } else { 128 - m };
                    ((v - 64) * (1 << 14)) as i32
                })
                .collect();
            mk_data(
                Representation::Wavetable,
                128,
                Layout::Mono,
                ObjectData::Wavetable(
                    Cycle::new(
                        &ObjectDescriptor::new(Representation::Wavetable, 128, Layout::Mono, None)
                            .unwrap(),
                        tri,
                    )
                    .unwrap(),
                ),
            )
        };
        let osc = mk_data(
            Representation::Oscillator,
            0,
            Layout::Mono,
            ObjectData::Oscillator(Oscillator::checked(220, 1 << 13).unwrap()),
        );
        let nse = mk_data(
            Representation::Noise,
            0,
            Layout::Mono,
            ObjectData::Noise(Noise::new(0xD1A7_2026)),
        );
        let cst = mk_data(
            Representation::Constant,
            0,
            Layout::Mono,
            ObjectData::Constant(Constant::new(65_536)),
        );
        let bank = mk_data(
            Representation::PartialBank,
            0,
            Layout::Mono,
            ObjectData::PartialBank(
                PartialBank::checked(
                    110,
                    vec![
                        Partial {
                            harmonic: 1,
                            amp_q16: 1 << 14,
                        },
                        Partial {
                            harmonic: 2,
                            amp_q16: 1 << 13,
                        },
                        Partial {
                            harmonic: 3,
                            amp_q16: 1 << 12,
                        },
                        Partial {
                            harmonic: 5,
                            amp_q16: 1 << 11,
                        },
                        Partial {
                            harmonic: 8,
                            amp_q16: 1 << 10,
                        },
                    ],
                )
                .unwrap(),
            ),
        );
        let resid = {
            let intrinsic: Vec<i32> = (0..512)
                .map(|i| ((i as i64 - 256) * (1 << 12)) as i32)
                .collect();
            let model = ResidualModel::Periodic {
                cycle: intrinsic[..64].to_vec(),
            };
            let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
            mk_data(
                Representation::PredictorResidual,
                512,
                Layout::Mono,
                ObjectData::PredictorResidual(
                    Residual::new(
                        &ObjectDescriptor::new(
                            Representation::PredictorResidual,
                            512,
                            Layout::Mono,
                            None,
                        )
                        .unwrap(),
                        model,
                        records,
                    )
                    .unwrap(),
                ),
            )
        };

        let mk = |object: ObjectId, trigger: i64| VoiceSpec {
            object,
            trigger_frame: trigger,
            note_off: None,
            start_pos_q24: 0,
            rate_q24: 1 << 24,
            object_channel: 0,
            route: Route::Mono(0),
            gain_q16: 1 << 15,
            pan_q16: 0,
            envelope: instant,
            loop_mode: LoopMode::Off,
            interp: Interp::Linear,
        };
        let mut events: Vec<TimelineEvent> = Vec::new();
        let voice = |events: &mut Vec<TimelineEvent>, s: VoiceSpec, gain: i32, route: Route| {
            let mut s = s;
            s.gain_q16 = gain.clamp(1, MAX_GAIN_Q16);
            s.route = route;
            events.push(TimelineEvent::VoiceOn(s));
        };
        voice(
            &mut events,
            {
                let mut s = mk(lit, 0);
                s.loop_mode = LoopMode::Region(LoopRegion::new(256, 1900).unwrap());
                s.rate_q24 = (1 << 24) + (1 << 16);
                s
            },
            g.literal,
            Route::Mono(0),
        );
        voice(
            &mut events,
            {
                let mut s = mk(lit, 0);
                s.loop_mode = LoopMode::Region(LoopRegion::new(256, 1900).unwrap());
                s.rate_q24 = (1 << 24) - (1 << 16);
                s
            },
            g.literal,
            Route::Mono(1),
        );
        voice(
            &mut events,
            {
                let mut s = mk(stereo, 0);
                s.loop_mode = LoopMode::Region(LoopRegion::new(0, 1024).unwrap());
                s.route = Route::StereoPair(0);
                s.pan_q16 = 0;
                s
            },
            g.stereo,
            Route::StereoPair(0),
        );
        voice(&mut events, mk(wt, 0), g.wavetable, Route::Mono(0));
        voice(&mut events, mk(wt, 1), g.wavetable, Route::Mono(1));
        voice(&mut events, mk(osc, 0), g.osc, Route::Mono(0));
        voice(&mut events, mk(osc, 1), g.osc, Route::Mono(1));
        voice(&mut events, mk(nse, 0), g.noise, Route::Mono(0));
        voice(&mut events, mk(bank, 0), g.bank, Route::Mono(1));
        voice(&mut events, mk(cst, 0), g.constant, Route::Mono(0));
        voice(
            &mut events,
            {
                let mut s = mk(resid, 0);
                s.loop_mode = LoopMode::Region(LoopRegion::new(0, 512).unwrap());
                s
            },
            g.residual,
            Route::Mono(1),
        );
        let world = World::new(RATE_HZ, 2, events).expect("d1 fixture world");
        (store, world)
    }

    fn peak(obs: &[i32]) -> i64 {
        obs.iter().map(|&x| i64::from(x).abs()).max().unwrap_or(0)
    }

    // ---------------------------------------------------------------------
    // Evidence records
    // ---------------------------------------------------------------------

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PathSession {
        path: String,
        endpoint: EndpointSnapshot,
        frames_committed: u64,
        chunks: u64,
        xruns: u64,
        completed: bool,
        shadow_exact: Option<bool>,
        window_hash: Option<String>,
        chunk_wall_mean_ms: Option<f64>,
        chunk_wall_median_ms: Option<f64>,
        chunk_wall_max_ms: Option<f64>,
        depth_min_frames: u64,
        depth_max_frames: u64,
        gpu_to_host_bytes: u64,
        host_copy_bytes: u64,
        endpoint_observation_bytes: u64,
        kernel_launches: u64,
        drain_state: Option<String>,
        detail: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct DeviceTrial {
        device: EndpointInfo,
        opened: bool,
        failure: Option<AlsaFailure>,
        snapshot: Option<EndpointSnapshot>,
        registration_rc: Option<i32>,
        registration_message: Option<String>,
        registered: bool,
        missing_symbol: Option<String>,
        range: Option<RegisterRange>,
        device_pointer: Option<String>,
        pointer_evidence: Option<PointerEvidence>,
        class: Verdict,
        class_detail: String,
    }

    impl DeviceTrial {
        fn failing(dev: &EndpointInfo, f: &AlsaFailure) -> DeviceTrial {
            let (v, d) = classify_failure(f);
            DeviceTrial {
                device: dev.clone(),
                opened: false,
                failure: Some(f.clone()),
                snapshot: None,
                registration_rc: None,
                registration_message: None,
                registered: false,
                missing_symbol: None,
                range: None,
                device_pointer: None,
                pointer_evidence: None,
                class: v,
                class_detail: d,
            }
        }
    }

    /// Outcome of one paced playback session (path-agnostic). The writer
    /// closure renders one contiguous mmap chunk and returns the codes it
    /// wrote; the session verifies them against the scalar oracle and commits.
    struct SessionOut {
        frames_committed: u64,
        chunks: u64,
        xruns: u64,
        wall: Vec<f64>,
        depth_min: u64,
        depth_max: u64,
        shadow_exact: Option<bool>,
        drain_state: Option<String>,
        detail: Option<String>,
    }

    fn run_session(
        pcm: &AlsaPcm,
        expected: &[i32],
        mut writer: impl FnMut(i64, u64, u64) -> std::result::Result<Vec<i32>, String>,
    ) -> SessionOut {
        let period = pcm.period_frames;
        let buffer = pcm.buffer_frames;
        let total = expected.len() as u64 / u64::from(pcm.request.channels);
        let mut frames_committed: u64 = 0;
        let mut chunks = 0u64;
        let mut xruns = 0u64;
        let mut wall: Vec<f64> = Vec::new();
        let mut depth_min = u64::MAX;
        let mut depth_max = 0u64;
        let mut shadow_exact = Some(true);
        let mut detail: Option<String> = None;
        let mut drain_state: Option<String> = None;
        let mut expected_pos = 0usize;
        let mut run = true;
        let mut stalls = 0u32;
        let mut started = false;
        while run && frames_committed < total {
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
                            run = false;
                            false
                        } else {
                            false
                        }
                    }
                    Err(_f) => {
                        // xrun/suspend while waiting: recover; the media
                        // timeline is preserved logically and the receipt
                        // records an explicit discontinuity (never a silent
                        // reset).
                        xruns += 1;
                        let _ = pcm.recover(-libc::EPIPE);
                        false
                    }
                },
                Err(f) => {
                    if f.rc == Some(-libc::EPIPE) {
                        xruns += 1;
                        let _ = pcm.recover(-libc::EPIPE);
                        false
                    } else {
                        detail = Some(format!("wait failed: {f}"));
                        false
                    }
                }
            };
            if !ready {
                continue;
            }
            let (offset, frames) = match pcm.mmap_begin(want) {
                Ok(r) => r,
                Err(f) => {
                    if f.rc == Some(-libc::EPIPE) {
                        xruns += 1;
                        let _ = pcm.recover(-libc::EPIPE);
                        continue;
                    }
                    detail = Some(format!("mmap_begin failed: {f}"));
                    break;
                }
            };
            if frames == 0 {
                detail = Some("mmap_begin returned 0 frames".into());
                break;
            }
            let t0 = Instant::now();
            let codes = match writer(frames_committed as i64, offset, frames) {
                Ok(c) => c,
                Err(e) => {
                    shadow_exact = Some(false);
                    detail = Some(e);
                    break;
                }
            };
            if codes.len() as u64 != frames * u64::from(pcm.request.channels) {
                detail = Some("writer returned a wrong code count".into());
                shadow_exact = Some(false);
                break;
            }
            // Shadow verification against the scalar oracle (in place for
            // D1; pre-copy for D0).
            let expect = &expected[expected_pos..expected_pos + codes.len()];
            expected_pos += codes.len();
            if codes != expect {
                let nbad = codes
                    .iter()
                    .zip(expect)
                    .filter(|(a, b)| a != b)
                    .take(3)
                    .map(|(a, b)| format!("{a} != {b}"))
                    .collect::<Vec<_>>()
                    .join("; ");
                shadow_exact = Some(false);
                detail = Some(format!(
                    "shadow mismatch at frame {}: {nbad} (+ more)",
                    frames_committed
                ));
                break;
            }
            match pcm.mmap_commit(offset, frames) {
                Ok(()) => wall.push(t0.elapsed().as_secs_f64() * 1e3),
                Err(f) => {
                    if f.rc == Some(-libc::EPIPE) {
                        xruns += 1;
                        let _ = pcm.recover(-libc::EPIPE);
                    } else {
                        detail = Some(format!("mmap_commit failed: {f}"));
                        break;
                    }
                }
            }
            frames_committed += frames;
            chunks += 1;
            // Explicit start after the first committed chunk (the sw
            // start-threshold does not reliably auto-start on every driver;
            // the C probe confirmed PREPARED persists after one period).
            if !started {
                started = true;
                match pcm.start() {
                    Ok(()) => {}
                    Err(f) => {
                        if f.rc == Some(-libc::EPIPE) {
                            xruns += 1;
                            let _ = pcm.recover(-libc::EPIPE);
                        } else {
                            detail = Some(format!("snd_pcm_start failed: {f}"));
                            break;
                        }
                    }
                }
            }
            if let Ok(avail) = pcm.avail() {
                let queued = buffer.saturating_sub(avail.min(buffer));
                depth_min = depth_min.min(queued);
                depth_max = depth_max.max(queued);
            }
        }
        if frames_committed > 0 {
            // Drain proves endpoint consumption of everything committed.
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

    fn finish_session(
        path: &str,
        pcm: &AlsaPcm,
        out: SessionOut,
        expected: &[i32],
        c: &Counters,
        launches: u64,
    ) -> PathSession {
        let mut wall = out.wall.clone();
        wall.sort_by(f64::total_cmp);
        let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
        PathSession {
            path: path.into(),
            endpoint: pcm.snapshot(),
            frames_committed: out.frames_committed,
            chunks: out.chunks,
            xruns: out.xruns,
            completed: out.detail.is_none() && out.frames_committed >= expected.len() as u64 / 2,
            shadow_exact: out.shadow_exact,
            window_hash: out
                .shadow_exact
                .unwrap_or(false)
                .then(|| hex(&observation_sha256(expected))),
            chunk_wall_mean_ms: (!wall.is_empty())
                .then(|| r3(wall.iter().sum::<f64>() / wall.len() as f64)),
            chunk_wall_median_ms: wall.get(wall.len() / 2).copied().map(r3),
            chunk_wall_max_ms: wall.last().copied().map(r3),
            depth_min_frames: out.depth_min,
            depth_max_frames: out.depth_max,
            gpu_to_host_bytes: c.gpu_to_host_pcm_bytes,
            host_copy_bytes: c.host_pcm_copy_bytes,
            endpoint_observation_bytes: c.endpoint_observation_bytes,
            kernel_launches: launches,
            drain_state: out.drain_state,
            detail: out.detail,
        }
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

    /// Deterministic default preference for endpoint trials (used only when
    /// VOLE_D1_DEVICE is unset): HDA analog first (a real on-board DAC), then
    /// other HDA, then USB, then everything else. This is a documented
    /// convenience order, never a result filter — every candidate still gets
    /// its own trial row.
    fn candidate_rank(dev: &EndpointInfo) -> (u8, String) {
        let kind = match (dev.driver.as_deref(), dev.pcm_label.as_deref()) {
            (Some("snd_hda_intel"), Some(l)) if l.contains("Analog") => 0u8,
            (Some("snd_hda_intel"), _) => 1,
            (Some("snd-usb-audio"), _) => 2,
            _ => 3,
        };
        (kind, dev.pcm_name.clone())
    }

    // ---------------------------------------------------------------------
    // Court
    // ---------------------------------------------------------------------

    pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
        let t0 = Instant::now();
        let audible = audible_requested();
        let mode = if audible { Mode::Audible } else { Mode::Quiet };
        let mut extras: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        extras.insert(
            "mode".into(),
            serde_json::json!(if audible {
                "audible"
            } else {
                "silence-safe probe"
            }),
        );

        // 0. PTX artifact + fixture + scalar authority window.
        let Some((ptx, artifact_sha)) = ptx_bytes()? else {
            let mut b = ReceiptBuilder::new("d1");
            b.result(Verdict::Inconclusive)
                .result_detail(format!(
                    "PTX artifact absent (looked at {DEFAULT_PTX}; set VOLE_CUDA_PTX to override)"
                ))
                .limitation("run scripts/build-cuda-device.sh first");
            let (_, path) = b.finish_write(receipts_root)?;
            println!("court d1: INCONCLUSIVE (PTX artifact absent)");
            println!("  receipt: {}", path.display());
            return Ok(Verdict::Inconclusive);
        };
        extras.insert(
            "artifact".into(),
            serde_json::json!({ "path": DEFAULT_PTX, "sha256": artifact_sha }),
        );
        let probe = match CudaProbe::capture(0) {
            Ok(Some(p)) => p,
            Ok(None) | Err(_) => {
                let mut b = ReceiptBuilder::new("d1");
                b.result(Verdict::UnsupportedByHardware)
                    .result_detail("CUDA driver/device unavailable on this host")
                    .extra("artifact_sha256", serde_json::json!(artifact_sha));
                let (_, path) = b.finish_write(receipts_root)?;
                println!("court d1: UNSUPPORTED_BY_HARDWARE (no CUDA driver/device)");
                println!("  receipt: {}", path.display());
                return Ok(Verdict::UnsupportedByHardware);
            }
        };
        extras.insert("cuda_probe".into(), serde_json::to_value(&probe)?);

        // Endpoints.
        let all = list_playback_endpoints();
        if all.is_empty() {
            let mut b = ReceiptBuilder::new("d1");
            b.result(Verdict::UnsupportedByHardware)
                .result_detail("no playback-capable ALSA PCMs found (/proc/asound/pcm)")
                .extra("cuda_probe", serde_json::to_value(&probe)?);
            let (_, path) = b.finish_write(receipts_root)?;
            println!("court d1: UNSUPPORTED_BY_HARDWARE (no playback endpoints)");
            println!("  receipt: {}", path.display());
            return Ok(Verdict::UnsupportedByHardware);
        }
        let candidates: Vec<EndpointInfo> = match device_override() {
            Some(name) => {
                let hit: Vec<EndpointInfo> =
                    all.iter().filter(|d| d.pcm_name == name).cloned().collect();
                if hit.is_empty() {
                    let mut b = ReceiptBuilder::new("d1");
                    b.result(Verdict::Inconclusive)
                        .result_detail(format!(
                            "VOLE_D1_DEVICE={name} not found among playback endpoints"
                        ))
                        .extra("endpoints", serde_json::to_value(&all)?);
                    let (_, path) = b.finish_write(receipts_root)?;
                    println!("court d1: INCONCLUSIVE (VOLE_D1_DEVICE {name} not found)");
                    println!("  receipt: {}", path.display());
                    return Ok(Verdict::Inconclusive);
                }
                hit
            }
            None => {
                let mut v = all.clone();
                v.sort_by_key(candidate_rank);
                v
            }
        };
        extras.insert("candidates".into(), serde_json::to_value(&candidates)?);

        let (store, world) = d1_fixture(mode);
        let flat = flatten(&store, &world)?;
        if u32::from(flat.output_channels) != CHANNELS {
            return Err(crate::error::Error::internal(
                "fixture channels != endpoint channels",
            ));
        }
        let oracle = ScalarOracle::new(world.clone());
        let mut sessions: Vec<PathSession> = Vec::new();
        let mut trials: Vec<DeviceTrial> = Vec::new();
        // Devices whose open already failed in pass A are not re-opened in
        // pass B (one trial per device).
        let mut open_failed: std::collections::HashSet<String> = std::collections::HashSet::new();

        // -----------------------------------------------------------------
        // Pass A — D0-mmap baseline on the first configurable endpoint
        // (0.5x duration; per-chunk metrics are directly comparable).
        // -----------------------------------------------------------------
        let d0_frames = media_frames(0.5);
        for dev in &candidates {
            let pcm = match AlsaPcm::open(&device_request(dev)) {
                Ok(p) => p,
                Err(f) => {
                    open_failed.insert(dev.pcm_name.clone());
                    trials.push(DeviceTrial::failing(dev, &f));
                    continue;
                }
            };
            // Configurable endpoint: run the D0-mmap baseline here and stop.
            if let Err(e) = pcm.set_blocking() {
                extras.insert(
                    "d0_set_blocking_error".into(),
                    serde_json::json!(e.to_string()),
                );
            }
            let expected = match oracle.observe(&store, 0, d0_frames) {
                Ok(w) => w,
                Err(e) => {
                    return Err(crate::error::Error::internal(format!(
                        "oracle window failed: {e}"
                    )));
                }
            };
            println!(
                "court d1: D0-mmap baseline on {} (peak |code| {} in {} frames)",
                dev.pcm_name,
                peak(&expected),
                expected.len() as u64 / u64::from(CHANNELS)
            );
            let mut kw = match KernelWorld::open(0, &ptx, flat.clone(), PERIOD_FRAMES as usize) {
                Ok(k) => k,
                Err(e) => {
                    return Err(crate::error::Error::internal(format!(
                        "KernelWorld::open: {e}"
                    )));
                }
            };
            let mut path_counters = Counters::new();
            let mut hostbuf: Vec<i32> = Vec::new();
            let out = run_session(&pcm, &expected, |start, _offset, frames| {
                // D0 path: render into host (DtoH + host copy), then copy the
                // codes into the endpoint region ourselves.
                hostbuf.clear();
                hostbuf.resize((frames * u64::from(CHANNELS)) as usize, 0);
                kw.render(Strategy::Standard, start, frames as usize, &mut hostbuf)
                    .map_err(|e| e.to_string())?;
                let bytes = frames * u64::from(CHANNELS) * 4;
                path_counters.gpu_to_host_pcm_bytes =
                    path_counters.gpu_to_host_pcm_bytes.saturating_add(bytes);
                path_counters.host_pcm_copy_bytes =
                    path_counters.host_pcm_copy_bytes.saturating_add(bytes);
                path_counters.endpoint_observation_bytes = path_counters
                    .endpoint_observation_bytes
                    .saturating_add(bytes);
                copy_into_region(&pcm, _offset, frames, &hostbuf).map_err(|e| e.to_string())?;
                Ok(hostbuf.clone())
            });
            let launches = kw.counters.kernel_launches;
            path_counters.kernel_launches = launches;
            sessions.push(finish_session(
                "d0-mmap",
                &pcm,
                out,
                &expected,
                &path_counters,
                launches,
            ));
            drop(kw);
            drop(pcm);
            break;
        }

        // -----------------------------------------------------------------
        // Pass B — D1 attempt: register the exact region on each candidate;
        // on the first success run the D1 session. One CUDA context for the
        // whole pass (registrations + launches share it).
        // -----------------------------------------------------------------
        let d1_frames = media_frames(1.0);
        let d1_expected = oracle.observe(&store, 0, d1_frames)?;
        let mut d1_cuda: Option<Cuda> = None;
        'd1: for dev in &candidates {
            if open_failed.contains(&dev.pcm_name) {
                // Open already failed in pass A; the trial is recorded.
                continue;
            }
            let pcm = match AlsaPcm::open(&device_request(dev)) {
                Ok(p) => p,
                Err(f) => {
                    trials.push(DeviceTrial::failing(dev, &f));
                    continue;
                }
            };
            if let Err(e) = pcm.set_blocking() {
                extras.insert(
                    "d1_set_blocking_error".into(),
                    serde_json::json!(e.to_string()),
                );
            }
            let cuda = match d1_cuda.take() {
                Some(c) => c,
                None => match Cuda::open(0) {
                    Ok(c) => c,
                    Err(e) => {
                        let mut b = ReceiptBuilder::new("d1");
                        b.result(Verdict::UnsupportedByHardware)
                            .result_detail(format!("CUDA open failed: {e}"));
                        let (_, path) = b.finish_write(receipts_root)?;
                        println!("court d1: UNSUPPORTED_BY_HARDWARE ({e})");
                        println!("  receipt: {}", path.display());
                        return Ok(Verdict::UnsupportedByHardware);
                    }
                },
            };
            let symbol_present = cuda.fns.d1_surface_complete();
            let host_register_supported = cuda.device.host_register_supported != 0;
            let base = pcm.area_base;
            let len = region_bytes(&pcm);
            let page = host_page_size();
            let range = RegisterRange::analyze(base, len as usize, page);
            // SAFETY: the ALSA mapping is live for the whole trial (pcm holds
            // it); the registration dies before pcm in this scope.
            let attempt = unsafe {
                crate::backend::cuda::direct::attempt_register(&cuda.fns, base, len as usize)
            };
            let mut trial = DeviceTrial {
                device: dev.clone(),
                opened: true,
                failure: None,
                snapshot: Some(pcm.snapshot()),
                registration_rc: None,
                registration_message: None,
                registered: false,
                missing_symbol: None,
                range: Some(range),
                device_pointer: None,
                pointer_evidence: None,
                class: Verdict::Inconclusive,
                class_detail: String::new(),
            };
            match attempt {
                RegistrationAttempt::Registered(r) => {
                    let dev_ptr = r.device_ptr.unwrap_or(0);
                    let pev = PointerEvidence::query(&cuda.fns, dev_ptr);
                    trial.registered = true;
                    trial.device_pointer = Some(format!("0x{dev_ptr:x}"));
                    trial.pointer_evidence = Some(pev.clone());
                    trial.class = Verdict::Supported;
                    trial.class_detail = "registered the exact endpoint region (DEVICEMAP)".into();
                    trials.push(trial);
                    // Run the D1 session on this device (the registration
                    // moves into the kernel world's context lifetime).
                    let mut kw = match KernelWorld::open_with(
                        cuda,
                        &ptx,
                        flat.clone(),
                        PERIOD_FRAMES as usize,
                    ) {
                        Ok(k) => k,
                        Err(e) => {
                            trials.last_mut().unwrap().class = Verdict::Inconclusive;
                            trials.last_mut().unwrap().class_detail =
                                format!("registered but KernelWorld::open_with failed: {e}");
                            drop(r);
                            drop(pcm);
                            continue;
                        }
                    };
                    let dev_base = dev_ptr;
                    let region_base = r.host_ptr as usize;
                    let frame_bytes = pcm.request.frame_bytes();
                    let mut shadow_counters = Counters::new();
                    println!(
                        "court d1: D1 session on {} (registered 0x{:x}+{}; device ptr 0x{:x})",
                        dev.pcm_name, base, len, dev_ptr
                    );
                    let out = run_session(&pcm, &d1_expected, |start, offset, frames| {
                        // Fused direct render: the kernel writes the final
                        // codes into the registered region at the chunk
                        // offset. No D0 block, no DtoH, no host copy.
                        let chunk_dev = dev_base + offset * frame_bytes;
                        kw.render_direct(start, frames as usize, chunk_dev)
                            .map_err(|e| e.to_string())?;
                        // Read the region back in place (the CUDA
                        // mapped-memory coherency contract makes the GPU
                        // writes visible after cuStreamSynchronize inside
                        // render_direct).
                        read_region(region_base, offset, frames, frame_bytes, CHANNELS as usize)
                    });
                    let launches = kw.counters.kernel_launches;
                    shadow_counters.endpoint_observation_bytes =
                        kw.counters.endpoint_observation_bytes;
                    shadow_counters.kernel_launches = launches;
                    let mut rec = finish_session(
                        "d1-direct",
                        &pcm,
                        out,
                        &d1_expected,
                        &shadow_counters,
                        launches,
                    );
                    let verdict = verdict_for_session(&rec);
                    rec.detail = rec.detail.clone().or(Some(verdict.1.to_string()));
                    sessions.push(rec);
                    // The device trial is already recorded (registered).
                    drop(kw);
                    drop(r);
                    drop(pcm);
                    break 'd1;
                }
                RegistrationAttempt::MissingSymbol(m) => {
                    trial.missing_symbol = Some(m);
                    trial.class = Verdict::UnsupportedByApi;
                    trial.class_detail =
                        "driver lacks the cuMemHostRegister surface (symbols listed)".into();
                    d1_cuda = Some(cuda);
                    drop(pcm);
                    trials.push(trial);
                    continue;
                }
                RegistrationAttempt::Failed { rc, message } => {
                    let (v, reason) =
                        HostRegistration::classify(rc, symbol_present, host_register_supported);
                    trial.registration_rc = Some(rc);
                    trial.registration_message = Some(message);
                    trial.class = v;
                    trial.class_detail = format!("{reason} (rc {rc})");
                    d1_cuda = Some(cuda);
                    drop(pcm);
                    trials.push(trial);
                    continue;
                }
            }
        }

        extras.insert("device_trials".into(), serde_json::to_value(&trials)?);
        extras.insert("sessions".into(), serde_json::to_value(&sessions)?);

        // Verdict.
        let d1_session: Option<&PathSession> = sessions.iter().find(|s| s.path == "d1-direct");
        let (verdict, detail): (Verdict, String) = if let Some(s) = d1_session {
            verdict_for_session(s)
        } else if trials.iter().any(|t| t.registered) {
            // Registered but the session never ran (shouldn't happen).
            (
                Verdict::Inconclusive,
                "registered but no session recorded".into(),
            )
        } else {
            let d0_ran = sessions.iter().any(|s| s.path == "d0-mmap");
            let failed_open = trials.iter().find(|t| !t.opened);
            let reg_fail = trials
                .iter()
                .find(|t| t.registration_rc.is_some() || t.missing_symbol.is_some());
            if d0_ran {
                let why = reg_fail
                    .map(|t| format!("{}: {}", t.device.pcm_name, t.class_detail))
                    .or_else(|| {
                        failed_open.map(|t| format!("{}: {}", t.device.pcm_name, t.class_detail))
                    })
                    .unwrap_or_else(|| "registration failed on every candidate".into());
                (
                    Verdict::FellBackToD0,
                    format!(
                        "D1 registration failed on every candidate; D0-mmap baseline ran. {why}"
                    ),
                )
            } else if let Some(t) = trials.first() {
                (t.class, format!("no endpoint usable: {}", t.class_detail))
            } else {
                (Verdict::Inconclusive, "no device was attempted".into())
            }
        };

        let mut counters = Counters::new();
        if let Some(s) = &sessions.first() {
            counters.endpoint_observation_bytes = s.endpoint_observation_bytes;
            counters.gpu_to_host_pcm_bytes = s.gpu_to_host_bytes;
            counters.host_pcm_copy_bytes = s.host_copy_bytes;
            counters.kernel_launches = s.kernel_launches;
            counters.quanta_submitted = s.chunks;
            counters.observe_endpoint_depth(s.depth_max_frames);
        }
        let ep = EndpointEvidence {
            directness: Some(
                if d1_session.is_some()
                    && d1_session.unwrap().completed
                    && d1_session.unwrap().shadow_exact == Some(true)
                    && d1_session.unwrap().xruns == 0
                {
                    crate::audio::directness::Directness::D1EndpointMapped
                        .label()
                        .into()
                } else if sessions.iter().any(|s| s.path == "d0-mmap") {
                    crate::audio::directness::Directness::D0Buffered
                        .label()
                        .into()
                } else {
                    "NONE".into()
                },
            ),
            topology: Some(crate::audio::topology::Topology::HostMapped.label().into()),
            registered_range: d1_session.map(|_| {
                trials
                    .iter()
                    .find(|t| t.registered)
                    .and_then(|t| t.range.as_ref())
                    .map(|r| r.label())
                    .unwrap_or_default()
            }),
            registration_result: d1_session.map(|_| {
                trials
                    .iter()
                    .find(|t| t.registered)
                    .and_then(|t| t.registration_message.clone())
                    .unwrap_or_else(|| "registered".into())
            }),
            device_pointer: d1_session
                .and_then(|_| trials.iter().find(|t| t.registered))
                .and_then(|t| t.device_pointer.clone()),
            synchronization_mechanism: Some(
                "cuStreamSynchronize after each fused kernel write; endpoint commit \
                 (snd_pcm_mmap_commit) only after sync"
                    .into(),
            ),
            fence_sync_evidence: Some(
                "commit ordering: kernel write -> cuStreamSynchronize -> in-place shadow \
                 verify -> snd_pcm_mmap_commit -> snd_pcm_drain at session end"
                    .into(),
            ),
            coherency_assumptions: Some(
                "CUDA mapped-host-memory (DEVICEMAP) coherency contract: GPU writes to the \
                 registered region are visible to the host after cuStreamSynchronize; the \
                 endpoint DMA reads the same physical pages the GPU wrote"
                    .into(),
            ),
            hidden_staging_investigation: Some(
                "path uses the actual ALSA hw:mmap region (access MMAP_INTERLEAVED on hw: \
                 device, no plug conversion). No application PCM staging exists in the D1 \
                 path; the region itself is the endpoint's DMA ring. D0-mmap baseline \
                 measures the exact copy bytes D1 removes"
                    .into(),
            ),
            endpoint_clock: d1_session.map(|s| {
                format!(
                    "alsa avail-paced; endpoint depth min {} max {} frames; drain state {:?}",
                    s.depth_min_frames, s.depth_max_frames, s.drain_state
                )
            }),
        };

        let params = CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1".into()),
            backend: Some("cuda-d1".into()),
            sample_rate_hz: Some(RATE_HZ),
            channels: Some(CHANNELS),
            quantum_frames: Some(PERIOD_FRAMES),
            content_kind: Some(if audible {
                "phase-h-d1-audible-demo".into()
            } else {
                "phase-h-d1-silence-safe-probe".into()
            }),
            ..Default::default()
        };
        let mut builder = ReceiptBuilder::new("d1");
        builder
            .result(verdict)
            .result_detail(detail.clone())
            .params(params)
            .counters(counters)
            .timing(RunTiming {
                total_ns: Some(t0.elapsed().as_nanos() as i64),
                ..Default::default()
            })
            .provenance(Provenance {
                gpu_artifact_hash: Some(artifact_sha.clone()),
                benchmark_order: vec!["d0-mmap".into(), "d1-direct".into()],
                ..Default::default()
            })
            .endpoint(ep);
        for (k, v) in extras {
            builder.extra(k, v);
        }
        builder.limitation(if audible {
            "audible content emitted (opt-in --emit-audio)"
        } else {
            "silence-safe content (peak codes <= 2^16 ~ -80 dBFS); endpoint consumption is \
             proven by commit/avail/drain evidence, not by listening"
        });
        let (env, path) = builder.finish_write(receipts_root)?;

        println!("court d1: {verdict}");
        println!("  {detail}");
        if let Some(d) = d1_session {
            println!(
                "  d1-direct: {} frames in {} chunks; shadow_exact={:?}; xruns={}; depth {}..{} frames",
                d.frames_committed,
                d.chunks,
                d.shadow_exact,
                d.xruns,
                d.depth_min_frames,
                d.depth_max_frames
            );
        }
        for s in &sessions {
            println!(
                "  {}: {} frames, {} chunks, gpu->host {} B, host copy {} B, endpoint obs {} B",
                s.path,
                s.frames_committed,
                s.chunks,
                s.gpu_to_host_bytes,
                s.host_copy_bytes,
                s.endpoint_observation_bytes
            );
        }
        for t in &trials {
            println!(
                "  {}: {} {}",
                t.device.pcm_name,
                t.class.label(),
                t.class_detail
            );
        }
        println!(
            "  git: {}",
            env.receipt
                .environment
                .git_identity()
                .unwrap_or_else(|| "?".into())
        );
        println!("  receipt: {}", path.display());
        Ok(verdict)
    }

    fn verdict_for_session(s: &PathSession) -> (Verdict, String) {
        match s.shadow_exact {
            Some(false) => (
                Verdict::FailedCorrectness,
                format!(
                    "D1 path wrote codes that differ from the scalar oracle: {}",
                    s.detail.clone().unwrap_or_default()
                ),
            ),
            Some(true) if s.completed && s.xruns == 0 => (
                Verdict::Supported,
                "registered the actual ALSA endpoint region; GPU wrote every chunk \
                 byte-exact vs the scalar oracle; committed; drained without xrun"
                    .to_string(),
            ),
            Some(true) if s.xruns > 0 => (
                Verdict::FailedDeadline,
                format!(
                    "shadow equality held but {} xrun(s) occurred (explicit discontinuities recorded)",
                    s.xruns
                ),
            ),
            Some(true) => (
                Verdict::FailedDeadline,
                format!(
                    "session incomplete: {}",
                    s.detail.clone().unwrap_or_default()
                ),
            ),
            None => (
                Verdict::FailedDeadline,
                format!(
                    "session ended before verification: {}",
                    s.detail.clone().unwrap_or_default()
                ),
            ),
        }
    }

    /// CPU copy of codes into the endpoint region at `offset` (D0 baseline).
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

    /// In-place host read of the codes the GPU wrote at `offset` (D1 shadow
    /// verification — a read of the registered mapping, not a copy).
    fn read_region(
        region_base: usize,
        offset: u64,
        frames: u64,
        frame_bytes: u64,
        channels: usize,
    ) -> std::result::Result<Vec<i32>, String> {
        let addr = region_base + (offset * frame_bytes) as usize;
        let len = (frames * channels as u64) as usize;
        // SAFETY: the registered ALSA mapping is live and readable; the GPU
        // writes are visible after cuStreamSynchronize (recorded coherency
        // contract).
        let codes: &[i32] = unsafe { std::slice::from_raw_parts(addr as *const i32, len) };
        Ok(codes.to_vec())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn window(mode: Mode) -> (Vec<i32>, i64) {
            let (store, world) = d1_fixture(mode);
            let out = ScalarOracle::new(world)
                .observe(&store, 0, 8192)
                .expect("fixture window");
            let p = peak(&out);
            (out, p)
        }

        #[test]
        fn quiet_mode_is_silence_safe_and_content_rich() {
            let (out, p) = window(Mode::Quiet);
            // Silence-safe: codes stay ~−80 dBFS or below. Content-rich:
            // nonzero codes exist (exact equality is a real statement).
            assert!(p <= 1 << 16, "quiet peak {p} too loud");
            assert!(out.iter().any(|&x| x != 0), "quiet fixture is all-zero");
        }

        #[test]
        fn audible_mode_is_audible_and_unclipped() {
            let (_out, p) = window(Mode::Audible);
            assert!(p >= 1 << 18, "audible peak {p} too quiet");
            assert!(p < (1 << 30), "audible peak {p} near clipping");
        }

        #[test]
        fn d1_fixture_flattens_exactly() {
            // The D1 court renders a *flattened* world on the GPU; the
            // fixture must be flatten-exact (parity anchor before hardware).
            let (store, world) = d1_fixture(Mode::Quiet);
            let flat = flatten(&store, &world).expect("flatten");
            let oracle = ScalarOracle::new(world.clone());
            for (start, frames) in [(0i64, 4096usize), (777, 512), (10_000, 2048)] {
                let want = oracle.observe(&store, start, frames).expect("oracle");
                let got = flat.render(start, frames);
                assert_eq!(want, got, "flat != scalar at [{start},+{frames})");
            }
        }
    }
}
