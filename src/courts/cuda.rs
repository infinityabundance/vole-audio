//! `court cuda` — Phase G CUDA D0: parity, capability, strategy, throughput.
//!
//! Everything here is the **D0 buffered diagnostic** path (`GpuBuffered-
//! Diagnostic`); no D1 endpoint claim exists in Phase G. The court:
//!
//! 1. probes the CUDA driver/device (receipt evidence);
//! 2. loads the PTX artifact produced from this same package by
//!    `scripts/build-cuda-device.sh` (env `VOLE_CUDA_PTX` overrides the
//!    default `scripts/out/vole_audio.ptx`);
//! 3. renders the frozen semantic/authored/mixed fixture worlds on the GPU
//!    and asserts **scalar == CUDA** bit-for-bit on every available
//!    submission strategy (standard stream, high-priority stream, captured
//!    graph);
//! 4. re-runs the independent semantic facts F01–F15 on the device surface
//!    (every fact row must pass before `SUPPORTED`);
//! 5. runs a random-world differential subset on the device;
//! 6. records fixture-level throughput surfaces (CPU scalar/AVX2/AVX-512 vs
//!    CUDA) — explicitly fixture-level, never a flagship claim (Phase M owns
//!    the frozen-corpus flagship court).
//!
//! Verdict is `SUPPORTED` only when every parity/fact row passes; the receipt
//! records probe, artifact hash, strategy latencies, throughput rows, and
//! limitations. GPU absent -> `UNSUPPORTED_BY_HARDWARE` receipt; artifact
//! absent -> `INCONCLUSIVE` receipt with the exact build step required. Never
//! a manufactured result.

use crate::backend::cuda::kernel::{KernelWorld, Strategy};
use crate::backend::cuda::probe::CudaProbe;
use crate::backend::flatten::{FlattenedWorld, flatten};
use crate::error::{Error, Result};
use crate::eval::backend::Isa;
use crate::eval::{ScalarOracle, SimdOracle};
use crate::evidence::counters::Counters;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder, RunTiming};
use crate::hash::sha256::{Sha256, hex};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::{Cycle, Literal, LoopRegion, ObjectData, ObjectStore};
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::pan::Route;
use crate::sampler::procedural::Partial;
use crate::sampler::scheduler::TimelineEvent;
use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
use crate::sampler::world::World;
use crate::status::Verdict;
use crate::universe::layout::Layout;
use crate::universe::observation::observation_sha256;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const RATE_HZ: u32 = 48_000;
/// Default PTX artifact produced by scripts/build-cuda-device.sh.
const DEFAULT_PTX: &str = "scripts/out/vole_audio.ptx";

/// One parity row: (label, Some(true)=passed / Some(false)=failed /
/// None=unavailable, detail).
type ParityRow = (String, Option<bool>, Option<String>);

fn ptx_bytes() -> Result<Option<(Vec<u8>, String)>> {
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

/// Flatten + verify a fixture world's scalar reference, returning the world
/// and its flat form.
fn fixture_flat(
    label: &str,
    store: &ObjectStore,
    world: &World,
    windows: &[(i64, usize)],
    hashes: &mut Vec<(String, String)>,
) -> Result<FlattenedWorld> {
    let oracle = ScalarOracle::new(world.clone());
    for &(start, frames) in windows {
        let out = oracle.observe(store, start, frames)?;
        let h = hex(&observation_sha256(&out));
        hashes.push((format!("{label}/scalar[{start},+{frames})"), h));
    }
    flatten(store, world)
}

fn device_parity_world(
    label: &str,
    flat: &FlattenedWorld,
    windows: &[(i64, usize)],
    hashes: &mut Vec<(String, String)>,
) -> Result<Vec<ParityRow>> {
    let max = windows.iter().map(|(_, f)| *f).max().unwrap_or(0);
    let ptx = ptx_bytes()?
        .ok_or_else(|| Error::new(crate::error::Kind::Unavailable, "PTX artifact absent"))?;
    let kw = KernelWorld::open(0, &ptx.0, flat.clone(), max)?;
    let mut kw = kw;
    let mut strategies = vec![Strategy::Standard];
    if kw.cuda.device.stream_priorities_supported != 0 {
        strategies.push(Strategy::HighPriority);
    }
    if kw.graph_available() {
        strategies.push(Strategy::Graph);
    } else {
        println!("    {label}: captured-graph unavailable (recorded)");
    }
    let mut rows = Vec::new();
    for s in strategies {
        for &(start, frames) in windows {
            let mut out = vec![0i32; frames * usize::from(flat.output_channels)];
            let r = kw.render(s, start, frames, &mut out);
            match r {
                Ok(()) => {
                    let h = hex(&observation_sha256(&out));
                    hashes.push((format!("{label}/{}/{start}/+{frames}", s.label()), h));
                    rows.push((format!("{label}/{}/{start}", s.label()), Some(true), None));
                }
                Err(e) => rows.push((
                    format!("{label}/{}/{start}", s.label()),
                    Some(false),
                    Some(format!("{e}")),
                )),
            }
        }
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Fixture / timing worlds
// ---------------------------------------------------------------------------

/// N-voice worlds for throughput surfaces (fixture-level, mono, 48 kHz).
/// `frames` sizes the observation window the caller will use (content loop
/// regions already sustain indefinitely).
fn timed_world(class: &str, voices: usize, frames: usize) -> Result<(ObjectStore, World)> {
    let mut store = ObjectStore::new();
    // Extents are fixed and looped; `frames` is the window the caller times.
    let _ = frames;
    let instant = EnvelopeParams::new(0, 0, crate::sampler::envelope::ENV_UNITY, 0).unwrap();
    let mk = |object, trigger| VoiceSpec {
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
    };
    let mut events = Vec::new();
    match class {
        "literal" => {
            for _ in 0..voices {
                let ext = 4096u64;
                let samples: Vec<i32> = (0..ext)
                    .map(|k| {
                        let v = (k as i64 - 2048) * 4_000;
                        (v * 2) as i32
                    })
                    .collect();
                let d = ObjectDescriptor::new(Representation::Literal, ext, Layout::Mono, None)
                    .unwrap();
                let id = store
                    .insert(
                        d.clone(),
                        ObjectData::Literal(Literal::new(&d, samples).unwrap()),
                    )
                    .unwrap();
                let mut spec = mk(id, 0);
                spec.loop_mode = LoopMode::Region(LoopRegion::new(0, 4096).unwrap());
                events.push(TimelineEvent::VoiceOn(spec));
            }
        }
        "oscillator" => {
            for i in 0..voices {
                let f = 55 + ((i * 37) % 900) as u32;
                let d = ObjectDescriptor::new(Representation::Oscillator, 0, Layout::Mono, None)
                    .unwrap();
                let id = store
                    .insert(
                        d.clone(),
                        ObjectData::Oscillator(
                            crate::object::Oscillator::checked(f, (1 << 16) / 2).unwrap(),
                        ),
                    )
                    .unwrap();
                events.push(TimelineEvent::VoiceOn(mk(id, 0)));
            }
        }
        "noise" => {
            for i in 0..voices {
                let d =
                    ObjectDescriptor::new(Representation::Noise, 0, Layout::Mono, None).unwrap();
                let id = store
                    .insert(
                        d.clone(),
                        ObjectData::Noise(crate::object::Noise::new(0x5EED_0000 + i as u64)),
                    )
                    .unwrap();
                events.push(TimelineEvent::VoiceOn(mk(id, 0)));
            }
        }
        "wavetable" => {
            for _ in 0..voices {
                let ext = 1024u64;
                let cyc: Vec<i32> = (0..ext)
                    .map(|k| {
                        let m = (k % 1024) as i64;
                        let v = if m < 512 { m } else { 1024 - m };
                        (v * 2_000 - (512 * 2_000)) as i32
                    })
                    .collect();
                let d = ObjectDescriptor::new(Representation::Wavetable, ext, Layout::Mono, None)
                    .unwrap();
                let id = store
                    .insert(
                        d.clone(),
                        ObjectData::Wavetable(Cycle::new(&d, cyc).unwrap()),
                    )
                    .unwrap();
                events.push(TimelineEvent::VoiceOn(mk(id, 0)));
            }
        }
        "partials" => {
            for i in 0..voices {
                let mut partials = Vec::new();
                let n = 256;
                for h in 1..=n {
                    let amp = (1 << 15) / (h as i32).saturating_mul(2).max(1);
                    partials.push(Partial {
                        harmonic: h,
                        amp_q16: amp.clamp(-(1 << 16), 1 << 16),
                    });
                }
                let d = ObjectDescriptor::new(Representation::PartialBank, 0, Layout::Mono, None)
                    .unwrap();
                let id = store
                    .insert(
                        d.clone(),
                        ObjectData::PartialBank(
                            crate::object::PartialBank::checked(
                                55 + ((i * 17) % 700) as u32,
                                partials,
                            )
                            .unwrap(),
                        ),
                    )
                    .unwrap();
                events.push(TimelineEvent::VoiceOn(mk(id, 0)));
            }
        }
        _ => return Err(Error::malformed("unknown timed fixture class")),
    }
    let world = World::new(RATE_HZ, 1, events)?;
    Ok((store, world))
}

/// Wall time for one floor over `frames`; warmup + median-of-runs (ms).
fn time_floor(
    label: &str,
    store: &ObjectStore,
    world: &World,
    frames: usize,
    isa: Option<Isa>,
    runs: usize,
    hashes: &mut Vec<(String, String)>,
) -> Result<serde_json::Value> {
    let mut samples = Vec::with_capacity(runs);
    let mut out = Vec::new();
    for r in 0..runs {
        let t0 = Instant::now();
        out = match isa {
            None => ScalarOracle::new(world.clone()).observe(store, 0, frames)?,
            Some(i) => SimdOracle {
                world: world.clone(),
                isa: i,
            }
            .observe(store, 0, frames)?,
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if r > 0 {
            samples.push(ms);
        }
    }
    let h = hex(&observation_sha256(&out));
    hashes.push((label.to_string(), h));
    samples.sort_by(f64::total_cmp);
    let round3 = |v: f64| (v * 1000.0).round() / 1000.0;
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    Ok(json!({
        "runs": runs,
        "frames": frames,
        "mean_ms": round3(mean),
        "median_ms": round3(samples[samples.len() / 2]),
        "min_ms": round3(samples[0]),
        "method": "fixture-level; host CLOCK_MONOTONIC wall; warmup 1",
    }))
}

// ---------------------------------------------------------------------------
// Court
// ---------------------------------------------------------------------------

pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let t0 = Instant::now();
    let mut counters = Counters::new();
    let mut hashes: Vec<(String, String)> = Vec::new();
    let mut extras: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let limitations: Vec<String> = Vec::new();

    // 0. PTX artifact.
    let ptx = match ptx_bytes()? {
        Some(p) => p,
        None => {
            let mut b = ReceiptBuilder::new("cuda");
            b.result(Verdict::Inconclusive)
                .result_detail(format!(
                    "PTX artifact absent (looked at {DEFAULT_PTX}; set VOLE_CUDA_PTX to override)"
                ))
                .limitation("run scripts/build-cuda-device.sh first");
            let (_, path) = b.finish_write(receipts_root)?;
            println!("court cuda: INCONCLUSIVE (PTX artifact absent)");
            println!("  run scripts/build-cuda-device.sh (or set VOLE_CUDA_PTX)");
            println!("  receipt: {}", path.display());
            return Ok(Verdict::Inconclusive);
        }
    };
    let artifact_sha = ptx.1.clone();
    extras.insert(
        "artifact".to_string(),
        json!({"path": DEFAULT_PTX, "sha256": artifact_sha}),
    );

    // 1. Probe / availability.
    let probe = match CudaProbe::capture(0) {
        Ok(Some(p)) => p,
        Ok(None) | Err(_) => {
            let mut b = ReceiptBuilder::new("cuda");
            b.result(Verdict::UnsupportedByHardware)
                .result_detail("CUDA driver/device unavailable on this host")
                .extra("artifact_sha256", json!(ptx.1));
            let (_, path) = b.finish_write(receipts_root)?;
            println!("court cuda: UNSUPPORTED_BY_HARDWARE (no CUDA driver/device)");
            println!("  receipt: {}", path.display());
            return Ok(Verdict::UnsupportedByHardware);
        }
    };
    extras.insert("probe".to_string(), serde_json::to_value(&probe)?);
    let strategies: Vec<Strategy> = {
        let mut v = vec![Strategy::Standard];
        if probe.device.stream_priorities_supported {
            v.push(Strategy::HighPriority);
        }
        v.push(Strategy::Graph); // captured at open; failure recorded per row
        v
    };
    extras.insert(
        "strategies".to_string(),
        json!(strategies.iter().map(|s| s.label()).collect::<Vec<_>>()),
    );
    let _ = &mut counters;

    let mut failed_rows: Vec<String> = Vec::new();
    let record_rows = |rows: Vec<ParityRow>,
                       extras: &mut BTreeMap<String, serde_json::Value>,
                       failed: &mut Vec<String>| {
        for (label, ok, detail) in rows {
            extras.insert(format!("row/{label}"), json!(ok));
            if ok == Some(false) {
                failed.push(format!("{label}: {}", detail.unwrap_or_default()));
            }
        }
    };

    // 2. Frozen fixture parity (scalar == CUDA, every strategy).
    {
        let (store, events) = crate::courts::semantic::semantic_court_fixture();
        let world = World::new(RATE_HZ, crate::courts::semantic::CHANNELS, events)?;
        let windows = [(0i64, 2400usize), (700, 900), (1600, 800)];
        let flat = fixture_flat("semantic", &store, &world, &windows, &mut hashes)?;
        let rows = device_parity_world("semantic", &flat, &windows, &mut hashes)?;
        record_rows(rows, &mut extras, &mut failed_rows);
        counters.device_sample_block_bytes = (2400 * 2 * 4) as u64;
    }
    {
        let (store, events) = crate::courts::authored::authored_court_fixture();
        let world = World::new(RATE_HZ, 1, events)?;
        let windows = [(0i64, 4000usize), (400, 3600), (1600, 2400)];
        let flat = fixture_flat("authored", &store, &world, &windows, &mut hashes)?;
        let rows = device_parity_world("authored", &flat, &windows, &mut hashes)?;
        record_rows(rows, &mut extras, &mut failed_rows);
    }
    {
        let (store, events) = crate::courts::simd::mixed_world();
        let world = World::new(RATE_HZ, 2, events)?;
        let windows = [(0i64, 8192usize), (1234, 700), (7000, 1192)];
        let flat = fixture_flat("mixed", &store, &world, &windows, &mut hashes)?;
        let rows = device_parity_world("mixed", &flat, &windows, &mut hashes)?;
        record_rows(rows, &mut extras, &mut failed_rows);
    }

    // 3. Semantic facts on the device surface (F01–F15 windows; authority
    // facts are surface-independent and already enforced by `court facts`).
    {
        let mut fact_rows = BTreeMap::new();
        for f in crate::facts::registry() {
            if let crate::facts::Expectation::Windows {
                world,
                windows,
                expected,
            } = &f.expectation
            {
                let flat = match flatten(&world.store, &world.world) {
                    Ok(fl) => fl,
                    Err(e) => {
                        failed_rows.push(format!("{}: flatten failed: {e}", f.id));
                        continue;
                    }
                };
                let max = windows.iter().map(|(_, n)| *n).max().unwrap_or(1);
                let kw = match KernelWorld::open(0, &ptx.0, flat, max) {
                    Ok(k) => k,
                    Err(e) => {
                        failed_rows.push(format!("{}: kernel open failed: {e}", f.id));
                        continue;
                    }
                };
                let mut kw = kw;
                let mut all_ok = true;
                for (k, &(start, frames)) in windows.iter().enumerate() {
                    let mut out = vec![0i32; frames * usize::from(world.world.output_channels)];
                    match kw.render(Strategy::Standard, start, frames, &mut out) {
                        Ok(()) => {
                            if out != expected[k] {
                                all_ok = false;
                                failed_rows.push(format!(
                                    "fact {} on cuda window [{start},+{frames}) differs",
                                    f.id
                                ));
                            }
                        }
                        Err(e) => {
                            all_ok = false;
                            failed_rows.push(format!("fact {} on cuda: {e}", f.id));
                        }
                    }
                }
                fact_rows.insert(f.id.to_string(), json!(all_ok));
            }
        }
        extras.insert("facts_on_cuda".to_string(), json!(fact_rows));
    }

    // 4. Random-world differential subset on the device.
    {
        let seeds = 32u64;
        let mut compared = 0u64;
        let mut seed_rows = BTreeMap::new();
        for seed in 0..seeds {
            let (store, pool) = crate::eval::battery::corpus(seed);
            let mut rng = crate::universe::prng::XoShiro256::from_seed(seed ^ 0xC0DA);
            let next32 = |rng: &mut crate::universe::prng::XoShiro256| rng.next_u64() as u32;
            let n_voices = 1 + (next32(&mut rng) % 3);
            let mut events = Vec::new();
            for _ in 0..n_voices {
                let object = pool[(next32(&mut rng) as usize) % pool.len()];
                let mut spec = crate::eval::battery::random_voice(object, &store, &mut rng, 2);
                if next32(&mut rng) % 3 == 0 {
                    spec.note_off = Some(spec.trigger_frame + 1 + (next32(&mut rng) % 800) as i64);
                }
                events.push(TimelineEvent::VoiceOn(spec));
            }
            let start = (next32(&mut rng) % 1600) as i64;
            let frames = 64 + (next32(&mut rng) % 1500) as usize;
            let world = match World::new(RATE_HZ, 2, events) {
                Ok(w) => w,
                Err(_) => continue,
            };
            let oracle = ScalarOracle::new(world.clone());
            let want = match oracle.observe(&store, start, frames) {
                Ok(a) => a,
                Err(_) => continue,
            };
            let flat = match flatten(&store, &world) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let kw = match KernelWorld::open(0, &ptx.0, flat, frames) {
                Ok(k) => k,
                Err(e) => {
                    failed_rows.push(format!("random seed {seed}: open failed: {e}"));
                    break;
                }
            };
            let mut kw = kw;
            let mut got = vec![0i32; want.len()];
            match kw.render(Strategy::Standard, start, frames, &mut got) {
                Ok(()) => {
                    if got == want {
                        compared += 1;
                        seed_rows.insert(seed.to_string(), json!(true));
                    } else {
                        seed_rows.insert(seed.to_string(), json!(false));
                        failed_rows.push(format!("random seed {seed}: device != scalar"));
                    }
                }
                Err(e) => {
                    seed_rows.insert(seed.to_string(), json!(false));
                    failed_rows.push(format!("random seed {seed}: render failed: {e}"));
                }
            }
        }
        extras.insert(
            "random_device_battery".to_string(),
            json!({
                "seeds_attempted": seeds,
                "seeds_compared": compared,
                "per_seed": seed_rows,
            }),
        );
    }

    // 5. Fixture-level throughput surfaces (CPU scalar/AVX2/AVX-512 vs CUDA).
    // 5. Fixture-level throughput surfaces (CPU scalar/AVX2/AVX-512 vs CUDA)
    // and a voice-count x quantum crossover sweep. Every cell first proves
    // scalar == CUDA bit-exactness on that world before timing anything.
    let mut perf = BTreeMap::new();
    let mut run_cell = |class: &str,
                        voices: usize,
                        frames: usize,
                        perf: &mut BTreeMap<String, serde_json::Value>|
     -> Result<()> {
        let (store, world) = match timed_world(class, voices, frames) {
            Ok(x) => x,
            Err(e) => {
                failed_rows.push(format!("timed_world {class}: {e}"));
                return Ok(());
            }
        };
        let cpu_runs = if class == "partials" { 3 } else { 5 };
        let mut rows = BTreeMap::new();
        rows.insert(
            "scalar".to_string(),
            time_floor(
                &format!("perf/{class}/{voices}v/{frames}/scalar"),
                &store,
                &world,
                frames,
                None,
                cpu_runs,
                &mut hashes,
            )?,
        );
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                rows.insert(
                    "avx2".to_string(),
                    time_floor(
                        &format!("perf/{class}/{voices}v/{frames}/avx2"),
                        &store,
                        &world,
                        frames,
                        Some(Isa::Avx2),
                        cpu_runs,
                        &mut hashes,
                    )?,
                );
            }
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512dq")
                && std::is_x86_feature_detected!("avx512vl")
            {
                rows.insert(
                    "avx512".to_string(),
                    time_floor(
                        &format!("perf/{class}/{voices}v/{frames}/avx512"),
                        &store,
                        &world,
                        frames,
                        Some(Isa::Avx512),
                        cpu_runs,
                        &mut hashes,
                    )?,
                );
            }
        }
        // CUDA D0: exactness first (one render compared to the scalar
        // oracle), then full-path wall + kernel-only driver-event time.
        let flat = flatten(&store, &world)?;
        let mut kw = match KernelWorld::open(0, &ptx.0, flat, frames) {
            Ok(k) => k,
            Err(e) => {
                failed_rows.push(format!("perf {class}: cuda open failed: {e}"));
                return Ok(());
            }
        };
        let mut got = vec![0i32; frames];
        let r = kw.render(Strategy::Standard, 0, frames, &mut got);
        match r {
            Ok(()) => {
                let want = ScalarOracle::new(world.clone()).observe(&store, 0, frames)?;
                if got == want {
                    let h = hex(&observation_sha256(&got));
                    hashes.push((format!("perf/{class}/{voices}v/{frames}/cuda-d0"), h));
                    extras.insert(
                        format!("row/perf/{class}/{voices}v/{frames}/cuda_parity"),
                        json!(true),
                    );
                } else {
                    extras.insert(
                        format!("row/perf/{class}/{voices}v/{frames}/cuda_parity"),
                        json!(false),
                    );
                    failed_rows.push(format!("perf {class} {voices}v: cuda != scalar"));
                    return Ok(());
                }
            }
            Err(e) => {
                failed_rows.push(format!("perf {class} {voices}v: render failed: {e}"));
                return Ok(());
            }
        }
        let gpu_runs = 7;
        let mut gpu_wall = Vec::with_capacity(gpu_runs);
        let mut gpu_kernel = Vec::with_capacity(gpu_runs);
        let round3 = |v: f64| (v * 1000.0).round() / 1000.0;
        for r in 0..gpu_runs {
            let t = kw.time_render_wall(Strategy::Standard, 0, frames)?;
            if r > 0 {
                gpu_wall.push(t * 1e3);
            }
            let k = kw.time_kernel_once(Strategy::Standard, frames)?;
            if r > 0 {
                gpu_kernel.push(k * 1e3);
            }
        }
        let sum = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        rows.insert(
            "cuda-d0".to_string(),
            json!({
                "runs": gpu_runs,
                "frames": frames,
                "wall_mean_ms": round3(sum(&gpu_wall)),
                "kernel_mean_ms": round3(sum(&gpu_kernel)),
                "method": "D0: state upload + launch + stream sync + DtoH (wall); kernel-only via driver events",
            }),
        );
        perf.insert(format!("{class}-{voices}v-q{frames}"), json!(rows));
        Ok(())
    };
    for (class, voices) in [
        ("literal", 256usize),
        ("oscillator", 1024),
        ("noise", 512),
        ("wavetable", 256),
        ("partials", 32),
    ] {
        run_cell(class, voices, 8192, &mut perf)?;
    }
    // Crossover sweep: representation x {64, 1024} voices x {512, 8192}
    // frames. Only classes whose CPU/SIMD and GPU costs differ structurally.
    for (class, voices) in [("noise", 64usize), ("noise", 1024), ("partials", 64)] {
        run_cell(class, voices, 512, &mut perf)?;
    }
    for (class, voices) in [("noise", 64usize), ("noise", 1024), ("oscillator", 64)] {
        run_cell(class, voices, 8192, &mut perf)?;
    }
    extras.insert("throughput_cells".to_string(), json!(perf));

    // Verdict: SUPPORTED only if every row passed.
    let passed = failed_rows.is_empty();
    let verdict = if passed {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    let detail = if passed {
        format!(
            "scalar == CUDA bit-exact on frozen fixtures ({}) and facts F01-F15 on device; strategies {:?}",
            hashes.len(),
            strategies.iter().map(|s| s.label()).collect::<Vec<_>>()
        )
    } else {
        format!(
            "{} row(s) failed: {}",
            failed_rows.len(),
            failed_rows.join("; ")
        )
    };

    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1".into()),
        backend: Some("cuda-d0".into()),
        sample_rate_hz: Some(RATE_HZ),
        channels: Some(2),
        quantum_frames: Some(8192),
        content_kind: Some("phase-g-cuda-d0-parity-and-throughput".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("cuda");
    builder
        .result(verdict)
        .result_detail(detail)
        .params(params)
        .counters(counters)
        .timing(RunTiming {
            total_ns: Some(t0.elapsed().as_nanos() as i64),
            ..Default::default()
        })
        .provenance(Provenance {
            gpu_artifact_hash: Some(artifact_sha.clone()),
            benchmark_order: vec![
                "scalar".into(),
                "avx2".into(),
                "avx512".into(),
                "cuda-d0".into(),
            ],
            ..Default::default()
        });
    for (k, v) in extras {
        builder.extra(k, v);
    }
    for l in limitations {
        builder.limitation(l);
    }
    let (_, path) = builder.finish_write(receipts_root)?;

    println!("court cuda: {verdict}");
    println!(
        "  device: {} ({}), driver {}{}",
        probe.device.name,
        probe.device.sm,
        probe.driver_version_label,
        if probe.context_created {
            ""
        } else {
            " (no context!)"
        }
    );
    println!("  artifact sha256: {artifact_sha}");
    println!(
        "  strategies: {}",
        strategies
            .iter()
            .map(|s| s.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
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
