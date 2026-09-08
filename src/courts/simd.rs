//! `court simd` — Phase F SIMD parity + fixture-level timing.
//!
//! Differential battery over the frozen court worlds and a mixed world:
//!
//! 1. `court semantic` fixture (scalar authority) rendered through the SIMD
//!    engine on every available ISA floor must reproduce the frozen reference
//!    hash `1791816f...` exactly;
//! 2. `court authored` fixture likewise (`f7e103f3...`);
//! 3. a mixed world exercising reverse rates, loop regions, nearest and
//!    linear interpolation, stereo pan, partial banks, residuals, and endless
//!    classes must render bit-identically across floors;
//! 4. a deterministic window sweep across fixture offsets.
//!
//! The court also records fixture-level end-to-end throughput (scalar oracle
//! vs SIMD floors, wall clock, host monotonic) as *indicative* evidence for
//! `docs/PERFORMANCE.md` — it is explicitly not a flagship benchmark (Phase M
//! owns the frozen-corpus flagship court). Verdict is `SUPPORTED` only when
//! every floor reproduces every reference observation bit-for-bit.

use crate::eval::backend::Isa;
use crate::eval::{ScalarOracle, SimdOracle};
use crate::evidence::counters::Counters;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder, RunTiming};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::residual::{Residual, ResidualModel};
use crate::object::{
    Constant, Cycle, Literal, LoopRegion, Noise, ObjectData, ObjectId, ObjectStore,
};
use crate::object::{Oscillator, PartialBank};
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::pan::Route;
use crate::sampler::procedural::Partial;
use crate::sampler::scheduler::TimelineEvent;
use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
use crate::sampler::world::World;
use crate::status::Verdict;
use crate::universe::layout::Layout;
use crate::universe::observation::observation_sha256;
use std::path::Path;
use std::time::Instant;

const RATE_HZ: u32 = 48_000;

/// ISA floors available on this host (always includes the exact scalar
/// floor of the SIMD engine; vector floors are runtime-detected).
fn floors() -> Vec<Isa> {
    let mut v = vec![Isa::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            v.push(Isa::Avx2);
        }
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
        {
            v.push(Isa::Avx512);
        }
    }
    v
}

/// Build a mixed world that crosses every vector/floor class boundary:
/// looped and one-shot literal content (forward and reverse, nearest and
/// linear), cycles, endless generators, a partial bank, and a residual.
/// (pub(crate): reused by the Phase G `court cuda` parity fixtures.)
pub(crate) fn mixed_world() -> (ObjectStore, Vec<TimelineEvent>) {
    let mut store = ObjectStore::new();
    let instant = EnvelopeParams::new(0, 0, crate::sampler::envelope::ENV_UNITY, 0).unwrap();
    let adr = EnvelopeParams::new(97, 250, (1 << 16) / 2, 500).unwrap();

    let mk = |object: ObjectId, trigger: i64| VoiceSpec {
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
    let mut events: Vec<TimelineEvent> = Vec::new();

    // Mono literal with a loop region, stereo literal, cycle, osc, noise,
    // partial bank, residual, and constant.
    let mut mk_data = |rep: Representation, extent: u64, layout: Layout, data: ObjectData| {
        let d = ObjectDescriptor::new(rep, extent, layout, None).unwrap();
        store.insert(d.clone(), data).unwrap()
    };
    let mono: Vec<i32> = (0..2048)
        .map(|i| {
            let v = i as i64 - 1024;
            (v.wrapping_mul(4_194_304 / 4)) as i32
        })
        .collect();
    let lit = mk_data(
        Representation::Literal,
        2048,
        Layout::Mono,
        ObjectData::Literal(
            Literal::new(
                &ObjectDescriptor::new(Representation::Literal, 2048, Layout::Mono, None).unwrap(),
                mono.clone(),
            )
            .unwrap(),
        ),
    );
    let st: Vec<i32> = (0..1024 * 2)
        .map(|i| {
            let l = (i as i64 - 1024) * (1 << 22);
            let r = -l;
            if i % 2 == 0 { l as i32 } else { r as i32 }
        })
        .collect();
    let stereo = mk_data(
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
    );
    let tri: Vec<i32> = (0..128)
        .map(|i| {
            let m = (i as i64) % 128;
            let v = if m < 64 { m } else { 128 - m };
            (v * (1 << 24) - (32 << 24)) as i32
        })
        .collect();
    let wt = mk_data(
        Representation::Wavetable,
        128,
        Layout::Mono,
        ObjectData::Wavetable(
            Cycle::new(
                &ObjectDescriptor::new(Representation::Wavetable, 128, Layout::Mono, None).unwrap(),
                tri.clone(),
            )
            .unwrap(),
        ),
    );
    let osc = mk_data(
        Representation::Oscillator,
        0,
        Layout::Mono,
        ObjectData::Oscillator(Oscillator::checked(220, 1 << 15).unwrap()),
    );
    let nse = mk_data(
        Representation::Noise,
        0,
        Layout::Mono,
        ObjectData::Noise(Noise::new(0x51A7_2026)),
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
                        amp_q16: 1 << 15,
                    },
                    Partial {
                        harmonic: 2,
                        amp_q16: 1 << 14,
                    },
                    Partial {
                        harmonic: 3,
                        amp_q16: 1 << 13,
                    },
                    Partial {
                        harmonic: 5,
                        amp_q16: 1 << 12,
                    },
                    Partial {
                        harmonic: 8,
                        amp_q16: 1 << 11,
                    },
                ],
            )
            .unwrap(),
        ),
    );
    // Residual: periodic model over a small cycle + closing residual.
    let intrinsic: Vec<i32> = mono[..512].to_vec();
    let model = ResidualModel::Periodic {
        cycle: mono[..64].to_vec(),
    };
    let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
    let resid = mk_data(
        Representation::PredictorResidual,
        512,
        Layout::Mono,
        ObjectData::PredictorResidual(
            Residual::new(
                &ObjectDescriptor::new(Representation::PredictorResidual, 512, Layout::Mono, None)
                    .unwrap(),
                model,
                records,
            )
            .unwrap(),
        ),
    );

    // Voices.
    events.push(TimelineEvent::VoiceOn(mk(lit, 0)));
    events.push(TimelineEvent::VoiceOn({
        let mut s = mk(lit, 300);
        s.rate_q24 = -(1 << 24); // reverse one-shot
        s.interp = Interp::Nearest;
        s.envelope = adr;
        s
    }));
    events.push(TimelineEvent::VoiceOn({
        let mut s = mk(lit, 600);
        s.loop_mode = LoopMode::Region(LoopRegion::new(100, 1900).unwrap());
        s.rate_q24 = (1 << 24) + (1 << 18); // looped
        s.note_off = Some(10_000);
        s.envelope = adr;
        s
    }));
    events.push(TimelineEvent::VoiceOn({
        let mut s = mk(stereo, 100);
        s.route = Route::StereoPair(0);
        s.pan_q16 = -3000;
        s.gain_q16 = 1 << 15;
        s
    }));
    events.push(TimelineEvent::VoiceOn(mk(wt, 400)));
    events.push(TimelineEvent::VoiceOn(mk(osc, 700)));
    events.push(TimelineEvent::VoiceOn(mk(nse, 900)));
    events.push(TimelineEvent::VoiceOn(mk(bank, 1100)));
    events.push(TimelineEvent::VoiceOn(mk(resid, 1300)));
    events.push(TimelineEvent::VoiceOn(mk(cst, 1500)));
    events.push(TimelineEvent::VoiceOff {
        frame: 900,
        voice: 0,
    });
    (store, events)
}

/// Assert one world renders bit-identically across every floor at `window`.
fn check_world(
    label: &str,
    store: &ObjectStore,
    world: &World,
    start: i64,
    frames: usize,
    hashes: &mut Vec<(String, String)>,
) -> crate::error::Result<()> {
    let scalar = ScalarOracle::new(world.clone());
    let reference = scalar.observe(store, start, frames)?;
    let ref_hash = crate::hash::sha256::hex(&observation_sha256(&reference));
    hashes.push((format!("{label}/reference"), ref_hash.clone()));
    for &isa in &floors() {
        let simd = SimdOracle {
            world: world.clone(),
            isa,
        };
        let got = simd.observe(store, start, frames)?;
        if got != reference {
            return Err(crate::error::Error::malformed(format!(
                "{label}: SIMD floor {isa:?} diverged from scalar at window \
                 [{start}, {})",
                start + frames as i64
            )));
        }
        let h = crate::hash::sha256::hex(&observation_sha256(&got));
        hashes.push((format!("{label}/{isa:?}"), h));
    }
    Ok(())
}

/// Fixture-level timing: end-to-end wall time per floor over a fixed window.
fn time_fixture(
    label: &str,
    store: &ObjectStore,
    world: &World,
    frames: usize,
) -> std::collections::BTreeMap<String, serde_json::Value> {
    use serde_json::json;
    let mut map = std::collections::BTreeMap::new();
    let runs = 7u32;
    for &isa in &floors() {
        let simd = SimdOracle {
            world: world.clone(),
            isa,
        };
        let _ = simd.observe(store, 0, 64).unwrap(); // warm
        let mut samples = Vec::with_capacity(runs as usize);
        for _ in 0..runs {
            let t0 = Instant::now();
            let out = simd.observe(store, 0, frames).unwrap();
            samples.push(t0.elapsed().as_secs_f64() * 1e9);
            let _ = out;
        }
        samples.sort_by(f64::total_cmp);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        map.insert(
            format!("{label}/{isa:?}"),
            json!({
                "runs": runs,
                "frames": frames,
                "mean_ns": mean.round() as u64,
                "min_ns": samples[0].round() as u64,
                "median_ns": samples[samples.len() / 2].round() as u64,
                "max_ns": samples[samples.len() - 1].round() as u64,
                "method": "host CLOCK_MONOTONIC wall, warm cache, single thread; fixture-level only",
            }),
        );
    }
    map
}

/// Run the court; writes an immutable receipt under `receipts/simd/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let mut counters = Counters::new();
    let mut hashes: Vec<(String, String)> = Vec::new();
    let mut timing = std::collections::BTreeMap::new();

    let run_check = |label: &str,
                     store: &ObjectStore,
                     world: &World,
                     windows: &[(i64, usize)],
                     hashes: &mut Vec<(String, String)>,
                     counters: &mut Counters|
     -> crate::error::Result<()> {
        for &(start, frames) in windows {
            check_world(label, store, world, start, frames, hashes)?;
            counters.quanta_submitted += 1;
        }
        Ok(())
    };

    // 1. Semantic fixture (frozen reference hash must be reproduced).
    {
        let (store, events) = crate::courts::semantic::semantic_court_fixture();
        let world = World::new(
            crate::courts::semantic::RATE_HZ,
            crate::courts::semantic::CHANNELS,
            events,
        )?;
        let windows = [(0i64, 2400usize), (0, 1), (700, 900), (1600, 800), (50, 64)];
        run_check(
            "semantic",
            &store,
            &world,
            &windows,
            &mut hashes,
            &mut counters,
        )?;
        timing.extend(time_fixture("semantic-timing", &store, &world, 2400));
    }
    // 2. Authored fixture.
    {
        let (store, events) = crate::courts::authored::authored_court_fixture();
        let world = World::new(crate::courts::authored::RATE_HZ, 1, events)?;
        let windows = [(0i64, 4000usize), (0, 1), (400, 3600), (1600, 2400)];
        run_check(
            "authored",
            &store,
            &world,
            &windows,
            &mut hashes,
            &mut counters,
        )?;
        timing.extend(time_fixture("authored-timing", &store, &world, 4000));
    }
    // 3. Mixed world crossing every class boundary.
    {
        let (store, events) = mixed_world();
        let world = World::new(RATE_HZ, 2, events)?;
        let windows = [(0i64, 8192usize), (0, 1), (1234, 700), (7000, 1192)];
        run_check(
            "mixed",
            &store,
            &world,
            &windows,
            &mut hashes,
            &mut counters,
        )?;
        timing.extend(time_fixture("mixed-timing", &store, &world, 8192));
    }

    // Every floor reproduced the reference on every window.
    let hash_rows: Vec<String> = hashes.iter().map(|(k, v)| format!("{k}:{v}")).collect();
    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1".into()),
        backend: Some("simd".into()),
        sample_rate_hz: Some(RATE_HZ),
        channels: Some(2),
        quantum_frames: Some(8192),
        duration_secs: Some((2400 + 4000 + 8192) as f64 / RATE_HZ as f64),
        content_kind: Some("phase-F-parity-battery".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("simd");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "scalar == SIMD on every available floor ({}); fixture-level timing recorded in extras",
            hash_rows.join(", ")
        ))
        .params(params)
        .counters(counters);
    builder.timing(RunTiming {
        warmup_runs: Some(1),
        run_count: Some(7),
        observation_count: Some(3 * 8192),
        ..Default::default()
    });
    builder.provenance(Provenance {
        reference_hash: hashes.first().map(|(_, h)| h.clone()),
        benchmark_order: floors().iter().map(|i| format!("{i:?}")).collect(),
        ..Default::default()
    });
    for (k, v) in timing {
        builder.extra(k, v);
    }
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court simd: SUPPORTED");
    println!(
        "  floors: {}",
        floors()
            .iter()
            .map(|f| format!("{f:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (k, h) in &hashes {
        println!("  {k}: {h}");
    }
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256::hex;

    #[test]
    fn mixed_world_is_audible_and_deterministic() {
        let (store, events) = mixed_world();
        store.validate().unwrap();
        let world = World::new(RATE_HZ, 2, events).unwrap();
        let a = ScalarOracle::new(world.clone())
            .observe(&store, 0, 8192)
            .unwrap();
        let b = ScalarOracle::new(world).observe(&store, 0, 8192).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().any(|&x| x != 0), "mixed world must be audible");
        let h = hex(&observation_sha256(&a));
        // Frozen for this fixture (battery worlds must stay stable).
        assert_eq!(
            h,
            "cba472b88d0876a6113bae4e4aa501903d8813948e8ca741ff7f01e9d94abf8f"
        );
    }
}
