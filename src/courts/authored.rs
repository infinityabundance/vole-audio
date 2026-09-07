//! `court authored` — procedural SampleObject battery.
//!
//! Builds one of every procedural class (silence, constant, wavetable,
//! single-cycle, exact-repeat, oscillator, partial bank, deterministic noise)
//! plus a reference to a wavetable, and verifies through the scalar oracle:
//!
//! 1. deterministic repeated observation (hash equality);
//! 2. chunked == contiguous;
//! 3. zero resident sample-domain bytes for endless classes (the "no
//!    full-object PCM" evidence: procedural observation is generated, not
//!    stored);
//! 4. hostile spec rejection (loop region on an oscillator, nonzero start on
//!    endless sources) returns errors, not panics.
//!
//! No GPU or audio hardware is needed.

use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::{Constant, Cycle, LoopRegion, Noise, ObjectData, ObjectId, ObjectStore};
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

const RATE_HZ: u32 = 48_000;

/// Frozen reference vector for the authored fixture observation hash.
pub const AUTHORED_COURT_REFERENCE_SHA256: &str =
    "f91b5b4228022d46a609a2fe7fc862e6e72c4058405c4b72316b202feb14fe87";

fn build_fixture() -> (ObjectStore, Vec<TimelineEvent>) {
    let mut store = ObjectStore::new();
    let mut events: Vec<TimelineEvent> = Vec::new();
    let instant = EnvelopeParams::new(0, 0, crate::sampler::envelope::ENV_UNITY, 0).unwrap();
    let mk = |object: ObjectId, trigger: i64, gain: i32| VoiceSpec {
        object,
        trigger_frame: trigger,
        note_off: None,
        start_pos_q24: 0,
        rate_q24: 1 << 24,
        object_channel: 0,
        route: Route::Mono(0),
        gain_q16: gain,
        pan_q16: 0,
        envelope: instant,
        loop_mode: LoopMode::Off,
        interp: Interp::Linear,
    };

    let e0 = ObjectDescriptor::new(Representation::Silence, 0, Layout::Mono, None).unwrap();
    let sil = store.insert(e0.clone(), ObjectData::Silence).unwrap();

    let c0 = ObjectDescriptor::new(Representation::Constant, 0, Layout::Mono, None).unwrap();
    let cst = store
        .insert(c0.clone(), ObjectData::Constant(Constant::new(65_536)))
        .unwrap();

    let n0 = ObjectDescriptor::new(Representation::Noise, 0, Layout::Mono, None).unwrap();
    let nse = store
        .insert(n0.clone(), ObjectData::Noise(Noise::new(0x5EED_2026)))
        .unwrap();

    let o0 = ObjectDescriptor::new(Representation::Oscillator, 0, Layout::Mono, None).unwrap();
    let osc = store
        .insert(
            o0.clone(),
            ObjectData::Oscillator(Oscillator::checked(220, 1 << 16).unwrap()),
        )
        .unwrap();

    let b0 = ObjectDescriptor::new(Representation::PartialBank, 0, Layout::Mono, None).unwrap();
    let bank = store
        .insert(
            b0.clone(),
            ObjectData::PartialBank(
                PartialBank::checked(
                    55,
                    vec![
                        Partial {
                            harmonic: 1,
                            amp_q16: 1 << 16,
                        },
                        Partial {
                            harmonic: 2,
                            amp_q16: 1 << 15,
                        },
                        Partial {
                            harmonic: 3,
                            amp_q16: 1 << 14,
                        },
                        Partial {
                            harmonic: 4,
                            amp_q16: 1 << 13,
                        },
                        Partial {
                            harmonic: 5,
                            amp_q16: 1 << 12,
                        },
                        Partial {
                            harmonic: 6,
                            amp_q16: 1 << 11,
                        },
                        Partial {
                            harmonic: 7,
                            amp_q16: 1 << 10,
                        },
                        Partial {
                            harmonic: 8,
                            amp_q16: 1 << 9,
                        },
                    ],
                )
                .unwrap(),
            ),
        )
        .unwrap();

    // A 96-frame triangle-ish cycle.
    let tri: Vec<i32> = (0..96)
        .map(|i| {
            let m = (i as i64) % 96;
            let v = if m < 48 { m } else { 96 - m };
            (v * (1 << 24) - (24 << 24)) as i32
        })
        .collect();

    let w0 = ObjectDescriptor::new(Representation::Wavetable, 96, Layout::Mono, None).unwrap();
    let wt = store
        .insert(
            w0.clone(),
            ObjectData::Wavetable(Cycle::new(&w0, tri.clone()).unwrap()),
        )
        .unwrap();
    let s0 = ObjectDescriptor::new(Representation::SingleCycle, 96, Layout::Mono, None).unwrap();
    let sc = store
        .insert(
            s0.clone(),
            ObjectData::SingleCycle(Cycle::new(&s0, tri.clone()).unwrap()),
        )
        .unwrap();
    let r0 = ObjectDescriptor::new(Representation::ExactRepeat, 96, Layout::Mono, None).unwrap();
    let er = store
        .insert(
            r0.clone(),
            ObjectData::ExactRepeat(Cycle::new(&r0, tri.clone()).unwrap()),
        )
        .unwrap();

    // Reference to the wavetable at half transpose.
    let rd = ObjectDescriptor::new(Representation::Referenced, 96, Layout::Mono, None).unwrap();
    let rf = store
        .insert(
            rd.clone(),
            ObjectData::Referenced(
                crate::object::reference::Referenced::checked(
                    store.get(wt).unwrap().content_id,
                    1 << 23,
                    None,
                )
                .unwrap(),
            ),
        )
        .unwrap();

    let half = 1 << 15;
    let third = (1 << 16) / 3;
    for (i, id) in [sil, cst, nse, osc, bank].iter().enumerate() {
        events.push(TimelineEvent::VoiceOn(mk(*id, (i as i64) * 97, half)));
    }
    for (i, id) in [wt, sc, er, rf].iter().enumerate() {
        events.push(TimelineEvent::VoiceOn(mk(
            *id,
            400 + (i as i64) * 400,
            third,
        )));
    }
    (store, events)
}

/// Run the court; writes an immutable receipt under `receipts/authored/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let (store, events) = build_fixture();
    store.validate()?;
    let world = World::new(RATE_HZ, 1, events)?;
    let oracle = crate::eval::ScalarOracle::new(world);

    // 1. Deterministic repeated observation.
    let total = 4000usize;
    let a = oracle.observe(&store, 0, total)?;
    let b = oracle.observe(&store, 0, total)?;
    if a != b {
        return fail(receipts_root, "repeated observation differs");
    }
    let hash_a = observation_sha256(&a);

    // 2. Chunked == contiguous.
    let mut joined = Vec::new();
    for (s, l) in [(0usize, 1200usize), (1200, 1500), (2700, 1300)] {
        joined.extend_from_slice(&oracle.observe(&store, s as i64, l)?);
    }
    if joined != a {
        return fail(receipts_root, "chunked observation != contiguous");
    }

    // 3. Zero resident sample bytes for the endless classes.
    for obj in store.iter() {
        if obj.data.is_endless() && obj.data.resident_sample_bytes() != 0 {
            return fail(receipts_root, "endless object holds resident sample bytes");
        }
    }

    // 4. Hostile specs are rejected cleanly (resolution happens at observe
    //    time, so rejection is observed as an `Err` from `observe`).
    let spec_of = |object| VoiceSpec {
        object,
        trigger_frame: 0,
        note_off: None,
        start_pos_q24: 0,
        rate_q24: 1 << 24,
        object_channel: 0,
        route: Route::Mono(0),
        gain_q16: 1 << 15,
        pan_q16: 0,
        envelope: EnvelopeParams::new(0, 0, crate::sampler::envelope::ENV_UNITY, 0).unwrap(),
        loop_mode: LoopMode::Region(LoopRegion::new(1, 2).unwrap()),
        interp: Interp::Linear,
    };
    // Loop region on an oscillator is invalid.
    let osc_id = store
        .iter()
        .find(|o| matches!(o.data, ObjectData::Oscillator(_)))
        .unwrap()
        .id;
    let w = World::new(RATE_HZ, 1, vec![TimelineEvent::VoiceOn(spec_of(osc_id))]).unwrap();
    if oracle_observe_ok(&w, &store) {
        return fail(receipts_root, "oscillator accepted a loop region");
    }
    // Nonzero start on a wavetable-cycle voice is invalid.
    let wt_id = store
        .iter()
        .find(|o| matches!(o.data, ObjectData::Wavetable(_)))
        .unwrap()
        .id;
    let mut s = spec_of(wt_id);
    s.loop_mode = LoopMode::Off;
    s.start_pos_q24 = 3 << 24;
    let w2 = World::new(RATE_HZ, 1, vec![TimelineEvent::VoiceOn(s)]).unwrap();
    if oracle_observe_ok(&w2, &store) {
        return fail(receipts_root, "cycle voice accepted a nonzero start");
    }

    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1".into()),
        backend: Some("scalar".into()),
        sample_rate_hz: Some(RATE_HZ),
        channels: Some(1),
        quantum_frames: Some(total as u32),
        duration_secs: Some(total as f64 / RATE_HZ as f64),
        content_kind: Some("procedural-battery".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("authored");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "procedural battery passed; reference sha256 {}",
            crate::hash::sha256::hex(&hash_a)
        ))
        .params(params);
    builder.provenance(crate::evidence::receipt::Provenance {
        reference_hash: Some(crate::hash::sha256::hex(&hash_a)),
        ..Default::default()
    });
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court authored: SUPPORTED");
    println!("  reference sha256: {}", crate::hash::sha256::hex(&hash_a));
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}

/// True if a 1-frame observation succeeds (resolution errors surface here).
fn oracle_observe_ok(world: &World, store: &ObjectStore) -> bool {
    world.observe(store, 0, 1).is_ok()
}

/// Honest failure receipt (negative results are results).
fn fail(receipts_root: &Path, why: &str) -> crate::error::Result<Verdict> {
    let mut builder = ReceiptBuilder::new("authored");
    builder
        .result(Verdict::FailedCorrectness)
        .result_detail(format!("authored battery failed: {why}"));
    let (_, path) = builder.finish_write(receipts_root)?;
    eprintln!("court authored: FAILED_CORRECTNESS ({why})");
    eprintln!("  receipt: {}", path.display());
    Ok(Verdict::FailedCorrectness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256::hex;

    #[test]
    fn fixture_reference_hash_is_frozen() {
        let (store, events) = build_fixture();
        let world = World::new(RATE_HZ, 1, events).unwrap();
        let oracle = crate::eval::ScalarOracle::new(world);
        let out = oracle.observe(&store, 0, 4000).unwrap();
        let h = hex(&observation_sha256(&out));
        assert_eq!(h, AUTHORED_COURT_REFERENCE_SHA256);
    }

    #[test]
    fn fixture_store_validates() {
        let (store, _) = build_fixture();
        store.validate().expect("authored fixture store valid");
    }
}
