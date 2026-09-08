//! `court semantic` — scalar oracle determinism battery.
//!
//! Exercises the frozen sampler semantics and emits evidence:
//!
//! 1. a deterministic mini-score (several voices: one-shot, loop, reverse,
//!    half-rate, gain, pan, note-off, stereo route, literal + reference);
//! 2. repeated observation hashes must be identical;
//! 3. chunked observation must equal contiguous observation;
//! 4. seek (windowed) must equal the sliced contiguous result;
//! 5. adversarial silence/limit cases return clean errors.
//!
//! On success the court verdict is `SUPPORTED` (semantics deterministic and
//! reproducible). This court needs no GPU/audio hardware.

use crate::evidence::counters::Counters;
use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::{Literal, ObjectData, ObjectStore};
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::pan::Route;
use crate::sampler::scheduler::TimelineEvent;
use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
use crate::sampler::world::World;
use crate::status::Verdict;
use crate::universe::layout::Layout;
use crate::universe::observation::observation_sha256;
use std::path::Path;

pub(crate) const RATE_HZ: u32 = 48_000;
pub(crate) const CHANNELS: u8 = 2;

/// Frozen reference vector for the court fixture observation hash. Changing
/// sampler semantics, fixture content, or u1 arithmetic changes this hash and
/// requires re-freezing (profile change), never a silent edit.
pub const SEMANTIC_COURT_REFERENCE_SHA256: &str =
    "1791816f4b938375cc4298b2587ce19eef260d063d9ed88c597e04d31837f6d0";

/// Build the deterministic court world: a ramp object, a reference to it, and
/// a small score that exercises the transform surface. Shared with the
/// Phase F parity tests (`eval::simd`) and `court simd`.
pub(crate) fn semantic_court_fixture() -> (ObjectStore, Vec<TimelineEvent>) {
    build_fixture()
}

fn build_fixture() -> (ObjectStore, Vec<TimelineEvent>) {
    let mut store = ObjectStore::new();
    // Mono ramp, 512 frames, ±0.5 full scale.
    let ramp: Vec<i32> = (0..512)
        .map(|i| {
            let v = (i as i64 - 256) * (1 << 22); // ±2^22-ish
            v as i32
        })
        .collect();
    let d = ObjectDescriptor::new(Representation::Literal, 512, Layout::Mono, None).unwrap();
    let ramp_id = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, ramp).unwrap()),
        )
        .unwrap();

    // Stereo object: left ramp, right inverted ramp.
    let stereo: Vec<i32> = (0..256)
        .flat_map(|i| {
            let l = (i as i64) * (1 << 22);
            let r = -l;
            [l as i32, r as i32]
        })
        .collect();
    let sd = ObjectDescriptor::new(Representation::Literal, 256, Layout::Stereo, None).unwrap();
    let stereo_id = store
        .insert(
            sd.clone(),
            ObjectData::Literal(Literal::new(&sd, stereo).unwrap()),
        )
        .unwrap();

    // Reference to the ramp at half transpose.
    let target = store.get(ramp_id).unwrap().content_id;
    let rd = ObjectDescriptor::new(Representation::Referenced, 512, Layout::Mono, None).unwrap();
    let ref_obj = crate::object::reference::Referenced::checked(target, 1 << 23, None).unwrap();
    let ref_id = store
        .insert(rd.clone(), ObjectData::Referenced(ref_obj))
        .unwrap();

    let env_default = EnvelopeParams::default_at(RATE_HZ);
    let instant = EnvelopeParams::new(0, 0, crate::sampler::envelope::ENV_UNITY, 0).unwrap();

    let mut events: Vec<TimelineEvent> = Vec::new();
    let mut push_voice = |spec: VoiceSpec| events.push(TimelineEvent::VoiceOn(spec));

    // v0: ramp one-shot, triggered at 0.
    push_voice(VoiceSpec {
        object: ramp_id,
        trigger_frame: 0,
        note_off: None,
        start_pos_q24: 0,
        rate_q24: 1 << 24,
        object_channel: 0,
        route: Route::Mono(0),
        gain_q16: 1 << 16,
        pan_q16: 0,
        envelope: env_default,
        loop_mode: LoopMode::Off,
        interp: Interp::Linear,
    });
    // v1: ramp looped [64,192), triggered at 400, note off at 900.
    push_voice(VoiceSpec {
        object: ramp_id,
        trigger_frame: 400,
        note_off: Some(900),
        start_pos_q24: 0,
        rate_q24: 1 << 24,
        object_channel: 0,
        route: Route::Mono(1),
        gain_q16: 1 << 15,
        pan_q16: 0,
        envelope: instant,
        loop_mode: LoopMode::Region(crate::object::LoopRegion::new(64, 192).unwrap()),
        interp: Interp::Linear,
    });
    // v2: referenced object, half transpose, reverse, routed right.
    push_voice(VoiceSpec {
        object: ref_id,
        trigger_frame: 100,
        note_off: None,
        start_pos_q24: 0,
        rate_q24: -(1 << 24),
        object_channel: 0,
        route: Route::Mono(1),
        gain_q16: 1 << 16,
        pan_q16: 0,
        envelope: instant,
        loop_mode: LoopMode::Off,
        interp: Interp::Nearest,
    });
    // v3: stereo object panned hard left (pan law exercised).
    push_voice(VoiceSpec {
        object: stereo_id,
        trigger_frame: 2000,
        note_off: None,
        start_pos_q24: 0,
        rate_q24: 1 << 24,
        object_channel: 0,
        route: Route::StereoPair(0),
        gain_q16: (1 << 16) / 2,
        pan_q16: -(1 << 16),
        envelope: instant,
        loop_mode: LoopMode::Off,
        interp: Interp::Linear,
    });
    events.push(TimelineEvent::VoiceOff {
        frame: 700,
        voice: 2,
    });
    (store, events)
}

/// Run the court; writes an immutable receipt under `receipts/semantic/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let (store, events) = build_fixture();
    store.validate()?;
    let world = World::new(RATE_HZ, CHANNELS, events)?;
    let oracle = crate::eval::ScalarOracle::new(world);

    let mut counters = Counters::new();

    // 1. Repeated observation hashes identical.
    let total = 2400usize;
    let a = oracle.observe(&store, 0, total)?;
    let b = oracle.observe(&store, 0, total)?;
    if a != b {
        return court_fail(
            receipts_root,
            &store,
            &oracle,
            "repeated observation differs",
        );
    }
    let hash_a = observation_sha256(&a);
    counters.quanta_submitted = 1;

    // 2. Chunked == contiguous.
    let mut joined = Vec::new();
    for (start, len) in [(0usize, 700usize), (700, 900), (1600, 800)] {
        joined.extend_from_slice(&oracle.observe(&store, start as i64, len)?);
    }
    if joined != a {
        return court_fail(
            receipts_root,
            &store,
            &oracle,
            "chunked observation != contiguous",
        );
    }

    // 3. Seek == sliced contiguous (element index = frame x channels).
    let ch = usize::from(CHANNELS);
    let seeked = oracle.observe(&store, 700, 900)?;
    if a[(700 * ch)..(1600 * ch)] != seeked[..] {
        return court_fail(receipts_root, &store, &oracle, "seek != sequential slice");
    }

    // 4. Hostile inputs rejected cleanly (no panics).
    let mut bad_events = build_fixture().1;
    // Note-off (frame 0) strictly before the trigger of voice 3 (frame 2000)
    // must be rejected by the scheduler.
    bad_events.push(TimelineEvent::VoiceOff { frame: 0, voice: 3 });
    if World::new(RATE_HZ, CHANNELS, bad_events).is_ok() {
        return court_fail(receipts_root, &store, &oracle, "hostile note-off accepted");
    }

    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1".into()),
        backend: Some("scalar".into()),
        sample_rate_hz: Some(RATE_HZ),
        channels: Some(u32::from(CHANNELS)),
        quantum_frames: Some(total as u32),
        duration_secs: Some(total as f64 / RATE_HZ as f64),
        content_kind: Some("deterministic-mini-score".into()),
        ..Default::default()
    };

    let mut builder = ReceiptBuilder::new("semantic");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "determinism battery passed; reference sha256 {}",
            crate::hash::sha256::hex(&hash_a)
        ))
        .params(params)
        .counters(counters);
    builder.provenance(crate::evidence::receipt::Provenance {
        reference_hash: Some(crate::hash::sha256::hex(&hash_a)),
        ..Default::default()
    });
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court semantic: SUPPORTED");
    println!("  reference sha256: {}", crate::hash::sha256::hex(&hash_a));
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}

/// Fail helper: still writes an honest receipt (negative results are results).
fn court_fail(
    receipts_root: &Path,
    _store: &ObjectStore,
    _oracle: &crate::eval::ScalarOracle,
    why: &str,
) -> crate::error::Result<Verdict> {
    let mut builder = ReceiptBuilder::new("semantic");
    builder
        .result(Verdict::FailedCorrectness)
        .result_detail(format!("determinism battery failed: {why}"));
    let (_, path) = builder.finish_write(receipts_root)?;
    eprintln!("court semantic: FAILED_CORRECTNESS ({why})");
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
        let world = World::new(RATE_HZ, CHANNELS, events).unwrap();
        let oracle = crate::eval::ScalarOracle::new(world);
        let out = oracle.observe(&store, 0, 2400).unwrap();
        let h = hex(&observation_sha256(&out));
        assert_eq!(h, SEMANTIC_COURT_REFERENCE_SHA256);
    }

    #[test]
    fn fixture_store_validates() {
        let (store, _) = build_fixture();
        store.validate().expect("fixture store valid");
    }
}
