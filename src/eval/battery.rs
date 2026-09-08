//! Shared randomized differential-battery corpus/voice generators (Phase F/G).
//!
//! `corpus(seed)` builds a store with one object of every representation
//! class plus a reference; `random_voice` draws an adversarial voice spec
//! (random/adversarial rates incl. reverse/zero/fractional, envelopes with
//! zero-length segments and extreme sustains, pan/gain across the domain,
//! loop modes, both interpolators). Deterministic from the seed.
//!
//! The SIMD parity battery (eval::simd tests), the flat-evaluator battery
//! (backend::flatten tests), and the device battery (`court cuda`) all draw
//! from the same distribution, so every backend is exercised over identical
//! worlds. Host-only (`std`).

use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::{
    self, Constant, Cycle, Literal, Noise, ObjectData, ObjectId, ObjectStore, Oscillator,
};
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::pan::Route;
use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
use crate::universe::layout::Layout;
use crate::universe::prng::XoShiro256;

/// Build a corpus with one object of every representation class and one
/// reference. Deterministic from `seed`.
pub fn corpus(seed: u64) -> (ObjectStore, Vec<ObjectId>) {
    let mut store = ObjectStore::new();
    let mut rng = XoShiro256::from_seed(seed ^ 0x5EED);
    let mut ids: Vec<ObjectId> = Vec::new();
    let next32 = |rng: &mut XoShiro256| rng.next_u64() as u32;
    let insert = |store: &mut ObjectStore,
                  rep: Representation,
                  extent: u64,
                  layout: Layout,
                  loop_region: Option<object::LoopRegion>,
                  data: ObjectData|
     -> ObjectId {
        let d = ObjectDescriptor::new(rep, extent, layout, loop_region).unwrap();
        store.insert(d.clone(), data).unwrap()
    };

    // Literal mono one-shot.
    let ext = 1 + (next32(&mut rng) % 4096) as u64;
    let samples: Vec<i32> = (0..ext).map(|_| next32(&mut rng) as i32).collect();
    let d = ObjectDescriptor::new(Representation::Literal, ext, Layout::Mono, None).unwrap();
    let id = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, samples).unwrap()),
        )
        .unwrap();
    ids.push(id);
    // Literal stereo one-shot.
    let ext = 1 + (next32(&mut rng) % 1024) as u64;
    let samples: Vec<i32> = (0..ext * 2).map(|_| next32(&mut rng) as i32).collect();
    let d = ObjectDescriptor::new(Representation::Literal, ext, Layout::Stereo, None).unwrap();
    let id = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, samples).unwrap()),
        )
        .unwrap();
    ids.push(id);
    // Literal mono with a declared loop region (used by LoopMode::Default).
    let ext = 64 + (next32(&mut rng) % 4096) as u64;
    let a = u64::from(next32(&mut rng)) % ext;
    let b = a + 1 + (u64::from(next32(&mut rng)) % (ext - a));
    let lr = object::LoopRegion::new(a, b).unwrap();
    let samples: Vec<i32> = (0..ext).map(|_| next32(&mut rng) as i32).collect();
    let d = ObjectDescriptor::new(Representation::Literal, ext, Layout::Mono, Some(lr)).unwrap();
    let id = store
        .insert(
            d.clone(),
            ObjectData::Literal(Literal::new(&d, samples).unwrap()),
        )
        .unwrap();
    ids.push(id);
    // Wavetable / single-cycle / exact-repeat (cycle classes, mono).
    let ext = 8 + (next32(&mut rng) % 2040) as u64;
    let cyc: Vec<i32> = (0..ext).map(|_| next32(&mut rng) as i32).collect();
    for rep in [
        Representation::Wavetable,
        Representation::SingleCycle,
        Representation::ExactRepeat,
    ] {
        let d = ObjectDescriptor::new(rep, ext, Layout::Mono, None).unwrap();
        let id = store
            .insert(
                d.clone(),
                ObjectData::Wavetable(Cycle::new(&d, cyc.clone()).unwrap()),
            )
            .unwrap();
        ids.push(id);
    }
    // Endless classes.
    let od = ObjectDescriptor::new(Representation::Oscillator, 0, Layout::Mono, None).unwrap();
    ids.push(
        store
            .insert(
                od.clone(),
                ObjectData::Oscillator(
                    Oscillator::checked(1 + next32(&mut rng) % 24_000, (1 << 16) / 2).unwrap(),
                ),
            )
            .unwrap(),
    );
    let nd = ObjectDescriptor::new(Representation::Noise, 0, Layout::Mono, None).unwrap();
    ids.push(
        store
            .insert(
                nd.clone(),
                ObjectData::Noise(Noise::new(next32(&mut rng).into())),
            )
            .unwrap(),
    );
    let cd = ObjectDescriptor::new(Representation::Constant, 0, Layout::Mono, None).unwrap();
    ids.push(
        store
            .insert(
                cd.clone(),
                ObjectData::Constant(Constant::new((next32(&mut rng) as i32) >> 1)),
            )
            .unwrap(),
    );
    // Partial bank (mono).
    let bd = ObjectDescriptor::new(Representation::PartialBank, 0, Layout::Mono, None).unwrap();
    let n_partials = 1 + next32(&mut rng) % 6;
    let mut partials = Vec::new();
    let mut h = 1 + next32(&mut rng) % 8;
    for _ in 0..n_partials {
        let amp = ((next32(&mut rng) as i32) % (1 << 16)) - (1 << 15);
        partials.push(crate::sampler::procedural::Partial {
            harmonic: h,
            amp_q16: amp.clamp(-(1 << 16), 1 << 16),
        });
        h += 1 + next32(&mut rng) % 8;
    }
    ids.push(
        store
            .insert(
                bd.clone(),
                ObjectData::PartialBank(
                    crate::object::PartialBank::checked(1 + next32(&mut rng) % 2000, partials)
                        .unwrap(),
                ),
            )
            .unwrap(),
    );
    // Residual-governed mono: periodic model + closing residual over a
    // random intrinsic.
    let ext = 128 + (next32(&mut rng) % 512) as u64;
    let intrinsic: Vec<i32> = (0..ext).map(|_| next32(&mut rng) as i32).collect();
    let cycle_len = 1 + (next32(&mut rng) % 64) as usize;
    let cycle: Vec<i32> = intrinsic[..cycle_len].to_vec();
    let periodic = object::residual::ResidualModel::Periodic { cycle };
    // Full-scale random deltas can exceed the i32 code domain under the
    // periodic model; fall back to the Zero model (deltas == intrinsic,
    // always representable) so the residual class stays covered.
    let (model, records) =
        match object::residual::Residual::closing_residual(&intrinsic, 1, &periodic) {
            Some(r) => (periodic, r),
            None => {
                let z = object::residual::ResidualModel::Zero;
                let r = object::residual::Residual::closing_residual(&intrinsic, 1, &z)
                    .expect("zero model always closes");
                (z, r)
            }
        };
    let rd =
        ObjectDescriptor::new(Representation::PredictorResidual, ext, Layout::Mono, None).unwrap();
    let residual = object::residual::Residual::new(&rd, model, records).unwrap();
    ids.push(
        store
            .insert(rd.clone(), ObjectData::PredictorResidual(residual))
            .unwrap(),
    );
    // Reference to the first literal at a random transpose.
    let target = store.get(ids[0]).unwrap().content_id;
    let tr = (next32(&mut rng) % (1 << 25)) + 1; // [1, 2) transpose
    let rid = insert(
        &mut store,
        Representation::Referenced,
        ext,
        Layout::Mono,
        None,
        ObjectData::Referenced(
            object::reference::Referenced::checked(target, tr as i64, None).unwrap(),
        ),
    );
    ids.push(rid);
    store.validate().unwrap();
    (store, ids)
}

/// Random voice spec over `objects`, deterministic from `rng`.
pub fn random_voice(
    object: ObjectId,
    store: &ObjectStore,
    rng: &mut XoShiro256,
    out_channels: u8,
) -> VoiceSpec {
    let next32 = |rng: &mut XoShiro256| rng.next_u64() as u32;
    let obj = store.get(object).unwrap();
    let layout = obj.descriptor.layout;
    let extent = obj.descriptor.extent_frames;
    let endless = obj.data.is_endless();
    let cycle = obj.data.is_periodic();
    // Envelope: adversarial random parameters (incl. zero-length segments
    // and full/zero sustain).
    let attack = next32(rng) % 300;
    let decay = next32(rng) % 600;
    let sustain = ((next32(rng) as i64) % (1 << 17)) as i32;
    let sustain = sustain.clamp(0, 1 << 16);
    let release = next32(rng) % 300;
    let envelope = EnvelopeParams::new(attack, decay, sustain, release).unwrap();
    // Rate: wide distribution incl. reverse, zero, fractional, and
    // near-domain extremes.
    let rate = match next32(rng) % 8 {
        0 => 0,
        1 => 1 << 24,
        2 => -(1 << 24),
        3 => (1 << 24) / 2,
        4 => -((1 << 24) / 3),
        5 => {
            let e = 8 + next32(rng) % 33;
            let mag = 1i64 << e;
            if next32(rng) & 1 == 0 { mag } else { -mag }
        }
        _ => ((next32(rng) as i64) % (1 << 25)) - (1 << 24),
    };
    let rate = rate.clamp(-(1i64 << 40), 1 << 40);
    let rate = if rate != 0 && rate.unsigned_abs() < (1 << 8) {
        1 << 8
    } else {
        rate
    };
    // Start position (content only).
    let start_pos_q24 = if !endless && !cycle && extent != 0 {
        (((next32(rng) as u64) % extent) << 24) as i64
    } else {
        0
    };
    // Routing: mono on any layout; stereo pair only for stereo objects.
    let (_object_channel, route) = if layout == Layout::Stereo && next32(rng) % 3 == 0 {
        let base = (next32(rng) % u32::from(out_channels).max(1)) as u8;
        if out_channels >= 2 && base < out_channels - 1 {
            (0, Route::StereoPair(base))
        } else {
            (0, Route::Mono(0))
        }
    } else {
        (
            0,
            Route::Mono((next32(rng) % u32::from(out_channels)) as u8),
        )
    };
    let gain_q16 = ((next32(rng) as i32) % (1 << 18)) - (1 << 17);
    let pan_q16 = ((next32(rng) as i64) % ((1 << 17) + 1)) as i32 - (1 << 16);
    // Loop mode: literal content may loop; endless/cycle classes reject
    // voice loop regions.
    let loop_mode = if endless || cycle {
        LoopMode::Off
    } else {
        match next32(rng) % 3 {
            0 => LoopMode::Off,
            1 => LoopMode::Default,
            _ => {
                // Random region strictly inside the extent (validated by
                // resolve; invalid draws make the world unrenderable and
                // are skipped by the battery).
                let ext = extent.max(2);
                let a = u64::from(next32(rng)) % (ext - 1);
                let b = a + 1 + ((next32(rng) as u64) % (ext - a - 1));
                LoopMode::Region(object::LoopRegion::new(a, b).unwrap())
            }
        }
    };
    let interp = if next32(rng) & 1 == 0 {
        Interp::Nearest
    } else {
        Interp::Linear
    };
    VoiceSpec {
        object,
        trigger_frame: (next32(rng) % 4000) as i64,
        note_off: None,
        start_pos_q24,
        rate_q24: rate,
        object_channel: 0,
        route,
        gain_q16: gain_q16.clamp(-(1 << 17), 1 << 17),
        pan_q16,
        envelope,
        loop_mode,
        interp,
    }
}
