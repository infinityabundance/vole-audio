//! GPU world flattening (Phase G) — host side, backend-neutral.
//!
//! `flatten(store, world)` converts a validated world into the flat
//! plain-data state the device kernels consume (`device::kernel_shared`):
//! a `Vec<FlatVoice>` and two arenas (interleaved i32 content, `Partial`
//! records). All reference chains are resolved and all transpose composition
//! happens here (voices arrive with `eff_rate_q24` and a concrete target),
//! exactly as `World::resolve_all` produces for the host engines.
//!
//! Capability matrix (frozen for the first CUDA kernel; mirrors the Phase F
//! lesson that classes vectorize at different costs):
//!
//! ```text
//! Silence/Constant/Noise/Oscillator/PartialBank  CUDA_NATIVE
//! Literal/Wavetable/SingleCycle/ExactRepeat      CUDA_NATIVE
//! PredictorResidual                              CUDA_FALLBACK (closure is
//!                                                materialized host-side into
//!                                                the arena; bytes counted)
//! Referenced                                     resolved before upload
//! ```
//!
//! Exactness contract: for every (world, window),
//! `FlattenedWorld::render == World::observe` bit for bit (enforced by the
//! tests here and by `court cuda` on the device).

use crate::device::kernel_shared::{
    F_INTERP_NEAREST, F_LOOP_REGION, F_ROUTE_STEREO, FlatEnvelope, FlatState, FlatVoice,
    NO_NOTE_OFF, VoiceClass, flat_state, render_window,
};
use crate::error::{Error, Result};
use crate::limits::MAX_ACTIVE_VOICES;
use crate::object::{ObjectData, ObjectId, ObjectStore};
use crate::sampler::pan::Route;
use crate::sampler::procedural::Partial;
use crate::sampler::voice::{Interp, ResolvedVoice};
use crate::sampler::world::World;
use std::collections::{HashMap, HashSet};

/// Ceiling for the flattened content arena (host memory + upload size).
/// Generous for diagnostics; real corpora are far smaller. Hostile objects
/// beyond this are rejected at flatten time, never silently truncated.
pub const MAX_FLAT_ARENA_BYTES: u64 = 1 << 30;

/// One flattened world (host view of the GPU-resident state).
#[derive(Debug, Clone)]
pub struct FlattenedWorld {
    pub voices: Vec<FlatVoice>,
    /// Interleaved content arena (i32 codes).
    pub samples: Vec<i32>,
    /// Partial-record arena.
    pub partials: Vec<Partial>,
    pub output_channels: u8,
    /// Per-class voice counts (index = `VoiceClass` code 0..=6).
    pub class_counts: [u32; 7],
    // --- exposure accounting (paper §41, honest D0 numbers) ---
    /// Bytes of residual closure materialized host-side (CUDA_FALLBACK).
    pub fallback_closure_bytes: u64,
    /// Total bytes uploaded for this world (voices + arenas).
    pub upload_bytes: u64,
}

impl FlattenedWorld {
    /// Flat state for one observation window.
    pub fn state(&self, start_frame: i64, frames: usize) -> FlatState {
        flat_state(
            start_frame,
            frames as u32,
            self.output_channels,
            self.voices.len(),
            self.partials.len(),
            self.samples.len(),
        )
    }

    /// Host-side sequential render over the exact device math.
    pub fn render(&self, start_frame: i64, frames: usize) -> Vec<i32> {
        let state = self.state(start_frame, frames);
        render_window(&state, &self.voices, &self.samples, &self.partials)
    }
}

/// Build the flat state for one (store, world).
pub fn flatten(store: &ObjectStore, world: &World) -> Result<FlattenedWorld> {
    let resolved = world.resolve_all(store)?;
    if resolved.len() as u64 > u64::from(MAX_ACTIVE_VOICES) {
        return Err(Error::limit("world exceeds MAX_ACTIVE_VOICES"));
    }
    let mut out = FlattenedWorld {
        voices: Vec::with_capacity(resolved.len()),
        samples: Vec::new(),
        partials: Vec::new(),
        output_channels: world.output_channels,
        class_counts: [0; 7],
        fallback_closure_bytes: 0,
        upload_bytes: 0,
    };

    // Content segment dedupe by resolved target id (content is unique per id
    // in the store; references already collapsed).
    let mut segment_of: HashMap<ObjectId, u64> = HashMap::new();
    let mut partial_seg_of: HashMap<ObjectId, (u32, u32)> = HashMap::new();
    // Residual objects whose closure was materialized (fallback accounting).
    let mut fallback_targets: HashSet<ObjectId> = HashSet::new();

    for rv in &resolved {
        let obj = store.get(rv.target_id)?;
        let channels = obj.descriptor.layout.count();
        let class = match &obj.data {
            ObjectData::Silence => VoiceClass::Silence,
            ObjectData::Constant(c) => {
                let mut v = base_record(rv, VoiceClass::Constant, channels);
                v.level = c.level;
                push_voice(&mut out, v, VoiceClass::Constant);
                continue;
            }
            ObjectData::Noise(n) => {
                let mut v = base_record(rv, VoiceClass::Noise, channels);
                v.noise_seed = n.seed;
                push_voice(&mut out, v, VoiceClass::Noise);
                continue;
            }
            ObjectData::Oscillator(o) => {
                let mut v = base_record(rv, VoiceClass::Oscillator, channels);
                v.osc_incr = rv.osc_incr;
                v.amp_q16 = o.amp_q16;
                push_voice(&mut out, v, VoiceClass::Oscillator);
                continue;
            }
            ObjectData::PartialBank(b) => {
                let (off, len) = match partial_seg_of.get(&rv.target_id) {
                    Some(&seg) => seg,
                    None => {
                        let off = out.partials.len() as u32;
                        out.partials.extend_from_slice(&b.partials);
                        let seg = (off, b.partials.len() as u32);
                        partial_seg_of.insert(rv.target_id, seg);
                        seg
                    }
                };
                let mut v = base_record(rv, VoiceClass::PartialBank, channels);
                v.osc_incr = rv.bank_incr;
                v.partials_offset = off;
                v.partials_len = len;
                push_voice(&mut out, v, VoiceClass::PartialBank);
                continue;
            }
            ObjectData::Wavetable(c) | ObjectData::SingleCycle(c) | ObjectData::ExactRepeat(c) => {
                let off =
                    arena_segment(&mut out.samples, &mut segment_of, rv.target_id, &c.samples)?;
                let mut v = base_record(rv, VoiceClass::Cycle, channels);
                v.arena_offset = off;
                push_voice(&mut out, v, VoiceClass::Cycle);
                continue;
            }
            ObjectData::Literal(l) => {
                let off =
                    arena_segment(&mut out.samples, &mut segment_of, rv.target_id, &l.samples)?;
                let mut v = base_record(rv, VoiceClass::Literal, channels);
                v.arena_offset = off;
                push_voice(&mut out, v, VoiceClass::Literal);
                continue;
            }
            ObjectData::PredictorResidual(r) => {
                // CUDA_FALLBACK: materialize the exact closure H+R into the
                // arena. Recorded separately in the exposure counters.
                let extent = obj.descriptor.extent_frames;
                let slots = usize::try_from(extent)
                    .ok()
                    .and_then(|e| e.checked_mul(usize::from(channels)))
                    .ok_or_else(|| Error::limit("residual extent too large for flatten"))?;
                let mut closure: Vec<i32> = Vec::with_capacity(slots);
                for f in 0..extent {
                    for ch in 0..u32::from(channels) {
                        closure.push(r.closure_sample(f, ch as u8));
                    }
                }
                let off = arena_segment(&mut out.samples, &mut segment_of, rv.target_id, &closure)?;
                let mut v = base_record(rv, VoiceClass::Literal, channels);
                v.arena_offset = off;
                push_voice(&mut out, v, VoiceClass::Literal);
                fallback_targets.insert(rv.target_id);
                continue;
            }
            ObjectData::Referenced(_) => {
                return Err(Error::dependency(
                    "flatten: resolved voice target is still Referenced (resolution bug)",
                ));
            }
        };
        let _ = class; // exhaustiveness anchor (all arms `continue`)
    }

    // Fallback accounting: count each materialized residual closure once.
    for id in &fallback_targets {
        let obj = store.get(*id)?;
        out.fallback_closure_bytes = out.fallback_closure_bytes.saturating_add(
            obj.descriptor.extent_frames * u64::from(obj.descriptor.layout.count()) * 4,
        );
    }

    out.upload_bytes = (out.voices.len() as u64) * 128
        + (out.samples.len() as u64) * 4
        + (out.partials.len() as u64) * 8;
    Ok(out)
}

fn push_voice(out: &mut FlattenedWorld, v: FlatVoice, class: VoiceClass) {
    out.class_counts[class as usize] += 1;
    out.voices.push(v);
}

/// Base flat record for a resolved voice (loop/route/envelope/rate facts;
/// class-specific payload fields are set by the caller).
fn base_record(rv: &ResolvedVoice, class: VoiceClass, object_channels: u8) -> FlatVoice {
    let spec = &rv.spec;
    let (route_stereo, route_channel) = match spec.route {
        Route::Mono(ch) => (false, ch),
        Route::StereoPair(base) => (true, base),
    };
    let mut flags = 0u8;
    if route_stereo {
        flags |= F_ROUTE_STEREO;
    }
    if matches!(spec.interp, Interp::Nearest) {
        flags |= F_INTERP_NEAREST;
    }
    let (has_loop, loop_start, loop_end) = match rv.loop_region {
        Some(l) => (true, l.start_frame, l.end_frame),
        None => (false, 0, 0),
    };
    if has_loop {
        flags |= F_LOOP_REGION;
    }
    FlatVoice {
        trigger_frame: spec.trigger_frame,
        note_off: spec.note_off.unwrap_or(NO_NOTE_OFF),
        // Endless classes require start 0 at resolve; content keeps it.
        start_pos_q24: spec.start_pos_q24,
        eff_rate_q24: rv.eff_rate_q24,
        extent_frames: rv.extent_frames,
        loop_start,
        loop_end,
        arena_offset: 0,
        osc_incr: 0,
        noise_seed: 0,
        envelope: FlatEnvelope::from_params(spec.envelope),
        gain_q16: spec.gain_q16,
        pan_q16: spec.pan_q16,
        amp_q16: 0,
        level: 0,
        partials_offset: 0,
        partials_len: 0,
        class,
        flags,
        route_channel,
        object_channel: spec.object_channel,
        object_channels,
        _pad: [0; 3],
    }
}

/// Append one interleaved content segment (deduped by target id) and return
/// its element offset. Enforces the arena budget.
fn arena_segment(
    samples: &mut Vec<i32>,
    segment_of: &mut HashMap<ObjectId, u64>,
    id: ObjectId,
    content: &[i32],
) -> Result<u64> {
    if let Some(&off) = segment_of.get(&id) {
        return Ok(off);
    }
    if (samples.len() as u64 + content.len() as u64) * 4 > MAX_FLAT_ARENA_BYTES {
        return Err(Error::limit("flattened arena exceeds MAX_FLAT_ARENA_BYTES"));
    }
    let off = samples.len() as u64;
    samples.extend_from_slice(content);
    segment_of.insert(id, off);
    Ok(off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::ScalarOracle;

    /// Flatten a world and assert bit equality with the scalar oracle over
    /// several windows (the flat evaluator's host-side parity anchor).
    fn assert_flat_parity(
        label: &str,
        store: &ObjectStore,
        world: &World,
        windows: &[(i64, usize)],
    ) {
        let oracle = ScalarOracle::new(world.clone());
        let flat = flatten(store, world).expect("flatten");
        for &(start, frames) in windows {
            let a = oracle
                .observe(store, start, frames)
                .expect("scalar observe");
            let b = flat.render(start, frames);
            assert_eq!(
                a,
                b,
                "{label}: flat != scalar at window [{start}, {})",
                start + frames as i64
            );
        }
    }

    #[test]
    fn flat_matches_scalar_on_court_fixtures() {
        let (store, events) = crate::courts::semantic::semantic_court_fixture();
        let world = World::new(
            crate::courts::semantic::RATE_HZ,
            crate::courts::semantic::CHANNELS,
            events,
        )
        .unwrap();
        assert_flat_parity(
            "semantic",
            &store,
            &world,
            &[(0, 2400), (0, 1), (700, 900), (1600, 800), (50, 64)],
        );

        let (store, events) = crate::courts::authored::authored_court_fixture();
        let world = World::new(crate::courts::authored::RATE_HZ, 1, events).unwrap();
        assert_flat_parity(
            "authored",
            &store,
            &world,
            &[(0, 4000), (0, 1), (400, 3600), (1600, 2400)],
        );
    }

    #[test]
    fn flat_matches_scalar_on_random_battery() {
        // Same corpus/voice generator as the SIMD differential battery (the
        // proven scalar==SIMD distribution): every class, adversarial rates,
        // envelopes, pan, loops, note-offs, reverse.
        use crate::eval::battery::{corpus, random_voice};
        use crate::sampler::scheduler::TimelineEvent;
        let seeds = 3000u64;
        let mut compared = 0u64;
        let mut audible = 0u64;
        for seed in 0..seeds {
            let (store, pool) = corpus(seed);
            let mut rng = crate::universe::prng::XoShiro256::from_seed(seed ^ 0xFAC7);
            let next32 = |rng: &mut crate::universe::prng::XoShiro256| rng.next_u64() as u32;
            let n_voices = 1 + (next32(&mut rng) % 4);
            let mut events: Vec<TimelineEvent> = Vec::new();
            for _ in 0..n_voices {
                let object = pool[(next32(&mut rng) as usize) % pool.len()];
                let mut spec = random_voice(object, &store, &mut rng, 2);
                // Half the worlds get a note-off after the trigger.
                if next32(&mut rng) % 2 == 0 {
                    spec.note_off = Some(spec.trigger_frame + 1 + (next32(&mut rng) % 600) as i64);
                }
                events.push(TimelineEvent::VoiceOn(spec));
            }
            let start = (next32(&mut rng) % 1200) as i64;
            let frames = 1 + (next32(&mut rng) % 1600) as usize;
            let world = match World::new(48_000, 2, events) {
                Ok(w) => w,
                Err(_) => continue, // out-of-domain random draw; not a parity case
            };
            let oracle = ScalarOracle::new(world.clone());
            let a = match oracle.observe(&store, start, frames) {
                Ok(a) => a,
                Err(_) => continue, // resolution rejects an edge draw
            };
            if a.iter().any(|&x| x != 0) {
                audible += 1;
            }
            let flat = match flatten(&store, &world) {
                Ok(f) => f,
                Err(e) => panic!("flatten failed on world scalar rendered (seed {seed}): {e}"),
            };
            let b = flat.render(start, frames);
            assert_eq!(
                a,
                b,
                "flat != scalar at seed {seed}, window [{start}, {})",
                start + frames as i64
            );
            compared += 1;
        }
        assert!(
            compared > seeds / 2,
            "battery compared too few worlds ({compared})"
        );
        assert!(
            audible > seeds / 10,
            "battery worlds were suspiciously silent ({audible}/{compared})"
        );
    }
}
