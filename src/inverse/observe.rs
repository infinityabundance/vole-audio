//! Exact observation/reconstruction helpers for the inverse compiler.
//!
//! Two *independent* reconstructions of a candidate hypothesis are used for
//! acceptance, and both must equal the observed window:
//!
//! * **intrinsic closure** — the representation's own exact reconstruction
//!   (`Literal` bytes, a cycle's wrapping expansion, `H + R` closure), computed
//!   without the sampler; and
//! * **scalar-oracle observation** — the normative evaluator: the object is
//!   inserted into a store and observed through `World` with an identity voice
//!   (unity rate/gain/pan, instant full-sustain envelope, nearest
//!   interpolation at integer frames, start 0, no loop).
//!
//! The identity voice is the *only* voice configuration under which the
//! observation equals the intrinsic content; `sampler::gain` proves the unity
//! chain is the identity, and integer positions with `Interp::Nearest` read
//! the exact frame. Requiring both reconstructions is what makes acceptance an
//! exactness claim rather than a plausibility claim.

use crate::error::{Error, Kind, Result};
use crate::evidence::timing::Stopwatch;
use crate::object::{ObjectData, ObjectDescriptor, ObjectId, ObjectStore};
use crate::sampler::envelope::{ENV_UNITY, EnvelopeParams};
use crate::sampler::pan::Route;
use crate::sampler::scheduler::TimelineEvent;
use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
use crate::sampler::world::World;
use crate::universe::layout::Layout;

/// Nominal rate for inverse observation. Content classes (literal / cycle /
/// residual / silence / constant) are rate-independent; the rate exists only
/// to build a legal world.
pub const OBSERVE_RATE_HZ: u32 = crate::limits::DEFAULT_SAMPLE_RATE_HZ;

/// The identity voice for one object channel.
fn identity_voice(object: ObjectId, channel: u8) -> Result<VoiceSpec> {
    let envelope = EnvelopeParams::new(0, 0, ENV_UNITY, 0)
        .ok_or_else(|| Error::internal("instant unity envelope must be legal"))?;
    Ok(VoiceSpec {
        object,
        trigger_frame: 0,
        note_off: None,
        start_pos_q24: 0,
        rate_q24: 1 << 24,
        object_channel: channel,
        route: Route::Mono(channel),
        gain_q16: 1 << 16,
        pan_q16: 0,
        envelope,
        loop_mode: LoopMode::Off,
        interp: Interp::Nearest,
    })
}

/// Observe `[0, frames)` of one object through the scalar oracle (semantic
/// authority), one identity voice per channel.
pub fn observe_object(
    store: &ObjectStore,
    id: ObjectId,
    channels: u8,
    frames: usize,
) -> Result<Vec<i32>> {
    observe_window(store, id, channels, 0, frames)
}

/// Observe `[start, start + frames)` of one object through the scalar oracle.
pub fn observe_window(
    store: &ObjectStore,
    id: ObjectId,
    channels: u8,
    start: i64,
    frames: usize,
) -> Result<Vec<i32>> {
    if frames == 0 {
        return Err(Error::malformed("zero-frame inverse observation"));
    }
    if frames > crate::limits::MAX_QUANTUM_FRAMES as usize {
        return Err(Error::limit(
            "inverse observation exceeds the quantum ceiling",
        ));
    }
    if channels == 0 || u32::from(channels) > crate::limits::MAX_CHANNELS {
        return Err(Error::malformed(
            "inverse observation channel count out of domain",
        ));
    }
    let events: Vec<TimelineEvent> = (0..channels)
        .map(|c| identity_voice(id, c).map(TimelineEvent::VoiceOn))
        .collect::<Result<Vec<_>>>()?;
    let world = World::new(OBSERVE_RATE_HZ, channels, events)?;
    crate::eval::ScalarOracle::new(world).observe(store, start, frames)
}

/// Exact intrinsic reconstruction of a candidate payload over `frames`
/// frames — the representation's own closure, independent of the sampler.
///
/// `Silence`/`Constant` are endless (extent 0); their reconstruction is
/// defined for any `frames`. Cycle classes wrap their stored cycle. Literal
/// and residual-governed objects must cover `frames`.
pub fn intrinsic_reconstruction(
    descriptor: &ObjectDescriptor,
    data: &ObjectData,
    frames: u64,
) -> Result<Vec<i32>> {
    let channels = usize::from(descriptor.layout.count());
    let total = usize::try_from(frames)
        .ok()
        .and_then(|f| f.checked_mul(channels))
        .ok_or_else(|| Error::limit("intrinsic reconstruction overflow"))?;
    let mut out = vec![0i32; total];

    match data {
        ObjectData::Silence => {}
        ObjectData::Constant(c) => out.fill(c.level),
        ObjectData::Literal(l) => {
            if l.samples.len() != total {
                return Err(Error::malformed(
                    "literal candidate does not cover the observation window",
                ));
            }
            out.copy_from_slice(&l.samples);
        }
        ObjectData::Wavetable(cycle)
        | ObjectData::SingleCycle(cycle)
        | ObjectData::ExactRepeat(cycle) => {
            if channels == 0 || cycle.samples.len() % channels != 0 {
                return Err(Error::malformed("cycle payload is not frame-aligned"));
            }
            let period = cycle.samples.len() / channels;
            if period == 0 {
                return Err(Error::malformed("empty cycle payload"));
            }
            for f in 0..frames as usize {
                let src = (f % period) * channels;
                out[f * channels..f * channels + channels]
                    .copy_from_slice(&cycle.samples[src..src + channels]);
            }
        }
        ObjectData::PredictorResidual(r) => {
            for f in 0..frames {
                for ch in 0..channels {
                    out[f as usize * channels + ch] = r.closure_sample(f, ch as u8);
                }
            }
        }
        ObjectData::Referenced(_) => {
            // A reference is resolved before observation; its intrinsic
            // reconstruction is the target's, handled by the caller.
            return Err(Error::new(
                Kind::Unsupported,
                "intrinsic reconstruction of a reference must resolve its target first",
            ));
        }
        _ => {
            return Err(Error::new(
                Kind::Unsupported,
                "intrinsic reconstruction is not defined for this representation in Phase K",
            ));
        }
    }
    Ok(out)
}

/// Timed intrinsic reconstruction (Phase-K measured quantity).
pub fn timed_intrinsic_reconstruction(
    descriptor: &ObjectDescriptor,
    data: &ObjectData,
    frames: u64,
) -> Result<(Vec<i32>, u64)> {
    let sw = Stopwatch::start();
    let out = intrinsic_reconstruction(descriptor, data, frames)?;
    Ok((out, sw.elapsed_ns().max(0) as u64))
}

/// Layout for an intrinsic channel count.
pub fn layout_of(channels: u8) -> Result<Layout> {
    Layout::checked(channels).ok_or_else(|| {
        Error::malformed(
            "intrinsic channel count out of domain (channels must be 1..=MAX_CHANNELS)",
        )
    })
}
