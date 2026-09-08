//! Voice semantics (frozen).
//!
//! A `VoiceSpec` is the immutable description of one voice instance: what it
//! plays, when, and how. A `ResolvedVoice` pins object resolution (reference
//! chains collapsed, transpose applied) and the *play source class*, so the
//! per-frame contribution is a pure function of `(resolved, frame)` — no
//! sequential state. Observation is randomly accessible and chunk-safe.
//!
//! Contribution chain per frame (frozen; U1_SPEC §12):
//!
//! ```text
//! content sources: position u(t) = start + rate*(t - trigger)  (Q24)
//!                  read w(u) with continuation rules
//! endless sources: generator(t)  (phase accumulation / noise / level)
//! envelope level e(t) (Q16 analytic), then gain/pan chain (Q16)
//! contribution = sat(rnd(obs * m, 16))                         (voice bus)
//! ```
//!
//! Class rules: literal content honors the voice loop mode and has a natural
//! end; cycle content (wavetable family) always wraps its cycle and is
//! endless; procedural sources (silence/constant/oscillator/partial bank/
//! noise) are endless and position-independent (rate still scales oscillator
//! pitch through `eff_incr`).

use crate::error::{Error, Result};
use crate::limits::FIXED_Q;
use crate::object::{LoopRegion, ObjectData, ObjectId, ObjectStore, SampleObject};
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::gain;
use crate::sampler::pan::{Route, pan_gains};
use crate::sampler::procedural;
use crate::sampler::rate;

/// Voice loop mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Use the resolved object's declared loop region (if any). Cycle objects
    /// always wrap regardless; procedural objects ignore it.
    Default,
    /// Force one-shot (ignore any object loop).
    Off,
    /// Explicit region (overrides the object's).
    Region(LoopRegion),
}

/// Interpolation kind for content reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    Nearest,
    Linear,
}

/// Immutable voice specification (spawn-time parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceSpec {
    pub object: ObjectId,
    pub trigger_frame: i64,
    /// Earliest note-off frame; `None` = sustain until natural end.
    pub note_off: Option<i64>,
    /// Start position in Q24 object frames (content sources; must be 0 for
    /// endless sources).
    pub start_pos_q24: i64,
    /// Signed rate in Q24 frames per output frame.
    pub rate_q24: i64,
    /// Object channel to read for a mono route; stereo uses ch/ch+1.
    pub object_channel: u8,
    pub route: Route,
    pub gain_q16: i32,
    pub pan_q16: i32,
    pub envelope: EnvelopeParams,
    pub loop_mode: LoopMode,
    pub interp: Interp,
}

impl VoiceSpec {
    /// Validate everything that does not require the store.
    pub fn validate(&self) -> Result<()> {
        if !rate::checked_rate(self.rate_q24) {
            return Err(Error::limit(format!(
                "voice rate {} outside frozen domain",
                self.rate_q24
            )));
        }
        if !gain::checked_gain(self.gain_q16) {
            return Err(Error::limit("voice gain outside frozen domain"));
        }
        if !crate::sampler::pan::checked_pan(self.pan_q16) {
            return Err(Error::limit("voice pan outside frozen domain"));
        }
        if self.note_off.is_some_and(|off| off < self.trigger_frame) {
            return Err(Error::malformed("note_off before trigger"));
        }
        Ok(())
    }
}

/// Resolved voice: spec + pinned object facts (immutable).
#[derive(Debug, Clone)]
pub struct ResolvedVoice {
    pub spec: VoiceSpec,
    /// Final (non-referenced) object this voice reads.
    pub target_id: ObjectId,
    /// Effective per-frame rate (transpose composed), validated domain.
    pub eff_rate_q24: i64,
    /// Extent in frames (content objects).
    pub extent_frames: u64,
    /// Effective loop region (literal content only).
    pub loop_region: Option<LoopRegion>,
    /// Endless procedural / periodic-cycle flag.
    pub endless: bool,
    /// Periodic-cycle (wavetable family) flag.
    pub periodic: bool,
    /// Precomputed oscillator phase increment (endless oscillators).
    pub osc_incr: u64,
    /// Partial-bank base phase increment.
    pub bank_incr: u64,
}

impl ResolvedVoice {
    /// Resolve a spec against a store + nominal sample rate.
    pub fn resolve(store: &ObjectStore, spec: VoiceSpec, fs_hz: u32) -> Result<ResolvedVoice> {
        spec.validate()?;
        let _obj = store.get(spec.object)?;
        let (target_id, obj, transpose) = SampleObject::resolve_target(store, spec.object)?;

        // Effective rate = rate * transpose (one rounding; revalidated).
        let eff = crate::object::compose_transpose(spec.rate_q24, transpose);
        if !rate::checked_rate(eff) {
            return Err(Error::limit(format!(
                "effective rate {eff} outside frozen domain after transpose"
            )));
        }

        let endless = obj.data.is_endless();
        let periodic = obj.data.is_periodic();
        let extent = obj.descriptor.extent_frames;
        let channels = usize::from(obj.descriptor.layout.count());
        let obj_ch = usize::from(spec.object_channel);
        if obj_ch >= channels {
            return Err(Error::malformed(format!(
                "voice object_channel {obj_ch} >= object channels {channels}"
            )));
        }
        if let Route::StereoPair(base) = spec.route {
            if usize::from(base) + 1 >= crate::limits::MAX_CHANNELS as usize {
                return Err(Error::limit("stereo route base out of range"));
            }
            if obj_ch + 1 >= channels {
                return Err(Error::malformed(
                    "stereo route needs object_channel+1 < object channels",
                ));
            }
        }

        // Loop/position rules per class.
        let loop_region = if periodic {
            // Cycle objects always wrap; a voice loop mode must not conflict.
            if !matches!(spec.loop_mode, LoopMode::Default | LoopMode::Off) {
                return Err(Error::malformed(
                    "cycle objects do not accept voice loop regions",
                ));
            }
            None
        } else if endless {
            if !matches!(spec.loop_mode, LoopMode::Default | LoopMode::Off) {
                return Err(Error::malformed(
                    "endless procedural objects do not accept voice loop regions",
                ));
            }
            if spec.start_pos_q24 != 0 {
                return Err(Error::malformed(
                    "endless procedural objects require start position 0",
                ));
            }
            None
        } else {
            match spec.loop_mode {
                LoopMode::Off => None,
                LoopMode::Region(r) => {
                    if r.end_frame > extent {
                        return Err(Error::malformed("loop region beyond object extent"));
                    }
                    Some(r)
                }
                LoopMode::Default => obj.descriptor.loop_region,
            }
        };

        // Start position inside [0, extent<<24) for content objects.
        if !endless && !periodic {
            let max_pos = (extent << FIXED_Q) as i64;
            if !(0..max_pos).contains(&spec.start_pos_q24) {
                return Err(Error::malformed(format!(
                    "start position {} outside object [0, {})",
                    spec.start_pos_q24, extent
                )));
            }
        } else if spec.start_pos_q24 != 0 {
            return Err(Error::malformed(
                "periodic/endless objects require start position 0",
            ));
        }

        // Precompute oscillator/partial phase increments (endless oscillators).
        let osc_incr = match &obj.data {
            ObjectData::Oscillator(o) => procedural::eff_incr(o.freq_hz, eff, fs_hz),
            _ => 0,
        };
        let bank_incr = match &obj.data {
            ObjectData::PartialBank(b) => procedural::eff_incr(b.freq_hz, eff, fs_hz),
            _ => 0,
        };

        Ok(ResolvedVoice {
            spec,
            target_id,
            eff_rate_q24: eff,
            extent_frames: extent,
            loop_region,
            endless,
            periodic,
            osc_incr,
            bank_incr,
        })
    }

    /// Natural end frame. Only one-shot literal content ends; cycles and
    /// procedural sources sustain until note-off.
    pub(crate) fn end_frame(&self) -> Option<i64> {
        if self.endless || self.periodic || self.loop_region.is_some() {
            return None;
        }
        rate::one_shot_end_frame(
            self.spec.start_pos_q24,
            self.eff_rate_q24,
            self.extent_frames,
            self.spec.trigger_frame,
        )
    }

    /// Raw content position (Q24) at media frame t.
    #[inline]
    pub fn position_at(&self, t: i64) -> i64 {
        let d = t - self.spec.trigger_frame;
        self.spec
            .start_pos_q24
            .wrapping_add(rate::advance(self.eff_rate_q24, d))
    }

    /// The exact contribution of this voice at media frame `t` — the single
    /// semantic source of truth shared by the scalar world engine, the SIMD
    /// engine, and (later) flattened GPU evaluation. `None` slots mean the
    /// voice is silent at `t` (not yet triggered, ended, envelope zero, or
    /// fully released). The two slots are the routed output channels
    /// (mono: `a` only; stereo pair: `a` = left base, `b` = base+1).
    pub fn contribution_at(&self, store: &ObjectStore, t: i64) -> Contribution {
        let spec = &self.spec;
        if self.end_frame().is_some_and(|end| t >= end) {
            return Contribution::default();
        }
        let env = spec.envelope.level_at(spec.trigger_frame, spec.note_off, t);
        if env == 0 {
            return Contribution::default();
        }
        if spec
            .envelope
            .silent_at(spec.trigger_frame, spec.note_off, t)
        {
            return Contribution::default();
        }

        let obj = match store.get(self.target_id) {
            Ok(o) => o,
            Err(_) => return Contribution::default(), // store validated; unreachable
        };
        let channels = usize::from(obj.descriptor.layout.count());
        let obj_ch = usize::from(spec.object_channel);
        let obs_of = |ch: usize| -> i32 { self.observe_channel(&obj.data, channels, ch, t) };

        let mut out = Contribution::default();
        match spec.route {
            Route::Mono(out_ch) => {
                let m = gain::channel_multiplier(
                    gain::env_gain_multiplier(env, spec.gain_q16),
                    1 << 16,
                );
                out.a = Some((out_ch, gain::contribution(obs_of(obj_ch), m)));
            }
            Route::StereoPair(base) => {
                let (gl, gr) = pan_gains(spec.pan_q16);
                let l_obs = obs_of(obj_ch);
                let r_obs = obs_of(obj_ch + 1); // resolve guarantees ch+1 < channels
                let ml =
                    gain::channel_multiplier(gain::env_gain_multiplier(env, spec.gain_q16), gl);
                let mr =
                    gain::channel_multiplier(gain::env_gain_multiplier(env, spec.gain_q16), gr);
                out.a = Some((base, gain::contribution(l_obs, ml)));
                out.b = Some((base + 1, gain::contribution(r_obs, mr)));
            }
        }
        out
    }
}

/// One voice's routed contributions at a frame (exact per-frame semantics).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Contribution {
    /// First routed contribution: `(output channel, voice-bus value)`.
    pub a: Option<(u8, i32)>,
    /// Second routed contribution (stereo pair right side).
    pub b: Option<(u8, i32)>,
}

impl ResolvedVoice {
    /// Sample one object channel at media frame `t` (exact class dispatch;
    /// the single source also used by the scalar world engine).
    #[inline]
    pub(crate) fn observe_channel(
        &self,
        data: &ObjectData,
        channels: usize,
        ch: usize,
        t: i64,
    ) -> i32 {
        match data {
            ObjectData::Silence => 0,
            ObjectData::Constant(c) => c.level,
            ObjectData::Noise(n) => crate::universe::prng::noise_sample(n.seed, t),
            ObjectData::Oscillator(o) => procedural::osc_sample(
                procedural::osc_phase(0, self.osc_incr, self.spec.trigger_frame, t),
                o.amp_q16,
            ),
            ObjectData::PartialBank(b) => procedural::partial_bank_sample(
                &b.partials,
                self.bank_incr,
                self.spec.trigger_frame,
                t,
            ),
            data @ (ObjectData::Literal(_)
            | ObjectData::Wavetable(_)
            | ObjectData::SingleCycle(_)
            | ObjectData::ExactRepeat(_)
            | ObjectData::PredictorResidual(_)) => {
                let w = self.position_at(t);
                self.read_content(data, channels, ch, w)
            }
            ObjectData::Referenced(_) => unreachable!("resolved target is never Referenced"),
        }
    }

    /// Interpolated content read at a Q24 position for one channel, with the
    /// frozen continuation rules:
    ///   cycle content: `[0, extent)` always wraps (neighbor at `extent-1` is
    ///   0); literal and residual-governed content honor the voice loop
    ///   region (neighbor at `b-1` is `a`) or hold the final sample.
    /// Residual-governed reads evaluate the *closure* (H+R) at the two
    /// neighbor frames, so integer-frame reads reproduce the intrinsic
    /// exactly.
    #[inline]
    pub(crate) fn read_content(
        &self,
        data: &ObjectData,
        channels: usize,
        ch: usize,
        w_q24: i64,
    ) -> i32 {
        let is_cycle = matches!(
            data,
            ObjectData::Wavetable(_) | ObjectData::SingleCycle(_) | ObjectData::ExactRepeat(_)
        );
        let sample_at = |i: usize| -> i32 {
            match data {
                ObjectData::Literal(l) => l.samples[i * channels + ch],
                ObjectData::Wavetable(c)
                | ObjectData::SingleCycle(c)
                | ObjectData::ExactRepeat(c) => c.samples[i * channels + ch],
                ObjectData::PredictorResidual(r) => r.closure_sample(i as u64, ch as u8),
                _ => unreachable!(),
            }
        };
        let (region_last, wrap_to) = if is_cycle {
            (self.extent_frames - 1, Some(0u64)) // cycle content wraps
        } else {
            match self.loop_region {
                Some(l) => (l.end_frame - 1, Some(l.start_frame)),
                None => (self.extent_frames - 1, None),
            }
        };
        let w = if is_cycle {
            rate::wrap_loop_q24(w_q24, 0, self.extent_frames)
        } else {
            match self.loop_region {
                Some(l) => rate::wrap_loop_q24(w_q24, l.start_frame, l.end_frame),
                None => w_q24,
            }
        };
        let idx0 = ((w >> FIXED_Q) as usize).min(region_last as usize);
        let frac = (w as u64 & 0xFF_FFFF) as u32;
        let next = if idx0 == region_last as usize {
            wrap_to.map(|a| a as usize).unwrap_or(idx0)
        } else {
            idx0 + 1
        };
        match self.spec.interp {
            Interp::Linear => {
                crate::universe::arithmetic::lerp_i32(sample_at(idx0), sample_at(next), frac)
            }
            Interp::Nearest => {
                if frac >= (1 << 23) && next != idx0 {
                    sample_at(next)
                } else {
                    sample_at(idx0)
                }
            }
        }
    }
}
