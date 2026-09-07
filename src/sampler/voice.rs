//! Voice semantics (frozen).
//!
//! A `VoiceSpec` is the immutable description of one voice instance: what it
//! plays, when, and how. A `ResolvedVoice` additionally pins the object
//! resolution (reference chains collapsed, transpose applied) so the
//! per-frame contribution is a *pure function* of `(resolved, frame)` — no
//! sequential state, so observation is randomly accessible and chunk-safe.
//!
//! Contribution chain per frame (all frozen, see the module docs):
//!   position u(t) = start + rate*(t - trigger)          (Q24)
//!   envelope level e(t)                                 (Q16 analytic)
//!   read obs = object channel sample(s) at w(u)         (linear/nearest)
//!   per-channel multiplier m = chain(env, gain, pan)    (Q16)
//!   contribution = sat(rnd(obs * m, 16))                (voice bus)

use crate::error::{Error, Result};
use crate::limits::FIXED_Q;
use crate::object::{LoopRegion, ObjectData, ObjectId, ObjectStore, SampleObject};
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::gain;
use crate::sampler::mix::Mixer;
use crate::sampler::pan::{Route, pan_gains};
use crate::sampler::rate;

/// Voice loop mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Use the resolved object's declared loop region (if any).
    Default,
    /// Force one-shot (ignore any object loop).
    Off,
    /// Explicit region (overrides the object's).
    Region(LoopRegion),
}

/// Interpolation kind for object reads.
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
    /// Start position in Q24 object frames.
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
    /// The literal object this voice ultimately reads.
    pub literal_id: ObjectId,
    /// Effective per-frame rate (transpose composed), validated domain.
    pub eff_rate_q24: i64,
    /// Object extent in frames (of the literal).
    pub extent_frames: u64,
    /// Effective loop region (object/voice semantics resolved).
    pub loop_region: Option<LoopRegion>,
}

impl ResolvedVoice {
    /// Resolve a spec against a store (reference chains collapsed).
    pub fn resolve(store: &ObjectStore, spec: VoiceSpec) -> Result<ResolvedVoice> {
        spec.validate()?;
        // Existence check; resolution follows below.
        let _obj = store.get(spec.object)?;
        let (literal_id, _literal, transpose) = SampleObject::resolve_literal(store, spec.object)?;
        let literal = store.get(literal_id)?;

        // Effective rate = rate * transpose (one rounding; host-side i64 is
        // exact here; magnitude revalidated).
        let eff = crate::object::compose_transpose(spec.rate_q24, transpose);
        if !rate::checked_rate(eff) {
            return Err(Error::limit(format!(
                "effective rate {eff} outside frozen domain after transpose"
            )));
        }

        // Object channel must exist in the literal layout.
        let ch = usize::from(literal.descriptor.layout.count());
        let obj_ch = usize::from(spec.object_channel);
        if obj_ch >= ch {
            return Err(Error::malformed(format!(
                "voice object_channel {obj_ch} >= object channels {ch}"
            )));
        }
        if let Route::StereoPair(base) = spec.route {
            if usize::from(base) + 1 >= crate::limits::MAX_CHANNELS as usize {
                return Err(Error::limit("stereo route base out of range"));
            }
            if obj_ch + 1 >= ch {
                return Err(Error::malformed(
                    "stereo route needs object_channel+1 < object channels",
                ));
            }
        }

        // Effective loop region.
        let loop_region = match spec.loop_mode {
            LoopMode::Off => None,
            LoopMode::Region(r) => {
                if r.end_frame > literal.descriptor.extent_frames {
                    return Err(Error::malformed("loop region beyond object extent"));
                }
                Some(r)
            }
            LoopMode::Default => literal.descriptor.loop_region,
        };

        // Start position inside [0, extent<<24).
        let extent = literal.descriptor.extent_frames;
        let max_pos = (extent << FIXED_Q) as i64;
        if !(0..max_pos).contains(&spec.start_pos_q24) {
            return Err(Error::malformed(format!(
                "start position {} outside object [0, {})",
                spec.start_pos_q24, extent
            )));
        }

        Ok(ResolvedVoice {
            spec,
            literal_id,
            eff_rate_q24: eff,
            extent_frames: extent,
            loop_region,
        })
    }

    /// Natural end frame (one-shot only). Loops have none.
    fn end_frame(&self) -> Option<i64> {
        if self.loop_region.is_some() {
            return None;
        }
        rate::one_shot_end_frame(
            self.spec.start_pos_q24,
            self.eff_rate_q24,
            self.extent_frames,
            self.spec.trigger_frame,
        )
    }

    /// Raw position (Q24) at media frame t.
    #[inline]
    pub fn position_at(&self, t: i64) -> i64 {
        let d = t - self.spec.trigger_frame;
        self.spec
            .start_pos_q24
            .wrapping_add(rate::advance(self.eff_rate_q24, d))
    }

    /// Contribution of this voice to `(frame, channel)` given an object
    /// channel sample; route/pan/gain/envelope applied.
    #[inline]
    fn contribute_channel(
        &self,
        obs: i32,
        channel_gain_q16: i32,
        env_q16: i32,
        m: &mut Mixer,
        frame: usize,
        ch: u8,
    ) {
        let e = gain::env_gain_multiplier(env_q16, self.spec.gain_q16);
        let mch = gain::channel_multiplier(e, channel_gain_q16);
        let c = gain::contribution(obs, mch);
        m.add(frame, ch, c);
    }

    /// Render this voice into the mixer for one frame `t` (output frame index
    /// `frame_idx`). Reads object channel planes via the store.
    pub fn render_frame(&self, store: &ObjectStore, mixer: &mut Mixer, t: i64, frame_idx: usize) {
        // Voice silence conditions: envelope silent or natural end passed.
        let env = self
            .spec
            .envelope
            .level_at(self.spec.trigger_frame, self.spec.note_off, t);
        if env == 0 {
            return;
        }
        if self.end_frame().is_some_and(|end| t >= end) {
            return;
        }
        let literal = match store.get(self.literal_id) {
            Ok(l) => l,
            Err(_) => return, // store is validated; unreachable in practice
        };
        let ObjectData::Literal(lit) = &literal.data else {
            return;
        };
        let channels = usize::from(literal.descriptor.layout.count());
        let obj_ch = usize::from(self.spec.object_channel);
        let w = self.position_at(t);
        let w_wrapped = match self.loop_region {
            Some(l) => rate::wrap_loop_q24(w, l.start_frame, l.end_frame),
            None => w,
        };
        let read = |ch: usize| -> i32 {
            let idx = (w_wrapped >> FIXED_Q) as usize;
            let last = self.extent_frames as usize - 1;
            let a = lit.sample(idx.min(last), ch, channels);
            let next_idx = match self.loop_region {
                Some(l) if idx + 1 >= l.end_frame as usize => l.start_frame as usize,
                _ => (idx + 1).min(last),
            };
            let frac = (w_wrapped as u64 & 0xFF_FFFF) as u32;
            match self.spec.interp {
                Interp::Linear => {
                    let b = lit.sample(next_idx, ch, channels);
                    crate::universe::arithmetic::lerp_i32(a, b, frac)
                }
                Interp::Nearest => {
                    if frac >= (1 << 23) {
                        lit.sample(next_idx, ch, channels)
                    } else {
                        a
                    }
                }
            }
        };

        match self.spec.route {
            Route::Mono(out_ch) => {
                let obs = read(obj_ch);
                self.contribute_channel(obs, 1 << 16, env, mixer, frame_idx, out_ch);
            }
            Route::StereoPair(base) => {
                let (gl, gr) = pan_gains(self.spec.pan_q16);
                let l_obs = read(obj_ch);
                let r_obs = read(obj_ch + 1);
                self.contribute_channel(l_obs, gl, env, mixer, frame_idx, base);
                self.contribute_channel(r_obs, gr, env, mixer, frame_idx, base + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Literal;
    use crate::object::descriptor::{ObjectDescriptor, Representation};
    use crate::sampler::envelope::ENV_UNITY;
    use crate::universe::layout::Layout;

    fn store_with(samples: Vec<i32>, chans: u8) -> (ObjectStore, ObjectId) {
        let mut store = ObjectStore::new();
        let frames = (samples.len() / chans as usize) as u64;
        let layout = Layout::checked(chans).unwrap();
        let d = ObjectDescriptor::new(Representation::Literal, frames, layout, None).unwrap();
        let id = store
            .insert(
                d.clone(),
                ObjectData::Literal(Literal::new(&d, samples).unwrap()),
            )
            .unwrap();
        (store, id)
    }

    const FR: i64 = 1 << FIXED_Q;

    fn base_spec(id: ObjectId) -> VoiceSpec {
        VoiceSpec {
            object: id,
            trigger_frame: 0,
            note_off: None,
            start_pos_q24: 0,
            rate_q24: FR,
            object_channel: 0,
            route: Route::Mono(0),
            gain_q16: 1 << 16,
            pan_q16: 0,
            envelope: EnvelopeParams::new(0, 0, ENV_UNITY, 0).unwrap(),
            loop_mode: LoopMode::Off,
            interp: Interp::Nearest,
        }
    }

    #[test]
    fn renders_constant_at_unity() {
        let (store, id) = store_with(vec![1000; 8], 1);
        let spec = base_spec(id);
        let rv = ResolvedVoice::resolve(&store, spec).unwrap();
        let mut mix = Mixer::new(1, 4);
        for f in 0..4 {
            rv.render_frame(&store, &mut mix, f as i64, f);
        }
        let mut out = [0i32; 4];
        mix.finalize_interleaved(&mut out);
        assert_eq!(out, [1000; 4]);
    }

    #[test]
    fn one_shot_ends_at_extent() {
        let (store, id) = store_with(vec![1000; 4], 1);
        let spec = base_spec(id);
        let rv = ResolvedVoice::resolve(&store, spec).unwrap();
        let mut mix = Mixer::new(1, 8);
        for f in 0..8 {
            rv.render_frame(&store, &mut mix, f as i64, f);
        }
        let mut out = [0i32; 8];
        mix.finalize_interleaved(&mut out);
        assert_eq!(out, [1000, 1000, 1000, 1000, 0, 0, 0, 0]);
    }

    #[test]
    fn reverse_plays_backwards() {
        // ramp 0..4; reverse rate -1: outputs 3,2,1,0 then silence.
        let (store, id) = store_with(vec![0, 10, 20, 30], 1);
        let mut spec = base_spec(id);
        spec.rate_q24 = -FR;
        spec.start_pos_q24 = 3 * FR;
        let rv = ResolvedVoice::resolve(&store, spec).unwrap();
        let mut mix = Mixer::new(1, 6);
        for f in 0..6 {
            rv.render_frame(&store, &mut mix, f as i64, f);
        }
        let mut out = [0i32; 6];
        mix.finalize_interleaved(&mut out);
        assert_eq!(out, [30, 20, 10, 0, 0, 0]);
    }

    #[test]
    fn loop_wraps_forward_and_reverse() {
        let (store, id) = store_with(vec![0, 10, 20, 30, 40, 50], 1);
        let mut spec = base_spec(id);
        spec.loop_mode = LoopMode::Region(LoopRegion::new(1, 4).unwrap());
        spec.start_pos_q24 = FR; // begin inside the loop region
        // Forward at unity from frame 1: 10,20,30,10,20,30,...
        let rv = ResolvedVoice::resolve(&store, spec).unwrap();
        let mut mix = Mixer::new(1, 8);
        for f in 0..8 {
            rv.render_frame(&store, &mut mix, f as i64, f);
        }
        let mut out = [0i32; 8];
        mix.finalize_interleaved(&mut out);
        assert_eq!(out, [10, 20, 30, 10, 20, 30, 10, 20]);

        // Reverse from position 4 (just under the end... start at frame 3):
        // 30,20,10,30,20,10,...
        let mut spec2 = base_spec(id);
        spec2.loop_mode = LoopMode::Region(LoopRegion::new(1, 4).unwrap());
        spec2.start_pos_q24 = 3 * FR;
        spec2.rate_q24 = -FR;
        let rv2 = ResolvedVoice::resolve(&store, spec2).unwrap();
        let mut mix2 = Mixer::new(1, 8);
        for f in 0..8 {
            rv2.render_frame(&store, &mut mix2, f as i64, f);
        }
        let mut out2 = [0i32; 8];
        mix2.finalize_interleaved(&mut out2);
        assert_eq!(out2, [30, 20, 10, 30, 20, 10, 30, 20]);
    }

    #[test]
    fn stereo_route_and_pan() {
        let (store, id) = store_with(vec![100, 200, 100, 200], 2); // 2 frames stereo
        let mut spec = base_spec(id);
        spec.route = Route::StereoPair(0);
        spec.pan_q16 = -(1 << 16); // hard left
        let rv = ResolvedVoice::resolve(&store, spec).unwrap();
        let mut mix = Mixer::new(2, 2);
        for f in 0..2 {
            rv.render_frame(&store, &mut mix, f as i64, f);
        }
        let mut out = [0i32; 4];
        mix.finalize_interleaved(&mut out);
        // Left = full gain of object ch0 (100); right = 0.
        assert_eq!(out, [100, 0, 100, 0]);
    }

    #[test]
    fn invalid_specs_are_rejected() {
        let (store, id) = store_with(vec![0; 4], 1);
        // Gain beyond ceiling.
        let mut spec = base_spec(id);
        spec.gain_q16 = (1 << 17) + 1;
        assert!(ResolvedVoice::resolve(&store, spec).is_err());
        // Object channel out of range.
        let mut spec = base_spec(id);
        spec.object_channel = 2;
        assert!(ResolvedVoice::resolve(&store, spec).is_err());
        // Start position beyond extent.
        let mut spec = base_spec(id);
        spec.start_pos_q24 = 4 * FR;
        assert!(ResolvedVoice::resolve(&store, spec).is_err());
        // Loop region beyond extent.
        let mut spec = base_spec(id);
        spec.loop_mode = LoopMode::Region(LoopRegion::new(2, 9).unwrap());
        assert!(ResolvedVoice::resolve(&store, spec).is_err());
    }
}
