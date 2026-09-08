//! Shared flat evaluation core (Phase G) — `no_std`-clean, compiled unchanged
//! for scalar host tests and for GPU device targets.
//!
//! The scalar world (`sampler::world::World::observe`) owns semantic
//! authority. For a GPU backend the world must be *flattened* into plain-data
//! records and arenas (no `ObjectStore`, no `Vec`-backed payloads, no
//! `World`) that a kernel can read from global memory. This module is the
//! device-side mirror of that semantics:
//!
//! * `FlatVoice` — every resolved-voice fact the per-frame math needs
//!   (`ResolvedVoice` + `VoiceSpec` + the resolved target's content facts),
//!   all `#[repr(C)]` so host and device compilers agree on layout.
//! * `render_sample` — the exact contribution mix for one output sample slot
//!   `(frame, channel)`: identical branch structure and arithmetic to
//!   `sampler::voice::contribution_at` + `World::observe`, calling the same
//!   frozen `no_std` functions (`envelope`, `gain`, `pan`, `rate`,
//!   `procedural`, `universe::arithmetic/phase/prng`). Only the reduction
//!   order per (frame, channel) may differ (i64 addition is associative); the
//!   set of added terms is identical.
//! * `render_window` — host-side sequential driver over `render_sample`
//!   (parity anchor and CPU reference for the GPU court).
//!
//! Exactness contract: for every flattened (world, window),
//! `render_window == World::observe` bit for bit, verified by the random
//! differential battery in `backend::flatten` and by `court cuda`.
//!
//! Every struct here is `#[repr(C)]` plain data: kernels never see `Vec`,
//! `Option` niches across the boundary, or heap.

use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::procedural::{Partial, osc_phase, osc_sample, partial_bank_sample};
use crate::sampler::{gain, pan, rate};
use crate::universe::arithmetic::{lerp_i32, sat_i32};
use crate::universe::prng::noise_sample;

/// Sentinel for "no note-off" (real note-offs are far below `i64::MAX`).
pub const NO_NOTE_OFF: i64 = i64::MAX;

/// Voice class codes (frozen). `backend::flatten` maps each resolved target
/// representation to exactly one class; `CUDA_FALLBACK` classes
/// (`PredictorResidual`, `Referenced`) are resolved into arena content on the
/// host before upload and arrive here as `Literal` (counted separately in the
/// receipts).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceClass {
    /// Endless procedural silence.
    Silence = 0,
    /// Endless procedural constant level.
    Constant = 1,
    /// Endless deterministic noise (stream keyed by media frame).
    Noise = 2,
    /// Endless oscillator (u64 phase DDS over the frozen sine table).
    Oscillator = 3,
    /// Endless partial bank.
    PartialBank = 4,
    /// Periodic cycle content: reads always wrap `[0, extent)`.
    Cycle = 5,
    /// Finite content (literal; residual-governed objects arrive here as a
    /// host-resolved closure arena).
    Literal = 6,
}

impl VoiceClass {
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// Content classes read the sample arena (cycles + literals).
    pub const fn is_content(self) -> bool {
        matches!(self, VoiceClass::Cycle | VoiceClass::Literal)
    }

    /// Endless classes have no natural end (mirror of `ObjectData::is_endless`).
    pub const fn is_endless(self) -> bool {
        !matches!(self, VoiceClass::Cycle | VoiceClass::Literal)
    }
}

/// Flat-voice flag bits.
pub const F_ROUTE_STEREO: u8 = 1 << 0;
pub const F_INTERP_NEAREST: u8 = 1 << 1;
/// Content voices honoring a voice loop region (`loop_start`/`loop_end`).
pub const F_LOOP_REGION: u8 = 1 << 2;

/// Flat envelope parameters (mirror of `sampler::envelope::EnvelopeParams`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FlatEnvelope {
    pub attack_frames: u32,
    pub decay_frames: u32,
    pub sustain_q16: i32,
    pub release_frames: u32,
}

impl FlatEnvelope {
    #[inline]
    pub const fn from_params(p: EnvelopeParams) -> FlatEnvelope {
        FlatEnvelope {
            attack_frames: p.attack_frames,
            decay_frames: p.decay_frames,
            sustain_q16: p.sustain_q16,
            release_frames: p.release_frames,
        }
    }

    #[inline]
    pub const fn to_params(self) -> EnvelopeParams {
        EnvelopeParams {
            attack_frames: self.attack_frames,
            decay_frames: self.decay_frames,
            sustain_q16: self.sustain_q16,
            release_frames: self.release_frames,
        }
    }
}

/// One flattened resolved voice (plain data; host fills, device reads).
///
/// Field semantics are exactly those of `sampler::voice::ResolvedVoice` plus
/// the resolved target object's content facts; see that module for the
/// authoritative comments. Every content/procedural parameter is either
/// stored inline or addressed into the flat arenas (`arena_offset`,
/// `partials_offset`) — no pointers, no heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FlatVoice {
    // --- scalars first (natural alignment; no packing surprises) ---
    pub trigger_frame: i64,
    /// `NO_NOTE_OFF` when the voice sustains until its natural end.
    pub note_off: i64,
    pub start_pos_q24: i64,
    /// Effective per-output-frame rate (transpose composed, Q24).
    pub eff_rate_q24: i64,
    /// Intrinsic extent in frames (content voices).
    pub extent_frames: u64,
    /// Loop region (content voices with `F_LOOP_REGION`).
    pub loop_start: u64,
    pub loop_end: u64,
    /// Element offset into the i32 sample arena (content voices).
    pub arena_offset: u64,
    /// Oscillator phase increment per output frame, or partial-bank base.
    pub osc_incr: u64,
    /// Noise stream seed.
    pub noise_seed: u64,
    /// Attack/decay/sustain/release (Q16 sustain; frames for segments).
    pub envelope: FlatEnvelope,
    pub gain_q16: i32,
    pub pan_q16: i32,
    /// Oscillator amplitude (Q16).
    pub amp_q16: i32,
    /// Constant level (class `Constant`).
    pub level: i32,
    /// Element offset into the partial arena (`Partial` records).
    pub partials_offset: u32,
    pub partials_len: u32,
    pub class: VoiceClass,
    pub flags: u8,
    /// Mono route: the output channel; stereo route: the base channel.
    pub route_channel: u8,
    /// Object channel read for a mono route / the left of a stereo pair.
    pub object_channel: u8,
    /// Channel count of the resolved target object's intrinsic layout.
    pub object_channels: u8,
    /// Explicit padding so every byte of the record is defined (the flat
    /// records are uploaded as raw bytes; no uninitialized padding may exist).
    pub _pad: [u8; 3],
}

impl FlatVoice {
    /// Natural end frame, mirroring `ResolvedVoice::end_frame`.
    #[inline]
    fn end_frame(&self) -> Option<i64> {
        if self.class.is_endless()
            || matches!(self.class, VoiceClass::Cycle)
            || self.flags & F_LOOP_REGION != 0
        {
            return None;
        }
        rate::one_shot_end_frame(
            self.start_pos_q24,
            self.eff_rate_q24,
            self.extent_frames,
            self.trigger_frame,
        )
    }

    /// Raw content position (Q24) at media frame `t` — mirror of
    /// `ResolvedVoice::position_at`.
    #[inline]
    fn position_at(&self, t: i64) -> i64 {
        let d = t - self.trigger_frame;
        self.start_pos_q24
            .wrapping_add(rate::advance(self.eff_rate_q24, d))
    }

    /// One content sample at intrinsic frame `i`, channel `ch`, from the
    /// flat arena. `i < extent_frames` and `ch < object_channels` are
    /// guaranteed by the read path (host validated at flatten time).
    #[inline]
    fn arena_sample(&self, samples: &[i32], i: u64, ch: u8) -> i32 {
        let idx = self.arena_offset as usize
            + (i as usize) * usize::from(self.object_channels)
            + usize::from(ch);
        samples[idx]
    }

    /// Content read at a Q24 position for one channel — exact mirror of
    /// `ResolvedVoice::read_content` (cycle wrap / loop region / hold-final,
    /// linear or nearest with the frozen continuation rules).
    #[inline]
    fn read_content(&self, samples: &[i32], ch: u8, w_q24: i64) -> i32 {
        let is_cycle = matches!(self.class, VoiceClass::Cycle);
        let (region_last, wrap_to) = if is_cycle {
            (self.extent_frames - 1, Some(0u64))
        } else if self.flags & F_LOOP_REGION != 0 {
            (self.loop_end - 1, Some(self.loop_start))
        } else {
            (self.extent_frames - 1, None)
        };
        let w = if is_cycle {
            rate::wrap_loop_q24(w_q24, 0, self.extent_frames)
        } else if self.flags & F_LOOP_REGION != 0 {
            rate::wrap_loop_q24(w_q24, self.loop_start, self.loop_end)
        } else {
            w_q24
        };
        let idx0 = ((w >> crate::limits::FIXED_Q) as u64).min(region_last);
        let frac = (w as u64 & 0xFF_FFFF) as u32;
        let next = if idx0 == region_last {
            wrap_to.unwrap_or(idx0)
        } else {
            idx0 + 1
        };
        let a = self.arena_sample(samples, idx0, ch);
        let b = self.arena_sample(samples, next, ch);
        if self.flags & F_INTERP_NEAREST != 0 {
            if frac >= (1 << 23) && next != idx0 {
                b
            } else {
                a
            }
        } else {
            lerp_i32(a, b, frac)
        }
    }

    /// Object-domain observation of one channel at media frame `t` — exact
    /// mirror of `ResolvedVoice::observe_channel` over the flat arenas.
    #[inline]
    fn observe_channel(&self, samples: &[i32], partials: &[Partial], ch: u8, t: i64) -> i32 {
        match self.class {
            VoiceClass::Silence => 0,
            VoiceClass::Constant => self.level,
            VoiceClass::Noise => noise_sample(self.noise_seed, t),
            VoiceClass::Oscillator => osc_sample(
                osc_phase(0, self.osc_incr, self.trigger_frame, t),
                self.amp_q16,
            ),
            VoiceClass::PartialBank => partial_bank_sample(
                &partials[self.partials_offset as usize
                    ..(self.partials_offset as usize + self.partials_len as usize)],
                self.osc_incr,
                self.trigger_frame,
                t,
            ),
            VoiceClass::Cycle | VoiceClass::Literal => {
                let w = self.position_at(t);
                self.read_content(samples, ch, w)
            }
        }
    }

    /// Voice-bus contribution routed to output channel `c` at media frame
    /// `t`, or 0 when the voice is silent/not routed there. Exact mirror of
    /// `sampler::voice::contribution_at` restricted to one channel: same
    /// gates, same multiplier composition, same final `gain::contribution`
    /// saturation.
    #[inline]
    pub fn contribution_to_channel(
        &self,
        samples: &[i32],
        partials: &[Partial],
        t: i64,
        c: u8,
    ) -> i32 {
        if t < self.trigger_frame {
            return 0;
        }
        if self.end_frame().is_some_and(|end| t >= end) {
            return 0;
        }
        let note_off = if self.note_off == NO_NOTE_OFF {
            None
        } else {
            Some(self.note_off)
        };
        let env = self.envelope.to_params();
        if env.silent_at(self.trigger_frame, note_off, t) {
            return 0;
        }
        let env_level = env.level_at(self.trigger_frame, note_off, t);
        if env_level == 0 {
            return 0;
        }
        let eg = gain::env_gain_multiplier(env_level, self.gain_q16);
        if self.flags & F_ROUTE_STEREO != 0 {
            // Stereo pair across (base, base+1); left reads `object_channel`,
            // right reads `object_channel + 1`.
            if c != self.route_channel && c != self.route_channel + 1 {
                return 0;
            }
            let (gl, gr) = pan::pan_gains(self.pan_q16);
            let (side_is_left, pg) = if c == self.route_channel {
                (true, gl)
            } else {
                (false, gr)
            };
            let obj_ch = if side_is_left {
                self.object_channel
            } else {
                self.object_channel + 1
            };
            let m = gain::channel_multiplier(eg, pg);
            let obs = self.observe_channel(samples, partials, obj_ch, t);
            gain::contribution(obs, m)
        } else {
            if c != self.route_channel {
                return 0;
            }
            let m = gain::channel_multiplier(eg, 1 << 16);
            let obs = self.observe_channel(samples, partials, self.object_channel, t);
            gain::contribution(obs, m)
        }
    }
}

/// Flat render window state (kernel parameters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FlatState {
    /// First media frame of the window.
    pub start_frame: i64,
    /// Frame count of the window.
    pub frames: u32,
    /// Output channel count.
    pub channels: u8,
    pub _pad: [u8; 3],
    /// Voice record count (arena length in `FlatVoice` units).
    pub voices_len: u32,
    /// Partial record count (arena length in `Partial` units).
    pub partials_len: u32,
    /// Sample arena length in i32 elements (content voices).
    pub samples_len: u64,
}

impl FlatState {
    /// Total interleaved sample slots in the window.
    #[inline]
    pub fn total_samples(&self) -> usize {
        self.frames as usize * usize::from(self.channels)
    }

    /// Split a flat sample slot index into (frame offset, channel).
    #[inline]
    pub fn frame_channel(&self, g: usize) -> (u32, u8) {
        (
            (g / usize::from(self.channels)) as u32,
            (g % usize::from(self.channels)) as u8,
        )
    }

    /// Layout sanity: legal channel count / quantum / window size.
    pub fn check_layout(&self) -> Option<()> {
        let ch = usize::from(self.channels);
        if !(1..=crate::limits::MAX_CHANNELS as usize).contains(&ch) {
            return None;
        }
        if self.frames as u64 > crate::limits::MAX_QUANTUM_FRAMES as u64 {
            return None;
        }
        if (self.frames as u64).saturating_mul(ch as u64) > crate::limits::MAX_OBSERVATION_FRAMES {
            return None;
        }
        Some(())
    }
}

/// Build a flat state value.
pub fn flat_state(
    start_frame: i64,
    frames: u32,
    channels: u8,
    voices: usize,
    partials: usize,
    samples: usize,
) -> FlatState {
    FlatState {
        start_frame,
        frames,
        channels,
        _pad: [0; 3],
        voices_len: voices as u32,
        partials_len: partials as u32,
        samples_len: samples as u64,
    }
}

/// Mix value for one output sample slot `g` (`0 <= g < total_samples`):
/// the exact per-(frame, channel) reduction of `World::observe`.
///
/// This is the function executed per thread on the device; the host
/// `render_window` drives it sequentially with the identical code.
#[inline]
pub fn render_sample(
    state: &FlatState,
    voices: &[FlatVoice],
    samples: &[i32],
    partials: &[Partial],
    g: usize,
) -> i32 {
    let (frame_off, ch) = state.frame_channel(g);
    let t = state.start_frame + i64::from(frame_off);
    let mut acc: i64 = 0;
    for v in voices {
        acc += i64::from(v.contribution_to_channel(samples, partials, t, ch));
    }
    sat_i32(acc)
}

/// Host-side sequential renderer over the exact device math (parity anchor
/// and CPU reference; identical result to the kernel by construction of
/// `render_sample`). Returns canonical interleaved i32 codes.
///
/// Host-only (`std`): device kernels drive `render_sample` per thread instead.
#[cfg(feature = "std")]
pub fn render_window(
    state: &FlatState,
    voices: &[FlatVoice],
    samples: &[i32],
    partials: &[Partial],
) -> Vec<i32> {
    debug_assert!(state.check_layout().is_some());
    let total = state.total_samples();
    let mut out = Vec::with_capacity(total);
    for g in 0..total {
        out.push(render_sample(state, voices, samples, partials, g));
    }
    out
}

/// Static layout anchors: the flat records are plain C structs so
/// host-built arena bytes and the device kernel agree on every compiler.
const _: () = {
    assert!(core::mem::size_of::<FlatVoice>() == 128);
    assert!(core::mem::size_of::<FlatState>() == 32);
    assert!(core::mem::size_of::<FlatEnvelope>() == 16);
    assert!(core::mem::size_of::<Partial>() == 8);
    assert!(core::mem::align_of::<FlatVoice>() == 8);
    assert!(core::mem::align_of::<Partial>() == 4);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_layout_splits_samples() {
        let s = flat_state(1000, 4, 2, 3, 0, 0);
        assert_eq!(s.total_samples(), 8);
        assert_eq!(s.frame_channel(0), (0, 0));
        assert_eq!(s.frame_channel(1), (0, 1));
        assert_eq!(s.frame_channel(7), (3, 1));
        assert!(s.check_layout().is_some());
    }

    #[test]
    fn state_layout_rejects_illegal_shapes() {
        let bad_ch = flat_state(0, 4, 0, 0, 0, 0);
        assert!(bad_ch.check_layout().is_none());
        let huge = flat_state(0, crate::limits::MAX_QUANTUM_FRAMES + 1, 2, 0, 0, 0);
        assert!(huge.check_layout().is_none());
    }

    #[test]
    fn class_semantics_are_consistent() {
        assert!(VoiceClass::Cycle.is_content() && !VoiceClass::Cycle.is_endless());
        assert!(VoiceClass::Literal.is_content() && !VoiceClass::Literal.is_endless());
        assert!(!VoiceClass::Noise.is_content() && VoiceClass::Noise.is_endless());
        assert_eq!(VoiceClass::Oscillator.code(), 3);
    }

    #[test]
    fn sentinel_never_collides_with_real_note_offs() {
        // Real note-offs live far below i64::MAX (frame coordinates bounded
        // by limits); the sentinel is unreachable as a real frame.
        assert!(NO_NOTE_OFF > crate::limits::MAX_OBJECT_FRAMES as i64 * 2);
    }
}
