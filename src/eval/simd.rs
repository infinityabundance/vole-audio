//! Host SIMD engine (Phase F).
//!
//! `SimdOracle` produces the *same* canonical interleaved observation as the
//! scalar oracle (`eval::scalar`) — bit for bit — via a planned, hoisted
//! renderer. Semantic authority stays with `sampler::voice::contribution_at`
//! and `World::observe`; this engine re-derives the identical per-frame
//! values with loop-invariant work hoisted out of the frame loop (per-voice
//! object handle, natural-end clamp, release window, envelope-segment
//! decomposition with the release level computed once instead of per frame).
//! It is the honest CPU baseline the GPU phases must beat.
//!
//! Structure:
//!
//! * `plan_window` — for one resolved voice and observation window, computes
//!   the exact frame span the voice can contribute to and the envelope-segment
//!   decomposition of that span (identical branch selection to
//!   `EnvelopeParams::level_at`).
//! * scalar floor — per-frame exact evaluation for every representation class
//!   (used on hosts without a vector ISA and as the reference for kernel
//!   differential tests).
//! * vector kernels (`eval::x86`, x86-64 only) — frame-blocked AVX2/AVX-512
//!   evaluation with identical per-lane integer math.
//!
//! Exactness contract: for every (world, store, window), the engine output
//! must equal `World::observe`. Per-(voice, frame) values must match exactly;
//! only the mix-accumulation order may differ (i64 addition is associative).
//!
//! Mixing-bound note: voices add `gain::contribution` values (voice-bus
//! saturated, `|c| <= 2^31 - 1`) into the i64 mixer, exactly like the scalar
//! world — the U1_SPEC overflow proof applies unchanged.

use crate::error::{Error, Result};
use crate::eval::backend::Isa;
use crate::object::ObjectStore;
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::mix::Mixer;
use crate::sampler::pan::{Route, pan_gains};
use crate::sampler::voice::ResolvedVoice;
use crate::sampler::world::World;

/// Host SIMD oracle: same observation surface as the scalar oracle.
#[derive(Debug, Clone)]
pub struct SimdOracle {
    pub world: World,
    /// Concrete ISA floor used for rendering (runtime-detected by default;
    /// tests may pin a floor).
    pub isa: Isa,
}

impl SimdOracle {
    pub fn new(world: World) -> SimdOracle {
        SimdOracle {
            world,
            isa: crate::eval::backend::detect_isa(),
        }
    }

    /// Observe `[start, start+frames)` in canonical interleaved i32 codes,
    /// bit-identical to the scalar oracle.
    pub fn observe(
        &self,
        store: &ObjectStore,
        start_frame: i64,
        frames: usize,
    ) -> Result<Vec<i32>> {
        render_engine(&self.world, store, start_frame, frames, self.isa)
    }

    /// Observe and return the canonical SHA-256 of the window.
    pub fn observe_hash(
        &self,
        store: &ObjectStore,
        start_frame: i64,
        frames: usize,
    ) -> Result<[u8; 32]> {
        let out = self.observe(store, start_frame, frames)?;
        Ok(crate::universe::observation::observation_sha256(&out))
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// One envelope segment of a voice's contribution span. Segments tile
/// `[lo, hi)` contiguously; `kind` selects the exact `Envelope::level_at`
/// branch that applies to every frame in the segment.
/// One envelope segment of a voice's contribution span. Segments tile
/// `[lo, hi)` contiguously; `kind` selects the exact `Envelope::level_at`
/// branch that applies to every frame in the segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segment {
    pub(crate) kind: SegKind,
    /// First frame (inclusive, media frame).
    pub(crate) lo: i64,
    /// Last frame (exclusive).
    pub(crate) hi: i64,
    /// Frame at which the segment's ramp index `k` is zero.
    pub(crate) anchor: i64,
}

/// Envelope envelope-level formula per segment (U1_SPEC §"Sampler transforms
/// — envelope" + `Envelope::level_at` freeze).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegKind {
    /// `k = t - anchor` in `[0, a)`: `level = round_half_up(UNITY * k / a)`.
    Attack { a: i64 },
    /// `k` in `[0, d)`: `level = UNITY - round_half_up(span * k / d)`,
    /// `span = UNITY - sustain`.
    Decay { d: i64, span: i64 },
    /// `level = sustain` (constant; zero sustains add 0 to the mix, which is
    /// identical to the oracle skipping those frames).
    Sustain { level: i32 },
    /// `k` in `[0, r)`: `level = l0 - min(l0, round_half_up(l0 * k / r))`.
    Release { r: i64, l0: i32 },
}

const ENV_UNITY_I64: i64 = 1i64 << 16;

/// `round_half_up(num / den)` for `num >= 0, den > 0` — the envelope's exact
/// rounding rule (mirror of `Envelope`'s private helper). Shared with the
/// vector kernels (`eval::x86`).
#[inline]
pub(crate) fn seg_round(num: i64, den: i64) -> i64 {
    debug_assert!(num >= 0 && den > 0);
    (num + den / 2) / den
}

/// Envelope level (Q16) at frame `t` for a segment kind + anchor — the single
/// segment-level formula shared by `Segment::level_at`, the vector kernels,
/// and kernel tail frames.
#[inline]
pub(crate) fn seg_level(kind: SegKind, anchor: i64, t: i64) -> i32 {
    let k = t - anchor;
    debug_assert!(k >= 0);
    match kind {
        SegKind::Attack { a } => seg_round(ENV_UNITY_I64 * k, a) as i32,
        SegKind::Decay { d, span } => (ENV_UNITY_I64 - seg_round(span * k, d)) as i32,
        SegKind::Sustain { level } => level,
        SegKind::Release { r, l0 } => {
            let drop = seg_round(i64::from(l0) * k, r).min(i64::from(l0));
            (i64::from(l0) - drop) as i32
        }
    }
}

impl Segment {
    /// Envelope level (Q16) at a frame `t in [lo, hi)` — must equal
    /// `EnvelopeParams::level_at(t_on, note_off, t)` for the voice this
    /// segment was planned from (property-tested).
    #[inline]
    fn level_at(&self, t: i64) -> i32 {
        seg_level(self.kind, self.anchor, t)
    }
}

/// Window plan for one voice: the (possibly empty) frame span the voice can
/// contribute to, decomposed into envelope segments.
struct VoiceWindow {
    /// First frame the voice can contribute in the window (inclusive).
    lo: i64,
    /// One past the last contributing frame (exclusive).
    hi: i64,
    segments: Vec<Segment>,
}

/// Plan the contribution span + envelope segments of one voice over the
/// observation window `[win_lo, win_hi)`.
///
/// The span is clipped so that every remaining frame is a frame the oracle
/// could contribute on:
///
/// * `t >= trigger_frame` (nothing before the trigger);
/// * `t < end_frame` for one-shot content (content reads stay in-domain);
/// * `t < note_off + release` for released voices (after that the envelope is
///   zero and the oracle's `silent_at` skips the frame — a zero level adds 0
///   to the mix, identical).
///
/// Zero-envelope frames that remain inside the span (attack `k = 0`, zero
/// sustain) are kept: their contribution is exactly 0 through the gain chain,
/// which is bit-identical to the oracle skipping them. Only content reads are
/// excluded — the `end_frame` clamp guarantees in-domain reads.
fn plan_window(
    env: &EnvelopeParams,
    t_on: i64,
    note_off: Option<i64>,
    end_frame: Option<i64>,
    win_lo: i64,
    win_hi: i64,
) -> Result<VoiceWindow> {
    let a = i64::from(env.attack_frames);
    let d = i64::from(env.decay_frames);
    let r = i64::from(env.release_frames);
    let s = i64::from(env.sustain_q16);

    let lo = win_lo.max(t_on);
    // Content end (one-shot) clamps reads; endless/periodic/looped have None.
    let mut hi = win_hi.min(end_frame.unwrap_or(i64::MAX));
    if let Some(off) = note_off {
        // Nothing audible from `off + release` on (`silent_at`), and a zero
        // release silences the voice at the note-off frame itself.
        let cap = if r > 0 { off + r } else { off };
        hi = hi.min(cap);
    }
    if lo >= hi {
        return Ok(VoiceWindow {
            lo,
            hi,
            segments: Vec::new(),
        });
    }

    // Candidate envelope boundaries strictly inside `(lo, hi)`.
    let decay_start = t_on + a; // == t_on when a == 0 (attack empty)
    let decay_end = decay_start + d; // == decay_start when d == 0
    let mut boundaries: Vec<i64> = Vec::with_capacity(5);
    for b in [decay_start, decay_end] {
        if b > lo && b < hi {
            boundaries.push(b);
        }
    }
    if let Some(off) = note_off
        && r > 0
    {
        if off > lo && off < hi {
            boundaries.push(off);
        }
        let rel_end = off + r;
        if rel_end > lo && rel_end < hi {
            boundaries.push(rel_end);
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    // Classify every frame in `[lo, hi)`; the kind changes only at a boundary.
    let span = ENV_UNITY_I64 - s; // decay drop span
    let classify = |t: i64| -> SegKind {
        if let Some(off) = note_off {
            if t >= off {
                debug_assert!(r > 0, "release frames excluded when r == 0");
                let l0 = env.level_at(t_on, None, off);
                SegKind::Release { r, l0 }
            } else if t < decay_start {
                debug_assert!(a > 0, "no attack frames when a == 0");
                SegKind::Attack { a }
            } else if t < decay_end {
                debug_assert!(d > 0, "no decay frames when d == 0");
                SegKind::Decay { d, span }
            } else {
                SegKind::Sustain {
                    level: env.sustain_q16,
                }
            }
        } else if t < decay_start {
            debug_assert!(a > 0, "no attack frames when a == 0");
            SegKind::Attack { a }
        } else if t < decay_end {
            debug_assert!(d > 0, "no decay frames when d == 0");
            SegKind::Decay { d, span }
        } else {
            SegKind::Sustain {
                level: env.sustain_q16,
            }
        }
    };

    // Walk boundaries emitting maximal segments.
    let mut segments: Vec<Segment> = Vec::with_capacity(5);
    let mut prev_kind: Option<SegKind> = None;
    // Iterate over `[lo, hi)` split points: every boundary plus `hi`.
    let mut points: Vec<i64> = Vec::with_capacity(boundaries.len() + 2);
    points.push(lo);
    points.extend_from_slice(&boundaries);
    points.push(hi);
    for w in points.windows(2) {
        let (s, e) = (w[0], w[1]);
        if e <= s {
            continue;
        }
        let kind = classify(s);
        let anchor = match kind {
            SegKind::Attack { .. } => t_on,
            SegKind::Decay { .. } => decay_start,
            SegKind::Sustain { .. } => s,
            SegKind::Release { .. } => note_off.expect("release classified only with a note-off"),
        };
        if prev_kind.as_ref() == Some(&kind) {
            let last = segments.last_mut().expect("segments nonempty");
            last.hi = e;
        } else {
            segments.push(Segment {
                kind,
                lo: s,
                hi: e,
                anchor,
            });
        }
        prev_kind = Some(kind);
    }

    Ok(VoiceWindow { lo, hi, segments })
}

/// Render one voice's planned window into the mixer using the exact scalar
/// per-frame semantics (scalar floor; also the differential reference for the
/// vector kernels).
///
/// `t_frame` is the first window frame (`start_frame` of the observation);
/// every mixer index is `t - start_frame`.
fn render_window_scalar(
    voice: &ResolvedVoice,
    store: &ObjectStore,
    window: &VoiceWindow,
    start_frame: i64,
    mixer: &mut Mixer,
) {
    let spec = &voice.spec;
    let obj = match store.get(voice.target_id) {
        Ok(o) => o,
        // The store was validated before rendering; resolution succeeded for
        // every voice (`resolve_all`), so the target exists.
        Err(_) => return,
    };
    let obj_channels = usize::from(obj.descriptor.layout.count());
    let obj_ch = usize::from(spec.object_channel);
    let data = &obj.data;

    for seg in &window.segments {
        let mut t = seg.lo;
        while t < seg.hi {
            let env = seg.level_at(t);
            let obs_of = |ch: usize| voice.observe_channel(data, obj_channels, ch, t);
            match spec.route {
                Route::Mono(out_ch) => {
                    let m = crate::sampler::gain::channel_multiplier(
                        crate::sampler::gain::env_gain_multiplier(env, spec.gain_q16),
                        1 << 16,
                    );
                    let c = crate::sampler::gain::contribution(obs_of(obj_ch), m);
                    mixer.add((t - start_frame) as usize, out_ch, c);
                }
                Route::StereoPair(base) => {
                    let (gl, gr) = pan_gains(spec.pan_q16);
                    let l = obs_of(obj_ch);
                    let r = obs_of(obj_ch + 1);
                    let e = crate::sampler::gain::env_gain_multiplier(env, spec.gain_q16);
                    let ml = crate::sampler::gain::channel_multiplier(e, gl);
                    let mr = crate::sampler::gain::channel_multiplier(e, gr);
                    let fi = (t - start_frame) as usize;
                    mixer.add(fi, base, crate::sampler::gain::contribution(l, ml));
                    mixer.add(fi, base + 1, crate::sampler::gain::contribution(r, mr));
                }
            }
            t += 1;
        }
    }
}

/// Render `[start_frame, start_frame+frames)` with the requested ISA floor.
pub(crate) fn render_engine(
    world: &World,
    store: &ObjectStore,
    start_frame: i64,
    frames: usize,
    isa: Isa,
) -> Result<Vec<i32>> {
    if frames > crate::limits::MAX_QUANTUM_FRAMES as usize {
        return Err(Error::limit("observation window exceeds quantum ceiling"));
    }
    let resolved = world.resolve_all(store)?;
    let channels = usize::from(world.output_channels);
    let mut mixer = Mixer::new(channels, frames);
    let win_hi = start_frame + frames as i64;

    for voice in &resolved {
        let spec = &voice.spec;
        if spec.trigger_frame >= win_hi {
            continue;
        }
        let window = plan_window(
            &spec.envelope,
            spec.trigger_frame,
            spec.note_off,
            voice.end_frame(),
            start_frame,
            win_hi,
        )?;
        if window.lo >= window.hi {
            continue;
        }
        #[cfg(target_arch = "x86_64")]
        if isa != Isa::Scalar
            && render_window_vector(voice, store, &window, start_frame, isa, &mut mixer)
        {
            continue;
        }
        render_window_scalar(voice, store, &window, start_frame, &mut mixer);
    }

    let mut out = vec![0i32; frames * channels];
    mixer.finalize_interleaved(&mut out);
    Ok(out)
}

/// Add one scalar oracle contribution (tail frames of vector-class voices).
#[cfg(target_arch = "x86_64")]
fn add_tail_contribution(
    store: &ObjectStore,
    voice: &ResolvedVoice,
    mixer: &mut Mixer,
    start_frame: i64,
    t: i64,
) {
    let c = voice.contribution_at(store, t);
    let fi = (t - start_frame) as usize;
    if let Some((ch, v)) = c.a {
        mixer.add(fi, ch, v);
    }
    if let Some((ch, v)) = c.b {
        mixer.add(fi, ch, v);
    }
}

/// Vector block width for an ISA floor.
#[cfg(target_arch = "x86_64")]
fn block_width(isa: Isa) -> usize {
    match isa {
        Isa::Avx512 => crate::eval::x86::kavx512::BLOCK_W,
        Isa::Avx2 => crate::eval::x86::kavx2::BLOCK_W,
        Isa::Scalar => 0,
    }
}

/// Render a content-class voice window through the vector kernels (whole
/// blocks) and the scalar oracle (tail frames).
#[cfg(target_arch = "x86_64")]
fn render_content_window(
    voice: &ResolvedVoice,
    store: &ObjectStore,
    window: &VoiceWindow,
    start_frame: i64,
    isa: Isa,
    mixer: &mut Mixer,
    ctx: &crate::eval::x86::ContentCtx,
) {
    use crate::eval::x86;
    let w = block_width(isa) as i64;
    for seg in &window.segments {
        let mut t = seg.lo;
        while t + w <= seg.hi {
            unsafe {
                match isa {
                    Isa::Avx512 => x86::kavx512::content_block(
                        ctx,
                        seg.kind,
                        seg.anchor,
                        t,
                        start_frame,
                        mixer,
                    ),
                    Isa::Avx2 => {
                        x86::kavx2::content_block(ctx, seg.kind, seg.anchor, t, start_frame, mixer)
                    }
                    Isa::Scalar => unreachable!(),
                }
            }
            t += w;
        }
        for f in t..seg.hi {
            add_tail_contribution(store, voice, mixer, start_frame, f);
        }
    }
}

/// Render an endless-class voice window through the vector kernels (whole
/// blocks) and the scalar oracle (tail frames).
#[cfg(target_arch = "x86_64")]
fn render_endless_window(
    voice: &ResolvedVoice,
    store: &ObjectStore,
    window: &VoiceWindow,
    start_frame: i64,
    isa: Isa,
    mixer: &mut Mixer,
    ctx: &crate::eval::x86::EndlessCtx,
) {
    use crate::eval::x86;
    let w = block_width(isa) as i64;
    for seg in &window.segments {
        let mut t = seg.lo;
        while t + w <= seg.hi {
            unsafe {
                match isa {
                    Isa::Avx512 => x86::kavx512::endless_block(
                        ctx,
                        seg.kind,
                        seg.anchor,
                        t,
                        start_frame,
                        mixer,
                    ),
                    Isa::Avx2 => {
                        x86::kavx2::endless_block(ctx, seg.kind, seg.anchor, t, start_frame, mixer)
                    }
                    Isa::Scalar => unreachable!(),
                }
            }
            t += w;
        }
        for f in t..seg.hi {
            add_tail_contribution(store, voice, mixer, start_frame, f);
        }
    }
}

/// Attempt a vector-kernel render of one voice's planned window. Returns true
/// when the voice was rendered (any class the kernels cover); false means the
/// caller must fall back to the scalar floor.
#[cfg(target_arch = "x86_64")]
fn render_window_vector(
    voice: &ResolvedVoice,
    store: &ObjectStore,
    window: &VoiceWindow,
    start_frame: i64,
    isa: Isa,
    mixer: &mut Mixer,
) -> bool {
    use crate::eval::x86::{EndlessCtx, EndlessKind};
    use crate::object::ObjectData;

    if isa == Isa::Scalar {
        return false;
    }
    let spec = &voice.spec;
    let obj = match store.get(voice.target_id) {
        Ok(o) => o,
        Err(_) => return false,
    };
    let obj_channels = usize::from(obj.descriptor.layout.count());
    let obj_ch = usize::from(spec.object_channel);
    let (gl, gr) = pan_gains(spec.pan_q16);

    match &obj.data {
        ObjectData::Literal(l) => {
            let ctx = content_ctx(&l.samples, voice, obj, obj_channels, obj_ch, spec, gl, gr);
            render_content_window(voice, store, window, start_frame, isa, mixer, &ctx);
            true
        }
        ObjectData::Wavetable(c) | ObjectData::SingleCycle(c) | ObjectData::ExactRepeat(c) => {
            let ctx = content_ctx(&c.samples, voice, obj, obj_channels, obj_ch, spec, gl, gr);
            render_content_window(voice, store, window, start_frame, isa, mixer, &ctx);
            true
        }
        ObjectData::Silence
        | ObjectData::Constant(_)
        | ObjectData::Noise(_)
        | ObjectData::Oscillator(_) => {
            let Route::Mono(out_ch) = spec.route else {
                return false; // endless objects are mono; stereo routes stay scalar
            };
            let kind = match &obj.data {
                ObjectData::Silence => EndlessKind::Silence,
                ObjectData::Constant(c) => EndlessKind::Constant(c.level),
                ObjectData::Noise(n) => EndlessKind::Noise { seed: n.seed },
                ObjectData::Oscillator(o) => EndlessKind::Oscillator {
                    incr: voice.osc_incr,
                    amp_q16: o.amp_q16,
                },
                _ => unreachable!(),
            };
            let ctx = EndlessCtx {
                kind,
                t_on: spec.trigger_frame,
                gain_q16: spec.gain_q16,
                out_ch,
            };
            render_endless_window(voice, store, window, start_frame, isa, mixer, &ctx);
            true
        }
        ObjectData::PartialBank(_)
        | ObjectData::PredictorResidual(_)
        | ObjectData::Referenced(_) => false,
    }
}

/// Build the content kernel context for a content object voice.
///
/// The argument count reflects the fixed content kernel parameter set (one
/// per frozen sampler transform); the parameters are all resolved voice/object
/// facts, not speculative extension points.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn content_ctx<'a>(
    samples: &'a [i32],
    voice: &ResolvedVoice,
    obj: &'a crate::object::SampleObject,
    obj_channels: usize,
    obj_ch: usize,
    spec: &crate::sampler::voice::VoiceSpec,
    gl: i32,
    gr: i32,
) -> crate::eval::x86::ContentCtx<'a> {
    use crate::eval::x86::ContentCtx;
    let extent = obj.descriptor.extent_frames;
    let is_cycle = obj.data.is_periodic();
    let (has_region, region_a, region_b) = if is_cycle {
        (true, 0u64, extent)
    } else {
        match voice.loop_region {
            Some(l) => (true, l.start_frame, l.end_frame),
            None => (false, 0, extent),
        }
    };
    // Single-correction wrap exactness bound for the widest floor (W = 8):
    // |rate| * (W - 1) <= len  (Q24).
    let len_w = u128::from((region_b - region_a) << crate::limits::FIXED_Q);
    let wrap_fast =
        region_b > region_a && u128::from(voice.eff_rate_q24.unsigned_abs()) * 7 <= len_w;
    let stereo_base = match spec.route {
        Route::StereoPair(base) => Some(base),
        Route::Mono(_) => None,
    };
    ContentCtx {
        samples,
        obj_channels,
        ch_a: obj_ch,
        ch_b: obj_ch + 1,
        read_stereo: stereo_base.is_some(),
        is_cycle,
        has_region,
        region_a,
        region_b,
        extent_frames: extent,
        start_pos_q24: spec.start_pos_q24,
        rate_q24: voice.eff_rate_q24,
        t_on: spec.trigger_frame,
        gain_q16: spec.gain_q16,
        wrap_fast,
        linear: matches!(spec.interp, crate::sampler::voice::Interp::Linear),
        mono_ch: match spec.route {
            Route::Mono(ch) => Some(ch),
            Route::StereoPair(_) => None,
        },
        stereo_base,
        gl,
        gr,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn segment_levels_equal_envelope_level_at() {
        // Every segment formula must reproduce `Envelope::level_at` for every
        // frame it covers, across adversarial parameters and note-off points.
        use crate::sampler::envelope::EnvelopeParams;
        let unity = 1 << 16;
        for &(attack, decay, sustain, release) in &[
            (0u32, 0u32, unity, 0u32),
            (1, 0, unity, 0),
            (0, 1, unity / 2, 0),
            (64, 256, unity / 2, 512),
            (3, 7, 0, 9),
            (1 << 20, 1 << 20, unity, 1 << 20),
            (17, 0, unity / 3, 5),
            (0, 0, 0, 0),
        ] {
            let env = EnvelopeParams::new(attack, decay, sustain, release).unwrap();
            let t_on = 123_456;
            for &off in &[None, Some(t_on), Some(t_on + 3), Some(t_on + 100_000)] {
                let (win_lo, win_hi) = (t_on, t_on + 200_000);
                let plan = plan_window(&env, t_on, off, None, win_lo, win_hi).unwrap();
                // The plan covers `[lo, hi)` (hi may be clamped by the note-off
                // release window); segments tile it contiguously.
                let mut cursor = plan.lo;
                for seg in &plan.segments {
                    assert_eq!(seg.lo, cursor, "segments must tile contiguously");
                    assert!(seg.hi > seg.lo);
                    cursor = seg.hi;
                }
                assert_eq!(cursor, plan.hi);
                for seg in &plan.segments {
                    for t in seg.lo..seg.hi {
                        let expected = env.level_at(t_on, off, t);
                        assert_eq!(
                            seg.level_at(t),
                            expected,
                            "segment level != level_at at t={t} (params {attack},{decay},{sustain},{release}, off {off:?})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn segment_levels_equal_envelope_for_window_slices() {
        // Same contract, but with observation windows that slice mid-segment
        // (the planner only sees `[win_lo, win_hi)`).
        use crate::sampler::envelope::EnvelopeParams;
        let env = EnvelopeParams::new(48, 1920, (1 << 16) / 2, 2880).unwrap();
        let t_on = 5000;
        let off = Some(50_000);
        for (win_lo, win_hi) in [
            (t_on, t_on + 10),
            (t_on + 5, t_on + 7),
            (t_on + 40, t_on + 60), // attack/decay boundary
            (t_on + 1900, t_on + 2100),
            (44_000, 56_000), // note-off in the middle of the window
            (55_000, 200_000),
            (60_000, 200_000), // fully released: empty plan
            (0, 4900),         // entirely before the trigger: empty plan
        ] {
            let plan = plan_window(&env, t_on, off, None, win_lo, win_hi).unwrap();
            if win_hi <= win_lo.max(t_on) {
                assert!(plan.segments.is_empty());
                continue;
            }
            let mut cursor = win_lo.max(t_on);
            for seg in &plan.segments {
                assert_eq!(seg.lo, cursor);
                cursor = seg.hi;
            }
            // Fully-released windows have nothing after `off + release`.
            let cap = 50_000 + 2880;
            if win_lo >= cap {
                assert!(plan.segments.is_empty());
            }
            for seg in &plan.segments {
                for t in seg.lo..seg.hi {
                    assert_eq!(seg.level_at(t), env.level_at(t_on, off, t));
                }
            }
        }
    }

    #[test]
    fn one_shot_content_end_clamps_segments() {
        // A one-shot literal ends at its natural end frame; the plan must not
        // contain frames at/after it (content reads stay in-domain).
        use crate::object::descriptor::{ObjectDescriptor, Representation};
        use crate::object::{Literal, ObjectData, ObjectStore};
        use crate::sampler::envelope::EnvelopeParams;
        use crate::sampler::voice::{Interp, LoopMode, VoiceSpec};
        let mut store = ObjectStore::new();
        let d = ObjectDescriptor::new(
            Representation::Literal,
            100,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap();
        let samples: Vec<i32> = (0..100).map(|i| i << 20).collect();
        let id = store
            .insert(
                d.clone(),
                ObjectData::Literal(Literal::new(&d, samples).unwrap()),
            )
            .unwrap();
        let spec = VoiceSpec {
            object: id,
            trigger_frame: 10,
            note_off: None,
            start_pos_q24: 0,
            rate_q24: 1 << 24,
            object_channel: 0,
            route: Route::Mono(0),
            gain_q16: 1 << 16,
            pan_q16: 0,
            envelope: EnvelopeParams::new(0, 0, 1 << 16, 0).unwrap(),
            loop_mode: LoopMode::Off,
            interp: Interp::Linear,
        };
        let voice = ResolvedVoice::resolve(&store, spec, 48_000).unwrap();
        let end = voice.end_frame().unwrap();
        assert_eq!(end, 110); // trigger 10 + extent 100 at unity rate
        let plan =
            plan_window(&voice.spec.envelope, 10, None, voice.end_frame(), 0, 10_000).unwrap();
        assert_eq!(plan.hi, 110);
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].lo, 10);
        assert_eq!(plan.segments[0].hi, 110);
    }

    // -------------------------------------------------------------------
    // Differential parity: SimdOracle (scalar floor) == ScalarOracle.
    // -------------------------------------------------------------------

    /// Two engines over the same world: the scalar oracle (authority) and the
    /// SIMD engine pinned to its scalar floor. All parity tests compare
    /// buffers element-wise plus canonical hashes.
    fn floor_pair(world: &World) -> (crate::eval::ScalarOracle, SimdOracle) {
        let scalar = crate::eval::ScalarOracle::new(world.clone());
        let simd = SimdOracle {
            world: world.clone(),
            isa: Isa::Scalar,
        };
        (scalar, simd)
    }

    fn assert_window_parity(
        store: &ObjectStore,
        scalar: &crate::eval::ScalarOracle,
        simd: &SimdOracle,
        start: i64,
        frames: usize,
    ) {
        let a = scalar.observe(store, start, frames).unwrap();
        let b = simd.observe(store, start, frames).unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(
            a,
            b,
            "simd != scalar at window [{start}, {})",
            start + frames as i64
        );
        assert_eq!(
            scalar.observe_hash(store, start, frames).unwrap(),
            simd.observe_hash(store, start, frames).unwrap()
        );
    }

    #[test]
    fn semantic_fixture_parity() {
        let (store, events) = crate::courts::semantic::semantic_court_fixture();
        let world = World::new(
            crate::courts::semantic::RATE_HZ,
            crate::courts::semantic::CHANNELS,
            events,
        )
        .unwrap();
        let (scalar, simd) = floor_pair(&world);
        for (start, frames) in [
            (0i64, 2400usize),
            (0, 1),
            (1, 2399),
            (700, 900),
            (1600, 800),
            (50, 64),
            (2390, 10),
        ] {
            assert_window_parity(&store, &scalar, &simd, start, frames);
        }
    }

    #[test]
    fn authored_fixture_parity() {
        let (store, events) = crate::courts::authored::authored_court_fixture();
        let world = World::new(crate::courts::authored::RATE_HZ, 1, events).unwrap();
        let (scalar, simd) = floor_pair(&world);
        for (start, frames) in [
            (0i64, 4000usize),
            (0, 1),
            (3, 5000),
            (400, 3600),
            (1600, 2400),
        ] {
            assert_window_parity(&store, &scalar, &simd, start, frames);
        }
    }

    #[test]
    fn fixture_hashes_are_reproduced_by_the_engine() {
        // The frozen court reference hashes are produced unchanged through
        // the SIMD engine — the cross-backend exactness invariant.
        let (store, events) = crate::courts::semantic::semantic_court_fixture();
        let world = World::new(
            crate::courts::semantic::RATE_HZ,
            crate::courts::semantic::CHANNELS,
            events,
        )
        .unwrap();
        let simd = SimdOracle::new(world);
        let h = simd.observe_hash(&store, 0, 2400).unwrap();
        assert_eq!(
            crate::hash::sha256::hex(&h),
            crate::courts::semantic::SEMANTIC_COURT_REFERENCE_SHA256
        );

        let (store, events) = crate::courts::authored::authored_court_fixture();
        let world = World::new(crate::courts::authored::RATE_HZ, 1, events).unwrap();
        let simd = SimdOracle::new(world);
        let h = simd.observe_hash(&store, 0, 4000).unwrap();
        assert_eq!(
            crate::hash::sha256::hex(&h),
            crate::courts::authored::AUTHORED_COURT_REFERENCE_SHA256
        );
    }

    // -------------------------------------------------------------------
    // Randomized differential battery.
    //
    // The corpus/voice generators now live in `crate::eval::battery` (shared
    // with backend::flatten parity and court cuda) — the SIMD floors consume
    // the same distribution here.
    use crate::eval::battery::{corpus, random_voice};

    #[test]
    fn random_battery_floor_parity() {
        use crate::sampler::scheduler::TimelineEvent;
        let seeds = 5000u64;
        let mut compared = 0u64;
        let mut audible = 0u64;
        for seed in 0..seeds {
            let (store, pool) = corpus(seed);
            let mut rng = crate::universe::prng::XoShiro256::from_seed(seed ^ 0xBEEF);
            let next32 = |rng: &mut crate::universe::prng::XoShiro256| rng.next_u64() as u32;
            let out_channels: u8 = 2;
            let n_voices = 1 + (next32(&mut rng) % 4);
            let mut events = Vec::new();
            for i in 0..n_voices {
                let object = pool[(i as usize) % pool.len()];
                let mut spec = random_voice(object, &store, &mut rng, out_channels);
                if next32(&mut rng) % 2 == 0 {
                    spec.note_off = Some(spec.trigger_frame + (next32(&mut rng) % 1200) as i64);
                }
                events.push(TimelineEvent::VoiceOn(spec));
            }
            let world = match World::new(48_000, out_channels, events) {
                Ok(w) => w,
                Err(_) => continue,
            };
            let (scalar, simd) = floor_pair(&world);
            let start = (next32(&mut rng) % 2000) as i64;
            let frames = 1 + (next32(&mut rng) % 700) as usize;
            let a = match scalar.observe(&store, start, frames) {
                Ok(a) => a,
                Err(_) => continue, // invalid draw (e.g. bad loop region)
            };
            let b = match simd.observe(&store, start, frames) {
                Ok(b) => b,
                Err(e) => panic!("simd failed on world the scalar rendered (seed {seed}): {e}"),
            };
            assert_eq!(
                a,
                b,
                "simd floor != scalar at seed {seed}, window [{start}, {})",
                start + frames as i64
            );
            if a.iter().any(|&x| x != 0) {
                audible += 1;
            }
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

    /// Vector ISA floors available at runtime on this host.
    fn available_vector_isas() -> Vec<Isa> {
        let mut v = Vec::new();
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

    #[test]
    fn vector_floors_reproduce_frozen_fixture_hashes() {
        // The frozen court reference hashes through every available vector
        // floor — the cross-backend exactness invariant, kernel paths
        // included (long windows force whole kernel blocks).
        for isa in available_vector_isas() {
            let (store, events) = crate::courts::semantic::semantic_court_fixture();
            let world = World::new(
                crate::courts::semantic::RATE_HZ,
                crate::courts::semantic::CHANNELS,
                events,
            )
            .unwrap();
            let simd = SimdOracle { world, isa };
            let h = simd.observe_hash(&store, 0, 2400).unwrap();
            assert_eq!(
                crate::hash::sha256::hex(&h),
                crate::courts::semantic::SEMANTIC_COURT_REFERENCE_SHA256,
                "semantic fixture hash broken on {isa:?}"
            );

            let (store, events) = crate::courts::authored::authored_court_fixture();
            let world = World::new(crate::courts::authored::RATE_HZ, 1, events).unwrap();
            let simd = SimdOracle { world, isa };
            let h = simd.observe_hash(&store, 0, 4000).unwrap();
            assert_eq!(
                crate::hash::sha256::hex(&h),
                crate::courts::authored::AUTHORED_COURT_REFERENCE_SHA256,
                "authored fixture hash broken on {isa:?}"
            );
        }
    }

    #[test]
    fn vector_floors_parity_on_fixture_windows() {
        for isa in available_vector_isas() {
            let (store, events) = crate::courts::semantic::semantic_court_fixture();
            let world = World::new(
                crate::courts::semantic::RATE_HZ,
                crate::courts::semantic::CHANNELS,
                events,
            )
            .unwrap();
            let (scalar, simd) = (
                crate::eval::ScalarOracle::new(world.clone()),
                SimdOracle { world, isa },
            );
            for (start, frames) in [(0i64, 2400usize), (1, 2399), (700, 900), (2390, 10), (0, 3)] {
                assert_window_parity(&store, &scalar, &simd, start, frames);
            }

            let (store, events) = crate::courts::authored::authored_court_fixture();
            let world = World::new(crate::courts::authored::RATE_HZ, 1, events).unwrap();
            let (scalar, simd) = (
                crate::eval::ScalarOracle::new(world.clone()),
                SimdOracle { world, isa },
            );
            for (start, frames) in [(0i64, 4000usize), (0, 33), (1999, 2001), (3000, 1000)] {
                assert_window_parity(&store, &scalar, &simd, start, frames);
            }
        }
    }

    #[test]
    fn random_battery_vector_parity() {
        // Every available vector floor vs the scalar oracle over random
        // worlds; windows are long enough to force whole kernel blocks.
        let isas = available_vector_isas();
        if isas.is_empty() {
            return;
        }
        use crate::sampler::scheduler::TimelineEvent;
        let seeds = 2000u64;
        let mut compared = [0u64; 8];
        for seed in 0..seeds {
            let (store, pool) = corpus(seed);
            let mut rng = crate::universe::prng::XoShiro256::from_seed(seed ^ 0xFEED);
            let next32 = |rng: &mut crate::universe::prng::XoShiro256| rng.next_u64() as u32;
            let out_channels: u8 = 2;
            let n_voices = 1 + (next32(&mut rng) % 3);
            let mut events = Vec::new();
            for i in 0..n_voices {
                let object = pool[(i as usize) % pool.len()];
                let mut spec = random_voice(object, &store, &mut rng, out_channels);
                if next32(&mut rng) % 2 == 0 {
                    spec.note_off = Some(spec.trigger_frame + (next32(&mut rng) % 1200) as i64);
                }
                events.push(TimelineEvent::VoiceOn(spec));
            }
            let world = match World::new(48_000, out_channels, events) {
                Ok(w) => w,
                Err(_) => continue,
            };
            let start = (next32(&mut rng) % 1000) as i64;
            let frames = 256 + (next32(&mut rng) % 700) as usize;
            let scalar = crate::eval::ScalarOracle::new(world.clone());
            let a = match scalar.observe(&store, start, frames) {
                Ok(a) => a,
                Err(_) => continue,
            };
            for (k, &isa) in isas.iter().enumerate() {
                let simd = SimdOracle {
                    world: world.clone(),
                    isa,
                };
                let b = match simd.observe(&store, start, frames) {
                    Ok(b) => b,
                    Err(e) => panic!("vector floor {isa:?} failed (seed {seed}): {e}"),
                };
                assert_eq!(
                    a,
                    b,
                    "vector floor {isa:?} != scalar at seed {seed}, window [{start}, {})",
                    start + frames as i64
                );
                compared[k] += 1;
            }
        }
        for (k, &isa) in isas.iter().enumerate() {
            assert!(
                compared[k] > seeds / 3,
                "battery compared too few worlds for {isa:?} ({})",
                compared[k]
            );
        }
    }
}
