//! Phase 7C.2-E: fractional long-term prediction and an interleaved-track
//! algebraic innovation — the `voice.exp2` excitation core.
//!
//! `exp1`'s CELP ([`crate::voice::celp`]) keeps a **integer** single-tap
//! adaptive codebook and a **free combinatorial** pulse codebook (any
//! `count`-subset of the 80 subframe positions). It is the frozen control and
//! this module does not touch it.
//!
//! Two measured weaknesses motivate a second, exp2-only core:
//!
//! 1. **Integer pitch resolution.** At 16 kHz one lag step is 62.5 µs. For a
//!    200 Hz voice the period is 80 samples, so a one-sample error is a 1.25 %
//!    period error — a large phase drift over a 5 ms subframe. The measured
//!    effect is that the long-term predictor adds only ~0.66 dB on real speech
//!    (`voice_bench pg`). This module resolves the lag at **quarter-sample**
//!    resolution by interpolating the reconstructed excitation with a fixed
//!    4-tap Lagrange filter.
//! 2. **A free pulse codebook does not scale.** A single gain multiplies unit
//!    pulses, so adding a pulse always adds energy: with more pulses the
//!    excitation cannot refine, it can only coarsen. `voice_bench cmp` measures
//!    the consequence directly — CELP reaches 10.66 dB with one pulse per
//!    subframe and *falls* to 3.83 dB with two. Interleaved tracks (4 tracks of
//!    20 positions, `k` pulses per track) spread the innovation by construction,
//!    carry the pulse count implicitly in the codebook class instead of in a
//!    transmitted field, and fall in cost per pulse as `k` rises.
//!
//! Wire layout is owned by [`crate::voice::exp2`]; this module is the mechanism.

use crate::error::{Error, Result};
use crate::voice::celp;
use crate::voice::predict as vp;

/// Samples per excitation subframe (5 ms at 16 kHz).
pub const SUB_LEN: usize = celp::SUB_LEN;
/// Interleaved tracks in the algebraic innovation.
pub const TRACKS: usize = 4;
/// Positions per track.
pub const TRACK_POS: usize = SUB_LEN / TRACKS;
/// Fractional bits of the lag: quarter-sample resolution.
pub const FRAC_BITS: u8 = 2;
/// Bits of the fractional lag anchor.
pub const LAG_Q_BITS: u8 = celp::LAG_BITS + FRAC_BITS;
/// Largest pulses carried per track.
pub const MAX_PER_TRACK: usize = 2;
/// Smallest fractional lag code (quarter samples).
pub const LAG_Q_MIN: i32 = vp::MIN_LAG as i32 * (1 << FRAC_BITS);
/// Largest fractional lag code (quarter samples).
pub const LAG_Q_MAX: i32 = vp::MAX_LAG as i32 * (1 << FRAC_BITS);
/// Pitch-gain levels (shared with the control).
pub const PITCH_GAIN_LEVELS: i32 = celp::PITCH_GAIN_LEVELS;
/// Innovation-gain levels (shared with the control).
pub const GAIN_LEVELS: i32 = celp::GAIN_LEVELS;
/// The pitch-gain codes the adaptive search tries. Four gains keep the
/// always-tried analysis inside the encode deadline; the ladder is the one the
/// measured cost curve could afford, not a free choice.
const PITCH_GAIN_TRIALS: [i32; 4] = [0, 10, 20, 31];
/// Positions the focused search keeps per track, ranked by `|d|`.
const FOCUS: usize = 6;
/// Integer lags examined either side of the running centre.
const NEAR: i32 = 4;
/// Stride of the coarse integer lag sweep, in samples.
const COARSE_STRIDE: i32 = 32;
/// Coordinate-ascent sweeps over the tracks.
const SWEEPS: usize = 2;

/// 4-tap Lagrange interpolation at quarter-sample offsets.
///
/// `INTERP[f][m]` multiplies `u(n − i + 1 − m)` for an integer part `i` and
/// fraction `f/4`, and `INTERP[0] = [0, 1, 0, 0]` makes the integer case exact.
/// The coefficients solve the cubic through the samples at local offsets
/// `−1, 0, 1, 2`, evaluated at `x = −f/4`.
const INTERP: [[f64; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [0.117_187_5, 1.054_687_5, -0.210_937_5, 0.039_062_5],
    [0.312_5, 0.937_5, -0.312_5, 0.062_5],
    [0.601_562_5, 0.601_562_5, -0.257_812_5, 0.054_687_5],
];

/// Binomial coefficient, saturating. Bounded by `C(20, 2)` here.
fn comb(n: u32, k: u32) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc = 1u64;
    for i in 0..k {
        acc = acc.saturating_mul(u64::from(n - i)) / u64::from(i + 1);
    }
    acc
}

/// Bits one track needs for `k` signed pulses: `ceil(log2(C(20,k)·2^k))`.
pub fn track_bits(k: usize) -> u8 {
    let combos = comb(TRACK_POS as u32, k as u32).saturating_mul(1u64 << k);
    if combos <= 1 {
        0
    } else {
        (combos - 1).ilog2() as u8 + 1
    }
}

/// Bits a whole subframe's innovation needs: four tracks of `k` pulses.
pub fn innovation_bits(per_track: usize) -> usize {
    TRACKS * usize::from(track_bits(per_track))
}

/// Interpolated excitation `u(n − τ)` for a fractional lag `lag_q = 4i + f`.
///
/// Reads only the bounded excitation ring, so it is exactly reproducible on both
/// sides and depends on nothing outside the packet's own history.
fn adaptive_sample(state: &vp::VoiceState, lag_q: i32) -> f64 {
    let i = (lag_q >> FRAC_BITS) as usize;
    let f = (lag_q & ((1 << FRAC_BITS) - 1)) as usize;
    let c = &INTERP[f];
    c[0] * state.excitation(i)
        + c[1] * state.excitation(i.saturating_sub(1))
        + c[2] * state.excitation(i.saturating_sub(2))
        + c[3] * state.excitation(i.saturating_sub(3))
}

/// The `exp2` synthesis loop: the shared short-term recursion with a
/// **fractional** long-term predictor.
///
/// Identical in structure to [`vp::synthesize_celp`] except that the adaptive
/// term is the interpolated excitation at a quarter-sample lag. It is a separate
/// function on purpose: the control profile's loop must stay bit-exact.
pub fn synthesize(
    state: &mut vp::VoiceState,
    k_q: &[i32],
    width: u8,
    sub_len: usize,
    ltp: &[(i32, i32)],
    fixed: &[f64],
) -> Vec<f64> {
    let w = vp::weights_of(k_q, width);
    synthesize_w(state, &w, sub_len, ltp, fixed)
}

/// [`synthesize`] with the short-term weights supplied by the caller.
///
/// A search evaluates hundreds of candidates against one spectrum, so building
/// the weights inside the loop is the difference between a few hundred
/// microseconds and several milliseconds per frame. This is the entry point the
/// analysis uses.
pub fn synthesize_w(
    state: &mut vp::VoiceState,
    w: &[f64],
    sub_len: usize,
    ltp: &[(i32, i32)],
    fixed: &[f64],
) -> Vec<f64> {
    let sub_len = sub_len.max(1);
    let mut out = Vec::with_capacity(fixed.len());
    for (n, &e) in fixed.iter().enumerate() {
        let (lag_q, pg) = ltp.get(n / sub_len).copied().unwrap_or((0, 0));
        let gain = celp::pitch_gain_of(pg);
        let mut st = 0.0f64;
        for (j, &wj) in w.iter().enumerate() {
            st += wj * state.at(j);
        }
        let adaptive = if lag_q > 0 {
            gain * adaptive_sample(state, lag_q)
        } else {
            0.0
        };
        let u = (adaptive + e).clamp(-vp::SATURATION, vp::SATURATION);
        let xh = (st + u).clamp(-vp::SATURATION, vp::SATURATION);
        state.push(xh, u);
        out.push(xh);
    }
    out
}

/// One subframe's transmitted parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subframe {
    /// Fractional pitch lag in quarter samples (`LAG_Q_MIN..=LAG_Q_MAX`).
    pub lag_q: i32,
    /// Pitch-gain code.
    pub pitch_gain: i32,
    /// Innovation-gain code.
    pub gain: i32,
    /// `(position, positive)` per pulse, ascending by position.
    pub pulses: Vec<(u8, bool)>,
}

/// One frame's transmitted parameters. `per_track` is the codebook class, so the
/// pulse count is implicit and never spends a transmitted field.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Params {
    /// Pulses per track (1 or 2 for the declared classes).
    pub per_track: usize,
    /// One entry per subframe.
    pub subframes: Vec<Subframe>,
}

impl Params {
    /// Total innovation pulses across the frame.
    pub fn pulses_per_subframe(&self) -> usize {
        self.per_track * TRACKS
    }
}

/// The reconstructed excitation for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Shot {
    /// `(fractional lag, pitch-gain code)` per subframe.
    pub pitch: Vec<(i32, i32)>,
    /// The fixed excitation, one sample per frame sample.
    pub fixed: Vec<f64>,
}

impl Shot {
    /// Run this shot through the `exp2` synthesis loop.
    pub fn render(&self, state: &mut vp::VoiceState, k_q: &[i32], width: u8) -> Vec<f64> {
        synthesize(state, k_q, width, SUB_LEN, &self.pitch, &self.fixed)
    }
}

/// Rebuild the shot from transmitted parameters. Encoder and decoder run this
/// identical code, so the reconstruction cannot drift.
pub fn reconstruct(params: &Params, n: usize) -> Result<Shot> {
    let subs = celp::subframes(n);
    if params.subframes.len() != subs {
        return Err(Error::malformed("voice.exp2 ACELP subframe count mismatch"));
    }
    if !(1..=MAX_PER_TRACK).contains(&params.per_track) {
        return Err(Error::malformed(
            "voice.exp2 ACELP codebook class out of range",
        ));
    }
    let mut shot = Shot {
        pitch: Vec::with_capacity(subs),
        fixed: Vec::with_capacity(n),
    };
    for sub in &params.subframes {
        if sub.lag_q < LAG_Q_MIN || sub.lag_q > LAG_Q_MAX {
            return Err(Error::malformed("voice.exp2 ACELP lag out of range"));
        }
        if !(0..PITCH_GAIN_LEVELS).contains(&sub.pitch_gain) {
            return Err(Error::malformed("voice.exp2 ACELP pitch gain out of range"));
        }
        if !(0..GAIN_LEVELS).contains(&sub.gain) {
            return Err(Error::malformed("voice.exp2 ACELP gain out of range"));
        }
        let take = SUB_LEN.min(n - shot.fixed.len());
        let mut block = vec![0.0f64; take];
        let scale = celp::gain_of(sub.gain);
        // Track legality is part of the wire contract: one pulse per track in
        // its own slot, at most `per_track`, ascending.
        let mut per_slot = [0usize; TRACKS];
        let mut last: i32 = -1;
        for &(pos, positive) in &sub.pulses {
            let pos = usize::from(pos);
            if pos >= take {
                return Err(Error::malformed(
                    "voice.exp2 ACELP pulse position out of range",
                ));
            }
            if (pos as i32) <= last {
                return Err(Error::malformed("voice.exp2 ACELP pulses not canonical"));
            }
            last = pos as i32;
            let slot = pos % TRACKS;
            per_slot[slot] += 1;
            if per_slot[slot] > params.per_track {
                return Err(Error::malformed("voice.exp2 ACELP track overloaded"));
            }
            block[pos] = if positive { scale } else { -scale };
        }
        shot.pitch.push((sub.lag_q, sub.pitch_gain));
        shot.fixed.extend_from_slice(&block);
    }
    Ok(shot)
}

/// Rank one track's signed pulses onto `0..C(20,k)·2^k`.
pub fn rank_track(positions: &[usize], signs: &[bool]) -> u64 {
    let mut rank = 0u64;
    for (i, &p) in positions.iter().enumerate() {
        rank += comb(p as u32, i as u32 + 1);
    }
    let mut s = 0u64;
    for (i, &b) in signs.iter().enumerate() {
        if b {
            s |= 1 << i;
        }
    }
    rank * (1u64 << positions.len()) + s
}

/// Inverse of [`rank_track`]: `(positions, signs)` for a track index.
pub fn unrank_track(idx: u64, k: usize) -> (Vec<usize>, Vec<bool>) {
    let signs = (0..k).map(|i| (idx >> i) & 1 == 1).collect::<Vec<_>>();
    let mut rank = idx >> k;
    let mut pos = vec![0usize; k];
    let mut upper = (TRACK_POS as u32).saturating_sub(1);
    for i in (1..=k as u32).rev() {
        let mut p = upper;
        while p >= i && comb(p, i) > rank {
            p -= 1;
        }
        pos[i as usize - 1] = p as usize;
        rank -= comb(p, i);
        upper = p.saturating_sub(1);
    }
    (pos, signs)
}

/// Perceptually weighted error energy under the shared weighting filter. The
/// precomputed [`vp::Weighting`] is threaded through the whole analysis so the
/// filter is built once per frame, not once per candidate.
fn weighted(target: &[f64], out: &[f64], wt: &vp::Weighting) -> f64 {
    wt.error(target, out)
}

/// Closed-loop CELP analysis of one frame with fractional LTP and a
/// track-structured innovation.
///
/// Returns the transmitted parameters, the exact weighted distortion of the
/// chosen per-subframe solutions, and the reconstructed shot the decoder will
/// rebuild. Joint selection is by construction: each subframe's pitch candidate
/// is scored *with* the innovation that will accompany it, and the innovation
/// gain is re-evaluated against the real synthesis loop.
pub fn analyse(
    state: &vp::VoiceState,
    frame: &[f64],
    model: &vp::FrameModel,
    k_q: &[i32],
    per_track: usize,
) -> (Params, f64, Shot) {
    let per_track = per_track.clamp(1, MAX_PER_TRACK);
    // Built once per frame: the spectrum is fixed for the whole analysis, and
    // rebuilding the weighting filter per candidate dominated the encode cost.
    let wt = vp::Weighting::new(k_q, model.width);
    let w = wt.weights();
    let mut probe = state.clone();
    let mut params = Params {
        per_track,
        subframes: Vec::new(),
    };
    let mut shot = Shot {
        pitch: Vec::new(),
        fixed: Vec::with_capacity(frame.len()),
    };
    let mut distortion = 0.0f64;
    let mut centre = if model.lag > 0 { model.lag } else { 80 };

    for sub in 0..celp::subframes(frame.len()) {
        let lo = sub * SUB_LEN;
        let hi = (lo + SUB_LEN).min(frame.len());
        let target = &frame[lo..hi];
        let len = target.len();
        let zeros = vec![0.0f64; len];

        // --- Fractional lag: coarse integers, then quarter-sample refinement ---
        let mut lags_q: Vec<i32> = Vec::new();
        let grid_centre = (centre * (1 << FRAC_BITS)).clamp(LAG_Q_MIN, LAG_Q_MAX);
        for d in -NEAR..=NEAR {
            let l = (centre + d) * (1 << FRAC_BITS);
            if (LAG_Q_MIN..=LAG_Q_MAX).contains(&l) {
                lags_q.push(l);
            }
        }
        let mut l = vp::MIN_LAG as i32;
        while l <= vp::MAX_LAG as i32 {
            lags_q.push(l * (1 << FRAC_BITS));
            l += COARSE_STRIDE;
        }
        lags_q.sort_unstable();
        lags_q.dedup();

        let mut best_lag: Option<(i32, i32, f64)> = None;
        for &lag_q in &lags_q {
            if let Some(c) = evaluate_lag(&probe, w, &wt, target, lag_q, &zeros)
                && best_lag.as_ref().is_none_or(|b| c.2 < b.2)
            {
                best_lag = Some(c);
            }
        }
        let (mut best_q, mut best_pg, mut best_d) = match best_lag {
            Some(b) => b,
            None => (grid_centre, 0, f64::INFINITY),
        };
        // Quarter-sample refinement around the winning integer lag.
        let base_i = best_q >> FRAC_BITS;
        for f in 1..(1 << FRAC_BITS) {
            for d in -1..=1 {
                let l = (base_i + d) * (1 << FRAC_BITS) + f;
                if l == best_q || !(LAG_Q_MIN..=LAG_Q_MAX).contains(&l) {
                    continue;
                }
                if let Some(c) = evaluate_lag(&probe, w, &wt, target, l, &zeros)
                    && c.2 < best_d
                {
                    best_q = c.0;
                    best_pg = c.1;
                    best_d = c.2;
                }
            }
        }
        if best_q > 0 {
            centre = (best_q / (1 << FRAC_BITS)).clamp(vp::MIN_LAG as i32, vp::MAX_LAG as i32);
        }

        // --- Track-structured innovation against the chosen adaptive term ---
        // The adaptive term is re-synthesised on its own: the loop is recursive
        // in the pitch gain (the gain sits inside the long-term feedback), so the
        // adaptive output is not a scaled copy of a unity-gain probe.
        let mut p_adapt = probe.clone();
        let adaptive = synthesize_w(&mut p_adapt, w, len, &[(best_q, best_pg)], &zeros);
        let residual: Vec<f64> = target.iter().zip(&adaptive).map(|(x, y)| x - y).collect();

        let mut impulse = vec![0.0f64; len];
        if len > 0 {
            impulse[0] = 1.0;
        }
        let mut h_probe = vp::VoiceState::new();
        let h = synthesize_w(&mut h_probe, w, len, &[(best_q, best_pg)], &impulse);
        let d: Vec<f64> = (0..len)
            .map(|n| (n..len).map(|i| residual[i] * h[i - n]).sum())
            .collect();
        let res_scale = (residual.iter().map(|x| x * x).sum::<f64>() / len.max(1) as f64).sqrt();

        let (pulses, gain, dist) = search_tracks(
            &probe,
            w,
            &wt,
            target,
            (best_q, best_pg),
            &d,
            &h,
            per_track,
            res_scale,
        );

        // --- Advance the shared probe with the accepted shot ---
        let scale = celp::gain_of(gain);
        let mut fixed = vec![0.0f64; len];
        for &(pos, positive) in &pulses {
            let pos = usize::from(pos);
            if pos < fixed.len() {
                fixed[pos] = if positive { scale } else { -scale };
            }
        }
        synthesize_w(&mut probe, w, len, &[(best_q, best_pg)], &fixed);
        distortion += dist;
        shot.fixed.extend_from_slice(&fixed);
        shot.pitch.push((best_q, best_pg));
        params.subframes.push(Subframe {
            lag_q: best_q,
            pitch_gain: best_pg,
            gain,
            pulses,
        });
    }
    (params, distortion, shot)
}

/// Score one fractional lag over the pitch-gain ladder and return
/// `(lag_q, pitch_gain, weighted error)`.
///
/// The long-term gain multiplies the *interpolated reconstructed excitation*,
/// so it sits inside the predictor's own feedback loop. The synthesised output
/// is therefore a polynomial in the gain, not a linear function of it, and each
/// gain candidate has to be synthesised explicitly. That is why this is the
/// dominant cost of the analysis and why the lag set is kept focused.
fn evaluate_lag(
    probe: &vp::VoiceState,
    w: &[f64],
    wt: &vp::Weighting,
    target: &[f64],
    lag_q: i32,
    zeros: &[f64],
) -> Option<(i32, i32, f64)> {
    let mut best: Option<(i32, i32, f64)> = None;
    for &g in &PITCH_GAIN_TRIALS {
        let mut p = probe.clone();
        let out = synthesize_w(&mut p, w, target.len(), &[(lag_q, g)], zeros);
        let d = weighted(target, &out, wt);
        if best.is_none_or(|b| d < b.2) {
            best = Some((lag_q, g, d));
        }
    }
    best
}

/// Focused position set per track: the `FOCUS` positions with the largest `|d|`.
fn focus_positions(d: &[f64], len: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); TRACKS];
    for (t, slot) in out.iter_mut().enumerate() {
        let mut slots: Vec<usize> = (t..len).step_by(TRACKS).collect();
        slots.sort_by(|&a, &b| d[b].abs().total_cmp(&d[a].abs()));
        slots.truncate(FOCUS);
        slots.sort_unstable();
        *slot = slots;
    }
    out
}

/// One innovation candidate: focused indices with signs.
type Set = Vec<(usize, bool)>;

/// `(dc, cc)` of a pulse set restricted to focused positions.
fn criterion(set: &Set, focus: &[usize], d: &[f64], phi: &[Vec<f64>]) -> (f64, f64) {
    let mut dc = 0.0f64;
    let mut cc = 0.0f64;
    for &(a, sa) in set {
        let da = if sa { 1.0 } else { -1.0 };
        dc += da * d[focus[a]];
        for &(b, sb) in set {
            let db = if sb { 1.0 } else { -1.0 };
            cc += da * db * phi[a][b];
        }
    }
    (dc, cc)
}

/// All `k`-subsets of `slots` with every sign combination.
fn subsets(slots: &[usize], k: usize) -> Vec<Set> {
    let mut out = Vec::new();
    let n = slots.len();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        for mask in 0..(1u32 << k) {
            out.push(
                idx.iter()
                    .enumerate()
                    .map(|(b, &s)| (slots[s], mask >> b & 1 == 1))
                    .collect(),
            );
        }
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
        }
    }
}

/// Coordinate-ascent search of the track innovation, then closed-loop gain
/// selection. Returns `(pulses, gain code, weighted error)`.
#[allow(clippy::too_many_arguments)]
fn search_tracks(
    probe: &vp::VoiceState,
    w: &[f64],
    wt: &vp::Weighting,
    target: &[f64],
    pitch: (i32, i32),
    d: &[f64],
    h: &[f64],
    per_track: usize,
    res_scale: f64,
) -> (Vec<(u8, bool)>, i32, f64) {
    let len = target.len();
    let focus_per_track = focus_positions(d, len);
    // Flatten the focused set; `current` and `subsets` index into this list.
    let mut focus: Vec<usize> = Vec::new();
    let mut track_span: Vec<(usize, usize)> = Vec::new();
    for per_track_focus in focus_per_track.iter() {
        let start = focus.len();
        focus.extend_from_slice(per_track_focus);
        track_span.push((start, focus.len()));
    }
    let m = focus.len();
    if m == 0 || per_track == 0 {
        let dist = weighted_gain(probe, w, wt, target, pitch, &with_gain(&[], 0, len));
        return (Vec::new(), celp::gain_code(res_scale), dist);
    }
    // Φ over the focused positions only: the standard complexity control.
    let mut phi = vec![vec![0.0f64; m]; m];
    for a in 0..m {
        for b in 0..=a {
            let (i, j) = (focus[a], focus[b]);
            let s: f64 = (i.max(j)..len).map(|k| h[k - i] * h[k - j]).sum();
            phi[a][b] = s;
            phi[b][a] = s;
        }
    }

    // Initialise each track from the largest |d| positions in the track.
    let mut current: Vec<Set> = vec![Vec::new(); TRACKS];
    for t in 0..TRACKS {
        let (s, e) = track_span[t];
        let mut slots: Vec<usize> = (s..e).collect();
        slots.sort_by(|&a, &b| d[focus[b]].abs().total_cmp(&d[focus[a]].abs()));
        slots.truncate(per_track.min(slots.len()));
        slots.sort_unstable();
        current[t] = best_signs(&slots, &focus, d);
    }

    // Coordinate ascent: one track at a time, all its combinations and signs.
    for _ in 0..SWEEPS {
        let mut moved = false;
        for t in 0..TRACKS {
            let (s, e) = track_span[t];
            let slots: Vec<usize> = (s..e).collect();
            if slots.len() < per_track {
                continue;
            }
            let mut best_score = f64::NEG_INFINITY;
            let mut best_set = current[t].clone();
            for cand in subsets(&slots, per_track) {
                let mut whole = cand.clone();
                for (u, other) in current.iter().enumerate() {
                    if u != t {
                        whole.extend_from_slice(other);
                    }
                }
                let (dc, cc) = criterion(&whole, &focus, d, &phi);
                let score = if cc <= 1e-12 { 0.0 } else { dc * dc / cc };
                if score > best_score {
                    best_score = score;
                    best_set = cand;
                }
            }
            if best_set != current[t] {
                moved = true;
                current[t] = best_set;
            }
        }
        if !moved {
            break;
        }
    }

    // Flatten to absolute positions in canonical ascending order.
    let mut pulses: Vec<(u8, bool)> = Vec::new();
    for set in &current {
        for &(a, s) in set {
            pulses.push((focus[a] as u8, s));
        }
    }
    pulses.sort_unstable_by_key(|&(p, _)| p);
    pulses.dedup_by_key(|&mut (p, _)| p);

    // Closing the loop: the winning set's gain is picked by the *real*
    // synthesis, not by the correlation proxy. The proxy's `dc/cc` is the
    // optimal continuous gain of the unweighted criterion; the weighted
    // optimum can differ, so a small ladder brackets it.
    let flat: Set = current.iter().flat_map(|set| set.iter().copied()).collect();
    let (dc, cc) = criterion(&flat, &focus, d, &phi);
    let optimum = if cc <= 1e-12 {
        0
    } else {
        celp::gain_code(dc / cc)
    };
    let mut best = f64::INFINITY;
    let mut best_gain = celp::gain_code(res_scale);
    for candidate in (optimum - 2)..=(optimum + 2) {
        let candidate = candidate.clamp(0, GAIN_LEVELS - 1);
        let dist = weighted_gain(
            probe,
            w,
            wt,
            target,
            pitch,
            &with_gain(&pulses, candidate, len),
        );
        if dist < best {
            best = dist;
            best_gain = candidate;
        }
    }
    (pulses, best_gain, best)
}

/// Best sign pattern for one index combination, by the correlation proxy.
fn best_signs(slots: &[usize], focus: &[usize], d: &[f64]) -> Set {
    let k = slots.len();
    let mut best: Set = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    for mask in 0..(1u32 << k) {
        let set: Set = slots
            .iter()
            .enumerate()
            .map(|(b, &s)| (s, mask >> b & 1 == 1))
            .collect();
        let dc: f64 = set
            .iter()
            .map(|&(a, s)| if s { d[focus[a]] } else { -d[focus[a]] })
            .sum();
        if dc > best_score {
            best_score = dc;
            best = set;
        }
    }
    best
}

/// Build an explicit fixed excitation of `len` samples for a pulse set.
fn with_gain(pulses: &[(u8, bool)], gain: i32, len: usize) -> Vec<f64> {
    let mut fixed = vec![0.0f64; len.max(SUB_LEN)];
    let scale = celp::gain_of(gain);
    for &(p, s) in pulses {
        fixed[usize::from(p)] = if s { scale } else { -scale };
    }
    fixed
}

/// Weighted error of an explicit pulse set through the real synthesis loop.
fn weighted_gain(
    probe: &vp::VoiceState,
    w: &[f64],
    wt: &vp::Weighting,
    target: &[f64],
    pitch: (i32, i32),
    fixed: &[f64],
) -> f64 {
    let mut p = probe.clone();
    let out = synthesize_w(&mut p, w, target.len(), &[pitch], fixed);
    weighted(target, &out, wt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_codebook_is_a_bijection() {
        // Allocation discipline: the exhaustive walk is hard-capped so the test
        // cannot grow with the codebook, and the set holds three-element tuples.
        const MAX_INDICES: u64 = 4_096;
        for k in 1..=MAX_PER_TRACK {
            let total = comb(TRACK_POS as u32, k as u32) * (1u64 << k);
            assert!(
                total <= MAX_INDICES,
                "k={k} would need {total} indices, above the {MAX_INDICES} test cap"
            );
            assert_eq!(
                u64::from(track_bits(k)),
                (total - 1).ilog2() as u64 + 1,
                "track_bits must be exact for k={k}"
            );
            // Every index unranks to a distinct, legal set that reranks back.
            let mut seen = std::collections::HashSet::new();
            for idx in 0..total {
                let (pos, sig) = unrank_track(idx, k);
                assert_eq!(pos.len(), k);
                let mut sorted = pos.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, pos, "positions must ascend");
                assert!(pos.iter().all(|&p| p < TRACK_POS));
                assert_eq!(rank_track(&pos, &sig), idx, "round trip at idx {idx}");
                assert!(seen.insert((pos.clone(), sig.clone())));
            }
            assert_eq!(seen.len() as u64, total);
        }
    }

    #[test]
    fn integer_fraction_is_exact_and_fractional_filter_is_normalised() {
        // INTERP[0] is the identity tap, and every fractional filter is a
        // partition of unity (its coefficients sum to one).
        assert_eq!(INTERP[0], [0.0, 1.0, 0.0, 0.0]);
        for (f, row) in INTERP.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "filter {f} sums to {s}");
        }
        // A constant history must interpolate to the same constant at every
        // fraction, which the partition-of-unity property guarantees.
        let mut st = vp::VoiceState::new();
        for _ in 0..256 {
            st.push(3.0, 5.0);
        }
        for f in 0..(1 << FRAC_BITS) {
            let lag_q = 40 * (1 << FRAC_BITS) + f;
            assert!((adaptive_sample(&st, lag_q) - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn the_long_term_gain_is_recursive_so_scoring_must_resynthesise() {
        // The pitch gain multiplies the interpolated *reconstructed excitation*,
        // so it sits inside the predictor feedback loop. `out(g)` is therefore a
        // polynomial in `g`, not `zir + g·q`: half gain is **not** half the
        // unity-gain response. This test pins that property down, because the
        // analysis cost and the search structure both depend on it.
        let k_q = vp::quantise_k(&[0.7, -0.5, 0.4, -0.3, 0.25, -0.2, 0.15, -0.1], 6);
        let mut st = vp::VoiceState::new();
        for i in 0..200 {
            st.push(
                (i as f64 * 0.03).sin() * 1000.0,
                (i as f64 * 0.11).sin() * 500.0,
            );
        }
        let zeros = vec![0.0f64; 80];
        let mut a = st.clone();
        let zir = synthesize(&mut a, &k_q, 6, 80, &[(0, 0)], &zeros);
        let mut b = st.clone();
        let unity = synthesize(&mut b, &k_q, 6, 80, &[(163, 20)], &zeros);
        let q: Vec<f64> = unity.iter().zip(&zir).map(|(x, y)| x - y).collect();
        let mut c = st.clone();
        let half = synthesize(&mut c, &k_q, 6, 80, &[(163, 10)], &zeros);
        let linear: Vec<f64> = zir.iter().zip(&q).map(|(z, x)| z + 0.5 * x).collect();
        let differs = half
            .iter()
            .zip(&linear)
            .any(|(h, l)| (h - l).abs() > 1e-6 * l.abs().max(1.0));
        assert!(
            differs,
            "if this ever becomes linear, the one-probe shortcut would be valid"
        );
        // And the true gain-1 response is exactly what the g=1 synthesis made.
        for i in 0..80 {
            assert!((unity[i] - (zir[i] + q[i])).abs() <= 1e-9 * unity[i].abs().max(1.0));
        }
    }

    #[test]
    fn reconstruction_rejects_malformed_parameters() {
        let good = Params {
            per_track: 1,
            subframes: (0..4)
                .map(|s| Subframe {
                    lag_q: 160 + s * 3,
                    pitch_gain: 20,
                    gain: 20,
                    // One pulse per track: slots 0, 1, 2, 3.
                    pulses: vec![(0, true), (1, false), (2, true), (3, false)],
                })
                .collect(),
        };
        assert!(reconstruct(&good, 320).is_ok());
        // A pulse that overloads its track is not a legal wire state.
        let mut overload = good.clone();
        overload.subframes[0].pulses.push((4, true)); // slot 0 already has one
        overload.subframes[0]
            .pulses
            .sort_unstable_by_key(|&(p, _)| p);
        assert!(reconstruct(&overload, 320).is_err());
        // Pulses out of canonical order are rejected.
        let mut unsorted = good.clone();
        unsorted.subframes[0].pulses = vec![(3, true), (1, false), (2, true), (0, false)];
        assert!(reconstruct(&unsorted, 320).is_err());
        // Out-of-range lag.
        let mut bad_lag = good.clone();
        bad_lag.subframes[0].lag_q = LAG_Q_MAX + 4;
        assert!(reconstruct(&bad_lag, 320).is_err());
    }
}
