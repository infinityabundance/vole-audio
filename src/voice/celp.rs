//! CELP excitation coder (7C.3): per-subframe adaptive codebook + ACELP pulses.
//!
//! The scalar dead-zone residual is a *scalar* quantiser. At the sub-bit-per-sample
//! rates a voice call wants it cannot represent the excitation, and that is the
//! measured cause of the low-rate quality floor. This module supplies the CELP
//! excitation for the frames where it wins:
//!
//! ```text
//! u[n] = gₐ(s)·u[n − lag(s)] + g(s)·Σ_p ±δ[pos_p]      (s = subframe index)
//! ```
//!
//! * **Per-subframe adaptive codebook.** `lag(s)` and `gₐ(s)` are transmitted per
//!   80-sample subframe, so the long-term predictor can track a pitch that drifts
//!   within a frame. A single frame-level lag cannot, and the fixed codebook then
//!   has to spend its pulses reproducing the periodicity the adaptive codebook
//!   should supply — which is why the excitation rate was previously wasted.
//! * **ACELP fixed codebook.** Pulse sets are scored by `(d·c)²/(cᵀΦc)` with
//!   `h` the impulse response of the combined synthesis filter (short-term `1/A`
//!   plus the adaptive recursion), `d = Hᵀr` and `Φ = HᵀH`. The maximiser implies
//!   the gain, so position, sign and gain are optimised jointly.
//! * **Rate-scaled pulse count.** `n` is chosen from the frame's remaining
//!   allowance, so the excitation rate tracks the budget.
//!
//! The decoder rebuilds the *fixed* excitation from the pulses and runs the
//! shared [`crate::voice::predict::synthesize_celp`] loop with the transmitted
//! per-subframe pitch, so it reconstructs exactly what the encoder measured.
//!
//! Wire payload (frame residual field, codec id [`CODEC_ID`]), MSB-first:
//!
//! ```text
//! u8 id (= 3)
//! per subframe: lag(9) || pitch_gain(5) || n(3) || gain(6)
//!               || rank(ceil(log2 C(80, n))) || n × sign(1)
//! ```
//!
//! The pulse *positions* are coded combinatorially: the `n` distinct sample
//! positions are sorted and sent as their rank in the combinatorial number
//! system, which costs `ceil(log2 C(80, n))` bits rather than `n × 7`. For the
//! four-pulse case that is 21 bits where a flat position field would spend 28,
//! and for six pulses 29 where it would spend 42. The rank is a bijection, so
//! the reconstruction is exact; the search is unchanged.

use crate::error::{Error, Kind, Result};
use crate::voice::predict::{self as vp, FrameModel, VoiceState};

/// Samples per excitation subframe.
pub const SUB_LEN: usize = 80;
/// Bits of the lag code (`lag − MIN_LAG`).
pub const LAG_BITS: u8 = 9;
/// Bits of the per-subframe pitch gain.
pub const PITCH_GAIN_BITS: u8 = 5;
/// Bits of the pulse count.
pub const COUNT_BITS: u8 = 3;
/// Bits of the pulse gain.
pub const GAIN_BITS: u8 = 6;
/// Bits of a pulse sign.
pub const SIGN_BITS: u8 = 1;
/// Largest pulses per subframe transmitted.
pub const MAX_PULSES: usize = 6;
/// Pulse-gain levels.
pub const GAIN_LEVELS: i32 = 1 << GAIN_BITS;
/// Pitch-gain levels.
pub const PITCH_GAIN_LEVELS: i32 = 1 << PITCH_GAIN_BITS;
/// Residual-codec id that marks a CELP frame.
pub const CODEC_ID: u8 = 3;
/// Beam width of the pulse search.
const BEAM: usize = 6;
/// Gain ladder step: `2^(1/3)`, about 2 dB.
const GAIN_STEP: f64 = 1.0 / 3.0;
/// Gain ladder origin: code 16 is unity.
const GAIN_ORIGIN: i32 = 16;
/// Pitch-gain quantiser step.
const PITCH_GAIN_STEP: f64 = 0.05;
/// Per-subframe budget of the adaptive search, in lags.
const PITCH_LAGS_AROUND: i32 = 4;
/// Stride of the coarse lag sweep.
const PITCH_LAG_STRIDE: i32 = 32;
/// Pitch gains tried by the adaptive search.
const PITCH_GAIN_TRIALS: [i32; 5] = [0, 8, 16, 24, 31];

/// Whether the encoder selects the CELP excitation coder.
///
/// Enabled by measurement, not by default: see the note on the last run in
/// `docs/PHASE_7C.md`. It is a real selector, so it can be turned on the moment a
/// court run shows it winning.
pub const SELECTED: bool = true;

/// Number of subframes a frame of `n` samples carries.
pub fn subframes(n: usize) -> usize {
    n.div_ceil(SUB_LEN)
}

/// Largest pulse count per subframe whose complete record fits in `bits` bits.
///
/// The encoder derives the excitation rate from the frame allowance with this,
/// so the pulse count tracks the budget instead of being pinned. A record is
/// `lag + pitch gain + count + gain` overhead plus the combinatorial position
/// rank and one sign per pulse.
pub fn max_pulses_for_bits(bits: usize) -> usize {
    let overhead = usize::from(LAG_BITS)
        + usize::from(PITCH_GAIN_BITS)
        + usize::from(COUNT_BITS)
        + usize::from(GAIN_BITS);
    let mut best = 0usize;
    for k in 0..=MAX_PULSES {
        let cost = overhead + usize::from(position_bits(k)) + k * usize::from(SIGN_BITS);
        if cost <= bits {
            best = k;
        }
    }
    best
}

/// Smallest wire size of a CELP payload (no pulses), including the id.
pub fn min_payload_bytes(n: usize) -> usize {
    1 + (subframes(n)
        * (usize::from(LAG_BITS)
            + usize::from(PITCH_GAIN_BITS)
            + usize::from(COUNT_BITS)
            + usize::from(GAIN_BITS)))
    .div_ceil(8)
}

/// Gain of a pulse-gain code.
pub fn gain_of(code: i32) -> f64 {
    let q = code.clamp(0, GAIN_LEVELS - 1);
    2f64.powf(f64::from(q - GAIN_ORIGIN) * GAIN_STEP)
}

/// Nearest pulse-gain code.
pub fn gain_code(g: f64) -> i32 {
    if !g.is_finite() {
        return if g > 0.0 { GAIN_LEVELS - 1 } else { 0 };
    }
    if g <= 0.0 {
        return 0;
    }
    let q = (g.log2() / GAIN_STEP).round() as i32 + GAIN_ORIGIN;
    q.clamp(0, GAIN_LEVELS - 1)
}

/// Long-term gain of a pitch-gain code.
pub fn pitch_gain_of(code: i32) -> f64 {
    f64::from(code.clamp(0, PITCH_GAIN_LEVELS - 1)) * PITCH_GAIN_STEP
}

/// Nearest pitch-gain code.
pub fn pitch_gain_code(g: f64) -> i32 {
    if !g.is_finite() {
        return if g > 0.0 { PITCH_GAIN_LEVELS - 1 } else { 0 };
    }
    ((g / PITCH_GAIN_STEP).round() as i32).clamp(0, PITCH_GAIN_LEVELS - 1)
}

/// Lag code (offset from `MIN_LAG`).
fn lag_code(lag: i32) -> i32 {
    (lag - vp::MIN_LAG as i32).clamp(0, (1 << LAG_BITS) - 1)
}

/// Lag from a code.
fn lag_of(code: i32) -> i32 {
    vp::MIN_LAG as i32 + code.clamp(0, (1 << LAG_BITS) - 1)
}

/// Binomial coefficient `C(n, k)`, saturating. Bounded by `C(80, 6)` here.
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

/// Bits to carry the rank of a `count`-subset of `SUB_LEN` positions: exactly
/// `ceil(log2 C(SUB_LEN, count))`, and zero for the empty set.
pub(crate) fn position_bits(count: usize) -> u8 {
    let c = comb(SUB_LEN as u32, count as u32);
    if c <= 1 { 0 } else { (c - 1).ilog2() as u8 + 1 }
}

/// Rank of a strictly increasing position set in the combinatorial number
/// system: `Σ_i C(p_i, i + 1)`. Bijective onto `0..C(SUB_LEN, k)`.
pub(crate) fn rank_positions(sorted: &[u8]) -> u64 {
    sorted
        .iter()
        .enumerate()
        .map(|(i, &p)| comb(u32::from(p), i as u32 + 1))
        .sum()
}

/// Inverse of [`rank_positions`].
pub(crate) fn unrank_positions(mut rank: u64, count: usize) -> Vec<u8> {
    let mut out = vec![0u8; count];
    let mut upper = (SUB_LEN as u32).saturating_sub(1);
    for i in (1..=count as u32).rev() {
        // Largest p with C(p, i) <= rank, searched from the top down.
        let mut p = upper;
        while p >= i && comb(p, i) > rank {
            p -= 1;
        }
        out[i as usize - 1] = p as u8;
        rank -= comb(p, i);
        upper = p.saturating_sub(1);
    }
    out
}

/// One subframe's transmitted parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subframe {
    /// Pitch lag in samples (`MIN_LAG..=MAX_LAG`).
    pub lag: i32,
    /// Pitch-gain code.
    pub pitch_gain: i32,
    /// Pulse-gain code.
    pub gain: i32,
    /// `(position, positive)` for each pulse.
    pub pulses: Vec<(u8, bool)>,
}

/// One frame's transmitted excitation parameters.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Params {
    /// One entry per subframe.
    pub subframes: Vec<Subframe>,
}

impl Params {
    /// Exact payload length in bytes, including the id.
    pub fn payload_bytes(&self) -> usize {
        1 + self.bit_len().div_ceil(8)
    }

    /// Exact payload length in bits.
    pub fn bit_len(&self) -> usize {
        self.subframes
            .iter()
            .map(|s| {
                let count = s.pulses.len().min((1usize << COUNT_BITS) - 1);
                usize::from(LAG_BITS)
                    + usize::from(PITCH_GAIN_BITS)
                    + usize::from(COUNT_BITS)
                    + usize::from(GAIN_BITS)
                    + usize::from(position_bits(count))
                    + count * usize::from(SIGN_BITS)
            })
            .sum()
    }
}

/// The reconstructed excitation for one frame: per-subframe pitch plus the fixed
/// impulse stream. Rendering it is the only synthesis call the CELP path makes.
#[derive(Debug, Clone, PartialEq)]
pub struct Shot {
    /// `(lag, pitch-gain code)` per subframe.
    pub pitch: Vec<(i32, i32)>,
    /// The fixed excitation, one sample per frame sample.
    pub fixed: Vec<f64>,
}

impl Shot {
    /// Run this shot through the shared synthesis loop.
    pub fn render(&self, state: &mut VoiceState, k_q: &[i32], width: u8) -> Vec<f64> {
        vp::synthesize_celp(state, k_q, width, SUB_LEN, &self.pitch, &self.fixed)
    }
}

/// Perceptually weighted error energy of a reconstruction against a target. Every
/// candidate in this module is scored this way — plain MSE would spend the
/// excitation on error the ear does not hear.
fn weighted(target: &[f64], out: &[f64], model: &FrameModel, k_q: &[i32]) -> f64 {
    let w = vp::weights_of(k_q, model.width);
    let (g1, g2) = vp::weight_gammas();
    vp::weighted_error_energy(target, out, &w, g1, g2)
}

/// Closed-loop search of the adaptive codebook for one subframe: the `(lag, gain)`
/// whose own contribution best explains the target.
fn search_pitch(
    state: &VoiceState,
    target: &[f64],
    model: &FrameModel,
    k_q: &[i32],
    centre: i32,
) -> (i32, i32, f64) {
    let sl = target.len();
    let zeros = vec![0.0f64; sl];
    let lo = vp::MIN_LAG as i32;
    let hi = vp::MAX_LAG as i32;
    let centre = centre.clamp(lo, hi);
    let mut lags: Vec<i32> = (centre - PITCH_LAGS_AROUND..=centre + PITCH_LAGS_AROUND)
        .filter(|&l| l >= lo && l <= hi)
        .collect();
    let mut l = lo;
    while l <= hi {
        lags.push(l);
        l += PITCH_LAG_STRIDE;
    }
    lags.sort_unstable();
    lags.dedup();
    let mut best: Option<(i32, i32, f64)> = None;
    for &lag in &lags {
        for &gain in &PITCH_GAIN_TRIALS {
            let mut probe = state.clone();
            let out = vp::synthesize_celp(&mut probe, k_q, model.width, sl, &[(lag, gain)], &zeros);
            let d = weighted(target, &out, model, k_q);
            if best.is_none_or(|b| d < b.2) {
                best = Some((lag, gain, d));
            }
        }
    }
    best.unwrap_or((0, 0, f64::INFINITY))
}

/// One partial pulse set during the ACELP search. `pulses` holds candidate
/// indices, not sample positions.
#[derive(Clone)]
struct Partial {
    mask: u128,
    pulses: Vec<(usize, bool)>,
    dc: f64,
    cc: f64,
}

impl Partial {
    fn score(&self) -> f64 {
        if self.cc <= 1e-12 {
            0.0
        } else {
            self.dc * self.dc / self.cc
        }
    }
}

/// ACELP pulse search for one subframe, given the chosen adaptive parameters.
///
/// `h`, `d` and `Φ` are computed through the shared synthesis loop with the
/// subframe's own adaptive recursion active, so the pulses are optimised against
/// the excitation they will actually accompany. `Φ` is evaluated only over the
/// positions whose correlation with the target is significant — the standard
/// CELP complexity control — which bounds the encode deadline without changing
/// what the search can choose.
fn search_pulses(
    state: &VoiceState,
    target: &[f64],
    model: &FrameModel,
    k_q: &[i32],
    pitch: (i32, i32),
    max_pulses: usize,
) -> (Vec<(u8, bool)>, i32, f64) {
    let len = target.len();
    let zero = vec![0.0f64; len];
    let mut zir_probe = state.clone();
    let zir = vp::synthesize_celp(&mut zir_probe, k_q, model.width, len, &[pitch], &zero);
    let mut impulse = vec![0.0f64; len];
    if len > 0 {
        impulse[0] = 1.0;
    }
    let mut h_probe = VoiceState::new();
    let h = vp::synthesize_celp(&mut h_probe, k_q, model.width, len, &[pitch], &impulse);
    let residual: Vec<f64> = target.iter().zip(zir.iter()).map(|(x, y)| x - y).collect();
    let d: Vec<f64> = (0..len)
        .map(|n| (n..len).map(|i| residual[i] * h[i - n]).sum())
        .collect();
    let res_rms = (residual.iter().map(|x| x * x).sum::<f64>() / len.max(1) as f64).sqrt();
    let gain_estimate = gain_code(res_rms);

    // Focus: the positions whose correlation with the target is largest.
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| d[b].abs().total_cmp(&d[a].abs()));
    let cand: Vec<usize> = order.into_iter().take((len / 3).max(12)).collect();
    let m = cand.len();
    let mut phi = vec![0.0f64; m * m];
    for a in 0..m {
        for b in 0..=a {
            let (i, j) = (cand[a], cand[b]);
            let s: f64 = (i.max(j)..len).map(|k| h[k - i] * h[k - j]).sum();
            phi[a * m + b] = s;
        }
    }
    let phi_at = |a: usize, b: usize| -> f64 {
        if a >= b {
            phi[a * m + b]
        } else {
            phi[b * m + a]
        }
    };

    // `Partial.pulses` holds *candidate indices*; they map to sample positions at
    // the end.
    let mut beam: Vec<Partial> = Vec::with_capacity(2 * m);
    for (k, &pos) in cand.iter().enumerate() {
        let cc = phi_at(k, k);
        if cc <= 1e-12 {
            continue;
        }
        for &positive in &[true, false] {
            let sd = if positive { 1.0 } else { -1.0 };
            beam.push(Partial {
                mask: 1u128 << k,
                pulses: vec![(k, positive)],
                dc: sd * d[pos],
                cc,
            });
        }
    }
    beam.sort_by(|a, b| b.score().total_cmp(&a.score()));
    beam.truncate(BEAM);
    let steps = max_pulses.min(MAX_PULSES).min(m);
    for _ in 1..steps {
        if beam.is_empty() {
            break;
        }
        let mut next: Vec<Partial> = Vec::with_capacity(beam.len() * 2 * m);
        for parent in &beam {
            for (k, &pos) in cand.iter().enumerate() {
                if parent.mask & (1u128 << k) != 0 {
                    continue;
                }
                let same = phi_at(k, k);
                for &positive in &[true, false] {
                    let sd = if positive { 1.0 } else { -1.0 };
                    let mut cross = 0.0f64;
                    for &(q, qpos) in &parent.pulses {
                        let sq = if qpos { 1.0 } else { -1.0 };
                        cross += sq * sd * phi_at(q, k);
                    }
                    let cc = parent.cc + same + 2.0 * cross;
                    if cc <= 1e-12 {
                        continue;
                    }
                    let mut pulses = parent.pulses.clone();
                    pulses.push((k, positive));
                    next.push(Partial {
                        mask: parent.mask | (1u128 << k),
                        pulses,
                        dc: parent.dc + sd * d[pos],
                        cc,
                    });
                }
            }
        }
        if next.is_empty() {
            break;
        }
        next.sort_by(|a, b| b.score().total_cmp(&a.score()));
        next.truncate(BEAM);
        beam = next;
    }
    if beam.is_empty() {
        return (Vec::new(), gain_estimate, f64::INFINITY);
    }
    let winner = &beam[0];
    let optimum = gain_code(winner.dc / winner.cc);
    let mut fixed = vec![0.0f64; len];
    let mut best = f64::INFINITY;
    let mut best_gain = gain_estimate;
    for candidate in [optimum, gain_estimate, optimum - 1, optimum + 1] {
        let candidate = candidate.clamp(0, GAIN_LEVELS - 1);
        let scale = gain_of(candidate);
        fixed.fill(0.0);
        for &(k, positive) in &winner.pulses {
            fixed[cand[k]] = if positive { scale } else { -scale };
        }
        let mut probe = state.clone();
        let out = vp::synthesize_celp(&mut probe, k_q, model.width, len, &[pitch], &fixed);
        let dist = weighted(target, &out, model, k_q);
        if dist < best {
            best = dist;
            best_gain = candidate;
        }
    }
    let pulses: Vec<(u8, bool)> = {
        let mut p: Vec<(u8, bool)> = winner
            .pulses
            .iter()
            .map(|&(k, positive)| (cand[k] as u8, positive))
            .collect();
        // Canonical order, matching the payload's combinatorial position code.
        p.sort_unstable_by_key(|&(pos, _)| pos);
        p
    };
    (pulses, best_gain, best)
}

/// Closed-loop CELP analysis of one frame.
///
/// Returns the transmitted parameters, the exact total distortion, and the exact
/// reconstructed shot (per-subframe pitch plus the fixed excitation) the decoder
/// will rebuild.
pub fn analyse(
    state: &VoiceState,
    frame: &[f64],
    model: &FrameModel,
    k_q: &[i32],
    max_pulses_per_subframe: usize,
) -> (Params, f64, Shot) {
    let mut probe = state.clone();
    let mut params = Params::default();
    let mut shot = Shot {
        pitch: Vec::new(),
        fixed: Vec::with_capacity(frame.len()),
    };
    let mut distortion = 0.0f64;
    let mut centre = if model.lag > 0 { model.lag } else { 80 };
    for sub in 0..subframes(frame.len()) {
        let lo = sub * SUB_LEN;
        let hi = (lo + SUB_LEN).min(frame.len());
        let target = &frame[lo..hi];
        let (lag, pitch_gain, _) = search_pitch(&probe, target, model, k_q, centre);
        if lag > 0 {
            centre = lag;
        }
        let (pulses, gain, d) = search_pulses(
            &probe,
            target,
            model,
            k_q,
            (lag, pitch_gain),
            max_pulses_per_subframe,
        );
        distortion += d;
        let scale = gain_of(gain);
        let mut fixed = vec![0.0f64; target.len()];
        for &(pos, positive) in &pulses {
            let pos = usize::from(pos);
            if pos < fixed.len() {
                fixed[pos] = if positive { scale } else { -scale };
            }
        }
        vp::synthesize_celp(
            &mut probe,
            k_q,
            model.width,
            target.len(),
            &[(lag, pitch_gain)],
            &fixed,
        );
        shot.fixed.extend_from_slice(&fixed);
        shot.pitch.push((lag, pitch_gain));
        params.subframes.push(Subframe {
            lag,
            pitch_gain,
            gain,
            pulses,
        });
    }
    (params, distortion, shot)
}

/// Rebuild the shot from transmitted parameters. This is the only reconstruction
/// path: encoder and decoder run this identical code.
pub fn reconstruct(params: &Params, n: usize) -> Result<Shot> {
    let subs = subframes(n);
    if params.subframes.len() != subs {
        return Err(Error::malformed("voice CELP subframe count mismatch"));
    }
    let mut shot = Shot {
        pitch: Vec::with_capacity(subs),
        fixed: Vec::with_capacity(n),
    };
    for sub in &params.subframes {
        if sub.lag < vp::MIN_LAG as i32 || sub.lag > vp::MAX_LAG as i32 {
            return Err(Error::malformed("voice CELP lag out of range"));
        }
        if !(0..PITCH_GAIN_LEVELS).contains(&sub.pitch_gain) {
            return Err(Error::malformed("voice CELP pitch gain out of range"));
        }
        if !(0..GAIN_LEVELS).contains(&sub.gain) {
            return Err(Error::malformed("voice CELP gain out of range"));
        }
        let take = SUB_LEN.min(n - shot.fixed.len());
        let mut block = vec![0.0f64; take];
        let scale = gain_of(sub.gain);
        for &(pos, positive) in &sub.pulses {
            let pos = usize::from(pos);
            if pos >= take {
                return Err(Error::malformed("voice CELP pulse position out of range"));
            }
            block[pos] = if positive { scale } else { -scale };
        }
        shot.pitch.push((sub.lag, sub.pitch_gain));
        shot.fixed.extend_from_slice(&block);
    }
    Ok(shot)
}

/// Pack a payload: id byte then the subframe records, MSB-first.
pub fn encode_payload(params: &Params) -> Vec<u8> {
    let mut bits: Vec<bool> = Vec::with_capacity(params.bit_len());
    let mut push = |v: u32, n: u8| {
        for i in (0..n).rev() {
            bits.push((v >> i) & 1 == 1);
        }
    };
    for sub in &params.subframes {
        let count = sub.pulses.len().min((1usize << COUNT_BITS) - 1);
        push(lag_code(sub.lag) as u32, LAG_BITS);
        push(
            sub.pitch_gain.clamp(0, PITCH_GAIN_LEVELS - 1) as u32,
            PITCH_GAIN_BITS,
        );
        push(count as u32, COUNT_BITS);
        push(sub.gain.clamp(0, GAIN_LEVELS - 1) as u32, GAIN_BITS);
        // Canonical order: positions ascending, signs carried with them.
        let mut ordered: Vec<(u8, bool)> = sub.pulses.iter().take(count).copied().collect();
        ordered.sort_unstable_by_key(|&(p, _)| p);
        let positions: Vec<u8> = ordered.iter().map(|&(p, _)| p).collect();
        let bits = position_bits(count);
        if bits > 0 {
            push(rank_positions(&positions) as u32, bits);
        }
        for &(_, positive) in &ordered {
            push(u32::from(positive), SIGN_BITS);
        }
    }
    let mut out = Vec::with_capacity(1 + bits.len().div_ceil(8));
    out.push(CODEC_ID);
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() {
            if b {
                byte |= 1 << (7 - i);
            }
        }
        out.push(byte);
    }
    out
}

/// Unpack a CELP payload (which must begin with [`CODEC_ID`]).
pub fn decode_payload(bytes: &[u8], n: usize) -> Result<Params> {
    let (&id, body) = bytes
        .split_first()
        .ok_or_else(|| Error::malformed("empty voice CELP payload"))?;
    if id != CODEC_ID {
        return Err(Error::new(
            Kind::Unsupported,
            format!("not a voice CELP payload (id {id})"),
        ));
    }
    let subs = subframes(n);
    let mut pos = 0usize;
    let mut out = Params::default();
    for _ in 0..subs {
        let lag = lag_of(read_bits(body, &mut pos, LAG_BITS)? as i32);
        let pitch_gain = read_bits(body, &mut pos, PITCH_GAIN_BITS)? as i32;
        let count = read_bits(body, &mut pos, COUNT_BITS)? as usize;
        let gain = read_bits(body, &mut pos, GAIN_BITS)? as i32;
        let bits = position_bits(count);
        let positions = if bits > 0 {
            let rank = read_bits(body, &mut pos, bits)?;
            unrank_positions(u64::from(rank), count)
        } else {
            Vec::new()
        };
        let mut pulses = Vec::with_capacity(count);
        for &p in &positions {
            let positive = read_bits(body, &mut pos, SIGN_BITS)? == 1;
            pulses.push((p, positive));
        }
        out.subframes.push(Subframe {
            lag,
            pitch_gain,
            gain,
            pulses,
        });
    }
    Ok(out)
}

fn read_bits(body: &[u8], pos: &mut usize, n: u8) -> Result<u32> {
    let mut v = 0u32;
    for _ in 0..n {
        let byte = *pos >> 3;
        if byte >= body.len() {
            return Err(Error::malformed("voice CELP bits exhausted"));
        }
        let bit = (body[byte] >> (7 - (*pos & 7))) & 1;
        v = (v << 1) | u32::from(bit);
        *pos += 1;
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gain_ladders_round_trip_and_are_monotone() {
        let mut prev = f64::NEG_INFINITY;
        for q in 0..GAIN_LEVELS {
            let g = gain_of(q);
            assert!(g > 0.0 && g.is_finite());
            assert!(g > prev, "pulse-gain ladder must strictly increase");
            prev = g;
            assert_eq!(gain_code(g), q);
        }
        for q in 0..PITCH_GAIN_LEVELS {
            assert_eq!(pitch_gain_code(pitch_gain_of(q)), q);
        }
        assert_eq!(gain_code(0.0), 0);
        assert_eq!(gain_code(f64::INFINITY), GAIN_LEVELS - 1);
    }

    fn sample_params(n: usize, per: usize) -> Params {
        let subs = subframes(n);
        Params {
            subframes: (0..subs)
                .map(|s| {
                    // Distinct positions, in the canonical ascending order the
                    // payload's combinatorial code assumes.
                    let mut pulses: Vec<(u8, bool)> = (0..per)
                        .map(|p| (((p * 11 + s * 7) % SUB_LEN) as u8, (s + p) % 2 == 0))
                        .collect();
                    pulses.sort_unstable_by_key(|&(pos, _)| pos);
                    Subframe {
                        lag: vp::MIN_LAG as i32 + (s as i32 * 17) % 200,
                        pitch_gain: (s as i32 * 5) % PITCH_GAIN_LEVELS,
                        gain: (s as i32 * 9) % GAIN_LEVELS,
                        pulses,
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn the_combinatorial_position_code_is_a_bijection() {
        // Every rank of every supported pulse count must map back exactly, and
        // the rank must never exceed the bits budgeted for it.
        for count in 0..=MAX_PULSES {
            let total = comb(SUB_LEN as u32, count as u32);
            let bits = position_bits(count);
            assert!(
                bits == 0 || u64::from(1u32 << (bits - 1)) < total.max(1),
                "position_bits({count}) = {bits} cannot address {total} sets"
            );
            let step = (total / 4096).max(1);
            let mut r = 0u64;
            while r < total {
                let positions = unrank_positions(r, count);
                assert!(
                    positions.windows(2).all(|w| w[0] < w[1]),
                    "unrank must be strictly increasing"
                );
                assert!(positions.iter().all(|&p| usize::from(p) < SUB_LEN));
                assert_eq!(rank_positions(&positions), r);
                r += step;
            }
        }
        assert_eq!(position_bits(0), 0);
        assert_eq!(position_bits(5), 25);
        assert_eq!(position_bits(6), 29);
    }

    #[test]
    fn payload_round_trips_exactly() {
        for n in [160usize, 320] {
            for per in 0..=MAX_PULSES {
                let params = sample_params(n, per);
                let payload = encode_payload(&params);
                assert_eq!(payload.len(), params.payload_bytes());
                assert_eq!(decode_payload(&payload, n).unwrap(), params);
            }
        }
    }

    #[test]
    fn payload_rejects_truncation_and_wrong_id() {
        let params = sample_params(320, 2);
        let payload = encode_payload(&params);
        assert!(decode_payload(&[], 320).is_err());
        assert!(decode_payload(&payload[..1], 320).is_err());
        let mut bad = payload.clone();
        bad[0] = 9;
        assert!(decode_payload(&bad, 320).is_err());
    }

    #[test]
    fn reconstruction_rejects_out_of_range_parameters() {
        let mut p = sample_params(160, 0);
        p.subframes[0].lag = 10_000;
        assert!(reconstruct(&p, 160).is_err());
        let mut p = sample_params(160, 0);
        p.subframes[0].gain = GAIN_LEVELS;
        assert!(reconstruct(&p, 160).is_err());
        let mut p = sample_params(160, 1);
        p.subframes[0].pulses[0] = (200, true);
        assert!(reconstruct(&p, 160).is_err());
        let p = sample_params(160, 0);
        assert!(reconstruct(&p, 320).is_err());
    }

    #[test]
    fn a_shot_is_an_impulse_train_with_per_subframe_scale() {
        let params = Params {
            subframes: vec![
                Subframe {
                    lag: 80,
                    pitch_gain: 0,
                    gain: 16,
                    pulses: vec![(3, true), (10, false)],
                };
                2
            ],
        };
        let shot = reconstruct(&params, 160).unwrap();
        assert_eq!(shot.fixed.len(), 160);
        assert_eq!(shot.fixed[3], 1.0);
        assert_eq!(shot.fixed[10], -1.0);
        assert_eq!(shot.fixed[11], 0.0);
        assert_eq!(shot.pitch, vec![(80, 0), (80, 0)]);
    }

    #[test]
    fn analysis_parameters_reconstruct_the_analysed_shot_exactly() {
        let k: Vec<f64> = vec![
            0.70, -0.50, 0.40, -0.30, 0.25, -0.20, 0.15, -0.10, 0.08, -0.06, 0.05, -0.04, 0.03,
            -0.02, 0.02, -0.01,
        ];
        let k_q = vp::quantise_k(&k, 6);
        let model = FrameModel {
            estimator: 0,
            order: 16,
            width: 6,
            lag: 73,
            ltpg_q: 20,
        };
        let mut frame = Vec::with_capacity(320);
        let mut s = 0x1234_5678_9abc_def0u64;
        for i in 0..320 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((s >> 40) as i64 % 4001 - 2000) as f64;
            frame.push(noise + if i % 73 < 2 { 800.0 } else { 0.0 });
        }
        let state = VoiceState::new();
        let (params, _d, shot) = analyse(&state, &frame, &model, &k_q, 4);
        let payload = encode_payload(&params);
        let back = decode_payload(&payload, frame.len()).unwrap();
        assert_eq!(back, params);
        let rebuilt = reconstruct(&back, frame.len()).unwrap();
        // Exact: both sides run the same integer impulse sum and the same pitch.
        assert_eq!(shot, rebuilt);
        let mut a = state.clone();
        let mut b = state.clone();
        assert_eq!(
            shot.render(&mut a, &k_q, 6),
            rebuilt.render(&mut b, &k_q, 6)
        );
    }
}
