//! Shared substrate for the Phase O learned courts (`O.46`–`O.58`).
//!
//! Every learned court competes a learned candidate against the same strong,
//! non-learned baselines:
//!
//! * the existing bounded VOLE inverse compiler's best accepted candidate
//!   (procedural or literal) — `u1_best`;
//! * the in-process FLAC-5 baseline (B1) where the channel count allows it;
//! * simple exact predictors with the *same* residual codec family and the same
//!   complete-cost accounting, so the comparison is not rigged by pricing.
//!
//! All learned candidates are built through
//! [`LearnedObject::from_intrinsic`], which refuses to produce an object unless
//! the canonical closure is exact.

use crate::baseline::flac::{B1_LEVEL_PRIMARY, b1_flac_artifact};
use crate::error::{Error, Result};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::{Intrinsic, SearchBudget, propose::ReferenceLibrary};
use crate::learned::accounting::{LearnedCost, SharedModelCost};
use crate::learned::arithmetic::{Activation, quantize_bias, quantize_weight};
use crate::learned::finite_field::LinearPredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::stateful::{StateCheckpoint, StatefulPredictor};
use crate::learned::train::linear::fit_linear_object;
use crate::learned::train::{TrainBudget, TrainStats};
use crate::learned::transfer::TransferOperator;
use crate::object::id::ContentId;

/// Frozen protocol schema for every Phase O receipt.
pub const LEARNED_PROTOCOL: &str = "vole.audio.learned.protocol.v1";

/// The bounded inverse-compiler budget used for the u1 baseline.
pub fn search_budget() -> SearchBudget {
    SearchBudget::default()
}

/// Default learned training budget.
pub fn train_budget() -> TrainBudget {
    TrainBudget::default()
}

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

/// The existing VOLE inverse compiler's cheapest accepted candidate.
pub fn u1_best(id: &str, samples: &[i32], channels: u8) -> Result<(u64, String)> {
    let intrinsic = Intrinsic::new(id, channels, samples.to_vec())?;
    let report = crate::inverse::compile(&intrinsic, &ReferenceLibrary::new(), search_budget())?;
    let best = report
        .cheapest()
        .ok_or_else(|| Error::internal("the inverse compiler produced no candidate"))?;
    Ok((best.cost.complete_bytes, best.kind.name().to_string()))
}

/// The in-process FLAC-5 (B1) baseline, when the geometry is in FLAC's domain.
pub fn flac5(samples: &[i32], channels: u8, rate: u32) -> Option<u64> {
    b1_flac_artifact(samples, channels, rate, B1_LEVEL_PRIMARY)
        .ok()
        .map(|a| a.encoding.encoded_bytes)
}

/// Canonical U1 literal byte floor (non-entropy).
pub fn literal_floor(samples: &[i32], channels: u8) -> u64 {
    let frames = (samples.len() / usize::from(channels).max(1)) as u64;
    crate::inverse::cost::canonical_u1_literal_bytes(frames, channels)
}

/// Complete bytes of a learned object (accounting, not an estimate).
pub fn learned_bytes(o: &LearnedObject) -> Result<u64> {
    Ok(LearnedCost::of(o)?.complete_bytes)
}

// ---------------------------------------------------------------------------
// Simple exact predictor baselines (same codec family, same accounting)
// ---------------------------------------------------------------------------

/// A simple, non-learned predictor family used as an exact baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimplePredictor {
    /// Zero hypothesis.
    Zero,
    /// Constant at the window mean.
    Constant,
    /// Previous-sample (delta) hypothesis.
    Previous,
    /// Fixed periodic hypothesis at lag `p`.
    Periodic(u16),
}

impl SimplePredictor {
    /// The candidate's canonical label.
    #[allow(dead_code)]
    pub const fn label(self) -> &'static str {
        match self {
            SimplePredictor::Zero => "zero",
            SimplePredictor::Constant => "constant",
            SimplePredictor::Previous => "previous_sample",
            SimplePredictor::Periodic(_) => "periodic",
        }
    }
}

/// Build the canonical object for a simple predictor (`O.18`, `O.39`).
pub fn simple_object(
    samples: &[i32],
    channels: u8,
    frames: u64,
    rate: u32,
    kind: SimplePredictor,
) -> Result<LearnedObject> {
    let c = usize::from(channels);
    let model = match kind {
        SimplePredictor::Zero => LinearPredictor {
            channels,
            taps: 1,
            weights: vec![0; c * c],
            bias: vec![0; c],
            block_frames: None,
        },
        SimplePredictor::Constant => {
            let mut bias = vec![0i32; c];
            for ch in 0..c {
                let sum: i128 = (0..frames as usize)
                    .map(|t| i128::from(samples[t * c + ch]))
                    .sum();
                let mean = (sum / i128::from(frames)) as f64;
                // The bias is Q12, so `H = bias >> 12`; this needs
                // `mean · 2^12`, which saturates for extreme DC levels. The
                // strong code-domain constant baseline is the u1 inverse
                // compiler's `Constant` candidate (`u1_best`).
                bias[ch] = quantize_bias(mean);
            }
            LinearPredictor {
                channels,
                taps: 1,
                weights: vec![0; c * c],
                bias,
                block_frames: None,
            }
        }
        SimplePredictor::Previous => {
            let mut weights = vec![0i16; c * c];
            for ch in 0..c {
                weights[ch * c + ch] = quantize_weight(1.0);
            }
            LinearPredictor {
                channels,
                taps: 1,
                weights,
                bias: vec![0; c],
                block_frames: None,
            }
        }
        SimplePredictor::Periodic(p) => {
            let p = p.max(1);
            let mut weights = vec![0i16; usize::from(p) * c * c];
            for ch in 0..c {
                weights[(usize::from(p) - 1) * c * c + ch * c + ch] = quantize_weight(1.0);
            }
            LinearPredictor {
                channels,
                taps: p,
                weights,
                bias: vec![0; c],
                block_frames: None,
            }
        }
    };
    LearnedObject::from_intrinsic(
        LearnedModel::Linear(model),
        channels,
        frames,
        rate,
        Vec::new(),
        samples,
    )
}

/// Candidate periodic lags searched by the simple periodic baseline.
pub const PERIODIC_CANDIDATES: [u16; 14] = [2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 100, 128];

/// The cheapest simple predictor for a window (deterministic tie order).
///
/// Uses [`crate::learned::bounds::admissible_select`] with a **provable**
/// model-byte lower bound: a candidate's complete bytes are at least its weight
/// and bias bytes, which are known before fitting, so a candidate whose model
/// lower bound cannot beat the incumbent is never built. The winner is
/// identical to exhaustive in-order evaluation.
///
/// Candidates whose exact residual does not fit the canonical i32 residual
/// domain (for example full-scale toggling content) are skipped honestly; an
/// error is returned only when *no* simple candidate can close.
pub fn best_simple(
    samples: &[i32],
    channels: u8,
    frames: u64,
    rate: u32,
) -> Result<(LearnedObject, u64)> {
    let c = usize::from(channels);
    // Exhaustive candidate order: the zero/constant/previous trio, then the
    // frozen periodic ladder.
    let mut plan: Vec<SimplePredictor> = vec![
        SimplePredictor::Zero,
        SimplePredictor::Constant,
        SimplePredictor::Previous,
    ];
    for p in PERIODIC_CANDIDATES {
        if u64::from(p) < frames {
            plan.push(SimplePredictor::Periodic(p));
        }
    }
    // Admissible bound: `taps * C * C` i16 weights plus `C` i32 biases. The
    // canonical model bytes always include these, so the bound is valid.
    let lower_bound = |kind: SimplePredictor| -> u64 {
        let taps = match kind {
            SimplePredictor::Periodic(p) => usize::from(p).max(1),
            _ => 1,
        };
        (taps * c * c * 2 + c * 4) as u64
    };
    let mut accounting_error: Option<Error> = None;
    let report = crate::learned::bounds::admissible_select(
        plan.len(),
        |i| lower_bound(plan[i]),
        |i| match simple_object(samples, channels, frames, rate, plan[i]) {
            Ok(o) => match learned_bytes(&o) {
                Ok(b) => b,
                Err(e) => {
                    accounting_error.get_or_insert(e);
                    u64::MAX
                }
            },
            Err(_) => u64::MAX,
        },
    );
    if let Some(e) = accounting_error {
        return Err(e);
    }
    let idx = report
        .winner
        .filter(|_| report.winner_cost != u64::MAX)
        .ok_or_else(|| Error::internal("no simple predictor candidate could close"))?;
    let object = simple_object(samples, channels, frames, rate, plan[idx])?;
    Ok((object, report.winner_cost))
}

/// The cheapest simple predictor's complete bytes, or `u64::MAX` when no simple
/// candidate can close (an honest negative, never a court failure).
pub fn best_simple_bytes(samples: &[i32], channels: u8, frames: u64, rate: u32) -> u64 {
    best_simple(samples, channels, frames, rate)
        .map(|(_, b)| b)
        .unwrap_or(u64::MAX)
}

/// A resilient linear fit: `None` when the family cannot close the window
/// exactly (for example an uncloseable i32 residual gap).
pub fn try_fit_linear(
    samples: &[i32],
    channels: u8,
    frames: u64,
    rate: u32,
    taps: u16,
    block_frames: Option<u32>,
) -> Option<LearnedObject> {
    crate::learned::train::linear::fit_linear_object(
        samples,
        channels,
        frames,
        rate,
        taps,
        block_frames,
        frames as usize,
        &train_budget(),
    )
    .ok()
    .map(|(o, _)| o)
}

/// A resilient linear fit compiled into the **Exp2** residual family.
#[allow(dead_code)]
pub fn try_fit_linear_exp2(
    samples: &[i32],
    channels: u8,
    frames: u64,
    rate: u32,
    taps: u16,
    block_frames: Option<u32>,
) -> Option<LearnedObject> {
    crate::learned::train::linear::fit_linear_object_exp2(
        samples,
        channels,
        frames,
        rate,
        taps,
        block_frames,
        frames as usize,
        &train_budget(),
    )
    .ok()
    .map(|(o, _)| o)
}

/// A resilient quantization-aware fit.
pub fn try_fit_qat(
    samples: &[i32],
    channels: u8,
    frames: u64,
    rate: u32,
    taps: u16,
    refine_iterations: u64,
) -> Option<LearnedObject> {
    crate::learned::train::quant_aware::fit_qat_linear(
        samples,
        channels,
        frames,
        rate,
        taps,
        None,
        frames as usize,
        &train_budget(),
        refine_iterations,
    )
    .ok()
    .map(|(o, _)| o)
}

/// A resilient nonlinear fit.
pub fn try_fit_nonlinear(
    samples: &[i32],
    frames: u64,
    rate: u32,
    taps: u16,
    bins: u32,
) -> Option<LearnedObject> {
    crate::learned::train::finite_field::fit_nonlinear_object(
        samples,
        frames,
        rate,
        taps,
        bins,
        &train_budget(),
        0,
    )
    .ok()
    .map(|(o, _)| o)
}

// ---------------------------------------------------------------------------
// Learned candidates
// ---------------------------------------------------------------------------

/// A bounded learned linear candidate: the object, its complete bytes and the
/// tap count that produced it.
pub type LearnedLinearCandidate = (LearnedObject, u64, u16);

/// Fit the learned linear finite-field family over a bounded tap sweep and
/// return the cheapest exact candidate plus its training statistics.
pub fn best_learned_linear(
    samples: &[i32],
    channels: u8,
    frames: u64,
    rate: u32,
    taps: &[u16],
    block_frames: Option<u32>,
) -> Result<(Option<LearnedLinearCandidate>, TrainStats)> {
    let budget = train_budget();
    let mut stats = TrainStats::default();
    let mut best: Option<LearnedLinearCandidate> = None;
    for &k in taps {
        if u64::from(k) >= frames {
            continue;
        }
        let (o, st) = match fit_linear_object(
            samples,
            channels,
            frames,
            rate,
            k,
            block_frames,
            frames as usize,
            &budget,
        ) {
            Ok(v) => v,
            Err(_) => {
                stats.rejected += 1;
                continue;
            }
        };
        stats.merge(&st);
        if !o.verify(samples) {
            stats.rejected += 1;
            continue;
        }
        let b = learned_bytes(&o)?;
        if best.as_ref().is_none_or(|(_, bb, _)| b < *bb) {
            best = Some((o, b, k));
        }
    }
    Ok((best, stats))
}

/// Build a stateful predictor that realizes a fitted FIR as a shift register,
/// with canonical checkpoints derived from the source.
#[allow(clippy::needless_range_loop)]
pub fn stateful_from_fir(
    fir: &LinearPredictor,
    samples: &[i32],
    frames: usize,
    interval: u32,
) -> Result<StatefulPredictor> {
    // Mono-only realization (the FIR's own cross terms are preserved for C=1).
    if fir.channels != 1 {
        return Err(Error::malformed(
            "stateful realization is mono-only in this build",
        ));
    }
    let k = fir.tap_count();
    let mut out_w = vec![0i16; k];
    for j in 0..k {
        // state[j] holds X_hat[t-1-j]; tap lag = j+1.
        out_w[j] = fir.weights[k - 1 - j];
    }
    let mut rec_w = vec![0i16; k * k];
    for j in 1..k {
        rec_w[j * k + (j - 1)] = quantize_weight(1.0);
    }
    let mut in_w = vec![0i16; k];
    in_w[0] = quantize_weight(1.0);
    let mut p = StatefulPredictor {
        channels: 1,
        state_dim: k as u16,
        out_w,
        out_b: fir.bias.clone(),
        rec_w,
        rec_b: vec![0; k],
        in_w,
        activation: Activation::Identity,
        checkpoint_interval: interval,
        checkpoints: Vec::new(),
    };
    p.validate()?;
    let cps = p.derive_checkpoints(samples, frames)?;
    p.checkpoints = cps;
    p.validate()?;
    Ok(p)
}

/// A deterministic zero-order stateful predictor (`H = X_hat[t-1]`).
#[allow(dead_code)]
pub fn delay_stateful(frames: usize, interval: u32) -> Result<StatefulPredictor> {
    let p = StatefulPredictor {
        channels: 1,
        state_dim: 1,
        out_w: vec![quantize_weight(1.0)],
        out_b: vec![0],
        rec_w: vec![0],
        rec_b: vec![0],
        in_w: vec![quantize_weight(1.0)],
        activation: Activation::Identity,
        checkpoint_interval: interval,
        checkpoints: Vec::new(),
    };
    p.validate()?;
    let _ = frames;
    Ok(p)
}

/// Build a transfer operator by ridge-fitting the causal cross-channel FIR.
#[allow(clippy::needless_range_loop)]
pub fn fit_transfer(
    source: &[i32],
    source_channels: u8,
    target: &[i32],
    target_channels: u8,
    frames: usize,
    taps: u16,
    delay: i64,
) -> Result<TransferOperator> {
    // Reuse the ridge machinery by constructing an "aligned source" whose
    // history lines up with the delay, then fit a per-output least-squares FIR.
    let sc = usize::from(source_channels);
    let tc = usize::from(target_channels);
    let k = usize::from(taps);
    let n_feat = k * sc + 1;
    let mut a = vec![vec![0f64; n_feat]; n_feat];
    let mut b = vec![vec![0f64; tc]; n_feat];
    let mut row = vec![0f64; n_feat];
    for t in 0..frames {
        for kk in 1..=k {
            let si = t as i64 - delay - (kk as i64 - 1);
            for i in 0..sc {
                row[(kk - 1) * sc + i] = if si >= 0 && (si as usize) < frames {
                    f64::from(source[si as usize * sc + i])
                } else {
                    0.0
                };
            }
        }
        row[n_feat - 1] = 1.0;
        for j in 0..n_feat {
            let fj = row[j];
            if fj == 0.0 {
                continue;
            }
            for o in 0..tc {
                b[j][o] += fj * f64::from(target[t * tc + o]);
            }
            for jj in 0..=j {
                a[j][jj] += fj * row[jj];
            }
        }
    }
    for j in 0..n_feat {
        for jj in 0..j {
            a[jj][j] = a[j][jj];
        }
    }
    let scale = (0..n_feat)
        .map(|j| a[j][j].abs())
        .fold(0.0f64, f64::max)
        .max(1.0);
    for j in 0..n_feat - 1 {
        a[j][j] += 1e-6 + 1e-9 * scale;
    }
    a[n_feat - 1][n_feat - 1] += 1e-12 * scale;
    let mut weights = vec![0i16; k * tc * sc];
    let mut bias = vec![0i32; tc];
    for o in 0..tc {
        let rhs: Vec<f64> = (0..n_feat).map(|j| b[j][o]).collect();
        let sol = solve_small(&a, &rhs)
            .ok_or_else(|| Error::internal("transfer normal equations are singular"))?;
        for kk in 1..=k {
            for i in 0..sc {
                weights[((kk - 1) * tc + o) * sc + i] = quantize_weight(sol[(kk - 1) * sc + i]);
            }
        }
        bias[o] = quantize_bias(sol[n_feat - 1]);
    }
    let op = TransferOperator {
        channels: target_channels,
        source_channels,
        taps,
        delay,
        weights,
        bias,
    };
    op.validate()?;
    Ok(op)
}

#[allow(clippy::needless_range_loop)]
fn solve_small(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut rr = r.clone();
            rr.push(b[i]);
            rr
        })
        .collect();
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col][col].abs();
        for r in col + 1..n {
            if m[r][col].abs() > best {
                best = m[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        let d = m[col][col];
        for r in col + 1..n {
            let f = m[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for cc in col..=n {
                m[r][cc] -= f * m[col][cc];
            }
        }
    }
    let mut x = vec![0f64; n];
    for i in (0..n).rev() {
        let mut s = m[i][n];
        for j in i + 1..n {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Projection / receipt helpers
// ---------------------------------------------------------------------------

/// Push a length-prefixed label into a projection buffer.
pub fn push_label(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Push a `u64` into a projection buffer.
pub fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// SHA-256 hex of a projection buffer.
pub fn projection_hash(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Deterministic splitmix64 (statistics helper; never the semantic universe PRNG).
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Exact two-sided Wilcoxon signed-rank over paired differences.
///
/// Ranks are average-free (deterministic by index) and zero differences are
/// dropped. The null distribution of the positive-rank sum is computed exactly
/// by dynamic programming. Returns `(w_plus, w_minus, p_two_sided_ppm)`.
pub fn wilcoxon_exact(diffs: &[i64]) -> (u64, u64, u64) {
    let mut items: Vec<(u64, bool)> = diffs
        .iter()
        .filter(|&&d| d != 0)
        .map(|&d| (d.unsigned_abs(), d > 0))
        .collect();
    items.sort_unstable_by_key(|&(m, _)| m);
    let n = items.len();
    if n == 0 {
        return (0, 0, 1_000_000);
    }
    let mut w_plus = 0u64;
    let mut w_minus = 0u64;
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j < n && items[j].0 == items[i].0 {
            j += 1;
        }
        let rank_sum: u64 = ((i + 1) as u64..=(j as u64)).sum();
        let group = (j - i) as u64;
        for &(_, positive) in &items[i..j] {
            if positive {
                w_plus += rank_sum / group;
            } else {
                w_minus += rank_sum / group;
            }
        }
        i = j;
    }
    let total: u64 = (n as u64) * (n as u64 + 1) / 2;
    let mut dp = vec![0u64; (total + 1) as usize];
    dp[0] = 1;
    for r in 1..=n as u64 {
        for s in (r..=total).rev() {
            let add = dp[(s - r) as usize];
            dp[s as usize] += add;
        }
    }
    let outcomes: u64 = 1u64 << n;
    let w = w_plus.min(w_minus);
    let tail: u64 = dp[..=(w as usize)].iter().sum();
    let p_ppm = ((2 * tail).min(outcomes) * 1_000_000 + outcomes / 2) / outcomes;
    (w_plus, w_minus, p_ppm)
}

/// Deterministic bootstrap CI (95%) for the median paired ratio in ppm.
pub fn bootstrap_median_ppm(nums: &[u64], dens: &[u64], seed: u64, rounds: u32) -> (u64, u64) {
    if nums.is_empty() {
        return (0, 0);
    }
    let n = nums.len();
    let mut state = seed | 1;
    let mut medians: Vec<u64> = Vec::with_capacity(rounds as usize);
    let mut sample: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..rounds {
        sample.clear();
        for _ in 0..n {
            let idx = (splitmix64(&mut state) % n as u64) as usize;
            let ratio = (nums[idx] + 1)
                .saturating_mul(1_000_000)
                .checked_div(dens[idx] + 1)
                .unwrap_or(u64::MAX);
            sample.push(ratio);
        }
        sample.sort_unstable();
        medians.push(sample[n / 2]);
    }
    medians.sort_unstable();
    let lo = medians[(rounds as f64 * 0.025) as usize];
    let hi = medians[((rounds as f64 * 0.975) as usize).min(rounds as usize - 1)];
    (lo, hi)
}

/// A shared-model regime helper: one learned model reused across objects.
#[allow(dead_code)]
pub fn shared_regime(
    shared_model_bytes: u64,
    per_object_incremental_bytes: u64,
    independent_each: u64,
    max_n: u64,
) -> (SharedModelCost, Option<u64>) {
    let cost = SharedModelCost {
        shared_model_bytes,
        per_object_incremental_bytes,
    };
    let n_star = cost.crossover(independent_each, max_n);
    (cost, n_star)
}

/// A dependency content id for a synthetic source object.
pub fn source_content_id(id: &str) -> ContentId {
    ContentId::from_bytes(Sha256::digest(id.as_bytes()))
}

/// A checkpoint summary for receipts.
#[allow(dead_code)]
pub fn checkpoint_summary(cps: &[StateCheckpoint]) -> serde_json::Value {
    serde_json::json!({
        "count": cps.len(),
        "first_frame": cps.first().map(|c| c.frame),
        "last_frame": cps.last().map(|c| c.frame),
        "state_dim": cps.first().map(|c| c.state.len()),
    })
}

/// Freeze-or-write a learned court's receipt.
///
/// The frozen hash is compared against the projection before the receipt is
/// written; an empty frozen hash means "first run" and only reports the
/// observed value.
#[allow(clippy::too_many_arguments)]
pub fn finish(
    court: &str,
    receipts_root: &std::path::Path,
    frozen: &str,
    projection: &[u8],
    verdict: crate::status::Verdict,
    detail: String,
    extras: Vec<(&str, serde_json::Value)>,
) -> Result<crate::status::Verdict> {
    finish_with_profile(
        court,
        crate::learned::profile::LearnedProfile::Exp1,
        receipts_root,
        frozen,
        projection,
        verdict,
        detail,
        extras,
    )
}

/// Freeze-or-write an **Exp2** learned court's receipt.
#[allow(clippy::too_many_arguments)]
pub fn finish_exp2(
    court: &str,
    receipts_root: &std::path::Path,
    frozen: &str,
    projection: &[u8],
    verdict: crate::status::Verdict,
    detail: String,
    extras: Vec<(&str, serde_json::Value)>,
) -> Result<crate::status::Verdict> {
    finish_with_profile(
        court,
        crate::learned::profile::LearnedProfile::Exp2,
        receipts_root,
        frozen,
        projection,
        verdict,
        detail,
        extras,
    )
}

/// Freeze-or-write an **Exp3** learned court's receipt.
#[allow(clippy::too_many_arguments)]
pub fn finish_exp3(
    court: &str,
    receipts_root: &std::path::Path,
    frozen: &str,
    projection: &[u8],
    verdict: crate::status::Verdict,
    detail: String,
    extras: Vec<(&str, serde_json::Value)>,
) -> Result<crate::status::Verdict> {
    finish_with_profile(
        court,
        crate::learned::profile::LearnedProfile::Exp3,
        receipts_root,
        frozen,
        projection,
        verdict,
        detail,
        extras,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn finish_with_profile(
    court: &str,
    profile: crate::learned::profile::LearnedProfile,
    receipts_root: &std::path::Path,
    frozen: &str,
    projection: &[u8],
    verdict: crate::status::Verdict,
    detail: String,
    extras: Vec<(&str, serde_json::Value)>,
) -> Result<crate::status::Verdict> {
    use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
    let observed = projection_hash(projection);
    if !frozen.is_empty() && observed != frozen {
        let mut b = ReceiptBuilder::new(court);
        b.result(crate::status::Verdict::FailedCorrectness)
            .result_detail(format!(
                "static result hash changed: frozen {frozen}, observed {observed}"
            ));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court {court}: FAILED_CORRECTNESS (static result hash changed)");
        eprintln!("  receipt: {}", path.display());
        return Ok(crate::status::Verdict::FailedCorrectness);
    }
    if frozen.is_empty() {
        eprintln!("court {court}: frozen result hash is unset; observed {observed}");
    }
    let schema = match profile {
        crate::learned::profile::LearnedProfile::Exp1 => {
            crate::learned::profile::LEARNED_EVIDENCE_SCHEMA
        }
        crate::learned::profile::LearnedProfile::Exp2 => {
            crate::learned::profile::LEARNED_EXP2_EVIDENCE_SCHEMA
        }
        crate::learned::profile::LearnedProfile::Exp3 => {
            crate::learned::profile::LEARNED_EXP3_EVIDENCE_SCHEMA
        }
    };
    let mut builder = ReceiptBuilder::new(court);
    builder
        .result(verdict)
        .result_detail(format!("{detail}; result sha256 {observed}"))
        .params(CourtParams {
            universe: Some(crate::learned::profile::LEARNED_UNIVERSE.into()),
            profile: Some(profile.name().into()),
            backend: Some("scalar canonical learned evaluator".into()),
            content_kind: Some("Phase O learned deterministic prediction".into()),
            ..Default::default()
        })
        .provenance(Provenance {
            reference_hash: Some(observed.clone()),
            ..Default::default()
        })
        .extra(
            "protocol",
            serde_json::json!({
                "protocol_schema": LEARNED_PROTOCOL,
                "schema": schema,
                "profile": profile.name(),
                "version": profile.version(),
                "semantic_authority": "none: a learned hypothesis is a candidate family, \
                                       never truth; scalar exact closure decides acceptance",
            }),
        );
    for (k, v) in extras {
        builder.extra(k, v);
    }
    builder.limitation(
        "learned prediction is an experimental profile: it does not change u1/v1, the literal \
         fallback remains mandatory, and negative learned results are preserved rather than hidden",
    );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court {court}: {verdict}");
    println!("  result sha256: {observed}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_baselines_are_exact_and_price_consistently() {
        let samples: Vec<i32> = (0..512).map(|i| ((i * 37) % 401) - 200).collect();
        let (o, bytes) = best_simple(&samples, 1, 512, 48_000).unwrap();
        assert!(o.verify(&samples));
        assert_eq!(bytes, learned_bytes(&o).unwrap());
    }

    #[test]
    fn periodic_baseline_captures_an_exact_period() {
        let samples: Vec<i32> = (0..512)
            .map(|i| if i % 8 < 4 { 12345 } else { -12345 })
            .collect();
        let (o, bytes) = best_simple(&samples, 1, 512, 48_000).unwrap();
        assert!(o.verify(&samples));
        // A period-8 predictor should be very cheap (model + framing only).
        assert!(bytes < 256, "periodic baseline cost {bytes}");
    }

    #[test]
    fn stateful_realization_of_a_fir_is_exact() {
        let samples: Vec<i32> = (0..512).map(|i| ((i * 17) % 251) - 125).collect();
        let (obj, _) =
            fit_linear_object(&samples, 1, 512, 48_000, 2, None, 512, &train_budget()).unwrap();
        let LearnedModel::Linear(fir) = &obj.model else {
            panic!("expected linear")
        };
        let p = stateful_from_fir(fir, &samples, 512, 64).unwrap();
        let so = LearnedObject::from_intrinsic(
            LearnedModel::Stateful(p),
            1,
            512,
            48_000,
            Vec::new(),
            &samples,
        )
        .unwrap();
        assert!(so.verify(&samples));
        assert_eq!(so.materialize_range(300, 40).unwrap(), &samples[300..340]);
    }
}
