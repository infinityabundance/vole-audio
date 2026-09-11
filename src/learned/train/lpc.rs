//! Dense short-term LPC fitting (Report 3, Seals S2–S3).
//!
//! Speech is the canonical dense all-pole problem, and FLAC-5's real-speech
//! advantage is a *local, dense, apodized* LPC pipeline: a Tukey(0.5)-windowed
//! autocorrelation, a Levinson–Durbin recursion, a chosen integer QLP predictor
//! per analysis block, and **error-feedback coefficient quantisation** with a
//! variable coefficient precision and right shift. This module reproduces that
//! model class inside VOLE's exact residual-closure constitution.
//!
//! The float analysis is disposable proposal machinery. The stored predictor is
//! the canonical integer [`LpcPredictor`] (kind `12`) — packed coefficients at a
//! declared precision, plus an arithmetic right shift — and only the quantized
//! evaluator has semantic authority. Per-block coefficients are realised through
//! the existing [`SegmentedModel`] (each block is an independent segment), so the
//! canonical container is unchanged.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::lpc::LpcPredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::segmented::{Segment, SegmentedModel};
use crate::learned::train::{TrainBudget, TrainStats};

/// The QLP fixed-point shift used when no other is better (Seal S2 default).
pub const LPC_SHIFT: u8 = 12;

/// Coefficient precisions searched by the Seal S3 sweep (bits, including sign).
pub const SEARCH_PRECISIONS: [u8; 4] = [10, 12, 14, 16];

/// Prediction right shifts searched by the Seal S3 sweep.
pub const SEARCH_SHIFTS: [u8; 3] = [10, 12, 14];

/// The coefficient quantiser searched by the Seal S3 sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantizer {
    /// Round every coefficient independently.
    Independent,
    /// Carry quantisation error forward across the coefficient vector.
    ErrorFeedback,
}

/// A Tukey window with taper fraction `alpha` (FLAC uses `tukey(0.5)`).
pub fn tukey_window(len: usize, alpha: f64) -> Vec<f64> {
    let mut w = vec![0.0f64; len];
    if len == 0 {
        return w;
    }
    if len == 1 {
        w[0] = 1.0;
        return w;
    }
    let n1 = (len - 1) as f64;
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return vec![1.0; len];
    }
    for (i, wi) in w.iter_mut().enumerate() {
        let x = i as f64 / n1;
        *wi = if x < alpha / 2.0 {
            0.5 * (1.0 + (std::f64::consts::PI * (2.0 * x / alpha - 1.0)).cos())
        } else if x <= 1.0 - alpha / 2.0 {
            1.0
        } else {
            0.5 * (1.0 + (std::f64::consts::PI * (2.0 * x / alpha - 2.0 / alpha + 1.0)).cos())
        };
    }
    w
}

/// Windowed autocorrelation `R[0..=order]` of a block.
pub fn autocorrelation(block: &[i32], order: usize, alpha: f64) -> Vec<f64> {
    let w = tukey_window(block.len(), alpha);
    let xw: Vec<f64> = block
        .iter()
        .zip(&w)
        .map(|(&x, &ww)| f64::from(x) * ww)
        .collect();
    let n = xw.len();
    let mut r = vec![0.0f64; order + 1];
    for (k, rk) in r.iter_mut().enumerate() {
        let mut s = 0.0f64;
        for i in k..n {
            s += xw[i] * xw[i - k];
        }
        *rk = s;
    }
    r
}

/// Levinson–Durbin recursion. Returns the *predictor* coefficients `c_j` for
/// orders `1..=max_order`: `H[n] = Σ_j c_j · x[n-j]`.
pub fn levinson_durbin(r: &[f64], max_order: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(max_order);
    if r.is_empty() || r[0] <= 0.0 || !r[0].is_finite() {
        return out;
    }
    let mut a = vec![0.0f64; max_order + 1];
    let mut e = r[0];
    for i in 1..=max_order {
        let mut acc = r[i];
        for j in 1..i {
            acc -= a[j] * r[i - j];
        }
        let k = acc / e;
        if !k.is_finite() {
            break;
        }
        // The update must read the *previous* coefficients: `a[i-j]` would
        // otherwise alias values already rewritten in this iteration (the
        // classic in-place Levinson–Durbin defect, which diverges for i >= 4).
        let old = a.clone();
        for j in 1..i {
            a[j] = old[j] - k * old[i - j];
        }
        a[i] = k;
        e *= 1.0 - k * k;
        let coeffs: Vec<f64> = (1..=i).map(|j| a[j]).collect();
        out.push(coeffs);
        if e <= 0.0 || !e.is_finite() {
            break;
        }
    }
    out
}

fn round_half_away(v: f64) -> i64 {
    if v >= 0.0 {
        (v + 0.5).floor() as i64
    } else {
        (v - 0.5).ceil() as i64
    }
}

/// Quantise real predictor coefficients at a declared precision/shift.
pub fn quantize(coeffs: &[f64], shift: u8, precision: u8, quantizer: Quantizer) -> Vec<i32> {
    let scale = f64::from(1u32 << shift);
    let lo = -(1i64 << (precision - 1));
    let hi = (1i64 << (precision - 1)) - 1;
    match quantizer {
        Quantizer::Independent => coeffs
            .iter()
            .map(|&c| round_half_away(c * scale).clamp(lo, hi) as i32)
            .collect(),
        Quantizer::ErrorFeedback => {
            let mut err = 0.0f64;
            let mut out = Vec::with_capacity(coeffs.len());
            for &c in coeffs {
                err += c * scale;
                let q = round_half_away(err).clamp(lo, hi);
                err -= q as f64;
                out.push(q as i32);
            }
            out
        }
    }
}

fn block_ranges(frames: usize, block: usize) -> Vec<(usize, usize)> {
    if block == 0 {
        return vec![(0, frames)];
    }
    (0..frames)
        .step_by(block)
        .map(|b| (b, (b + block).min(frames)))
        .collect()
}

/// Zigzag-map a residual value to its Rice input.
#[inline]
fn zigzag(v: i32) -> u64 {
    let v = i64::from(v);
    ((v << 1) ^ (v >> 63)) as u64
}

/// An optimal-Rice bit estimate for a residual block (encoder-side heuristic).
fn rice_cost_bits(residual: &[i32]) -> u64 {
    if residual.is_empty() {
        return 0;
    }
    let mapped: Vec<u64> = residual.iter().map(|&v| zigzag(v)).collect();
    let mut sums = [0u64; 16];
    for &u in &mapped {
        for (k, s) in sums.iter_mut().enumerate() {
            *s += u >> k;
        }
    }
    let n = mapped.len() as u64;
    let mut best = u64::MAX;
    for (k, &s) in sums.iter().enumerate() {
        let bits = s + n + n * (k as u64);
        best = best.min(bits);
    }
    best
}

/// The exact residual of one block under a predictor, from zero local history.
fn block_residual(block: &[i32], coeffs: &[i32], shift: u8) -> Vec<i32> {
    let mut out = vec![0i32; block.len()];
    for (t, o) in out.iter_mut().enumerate() {
        let mut acc = 0i64;
        for (j, &c) in coeffs.iter().enumerate() {
            if t > j {
                acc += i64::from(c) * i64::from(block[t - 1 - j]);
            }
        }
        *o = block[t] - ((acc >> shift) as i32);
    }
    out
}

/// Analyse one block and return the Levinson predictor coefficients per order.
fn analyse_block(block: &[i32], max_order: usize) -> Vec<Vec<f64>> {
    let r = autocorrelation(block, max_order, 0.5);
    levinson_durbin(&r, max_order)
}

/// Burg's maximum-entropy estimator (forward/backward error minimisation).
/// Returns the predictor coefficients `c_j` (`x_hat[n] = Σ c_j x[n-j]`) for each
/// order `1..=max_order`.
pub fn burg_sets(block: &[i32], max_order: usize) -> Vec<Vec<f64>> {
    let n = block.len();
    if n == 0 || max_order == 0 || n <= max_order + 1 {
        return Vec::new();
    }
    let x: Vec<f64> = block.iter().map(|&v| f64::from(v)).collect();
    let mut f = x.clone();
    let mut b = x.clone();
    let mut a = vec![0.0f64; max_order + 1];
    a[0] = 1.0;
    let mut out = Vec::with_capacity(max_order);
    for m in 1..=max_order {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in m..n {
            num += f[i] * b[i - 1];
            den += f[i] * f[i] + b[i - 1] * b[i - 1];
        }
        if den <= 0.0 {
            break;
        }
        let k = 2.0 * num / den;
        let old = a.clone();
        for j in 1..m {
            a[j] = old[j] - k * old[m - j];
        }
        a[m] = -k;
        for i in (m..n).rev() {
            let fi = f[i];
            let bi = b[i - 1];
            f[i] = fi - k * bi;
            b[i] = bi - k * fi;
        }
        out.push((1..=m).map(|j| -a[j]).collect());
    }
    out
}

/// Covariance / least-squares estimator: the coefficient vector minimising the
/// exact squared error on the block, solved through the normal equations.
/// Returns `None` for a singular system.
#[allow(clippy::needless_range_loop)]
pub fn lsq_coefficients(block: &[i32], order: usize) -> Option<Vec<f64>> {
    let n = block.len();
    if order == 0 || n <= order + 1 {
        return None;
    }
    let mut a = vec![vec![0.0f64; order]; order];
    let mut rhs = vec![0.0f64; order];
    for t in order..n {
        let y = f64::from(block[t]);
        for i in 1..=order {
            let xi = f64::from(block[t - i]);
            rhs[i - 1] += xi * y;
            for j in 1..=i {
                a[i - 1][j - 1] += xi * f64::from(block[t - j]);
            }
        }
    }
    for i in 0..order {
        for j in 0..i {
            a[j][i] = a[i][j];
        }
    }
    solve_linear(a, rhs)
}

/// Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn solve_linear(mut m: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
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
        b.swap(col, piv);
        let d = m[col][col];
        for r in col + 1..n {
            let fac = m[r][col] / d;
            if fac == 0.0 {
                continue;
            }
            for c in col..n {
                m[r][c] -= fac * m[col][c];
            }
            b[r] -= fac * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= m[i][j] * x[j];
        }
        if m[i][i] == 0.0 {
            return None;
        }
        x[i] = s / m[i][i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(x)
}

/// The estimator that proposed a coefficient set (evidence only).
pub const ESTIMATORS: [&str; 3] = ["autocorrelation_levinson", "burg", "covariance_lsq"];

/// Levinson reflection (PARCOR) coefficients `k_1..k_max_order`.
pub fn levinson_reflections(r: &[f64], max_order: usize) -> Vec<f64> {
    let mut out = Vec::new();
    if r.is_empty() || r[0] <= 0.0 || !r[0].is_finite() {
        return out;
    }
    let mut a = vec![0.0f64; max_order + 1];
    let mut e = r[0];
    for i in 1..=max_order {
        let mut acc = r[i];
        for j in 1..i {
            acc -= a[j] * r[i - j];
        }
        let k = acc / e;
        if !k.is_finite() {
            break;
        }
        out.push(k);
        let old = a.clone();
        for j in 1..i {
            a[j] = old[j] - k * old[i - j];
        }
        a[i] = k;
        e *= 1.0 - k * k;
        if e <= 0.0 || !e.is_finite() {
            break;
        }
    }
    out
}

/// Burg reflection (PARCOR) coefficients `k_1..k_max_order`.
pub fn burg_reflections(block: &[i32], max_order: usize) -> Vec<f64> {
    let n = block.len();
    let mut out = Vec::new();
    if n == 0 || max_order == 0 || n <= max_order + 1 {
        return out;
    }
    let x: Vec<f64> = block.iter().map(|&v| f64::from(v)).collect();
    let mut f = x.clone();
    let mut b = x.clone();
    for m in 1..=max_order {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in m..n {
            num += f[i] * b[i - 1];
            den += f[i] * f[i] + b[i - 1] * b[i - 1];
        }
        if den <= 0.0 {
            break;
        }
        let k = 2.0 * num / den;
        out.push(k);
        for i in (m..n).rev() {
            let fi = f[i];
            let bi = b[i - 1];
            f[i] = fi - k * bi;
            b[i] = bi - k * fi;
        }
    }
    out
}

/// Quantise reflection coefficients to `Q(shift)`, keeping `|k| < 1`.
pub fn quantize_reflections(ks: &[f64], shift: u8) -> Vec<i32> {
    let limit = 1i64 << shift;
    ks.iter()
        .map(|&k| {
            let q = round_half_away(k * f64::from(1u32 << shift));
            q.clamp(-(limit - 1), limit - 1) as i32
        })
        .collect()
}

/// Build the lattice object for one block with the given order, or `None`.
fn lattice_object(
    block: &[i32],
    order: usize,
    reflection: &[f64],
    shift: u8,
) -> Option<(Vec<i32>, i32)> {
    if reflection.len() < order {
        return None;
    }
    let ks = quantize_reflections(&reflection[..order], shift);
    let p = crate::learned::lattice::LatticePredictor {
        channels: 1,
        order: order as u16,
        shift,
        reflection: ks,
        block_frames: None,
    };
    if p.validate().is_err() {
        return None;
    }
    let h = p.hypothesis_all_from_source(block, block.len()).ok()?;
    let residual: Vec<i32> = block
        .iter()
        .zip(&h)
        .map(|(&x, &hh)| x.wrapping_sub(hh))
        .collect();
    Some((residual, order as i32))
}

/// Fit a per-block lattice object over the frozen order ladder.
pub fn fit_lattice_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    block_frames: u32,
    max_order: u16,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    if source.len() != frames as usize {
        return Err(Error::malformed(
            "lattice fit is mono-only and expects one sample per frame",
        ));
    }
    if block_frames == 0 {
        return Err(Error::malformed("lattice block size must be positive"));
    }
    let max_order = max_order.min(crate::learned::lattice::MAX_LATTICE_ORDER) as usize;
    let shift = crate::learned::lattice::LATTICE_SHIFT;
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let ranges = block_ranges(frames as usize, block_frames as usize);
    let mut segments = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        stats.candidates += 1;
        let block = &source[from..to];
        let r = autocorrelation(block, max_order, 0.5);
        let lev = levinson_reflections(&r, max_order);
        let burg = burg_reflections(block, max_order);
        let mut best: Option<(i32, u8, Vec<i32>, u64)> = None;
        for &shift in &[8u8, 10, 12, 13, 14, 15] {
            for order in 1..=max_order {
                for refl in [&lev, &burg] {
                    let Some((residual, _)) = lattice_object(block, order, refl, shift) else {
                        continue;
                    };
                    let model_bits = (order as u64) * 16 + 16;
                    let cost = model_bits + rice_cost_bits(&residual);
                    if best.as_ref().is_none_or(|(_, _, _, c)| cost < *c) {
                        let ks = quantize_reflections(&refl[..order], shift);
                        best = Some((order as i32, shift, ks, cost));
                    }
                }
            }
        }
        let (order, shift, ks, _) = best.unwrap_or((1, shift, vec![0], u64::MAX));
        let p = crate::learned::lattice::LatticePredictor {
            channels: 1,
            order: order as u16,
            shift,
            reflection: ks,
            block_frames: None,
        };
        p.validate()?;
        segments.push(Segment {
            frames: (to - from) as u32,
            model: Box::new(LearnedModel::Lattice(p)),
        });
    }
    let o = build_segmented(source, frames, sample_rate_hz, segments)?;
    if !o.verify(source) {
        return Err(Error::internal("lattice candidate did not close exactly"));
    }
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((o, stats))
}

/// All estimator proposals for one order, in deterministic order.
fn estimator_proposals(
    block: &[i32],
    levinson: &[Vec<f64>],
    burg: &[Vec<f64>],
    order: usize,
) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    if let Some(c) = levinson.get(order - 1) {
        out.push(c.clone());
    }
    if let Some(c) = burg.get(order - 1) {
        out.push(c.clone());
    }
    if let Some(c) = lsq_coefficients(block, order) {
        out.push(c);
    }
    out
}

/// One block's chosen predictor parameters.
struct BlockChoice {
    coeffs: Vec<i32>,
    order: usize,
    precision: u8,
    shift: u8,
}

/// Choose `(order, precision, shift, quantizer)` for one block by minimising the
/// encoder-side estimate `model bits + optimal-Rice residual bits`.
fn choose_block(block: &[i32], max_order: usize) -> BlockChoice {
    let levinson = analyse_block(block, max_order);
    let burg = burg_sets(block, max_order);
    let mut best: Option<(u64, BlockChoice)> = None;
    for order in 1..=max_order {
        for coeffs_real in estimator_proposals(block, &levinson, &burg, order) {
            for &precision in &SEARCH_PRECISIONS {
                for &shift in &SEARCH_SHIFTS {
                    for quantizer in [Quantizer::Independent, Quantizer::ErrorFeedback] {
                        let coeffs = quantize(&coeffs_real, shift, precision, quantizer);
                        let residual = block_residual(block, &coeffs, shift);
                        let model_bits = (order as u64) * u64::from(precision) + 16;
                        let cost = model_bits + rice_cost_bits(&residual);
                        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                            best = Some((
                                cost,
                                BlockChoice {
                                    coeffs,
                                    order,
                                    precision,
                                    shift,
                                },
                            ));
                        }
                    }
                }
            }
        }
    }
    best.map(|(_, c)| c).unwrap_or(BlockChoice {
        coeffs: vec![0],
        order: 1,
        precision: 10,
        shift: LPC_SHIFT,
    })
}

fn predictor(choice: &BlockChoice) -> LpcPredictor {
    LpcPredictor {
        channels: 1,
        order: choice.order as u16,
        precision: choice.precision,
        shift: choice.shift,
        coeffs: choice.coeffs.clone(),
        block_frames: None,
    }
}

/// The encoder-side estimate for one block choice.
fn estimate_choice(block: &[i32], c: &BlockChoice) -> u64 {
    let residual = block_residual(block, &c.coeffs, c.shift);
    (c.order as u64) * u64::from(c.precision) + 16 + rice_cost_bits(&residual)
}

/// Fit dense LPC with a **per-block forward/reverse direction choice**.
pub fn fit_lpc_bidir_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    block_frames: u32,
    max_order: u16,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    if source.len() != frames as usize {
        return Err(Error::malformed(
            "bidirectional LPC fit is mono-only and expects one sample per frame",
        ));
    }
    if block_frames == 0 {
        return Err(Error::malformed(
            "bidirectional LPC block size must be positive",
        ));
    }
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let max_order = (max_order as usize).min(crate::limits::MAX_LEARNED_TAPS as usize);
    let ranges = block_ranges(frames as usize, block_frames as usize);
    let mut segments = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        stats.candidates += 2;
        let block = &source[from..to];
        let rev: Vec<i32> = block.iter().rev().copied().collect();
        let cf = choose_block(block, max_order);
        let cr = choose_block(&rev, max_order);
        let cost_f = estimate_choice(block, &cf);
        let cost_r = estimate_choice(&rev, &cr);
        let model = if cost_r < cost_f {
            LearnedModel::Reverse(crate::learned::reverse::ReversePredictor {
                channels: 1,
                inner: Box::new(LearnedModel::Lpc(predictor(&cr))),
            })
        } else {
            LearnedModel::Lpc(predictor(&cf))
        };
        segments.push(Segment {
            frames: (to - from) as u32,
            model: Box::new(model),
        });
    }
    let o = build_segmented(source, frames, sample_rate_hz, segments)?;
    if !o.verify(source) {
        return Err(Error::internal(
            "bidirectional LPC candidate did not close exactly",
        ));
    }
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((o, stats))
}

fn build_segmented(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    segments: Vec<Segment>,
) -> Result<LearnedObject> {
    let model = LearnedModel::Segmented(SegmentedModel {
        channels: 1,
        segments,
    });
    LearnedObject::from_intrinsic_exp2(model, 1, frames, sample_rate_hz, Vec::new(), source)
}

fn complete_bytes(o: &LearnedObject) -> Result<u64> {
    Ok(crate::learned::accounting::LearnedCost::of(o)?.complete_bytes)
}

/// Fit a dense per-block LPC object with the Seal S3 parameter search.
///
/// Per-block coefficients use the existing segmented container; the residual is
/// the concatenation of the per-block residuals and is encoded by the Exp2
/// residual family.
pub fn fit_lpc_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    block_frames: u32,
    max_order: u16,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    if source.len() != frames as usize {
        return Err(Error::malformed(
            "dense LPC fit is mono-only and expects one sample per frame",
        ));
    }
    if block_frames == 0 {
        return Err(Error::malformed("dense LPC block size must be positive"));
    }
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let max_order = (max_order as usize).min(crate::limits::MAX_LEARNED_TAPS as usize);
    let ranges = block_ranges(frames as usize, block_frames as usize);
    let mut segments = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        stats.candidates += 1;
        let block = &source[from..to];
        let choice = choose_block(block, max_order);
        let p = predictor(&choice);
        if p.validate().is_err() {
            stats.rejected += 1;
            continue;
        }
        segments.push(Segment {
            frames: (to - from) as u32,
            model: Box::new(LearnedModel::Lpc(p)),
        });
    }
    if segments.is_empty() {
        return Err(Error::internal("no dense LPC block could be analysed"));
    }
    let o = build_segmented(source, frames, sample_rate_hz, segments)?;
    if !o.verify(source) {
        return Err(Error::internal("dense LPC candidate did not close exactly"));
    }
    let _ = complete_bytes(&o)?;
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((o, stats))
}

/// Fit a per-block pole-zero (ARMA) object over the frozen `(p, q)` ladder.
pub fn fit_pz_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    block_frames: u32,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    if source.len() != frames as usize {
        return Err(Error::malformed(
            "pole-zero fit is mono-only and expects one sample per frame",
        ));
    }
    if block_frames == 0 {
        return Err(Error::malformed("pole-zero block size must be positive"));
    }
    let shift = crate::learned::polezero::PZ_SHIFT;
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let ranges = block_ranges(frames as usize, block_frames as usize);
    let mut segments = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        stats.candidates += 1;
        let block = &source[from..to];
        let max_p = 8usize;
        let lev = analyse_block(block, max_p);
        let mut best: Option<(crate::learned::polezero::PoleZeroPredictor, u64)> = None;
        for &(p, q) in &crate::learned::polezero::PZ_LADDER {
            let p = usize::from(p);
            let q = usize::from(q);
            let Some(ar_real) = lev.get(p - 1) else {
                continue;
            };
            let ar = quantize(ar_real, shift, 16, Quantizer::ErrorFeedback);
            // AR residual (the intermediate signal the MA stage predicts).
            let mut e = vec![0i32; block.len()];
            for t in p..block.len() {
                let mut acc = 0i64;
                for (i, &c) in ar.iter().enumerate() {
                    acc += i64::from(c) * i64::from(block[t - 1 - i]);
                }
                e[t] = block[t].wrapping_sub((acc >> shift) as i32);
            }
            let ma_real = levinson_durbin(&autocorrelation(&e[p..], q, 0.5), q);
            let ma = match ma_real.last() {
                Some(c) => quantize(c, shift, 16, Quantizer::ErrorFeedback),
                None => vec![0i32; q],
            };
            let m = crate::learned::polezero::PoleZeroPredictor {
                channels: 1,
                order_ar: p as u16,
                order_ma: q as u16,
                shift,
                ar: ar.clone(),
                ma,
                block_frames: None,
            };
            if m.validate().is_err() {
                continue;
            }
            let Ok(h) = m.hypothesis_all_from_source(block, block.len()) else {
                continue;
            };
            let residual: Vec<i32> = block
                .iter()
                .zip(&h)
                .map(|(&x, &hh)| x.wrapping_sub(hh))
                .collect();
            let cost = ((p + q) as u64) * 16 + 32 + rice_cost_bits(&residual);
            if best.as_ref().is_none_or(|(_, c)| cost < *c) {
                best = Some((m, cost));
            }
        }
        let (m, _) = best.unwrap_or_else(|| {
            (
                crate::learned::polezero::PoleZeroPredictor {
                    channels: 1,
                    order_ar: 1,
                    order_ma: 1,
                    shift,
                    ar: vec![0],
                    ma: vec![0],
                    block_frames: None,
                },
                u64::MAX,
            )
        });
        m.validate()?;
        segments.push(Segment {
            frames: (to - from) as u32,
            model: Box::new(LearnedModel::PoleZero(m)),
        });
    }
    let o = build_segmented(source, frames, sample_rate_hz, segments)?;
    if !o.verify(source) {
        return Err(Error::internal("pole-zero candidate did not close exactly"));
    }
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    Ok((o, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_is_symmetric_and_peak_flat() {
        let w = tukey_window(64, 0.5);
        assert_eq!(w.len(), 64);
        assert!((w[32] - 1.0).abs() < 1e-12);
        assert!(w[0] < 1e-9);
        assert!((w[0] - w[63]).abs() < 1e-9);
    }

    #[test]
    fn levinson_recovers_an_ar2() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut x = vec![0i32, 0];
        for t in 2..8192 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let d = (s >> 40) as i64 % 2001 - 1000;
            let v = i64::from(x[t - 1]) / 2 + i64::from(x[t - 2]) / 4 + d;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let r = autocorrelation(&x, 4, 0.5);
        let sets = levinson_durbin(&r, 4);
        let c2 = &sets[1];
        assert!((c2[0] - 0.5).abs() < 0.1, "c1 = {}", c2[0]);
        assert!((c2[1] - 0.25).abs() < 0.1, "c2 = {}", c2[1]);
    }

    #[test]
    fn error_feedback_preserves_the_aggregate_better_than_independent_rounding() {
        // A coefficient vector whose independent rounding accumulates error:
        // error-feedback must keep the running error smaller.
        let coeffs = [0.3004f64, 0.3004, 0.3004, 0.3004, 0.3004];
        let indep = quantize(&coeffs, 0, 16, Quantizer::Independent);
        let ef = quantize(&coeffs, 0, 16, Quantizer::ErrorFeedback);
        let sum_indep: i64 = indep.iter().map(|&v| i64::from(v)).sum();
        let sum_ef: i64 = ef.iter().map(|&v| i64::from(v)).sum();
        let target = (coeffs.iter().sum::<f64>()).round() as i64;
        assert!(
            (sum_ef - target).abs() <= (sum_indep - target).abs(),
            "ef {sum_ef} indep {sum_indep} target {target}"
        );
    }

    #[test]
    fn burg_and_lsq_recover_an_ar2() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut x = vec![0i32, 0];
        for t in 2..8192 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let d = (s >> 40) as i64 % 2001 - 1000;
            let v = i64::from(x[t - 1]) / 2 + i64::from(x[t - 2]) / 4 + d;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let b = burg_sets(&x, 4);
        assert!((b[1][0] - 0.5).abs() < 0.12, "burg c1 = {}", b[1][0]);
        assert!((b[1][1] - 0.25).abs() < 0.12, "burg c2 = {}", b[1][1]);
        let c = lsq_coefficients(&x, 2).unwrap();
        assert!((c[0] - 0.5).abs() < 0.05, "lsq c1 = {}", c[0]);
        assert!((c[1] - 0.25).abs() < 0.05, "lsq c2 = {}", c[1]);
    }

    #[test]
    fn lpc_fit_closes_exactly() {
        let x: Vec<i32> = (0..16384).map(|i| ((i * 37) % 2003) - 1000).collect();
        let budget = TrainBudget::default();
        let (o, st) = fit_lpc_object(&x, 16384, 16_000, 4096, 8, &budget).unwrap();
        assert!(st.candidates >= 1);
        assert!(o.verify(&x));
        assert!(matches!(o.model, LearnedModel::Segmented(_)));
    }
}
