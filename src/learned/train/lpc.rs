//! Dense short-term LPC fitting (Report 3, **Seal S2**).
//!
//! Speech is the canonical dense all-pole problem, and FLAC-5's real-speech
//! advantage is a *local, dense, apodized* LPC pipeline: a Tukey(0.5)-windowed
//! autocorrelation, a Levinson–Durbin recursion, and a chosen integer QLP
//! predictor per analysis block. This module reproduces that model class inside
//! VOLE's exact residual-closure constitution.
//!
//! The float analysis is disposable proposal machinery. The stored predictor is
//! the canonical integer [`LpcPredictor`] (kind `12`) — coefficients, precision
//! and an arithmetic right shift — and only the quantized evaluator has
//! semantic authority. Per-block coefficients are realised through the existing
//! [`SegmentedModel`] (each block is an independent segment), so the canonical
//! format is unchanged.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::lpc::LpcPredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::segmented::{Segment, SegmentedModel};
use crate::learned::train::{TrainBudget, TrainStats};

/// The QLP fixed-point shift used by Seal S2 (S3 searches precision/shift).
pub const LPC_SHIFT: u8 = 12;

/// The declared coefficient precision (informational at this seal).
pub const LPC_PRECISION: u8 = 16;

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
/// orders `1..=order`: `H[n] = Σ_j c_j · x[n-j]`.
///
/// `c[order][j-1]` is the j-th coefficient of the order-`order` predictor.
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
        a[i] = k;
        for j in 1..i {
            let aj = a[j];
            a[j] = aj - k * a[i - j];
        }
        e *= 1.0 - k * k;
        // Levinson recursion for `x_hat[n] = sum_j a[j] x[n-j]` (the normal
        // equations `sum_j c_j R[|i-j|] = R[i]`), so the predictor coefficients
        // are `a[1..=i]` directly.
        let coeffs: Vec<f64> = (1..=i).map(|j| a[j]).collect();
        out.push(coeffs);
        if e <= 0.0 || !e.is_finite() {
            break;
        }
    }
    out
}

/// Quantize real predictor coefficients to the canonical integer QLP form.
pub fn quantize_coeffs(coeffs: &[f64], shift: u8) -> Vec<i32> {
    let scale = f64::from(1u32 << shift);
    coeffs
        .iter()
        .map(|&c| {
            let scaled = c * scale;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5).floor()
            } else {
                (scaled - 0.5).ceil()
            };
            if rounded >= f64::from(i32::MAX) {
                i32::MAX
            } else if rounded <= f64::from(i32::MIN) {
                i32::MIN
            } else {
                rounded as i32
            }
        })
        .collect()
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
    let mut best = u64::MAX;
    for k in 0..=15u32 {
        let mut bits = 0u64;
        for &u in &mapped {
            bits = bits.saturating_add((u >> k) + 1 + u64::from(k));
        }
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
        let pred = acc >> shift;
        *o = block[t] - (pred as i32);
    }
    out
}

/// Analyse one block and return the Levinson predictor coefficients per order.
fn analyse_block(block: &[i32], max_order: usize) -> Vec<Vec<f64>> {
    let r = autocorrelation(block, max_order, 0.5);
    levinson_durbin(&r, max_order)
}

fn predictor(coeffs: Vec<i32>, order: usize, shift: u8) -> LpcPredictor {
    LpcPredictor {
        channels: 1,
        order: order as u16,
        precision: LPC_PRECISION,
        shift,
        coeffs,
        block_frames: None,
    }
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

/// Fit one order `P` as a segmented per-block LPC object (a single order for
/// every block).
fn fit_order(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    block_frames: u32,
    order: usize,
    shift: u8,
) -> Result<LearnedObject> {
    let ranges = block_ranges(frames as usize, block_frames as usize);
    let mut segments = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        let block = &source[from..to];
        let sets = analyse_block(block, order);
        let coeffs_real = sets.last().cloned().unwrap_or_else(|| vec![0.0; order]);
        let coeffs = quantize_coeffs(&coeffs_real, shift);
        let predictor = predictor(coeffs, order, shift);
        predictor.validate()?;
        segments.push(Segment {
            frames: (to - from) as u32,
            model: Box::new(LearnedModel::Lpc(predictor)),
        });
    }
    build_segmented(source, frames, sample_rate_hz, segments)
}

/// Fit with **per-block order selection**: each block chooses the order
/// minimising its own `model bits + optimal-Rice residual bits` (the encoder-side
/// heuristic FLAC uses when it picks a subframe order), then the whole object is
/// measured by its complete bytes.
fn fit_per_block(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    block_frames: u32,
    max_order: usize,
    shift: u8,
) -> Result<LearnedObject> {
    let ranges = block_ranges(frames as usize, block_frames as usize);
    let mut segments = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        let block = &source[from..to];
        let sets = analyse_block(block, max_order);
        let mut best: Option<(Vec<i32>, usize, u64)> = None;
        for (idx, coeffs_real) in sets.iter().enumerate() {
            let order = idx + 1;
            let coeffs = quantize_coeffs(coeffs_real, shift);
            if coeffs
                .iter()
                .any(|&q| !(i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&q))
            {
                continue;
            }
            let residual = block_residual(block, &coeffs, shift);
            let model_bits = (order as u64) * 16 + 16;
            let cost = model_bits + rice_cost_bits(&residual);
            if best.as_ref().is_none_or(|(_, _, c)| cost < *c) {
                best = Some((coeffs, order, cost));
            }
        }
        let (coeffs, order, _) = best.unwrap_or_else(|| (vec![0i32; 1], 1, u64::MAX));
        let predictor = predictor(coeffs, order, shift);
        predictor.validate()?;
        segments.push(Segment {
            frames: (to - from) as u32,
            model: Box::new(LearnedModel::Lpc(predictor)),
        });
    }
    build_segmented(source, frames, sample_rate_hz, segments)
}

/// Fit and select a dense per-block LPC object over orders `1..=max_order`.
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
    let mut best: Option<(LearnedObject, u64)> = None;
    let max_order = max_order.min(crate::limits::MAX_LEARNED_TAPS as u16);
    // (a) a single order shared by every block.
    for order in 1..=usize::from(max_order) {
        stats.candidates += 1;
        let o = match fit_order(
            source,
            frames,
            sample_rate_hz,
            block_frames,
            order,
            LPC_SHIFT,
        ) {
            Ok(o) => o,
            Err(_) => {
                stats.rejected += 1;
                continue;
            }
        };
        if !o.verify(source) {
            stats.rejected += 1;
            continue;
        }
        let b = complete_bytes(&o)?;
        if best.as_ref().is_none_or(|(_, bb)| b < *bb) {
            best = Some((o, b));
        }
    }
    // (b) per-block order selection (FLAC-style local subframe choice).
    stats.candidates += 1;
    match fit_per_block(
        source,
        frames,
        sample_rate_hz,
        block_frames,
        usize::from(max_order),
        LPC_SHIFT,
    ) {
        Ok(o) if o.verify(source) => {
            let b = complete_bytes(&o)?;
            if best.as_ref().is_none_or(|(_, bb)| b < *bb) {
                best = Some((o, b));
            }
        }
        _ => stats.rejected += 1,
    }
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    let (o, _) = best.ok_or_else(|| Error::internal("no dense LPC candidate could close"))?;
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
        // An AR(2) with a1=0.5, a2=0.25 -> predictor c = [0.5, 0.25].
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
    fn lpc_fit_closes_exactly() {
        let x: Vec<i32> = (0..16384).map(|i| ((i * 37) % 2003) - 1000).collect();
        let budget = TrainBudget::default();
        let (o, st) = fit_lpc_object(&x, 16384, 16_000, 4096, 8, &budget).unwrap();
        assert!(st.candidates >= 1);
        assert!(o.verify(&x));
        assert!(matches!(o.model, LearnedModel::Segmented(_)));
    }
}
