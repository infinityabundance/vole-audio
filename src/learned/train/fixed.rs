//! Fitting for the coefficient-free fixed finite-difference family (Seal S1).
//!
//! There is nothing to fit: the coefficients are fixed by the order. The
//! "search" is a bounded sweep over the frozen order ladder (`0..=max_order`)
//! and block-local reset sizes, selecting the smallest **complete** artifact.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::learned::fixed::{FixedDifferencePredictor, MAX_FIXED_ORDER};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::train::{TrainBudget, TrainStats};

/// Build the exact fixed-difference object for one `(order, block_frames)`.
pub fn fixed_diff_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    order: u8,
    block_frames: Option<u32>,
) -> Result<LearnedObject> {
    let model = LearnedModel::Fixed(FixedDifferencePredictor {
        channels: 1,
        order,
        block_frames,
    });
    LearnedObject::from_intrinsic_exp2(model, 1, frames, sample_rate_hz, Vec::new(), source)
}

fn complete_bytes(o: &LearnedObject) -> Result<u64> {
    Ok(crate::learned::accounting::LearnedCost::of(o)?.complete_bytes)
}

/// Sweep orders `0..=max_order` at one block size and return the cheapest exact
/// candidate.
pub fn fit_fixed_diff_object(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    max_order: u8,
    block_frames: Option<u32>,
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    budget.validate()?;
    if source.len() != frames as usize {
        return Err(Error::malformed(
            "fixed-difference fit is mono-only and expects one sample per frame",
        ));
    }
    let max_order = max_order.min(MAX_FIXED_ORDER);
    let sw = Stopwatch::start();
    let mut stats = TrainStats::default();
    let mut best: Option<(LearnedObject, u64)> = None;
    for order in 0..=max_order {
        stats.candidates += 1;
        let o = match fixed_diff_object(source, frames, sample_rate_hz, order, block_frames) {
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
    stats.fit_ns = sw.elapsed_ns().max(0) as u64;
    let (o, _b) =
        best.ok_or_else(|| Error::internal("no fixed-difference candidate could close"))?;
    Ok((o, stats))
}

/// Sweep orders **and** a frozen block-size ladder, returning the cheapest exact
/// candidate across the whole surface.
pub fn fit_fixed_diff_sweep(
    source: &[i32],
    frames: u64,
    sample_rate_hz: u32,
    max_order: u8,
    block_sizes: &[Option<u32>],
    budget: &TrainBudget,
) -> Result<(LearnedObject, TrainStats)> {
    let mut best: Option<(LearnedObject, u64)> = None;
    let mut stats = TrainStats::default();
    for &bf in block_sizes {
        if let Ok((o, st)) =
            fit_fixed_diff_object(source, frames, sample_rate_hz, max_order, bf, budget)
        {
            stats.merge(&st);
            let b = complete_bytes(&o)?;
            if best.as_ref().is_none_or(|(_, bb)| b < *bb) {
                best = Some((o, b));
            }
        }
    }
    best.map(|(o, _)| (o, stats))
        .ok_or_else(|| Error::internal("no fixed-difference candidate could close"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_diff_sweep_closes_and_selects() {
        let x: Vec<i32> = (0..4096).map(|i| (i * 11) % 997 - 400).collect();
        let budget = TrainBudget::default();
        let (o, st) =
            fit_fixed_diff_sweep(&x, 4096, 48_000, 4, &[None, Some(4096)], &budget).unwrap();
        assert!(st.candidates >= 1);
        assert!(o.verify(&x));
        assert!(matches!(o.model, LearnedModel::Fixed(_)));
    }
}
