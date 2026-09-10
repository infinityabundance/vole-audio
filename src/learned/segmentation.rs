//! Complete-byte adaptive segmentation (Exp2, priority `2`).
//!
//! A stationary model is punished by a non-stationary region. The planner
//! treats segmentation as a shortest path over a frozen boundary grid: a node
//! is a boundary position, an edge `(i, j)` is the actual complete cost of
//! representing frames `[i, j)` with its own learned hypothesis. The chosen
//! path minimizes the measured physical bytes.
//!
//! Two invariants make this structurally safe:
//!
//! * the **unsplit path** (a single edge `0 → n`) is always a candidate, so
//!   segmentation can never enlarge the portfolio;
//! * the final object is rebuilt and re-measured, and the cheaper of the planned
//!   segmentation and the unsplit candidate wins.
//!
//! Ties prefer fewer segments, then earlier boundaries (deterministic).

use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::segmented::{Segment, SegmentedModel};

/// The frozen boundary grid (frames), coarsest first.
pub const DEFAULT_BOUNDARY_GRID: [usize; 5] = [4096, 2048, 1024, 512, 256];

/// The smallest alignment unit of the frozen grid.
pub fn grid_unit(grid: &[usize]) -> usize {
    grid.iter().copied().min().unwrap_or(1).max(1)
}

/// Plan a segmentation of `[0, n)` minimizing the sum of `edge` costs.
///
/// `edge(i, j)` returns the actual complete cost of representing `[i, j)`, or
/// `None` when no candidate can close that region. The unsplit edge `(0, n)` is
/// always considered. Returns the ordered region list, or an empty vector when
/// not even the unsplit edge closes.
pub fn plan<F>(n: usize, grid: &[usize], mut edge: F) -> Vec<(usize, usize)>
where
    F: FnMut(usize, usize) -> Option<u64>,
{
    if n == 0 {
        return Vec::new();
    }
    let unit = grid_unit(grid);
    let nodes = n.div_ceil(unit);
    // Terminal node index is `nodes` (position `n`); node i is position i*unit.
    let mut cost = vec![u64::MAX; nodes + 1];
    let mut parts = vec![u64::MAX; nodes + 1];
    let mut choice = vec![0usize; nodes];
    cost[nodes] = 0;
    parts[nodes] = 0;
    for i in (0..nodes).rev() {
        let p = i * unit;
        let mut best_cost = u64::MAX;
        let mut best_parts = u64::MAX;
        let mut best_len = 0usize;
        // Candidate rungs that land on a grid node.
        for &rung in grid {
            let end = p + rung;
            if end > n || !end.is_multiple_of(unit) {
                continue;
            }
            let next = end / unit;
            if cost[next] == u64::MAX {
                continue;
            }
            if let Some(e) = edge(p, end) {
                let total = e.saturating_add(cost[next]);
                let total_parts = parts[next].saturating_add(1);
                if total < best_cost || (total == best_cost && total_parts < best_parts) {
                    best_cost = total;
                    best_parts = total_parts;
                    best_len = rung;
                }
            }
        }
        // Exact tail to `n` (covers `n` not aligned to the unit).
        let tail = n - p;
        if !n.is_multiple_of(unit)
            && tail <= grid_unit(grid) * 4
            && let Some(e) = edge(p, n)
        {
            let total = e.saturating_add(cost[nodes]);
            let total_parts = 1u64;
            if total < best_cost || (total == best_cost && total_parts < best_parts) {
                best_cost = total;
                best_parts = total_parts;
                best_len = tail;
            }
        }
        // The unsplit path: a direct edge from the root to `n`. It is a normal
        // (cost, parts) candidate, so it only wins when it is genuinely cheaper.
        if i == 0
            && let Some(e) = edge(0, n)
        {
            let total_parts = 1u64;
            if e < best_cost || (e == best_cost && total_parts < best_parts) {
                best_cost = e;
                best_parts = total_parts;
                best_len = n;
            }
        }
        cost[i] = best_cost;
        parts[i] = best_parts;
        choice[i] = best_len;
    }
    let mut regions = Vec::new();
    let mut i = 0usize;
    while i < nodes {
        let p = i * unit;
        let l = choice[i];
        if l == 0 {
            regions.clear();
            return regions;
        }
        let end = (p + l).min(n);
        regions.push((p, end));
        if end >= n {
            break;
        }
        i = end / unit;
    }
    regions
}

/// Build the cheapest segmented Exp2 object for `source` using `fit_region`,
/// or `None` when no representation closes.
///
/// The planned segmentation and the unsplit candidate are both constructed and
/// measured; the smaller complete cost wins, so the result is never larger than
/// the best single-segment candidate.
pub fn build_segmented<F>(
    source: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
    grid: &[usize],
    mut fit_region: F,
) -> Option<LearnedObject>
where
    F: FnMut(usize, usize) -> Option<LearnedObject>,
{
    let n = frames as usize;
    if n == 0 {
        return None;
    }
    let c = usize::from(channels);
    // Memoize region fits so planning and construction share one fit per region.
    let mut cache: std::collections::HashMap<(usize, usize), Option<LearnedObject>> =
        std::collections::HashMap::new();
    let mut edge = |a: usize, b: usize| -> Option<u64> {
        let o = cache.entry((a, b)).or_insert_with(|| fit_region(a, b));
        o.as_ref().map(marginal_bytes)
    };
    let regions = plan(n, grid, &mut edge);
    let unsplit = cache.get(&(0, n)).and_then(|o| o.clone());
    if regions.is_empty() {
        return unsplit;
    }
    // Assemble the segmented model from the planned regions.
    let mut segments = Vec::with_capacity(regions.len());
    let mut ok = true;
    for &(a, b) in &regions {
        let Some(o) = cache.get(&(a, b)).and_then(|o| o.clone()) else {
            ok = false;
            break;
        };
        segments.push(Segment {
            frames: (b - a) as u32,
            model: Box::new(o.model),
        });
    }
    let planned = if ok {
        let m = SegmentedModel { channels, segments };
        if m.frames() == frames && m.validate().is_ok() {
            LearnedObject::from_intrinsic_exp2(
                LearnedModel::Segmented(m),
                channels,
                frames,
                sample_rate_hz,
                Vec::new(),
                source,
            )
            .ok()
            .filter(|o| o.verify(&source[..n * c]))
        } else {
            None
        }
    } else {
        None
    };
    match (planned, unsplit) {
        (Some(p), Some(u)) => {
            let pb = crate::learned::accounting::LearnedCost::of(&p)
                .ok()?
                .complete_bytes;
            let ub = crate::learned::accounting::LearnedCost::of(&u)
                .ok()?
                .complete_bytes;
            if pb <= ub { Some(p) } else { Some(u) }
        }
        (Some(p), None) => Some(p),
        (None, u) => u,
    }
}

/// Marginal bytes a segment contributes inside a segmented object: its model
/// bytes, the residual codec id, its residual payload and its framing.
fn marginal_bytes(o: &LearnedObject) -> u64 {
    let model = o.model.canonical_bytes().len() as u64;
    model
        .saturating_add(1)
        .saturating_add(o.residual_bytes.len() as u64)
        .saturating_add(12)
}

/// Outcome of a segmentation experiment, for evidence.
pub struct SegmentationOutcome {
    pub object: LearnedObject,
    pub regions: Vec<(usize, usize)>,
    pub segmented_bytes: u64,
    pub unsplit_bytes: u64,
}

/// Convenience wrapper returning the planned regions alongside the winner.
pub fn build_segmented_with_regions<F>(
    source: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
    grid: &[usize],
    mut fit_region: F,
) -> Option<SegmentationOutcome>
where
    F: FnMut(usize, usize) -> Option<LearnedObject>,
{
    let n = frames as usize;
    if n == 0 {
        return None;
    }
    let mut cache: std::collections::HashMap<(usize, usize), Option<LearnedObject>> =
        std::collections::HashMap::new();
    let regions = {
        let mut edge = |a: usize, b: usize| -> Option<u64> {
            let o = cache.entry((a, b)).or_insert_with(|| fit_region(a, b));
            o.as_ref().map(marginal_bytes)
        };
        plan(n, grid, &mut edge)
    };
    let unsplit = cache.get(&(0, n)).and_then(|o| o.clone());
    let planned = assemble(source, channels, frames, sample_rate_hz, &regions, &cache);
    let (object, segmented_bytes, unsplit_bytes) = match (planned, unsplit) {
        (Some(p), Some(u)) => {
            let pb = crate::learned::accounting::LearnedCost::of(&p)
                .ok()?
                .complete_bytes;
            let ub = crate::learned::accounting::LearnedCost::of(&u)
                .ok()?
                .complete_bytes;
            if pb <= ub { (p, pb, ub) } else { (u, ub, pb) }
        }
        (Some(p), None) => {
            let pb = crate::learned::accounting::LearnedCost::of(&p)
                .ok()?
                .complete_bytes;
            (p, pb, pb)
        }
        (None, Some(u)) => {
            let ub = crate::learned::accounting::LearnedCost::of(&u)
                .ok()?
                .complete_bytes;
            (u, ub, ub)
        }
        (None, None) => return None,
    };
    Some(SegmentationOutcome {
        object,
        regions,
        segmented_bytes,
        unsplit_bytes,
    })
}

/// Assemble a segmented object from already-fitted region candidates.
fn assemble(
    source: &[i32],
    channels: u8,
    frames: u64,
    sample_rate_hz: u32,
    regions: &[(usize, usize)],
    cache: &std::collections::HashMap<(usize, usize), Option<LearnedObject>>,
) -> Option<LearnedObject> {
    let n = frames as usize;
    let c = usize::from(channels);
    if regions.is_empty() {
        return None;
    }
    let mut segments = Vec::with_capacity(regions.len());
    for &(a, b) in regions {
        let o = cache.get(&(a, b)).and_then(|o| o.clone())?;
        segments.push(Segment {
            frames: (b - a) as u32,
            model: Box::new(o.model),
        });
    }
    let m = SegmentedModel { channels, segments };
    if m.frames() != frames || m.validate().is_err() {
        return None;
    }
    LearnedObject::from_intrinsic_exp2(
        LearnedModel::Segmented(m),
        channels,
        frames,
        sample_rate_hz,
        Vec::new(),
        source,
    )
    .ok()
    .filter(|o| o.verify(&source[..n * c]))
}

/// A single-region (unsplit) fit helper used by courts.
pub fn unsplit<F>(frames: u64, mut fit_region: F) -> Option<(LearnedObject, u64)>
where
    F: FnMut(usize, usize) -> Option<LearnedObject>,
{
    let o = fit_region(0, frames as usize)?;
    let b = crate::learned::accounting::LearnedCost::of(&o)
        .ok()?
        .complete_bytes;
    Some((o, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::quantize_weight;
    use crate::learned::finite_field::LinearPredictor;

    fn previous_sample_region(
        source: &[i32],
    ) -> impl FnMut(usize, usize) -> Option<LearnedObject> + '_ {
        move |a: usize, b: usize| {
            let p = LinearPredictor {
                channels: 1,
                taps: 1,
                weights: vec![quantize_weight(1.0)],
                bias: vec![0],
                block_frames: None,
            };
            LearnedObject::from_intrinsic_exp2(
                LearnedModel::Linear(p),
                1,
                (b - a) as u64,
                48_000,
                Vec::new(),
                &source[a..b],
            )
            .ok()
        }
    }

    #[test]
    fn plan_prefers_the_unsplit_path_when_it_is_cheapest() {
        // Constant edge cost: many segments would be worse, so unsplit wins.
        let edge = |_a: usize, _b: usize| Some(100u64);
        let regions = plan(4096, &DEFAULT_BOUNDARY_GRID, edge);
        assert_eq!(regions, vec![(0, 4096)]);
    }

    #[test]
    fn plan_finds_a_cheaper_split_when_edges_are_nonuniform() {
        // Frames [0, 2048) cheap as one region; [2048, 4096) cheaper split.
        let edge = |a: usize, b: usize| -> Option<u64> {
            if a == 0 && b == 2048 {
                Some(50)
            } else if (a == 2048 && b == 3072) || (a == 3072 && b == 4096) {
                Some(10)
            } else if a == 2048 && b == 4096 {
                Some(500)
            } else if a == 0 && b == 4096 {
                Some(900)
            } else {
                None
            }
        };
        let regions = plan(4096, &DEFAULT_BOUNDARY_GRID, edge);
        assert_eq!(regions, vec![(0, 2048), (2048, 3072), (3072, 4096)]);
    }

    #[test]
    fn plan_is_deterministic() {
        let edge = |a: usize, b: usize| Some(((b - a) as u64 / 256).max(1) * 7);
        let a = plan(5000, &DEFAULT_BOUNDARY_GRID, edge);
        let b = plan(5000, &DEFAULT_BOUNDARY_GRID, edge);
        assert_eq!(a, b);
        // Coverage is exact and ordered.
        let mut pos = 0usize;
        for (s, e) in &a {
            assert_eq!(*s, pos);
            assert!(e > s);
            pos = *e;
        }
        assert_eq!(pos, 5000);
    }

    #[test]
    fn segmented_build_never_exceeds_unsplit() {
        let source: Vec<i32> = (0..4096)
            .map(|i| if i < 2048 { i * 3 } else { -i * 5 })
            .collect();
        let outcome = build_segmented_with_regions(
            &source,
            1,
            4096,
            48_000,
            &DEFAULT_BOUNDARY_GRID,
            previous_sample_region(&source),
        )
        .unwrap();
        assert!(outcome.object.verify(&source));
        assert!(outcome.segmented_bytes <= outcome.unsplit_bytes);
    }
}
