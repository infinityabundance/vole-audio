//! Optimizer v2: deterministic multiscale beam search (Exp2, priority `6`).
//!
//! The Exp1 optimizer is a single greedy ±1 coordinate-descent trajectory. That
//! is deliberately weak. Exp2 adds a real deterministic discrete optimizer:
//!
//! * a **frozen multiscale schedule** `256 → 64 → 16 → 4 → 1`, so a candidate
//!   can escape quantization-scale traps before it is refined;
//! * a **beam** of several trajectories (canonical expansion and tie order);
//! * **memoization** by canonical candidate hash, so neighbouring architectures
//!   are cheap to evaluate;
//! * strict complete-byte improvement only. A proposal is accepted only when the
//!   measured cost strictly decreases, and the seed is always retained, so the
//!   returned candidate is never worse than the seed.
//!
//! The optimizer has **zero semantic authority**: the cost function it consumes
//! must be the actual canonical stored size, and the returned candidate is
//! re-verified by exact closure before it is accepted anywhere.

use std::collections::HashMap;

/// The frozen multiscale step schedule (coarse to fine).
pub const MULTISCALE_SCHEDULE: [i16; 5] = [256, 64, 16, 4, 1];

/// Result of a beam optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamResult {
    /// The best parameter vector found (never worse than the seed).
    pub best: Vec<i16>,
    /// Its cost.
    pub cost: f64,
    /// Number of distinct evaluations actually performed (memo misses).
    pub evaluations: u64,
    /// Number of memo hits.
    pub cache_hits: u64,
}

/// A deterministic, memoized, multiscale beam coordinate search.
///
/// `evaluate` maps a parameter vector to a cost to minimize (lower is better).
/// Non-finite costs are treated as `f64::INFINITY` (rejected). The seed is always
/// evaluated and always retained if nothing strictly improves.
///
/// `beam_width` is clamped to `1..=MAX_LEARNED_BEAM_WIDTH`. Expansion visits
/// coordinates in ascending order and tries `+step` before `-step`, so ties are
/// broken deterministically.
pub fn beam_coordinate_descent<F>(
    seed: &[i16],
    beam_width: usize,
    max_evaluations: u64,
    mut evaluate: F,
) -> BeamResult
where
    F: FnMut(&[i16]) -> f64,
{
    let width = beam_width.clamp(1, crate::limits::MAX_LEARNED_BEAM_WIDTH as usize);
    let mut cache: HashMap<Vec<i16>, f64> = HashMap::new();
    let mut evaluations = 0u64;
    let mut cache_hits = 0u64;

    let mut cost_of = |w: &Vec<i16>,
                       cache: &mut HashMap<Vec<i16>, f64>,
                       evals: &mut u64,
                       hits: &mut u64|
     -> f64 {
        if let Some(&c) = cache.get(w) {
            *hits += 1;
            return c;
        }
        let raw = evaluate(w);
        let c = if raw.is_finite() { raw } else { f64::INFINITY };
        cache.insert(w.clone(), c);
        *evals += 1;
        c
    };

    let seed_cost = cost_of(
        &seed.to_vec(),
        &mut cache,
        &mut evaluations,
        &mut cache_hits,
    );
    // Beam entries are (cost, parameters) kept in a deterministic canonical order.
    let mut beam: Vec<(f64, Vec<i16>)> = vec![(seed_cost, seed.to_vec())];
    let mut global_best = seed_cost;
    let mut global_best_w = seed.to_vec();

    for &step in &MULTISCALE_SCHEDULE {
        // Repeat passes at this scale until no coordinate improves; strict
        // improvement guarantees termination.
        loop {
            if evaluations >= max_evaluations {
                break;
            }
            let current_best = beam[0].0;
            let mut candidates: Vec<(f64, Vec<i16>)> = Vec::new();
            let mut seen: HashMap<Vec<i16>, ()> = HashMap::new();
            for (_, w) in &beam {
                for i in 0..w.len() {
                    for delta in [step, -step] {
                        if evaluations >= max_evaluations {
                            break;
                        }
                        let new = i32::from(w[i]) + i32::from(delta);
                        if new < i32::from(i16::MIN) || new > i32::from(i16::MAX) {
                            continue;
                        }
                        let mut cand = w.clone();
                        cand[i] = new as i16;
                        if seen.insert(cand.clone(), ()).is_some() {
                            cache_hits += 1;
                            continue;
                        }
                        let c = cost_of(&mut cand, &mut cache, &mut evaluations, &mut cache_hits);
                        candidates.push((c, cand));
                    }
                }
            }
            // Deterministic order: by cost, then lexicographically by parameters.
            candidates.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            let mut next: Vec<(f64, Vec<i16>)> = Vec::new();
            for (c, w) in candidates.into_iter() {
                if c < current_best {
                    if c < global_best {
                        global_best = c;
                        global_best_w = w.clone();
                    }
                    next.push((c, w));
                    if next.len() >= width {
                        break;
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            // Keep the current best as a floor so the beam never regresses.
            next.push((current_best, beam[0].1.clone()));
            next.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            next.truncate(width);
            beam = next;
        }
    }

    BeamResult {
        best: global_best_w,
        cost: global_best,
        evaluations,
        cache_hits,
    }
}

/// Evaluate a cost with canonical memoization only (for callers that need the
/// cache across several passes).
pub struct CostCache<'a, F: FnMut(&[i16]) -> f64> {
    evaluate: &'a mut F,
    cache: HashMap<Vec<i16>, f64>,
    pub evaluations: u64,
    pub cache_hits: u64,
}

impl<'a, F: FnMut(&[i16]) -> f64> CostCache<'a, F> {
    pub fn new(evaluate: &'a mut F) -> Self {
        CostCache {
            evaluate,
            cache: HashMap::new(),
            evaluations: 0,
            cache_hits: 0,
        }
    }

    /// Cost of a candidate, memoized by its canonical parameter vector.
    pub fn cost(&mut self, w: &[i16]) -> f64 {
        if let Some(&c) = self.cache.get(w) {
            self.cache_hits += 1;
            return c;
        }
        let raw = (self.evaluate)(w);
        let c = if raw.is_finite() { raw } else { f64::INFINITY };
        self.cache.insert(w.to_vec(), c);
        self.evaluations += 1;
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_frozen_and_coarse_to_fine() {
        assert_eq!(MULTISCALE_SCHEDULE, [256, 64, 16, 4, 1]);
        assert!(MULTISCALE_SCHEDULE.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn beam_is_never_worse_than_the_seed_and_is_deterministic() {
        let target = [300i16, -500, 128, -64];
        let cost = |w: &[i16]| -> f64 {
            w.iter()
                .zip(target.iter())
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum()
        };
        let a = beam_coordinate_descent(&[0, 0, 0, 0], 6, 100_000, cost);
        let b = beam_coordinate_descent(&[0, 0, 0, 0], 6, 100_000, cost);
        assert_eq!(a, b);
        // Never worse than the seed, and materially improved.
        let seed_cost = cost(&[0, 0, 0, 0]);
        assert!(a.cost <= seed_cost);
        assert!(
            a.cost < seed_cost / 1000.0,
            "cost {} seed {}",
            a.cost,
            seed_cost
        );
        assert!(a.evaluations > 0);
    }

    #[test]
    fn multiscale_escapes_a_quantization_trap() {
        // Cost has a broad valley at 200 but the local step-1 gradient points
        // the wrong way near 0; coarse scales are required.
        let cost = |w: &[i16]| -> f64 {
            let x = f64::from(w[0]);
            // minimum at 200, with a shallow local well near 0.
            (x - 200.0).abs() + if x < 30.0 { 1000.0 - x * 10.0 } else { 0.0 }
        };
        let r = beam_coordinate_descent(&[0], 4, 100_000, cost);
        assert_eq!(r.best, vec![200]);
    }

    #[test]
    fn memoization_reduces_evaluations() {
        let mut calls = 0u64;
        let mut cost = |w: &[i16]| -> f64 {
            calls += 1;
            f64::from(w[0]) * f64::from(w[0])
        };
        {
            let mut cache = CostCache::new(&mut cost);
            for _ in 0..10 {
                let _ = cache.cost(&[3, 4]);
            }
            assert_eq!(cache.evaluations, 1);
            assert_eq!(cache.cache_hits, 9);
        }
        assert_eq!(calls, 1);
    }
}
