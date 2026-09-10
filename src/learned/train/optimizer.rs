//! Specialized optimizers (`O.20`).
//!
//! No general-purpose automatic differentiation. Fitting uses closed-form least
//! squares (see [`super::linear`]) and small, explicitly implemented coordinate
//! descent over the *canonical quantized* parameters. This is deliberately the
//! weakest optimizer that can still refine a hypothesis, so the evidence can
//! never be confused with the power of a training framework.

/// Greedy coordinate descent over `i16` weights.
///
/// `evaluate` returns a cost to minimize (lower is better). The search tries
/// `±1` (and optionally `±step`) per coordinate and keeps strict improvements,
/// visiting coordinates in canonical order for determinism. Returns the best
/// weights found and the number of evaluations actually executed.
pub fn coordinate_descent_weights<F>(
    weights: &[i16],
    step: i16,
    max_iterations: u64,
    mut evaluate: F,
) -> (Vec<i16>, u64)
where
    F: FnMut(&[i16]) -> f64,
{
    let mut best = weights.to_vec();
    let mut best_cost = evaluate(&best);
    let mut iterations = 1u64;
    let mut passes = 0u64;
    // Bound the outer passes so the total evaluation budget is respected even
    // when no coordinate improves.
    let max_passes = max_iterations.div_ceil(weights.len().max(1) as u64).max(1);
    let mut improved = true;
    while improved && iterations < max_iterations && passes < max_passes {
        improved = false;
        passes += 1;
        for i in 0..best.len() {
            for delta in [step, -step] {
                if iterations >= max_iterations {
                    break;
                }
                let old = best[i];
                let new = i32::from(old) + i32::from(delta);
                if new < i32::from(i16::MIN) || new > i32::from(i16::MAX) {
                    continue;
                }
                best[i] = new as i16;
                let cost = evaluate(&best);
                iterations += 1;
                if cost < best_cost {
                    best_cost = cost;
                    improved = true;
                } else {
                    best[i] = old;
                }
            }
        }
    }
    (best, iterations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descent_finds_the_optimum_of_a_convex_quadratic() {
        // Cost = sum((w_i - target_i)^2), target = [10, -20, 30].
        let target = [10i16, -20, 30];
        let cost = |w: &[i16]| -> f64 {
            w.iter()
                .zip(target.iter())
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum()
        };
        let (best, iters) = coordinate_descent_weights(&[0, 0, 0], 2, 4096, cost);
        assert_eq!(best, target.to_vec());
        assert!(iters > 0);
    }

    #[test]
    fn descent_is_deterministic_and_bounded() {
        let cost = |w: &[i16]| -> f64 { w.iter().map(|&v| f64::from(v) * f64::from(v)).sum() };
        let a = coordinate_descent_weights(&[5, -5, 5], 1, 100, cost);
        let b = coordinate_descent_weights(&[5, -5, 5], 1, 100, cost);
        assert_eq!(a, b);
        assert!(a.1 <= 100);
    }
}
