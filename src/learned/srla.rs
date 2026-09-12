//! Phase 2A mechanism 2 — code-length-shaped coefficient refinement (SRLA).
//!
//! SRLA refines a quantized predictor's integer coefficients against a
//! continuous approximation of the Recursive Golomb–Rice code length of the
//! prediction residual, rather than against squared error. VOLE's twist is that
//! the approximation is only a **proposal mechanism**: the refined candidate is
//! kept only if its *exact* canonical complete bytes improve, so the physical
//! artifact stays the authority.
//!
//! This module provides the objective ([`residual_rice_bits`], an exact minimum
//! over the Rice ladder — at least as faithful as SRLA's continuous proxy) and a
//! deterministic hill climb over `±1` coefficient perturbations. Acceptance is
//! the caller's job: [`refine_coefficients`] returns the best proposal, and the
//! court compares exact bytes before and after.

/// Exact Rice bit cost of `values` at parameter `k`.
fn rice_bits(values: &[u64], k: u32) -> u64 {
    let mut bits = 0u64;
    for &v in values {
        bits = bits
            .saturating_add(v >> k)
            .saturating_add(1)
            .saturating_add(u64::from(k));
    }
    bits
}

/// Minimum Rice bit cost over the frozen ladder `k = 0..=20`.
pub fn best_rice_bits(values: &[u64]) -> u64 {
    let mut best = u64::MAX;
    for k in 0..=20u32 {
        let b = rice_bits(values, k);
        if b < best {
            best = b;
        }
    }
    if best == u64::MAX { 0 } else { best }
}

fn predict(source: &[i32], coeffs: &[i32], shift: u32, t: usize) -> i64 {
    let mut s = 0i64;
    for (i, &c) in coeffs.iter().enumerate() {
        if let Some(idx) = t.checked_sub(1 + i) {
            s += i64::from(c) * i64::from(source[idx]);
        }
    }
    s >> shift
}

/// Rice-objective cost of the open-loop residual under `coeffs`.
pub fn residual_rice_bits(source: &[i32], coeffs: &[i32], shift: u32) -> u64 {
    if coeffs.is_empty() || source.is_empty() {
        return 0;
    }
    let start = coeffs.len().min(source.len());
    let mut mags = Vec::with_capacity(source.len() - start);
    for t in start..source.len() {
        let r = i64::from(source[t]) - predict(source, coeffs, shift, t);
        let zz = ((r << 1) ^ (r >> 63)) as u64;
        mags.push(zz);
    }
    best_rice_bits(&mags)
}

/// Deterministic hill climb: repeatedly try `±1` on each coefficient in order
/// and keep strict Rice-objective improvements until a pass changes nothing.
pub fn refine_coefficients(source: &[i32], coeffs: &[i32], shift: u32, passes: usize) -> Vec<i32> {
    let mut best = coeffs.to_vec();
    let mut best_bits = residual_rice_bits(source, &best, shift);
    for _ in 0..passes {
        let mut improved = false;
        for i in 0..best.len() {
            for delta in [-1i32, 1] {
                let mut cand = best.clone();
                let Some(v) = cand[i].checked_add(delta) else {
                    continue;
                };
                cand[i] = v;
                let bits = residual_rice_bits(source, &cand, shift);
                if bits < best_bits {
                    best = cand;
                    best_bits = bits;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rice_objective_prefers_the_better_predictor() {
        let source: Vec<i32> = (0..2048).map(|i| ((i * 89) % 15013) - 7000).collect();
        let good = vec![1900i32, -800];
        let bad = vec![100i32, -20];
        assert!(residual_rice_bits(&source, &good, 12) < residual_rice_bits(&source, &bad, 12));
    }

    #[test]
    fn refinement_never_worsens_the_objective() {
        let source: Vec<i32> = (0..4096).map(|i| ((i * 137) % 20011) - 10000).collect();
        let seed = vec![1500i32, -600, 120, -12];
        let before = residual_rice_bits(&source, &seed, 12);
        let after = residual_rice_bits(&source, &refine_coefficients(&source, &seed, 12, 6), 12);
        assert!(after <= before, "before {before} after {after}");
    }

    #[test]
    fn best_rice_bits_matches_a_brute_force_scan() {
        let values = vec![0u64, 1, 2, 5, 40, 300, 7000];
        let expected = (0..=20u32).map(|k| rice_bits(&values, k)).min().unwrap();
        assert_eq!(best_rice_bits(&values), expected);
    }
}
