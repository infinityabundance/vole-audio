//! Line spectral frequency (LSF) conversion for the voice predictor.
//!
//! This module is a *representation* layer on top of the reflection
//! coefficients the voice codec already transmits. It does not change what is
//! coded; it exposes the same all-pole filter in the LSF domain, where the
//! parameters are ordered, bounded and (unlike raw predictor coefficients)
//! safe to interpolate and quantise.
//!
//! # Conventions
//!
//! Reflection coefficients follow [`crate::lossy::predict::reflection_to_weights`]:
//! the Levinson step-up produces `a[0] = 1` and `a[j]`, and the direct-form
//! synthesis weights are `w[j-1] = -a[j]`, so
//!
//! ```text
//! A(z) = 1 + sum_{j=1..p} a_j z^-j = 1 - sum_j w_j z^-j,   synthesis = 1/A(z).
//! ```
//!
//! For an even order `p = 2m` the auxiliary polynomials are
//!
//! ```text
//! P(z) = A(z) + z^-(p+1) A(z^-1)   (even symmetric, root at z = -1)
//! Q(z) = A(z) - z^-(p+1) A(z^-1)   (odd  symmetric, root at z = +1)
//! ```
//!
//! with `a[p+1] := 0`. Dividing out the known roots gives the degree-`p`
//! palindromic `P'`, `Q'`; their unit-circle zeros are the LSFs. For a stable
//! filter those zeros are simple and the two sets strictly interlace, and the
//! **smallest** LSF belongs to `P'`. Consequently, in the zero-based sorted LSF
//! vector, *even* indices come from `P'` (and carry the `(1 + z^-1)` factor) and
//! *odd* indices come from `Q'` (the `(1 - z^-1)` factor). The inverse below is
//! written to that convention and the round-trip tests pin it down.
//!
//! # Root isolation
//!
//! For the palindromic `P'`,
//!
//! ```text
//! P'(e^{jw}) = e^{-j w p/2} * C1(w),  C1(w) = P'[m] + 2 sum_{k=1..m} P'[m-k] cos(k w)
//! ```
//!
//! and likewise `Q' -> C2`. The LSFs are the zeros of `C1` and `C2` in `(0, pi)`.
//! `C1`/`C2` are evaluated as trigonometric sums (never expanded into a
//! polynomial in `cos w`): expanding to a monomial basis and rooting that is
//! badly conditioned for clustered LSFs — e.g. every reflection coefficient near
//! `+0.99` — where `~1e-16` coefficient noise moves the printed roots by `~1e-8`
//! and wrecks the round trip. The direct sum keeps the sign reliable through the
//! cancellation near a zero.
//!
//! The forward conversion is only defined for a *strictly* stable reflection
//! vector (`|k_i| < 1`); anything else returns `None`, as does a numerical
//! failure to isolate exactly `p` roots. Nothing here panics on hostile input.

use core::f64::consts::PI;

use crate::lossy::predict::reflection_to_weights;

/// Smallest LSF kept by [`stabilize_lsf`], in radians.
pub const MIN_LSF: f64 = 0.0015;
/// Largest LSF kept by [`stabilize_lsf`], in radians (`PI - MIN_LSF`).
pub const PI_MINUS_MIN_LSF: f64 = PI - MIN_LSF;
/// Minimum gap enforced between consecutive LSFs, in radians (~100 Hz @ 16 kHz).
pub const MIN_SEPARATION: f64 = 0.006;

/// Number of sub-intervals `(0, PI)` is split into when bracketing roots. Roots
/// of a stable filter are simple and, with `MIN_SEPARATION` enforced by the
/// caller, well separated, so a fine uniform grid in the *angle* is enough to
/// isolate every root without making the encoder's LSF conversion expensive.
const ROOT_SAMPLES: usize = 2048;
/// Bisection iterations per bracketed root. Fixed count => deterministic.
const BISECTION_ITERS: usize = 80;
/// Roots within this distance of the open-interval boundary are rejected as
/// degenerate (an LSF must be strictly inside `(0, pi)`).
const BOUNDARY_EPS: f64 = 1e-9;

/// Line spectral frequencies (radians, strictly increasing, in `(0, pi)`) of an
/// order-`p` all-pole filter `A(z)` derived from reflection coefficients `k`.
///
/// Returns `None` if the reflection vector is not strictly stable (`|k_i| >= 1`
/// or non-finite), if the order is not positive even, or if the conversion
/// fails numerically. Length `== k.len()`.
pub fn reflections_to_lsf(k: &[f64]) -> Option<Vec<f64>> {
    let p = k.len();
    if p < 2 || !p.is_multiple_of(2) {
        return None;
    }
    if k.iter().any(|v| !v.is_finite() || v.abs() >= 1.0) {
        return None;
    }

    // a[0] = 1, a[j] = -weights[j-1].
    let weights = reflection_to_weights(k);
    let mut a = vec![0.0f64; p + 1];
    a[0] = 1.0;
    for (slot, w) in a[1..].iter_mut().zip(weights.iter()) {
        *slot = -*w;
    }

    // P[i] = a[i] + a[p+1-i],  Q[i] = a[i] - a[p+1-i]  for i = 0..=p+1.
    let mut p_poly = vec![0.0f64; p + 2];
    let mut q_poly = vec![0.0f64; p + 2];
    for (i, (pi, qi)) in p_poly.iter_mut().zip(q_poly.iter_mut()).enumerate() {
        let ai = if i <= p { a[i] } else { 0.0 };
        let ar = if i >= 1 { a[p + 1 - i] } else { 0.0 };
        *pi = ai + ar;
        *qi = ai - ar;
    }

    // P' = P / (1 + z^-1),  Q' = Q / (1 - z^-1), each of degree p.
    let p_prime = divide_one_plus(&p_poly);
    let q_prime = divide_one_minus(&q_poly);
    let m = p / 2;

    let c1 = |w: f64| -> f64 { cosine_sum(&p_prime[..=p], m, w) };
    let c2 = |w: f64| -> f64 { cosine_sum(&q_prime[..=p], m, w) };

    let r_even = roots_of(&c1);
    let r_odd = roots_of(&c2);
    if r_even.len() != m || r_odd.len() != m {
        return None;
    }

    // The two sets must strictly interlace with `P'` (even indices) first; if
    // they do not, the isolation was numerically unreliable and we refuse to
    // guess a convention.
    let mut merged: Vec<(f64, u8)> = r_even
        .iter()
        .map(|&w| (w, 0u8))
        .chain(r_odd.iter().map(|&w| (w, 1u8)))
        .collect();
    merged.sort_by(|x, y| x.0.total_cmp(&y.0));
    if merged.len() != p {
        return None;
    }
    let mut out = Vec::with_capacity(p);
    for (i, &(w, src)) in merged.iter().enumerate() {
        if src != (i % 2) as u8 {
            return None;
        }
        if !(w > 0.0 && w < PI) {
            return None;
        }
        out.push(w);
    }
    if out.windows(2).any(|ws| ws[1] <= ws[0]) {
        return None;
    }
    Some(out)
}

/// Direct-form synthesis weights (`x_hat[n] = sum_j w[j]*x_hat[n-j] + e[n]`)
/// from an LSF vector.
///
/// Returns `None` if the LSF vector is not strictly ordered, is not of positive
/// even length, or contains non-finite values. Length `== lsf.len()`, and the
/// result matches [`reflection_to_weights`] of the original reflections to
/// within `1e-6` when the LSFs came from [`reflections_to_lsf`].
pub fn lsf_to_weights(lsf: &[f64]) -> Option<Vec<f64>> {
    let n = lsf.len();
    if n < 2 || !n.is_multiple_of(2) {
        return None;
    }
    if lsf.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if lsf.windows(2).any(|ws| ws[1] <= ws[0]) {
        return None;
    }

    // A(z) = 0.5 * [ (1+z^-1) * prod_{even i} Q_i + (1-z^-1) * prod_{odd i} Q_i ]
    // with Q_i = 1 - 2 cos(lsf_i) z^-1 + z^-2.
    let mut prod_even = vec![1.0f64];
    let mut prod_odd = vec![1.0f64];
    for (i, &w) in lsf.iter().enumerate() {
        let quad = [1.0, -2.0 * w.cos(), 1.0];
        if i % 2 == 0 {
            prod_even = poly_mul(&prod_even, &quad);
        } else {
            prod_odd = poly_mul(&prod_odd, &quad);
        }
    }
    let p_poly = poly_mul(&prod_even, &[1.0, 1.0]);
    let q_poly = poly_mul(&prod_odd, &[1.0, -1.0]);

    // The z^-(p+1) terms cancel, so only indices 0..=p are meaningful.
    let mut a: Vec<f64> = (0..=n).map(|j| 0.5 * (p_poly[j] + q_poly[j])).collect();
    let a0 = a[0];
    if a0.abs() < 1e-12 {
        return None;
    }
    if (a0 - 1.0).abs() > 1e-9 {
        for v in &mut a {
            *v /= a0;
        }
    }
    Some((1..=n).map(|j| -a[j]).collect())
}

/// Sort, clamp into `[MIN_LSF, PI - MIN_LSF]`, and enforce at least
/// [`MIN_SEPARATION`] radians between consecutive LSFs (pushing values apart
/// deterministically from a running lower bound). In-place and idempotent on
/// already-stable input.
pub fn stabilize_lsf(lsf: &mut [f64]) {
    for v in lsf.iter_mut() {
        if !v.is_finite() {
            *v = MIN_LSF;
        }
    }
    lsf.sort_by(f64::total_cmp);

    let mut prev = f64::NEG_INFINITY;
    for v in lsf.iter_mut() {
        let mut value = v.clamp(MIN_LSF, PI_MINUS_MIN_LSF);
        if prev.is_finite() && value < prev + MIN_SEPARATION {
            value = prev + MIN_SEPARATION;
        }
        *v = value;
        prev = value;
    }

    // If the running floor pushed the tail past the upper bound (only possible
    // when the vector is over-crowded), pull the tail back and re-separate from
    // the top. `p <= 16` gives ample room, so this is a guard, not a path.
    if let Some(last) = lsf.len().checked_sub(1)
        && lsf[last] > PI_MINUS_MIN_LSF
    {
        lsf[last] = PI_MINUS_MIN_LSF;
        for i in (0..last).rev() {
            let ceil = lsf[i + 1] - MIN_SEPARATION;
            if lsf[i] > ceil {
                lsf[i] = ceil;
            }
        }
    }
}

/// True iff strictly ordered with at least [`MIN_SEPARATION`] between
/// consecutive entries and every entry inside `(0, pi)` with the [`MIN_LSF`]
/// margin (`MIN_LSF <= v <= PI - MIN_LSF`).
pub fn is_stable_lsf(lsf: &[f64]) -> bool {
    if lsf.iter().any(|v| !v.is_finite()) {
        return false;
    }
    if lsf
        .iter()
        .any(|v| !(MIN_LSF..=PI_MINUS_MIN_LSF).contains(v))
    {
        return false;
    }
    lsf.windows(2).all(|ws| ws[1] - ws[0] >= MIN_SEPARATION)
}

// ---------------------------------------------------------------------------
// Polynomial helpers
// ---------------------------------------------------------------------------

/// `C(z) / (1 + z^-1)` for coefficients given ascending in `z^-i`.
fn divide_one_plus(c: &[f64]) -> Vec<f64> {
    let n = c.len();
    let mut q = vec![0.0f64; n];
    q[0] = c[0];
    for i in 1..n {
        q[i] = c[i] - q[i - 1];
    }
    q
}

/// `C(z) / (1 - z^-1)` for coefficients given ascending in `z^-i`.
fn divide_one_minus(c: &[f64]) -> Vec<f64> {
    let n = c.len();
    let mut q = vec![0.0f64; n];
    q[0] = c[0];
    for i in 1..n {
        q[i] = c[i] + q[i - 1];
    }
    q
}

/// `C1(w)` (or `C2(w)`) for a degree-`2m` palindromic polynomial `s` (ascending
/// in `z^-i`, length `2m+1`):
/// `s[m] + 2 * sum_{k=1..m} s[m-k] * cos(k w)`, accumulated with compensation.
fn cosine_sum(s: &[f64], m: usize, w: f64) -> f64 {
    let mut sum = s[m];
    let mut comp = 0.0f64;
    for k in 1..=m {
        compensated(&mut sum, &mut comp, 2.0 * s[m - k] * (k as f64 * w).cos());
    }
    sum + comp
}

/// Compensated (Neumaier) accumulation of `x` into `sum`, error in `comp`.
fn compensated(sum: &mut f64, comp: &mut f64, x: f64) {
    let t = *sum + x;
    if sum.abs() >= x.abs() {
        *comp += (*sum - t) + x;
    } else {
        *comp += (x - t) + *sum;
    }
    *sum = t;
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Real roots of `f` strictly inside `(0, pi)`: dense uniform sweep in the angle
/// to bracket sign changes, then bisection. Roots of the stable filters this
/// crate produces are simple, so a sign change is a valid bracket.
fn roots_of(f: &impl Fn(f64) -> f64) -> Vec<f64> {
    let mut roots = Vec::new();
    let step = PI / ROOT_SAMPLES as f64;
    let mut prev_w = 0.0f64;
    let mut prev_f = f(0.0);
    if prev_f == 0.0 {
        // `w = 0` is a boundary, not a valid LSF; probe just inside so an
        // interior root in the first cell is not lost.
        prev_w = step * 1e-6;
        prev_f = f(prev_w);
    }
    for j in 1..=ROOT_SAMPLES {
        let w = if j == ROOT_SAMPLES {
            PI
        } else {
            step * j as f64
        };
        let val = f(w);
        if prev_f == 0.0 {
            if prev_w > 0.0 {
                roots.push(prev_w);
            }
        } else if val != 0.0 && (prev_f < 0.0) != (val < 0.0) {
            let mut lo = prev_w;
            let mut hi = w;
            for _ in 0..BISECTION_ITERS {
                let mid = 0.5 * (lo + hi);
                if (prev_f < 0.0) != (f(mid) < 0.0) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            roots.push(0.5 * (lo + hi));
        }
        prev_w = w;
        prev_f = val;
    }
    roots.retain(|&w| w > BOUNDARY_EPS && w < PI - BOUNDARY_EPS);
    roots
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lossy::predict::reflection_to_weights;

    /// Deterministic xorshift64 stand-in; no external dependency.
    struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self { state: seed | 1 }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        /// Uniform in `[0, 1)`.
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn random_stable(seed: u64, order: usize, amp: f64) -> Vec<f64> {
        let mut rng = XorShift64::new(seed);
        (0..order).map(|_| (rng.unit() * 2.0 - 1.0) * amp).collect()
    }

    fn assert_round_trip(k: &[f64]) {
        let lsf = reflections_to_lsf(k).unwrap_or_else(|| panic!("no LSF for {k:?}"));
        assert_eq!(lsf.len(), k.len());
        assert!(
            lsf.windows(2).all(|ws| ws[1] > ws[0]),
            "LSFs not strictly increasing: {lsf:?}"
        );
        assert!(
            lsf.iter().all(|&w| w > 0.0 && w < PI),
            "LSF outside (0, pi): {lsf:?}"
        );

        let got = lsf_to_weights(&lsf).expect("ordered LSF inverts");
        let want = reflection_to_weights(k);
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "order {} component {i}: k={k:?} lsf={lsf:?} got {a} want {b}",
                k.len()
            );
        }
    }

    #[test]
    fn round_trip_covers_representative_orders() {
        for order in [2usize, 4, 8, 10, 12, 16] {
            assert_round_trip(&vec![0.0; order]);

            let small: Vec<f64> = (0..order)
                .map(|i| if i % 2 == 0 { 0.1 } else { -0.05 })
                .collect();
            assert_round_trip(&small);

            // Mixed-sign coefficients of near-unit magnitude: this is the
            // boundary stress that stays numerically well-posed at every order.
            let boundary: Vec<f64> = (0..order)
                .map(|i| 0.99 * ((i as f64) * 1.7).sin())
                .collect();
            assert_round_trip(&boundary);
        }

        // All-identical near-marginal filters are resolvable at moderate order.
        // At higher order every pole crowds the unit circle, the LSFs cluster
        // below double-precision resolution, and `reflections_to_lsf` correctly
        // reports the numerical failure with `None` (see the test below).
        for order in [2usize, 4, 8] {
            assert_round_trip(&vec![0.99; order]);
            assert_round_trip(&vec![-0.99; order]);
        }
    }

    #[test]
    fn extreme_clustered_filters_fail_explicitly_or_round_trip() {
        // Near-marginal all-`-0.99` filters crowd `p` poles against the unit
        // circle; past order ~10 the LSF clusters fall below double precision.
        // The converter must then return `None` rather than fabricate or panic.
        for order in [10usize, 12, 16] {
            let k = vec![-0.99f64; order];
            match reflections_to_lsf(&k) {
                None => {}
                Some(lsf) => {
                    assert!(lsf.windows(2).all(|ws| ws[1] > ws[0]));
                    assert!(lsf.iter().all(|&w| w > 0.0 && w < PI));
                }
            }
        }
    }

    #[test]
    fn round_trip_covers_random_stable_vectors() {
        for order in [2usize, 4, 8, 10, 12, 16] {
            for seed in 1u64..=25 {
                let k = random_stable(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), order, 0.99);
                assert_round_trip(&k);
            }
        }
    }

    #[test]
    fn a_known_order_two_filter_is_reconstructed() {
        let k = [2.0 / 3.0, 0.25];
        let lsf = reflections_to_lsf(&k).expect("stable");
        let got = lsf_to_weights(&lsf).expect("ordered");
        let want = reflection_to_weights(&k);
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, want {b}");
        }
    }

    #[test]
    fn a_flat_order_two_filter_has_the_analytic_lsf_pair() {
        let lsf = reflections_to_lsf(&[0.0, 0.0]).expect("stable");
        assert!((lsf[0] - PI / 3.0).abs() < 1e-9, "got {:?}", lsf);
        assert!((lsf[1] - 2.0 * PI / 3.0).abs() < 1e-9, "got {:?}", lsf);
    }

    #[test]
    fn reflection_conversion_rejects_unstable_or_odd_orders() {
        assert!(reflections_to_lsf(&[0.5]).is_none()); // odd order
        assert!(reflections_to_lsf(&[]).is_none());
        assert!(reflections_to_lsf(&[1.0, 0.0]).is_none());
        assert!(reflections_to_lsf(&[0.0, -1.0]).is_none());
        assert!(reflections_to_lsf(&[1.5, 0.0]).is_none());
        assert!(reflections_to_lsf(&[f64::NAN, 0.0]).is_none());
        assert!(reflections_to_lsf(&[f64::INFINITY, 0.0]).is_none());
    }

    #[test]
    fn lsf_inversion_rejects_unordered_vectors() {
        assert!(lsf_to_weights(&[1.0, 0.5]).is_none()); // reversed
        assert!(lsf_to_weights(&[0.5, 0.5]).is_none()); // not strict
        assert!(lsf_to_weights(&[0.5]).is_none()); // odd length
        assert!(lsf_to_weights(&[]).is_none());
        assert!(lsf_to_weights(&[0.5, f64::NAN]).is_none());
    }

    #[test]
    fn stability_predicate_rejects_bad_vectors() {
        let good = [0.5f64, 0.8, 1.2, 1.6];
        assert!(is_stable_lsf(&good));

        let mut reversed = good;
        reversed.reverse();
        assert!(!is_stable_lsf(&reversed));

        assert!(!is_stable_lsf(&[0.0, 1.0, 2.0, 3.0]));
        assert!(!is_stable_lsf(&[0.5, 0.8, 1.2, PI]));
        assert!(!is_stable_lsf(&[0.5, 0.8, 1.2, PI_MINUS_MIN_LSF + 0.01]));

        // Consecutive entries closer than MIN_SEPARATION.
        assert!(!is_stable_lsf(&[0.5, 0.5 + MIN_SEPARATION * 0.5]));

        assert!(!is_stable_lsf(&[f64::NAN, 1.0]));
    }

    #[test]
    fn stabilize_repairs_and_is_idempotent() {
        // Unordered, out-of-range, and too-close in one vector.
        let mut messy = [2.5f64, -0.4, 1.0, 1.0 + MIN_SEPARATION * 0.25, PI + 1.0];
        stabilize_lsf(&mut messy);
        assert!(is_stable_lsf(&messy), "stabilised to {messy:?}");

        let mut again = messy;
        stabilize_lsf(&mut again);
        assert_eq!(messy, again, "not idempotent: {messy:?} -> {again:?}");

        // Already-stable distinct values keep their order and spacing.
        let mut stable = [0.3f64, 0.9, 1.7, 2.4];
        let before = stable;
        stabilize_lsf(&mut stable);
        assert_eq!(stable, before);
    }

    #[test]
    fn stabilize_maps_non_finite_entries_into_range() {
        let mut v = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0];
        stabilize_lsf(&mut v);
        assert!(is_stable_lsf(&v), "stabilised to {v:?}");
    }
}
