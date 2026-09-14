//! Phase 7C.2-G: an orthonormal in-frame transform and a PVQ shape quantiser.
//!
//! # Why not the existing MDCT
//!
//! `crate::lossy::mdct::Mdct` is a **framing** transform: `n` coefficients from
//! a `2n`-sample windowed input, reconstructed by windowed overlap-add across
//! frames. Its synthesis returns `w[m]·x[m]`, so recovering the input in-frame
//! means dividing by the sine window, whose smallest value is `sin(π/4n) ≈
//! π/4n` at the frame edges. At `n = 160` that is a ≈200× amplification of edge
//! quantisation noise — precisely the aliasing the overlap-add exists to cancel.
//! A per-frame escape excitation has no overlap-add, so it cannot use it.
//!
//! This module therefore provides what the escape mode actually needs: a
//! **self-inverse orthonormal** transform (DCT-IV) with no window and no
//! cross-frame state, so packet independence is untouched, and a PVQ
//! quantiser whose cardinality is analytic.
//!
//! # Why PVQ
//!
//! Per the charter §12: PVQ's codebook cardinality is known in closed form, so a
//! pulse count `K` fixes the exact transmitted rate without a learned
//! probability table. That is what makes it a clean fit for a hard-bit-budget
//! selector. The frame is split into fixed blocks so the index width stays
//! inside a `u32`, which keeps the wire writer's exact accounting simple.

/// Samples per PVQ block.
///
/// Deliberately the codec's own 5 ms subframe ([`crate::voice::celp::SUB_LEN`]).
/// The block size sets the index cost per pulse: at 40 samples a *single* pulse
/// already costs 7 bits, because the codebook must also express "this block's
/// shape", so small blocks are bit-inefficient at low rates. At 80 samples one
/// pulse costs 8 bits *for a whole 5 ms block*, which is what lets the escape
/// mode reach 6 kbps at all.
pub const BLOCK: usize = 80;
/// Largest pulses per block whose index still fits the wire's 32-bit field.
///
/// The ceiling is the index width, not the quantiser's quality: `V(80,6)`
/// already exceeds `2^32`. Verified by test rather than assumed.
pub const MAX_PULSES: usize = 5;
/// Bits of the per-frame pulse-count field.
pub const K_BITS: u8 = 3;

/// Orthonormal DCT-IV. The transform is its own inverse (`C·C = I`), so
/// [`dct4`] *is* the inverse and the reconstruction needs no second matrix.
///
/// `t[k] = sqrt(2/N) Σ_m x[m]·cos(π/N·(m+½)(k+½))`. Orthonormal means the
/// coefficient vector has exactly the input's energy, so PVQ's unit-norm shape
/// and the transmitted gain reconstruct the residual with no hidden scale
/// factor — which is what lets the encoder measure the true distortion.
pub fn dct4(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let s = (2.0 / n as f64).sqrt();
    let mut out = vec![0.0f64; n];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (m, &xm) in x.iter().enumerate() {
            acc +=
                xm * (std::f64::consts::PI / n as f64 * (m as f64 + 0.5) * (k as f64 + 0.5)).cos();
        }
        *slot = s * acc;
    }
    out
}

/// The inverse, which for DCT-IV is the transform itself.
pub fn idct4(c: &[f64]) -> Vec<f64> {
    dct4(c)
}

/// `V(n, k)`: the number of length-`n` integer vectors with `Σ|x_i| = k`, which
/// is the PVQ codebook size at that rate. Saturating, so an out-of-range `k`
/// cannot panic — but [`MAX_PULSES`] is chosen so the values that matter fit.
///
/// Recurrence: the first coordinate is `0` (`V(n−1,k)` ways) or has magnitude
/// drawn from `1..=k`, each magnitude with two signs. Collapsing the sum gives
/// `V(n,k) = V(n−1,k) + V(n,k−1) + V(n−1,k−1)`.
pub fn count(n: usize, k: usize) -> u128 {
    if k == 0 {
        return 1;
    }
    if n == 0 {
        return 0;
    }
    let mut prev = vec![0u128; k + 1];
    prev[0] = 1;
    for _ in 0..n {
        let mut cur = vec![0u128; k + 1];
        cur[0] = 1;
        for j in 1..=k {
            cur[j] = prev[j]
                .saturating_add(cur[j - 1])
                .saturating_add(prev[j - 1]);
        }
        prev = cur;
    }
    prev[k]
}

/// Bits the PVQ index of a `k`-pulse block occupies.
pub fn index_bits(k: usize) -> u8 {
    let v = count(BLOCK, k);
    if v <= 1 { 0 } else { (v - 1).ilog2() as u8 + 1 }
}

/// Total innovation bits for `blocks` blocks at `k` pulses each.
pub fn shape_bits(blocks: usize, k: usize) -> usize {
    blocks * usize::from(index_bits(k))
}

/// Rank a signed vector with `Σ|x_i| = k` onto `0..V(n,k)`.
///
/// Ordering: at each position, `0` first, then magnitudes ascending, and within
/// a magnitude the positive sign before the negative.
pub fn rank(y: &[i32]) -> u128 {
    let n = y.len();
    let mut idx = 0u128;
    let mut k = y.iter().map(|v| v.unsigned_abs() as usize).sum::<usize>();
    for (i, &u) in y.iter().enumerate() {
        if u == 0 {
            continue;
        }
        let rest = n - i - 1;
        // Every vector that keeps this position at zero sorts before this one.
        idx += count(rest, k);
        let m = u.unsigned_abs() as usize;
        for j in 1..m {
            idx += 2 * count(rest, k - j);
        }
        if u < 0 {
            idx += count(rest, k - m);
        }
        k -= m;
    }
    idx
}

/// Inverse of [`rank`]: the signed vector for a `k`-pulse index.
pub fn unrank(idx: u128, n: usize, k: usize) -> Vec<i32> {
    let mut y = vec![0i32; n];
    let mut idx = idx;
    let mut kk = k;
    for (i, slot) in y.iter_mut().enumerate() {
        if kk == 0 {
            break;
        }
        let rest = n - i - 1;
        let zero = count(rest, kk);
        if idx < zero {
            continue;
        }
        idx -= zero;
        let mut m = 1usize;
        while m <= kk {
            let c = count(rest, kk - m);
            let group = 2 * c;
            if idx < group {
                if idx < c {
                    *slot = m as i32;
                } else {
                    *slot = -(m as i32);
                    idx -= c;
                }
                kk -= m;
                break;
            }
            idx -= group;
            m += 1;
        }
    }
    y
}

/// Greedy PVQ shape search: place `k` unit pulses so the unit-norm shape
/// `y/‖y‖` best matches `x` in the normalised-correlation sense.
///
/// The criterion for adding a signed unit pulse `d` at position `i` is
/// `(x·y')² / (y'·y')`, maintained incrementally so a step is `O(n)`. The
/// search is deterministic and always returns exactly `k` pulses, which is what
/// makes the rate exactly predictable from `k` alone.
pub fn encode_shape(x: &[f64], k: usize) -> Vec<i32> {
    let n = x.len();
    let mut y = vec![0i32; n];
    if n == 0 || k == 0 {
        return y;
    }
    let mut xy = 0.0f64;
    let mut yy = 0.0f64;
    for _ in 0..k {
        let mut best = (0usize, 1i32, f64::NEG_INFINITY);
        for i in 0..n {
            // `nxy²` is identical for both signs, so the score ties. Try the
            // sign aligned with the target first and keep strict `>`, so the tie
            // breaks toward the correlation the pulse is supposed to capture.
            let pref: [i32; 2] = if x[i] >= 0.0 { [1, -1] } else { [-1, 1] };
            for d in pref {
                let fd = f64::from(d);
                let nxy = xy + fd * x[i];
                let nyy = yy + 2.0 * fd * f64::from(y[i]) + 1.0;
                if nyy <= 0.0 {
                    continue;
                }
                let score = nxy * nxy / nyy;
                if score > best.2 {
                    best = (i, d, score);
                }
            }
        }
        let (i, d, _) = best;
        y[i] += d;
        let fd = f64::from(d);
        xy += fd * x[i];
        yy += 2.0 * fd * f64::from(y[i] - d) + 1.0;
    }
    y
}

/// Split a signal into `BLOCK`-sample blocks.
pub fn blocks(signal: &[f64]) -> Vec<&[f64]> {
    signal.chunks(BLOCK).collect()
}

/// Transform every block and return the flattened coefficients, so PVQ operates
/// on `BLOCK`-dimensional subspaces.
pub fn forward_blocks(signal: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(signal.len());
    for b in signal.chunks(BLOCK) {
        out.extend_from_slice(&dct4(b));
    }
    out
}

/// Inverse of [`forward_blocks`].
pub fn inverse_blocks(coeffs: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(coeffs.len());
    for b in coeffs.chunks(BLOCK) {
        out.extend_from_slice(&idct4(b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct4_is_orthonormal_and_self_inverse() {
        let x: Vec<f64> = (0..BLOCK)
            .map(|i| 300.0 * (0.31 * i as f64).sin() + 120.0 * (1.7 * i as f64).cos())
            .collect();
        let c = dct4(&x);
        // Energy preserving.
        let ex: f64 = x.iter().map(|v| v * v).sum();
        let ec: f64 = c.iter().map(|v| v * v).sum();
        assert!((ex - ec).abs() <= 1e-9 * ex.max(1.0), "{ex} vs {ec}");
        // Self-inverse.
        let back = idct4(&c);
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0));
        }
        // Orthonormal basis check on an impulse.
        let mut e = vec![0.0f64; BLOCK];
        e[7] = 1.0;
        let ce = dct4(&e);
        assert!((ce.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pvq_count_matches_the_recurrence_and_the_expected_small_values() {
        // Hand-checkable: one pulse in n dimensions has 2n signed vectors.
        for n in 1..=12 {
            assert_eq!(count(n, 1), 2 * n as u128);
            assert_eq!(count(n, 0), 1);
        }
        assert_eq!(count(0, 0), 1);
        assert_eq!(count(0, 3), 0);
        // V(2,2) = [+-2,0] x2, [0,+-2] x2, [+-1,+-1] x4 = 8.
        assert_eq!(count(2, 2), 8);
        // The shipped pulse cap must fit the wire's 32-bit index field.
        for k in 0..=MAX_PULSES {
            assert!(
                count(BLOCK, k) <= u128::from(u32::MAX) + 1,
                "k={k} index does not fit u32 ({} values)",
                count(BLOCK, k)
            );
            if k > 0 {
                assert_eq!(index_bits(k), (count(BLOCK, k) - 1).ilog2() as u8 + 1);
            } else {
                assert_eq!(index_bits(0), 0);
            }
        }
        // Bits grow with pulses, monotonically.
        let mut prev = 0u8;
        for k in 0..=MAX_PULSES {
            let b = index_bits(k);
            assert!(b >= prev, "index_bits must be monotone");
            prev = b;
        }
    }

    #[test]
    fn rank_and_unrank_are_a_bijection() {
        // Allocation discipline: this test must never build a set proportional to
        // the codebook size in *heap objects*. It stores one bit per reachable
        // index in a flat bit vec, and the total is hard-capped, so peak memory
        // is bounded by `MAX_INDICES` bytes (1 byte per `bool`) — a few MB at
        // worst — no matter what the recurrence returns.
        const MAX_INDICES: u128 = 4_000_000;
        for n in [1usize, 2, 3, 4, 5, BLOCK] {
            for k in 0..=if n <= 5 { 4 } else { 3 } {
                let total = count(n, k);
                assert!(
                    total <= MAX_INDICES,
                    "n={n} k={k} would need {total} indices, above the {MAX_INDICES} test cap"
                );
                let total = total as usize;
                let mut seen = vec![false; total];
                for idx in 0..total as u128 {
                    let y = unrank(idx, n, k);
                    assert_eq!(y.len(), n);
                    assert_eq!(
                        y.iter().map(|v| v.unsigned_abs() as usize).sum::<usize>(),
                        k,
                        "unrank must land on the L1 sphere"
                    );
                    let back = rank(&y);
                    assert_eq!(back, idx, "round trip failed at n={n} k={k} idx={idx}");
                    // Distinctness without a second heap structure: two different
                    // indices producing the same rank would be a collision.
                    let slot = &mut seen[back as usize];
                    assert!(!*slot, "rank {back} produced twice at n={n} k={k}");
                    *slot = true;
                }
                assert!(seen.iter().all(|s| *s), "some index was unreachable");
            }
        }
    }

    #[test]
    fn greedy_shape_places_exactly_k_pulses_and_improves_monotonically() {
        let x: Vec<f64> = (0..BLOCK).map(|i| (0.4 * i as f64).sin()).collect();
        let nx: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let corr = |y: &[i32]| -> f64 {
            let ny: f64 = y.iter().map(|v| f64::from(v * v)).sum::<f64>().sqrt();
            if ny <= 0.0 {
                return 0.0;
            }
            x.iter().zip(y).map(|(a, b)| a * f64::from(*b)).sum::<f64>() / (nx * ny)
        };
        let mut prev = -1.0;
        for k in 0..=MAX_PULSES {
            let y = encode_shape(&x, k);
            assert_eq!(
                y.iter().map(|v| v.unsigned_abs() as usize).sum::<usize>(),
                k,
                "exactly k pulses are placed"
            );
            let c = corr(&y);
            assert!(c >= prev - 1e-12, "correlation fell to {c} at k={k}");
            prev = c;
        }
        // More pulses is a real improvement, not a wash.
        assert!(
            corr(&encode_shape(&x, MAX_PULSES)) > corr(&encode_shape(&x, 1)) + 0.1,
            "the pulse budget must buy shape accuracy"
        );
        // A target that is itself a PVQ vector must be recovered exactly: this
        // is the sharp statement of what the greedy search is supposed to do.
        let mut t = vec![0.0f64; BLOCK];
        t[3] = 1.0;
        t[10] = -2.0;
        t[20] = 1.0;
        let want: Vec<i32> = (0..BLOCK)
            .map(|i| match i {
                3 => 1,
                10 => -2,
                20 => 1,
                _ => 0,
            })
            .collect();
        assert_eq!(encode_shape(&t, 4), want, "k-sparse targets must be exact");
        // The first pulse must sit at the largest magnitude and take its sign.
        // The score `(x_i)²` is identical for both signs, so a search that does
        // not break the tie toward the target picks the wrong one.
        let argmax = x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .unwrap();
        let y1 = encode_shape(&x, 1);
        assert_eq!(y1.iter().position(|v| *v != 0), Some(argmax));
        assert_eq!(y1[argmax], if x[argmax] >= 0.0 { 1 } else { -1 });
        // A negated target must produce a negated pulse.
        let neg: Vec<f64> = x.iter().map(|v| -v).collect();
        let yn = encode_shape(&neg, 1);
        assert!(yn.iter().all(|v| *v <= 0), "sign must follow the target");
    }

    #[test]
    fn block_transforms_round_trip_and_stay_within_a_block() {
        let signal: Vec<f64> = (0..320).map(|i| 500.0 * (0.07 * i as f64).sin()).collect();
        let c = forward_blocks(&signal);
        assert_eq!(c.len(), signal.len());
        let back = inverse_blocks(&c);
        for (a, b) in signal.iter().zip(&back) {
            assert!((a - b).abs() <= 1e-8 * a.abs().max(1.0));
        }
        // A change in one coefficient must not leak into a neighbouring block.
        let mut perturbed = c.clone();
        perturbed[BLOCK + 3] += 1.0;
        let p = inverse_blocks(&perturbed);
        for (i, (a, b)) in back.iter().zip(&p).enumerate() {
            if i < BLOCK {
                assert_eq!(a, b, "block 0 leaked at {i}");
            }
        }
    }
}
