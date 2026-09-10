//! Shared period-scan semantics (Phase L) — `no_std`, compiled for the host
//! (parity reference) and for both device targets.
//!
//! The inverse compiler's dominant search cost is the **bounded period scan**:
//! for every candidate period `p`, how many frames disagree with the periodic
//! hypothesis `cycle = X[0..p]`? For mono content that count *is* the periodic
//! residual record count, because `ResidualModel::Periodic` is mono and
//! `channel 0`'s delta is `X[f] - cycle[f % p]`, which is zero exactly when the
//! frames agree. The scan is therefore the exact p-dependent part of
//! `Residual::closing_residual` — nothing is approximated.
//!
//! Because the host reference and the device kernels call *this same
//! function*, `scalar == parallel == CUDA == ROCm` on the scan is a structural
//! property rather than a coincidence. `court inverse-search` re-verifies it
//! anyway, and every proposal the scan ranks is re-verified by the normative
//! exact evaluator: the scan has **zero authority**.

/// Sentinel for a period whose periodic hypothesis cannot close the window:
/// at least one delta `X[f] - X[f % p]` falls outside the i32 code domain, so
/// `Residual::new` would reject the object. Real counts are bounded by the
/// frame count (≤ `MAX_QUANTUM_FRAMES`), so the sentinel is unambiguous.
pub const PERIOD_NOT_CLOSEABLE: u32 = u32::MAX;

/// True when `a - b` fits the i32 residual delta domain (the Phase-E closure
/// rule).
#[inline]
pub fn delta_fits_i32(a: i32, b: i32) -> bool {
    let d = (a as i64) - (b as i64);
    d >= i32::MIN as i64 && d <= i32::MAX as i64
}

/// Exact periodic-hypothesis residual record count for `period` over
/// `x[..frames]` (mono channel 0), or [`PERIOD_NOT_CLOSEABLE`].
///
/// Frames `f < period` compare against themselves (`X[f % p] == X[f]`), so the
/// scan starts at `f == period`. Identical to the record count
/// `Residual::closing_residual(x, 1, Periodic{x[..p]})` produces.
pub fn period_records(x: &[i32], frames: u32, period: u32) -> u32 {
    if period == 0 || frames < 2 || period >= frames {
        return PERIOD_NOT_CLOSEABLE;
    }
    let mut count = 0u32;
    let mut f = period;
    while f < frames {
        let a = x[f as usize];
        let b = x[(f % period) as usize];
        if !delta_fits_i32(a, b) {
            return PERIOD_NOT_CLOSEABLE;
        }
        if a != b {
            count += 1;
        }
        f += 1;
    }
    count
}

/// Fill `out[i]` with the record count for period `i + 1` (index 0 = period 1)
/// for `i < min(period_limit, frames - 1, out.len())`. Returns how many cells
/// were written. Periods `>= frames` are never scanned: they carry no
/// information (a period equal to the frame count would trivially "close").
pub fn scan_into(x: &[i32], frames: u32, period_limit: u32, out: &mut [u32]) -> u32 {
    if frames < 2 || period_limit == 0 {
        return 0;
    }
    let limit = (period_limit as usize)
        .min(frames as usize - 1)
        .min(out.len());
    for (i, slot) in out[..limit].iter_mut().enumerate() {
        *slot = period_records(x, frames, (i + 1) as u32);
    }
    limit as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_domain_matches_the_i32_residual_rule() {
        assert!(delta_fits_i32(0, 0));
        assert!(delta_fits_i32(i32::MAX, 0));
        assert!(delta_fits_i32(i32::MIN, 0));
        assert!(!delta_fits_i32(i32::MAX, -1));
        assert!(!delta_fits_i32(i32::MIN, 1));
    }

    #[test]
    fn exact_period_reports_zero_records() {
        // Perfect period 8.
        let x: Vec<i32> = (0..256).map(|i| (i % 8) * 1000).collect();
        assert_eq!(period_records(&x, 256, 8), 0);
        // Periods that do not divide the pattern disagree somewhere.
        assert!(period_records(&x, 256, 7) > 0);
    }

    #[test]
    fn non_closeable_periods_are_flagged_not_counted() {
        // Alternating extremes: delta can exceed i32.
        let x = vec![i32::MAX, i32::MIN];
        assert_eq!(period_records(&x, 2, 1), PERIOD_NOT_CLOSEABLE);
        // A period >= frames carries no information.
        assert_eq!(period_records(&x, 2, 2), PERIOD_NOT_CLOSEABLE);
        assert_eq!(period_records(&x, 2, 0), PERIOD_NOT_CLOSEABLE);
    }

    #[test]
    fn scan_covers_only_meaningful_periods() {
        let x: Vec<i32> = (0..16).map(|i| i * 3).collect();
        let mut out = vec![0u32; 32];
        let n = scan_into(&x, 16, 32, &mut out);
        // bounded by frames - 1
        assert_eq!(n, 15);
        assert_eq!(out[0], 15, "period 1 disagrees on 15 of 16 frames");
        for slot in &out[n as usize..] {
            assert_eq!(*slot, 0, "cells beyond the scan stay untouched");
        }
    }

    #[test]
    fn scan_is_bounded_by_the_output_buffer() {
        let x: Vec<i32> = (0..64).collect();
        let mut out = vec![0u32; 3];
        assert_eq!(scan_into(&x, 64, 512, &mut out), 3);
    }

    #[test]
    fn empty_and_degenerate_inputs_are_safe() {
        let mut out = vec![0u32; 4];
        assert_eq!(scan_into(&[], 0, 8, &mut out), 0);
        assert_eq!(scan_into(&[1], 1, 8, &mut out), 0);
        assert_eq!(scan_into(&[1, 2], 2, 0, &mut out), 0);
    }
}
