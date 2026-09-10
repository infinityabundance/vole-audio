//! Bounded period-scan search surfaces (Phase L).
//!
//! The inverse compiler's dominant search cost is the bounded period scan
//! (see `device::search_shared`). This module owns the **host** surfaces for
//! that scan and the deterministic ranking rule, so the same ranking can be
//! fed by a sequential scan, a multi-threaded scan, or a device kernel:
//!
//! ```text
//! scalar      one thread, sequential periods
//! parallel    host threads over disjoint period ranges
//! CUDA        backend::cuda::SearchWorld  (one device thread per period)
//! ROCm        backend::rocm::SearchWorldRocm
//! ```
//!
//! Placement is a **performance** question only: every surface calls the same
//! `period_records` semantics and must produce identical counts, and every
//! proposal the ranking yields is re-verified by the normative exact evaluator.
//! `court inverse-search` asserts both.

use crate::device::search_shared::{PERIOD_NOT_CLOSEABLE, scan_into};
use crate::error::{Error, Result};

/// Which surface produced a scan (evidence label only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanSurface {
    Scalar,
    Parallel { threads: usize },
    Cuda,
    Rocm,
}

impl ScanSurface {
    pub const fn label(self) -> &'static str {
        match self {
            ScanSurface::Scalar => "scalar",
            ScanSurface::Parallel { .. } => "parallel",
            ScanSurface::Cuda => "cuda",
            ScanSurface::Rocm => "rocm",
        }
    }
}

/// A completed period scan: `counts[i]` is the exact periodic residual record
/// count for period `i + 1`, or [`PERIOD_NOT_CLOSEABLE`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeriodScan {
    pub counts: Vec<u32>,
}

impl PeriodScan {
    /// Number of candidate periods covered (`min(period_limit, frames - 1)`).
    pub fn limit(&self) -> u32 {
        self.counts.len() as u32
    }

    /// Ranked candidates: `(period, records)` for closeable periods, ordered
    /// by (record count ascending, period ascending) and truncated to `keep`,
    /// then returned in ascending period order (deterministic).
    pub fn ranked(&self, keep: usize) -> Vec<(u32, u32)> {
        rank_periods(&self.counts, keep)
    }

    /// The periods a bounded search would propose, ascending.
    pub fn periods(&self, keep: usize) -> Vec<u32> {
        self.ranked(keep).into_iter().map(|(p, _)| p).collect()
    }
}

/// The frozen ranking rule: closeable periods sorted by (record count,
/// period), truncated to `keep`, then re-sorted by period for determinism.
pub fn rank_periods(counts: &[u32], keep: usize) -> Vec<(u32, u32)> {
    if keep == 0 {
        return Vec::new();
    }
    let mut rows: Vec<(u32, u32)> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != PERIOD_NOT_CLOSEABLE)
        .map(|(i, &c)| ((i + 1) as u32, c))
        .collect();
    rows.sort_by_key(|&(period, records)| (records, period));
    rows.truncate(keep);
    rows.sort_by_key(|&(period, _)| period);
    rows
}

/// Sequential host scan (the reference surface).
pub fn scan_scalar(x: &[i32], frames: u32, period_limit: u32) -> PeriodScan {
    let limit = scan_bound(frames, period_limit);
    let mut counts = vec![0u32; limit as usize];
    scan_into(x, frames, period_limit, &mut counts);
    PeriodScan { counts }
}

/// Multi-threaded host scan. Every period is computed independently from the
/// same input, so the result is identical to [`scan_scalar`] for any thread
/// count (asserted by `court inverse-search`).
pub fn scan_parallel(
    x: &[i32],
    frames: u32,
    period_limit: u32,
    threads: usize,
) -> Result<PeriodScan> {
    let limit = scan_bound(frames, period_limit);
    if threads <= 1 || limit <= 1 {
        return Ok(scan_scalar(x, frames, period_limit));
    }
    let workers = threads.min(limit as usize);
    let mut counts = vec![0u32; limit as usize];
    let chunk = (limit as usize).div_ceil(workers);
    std::thread::scope(|scope| {
        for (index, out) in counts.chunks_mut(chunk).enumerate() {
            let base = index * chunk;
            let x = &x[..frames as usize];
            scope.spawn(move || {
                for (i, slot) in out.iter_mut().enumerate() {
                    let period = (base + i + 1) as u32;
                    *slot = crate::device::search_shared::period_records(x, frames, period);
                }
            });
        }
    });
    Ok(PeriodScan { counts })
}

/// The scan range for a window: `min(period_limit, frames - 1)`, never 0 for a
/// usable input.
pub fn scan_bound(frames: u32, period_limit: u32) -> u32 {
    if frames < 2 {
        return 0;
    }
    period_limit.min(frames - 1)
}

/// Validate a window/period request before touching any surface.
pub fn validate_request(x: &[i32], frames: u32, period_limit: u32) -> Result<()> {
    if frames < 2 {
        return Err(Error::malformed("period scan needs at least 2 frames"));
    }
    if x.len() != frames as usize {
        return Err(Error::malformed("period-scan window length mismatch"));
    }
    if period_limit == 0 {
        return Err(Error::malformed("period scan needs a nonzero period bound"));
    }
    Ok(())
}

/// True when two scans produced identical per-period counts (the cross-surface
/// equality property search placement must never break).
pub fn scans_agree(a: &PeriodScan, b: &PeriodScan) -> bool {
    a.counts == b.counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::corpus;
    use crate::object::residual::{Residual, ResidualModel};

    /// The scan must equal the normative residual construction exactly: for
    /// every period, the record count is the length of the residual that
    /// `Residual::closing_residual` produces, and non-closeable periods are
    /// exactly the ones it rejects.
    #[test]
    fn scan_equals_the_normative_residual_construction() {
        for name in [
            "silence",
            "dc",
            "single-sine",
            "harmonic-tone",
            "quasi-periodic",
            "impulse-train",
            "am-signal",
            "white-noise",
        ] {
            let fx = corpus::named(name).unwrap();
            if fx.channels != 1 {
                continue;
            }
            let frames = fx.frames().min(2048);
            let x = &fx.samples[..frames];
            let limit = 96u32;
            let scan = scan_scalar(x, frames as u32, limit);
            for p in 1..=scan.limit() {
                let cycle = x[..p as usize].to_vec();
                let model = ResidualModel::Periodic { cycle };
                let by_construction = Residual::closing_residual(x, 1, &model);
                let got = scan.counts[(p - 1) as usize];
                match by_construction {
                    Some(records) => assert_eq!(
                        got as usize,
                        records.len(),
                        "{name}: period {p} record count mismatch"
                    ),
                    None => assert_eq!(
                        got, PERIOD_NOT_CLOSEABLE,
                        "{name}: period {p} should be flagged non-closeable"
                    ),
                }
            }
        }
    }

    #[test]
    fn parallel_equals_scalar_for_every_thread_count() {
        let fx = corpus::named("harmonic-tone").unwrap();
        let frames = 2048usize;
        let x = &fx.samples[..frames];
        let scalar = scan_scalar(x, frames as u32, 128);
        for threads in [1usize, 2, 3, 5, 8, 64, 1000] {
            let par = scan_parallel(x, frames as u32, 128, threads).unwrap();
            assert!(scans_agree(&scalar, &par), "threads={threads} diverged");
        }
    }

    #[test]
    fn ranking_is_deterministic_and_prefers_fewer_records() {
        let counts = vec![5u32, 0, 3, PERIOD_NOT_CLOSEABLE, 0];
        // keep 3: candidates are (1,5), (2,0), (3,3), (5,0); ranked by
        // (records, period): (2,0), (5,0), (3,3) -> ascending period: 2,3,5.
        assert_eq!(rank_periods(&counts, 3), vec![(2, 0), (3, 3), (5, 0)]);
        // keep 0 disables the family.
        assert!(rank_periods(&counts, 0).is_empty());
        // Non-closeable periods are never proposed.
        assert!(!rank_periods(&counts, 5).iter().any(|&(p, _)| p == 4));
    }

    #[test]
    fn scan_bound_is_never_degenerate() {
        assert_eq!(scan_bound(4096, 512), 512);
        assert_eq!(scan_bound(100, 512), 99);
        assert_eq!(scan_bound(1, 512), 0);
        assert_eq!(scan_bound(0, 512), 0);
    }

    #[test]
    fn request_validation_rejects_malformed_input() {
        assert!(validate_request(&[0, 1, 2, 3], 4, 2).is_ok());
        assert!(validate_request(&[0], 1, 2).is_err());
        assert!(validate_request(&[0, 1, 2], 4, 2).is_err());
        assert!(validate_request(&[0, 1, 2, 3], 4, 0).is_err());
    }
}
