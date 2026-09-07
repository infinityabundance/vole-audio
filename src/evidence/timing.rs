//! Timing domains and tail statistics.
//!
//! Two timing domains exist (paper §"Timing"):
//!   1. host/end-to-end: `CLOCK_MONOTONIC_RAW` (or equivalent precise monotonic
//!      source), used for audio deadlines and submission latency;
//!   2. GPU kernel time: CUDA/HIP events (in the backend modules).
//!
//! GPU event time is never substituted for end-to-end audio deadline latency.
//! Raw samples are stored; percentiles are only reported when the sample count
//! supports a meaningful estimate (see `min_samples_for`).

use core::fmt;

/// A monotonic instant in nanoseconds (host domain).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstantNs(pub i64);

/// A duration in nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct DurationNs(pub i64);

impl DurationNs {
    pub const ZERO: DurationNs = DurationNs(0);

    pub fn from_secs_f64(s: f64) -> DurationNs {
        DurationNs((s * 1e9).round() as i64)
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1e9
    }

    pub fn as_micros_f64(self) -> f64 {
        self.0 as f64 / 1e3
    }

    pub fn as_millis_f64(self) -> f64 {
        self.0 as f64 / 1e6
    }
}

impl fmt::Display for DurationNs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1_000_000_000 {
            write!(f, "{:.6} s", self.as_secs_f64())
        } else if self.0 >= 1_000_000 {
            write!(f, "{:.3} ms", self.as_millis_f64())
        } else if self.0 >= 1_000 {
            write!(f, "{:.1} us", self.as_micros_f64())
        } else {
            write!(f, "{} ns", self.0)
        }
    }
}

/// Monotonic raw clock (host). Not available on device targets.
#[cfg(feature = "std")]
pub fn monotonic_raw_ns() -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: timespec is a valid out-pointer for clock_gettime; the call
    // initializes it. CLOCK_MONOTONIC_RAW exists on Linux (and musl).
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts) };
    debug_assert_eq!(rc, 0, "clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    ts.tv_sec * 1_000_000_000 + ts.tv_nsec
}

/// Host-domain stopwatch.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch {
    start: i64,
}

#[cfg(feature = "std")]
impl Stopwatch {
    pub fn start() -> Self {
        Self {
            start: monotonic_raw_ns(),
        }
    }

    pub fn elapsed_ns(&self) -> i64 {
        monotonic_raw_ns() - self.start
    }

    pub fn elapsed(&self) -> DurationNs {
        DurationNs(self.elapsed_ns())
    }
}

/// Minimum sample count required before a percentile `p` (in (0,1]) may be
/// reported from *raw stored samples*.
///
/// Policy: at least 20 observations are expected beyond the percentile:
/// `n >= ceil(20 / (1 - p))` (for p < 1). For p == 1 (max) one sample suffices.
/// This prevents reporting p99.9 from a hundred samples.
///
/// The ceiling is computed with integer verification against the floating
/// point `p` so ladder values stay exact (p50=40, p90=200, p99=2000,
/// p99.9=20000) despite 0.1/0.01-style representation error.
pub fn min_samples_for(p: f64) -> usize {
    debug_assert!((0.0..=1.0).contains(&p));
    if p >= 1.0 {
        return 1;
    }
    let denom = 1.0 - p;
    let mut n = ((20.0 / denom) as usize).max(1);
    // Climb until n * denom >= 20 (tolerance only for representation noise).
    while (n as f64) * denom < 20.0 - 1e-12 {
        n += 1;
    }
    n
}

/// Which percentiles may be reported given N raw samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReportableTail {
    pub n: usize,
    pub p50: bool,
    pub p90: bool,
    pub p99: bool,
    pub p999: bool,
    pub max: bool,
}

impl ReportableTail {
    /// Decide reportability for the whole ladder from one N.
    pub fn for_n(n: usize) -> Self {
        Self {
            n,
            p50: n >= min_samples_for(0.50),
            p90: n >= min_samples_for(0.90),
            p99: n >= min_samples_for(0.99),
            p999: n >= min_samples_for(0.999),
            max: n >= 1,
        }
    }

    /// Human explanation for the deepest reported percentile.
    pub fn policy_note(&self) -> String {
        let deepest = if self.p999 {
            "p99.9"
        } else if self.p99 {
            "p99"
        } else if self.p90 {
            "p90"
        } else if self.p50 {
            "p50"
        } else {
            "none (max_observed only)"
        };
        format!(
            "n={} raw samples; deepest reported percentile: {} \
             (rule: n >= ceil(20/(1-p)))",
            self.n, deepest
        )
    }
}

/// Tail summary over raw latency samples (ns). All values are ns.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TailSummary {
    pub n: usize,
    pub min_ns: Option<u64>,
    pub max_ns: Option<u64>,
    /// Arithmetic mean in ns (as f64 for readability).
    pub mean_ns: Option<f64>,
    pub p50_ns: Option<u64>,
    pub p90_ns: Option<u64>,
    pub p99_ns: Option<u64>,
    pub p999_ns: Option<u64>,
    /// Which deeper percentiles are *not* reported and why.
    pub policy_note: String,
}

impl TailSummary {
    /// Summarize raw samples (does not mutate them).
    ///
    /// Percentiles follow the nearest-rank method on sorted raw samples.
    /// Percentiles that N cannot support are reported as `None` and the policy
    /// note says why.
    pub fn summarize(raw_ns: &[u64]) -> Self {
        let mut sorted = raw_ns.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        let reportable = ReportableTail::for_n(n);
        let nr = |p: f64| -> Option<u64> {
            if n == 0 {
                return None;
            }
            // nearest-rank
            let rank = ((p * n as f64).ceil() as usize).clamp(1, n);
            Some(sorted[rank - 1])
        };
        let mean = if n == 0 {
            None
        } else {
            Some(sorted.iter().map(|&v| v as f64).sum::<f64>() / n as f64)
        };
        TailSummary {
            n,
            min_ns: sorted.first().copied(),
            max_ns: sorted.last().copied(),
            mean_ns: mean,
            p50_ns: if reportable.p50 { nr(0.50) } else { None },
            p90_ns: if reportable.p90 { nr(0.90) } else { None },
            p99_ns: if reportable.p99 { nr(0.99) } else { None },
            p999_ns: if reportable.p999 { nr(0.999) } else { None },
            policy_note: reportable.policy_note(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_samples_policy() {
        assert_eq!(min_samples_for(1.0), 1);
        assert_eq!(min_samples_for(0.50), 40);
        assert_eq!(min_samples_for(0.90), 200);
        assert_eq!(min_samples_for(0.99), 2_000);
        assert_eq!(min_samples_for(0.999), 20_000);
        let r = ReportableTail::for_n(100);
        assert!(r.p50 && !r.p90 && !r.p999);
        let r = ReportableTail::for_n(25_000);
        assert!(r.p50 && r.p90 && r.p99 && r.p999);
    }

    #[test]
    fn summary_computes_percentiles() {
        let raw: Vec<u64> = (0..40_000).collect();
        let s = TailSummary::summarize(&raw);
        assert_eq!(s.n, 40_000);
        assert_eq!(s.min_ns, Some(0));
        assert_eq!(s.max_ns, Some(39_999));
        assert_eq!(s.p50_ns, Some(19_999)); // nearest-rank: ceil(.5*40000)=20000 -> sorted[19999] = 19999
        assert!(s.p99_ns.is_some() && s.p999_ns.is_some());
        assert!(s.policy_note.contains("p99.9"));
    }

    #[test]
    fn summary_guards_small_n() {
        let raw: Vec<u64> = vec![1, 2, 3];
        let s = TailSummary::summarize(&raw);
        assert_eq!(s.p50_ns, None);
        assert_eq!(s.max_ns, Some(3));
        assert!(s.policy_note.contains("max_observed"));
    }
}
