//! Clock domains and xrun/epoch policy.
//!
//! Clock state is part of the architecture (U1_SPEC §"Clock and xrun
//! recovery"). The endpoint clock is instrumented on the host
//! (`CLOCK_MONOTONIC_RAW`) and never substituted for media time; this module
//! defines the *policy vocabulary* for discontinuities and the host-side
//! frame<->time conversions with explicit rounding.

use crate::universe::time::NominalRate;
use core::fmt;

/// Policy applied when the endpoint underruns/overruns or the media timeline
/// must be re-anchored. The chosen policy is recorded in receipts; the
/// sampler never silently resets state and pretends continuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XrunPolicy {
    /// Preserve the defined media timeline; insert an explicit discontinuity
    /// marker in the observation stream (no samples are fabricated).
    PreserveTimeline,
    /// Insert an explicit discontinuity (gap/silence) and continue.
    Discontinuity,
    /// Restart the media epoch (`EpochId` increments; state is re-derived
    /// deterministically under the new epoch).
    RestartEpoch,
}

impl XrunPolicy {
    pub const fn label(self) -> &'static str {
        match self {
            XrunPolicy::PreserveTimeline => "PRESERVE_TIMELINE",
            XrunPolicy::Discontinuity => "DISCONTINUITY",
            XrunPolicy::RestartEpoch => "RESTART_EPOCH",
        }
    }
}

impl fmt::Display for XrunPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Host-side frame<->microsecond conversion with documented rounding
/// (round-half-up). Used for endpoint scheduling math on the host only; the
/// media timeline itself is integer frames and never goes through floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostClock {
    pub rate: NominalRate,
}

impl HostClock {
    pub const fn new(rate: NominalRate) -> Self {
        Self { rate }
    }

    /// Frames -> microseconds (round half up).
    pub fn frames_to_micros(&self, frames: u64) -> u64 {
        (frames * 1_000_000 + u64::from(self.rate.to_hz()) / 2) / u64::from(self.rate.to_hz())
    }

    /// Microseconds -> frames (round half up).
    pub fn micros_to_frames(&self, micros: u64) -> u64 {
        (micros * u64::from(self.rate.to_hz()) + 500_000) / 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xrun_policy_labels() {
        assert_eq!(XrunPolicy::PreserveTimeline.label(), "PRESERVE_TIMELINE");
        assert_eq!(XrunPolicy::Discontinuity.label(), "DISCONTINUITY");
        assert_eq!(XrunPolicy::RestartEpoch.label(), "RESTART_EPOCH");
    }

    #[test]
    fn host_clock_roundtrip() {
        let c = HostClock::new(NominalRate::new(48_000));
        assert_eq!(c.frames_to_micros(48_000), 1_000_000);
        assert_eq!(c.micros_to_frames(1_000_000), 48_000);
        assert_eq!(c.frames_to_micros(24_000), 500_000);
        assert_eq!(c.micros_to_frames(10_417), 500);
        // round-trip sanity at a non-divisible point
        let f = 123_457u64;
        let us = c.frames_to_micros(f);
        let back = c.micros_to_frames(us);
        assert!((back as i64 - f as i64).abs() <= 1);
    }
}
