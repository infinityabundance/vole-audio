//! Media clock and xrun recovery (Phase N, contract §37).
//!
//! Clock state is part of the architecture: media epoch, logical frame, nominal
//! rate and an explicit discontinuity count. Recovery never silently resets and
//! pretends continuity — the policy is named in the outcome so a receipt can say
//! exactly what happened.
//!
//! ```text
//! PreserveTimeline  the media timeline continues; the endpoint skips the
//!                   missed frames (no samples are fabricated)
//! Discontinuity     the media frame is unchanged; explicit silence is inserted
//!                   (a real discontinuity in the observation)
//! RestartEpoch      the epoch increments and the logical frame restarts at 0
//! ```

use crate::universe::clock::XrunPolicy;
use crate::universe::time::{EpochId, MediaFrame, NominalRate};

/// The deterministic result of a recovery decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryOutcome {
    pub policy: XrunPolicy,
    /// Epoch after recovery.
    pub epoch: u32,
    /// Logical media frame after recovery.
    pub logical_frame: i64,
    /// Media frames the endpoint skipped (timeline preserved).
    pub skipped_frames: u64,
    /// Frames of explicit silence inserted (observation discontinuity).
    pub inserted_frames: u64,
    pub preserves_timeline: bool,
    pub inserts_discontinuity: bool,
    pub restarts_epoch: bool,
}

/// A media clock with an explicit discontinuity counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaClock {
    epoch: EpochId,
    frame: MediaFrame,
    rate: NominalRate,
    discontinuities: u64,
}

impl MediaClock {
    pub const fn new(rate: NominalRate) -> Self {
        MediaClock {
            epoch: EpochId(0),
            frame: MediaFrame::new(0),
            rate,
            discontinuities: 0,
        }
    }

    pub const fn epoch(&self) -> u32 {
        self.epoch.0
    }

    pub const fn frame(&self) -> i64 {
        self.frame.to_i64()
    }

    pub const fn rate(&self) -> u32 {
        self.rate.to_hz()
    }

    pub const fn discontinuities(&self) -> u64 {
        self.discontinuities
    }

    /// Advance the logical frame by `frames` (normal playback progress).
    pub fn advance(&mut self, frames: u64) {
        self.frame = MediaFrame::new(self.frame.to_i64() + frames as i64);
    }

    /// Re-anchor the clock to an explicit epoch and logical frame (a checkpoint
    /// resync or an epoch-advancing control frame). Counted as a discontinuity.
    pub fn reset_to(&mut self, epoch: u32, frame: i64) {
        self.epoch = EpochId(epoch);
        self.frame = MediaFrame::new(frame);
        self.discontinuities += 1;
    }

    /// Apply an xrun policy for `missed_frames` and return the deterministic
    /// outcome. The clock is updated to the post-recovery state.
    pub fn on_xrun(&mut self, policy: XrunPolicy, missed_frames: u64) -> RecoveryOutcome {
        let missed = missed_frames as i64;
        match policy {
            XrunPolicy::PreserveTimeline => {
                // The media timeline continues: the missed frames are consumed
                // by the endpoint without being rendered. No samples fabricated.
                self.frame = MediaFrame::new(self.frame.to_i64() + missed);
                self.discontinuities += 1;
                RecoveryOutcome {
                    policy,
                    epoch: self.epoch(),
                    logical_frame: self.frame(),
                    skipped_frames: missed_frames,
                    inserted_frames: 0,
                    preserves_timeline: true,
                    inserts_discontinuity: false,
                    restarts_epoch: false,
                }
            }
            XrunPolicy::Discontinuity => {
                // The logical frame is unchanged; explicit silence is inserted, so
                // the observation contains a real, counted discontinuity.
                self.discontinuities += 1;
                RecoveryOutcome {
                    policy,
                    epoch: self.epoch(),
                    logical_frame: self.frame(),
                    skipped_frames: 0,
                    inserted_frames: missed_frames,
                    preserves_timeline: true,
                    inserts_discontinuity: true,
                    restarts_epoch: false,
                }
            }
            XrunPolicy::RestartEpoch => {
                self.epoch = self.epoch.next();
                self.frame = MediaFrame::new(0);
                self.discontinuities += 1;
                RecoveryOutcome {
                    policy,
                    epoch: self.epoch(),
                    logical_frame: 0,
                    skipped_frames: 0,
                    inserted_frames: 0,
                    preserves_timeline: false,
                    inserts_discontinuity: true,
                    restarts_epoch: true,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policies_are_distinct_and_deterministic() {
        let rate = NominalRate::new(48_000);

        let mut preserve = MediaClock::new(rate);
        preserve.advance(1000);
        let o = preserve.on_xrun(XrunPolicy::PreserveTimeline, 128);
        assert!(o.preserves_timeline && !o.inserts_discontinuity && !o.restarts_epoch);
        assert_eq!(o.skipped_frames, 128);
        assert_eq!(o.logical_frame, 1128);
        assert_eq!(preserve.frame(), 1128);

        let mut disc = MediaClock::new(rate);
        disc.advance(1000);
        let o = disc.on_xrun(XrunPolicy::Discontinuity, 128);
        assert!(o.inserts_discontinuity && !o.restarts_epoch);
        assert_eq!(o.inserted_frames, 128);
        assert_eq!(o.logical_frame, 1000, "the media frame does not advance");

        let mut restart = MediaClock::new(rate);
        restart.advance(1000);
        let o = restart.on_xrun(XrunPolicy::RestartEpoch, 128);
        assert!(o.restarts_epoch && !o.preserves_timeline);
        assert_eq!(o.epoch, 1);
        assert_eq!(o.logical_frame, 0);
        assert_eq!(restart.epoch(), 1);
    }

    #[test]
    fn recovery_is_reproducible() {
        let mut a = MediaClock::new(NominalRate::new(44_100));
        let mut b = MediaClock::new(NominalRate::new(44_100));
        a.advance(500);
        b.advance(500);
        for (x, y) in [
            (
                a.on_xrun(XrunPolicy::PreserveTimeline, 64),
                b.on_xrun(XrunPolicy::PreserveTimeline, 64),
            ),
            (
                a.on_xrun(XrunPolicy::Discontinuity, 32),
                b.on_xrun(XrunPolicy::Discontinuity, 32),
            ),
            (
                a.on_xrun(XrunPolicy::RestartEpoch, 0),
                b.on_xrun(XrunPolicy::RestartEpoch, 0),
            ),
        ] {
            assert_eq!(x, y);
        }
        assert_eq!(a, b);
    }
}
