//! Deterministic transport (Phase N, contract §36).
//!
//! A small framing model for `OBJECT`, `EVENT`, `STATE`, `CHECKPOINT`,
//! `DEPENDENCY`, `CLOCK` and `INTEGRITY` frames. It transports procedural/state
//! information — never mandatory PCM — while the literal fallback rides as an
//! `OBJECT` payload. It is deliberately not a networking stack.
//!
//! [`TransportReceiver`] is the ordered, bounded state machine that handles
//! sequencing, duplicate frames, stale epochs, gaps, late events, checkpoint
//! resync, missing dependencies and bounded resource use. Every decision is an
//! [`Outcome`] the caller can record in a receipt.

pub mod checkpoint;
pub mod clock;
pub mod event;
pub mod frame;
pub mod integrity;

use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;
use crate::limits::MAX_PENDING_EVENTS;
use crate::object::id::ContentId;
use crate::universe::clock::XrunPolicy;
use crate::universe::time::NominalRate;

pub use checkpoint::{Checkpoint, decode_checkpoint, encode_checkpoint};
pub use clock::{MediaClock, RecoveryOutcome};
pub use event::{decode_event, encode_event};
pub use frame::{Frame, FrameKind, NO_MEDIA_FRAME, decode_frame, decode_stream, encode_frame};
pub use integrity::{
    IntegrityReport, integrity_frame, report as stream_report, verify_integrity_frame,
};

use std::collections::BTreeSet;

/// What the receiver did with one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Object {
        content_id: ContentId,
    },
    Dependency {
        content_id: ContentId,
    },
    Event {
        late: bool,
    },
    State,
    Clock {
        epoch: u32,
        resync: bool,
    },
    Integrity,
    Checkpoint {
        resync: bool,
    },
    /// The frame belongs to an epoch older than the current one; dropped.
    StaleEpoch {
        frame_epoch: u32,
        current: u32,
    },
    /// The frame's sequence was already consumed; dropped.
    Duplicate {
        sequence: u64,
    },
    /// The sequence jumped; the receiver records the gap and accepts forward.
    SequenceGap {
        expected: u64,
        got: u64,
    },
}

/// Counters over a receiver's lifetime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReceiverStats {
    pub accepted: u64,
    pub duplicates: u64,
    pub stale_epochs: u64,
    pub gaps: u64,
    pub missing_sequence_frames: u64,
    pub late_events: u64,
    pub events: u64,
    pub resyncs: u64,
    pub objects: u64,
    pub dependencies: u64,
    pub checkpoints: u64,
}

/// The bounded, ordered transport receiver.
#[derive(Debug, Clone)]
pub struct TransportReceiver {
    clock: MediaClock,
    next_sequence: u64,
    declared_dependencies: BTreeSet<[u8; 32]>,
    resolved_content: BTreeSet<[u8; 32]>,
    pending_events: u64,
    stats: ReceiverStats,
}

impl Default for TransportReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl TransportReceiver {
    /// A receiver that expects epoch 0, sequence 0, at the u1 default rate.
    pub fn new() -> Self {
        Self::at_rate(NominalRate::new(crate::limits::DEFAULT_SAMPLE_RATE_HZ))
    }

    /// A receiver for a specific nominal rate.
    pub fn at_rate(rate: NominalRate) -> Self {
        TransportReceiver {
            clock: MediaClock::new(rate),
            next_sequence: 0,
            declared_dependencies: BTreeSet::new(),
            resolved_content: BTreeSet::new(),
            pending_events: 0,
            stats: ReceiverStats::default(),
        }
    }

    pub fn epoch(&self) -> u32 {
        self.clock.epoch()
    }

    pub fn cursor(&self) -> i64 {
        self.clock.frame()
    }

    pub fn stats(&self) -> ReceiverStats {
        self.stats
    }

    /// Advance the logical media cursor (normal playback progress).
    pub fn advance_cursor(&mut self, frames: u64) {
        self.clock.advance(frames);
    }

    /// Release `n` accepted events from the pending buffer once the sampler has
    /// consumed them. Without this the receiver's event bound is a per-epoch
    /// total, not a live pending count; a checkpoint resets it too.
    pub fn consume_events(&mut self, n: u64) {
        self.pending_events = self.pending_events.saturating_sub(n);
    }

    /// Events accepted but not yet released by [`Self::consume_events`].
    pub fn pending_events(&self) -> u64 {
        self.pending_events
    }

    /// Apply an xrun policy to the receiver's clock, returning the deterministic
    /// outcome that a receipt must record.
    pub fn on_xrun(&mut self, policy: XrunPolicy, missed_frames: u64) -> RecoveryOutcome {
        self.clock.on_xrun(policy, missed_frames)
    }

    /// Dependencies that have been declared but not yet satisfied by an object.
    pub fn unresolved_dependencies(&self) -> Vec<ContentId> {
        self.declared_dependencies
            .difference(&self.resolved_content)
            .map(|b| ContentId::from_bytes(*b))
            .collect()
    }

    /// Accept one frame, returning what the receiver did with it.
    pub fn push(&mut self, frame: &Frame) -> Result<Outcome> {
        // Stale epoch: never silently reinterpret an old epoch's frame.
        if frame.epoch < self.epoch() {
            self.stats.stale_epochs += 1;
            return Ok(Outcome::StaleEpoch {
                frame_epoch: frame.epoch,
                current: self.epoch(),
            });
        }
        // Duplicate: a sequence already consumed is dropped, not replayed.
        if frame.sequence < self.next_sequence {
            self.stats.duplicates += 1;
            return Ok(Outcome::Duplicate {
                sequence: frame.sequence,
            });
        }
        // Gap: record the missing count and accept forward.
        let mut outcome = None;
        if frame.sequence > self.next_sequence {
            let missing = frame.sequence - self.next_sequence;
            self.stats.gaps += 1;
            self.stats.missing_sequence_frames += missing;
            outcome = Some(Outcome::SequenceGap {
                expected: self.next_sequence,
                got: frame.sequence,
            });
        }
        // Epoch advance is explicit: only a CLOCK frame may change the epoch
        // forward, so a stray frame cannot silently re-anchor the stream.
        let mut epoch_advanced = false;
        if frame.epoch > self.epoch() {
            if frame.kind == FrameKind::Clock {
                self.clock.reset_to(frame.epoch, 0);
                self.next_sequence = frame.sequence;
                self.stats.resyncs += 1;
                epoch_advanced = true;
            } else {
                self.stats.stale_epochs += 1;
                return Ok(Outcome::StaleEpoch {
                    frame_epoch: frame.epoch,
                    current: self.epoch(),
                });
            }
        }

        let kind_outcome = match frame.kind {
            FrameKind::Object => {
                let content_id = ContentId::from_bytes(Sha256::digest(&frame.payload));
                self.resolved_content.insert(content_id.to_bytes());
                self.stats.objects += 1;
                Outcome::Object { content_id }
            }
            FrameKind::Dependency => {
                let bytes: [u8; 32] = frame.payload.as_slice().try_into().map_err(|_| {
                    Error::malformed("dependency frame payload must be a 32-byte content id")
                })?;
                self.declared_dependencies.insert(bytes);
                self.stats.dependencies += 1;
                Outcome::Dependency {
                    content_id: ContentId::from_bytes(bytes),
                }
            }
            FrameKind::Event => {
                let event = decode_event(&frame.payload)?;
                if self.pending_events >= u64::from(MAX_PENDING_EVENTS) {
                    return Err(Error::limit("transport event buffer exceeds the bound"));
                }
                self.pending_events += 1;
                let late = event.media_frame.to_i64() < self.cursor();
                if late {
                    self.stats.late_events += 1;
                }
                self.stats.events += 1;
                Outcome::Event { late }
            }
            FrameKind::State => Outcome::State,
            FrameKind::Clock => Outcome::Clock {
                epoch: self.epoch(),
                resync: epoch_advanced,
            },
            FrameKind::Integrity => Outcome::Integrity,
            FrameKind::Checkpoint => {
                let cp = checkpoint::decode_checkpoint(&frame.payload)?;
                let resync = cp.epoch != self.epoch() || cp.logical_frame != self.cursor();
                self.clock.reset_to(cp.epoch, cp.logical_frame);
                self.next_sequence = frame.sequence;
                self.pending_events = 0;
                if resync {
                    self.stats.resyncs += 1;
                }
                self.stats.checkpoints += 1;
                Outcome::Checkpoint { resync }
            }
        };

        self.next_sequence = frame.sequence + 1;
        self.stats.accepted += 1;
        Ok(outcome.unwrap_or(kind_outcome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::universe::event::{Event, EventClass};
    use crate::universe::time::MediaFrame;

    fn object_frame(seq: u64, payload: Vec<u8>) -> Frame {
        Frame::new(FrameKind::Object, 0, seq, NO_MEDIA_FRAME).with_payload(payload)
    }

    fn event_frame(seq: u64, at: i64) -> Frame {
        let e = Event::new(MediaFrame::new(at), EventClass::Param, seq, 1);
        Frame::new(FrameKind::Event, 0, seq, at).with_payload(encode_event(&e))
    }

    #[test]
    fn duplicates_stale_epochs_and_gaps_are_classified() {
        let mut r = TransportReceiver::new();
        assert!(matches!(
            r.push(&object_frame(0, vec![1])).unwrap(),
            Outcome::Object { .. }
        ));
        assert_eq!(
            r.push(&object_frame(0, vec![1])).unwrap(),
            Outcome::Duplicate { sequence: 0 }
        );
        assert_eq!(
            r.push(&object_frame(3, vec![2])).unwrap(),
            Outcome::SequenceGap {
                expected: 1,
                got: 3
            }
        );
        assert_eq!(r.stats().missing_sequence_frames, 2);
        // A frame from an older epoch is dropped.
        let mut r2 = TransportReceiver::new();
        r2.push(&Frame::new(FrameKind::Clock, 2, 0, NO_MEDIA_FRAME))
            .unwrap();
        assert_eq!(
            r2.push(&Frame::new(FrameKind::State, 0, 1, NO_MEDIA_FRAME))
                .unwrap(),
            Outcome::StaleEpoch {
                frame_epoch: 0,
                current: 2
            }
        );
    }

    #[test]
    fn only_a_clock_frame_may_advance_the_epoch() {
        let mut r = TransportReceiver::new();
        let stray = Frame::new(FrameKind::State, 5, 0, NO_MEDIA_FRAME);
        assert!(matches!(
            r.push(&stray).unwrap(),
            Outcome::StaleEpoch { .. }
        ));
        assert_eq!(r.epoch(), 0);
        let clock = Frame::new(FrameKind::Clock, 5, 0, NO_MEDIA_FRAME);
        assert_eq!(
            r.push(&clock).unwrap(),
            Outcome::Clock {
                epoch: 5,
                resync: true
            }
        );
        assert_eq!(r.epoch(), 5);
        assert_eq!(r.stats().resyncs, 1);
        // A clock frame that does not advance the epoch is not a resync.
        let same = Frame::new(FrameKind::Clock, 5, 1, NO_MEDIA_FRAME);
        assert_eq!(
            r.push(&same).unwrap(),
            Outcome::Clock {
                epoch: 5,
                resync: false
            }
        );
        assert_eq!(r.stats().resyncs, 1);
    }

    #[test]
    fn pending_events_are_bounded_and_released() {
        let mut r = TransportReceiver::new();
        for i in 0..5u64 {
            let e = Event::new(MediaFrame::new(0), EventClass::Param, i, 1);
            let f = Frame::new(FrameKind::Event, 0, i, 0).with_payload(encode_event(&e));
            r.push(&f).unwrap();
        }
        assert_eq!(r.pending_events(), 5);
        r.consume_events(2);
        assert_eq!(r.pending_events(), 3);
        r.consume_events(100);
        assert_eq!(r.pending_events(), 0);
    }

    #[test]
    fn late_events_and_checkpoint_resync() {
        let mut r = TransportReceiver::new();
        r.advance_cursor(10_000);
        assert_eq!(
            r.push(&event_frame(0, 9_999)).unwrap(),
            Outcome::Event { late: true }
        );
        assert_eq!(r.stats().late_events, 1);
        let cp = Checkpoint::new(0, 500, vec![0xAB]);
        let frame = Frame::new(FrameKind::Checkpoint, 0, 1, 500)
            .with_payload(checkpoint::encode_checkpoint(&cp).unwrap());
        assert_eq!(
            r.push(&frame).unwrap(),
            Outcome::Checkpoint { resync: true }
        );
        assert_eq!(r.cursor(), 500);
        assert_eq!(r.stats().resyncs, 1);
        assert_eq!(
            r.push(&event_frame(2, 700)).unwrap(),
            Outcome::Event { late: false }
        );
    }

    #[test]
    fn missing_dependencies_are_tracked_until_resolved() {
        let mut r = TransportReceiver::new();
        let payload = b"canonical-object-bytes".to_vec();
        let resolved = ContentId::from_bytes(Sha256::digest(&payload));
        let needed = ContentId::from_bytes([7u8; 32]);
        // Declare the dependency that the object will satisfy.
        let dep = Frame::new(FrameKind::Dependency, 0, 0, NO_MEDIA_FRAME)
            .with_payload(resolved.to_bytes().to_vec());
        r.push(&dep).unwrap();
        assert_eq!(r.unresolved_dependencies(), vec![resolved]);
        r.push(&object_frame(1, payload)).unwrap();
        assert!(r.unresolved_dependencies().is_empty());
        // An unrelated declaration stays unresolved.
        let other = Frame::new(FrameKind::Dependency, 0, 2, NO_MEDIA_FRAME)
            .with_payload(needed.to_bytes().to_vec());
        r.push(&other).unwrap();
        assert_eq!(r.unresolved_dependencies(), vec![needed]);
    }

    #[test]
    fn xrun_policies_are_deterministic_on_the_receiver() {
        let mut a = TransportReceiver::new();
        a.advance_cursor(1_000);
        let mut b = TransportReceiver::new();
        b.advance_cursor(1_000);
        assert_eq!(
            a.on_xrun(XrunPolicy::PreserveTimeline, 64),
            b.on_xrun(XrunPolicy::PreserveTimeline, 64)
        );
        assert_eq!(a.cursor(), 1_064);
        let or = a.on_xrun(XrunPolicy::RestartEpoch, 0);
        assert!(or.restarts_epoch);
        assert_eq!(a.cursor(), 0);
        assert_eq!(a.epoch(), 1);
    }
}
