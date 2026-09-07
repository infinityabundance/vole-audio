//! Event model with a frozen total order.
//!
//! Two simultaneous events must never depend on hashmap iteration, thread
//! scheduling, GPU lane ordering, or allocation order. u1 therefore defines
//! the total order:
//!
//! ```text
//! (media_frame, event_class_priority, sequence)
//! ```
//!
//! where `sequence` is the monotonically assigned arrival order within the
//! frame. `EventClass` carries a fixed priority so that, e.g., a voice stop
//! deterministically precedes a parameter change at the same frame.

use crate::universe::time::MediaFrame;
use core::cmp::Ordering;
use core::fmt;

/// Event classes with frozen priorities (lower number = earlier at the same
/// frame). The exact priorities are part of the u1 freeze.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EventClass {
    /// Voice trigger/start.
    Start = 0,
    /// Voice stop (note-off / release).
    Stop = 1,
    /// Voice parameter update (rate, gain, pan, ...).
    Param = 2,
    /// World/global state update.
    World = 3,
    /// Epoch/clock control.
    Clock = 4,
    /// Diagnostic/evidence marker (never affects media semantics).
    Diagnostic = 5,
}

impl EventClass {
    pub const fn priority(self) -> u8 {
        self as u8
    }
}

/// A scheduler event with a frozen total order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Event {
    pub media_frame: MediaFrame,
    pub class: EventClass,
    /// Monotonically assigned arrival sequence (unique within a run).
    pub sequence: u64,
    /// Payload selector; decoding depends on the sampler layer.
    pub kind: u16,
}

impl Event {
    pub const fn new(media_frame: MediaFrame, class: EventClass, sequence: u64, kind: u16) -> Self {
        Self {
            media_frame,
            class,
            sequence,
            kind,
        }
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Total order: frame, class priority, arrival sequence.
        self.media_frame
            .cmp(&other.media_frame)
            .then_with(|| self.class.priority().cmp(&other.class.priority()))
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "event@{}:{}:{} kind={}",
            self.media_frame,
            self.class.priority(),
            self.sequence,
            self.kind
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn total_order_is_frame_class_then_sequence() {
        let a = Event::new(MediaFrame::new(10), EventClass::Start, 1, 0);
        let b = Event::new(MediaFrame::new(10), EventClass::Start, 2, 0);
        let d = Event::new(MediaFrame::new(11), EventClass::Start, 0, 0);
        assert!(a < b); // same frame/class -> sequence
        assert!(a < d); // frame dominates
        // Priority rule: lower priority value = earlier. Start=0 < Stop=1.
        let s = Event::new(MediaFrame::new(10), EventClass::Start, 99, 0);
        let t = Event::new(MediaFrame::new(10), EventClass::Stop, 0, 0);
        assert!(
            s < t,
            "Start (priority 0) must sort before Stop (priority 1)"
        );
    }

    #[test]
    fn sorting_is_independent_of_input_order() {
        let mk = |f, c, seq| Event::new(MediaFrame::new(f), c, seq, 0);
        let mut v1 = vec![
            mk(0, EventClass::Stop, 7),
            mk(0, EventClass::Start, 1),
            mk(0, EventClass::Start, 0),
            mk(5, EventClass::Param, 0),
            mk(0, EventClass::Stop, 0),
        ];
        let mut v2 = v1.clone();
        v2.reverse();
        // Deterministic regardless of insertion order (BTreeSet uses Ord).
        let s1: BTreeSet<_> = v1.drain(..).collect();
        let s2: BTreeSet<_> = v2.drain(..).collect();
        assert_eq!(s1, s2);
        let e: Vec<_> = s1.into_iter().collect();
        assert_eq!(e[0].class, EventClass::Start);
        assert_eq!(e[0].sequence, 0);
        assert_eq!(e[1].class, EventClass::Start);
        assert_eq!(e[1].sequence, 1);
        assert_eq!(e[2].class, EventClass::Stop);
        assert_eq!(e[2].sequence, 0);
    }

    #[test]
    fn priority_values_are_frozen() {
        assert_eq!(EventClass::Start.priority(), 0);
        assert_eq!(EventClass::Stop.priority(), 1);
        assert_eq!(EventClass::Param.priority(), 2);
        assert_eq!(EventClass::World.priority(), 3);
        assert_eq!(EventClass::Clock.priority(), 4);
        assert_eq!(EventClass::Diagnostic.priority(), 5);
    }
}
