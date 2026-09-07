//! Wavetable-family payload: one stored cycle.
//!
//! `Cycle` backs three representation tags (identity differs by tag):
//! `Wavetable`, `SingleCycle`, and `ExactRepeat`. A cycle is canonical
//! interleaved i32 content of `extent_frames` frames; playback of these
//! objects **always wraps the cycle** (periodic content; voice loop regions
//! are ignored and natural end is `None`).

use crate::hash::sha256::Sha256;
use crate::object::ObjectData;
use crate::object::descriptor::{ObjectDescriptor, Representation, canonical_header_bytes};
use crate::object::id::ContentId;

/// One stored cycle (interleaved samples, `extent_frames * channels` codes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle {
    pub samples: Vec<i32>,
}

impl Cycle {
    /// Validate length against descriptor extent/layout and ceilings.
    pub fn new(descriptor: &ObjectDescriptor, samples: Vec<i32>) -> Option<Cycle> {
        if descriptor.extent_frames == 0 {
            return None;
        }
        let ch = usize::from(descriptor.layout.count());
        let expect = usize::try_from(descriptor.extent_frames)
            .ok()?
            .checked_mul(ch)?;
        if samples.len() != expect {
            return None;
        }
        // Cycles are resident tables: their bytes count as a declared
        // dependency/table (never free), bounded by the table ceiling.
        if (samples.len() as u64) * 4 > u64::from(crate::limits::MAX_TABLE_BYTES) {
            return None;
        }
        Some(Cycle { samples })
    }
}

/// Canonical cycle payload bytes:
/// `header || cycle_len_frames(u64 LE) || sample_count(u64 LE) || codes (i32 LE)`.
pub fn cycle_canonical_bytes(
    descriptor: &ObjectDescriptor,
    cycle: &Cycle,
    rep: Representation,
) -> Vec<u8> {
    debug_assert!(matches!(
        rep,
        Representation::Wavetable | Representation::SingleCycle | Representation::ExactRepeat
    ));
    let mut d = descriptor.clone();
    d.representation = rep;
    let mut out = canonical_header_bytes(&d);
    out.extend_from_slice(&d.extent_frames.to_le_bytes());
    out.extend_from_slice(&(cycle.samples.len() as u64).to_le_bytes());
    for s in &cycle.samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Build a payload for any of the three tags.
pub fn make(
    rep: Representation,
    descriptor: &ObjectDescriptor,
    samples: Vec<i32>,
) -> Option<ObjectData> {
    if !matches!(
        rep,
        Representation::Wavetable | Representation::SingleCycle | Representation::ExactRepeat
    ) {
        return None;
    }
    let cycle = Cycle::new(descriptor, samples)?;
    Some(match rep {
        Representation::Wavetable => ObjectData::Wavetable(cycle),
        Representation::SingleCycle => ObjectData::SingleCycle(cycle),
        Representation::ExactRepeat => ObjectData::ExactRepeat(cycle),
        _ => unreachable!(),
    })
}

/// Content identity of a cycle payload under its tag.
pub fn content_id(descriptor: &ObjectDescriptor, cycle: &Cycle, rep: Representation) -> ContentId {
    ContentId(Sha256::digest(&cycle_canonical_bytes(
        descriptor, cycle, rep,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::universe::layout::Layout;

    #[test]
    fn cycle_length_must_match_descriptor() {
        let d = ObjectDescriptor::new(Representation::Wavetable, 8, Layout::Mono, None).unwrap();
        assert!(Cycle::new(&d, vec![0; 8]).is_some());
        assert!(Cycle::new(&d, vec![0; 7]).is_none());
        // Extent 0 cycles are rejected.
        let d0 = ObjectDescriptor::new(Representation::Wavetable, 0, Layout::Mono, None).unwrap();
        assert!(Cycle::new(&d0, vec![]).is_none());
    }

    #[test]
    fn tags_change_identity() {
        let d = ObjectDescriptor::new(Representation::Wavetable, 4, Layout::Mono, None).unwrap();
        let c = Cycle::new(&d, vec![1, 2, 3, 4]).unwrap();
        let a = content_id(&d, &c, Representation::Wavetable);
        let b = content_id(&d, &c, Representation::SingleCycle);
        assert_ne!(a, b);
        let c2 = content_id(&d, &c, Representation::ExactRepeat);
        assert_ne!(b, c2);
    }
}
