//! SampleObject descriptor and representation taxonomy.
//!
//! The descriptor is the immutable header of a SampleObject: identity,
//! representation class, intrinsic extent, channel layout, loop region,
//! dependency list, and hostile-input bounds. The canonical serialization of
//! (descriptor + payload) defines the content identity hash.
//!
//! Representation tags are frozen from day one so archives never renumber;
//! constructors/observers for later classes land in their phases, but the tag
//! space is fixed.

use crate::object::id::ContentId;
use crate::universe::layout::Layout;
use crate::universe::u1::PROFILE_TAG_BYTES;
use core::fmt;

/// Representation class of a SampleObject (paper §"SampleObject").
///
/// Tag numbers are canonical and frozen. A tag with no payload constructor in
/// this build is *unavailable*, never silently reinterpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Representation {
    /// Full sampled payload (universal fallback; always legal).
    Literal = 0x01,
    /// Procedural silence.
    Silence = 0x02,
    /// Procedural constant level.
    Constant = 0x03,
    /// Exact repetition of a stored motif.
    ExactRepeat = 0x04,
    /// Single-cycle waveform table (periodic).
    SingleCycle = 0x05,
    /// Arbitrary wavetable with loop semantics.
    Wavetable = 0x06,
    /// Phase-accumulating oscillator (u64 phase, frozen table).
    Oscillator = 0x07,
    /// Bank of harmonic/partial generators.
    PartialBank = 0x08,
    /// Alias to another SampleObject (+ optional transpose/loop override).
    Referenced = 0x09,
    /// Composition of objects (graph node).
    Compound = 0x0A,
    /// Residual-governed object: model + explicit residual.
    PredictorResidual = 0x0B,
}

impl Representation {
    pub const fn tag(self) -> u8 {
        self as u8
    }

    pub const fn from_tag(t: u8) -> Option<Representation> {
        match t {
            0x01 => Some(Representation::Literal),
            0x02 => Some(Representation::Silence),
            0x03 => Some(Representation::Constant),
            0x04 => Some(Representation::ExactRepeat),
            0x05 => Some(Representation::SingleCycle),
            0x06 => Some(Representation::Wavetable),
            0x07 => Some(Representation::Oscillator),
            0x08 => Some(Representation::PartialBank),
            0x09 => Some(Representation::Referenced),
            0x0A => Some(Representation::Compound),
            0x0B => Some(Representation::PredictorResidual),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Representation::Literal => "literal",
            Representation::Silence => "silence",
            Representation::Constant => "constant",
            Representation::ExactRepeat => "exact_repeat",
            Representation::SingleCycle => "single_cycle",
            Representation::Wavetable => "wavetable",
            Representation::Oscillator => "oscillator",
            Representation::PartialBank => "partial_bank",
            Representation::Referenced => "referenced",
            Representation::Compound => "compound",
            Representation::PredictorResidual => "predictor_residual",
        }
    }
}

impl fmt::Display for Representation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Loop region in object-intrinsic frames (half-open `[start, end)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopRegion {
    pub start_frame: u64,
    pub end_frame: u64,
}

impl LoopRegion {
    pub const fn new(start: u64, end: u64) -> Option<LoopRegion> {
        if end > start && end <= crate::limits::MAX_OBJECT_FRAMES {
            Some(LoopRegion {
                start_frame: start,
                end_frame: end,
            })
        } else {
            None
        }
    }

    pub const fn len_frames(&self) -> u64 {
        self.end_frame - self.start_frame
    }

    /// Whole-extent loop.
    pub const fn whole(extent: u64) -> Option<LoopRegion> {
        if extent == 0 {
            None
        } else {
            Some(LoopRegion {
                start_frame: 0,
                end_frame: extent,
            })
        }
    }
}

/// Immutable object descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectDescriptor {
    pub representation: Representation,
    /// Intrinsic extent in object frames.
    pub extent_frames: u64,
    /// Channel layout of the object's intrinsic domain.
    pub layout: Layout,
    /// Optional intrinsic loop region.
    pub loop_region: Option<LoopRegion>,
    /// Direct dependency **content ids** (reference/compound/residual targets).
    /// Content ids are archive-stable, so identity is portable.
    pub dependencies: Vec<ContentId>,
}

impl ObjectDescriptor {
    pub const fn new(
        representation: Representation,
        extent_frames: u64,
        layout: Layout,
        loop_region: Option<LoopRegion>,
    ) -> Option<ObjectDescriptor> {
        if extent_frames > crate::limits::MAX_OBJECT_FRAMES {
            return None;
        }
        // Loop region must lie within extent.
        if matches!(loop_region, Some(l) if l.end_frame > extent_frames) {
            return None;
        }
        Some(ObjectDescriptor {
            representation,
            extent_frames,
            layout,
            loop_region,
            dependencies: Vec::new(),
        })
    }

    /// Bounded dependency count check (hostile input).
    pub fn check_dependency_budget(&self) -> bool {
        self.dependencies.len() as u32 <= crate::limits::MAX_DEPENDENCIES_PER_OBJECT
    }
}

/// Canonical header bytes for object serialization:
/// `PROFILE_TAG || representation_tag || extent(u64 LE) || layout(u8) ||
/// loop_flag(u8) || loop_start(u64 LE) || loop_end(u64 LE)`.
///
/// This is the identity-relevant header; payload encoding is representation-
/// specific and appended by each representation module.
pub fn canonical_header_bytes(d: &ObjectDescriptor) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + 1 + 8 + 1 + 1 + 16);
    out.extend_from_slice(PROFILE_TAG_BYTES);
    out.push(d.representation.tag());
    out.extend_from_slice(&d.extent_frames.to_le_bytes());
    out.push(d.layout.count());
    match d.loop_region {
        Some(l) => {
            out.push(1);
            out.extend_from_slice(&l.start_frame.to_le_bytes());
            out.extend_from_slice(&l.end_frame.to_le_bytes());
        }
        None => {
            out.push(0);
            out.extend_from_slice(&0u64.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_tags_are_frozen() {
        let pairs = [
            (Representation::Literal, 0x01),
            (Representation::Silence, 0x02),
            (Representation::Constant, 0x03),
            (Representation::ExactRepeat, 0x04),
            (Representation::SingleCycle, 0x05),
            (Representation::Wavetable, 0x06),
            (Representation::Oscillator, 0x07),
            (Representation::PartialBank, 0x08),
            (Representation::Referenced, 0x09),
            (Representation::Compound, 0x0A),
            (Representation::PredictorResidual, 0x0B),
        ];
        for (r, t) in pairs {
            assert_eq!(r.tag(), t);
            assert_eq!(Representation::from_tag(t), Some(r));
        }
        assert_eq!(Representation::from_tag(0), None);
        assert_eq!(Representation::from_tag(0x0C), None);
    }

    #[test]
    fn descriptor_validation() {
        assert!(
            ObjectDescriptor::new(
                Representation::Literal,
                100,
                Layout::Mono,
                LoopRegion::new(0, 50)
            )
            .is_some()
        );
        // Loop beyond extent is rejected.
        assert!(
            ObjectDescriptor::new(
                Representation::Literal,
                100,
                Layout::Mono,
                LoopRegion::new(50, 200)
            )
            .is_none()
        );
        // Extent ceiling.
        assert!(
            ObjectDescriptor::new(
                Representation::Literal,
                crate::limits::MAX_OBJECT_FRAMES + 1,
                Layout::Mono,
                None
            )
            .is_none()
        );
    }

    #[test]
    fn header_bytes_are_canonical() {
        let d = ObjectDescriptor::new(
            Representation::Literal,
            100,
            Layout::Stereo,
            LoopRegion::new(4, 90),
        )
        .unwrap();
        let b = canonical_header_bytes(&d);
        assert!(b.starts_with(b"vole.audio.u1/u1/v1"));
        // profile(19) | tag | extent u64 | layout u8 | loop_flag u8 | ...
        assert_eq!(b[19], Representation::Literal.tag());
        assert_eq!(b[28], 2); // layout count (stereo)
        assert_eq!(b[29], 1); // loop flag
        let mut e = ObjectDescriptor::new(
            Representation::Literal,
            100,
            Layout::Stereo,
            LoopRegion::new(4, 90),
        )
        .unwrap();
        e.loop_region = None;
        assert_ne!(b, canonical_header_bytes(&e));
    }
}
