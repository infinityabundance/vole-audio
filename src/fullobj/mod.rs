//! Full-object archival container (Phase M): exact U1 SampleObjects as segments.
//!
//! The Phase-K inverse compiler is deliberately bounded to one observation
//! window ([`crate::inverse::MAX_INVERSE_FRAMES`] = 65,536 frames). A frozen
//! corpus object can be much larger (up to 144,000 frames in the flagship
//! population). This module is the **object-above-objects** structure that lets
//! the existing, unmodified bounded compiler explain a whole object:
//!
//! ```text
//! full frozen object
//!        │
//!        ▼
//! deterministic segmentation (65,536-frame ceiling, inherited from Phase K)
//!        │
//!        ├── segment 0 → exact U1 SampleObject
//!        ├── segment 1 → exact U1 SampleObject
//!        └── ...
//!        │
//!        ▼
//! canonical container: header ∥ segment index ∥ payloads ∥ integrity
//! ```
//!
//! Three things this container is deliberately **not**:
//!
//! * it is **not** a new U1 `Representation` tag. The frozen representation
//!   taxonomy is untouched; this is an archival structure whose segments are
//!   ordinary U1 SampleObjects. Phase K is unchanged.
//! * it is **not** content-adaptive. Segmentation is a frozen rule —
//!   `min(65,536, remaining)` consecutive intrinsic ranges — decided before any
//!   result exists, so boundary selection can never become a hidden search.
//! * it is **not** a cost estimate. Every segment stores the actual canonical
//!   serialized bytes of its selected representation, and the container's
//!   overhead is real serialized bytes (`header.len() + index.len() +
//!   Σ payload.len() + integrity`).
//!
//! Segment selection is frozen: the accepted exact candidate with the minimum
//! Phase-K `complete_bytes`, ties broken by the deterministic proposal order. No
//! measured quantity and no weighted score participates.
//!
//! Semantics are preserved, not just bytes: the root carries the finite extent
//! and the declared loop/one-shot identity, and observation across the root
//! reproduces both.

pub mod reader;

pub use reader::{FullObjectReader, ReadStats, VerifiedFullObject};

use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;
use crate::inverse::{
    self, CandidateKind, Intrinsic, ReferenceLibrary, SearchBudget, SelectedPayload,
};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::id::ContentId;
use crate::object::wavetable::Cycle;
use crate::object::{Constant, Literal, ObjectData};
use crate::universe::u1::PROFILE_TAG_BYTES;

/// Container format tag (15 bytes, like the entropy containers).
pub const FORMAT_TAG: &[u8; 15] = b"vole.fullobj.p1";
/// Container format version. Bumping this is a new container identity.
pub const FORMAT_VERSION: u8 = 1;

/// The frozen maximum segment length. Inherited *verbatim* from the Phase-K
/// observation ceiling so the choice is a consequence of already-frozen
/// machinery, not a tuned parameter.
pub const MAX_SEGMENT_FRAMES: u64 = crate::inverse::MAX_INVERSE_FRAMES;

/// Fixed container header length.
pub const HEADER_BYTES: usize = 15 + 1 + 4 + 1 + 1 + 4 + 8 + 4 + 8;
/// Fixed segment index record length:
/// `start(u64) ∥ frames(u64) ∥ representation_tag(u8) ∥ candidate_tag(u8) ∥
/// content_id(32) ∥ payload_offset(u64) ∥ payload_length(u64)`.
pub const INDEX_RECORD_BYTES: usize = 8 + 8 + 1 + 1 + 32 + 8 + 8;
/// Trailing integrity digest length (SHA-256 over everything before it).
pub const INTEGRITY_BYTES: u64 = 32;

/// Semantics code for a one-shot (finite, non-looping) root.
pub const SEMANTICS_ONE_SHOT: u8 = 0;
/// Semantics code for a looping root.
pub const SEMANTICS_LOOP: u8 = 1;

/// The root's declared identity semantics. A container that reproduced every
/// sample but forgot that an object loops would not be semantically equivalent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullSemantics {
    OneShot,
    Loop { period_frames: u32 },
}

impl FullSemantics {
    pub const fn kind_code(self) -> u8 {
        match self {
            FullSemantics::OneShot => SEMANTICS_ONE_SHOT,
            FullSemantics::Loop { .. } => SEMANTICS_LOOP,
        }
    }

    pub const fn period_frames(self) -> u32 {
        match self {
            FullSemantics::OneShot => 0,
            FullSemantics::Loop { period_frames } => period_frames,
        }
    }

    pub const fn is_loop(self) -> bool {
        matches!(self, FullSemantics::Loop { .. })
    }
}

/// One deterministic intrinsic segment `[start_frame, start_frame + frame_count)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub start_frame: u64,
    pub frame_count: u64,
}

/// The frozen segmentation rule: consecutive intrinsic ranges of at most
/// [`MAX_SEGMENT_FRAMES`] frames. No content-sensitive boundaries.
pub fn segment_plan(total_frames: u64) -> Result<Vec<Segment>> {
    if total_frames == 0 {
        return Err(Error::malformed("full object has no frames"));
    }
    if total_frames > crate::limits::MAX_OBJECT_FRAMES {
        return Err(Error::limit("full object exceeds the frame ceiling"));
    }
    let count = total_frames.div_ceil(MAX_SEGMENT_FRAMES) as usize;
    let mut out = Vec::with_capacity(count);
    let mut start = 0u64;
    while start < total_frames {
        let frame_count = (total_frames - start).min(MAX_SEGMENT_FRAMES);
        out.push(Segment {
            start_frame: start,
            frame_count,
        });
        start += frame_count;
    }
    debug_assert_eq!(out.len(), count);
    Ok(out)
}

/// Candidate-family name for a stored tag (evidence only; decode never depends
/// on it).
pub fn candidate_tag_name(tag: u8) -> &'static str {
    match tag {
        1 => "literal",
        2 => "silence",
        3 => "constant",
        4 => "exact_repeat",
        5 => "residual_zero",
        6 => "residual_constant",
        7 => "residual_periodic",
        8 => "shared_reference",
        _ => "unknown",
    }
}

/// One compiled segment: its plan, the selected representation, and the exact
/// canonical bytes that were stored.
#[derive(Debug, Clone)]
pub struct SegmentSelection {
    pub plan: Segment,
    pub kind: CandidateKind,
    pub representation: Representation,
    pub content_id: ContentId,
    /// The Phase-K selection objective (the candidate's `complete_bytes`).
    pub objective_bytes: u64,
    /// The actual canonical serialized payload (`payload.len()`).
    pub stored_bytes: u64,
    pub payload: Vec<u8>,
    pub proposed: usize,
    pub accepted: usize,
    pub search_ns: u64,
}

/// A compiled full object: metadata, per-segment selections, real byte
/// accounting, and the canonical serialized container.
#[derive(Debug, Clone)]
pub struct FullObject {
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub total_frames: u64,
    pub semantics: FullSemantics,
    pub segments: Vec<SegmentSelection>,
    pub header_bytes: u64,
    pub index_bytes: u64,
    pub payload_bytes: u64,
    pub integrity_bytes: u64,
    /// The canonical serialized container (the actual stored artifact).
    pub bytes: Vec<u8>,
    /// Total measured compile time (ns); never part of the artifact.
    pub compile_ns: u64,
}

impl FullObject {
    /// Real stored bytes: header + index + payloads + integrity.
    pub fn complete_bytes(&self) -> u64 {
        self.header_bytes
            .saturating_add(self.index_bytes)
            .saturating_add(self.payload_bytes)
            .saturating_add(self.integrity_bytes)
    }

    /// `Σ` of the per-segment Phase-K selection objectives (the economics that
    /// chose the segments).
    pub fn objective_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.objective_bytes).sum()
    }

    /// Container identity (SHA-256 over the canonical bytes).
    pub fn sha256(&self) -> [u8; 32] {
        Sha256::digest(&self.bytes)
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// True when every segment selected the universal literal fallback.
    pub fn all_literal(&self) -> bool {
        self.segments
            .iter()
            .all(|s| s.kind == CandidateKind::Literal)
    }

    /// True when no segment selected the literal fallback.
    pub fn none_literal(&self) -> bool {
        self.segments
            .iter()
            .all(|s| s.kind != CandidateKind::Literal)
    }

    /// True for a mix of literal and procedural segments.
    pub fn mixed(&self) -> bool {
        !self.all_literal() && !self.none_literal()
    }
}

/// Compile one full object: segment it with the frozen rule, run the bounded
/// Phase-K compiler on each segment against an **empty** reference library
/// (standalone, no corpus-level deduplication), select the minimum-`complete_bytes`
/// accepted candidate, serialize it, and build the canonical container.
pub fn compile_full_object(
    name: &str,
    sample_rate_hz: u32,
    channels: u8,
    total_frames: u64,
    semantics: FullSemantics,
    samples: &[i32],
    budget: SearchBudget,
) -> Result<FullObject> {
    if channels == 0 || u32::from(channels) > crate::limits::MAX_CHANNELS {
        return Err(Error::malformed("full object channel count out of domain"));
    }
    if sample_rate_hz == 0 || sample_rate_hz > crate::limits::MAX_SAMPLE_RATE_HZ {
        return Err(Error::malformed("full object sample rate out of domain"));
    }
    let ch = usize::from(channels);
    let expect = usize::try_from(total_frames)
        .ok()
        .and_then(|f| f.checked_mul(ch))
        .ok_or_else(|| Error::limit("full object sample count overflow"))?;
    if samples.len() != expect {
        return Err(Error::malformed(format!(
            "full object holds {} samples but {total_frames} frames x {channels} channels need {expect}",
            samples.len()
        )));
    }
    if let FullSemantics::Loop { period_frames } = semantics
        && (period_frames == 0 || u64::from(period_frames) > total_frames)
    {
        return Err(Error::malformed("loop period must be 1..=total frames"));
    }
    // The container itself is host-side, but each segment must be a legal
    // intrinsic window: the frozen segmentation rule guarantees this.
    let plan = segment_plan(total_frames)?;
    let sw = crate::evidence::timing::Stopwatch::start();
    let mut segments = Vec::with_capacity(plan.len());
    for (idx, seg) in plan.iter().enumerate() {
        let lo = seg.start_frame as usize * ch;
        let hi = (seg.start_frame + seg.frame_count) as usize * ch;
        let window = samples[lo..hi].to_vec();
        let intrinsic = Intrinsic::new(
            format!("{name}#{idx}@{}", seg.start_frame),
            channels,
            window,
        )?;
        // Empty library: standalone pricing, never corpus-level amortization.
        let (report, selected) =
            inverse::compile_selected(&intrinsic, &ReferenceLibrary::new(), budget, None)?;
        let selected: SelectedPayload = selected.ok_or_else(|| {
            Error::internal(format!(
                "segment {} produced no accepted candidate (Literal must always be accepted)",
                seg.start_frame
            ))
        })?;
        if selected.stored_bytes() == 0 {
            return Err(Error::internal(
                "selected representation serialized to zero bytes",
            ));
        }
        // The selection objective must be the physical artifact size, or
        // "minimum complete bytes" is not a storage claim. This is the
        // full-object form of the entropy cost invariant.
        if selected.objective_bytes != selected.stored_bytes() {
            return Err(Error::internal(format!(
                "segment {}: selection objective {} != stored bytes {} (the storage oracle \
                 must equal the physical artifact)",
                seg.start_frame,
                selected.objective_bytes,
                selected.stored_bytes()
            )));
        }
        segments.push(SegmentSelection {
            plan: *seg,
            kind: selected.kind,
            representation: selected.representation,
            content_id: selected.content_id,
            objective_bytes: selected.objective_bytes,
            stored_bytes: selected.stored_bytes(),
            payload: selected.bytes,
            proposed: report.proposed,
            accepted: report.accepted.len(),
            search_ns: report.search_ns,
        });
    }

    // Canonical serialization. Header, index and payload offsets are all fixed
    // little-endian fields; payload offsets are absolute container offsets.
    let index_bytes = (segments.len() * INDEX_RECORD_BYTES) as u64;
    let payload_start = HEADER_BYTES as u64 + index_bytes;
    let mut header: Vec<u8> = Vec::with_capacity(HEADER_BYTES);
    header.extend_from_slice(FORMAT_TAG);
    header.push(FORMAT_VERSION);
    header.extend_from_slice(&sample_rate_hz.to_le_bytes());
    header.push(channels);
    header.push(semantics.kind_code());
    header.extend_from_slice(&semantics.period_frames().to_le_bytes());
    header.extend_from_slice(&total_frames.to_le_bytes());
    header.extend_from_slice(&(segments.len() as u32).to_le_bytes());
    header.extend_from_slice(&MAX_SEGMENT_FRAMES.to_le_bytes());
    debug_assert_eq!(header.len(), HEADER_BYTES);

    let mut index: Vec<u8> = Vec::with_capacity(segments.len() * INDEX_RECORD_BYTES);
    let mut payloads: Vec<u8> = Vec::new();
    let mut offset = payload_start;
    for s in &segments {
        index.extend_from_slice(&s.plan.start_frame.to_le_bytes());
        index.extend_from_slice(&s.plan.frame_count.to_le_bytes());
        index.push(s.representation.tag());
        index.push(s.kind.tag());
        index.extend_from_slice(&s.content_id.to_bytes());
        index.extend_from_slice(&offset.to_le_bytes());
        index.extend_from_slice(&s.stored_bytes.to_le_bytes());
        payloads.extend_from_slice(&s.payload);
        offset += s.stored_bytes;
    }
    debug_assert_eq!(index.len() as u64, index_bytes);

    let mut bytes = Vec::with_capacity(offset as usize + INTEGRITY_BYTES as usize);
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&index);
    bytes.extend_from_slice(&payloads);
    let digest = Sha256::digest(&bytes);
    bytes.extend_from_slice(&digest);
    debug_assert_eq!(bytes.len() as u64, offset + INTEGRITY_BYTES);

    let compile_ns = sw.elapsed_ns().max(0) as u64;
    Ok(FullObject {
        sample_rate_hz,
        channels,
        total_frames,
        semantics,
        segments,
        header_bytes: HEADER_BYTES as u64,
        index_bytes,
        payload_bytes: payloads.len() as u64,
        integrity_bytes: INTEGRITY_BYTES,
        bytes,
        compile_ns,
    })
}

/// One decoded index record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedSegment {
    pub plan: Segment,
    pub representation: Representation,
    pub candidate_tag: u8,
    pub content_id: ContentId,
    pub payload_offset: u64,
    pub payload_length: u64,
}

/// A decoded container: the root's identity plus its segment index. Payload
/// bytes are borrowed from the container slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFullObject {
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub total_frames: u64,
    pub semantics: FullSemantics,
    pub segment_frames: u64,
    pub segments: Vec<DecodedSegment>,
    pub header_bytes: u64,
    pub index_bytes: u64,
    pub payload_bytes: u64,
    pub integrity_bytes: u64,
}

/// The set of candidate tags a stored representation may declare. The index is
/// canonical, so an incompatible pairing (e.g. `silence` with
/// `residual_periodic`) is a malformed container, not harmless metadata.
fn candidate_tag_matches(rep: Representation, tag: u8) -> bool {
    match rep {
        Representation::Literal => tag == CandidateKind::Literal.tag(),
        Representation::Silence => tag == CandidateKind::Silence.tag(),
        Representation::Constant => tag == CandidateKind::Constant.tag(),
        Representation::ExactRepeat => tag == CandidateKind::ExactRepeat.tag(),
        Representation::PredictorResidual => matches!(
            tag,
            x if x == CandidateKind::ResidualZero.tag()
                || x == CandidateKind::ResidualConstant.tag()
                || x == CandidateKind::ResidualPeriodic.tag()
        ),
        Representation::Referenced => tag == CandidateKind::SharedReference.tag(),
        _ => false,
    }
}

fn le_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([
        b[at],
        b[at + 1],
        b[at + 2],
        b[at + 3],
        b[at + 4],
        b[at + 5],
        b[at + 6],
        b[at + 7],
    ])
}

/// Parse and integrity-check a canonical full-object container.
pub fn decode_full_object(bytes: &[u8]) -> Result<DecodedFullObject> {
    let min = HEADER_BYTES + INTEGRITY_BYTES as usize;
    if bytes.len() < min {
        return Err(Error::malformed("full-object container is truncated"));
    }
    let body_len = bytes.len() - INTEGRITY_BYTES as usize;
    let (body, integrity) = bytes.split_at(body_len);
    if Sha256::digest(body).as_slice() != integrity {
        return Err(Error::malformed(
            "full-object container integrity digest does not verify",
        ));
    }
    if &body[..15] != FORMAT_TAG {
        return Err(Error::malformed("full-object container tag mismatch"));
    }
    if body[15] != FORMAT_VERSION {
        return Err(Error::malformed("full-object container version mismatch"));
    }
    let sample_rate_hz = u32::from_le_bytes([body[16], body[17], body[18], body[19]]);
    let channels = body[20];
    let semantics_kind = body[21];
    let period_frames = u32::from_le_bytes([body[22], body[23], body[24], body[25]]);
    let total_frames = le_u64(body, 26);
    let segment_count = u32::from_le_bytes([body[34], body[35], body[36], body[37]]) as usize;
    let segment_frames = le_u64(body, 38);

    if channels == 0 || u32::from(channels) > crate::limits::MAX_CHANNELS {
        return Err(Error::malformed("full-object channel count out of domain"));
    }
    if sample_rate_hz == 0 || sample_rate_hz > crate::limits::MAX_SAMPLE_RATE_HZ {
        return Err(Error::malformed("full-object sample rate out of domain"));
    }
    // The frozen segmentation rule is part of the format: a container that
    // declares a different ceiling or a different plan is rejected.
    if segment_frames != MAX_SEGMENT_FRAMES {
        return Err(Error::malformed(
            "full-object segment ceiling is not the frozen value",
        ));
    }
    let plan = segment_plan(total_frames)?;
    if segment_count != plan.len() {
        return Err(Error::malformed(
            "full-object segment count does not match the frozen segmentation rule",
        ));
    }
    let semantics = match semantics_kind {
        SEMANTICS_ONE_SHOT => {
            if period_frames != 0 {
                return Err(Error::malformed("one-shot root carries a loop period"));
            }
            FullSemantics::OneShot
        }
        SEMANTICS_LOOP => {
            if period_frames == 0 || u64::from(period_frames) > total_frames {
                return Err(Error::malformed("loop root period out of domain"));
            }
            FullSemantics::Loop { period_frames }
        }
        _ => return Err(Error::malformed("unknown full-object semantics code")),
    };

    let index_bytes = segment_count * INDEX_RECORD_BYTES;
    if body.len() < HEADER_BYTES + index_bytes {
        return Err(Error::malformed("full-object index is truncated"));
    }
    let payload_start = (HEADER_BYTES + index_bytes) as u64;
    let payload_end = body_len as u64;
    let mut segments = Vec::with_capacity(segment_count);
    // The serializer writes the payloads as one canonical contiguous block, in
    // segment order, immediately after the index. Requiring that layout rejects
    // gaps, overlaps, aliased/duplicated offsets and unreferenced trailing bytes
    // — noncanonical encodings of a format that claims to be canonical.
    let mut expected_offset = payload_start;
    for (i, expect) in plan.iter().enumerate() {
        let at = HEADER_BYTES + i * INDEX_RECORD_BYTES;
        let start_frame = le_u64(body, at);
        let frame_count = le_u64(body, at + 8);
        let representation = Representation::from_tag(body[at + 16])
            .ok_or_else(|| Error::malformed("unknown segment representation tag"))?;
        let candidate_tag = body[at + 17];
        if !candidate_tag_matches(representation, candidate_tag) {
            return Err(Error::malformed(
                "full-object segment candidate tag is incompatible with its representation",
            ));
        }
        let mut cid = [0u8; 32];
        cid.copy_from_slice(&body[at + 18..at + 50]);
        let payload_offset = le_u64(body, at + 50);
        let payload_length = le_u64(body, at + 58);
        // The plan is frozen, so a segment whose boundaries differ is invalid.
        if start_frame != expect.start_frame || frame_count != expect.frame_count {
            return Err(Error::malformed(
                "full-object segment boundaries differ from the frozen plan",
            ));
        }
        if payload_length == 0 {
            return Err(Error::malformed("full-object segment payload is empty"));
        }
        if payload_offset != expected_offset {
            return Err(Error::malformed(
                "full-object payload offsets are not the canonical contiguous layout",
            ));
        }
        let end = payload_offset
            .checked_add(payload_length)
            .ok_or_else(|| Error::malformed("full-object payload range overflow"))?;
        if end > payload_end {
            return Err(Error::malformed(
                "full-object segment payload range outside the payload region",
            ));
        }
        expected_offset = end;
        segments.push(DecodedSegment {
            plan: *expect,
            representation,
            candidate_tag,
            content_id: ContentId::from_bytes(cid),
            payload_offset,
            payload_length,
        });
    }
    if expected_offset != payload_end {
        return Err(Error::malformed(
            "full-object container has unreferenced trailing payload bytes",
        ));
    }

    Ok(DecodedFullObject {
        sample_rate_hz,
        channels,
        total_frames,
        semantics,
        segment_frames,
        segments,
        header_bytes: HEADER_BYTES as u64,
        index_bytes: index_bytes as u64,
        payload_bytes: payload_end - payload_start,
        integrity_bytes: INTEGRITY_BYTES,
    })
}

/// Reconstruct the exact semantic U1 object stored by one segment: the
/// descriptor and `ObjectData` the index's `content_id` claims.
pub(crate) fn segment_object(
    rep: Representation,
    payload: &[u8],
    channels: u8,
    frame_count: u64,
) -> Result<(ObjectDescriptor, ObjectData)> {
    match rep {
        Representation::Literal => {
            let rl = crate::entropy::represent::parse_literal_container(payload)?;
            if rl.descriptor.extent_frames != frame_count
                || rl.descriptor.layout.count() != channels
            {
                return Err(Error::malformed(
                    "stored literal segment disagrees with the index geometry",
                ));
            }
            let samples = rl.materialize_full()?;
            let lit = Literal::new(&rl.descriptor, samples)
                .ok_or_else(|| Error::malformed("stored literal segment payload out of domain"))?;
            Ok((rl.descriptor, ObjectData::Literal(lit)))
        }
        Representation::PredictorResidual => {
            let rr = crate::entropy::represent::parse_residual_container(payload)?;
            if rr.descriptor.extent_frames != frame_count
                || rr.descriptor.layout.count() != channels
            {
                return Err(Error::malformed(
                    "stored residual segment disagrees with the index geometry",
                ));
            }
            let residual = rr.reconstruct_full()?;
            Ok((rr.descriptor, ObjectData::PredictorResidual(residual)))
        }
        Representation::Silence | Representation::Constant | Representation::ExactRepeat => {
            let (descriptor, data) = parse_canonical_segment(payload)?;
            if descriptor.layout.count() != channels {
                return Err(Error::malformed(
                    "stored canonical segment channel count disagrees with the index",
                ));
            }
            Ok((descriptor, data))
        }
        other => Err(Error::new(
            crate::error::Kind::Unsupported,
            format!("full-object container cannot decode segment representation {other}"),
        )),
    }
}

pub(crate) fn data_representation(data: &ObjectData) -> Representation {
    match data {
        ObjectData::Literal(_) => Representation::Literal,
        ObjectData::Referenced(_) => Representation::Referenced,
        ObjectData::Silence => Representation::Silence,
        ObjectData::Constant(_) => Representation::Constant,
        ObjectData::Wavetable(_) => Representation::Wavetable,
        ObjectData::SingleCycle(_) => Representation::SingleCycle,
        ObjectData::ExactRepeat(_) => Representation::ExactRepeat,
        ObjectData::Oscillator(_) => Representation::Oscillator,
        ObjectData::PartialBank(_) => Representation::PartialBank,
        ObjectData::Noise(_) => Representation::Noise,
        ObjectData::PredictorResidual(_) => Representation::PredictorResidual,
    }
}

/// Parse a canonical object in the subset the container stores: `Silence`,
/// `Constant` and the cycle family. Anything else is rejected (never silently
/// reinterpreted).
pub(crate) fn parse_canonical_segment(bytes: &[u8]) -> Result<(ObjectDescriptor, ObjectData)> {
    let header = PROFILE_TAG_BYTES.len() + 1 + 8 + 1 + 1 + 16;
    if bytes.len() < header {
        return Err(Error::malformed("canonical segment header is truncated"));
    }
    if &bytes[..PROFILE_TAG_BYTES.len()] != PROFILE_TAG_BYTES {
        return Err(Error::malformed("canonical segment profile tag mismatch"));
    }
    let mut at = PROFILE_TAG_BYTES.len();
    let representation = Representation::from_tag(bytes[at])
        .ok_or_else(|| Error::malformed("unknown canonical segment representation tag"))?;
    at += 1;
    let extent_frames = le_u64(bytes, at);
    at += 8;
    let channels = bytes[at];
    at += 1;
    let layout = crate::universe::layout::Layout::checked(channels)
        .ok_or_else(|| Error::malformed("canonical segment channel count out of domain"))?;
    let loop_flag = bytes[at];
    at += 1;
    let loop_start = le_u64(bytes, at);
    at += 8;
    let loop_end = le_u64(bytes, at);
    at += 8;
    let loop_region = match loop_flag {
        0 => None,
        1 => Some(
            crate::object::descriptor::LoopRegion::new(loop_start, loop_end)
                .ok_or_else(|| Error::malformed("canonical segment loop region out of domain"))?,
        ),
        _ => {
            return Err(Error::malformed(
                "canonical segment loop flag out of domain",
            ));
        }
    };
    let descriptor = ObjectDescriptor::new(representation, extent_frames, layout, loop_region)
        .ok_or_else(|| Error::malformed("canonical segment descriptor out of domain"))?;
    let payload = &bytes[at..];
    let data = match representation {
        Representation::Silence => {
            if !payload.is_empty() {
                return Err(Error::malformed("silence segment carries a payload"));
            }
            ObjectData::Silence
        }
        Representation::Constant => {
            if payload.len() != 4 {
                return Err(Error::malformed("constant segment payload is not 4 bytes"));
            }
            ObjectData::Constant(Constant::new(i32::from_le_bytes([
                payload[0], payload[1], payload[2], payload[3],
            ])))
        }
        Representation::ExactRepeat => {
            if payload.len() < 16 {
                return Err(Error::malformed("cycle segment payload is truncated"));
            }
            let len_frames = le_u64(payload, 0);
            let sample_count = le_u64(payload, 8);
            if len_frames != extent_frames {
                return Err(Error::malformed(
                    "cycle segment length disagrees with its header",
                ));
            }
            let body = &payload[16..];
            if body.len() as u64 != sample_count * 4 {
                return Err(Error::malformed("cycle segment sample block is truncated"));
            }
            let mut samples = Vec::with_capacity(sample_count as usize);
            for w in body.as_chunks::<4>().0 {
                samples.push(i32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            }
            let cycle = Cycle::new(&descriptor, samples)
                .ok_or_else(|| Error::malformed("cycle segment payload out of domain"))?;
            ObjectData::ExactRepeat(cycle)
        }
        other => {
            return Err(Error::new(
                crate::error::Kind::Unsupported,
                format!("canonical segment representation {other} is not storable in v1"),
            ));
        }
    };
    Ok((descriptor, data))
}

/// Borrow one segment's stored payload from a container.
pub(crate) fn segment_payload<'a>(bytes: &'a [u8], seg: &DecodedSegment) -> Result<&'a [u8]> {
    let lo = seg.payload_offset as usize;
    let hi = lo + seg.payload_length as usize;
    bytes
        .get(lo..hi)
        .ok_or_else(|| Error::malformed("full-object segment payload outside the container"))
}

/// Bind every segment's index `content_id` to the semantic object it stores,
/// without retaining the reconstructed waveform.
pub(crate) fn verify_segment_bindings(bytes: &[u8], decoded: &DecodedFullObject) -> Result<()> {
    for seg in &decoded.segments {
        let payload = segment_payload(bytes, seg)?;
        let (descriptor, data) = segment_object(
            seg.representation,
            payload,
            decoded.channels,
            seg.plan.frame_count,
        )?;
        if data_representation(&data) != seg.representation {
            return Err(Error::malformed(
                "stored segment representation disagrees with the index",
            ));
        }
        if crate::object::canonical_content_id(&descriptor, &data) != seg.content_id {
            return Err(Error::malformed(
                "full-object index content_id does not match the stored segment",
            ));
        }
    }
    Ok(())
}

/// A container that has been fully decoded and materialized: the exact samples
/// plus the root's declared semantics.
#[derive(Debug, Clone)]
pub struct MaterializedFullObject {
    pub decoded: DecodedFullObject,
    pub samples: Vec<i32>,
}

impl MaterializedFullObject {
    pub fn channels(&self) -> usize {
        usize::from(self.decoded.channels)
    }

    pub fn total_frames(&self) -> u64 {
        self.decoded.total_frames
    }

    /// Canonical interleaved samples of the finite extent.
    pub fn samples(&self) -> &[i32] {
        &self.samples
    }

    /// Observe the root at `[start, start + frames)`, honouring the declared
    /// semantics:
    ///
    /// * within the finite extent, the stored samples are authoritative;
    /// * beyond a **loop** root's finite extent, the loop region `[0, period)`
    ///   is repeated; a **one-shot** root has no continuation and errors.
    pub fn observe(&self, start: u64, frames: u64) -> Result<Vec<i32>> {
        let ch = self.channels();
        if frames == 0 {
            return Err(Error::malformed("zero-frame full-object observation"));
        }
        let end = start
            .checked_add(frames)
            .ok_or_else(|| Error::limit("full-object observation overflow"))?;
        let mut out = vec![0i32; frames as usize * ch];
        match self.decoded.semantics {
            FullSemantics::OneShot => {
                if end > self.decoded.total_frames {
                    return Err(Error::malformed(
                        "one-shot full-object observation runs past the finite extent",
                    ));
                }
                let lo = start as usize * ch;
                let hi = end as usize * ch;
                out.copy_from_slice(&self.samples[lo..hi]);
            }
            FullSemantics::Loop { period_frames } => {
                let period = u64::from(period_frames);
                for i in 0..frames {
                    let pos = start + i;
                    let src = if pos < self.decoded.total_frames {
                        pos
                    } else {
                        (pos - self.decoded.total_frames) % period
                    };
                    let s = src as usize * ch;
                    out[i as usize * ch..i as usize * ch + ch]
                        .copy_from_slice(&self.samples[s..s + ch]);
                }
            }
        }
        Ok(out)
    }
}

/// Decode a container and materialize its exact full extent.
pub fn materialize_full_object(bytes: &[u8]) -> Result<MaterializedFullObject> {
    let decoded = decode_full_object(bytes)?;
    verify_segment_bindings(bytes, &decoded)?;
    let mut samples =
        Vec::with_capacity(decoded.total_frames as usize * usize::from(decoded.channels));
    for seg in &decoded.segments {
        let payload = segment_payload(bytes, seg)?;
        let (descriptor, data) = segment_object(
            seg.representation,
            payload,
            decoded.channels,
            seg.plan.frame_count,
        )?;
        let mut seg_samples = crate::inverse::observe::intrinsic_reconstruction(
            &descriptor,
            &data,
            seg.plan.frame_count,
        )?;
        let expect = seg.plan.frame_count as usize * usize::from(decoded.channels);
        if seg_samples.len() != expect {
            return Err(Error::malformed("stored segment has the wrong length"));
        }
        samples.append(&mut seg_samples);
    }
    let expect = decoded.total_frames as usize * usize::from(decoded.channels);
    if samples.len() != expect {
        return Err(Error::malformed(format!(
            "materialized {} samples, expected {expect}",
            samples.len()
        )));
    }
    Ok(MaterializedFullObject { decoded, samples })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::SearchBudget;

    /// Deterministic fixture generator (tests only).
    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn budget() -> SearchBudget {
        SearchBudget {
            max_period_scan: 256,
            max_residual_period_candidates: 2,
            max_candidates: 16,
            ..SearchBudget::default()
        }
    }

    fn tone(frames: usize, channels: usize) -> Vec<i32> {
        (0..frames)
            .flat_map(|f| {
                (0..channels).map(move |c| (((f % 64) as i32) << 14).wrapping_sub((c as i32) * 7))
            })
            .collect()
    }

    fn noise(frames: usize, channels: usize, seed: u64) -> Vec<i32> {
        let mut s = seed;
        (0..frames * channels)
            .map(|_| splitmix(&mut s) as i32)
            .collect()
    }

    #[test]
    fn segmentation_rule_is_frozen_at_the_phase_k_ceiling() {
        assert_eq!(MAX_SEGMENT_FRAMES, 65_536);
        assert_eq!(MAX_SEGMENT_FRAMES, crate::inverse::MAX_INVERSE_FRAMES);
        for (total, expected) in [
            (1u64, vec![(0, 1)]),
            (65_535, vec![(0, 65_535)]),
            (65_536, vec![(0, 65_536)]),
            (65_537, vec![(0, 65_536), (65_536, 1)]),
            (131_072, vec![(0, 65_536), (65_536, 65_536)]),
            (131_073, vec![(0, 65_536), (65_536, 65_536), (131_072, 1)]),
        ] {
            let plan = segment_plan(total).unwrap();
            let got: Vec<(u64, u64)> = plan
                .iter()
                .map(|s| (s.start_frame, s.frame_count))
                .collect();
            assert_eq!(got, expected, "total {total}");
        }
        assert!(segment_plan(0).is_err());
    }

    #[test]
    fn container_round_trips_every_boundary_length_and_channel_count() {
        for channels in [1u8, 2, 3] {
            for total in [1u64, 65_535, 65_536, 65_537, 131_072, 131_073] {
                let ch = usize::from(channels);
                let samples = tone(total as usize, ch);
                let obj = compile_full_object(
                    "boundary",
                    48_000,
                    channels,
                    total,
                    FullSemantics::OneShot,
                    &samples,
                    budget(),
                )
                .unwrap_or_else(|e| panic!("{channels}ch/{total}f: {e}"));
                assert_eq!(obj.complete_bytes(), obj.bytes.len() as u64);
                assert_eq!(
                    obj.complete_bytes(),
                    obj.header_bytes + obj.index_bytes + obj.payload_bytes + obj.integrity_bytes
                );
                // "Minimum complete bytes" must be a physical-storage claim.
                for s in &obj.segments {
                    assert_eq!(
                        s.objective_bytes, s.stored_bytes,
                        "{channels}ch/{total}f segment @{}: objective != stored",
                        s.plan.start_frame
                    );
                }
                let mat = materialize_full_object(&obj.bytes)
                    .unwrap_or_else(|e| panic!("{channels}ch/{total}f decode: {e}"));
                assert_eq!(
                    mat.samples(),
                    &samples[..],
                    "{channels}ch/{total}f reconstruction"
                );
                assert_eq!(mat.total_frames(), total);
                assert_eq!(mat.decoded.semantics, FullSemantics::OneShot);
            }
        }
    }

    #[test]
    fn reconstruction_is_exact_on_noise_and_mixed_content() {
        // High-entropy content (literal segments) plus a silent tail (procedural
        // segments): the container must be exact across the boundary either way.
        let total = 131_073usize;
        let mut samples = noise(65_536, 1, 0x1234_5678_9abc_def0);
        samples.extend(std::iter::repeat_n(0i32, total - 65_536));
        let obj = compile_full_object(
            "mixed",
            48_000,
            1,
            total as u64,
            FullSemantics::OneShot,
            &samples,
            budget(),
        )
        .unwrap();
        assert!(obj.mixed(), "expected a mixed literal/procedural object");
        let mat = materialize_full_object(&obj.bytes).unwrap();
        assert_eq!(mat.samples(), &samples[..]);
    }

    #[test]
    fn observation_across_segment_boundaries_is_exact() {
        let total = 131_073u64;
        let samples = tone(total as usize, 2);
        let obj = compile_full_object(
            "boundary-obs",
            48_000,
            2,
            total,
            FullSemantics::OneShot,
            &samples,
            budget(),
        )
        .unwrap();
        let mat = materialize_full_object(&obj.bytes).unwrap();
        let ch = 2usize;
        for start in [65_535u64, 65_536, 65_537, 131_071] {
            for frames in [1u64, 2, 3] {
                if start + frames > total {
                    continue;
                }
                let got = mat.observe(start, frames).unwrap();
                let lo = start as usize * ch;
                let hi = (start + frames) as usize * ch;
                assert_eq!(got, &samples[lo..hi], "start {start} frames {frames}");
            }
        }
        // A window spanning the first boundary.
        let got = mat.observe(65_535, 3).unwrap();
        assert_eq!(got, &samples[65_535 * ch..65_538 * ch]);
        // The last frame.
        let got = mat.observe(total - 1, 1).unwrap();
        assert_eq!(
            got,
            &samples[(total as usize - 1) * ch..total as usize * ch]
        );
    }

    #[test]
    fn loop_semantics_survive_the_container_and_wrap() {
        // A 64-frame-periodic loop longer than one segment.
        let period = 64usize;
        let total = 65_537u64;
        let base = tone(period, 1);
        let samples: Vec<i32> = (0..total as usize).map(|f| base[f % period]).collect();
        let obj = compile_full_object(
            "loop",
            48_000,
            1,
            total,
            FullSemantics::Loop {
                period_frames: period as u32,
            },
            &samples,
            budget(),
        )
        .unwrap();
        let mat = materialize_full_object(&obj.bytes).unwrap();
        assert_eq!(mat.samples(), &samples[..]);
        assert_eq!(
            mat.decoded.semantics,
            FullSemantics::Loop { period_frames: 64 }
        );
        // Within the extent: exact.
        assert_eq!(mat.observe(65_529, 8).unwrap(), &samples[65_529..65_537]);
        // A window straddling the finite end wraps into the declared loop region.
        let straddle = mat.observe(total - 1, 3).unwrap();
        assert_eq!(straddle, vec![samples[65_536], samples[0], samples[1]]);
        // Beyond the extent: the declared loop region repeats.
        assert_eq!(mat.observe(total, 4).unwrap(), &samples[0..4]);
        assert_eq!(mat.observe(total + 3, 2).unwrap(), &samples[3..5]);

        // A one-shot root has no continuation.
        let one = compile_full_object(
            "one-shot",
            48_000,
            1,
            total,
            FullSemantics::OneShot,
            &samples,
            budget(),
        )
        .unwrap();
        let mat = materialize_full_object(&one.bytes).unwrap();
        assert!(mat.observe(total, 1).is_err());
    }

    #[test]
    fn standalone_pricing_never_uses_corpus_deduplication() {
        // Two identical objects compiled separately must each pay in full: no
        // cross-object reference may appear (`shared_reference` is never
        // selected with an empty library).
        let samples = tone(4096, 1);
        for _ in 0..2 {
            let obj = compile_full_object(
                "standalone",
                48_000,
                1,
                4096,
                FullSemantics::OneShot,
                &samples,
                budget(),
            )
            .unwrap();
            assert!(
                obj.segments
                    .iter()
                    .all(|s| s.kind != CandidateKind::SharedReference)
            );
        }
    }

    /// Recompute the trailing integrity digest so a structural mutation reaches
    /// the parser instead of failing the outer digest.
    fn reseal(bytes: &mut [u8]) {
        let n = bytes.len();
        let body = n - INTEGRITY_BYTES as usize;
        let digest = Sha256::digest(&bytes[..body]);
        bytes[body..].copy_from_slice(&digest);
    }

    #[test]
    fn resealed_structural_mutations_must_fail() {
        // A container whose outer digest is *valid*: only the format's own
        // invariants can reject these mutations.
        let samples = tone(40_000, 2);
        let obj = compile_full_object(
            "resealed",
            48_000,
            2,
            40_000,
            FullSemantics::OneShot,
            &samples,
            budget(),
        )
        .unwrap();
        let index0 = HEADER_BYTES;
        type Mutation = Box<dyn Fn(&mut [u8])>;
        let cases: Vec<(&str, Mutation)> = vec![
            (
                "version",
                Box::new(|b: &mut [u8]| b[15] = FORMAT_VERSION.wrapping_add(1)),
            ),
            (
                "ceiling",
                Box::new(|b: &mut [u8]| b[38..46].copy_from_slice(&32_768u64.to_le_bytes())),
            ),
            ("semantics", Box::new(|b: &mut [u8]| b[21] = 0x7F)),
            (
                "one_shot_with_period",
                Box::new(|b: &mut [u8]| b[22..26].copy_from_slice(&5u32.to_le_bytes())),
            ),
            (
                "segment_count",
                Box::new(|b: &mut [u8]| {
                    let n = u32::from_le_bytes([b[34], b[35], b[36], b[37]]);
                    b[34..38].copy_from_slice(&n.wrapping_add(1).to_le_bytes());
                }),
            ),
            (
                "boundary",
                Box::new(move |b: &mut [u8]| {
                    let v = u64::from_le_bytes(b[index0..index0 + 8].try_into().unwrap());
                    b[index0..index0 + 8].copy_from_slice(&v.wrapping_add(1).to_le_bytes());
                }),
            ),
            (
                "representation_tag",
                Box::new(move |b: &mut [u8]| b[index0 + 16] = 0x00),
            ),
            (
                "candidate_tag",
                Box::new(move |b: &mut [u8]| b[index0 + 17] = 0xFF),
            ),
            (
                "content_id",
                Box::new(move |b: &mut [u8]| b[index0 + 18] ^= 0xFF),
            ),
            (
                "offset_gap",
                Box::new(move |b: &mut [u8]| {
                    let v = u64::from_le_bytes(b[index0 + 50..index0 + 58].try_into().unwrap());
                    b[index0 + 50..index0 + 58].copy_from_slice(&v.wrapping_add(1).to_le_bytes());
                }),
            ),
            (
                "payload_short",
                Box::new(move |b: &mut [u8]| {
                    let v = u64::from_le_bytes(b[index0 + 58..index0 + 66].try_into().unwrap());
                    if v > 1 {
                        b[index0 + 58..index0 + 66].copy_from_slice(&(v - 1).to_le_bytes());
                    }
                }),
            ),
        ];
        for (label, mutate) in &cases {
            let mut candidate = obj.bytes.clone();
            mutate(&mut candidate);
            reseal(&mut candidate);
            // A resealed structural mutation may be rejected at parse time or at
            // materialization time (the `content_id` binding is the latter), but
            // it must never be usable.
            let usable = decode_full_object(&candidate).is_ok()
                && materialize_full_object(&candidate).is_ok();
            assert!(!usable, "resealed mutation '{label}' remained usable");
        }
        // The unmutated container still decodes and materializes exactly.
        assert_eq!(
            materialize_full_object(&obj.bytes).unwrap().samples(),
            &samples[..]
        );
    }

    #[test]
    fn hostile_containers_fail_typed() {
        let samples = tone(4096, 1);
        let obj = compile_full_object(
            "hostile",
            48_000,
            1,
            4096,
            FullSemantics::OneShot,
            &samples,
            budget(),
        )
        .unwrap();
        // Truncated.
        assert!(decode_full_object(&obj.bytes[..obj.bytes.len() - 4]).is_err());
        // Corrupted body -> integrity failure.
        let mut corrupted = obj.bytes.clone();
        corrupted[HEADER_BYTES + 3] ^= 0xFF;
        assert!(decode_full_object(&corrupted).is_err());
        // Wrong version.
        let mut version = obj.bytes.clone();
        version[15] = 2;
        assert!(decode_full_object(&version).is_err());
        // Wrong segment ceiling (a plan the format does not allow).
        let mut ceiling = obj.bytes.clone();
        ceiling[38..46].copy_from_slice(&32_768u64.to_le_bytes());
        assert!(decode_full_object(&ceiling).is_err());
        // Empty payload declared in the index.
        let mut empty = obj.bytes.clone();
        let len_at = HEADER_BYTES + 58;
        empty[len_at..len_at + 8].copy_from_slice(&0u64.to_le_bytes());
        assert!(decode_full_object(&empty).is_err());
        // The unmutated container decodes.
        assert!(decode_full_object(&obj.bytes).is_ok());
    }

    #[test]
    fn selection_is_minimum_complete_bytes_with_deterministic_ties() {
        // Silence > one segment: every segment must select the procedural
        // explanation, and compiling twice must be byte-identical.
        let total = 131_073u64;
        let samples = vec![0i32; total as usize];
        let a = compile_full_object(
            "silence",
            48_000,
            1,
            total,
            FullSemantics::OneShot,
            &samples,
            budget(),
        )
        .unwrap();
        let b = compile_full_object(
            "silence",
            48_000,
            1,
            total,
            FullSemantics::OneShot,
            &samples,
            budget(),
        )
        .unwrap();
        assert_eq!(a.bytes, b.bytes, "compilation must be deterministic");
        assert!(a.none_literal());
        for s in &a.segments {
            assert_eq!(s.kind, CandidateKind::Silence);
            assert!(s.stored_bytes > 0);
            assert!(s.accepted >= 1 && s.proposed >= s.accepted);
        }
        assert_eq!(
            materialize_full_object(&a.bytes).unwrap().samples(),
            &samples[..]
        );
    }
}
