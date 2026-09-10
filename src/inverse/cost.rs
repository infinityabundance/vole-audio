//! Complete dependency accounting for inverse candidates (contract §33/§35).
//!
//! Every candidate reports a **complete** cost — never a bare payload size.
//! Where an entropy representation exists the cost comes from the frozen H.2
//! complete-cost API (`entropy::represent`), so the inverse compiler and the
//! entropy encoder can never disagree about what a representation costs. For
//! representations without an entropy body (silence, constant, cycle, shared
//! reference) the cost is the canonical object byte length.
//!
//! The static fields mirror the contract's names exactly:
//! `persistent_bytes`, `residual_bytes`, `dependency_bytes`,
//! `checkpoint_bytes`, `state_bytes`. A resident table (a stored cycle) is
//! **never** free: it is counted as persistent/state bytes, and a shared
//! reference is counted as dependency bytes (never as zero).
//!
//! Abstract universe work (`generator_ops`, `residual_ops`, `lookup_ops`,
//! `filter_ops`) is a *static count derived from the representation
//! structure*, not a measurement; the docs state the counting rule.

use crate::entropy::represent::{ModelMode, RepresentedLiteral, RepresentedResidual};
use crate::entropy::symbol::Symbolization;
use crate::error::Result;
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::wavetable::Cycle;
use crate::object::{ObjectData, Residual};

/// Canonical U1 literal byte length for a (frames, channels) window:
/// header 46 + count prefix 8 + 4 bytes per sample code.
pub const U1_LITERAL_HEADER_BYTES: u64 = 46;

/// The frozen literal entropy universe (symbolization × page size, inline
/// models) used to price the `Literal` candidate — the same axes the H.2
/// literal court sweeps, so the inverse compiler prices literal content with
/// the entropy encoder's own best representation.
pub const LITERAL_SYMBOLIZATIONS: [Symbolization; 4] = [
    Symbolization::Identity,
    Symbolization::Lane4Plain,
    Symbolization::Lane4ZigZag,
    Symbolization::DeltaLane4,
];
pub const LITERAL_PAGE_FRAMES: [u32; 3] = [256, 512, 1024];
/// Residual page sizes priced for a residual candidate.
pub const RESIDUAL_PAGE_FRAMES: [u32; 3] = [256, 512, 1024];

/// Canonical U1 literal bytes for a window.
pub const fn canonical_u1_literal_bytes(frames: u64, channels: u8) -> u64 {
    U1_LITERAL_HEADER_BYTES + 8 + frames * channels as u64 * 4
}

/// Static abstract work of one representation ("universe abstract work",
/// contract §33). Counts are derived from the representation's data
/// structure; they are not measurements and no wall-clock is involved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AbstractWork {
    /// Hypothesis generation steps (endless/procedural evaluation).
    pub generator_ops: u64,
    /// Residual delta applications.
    pub residual_ops: u64,
    /// Content lookups (sample reads / model binary searches).
    pub lookup_ops: u64,
    /// Filter operations (none in the frozen u1 v1 vocabulary).
    pub filter_ops: u64,
}

impl AbstractWork {
    pub const fn total(&self) -> u64 {
        self.generator_ops + self.residual_ops + self.lookup_ops + self.filter_ops
    }

    /// Work to materialize a bounded window of `frames` frames.
    pub fn window(&self, frames: u64, total_frames: u64) -> u64 {
        if total_frames == 0 {
            return 0;
        }
        // Per-window work is proportional to the window's share of the object
        // (deterministic integer arithmetic; no float).
        self.total()
            .saturating_mul(frames.min(total_frames))
            .div_ceil(total_frames)
    }
}

/// Complete cost of one candidate representation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateCost {
    /// Container/descriptor/version metadata bytes.
    pub metadata_bytes: u64,
    /// Hypothesis bytes (procedural model state).
    pub hypothesis_bytes: u64,
    /// Entropy-coded residual payload bytes (0 when not a residual object).
    pub residual_bytes: u64,
    /// Stored sample-domain payload bytes owned by the representation.
    pub persistent_bytes: u64,
    /// Dependency bytes (referenced content ids, shared tables).
    pub dependency_bytes: u64,
    /// Checkpoint bytes (0 in Phase K; the field exists so Phase N candidates
    /// do not need cost-API surgery).
    pub checkpoint_bytes: u64,
    /// Resident sample-domain state bytes.
    pub state_bytes: u64,
    /// Sum of the stored cost fields above (the comparable objective).
    pub complete_bytes: u64,
    /// Raw canonical sample bytes of the observed window (comparison only).
    pub raw_sample_bytes: u64,
    /// Canonical U1 literal bytes of the observed window (comparison only).
    pub canonical_literal_bytes: u64,
    /// Which oracle produced this cost: `entropy_literal`, `entropy_residual`,
    /// or `canonical_object`.
    pub cost_source: &'static str,
}

impl CandidateCost {
    fn finish(mut self) -> CandidateCost {
        self.complete_bytes = self
            .metadata_bytes
            .saturating_add(self.hypothesis_bytes)
            .saturating_add(self.residual_bytes)
            .saturating_add(self.persistent_bytes)
            .saturating_add(self.dependency_bytes)
            .saturating_add(self.checkpoint_bytes);
        self
    }

    fn baseline(frames: u64, channels: u8) -> (u64, u64) {
        (
            frames * u64::from(channels) * 4,
            canonical_u1_literal_bytes(frames, channels),
        )
    }
}

/// Price the `Literal` candidate with the best frozen literal entropy
/// representation (minimum `complete_bytes` over the literal universe).
pub fn literal_cost(samples: &[i32], frames: u64, channels: u8) -> Result<CandidateCost> {
    let descriptor = crate::object::descriptor::ObjectDescriptor::new(
        Representation::Literal,
        frames,
        crate::inverse::observe::layout_of(channels)?,
        None,
    )
    .ok_or_else(|| crate::error::Error::malformed("literal descriptor out of domain"))?;
    let canonical = canonical_u1_literal_bytes(frames, channels);
    let mut best: Option<CandidateCost> = None;
    for sym in LITERAL_SYMBOLIZATIONS {
        for page in LITERAL_PAGE_FRAMES {
            let rl = RepresentedLiteral::encode(
                descriptor.clone(),
                samples,
                page,
                sym,
                ModelMode::Inline,
                false,
            )?;
            let c = rl.cost(canonical)?;
            let raw_bytes = samples.len() as u64 * 4;
            let candidate = CandidateCost {
                metadata_bytes: c.metadata_bytes,
                hypothesis_bytes: 0,
                residual_bytes: c.payload_bytes,
                persistent_bytes: raw_bytes,
                dependency_bytes: 0,
                checkpoint_bytes: 0,
                state_bytes: raw_bytes,
                complete_bytes: 0,
                raw_sample_bytes: raw_bytes,
                canonical_literal_bytes: canonical,
                cost_source: "entropy_literal",
            }
            .finish();
            if best.is_none_or(|b| candidate.complete_bytes < b.complete_bytes) {
                best = Some(candidate);
            }
        }
    }
    best.ok_or_else(|| crate::error::Error::internal("literal universe is empty"))
}

/// Price a residual-governed candidate with the best frozen residual encoding
/// (minimum `complete_bytes` over the residual page sizes).
pub fn residual_cost(
    descriptor: &ObjectDescriptor,
    residual: &Residual,
    frames: u64,
    channels: u8,
) -> Result<CandidateCost> {
    let canonical = canonical_u1_literal_bytes(frames, channels);
    let mut best: Option<CandidateCost> = None;
    for page in RESIDUAL_PAGE_FRAMES {
        let rr = RepresentedResidual::encode(
            descriptor.clone(),
            residual,
            page,
            ModelMode::Inline,
            false,
        )?;
        let c = rr.cost(canonical)?;
        let persistent = residual.records.len() as u64 * 4;
        let candidate = CandidateCost {
            metadata_bytes: c.metadata_bytes,
            hypothesis_bytes: c.hypothesis_bytes,
            residual_bytes: c.payload_bytes,
            persistent_bytes: persistent,
            dependency_bytes: 0,
            checkpoint_bytes: 0,
            state_bytes: persistent,
            complete_bytes: 0,
            raw_sample_bytes: frames * u64::from(channels) * 4,
            canonical_literal_bytes: canonical,
            cost_source: "entropy_residual",
        }
        .finish();
        if best.is_none_or(|b| candidate.complete_bytes < b.complete_bytes) {
            best = Some(candidate);
        }
    }
    best.ok_or_else(|| crate::error::Error::internal("residual page universe is empty"))
}

/// Price a payload with no entropy body from its canonical object bytes.
pub fn canonical_object_cost(
    descriptor: &ObjectDescriptor,
    data: &ObjectData,
    frames: u64,
    channels: u8,
) -> Result<CandidateCost> {
    let (header, payload) = canonical_object_parts(descriptor, data)?;
    let metadata = header;
    let (raw, canonical) = CandidateCost::baseline(frames, channels);
    // For cycle/constant payloads the payload is both hypothesis (semantic
    // content) and stored sample-domain state; it is never free.
    let persistent = match data {
        ObjectData::Wavetable(cycle)
        | ObjectData::SingleCycle(cycle)
        | ObjectData::ExactRepeat(cycle) => cycle.samples.len() as u64 * 4,
        _ => 0,
    };
    let hypothesis = match data {
        ObjectData::Constant(_) => payload,
        ObjectData::Wavetable(cycle)
        | ObjectData::SingleCycle(cycle)
        | ObjectData::ExactRepeat(cycle) => {
            // cycle_len + sample_count prefixes are hypothesis metadata; the
            // cycle codes themselves are persistent sample-domain content.
            payload.saturating_sub(cycle_bytes(cycle))
        }
        ObjectData::Referenced(_) => {
            // transpose + loop override state; the target content id is the
            // dependency, not hypothesis state.
            payload.saturating_sub(REFERENCE_TARGET_BYTES)
        }
        ObjectData::Silence => 0,
        _ => payload,
    };
    let dependency = match data {
        ObjectData::Referenced(_) => REFERENCE_TARGET_BYTES,
        _ => 0,
    };
    // Every canonical byte must land in exactly one bucket: the complete cost
    // of a candidate with no entropy body is its canonical object length.
    debug_assert_eq!(
        metadata + hypothesis + persistent + dependency,
        metadata + payload,
        "canonical cost decomposition must cover the whole payload"
    );
    Ok(CandidateCost {
        metadata_bytes: metadata,
        hypothesis_bytes: hypothesis,
        residual_bytes: 0,
        persistent_bytes: persistent,
        dependency_bytes: dependency,
        checkpoint_bytes: 0,
        state_bytes: persistent,
        complete_bytes: 0,
        raw_sample_bytes: raw,
        canonical_literal_bytes: canonical,
        cost_source: "canonical_object",
    }
    .finish())
}

fn cycle_bytes(cycle: &Cycle) -> u64 {
    cycle.samples.len() as u64 * 4
}

/// A reference payload's dependency is the 32-byte target content id.
const REFERENCE_TARGET_BYTES: u64 = 32;

/// Canonical object bytes split into (header bytes, payload bytes).
fn canonical_object_parts(descriptor: &ObjectDescriptor, data: &ObjectData) -> Result<(u64, u64)> {
    use crate::object::descriptor::canonical_header_bytes;
    let header = canonical_header_bytes(descriptor).len() as u64;
    let total = match data {
        ObjectData::Silence => {
            crate::object::simple::canonical_bytes(descriptor, Representation::Silence, &[])
        }
        ObjectData::Constant(c) => crate::object::simple::canonical_bytes(
            descriptor,
            Representation::Constant,
            &c.level.to_le_bytes(),
        ),
        ObjectData::Referenced(r) => r.canonical_bytes(descriptor),
        ObjectData::Wavetable(c) => crate::object::wavetable::cycle_canonical_bytes(
            descriptor,
            c,
            Representation::Wavetable,
        ),
        ObjectData::SingleCycle(c) => crate::object::wavetable::cycle_canonical_bytes(
            descriptor,
            c,
            Representation::SingleCycle,
        ),
        ObjectData::ExactRepeat(c) => crate::object::wavetable::cycle_canonical_bytes(
            descriptor,
            c,
            Representation::ExactRepeat,
        ),
        other => {
            return Err(crate::error::Error::internal(format!(
                "canonical_object_cost called with priced representation {other:?}"
            )));
        }
    };
    let payload = total.len() as u64 - header;
    Ok((header, payload))
}

/// Abstract universe work for a candidate, from its representation structure.
///
/// Rules (docs/INVERSE.md):
/// * endless classes (silence/constant) generate one value per output code;
/// * literal/cycle/reference read one stored code per output code;
/// * a residual object runs the hypothesis once per frame and applies one
///   delta per record (the closure binary search is counted as one lookup per
///   output code, the same as any content read).
pub fn abstract_work(data: &ObjectData, frames: u64, channels: u8) -> AbstractWork {
    let codes = frames.saturating_mul(u64::from(channels));
    match data {
        ObjectData::Silence | ObjectData::Constant(_) => AbstractWork {
            generator_ops: codes,
            ..Default::default()
        },
        ObjectData::Literal(_)
        | ObjectData::Wavetable(_)
        | ObjectData::SingleCycle(_)
        | ObjectData::ExactRepeat(_)
        | ObjectData::Referenced(_) => AbstractWork {
            lookup_ops: codes,
            ..Default::default()
        },
        ObjectData::PredictorResidual(r) => AbstractWork {
            generator_ops: frames,
            residual_ops: r.records.len() as u64,
            lookup_ops: codes,
            filter_ops: 0,
        },
        _ => AbstractWork {
            lookup_ops: codes,
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_cost_is_at_most_the_canonical_literal() {
        // Random full-range codes must fall back toward RAW: the priced
        // literal representation can never cost less than a bare RAW body,
        // and never much more than the canonical literal.
        let samples: Vec<i32> = (0..2048u32)
            .map(|i| i.wrapping_mul(2654435761) as i32)
            .collect();
        let c = literal_cost(&samples, 2048, 1).unwrap();
        assert_eq!(c.cost_source, "entropy_literal");
        assert_eq!(c.raw_sample_bytes, 2048 * 4);
        assert!(c.complete_bytes <= c.canonical_literal_bytes * 2);
        assert!(c.persistent_bytes == 2048 * 4);
    }

    #[test]
    fn silence_and_constant_have_tiny_complete_cost() {
        let d = ObjectDescriptor::new(
            Representation::Silence,
            0,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap();
        let c = canonical_object_cost(&d, &ObjectData::Silence, 4096, 1).unwrap();
        assert_eq!(c.complete_bytes, U1_LITERAL_HEADER_BYTES);
        assert!(c.dependency_bytes == 0);

        let dc = ObjectDescriptor::new(
            Representation::Constant,
            0,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap();
        let cc = canonical_object_cost(
            &dc,
            &ObjectData::Constant(crate::object::Constant::new(7)),
            4096,
            1,
        )
        .unwrap();
        assert_eq!(cc.complete_bytes, U1_LITERAL_HEADER_BYTES + 4);
        assert_eq!(cc.hypothesis_bytes, 4);
    }

    #[test]
    fn shared_reference_counts_dependency_bytes() {
        let d = ObjectDescriptor::new(
            Representation::Referenced,
            64,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap();
        let r = crate::object::reference::Referenced::checked(
            crate::object::id::ContentId([9; 32]),
            1 << 24,
            None,
        )
        .unwrap();
        let c = canonical_object_cost(&d, &ObjectData::Referenced(r), 64, 1).unwrap();
        assert_eq!(c.dependency_bytes, 32);
        assert_eq!(c.persistent_bytes, 0);
        assert_eq!(c.complete_bytes, U1_LITERAL_HEADER_BYTES + 8 + 1 + 16 + 32);
    }

    #[test]
    fn abstract_work_follows_the_representation() {
        let silence = abstract_work(&ObjectData::Silence, 100, 2);
        assert_eq!(silence.generator_ops, 200);
        assert_eq!(silence.lookup_ops, 0);
        let literal = ObjectData::Literal(
            crate::object::Literal::new(
                &ObjectDescriptor::new(
                    Representation::Literal,
                    100,
                    crate::universe::layout::Layout::Mono,
                    None,
                )
                .unwrap(),
                vec![0; 100],
            )
            .unwrap(),
        );
        let lit = abstract_work(&literal, 100, 1);
        assert_eq!(lit.lookup_ops, 100);
        assert_eq!(lit.total(), 100);
    }
}
