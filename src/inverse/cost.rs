//! Complete dependency accounting for inverse candidates (contract §33/§35).
//!
//! Every candidate reports a **complete** cost — never a bare payload size.
//! Where an entropy representation exists the cost **is** the frozen H.2
//! complete-cost value ([`crate::entropy::represent`]): the eight storage
//! components are carried through unchanged and `complete_bytes` is the H.2
//! `complete_bytes`, so the inverse compiler can never disagree with the
//! entropy encoder about what a representation costs.
//!
//! Four quantities are deliberately kept apart, because they are different
//! measurements (H.2 memory-path doctrine: a persistent representation is not
//! a transient materialization and is not verification instrumentation):
//!
//! ```text
//! STORAGE COST       physical representation bytes
//!                      complete_bytes = metadata + hypothesis + model
//!                                     + payload + index + checkpoint
//!                                     + dependency + integrity
//! REPRESENTATION     does the CHOSEN representation persist baked samples?
//! PERSISTENCE          persistent_sample_domain_bytes
//! TRANSIENT          what must be decoded/materialized to observe it?
//! MATERIALIZATION      decoded_sample_state_bytes
//!                      decoded_residual_state_bytes
//!                      decoded_window_state_bytes
//! BASELINE           the original/raw/canonical sample bytes
//!                      raw_sample_bytes, canonical_literal_bytes
//! ```
//!
//! Only the eight storage components are summed. An **entropy-coded** literal
//! or residual persists *no* baked sample-domain content — its persistence is
//! the entropy state — so `persistent_sample_domain_bytes` is 0 and the
//! decoded content is reported as transient materialization. A canonical cycle
//! *does* persist its table, so that is persistent sample-domain bytes (and
//! also storage `payload`; a resident table is never free).
//!
//! For representations with no entropy body the decomposition is over the
//! canonical object bytes and sums to exactly that length (asserted), with
//! component names that stay semantically truthful: a constant's level and a
//! cycle's framing are `hypothesis_bytes` (deterministic model state), a
//! stored table is `payload_bytes`, and a reference's target content id is
//! `dependency_bytes`.
//!
//! Abstract universe work (`generator_ops`, `residual_ops`, `lookup_ops`,
//! `filter_ops`) is a *static count derived from the representation
//! structure*, not a measurement.

use crate::entropy::represent::{ModelMode, RepresentedLiteral, RepresentedResidual};
use crate::entropy::symbol::Symbolization;
use crate::error::{Error, Result};
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

/// A reference payload's dependency is the 32-byte target content id.
const REFERENCE_TARGET_BYTES: u64 = 32;
/// A cycle payload's framing: `cycle_len(u64) || sample_count(u64)`.
const CYCLE_FRAMING_BYTES: u64 = 16;

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
    // --- STORAGE COST: these eight components sum to `complete_bytes` ---
    /// Container/descriptor/version metadata bytes.
    pub metadata_bytes: u64,
    /// Deterministic hypothesis state (residual model, constant level, cycle
    /// framing, reference transpose/loop parameters).
    pub hypothesis_bytes: u64,
    /// Entropy model bytes (inline models in full; shared models once).
    pub model_bytes: u64,
    /// Entropy payload bytes (rANS/RAW bodies), or the stored table bytes of a
    /// canonical cycle object.
    pub payload_bytes: u64,
    /// Page-index bytes.
    pub index_bytes: u64,
    /// Checkpoint bytes (0 in Phase K; the field exists so Phase N candidates
    /// need no cost-API surgery).
    pub checkpoint_bytes: u64,
    /// Dependency bytes (referenced content ids, shared tables).
    pub dependency_bytes: u64,
    /// Integrity digest bytes.
    pub integrity_bytes: u64,
    /// Sum of the eight storage components. For an entropy-carrying
    /// representation this is exactly the H.2 `CompleteCost::complete_bytes`.
    pub complete_bytes: u64,

    // --- BASELINE: comparison only, never part of the sum ---
    /// Raw canonical sample bytes of the observed window.
    pub raw_sample_bytes: u64,
    /// Canonical U1 literal bytes of the observed window.
    pub canonical_literal_bytes: u64,

    // --- REPRESENTATION PERSISTENCE: never part of the sum ---
    /// Baked sample-domain bytes the **chosen representation** persists.
    /// Zero for an entropy-coded representation (its persistence is the
    /// entropy state); the stored table bytes for a canonical cycle.
    pub persistent_sample_domain_bytes: u64,

    // --- TRANSIENT MATERIALIZATION: never part of the sum ---
    /// Sample-domain content decoded to observe an entropy-coded literal.
    pub decoded_sample_state_bytes: u64,
    /// Residual record values decoded to observe an entropy-coded residual.
    pub decoded_residual_state_bytes: u64,
    /// Reconstruction window held while observing (frames × channels × 4).
    pub decoded_window_state_bytes: u64,

    /// Which oracle produced the storage cost: `entropy_literal`,
    /// `entropy_residual`, or `canonical_object`.
    pub cost_source: &'static str,
}

impl CandidateCost {
    /// Recompute `complete_bytes` from the eight storage components.
    fn finish(mut self) -> CandidateCost {
        self.complete_bytes = self
            .metadata_bytes
            .saturating_add(self.hypothesis_bytes)
            .saturating_add(self.model_bytes)
            .saturating_add(self.payload_bytes)
            .saturating_add(self.index_bytes)
            .saturating_add(self.checkpoint_bytes)
            .saturating_add(self.dependency_bytes)
            .saturating_add(self.integrity_bytes);
        self
    }

    /// The eight storage components must sum to `complete_bytes`.
    pub fn decomposition_is_consistent(&self) -> bool {
        self.metadata_bytes
            .saturating_add(self.hypothesis_bytes)
            .saturating_add(self.model_bytes)
            .saturating_add(self.payload_bytes)
            .saturating_add(self.index_bytes)
            .saturating_add(self.checkpoint_bytes)
            .saturating_add(self.dependency_bytes)
            .saturating_add(self.integrity_bytes)
            == self.complete_bytes
    }

    /// Total transient sample-domain materialization (decoded content only).
    pub fn decoded_state_bytes(&self) -> u64 {
        self.decoded_sample_state_bytes
            .saturating_add(self.decoded_residual_state_bytes)
            .saturating_add(self.decoded_window_state_bytes)
    }
}

/// Carry the frozen H.2 cost through unchanged: the eight storage components
/// and `complete_bytes` are the H.2 values, verbatim. An entropy-coded
/// representation persists no baked sample-domain content (its persistence is
/// the entropy state), so `persistent_sample_domain_bytes` is 0; the decoded
/// content is reported as transient materialization.
fn from_h2(
    h2: &crate::entropy::accounting::CompleteCost,
    source: &'static str,
    decoded_sample_state_bytes: u64,
    decoded_residual_state_bytes: u64,
    decoded_window_state_bytes: u64,
) -> CandidateCost {
    CandidateCost {
        metadata_bytes: h2.metadata_bytes,
        hypothesis_bytes: h2.hypothesis_bytes,
        model_bytes: h2.model_bytes,
        payload_bytes: h2.payload_bytes,
        index_bytes: h2.index_bytes,
        checkpoint_bytes: h2.checkpoint_bytes,
        dependency_bytes: h2.dependency_bytes,
        integrity_bytes: h2.integrity_bytes,
        complete_bytes: h2.complete_bytes,
        raw_sample_bytes: h2.raw_sample_bytes,
        canonical_literal_bytes: h2.canonical_literal_bytes,
        persistent_sample_domain_bytes: 0,
        decoded_sample_state_bytes,
        decoded_residual_state_bytes,
        decoded_window_state_bytes,
        cost_source: source,
    }
}

/// Price the `Literal` candidate with the best frozen literal entropy
/// representation (minimum `complete_bytes` over the literal universe). The
/// reported storage cost is the H.2 cost; the decoded samples are transient
/// materialization, not persistence, and are never summed.
pub fn literal_cost(samples: &[i32], frames: u64, channels: u8) -> Result<CandidateCost> {
    let descriptor = ObjectDescriptor::new(
        Representation::Literal,
        frames,
        crate::inverse::observe::layout_of(channels)?,
        None,
    )
    .ok_or_else(|| Error::malformed("literal descriptor out of domain"))?;
    let canonical = canonical_u1_literal_bytes(frames, channels);
    let sample_bytes = samples.len() as u64 * 4;
    let window_bytes = frames * u64::from(channels) * 4;
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
            let h2 = rl.cost(canonical)?;
            let candidate = from_h2(&h2, "entropy_literal", sample_bytes, 0, window_bytes);
            debug_assert!(candidate.decomposition_is_consistent());
            if best.is_none_or(|b| candidate.complete_bytes < b.complete_bytes) {
                best = Some(candidate);
            }
        }
    }
    best.ok_or_else(|| Error::internal("literal universe is empty"))
}

/// Price a residual-governed candidate with the best frozen residual encoding
/// (minimum `complete_bytes` over the residual page sizes). The storage cost is
/// the H.2 cost; the residual deltas are transient materialization, not
/// persistence, and are never charged a second time.
pub fn residual_cost(
    descriptor: &ObjectDescriptor,
    residual: &Residual,
    frames: u64,
    channels: u8,
) -> Result<CandidateCost> {
    let canonical = canonical_u1_literal_bytes(frames, channels);
    let delta_bytes = residual.records.len() as u64 * 4;
    let window_bytes = frames * u64::from(channels) * 4;
    let mut best: Option<CandidateCost> = None;
    for page in RESIDUAL_PAGE_FRAMES {
        let rr = RepresentedResidual::encode(
            descriptor.clone(),
            residual,
            page,
            ModelMode::Inline,
            false,
        )?;
        let h2 = rr.cost(canonical)?;
        let candidate = from_h2(&h2, "entropy_residual", 0, delta_bytes, window_bytes);
        debug_assert!(candidate.decomposition_is_consistent());
        if best.is_none_or(|b| candidate.complete_bytes < b.complete_bytes) {
            best = Some(candidate);
        }
    }
    best.ok_or_else(|| Error::internal("residual page universe is empty"))
}

/// Price a payload with no entropy body from its canonical object bytes. The
/// decomposition sums to exactly the canonical representation length, with
/// truthful component names.
pub fn canonical_object_cost(
    descriptor: &ObjectDescriptor,
    data: &ObjectData,
    frames: u64,
    channels: u8,
) -> Result<CandidateCost> {
    let (header, payload_total) = canonical_object_parts(descriptor, data)?;
    let raw = frames * u64::from(channels) * 4;
    let canonical = canonical_u1_literal_bytes(frames, channels);

    // Storage decomposition and representation persistence per class.
    let (hypothesis, payload, dependency, persistent) = match data {
        ObjectData::Silence => (0, 0, 0, 0),
        // The level IS the deterministic model state.
        ObjectData::Constant(_) => (4, 0, 0, 0),
        // Framing is hypothesis state; the stored table is payload AND
        // persistent sample-domain content (a resident table is never free).
        ObjectData::Wavetable(c) | ObjectData::SingleCycle(c) | ObjectData::ExactRepeat(c) => {
            (CYCLE_FRAMING_BYTES, cycle_bytes(c), 0, cycle_bytes(c))
        }
        // Transpose + loop override are reference parameters; the target
        // content id is the dependency. The target owns its own content.
        ObjectData::Referenced(_) => (
            payload_total.saturating_sub(REFERENCE_TARGET_BYTES),
            0,
            REFERENCE_TARGET_BYTES,
            0,
        ),
        other => {
            return Err(Error::internal(format!(
                "canonical_object_cost called with priced representation {other:?}"
            )));
        }
    };

    let cost = CandidateCost {
        metadata_bytes: header,
        hypothesis_bytes: hypothesis,
        model_bytes: 0,
        payload_bytes: payload,
        index_bytes: 0,
        checkpoint_bytes: 0,
        dependency_bytes: dependency,
        integrity_bytes: 0,
        complete_bytes: 0,
        raw_sample_bytes: raw,
        canonical_literal_bytes: canonical,
        persistent_sample_domain_bytes: persistent,
        decoded_sample_state_bytes: 0,
        decoded_residual_state_bytes: 0,
        decoded_window_state_bytes: 0,
        cost_source: "canonical_object",
    }
    .finish();
    debug_assert_eq!(
        cost.complete_bytes,
        header + payload_total,
        "canonical decomposition must cover the whole canonical object"
    );
    Ok(cost)
}

fn cycle_bytes(cycle: &Cycle) -> u64 {
    cycle.samples.len() as u64 * 4
}

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
            return Err(Error::internal(format!(
                "canonical_object_cost called with priced representation {other:?}"
            )));
        }
    };
    let payload = total.len() as u64 - header;
    Ok((header, payload))
}

/// Sample-domain bytes a candidate representation instantiates in memory
/// while the search proposes/verifies it (a residual record vector, a literal
/// sample vector, a stored cycle table). This is **search-time allocation**,
/// distinct from representation persistence and from decoded state.
pub fn semantic_state_bytes(data: &ObjectData) -> u64 {
    match data {
        ObjectData::Literal(l) => l.samples.len() as u64 * 4,
        ObjectData::Wavetable(c) | ObjectData::SingleCycle(c) | ObjectData::ExactRepeat(c) => {
            cycle_bytes(c)
        }
        ObjectData::PredictorResidual(r) => r.records.len() as u64 * 4,
        ObjectData::Referenced(_) | ObjectData::Silence | ObjectData::Constant(_) => 0,
        _ => 0,
    }
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

    /// The literal candidate's storage cost must be exactly the H.2 complete
    /// cost — not the H.2 cost plus the raw samples it replaced.
    #[test]
    fn literal_cost_equals_the_h2_complete_cost() {
        // Full-range noise: the H.2 literal representation is RAW-fallback
        // pages, so the H.2 complete cost is metadata + payload + index.
        let samples: Vec<i32> = (0..4096u32)
            .map(|i| (i.wrapping_mul(2654435761)) as i32)
            .collect();
        let c = literal_cost(&samples, 4096, 1).unwrap();
        assert!(c.decomposition_is_consistent());
        assert_eq!(c.cost_source, "entropy_literal");
        assert_eq!(c.raw_sample_bytes, 4096 * 4);
        assert_eq!(c.canonical_literal_bytes, 46 + 8 + 4096 * 4);

        // The storage cost must not include the raw sample baseline.
        assert!(
            c.complete_bytes < c.raw_sample_bytes + c.metadata_bytes + c.index_bytes + 1,
            "storage cost must not include the raw sample baseline ({} vs raw {})",
            c.complete_bytes,
            c.raw_sample_bytes
        );
        // An entropy-coded literal persists NO baked sample-domain content:
        // the decoded samples are transient materialization.
        assert_eq!(c.persistent_sample_domain_bytes, 0);
        assert_eq!(c.decoded_sample_state_bytes, 4096 * 4);
        assert_eq!(c.decoded_window_state_bytes, 4096 * 4);
        assert_eq!(c.decoded_residual_state_bytes, 0);
    }

    #[test]
    fn entropy_literal_matches_the_h2_decomposition_verbatim() {
        // Structured content so the rANS path (not RAW fallback) is used.
        let samples: Vec<i32> = (0..4096i32).map(|i| (i % 64) << 16).collect();
        let c = literal_cost(&samples, 4096, 1).unwrap();
        assert!(c.decomposition_is_consistent());
        let components = c.metadata_bytes
            + c.hypothesis_bytes
            + c.model_bytes
            + c.payload_bytes
            + c.index_bytes
            + c.checkpoint_bytes
            + c.dependency_bytes
            + c.integrity_bytes;
        assert_eq!(components, c.complete_bytes);
        assert!(c.complete_bytes < c.canonical_literal_bytes);
        assert_eq!(c.persistent_sample_domain_bytes, 0);
    }

    #[test]
    fn h2_model_and_index_components_are_carried_through() {
        // A structured literal is rANS-coded, so the H.2 cost must carry
        // non-zero model and index components (the old adaptation dropped
        // them).
        let samples: Vec<i32> = (0..4096i32).map(|i| (i % 64) << 16).collect();
        let c = literal_cost(&samples, 4096, 1).unwrap();
        assert!(c.index_bytes > 0, "page index must be accounted");
        assert!(
            c.model_bytes > 0,
            "inline rANS models must be accounted, not dropped"
        );
        assert!(c.payload_bytes > 0);
    }

    #[test]
    fn silence_and_constant_component_names_are_truthful() {
        let d = ObjectDescriptor::new(
            Representation::Silence,
            0,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap();
        let c = canonical_object_cost(&d, &ObjectData::Silence, 4096, 1).unwrap();
        assert_eq!(c.complete_bytes, U1_LITERAL_HEADER_BYTES);
        assert!(c.decomposition_is_consistent());
        assert_eq!(c.hypothesis_bytes, 0);
        assert_eq!(c.payload_bytes, 0);
        assert_eq!(c.persistent_sample_domain_bytes, 0);

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
        // The level is deterministic model state, not opaque payload.
        assert_eq!(cc.hypothesis_bytes, 4);
        assert_eq!(cc.payload_bytes, 0);
        assert!(cc.decomposition_is_consistent());
    }

    #[test]
    fn cycle_table_is_payload_and_persistent_state_with_truthful_framing() {
        let d = ObjectDescriptor::new(
            Representation::ExactRepeat,
            64,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap();
        let cycle = Cycle::new(&d, (0..64).collect()).unwrap();
        let c = canonical_object_cost(&d, &ObjectData::ExactRepeat(cycle), 4096, 1).unwrap();
        assert!(c.decomposition_is_consistent());
        // header 46 + framing 16 + table 256
        assert_eq!(c.complete_bytes, 46 + 16 + 256);
        assert_eq!(c.hypothesis_bytes, CYCLE_FRAMING_BYTES);
        assert_eq!(c.payload_bytes, 256);
        // A canonical cycle DOES persist its baked table.
        assert_eq!(c.persistent_sample_domain_bytes, 256);
        // Nothing is decoded transiently for a canonical object.
        assert_eq!(c.decoded_state_bytes(), 0);
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
        assert!(c.decomposition_is_consistent());
        assert_eq!(c.dependency_bytes, REFERENCE_TARGET_BYTES);
        assert_eq!(c.complete_bytes, U1_LITERAL_HEADER_BYTES + 8 + 1 + 16 + 32);
        // Transpose + loop override are reference parameters (hypothesis).
        assert_eq!(c.hypothesis_bytes, 8 + 1 + 16);
        assert_eq!(c.payload_bytes, 0);
        assert_eq!(c.persistent_sample_domain_bytes, 0);
    }

    #[test]
    fn residual_cost_does_not_double_charge_the_deltas() {
        use crate::object::residual::{Residual, ResidualModel};
        let intrinsic: Vec<i32> = (0..256)
            .map(|i| if i % 32 == 0 { 1000 } else { 0 })
            .collect();
        let model = ResidualModel::Zero;
        let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
        let descriptor = ObjectDescriptor::new(
            Representation::PredictorResidual,
            256,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap();
        let residual = Residual::new(&descriptor, model, records).unwrap();
        let c = residual_cost(&descriptor, &residual, 256, 1).unwrap();
        assert!(c.decomposition_is_consistent());
        assert_eq!(c.cost_source, "entropy_residual");
        // The deltas are transient decoded state, not persistence.
        assert_eq!(c.persistent_sample_domain_bytes, 0);
        assert_eq!(c.decoded_residual_state_bytes, 8 * 4);
        assert_eq!(c.decoded_window_state_bytes, 256 * 4);
        assert_eq!(c.decoded_sample_state_bytes, 0);
        // Storage is the H.2 residual cost (the sweep takes the minimum over
        // page sizes, so it cannot exceed any single encoding) and carries the
        // H.2 payload unchanged.
        let rr = RepresentedResidual::encode(
            descriptor.clone(),
            &residual,
            1024,
            ModelMode::Inline,
            false,
        )
        .unwrap();
        let h2 = rr.cost(canonical_u1_literal_bytes(256, 1)).unwrap();
        assert!(c.complete_bytes <= h2.complete_bytes);
        assert!(c.payload_bytes <= h2.payload_bytes);
    }

    #[test]
    fn abstract_work_follows_the_representation() {
        let silence = abstract_work(&ObjectData::Silence, 100, 2);
        assert_eq!(silence.generator_ops, 200);
        assert_eq!(silence.lookup_ops, 0);
        assert_eq!(silence.total(), 200);
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
