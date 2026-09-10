//! Complete dependency accounting for inverse candidates (contract §33/§35).
//!
//! Every candidate reports a **complete** cost — never a bare payload size.
//! Where an entropy representation exists the cost **is** the frozen H.2
//! complete-cost value ([`crate::entropy::represent`]): the eight storage
//! components are carried through unchanged and `complete_bytes` is the H.2
//! `complete_bytes`, so the inverse compiler can never disagree with the
//! entropy encoder about what a representation costs.
//!
//! Three quantities are deliberately kept apart, because they are different
//! measurements:
//!
//! ```text
//! STORAGE COST        physical representation bytes
//!                       complete_bytes = metadata + hypothesis + model
//!                                      + payload + index + checkpoint
//!                                      + dependency + integrity
//! STATE / EXPOSURE    sample-domain content that may exist while evaluating
//!                       persistent_sample_domain_bytes, state_bytes
//!                       (NEVER added to complete_bytes)
//! BASELINE            the original/raw/canonical sample bytes
//!                       raw_sample_bytes, canonical_literal_bytes
//!                       (NEVER part of the sum)
//! ```
//!
//! For representations with no entropy body (silence, constant, cycle,
//! reference) the decomposition is over the canonical object bytes and sums
//! to exactly that length (asserted). A resident table (a stored cycle) is
//! storage `payload`, not a free dependency; a shared reference's target
//! content id is `dependency_bytes`, never zero.
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
    /// Hypothesis bytes (procedural model state).
    pub hypothesis_bytes: u64,
    /// Entropy model bytes (inline models in full; shared models once).
    pub model_bytes: u64,
    /// Entropy payload bytes (rANS/RAW bodies), or canonical object payload
    /// bytes for representations with no entropy body.
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

    // --- STATE / EXPOSURE: sample-domain residency, never part of the sum ---
    /// Sample-domain content the representation owns persistently (literal
    /// samples, cycle samples, residual deltas).
    pub persistent_sample_domain_bytes: u64,
    /// Sample-domain state resident while materializing one observation
    /// window (the decoded window for entropy-carrying representations, the
    /// owned content for baked ones).
    pub state_bytes: u64,

    /// Which oracle produced the storage cost: `entropy_literal`,
    /// `entropy_residual`, or `canonical_object`.
    pub cost_source: &'static str,
}

impl CandidateCost {
    /// Recompute `complete_bytes` from the eight storage components and assert
    /// the decomposition invariant.
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

    /// Componentwise storage check used by hostile tests: the eight components
    /// must sum to `complete_bytes`.
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

    fn window_baseline(frames: u64, channels: u8) -> (u64, u64) {
        (
            frames * u64::from(channels) * 4,
            canonical_u1_literal_bytes(frames, channels),
        )
    }
}

/// Carry the frozen H.2 cost through unchanged: the eight storage components
/// and `complete_bytes` are the H.2 values, verbatim. Sample-domain state is
/// reported separately and is **not** added to the storage cost.
fn from_h2(
    h2: &crate::entropy::accounting::CompleteCost,
    source: &'static str,
    persistent_sample_domain_bytes: u64,
    state_bytes: u64,
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
        persistent_sample_domain_bytes,
        state_bytes,
        cost_source: source,
    }
}

/// Price the `Literal` candidate with the best frozen literal entropy
/// representation (minimum `complete_bytes` over the literal universe). The
/// reported storage cost is the H.2 cost; the decoded samples are reported as
/// sample-domain state, never as storage.
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
            let candidate = from_h2(&h2, "entropy_literal", sample_bytes, sample_bytes);
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
/// the H.2 cost; the residual deltas are sample-domain state and are **not**
/// charged a second time.
pub fn residual_cost(
    descriptor: &ObjectDescriptor,
    residual: &Residual,
    frames: u64,
    channels: u8,
) -> Result<CandidateCost> {
    let canonical = canonical_u1_literal_bytes(frames, channels);
    let delta_bytes = residual.records.len() as u64 * 4;
    let decoded_window = frames * u64::from(channels) * 4;
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
        let candidate = from_h2(&h2, "entropy_residual", delta_bytes, decoded_window);
        debug_assert!(candidate.decomposition_is_consistent());
        if best.is_none_or(|b| candidate.complete_bytes < b.complete_bytes) {
            best = Some(candidate);
        }
    }
    best.ok_or_else(|| Error::internal("residual page universe is empty"))
}

/// Price a payload with no entropy body from its canonical object bytes. The
/// decomposition sums to exactly the canonical representation length.
pub fn canonical_object_cost(
    descriptor: &ObjectDescriptor,
    data: &ObjectData,
    frames: u64,
    channels: u8,
) -> Result<CandidateCost> {
    let (header, payload) = canonical_object_parts(descriptor, data)?;
    let (raw, canonical) = CandidateCost::window_baseline(frames, channels);

    // Sample-domain residency (exposure/state; never part of the sum).
    let (persistent, state) = sample_domain_state(data, frames, channels);

    // Storage decomposition. For a cycle the stored table is `payload` (a
    // resident table is not free); for a reference the target content id is
    // `dependency` and the transpose/loop state is `payload`.
    let dependency = match data {
        ObjectData::Referenced(_) => REFERENCE_TARGET_BYTES,
        _ => 0,
    };
    let payload_bytes = payload.saturating_sub(dependency);
    let cost = CandidateCost {
        metadata_bytes: header,
        hypothesis_bytes: 0,
        model_bytes: 0,
        payload_bytes,
        index_bytes: 0,
        checkpoint_bytes: 0,
        dependency_bytes: dependency,
        integrity_bytes: 0,
        complete_bytes: 0,
        raw_sample_bytes: raw,
        canonical_literal_bytes: canonical,
        persistent_sample_domain_bytes: persistent,
        state_bytes: state,
        cost_source: "canonical_object",
    }
    .finish();
    debug_assert_eq!(
        cost.complete_bytes,
        header + payload,
        "canonical decomposition must cover the whole canonical object"
    );
    Ok(cost)
}

/// Sample-domain residency of a candidate payload (exposure/state).
fn sample_domain_state(data: &ObjectData, frames: u64, channels: u8) -> (u64, u64) {
    match data {
        ObjectData::Literal(l) => {
            let bytes = l.samples.len() as u64 * 4;
            (bytes, bytes)
        }
        ObjectData::Wavetable(c) | ObjectData::SingleCycle(c) | ObjectData::ExactRepeat(c) => {
            let bytes = cycle_bytes(c);
            (bytes, bytes)
        }
        ObjectData::PredictorResidual(r) => {
            (r.records.len() as u64 * 4, frames * u64::from(channels) * 4)
        }
        ObjectData::Referenced(_) | ObjectData::Silence | ObjectData::Constant(_) => (0, 0),
        _ => (0, 0),
    }
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

        let descriptor = ObjectDescriptor::new(
            Representation::Literal,
            4096,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap();
        // The reported storage cost equals the H.2 cost for the same encoding.
        let rl = RepresentedLiteral::encode(
            descriptor,
            &samples,
            1024,
            Symbolization::Identity,
            ModelMode::Inline,
            false,
        )
        .unwrap();
        let h2 = rl.cost(canonical_u1_literal_bytes(4096, 1)).unwrap();
        // The unbounded-symbolization sweep chooses the minimum, which can
        // only be <= the identity/1024 cell.
        assert!(c.complete_bytes <= h2.complete_bytes);
        // The raw sample bytes are a baseline and are NOT in the sum: a
        // 4096-frame mono window has 16384 raw bytes, so a cost that folded
        // them in would exceed 16384.
        assert!(
            c.complete_bytes < c.raw_sample_bytes + c.metadata_bytes + c.index_bytes + 1,
            "storage cost must not include the raw sample baseline ({} vs raw {})",
            c.complete_bytes,
            c.raw_sample_bytes
        );
        // Sample-domain state is reported but not summed.
        assert_eq!(c.persistent_sample_domain_bytes, 4096 * 4);
        assert_eq!(c.state_bytes, 4096 * 4);
    }

    #[test]
    fn h2_decomposition_is_carried_through_verbatim() {
        // Structured content so the rANS path (not RAW fallback) is exercised.
        let samples: Vec<i32> = (0..4096i32).map(|i| (i % 64) << 16).collect();
        let c = literal_cost(&samples, 4096, 1).unwrap();
        assert!(c.decomposition_is_consistent());
        // Recompute directly through the H.2 API for the chosen cell is not
        // possible from the outside (the sweep chooses the best); assert the
        // invariant that the sum is exactly the eight components and that no
        // baseline leaked in.
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
    }

    #[test]
    fn silence_and_constant_cost_is_their_canonical_length() {
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
        assert!(cc.decomposition_is_consistent());
    }

    #[test]
    fn cycle_table_is_storage_payload_and_also_reported_as_state() {
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
        // header 46 + cycle_len 8 + count 8 + 64*4 = 318
        assert_eq!(c.complete_bytes, 46 + 16 + 256);
        // The stored table is storage payload (never free) and is also the
        // resident sample-domain state.
        assert_eq!(c.payload_bytes, 16 + 256);
        assert_eq!(c.persistent_sample_domain_bytes, 256);
        assert_eq!(c.state_bytes, 256);
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
        // Deltas are exposure/state, not storage.
        assert_eq!(c.persistent_sample_domain_bytes, 8 * 4);
        assert_eq!(c.state_bytes, 256 * 4);
        // The storage cost is the H.2 residual cost (the sweep takes the
        // minimum over page sizes, so it cannot exceed any single encoding)
        // and it carries the H.2 payload unchanged — the unencoded delta bytes
        // are never added on top.
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
        assert!(c.complete_bytes < c.raw_sample_bytes + c.persistent_sample_domain_bytes);
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
