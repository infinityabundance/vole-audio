//! Canonical serialization of an accepted candidate's **selected**
//! representation.
//!
//! The inverse compiler prices candidates with the frozen H.2/canonical cost
//! oracles; this module turns the *same* representation into its physical
//! canonical bytes, so the full-object container stores real serialized
//! payloads rather than an accounting estimate. Pricing and serialization share
//! one iteration over each frozen encoding universe (`cost::best_literal`,
//! `cost::best_residual`), so they can never disagree about which encoding won.
//!
//! * an entropy-coded `Literal` stores its `vole.entropy.p1` literal container;
//! * an entropy-coded `PredictorResidual` stores its residual container;
//! * a canonical object (`Silence`, `Constant`, cycle family) stores its
//!   canonical object bytes (descriptor header + payload).

use crate::entropy::represent::{literal_container_bytes, residual_container_bytes};
use crate::error::Result;
use crate::object::canonical_content_id;
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::id::ContentId;
use crate::object::{ObjectData, canonical_object_bytes};

use super::CandidateKind;
use super::cost;

/// One selected segment representation, in its physical canonical byte form.
#[derive(Debug, Clone)]
pub struct SelectedPayload {
    /// The candidate family that won the segment.
    pub kind: CandidateKind,
    /// The U1 representation of the stored bytes.
    pub representation: Representation,
    /// The U1 content identity of the accepted object.
    pub content_id: ContentId,
    /// The Phase-K selection objective (the candidate's `complete_bytes`).
    pub objective_bytes: u64,
    /// The actual canonical serialized payload.
    pub bytes: Vec<u8>,
}

impl SelectedPayload {
    /// The physical payload length (`bytes.len()`), never an estimate.
    pub fn stored_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }
}

/// Serialize the selected candidate's representation to its canonical bytes.
///
/// `frames`/`channels` are the segment's intrinsic window, which the descriptor
/// must already cover.
pub fn serialize_candidate(
    kind: CandidateKind,
    descriptor: &ObjectDescriptor,
    data: &ObjectData,
    frames: u64,
    channels: u8,
) -> Result<SelectedPayload> {
    let (representation, bytes, objective_bytes) = match data {
        ObjectData::Literal(l) => {
            let (rl, h2) = cost::best_literal(&l.samples, frames, channels)?;
            (
                Representation::Literal,
                literal_container_bytes(&rl)?,
                h2.complete_bytes,
            )
        }
        ObjectData::PredictorResidual(r) => {
            let (rr, h2) = cost::best_residual(descriptor, r, frames, channels)?;
            (
                Representation::PredictorResidual,
                residual_container_bytes(&rr)?,
                h2.complete_bytes,
            )
        }
        other => {
            let bytes = canonical_object_bytes(descriptor, other);
            let cost = cost::canonical_object_cost(descriptor, other, frames, channels)?;
            (descriptor.representation, bytes, cost.complete_bytes)
        }
    };
    Ok(SelectedPayload {
        kind,
        representation,
        content_id: canonical_content_id(descriptor, data),
        objective_bytes,
        bytes,
    })
}
