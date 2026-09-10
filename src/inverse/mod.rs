//! Bounded inverse proceduralization (Phase K) — the inverse compiler.
//!
//! The forward architecture answers *"how do I materialize this SampleObject
//! exactly?"*. The inverse compiler answers the dual question:
//!
//! > Given an observed bounded sample window, which deterministic
//! > `SampleObject` explanation is the cheapest **exact** explanation?
//!
//! It is a **bounded deterministic proposal search** over the frozen `u1`
//! semantic vocabulary — never a new semantic authority:
//!
//! ```text
//! observed window X_O
//!       │
//!       ▼  bounded proposals (deterministic; no randomness, no tuning)
//! candidate hypotheses (literal | silence | constant | exact-repeat |
//!                       residual{zero,constant,periodic} | shared reference)
//!       │
//!       ▼  exact acceptance, two independent reconstructions, both == X_O:
//!          (a) intrinsic closure of the representation
//!          (b) normative scalar-oracle observation
//!       │
//!       ▼  complete dependency accounting (H.2 complete-cost is the oracle)
//!       │
//!       ▼  deterministic Pareto frontier over static (bytes, work, seek work)
//! ```
//!
//! Rules (implementation contract §33):
//!
//! * the search is **bounded** and deterministic — no randomness anywhere;
//! * a candidate is accepted only when it reconstructs `X_O` **exactly**
//!   through the normative scalar evaluator *and* its intrinsic closure is
//!   exact; a merely-close hypothesis is rejected;
//! * `Literal` is **always** a candidate (the universal fallback);
//! * every accepted candidate reports a **complete** cost (persistent,
//!   residual, dependency, checkpoint, state bytes, plus the H.2
//!   `CompleteCost` where an entropy representation exists) — never a bare
//!   payload size;
//! * the frontier is a real Pareto set over static objective vectors; costs
//!   are never collapsed into a magical weighted score;
//! * measured quantities (proposal/materialization/seek time, accounted
//!   memory) are reported per candidate but are **not** frontier objectives.
//!
//! The inverse compiler has **zero decoder authority**: its output is a
//! proposal, and every accepted proposal has been re-verified by the exact
//! evaluator. See `docs/INVERSE.md` for the accounting rules and the
//! explicitly deferred candidate families.

pub mod cost;
pub mod frontier;
pub mod observe;
pub mod propose;

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::id::ContentId;
use crate::object::{ObjectData, ObjectStore};

pub use cost::{AbstractWork, CandidateCost};
pub use frontier::Frontier;
pub use propose::ReferenceLibrary;

/// Bounded seek window for the measured seek-latency cell.
pub const SEEK_FRAMES: u32 = 512;

/// Largest intrinsic window the inverse compiler accepts (one observation
/// window, bounded by the quantum ceiling).
pub const MAX_INVERSE_FRAMES: u64 = crate::limits::MAX_QUANTUM_FRAMES as u64;

/// Search bounds. Every bound is explicit and frozen by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchBudget {
    /// Largest period evaluated by the bounded residual period scan.
    pub max_period_scan: u32,
    /// How many scanned periods may become residual candidates.
    pub max_residual_period_candidates: usize,
    /// Hard ceiling on proposed candidates.
    pub max_candidates: usize,
}

impl Default for SearchBudget {
    fn default() -> Self {
        SearchBudget {
            max_period_scan: 512,
            max_residual_period_candidates: 4,
            max_candidates: 64,
        }
    }
}

/// One observed intrinsic window (canonical interleaved i32 codes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intrinsic {
    pub name: String,
    pub channels: u8,
    pub frames: u64,
    /// `frames * channels` canonical codes.
    pub samples: Vec<i32>,
}

impl Intrinsic {
    /// Build and validate an intrinsic window.
    pub fn new(name: impl Into<String>, channels: u8, samples: Vec<i32>) -> Result<Intrinsic> {
        let ch = usize::from(channels);
        if channels == 0 || u32::from(channels) > crate::limits::MAX_CHANNELS {
            return Err(Error::malformed("intrinsic channel count out of domain"));
        }
        if samples.is_empty() || !samples.len().is_multiple_of(ch) {
            return Err(Error::malformed(
                "intrinsic samples must be a non-empty frame-aligned window",
            ));
        }
        let frames = (samples.len() / ch) as u64;
        if frames > MAX_INVERSE_FRAMES {
            return Err(Error::limit(
                "intrinsic window exceeds the inverse observation ceiling",
            ));
        }
        Ok(Intrinsic {
            name: name.into(),
            channels,
            frames,
            samples,
        })
    }

    /// SHA-256 over the canonical interleaved codes.
    pub fn content_sha256(&self) -> [u8; 32] {
        crate::universe::observation::observation_sha256(&self.samples)
    }
}

/// Candidate family. Tag numbers are stable (receipts may key on them).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum CandidateKind {
    Literal = 1,
    Silence = 2,
    Constant = 3,
    ExactRepeat = 4,
    ResidualZero = 5,
    ResidualConstant = 6,
    ResidualPeriodic = 7,
    SharedReference = 8,
}

impl CandidateKind {
    pub const fn tag(self) -> u8 {
        self as u8
    }

    pub const fn name(self) -> &'static str {
        match self {
            CandidateKind::Literal => "literal",
            CandidateKind::Silence => "silence",
            CandidateKind::Constant => "constant",
            CandidateKind::ExactRepeat => "exact_repeat",
            CandidateKind::ResidualZero => "residual_zero",
            CandidateKind::ResidualConstant => "residual_constant",
            CandidateKind::ResidualPeriodic => "residual_periodic",
            CandidateKind::SharedReference => "shared_reference",
        }
    }
}

/// One proposed hypothesis (pre-acceptance).
#[derive(Debug, Clone)]
pub struct Candidate {
    pub kind: CandidateKind,
    pub label: String,
    /// The proposed object, for object candidates.
    pub object: Option<(ObjectDescriptor, ObjectData)>,
    /// The library target content id, for a shared-reference candidate.
    pub reference: Option<ContentId>,
    /// Proposal wall time attributed to this candidate (ns).
    pub proposal_ns: u64,
}

/// One exactly-accepted candidate with its complete accounting.
#[derive(Debug, Clone)]
pub struct Acceptance {
    pub kind: CandidateKind,
    pub label: String,
    pub cost: CandidateCost,
    pub work: AbstractWork,
    pub content_id: ContentId,
    /// For a shared reference: the content id of the referenced library
    /// object (the dependency target). `None` for object candidates.
    pub reference_target: Option<ContentId>,
    /// Intrinsic closure equals the window (always true for accepted).
    pub intrinsic_exact: bool,
    /// Scalar-oracle observation equals the window (always true for accepted).
    pub evaluator_exact: bool,
    /// A bounded seek window at `seek_start` equals the same slice of the
    /// window (always true for accepted).
    pub seek_exact: bool,
    /// Measured materialization time of the full window (ns).
    pub materialize_ns: u64,
    /// Measured time to materialize one bounded seek window (ns).
    pub seek_latency_ns: u64,
    /// Measured intrinsic-closure reconstruction time (ns).
    pub intrinsic_ns: u64,
    pub seek_start: u64,
    pub seek_frames: u32,
    /// Accounted sample-domain transient for one full observation (bytes).
    /// This is an accounting figure over the buffers the evaluator allocates,
    /// not an allocator-instrumented peak; the docs say so.
    pub accounted_peak_bytes: u64,
    /// Proposal wall time attributed to this candidate (ns).
    pub proposal_ns: u64,
    pub seek_ops: u64,
    pub total_ops: u64,
}

impl Acceptance {
    /// Static objective vector (all minimized): complete bytes, total abstract
    /// work, seek work. Deterministic — no measured quantity appears here.
    pub fn objective(&self) -> [u64; 3] {
        [self.cost.complete_bytes, self.total_ops, self.seek_ops]
    }
}

/// Result of one bounded inverse search over one intrinsic window.
#[derive(Debug, Clone)]
pub struct SearchReport {
    pub intrinsic_name: String,
    pub intrinsic_sha256: [u8; 32],
    pub channels: u8,
    pub frames: u64,
    pub proposed: usize,
    pub evaluated: usize,
    pub rejected: usize,
    pub accepted: Vec<Acceptance>,
    pub frontier: Frontier,
    pub budget: SearchBudget,
    /// Total measured search time (ns).
    pub search_ns: u64,
}

impl SearchReport {
    /// Cheapest accepted candidate by complete bytes (ties -> first accepted).
    pub fn cheapest(&self) -> Option<&Acceptance> {
        self.accepted.iter().min_by_key(|a| a.cost.complete_bytes)
    }

    /// Cheapest candidate that is on the frontier.
    pub fn cheapest_on_frontier(&self) -> Option<&Acceptance> {
        self.frontier.cheapest(&self.accepted)
    }

    /// True when the literal fallback was accepted (it must always be).
    pub fn literal_accepted(&self) -> bool {
        self.accepted
            .iter()
            .any(|a| a.kind == CandidateKind::Literal)
    }
}

/// Run the bounded inverse search for one intrinsic window against a reference
/// library (whose objects may be proposed as exact shared references).
pub fn compile(
    intrinsic: &Intrinsic,
    library: &ReferenceLibrary,
    budget: SearchBudget,
) -> Result<SearchReport> {
    let sw = Stopwatch::start();
    let candidates = propose::propose(intrinsic, library, budget)?;
    let proposed = candidates.len();
    let mut accepted: Vec<Acceptance> = Vec::with_capacity(proposed);
    let mut rejected = 0usize;
    for cand in &candidates {
        match accept(intrinsic, library, cand)? {
            Some(a) => accepted.push(a),
            None => rejected += 1,
        }
    }
    let frontier = Frontier::build(&accepted);
    let search_ns = sw.elapsed_ns().max(0) as u64;
    Ok(SearchReport {
        intrinsic_name: intrinsic.name.clone(),
        intrinsic_sha256: intrinsic.content_sha256(),
        channels: intrinsic.channels,
        frames: intrinsic.frames,
        proposed,
        evaluated: proposed,
        rejected,
        accepted,
        frontier,
        budget,
        search_ns,
    })
}

/// Exact acceptance of one candidate: build the evaluation store, require the
/// intrinsic closure and the scalar-oracle observation to both equal the
/// window, then price and time the candidate.
fn accept(
    intrinsic: &Intrinsic,
    library: &ReferenceLibrary,
    cand: &Candidate,
) -> Result<Option<Acceptance>> {
    let layout = observe::layout_of(intrinsic.channels)?;
    let mut store: ObjectStore = library.store().clone();
    let (descriptor, data) = match cand.kind {
        CandidateKind::SharedReference => {
            let target = cand
                .reference
                .ok_or_else(|| Error::internal("reference candidate without a target"))?;
            if store.id_of_content(&target).is_none() {
                return Ok(None);
            }
            let descriptor =
                ObjectDescriptor::new(Representation::Referenced, intrinsic.frames, layout, None)
                    .ok_or_else(|| Error::malformed("reference descriptor out of domain"))?;
            let payload = crate::object::reference::Referenced::checked(target, 1 << 24, None)
                .ok_or_else(|| Error::malformed("reference payload out of domain"))?;
            (descriptor, ObjectData::Referenced(payload))
        }
        _ => cand
            .object
            .clone()
            .ok_or_else(|| Error::internal("object candidate without a payload"))?,
    };

    let id = store.insert(descriptor.clone(), data.clone())?;

    // (a) Intrinsic closure. A reference's closure is its target's closure.
    let (intrinsic_samples, intrinsic_ns) = match &data {
        ObjectData::Referenced(r) => {
            let target = store
                .id_of_content(&r.target_content)
                .ok_or_else(|| Error::dependency("reference target absent from the store"))?;
            let obj = store.get(target)?;
            observe::timed_intrinsic_reconstruction(&obj.descriptor, &obj.data, intrinsic.frames)?
        }
        _ => observe::timed_intrinsic_reconstruction(&descriptor, &data, intrinsic.frames)?,
    };
    if intrinsic_samples != intrinsic.samples {
        return Ok(None);
    }

    // (b) Normative scalar-oracle observation.
    let sw = Stopwatch::start();
    let observed =
        observe::observe_object(&store, id, intrinsic.channels, intrinsic.frames as usize)?;
    let materialize_ns = sw.elapsed_ns().max(0) as u64;
    if observed != intrinsic.samples {
        return Err(Error::internal(format!(
            "inverse candidate '{}' closed intrinsically but the scalar evaluator disagrees",
            cand.label
        )));
    }

    // Bounded seek: materialize one window at a content-derived position and
    // require the same slice.
    let seek_frames = SEEK_FRAMES.min(intrinsic.frames as u32).max(1);
    let span = intrinsic.frames - u64::from(seek_frames);
    let hash = intrinsic.content_sha256();
    let seek_start = if span == 0 {
        0
    } else {
        let word = u64::from_le_bytes([
            hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
        ]);
        word % (span + 1)
    };
    let sw = Stopwatch::start();
    let window = observe::observe_window(
        &store,
        id,
        intrinsic.channels,
        seek_start as i64,
        seek_frames as usize,
    )?;
    let seek_latency_ns = sw.elapsed_ns().max(0) as u64;
    let ch = usize::from(intrinsic.channels);
    let start_idx = seek_start as usize * ch;
    let expect = &intrinsic.samples[start_idx..start_idx + seek_frames as usize * ch];
    if window != expect {
        return Err(Error::internal(format!(
            "inverse candidate '{}' reproduced the full window but not a seek window",
            cand.label
        )));
    }

    // Complete cost.
    let cost = match &data {
        ObjectData::Literal(l) => {
            cost::literal_cost(&l.samples, intrinsic.frames, intrinsic.channels)?
        }
        ObjectData::PredictorResidual(r) => {
            cost::residual_cost(&descriptor, r, intrinsic.frames, intrinsic.channels)?
        }
        other => {
            cost::canonical_object_cost(&descriptor, other, intrinsic.frames, intrinsic.channels)?
        }
    };
    let work = cost::abstract_work(&data, intrinsic.frames, intrinsic.channels);
    let accounted_peak_bytes = (intrinsic.samples.len() as u64) * 4
        + cost.persistent_bytes
        + u64::from(seek_frames) * u64::from(intrinsic.channels) * 4;

    Ok(Some(Acceptance {
        kind: cand.kind,
        label: cand.label.clone(),
        cost,
        work,
        content_id: store
            .get(id)
            .map(|o| o.content_id)
            .map_err(|e| Error::internal(format!("accepted candidate vanished: {e}")))?,
        reference_target: cand.reference,
        intrinsic_exact: true,
        evaluator_exact: true,
        seek_exact: true,
        materialize_ns,
        seek_latency_ns,
        intrinsic_ns,
        seek_start,
        seek_frames,
        accounted_peak_bytes,
        proposal_ns: cand.proposal_ns,
        seek_ops: work.window(u64::from(seek_frames), intrinsic.frames),
        total_ops: work.total(),
    }))
}
