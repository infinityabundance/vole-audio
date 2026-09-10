//! Bounded deterministic candidate proposals (contract §33).
//!
//! There is **no randomness** here. Every proposal is derived from the
//! observed window by a closed-form or bounded-scan rule, so the same input
//! always yields the same candidate set. The families are deliberately
//! restricted to explanations that can be validated rigorously by the exact
//! evaluator:
//!
//! | family | hypothesis |
//! | ------ | ---------- |
//! | `Literal` | the canonical samples themselves (universal fallback) |
//! | `Silence` | zero everywhere |
//! | `Constant` | one level (the deterministic mode of channel 0) |
//! | `ExactRepeat` | the minimal exact frame period (string-period detection) |
//! | `ResidualZero` | zero hypothesis + exact sparse residual |
//! | `ResidualConstant` | mode-level hypothesis + exact sparse residual |
//! | `ResidualPeriodic` | bounded period scan, best K by residual record count |
//! | `SharedReference` | an object already in the reference library whose observation equals the window |
//!
//! Families the contract lists but that require a *residual-model vocabulary
//! extension* beyond the frozen u1 v1 models (delta/linear predictors,
//! partial/harmonic hypotheses) are documented as deferred in
//! `docs/INVERSE.md`; they are not silently approximated by a v1 model.
//! `Literal` is always proposed.

use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::id::ContentId;
use crate::object::wavetable::Cycle;
use crate::object::{
    Constant, Literal, ObjectData, ObjectStore, Residual, ResidualModel, ResidualRecord,
};
use crate::universe::layout::Layout;

use super::observe;
use super::{Candidate, CandidateKind, Intrinsic, SearchBudget};

/// Smallest number of frames a period scan can distinguish.
const MIN_PERIOD: u32 = 1;

/// Propose the bounded candidate set for one intrinsic window.
pub fn propose(
    intrinsic: &Intrinsic,
    library: &ReferenceLibrary,
    budget: SearchBudget,
) -> Result<Vec<Candidate>> {
    let layout = observe::layout_of(intrinsic.channels)?;
    let frames = intrinsic.frames as usize;
    let channels = usize::from(intrinsic.channels);
    let mut out: Vec<Candidate> = Vec::with_capacity(budget.max_candidates.min(16));

    // 1. Literal — always a candidate (universal fallback).
    let sw = Stopwatch::start();
    let lit_desc = ObjectDescriptor::new(Representation::Literal, intrinsic.frames, layout, None)
        .ok_or_else(|| Error::malformed("literal descriptor out of domain"))?;
    let literal = Literal::new(&lit_desc, intrinsic.samples.clone())
        .ok_or_else(|| Error::malformed("literal payload out of domain"))?;
    out.push(Candidate {
        kind: CandidateKind::Literal,
        label: "literal".to_string(),
        object: Some((lit_desc, ObjectData::Literal(literal))),
        reference: None,
        proposal_ns: sw.elapsed_ns().max(0) as u64,
    });

    // 2. Silence (endless, extent 0).
    let sw = Stopwatch::start();
    let sil_desc = ObjectDescriptor::new(Representation::Silence, 0, layout, None)
        .ok_or_else(|| Error::malformed("silence descriptor out of domain"))?;
    out.push(Candidate {
        kind: CandidateKind::Silence,
        label: "silence".to_string(),
        object: Some((sil_desc, ObjectData::Silence)),
        reference: None,
        proposal_ns: sw.elapsed_ns().max(0) as u64,
    });

    // 3. Constant (endless, extent 0) at the deterministic mode of channel 0.
    let sw = Stopwatch::start();
    let level = mode_level(&intrinsic.samples, channels, frames);
    let cst_desc = ObjectDescriptor::new(Representation::Constant, 0, layout, None)
        .ok_or_else(|| Error::malformed("constant descriptor out of domain"))?;
    out.push(Candidate {
        kind: CandidateKind::Constant,
        label: format!("constant(level={level})"),
        object: Some((cst_desc, ObjectData::Constant(Constant::new(level)))),
        reference: None,
        proposal_ns: sw.elapsed_ns().max(0) as u64,
    });

    // 4. ExactRepeat at the minimal exact frame period (< frames).
    let sw = Stopwatch::start();
    if frames >= 2 {
        let period = minimal_frame_period(&intrinsic.samples, channels, frames);
        if period >= 1
            && period < frames
            && let Some((desc, data)) = exact_repeat(intrinsic, layout, period)
        {
            out.push(Candidate {
                kind: CandidateKind::ExactRepeat,
                label: format!("exact_repeat(period={period})"),
                object: Some((desc, data)),
                reference: None,
                proposal_ns: sw.elapsed_ns().max(0) as u64,
            });
        }
    }

    // 5. Shared reference: an object already in the library whose observation
    //    equals the window exactly (archive-level deduplication).
    let sw = Stopwatch::start();
    if let Some(target) = library.exact_match(&intrinsic.samples, intrinsic.channels, frames)? {
        out.push(Candidate {
            kind: CandidateKind::SharedReference,
            label: format!("shared_reference({})", &target.to_string()[..12]),
            object: None,
            reference: Some(target),
            proposal_ns: sw.elapsed_ns().max(0) as u64,
        });
    }

    // 6. PredictorResidual(Zero) — the literal-coded residual floor.
    let sw = Stopwatch::start();
    let zero = ResidualModel::Zero;
    if let Some(cand) = residual_candidate(
        intrinsic,
        layout,
        zero,
        CandidateKind::ResidualZero,
        "residual(zero)".to_string(),
        sw.elapsed_ns().max(0) as u64,
    )? {
        out.push(cand);
    }

    // 7. PredictorResidual(Constant) at the mode level.
    let sw = Stopwatch::start();
    if let Some(cand) = residual_candidate(
        intrinsic,
        layout,
        ResidualModel::Constant(level),
        CandidateKind::ResidualConstant,
        format!("residual(constant={level})"),
        sw.elapsed_ns().max(0) as u64,
    )? {
        out.push(cand);
    }

    // 8. PredictorResidual(Periodic p) for the best K bounded-scan periods
    //    (the v1 periodic model is mono, so this family is mono-only).
    if intrinsic.channels == 1 {
        let sw = Stopwatch::start();
        let scanned = scan_periods(
            &intrinsic.samples,
            frames,
            budget.max_period_scan,
            budget.max_residual_period_candidates,
        );
        let mut proposal_ns = sw.elapsed_ns().max(0) as u64;
        for (period, _records) in scanned {
            let cycle = intrinsic.samples[..period].to_vec();
            if let Some(cand) = residual_candidate(
                intrinsic,
                layout,
                ResidualModel::Periodic { cycle },
                CandidateKind::ResidualPeriodic,
                format!("residual(period={period})"),
                proposal_ns,
            )? {
                out.push(cand);
            }
            proposal_ns = 0; // attributed once to the scan
        }
    }

    // Bounded: truncate deterministically (families are ordered by increasing
    // representational complexity, and residual period candidates were already
    // ranked by residual sparsity).
    if out.len() > budget.max_candidates {
        out.truncate(budget.max_candidates);
    }
    Ok(out)
}

/// Build the `ExactRepeat` payload for an exact frame period.
fn exact_repeat(
    intrinsic: &Intrinsic,
    layout: Layout,
    period: usize,
) -> Option<(ObjectDescriptor, ObjectData)> {
    let channels = usize::from(intrinsic.channels);
    let cycle_samples = intrinsic.samples[..period * channels].to_vec();
    let desc = ObjectDescriptor::new(Representation::ExactRepeat, period as u64, layout, None)?;
    let cycle = Cycle::new(&desc, cycle_samples)?;
    Some((desc, ObjectData::ExactRepeat(cycle)))
}

/// Build a residual candidate for one hypothesis; `None` when the hypothesis
/// cannot close the window in the i32 delta domain (dropped, never
/// approximated).
fn residual_candidate(
    intrinsic: &Intrinsic,
    layout: Layout,
    model: ResidualModel,
    kind: CandidateKind,
    label: String,
    proposal_ns: u64,
) -> Result<Option<Candidate>> {
    let channels = intrinsic.channels;
    let records = match Residual::closing_residual(&intrinsic.samples, channels, &model) {
        Some(r) => r,
        None => return Ok(None),
    };
    let desc = ObjectDescriptor::new(
        Representation::PredictorResidual,
        intrinsic.frames,
        layout,
        None,
    )
    .ok_or_else(|| Error::malformed("residual descriptor out of domain"))?;
    let residual = match Residual::new(&desc, model, records) {
        Some(r) => r,
        None => return Ok(None),
    };
    Ok(Some(Candidate {
        kind,
        label,
        object: Some((desc, ObjectData::PredictorResidual(residual))),
        reference: None,
        proposal_ns,
    }))
}

/// Deterministic mode of channel 0 (ties broken by the smaller value).
fn mode_level(samples: &[i32], channels: usize, frames: usize) -> i32 {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<i32, u64> = BTreeMap::new();
    for f in 0..frames {
        *counts.entry(samples[f * channels]).or_insert(0) += 1;
    }
    let mut best = 0i32;
    let mut best_count = 0u64;
    for (value, count) in counts {
        if count > best_count {
            best = value;
            best_count = count;
        }
    }
    best
}

/// Minimal exact frame period of an interleaved window (the smallest `p` with
/// `X[f] == X[f - p]` for all `f >= p`), computed with a KMP prefix function
/// over frame blocks. `frames` is returned when no smaller period exists.
fn minimal_frame_period(samples: &[i32], channels: usize, frames: usize) -> usize {
    let eq = |a: usize, b: usize| -> bool {
        samples[a * channels..a * channels + channels]
            == samples[b * channels..b * channels + channels]
    };
    let mut pi = vec![0usize; frames];
    for i in 1..frames {
        let mut k = pi[i - 1];
        while k > 0 && !eq(i, k) {
            k = pi[k - 1];
        }
        if eq(i, k) {
            k += 1;
        }
        pi[i] = k;
    }
    frames - pi[frames - 1]
}

/// Bounded period scan for the periodic residual family: residual record count
/// for every `p` in `[MIN_PERIOD, min(max_scan, frames - 1)]`, ranked by
/// (record count ascending, period ascending) and truncated to `keep`.
fn scan_periods(samples: &[i32], frames: usize, max_scan: u32, keep: usize) -> Vec<(usize, usize)> {
    if frames < 2 || keep == 0 {
        return Vec::new();
    }
    let limit = (max_scan as usize).min(frames - 1).max(MIN_PERIOD as usize);
    let mut rows: Vec<(usize, usize)> = Vec::new();
    for p in MIN_PERIOD as usize..=limit {
        let cycle = samples[..p].to_vec();
        let model = ResidualModel::Periodic { cycle };
        let Some(records) = Residual::closing_residual(samples, 1, &model) else {
            continue;
        };
        rows.push((records.len(), p));
    }
    rows.sort_by_key(|&(count, p)| (count, p));
    rows.truncate(keep);
    // Stable, deterministic final order by period so proposal order is
    // independent of the ranking tie structure.
    rows.sort_by_key(|&(_count, p)| p);
    rows.into_iter().map(|(count, p)| (p, count)).collect()
}

/// A reference library: objects whose content is already stored, used to
/// propose exact shared references (archive-level deduplication).
#[derive(Debug, Clone, Default)]
pub struct ReferenceLibrary {
    store: ObjectStore,
}

impl ReferenceLibrary {
    pub fn new() -> ReferenceLibrary {
        ReferenceLibrary {
            store: ObjectStore::new(),
        }
    }

    /// Register an object; returns its content id.
    pub fn register(
        &mut self,
        descriptor: ObjectDescriptor,
        data: ObjectData,
    ) -> Result<ContentId> {
        let id = self.store.insert(descriptor, data)?;
        Ok(self.store.get(id)?.content_id)
    }

    /// Register canonical literal content.
    pub fn register_literal(&mut self, channels: u8, samples: Vec<i32>) -> Result<ContentId> {
        let frames = (samples.len() / usize::from(channels)) as u64;
        if frames * u64::from(channels) != samples.len() as u64 {
            return Err(Error::malformed("library literal is not frame-aligned"));
        }
        let layout = observe::layout_of(channels)?;
        let desc = ObjectDescriptor::new(Representation::Literal, frames, layout, None)
            .ok_or_else(|| Error::malformed("library literal descriptor out of domain"))?;
        let data = Literal::new(&desc, samples)
            .ok_or_else(|| Error::malformed("library literal payload out of domain"))?;
        self.register(desc, ObjectData::Literal(data))
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    pub fn store(&self) -> &ObjectStore {
        &self.store
    }

    /// True when the library observation over `frames` reproduces the window
    /// exactly; returns the matching target's content id (smallest id wins,
    /// deterministic).
    pub fn exact_match(
        &self,
        samples: &[i32],
        channels: u8,
        frames: usize,
    ) -> Result<Option<ContentId>> {
        let expected_len = frames
            .checked_mul(usize::from(channels))
            .ok_or_else(|| Error::limit("library match window overflow"))?;
        if samples.len() != expected_len {
            return Ok(None);
        }
        let mut candidates: Vec<&crate::object::SampleObject> = self
            .store
            .iter()
            .filter(|o| o.descriptor.layout.count() == channels)
            .collect();
        // Deterministic order by archive-local id.
        candidates.sort_by_key(|o| o.id);
        for obj in candidates {
            if let Ok(observed) = observe::observe_object(&self.store, obj.id, channels, frames)
                && observed == samples
            {
                return Ok(Some(obj.content_id));
            }
        }
        Ok(None)
    }
}

/// Deterministic ordering of residual records is the responsibility of
/// `Residual::new`; this re-export keeps the type visible to callers that
/// build residuals directly.
pub type Records = Vec<ResidualRecord>;
