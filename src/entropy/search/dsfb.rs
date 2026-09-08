//! DSFB zero-authority search governance (H.2.26–H.2.29; feature `dsfb`,
//! default-off).
//!
//! Owner of the DSFB integration boundary (`docs/DSFB_SEARCH.md`). DSFB is
//! encoder/search governance **only**: it decides which candidates of the
//! frozen universe (see [`super`]) are evaluated next and when a bounded
//! search may stop. It has **zero decoder authority**: it never changes
//! normative samples, never alters entropy decoding, never makes an invalid
//! candidate valid, never bypasses exact closure, never suppresses the
//! RAW/literal fallback, and never overrides the final exact complete-cost
//! comparison (the winner is always the measured minimum of the evaluated
//! cells).
//!
//! # Honest naming / provenance
//!
//! The published `dsfb` crate (0.1.2, vendored) implements the DSFB
//! **state estimator**: a drift–slew-fusion observer tracking
//! level/drift/slew (`phi`/`omega`/`alpha`) of a measurement series with
//! adaptive trust weighting. Its API fits the *regime-tracking* half of
//! search governance, and this module wraps it thinly ([`PublishedDsfbProbe`]):
//! one measurement channel fed the winner-quality series of the search, its
//! raw state classified into Stable/Drift/Slew for reporting (the same
//! role the companion `entropyfs` 0.7.17 `src/dsfb/drift.rs::classify`
//! reserves for reporting — the raw α accumulation has a permanent-velocity
//! problem, so the runtime stop decision does not trust it).
//!
//! The published API does **not** natively select candidates across the
//! dimension-valued universe, so the *minimum deterministic observer* needed
//! for that half lives here too: a two-timescale regime tracker over the
//! winner-quality series (mirroring the companion `MeasurementTracker`
//! semantics) plus a per-dimension improvement-evidence ordering that
//! prioritizes the dimensions that measured the largest complete-cost
//! improvements. That minimum observer is deterministic, self-contained, and
//! deliberately small; it is named and documented as vole-audio's DSFB
//! observer rather than presented as the published crate.
//!
//! Deterministic by construction: no randomness anywhere; identical inputs
//! produce the identical evaluation order, N, and J.

use super::{Candidate, EvaluatedCell, RAW_LITERAL_ANCHOR, evaluate_literal};
use crate::error::Result;
use crate::object::descriptor::ObjectDescriptor;

/// Budget of the guided strategy (candidate count). Shared with
/// [`super::BUDGET`].
pub const GUIDED_BUDGET: usize = super::BUDGET;

/// Warm-up gate of the regime tracker (classifier fires from observation 5,
/// mirroring the companion tracker).
pub const TRACKER_WARMUP_SAMPLES: u64 = 4;

/// Consecutive Stable classifications after warm-up that stop the search.
pub const STABLE_STALL: u64 = 2;

/// Slew trigger: a single-step |delta| above this fraction of the scale.
pub const SLEW_DELTA_THRESHOLD: f64 = 0.25;
/// Drift trigger: |drift rate| above this (per-step, on [0,1] scale).
pub const DRIFT_RATE_THRESHOLD: f64 = 0.004;
/// Slew persistence window (steps).
pub const SLEW_WINDOW: u64 = 8;
/// Recovery: measurement within this distance of the pre-slew EMA ends a
/// slew early.
pub const SLEW_RECOVERY_EPS: f64 = 0.1;
/// Raw-state classifier thresholds (report-only; see module docs).
pub const RAW_SLEW_ALPHA: f64 = 0.05;
pub const RAW_DRIFT_OMEGA: f64 = 0.02;

/// Representation-regime classification (mirrors the companion vocabulary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// No evidence yet.
    Unknown,
    /// Stable: residual structure constant, basis effective.
    Stable,
    /// Drift: residual structure changes slowly on a stable basis.
    Drift,
    /// Slew: residual structure changed abruptly (regime break).
    Slew,
}

impl Regime {
    pub const fn label(self) -> &'static str {
        match self {
            Regime::Unknown => "unknown",
            Regime::Stable => "stable",
            Regime::Drift => "drift",
            Regime::Slew => "slew",
        }
    }
}

/// Classify the raw (φ, ω, α) observer state into a regime (report-only —
/// the runtime stop decision uses [`RegimeTracker`]).
pub fn classify_raw(phi: f64, omega: f64, alpha: f64) -> Regime {
    let _ = phi;
    if alpha.abs() > RAW_SLEW_ALPHA {
        Regime::Slew
    } else if omega.abs() > RAW_DRIFT_OMEGA {
        Regime::Drift
    } else {
        Regime::Stable
    }
}

/// Two-timescale regime tracker over the bounded winner-quality series
/// [0, 1].
///
/// Semantics mirror the companion `entropyfs` 0.7.17
/// `src/dsfb/drift.rs::MeasurementTracker` (itself the storage adaptation of
/// the published DSFB drift/slew classification): a fast EMA (α = 0.2)
/// tracks current quality, a slow EMA (α = 0.1) of per-step deltas tracks
/// the drift rate, and a single-step jump beyond the slew threshold opens a
/// persistent slew window with early recovery. All thresholds are in the
/// measurement scale [0, 1]. Deterministic; no allocation.
#[derive(Debug, Clone, Copy)]
pub struct RegimeTracker {
    /// Fast EMA of the measurement (α = 0.2) — current quality.
    ema: f64,
    /// Slow EMA of per-step deltas (α = 0.1) — the drift rate.
    delta_ema: f64,
    /// Measurements seen.
    samples: u64,
    /// Slew persistence window remaining (steps).
    slew_window: u64,
    /// Fast EMA at the moment the current slew was declared.
    pre_slew_ema: f64,
}

impl Default for RegimeTracker {
    fn default() -> Self {
        Self {
            ema: 0.0,
            delta_ema: 0.0,
            samples: 0,
            slew_window: 0,
            pre_slew_ema: 0.0,
        }
    }
}

impl RegimeTracker {
    /// Feed one bounded measurement `m ∈ [0, 1]`; returns the classified
    /// regime. The EMAs always update (even inside a slew window), so a slew
    /// eventually expires and the tracker re-baselines.
    pub fn observe(&mut self, m: f64) -> Regime {
        debug_assert!((0.0..=1.0).contains(&m));
        let m = m.clamp(0.0, 1.0);
        self.samples += 1;
        if self.samples == 1 {
            self.ema = m;
            return Regime::Unknown;
        }
        let delta = m - self.ema;
        self.delta_ema = 0.9 * self.delta_ema + 0.1 * delta;
        self.ema = 0.8 * self.ema + 0.2 * m;
        if self.slew_window > 0 {
            self.slew_window -= 1;
            if (m - self.pre_slew_ema).abs() < SLEW_RECOVERY_EPS {
                self.slew_window = 0;
            } else if self.slew_window > 0 {
                return Regime::Slew;
            }
        }
        if self.samples > TRACKER_WARMUP_SAMPLES && delta.abs() > SLEW_DELTA_THRESHOLD {
            self.slew_window = SLEW_WINDOW;
            self.pre_slew_ema = self.ema;
            return Regime::Slew;
        }
        if self.samples > TRACKER_WARMUP_SAMPLES && self.delta_ema.abs() > DRIFT_RATE_THRESHOLD {
            Regime::Drift
        } else {
            Regime::Stable
        }
    }

    /// Samples observed.
    pub fn samples(&self) -> u64 {
        self.samples
    }
}

/// One step of the published-crate drift-slew observer state.
#[derive(Debug, Clone, Copy)]
pub struct PublishedDsfbState {
    pub phi: f64,
    pub omega: f64,
    pub alpha: f64,
}

/// Thin adapter over the published `dsfb` crate: a one-channel
/// [`::dsfb::DsfbObserver`] tracking the search's winner-quality series.
/// The published algorithm's raw state is reported per step and classified
/// for reporting; it never selects candidates and never decides the winner.
pub struct PublishedDsfbProbe {
    observer: ::dsfb::DsfbObserver,
}

impl core::fmt::Debug for PublishedDsfbProbe {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The published observer exposes no Debug/Clone; a stable label is
        // enough for diagnostics.
        f.debug_struct("PublishedDsfbProbe").finish_non_exhaustive()
    }
}

impl Default for PublishedDsfbProbe {
    fn default() -> Self {
        Self {
            observer: ::dsfb::DsfbObserver::new(::dsfb::DsfbParams::default(), 1),
        }
    }
}

impl PublishedDsfbProbe {
    /// Feed one bounded measurement; returns the corrected state estimate.
    pub fn observe(&mut self, m: f64) -> PublishedDsfbState {
        let state = self.observer.step(&[m.clamp(0.0, 1.0)], 1.0);
        PublishedDsfbState {
            phi: state.phi,
            omega: state.omega,
            alpha: state.alpha,
        }
    }
}

/// One governed evaluation step.
#[derive(Debug, Clone)]
pub struct GuidedStep {
    pub cell: EvaluatedCell,
    /// Best complete bytes before this step.
    pub best_before: Option<u64>,
    /// Winner-quality after this step: `1 - best/anchor` (the bounded
    /// measurement series fed to the regime machinery).
    pub quality: f64,
    /// Regime from the deterministic tracker after this step.
    pub regime: Regime,
    /// Raw state of the published-crate observer after this step.
    pub dsfb_state: PublishedDsfbState,
    /// Regime classified from the raw published state (report-only).
    pub dsfb_raw_regime: Regime,
}

impl GuidedStep {
    /// Improvement of this candidate over the RAW/literal anchor
    /// (`1 - cost/anchor`, clamped to >= 0); the per-dimension evidence
    /// signal.
    fn improvement_vs_anchor(&self, anchor_cost: u64) -> f64 {
        if anchor_cost == 0 {
            return 0.0;
        }
        (1.0 - self.cell.complete_bytes as f64 / anchor_cost as f64).max(0.0)
    }
}

/// Why the guided search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The bounded budget was exhausted.
    Budget,
    /// The regime tracker reported Stable for [`STABLE_STALL`] consecutive
    /// observations after warm-up (uniform/stable content: the RAW/literal
    /// anchor already dominates).
    StableStall,
    /// Every candidate of the universe was evaluated.
    PoolExhausted,
}

impl StopReason {
    pub const fn label(self) -> &'static str {
        match self {
            StopReason::Budget => "budget",
            StopReason::StableStall => "stable-stall",
            StopReason::PoolExhausted => "pool-exhausted",
        }
    }
}

/// Full trace of one DSFB-guided run.
#[derive(Debug, Clone)]
pub struct GuidedTrace {
    pub steps: Vec<GuidedStep>,
    pub stop_reason: StopReason,
}

/// Dimension-value evidence key for the deterministic improvement-ordering
/// observer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EvKey {
    Sym(u8),
    Page(u32),
    Mode(u8),
}

fn candidate_keys(c: Candidate) -> [EvKey; 3] {
    [
        EvKey::Sym(c.symbolization.code()),
        EvKey::Page(c.page_frames),
        EvKey::Mode(match c.model_mode {
            crate::entropy::represent::ModelMode::Inline => 0,
            crate::entropy::represent::ModelMode::Shared => 1,
        }),
    ]
}

/// The DSFB-guided strategy over the frozen universe.
///
/// # Governance (deterministic, documented)
///
/// 1. Evaluate the [`RAW_LITERAL_ANCHOR`] first — the RAW/literal fallback
///    can never be suppressed, and it fixes the improvement scale.
/// 2. Breadth (Unknown regime, no evidence yet): evaluate the other three
///    symbolizations at the anchor's page size and model mode, so every
///    symbolization value has one measurement to earn credit from.
/// 3. Remaining budget: repeatedly evaluate the unevaluated candidate with
///    the highest dimension-improvement score — the sum, over its three
///    dimension values, of the largest improvement (vs the RAW/literal
///    anchor) any evaluated cell of that value achieved. Dimensions that
///    measured no improvement get no credit, so the search concentrates on
///    the promising dimensions (page sizes and model modes of the winning
///    symbolization are explored first). Ties break by canonical universe
///    index (deterministic).
/// 4. After every step the winner-quality series feeds the regime machinery
///    (deterministic tracker + published-crate observer). A single-step
///    quality jump opens the slew window and keeps the search broad; on
///    uniform/stable content — nothing improves over the RAW/literal
///    fallback — the tracker reports Stable for [`STABLE_STALL`] consecutive
///    observations after warm-up and the search stops early. The budget caps
///    the search at [`GUIDED_BUDGET`] candidates.
pub fn run_dsfb_guided(
    descriptor: &ObjectDescriptor,
    samples: &[i32],
    canonical_literal_bytes: u64,
) -> Result<(crate::entropy::search::StrategyOutcome, GuidedTrace)> {
    let universe = super::frozen_universe();
    let eval = |c: Candidate| evaluate_literal(descriptor, samples, canonical_literal_bytes, c);

    let mut evaluated: Vec<EvaluatedCell> = Vec::with_capacity(GUIDED_BUDGET);
    let mut trace_steps: Vec<GuidedStep> = Vec::with_capacity(GUIDED_BUDGET);
    let mut tracker = RegimeTracker::default();
    let mut probe = PublishedDsfbProbe::default();

    // Evidence per dimension value: max improvement vs the anchor measured
    // on any evaluated cell carrying that value.
    let mut evidence: std::collections::BTreeMap<EvKey, f64> = std::collections::BTreeMap::new();

    // Anchor evaluation (RAW/literal fallback guarantee).
    let anchor = eval(RAW_LITERAL_ANCHOR)?;
    let anchor_cost = anchor.complete_bytes;
    let mut best = anchor_cost;
    let mut quality = 0.0; // 1 - best/anchor: the winner-quality series.
    let regime = tracker.observe(quality);
    let dsfb_state = probe.observe(quality);
    let dsfb_raw_regime = classify_raw(dsfb_state.phi, dsfb_state.omega, dsfb_state.alpha);
    trace_steps.push(GuidedStep {
        cell: anchor.clone(),
        best_before: None,
        quality,
        regime,
        dsfb_state,
        dsfb_raw_regime,
    });
    evaluated.push(anchor);

    // Breadth: one measurement per symbolization value (the remaining three
    // symbolizations at the anchor's page size and model mode).
    let breadth: Vec<Candidate> = super::UNIVERSE_SYMBOLIZATIONS
        .iter()
        .filter(|s| **s != RAW_LITERAL_ANCHOR.symbolization)
        .map(|s| Candidate {
            symbolization: *s,
            page_frames: RAW_LITERAL_ANCHOR.page_frames,
            model_mode: RAW_LITERAL_ANCHOR.model_mode,
        })
        .collect();

    let mut stable_streak: u64 = 0;
    let mut stop_reason = StopReason::Budget;

    for c in breadth {
        if evaluated.len() >= GUIDED_BUDGET {
            break;
        }
        let cell = eval(c)?;
        let step = govern_step(
            &cell,
            &mut best,
            &mut quality,
            anchor_cost,
            &mut tracker,
            &mut probe,
            &mut evidence,
        );
        evaluated.push(cell);
        trace_steps.push(step);
    }

    while evaluated.len() < GUIDED_BUDGET {
        // Deterministic next-candidate selection: highest dimension-
        // improvement score, ties by canonical universe index.
        let next = universe
            .iter()
            .filter(|c| !evaluated.iter().any(|e| e.candidate == **c))
            .max_by(|a, b| {
                let score = |c: &Candidate| -> f64 {
                    candidate_keys(*c)
                        .iter()
                        .map(|k| evidence.get(k).copied().unwrap_or(0.0))
                        .sum()
                };
                let sa = score(a);
                let sb = score(b);
                // f64 total ordering (deterministic): highest score wins;
                // ties fall through to the canonical universe index, lowest
                // first.
                sa.total_cmp(&sb).then_with(|| b.index().cmp(&a.index()))
            })
            .copied();

        let Some(candidate) = next else {
            stop_reason = StopReason::PoolExhausted;
            break;
        };

        let cell = eval(candidate)?;
        let step = govern_step(
            &cell,
            &mut best,
            &mut quality,
            anchor_cost,
            &mut tracker,
            &mut probe,
            &mut evidence,
        );

        // Regime-driven early stop: uniform/stable content (nothing beats
        // the RAW/literal fallback) stalls after warm-up.
        if tracker.samples() > TRACKER_WARMUP_SAMPLES {
            if step.regime == Regime::Stable {
                stable_streak += 1;
            } else {
                stable_streak = 0;
            }
            if stable_streak >= STABLE_STALL {
                stop_reason = StopReason::StableStall;
                evaluated.push(cell);
                trace_steps.push(step);
                break;
            }
        }

        evaluated.push(cell);
        trace_steps.push(step);
    }

    Ok((
        crate::entropy::search::StrategyOutcome {
            strategy: "dsfb-guided",
            evaluated,
        },
        GuidedTrace {
            steps: trace_steps,
            stop_reason,
        },
    ))
}

/// One governed step: update the running best, the winner-quality series,
/// the regime machinery, and the per-dimension improvement evidence.
#[allow(clippy::too_many_arguments)]
fn govern_step(
    cell: &EvaluatedCell,
    best: &mut u64,
    quality: &mut f64,
    anchor_cost: u64,
    tracker: &mut RegimeTracker,
    probe: &mut PublishedDsfbProbe,
    evidence: &mut std::collections::BTreeMap<EvKey, f64>,
) -> GuidedStep {
    let best_before = *best;
    if cell.complete_bytes < *best {
        *best = cell.complete_bytes;
    }
    *quality = 1.0 - *best as f64 / anchor_cost as f64;
    let regime = tracker.observe(*quality);
    let dsfb_state = probe.observe(*quality);
    let dsfb_raw_regime = classify_raw(dsfb_state.phi, dsfb_state.omega, dsfb_state.alpha);
    let step = GuidedStep {
        cell: cell.clone(),
        best_before: Some(best_before),
        quality: *quality,
        regime,
        dsfb_state,
        dsfb_raw_regime,
    };
    let improvement = step.improvement_vs_anchor(anchor_cost);
    if improvement > 0.0 {
        for k in candidate_keys(step.cell.candidate) {
            let entry = evidence.entry(k).or_insert(0.0);
            *entry = entry.max(improvement);
        }
    }
    step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::corpus;
    use crate::entropy::search::{literal_descriptor, run_exhaustive};

    fn run(name: &str, frames: usize) -> (ObjectDescriptor, Vec<i32>, u64) {
        let fx = corpus::named(name).expect("fixture");
        let ch = usize::from(fx.channels);
        let samples = fx.samples[..frames * ch].to_vec();
        let d = literal_descriptor(fx.channels, frames as u64).unwrap();
        let u1 = 46 + 8 + samples.len() as u64 * 4;
        (d, samples, u1)
    }

    fn guided(name: &str, frames: usize) -> (crate::entropy::search::StrategyOutcome, GuidedTrace) {
        let (d, samples, u1) = run(name, frames);
        run_dsfb_guided(&d, &samples, u1).unwrap()
    }

    #[test]
    fn uniform_content_stops_early_at_the_raw_anchor() {
        // White noise: the RAW/literal anchor already dominates; the regime
        // tracker stalls and the guided search stops well below the budget.
        let (outcome, trace) = guided("white-noise", 2048);
        assert!(outcome.n() < GUIDED_BUDGET, "N = {}", outcome.n());
        assert_eq!(trace.stop_reason, StopReason::StableStall);
        // The anchor was evaluated and is the best cell.
        assert_eq!(outcome.best().unwrap().candidate, RAW_LITERAL_ANCHOR);
        // First step is the anchor.
        assert_eq!(trace.steps[0].cell.candidate, RAW_LITERAL_ANCHOR);
    }

    #[test]
    fn structured_content_reaches_the_exhaustive_best_within_budget() {
        // A pure tone: delta-lane4 at the large page size (shared pool wins
        // on identical per-page models). The credit ordering must reach the
        // exhaustive winner inside the budget.
        let (d, samples, u1) = run("single-sine", 2048);
        let ex = run_exhaustive(&d, &samples, u1).unwrap();
        let (outcome, _) = run_dsfb_guided(&d, &samples, u1).unwrap();
        let j_ex = ex.j().unwrap();
        let j_dsfb = outcome.j().unwrap();
        assert_eq!(
            j_dsfb, j_ex,
            "guided J must equal exhaustive J on a tone (regret would be recorded by the court)"
        );
        assert!(outcome.n() < ex.n(), "N_guided < N_exhaustive");
    }

    #[test]
    fn guided_is_deterministic_and_subset_of_universe() {
        let (d, samples, u1) = run("harmonic-tone", 1024);
        let (a, ta) = run_dsfb_guided(&d, &samples, u1).unwrap();
        let (b, tb) = run_dsfb_guided(&d, &samples, u1).unwrap();
        assert_eq!(a.n(), b.n());
        for (x, y) in a.evaluated.iter().zip(b.evaluated.iter()) {
            assert_eq!(x.candidate, y.candidate);
            assert_eq!(x.complete_bytes, y.complete_bytes);
        }
        assert_eq!(ta.stop_reason, tb.stop_reason);
        // Every guided cell is a member of the one frozen universe, and the
        // anchor is always evaluated.
        let universe = crate::entropy::search::frozen_universe();
        for cell in &a.evaluated {
            assert!(universe.contains(&cell.candidate));
        }
        assert!(
            a.evaluated
                .iter()
                .any(|c| c.candidate == RAW_LITERAL_ANCHOR)
        );
    }

    #[test]
    fn improvement_vs_anchor_is_bounded() {
        let (outcome, trace) = guided("am-signal", 1024);
        let anchor_cost = trace.steps[0].cell.complete_bytes;
        for step in &trace.steps {
            let imp = step.improvement_vs_anchor(anchor_cost);
            assert!((0.0..=1.0).contains(&imp));
        }
        assert!(outcome.n() >= 1);
    }

    #[test]
    fn raw_classify_thresholds() {
        assert_eq!(classify_raw(0.0, 0.001, 0.0), Regime::Stable);
        assert_eq!(classify_raw(0.0, 0.05, 0.0), Regime::Drift);
        assert_eq!(classify_raw(0.0, 0.05, 0.1), Regime::Slew);
    }

    #[test]
    fn tracker_stable_on_flat_series_and_slew_on_jump() {
        let mut t = RegimeTracker::default();
        assert_eq!(t.observe(1.0), Regime::Unknown);
        for _ in 0..20 {
            assert_eq!(t.observe(1.0), Regime::Stable);
        }
        let mut s = RegimeTracker::default();
        s.observe(1.0);
        for _ in 0..10 {
            s.observe(1.0);
        }
        assert_eq!(s.observe(0.0), Regime::Slew);
    }
}
