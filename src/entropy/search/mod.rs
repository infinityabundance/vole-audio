//! Zero-authority entropy candidate-search governance (H.2.26–H.2.29).
//!
//! The frozen candidate universe for the H.2 search (owner:
//! `docs/DSFB_SEARCH.md`):
//!
//! ```text
//! candidates = symbolization {identity, lane4_plain, lane4_zigzag,
//!                             delta_lane4}
//!           x  page_frames   {256, 512, 1024}
//!           x  model_mode    {inline, shared}
//! ```
//!
//! (the fixture/content class is fixed by the corpus) — 24 candidates per
//! fixture. Every candidate is evaluated by the *same* deterministic
//! measurement: [`RepresentedLiteral::encode`] (per-page RAW fallback
//! H.2.5), exact reconstruction verification, and complete-cost accounting
//! (H.2.11). The candidate **set** is identical across strategies; the
//! strategies differ only in evaluation order and the bounded budget:
//!
//! * `exhaustive` — evaluate all 24 candidates;
//! * `fixed-heuristic` — a frozen deterministic ordering with a bounded
//!   budget that always evaluates the RAW/literal fallback first;
//! * `dsfb-guided` (feature `dsfb`) — the DSFB-governed evaluation order
//!   with a bounded budget and regime-driven early stop
//!   (`docs/DSFB_SEARCH.md`, implementation in [`dsfb`]).
//!
//! Search governance is encoder-side only: the winner is always chosen by
//! the exact measured complete cost of the evaluated candidates, DSFB has
//! zero decoder authority, and the RAW/literal fallback can never be
//! suppressed. Everything is deterministic; there is no randomness.
//!
//! The measured quantities are `N` (candidates evaluated) and `J` (best
//! complete bytes found by that strategy). See
//! [`crate::courts::dsfb_entropy`] for the comparison court.

use crate::entropy::accounting::CompleteCost;
use crate::entropy::represent::{ModelMode, RepresentedLiteral, literal_container_bytes};
use crate::entropy::symbol::Symbolization;
use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::universe::layout::Layout;

/// DSFB-governed strategy (`feature dsfb`, default-off). See the module
/// documentation of [`dsfb`] for the governance semantics.
#[cfg(feature = "dsfb")]
pub mod dsfb;

/// The frozen symbolization axis (id order; versioned, never renumbered).
pub const UNIVERSE_SYMBOLIZATIONS: [Symbolization; 4] = [
    Symbolization::Identity,
    Symbolization::Lane4Plain,
    Symbolization::Lane4ZigZag,
    Symbolization::DeltaLane4,
];

/// The frozen page-size axis.
pub const UNIVERSE_PAGE_FRAMES: [u32; 3] = [256, 512, 1024];

/// The frozen model-mode axis (inline before shared).
pub const UNIVERSE_MODES: [ModelMode; 2] = [ModelMode::Inline, ModelMode::Shared];

/// Universe size: 4 symbolizations x 3 page sizes x 2 model modes.
pub const UNIVERSE_SIZE: usize = 24;

/// Bounded budget shared by the non-exhaustive strategies (candidate count).
/// The companion implementation maps the "balanced" regime to a 12-candidate
/// budget (`entropyfs` 0.7.17 `src/dsfb/selection.rs`); our universe is
/// exactly twice that.
pub const BUDGET: usize = 12;

/// One cell of the frozen candidate universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub symbolization: Symbolization,
    pub page_frames: u32,
    pub model_mode: ModelMode,
}

impl Candidate {
    /// Stable index in the canonical universe enumeration (symbolization
    /// major, page size, model mode). Used for tie-breaks and stable ids.
    pub const fn index(self) -> usize {
        let s = match self.symbolization {
            Symbolization::Identity => 0,
            Symbolization::Lane4Plain => 1,
            Symbolization::Lane4ZigZag => 2,
            Symbolization::DeltaLane4 => 3,
        };
        let p = match self.page_frames {
            256 => 0,
            512 => 1,
            _ => 2,
        };
        let m = match self.model_mode {
            ModelMode::Inline => 0,
            ModelMode::Shared => 1,
        };
        s * 6 + p * 2 + m
    }

    /// Short stable name, e.g. `delta_lane4/p1024/shared`.
    pub fn name(self) -> String {
        let mode = match self.model_mode {
            ModelMode::Inline => "inline",
            ModelMode::Shared => "shared",
        };
        format!(
            "{}/p{}/{}",
            self.symbolization.name(),
            self.page_frames,
            mode
        )
    }
}

/// The RAW/literal fallback anchor: plain identity symbolization (the raw LE
/// sample-byte stream; per-page RAW fallback competes inside every encode),
/// the largest page size (best model/index amortization), inline models.
/// Both bounded strategies evaluate this cell first, so the RAW/literal
/// fallback is always measured and the search can never silently depend on an
/// unevaluated literal floor.
pub const RAW_LITERAL_ANCHOR: Candidate = Candidate {
    symbolization: Symbolization::Identity,
    page_frames: 1024,
    model_mode: ModelMode::Inline,
};

/// The frozen universe, in canonical enumeration order (`index()` order).
pub fn frozen_universe() -> Vec<Candidate> {
    let mut out = Vec::with_capacity(UNIVERSE_SIZE);
    for s in UNIVERSE_SYMBOLIZATIONS {
        for &p in &UNIVERSE_PAGE_FRAMES {
            for m in UNIVERSE_MODES {
                out.push(Candidate {
                    symbolization: s,
                    page_frames: p,
                    model_mode: m,
                });
            }
        }
    }
    debug_assert_eq!(out.len(), UNIVERSE_SIZE);
    debug_assert_eq!(out[RAW_LITERAL_ANCHOR.index()], RAW_LITERAL_ANCHOR);
    out
}

/// One measured evaluation of a candidate.
#[derive(Debug, Clone)]
pub struct EvaluatedCell {
    pub candidate: Candidate,
    /// Complete representation bytes of this candidate (H.2.11).
    pub complete_bytes: u64,
    /// The full cost ledger.
    pub cost: CompleteCost,
    /// Fraction of pages whose per-page RAW fallback won (H.2.5).
    pub raw_fallback_fraction: f64,
    /// Wall encode time of this cell, ns (court instrumentation).
    pub encode_ns: u64,
}

/// Outcome of one strategy over one fixture.
#[derive(Debug, Clone)]
pub struct StrategyOutcome {
    /// Strategy identity (`exhaustive`, `fixed-heuristic`, `dsfb-guided`).
    pub strategy: &'static str,
    /// Cells in evaluation order (measured, never fabricated).
    pub evaluated: Vec<EvaluatedCell>,
}

impl StrategyOutcome {
    /// `N` — candidates evaluated.
    pub fn n(&self) -> usize {
        self.evaluated.len()
    }

    /// Best cell by complete bytes (ties -> earliest evaluation).
    pub fn best(&self) -> Option<&EvaluatedCell> {
        self.evaluated.iter().min_by_key(|c| c.complete_bytes)
    }

    /// `J` — best complete bytes found by this strategy.
    pub fn j(&self) -> Option<u64> {
        self.best().map(|c| c.complete_bytes)
    }

    /// Measured regret of this strategy vs an oracle best (never negative).
    pub fn regret_vs(&self, oracle: u64) -> Option<u64> {
        self.j().map(|j| j.saturating_sub(oracle))
    }
}

/// Literal descriptor for a (channels, frames) fixture.
pub fn literal_descriptor(channels: u8, frames: u64) -> Result<ObjectDescriptor> {
    let layout =
        Layout::checked(channels).ok_or_else(|| Error::malformed("layout out of domain"))?;
    ObjectDescriptor::new(Representation::Literal, frames, layout, None)
        .ok_or_else(|| Error::malformed("literal descriptor out of domain"))
}

/// Evaluate one candidate over the canonical interleaved samples:
/// encode (per-page RAW fallback), exact-reconstruction verification, and
/// complete-cost accounting. Shared by every strategy — the measurement is
/// identical; only the selection differs.
pub fn evaluate_literal(
    descriptor: &ObjectDescriptor,
    samples: &[i32],
    canonical_literal_bytes: u64,
    candidate: Candidate,
) -> Result<EvaluatedCell> {
    let sw = Stopwatch::start();
    let rl = RepresentedLiteral::encode(
        descriptor.clone(),
        samples,
        candidate.page_frames,
        candidate.symbolization,
        candidate.model_mode,
        false,
    )?;
    let encode_ns = sw.elapsed_ns() as u64;
    // Exact reconstruction is a property of every evaluated cell.
    let full = rl.materialize_full()?;
    if full != samples {
        return Err(Error::internal(
            "evaluated candidate does not reconstruct exactly",
        ));
    }
    let cost = rl.cost(canonical_literal_bytes)?;
    let raw_pages = rl
        .pages
        .iter()
        .filter(|p| p.kind == crate::entropy::represent::PageKind::Raw)
        .count();
    let raw_fallback_fraction = if rl.pages.is_empty() {
        1.0
    } else {
        raw_pages as f64 / rl.pages.len() as f64
    };
    // The canonical container must be reachable for every evaluated cell
    // (persistence adapters store exactly these bytes).
    let _ = literal_container_bytes(&rl)?;
    Ok(EvaluatedCell {
        candidate,
        complete_bytes: cost.complete_bytes,
        cost,
        raw_fallback_fraction,
        encode_ns,
    })
}

/// The exhaustive strategy: evaluate every candidate in the universe.
pub fn run_exhaustive(
    descriptor: &ObjectDescriptor,
    samples: &[i32],
    canonical_literal_bytes: u64,
) -> Result<StrategyOutcome> {
    let mut evaluated = Vec::with_capacity(UNIVERSE_SIZE);
    for c in frozen_universe() {
        evaluated.push(evaluate_literal(
            descriptor,
            samples,
            canonical_literal_bytes,
            c,
        )?);
    }
    Ok(StrategyOutcome {
        strategy: "exhaustive",
        evaluated,
    })
}

/// Frozen deterministic ordering for the fixed heuristic.
///
/// Content-blind rule (documented, never tuned to the corpus): page size
/// descending (larger pages amortize model + page-index overhead), then
/// symbolization in frozen id order, then inline before shared. The first
/// cell of this order is the [`RAW_LITERAL_ANCHOR`], so the RAW/literal
/// fallback is always evaluated within the budget.
fn fixed_heuristic_order() -> Vec<Candidate> {
    let mut order = frozen_universe();
    order.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.page_frames),
            c.symbolization.code(),
            match c.model_mode {
                ModelMode::Inline => 0u8,
                ModelMode::Shared => 1,
            },
        )
    });
    order
}

/// The fixed-heuristic strategy: the first [`BUDGET`] cells of the frozen
/// deterministic ordering.
pub fn run_fixed_heuristic(
    descriptor: &ObjectDescriptor,
    samples: &[i32],
    canonical_literal_bytes: u64,
) -> Result<StrategyOutcome> {
    let mut evaluated = Vec::with_capacity(BUDGET);
    for c in fixed_heuristic_order().into_iter().take(BUDGET) {
        evaluated.push(evaluate_literal(
            descriptor,
            samples,
            canonical_literal_bytes,
            c,
        )?);
    }
    debug_assert!(evaluated.iter().any(|c| c.candidate == RAW_LITERAL_ANCHOR));
    Ok(StrategyOutcome {
        strategy: "fixed-heuristic",
        evaluated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::corpus;

    /// Small deterministic fixture (first `frames` of a frozen corpus
    /// fixture). 2048 frames keeps unit tests fast.
    fn small_fixture(name: &str, frames: usize) -> (Vec<i32>, u8) {
        let fx = corpus::named(name).expect("fixture");
        let ch = usize::from(fx.channels);
        (fx.samples[..frames * ch].to_vec(), fx.channels)
    }

    fn run(name: &str, frames: usize) -> (ObjectDescriptor, Vec<i32>, u64) {
        let (samples, channels) = small_fixture(name, frames);
        let d =
            literal_descriptor(channels, (samples.len() / usize::from(channels)) as u64).unwrap();
        let u1 = 46 + 8 + samples.len() as u64 * 4;
        (d, samples, u1)
    }

    #[test]
    fn universe_is_frozen_24_unique() {
        let u = frozen_universe();
        assert_eq!(u.len(), UNIVERSE_SIZE);
        let mut seen = std::collections::BTreeSet::new();
        for (i, c) in u.iter().enumerate() {
            assert_eq!(c.index(), i);
            assert!(seen.insert(c.index()), "duplicate candidate");
        }
        assert_eq!(RAW_LITERAL_ANCHOR.index(), 4);
        assert_eq!(RAW_LITERAL_ANCHOR.name(), "identity/p1024/inline");
    }

    #[test]
    fn fixed_heuristic_order_leads_with_raw_literal_anchor() {
        let order = fixed_heuristic_order();
        assert_eq!(order.len(), UNIVERSE_SIZE);
        assert_eq!(order[0], RAW_LITERAL_ANCHOR);
        // Rule: page sizes descending within the prefix.
        assert_eq!(order[0].page_frames, 1024);
        assert_eq!(order[7].page_frames, 1024);
        assert_eq!(order[8].page_frames, 512);
        assert_eq!(order[15].page_frames, 512);
    }

    #[test]
    fn exhaustive_evaluates_every_candidate() {
        let (d, samples, u1) = run("single-sine", 2048);
        let out = run_exhaustive(&d, &samples, u1).unwrap();
        assert_eq!(out.n(), UNIVERSE_SIZE);
        assert_eq!(out.strategy, "exhaustive");
        let j = out.j().unwrap();
        // The best cell must be the min over the measured cells (self-check).
        for c in &out.evaluated {
            assert!(c.complete_bytes >= j);
        }
        // Raw baseline is frames*channels*4 for every cell.
        for c in &out.evaluated {
            assert_eq!(c.cost.raw_sample_bytes, samples.len() as u64 * 4);
        }
    }

    #[test]
    fn strategies_share_one_candidate_set() {
        let (d, samples, u1) = run("harmonic-tone", 2048);
        let ex = run_exhaustive(&d, &samples, u1).unwrap();
        let fx = run_fixed_heuristic(&d, &samples, u1).unwrap();
        assert_eq!(fx.n(), BUDGET);
        assert!(fx.n() < ex.n(), "bounded budget strictly below exhaustive");
        // The RAW/literal anchor is always evaluated by the bounded strategy.
        assert!(
            fx.evaluated
                .iter()
                .any(|c| c.candidate == RAW_LITERAL_ANCHOR)
        );
        // Same evaluation function: identical candidates measure identically.
        let anchor = fx.evaluated[0].candidate;
        let in_ex = ex
            .evaluated
            .iter()
            .find(|c| c.candidate == anchor)
            .expect("anchor in the universe");
        let in_fx = fx.evaluated.iter().find(|c| c.candidate == anchor).unwrap();
        assert_eq!(in_ex.complete_bytes, in_fx.complete_bytes);
    }

    #[test]
    fn deterministic_no_randomness() {
        let (d, samples, u1) = run("am-signal", 1024);
        let a = run_exhaustive(&d, &samples, u1).unwrap();
        let b = run_exhaustive(&d, &samples, u1).unwrap();
        for (x, y) in a.evaluated.iter().zip(b.evaluated.iter()) {
            assert_eq!(x.complete_bytes, y.complete_bytes);
        }
    }
}
