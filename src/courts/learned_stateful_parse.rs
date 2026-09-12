//! `court learned-stateful-parse` — Phase 6 mechanism 1 (`StatefulSyntaxParse`).
//!
//! The fourth-pass program's first Phase-6 mechanism is a **stateful model
//! syntax**: a decoder-synchronized move-to-front carousel of the three most
//! recently used model tuples, so a repeated tuple costs two bits instead of a
//! full canonical model payload. The parser is a shortest path whose state is
//! `(position, carousel)` — choosing a tuple now changes its price later — kept
//! under a deterministic bounded beam.
//!
//! This court establishes the mechanism on three surfaces, each attributable and
//! never bundled:
//!
//! 1. **Synthetic repeated-regime fixtures** with a tiny fixed bank: it compares
//!    the bounded beam against exhaustive enumeration at widths 1/2/4/8/16, and
//!    measures the move-to-front saving against the *same parse* coded with a
//!    plain [`SegmentedModel`] (the exact ablation: identical residual, only the
//!    model syntax changes).
//! 2. **The frozen intrinsic corpus**: the same ablation over fitted regions.
//! 3. **The real speech effectiveness corpus** (LibriSpeech, CC BY 4.0), where
//!    the bank is the current speech portfolio itself: the parser chooses, per
//!    region, which already-fitted portfolio model to apply block-locally.
//!
//! Every assembled object must close exactly and round-trip through the
//! container. Selection never privileges a mechanism: `StatefulSyntax` is one
//! more candidate beside the existing portfolio, so the portfolio minimum can
//! only shrink. The held-out Mode-C split is deliberately untouched.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::arithmetic::quantize_weight;
use crate::learned::carousel::{
    ParseOutcome, RegionOption, StatefulSyntaxModel, SyntaxSegment, option_from_residual,
    parse_stateful, residual_under,
};
use crate::learned::corpus::intrinsic_cases;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::finite_field::LinearPredictor;
use crate::learned::fixed::FixedDifferencePredictor;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::segmented::{Segment, SegmentedModel};
use crate::learned::train::TrainBudget;
use crate::learned::train::linear::fit_linear_object_exp2;
use crate::learned::train::lpc::fit_lpc_object;
use crate::status::Verdict;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_STATEFUL_PARSE_SHA256: &str =
    "fe903ac63f40482bb620cdcffcb826f22c52a770ac70f4bd5e32185b111b6854";

/// Synthetic boundary grid (frames, coarsest first).
pub(crate) const SYNTH_GRID: [usize; 2] = [512, 256];
/// Real/intrinsic boundary grid. Chosen coarse enough that the node count stays
/// small (and the per-region fitting budget bounded) on full-length clips.
pub(crate) const FIT_GRID: [usize; 2] = [4096, 2048];
/// Beam widths measured against exhaustive enumeration.
const SYNTH_BEAMS: [usize; 5] = [1, 2, 4, 8, 16];
/// Production beam width for the fitted surfaces.
pub(crate) const FIT_BEAM: usize = 8;

fn bytes(o: &LearnedObject) -> Option<u64> {
    LearnedCost::of(o).ok().map(|c| c.complete_bytes)
}

// ---------------------------------------------------------------------------
// Synthetic fixtures
// ---------------------------------------------------------------------------

fn lcg(state: &mut u64) -> i32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) & 0xffff) as i32 - 32768
}

/// AR(1) samples whose predictor is exactly one bank model at this quantized
/// weight, so the matching tuple leaves only the injected noise.
fn ar_signal(n: usize, weight: f64, seed: u64) -> Vec<i32> {
    let q = i64::from(quantize_weight(weight));
    let mut s = seed | 1;
    let mut x = 0i64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = (q * x) >> 12;
        x += i64::from(lcg(&mut s)) >> 5;
        x = x.clamp(-(1 << 22), 1 << 22);
        out.push(x as i32);
    }
    out
}

fn ar_model(weight: f64) -> LearnedModel {
    LearnedModel::Linear(LinearPredictor {
        channels: 1,
        taps: 1,
        weights: vec![quantize_weight(weight)],
        bias: vec![0],
        block_frames: None,
    })
}

fn fixed_model(order: u8) -> LearnedModel {
    LearnedModel::Fixed(FixedDifferencePredictor {
        channels: 1,
        order,
        block_frames: None,
    })
}

pub(crate) fn synthetic_bank() -> Vec<LearnedModel> {
    vec![
        fixed_model(1),
        fixed_model(2),
        fixed_model(4),
        ar_model(0.95),
        ar_model(0.5),
        ar_model(0.0),
        ar_model(-0.5),
    ]
}

/// The frozen synthetic fixtures: regime sequences that force tuple reuse.
pub(crate) fn synthetic_fixtures() -> Vec<(&'static str, Vec<i32>)> {
    let mut out = Vec::new();
    {
        let mut s = Vec::new();
        s.extend(ar_signal(512, 0.95, 11));
        s.extend(ar_signal(512, -0.5, 22));
        s.extend(ar_signal(512, 0.95, 33));
        s.extend(ar_signal(512, -0.5, 44));
        out.push(("alternating_ab", s));
    }
    {
        let mut s = Vec::new();
        s.extend(ar_signal(512, 0.95, 101));
        s.extend(ar_signal(512, -0.5, 102));
        s.extend(ar_signal(512, 0.0, 103));
        s.extend(ar_signal(512, -0.5, 104));
        s.extend(ar_signal(512, 0.95, 105));
        out.push(("palindrome_abcba", s));
    }
    {
        let mut s = Vec::new();
        s.extend(ar_signal(512, 0.5, 201));
        s.extend(ar_signal(512, 0.5, 202));
        s.extend(ar_signal(512, -0.5, 203));
        s.extend(ar_signal(512, 0.5, 204));
        s.extend(ar_signal(512, -0.5, 205));
        s.extend(ar_signal(512, 0.5, 206));
        out.push(("drifting_ababab", s));
    }
    out
}

// ---------------------------------------------------------------------------
// Shared machinery
// ---------------------------------------------------------------------------

/// The result of one mechanism run on a mono signal.
struct MechRun {
    outcome: Option<ParseOutcome>,
    /// The `StatefulSyntax` object built from the chosen parse.
    dp_stateful: Option<LearnedObject>,
    /// The same parse and residual coded with a plain segmented model.
    dp_segmented_bytes: Option<u64>,
    /// The best single-model object over the whole extent (the unsplit floor).
    best_single: Option<LearnedObject>,
    exact: bool,
}

/// Build the `StatefulSyntax` object for a chosen parse.
pub(crate) fn build_stateful(
    source: &[i32],
    frames: usize,
    rate: u32,
    regions: &[(usize, usize)],
    models: &[LearnedModel],
) -> Option<LearnedObject> {
    if regions.is_empty() || regions.len() != models.len() {
        return None;
    }
    let segments = regions
        .iter()
        .zip(models)
        .map(|(&(a, b), m)| SyntaxSegment {
            frames: (b - a) as u32,
            model: Box::new(m.clone()),
        })
        .collect();
    let m = StatefulSyntaxModel {
        channels: 1,
        segments,
    };
    match LearnedObject::from_intrinsic_exp3(
        LearnedModel::StatefulSyntax(m),
        1,
        frames as u64,
        rate,
        Vec::new(),
        source,
    ) {
        Ok(o) if o.verify(source) => Some(o),
        _ => None,
    }
}

/// Exact byte delta of coding the same parse with a plain segmented model
/// instead of the move-to-front syntax.
///
/// Both wrappers sit in the identical container framing, so the complete cost
/// differs only by the canonical model length: `SegmentedModel` pays every
/// segment's full model payload, while `StatefulSyntaxModel` pays a fresh
/// payload only on a carousel miss. The residual is byte-identical (the same
/// per-segment block-local hypotheses), so this is an exact ablation without
/// building a second object. The `SegmentedModel` here is only serialized to
/// measure that length, never decoded.
fn segmented_syntax_model_len(regions: &[(usize, usize)], models: &[LearnedModel]) -> Option<u64> {
    if regions.is_empty() || regions.len() != models.len() {
        return None;
    }
    let segments = regions
        .iter()
        .zip(models)
        .map(|(&(a, b), m)| Segment {
            frames: (b - a) as u32,
            model: Box::new(m.clone()),
        })
        .collect();
    Some(
        SegmentedModel {
            channels: 1,
            segments,
        }
        .canonical_bytes()
        .len() as u64,
    )
}

/// Run the stateful parse and both ablations on one mono signal.
fn mechanism_run<F>(
    source: &[i32],
    rate: u32,
    grid: &[usize],
    beam: usize,
    bank: &[LearnedModel],
    mut extra: F,
) -> MechRun
where
    F: FnMut(usize, usize) -> Vec<LearnedModel>,
{
    let frames = source.len();
    let mut cache: HashMap<(usize, usize), Vec<LearnedModel>> = HashMap::new();
    let mut extra_call = |a: usize, b: usize| -> Vec<LearnedModel> {
        cache.entry((a, b)).or_insert_with(|| extra(a, b)).clone()
    };

    let outcome = {
        let options = |a: usize, b: usize| -> Vec<RegionOption> {
            let slice = &source[a..b];
            let mut opts: Vec<RegionOption> = Vec::new();
            for m in bank {
                if let Ok(r) = residual_under(m, slice, b - a) {
                    opts.push(option_from_residual(m.clone(), &r));
                }
            }
            for m in extra_call(a, b) {
                if let Ok(r) = residual_under(&m, slice, b - a) {
                    opts.push(option_from_residual(m, &r));
                }
            }
            opts
        };
        parse_stateful(frames, grid, beam, options)
    };

    let dp_stateful = outcome
        .as_ref()
        .and_then(|o| build_stateful(source, frames, rate, &o.regions, &o.models));
    // Ablation: the same parse and residual under plain segmented model syntax.
    let dp_segmented_bytes = dp_stateful.as_ref().and_then(|o| {
        let out = outcome.as_ref()?;
        let ss_len = o.model.canonical_bytes().len() as u64;
        let seg_len = segmented_syntax_model_len(&out.regions, &out.models)?;
        LearnedCost::of(o)
            .ok()
            .map(|c| c.complete_bytes - ss_len + seg_len)
    });

    // The unsplit floor: the best single model over the whole extent, charged
    // exactly through the container (not through the syntax wrapper). Only the
    // proxy-best model is assembled, so this never builds an object per bank
    // entry; the floor is dominated by the portfolio on the real surface anyway.
    let mut best_single: Option<LearnedObject> = None;
    let mut singles: Vec<LearnedModel> = bank.to_vec();
    singles.extend(extra_call(0, frames));
    let mut best_score = u64::MAX;
    for m in singles {
        let Ok(r) = residual_under(&m, source, frames) else {
            continue;
        };
        let payload = m.canonical_bytes().len() as u64;
        let score = crate::learned::carousel::proxy_residual_bits(&r).saturating_add(payload * 8);
        if score < best_score {
            best_score = score;
            best_single =
                LearnedObject::from_intrinsic_exp3(m, 1, frames as u64, rate, Vec::new(), source)
                    .ok()
                    .filter(|o| o.verify(source));
        }
    }

    let mut exact = outcome.is_some();
    // A parse that cannot be assembled, or whose object cannot be accounted,
    // is a failure, not a silent pass.
    exact &= dp_stateful
        .as_ref()
        .is_some_and(|o| o.verify(source) && LearnedCost::of(o).is_ok());
    exact &= best_single
        .as_ref()
        .is_some_and(|o| o.verify(source) && LearnedCost::of(o).is_ok());
    MechRun {
        outcome,
        dp_stateful,
        dp_segmented_bytes,
        best_single,
        exact,
    }
}

// ---------------------------------------------------------------------------
// Court
// ---------------------------------------------------------------------------

struct Gates {
    all_exact: bool,
    beam_matches_exhaustive: bool,
    mru_saves_on_a_fixture: bool,
}

fn synth_section(projection: &mut Vec<u8>, gates: &mut Gates) -> Vec<serde_json::Value> {
    let bank = synthetic_bank();
    let mut rows = Vec::new();
    for (name, source) in synthetic_fixtures() {
        let n = source.len();
        // Exhaustive parse for the true optimum and the byte ablations.
        let exhaustive = mechanism_run(&source, 48_000, &SYNTH_GRID, usize::MAX, &bank, |_, _| {
            Vec::new()
        });
        gates.all_exact &= exhaustive.exact;
        let Some(ex) = exhaustive.outcome.as_ref() else {
            common::push_label(projection, name);
            common::push_label(projection, "no-parse");
            continue;
        };
        let dp_bytes = exhaustive.dp_stateful.as_ref().and_then(bytes);
        let seg_bytes = exhaustive.dp_segmented_bytes;
        let single_bytes = exhaustive.best_single.as_ref().and_then(bytes);
        let mru_positive = dp_bytes
            .zip(seg_bytes)
            .is_some_and(|(d, s)| d <= s && ex.mru_hits >= 1);
        gates.mru_saves_on_a_fixture |= mru_positive;

        // Beam sweep. Only the widest measured beam is gated against exhaustive
        // enumeration; the whole ladder is reported so the width/quality curve is
        // attributable rather than assumed.
        let widest = SYNTH_BEAMS.iter().copied().max().unwrap_or(1);
        let mut beam_rows = Vec::new();
        for beam in SYNTH_BEAMS {
            let run = mechanism_run(&source, 48_000, &SYNTH_GRID, beam, &bank, |_, _| Vec::new());
            gates.all_exact &= run.exact;
            let bits = run.outcome.as_ref().map(|o| o.cost_bits);
            let equal = bits == Some(ex.cost_bits);
            if beam == widest {
                gates.beam_matches_exhaustive &= equal;
            }
            beam_rows.push(serde_json::json!({
                "beam": beam,
                "cost_bits": bits,
                "matches_exhaustive": equal,
            }));
        }

        common::push_label(projection, name);
        common::push_u64(projection, n as u64);
        common::push_u64(projection, ex.cost_bits);
        common::push_u64(projection, dp_bytes.unwrap_or(u64::MAX));
        common::push_u64(projection, seg_bytes.unwrap_or(u64::MAX));
        common::push_u64(projection, ex.mru_hits);
        common::push_u64(projection, ex.new_tuples);

        rows.push(serde_json::json!({
            "fixture": name,
            "frames": n,
            "exhaustive_cost_bits": ex.cost_bits,
            "segments": ex.regions.len(),
            "mru_hits": ex.mru_hits,
            "new_tuples": ex.new_tuples,
            "stateful_bytes": dp_bytes,
            "segmented_same_parse_bytes": seg_bytes,
            "best_single_bytes": single_bytes,
            "mru_saves": dp_bytes.zip(seg_bytes).map(|(d, s)| s as i64 - d as i64),
            "beam_sweep": beam_rows,
        }));
    }
    rows
}

fn intrinsic_section(
    projection: &mut Vec<u8>,
    gates: &mut Gates,
    budget: &TrainBudget,
) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();
    for case in intrinsic_cases().into_iter().take(6) {
        if case.channels != 1 {
            common::push_label(projection, case.id);
            common::push_label(projection, "multichannel-skip");
            continue;
        }
        let frames = case.samples.len() as u64;
        let rate = case.sample_rate_hz;
        let mut bank = Vec::new();
        if let Ok((o, _)) = fit_linear_object_exp2(
            &case.samples,
            1,
            frames,
            rate,
            4,
            None,
            frames as usize,
            budget,
        ) {
            bank.push(o.model);
        }
        for order in 1..=4u8 {
            bank.push(fixed_model(order));
        }
        let source = case.samples.clone();
        let run = mechanism_run(&source, rate, &FIT_GRID, FIT_BEAM, &bank, |a, b| {
            let mut out = Vec::new();
            if let Ok((o, _)) = fit_linear_object_exp2(
                &source[a..b],
                1,
                (b - a) as u64,
                rate,
                4,
                None,
                b - a,
                budget,
            ) {
                out.push(o.model);
            }
            out
        });
        gates.all_exact &= run.exact;
        let dp = run.dp_stateful.as_ref().and_then(bytes);
        let seg = run.dp_segmented_bytes;
        let single = run.best_single.as_ref().and_then(bytes);
        let (mru, new) = run
            .outcome
            .as_ref()
            .map(|o| (o.mru_hits, o.new_tuples))
            .unwrap_or((0, 0));
        common::push_label(projection, case.id);
        common::push_u64(projection, dp.unwrap_or(u64::MAX));
        common::push_u64(projection, seg.unwrap_or(u64::MAX));
        common::push_u64(projection, mru);
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "frames": frames,
            "stateful_bytes": dp,
            "segmented_same_parse_bytes": seg,
            "best_single_bytes": single,
            "mru_hits": mru,
            "new_tuples": new,
            "mru_saves": dp.zip(seg).map(|(d, s)| s as i64 - d as i64),
        }));
    }
    rows
}

fn real_section(
    projection: &mut Vec<u8>,
    gates: &mut Gates,
    budget: &TrainBudget,
    scratch: &Path,
) -> Result<Vec<serde_json::Value>> {
    let clips = load_cases(&effectiveness_clips(), 8, scratch)?;
    let mut rows = Vec::new();
    for case in &clips {
        let frames = case.frames();
        let rate = case.rate();
        let cands = speech::portfolio(case, budget)?;
        let mut portfolio_best = u64::MAX;
        let mut bank = Vec::new();
        for (_, o) in &cands {
            if let Some(b) = bytes(o) {
                portfolio_best = portfolio_best.min(b);
            }
            bank.push(o.model.clone());
        }
        let source = case.samples.clone();
        let run = mechanism_run(&source, rate, &FIT_GRID, FIT_BEAM, &bank, |a, b| {
            let mut out = Vec::new();
            if let Ok((o, _)) = fit_lpc_object(
                &source[a..b],
                (b - a) as u64,
                rate,
                (b - a) as u32,
                16,
                budget,
            ) {
                out.push(o.model);
            }
            out
        });
        gates.all_exact &= run.exact;
        let dp = run.dp_stateful.as_ref().and_then(bytes);
        let seg = run.dp_segmented_bytes;
        let single = run.best_single.as_ref().and_then(bytes);
        // The portfolio admits the new candidate, so it can only improve.
        let stateful_candidate = dp.unwrap_or(u64::MAX).min(single.unwrap_or(u64::MAX));
        let combined = portfolio_best.min(stateful_candidate);
        let (mru, new, segments) = run
            .outcome
            .as_ref()
            .map(|o| (o.mru_hits, o.new_tuples, o.regions.len()))
            .unwrap_or((0, 0, 0));
        common::push_label(projection, &case.clip.id);
        common::push_u64(projection, portfolio_best);
        common::push_u64(projection, stateful_candidate);
        common::push_u64(projection, combined);
        common::push_u64(projection, mru);
        rows.push(serde_json::json!({
            "id": case.clip.id,
            "frames": frames,
            "portfolio_bytes": portfolio_best,
            "stateful_syntax_bytes": dp,
            "segmented_same_parse_bytes": seg,
            "best_single_bytes": single,
            "stateful_candidate_bytes": stateful_candidate,
            "combined_portfolio_bytes": combined,
            "mru_saves": dp.zip(seg).map(|(d, s)| s as i64 - d as i64),
            "segments": segments,
            "mru_hits": mru,
            "new_tuples": new,
        }));
    }
    Ok(rows)
}

/// Run the court; writes `receipts/learned-stateful-parse/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.stateful_parse.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    let mut budget = common::train_budget();
    budget.max_iterations = 128;

    let mut gates = Gates {
        all_exact: true,
        beam_matches_exhaustive: true,
        mru_saves_on_a_fixture: false,
    };
    let synth = synth_section(&mut projection, &mut gates);
    let intrinsic = intrinsic_section(&mut projection, &mut gates, &budget);

    let (real, real_present) = if available() && decoder_available() {
        let scratch = PathBuf::from("target/real-corpus/scratch");
        let rows = real_section(&mut projection, &mut gates, &budget, &scratch)?;
        (rows, true)
    } else {
        (Vec::new(), false)
    };

    let supported =
        gates.all_exact && gates.beam_matches_exhaustive && gates.mru_saves_on_a_fixture;
    let verdict = if !supported {
        Verdict::FailedCorrectness
    } else if !real_present {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };

    let detail = format!(
        "stateful syntax parse: {} synthetic fixtures (beam == exhaustive, MRU ablation), {} \
         intrinsic cases, {} real effectiveness clips; move-to-front carousel over model tuples \
         with a `(position, carousel)` shortest path under a bounded beam",
        synth.len(),
        intrinsic.len(),
        real.len()
    );

    common::finish_exp3(
        "learned-stateful-parse",
        receipts_root,
        LEARNED_STATEFUL_PARSE_SHA256,
        &projection,
        verdict,
        detail,
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": gates.all_exact,
                    "beam_matches_exhaustive": gates.beam_matches_exhaustive,
                    "mru_saves_on_a_fixture": gates.mru_saves_on_a_fixture,
                }),
            ),
            ("synthetic", serde_json::json!(synth)),
            ("intrinsic", serde_json::json!(intrinsic)),
            ("real_effectiveness", serde_json::json!(real)),
            (
                "method",
                serde_json::json!({
                    "syntax": "00/01/10 = carousel slots MRU[0..2]; 11 = new tuple (varint \
                               length + canonical model bytes); move-to-front after every use",
                    "parser": "shortest path over (boundary, carousel) states; the carousel is \
                               part of the DP state because a choice changes later prices",
                    "beam": "deterministic bounded beam; widths 1/2/4/8/16 measured against \
                             exhaustive enumeration on the synthetic fixtures",
                    "price_model": "planning uses a cheap integer code-length proxy (min of \
                                    Exp-Golomb and best Rice); the assembled object is measured \
                                    exactly and compared against the unsplit floor",
                    "ablation": "the same parse and residual are also coded as a plain segmented \
                                 model, isolating the move-to-front model-syntax contribution",
                    "non_regression": "StatefulSyntax is one more portfolio candidate; the \
                                       portfolio minimum can only shrink",
                    "memory_bound": "the shortest path deduplicates exactly by carousel \
                                     (dominance) and enforces hard ceilings on grid nodes, \
                                     retained states per boundary and region alternatives, so an \
                                     unbounded parse can never allocate without limit",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the parser optimizes a code-length proxy, not the final entropy coder; the \
                     exact assembled bytes decide acceptance (IterativeReprice is the follow-up)",
                    "the real-corpus bank is the current speech portfolio plus a local LPC fit, \
                     applied block-locally by the parser",
                    "a move-to-front reference is only a win when a tuple genuinely recurs; the \
                     ablation reports the sign per surface",
                ]),
            ),
        ],
    )
}
