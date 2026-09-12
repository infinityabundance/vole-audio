//! `court learned-iterative-reprice` — Phase 6 mechanism 2 (`IterativeReprice`).
//!
//! Mechanism 1 selects a parse by minimizing a *proxy* price (the smaller of the
//! canonical Exp-Golomb length and the best Rice length). That proxy ignores the
//! residual distribution the chosen parse actually induces, so the parse and the
//! prices it is judged by are inconsistent.
//!
//! `IterativeReprice` is the coordinate descent that closes the loop:
//!
//! ```text
//! P_0 → Parse_0 → P_1 → Parse_1 → … → stop when physical bytes stop improving
//! ```
//!
//! `P_{k+1}` is fit from the exact residual produced by `Parse_k`; every region
//! alternative is repriced under it; the parser runs again. The loop stops when
//! the exact assembled bytes do not shrink, when the parse repeats, or at a
//! frozen iteration ceiling. Iteration 0 (the proxy parse) is always retained as
//! a candidate, so the result can never be larger than the single-pass parse.
//!
//! Nothing about the format changes: the winner is still an ordinary
//! `stateful_syntax` object whose exact bytes decide acceptance.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::courts::learned_stateful_parse as p6;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::carousel::{RegionOption, parse_stateful, proxy_residual_bits, residual_under};
use crate::learned::corpus::intrinsic_cases;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::reprice::PriceTable;
use crate::learned::train::linear::fit_linear_object_exp2;
use crate::learned::train::lpc::fit_lpc_object;
use crate::status::Verdict;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_ITERATIVE_REPRICE_SHA256: &str =
    "06b1806705f148f7ed7c66030421c293072ad1fd06cb5e2668021a2b51fdf08e";

/// Frozen iteration ceiling (additional parses after the proxy parse).
const MAX_REPRICE_ITERS: usize = 4;

type RegionModels = HashMap<(usize, usize), Vec<(LearnedModel, Vec<i32>)>>;

fn bytes(o: &LearnedObject) -> Option<u64> {
    LearnedCost::of(o).ok().map(|c| c.complete_bytes)
}

/// One surface's reprice outcome.
#[derive(Debug, Clone, Copy, Default)]
struct RepriceRun {
    baseline_bytes: Option<u64>,
    best_bytes: Option<u64>,
    iterations: u32,
    converged: bool,
    improved: bool,
    base_segments: usize,
    best_segments: usize,
    exact: bool,
}

/// Materialize (and cache) every region alternative, once, for one signal.
///
/// Options are fitted once and reused across iterations: repricing changes only
/// the prices, never the candidate set.
fn ensure_region<F>(
    cache: &mut RegionModels,
    extra_cache: &mut HashMap<(usize, usize), Vec<LearnedModel>>,
    extra: &mut F,
    source: &[i32],
    bank: &[LearnedModel],
    a: usize,
    b: usize,
) where
    F: FnMut(usize, usize) -> Vec<LearnedModel>,
{
    if cache.contains_key(&(a, b)) {
        return;
    }
    let slice = &source[a..b];
    let mut out: Vec<(LearnedModel, Vec<i32>)> = Vec::new();
    for m in bank {
        if let Ok(r) = residual_under(m, slice, b - a) {
            out.push((m.clone(), r));
        }
    }
    let locals = extra_cache
        .entry((a, b))
        .or_insert_with(|| extra(a, b))
        .clone();
    for m in locals {
        if let Ok(r) = residual_under(&m, slice, b - a) {
            out.push((m, r));
        }
    }
    cache.insert((a, b), out);
}

/// Run the reprice loop on one mono signal.
fn run_reprice<F>(
    source: &[i32],
    rate: u32,
    grid: &[usize],
    beam: usize,
    bank: &[LearnedModel],
    mut extra: F,
) -> RepriceRun
where
    F: FnMut(usize, usize) -> Vec<LearnedModel>,
{
    let frames = source.len();
    let mut cache: RegionModels = HashMap::new();
    let mut extra_cache: HashMap<(usize, usize), Vec<LearnedModel>> = HashMap::new();

    // Pass 0: collect every region alternative and parse under the proxy price.
    let parse0 = {
        let collect = |a: usize, b: usize| -> Vec<RegionOption> {
            ensure_region(&mut cache, &mut extra_cache, &mut extra, source, bank, a, b);
            cache
                .get(&(a, b))
                .map(|v| {
                    v.iter()
                        .map(|(m, r)| RegionOption {
                            model: m.clone(),
                            frames: r.len() as u32,
                            residual_bits: proxy_residual_bits(r),
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        parse_stateful(frames, grid, beam, collect)
    };
    let Some(p0) = parse0 else {
        return RepriceRun::default();
    };
    let Some(obj0) = p6::build_stateful(source, frames, rate, &p0.regions, &p0.models) else {
        return RepriceRun::default();
    };
    let Some(b0) = bytes(&obj0) else {
        return RepriceRun::default();
    };
    let mut exact = obj0.verify(source);
    let mut best_bytes = b0;
    let mut best_obj = obj0;
    let base_segments = p0.regions.len();
    let mut best_segments = base_segments;
    let mut prev_regions = p0.regions.clone();
    let mut prev_models = p0.models.clone();
    let mut table = PriceTable::fit(&best_obj.residual().unwrap_or_default());
    let mut iterations = 0u32;
    let mut converged = false;
    let mut improved = false;

    for _ in 0..MAX_REPRICE_ITERS {
        iterations += 1;
        let parse = {
            let opts = |a: usize, b: usize| -> Vec<RegionOption> {
                cache
                    .get(&(a, b))
                    .map(|v| {
                        v.iter()
                            .map(|(m, r)| RegionOption {
                                model: m.clone(),
                                frames: r.len() as u32,
                                residual_bits: table.price_bits(r),
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };
            parse_stateful(frames, grid, beam, opts)
        };
        let Some(pk) = parse else {
            converged = true;
            break;
        };
        if pk.regions == prev_regions && pk.models == prev_models {
            converged = true;
            break;
        }
        let Some(objk) = p6::build_stateful(source, frames, rate, &pk.regions, &pk.models) else {
            converged = true;
            break;
        };
        exact &= objk.verify(source);
        let Some(bk) = bytes(&objk) else {
            converged = true;
            break;
        };
        if bk < best_bytes {
            best_bytes = bk;
            best_obj = objk;
            best_segments = pk.regions.len();
            improved = true;
            table = PriceTable::fit(&best_obj.residual().unwrap_or_default());
            prev_regions = pk.regions;
            prev_models = pk.models;
        } else {
            converged = true;
            break;
        }
    }

    exact &= best_obj.verify(source) && LearnedCost::of(&best_obj).is_ok();
    RepriceRun {
        baseline_bytes: Some(b0),
        best_bytes: Some(best_bytes),
        iterations,
        converged,
        improved,
        base_segments,
        best_segments,
        exact,
    }
}

fn push_row(
    projection: &mut Vec<u8>,
    label: &str,
    run: &RepriceRun,
    gates: &mut bool,
) -> serde_json::Value {
    *gates &= run.exact;
    *gates &= run.best_bytes <= run.baseline_bytes;
    common::push_label(projection, label);
    common::push_u64(projection, run.baseline_bytes.unwrap_or(u64::MAX));
    common::push_u64(projection, run.best_bytes.unwrap_or(u64::MAX));
    common::push_u64(projection, u64::from(run.iterations));
    serde_json::json!({
        "id": label,
        "baseline_bytes": run.baseline_bytes,
        "best_bytes": run.best_bytes,
        "delta_bytes": run
            .baseline_bytes
            .zip(run.best_bytes)
            .map(|(b, x)| b as i64 - x as i64),
        "iterations": run.iterations,
        "converged": run.converged,
        "improved": run.improved,
        "segments_baseline": run.base_segments,
        "segments_best": run.best_segments,
    })
}

/// Run the court; writes `receipts/learned-iterative-reprice/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.iterative_reprice.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let mut all_ok = true;

    // Synthetic repeated-regime fixtures (exhaustive parse: isolates pricing).
    let bank = p6::synthetic_bank();
    let mut synthetic = Vec::new();
    for (name, source) in p6::synthetic_fixtures() {
        let run = run_reprice(
            &source,
            48_000,
            &p6::SYNTH_GRID,
            usize::MAX,
            &bank,
            |_, _| Vec::new(),
        );
        synthetic.push(push_row(&mut projection, name, &run, &mut all_ok));
    }

    // Frozen intrinsic corpus.
    let mut intrinsic = Vec::new();
    for case in intrinsic_cases().into_iter().take(6) {
        if case.channels != 1 {
            common::push_label(&mut projection, case.id);
            common::push_label(&mut projection, "multichannel-skip");
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
            &budget,
        ) {
            bank.push(o.model);
        }
        for order in 1..=4u8 {
            bank.push(LearnedModel::Fixed(
                crate::learned::fixed::FixedDifferencePredictor {
                    channels: 1,
                    order,
                    block_frames: None,
                },
            ));
        }
        let source = case.samples.clone();
        let run = run_reprice(&source, rate, &p6::FIT_GRID, p6::FIT_BEAM, &bank, |a, b| {
            let mut out = Vec::new();
            if let Ok((o, _)) = fit_linear_object_exp2(
                &source[a..b],
                1,
                (b - a) as u64,
                rate,
                4,
                None,
                b - a,
                &budget,
            ) {
                out.push(o.model);
            }
            out
        });
        intrinsic.push(push_row(&mut projection, case.id, &run, &mut all_ok));
    }

    // Real speech effectiveness corpus.
    let mut real = Vec::new();
    let mut real_present = false;
    if available() && decoder_available() {
        real_present = true;
        let scratch = PathBuf::from("target/real-corpus/scratch");
        for case in load_cases(&effectiveness_clips(), 8, &scratch)? {
            let rate = case.rate();
            let cands = speech::portfolio(&case, &budget)?;
            let bank: Vec<LearnedModel> = cands.into_iter().map(|(_, o)| o.model).collect();
            let source = case.samples.clone();
            let run = run_reprice(&source, rate, &p6::FIT_GRID, p6::FIT_BEAM, &bank, |a, b| {
                let mut out = Vec::new();
                if let Ok((o, _)) = fit_lpc_object(
                    &source[a..b],
                    (b - a) as u64,
                    rate,
                    (b - a) as u32,
                    16,
                    &budget,
                ) {
                    out.push(o.model);
                }
                out
            });
            real.push(push_row(&mut projection, &case.clip.id, &run, &mut all_ok));
        }
    }

    let verdict = if !all_ok {
        Verdict::FailedCorrectness
    } else if !real_present {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };
    let improved = synthetic
        .iter()
        .chain(intrinsic.iter())
        .chain(real.iter())
        .filter(|r| r["improved"] == serde_json::json!(true))
        .count();

    common::finish_exp3(
        "learned-iterative-reprice",
        receipts_root,
        LEARNED_ITERATIVE_REPRICE_SHA256,
        &projection,
        verdict,
        format!(
            "iterative reprice: {} synthetic + {} intrinsic + {} real surfaces; {} improved over \
             the single-pass proxy parse; coordinate descent between the parse and the entropy \
             prices it induces, accepted only on exact assembled bytes",
            synthetic.len(),
            intrinsic.len(),
            real.len(),
            improved
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_ok,
                    "non_regression": true,
                }),
            ),
            ("synthetic", serde_json::json!(synthetic)),
            ("intrinsic", serde_json::json!(intrinsic)),
            ("real_effectiveness", serde_json::json!(real)),
            (
                "method",
                serde_json::json!({
                    "price_model": "order-0 empirical magnitude-bucket distribution, one bit of \
                                    sign per nonzero sample, Q8 integer costs",
                    "loop": "P0 -> Parse0 -> P1 -> Parse1 -> ... where P_{k+1} is fit from the exact \
                             residual of Parse_k",
                    "stop": "exact assembled bytes stop shrinking, or the parse repeats, or the \
                             frozen iteration ceiling is reached",
                    "retention": "iteration 0 is always a candidate, so the result can never be \
                                  larger than the single-pass parse",
                    "format": "unchanged: the winner is an ordinary stateful_syntax object",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the price model is order-0 magnitude; a context-conditional model is a later \
                     rung and would need the context to remain decoder-visible for the parse to \
                     stay separable",
                    "each iteration costs one exact assembly; cheap reparse plus one assembly per \
                     accepted step bounds the work",
                ]),
            ),
        ],
    )
}
