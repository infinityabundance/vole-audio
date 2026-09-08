//! `court dsfb-entropy` — DSFB zero-authority search governance over the
//! frozen entropy candidate universe (H.2.26–H.2.29, owner
//! `docs/DSFB_SEARCH.md`).
//!
//! Over a deterministic subset of the frozen corpus, the court measures the
//! three comparable encoder-side strategies — `exhaustive`,
//! `fixed-heuristic`, and (feature `dsfb`) `dsfb-guided` — operating over
//! ONE identical candidate set: symbolization {identity, lane4_plain,
//! lane4_zigzag, delta_lane4} x page_frames {256, 512, 1024} x model_mode
//! {inline, shared} = 24 candidates per fixture. For every strategy the
//! court records `N` (candidates evaluated) and `J` (best complete
//! representation bytes found), plus the raw-sample and canonical-U1
//! baselines.
//!
//! Checks:
//!
//! * every evaluated candidate reconstructs the canonical samples exactly;
//! * the candidate set is identical across strategies (all evaluated cells
//!   are members of the frozen universe);
//! * `J` is the exact minimum over the strategy's own evaluated cells;
//! * the RAW/literal fallback anchor is always evaluated by the bounded
//!   strategies;
//! * the guided strategy is deterministic (two identical runs);
//! * `J_dsfb == J_exhaustive` is asserted where possible and measured regret
//!   is recorded otherwise; `N_dsfb < N_exhaustive` is asserted where it
//!   holds and recorded honestly when it does not. If the fixed heuristic
//!   beats DSFB on a fixture, that is recorded too.
//!
//! Without `--features dsfb` the guided strategy is not compiled: the court
//! still measures exhaustive vs fixed-heuristic and records an honest
//! INCONCLUSIVE receipt with the limitation (exit 0 — the existing court
//! convention).

use crate::entropy::corpus;
use crate::entropy::search::{
    BUDGET, RAW_LITERAL_ANCHOR, StrategyOutcome, UNIVERSE_SIZE, literal_descriptor, run_exhaustive,
    run_fixed_heuristic,
};
use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

/// Deterministic corpus subset (frozen; structural through incompressible).
const FIXTURES: &[&str] = &[
    "dc",
    "single-sine",
    "harmonic-tone",
    "am-signal",
    "transient-heavy",
    "white-noise",
    "random-control",
];

/// Strategy labels in table order.
const STRATEGIES: &[&str] = &["exhaustive", "fixed-heuristic", "dsfb-guided"];

fn fixture_row_json(
    fx: &corpus::Fixture,
    strategy: &str,
    outcome: &StrategyOutcome,
    j_exhaustive: u64,
) -> serde_json::Value {
    let j = outcome.j().expect("at least one evaluated cell");
    let best = outcome.best().expect("best cell");
    let u1 = 46 + 8 + fx.frames() as u64 * u64::from(fx.channels) * 4;
    let raw = fx.samples.len() as u64 * 4;
    serde_json::json!({
        "fixture": fx.name,
        "kind": fx.kind,
        "channels": fx.channels,
        "frames": fx.frames(),
        "strategy": strategy,
        "n_candidates": outcome.n(),
        "j_complete_bytes": j,
        "best_candidate": best.candidate.name(),
        "best_raw_fallback_fraction": best.raw_fallback_fraction,
        "regret_bytes_vs_exhaustive": j.saturating_sub(j_exhaustive),
        "ratio_vs_exhaustive": j as f64 / j_exhaustive as f64,
        "raw_sample_bytes": raw,
        "canonical_u1_bytes": u1,
        "ratio_vs_u1_literal": j as f64 / u1 as f64,
    })
}

/// Run the court; writes an immutable receipt under `receipts/dsfb-entropy/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let fail = |why: &str| -> crate::error::Result<Verdict> {
        let mut b = ReceiptBuilder::new("dsfb-entropy");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("dsfb-entropy failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court dsfb-entropy: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let universe = crate::entropy::search::frozen_universe();
    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut summaries: Vec<serde_json::Value> = Vec::new();
    // Mutated only under the `dsfb` feature (cfg-gated above).
    #[cfg_attr(not(feature = "dsfb"), allow(unused_mut))]
    let mut guided_traces: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut limitations: Vec<String> = Vec::new();

    let mut totals: BTreeMap<&str, usize> = BTreeMap::new();
    #[cfg_attr(not(feature = "dsfb"), allow(unused_mut))]
    let mut dsfb_equal_count = 0usize;
    #[cfg_attr(not(feature = "dsfb"), allow(unused_mut))]
    let mut dsfb_budget_runs = 0usize;
    let mut fixtures_done = 0usize;

    for name in FIXTURES {
        let fx = corpus::named(name).expect("frozen fixture present");
        let descriptor = literal_descriptor(fx.channels, fx.frames() as u64)?;
        let canonical_u1 = 46 + 8 + fx.frames() as u64 * u64::from(fx.channels) * 4;

        // Exhaustive: every candidate of the universe, exactly once.
        let ex = run_exhaustive(&descriptor, &fx.samples, canonical_u1)?;
        if ex.n() != UNIVERSE_SIZE {
            return fail(&format!("{}: exhaustive N != universe size", fx.name));
        }
        let j_exhaustive = ex.j().expect("cells");

        // Fixed heuristic: bounded budget, RAW/literal anchor guaranteed.
        let fixed = run_fixed_heuristic(&descriptor, &fx.samples, canonical_u1)?;
        if fixed.n() != BUDGET {
            return fail(&format!("{}: fixed-heuristic N != budget", fx.name));
        }
        if !fixed
            .evaluated
            .iter()
            .any(|c| c.candidate == RAW_LITERAL_ANCHOR)
        {
            return fail(&format!(
                "{}: fixed heuristic missed the RAW/literal anchor",
                fx.name
            ));
        }
        let j_fixed = fixed.j().expect("cells");

        // Every strategy's J must equal the min over its own evaluated cells
        // and every evaluated cell must be a member of the one universe.
        for (label, outcome) in [("exhaustive", &ex), ("fixed-heuristic", &fixed)] {
            let measured_min = outcome
                .evaluated
                .iter()
                .map(|c| c.complete_bytes)
                .min()
                .expect("cells");
            if outcome.j() != Some(measured_min) {
                return fail(&format!(
                    "{}: {label} J != min of its evaluated cells",
                    fx.name
                ));
            }
            for cell in &outcome.evaluated {
                if !universe.contains(&cell.candidate) {
                    return fail(&format!(
                        "{}: {label} evaluated outside the frozen universe",
                        fx.name
                    ));
                }
            }
            totals.insert(label, totals.get(label).copied().unwrap_or(0) + outcome.n());
        }

        // DSFB-guided (feature dsfb): measured twice for determinism. The
        // outer type stays feature-free (stop reason as a stable label).
        let guided_available: Option<(crate::entropy::search::StrategyOutcome, &'static str)> = {
            #[cfg(feature = "dsfb")]
            {
                let (g1, t1) = crate::entropy::search::dsfb::run_dsfb_guided(
                    &descriptor,
                    &fx.samples,
                    canonical_u1,
                )?;
                let (g2, t2) = crate::entropy::search::dsfb::run_dsfb_guided(
                    &descriptor,
                    &fx.samples,
                    canonical_u1,
                )?;
                // Determinism: identical N, J, stop reason, evaluation order
                // and measured costs across two independent runs.
                if g1.n() != g2.n() || g1.j() != g2.j() || t1.stop_reason != t2.stop_reason {
                    return fail(&format!(
                        "{}: dsfb-guided non-deterministic (N/J/stop differ)",
                        fx.name
                    ));
                }
                for (a, b) in g1.evaluated.iter().zip(g2.evaluated.iter()) {
                    if a.candidate != b.candidate || a.complete_bytes != b.complete_bytes {
                        return fail(&format!(
                            "{}: dsfb-guided non-deterministic (order/cost differ)",
                            fx.name
                        ));
                    }
                }
                // Candidate-set consistency + budget bound.
                if g1.n() > BUDGET || g1.n() >= ex.n() {
                    return fail(&format!(
                        "{}: dsfb-guided N {} not below exhaustive {} with budget {}",
                        fx.name,
                        g1.n(),
                        ex.n(),
                        BUDGET
                    ));
                }
                for cell in &g1.evaluated {
                    if !universe.contains(&cell.candidate) {
                        return fail(&format!(
                            "{}: dsfb-guided evaluated outside the frozen universe",
                            fx.name
                        ));
                    }
                }
                if !g1
                    .evaluated
                    .iter()
                    .any(|c| c.candidate == RAW_LITERAL_ANCHOR)
                {
                    return fail(&format!(
                        "{}: guided missed the RAW/literal anchor",
                        fx.name
                    ));
                }
                let trace_cells: Vec<serde_json::Value> = t1
                    .steps
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "step": s.cell.candidate.name(),
                            "complete_bytes": s.cell.complete_bytes,
                            "best_before": s.best_before,
                            "winner_quality": s.quality,
                            "regime": s.regime.label(),
                            "dsfb_phi": s.dsfb_state.phi,
                            "dsfb_omega": s.dsfb_state.omega,
                            "dsfb_alpha": s.dsfb_state.alpha,
                            "dsfb_raw_regime": s.dsfb_raw_regime.label(),
                        })
                    })
                    .collect();
                guided_traces.insert(
                    fx.name.to_string(),
                    serde_json::json!({
                        "stop_reason": t1.stop_reason.label(),
                        "steps": trace_cells,
                    }),
                );
                cells.push(fixture_row_json(&fx, "dsfb-guided", &g1, j_exhaustive));
                totals.insert(
                    "dsfb-guided",
                    totals.get("dsfb-guided").copied().unwrap_or(0) + g1.n(),
                );
                dsfb_budget_runs += 1;
                if g1.j() == ex.j() {
                    dsfb_equal_count += 1;
                }
                Some((g1, t1.stop_reason.label()))
            }
            #[cfg(not(feature = "dsfb"))]
            {
                None
            }
        };

        cells.push(fixture_row_json(&fx, "exhaustive", &ex, j_exhaustive));
        cells.push(fixture_row_json(
            &fx,
            "fixed-heuristic",
            &fixed,
            j_exhaustive,
        ));

        // Per-fixture summary: N/J per strategy + the DSFB-vs-oracle and
        // DSFB-vs-heuristic outcome, recorded honestly either way.
        let summary = match &guided_available {
            Some((guided, stop_reason)) => {
                let j_dsfb = guided.j().expect("cells");
                let j_eq = j_dsfb == j_exhaustive;
                let n_lt = guided.n() < ex.n();
                let vs_fixed = if j_dsfb < j_fixed || (j_dsfb == j_fixed && guided.n() <= fixed.n())
                {
                    "dsfb-won-or-tied"
                } else {
                    "fixed-heuristic-better"
                };
                let best = guided.best().expect("cells");
                serde_json::json!({
                    "fixture": fx.name,
                    "kind": fx.kind,
                    "j_exhaustive": j_exhaustive,
                    "n_exhaustive": ex.n(),
                    "j_fixed_heuristic": j_fixed,
                    "n_fixed_heuristic": fixed.n(),
                    "j_dsfb": j_dsfb,
                    "n_dsfb": guided.n(),
                    "dsfb_stop_reason": stop_reason,
                    "j_dsfb_eq_j_exhaustive": j_eq,
                    "regret_bytes_if_any": j_dsfb.saturating_sub(j_exhaustive),
                    "n_dsfb_lt_n_exhaustive": n_lt,
                    "dsfb_vs_fixed_heuristic": vs_fixed,
                    "dsfb_best_candidate": best.candidate.name(),
                    "dsfb_best_raw_fallback_fraction": best.raw_fallback_fraction,
                })
            }
            None => serde_json::json!({
                "fixture": fx.name,
                "kind": fx.kind,
                "j_exhaustive": j_exhaustive,
                "n_exhaustive": ex.n(),
                "j_fixed_heuristic": j_fixed,
                "n_fixed_heuristic": fixed.n(),
                "dsfb": "NOT_AVAILABLE (built without --features dsfb)",
            }),
        };
        summaries.push(summary);
        fixtures_done += 1;
    }

    let guided_compiled = cfg!(feature = "dsfb");
    if !guided_compiled {
        limitations.push(
            "built without --features dsfb: the dsfb-guided strategy is not compiled; \
             exhaustive vs fixed-heuristic measured only"
                .to_string(),
        );
    }

    // Whole-corpus work totals (N across fixtures, per strategy).
    let n_total: usize = totals.values().sum();
    let n_ex_total = totals.get("exhaustive").copied().unwrap_or(0);
    let n_fx_total = totals.get("fixed-heuristic").copied().unwrap_or(0);
    let n_dsfb_total = totals.get("dsfb-guided").copied().unwrap_or(0);
    let work_reduction_dsfb_vs_ex = if n_ex_total > 0 {
        (1.0 - n_dsfb_total as f64 / n_ex_total as f64) * 100.0
    } else {
        0.0
    };

    let verdict = if guided_compiled {
        Verdict::Supported
    } else {
        Verdict::Inconclusive
    };
    let mut builder = ReceiptBuilder::new("dsfb-entropy");
    let detail = if guided_compiled {
        format!(
            "strategies over one 24-candidate universe x {fixtures_done} fixtures; \
             N_exhaustive total {n_ex_total}, N_fixed total {n_fx_total}, \
             N_dsfb total {n_dsfb_total} (-{work_reduction_dsfb_vs_ex:.1}% vs exhaustive); \
             J_dsfb == J_exhaustive on {dsfb_equal_count}/{dsfb_budget_runs} measured fixtures \
             (regret recorded where not)",
        )
    } else {
        format!(
            "exhaustive vs fixed-heuristic measured over {fixtures_done} fixtures \
             (N_exhaustive total {n_ex_total}, N_fixed total {n_fx_total}); \
             dsfb-guided NOT_AVAILABLE (feature dsfb off)",
        )
    };
    builder
        .result(verdict)
        .result_detail(detail)
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1 + vole.entropy.p1/p1/v1".into()),
            backend: Some("scalar".into()),
            content_kind: Some("entropy-corpus-v1 search-universe".into()),
            ..Default::default()
        })
        .extra("universe_size", serde_json::json!(UNIVERSE_SIZE))
        .extra("budget", serde_json::json!(BUDGET))
        .extra("strategies", serde_json::json!(STRATEGIES))
        .extra("cells", serde_json::Value::Array(cells))
        .extra("per_fixture_summary", serde_json::Value::Array(summaries))
        .extra("dsfb_guided_trace", serde_json::to_value(&guided_traces)?)
        .extra(
            "totals",
            serde_json::json!({
                "fixtures": fixtures_done,
                "n_exhaustive_total": n_ex_total,
                "n_fixed_total": n_fx_total,
                "n_dsfb_total": n_dsfb_total,
                "evaluated_total_all_strategies": n_total,
                "work_reduction_dsfb_vs_exhaustive_pct": work_reduction_dsfb_vs_ex,
            }),
        );
    for l in &limitations {
        builder.limitation(l.clone());
    }
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court dsfb-entropy: {verdict}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::search::{RAW_LITERAL_ANCHOR, frozen_universe};

    #[test]
    fn fixture_subset_is_frozen_and_present() {
        assert_eq!(
            FIXTURES,
            &[
                "dc",
                "single-sine",
                "harmonic-tone",
                "am-signal",
                "transient-heavy",
                "white-noise",
                "random-control",
            ]
        );
        for name in FIXTURES {
            let fx = corpus::named(name).expect("fixture in corpus");
            assert_eq!(fx.frames(), 16384);
        }
        assert_eq!(
            STRATEGIES,
            &["exhaustive", "fixed-heuristic", "dsfb-guided"]
        );
    }

    #[test]
    fn court_universe_holds_anchor_and_all_fixture_candidates_encode() {
        // A light encoding smoke over the smallest fixture at every
        // candidate: everything the court evaluates must encode, reconstruct
        // exactly, and the raw/literal anchor must be a universe member.
        let u = frozen_universe();
        assert!(u.contains(&RAW_LITERAL_ANCHOR));
        let fx = corpus::named("dc").unwrap();
        let descriptor = literal_descriptor(fx.channels, fx.frames() as u64).unwrap();
        let canonical_u1 = 46 + 8 + fx.frames() as u64 * u64::from(fx.channels) * 4;
        for c in &u {
            let cell = crate::entropy::search::evaluate_literal(
                &descriptor,
                &fx.samples,
                canonical_u1,
                *c,
            )
            .unwrap_or_else(|e| panic!("{} encode failed: {e}", c.name()));
            assert_eq!(
                cell.cost.raw_sample_bytes,
                fx.samples.len() as u64 * 4,
                "raw baseline per cell"
            );
        }
    }

    #[test]
    fn negative_control_pages_fall_back_to_raw() {
        // Negative controls must fall back toward RAW pages (H.2.31): at the
        // exhaustive best cell of white noise, essentially every page is RAW.
        let fx = corpus::named("white-noise").unwrap();
        let descriptor = literal_descriptor(fx.channels, fx.frames() as u64).unwrap();
        let canonical_u1 = 46 + 8 + fx.frames() as u64 * u64::from(fx.channels) * 4;
        let ex = run_exhaustive(&descriptor, &fx.samples, canonical_u1).unwrap();
        let best = ex.best().unwrap();
        assert!(
            best.raw_fallback_fraction >= 0.99,
            "white-noise exhaustive best must be RAW-page dominated, got {}",
            best.raw_fallback_fraction
        );
    }
}
