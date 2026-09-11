//! `court learned-speech` — the real-speech portfolio court (Report 3 campaign).
//!
//! Runs the **current** speech candidate portfolio over the frozen real +
//! Mode-C corpus (the sign-extended-i16 experimental domain, unchanged from
//! Seal K) and reports paired statistics against FLAC-5. The portfolio grows
//! one seal at a time; every receipt records the active candidate set and the
//! per-family standalone bytes, so each seal's contribution is attributable.
//!
//! Seal S1 adds the coefficient-free **fixed finite-difference** family
//! (`learned::fixed`, orders 0–4, block-size ladder) beside the wired baseline
//! (4-tap dense ridge, sparse ≤10 lags, 2-stage hierarchy).
//!
//! The court is exact: every candidate must close the intrinsic sample for
//! sample before it can be priced.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{
    RealCase, available, decoder_available, effectiveness_clips, load_cases, mode_c_clips,
    real_corpus_sha256,
};
use crate::learned::object::LearnedObject;
use crate::learned::train::TrainBudget;
use crate::learned::train::fixed::fit_fixed_diff_sweep;
use crate::learned::train::hierarchy::fit_hierarchy_object;
use crate::learned::train::linear::fit_linear_object_exp2;
use crate::learned::train::lpc::fit_lpc_object;
use crate::learned::train::sparse::fit_sparse_object;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
/// Frozen static-result hash (empty means "not yet frozen").
/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_SPEECH_SHA256: &str =
    "2ea09b963bc64a5adb8f95fdcde09d322c426cc7fd50f349e2db4c46a80cfc5d";

const CLIPS_PER_SPLIT: usize = 8;

/// The active candidate families (grown one seal at a time).
pub const ACTIVE_FAMILIES: &[&str] = &["dense4", "sparse10", "hier2", "fixed_diff", "lpc"];

fn bytes(o: &LearnedObject) -> Option<u64> {
    LearnedCost::of(o).ok().map(|c| c.complete_bytes)
}

/// Build the active portfolio for one case. Returns named exact candidates in the
/// Exp3 profile (every Exp2 candidate plus the Seal S4 residual codecs).
fn portfolio(case: &RealCase, budget: &TrainBudget) -> Result<Vec<(&'static str, LearnedObject)>> {
    let frames = case.frames();
    let rate = case.rate();
    let mut raw: Vec<(&'static str, LearnedObject)> = Vec::new();

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
        raw.push(("dense4", o));
    }
    if let Ok((o, _)) = fit_sparse_object(&case.samples, frames, rate, 10, None, budget) {
        raw.push(("sparse10", o));
    }
    if let Ok((o, _)) = fit_hierarchy_object(&case.samples, frames, rate, 2, budget) {
        raw.push(("hier2", o));
    }
    // Seal S1: fixed finite differences over a frozen block-size ladder.
    let ladder = [None, Some(4096u32), Some(2048), Some(1024), Some(512)];
    if let Ok((o, _)) = fit_fixed_diff_sweep(&case.samples, frames, rate, 4, &ladder, budget) {
        raw.push(("fixed_diff", o));
    }
    // Seal S2/S3: dense per-block all-pole LPC (Tukey 0.5 + autocorrelation +
    // Levinson-Durbin) with precision/shift/quantiser search, orders 1..=8.
    let mut best_lpc: Option<LearnedObject> = None;
    let mut best_lpc_bytes = u64::MAX;
    for block in [4096u32, 2048, 1024] {
        if let Ok((o, _)) = fit_lpc_object(&case.samples, frames, rate, block, 8, budget)
            && let Some(b) = bytes(&o)
            && b < best_lpc_bytes
        {
            best_lpc_bytes = b;
            best_lpc = Some(o);
        }
    }
    if let Some(o) = best_lpc {
        raw.push(("lpc", o));
    }
    // Re-encode every Exp2 candidate under Exp3 so the Seal S4 residual codecs
    // (general Golomb, centered Golomb) are selectable.
    let mut out = Vec::with_capacity(raw.len());
    for (name, o) in raw {
        let upgraded = LearnedObject::from_intrinsic_exp3(
            o.model,
            o.channels,
            o.frames,
            o.sample_rate_hz,
            o.dependencies,
            &case.samples,
        )?;
        out.push((name, upgraded));
    }
    Ok(out)
}

struct SplitReport {
    rows: Vec<serde_json::Value>,
    portfolio: Vec<u64>,
    flac: Vec<u64>,
    baseline: Vec<u64>,
}

fn run_split(cases: &[RealCase], budget: &TrainBudget) -> Result<(SplitReport, bool)> {
    let mut report = SplitReport {
        rows: Vec::new(),
        portfolio: Vec::new(),
        flac: Vec::new(),
        baseline: Vec::new(),
    };
    let mut all_exact = true;
    for case in cases {
        let frames = case.frames();
        let cands = portfolio(case, budget)?;
        let mut family_bytes = serde_json::Map::new();
        let mut family_detail = serde_json::Map::new();
        let mut best: Option<(String, u64)> = None;
        let mut baseline: Option<u64> = None;
        for (name, o) in &cands {
            all_exact &= o.verify(&case.samples);
            let cost = LearnedCost::of(o)?;
            let b = cost.complete_bytes;
            let residual = o.residual()?;
            let mean_abs = if residual.is_empty() {
                0.0
            } else {
                residual
                    .iter()
                    .map(|&v| f64::from(v.unsigned_abs()))
                    .sum::<f64>()
                    / residual.len() as f64
            };
            family_bytes.insert((*name).to_string(), serde_json::json!(b));
            family_detail.insert(
                (*name).to_string(),
                serde_json::json!({
                    "bytes": b,
                    "model_bytes": cost.model_bytes,
                    "residual_bytes": cost.residual_bytes,
                    "residual_codec": o.residual_codec.name(),
                    "residual_mean_abs": mean_abs,
                }),
            );
            if matches!(*name, "dense4" | "sparse10" | "hier2") {
                baseline = Some(baseline.map_or(b, |x| x.min(b)));
            }
            if best.as_ref().is_none_or(|(_, bb)| b < *bb) {
                best = Some(((*name).to_string(), b));
            }
        }
        let (winner, best_bytes) =
            best.ok_or_else(|| crate::error::Error::internal("empty speech portfolio"))?;
        let flac = common::flac5(&case.samples, 1, case.rate()).unwrap_or(u64::MAX);
        let baseline_bytes = baseline.unwrap_or(u64::MAX);
        report.portfolio.push(best_bytes);
        report.flac.push(flac);
        report.baseline.push(baseline_bytes);
        report.rows.push(serde_json::json!({
            "id": case.clip.id,
            "split": case.clip.split,
            "speaker": case.clip.speaker,
            "frames": frames,
            "winner": winner,
            "portfolio_bytes": best_bytes,
            "baseline_bytes": baseline_bytes,
            "flac_bytes": flac,
            "ratio_flac_over_portfolio": if best_bytes == 0 { 0.0 } else { flac as f64 / best_bytes as f64 },
            "families": serde_json::Value::Object(family_bytes),
            "family_detail": serde_json::Value::Object(family_detail),
        }));
    }
    Ok((report, all_exact))
}

fn summarize(label: &str, rep: &SplitReport) -> serde_json::Value {
    let n = rep.portfolio.len();
    let (mut wins, mut losses, mut bwins, mut blossses) = (0u64, 0u64, 0u64, 0u64);
    let mut diffs = Vec::with_capacity(n);
    let mut bdiffs = Vec::with_capacity(n);
    let (mut tp, mut tf, mut tb) = (0u64, 0u64, 0u64);
    for i in 0..n {
        let p = rep.portfolio[i];
        let f = rep.flac[i];
        let b = rep.baseline[i];
        tp = tp.saturating_add(p);
        tf = tf.saturating_add(f);
        tb = tb.saturating_add(b);
        if p < f {
            wins += 1;
        } else if p > f {
            losses += 1;
        }
        if p < b {
            bwins += 1;
        } else if p > b {
            blossses += 1;
        }
        diffs.push(f as i64 - p as i64);
        bdiffs.push(b as i64 - p as i64);
    }
    let (wp, wm, ppm) = common::wilcoxon_exact(&diffs);
    let (bp, bm, bppm) = common::wilcoxon_exact(&bdiffs);
    let (lo, hi) = common::bootstrap_median_ppm(&rep.flac, &rep.portfolio, 0x5EED_3001, 4000);
    serde_json::json!({
        "split": label,
        "clips": n,
        "total_portfolio_bytes": tp,
        "total_flac_bytes": tf,
        "total_baseline_bytes": tb,
        "wins_vs_flac": wins,
        "losses_vs_flac": losses,
        "wilcoxon_flac": {"w_plus": wp, "w_minus": wm, "p_two_sided_ppm": ppm},
        "wins_vs_baseline": bwins,
        "losses_vs_baseline": blossses,
        "wilcoxon_baseline": {"w_plus": bp, "w_minus": bm, "p_two_sided_ppm": bppm},
        "bootstrap_median_flac_over_portfolio_ppm": {"lo": lo, "hi": hi},
    })
}

/// Run the court; writes `receipts/learned-speech/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.speech.v1");
    common::push_label(&mut projection, &real_corpus_sha256());
    for f in ACTIVE_FAMILIES {
        common::push_label(&mut projection, f);
    }

    if !available() || !decoder_available() {
        return common::finish_exp2(
            "learned-speech",
            receipts_root,
            LEARNED_SPEECH_SHA256,
            &projection,
            Verdict::Inconclusive,
            "real corpus audio or the external `flac` decoder is absent".to_string(),
            vec![(
                "limitations",
                serde_json::json!([
                    "requires the uncommitted LibriSpeech audio bulk under corpus/real/ and the \
                     external `flac` decoder used only to read the frozen compressed clips"
                ]),
            )],
        );
    }

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let scratch = std::path::PathBuf::from("target/real-corpus/scratch");
    let effectiveness = load_cases(&effectiveness_clips(), CLIPS_PER_SPLIT, &scratch)?;
    let mode_c = load_cases(&mode_c_clips(), CLIPS_PER_SPLIT, &scratch)?;
    let (eff_report, eff_exact) = run_split(&effectiveness, &budget)?;
    let (mc_report, mc_exact) = run_split(&mode_c, &budget)?;
    let all_exact = eff_exact && mc_exact;
    let eff = summarize("effectiveness (dev-clean)", &eff_report);
    let mc = summarize("mode-c held-out (test-clean)", &mc_report);

    for row in eff_report.rows.iter().chain(mc_report.rows.iter()) {
        common::push_label(&mut projection, row["id"].as_str().unwrap_or(""));
        common::push_u64(
            &mut projection,
            row["portfolio_bytes"].as_u64().unwrap_or(u64::MAX),
        );
        common::push_u64(
            &mut projection,
            row["flac_bytes"].as_u64().unwrap_or(u64::MAX),
        );
        common::push_u64(
            &mut projection,
            row["baseline_bytes"].as_u64().unwrap_or(u64::MAX),
        );
    }

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp3(
        "learned-speech",
        receipts_root,
        LEARNED_SPEECH_SHA256,
        &projection,
        verdict,
        format!(
            "real-speech portfolio (families: {}) over {} effectiveness + {} held-out Mode-C \
             clips, paired against FLAC-5 with exact Wilcoxon signed-rank and a deterministic \
             bootstrap median CI",
            ACTIVE_FAMILIES.join(", "),
            effectiveness.len(),
            mode_c.len()
        ),
        vec![
            ("effectiveness", eff),
            ("mode_c", mc),
            ("effectiveness_clips", serde_json::json!(eff_report.rows)),
            ("mode_c_clips", serde_json::json!(mc_report.rows)),
            (
                "method",
                serde_json::json!({
                    "active_families": ACTIVE_FAMILIES,
                    "profile": crate::learned::profile::LEARNED_EXP3_PROFILE,
                    "sample_domain": "sign-extended i16 in i32 (the Seal-K experimental domain)",
                    "baseline": "the wired three-family Exp2 real-speech portfolio",
                    "attribution": "per-family standalone bytes are reported for every clip",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "object-specific fitting is representation selection, never a generalization \
                     claim",
                    "the portfolio grows one seal at a time; this receipt names its active set"
                ]),
            ),
        ],
    )
}
