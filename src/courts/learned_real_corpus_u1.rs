//! `court learned-real-corpus-u1` — U1-domain real-speech replay (Seal S0).
//!
//! The Seal-K court (`learned-exp2-real-corpus`) runs on a **sign-extended-i16
//! experimental domain** (`i32::from(v)`), which is *not* the frozen U1 s16
//! ingest (`i32 = i16 << 16`). Seal K is preserved exactly; this court is a
//! separate, explicitly-versioned replay that uses the true U1 mapping with
//! **new identities and hashes** (`corpus/real_manifest_u1.json`).
//!
//! It runs the same wired three-family Exp2 portfolio against FLAC-5 over the
//! U1-mapped samples, so the two sample domains can be compared without
//! rewriting any prior evidence. The U1 mapping gives every sample 16 low zero
//! bits, which libFLAC exploits through its wasted-bits mechanism; this court
//! measures exactly that effect rather than assuming it.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus_real::{
    RealCase, SampleDomain, available, decoder_available, load_cases_domain,
    u1_effectiveness_clips, u1_mode_c_clips, u1_real_corpus_sha256,
};
use crate::learned::object::LearnedObject;
use crate::learned::train::hierarchy::fit_hierarchy_object;
use crate::learned::train::linear::fit_linear_object_exp2;
use crate::learned::train::sparse::fit_sparse_object;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_REAL_CORPUS_U1_SHA256: &str =
    "d07cb3e7858efd1a1ab13c0adbd6610e7613e84fd3617b79c0d0125a48fdc00d";

const CLIPS_PER_SPLIT: usize = 8;

fn bytes(o: &LearnedObject) -> Option<u64> {
    crate::learned::accounting::LearnedCost::of(o)
        .ok()
        .map(|c| c.complete_bytes)
}

struct SplitReport {
    rows: Vec<serde_json::Value>,
    portfolio: Vec<u64>,
    flac: Vec<u64>,
}

fn run_split(
    cases: &[RealCase],
    budget: &crate::learned::train::TrainBudget,
) -> Result<(SplitReport, bool)> {
    let mut report = SplitReport {
        rows: Vec::new(),
        portfolio: Vec::new(),
        flac: Vec::new(),
    };
    let mut all_exact = true;
    for case in cases {
        let frames = case.frames();
        let ch = case.channels();
        let rate = case.rate();
        let linear = fit_linear_object_exp2(
            &case.samples,
            ch,
            frames,
            rate,
            4,
            None,
            frames as usize,
            budget,
        )
        .ok()
        .map(|(o, _)| o);
        let sparse = fit_sparse_object(&case.samples, frames, rate, 10, None, budget)
            .ok()
            .map(|(o, _)| o);
        let hierarchy = fit_hierarchy_object(&case.samples, frames, rate, 2, budget)
            .ok()
            .map(|(o, _)| o);
        for o in [linear.as_ref(), sparse.as_ref(), hierarchy.as_ref()]
            .into_iter()
            .flatten()
        {
            all_exact &= o.verify(&case.samples);
        }
        let best = [
            linear.as_ref().and_then(bytes),
            sparse.as_ref().and_then(bytes),
            hierarchy.as_ref().and_then(bytes),
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(u64::MAX);
        let flac = common::flac5(&case.samples, ch, rate).unwrap_or(u64::MAX);
        report.portfolio.push(best);
        report.flac.push(flac);
        report.rows.push(serde_json::json!({
            "id": case.clip.id,
            "split": case.clip.split,
            "speaker": case.clip.speaker,
            "frames": frames,
            "domain": "u1_s16_left_shift_16",
            "portfolio_bytes": if best == u64::MAX { None } else { Some(best) },
            "sparse_bytes": sparse.as_ref().and_then(bytes),
            "hierarchy_bytes": hierarchy.as_ref().and_then(bytes),
            "linear_bytes": linear.as_ref().and_then(bytes),
            "flac_bytes": if flac == u64::MAX { None } else { Some(flac) },
        }));
    }
    Ok((report, all_exact))
}

fn summarize(label: &str, rep: &SplitReport) -> serde_json::Value {
    let n = rep.portfolio.len();
    let mut wins = 0u64;
    let mut losses = 0u64;
    let mut diffs = Vec::with_capacity(n);
    let mut total_p = 0u64;
    let mut total_f = 0u64;
    for i in 0..n {
        let p = rep.portfolio[i];
        let f = rep.flac[i];
        total_p = total_p.saturating_add(p);
        total_f = total_f.saturating_add(f);
        if p < f {
            wins += 1;
        } else if p > f {
            losses += 1;
        }
        diffs.push(f as i64 - p as i64);
    }
    let (wp, wm, ppm) = common::wilcoxon_exact(&diffs);
    let (lo, hi) = common::bootstrap_median_ppm(&rep.flac, &rep.portfolio, 0x5EED_9001, 4000);
    serde_json::json!({
        "split": label,
        "clips": n,
        "total_portfolio_bytes": total_p,
        "total_flac_bytes": total_f,
        "wins_vs_flac": wins,
        "losses_vs_flac": losses,
        "wilcoxon_flac": {"w_plus": wp, "w_minus": wm, "p_two_sided_ppm": ppm},
        "bootstrap_median_flac_over_portfolio_ppm": {"lo": lo, "hi": hi},
    })
}

/// Run the court; writes `receipts/learned-real-corpus-u1/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.real-corpus-u1.v1");
    common::push_label(&mut projection, &u1_real_corpus_sha256());

    if !available() || !decoder_available() {
        return common::finish_exp2(
            "learned-real-corpus-u1",
            receipts_root,
            LEARNED_REAL_CORPUS_U1_SHA256,
            &projection,
            Verdict::Inconclusive,
            "the external `flac` decoder is absent; the U1-domain replay cannot load the corpus"
                .to_string(),
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
    let effectiveness = load_cases_domain(
        &u1_effectiveness_clips(),
        CLIPS_PER_SPLIT,
        &scratch,
        SampleDomain::U1S16,
    )?;
    let mode_c = load_cases_domain(
        &u1_mode_c_clips(),
        CLIPS_PER_SPLIT,
        &scratch,
        SampleDomain::U1S16,
    )?;
    let (eff_report, eff_exact) = run_split(&effectiveness, &budget)?;
    let (mc_report, mc_exact) = run_split(&mode_c, &budget)?;
    let all_exact = eff_exact && mc_exact;
    let eff = summarize("effectiveness (dev-clean, U1 domain)", &eff_report);
    let mc = summarize("mode-c held-out (test-clean, U1 domain)", &mc_report);

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
    }
    common::push_u64(&mut projection, eff["wins_vs_flac"].as_u64().unwrap_or(0));
    common::push_u64(&mut projection, mc["wins_vs_flac"].as_u64().unwrap_or(0));

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp2(
        "learned-real-corpus-u1",
        receipts_root,
        LEARNED_REAL_CORPUS_U1_SHA256,
        &projection,
        verdict,
        format!(
            "U1-domain real-speech replay: {} effectiveness + {} held-out Mode-C clips mapped \
             through the frozen U1 s16 ingest (i32 = i16 << 16), paired against FLAC-5 with exact \
             Wilcoxon signed-rank",
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
                    "domain": "frozen U1 s16 ingest: i32 = (i16 as i32) << 16",
                    "identities": "corpus/real_manifest_u1.json (new identities + hashes; the \
                                   Seal-K manifest is untouched)",
                    "portfolio": "the wired three-family Exp2 real-speech portfolio",
                    "note": "the U1 mapping gives every sample 16 low zero bits; libFLAC exploits \
                             them via wasted bits, so this court measures the domain effect \
                             directly",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "this is a separate replay identity; it does not mutate or supersede Seal K",
                    "object-specific fitting is representation selection, never a generalization \
                     claim"
                ]),
            ),
        ],
    )
}
