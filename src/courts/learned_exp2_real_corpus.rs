//! `court learned-exp2-real-corpus` — real + held-out Mode-C corpus (Seal K).
//!
//! Runs the Exp2 portfolio over **real speech** (LibriSpeech, CC BY 4.0) and
//! reports paired statistics rather than aggregate bytes alone:
//!
//! * per-clip complete bytes for FLAC, the Exp1 dense-linear baseline, and the
//!   Exp2 portfolio minimum;
//! * paired win/equal/loss counts against FLAC and against Exp1;
//! * the **exact** two-sided Wilcoxon signed-rank p-value over the paired
//!   per-clip differences;
//! * a deterministic bootstrap confidence interval for the median paired ratio.
//!
//! The `test-clean` split (Mode C) is **held out** from architecture tuning and
//! is reported separately. Object-specific fitting is never described as
//! generalization.
//!
//! When the (uncommitted) audio bulk is absent the court reports
//! `INCONCLUSIVE` with an explicit limitation rather than fabricating a result.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus_real::{
    RealCase, available, decoder_available, effectiveness_clips, load_cases, mode_c_clips,
    real_corpus_sha256,
};
use crate::learned::object::LearnedObject;
use crate::learned::train::hierarchy::fit_hierarchy_object;
use crate::learned::train::linear::{fit_linear_object, fit_linear_object_exp2};
use crate::learned::train::sparse::fit_sparse_object;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_EXP2_REAL_CORPUS_SHA256: &str =
    "f6a87cc5a9e53401a6f9cb4d66ba6e42993deab47ffa600826f07d16e2d92446";

/// Clips per split used by the court (bounded encoder budget).
const CLIPS_PER_SPLIT: usize = 8;

fn bytes(o: &LearnedObject) -> Option<u64> {
    crate::learned::accounting::LearnedCost::of(o)
        .ok()
        .map(|c| c.complete_bytes)
}

/// Exact two-sided Wilcoxon signed-rank over paired differences.
///
/// Ranks are average-free (deterministic by index) and zero differences are
/// dropped. The null distribution of the positive-rank sum is computed exactly
/// by dynamic programming over the subset sums of `1..=n`. Returns
/// `(w_plus, w_minus, p_two_sided_ppm)` where the p-value is in parts per
/// million (integer, deterministic).
fn wilcoxon_exact(diffs: &[i64]) -> (u64, u64, u64) {
    let mut items: Vec<(u64, bool)> = diffs
        .iter()
        .filter(|&&d| d != 0)
        .map(|&d| (d.unsigned_abs(), d > 0))
        .collect();
    items.sort_unstable_by_key(|&(m, _)| m);
    let n = items.len();
    if n == 0 {
        return (0, 0, 1_000_000);
    }
    // Mid-ranks for ties.
    let mut w_plus = 0u64;
    let mut w_minus = 0u64;
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j < n && items[j].0 == items[i].0 {
            j += 1;
        }
        // Ranks i+1..=j (1-based); mid-rank sum for this tie group.
        let rank_sum: u64 = ((i + 1) as u64..=(j as u64)).sum();
        let group = (j - i) as u64;
        for &(_, positive) in &items[i..j] {
            if positive {
                w_plus += rank_sum / group;
            } else {
                w_minus += rank_sum / group;
            }
        }
        i = j;
    }
    // Exact null distribution of W+ (subset sums of 1..=n), n <= 64 is huge;
    // the court only runs small paired samples (<= 16), so DP is cheap.
    let total: u64 = (n as u64) * (n as u64 + 1) / 2;
    let mut dp = vec![0u64; (total + 1) as usize];
    dp[0] = 1;
    for r in 1..=n as u64 {
        for s in (r..=total).rev() {
            let add = dp[(s - r) as usize];
            dp[s as usize] += add;
        }
    }
    let outcomes: u64 = 1u64 << n;
    let w = w_plus.min(w_minus);
    let tail: u64 = dp[..=(w as usize)].iter().sum();
    let p_ppm = ((2 * tail).min(outcomes) * 1_000_000 + outcomes / 2) / outcomes;
    (w_plus, w_minus, p_ppm)
}

/// Deterministic splitmix64.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic bootstrap CI (95%) for the median of paired ratios
/// `(num_i + 1) / (den_i + 1)` in parts per million.
fn bootstrap_median_ppm(nums: &[u64], dens: &[u64], seed: u64, rounds: u32) -> (u64, u64) {
    if nums.is_empty() {
        return (0, 0);
    }
    let n = nums.len();
    let mut state = seed | 1;
    let mut medians: Vec<u64> = Vec::with_capacity(rounds as usize);
    let mut sample: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..rounds {
        sample.clear();
        for _ in 0..n {
            let idx = (splitmix64(&mut state) % n as u64) as usize;
            let ratio = (nums[idx] + 1)
                .saturating_mul(1_000_000)
                .checked_div(dens[idx] + 1)
                .unwrap_or(u64::MAX);
            sample.push(ratio);
        }
        sample.sort_unstable();
        medians.push(sample[n / 2]);
    }
    medians.sort_unstable();
    let lo = medians[(rounds as f64 * 0.025) as usize];
    let hi = medians[((rounds as f64 * 0.975) as usize).min(rounds as usize - 1)];
    (lo, hi)
}

struct SplitReport {
    rows: Vec<serde_json::Value>,
    exp2: Vec<u64>,
    flac: Vec<u64>,
    exp1: Vec<u64>,
}

fn run_split(
    cases: &[RealCase],
    budget: &crate::learned::train::TrainBudget,
) -> Result<(SplitReport, bool)> {
    let mut report = SplitReport {
        rows: Vec::new(),
        exp2: Vec::new(),
        flac: Vec::new(),
        exp1: Vec::new(),
    };
    let mut all_exact = true;
    for case in cases {
        let frames = case.frames();
        let ch = case.channels();
        // Exp1 dense linear baseline.
        let exp1 = fit_linear_object(
            &case.samples,
            ch,
            frames,
            case.rate(),
            4,
            None,
            frames as usize,
            budget,
        )
        .ok()
        .map(|(o, _)| o);
        all_exact &= exp1.as_ref().is_none_or(|o| o.verify(&case.samples));
        let exp1_bytes = exp1.as_ref().and_then(bytes);
        // Exp2 portfolio: dense linear, sparse, 2-stage hierarchy.
        let exp2_linear = fit_linear_object_exp2(
            &case.samples,
            ch,
            frames,
            case.rate(),
            4,
            None,
            frames as usize,
            budget,
        )
        .ok()
        .map(|(o, _)| o);
        let sparse = fit_sparse_object(&case.samples, frames, case.rate(), 10, None, budget)
            .ok()
            .map(|(o, _)| {
                all_exact &= o.verify(&case.samples);
                bytes(&o).unwrap_or(u64::MAX)
            });
        let hierarchy = fit_hierarchy_object(&case.samples, frames, case.rate(), 2, budget)
            .ok()
            .map(|(o, _)| {
                all_exact &= o.verify(&case.samples);
                bytes(&o).unwrap_or(u64::MAX)
            });
        let best = [exp2_linear.as_ref().and_then(bytes), sparse, hierarchy]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(u64::MAX);
        let flac = common::flac5(&case.samples, ch, case.rate()).unwrap_or(u64::MAX);
        report.exp2.push(best);
        report.flac.push(flac);
        report.exp1.push(exp1_bytes.unwrap_or(u64::MAX));
        report.rows.push(serde_json::json!({
            "id": case.clip.id,
            "split": case.clip.split,
            "speaker": case.clip.speaker,
            "frames": frames,
            "exp1_linear_bytes": exp1_bytes,
            "exp2_best_bytes": if best == u64::MAX { None } else { Some(best) },
            "sparse_bytes": sparse,
            "hierarchy_bytes": hierarchy,
            "flac_bytes": if flac == u64::MAX { None } else { Some(flac) },
        }));
    }
    Ok((report, all_exact))
}

fn summarize(label: &str, rep: &SplitReport) -> serde_json::Value {
    let n = rep.exp2.len();
    let mut wins_flac = 0u64;
    let mut losses_flac = 0u64;
    let mut wins_exp1 = 0u64;
    let mut losses_exp1 = 0u64;
    let mut diffs_flac = Vec::with_capacity(n);
    let mut diffs_exp1 = Vec::with_capacity(n);
    let mut total_exp2 = 0u64;
    let mut total_flac = 0u64;
    for i in 0..n {
        let e = rep.exp2[i];
        let f = rep.flac[i];
        let x = rep.exp1[i];
        total_exp2 = total_exp2.saturating_add(e);
        total_flac = total_flac.saturating_add(f);
        if e < f {
            wins_flac += 1;
        } else if e > f {
            losses_flac += 1;
        }
        if e < x {
            wins_exp1 += 1;
        } else if e > x {
            losses_exp1 += 1;
        }
        diffs_flac.push(f as i64 - e as i64);
        diffs_exp1.push(x as i64 - e as i64);
    }
    let (wp_f, wm_f, p_f) = wilcoxon_exact(&diffs_flac);
    let (wp_x, wm_x, p_x) = wilcoxon_exact(&diffs_exp1);
    let (lo_f, hi_f) = bootstrap_median_ppm(&rep.flac, &rep.exp2, 0x5EED_1234, 4000);
    let (lo_x, hi_x) = bootstrap_median_ppm(&rep.exp1, &rep.exp2, 0x5EED_5678, 4000);
    serde_json::json!({
        "split": label,
        "clips": n,
        "total_exp2_bytes": total_exp2,
        "total_flac_bytes": total_flac,
        "wins_vs_flac": wins_flac,
        "losses_vs_flac": losses_flac,
        "wins_vs_exp1": wins_exp1,
        "losses_vs_exp1": losses_exp1,
        "wilcoxon_flac": {"w_plus": wp_f, "w_minus": wm_f, "p_two_sided_ppm": p_f},
        "wilcoxon_exp1": {"w_plus": wp_x, "w_minus": wm_x, "p_two_sided_ppm": p_x},
        "bootstrap_median_flac_over_exp2_ppm": {"lo": lo_f, "hi": hi_f},
        "bootstrap_median_exp1_over_exp2_ppm": {"lo": lo_x, "hi": hi_x},
    })
}

/// Run the court; writes `receipts/learned-exp2-real-corpus/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.exp2.real-corpus.v1");
    common::push_label(&mut projection, &real_corpus_sha256());

    if !available() || !decoder_available() {
        return common::finish_exp2(
            "learned-exp2-real-corpus",
            receipts_root,
            LEARNED_EXP2_REAL_CORPUS_SHA256,
            &projection,
            Verdict::Inconclusive,
            "real corpus audio or the external `flac` decoder is absent; the frozen manifest is \
             present but no result is possible"
                .to_string(),
            vec![(
                "limitations",
                serde_json::json!([
                    "the real-corpus audio bulk (LibriSpeech dev-clean/test-clean, CC BY 4.0) is \
                     not part of the repository; only the frozen manifest of identities and \
                     canonical i32 hashes is committed",
                    "place the clips under corpus/real/ to run this court"
                ]),
            )],
        );
    }

    let budget = {
        let mut b = common::train_budget();
        b.max_iterations = 128;
        b
    };
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
            row["exp2_best_bytes"].as_u64().unwrap_or(u64::MAX),
        );
        common::push_u64(
            &mut projection,
            row["flac_bytes"].as_u64().unwrap_or(u64::MAX),
        );
        common::push_u64(
            &mut projection,
            row["exp1_linear_bytes"].as_u64().unwrap_or(u64::MAX),
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
        "learned-exp2-real-corpus",
        receipts_root,
        LEARNED_EXP2_REAL_CORPUS_SHA256,
        &projection,
        verdict,
        format!(
            "Exp2 portfolio over real speech: {} effectiveness + {} held-out Mode-C clips, paired \
             against FLAC and the Exp1 dense-linear baseline with exact Wilcoxon signed-rank and a \
             deterministic bootstrap median CI",
            effectiveness.len(),
            mode_c.len()
        ),
        vec![
            ("effectiveness", eff),
            ("mode_c", mc),
            ("effectiveness_clips", serde_json::json!(eff_report.rows)),
            ("mode_c_clips", serde_json::json!(mc_report.rows)),
            (
                "statistics",
                serde_json::json!({
                    "wilcoxon": "exact two-sided signed-rank over paired per-clip byte differences; \
                                 zero differences dropped; ties use mid-ranks",
                    "bootstrap": "4000 deterministic resamples of the paired ratio \
                                  (numerator+1)/(denominator+1); 95% percentile interval; fixed seeds",
                    "object_specific_fitting_is_not_generalization": true,
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the real corpus is 16-bit mono speech at 16 kHz, truncated to 16384 frames \
                     per clip",
                    "Mode C is held out from architecture tuning and reported separately",
                    "per-clip object-specific fitting is representation selection, never a \
                     generalization claim",
                    "the audio bulk is externally sourced (LibriSpeech, OpenSLR SLR12, CC BY 4.0) \
                     and is not committed"
                ]),
            ),
        ],
    )
}
