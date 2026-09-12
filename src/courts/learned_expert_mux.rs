//! `court learned-expert-mux` — Phase 6 mechanism 4 (`SampleExpertMux`).
//!
//! `SampleExpertMux` (model kind 20) keeps several decoder-synchronized
//! backward-adaptive experts and selects **one** of them per microgroup
//! (`x̂ = P_{j*}`, never a mixture), transmitting a packed selector stream.
//! Every expert is stepped from the reconstructed value, so the experts stay
//! synchronised for free and their trajectories do not depend on the selector —
//! which makes the per-group choice exactly optimal for the fixed expert set.
//!
//! This court isolates the mechanism's contribution:
//!
//! * deterministic non-stationary fixtures whose best predictor changes across
//!   the extent (so selection can pay);
//! * the real speech effectiveness clips;
//!
//! and reports, per case, the mux size beside (a) the best *single* expert from
//! the same fitted pair and (b) the existing backward-adaptive family, plus the
//! selector usage histogram. Every object must close exactly. Held-out Mode C is
//! deliberately untouched.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::train::TrainBudget;
use crate::learned::train::adaptive::fit_adaptive_object;
use crate::learned::train::mux::fit_expert_mux_object;
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_EXPERT_MUX_SHA256: &str =
    "d04d6b71b47fd0b4703527bee5cd933e07f1efe78db5db481f5f7997b87015d4";

fn bytes(o: &LearnedObject) -> Option<u64> {
    LearnedCost::of(o).ok().map(|c| c.complete_bytes)
}

struct Row {
    id: String,
    population: &'static str,
    samples: usize,
    mux_bytes: u64,
    single_expert_bytes: u64,
    adaptive_bytes: u64,
    groups: usize,
    use0: u64,
    use1: u64,
    exact: bool,
}

/// Deterministic non-stationary fixtures: the best predictor changes mid-signal.
fn synthetic_fixtures() -> Vec<(&'static str, Vec<i32>)> {
    vec![
        (
            "regime_switch_ar",
            regime_signal(4096, &[(0, 0.95), (2048, -0.6)], 3),
        ),
        (
            "triple_regime",
            regime_signal(6144, &[(0, 0.9), (2048, 0.1), (4096, -0.75)], 9),
        ),
        ("drifting_ar", drifting_signal(6144, 0.2, 0.9, 11)),
    ]
}

fn regime_signal(n: usize, switches: &[(usize, f64)], seed: u64) -> Vec<i32> {
    let mut st = seed | 1;
    let mut x = 0i64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = switches
            .iter()
            .rev()
            .find(|(at, _)| i >= *at)
            .map(|(_, a)| *a)
            .unwrap_or(0.9);
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (((st >> 33) & 0xffff) as i64 - 32768) >> 4;
        x = ((a * x as f64).round() as i64) + noise;
        x = x.clamp(-(1 << 22), 1 << 22);
        out.push(x as i32);
    }
    out
}

fn drifting_signal(n: usize, lo: f64, hi: f64, seed: u64) -> Vec<i32> {
    let mut st = seed | 1;
    let mut x = 0i64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let frac = i as f64 / n as f64;
        let a = lo + (hi - lo) * frac;
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (((st >> 33) & 0xffff) as i64 - 32768) >> 4;
        x = ((a * x as f64).round() as i64) + noise;
        x = x.clamp(-(1 << 22), 1 << 22);
        out.push(x as i32);
    }
    out
}

/// Evaluate one signal: mux versus its best single expert and the adaptive family.
fn evaluate(
    id: String,
    population: &'static str,
    source: &[i32],
    rate: u32,
    budget: &TrainBudget,
) -> Result<Row> {
    let frames = source.len() as u64;
    let (mux_obj, _) = fit_expert_mux_object(source, frames, rate, budget)?;
    let mut exact = mux_obj.verify(source);
    let mux_bytes = bytes(&mux_obj).unwrap_or(u64::MAX);
    let LearnedModel::ExpertMux(m) = &mux_obj.model else {
        return Err(crate::error::Error::internal(
            "mux fit returned a non-mux model",
        ));
    };
    let (groups, use0, use1) = {
        let mut use_counts = [0u64; 2];
        for &s in &m.selectors {
            use_counts[usize::from(s).min(1)] += 1;
        }
        (m.selectors.len(), use_counts[0], use_counts[1])
    };
    // The best single expert from the same fitted pair: the no-selection floor.
    let mut single = u64::MAX;
    for e in &m.experts {
        if let Ok(o) = LearnedObject::from_intrinsic_exp2(
            LearnedModel::Adaptive(e.clone()),
            1,
            frames,
            rate,
            Vec::new(),
            source,
        ) && o.verify(source)
            && let Some(b) = bytes(&o)
        {
            single = single.min(b);
        }
    }
    // The existing backward-adaptive family (best over the tap ladder).
    let mut adaptive = u64::MAX;
    for taps in [8u16, 16] {
        if let Ok((o, _)) = fit_adaptive_object(source, frames, rate, taps, budget) {
            exact &= o.verify(source);
            if let Some(b) = bytes(&o) {
                adaptive = adaptive.min(b);
            }
        }
    }
    Ok(Row {
        id,
        population,
        samples: source.len(),
        mux_bytes,
        single_expert_bytes: single,
        adaptive_bytes: adaptive,
        groups,
        use0,
        use1,
        exact,
    })
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.exact;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.mux_bytes);
    common::push_u64(projection, row.single_expert_bytes);
    common::push_u64(projection, row.adaptive_bytes);
    common::push_u64(projection, row.use0);
    common::push_u64(projection, row.use1);
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "population": row.population,
        "samples": row.samples,
        "mux_bytes": row.mux_bytes,
        "best_single_expert_bytes": row.single_expert_bytes,
        "adaptive_family_bytes": row.adaptive_bytes,
        "gain_vs_single_expert_bytes": row.single_expert_bytes as i64 - row.mux_bytes as i64,
        "gain_vs_adaptive_bytes": row.adaptive_bytes as i64 - row.mux_bytes as i64,
        "groups": row.groups,
        "selector_usage": [row.use0, row.use1],
        "exact": row.exact,
    })
}

/// Run the court; writes `receipts/learned-expert-mux/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.expert_mux.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let mut all_exact = true;

    let mut synthetic = Vec::new();
    for (name, source) in synthetic_fixtures() {
        let row = evaluate(
            name.to_string(),
            "synthetic_nonstationary",
            &source,
            48_000,
            &budget,
        )?;
        push_row(&mut projection, &row, &mut all_exact);
        synthetic.push(row_json(&row));
    }

    let mut real = Vec::new();
    let mut real_present = false;
    if available() && decoder_available() {
        real_present = true;
        let scratch = PathBuf::from("target/real-corpus/scratch");
        for case in load_cases(&effectiveness_clips(), speech::CLIPS_PER_SPLIT, &scratch)? {
            let row = evaluate(
                case.clip.id.clone(),
                "speech_effectiveness",
                &case.samples,
                case.rate(),
                &budget,
            )?;
            push_row(&mut projection, &row, &mut all_exact);
            real.push(row_json(&row));
        }
    }

    let count_true = |rows: &[serde_json::Value], key: &str| -> u64 {
        rows.iter()
            .filter(|r| r[key].as_i64().is_some_and(|v| v > 0))
            .count() as u64
    };
    let syn_wins_vs_single = count_true(&synthetic, "gain_vs_single_expert_bytes");
    let real_wins_vs_single = count_true(&real, "gain_vs_single_expert_bytes");
    let real_wins_vs_adaptive = count_true(&real, "gain_vs_adaptive_bytes");

    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else if !real_present {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };

    common::finish_exp3(
        "learned-expert-mux",
        receipts_root,
        LEARNED_EXPERT_MUX_SHA256,
        &projection,
        verdict,
        format!(
            "hard-selection expert mux (model kind 20) over {} synthetic + {} real cases: beats \
             the best single expert on {syn_wins_vs_single}/{} synthetic and \
             {real_wins_vs_single}/{} real, and the adaptive family on {real_wins_vs_adaptive}/{} \
             real",
            synthetic.len(),
            real.len(),
            synthetic.len(),
            real.len(),
            real.len(),
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "synthetic_cases": synthetic.len(),
                    "real_cases": real.len(),
                }),
            ),
            ("synthetic", serde_json::json!(synthetic)),
            ("real_effectiveness", serde_json::json!(real)),
            (
                "method",
                serde_json::json!({
                    "selection": "x_hat = P_{j*}, a hard per-microgroup selection, never a mixture",
                    "experts": "backward-adaptive sign-sign LMS experts (short/fast vs long/slow), \
                                initialised from one ridge fit",
                    "selectors": "packed bit field, one entry per microgroup; 1 bit per group for \
                                  two experts",
                    "synchronisation": "every expert is stepped from the reconstructed value, so \
                                        the experts stay decoder-synchronised and their \
                                        trajectories are independent of the selector; the \
                                        per-group choice is therefore exactly optimal",
                    "ladder": "expert pairs ((8,8),(8,1)), ((16,8),(16,1)), ((8,8),(16,1)), \
                               ((16,4),(8,2)) and groups 64/128/256, selected by complete bytes",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the selector stream is charged as model bytes, so a constant selector (no \
                     regime change) is a near-negative by construction",
                    "the expert trajectories do not depend on the selection, which is what makes \
                     the choice separable; a variant whose experts only update when selected \
                     would become path-dependent and require a search",
                ]),
            ),
        ],
    )
}
