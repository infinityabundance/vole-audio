//! `court learned-shared` — shared-model amortization (`O.54`, `O.15`).
//!
//! One learned model may be shared across many objects when dependency
//! semantics permit. This court reports the shared-model bytes, the per-object
//! incremental bytes, the whole-corpus bytes, and the amortization crossover
//! `N*` — or states explicitly that no crossover was observed.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::accounting::{LearnedCost, SharedModelCost};
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::train::linear::fit_linear_object;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_SHARED_SHA256: &str =
    "eab4d545fabd7235eee32ab632ab9d30f26cec704e0dbf012f8239d4a41eb00b";

/// Corpus sizes evaluated for the crossover.
pub const CORPUS_SIZES: [u64; 7] = [1, 2, 4, 8, 16, 32, 64];

fn per_object_incremental(cost: &LearnedCost) -> u64 {
    // The shared model is stored once; each object stores framing, residual,
    // dependencies and integrity.
    cost.metadata_bytes
        .saturating_add(cost.residual_bytes)
        .saturating_add(cost.dependency_bytes)
        .saturating_add(cost.integrity_bytes)
}

/// Run the court; writes `receipts/learned-shared/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.shared.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let cases = intrinsic_cases();
    // Regime A: N copies of one object (a shared model is exactly applicable).
    let a_case = cases.iter().find(|c| c.id == "sine-440").ok_or_else(|| {
        crate::error::Error::internal("corpus is missing the shared-model anchor")
    })?;
    let (a_obj, _) = fit_linear_object(
        &a_case.samples,
        1,
        a_case.samples.len() as u64,
        a_case.sample_rate_hz,
        8,
        None,
        a_case.samples.len(),
        &common::train_budget(),
    )?;
    let a_cost = LearnedCost::of(&a_obj)?;
    let a_shared = a_cost.model_bytes;
    let a_incr = per_object_incremental(&a_cost);
    let a_independent = common::best_simple_bytes(
        &a_case.samples,
        1,
        a_case.samples.len() as u64,
        a_case.sample_rate_hz,
    );
    let a_regime = SharedModelCost {
        shared_model_bytes: a_shared,
        per_object_incremental_bytes: a_incr,
    };
    let a_n_star = a_regime.crossover(a_independent, 1024);

    // Regime B: N distinct objects sharing one model fitted on the first.
    let b_shared = a_shared;
    let mut b_incr_total = 0u64;
    let mut b_independent_total = 0u64;
    let mut b_count = 0u64;
    for case in cases.iter().take(8).filter(|c| c.channels == 1) {
        let frames = case.samples.len() as u64;
        // Reuse the fixed model (weights/bias from the anchor), recompute the
        // residual against the fixed model by building a zero-order object.
        let mut model = match &a_obj.model {
            crate::learned::model::LearnedModel::Linear(p) => p.clone(),
            _ => continue,
        };
        model.block_frames = None;
        let o = match crate::learned::object::LearnedObject::from_intrinsic(
            crate::learned::model::LearnedModel::Linear(model),
            1,
            frames,
            case.sample_rate_hz,
            Vec::new(),
            &case.samples,
        ) {
            Ok(o) => o,
            Err(_) => continue,
        };
        let cost = LearnedCost::of(&o)?;
        b_incr_total = b_incr_total.saturating_add(per_object_incremental(&cost));
        let ind = common::best_simple_bytes(&case.samples, 1, frames, case.sample_rate_hz);
        b_independent_total = b_independent_total.saturating_add(ind);
        b_count += 1;
        let _ = o;
    }
    let b_incr_each = b_incr_total.checked_div(b_count).unwrap_or(0);
    let b_independent_each = b_independent_total.checked_div(b_count).unwrap_or(0);
    let b_regime = SharedModelCost {
        shared_model_bytes: b_shared,
        per_object_incremental_bytes: b_incr_each,
    };
    let b_n_star = b_regime.crossover(b_independent_each, 1024);

    // Thresholds: a fixed independent total.
    let mut a_sizes = Vec::new();
    for n in CORPUS_SIZES {
        a_sizes.push(serde_json::json!({
            "n": n,
            "shared_bytes": a_regime.corpus_bytes(n),
            "independent_bytes": a_independent.saturating_mul(n),
        }));
    }
    let mut b_sizes = Vec::new();
    for n in CORPUS_SIZES {
        b_sizes.push(serde_json::json!({
            "n": n,
            "shared_bytes": b_regime.corpus_bytes(n),
            "independent_bytes": b_independent_each.saturating_mul(n),
        }));
    }

    common::push_u64(&mut projection, a_shared);
    common::push_u64(&mut projection, a_incr);
    common::push_u64(&mut projection, a_n_star.unwrap_or(u64::MAX));
    common::push_u64(&mut projection, b_shared);
    common::push_u64(&mut projection, b_incr_each);
    common::push_u64(&mut projection, b_n_star.unwrap_or(u64::MAX));

    common::finish(
        "learned-shared",
        receipts_root,
        LEARNED_SHARED_SHA256,
        &projection,
        Verdict::Supported,
        format!(
            "shared-model amortization: identical-object regime N* = {:?}; distinct-object regime \
             N* = {:?}",
            a_n_star, b_n_star
        ),
        vec![
            (
                "identical_objects",
                serde_json::json!({
                    "shared_model_bytes": a_shared,
                    "per_object_incremental_bytes": a_incr,
                    "independent_bytes_each": a_independent,
                    "amortization_crossover_objects": a_n_star,
                    "corpus": a_sizes,
                }),
            ),
            (
                "distinct_objects",
                serde_json::json!({
                    "shared_model_bytes": b_shared,
                    "per_object_incremental_bytes": b_incr_each,
                    "independent_bytes_each": b_independent_each,
                    "amortization_crossover_objects": b_n_star,
                    "objects": b_count,
                    "corpus": b_sizes,
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "a shared-model win is never reported without the corpus size required to obtain it",
                    "NO_CROSSOVER (null) means the shared regime never became cheaper within the search bound"
                ]),
            ),
        ],
    )
}
