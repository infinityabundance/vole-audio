//! `court learned-residual` — residual shape and residual-aware training (`O.51`, `O.11`).
//!
//! Prediction error alone is not a result. This court records the shape of the
//! exact residual (zero fraction, nonzero density, run lengths, magnitude
//! statistics, an entropy estimate), the codec actually selected, and the
//! difference between a conventional MSE-optimal fit and residual-cost-aware
//! (quantization-aware) training.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::train::objective::ResidualShape;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_RESIDUAL_SHA256: &str =
    "80c3fbbf9fcb0f018f0a036d4f5d7654400a32cd2e6983afc6820b0988cf1e60";

/// Run the court; writes `receipts/learned-residual/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.residual.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut mse_better = 0u64;
    let mut qat_better = 0u64;
    let mut tie = 0u64;

    for case in intrinsic_cases() {
        let ch = case.channels;
        let frames = case.samples.len() as u64 / u64::from(ch);
        let taps: u16 = 8;
        if u64::from(taps) >= frames {
            continue;
        }
        // Conventional (MSE-optimal) fit. A candidate whose exact residual does
        // not fit the canonical i32 domain is skipped honestly.
        let Some(mse_obj) =
            common::try_fit_linear(&case.samples, ch, frames, case.sample_rate_hz, taps, None)
        else {
            continue;
        };
        if !mse_obj.verify(&case.samples) {
            all_exact = false;
        }
        let mse_residual = mse_obj.residual()?;
        let mse_shape = ResidualShape::of(&mse_residual);
        let mse_bytes = common::learned_bytes(&mse_obj)?;

        // Residual-cost-aware (quantization-aware) fit, mono only.
        let (qat_bytes, qat_shape, qat_codec) = if ch == 1 {
            match common::try_fit_qat(&case.samples, 1, frames, case.sample_rate_hz, taps, 24) {
                Some(o) if o.verify(&case.samples) => {
                    let r = o.residual()?;
                    (
                        common::learned_bytes(&o)?,
                        ResidualShape::of(&r),
                        o.residual_codec.name(),
                    )
                }
                _ => (u64::MAX, mse_shape, "unavailable"),
            }
        } else {
            (u64::MAX, mse_shape, "mono-only")
        };

        match qat_bytes.cmp(&mse_bytes) {
            std::cmp::Ordering::Less => qat_better += 1,
            std::cmp::Ordering::Equal => tie += 1,
            std::cmp::Ordering::Greater => mse_better += 1,
        }

        let best = crate::learned::residual_codec::encode_best(&mse_residual);
        common::push_label(&mut projection, case.id);
        common::push_u64(&mut projection, mse_bytes);
        common::push_u64(&mut projection, qat_bytes);
        common::push_u64(&mut projection, mse_shape.zeros);
        common::push_u64(&mut projection, mse_shape.longest_zero_run);
        common::push_u64(&mut projection, mse_shape.max_abs);
        common::push_u64(&mut projection, best.complete_bytes());

        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "mse_fit_bytes": mse_bytes,
            "qat_fit_bytes": if qat_bytes == u64::MAX { None } else { Some(qat_bytes) },
            "selected_codec": best.codec.name(),
            "residual": {
                "values": mse_shape.values,
                "zero_fraction": mse_shape.zero_fraction,
                "nonzero_density": mse_shape.nonzero_density,
                "mean_abs_nonzero": mse_shape.mean_abs_nonzero,
                "max_abs": mse_shape.max_abs,
                "longest_zero_run": mse_shape.longest_zero_run,
                "varint_entropy_bits": mse_shape.varint_entropy_bits,
                "estimated_bits": mse_shape.estimated_bits(),
            },
            "qat_residual": {
                "zero_fraction": qat_shape.zero_fraction,
                "nonzero_density": qat_shape.nonzero_density,
                "codec": qat_codec,
            },
        }));
    }

    common::push_u64(&mut projection, qat_better);
    common::push_u64(&mut projection, tie);
    common::push_u64(&mut projection, mse_better);

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-residual",
        receipts_root,
        LEARNED_RESIDUAL_SHA256,
        &projection,
        verdict,
        format!(
            "residual shape and residual-aware training over {} windows: quantization-aware \
             cheaper {qat_better}, equal {tie}, conventional cheaper {mse_better}",
            rows.len()
        ),
        vec![
            (
                "objective",
                serde_json::json!({
                    "final_judge": "actual canonical encoded residual bytes (no proxy decides)",
                    "qat_cheaper": qat_better,
                    "equal": tie,
                    "conventional_cheaper": mse_better,
                }),
            ),
            ("objects", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "residual-cost-aware training is a bounded coordinate-descent refinement of the \
                     quantized weights; a wider search is a future extension",
                    "the shape statistics are recorded, never used as the final selection criterion"
                ]),
            ),
        ],
    )
}
