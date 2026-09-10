//! `court learned-quantization` — canonical precision and quantization-aware
//! training (`O.52`, `O.22`, `O.23`).
//!
//! The canonical learned container currently stores **i16 Q12 weights and i32
//! Q12 biases**. i8 and mixed-precision storage are reported explicitly as not
//! implemented rather than faked; what *is* compared is the real difference
//! between post-training quantization and quantization-aware training at the
//! implemented precision, with exact SIMD parity.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::eval::learned_simd;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::model::LearnedModel;
use crate::status::Verdict;
use std::path::Path;
use std::time::Instant;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_QUANTIZATION_SHA256: &str =
    "0d8b125065dd69245b340efdc97c9eb5b57c699b6ea2c052c3c36376147fd596";

/// Run the court; writes `receipts/learned-quantization/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.quantization.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut simd_all = true;
    let mut ptq_total = 0u64;
    let mut qat_total = 0u64;

    for case in intrinsic_cases()
        .into_iter()
        .filter(|c| c.channels == 1)
        .take(12)
    {
        let frames = case.samples.len() as u64;
        let taps: u16 = 8;
        let Some(ptq) =
            common::try_fit_linear(&case.samples, 1, frames, case.sample_rate_hz, taps, None)
        else {
            continue;
        };
        let Some(qat) =
            common::try_fit_qat(&case.samples, 1, frames, case.sample_rate_hz, taps, 16)
        else {
            continue;
        };
        all_exact &= ptq.verify(&case.samples) && qat.verify(&case.samples);

        // Scalar vs SIMD parity on the fitted model.
        let parity = if let LearnedModel::Linear(p) = &ptq.model {
            let mut ok = true;
            let mut h = vec![0i32; 1];
            let mut g = vec![0i32; 1];
            for t in p.tap_count()..case.samples.len().min(1024) {
                let hist = &case.samples[t - p.tap_count()..t];
                learned_simd::scalar_hypothesis(p, hist, &mut h);
                learned_simd::simd_hypothesis(p, hist, &mut g);
                ok &= h == g;
            }
            ok
        } else {
            false
        };
        simd_all &= parity;

        // Measure the scalar and SIMD per-frame hypothesis cost.
        let mut scalar_ns = 0u64;
        let mut simd_ns = 0u64;
        if let LearnedModel::Linear(p) = &ptq.model {
            let k = p.tap_count();
            let mut h = vec![0i32; 1];
            let t0 = Instant::now();
            for t in k..case.samples.len().min(4096) {
                learned_simd::scalar_hypothesis(p, &case.samples[t - k..t], &mut h);
            }
            scalar_ns = t0.elapsed().as_nanos() as u64;
            let t0 = Instant::now();
            for t in k..case.samples.len().min(4096) {
                learned_simd::simd_hypothesis(p, &case.samples[t - k..t], &mut h);
            }
            simd_ns = t0.elapsed().as_nanos() as u64;
        }

        let ptq_bytes = common::learned_bytes(&ptq)?;
        let qat_bytes = common::learned_bytes(&qat)?;
        ptq_total = ptq_total.saturating_add(ptq_bytes);
        qat_total = qat_total.saturating_add(qat_bytes);

        common::push_label(&mut projection, case.id);
        common::push_u64(&mut projection, ptq_bytes);
        common::push_u64(&mut projection, qat_bytes);
        common::push_u64(&mut projection, u64::from(parity));
        rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "ptq_bytes": ptq_bytes,
            "qat_bytes": qat_bytes,
            "simd_parity": parity,
            "scalar_hypothesis_ns": scalar_ns,
            "simd_hypothesis_ns": simd_ns,
        }));
    }

    common::push_u64(&mut projection, ptq_total);
    common::push_u64(&mut projection, qat_total);

    let verdict = if all_exact && simd_all {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-quantization",
        receipts_root,
        LEARNED_QUANTIZATION_SHA256,
        &projection,
        verdict,
        format!(
            "post-training quantization vs quantization-aware training at the implemented i16 Q12 \
             precision over {} windows (PTQ {ptq_total} B, QAT {qat_total} B); SIMD parity {:?}",
            rows.len(),
            simd_all
        ),
        vec![
            (
                "precisions",
                serde_json::json!([
                    {"precision": "i16_q12_weights_i32_q12_bias", "status": "IMPLEMENTED", "canonical": true},
                    {"precision": "i8", "status": "NOT_IMPLEMENTED",
                     "reason": "the canonical container has no i8 weight lattice in this profile; a \
                                weight-bits field plus a per-model scale is a declared format extension"},
                    {"precision": "mixed", "status": "NOT_IMPLEMENTED",
                     "reason": "mixed precision needs the same weight-bits/scale extension"}
                ]),
            ),
            (
                "simd",
                serde_json::json!({
                    "level": learned_simd::detect().name(),
                    "parity": simd_all,
                }),
            ),
            ("objects", serde_json::json!(rows)),
        ],
    )
}
