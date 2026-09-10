//! `court learned-gpu` — execution surfaces and direct-materialization
//! interaction (`O.56`, `O.57`, `O.33`, `O.36`).
//!
//! Measures the host surfaces that exist (scalar and SIMD, with exact parity)
//! and records the device surfaces honestly: the learned device kernel is not
//! part of the **frozen** device artifact (which must stay byte-identical for
//! Phases G/J), so CUDA/ROCm learned execution is an explicit unavailable
//! extension rather than a fabricated pass.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::eval::learned_simd;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::finite_field::LinearPredictor;
use crate::status::Verdict;
use std::path::Path;
use std::time::Instant;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_GPU_SHA256: &str =
    "542afe4f76793dde4dbe86477f17e71d89b476e8c58027bfded9dae9b8680259";

/// Tap counts compared across surfaces.
pub const GPU_TAPS: [u16; 6] = [4, 8, 16, 32, 64, 128];

fn predictor(taps: u16) -> LinearPredictor {
    LinearPredictor {
        channels: 1,
        taps,
        weights: (0..taps)
            .map(|i| (i as i16).wrapping_mul(97).wrapping_add(13))
            .collect(),
        bias: vec![-7],
        block_frames: None,
    }
}

/// Run the court; writes `receipts/learned-gpu/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.gpu.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let case = intrinsic_cases()
        .into_iter()
        .find(|c| c.id == "sine-440")
        .ok_or_else(|| crate::error::Error::internal("corpus is missing the surface anchor"))?;

    let mut rows = Vec::new();
    let mut parity_all = true;
    for taps in GPU_TAPS {
        let p = predictor(taps);
        let k = usize::from(taps);
        if k >= case.samples.len() {
            continue;
        }
        // Exact parity over many windows.
        let mut parity = true;
        let mut a = [0i32];
        let mut b = [0i32];
        for t in k..case.samples.len() {
            let hist = &case.samples[t - k..t];
            learned_simd::scalar_hypothesis(&p, hist, &mut a);
            learned_simd::simd_hypothesis(&p, hist, &mut b);
            if a != b {
                parity = false;
                break;
            }
        }
        parity_all &= parity;

        // Throughput: scalar vs SIMD per-frame hypothesis.
        let reps = 4;
        let t0 = Instant::now();
        for _ in 0..reps {
            for t in k..case.samples.len() {
                learned_simd::scalar_hypothesis(&p, &case.samples[t - k..t], &mut a);
            }
        }
        let scalar_ns = t0.elapsed().as_nanos() as u64 / reps;
        let t0 = Instant::now();
        for _ in 0..reps {
            for t in k..case.samples.len() {
                learned_simd::simd_hypothesis(&p, &case.samples[t - k..t], &mut b);
            }
        }
        let simd_ns = t0.elapsed().as_nanos() as u64 / reps;

        common::push_u64(&mut projection, u64::from(taps));
        common::push_u64(&mut projection, u64::from(parity));
        rows.push(serde_json::json!({
            "taps": taps,
            "scalar_ns": scalar_ns,
            "simd_ns": simd_ns,
            "simd_speedup": if simd_ns == 0 { 0.0 } else { scalar_ns as f64 / simd_ns as f64 },
            "exact_parity": parity,
        }));
    }

    common::push_u64(&mut projection, u64::from(parity_all));

    let verdict = if parity_all {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-gpu",
        receipts_root,
        LEARNED_GPU_SHA256,
        &projection,
        verdict,
        format!(
            "execution surfaces over {} tap counts: scalar == {} (parity {:?}); CUDA/ROCm learned \
             kernels are explicitly unavailable in this build",
            rows.len(),
            learned_simd::detect().name(),
            parity_all
        ),
        vec![
            (
                "surfaces",
                serde_json::json!({
                    "scalar": {"status": "SUPPORTED", "authority": "semantic"},
                    "simd": {"status": if parity_all { "SUPPORTED" } else { "FAILED_CORRECTNESS" },
                             "level": learned_simd::detect().name()},
                    "cuda": {"status": "NOT_IMPLEMENTED",
                             "reason": "the learned device kernel is not part of the frozen device \
                                        artifact; adding it would change the Phase G/J PTX bytes and \
                                        invalidate sealed device evidence. A separate learned device \
                                        artifact is a declared future extension."},
                    "rocm": {"status": "UNSUPPORTED_BY_HARDWARE",
                             "reason": "no AMD GPU / KFD / ROCm userspace on this host"},
                }),
            ),
            ("tap_sweep", serde_json::json!(rows)),
            (
                "direct_materialization",
                serde_json::json!({
                    "architecture": "a learned SampleObject is materialized by the same scalar/SIMD \
                                     evaluator that feeds the D0/D1 path; the learned representation \
                                     does not change D0/D1/D2 semantics",
                    "measured": false,
                    "reason": "no learned device artifact exists in this build, so no endpoint \
                               deadline or D1 byte-removal claim is made",
                }),
            ),
        ],
    )
}
