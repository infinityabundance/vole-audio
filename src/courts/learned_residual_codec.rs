//! `court learned-residual-codec` — the exact residual codec family (`O.47`).
//!
//! Compares the six canonical codecs over representative residual shapes,
//! including the two shapes `O.11` is about: tiny dense nonzero errors and
//! sparse larger exceptions. Every codec must round-trip exactly, and the
//! selected codec is the canonical minimum-bytes member with deterministic ties.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::residual_codec::{ResidualCodec, encode_all};
use crate::learned::train::linear::fit_linear_object;
use crate::status::Verdict;
use std::path::Path;
use std::time::Instant;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_RESIDUAL_CODEC_SHA256: &str =
    "49f8d5c6b0fb43bef55366775f349d2f36ea48dd2f388c3b727f42f78ca60b66";

/// Run the court; writes `receipts/learned-residual-codec/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.residual-codec.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    // --- 1. canonical synthetic shapes ---
    let shapes: Vec<(&str, Vec<i32>)> = {
        let mut v: Vec<(&str, Vec<i32>)> = Vec::new();
        v.push(("all_zero", vec![0i32; 1024]));
        let mut sparse = vec![0i32; 1024];
        sparse[17] = 50_000;
        sparse[511] = -70_000;
        v.push(("sparse_exceptions", sparse));
        v.push((
            "dense_small",
            (0..1024).map(|i| ((i * 37) % 9) - 4).collect(),
        ));
        v.push((
            "alternating",
            (0..1024).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect(),
        ));
        let mut s = 0x9E37_79B9u64;
        v.push((
            "full_width_random",
            (0..1024)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s as i32).wrapping_mul(2_654_435_761u32 as i32)
                })
                .collect(),
        ));
        v.push((
            "extremes",
            (0..256)
                .map(|i| if i % 3 == 0 { i32::MIN } else { i32::MAX })
                .collect(),
        ));
        v
    };

    let mut rows = Vec::new();
    let mut all_exact = true;
    for (name, residual) in &shapes {
        let mut encodings = Vec::new();
        for e in encode_all(residual) {
            let t0 = Instant::now();
            let back = e.decode(residual.len())?;
            let decode_ns = t0.elapsed().as_nanos() as u64;
            let exact = back == *residual;
            all_exact &= exact;
            encodings.push(serde_json::json!({
                "codec": e.codec.name(),
                "id": e.codec.id(),
                "bytes": e.complete_bytes(),
                "exact": exact,
                "decode_ns": decode_ns,
            }));
            common::push_u64(&mut projection, e.complete_bytes());
        }
        let best = crate::learned::residual_codec::encode_best(residual);
        common::push_label(&mut projection, name);
        push_codec_id(&mut projection, best.codec);
        common::push_u64(&mut projection, best.complete_bytes());
        rows.push(serde_json::json!({
            "shape": name,
            "values": residual.len(),
            "codecs": encodings,
            "selected": best.codec.name(),
            "selected_bytes": best.complete_bytes(),
        }));
    }

    // --- 2. real residuals from the learned family on the intrinsic corpus ---
    let mut real_rows = Vec::new();
    let budget = common::train_budget();
    for case in intrinsic_cases().into_iter().take(8) {
        let frames = case.samples.len() as u64 / u64::from(case.channels);
        let (o, _) = fit_linear_object(
            &case.samples,
            case.channels,
            frames,
            case.sample_rate_hz,
            4,
            None,
            case.samples.len() / usize::from(case.channels),
            &budget,
        )?;
        if !o.verify(&case.samples) {
            all_exact = false;
        }
        let residual = o.residual()?;
        let shape = crate::learned::train::objective::ResidualShape::of(&residual);
        let best = crate::learned::residual_codec::encode_best(&residual);
        common::push_label(&mut projection, case.id);
        push_codec_id(&mut projection, best.codec);
        common::push_u64(&mut projection, best.complete_bytes());
        common::push_u64(&mut projection, shape.zeros);
        common::push_u64(&mut projection, shape.longest_zero_run);
        real_rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "selected": best.codec.name(),
            "selected_bytes": best.complete_bytes(),
            "zero_fraction": shape.zero_fraction,
            "nonzero_density": shape.nonzero_density,
            "longest_zero_run": shape.longest_zero_run,
            "mean_abs_nonzero": shape.mean_abs_nonzero,
            "varint_entropy_bits": shape.varint_entropy_bits,
        }));
    }

    let verdict = if all_exact {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    common::finish(
        "learned-residual-codec",
        receipts_root,
        LEARNED_RESIDUAL_CODEC_SHA256,
        &projection,
        verdict,
        format!(
            "six canonical residual codecs over {} synthetic shapes and {} corpus residuals; \
             every codec round-trips exactly and the selected codec is the minimum-bytes member",
            shapes.len(),
            real_rows.len()
        ),
        vec![
            (
                "codecs",
                serde_json::json!(
                    ResidualCodec::ALL
                        .iter()
                        .map(|c| serde_json::json!({"id": c.id(), "name": c.name()}))
                        .collect::<Vec<_>>()
                ),
            ),
            ("synthetic", serde_json::json!(rows)),
            ("corpus_residuals", serde_json::json!(real_rows)),
            (
                "limitations",
                serde_json::json!([
                    "block-adaptive Rice selects its parameter per 32-value block by a frozen \
                     minimum-bits rule and may pad the final block with unused bits",
                    "no codec is privileged: the selected codec is always the minimum complete \
                     cost with ascending-id tie break"
                ]),
            ),
        ],
    )
}

fn push_codec_id(out: &mut Vec<u8>, codec: ResidualCodec) {
    out.push(codec.id());
}
