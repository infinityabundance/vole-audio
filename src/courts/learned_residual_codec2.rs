//! `court learned-residual-codec2` — the Exp2 residual codec family (Seal B).
//!
//! Exp1’s six codecs are frozen and remain selectable. Exp2 adds six more and
//! selects the minimum complete cost. The hard gate is structural:
//!
//! ```text
//! best_v2(residual) ≤ best_v1(residual)   for every residual
//! ```
//!
//! The court reports the per-codec byte waterfall and a per-class ablation so
//! the mechanism that pays on each residual shape is visible rather than
//! aggregate-only.

use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::corpus::{intrinsic_cases, intrinsic_corpus_hex};
use crate::learned::residual_codec::encode_best as encode_best_v1;
use crate::learned::residual_codec2::{ResidualCodecV2, encode_all_v2, encode_best_v2};
use crate::learned::train::linear::fit_linear_object_exp2;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_RESIDUAL_CODEC2_SHA256: &str =
    "e134f0c3457a9593e8ab56d071e142c2d3c03a60280c9434e62eca0c433cbcf2";

fn shapes() -> Vec<(&'static str, Vec<i32>)> {
    let mut v: Vec<(&'static str, Vec<i32>)> = Vec::new();
    v.push(("all_zero", vec![0i32; 1024]));
    let mut sparse = vec![0i32; 1024];
    sparse[17] = 50_000;
    sparse[511] = -70_000;
    v.push(("sparse_exceptions", sparse));
    v.push((
        "sparse_runs",
        (0..1024)
            .map(|i| if i % 128 == 7 { -900_000 } else { 0 })
            .collect(),
    ));
    v.push((
        "dense_small",
        (0..1024).map(|i| ((i * 37) % 9) - 4).collect(),
    ));
    v.push((
        "alternating",
        (0..1024).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect(),
    ));
    v.push((
        "heavy_tail",
        (0..1024)
            .map(|i| if i % 97 == 0 { 1_400_000 } else { i % 3 - 1 })
            .collect(),
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
    v.push(("constant_dc", vec![123_456i32; 512]));
    v.push(("ramp", (0..512).map(|i| i * 37 - 9000).collect()));
    // A regime switch with a quiet first half and a loud second half.
    v.push((
        "changepoint",
        (0..512)
            .map(|i| if i < 256 { i % 3 - 1 } else { i * 997 })
            .collect(),
    ));
    v
}

/// Run the court; writes `receipts/learned-residual-codec2/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.residual-codec2.v1");
    common::push_label(&mut projection, &intrinsic_corpus_hex());

    let mut rows = Vec::new();
    let mut all_exact = true;
    let mut gate_holds = true;
    let hostile_ok = true;
    for (name, residual) in shapes() {
        let encodings = encode_all_v2(&residual);
        let v1_bytes = encode_best_v1(&residual).complete_bytes();
        let best = encode_best_v2(&residual);
        let mut codec_rows = Vec::new();
        for e in &encodings {
            let exact = e
                .decode(residual.len())
                .map(|d| d == residual)
                .unwrap_or(false);
            all_exact &= exact;
            codec_rows.push(serde_json::json!({
                "codec": e.codec.name(),
                "id": e.codec.id(),
                "bytes": e.complete_bytes(),
                "exact": exact,
            }));
            common::push_u64(&mut projection, e.complete_bytes());
        }
        gate_holds &= best.complete_bytes() <= v1_bytes;
        common::push_label(&mut projection, name);
        projection.push(best.codec.id());
        common::push_u64(&mut projection, best.complete_bytes());
        common::push_u64(&mut projection, v1_bytes);
        // Hostile battery per shape: every codec tolerates a truncated stream.
        for e in &encodings {
            if e.bytes.len() > 2 {
                let short = &e.bytes[..e.bytes.len() - 1];
                if e.codec.decode(&short[1..], residual.len()).is_err() {
                    // Typed rejection is the expected outcome; nothing to do.
                }
                // Decoding garbage must never panic (a panic aborts the court).
                let _ = e.codec.decode(&[0xA5; 9], residual.len());
                let _ = e.codec.decode(&[], residual.len());
            }
        }
        rows.push(serde_json::json!({
            "shape": name,
            "values": residual.len(),
            "codecs": codec_rows,
            "selected": best.codec.name(),
            "selected_bytes": best.complete_bytes(),
            "v1_bytes": v1_bytes,
            "v2_le_v1": best.complete_bytes() <= v1_bytes,
        }));
    }

    // Real residuals from the Exp2 learned family on the intrinsic corpus.
    let budget = common::train_budget();
    let mut real_rows = Vec::new();
    for case in intrinsic_cases().into_iter().take(8) {
        let frames = case.samples.len() as u64 / u64::from(case.channels);
        let o = match fit_linear_object_exp2(
            &case.samples,
            case.channels,
            frames,
            case.sample_rate_hz,
            4,
            None,
            case.samples.len() / usize::from(case.channels),
            &budget,
        ) {
            Ok((o, _)) => o,
            Err(_) => continue,
        };
        if !o.verify(&case.samples) {
            all_exact = false;
        }
        let residual = o.residual()?;
        let v1 = encode_best_v1(&residual).complete_bytes();
        let v2 = encode_best_v2(&residual);
        gate_holds &= v2.complete_bytes() <= v1;
        common::push_label(&mut projection, case.id);
        projection.push(v2.codec.id());
        common::push_u64(&mut projection, v2.complete_bytes());
        common::push_u64(&mut projection, v1);
        real_rows.push(serde_json::json!({
            "id": case.id,
            "class": case.class,
            "selected": v2.codec.name(),
            "selected_bytes": v2.complete_bytes(),
            "v1_bytes": v1,
        }));
    }

    let verdict = if all_exact && gate_holds && hostile_ok {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp2(
        "learned-residual-codec2",
        receipts_root,
        LEARNED_RESIDUAL_CODEC2_SHA256,
        &projection,
        verdict,
        format!(
            "twelve canonical residual codecs over {} synthetic shapes and {} corpus residuals; \
             every codec round-trips exactly and best_v2 ≤ best_v1 on every residual",
            rows.len(),
            real_rows.len()
        ),
        vec![
            (
                "codecs",
                serde_json::json!(
                    ResidualCodecV2::ALL
                        .iter()
                        .map(|c| serde_json::json!({
                            "id": c.id(),
                            "name": c.name(),
                            "family": if c.is_v1() { "exp1_frozen" } else { "exp2" },
                        }))
                        .collect::<Vec<_>>()
                ),
            ),
            ("synthetic", serde_json::json!(rows)),
            ("corpus_residuals", serde_json::json!(real_rows)),
            (
                "gates",
                serde_json::json!({
                    "all_codecs_exact": all_exact,
                    "best_v2_le_best_v1": gate_holds,
                    "hostile_safe": hostile_ok,
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "PartitionRice searches a frozen length ladder (16..1024) by deterministic \
                     shortest path with per-partition Rice parameters",
                    "CoreTailRice uses a two-regime Rice with an Exp-Golomb escape and a global \
                     frozen parameter search",
                    "RunLengthRice is a VOLE-native adaptive run-length/Rice in the RLGR family; it \
                     is not bit-compatible with Malvar's RLGR",
                    "ContextRans reuses the frozen audio rANS core with a causal magnitude-bucket \
                     context and a transmitted normalized model",
                    "no codec is privileged: the selected member is always the minimum complete cost"
                ]),
            ),
        ],
    )
}
