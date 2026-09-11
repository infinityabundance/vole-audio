//! `court learned-speech-trace` — Seal S0 diagnostic for the real-speech court.
//!
//! **No codec changes.** This court answers two questions about the frozen
//! Seal-K result with evidence rather than assumption:
//!
//! 1. **What is FLAC-5 actually doing on these clips?** The *actual* in-process
//!    B1 artifact is parsed bit for bit (`baseline::flac_trace`) and every
//!    subframe is classified — constant / verbatim / fixed / LPC — with its
//!    order, coefficient precision, prediction shift, warmup bits, coefficient
//!    bits, partition order and residual payload bits. The trace re-reconstructs
//!    every sample, so it is an authority on the bytes, not a guess.
//! 2. **Where do VOLE's bytes go?** For the currently wired three-family Exp2
//!    portfolio (4-tap dense ridge over the whole clip, sparse ≤10 lags, 2-stage
//!    hierarchy) the court records the exact complete-byte decomposition
//!    (model / residual / metadata / integrity) and residual shape statistics
//!    (mean / median / p90 / p99 |residual|, zero fraction, sign skew).
//!
//! The court is diagnostic only: it never changes a representation, and it
//! reports the sign-extended-i16 experimental sample domain honestly.

use crate::baseline::flac_trace::{FlacTrace, SubframeKind, trace_flac};
use crate::courts::learned_common as common;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{
    RealCase, available, decoder_available, effectiveness_clips, load_cases, mode_c_clips,
    real_corpus_sha256,
};
use crate::learned::object::LearnedObject;
use crate::learned::train::hierarchy::fit_hierarchy_object;
use crate::learned::train::linear::fit_linear_object_exp2;
use crate::learned::train::sparse::fit_sparse_object;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_SPEECH_TRACE_SHA256: &str =
    "0661a292103e2b37ab90ca790cc678b36d8e0376ddc7b39cd5eec9db9ccf53e1";

const CLIPS_PER_SPLIT: usize = 8;

/// Aggregate FLAC structural accounting over one clip.
#[derive(Debug, Clone)]
struct FlacShape {
    frames: u64,
    constant: u64,
    verbatim: u64,
    fixed: u64,
    lpc: u64,
    /// `order -> count` for LPC subframes.
    lpc_orders: [u64; 33],
    /// `precision -> count`.
    lpc_precision: [u64; 33],
    /// `shift+16 -> count`.
    lpc_shift: [u64; 33],
    /// `partition_order -> count`.
    partition_orders: [u64; 16],
    metadata_bytes: u64,
    frame_header_bytes: u64,
    frame_footer_bytes: u64,
    subframe_header_bytes: u64,
    warmup_bytes: u64,
    coeff_bytes: u64,
    residual_header_bytes: u64,
    residual_payload_bytes: u64,
    total_bytes: u64,
    crc_ok: bool,
    reconstructed_ok: bool,
}

impl FlacShape {
    fn new() -> Self {
        FlacShape {
            frames: 0,
            constant: 0,
            verbatim: 0,
            fixed: 0,
            lpc: 0,
            lpc_orders: [0; 33],
            lpc_precision: [0; 33],
            lpc_shift: [0; 33],
            partition_orders: [0; 16],
            metadata_bytes: 0,
            frame_header_bytes: 0,
            frame_footer_bytes: 0,
            subframe_header_bytes: 0,
            warmup_bytes: 0,
            coeff_bytes: 0,
            residual_header_bytes: 0,
            residual_payload_bytes: 0,
            total_bytes: 0,
            crc_ok: true,
            reconstructed_ok: true,
        }
    }

    fn add(&mut self, t: &FlacTrace) {
        self.metadata_bytes += t.metadata_bytes as u64;
        self.frames += t.frames.len() as u64;
        for f in &t.frames {
            self.frame_header_bytes += u64::from(f.header_bits) / 8;
            self.frame_footer_bytes += 2;
            self.crc_ok &= f.crc8_ok && f.crc16_ok;
            for s in &f.subframes {
                self.subframe_header_bytes += u64::from(s.header_bits).div_ceil(8);
                self.warmup_bytes += s.warmup_bits / 8;
                self.coeff_bytes += s.coeff_bits / 8;
                match &s.kind {
                    SubframeKind::Constant { .. } => self.constant += 1,
                    SubframeKind::Verbatim => self.verbatim += 1,
                    SubframeKind::Fixed { .. } => self.fixed += 1,
                    SubframeKind::Lpc {
                        order,
                        precision,
                        shift,
                        ..
                    } => {
                        self.lpc += 1;
                        self.lpc_orders[(*order).min(32) as usize] += 1;
                        self.lpc_precision[(*precision).min(32) as usize] += 1;
                        let idx = (shift + 16).clamp(0, 32) as usize;
                        self.lpc_shift[idx] += 1;
                    }
                }
                if let Some(r) = &s.residual {
                    self.partition_orders[r.partition_order.min(15) as usize] += 1;
                    self.residual_header_bytes += u64::from(r.header_bits).div_ceil(8);
                    self.residual_payload_bytes += r.payload_bits / 8;
                }
            }
        }
        let mut total = t.metadata_bytes as u64;
        for f in &t.frames {
            total += f.frame_bytes as u64;
        }
        self.total_bytes += total;
    }

    fn dominant(&self, arr: &[u64; 33], offset: i64) -> Vec<serde_json::Value> {
        let mut v: Vec<(usize, u64)> = arr
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(i, c)| (i, *c))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter()
            .map(|(i, c)| serde_json::json!({"value": i as i64 - offset, "count": c}))
            .collect()
    }
}

/// Residual shape statistics for a dense residual vector.
struct ResidualStats {
    count: u64,
    mean_abs: f64,
    median_abs: i64,
    p90_abs: i64,
    p99_abs: i64,
    zero_fraction_ppm: u64,
    positive: u64,
    negative: u64,
}

fn residual_stats(residual: &[i32]) -> ResidualStats {
    let mut abs: Vec<i64> = residual.iter().map(|&v| i64::from(v).abs()).collect();
    abs.sort_unstable();
    let n = abs.len().max(1);
    let mean_abs = abs.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let at = |q: f64| -> i64 { abs[((q * (n - 1) as f64).round() as usize).min(n - 1)] };
    let zeros = residual.iter().filter(|&&v| v == 0).count() as u64;
    let positive = residual.iter().filter(|&&v| v > 0).count() as u64;
    let negative = residual.iter().filter(|&&v| v < 0).count() as u64;
    ResidualStats {
        count: residual.len() as u64,
        mean_abs,
        median_abs: at(0.5),
        p90_abs: at(0.9),
        p99_abs: at(0.99),
        zero_fraction_ppm: (zeros * 1_000_000) / n as u64,
        positive,
        negative,
    }
}

struct ClipRow {
    json: serde_json::Value,
}

fn analyse_case(case: &RealCase, budget: &crate::learned::train::TrainBudget) -> Result<ClipRow> {
    let frames = case.frames();
    let ch = case.channels();
    let rate = case.rate();

    // --- FLAC-5 actual artifact trace ---
    let artifact = crate::baseline::flac::b1_flac_artifact(
        &case.samples,
        ch,
        rate,
        crate::baseline::flac::B1_LEVEL_PRIMARY,
    )?;
    let trace = trace_flac(&artifact.bytes)?;
    let expect: Vec<i64> = case.samples.iter().map(|&v| i64::from(v)).collect();
    let mut shape = FlacShape {
        reconstructed_ok: trace.decoded == expect,
        ..FlacShape::new()
    };
    shape.add(&trace);

    // --- VOLE waterfall over the wired three-family portfolio ---
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

    let candidates: [(&str, Option<&LearnedObject>); 3] = [
        ("dense4", linear.as_ref()),
        ("sparse10", sparse.as_ref()),
        ("hier2", hierarchy.as_ref()),
    ];
    let mut waterfall = Vec::new();
    let mut best: Option<(&str, &LearnedObject, u64)> = None;
    for (name, opt) in candidates {
        let Some(o) = opt else { continue };
        if !o.verify(&case.samples) {
            return Err(crate::error::Error::internal(
                "diagnostic candidate failed exact closure",
            ));
        }
        let cost = LearnedCost::of(o)?;
        let residual = o.residual()?;
        let st = residual_stats(&residual);
        waterfall.push(serde_json::json!({
            "family": name,
            "model_bytes": cost.model_bytes,
            "residual_bytes": cost.residual_bytes,
            "metadata_bytes": cost.metadata_bytes,
            "integrity_bytes": cost.integrity_bytes,
            "complete_bytes": cost.complete_bytes,
            "residual_codec": o.residual_codec.name(),
            "mean_abs": st.mean_abs,
            "median_abs": st.median_abs,
            "p90_abs": st.p90_abs,
            "p99_abs": st.p99_abs,
            "zero_fraction_ppm": st.zero_fraction_ppm,
            "positive": st.positive,
            "negative": st.negative,
        }));
        if best.is_none_or(|(_, _, b)| cost.complete_bytes < b) {
            best = Some((name, o, cost.complete_bytes));
        }
    }
    let (best_family, best_obj, best_bytes) = best.ok_or_else(|| {
        crate::error::Error::internal("no diagnostic portfolio candidate closed exactly")
    })?;
    let best_cost = LearnedCost::of(best_obj)?;
    let best_res = residual_stats(&best_obj.residual()?);

    let flac_bytes = artifact.encoding.encoded_bytes;
    let row = serde_json::json!({
        "id": case.clip.id,
        "split": case.clip.split,
        "speaker": case.clip.speaker,
        "frames": frames,
        "flac_bytes": flac_bytes,
        "flac_shape": {
            "frames": shape.frames,
            "subframes": {"constant": shape.constant, "verbatim": shape.verbatim,
                          "fixed": shape.fixed, "lpc": shape.lpc},
            "lpc_orders_by_count": shape.dominant(&shape.lpc_orders, 0),
            "lpc_precision_by_count": shape.dominant(&shape.lpc_precision, 0),
            "lpc_shift_by_count": shape.dominant(&shape.lpc_shift, 16),
            "partition_orders": shape.partition_orders.iter().enumerate()
                .filter(|(_, c)| **c > 0).map(|(i, c)| serde_json::json!({"order": i, "count": c}))
                .collect::<Vec<_>>(),
            "bytes": {
                "metadata": shape.metadata_bytes,
                "frame_header": shape.frame_header_bytes,
                "frame_footer": shape.frame_footer_bytes,
                "subframe_header": shape.subframe_header_bytes,
                "warmup": shape.warmup_bytes,
                "coeff": shape.coeff_bytes,
                "residual_header": shape.residual_header_bytes,
                "residual_payload": shape.residual_payload_bytes,
                "total": shape.total_bytes,
            },
            "crc_ok": shape.crc_ok,
            "reconstructed_ok": shape.reconstructed_ok,
        },
        "vle_best_family": best_family,
        "vle_best_bytes": best_bytes,
        "vle_model_bytes": best_cost.model_bytes,
        "vle_residual_bytes": best_cost.residual_bytes,
        "vle_residual_codec": best_obj.residual_codec.name(),
        "vle_residual_count": best_res.count,
        "vle_residual_mean_abs": best_res.mean_abs,
        "vle_residual_median_abs": best_res.median_abs,
        "vle_residual_p90_abs": best_res.p90_abs,
        "vle_residual_p99_abs": best_res.p99_abs,
        "vle_residual_zero_fraction_ppm": best_res.zero_fraction_ppm,
        "vle_residual_positive": best_res.positive,
        "vle_residual_negative": best_res.negative,
        "waterfall": waterfall,
    });
    Ok(ClipRow { json: row })
}

/// Run the court; writes `receipts/learned-speech-trace/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.speech-trace.v1");
    common::push_label(&mut projection, &real_corpus_sha256());

    if !available() || !decoder_available() {
        return common::finish_exp2(
            "learned-speech-trace",
            receipts_root,
            LEARNED_SPEECH_TRACE_SHA256,
            &projection,
            Verdict::Inconclusive,
            "real corpus audio or the external `flac` decoder is absent; the diagnostic trace \
             cannot run"
                .to_string(),
            vec![(
                "limitations",
                serde_json::json!([
                    "the diagnostic requires the uncommitted LibriSpeech audio bulk under \
                     corpus/real/ and the external `flac` decoder used only to read the frozen \
                     compressed clips",
                    "no representation, profile or codec byte changes in this seal"
                ]),
            )],
        );
    }

    let mut budget = common::train_budget();
    budget.max_iterations = 128;
    let scratch = std::path::PathBuf::from("target/real-corpus/scratch");
    let effectiveness = load_cases(&effectiveness_clips(), CLIPS_PER_SPLIT, &scratch)?;
    let mode_c = load_cases(&mode_c_clips(), CLIPS_PER_SPLIT, &scratch)?;

    let mut rows = Vec::new();
    let mut all_ok = true;
    for case in effectiveness.iter().chain(mode_c.iter()) {
        let row = analyse_case(case, &budget)?;
        all_ok &= row.json["flac_shape"]["crc_ok"].as_bool().unwrap_or(false);
        all_ok &= row.json["flac_shape"]["reconstructed_ok"]
            .as_bool()
            .unwrap_or(false);
        projection.extend_from_slice(row.json["id"].as_str().unwrap_or("").as_bytes());
        common::push_u64(
            &mut projection,
            row.json["flac_bytes"].as_u64().unwrap_or(u64::MAX),
        );
        common::push_u64(
            &mut projection,
            row.json["vle_best_bytes"].as_u64().unwrap_or(u64::MAX),
        );
        common::push_u64(
            &mut projection,
            row.json["vle_residual_bytes"].as_u64().unwrap_or(u64::MAX),
        );
        common::push_u64(
            &mut projection,
            row.json["flac_shape"]["bytes"]["residual_payload"]
                .as_u64()
                .unwrap_or(u64::MAX),
        );
        rows.push(row.json);
    }

    let verdict = if all_ok {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_exp2(
        "learned-speech-trace",
        receipts_root,
        LEARNED_SPEECH_TRACE_SHA256,
        &projection,
        verdict,
        format!(
            "Seal S0 diagnostic: {} real clips traced bit-for-bit from the frozen B1 FLAC-5 \
             artifact (CRC-validated, fully reconstructed) beside the wired three-family Exp2 \
             VOLE byte waterfall",
            rows.len()
        ),
        vec![
            ("clips", serde_json::json!(rows)),
            (
                "method",
                serde_json::json!({
                    "flac_source": "the actual in-process B1 artifact bytes (libflac-rs pinned \
                                    =0.143.1), parsed and reconstructed sample for sample",
                    "flac_validation": "frame-header CRC-8 and frame-footer CRC-16 verified; \
                                        every subframe reconstructed and compared to the encoder \
                                        round trip; STREAMINFO sample count checked",
                    "vle_portfolio": "the currently wired three-family Exp2 real-speech portfolio \
                                      (4-tap dense ridge whole-clip, sparse <=10 lags, 2-stage \
                                      hierarchy); this is deliberately NOT the whole Exp2 profile",
                    "sample_domain": "sign-extended i16 in i32 (the Seal-K experimental domain); \
                                      this is NOT the frozen U1 s16 mapping (x << 16)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "diagnostic only: no codec, profile, container or residual byte changes",
                    "the sample domain is the sign-extended-i16 experimental domain, not U1; a \
                     separate U1-mapped real-corpus court with new identities is required",
                    "object-specific fitting is representation selection, never a generalization \
                     claim"
                ]),
            ),
        ],
    )
}
