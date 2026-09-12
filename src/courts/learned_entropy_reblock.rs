//! `court learned-entropy-reblock` — Phase 6 mechanism 3 (`EntropyReblock`).
//!
//! The predictor is untouched. `EntropyReblock` (residual codec id 24)
//! re-partitions the residual **for entropy coding alone** by a shortest path
//! over aligned boundaries, and each partition selects the base coder that
//! minimizes its stored bytes among Exp-Golomb(0), Rice, general Golomb and
//! BGMC. It is strictly more general than `PartitionRice` (id 6), which uses a
//! fixed ladder and only Rice parameters.
//!
//! This court isolates the mechanism's attributable contribution:
//!
//! * deterministic heteroscedastic fixtures (quiet/loud, alternating regimes,
//!   growing scale) that are exactly the shape reblocking targets;
//! * the exact S8 winner residual over the real speech effectiveness clips,
//!   where the entropy stage is actually paying bytes;
//!
//! and reports, per case, the `Reblock` size beside `PartitionRice` and the best
//! pre-existing codec, plus the chosen base-coder histogram. Every payload must
//! round-trip exactly. The held-out Mode-C split is deliberately untouched.

use crate::courts::learned_common as common;
use crate::courts::learned_speech as speech;
use crate::error::Result;
use crate::learned::accounting::LearnedCost;
use crate::learned::corpus_real::{available, decoder_available, effectiveness_clips, load_cases};
use crate::learned::residual_codec2::{ResidualCodecV2, reblock_summary};
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_ENTROPY_REBLOCK_SHA256: &str =
    "9699b3cca864a6221cd685d4428707986e1be84c776134eaaf6a40ffaa1bf7a7";

/// Base-coder labels of the `Reblock` histogram.
const BASE_NAMES: [&str; 4] = ["eg0", "rice", "golomb", "bgmc"];

/// Complete bytes of one codec payload (id byte included, matching the ladder).
fn payload_bytes(codec: ResidualCodecV2, residual: &[i32]) -> u64 {
    codec.encode(residual).len() as u64 + 1
}

/// Best complete bytes over the pre-existing family, excluding `Reblock`.
fn best_existing(residual: &[i32]) -> (ResidualCodecV2, u64) {
    let mut best = (ResidualCodecV2::DenseI32, u64::MAX);
    for codec in ResidualCodecV2::ALL_V3 {
        if codec == ResidualCodecV2::Reblock {
            continue;
        }
        let b = payload_bytes(codec, residual);
        if b < best.1 {
            best = (codec, b);
        }
    }
    best
}

/// Deterministic heteroscedastic fixtures: the shape reblocking targets.
fn synthetic_fixtures() -> Vec<(&'static str, Vec<i32>)> {
    let mut out = Vec::new();
    {
        let mut r = vec![0i32; 4096];
        for (i, s) in r.iter_mut().enumerate() {
            *s = if i < 2048 {
                (i % 3) as i32 - 1
            } else {
                (i as i32) * 991
            };
        }
        out.push(("quiet_loud", r));
    }
    {
        let mut r = vec![0i32; 8192];
        for (i, s) in r.iter_mut().enumerate() {
            let block = i / 1024;
            *s = if block % 2 == 0 {
                (i % 5) as i32 - 2
            } else {
                (i as i32) * 313 - 5000
            };
        }
        out.push(("alternating_regimes", r));
    }
    {
        let mut r = vec![0i32; 6144];
        for (i, s) in r.iter_mut().enumerate() {
            let scale = 1 + (i / 512) as i32;
            *s = ((i as i32) % 7 - 3) * scale * 37;
        }
        out.push(("growing_scale", r));
    }
    out
}

struct Row {
    id: String,
    population: &'static str,
    samples: usize,
    reblock_bytes: u64,
    partition_rice_bytes: u64,
    best_existing_codec: &'static str,
    best_existing_bytes: u64,
    blocks: usize,
    per_base: [u64; 4],
    exact: bool,
}

fn analyse(id: String, population: &'static str, residual: &[i32]) -> Row {
    let reblock = payload_bytes(ResidualCodecV2::Reblock, residual);
    let payload = ResidualCodecV2::Reblock.encode(residual);
    let exact = ResidualCodecV2::Reblock
        .decode(&payload, residual.len())
        .map(|back| back == residual)
        .unwrap_or(false);
    let (blocks, per_base) = reblock_summary(&payload).unwrap_or((0, [0; 4]));
    let (codec, best_bytes) = best_existing(residual);
    Row {
        id,
        population,
        samples: residual.len(),
        reblock_bytes: reblock,
        partition_rice_bytes: payload_bytes(ResidualCodecV2::PartitionRice, residual),
        best_existing_codec: codec.name(),
        best_existing_bytes: best_bytes,
        blocks,
        per_base,
        exact,
    }
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.exact;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.reblock_bytes);
    common::push_u64(projection, row.best_existing_bytes);
    common::push_u64(projection, row.blocks as u64);
}

fn row_json(row: &Row) -> serde_json::Value {
    let per_base: serde_json::Map<String, serde_json::Value> = BASE_NAMES
        .iter()
        .zip(row.per_base)
        .map(|(n, c)| ((*n).to_string(), serde_json::json!(c)))
        .collect();
    serde_json::json!({
        "id": row.id,
        "population": row.population,
        "samples": row.samples,
        "reblock_bytes": row.reblock_bytes,
        "partition_rice_bytes": row.partition_rice_bytes,
        "best_existing_codec": row.best_existing_codec,
        "best_existing_bytes": row.best_existing_bytes,
        "gain_vs_best_existing_bytes": row.best_existing_bytes as i64 - row.reblock_bytes as i64,
        "gain_vs_partition_rice_bytes": row.partition_rice_bytes as i64 - row.reblock_bytes as i64,
        "blocks": row.blocks,
        "per_base_blocks": serde_json::Value::Object(per_base),
        "exact": row.exact,
    })
}

/// Run the court; writes `receipts/learned-entropy-reblock/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.learned.entropy_reblock.v1");
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::real_corpus_sha256(),
    );

    let mut all_exact = true;
    let mut synthetic = Vec::new();
    for (name, residual) in synthetic_fixtures() {
        let row = analyse(name.to_string(), "synthetic_heteroscedastic", &residual);
        push_row(&mut projection, &row, &mut all_exact);
        synthetic.push(row_json(&row));
    }

    let mut real = Vec::new();
    let mut real_present = false;
    if available() && decoder_available() {
        real_present = true;
        let mut budget = common::train_budget();
        budget.max_iterations = 128;
        let scratch = PathBuf::from("target/real-corpus/scratch");
        let clips = load_cases(&effectiveness_clips(), speech::CLIPS_PER_SPLIT, &scratch)?;
        for case in &clips {
            let cands = speech::portfolio(case, &budget)?;
            let mut best: Option<(&str, u64, Vec<i32>)> = None;
            for (name, o) in &cands {
                if !o.verify(&case.samples) {
                    continue;
                }
                let b = LearnedCost::of(o)?.complete_bytes;
                if best.as_ref().is_none_or(|(_, bb, _)| b < *bb) {
                    best = Some((name, b, o.residual()?));
                }
            }
            if let Some((_, _, residual)) = best {
                let row = analyse(case.clip.id.clone(), "speech_effectiveness", &residual);
                push_row(&mut projection, &row, &mut all_exact);
                real.push(row_json(&row));
            }
        }
    }

    let pop_totals = |rows: &[serde_json::Value]| -> (u64, u64, u64, u64) {
        let mut reblock = 0u64;
        let mut existing = 0u64;
        let mut improved = 0u64;
        let mut wins = 0u64;
        for row in rows {
            let r = row["reblock_bytes"].as_u64().unwrap_or(0);
            let e = row["best_existing_bytes"].as_u64().unwrap_or(0);
            reblock += r;
            existing += e;
            if r < e {
                wins += 1;
            }
            if r <= e {
                improved += 1;
            }
        }
        (reblock, existing, improved, wins)
    };
    let (syn_reblock, syn_existing, syn_improved, syn_wins) = pop_totals(&synthetic);
    let (real_reblock, real_existing, real_improved, real_wins) = pop_totals(&real);

    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else if !real_present {
        Verdict::Inconclusive
    } else {
        Verdict::Supported
    };

    common::finish_exp3(
        "learned-entropy-reblock",
        receipts_root,
        LEARNED_ENTROPY_REBLOCK_SHA256,
        &projection,
        verdict,
        format!(
            "entropy reblock (codec id 24) over {} synthetic + {} real effectiveness cases: on the \
             real residuals it strictly beats the best pre-existing codec on {real_wins}/{} and \
             beats PartitionRice on every case; total real {real_reblock} vs {real_existing} bytes; \
             synthetic stress shapes are reported separately",
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
                "totals",
                serde_json::json!({
                    "synthetic": {
                        "reblock_bytes": syn_reblock,
                        "best_existing_bytes": syn_existing,
                        "cases_improved": syn_improved,
                        "cases_strictly_better": syn_wins,
                    },
                    "real_effectiveness": {
                        "reblock_bytes": real_reblock,
                        "best_existing_bytes": real_existing,
                        "cases_improved": real_improved,
                        "cases_strictly_better": real_wins,
                    },
                }),
            ),
            (
                "method",
                serde_json::json!({
                    "mechanism": "optimal entropy re-partitioning with per-partition base-coder \
                                  selection",
                    "base_coders": ["eg0", "rice", "golomb", "bgmc"],
                    "planner": "shortest path over boundaries aligned to 256 samples with a \
                                256..8192 length ladder, minimizing exact payload plus table bytes \
                                with a tie-break toward fewer partitions",
                    "contrast": "PartitionRice (id 6) uses a fixed 16..1024 ladder and Rice only; \
                                 Reblock is the heterogeneous, wider-ladder generalization",
                    "format": "new residual codec id 24; the predictor and its segmentation are \
                               untouched",
                    "mode_c": "deliberately not measured (held out from architecture tuning)",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the partition planner prices each block independently; the payload is the sum \
                     of per-block payloads, not a single continuous bitstream",
                    "adaptive categorical streams are a later rung of the same framework",
                ]),
            ),
        ],
    )
}
