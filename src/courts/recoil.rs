//! `court recoil-checkpoints` — Phase 6 mechanism 11 (`RecoilCheckpoints`).
//!
//! A single rANS stream is sequential: the decoder replays it from the front.
//! `RecoilCheckpoints` records `(state, rANS byte position)` at chosen symbol
//! boundaries so disjoint ranges can be decoded independently — decoder-chosen
//! parallelism bought with an explicit, small metadata cost.
//!
//! This court builds a static frequency table over each fixture's magnitude
//! buckets, encodes the bucket stream, builds a checkpoint index at a fixed
//! interval, and then decodes every checkpoint's range **independently**. The
//! reassembled segments must equal the full decode exactly, and the checkpoint
//! index's byte cost is reported against the stream. No coded symbol changes.

use crate::courts::learned_common as common;
use crate::entropy::corpus;
use crate::entropy::recoil::{
    checkpoints, decode, decode_segment, encode, encode_checkpoints, normalize_histogram,
};
use crate::error::Result;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const RECOIL_CHECKPOINTS_SHA256: &str =
    "159e4714d817582e569608d90b0f02ca16ab7c4caec29fcda7ccff3c1b3a49ef";

/// Checkpoint interval in symbols.
const INTERVAL: usize = 256;

/// Alphabet: magnitude bit-length buckets `0..=33`.
const ALPHABET: usize = 34;

fn bucket_of(v: i64) -> usize {
    let m = v.unsigned_abs();
    if m == 0 {
        0
    } else {
        (64 - m.leading_zeros()) as usize
    }
}

struct Row {
    id: String,
    symbols: usize,
    stream_bytes: u64,
    checkpoint_count: u64,
    checkpoint_bytes: u64,
    overhead_permille: u64,
    segments_match: bool,
}

fn analyse(id: String, values: &[i64]) -> Result<Row> {
    let symbols: Vec<usize> = values.iter().map(|&v| bucket_of(v)).collect();
    let mut counts = vec![0u64; ALPHABET];
    for &s in &symbols {
        counts[s] += 1;
    }
    let table = normalize_histogram(&counts);
    let stream = encode(&symbols, &table)?;
    let full = decode(&stream, symbols.len(), &table)?;
    let mut segments_match = full == symbols;
    let cps = checkpoints(&stream, symbols.len(), &table, INTERVAL)?;
    let index = encode_checkpoints(&cps);
    let mut reassembled = Vec::with_capacity(symbols.len());
    for (w, cp) in cps.iter().enumerate() {
        let start = w * INTERVAL;
        let count = INTERVAL.min(symbols.len() - start);
        let seg = decode_segment(&stream, symbols.len(), start, count, &table, Some(cp))?;
        segments_match &= seg == full[start..start + count];
        reassembled.extend_from_slice(&seg);
    }
    segments_match &= reassembled == full;
    let overhead_permille = if stream.is_empty() {
        0
    } else {
        (index.len() as u64 * 1000) / stream.len() as u64
    };
    Ok(Row {
        id,
        symbols: symbols.len(),
        stream_bytes: stream.len() as u64,
        checkpoint_count: cps.len() as u64,
        checkpoint_bytes: index.len() as u64,
        overhead_permille,
        segments_match,
    })
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.segments_match;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.symbols as u64);
    common::push_u64(projection, row.stream_bytes);
    common::push_u64(projection, row.checkpoint_count);
    common::push_u64(projection, row.checkpoint_bytes);
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "symbols": row.symbols,
        "stream_bytes": row.stream_bytes,
        "checkpoint_count": row.checkpoint_count,
        "checkpoint_bytes": row.checkpoint_bytes,
        "overhead_permille": row.overhead_permille,
        "segments_match": row.segments_match,
    })
}

/// Run the court; writes `receipts/recoil-checkpoints/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.recoil_checkpoints.v1");

    let names = [
        "silence",
        "dc",
        "single-sine",
        "harmonic-tone",
        "quasi-periodic",
        "impulse-train",
        "am-signal",
        "white-noise",
    ];
    let mut rows = Vec::new();
    let mut all_exact = true;
    for name in names {
        let Some(fx) = corpus::named(name) else {
            continue;
        };
        let frames = fx.frames().min(2048);
        if frames == 0 {
            continue;
        }
        let values: Vec<i64> = fx.samples[..frames].iter().map(|&s| i64::from(s)).collect();
        let row = analyse(name.to_string(), &values)?;
        let mut gates = true;
        push_row(&mut projection, &row, &mut gates);
        all_exact &= gates;
        rows.push(row);
    }

    // A synthetic skewed source exercises a wider alphabet than audio silence.
    let synthetic: Vec<i64> = (0..4096i64)
        .map(|i| {
            let v = ((i * 2654435761) % 100003) - 50001;
            if i % 13 == 0 { v / 256 } else { v }
        })
        .collect();
    let row = analyse("synthetic_skewed".to_string(), &synthetic)?;
    let mut gates = true;
    push_row(&mut projection, &row, &mut gates);
    all_exact &= gates;
    rows.push(row);

    let total_stream: u64 = rows.iter().map(|r| r.stream_bytes).sum();
    let total_index: u64 = rows.iter().map(|r| r.checkpoint_bytes).sum();
    let total_workers: u64 = rows.iter().map(|r| r.checkpoint_count).sum();
    let overhead_permille = total_index
        .checked_mul(1000)
        .and_then(|x| x.checked_div(total_stream))
        .unwrap_or(0);
    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish(
        "recoil-checkpoints",
        receipts_root,
        RECOIL_CHECKPOINTS_SHA256,
        &projection,
        verdict,
        format!(
            "checkpointed static rANS over {n} fixtures: {w} independent decode ranges reassemble \
             the full decode exactly; checkpoint index {ti} B on {ts} B of stream ({pk} per-mille \
             overhead) buys decoder-chosen parallelism at interval {iv}",
            n = rows.len(),
            w = total_workers,
            ti = total_index,
            ts = total_stream,
            pk = overhead_permille,
            iv = INTERVAL,
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "fixtures": rows.len(),
                }),
            ),
            ("fixtures", rows_json),
            (
                "method",
                serde_json::json!({
                    "checkpoint": "symbol_index, rANS decoder state and reader byte position at a \
                                   fixed symbol interval",
                    "claim": "each recorded range decodes independently and the reassembled \
                              segments equal the full sequential decode exactly",
                    "cost": "the checkpoint index is explicit metadata; the coded symbols are \
                             unchanged, so this is a size-for-speed trade, not a ratio claim",
                    "contrast": "an independent per-block rANS stream pays four state bytes per \
                                 block and loses cross-block modelling; a checkpoint pays a \
                                 fraction of that and leaves the stream intact",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the checkpoints are built by a full decode pass at encode time; the payoff is \
                     at decode time",
                    "the static table is order-0 here; an adaptive model would also need its state \
                     checkpointed, which this mechanism does not attempt",
                ]),
            ),
        ],
    )
}
