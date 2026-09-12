//! `court pivot-side-streams` — Phase 6 mechanism 15 (`PivotSideStreams`).
//!
//! Small high-volume side streams (model ids, expert selectors, scale classes,
//! MRU tokens) are commonly Huffman-coded with a serial tree walk. PivCo-style
//! coding stores the same prefix code but reorders its bits by tree level, so
//! each level is a flat pass over the still-active symbols that a vector unit
//! can partition. The layout is a deterministic permutation of the code bits and
//! carries no ISA width.
//!
//! This court builds side streams from the frozen fixtures, derives a canonical
//! Huffman codebook, encodes each stream in both the serial and the
//! level-transposed layouts, and requires both decoders to recover the stream
//! exactly. It reports sizes, alphabet and code depth: a layout/throughput
//! experiment, not a compression-ratio claim (the code bits are identical).

use crate::courts::learned_common as common;
use crate::entropy::corpus;
use crate::entropy::pivot::{
    decode_pivot, decode_serial, encode_pivot, encode_serial, huffman_codebook,
};
use crate::error::Result;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const PIVOT_SIDE_STREAMS_SHA256: &str =
    "5637779218d4350f3fc08c849e687864c517d559f6d117c58e6a3f70b906a7b2";

/// Side-stream alphabet: `bucket mod 8` of the sample magnitude.
const ALPHABET: usize = 8;

fn symbol_of(v: i64) -> usize {
    let m = v.unsigned_abs();
    let bucket = if m == 0 {
        0
    } else {
        (64 - m.leading_zeros()) as usize
    };
    bucket % ALPHABET
}

struct Row {
    id: String,
    symbols: usize,
    max_len: u8,
    serial_bytes: u64,
    pivot_bytes: u64,
    serial_ok: bool,
    pivot_ok: bool,
}

fn analyse(id: String, values: &[i64]) -> Result<Row> {
    let symbols: Vec<usize> = values.iter().map(|&v| symbol_of(v)).collect();
    let mut counts = vec![0u64; ALPHABET];
    for &s in &symbols {
        counts[s] += 1;
    }
    let cb = huffman_codebook(&counts);
    let serial = encode_serial(&symbols, &cb)?;
    let pivot = encode_pivot(&symbols, &cb)?;
    let serial_ok = decode_serial(&serial, symbols.len(), &cb)? == symbols;
    let pivot_ok = decode_pivot(&pivot, symbols.len(), &cb)? == symbols;
    Ok(Row {
        id,
        symbols: symbols.len(),
        max_len: cb.max_len,
        serial_bytes: serial.len() as u64,
        pivot_bytes: pivot.len() as u64,
        serial_ok,
        pivot_ok,
    })
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.serial_ok && row.pivot_ok;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.symbols as u64);
    common::push_u64(projection, u64::from(row.max_len));
    common::push_u64(projection, row.serial_bytes);
    common::push_u64(projection, row.pivot_bytes);
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "symbols": row.symbols,
        "max_code_len": row.max_len,
        "serial_bytes": row.serial_bytes,
        "pivot_bytes": row.pivot_bytes,
        "serial_ok": row.serial_ok,
        "pivot_ok": row.pivot_ok,
    })
}

/// Run the court; writes `receipts/pivot-side-streams/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.pivot_side_streams.v1");

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
        let n = fx.frames().min(4096);
        if n == 0 {
            continue;
        }
        let values: Vec<i64> = fx.samples[..n].iter().map(|&s| i64::from(s)).collect();
        let row = analyse(name.to_string(), &values)?;
        let mut gates = true;
        push_row(&mut projection, &row, &mut gates);
        all_exact &= gates;
        rows.push(row);
    }

    // A synthetic skewed side stream exercises deep codes.
    let synthetic: Vec<i64> = (0..8192i64)
        .map(|i| {
            let v = ((i * 2654435761) % 100003) - 50001;
            if i % 17 == 0 { v } else { v / 4096 }
        })
        .collect();
    let row = analyse("synthetic_skewed".to_string(), &synthetic)?;
    let mut gates = true;
    push_row(&mut projection, &row, &mut gates);
    all_exact &= gates;
    rows.push(row);

    let total_symbols: u64 = rows.iter().map(|r| r.symbols as u64).sum();
    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish(
        "pivot-side-streams",
        receipts_root,
        PIVOT_SIDE_STREAMS_SHA256,
        &projection,
        verdict,
        format!(
            "level-transposed prefix coding over {n} side streams / {ts} symbols: serial and \
             level-transposed decoders both recover every stream exactly, and the layouts carry \
             the same code bits (no ratio claim); the transposed layout makes each tree level a \
             flat partition pass",
            n = rows.len(),
            ts = total_symbols,
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
                    "codebook": "canonical Huffman code from the stream's frequencies, with a \
                                 deterministic tie-break and codes assigned by (length, symbol)",
                    "layouts": "serial concatenates each symbol's code bits; the level-transposed \
                                layout stores every first bit, then every active symbol's second \
                                bit, and so on",
                    "decoder": "the transposed decoder makes one flat pass per level and resolves a \
                                position from the prefix code itself; no symbol sequence is needed \
                                and there is no per-symbol pointer chase",
                    "format": "the two layouts contain the same code bits in a different order, so \
                               no ISA width is encoded",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the level pass is a scalar simulation of the partition semantics, not a \
                     measured SIMD speedup; the court gates decode equivalence and reports sizes",
                    "the codebook is derived here from the stream's own frequencies; a production \
                     side stream would transmit it (or use a frozen codebook)",
                ]),
            ),
        ],
    )
}
