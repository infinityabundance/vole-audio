//! `court page-batch-simd` — Phase 6 mechanism 12 (`PageBatchSIMD`).
//!
//! A canonical page is an independent rANS-coded unit with two implicit
//! alternating states. Because a page is self-contained, decoding several pages
//! at once is a host runtime choice: **no ISA width enters the format**, and the
//! same bytes decode identically sequentially or batched.
//!
//! This court splits each fixture's magnitude-bucket stream into canonical
//! pages, encodes each page with the two-state coder, then decodes the whole set
//! in batches of 4, 8 and 16 lanes with a lockstep lane loop — the exact
//! ordering a SIMD page-batch kernel must preserve — and requires the batched
//! result to equal the sequential page-by-page decode exactly on every lane
//! count.

use crate::courts::learned_common as common;
use crate::entropy::corpus;
use crate::entropy::page_batch::{PAGE_LANE_LADDER, decode_batch, decode_page, encode_page};
use crate::entropy::recoil::normalize_histogram;
use crate::error::Result;
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const PAGE_BATCH_SIMD_SHA256: &str =
    "524ec45fb08f466c7c5976bec926b69959110c444697ee57871b23ce8e78b9cf";

/// Canonical page size in symbols.
const PAGE_SYMBOLS: usize = 512;

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
    pages: usize,
    stream_bytes: u64,
    lanes_agree: Vec<u64>,
    all_agree: bool,
}

fn analyse(id: String, values: &[i64]) -> Result<Row> {
    let symbols: Vec<usize> = values.iter().map(|&v| bucket_of(v)).collect();
    let mut counts = vec![0u64; ALPHABET];
    for &s in &symbols {
        counts[s] += 1;
    }
    let table = normalize_histogram(&counts);

    let pages: Vec<Vec<usize>> = symbols
        .chunks(PAGE_SYMBOLS)
        .map(<[usize]>::to_vec)
        .collect();
    let encoded: Vec<Vec<u8>> = pages
        .iter()
        .map(|p| encode_page(p, &table))
        .collect::<Result<_>>()?;
    let sequential: Vec<Vec<usize>> = encoded
        .iter()
        .zip(&pages)
        .map(|(b, p)| decode_page(b, p.len(), &table))
        .collect::<Result<_>>()?;
    let refs: Vec<(&[u8], usize)> = encoded
        .iter()
        .zip(&pages)
        .map(|(b, p)| (b.as_slice(), p.len()))
        .collect();

    let mut lanes_agree = Vec::new();
    let mut all_agree = true;
    for lanes in PAGE_LANE_LADDER {
        let batched = decode_batch(&refs, &table, lanes)?;
        let agree = batched == sequential;
        all_agree &= agree;
        lanes_agree.push(if agree { lanes as u64 } else { 0 });
    }
    let stream_bytes: u64 = encoded.iter().map(|b| b.len() as u64).sum();
    Ok(Row {
        id,
        symbols: symbols.len(),
        pages: pages.len(),
        stream_bytes,
        lanes_agree,
        all_agree,
    })
}

fn push_row(projection: &mut Vec<u8>, row: &Row, gates: &mut bool) {
    *gates &= row.all_agree;
    common::push_label(projection, &row.id);
    common::push_u64(projection, row.symbols as u64);
    common::push_u64(projection, row.pages as u64);
    common::push_u64(projection, row.stream_bytes);
    for l in &row.lanes_agree {
        common::push_u64(projection, *l);
    }
}

fn row_json(row: &Row) -> serde_json::Value {
    serde_json::json!({
        "id": row.id,
        "symbols": row.symbols,
        "pages": row.pages,
        "stream_bytes": row.stream_bytes,
        "lanes_agree": row.lanes_agree,
        "all_agree": row.all_agree,
    })
}

/// Run the court; writes `receipts/page-batch-simd/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.page_batch_simd.v1");

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

    let total_pages: u64 = rows.iter().map(|r| r.pages as u64).sum();
    let total_symbols: u64 = rows.iter().map(|r| r.symbols as u64).sum();
    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else {
        Verdict::Supported
    };

    let rows_json: serde_json::Value = rows.iter().map(row_json).collect();
    common::finish(
        "page-batch-simd",
        receipts_root,
        PAGE_BATCH_SIMD_SHA256,
        &projection,
        verdict,
        format!(
            "two-state canonical rANS pages over {n} fixtures: {tp} pages / {ts} symbols decode \
             identically sequentially and batched at every lane count in {ladder:?}; no ISA width \
             is present in the format",
            n = rows.len(),
            tp = total_pages,
            ts = total_symbols,
            ladder = PAGE_LANE_LADDER,
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
                    "page": "an independent rANS unit; symbol i is coded by state i & 1 and the \
                             two states renormalize into one shared byte stream",
                    "batching": "pages are decoded in batches of 4/8/16 lanes with a lockstep lane \
                                 loop; each lane is self-contained",
                    "format": "no ISA width is encoded: the same page bytes decode identically \
                               sequentially or batched, and the lane count is a host runtime choice",
                    "contrast": "a single-state stream is strictly serial; two states recover most \
                                 of the dependency at no format cost",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the batch loop is a scalar simulation of the lane semantics, not a measured \
                     SIMD speedup; the court gates equivalence, not throughput",
                    "this makes no compression-ratio claim: page independence can cost a little \
                     ratio versus one long stream",
                ]),
            ),
        ],
    )
}
