//! `court facts` — independent semantic-oracle coverage (Phase G prerequisite).
//!
//! Differential parity (`scalar == SIMD == CUDA == ROCm`) proves *backend*
//! equality, but a bug shared by every backend survives it. Each semantic fact
//! (`crate::facts`, ids F01…) is therefore an independent statement about
//! `vole.audio.u1`: its expected samples are derived from first principles
//! (closed-form integer math, hand-enumerated boundary sequences, or vectors
//! produced by an independent implementation) and observed through the full
//! `World` path on every available host surface.
//!
//! The court verdict is `SUPPORTED` only when every fact passes on every
//! surface (scalar authority + every runtime-available SIMD floor); otherwise
//! it writes an honest `FAILED_CORRECTNESS` receipt. Needs no GPU/audio
//! hardware. Adding a representation/transform requires adding a fact — see
//! docs/SEMANTIC_FACTS.md.

use crate::evidence::receipt::{CourtParams, ReceiptBuilder, RunTiming};
use crate::status::Verdict;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

/// Run the court; writes an immutable receipt under `receipts/facts/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let t0 = Instant::now();
    let results = crate::facts::run_facts()?;

    let mut rows_total = 0usize;
    let mut rows_passed = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut per_fact: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for r in &results {
        let mut rows = Vec::new();
        for row in &r.rows {
            rows_total += 1;
            if row.passed {
                rows_passed += 1;
            } else {
                failures.push(format!(
                    "{} ({}): {}",
                    r.fact_id,
                    row.surface,
                    row.detail.as_deref().unwrap_or("no detail")
                ));
            }
            rows.push(json!({
                "surface": row.surface,
                "passed": row.passed,
                "detail": row.detail,
            }));
        }
        per_fact.insert(format!("{}/{}", r.fact_id, r.fact_name), json!(rows));
    }

    let passed = failures.is_empty();
    let verdict = if passed {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    let detail = if passed {
        format!(
            "all {} facts passed on every surface ({} rows: scalar authority + SIMD floors)",
            results.len(),
            rows_total
        )
    } else {
        format!(
            "{}/{} fact rows failed ({}): {}",
            failures.len(),
            rows_total,
            results.len(),
            failures.join("; ")
        )
    };

    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1".into()),
        backend: Some("facts".into()),
        sample_rate_hz: Some(48_000),
        channels: None,
        content_kind: Some("semantic-facts-independent-oracle-battery".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("facts");
    builder
        .result(verdict)
        .result_detail(detail)
        .params(params)
        .timing(RunTiming {
            total_ns: Some(t0.elapsed().as_nanos() as i64),
            ..Default::default()
        });
    for (k, v) in per_fact {
        builder.extra(format!("fact/{k}"), v);
    }
    let (_, path) = builder.finish_write(receipts_root)?;

    println!("court facts: {verdict}");
    println!(
        "  facts: {}   rows: {rows_passed}/{rows_total}",
        results.len()
    );
    for r in &results {
        let marks: Vec<&str> = r
            .rows
            .iter()
            .map(|row| if row.passed { "pass" } else { "FAIL" })
            .collect();
        println!("    {} {:<36} {}", r.fact_id, r.fact_name, marks.join(", "));
    }
    if !passed {
        for f in &failures {
            eprintln!("    failed: {f}");
        }
    }
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
