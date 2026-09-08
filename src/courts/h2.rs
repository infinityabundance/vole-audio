//! `court h2` — aggregate Phase H.2 seal (H.2.32, H.2.50).
//!
//! Runs every Phase H.2 court in sequence against the same receipts root
//! (each sub-court writes its own immutable receipt) and writes one aggregate
//! receipt recording each sub-verdict. The aggregate is SUPPORTED only when
//! every H.2 court is SUPPORTED; any negative sub-verdict is preserved and
//! reported (negative results are evidence, never deleted).
//!
//! Sub-courts (in charter order):
//! `entropy-rans`, `entropy-literal`, `entropy-residual`, `entropy-pages`,
//! `entropy-partial`, `entropy-simd`, `entropy-cuda`, `entropy-d1`,
//! `entropyfs`, `dsfb-entropy`.
//!
//! `entropyfs` and `dsfb-entropy` report INCONCLUSIVE + a limitation when
//! their optional features are off; the aggregate then reports the honest
//! INCONCLUSIVE with the reason (run with `--all-features` for the seal).

use crate::error::Result;
use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::status::Verdict;
use std::path::Path;

const H2_COURTS: &[&str] = &[
    "entropy-rans",
    "entropy-literal",
    "entropy-residual",
    "entropy-pages",
    "entropy-partial",
    "entropy-simd",
    "entropy-cuda",
    "entropy-d1",
    "entropyfs",
    "dsfb-entropy",
];

/// Court list for this run: `VOLE_H2_COURTS` (comma separated) overrides the
/// full sequence — used for partial re-seals and bisecting aggregate runs.
fn court_list() -> Vec<String> {
    if let Ok(over) = std::env::var("VOLE_H2_COURTS") {
        return over
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
    }
    H2_COURTS.iter().map(|s| s.to_string()).collect()
}

/// Run the court; writes an immutable receipt under `receipts/h2/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut worst: Option<Verdict> = None;
    let mut operational_error: Option<String> = None;
    let courts = court_list();

    for name in courts {
        println!("== court {name} (aggregate h2) ==");
        let v = match crate::courts::run(&name, receipts_root) {
            Ok(v) => v,
            Err(e) => {
                operational_error = Some(format!("{name}: {e}"));
                break;
            }
        };
        rows.push(serde_json::json!({ "court": name, "verdict": v.label() }));
        worst = Some(match worst {
            None => v,
            Some(w) => worst_of(w, v),
        });
    }

    if let Some(e) = operational_error {
        let mut b = ReceiptBuilder::new("h2");
        b.result(Verdict::Inconclusive)
            .result_detail(format!("h2 aggregate aborted on operational error: {e}"))
            .extra("rows", serde_json::Value::Array(rows));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court h2: INCONCLUSIVE ({e})");
        eprintln!("  receipt: {}", path.display());
        return Ok(Verdict::Inconclusive);
    }

    let verdict = worst.unwrap_or(Verdict::Inconclusive);
    let mut b = ReceiptBuilder::new("h2");
    let rows_json = serde_json::Value::Array(rows.clone());
    b.result(verdict)
        .result_detail(format!(
            "aggregate Phase H.2 seal: {} courts run; verdicts below",
            rows.len()
        ))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1 + vole.entropy.p1/p1/v1".into()),
            content_kind: Some("phase-h2 aggregate".into()),
            ..Default::default()
        })
        .extra("rows", rows_json);
    if verdict != Verdict::Supported {
        b.limitation(
            "aggregate reflects every sub-court verdict; run with --all-features on \
             supported hardware for the full seal",
        );
    }
    let (_, path) = b.finish_write(receipts_root)?;
    println!("court h2: {verdict}");
    for r in &rows {
        println!(
            "  {}: {}",
            r["court"].as_str().unwrap_or("?"),
            r["verdict"].as_str().unwrap_or("?")
        );
    }
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

/// Deterministic worst-verdict ordering (a negative dominates).
fn worst_of(a: Verdict, b: Verdict) -> Verdict {
    let rank = |v: Verdict| match v {
        Verdict::FailedCorrectness => 9,
        Verdict::FailedDeadline => 8,
        Verdict::FellBackToD0 => 7,
        Verdict::UnsupportedByHardware => 6,
        Verdict::UnsupportedByTopology => 5,
        Verdict::UnsupportedByApi => 4,
        Verdict::NotImplemented => 3,
        Verdict::NotApplicable => 2,
        Verdict::Inconclusive => 1,
        Verdict::Supported => 0,
    };
    if rank(a) >= rank(b) { a } else { b }
}
