//! `court all` — the Phase-M aggregate (contract §46).
//!
//! Runs every Phase-M court in sequence and is `SUPPORTED` only when all of them
//! are. Like the H.2 aggregate, it re-runs its sub-courts so the aggregate
//! receipt is backed by fresh sub-court receipts, not by memory of a previous
//! run.

use crate::error::Result;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::hash::sha256::{Sha256, hex};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const ALL_RESULT_SHA256: &str =
    "c3d4a47c38803f16bb38146d6364785bf305f5e835e777a4db03b99d11e9ebd5";

/// The frozen aggregate identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.all.protocol.v1";

/// The Phase-M courts this aggregate covers, in execution order.
pub const PHASE_M_COURTS: [&str; 9] = [
    "conventional",
    "corpus",
    "fullobj",
    "flagship",
    "runtime",
    "random-access",
    "negative",
    "depth",
    "interference",
];

/// Run the aggregate; writes `receipts/all/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut verdicts: Vec<(String, Verdict)> = Vec::with_capacity(PHASE_M_COURTS.len());
    let mut all_supported = true;
    for name in PHASE_M_COURTS {
        let verdict = super::run(name, receipts_root)?;
        all_supported &= verdict == Verdict::Supported;
        verdicts.push((name.to_string(), verdict));
    }

    let mut projection = Vec::new();
    projection.extend_from_slice(PROTOCOL_SCHEMA.as_bytes());
    projection.push(0);
    for (name, v) in &verdicts {
        projection.extend_from_slice(name.as_bytes());
        projection.push(0);
        projection.extend_from_slice(v.label().as_bytes());
        projection.push(0);
    }
    let result_hex = hex(&Sha256::digest(&projection));
    if ALL_RESULT_SHA256.is_empty() {
        eprintln!("court all: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != ALL_RESULT_SHA256 {
        let mut b = ReceiptBuilder::new("all");
        b.result(Verdict::FailedCorrectness).result_detail(format!(
            "aggregate static result hash changed: frozen {ALL_RESULT_SHA256}, observed {result_hex}"
        ));
        b.finish_write(receipts_root)?;
        eprintln!("court all: FAILED_CORRECTNESS (static result hash changed)");
        return Ok(Verdict::FailedCorrectness);
    }

    let verdict = if all_supported {
        Verdict::Supported
    } else {
        Verdict::Inconclusive
    };
    let breakdown: Vec<serde_json::Value> = verdicts
        .iter()
        .map(|(n, v)| serde_json::json!({ "court": n, "verdict": v.label() }))
        .collect();
    let mut builder = ReceiptBuilder::new("all");
    builder
        .result(verdict)
        .result_detail(format!(
            "Phase-M aggregate: {} of {} sub-courts SUPPORTED; result sha256 {result_hex}",
            verdicts
                .iter()
                .filter(|(_, v)| *v == Verdict::Supported)
                .count(),
            verdicts.len(),
        ))
        .params(CourtParams {
            content_kind: Some("Phase-M aggregate over its member courts".into()),
            ..Default::default()
        })
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
            ..Default::default()
        })
        .extra(
            "aggregate",
            serde_json::json!({
                "protocol_schema": PROTOCOL_SCHEMA,
                "courts": PHASE_M_COURTS,
                "verdicts": breakdown,
            }),
        )
        .limitation(
            "an aggregate is only as strong as its members: a member that reports an honest negative \
             (for example ROCm hardware-unavailable) is not an aggregate failure, so the aggregate is \
             SUPPORTED when every member is SUPPORTED and otherwise INCONCLUSIVE",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court all: {verdict}");
    for (n, v) in &verdicts {
        println!("  {n}: {v}");
    }
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
