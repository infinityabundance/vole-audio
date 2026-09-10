//! `court phase-n` — the Phase-N aggregate (contract §52).
//!
//! Runs every Phase-N court in sequence and is `SUPPORTED` only when all of them
//! are. Like the `all` (Phase M) and `h2` aggregates, it re-runs its sub-courts
//! so the aggregate receipt is backed by fresh sub-court receipts rather than by
//! memory of a previous run.

use crate::error::Result;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::hash::sha256::{Sha256, hex};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const PHASE_N_RESULT_SHA256: &str =
    "92df23d2b35338764986216a6a7d4904782d13c55be641c98d5e376b6d12b79a";

/// The frozen aggregate identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.phase_n.protocol.v1";

/// The Phase-N courts this aggregate covers, in execution order.
pub const PHASE_N_COURTS: [&str; 2] = ["archive", "transport"];

/// Run the aggregate; writes `receipts/phase-n/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut verdicts: Vec<(String, Verdict)> = Vec::with_capacity(PHASE_N_COURTS.len());
    let mut all_supported = true;
    for name in PHASE_N_COURTS {
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
    if PHASE_N_RESULT_SHA256.is_empty() {
        eprintln!("court phase-n: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != PHASE_N_RESULT_SHA256 {
        let mut b = ReceiptBuilder::new("phase-n");
        b.result(Verdict::FailedCorrectness).result_detail(format!(
            "aggregate static result hash changed: frozen {PHASE_N_RESULT_SHA256}, observed {result_hex}"
        ));
        b.finish_write(receipts_root)?;
        eprintln!("court phase-n: FAILED_CORRECTNESS (static result hash changed)");
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
    let mut builder = ReceiptBuilder::new("phase-n");
    builder
        .result(verdict)
        .result_detail(format!(
            "Phase-N aggregate: {} of {} sub-courts SUPPORTED; result sha256 {result_hex}",
            verdicts
                .iter()
                .filter(|(_, v)| *v == Verdict::Supported)
                .count(),
            verdicts.len(),
        ))
        .params(CourtParams {
            content_kind: Some("Phase-N aggregate over its member courts".into()),
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
                "courts": PHASE_N_COURTS,
                "verdicts": breakdown,
            }),
        )
        .limitation(
            "an aggregate is only as strong as its members: the archive court proves container \
             canonicality and the transport court proves framing/state-machine/recovery \
             determinism, but neither is a network stack and neither claims distributed transport",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court phase-n: {verdict}");
    for (n, v) in &verdicts {
        println!("  {n}: {v}");
    }
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
