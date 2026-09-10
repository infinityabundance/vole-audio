//! `court corpus` / `vole-audio corpus verify` — the flagship-corpus gate.
//!
//! Phase M freezes the corpus **before** any flagship result exists, and this
//! court is the executable proof that the frozen population is intact:
//!
//! * the embedded manifest parses and carries the frozen schema;
//! * its `corpus_sha256` covers exactly the objects it lists (identity bytes,
//!   not just sample bytes);
//! * the frozen membership (code) and the manifest agree in both directions —
//!   a missing or extra object is a failure;
//! * every object's class assignment, rate, channel count and frame count match
//!   the frozen identity;
//! * regenerating every object reproduces its canonical `i32` hash exactly.
//!
//! Nothing here is a performance claim. It is the gate that makes a later
//! B1/VOLE comparison state exactly which population, under which generator
//! parameters, produced it.

use crate::corpus::{Finding, VerifyReport};
use crate::error::Result;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

fn findings_json(findings: &[Finding]) -> Vec<serde_json::Value> {
    findings
        .iter()
        .map(|f| {
            serde_json::json!({
                "object": f.id,
                "kind": f.kind.as_str(),
                "detail": f.detail,
            })
        })
        .collect()
}

fn histogram(values: impl Iterator<Item = String>) -> serde_json::Value {
    let mut map: BTreeMap<String, usize> = BTreeMap::new();
    for v in values {
        *map.entry(v).or_insert(0) += 1;
    }
    serde_json::to_value(map).unwrap_or(serde_json::Value::Null)
}

/// Run the corpus gate; writes an immutable receipt under `receipts/corpus/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let manifest = crate::corpus::manifest()?;
    let report: VerifyReport = crate::corpus::verify_manifest(&manifest)?;
    let verdict = if report.ok() {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    // Class histograms are evidence about the population, not results about it.
    let by_source_structure = histogram(
        manifest
            .objects
            .iter()
            .map(|o| o.source_structure_class.clone()),
    );
    let by_amplitude = histogram(manifest.objects.iter().map(|o| o.amplitude_class.clone()));
    let by_channels = histogram(manifest.objects.iter().map(|o| o.channel_structure.clone()));
    let by_temporal = histogram(manifest.objects.iter().map(|o| o.temporal_class.clone()));
    let by_entropy = histogram(manifest.objects.iter().map(|o| o.entropy_class.clone()));
    let by_rate = histogram(
        manifest
            .objects
            .iter()
            .map(|o| o.sample_rate_hz.to_string()),
    );

    // The B1 denominator is derived from the format domain (channel count),
    // never read from the manifest's audit field.
    let (b1_comparable, b1_excluded) = crate::corpus::derived_b1_counts(&manifest.objects);

    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!("{} + corpus.v1", manifest.profile)),
        backend: Some("generated (deterministic regeneration + canonical hash)".into()),
        content_kind: Some("flagship corpus (frozen before results)".into()),
        ..Default::default()
    };

    let mut builder = ReceiptBuilder::new("corpus");
    builder
        .result(verdict)
        .result_detail(format!(
            "flagship corpus {} verified: {} objects regenerated and hash-matched, \
             {}/{} B1-comparable ({} excluded by format domain, derived from the \
             channel count); manifest sha256 {}",
            if report.ok() { "PASS" } else { "FAIL" },
            report.verified,
            b1_comparable,
            manifest.populations.whole_corpus_objects,
            b1_excluded,
            report.manifest_sha256,
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(report.manifest_sha256.clone()),
            corpus_hash: Some(report.corpus_sha256.clone()),
            ..Default::default()
        })
        .extra(
            "manifest",
            serde_json::json!({
                "schema": manifest.schema,
                "universe": manifest.universe,
                "profile": manifest.profile,
                "state": manifest.state,
                "manifest_sha256": report.manifest_sha256,
                "corpus_sha256": report.corpus_sha256,
                "objects": report.objects,
                "verified": report.verified,
            }),
        )
        .extra(
            "populations",
            serde_json::json!({
                "whole_corpus_objects": manifest.populations.whole_corpus_objects,
                "b1_comparable_objects": b1_comparable,
                "b1_excluded_objects": b1_excluded,
                "high_channel_stress_objects": manifest.populations.high_channel_stress_objects,
                "b1_domain_rule": "1..=8 channels are B1-comparable; >8 are \
                                   NOT_APPLICABLE_BY_FORMAT_DOMAIN and never enter a \
                                   B1-vs-VOLE aggregate",
                "derivation": "b1-comparable counts are derived from each object's channel \
                               count, not read from the manifest's audit field",
            }),
        )
        .extra("by_source_structure_class", by_source_structure)
        .extra("by_amplitude_class", by_amplitude)
        .extra("by_channel_structure", by_channels)
        .extra("by_temporal_class", by_temporal)
        .extra("by_entropy_class", by_entropy)
        .extra("by_sample_rate_hz", by_rate)
        .extra(
            "findings",
            serde_json::json!(findings_json(&report.findings)),
        )
        .extra(
            "real_audio_stratum",
            serde_json::json!({
                "status": "VACANT_DECLARED",
                "admission": "an external object is admitted only with source, license, \
                              original-content hash, an explicit ingest/conversion path and \
                              the canonical i32 hash; courts report external entries as \
                              NOT_AVAILABLE rather than inventing data",
                "consequence": "no production claim from this corpus rests on real \
                                recordings yet, and this receipt says so",
            }),
        )
        .extra("objects", serde_json::json!(manifest.objects.len()))
        .limitation(
            "objects are generated, never stored: verification regenerates each object from \
             the manifest's own generator parameters and requires an exact canonical i32 \
             hash match, so a mutated generator or a mutated object both fail",
        )
        .limitation(
            "this receipt is the corpus gate, not a performance result: it makes no claim \
             about size, speed or representation choice",
        )
        .limitation(
            "the corpus is deterministic generated material plus hostile adversarial \
             controls; the license-clean real-recording stratum is declared and currently \
             vacant (see `real_audio_stratum`)",
        );
    let (_, path) = builder.finish_write(receipts_root)?;

    println!("court corpus: {verdict}");
    println!(
        "  objects: {}  verified: {}  B1-comparable: {}  excluded (format domain): {}",
        report.objects, report.verified, b1_comparable, b1_excluded
    );
    println!("  manifest sha256: {}", report.manifest_sha256);
    println!("  corpus sha256:   {}", report.corpus_sha256);
    for f in &report.findings {
        println!("  FINDING {}: {} ({})", f.kind.as_str(), f.id, f.detail);
    }
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
