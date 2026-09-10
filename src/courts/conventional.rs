//! `court conventional` — Phase M flagship conventional baselines (contract §47).
//!
//! The canonical comparison must include the full `B0`–`B9` ladder. This court
//! owns the **conventional** rows over the **frozen flagship corpus** and
//! compares them against the exact same information the VOLE representation is
//! priced against — the canonical interleaved `i32` sample domain, with no
//! conversion of any kind:
//!
//! ```text
//!                     same exact i32 source
//!                              │
//!         ┌────────────────────┼────────────────────┐
//!         ▼                    ▼                    ▼
//!      B0 raw PCM          B1 FLAC (32-bit)     u1 literal
//!         │                    │                    │
//!   no compression      conventional exact     universal fallback
//!         │                    │                    │
//!         └────────────────────┼────────────────────┘
//!                              ▼
//!          size / ratio / encode+decode time / exactness
//! ```
//!
//! This is the **flagship B0/B1 conventional-baseline result**, not yet the
//! B1-vs-VOLE comparison: the third row is the canonical `u1` *literal* (the
//! universal fallback representation), not the inverse compiler's **selected**
//! VOLE representation for the object. That comparison needs an exact
//! full-object inverse-compilation container, which is a later increment.
//!
//! Three things this court refuses to do:
//!
//! * it never converts the source. The historical H.2 comparator (an external
//!   `flac` on a `>> 8` s24 conversion) stays frozen in
//!   `courts::entropy_common` as its own historical row; B1 is a *new*, exact,
//!   32-bit, in-process baseline. A B1 row that used the low 8 bits of the
//!   canonical domain for nothing would not be the same information problem;
//! * it never measures an unproven corpus: the frozen manifest is verified
//!   (canonical object comparison, membership, order, derived B1 eligibility)
//!   before a single baseline runs;
//! * it never lets a row disappear. `B2`–`B9` are present in the ladder
//!   manifest with their status, the five `>8`-channel objects appear explicitly
//!   as `NOT_APPLICABLE_BY_FORMAT_DOMAIN`, and every object the frozen
//!   membership defines is represented.

use crate::baseline::{
    B1_LEVEL_CONTROLS, B1_LEVEL_PRIMARY, FlacEncoding, b0_raw_pcm_bytes, b1_flac, b1_level_label,
    reference_flac,
};
use crate::corpus::{self};
use crate::error::{Error, Result};
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::hash::sha256::{Sha256, hex};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

/// Frozen static-result hash: the court fails if a change silently alters the
/// conventional-baseline results. Re-freeze only with a documented reason.
///
/// This binds the flagship (Phase-M) corpus population. The historical Seal-1
/// value over the frozen H.2 entropy corpus (`11f8683f…`) remains recorded in
/// that seal's receipts and ledger; this court now measures the flagship corpus.
pub const CONVENTIONAL_RESULT_SHA256: &str =
    "acfdaa32b9b69cdc0ee3ab2c0fb10387233603bc7a13d816575e12f7d2988b9b";

/// `B1` applicability status for an object inside FLAC's channel domain.
const B1_MEASURED: &str = "MEASURED";
/// `B1` applicability status for an object outside FLAC's format domain.
const B1_NOT_APPLICABLE: &str = "NOT_APPLICABLE_BY_FORMAT_DOMAIN";

/// Static identity of one object's conventional rows (no measured quantity), plus
/// the manifest/corpus binding. Order is significant: the object sequence *is*
/// part of the identity.
fn static_projection(manifest_sha: &str, corpus_sha: &str, cells: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for head in [manifest_sha, corpus_sha] {
        out.extend_from_slice(head.as_bytes());
        out.push(0);
    }
    for c in cells {
        out.extend_from_slice(c["id"].as_str().unwrap_or("").as_bytes());
        out.push(0);
        out.extend_from_slice(c["canonical_i32_sha256"].as_str().unwrap_or("").as_bytes());
        out.push(0);
        for key in [
            "sample_rate_hz",
            "channels",
            "frames",
            "b0_bytes",
            "u1_literal_bytes",
        ] {
            out.extend_from_slice(&c[key].as_u64().unwrap_or(u64::MAX).to_le_bytes());
        }
        let applicable = c["b1_status"].as_str() == Some(B1_MEASURED);
        out.push(applicable as u8);
        for key in [
            "b1_primary_bytes",
            "b1_control_0_bytes",
            "b1_control_8_bytes",
        ] {
            out.extend_from_slice(&c[key].as_u64().unwrap_or(u64::MAX).to_le_bytes());
        }
    }
    out
}

fn b1_cell(e: &FlacEncoding) -> serde_json::Value {
    serde_json::json!({
        "level": e.level,
        "label": b1_level_label(e.level),
        "bytes": e.encoded_bytes,
        "bits_per_sample": e.bits_per_sample,
        "source_bytes": e.source_bytes,
        "ratio_vs_source": e.ratio_vs_source(),
        "encode_ns": e.encode_ns,
        "decode_ns": e.decode_ns,
        "source_sha256": hex(&e.source_sha256),
        "decoded_sha256": hex(&e.decoded_sha256),
        "exact_roundtrip": e.exact_roundtrip,
        "md5_ok": e.md5_ok,
    })
}

/// Per-axis breakdown, over the `B1`-comparable subset for every byte figure.
fn axis_breakdown(cells: &[serde_json::Value], key: &str) -> serde_json::Value {
    #[derive(Default)]
    struct Agg {
        objects: usize,
        b1: usize,
        b0_bytes: u64,
        b1_bytes: u64,
    }
    let mut map: BTreeMap<String, Agg> = BTreeMap::new();
    for c in cells {
        let class = c[key]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| c[key].to_string());
        let e = map.entry(class).or_default();
        e.objects += 1;
        if c["b1_status"].as_str() == Some(B1_MEASURED) {
            e.b1 += 1;
            e.b0_bytes += c["b0_bytes"].as_u64().unwrap_or(0);
            e.b1_bytes += c["b1_primary_bytes"].as_u64().unwrap_or(0);
        }
    }
    let mut out = serde_json::Map::new();
    for (class, v) in map {
        out.insert(
            class,
            serde_json::json!({
                "objects": v.objects,
                "b1_comparable_objects": v.b1,
                "b1_excluded_objects": v.objects - v.b1,
                "b0_bytes_b1_comparable": v.b0_bytes,
                "b1_primary_bytes": v.b1_bytes,
                "ratio_b1_primary_vs_b0": if v.b0_bytes > 0 {
                    serde_json::json!(v.b1_bytes as f64 / v.b0_bytes as f64)
                } else {
                    serde_json::Value::Null
                },
            }),
        );
    }
    serde_json::Value::Object(out)
}

/// The `B0`–`B9` ladder manifest: every row visible, none deleted.
fn ladder_manifest() -> serde_json::Value {
    serde_json::json!([
        {"id": "B0", "baseline": "literal PCM", "status": "MEASURED",
         "where": "court conventional (this receipt)"},
        {"id": "B1", "baseline": "conventional lossless codec (FLAC, 32-bit, level 5)",
         "status": "MEASURED", "where": "court conventional (this receipt)"},
        {"id": "B2", "baseline": "PCM-resident sampler", "status": "NOT_IMPLEMENTED",
         "where": "Phase M"},
        {"id": "B3", "baseline": "disk-streaming sampler", "status": "NOT_IMPLEMENTED",
         "where": "Phase M", "must_record": ["preload", "storage type", "filesystem",
         "cache state", "read traffic", "seek behavior", "underruns"]},
        {"id": "B4", "baseline": "conventional compressed-file decode + playback",
         "status": "NOT_IMPLEMENTED", "where": "Phase M"},
        {"id": "B5", "baseline": "VOLE scalar (selected representation, full object)",
         "status": "NOT_IMPLEMENTED", "where": "requires the full-object inverse container"},
        {"id": "B6", "baseline": "VOLE CUDA buffered", "status": "MEASURED_ELSEWHERE",
         "where": "court cuda"},
        {"id": "B7", "baseline": "VOLE ROCm buffered", "status": "MEASURED_ELSEWHERE",
         "where": "court rocm-d0 (hardware-gated)"},
        {"id": "B8", "baseline": "VOLE D1 attempt", "status": "MEASURED_ELSEWHERE",
         "where": "court d1 / court entropy-d1"},
        {"id": "B9", "baseline": "VOLE D2 attempt", "status": "NOT_IMPLEMENTED",
         "where": "D2 is future conceptual; the row stays visible"}
    ])
}

/// Run the court; writes an immutable receipt under `receipts/conventional/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("conventional");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("flagship conventional baselines failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court conventional: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    // The corpus gate runs first: never measure a population the verifier does
    // not prove intact (canonical objects, membership, order, derived B1
    // eligibility).
    let manifest = corpus::manifest()?;
    let report = corpus::verify_manifest(&manifest)?;
    if !report.ok() {
        return fail("the frozen flagship corpus does not verify");
    }
    let specs = corpus::specs::specs();
    let spec_by_id: BTreeMap<&str, &corpus::generate::Spec> =
        specs.iter().map(|s| (s.id.as_str(), s)).collect();
    if manifest.objects.len() != specs.len() {
        return fail("manifest object count != frozen membership count");
    }
    // B1 eligibility is derived from the format domain, never the audit flag.
    let (b1_comparable_objects, b1_excluded_objects) = corpus::derived_b1_counts(&manifest.objects);

    let sw = Stopwatch::start();
    let mut cells: Vec<serde_json::Value> = Vec::with_capacity(manifest.objects.len());
    let mut excluded: Vec<serde_json::Value> = Vec::new();
    let mut b0_total = 0u64;
    let mut u1_total = 0u64;
    let mut b0_b1_total = 0u64;
    let mut b1_primary_total = 0u64;
    let mut b1_control_totals = [0u64; B1_LEVEL_CONTROLS.len()];
    let mut encode_ns_total = 0u64;
    let mut decode_ns_total = 0u64;
    let mut reference_bytes_total = 0u64;
    let mut reference_available = 0u64;
    let mut reference_exact = 0u64;
    let mut reference_version: Option<String> = None;
    let mut exact_round_trips = 0usize;

    for o in &manifest.objects {
        let Some(spec) = spec_by_id.get(o.id.as_str()) else {
            return fail(&format!("{}: not in the frozen membership", o.id));
        };
        let samples = corpus::generate::generate(spec)?;
        let canonical = hex(&corpus::generate::canonical_sha256(&samples));
        if canonical != o.canonical_i32_sha256 {
            return fail(&format!(
                "{}: regenerated content does not match the frozen hash",
                o.id
            ));
        }
        let frames = o.frames;
        let ch = o.channels;
        let b0 = b0_raw_pcm_bytes(&samples);
        let u1 = crate::courts::entropy_common::canonical_u1_literal_bytes(frames, ch);
        b0_total += b0;
        u1_total += u1;

        let mut cell = serde_json::json!({
            "id": o.id,
            "source_structure_class": o.source_structure_class,
            "amplitude_class": o.amplitude_class,
            "channel_structure": o.channel_structure,
            "temporal_class": o.temporal_class,
            "entropy_class": o.entropy_class,
            "sample_rate_hz": o.sample_rate_hz,
            "channels": ch,
            "frames": frames,
            "canonical_i32_sha256": canonical,
            "b0_bytes": b0,
            "u1_literal_bytes": u1,
        });

        // B1 is defined on 1..=8 channels (FLAC's own ceiling). A wider object
        // is reported visibly rather than silently dropped or approximated.
        if !corpus::generate::b1_comparable(ch) {
            cell["b1_status"] = serde_json::json!(B1_NOT_APPLICABLE);
            excluded.push(serde_json::json!({
                "id": o.id,
                "baseline": "B1",
                "status": B1_NOT_APPLICABLE,
                "channels": ch,
                "reason": format!(
                    "FLAC encodes at most {FLAC_MAX_CHANNELS} channels; this object has {ch}",
                    FLAC_MAX_CHANNELS = crate::baseline::FLAC_MAX_CHANNELS
                ),
            }));
            cells.push(cell);
            continue;
        }

        let primary = b1_flac(&samples, ch, o.sample_rate_hz, B1_LEVEL_PRIMARY)
            .map_err(|e| Error::internal(format!("{}: B1 primary: {e}", o.id)))?;
        if !primary.exact_roundtrip || primary.source_sha256 != primary.decoded_sha256 {
            return fail(&format!("{}: B1 round trip is not exact", o.id));
        }
        if !primary.md5_ok {
            return fail(&format!("{}: B1 STREAMINFO audio MD5 did not verify", o.id));
        }
        exact_round_trips += 1;

        let mut controls: Vec<FlacEncoding> = Vec::with_capacity(B1_LEVEL_CONTROLS.len());
        for level in B1_LEVEL_CONTROLS {
            let e = b1_flac(&samples, ch, o.sample_rate_hz, level).map_err(|err| {
                Error::internal(format!("{}: B1 control level {level}: {err}", o.id))
            })?;
            if !e.exact_roundtrip || e.source_sha256 != e.decoded_sha256 {
                return fail(&format!(
                    "{}: B1 control level {level} round trip is not exact",
                    o.id
                ));
            }
            exact_round_trips += 1;
            controls.push(e);
        }

        b0_b1_total += b0;
        b1_primary_total += primary.encoded_bytes;
        for (i, c) in controls.iter().enumerate() {
            b1_control_totals[i] += c.encoded_bytes;
        }
        encode_ns_total += primary.encode_ns;
        decode_ns_total += primary.decode_ns;

        // Reference oracle: identical exact i32 domain, identical level, no
        // padding; never authoritative and never part of the frozen vector.
        let reference = reference_flac(&samples, ch, o.sample_rate_hz, B1_LEVEL_PRIMARY)?;
        let reference_cell = match &reference {
            Some(r) => {
                reference_bytes_total += r.bytes;
                reference_available += 1;
                if r.exact_roundtrip {
                    reference_exact += 1;
                }
                reference_version.get_or_insert_with(|| r.version.clone());
                serde_json::json!({
                    "available": true,
                    "command": r.command,
                    "version": r.version,
                    "bytes": r.bytes,
                    "decode_ns": r.decode_ns,
                    "exact_roundtrip": r.exact_roundtrip,
                    "ratio_vs_b1": r.bytes as f64 / primary.encoded_bytes.max(1) as f64,
                })
            }
            None => serde_json::json!({"available": false, "status": "NOT_AVAILABLE"}),
        };

        cell["b1_status"] = serde_json::json!(B1_MEASURED);
        cell["b1_primary_bytes"] = serde_json::json!(primary.encoded_bytes);
        cell["b1_control_0_bytes"] = serde_json::json!(controls[0].encoded_bytes);
        cell["b1_control_8_bytes"] = serde_json::json!(controls[1].encoded_bytes);
        cell["b1_primary_ratio_vs_b0"] =
            serde_json::json!(primary.encoded_bytes as f64 / b0.max(1) as f64);
        cell["b1_primary_ratio_vs_u1_literal"] =
            serde_json::json!(primary.encoded_bytes as f64 / u1.max(1) as f64);
        cell["b1_primary"] = b1_cell(&primary);
        cell["b1_controls"] = serde_json::json!(controls.iter().map(b1_cell).collect::<Vec<_>>());
        cell["reference_flac"] = reference_cell;
        cells.push(cell);
    }

    let total_ns = sw.elapsed_ns().max(0) as u64;
    let object_count = cells.len();
    let result_hash = Sha256::digest(&static_projection(
        &report.manifest_sha256,
        &report.corpus_sha256,
        &cells,
    ));
    let result_hex = hex(&result_hash);
    if CONVENTIONAL_RESULT_SHA256.is_empty() {
        eprintln!("court conventional: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != CONVENTIONAL_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {CONVENTIONAL_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    let verdict = if object_count == 0 {
        Verdict::UnsupportedByHardware
    } else {
        Verdict::Supported
    };

    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!("{} + conventional.b1/v1", manifest.profile)),
        backend: Some("flac-in-process-pure-rust".into()),
        sample_rate_hz: None, // per-object, as frozen in the manifest
        quantum_frames: Some(crate::limits::DEFAULT_QUANTUM_FRAMES),
        content_kind: Some("flagship corpus (frozen before results): conventional baseline".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("conventional");
    builder
        .result(verdict)
        .result_detail(format!(
            "flagship conventional baselines over {} frozen objects ({} B1-comparable, {} \
             excluded by format domain); B0 {b0_total} B, B1(level {}) {b1_primary_total} B, \
             FLAC/u1-literal {:.3}; {} exact round trips verified; corpus sha256 {}, manifest \
             sha256 {}; result sha256 {result_hex}",
            object_count,
            b1_comparable_objects,
            b1_excluded_objects,
            B1_LEVEL_PRIMARY,
            b1_primary_total as f64 / u1_total.max(1) as f64,
            exact_round_trips,
            report.corpus_sha256,
            report.manifest_sha256,
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
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
                "b1_comparable_objects": b1_comparable_objects,
                "b1_excluded_objects": b1_excluded_objects,
                "derivation": "b1-comparable counts are derived from each object's channel \
                               count, not read from the manifest's audit field",
            }),
        )
        .extra(
            "b1_implementation",
            serde_json::json!({
                "crate": "libflac-rs",
                "version": "=0.143.1",
                "description": "pure-Rust, forbid(unsafe_code), byte-exact port of libFLAC 1.4.3",
                "bits_per_sample": 32,
                "compression_level_primary": B1_LEVEL_PRIMARY,
                "compression_level_controls": B1_LEVEL_CONTROLS,
                "conversion": "none (exact canonical interleaved i32; no shift, dither, \
                               normalisation or resampling)",
                "library_baseline_semantics": "libFLAC 1.4.3: at >=28 bits/sample the CONSTANT \
                                               subframe is not selected, so a zeroed block costs \
                                               about one bit per sample; the non-authoritative \
                                               `reference_flac` row records where a newer \
                                               reference encoder differs",
                "primary_choice": "level 5 is the official flac tool's default and libFLAC's \
                                   documented default, so it is the conventional target",
            }),
        )
        .extra("ladder", ladder_manifest())
        .extra(
            "aggregate",
            serde_json::json!({
                "objects": cells.len(),
                "b1_comparable_objects": b1_comparable_objects,
                "b1_excluded_objects": b1_excluded_objects,
                "b0_bytes": b0_total,
                "u1_literal_bytes": u1_total,
                "b0_bytes_b1_comparable": b0_b1_total,
                "b1_primary_bytes": b1_primary_total,
                "b1_control_0_bytes": b1_control_totals[0],
                "b1_control_8_bytes": b1_control_totals[1],
                "b1_primary_ratio_vs_b0_b1_comparable": b1_primary_total as f64
                    / b0_b1_total.max(1) as f64,
                "b1_primary_ratio_vs_u1_literal": b1_primary_total as f64
                    / u1_total.max(1) as f64,
                "b1_encode_ns_total": encode_ns_total,
                "b1_decode_ns_total": decode_ns_total,
                "exact_round_trips_verified": exact_round_trips,
                "reference_flac_available_objects": reference_available,
                "reference_flac_bytes": reference_bytes_total,
                "reference_flac_exact_round_trips": reference_exact,
                "reference_flac_version": reference_version,
                "reference_ratio_vs_b1": if reference_bytes_total > 0 {
                    serde_json::json!(reference_bytes_total as f64 / b1_primary_total.max(1) as f64)
                } else {
                    serde_json::Value::Null
                },
                "total_ns": total_ns,
            }),
        )
        .extra(
            "by_source_structure_class",
            axis_breakdown(&cells, "source_structure_class"),
        )
        .extra(
            "by_amplitude_class",
            axis_breakdown(&cells, "amplitude_class"),
        )
        .extra(
            "by_channel_structure",
            axis_breakdown(&cells, "channel_structure"),
        )
        .extra(
            "by_temporal_class",
            axis_breakdown(&cells, "temporal_class"),
        )
        .extra("by_entropy_class", axis_breakdown(&cells, "entropy_class"))
        .extra(
            "by_sample_rate_hz",
            axis_breakdown(&cells, "sample_rate_hz"),
        )
        .extra("unsupported_objects", serde_json::Value::Array(excluded))
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "this is the flagship B0/B1 CONVENTIONAL-BASELINE result, not yet the B1-vs-VOLE \
             comparison: the `u1_literal_bytes` row is the canonical universal fallback \
             representation, NOT the inverse compiler's selected VOLE representation for the \
             object. The selected-representation comparison needs an exact full-object inverse \
             container (the Phase-K compiler is bounded to 65,536 frames) and is a later \
             increment",
        )
        .limitation(
            "B1 is a comparator with zero VOLE semantic authority: it never decides anything \
             about a SampleObject, and its only correctness requirement is an exact round trip \
             of the same canonical i32 information the VOLE representation carries",
        )
        .limitation(
            "this is a size/exactness comparison at the object level, not a real-time or \
             playback comparison; B2–B4 (sampler/disk/compressed-file playback) are \
             NOT_IMPLEMENTED in this increment and remain visible in the ladder manifest",
        )
        .limitation(
            "the `reference_flac` row is NON-AUTHORITATIVE and outside the frozen result vector: \
             it runs the system `flac` at the same settings when installed, and is \
             NOT_AVAILABLE otherwise. B1 itself never depends on it. It exists because the \
             ported encoder is libFLAC 1.4.3, which does not select the CONSTANT subframe at \
             >=28 bits/sample (an all-zero 32-bit block costs about one bit per sample), while \
             newer reference encoders do; recording both is what stops that divergence from \
             silently flattering the comparison",
        )
        .limitation(
            "the five objects above FLAC's 8-channel ceiling appear explicitly as \
             NOT_APPLICABLE_BY_FORMAT_DOMAIN (in `unsupported_objects`) and their bytes never \
             enter a B1 aggregate; `b1_comparable` is derived from the channel count, not read \
             from the manifest",
        )
        .limitation(
            "the result is bound to the frozen population: the static projection covers the \
             manifest and corpus hashes, the object order, and each object's canonical i32 \
             hash, rate, channels, frames and byte rows",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court conventional: {verdict}");
    println!(
        "  objects: {} ({} B1-comparable, {} excluded by format domain)",
        object_count, b1_comparable_objects, b1_excluded_objects
    );
    println!(
        "  B0 {b0_total} B | B1(level {}) {b1_primary_total} B | u1 literal {u1_total} B",
        B1_LEVEL_PRIMARY
    );
    println!(
        "  FLAC/u1-literal: {:.3} | FLAC/B0 (B1-comparable): {:.3} | exact round trips: {}",
        b1_primary_total as f64 / u1_total.max(1) as f64,
        b1_primary_total as f64 / b0_b1_total.max(1) as f64,
        exact_round_trips
    );
    println!("  corpus sha256:   {}", report.corpus_sha256);
    println!("  manifest sha256: {}", report.manifest_sha256);
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_projection_is_order_and_binding_sensitive() {
        let a = serde_json::json!({
            "id": "a", "canonical_i32_sha256": "00", "sample_rate_hz": 48000,
            "channels": 1, "frames": 10, "b0_bytes": 40, "u1_literal_bytes": 94,
            "b1_status": B1_MEASURED, "b1_primary_bytes": 30,
            "b1_control_0_bytes": 40, "b1_control_8_bytes": 28,
        });
        let b = serde_json::json!({
            "id": "b", "canonical_i32_sha256": "11", "sample_rate_hz": 44100,
            "channels": 2, "frames": 20, "b0_bytes": 160, "u1_literal_bytes": 214,
            "b1_status": B1_NOT_APPLICABLE, "b1_primary_bytes": 0,
            "b1_control_0_bytes": 0, "b1_control_8_bytes": 0,
        });
        let base = static_projection("m", "c", &[a.clone(), b.clone()]);
        assert_ne!(base, static_projection("m2", "c", &[a.clone(), b.clone()]));
        assert_ne!(base, static_projection("m", "c2", &[a.clone(), b.clone()]));
        assert_ne!(base, static_projection("m", "c", &[b, a]));
    }

    #[test]
    fn axis_breakdown_excludes_non_b1_bytes() {
        let cells = vec![
            serde_json::json!({
                "entropy_class": "noise", "b1_status": B1_MEASURED,
                "b0_bytes": 100, "b1_primary_bytes": 60,
            }),
            serde_json::json!({
                "entropy_class": "noise", "b1_status": B1_NOT_APPLICABLE,
                "b0_bytes": 999, "b1_primary_bytes": 0,
            }),
        ];
        let b = axis_breakdown(&cells, "entropy_class");
        let noise = &b["noise"];
        assert_eq!(noise["objects"], 2);
        assert_eq!(noise["b1_comparable_objects"], 1);
        assert_eq!(noise["b1_excluded_objects"], 1);
        assert_eq!(noise["b0_bytes_b1_comparable"], 100);
        assert_eq!(noise["b1_primary_bytes"], 60);
    }
}
