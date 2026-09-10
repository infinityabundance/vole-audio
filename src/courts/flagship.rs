//! `court flagship` — the true flagship result (Phase M, Seal 6):
//! **B1 FLAC versus current bounded VOLE inverse selection**.
//!
//! Every object of the frozen flagship corpus is compiled to a full-object
//! archival container by the frozen Seal-5 mechanism (minimum Phase-K
//! `complete_bytes` per 65,536-frame segment, empty reference library,
//! semantics preserved), and the exactly-0110 B1-comparable objects are priced
//! against their frozen FLAC level-5 bytes.
//!
//! The claim boundary is explicit in the name: this is the current bounded
//! inverse compiler's **selected** representation, whose proposal vocabulary is
//! deliberately limited (literal, silence, constant, exact-repeat, residual
//! zero/constant/periodic, shared reference). It is **not** "optimal VOLE": an
//! object frozen as `oscillator` may legitimately compile to literal or
//! exact-repeat, and that is a result of the compiler that exists today.
//!
//! The court reports, per object: B0, B1, literal-equivalent, the container's
//! real complete bytes with its header/index/integrity overhead, the selected
//! representation per segment, and the comparison ratios. It reports the three
//! comparison buckets (VOLE cheaper / ≈ B1 / VOLE larger) rather than only an
//! aggregate, because an aggregate can be dominated by a few large or
//! high-channel objects.

use crate::baseline::{B1_LEVEL_PRIMARY, b0_raw_pcm_bytes, b1_flac};
use crate::corpus::generate::{self, Spec};
use crate::corpus::{self};
use crate::error::{Error, Result};
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::fullobj::{self, FullSemantics};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

/// Frozen static-result hash: the court fails if a change silently alters the
/// flagship comparison. Empty means "not yet frozen" (the observed value is
/// printed); re-freeze only with a documented reason.
pub const FLAGSHIP_RESULT_SHA256: &str =
    "8f37fab06e5fd088369f0ff6199ad3660a2b6ccb918da3060af65a4e68cd2419";

/// The frozen FLAC level-5 total over the 110 B1-comparable objects, as sealed
/// by `court conventional` (Seal 4). The flagship court recomputes B1
/// in-process and requires this total to match, so the comparison cannot drift
/// away from the sealed conventional baseline.
pub const EXPECTED_B1_PRIMARY_TOTAL: u64 = 25_577_431;

/// The frozen search budget for this result: the current default bounded
/// search. Recorded in the receipt so the comparison states exactly which
/// compiler configuration produced it.
pub fn budget() -> SearchBudget {
    SearchBudget::default()
}

/// The frozen corpus root semantics for a spec.
fn semantics_of(spec: &Spec) -> FullSemantics {
    match spec.semantics.period_frames() {
        Some(period_frames) => FullSemantics::Loop { period_frames },
        None => FullSemantics::OneShot,
    }
}

/// Per-axis surface. Counts are over all objects in the class; every byte sum is
/// reported for both populations so no B1 ratio can silently mix them, and the
/// ratios use the B1-comparable population only.
fn axis_surface(cells: &[serde_json::Value], key: &str) -> serde_json::Value {
    #[derive(Default)]
    struct Agg {
        objects: usize,
        b1: usize,
        b0_all: u64,
        b0_comparable: u64,
        b1_bytes: u64,
        vole_comparable: u64,
        literal_all: u64,
        literal_comparable: u64,
        vole_wins: usize,
    }
    let mut map: BTreeMap<String, Agg> = BTreeMap::new();
    for c in cells {
        let class = c[key]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| c[key].to_string());
        let e = map.entry(class).or_default();
        e.objects += 1;
        e.b0_all += c["b0_bytes"].as_u64().unwrap_or(0);
        e.literal_all += c["literal_equivalent_bytes"].as_u64().unwrap_or(0);
        if c["b1_comparable"].as_bool().unwrap_or(false) {
            e.b1 += 1;
            let b1 = c["b1_bytes"].as_u64().unwrap_or(0);
            let vole = c["vole_complete_bytes"].as_u64().unwrap_or(0);
            e.b0_comparable += c["b0_bytes"].as_u64().unwrap_or(0);
            e.literal_comparable += c["literal_equivalent_bytes"].as_u64().unwrap_or(0);
            e.b1_bytes += b1;
            e.vole_comparable += vole;
            if vole < b1 {
                e.vole_wins += 1;
            }
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
                "b0_bytes_all_objects": v.b0_all,
                "b0_bytes_b1_comparable": v.b0_comparable,
                "literal_equivalent_bytes_all_objects": v.literal_all,
                "literal_equivalent_bytes_b1_comparable": v.literal_comparable,
                "b1_bytes_b1_comparable": v.b1_bytes,
                "vole_complete_bytes_b1_comparable": v.vole_comparable,
                "b1_over_vole_b1_comparable": ratio(v.b1_bytes, v.vole_comparable),
                "vole_over_b1_b1_comparable": ratio(v.vole_comparable, v.b1_bytes),
                "objects_vole_cheaper_than_b1": v.vole_wins,
            }),
        );
    }
    serde_json::Value::Object(out)
}

fn ratio(num: u64, den: u64) -> serde_json::Value {
    if den == 0 {
        serde_json::Value::Null
    } else {
        serde_json::json!(num as f64 / den as f64)
    }
}

/// Deterministic static-result projection (no measured quantity).
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
        out.extend_from_slice(&c["sample_rate_hz"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["channels"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["frames"].as_u64().unwrap_or(0).to_le_bytes());
        out.push(c["semantics_code"].as_u64().unwrap_or(0) as u8);
        out.extend_from_slice(&c["loop_period_frames"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["b0_bytes"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(
            &c["literal_equivalent_bytes"]
                .as_u64()
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&c["b1_bytes"].as_u64().unwrap_or(u64::MAX).to_le_bytes());
        out.extend_from_slice(&c["vole_complete_bytes"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(
            &c["vole_objective_bytes"]
                .as_u64()
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &c["container_header_bytes"]
                .as_u64()
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &c["container_index_bytes"]
                .as_u64()
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &c["container_integrity_bytes"]
                .as_u64()
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&c["segment_count"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&[if c["all_literal"].as_bool().unwrap_or(false) {
            1
        } else {
            0
        }]);
        out.extend_from_slice(&[if c["all_procedural"].as_bool().unwrap_or(false) {
            1
        } else {
            0
        }]);
        if let Some(segs) = c["segments"].as_array() {
            for s in segs {
                out.extend_from_slice(&s["start_frame"].as_u64().unwrap_or(0).to_le_bytes());
                out.extend_from_slice(&s["frame_count"].as_u64().unwrap_or(0).to_le_bytes());
                out.push(s["representation_tag"].as_u64().unwrap_or(0) as u8);
                out.push(s["candidate_tag"].as_u64().unwrap_or(0) as u8);
                out.extend_from_slice(&s["objective_bytes"].as_u64().unwrap_or(0).to_le_bytes());
                out.extend_from_slice(&s["stored_bytes"].as_u64().unwrap_or(0).to_le_bytes());
                if let Some(cid) = s["content_id"].as_str() {
                    out.extend_from_slice(cid.as_bytes());
                }
                out.push(0);
            }
        }
        out.push(0xEE);
    }
    out
}

/// Run the court; writes an immutable receipt under `receipts/flagship/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("flagship");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("flagship B1-vs-VOLE comparison failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court flagship: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    // The corpus gate runs first: never measure a population the verifier does
    // not prove intact.
    let manifest = corpus::manifest()?;
    let report = corpus::verify_manifest(&manifest)?;
    if !report.ok() {
        return fail("the frozen flagship corpus does not verify");
    }
    let specs = corpus::specs::specs();
    let spec_by_id: BTreeMap<&str, &Spec> = specs.iter().map(|s| (s.id.as_str(), s)).collect();
    let (b1_comparable_objects, _) = corpus::derived_b1_counts(&manifest.objects);

    let sw = Stopwatch::start();
    let mut cells: Vec<serde_json::Value> = Vec::with_capacity(manifest.objects.len());
    // Every byte total is tracked over both populations: `_all` is all 115
    // objects, `_comparable` is the 110 inside FLAC's format domain. A ratio
    // against B1 must only ever use the comparable population.
    let mut b0_all = 0u64;
    let mut b0_comparable = 0u64;
    let mut literal_all = 0u64;
    let mut literal_comparable = 0u64;
    let mut b1_total = 0u64; // comparable only
    let mut vole_all = 0u64;
    let mut vole_comparable = 0u64;
    let mut segment_total = 0usize;
    let mut all_literal_objects = 0usize;
    let mut all_procedural_objects = 0usize;
    let mut mixed_objects = 0usize;
    let mut vole_lt_b1 = 0usize;
    let mut vole_eq_b1 = 0usize;
    let mut vole_within_1pct = 0usize;
    let mut vole_gt_b1 = 0usize;
    let mut selected_kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut selected_representations: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut b1_available = 0usize;
    let mut b1_exact = 0usize;

    for o in &manifest.objects {
        let spec = spec_by_id
            .get(o.id.as_str())
            .ok_or_else(|| Error::internal(format!("{}: not in the frozen membership", o.id)))?;
        let samples = generate::generate(spec)?;
        let canonical = hex(&generate::canonical_sha256(&samples));
        if canonical != o.canonical_i32_sha256 {
            return fail(&format!("{}: regenerated content changed", o.id));
        }
        let ch = o.channels;
        let frames = o.frames;
        let comparable = generate::b1_comparable(ch);
        let b0 = b0_raw_pcm_bytes(&samples);
        let literal = crate::inverse::cost::canonical_u1_literal_bytes(frames, ch);
        b0_all += b0;
        literal_all += literal;
        if comparable {
            b0_comparable += b0;
            literal_comparable += literal;
        }

        // B1 (FLAC level 5) over the exact same canonical i32 domain.
        let (b1_bytes, b1_cell) = if comparable {
            let e = b1_flac(&samples, ch, o.sample_rate_hz, B1_LEVEL_PRIMARY)
                .map_err(|err| Error::internal(format!("{}: B1: {err}", o.id)))?;
            if !e.exact_roundtrip || e.source_sha256 != e.decoded_sha256 || !e.md5_ok {
                return fail(&format!("{}: B1 is not exact", o.id));
            }
            b1_available += 1;
            if e.exact_roundtrip {
                b1_exact += 1;
            }
            b1_total += e.encoded_bytes;
            (
                Some(e.encoded_bytes),
                serde_json::json!({
                    "status": "MEASURED",
                    "level": B1_LEVEL_PRIMARY,
                    "bytes": e.encoded_bytes,
                    "encode_ns": e.encode_ns,
                    "decode_ns": e.decode_ns,
                    "exact_roundtrip": e.exact_roundtrip,
                    "md5_ok": e.md5_ok,
                }),
            )
        } else {
            (
                None,
                serde_json::json!({
                    "status": "NOT_APPLICABLE_BY_FORMAT_DOMAIN",
                    "reason": format!("FLAC encodes at most 8 channels; this object has {ch}"),
                }),
            )
        };

        // VOLE: the frozen full-object container over the same samples.
        let semantics = semantics_of(spec);
        let vole = fullobj::compile_full_object(
            &o.id,
            o.sample_rate_hz,
            ch,
            frames,
            semantics,
            &samples,
            budget(),
        )?;
        let mat = fullobj::materialize_full_object(&vole.bytes)?;
        if mat.samples() != &samples[..] {
            return fail(&format!("{}: VOLE reconstruction is not exact", o.id));
        }
        if mat.decoded.semantics != semantics {
            return fail(&format!("{}: VOLE semantics were not preserved", o.id));
        }
        let vole_bytes = vole.complete_bytes();
        vole_all += vole_bytes;
        if b1_bytes.is_some() {
            vole_comparable += vole_bytes;
        }
        segment_total += vole.segment_count();

        let (all_literal, all_procedural, mixed) =
            (vole.all_literal(), vole.none_literal(), vole.mixed());
        if all_literal {
            all_literal_objects += 1;
        } else if all_procedural {
            all_procedural_objects += 1;
        } else {
            mixed_objects += 1;
        }

        let mut seg_cells = Vec::with_capacity(vole.segments.len());
        let mut object_representations: Vec<&'static str> = Vec::new();
        for s in &vole.segments {
            *selected_kinds.entry(s.kind.name()).or_insert(0) += 1;
            *selected_representations
                .entry(s.representation.name())
                .or_insert(0) += 1;
            if !object_representations.contains(&s.representation.name()) {
                object_representations.push(s.representation.name());
            }
            seg_cells.push(serde_json::json!({
                "start_frame": s.plan.start_frame,
                "frame_count": s.plan.frame_count,
                "kind": s.kind.name(),
                "candidate_tag": s.kind.tag(),
                "representation": s.representation.name(),
                "representation_tag": s.representation.tag(),
                "content_id": s.content_id.to_string(),
                "objective_bytes": s.objective_bytes,
                "stored_bytes": s.stored_bytes,
                "proposed": s.proposed,
                "accepted": s.accepted,
            }));
        }

        if let Some(b1) = b1_bytes {
            if vole_bytes < b1 {
                vole_lt_b1 += 1;
            } else if vole_bytes > b1 {
                vole_gt_b1 += 1;
            } else {
                vole_eq_b1 += 1;
            }
            let diff = vole_bytes.abs_diff(b1);
            if diff * 100 <= b1.max(1) {
                vole_within_1pct += 1;
            }
        }

        cells.push(serde_json::json!({
            "id": o.id,
            "source_structure_class": o.source_structure_class,
            "amplitude_class": o.amplitude_class,
            "channel_structure": o.channel_structure,
            "temporal_class": o.temporal_class,
            "entropy_class": o.entropy_class,
            "sample_rate_hz": o.sample_rate_hz,
            "channels": ch,
            "frames": frames,
            "semantics": o.semantics,
            "semantics_code": semantics.kind_code(),
            "loop_period_frames": semantics.period_frames(),
            "canonical_i32_sha256": canonical,
            "b1_comparable": b1_bytes.is_some(),
            "b0_bytes": b0,
            "literal_equivalent_bytes": literal,
            "b1_bytes": b1_bytes,
            "b1": b1_cell,
            "full_object_sha256": hex(&vole.sha256()),
            "vole_complete_bytes": vole_bytes,
            "vole_objective_bytes": vole.objective_bytes(),
            "container_header_bytes": vole.header_bytes,
            "container_index_bytes": vole.index_bytes,
            "container_payload_bytes": vole.payload_bytes,
            "container_integrity_bytes": vole.integrity_bytes,
            "segment_count": vole.segment_count(),
            "segments": seg_cells,
            "selected_representations": object_representations,
            "all_literal": all_literal,
            "all_procedural": all_procedural,
            "mixed": mixed,
            "exact_reconstruction": true,
            "semantics_preserved": true,
            "b1_over_vole": b1_bytes.map(|b| ratio(b, vole_bytes)),
            "vole_over_b1": b1_bytes.map(|b| ratio(vole_bytes, b)),
            "literal_over_vole": ratio(literal, vole_bytes),
            "compile_ns": vole.compile_ns,
        }));
    }

    let total_ns = sw.elapsed_ns().max(0) as u64;
    let object_count = cells.len();
    if b1_total != EXPECTED_B1_PRIMARY_TOTAL {
        return fail(&format!(
            "recomputed B1 total {b1_total} does not match the sealed conventional total \
             {EXPECTED_B1_PRIMARY_TOTAL}"
        ));
    }

    let result_hex = hex(&Sha256::digest(&static_projection(
        &report.manifest_sha256,
        &report.corpus_sha256,
        &cells,
    )));
    if FLAGSHIP_RESULT_SHA256.is_empty() {
        eprintln!("court flagship: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != FLAGSHIP_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {FLAGSHIP_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    let verdict = Verdict::Supported;
    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!(
            "{} + fullobj.v1 + conventional.b1/v1",
            manifest.profile
        )),
        backend: Some("full-object container over the bounded Phase-K inverse compiler".into()),
        sample_rate_hz: None, // per object, as frozen
        quantum_frames: Some(crate::limits::DEFAULT_QUANTUM_FRAMES),
        content_kind: Some("flagship corpus (frozen before results): B1 vs selected VOLE".into()),
        ..Default::default()
    };
    let b = budget();
    let mut builder = ReceiptBuilder::new("flagship");
    builder
        .result(verdict)
        .result_detail(format!(
            "B1 FLAC versus current bounded VOLE inverse selection over {} frozen objects \
             ({b1_comparable_objects} B1-comparable); over the SAME comparable population: \
             B1(level {B1_LEVEL_PRIMARY}) {b1_total} B, VOLE {vole_comparable} B, B1/VOLE {:.3} \
             (VOLE {:.3}x B1); VOLE cheaper on {vole_lt_b1}, equal on {vole_eq_b1}, larger on \
             {vole_gt_b1} of the comparable objects; every extent reconstructed exactly; \
             corpus sha256 {}, manifest sha256 {}; result sha256 {result_hex}",
            cells.len(),
            b1_total as f64 / vole_comparable.max(1) as f64,
            vole_comparable as f64 / b1_total.max(1) as f64,
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
            "claim",
            serde_json::json!({
                "name": "B1 FLAC versus current bounded VOLE inverse selection",
                "is_not": "not optimal VOLE: the inverse compiler's proposal vocabulary is \
                           deliberately limited (literal, silence, constant, exact-repeat, \
                           residual zero/constant/periodic, shared reference). A source frozen \
                           as `oscillator` may legitimately compile to literal or exact-repeat.",
                "selection": "accepted exact candidates -> minimum Phase-K complete_bytes -> \
                              deterministic proposal-order tie break",
                "standalone": "each object is compiled with an empty reference library, so no \
                               corpus-level deduplication can flatter the comparison; the \
                               conventional baseline is priced standalone per file too",
            }),
        )
        .extra(
            "search_budget",
            serde_json::json!({
                "max_period_scan": b.max_period_scan,
                "max_residual_period_candidates": b.max_residual_period_candidates,
                "max_candidates": b.max_candidates,
                "placement": format!("{:?}", b.placement),
            }),
        )
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
            "aggregate",
            serde_json::json!({
                "objects": cells.len(),
                "b1_comparable_objects": b1_comparable_objects,
                "b1_excluded_objects": cells.len() - b1_comparable_objects,
                "b1_exact_round_trips": b1_exact,
                "b1_available_objects": b1_available,
                // Population split: a B1 ratio must only ever use the comparable
                // population. `_all` figures are reported for completeness and
                // must never be divided by a B1 total.
                "b0_bytes_all_objects": b0_all,
                "b0_bytes_b1_comparable": b0_comparable,
                "literal_equivalent_bytes_all_objects": literal_all,
                "literal_equivalent_bytes_b1_comparable": literal_comparable,
                "b1_primary_bytes": b1_total,
                "expected_b1_primary_total": EXPECTED_B1_PRIMARY_TOTAL,
                "vole_complete_bytes_all_objects": vole_all,
                "vole_complete_bytes_b1_comparable": vole_comparable,
                "b1_over_vole_b1_comparable": ratio(b1_total, vole_comparable),
                "vole_over_b1_b1_comparable": ratio(vole_comparable, b1_total),
                "literal_over_vole_b1_comparable": ratio(literal_comparable, vole_comparable),
                "segments_total": segment_total,
                "objects_all_literal": all_literal_objects,
                "objects_all_procedural": all_procedural_objects,
                "objects_mixed": mixed_objects,
                "objects_vole_cheaper_than_b1": vole_lt_b1,
                "objects_vole_equal_b1": vole_eq_b1,
                "objects_vole_within_one_percent_of_b1": vole_within_1pct,
                "objects_vole_larger_than_b1": vole_gt_b1,
                "selected_candidate_kinds": selected_kinds,
                "selected_representations": selected_representations,
                "total_ns": total_ns,
            }),
        )
        .extra(
            "by_source_structure_class",
            axis_surface(&cells, "source_structure_class"),
        )
        .extra(
            "by_amplitude_class",
            axis_surface(&cells, "amplitude_class"),
        )
        .extra(
            "by_channel_structure",
            axis_surface(&cells, "channel_structure"),
        )
        .extra("by_temporal_class", axis_surface(&cells, "temporal_class"))
        .extra("by_entropy_class", axis_surface(&cells, "entropy_class"))
        .extra("by_sample_rate_hz", axis_surface(&cells, "sample_rate_hz"))
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "this is the flagship B1-vs-VOLE result for the compiler that exists today, not an \
             optimality claim: the proposal vocabulary is bounded and deterministic, and the \
             full-object container is a frozen archival structure (header ∥ segment index ∥ \
             payloads ∥ integrity), not a new U1 Representation",
        )
        .limitation(
            "the container's complete bytes are real serialized bytes (header + segment index + \
             Σ payload + integrity). Per-segment objective_bytes is the Phase-K selection \
             objective; both are recorded",
        )
        .limitation(
            "the five objects above FLAC's 8-channel ceiling are NOT_APPLICABLE_BY_FORMAT_DOMAIN \
             and never enter a B1 aggregate; the B1 total is recomputed in-process and required \
             to equal the sealed `court conventional` total, so the two courts cannot drift",
        )
        .limitation(
            "the axes are descriptive surfaces of this frozen population, not controlled causal \
             effects: objects differ by rate and structure at once",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court flagship: {verdict}");
    println!(
        "  objects: {} ({} B1-comparable) | segments: {segment_total}",
        object_count, b1_comparable_objects
    );
    println!(
        "  B0 all/comparable {b0_all}/{b0_comparable} B | literal {literal_all}/{literal_comparable} B"
    );
    println!(
        "  B1 {b1_total} B | VOLE all/comparable {vole_all}/{vole_comparable} B | B1/VOLE(comparable) {:.3}",
        b1_total as f64 / vole_comparable.max(1) as f64
    );
    println!(
        "  VOLE cheaper/equal/larger than B1: {vole_lt_b1}/{vole_eq_b1}/{vole_gt_b1} \
         (within 1%: {vole_within_1pct})"
    );
    println!(
        "  objects all-literal/procedural/mixed: {all_literal_objects}/{all_procedural_objects}/{mixed_objects}"
    );
    println!("  selected kinds: {selected_kinds:?}");
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
