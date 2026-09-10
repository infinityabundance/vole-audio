//! `court conventional` — Phase M conventional baselines (contract §47).
//!
//! The canonical comparison must include the full `B0`–`B9` ladder. This court
//! owns the **conventional** rows and compares them against the exact same
//! information the VOLE representation is priced against — the canonical
//! interleaved `i32` sample domain, with no conversion of any kind:
//!
//! ```text
//!                     same exact i32 source
//!                              │
//!         ┌────────────────────┼────────────────────┐
//!         ▼                    ▼                    ▼
//!      B0 raw PCM          B1 FLAC (32-bit)     VOLE u1 literal
//!         │                    │                    │
//!   no compression      conventional exact     canonical bytes
//!         │                    │                    │
//!         └────────────────────┼────────────────────┘
//!                              ▼
//!          size / ratio / encode+decode time / exactness
//! ```
//!
//! Two things this court refuses to do:
//!
//! * it never converts the source. The historical H.2 comparator (an external
//!   `flac` on a `>> 8` s24 conversion) stays frozen in
//!   `courts::entropy_common` as its own historical row; B1 is a *new*, exact,
//!   32-bit, in-process baseline. A B1 row that used the low 8 bits of the
//!   canonical domain for nothing would not be the same information problem;
//! * it never lets a row disappear. `B2`–`B9` are present in the ladder
//!   manifest with their status, including the ones that are
//!   `NOT_IMPLEMENTED` in this increment and the ones measured by other courts.

use crate::baseline::{
    B1_LEVEL_CONTROLS, B1_LEVEL_PRIMARY, FlacEncoding, b0_raw_pcm_bytes, b1_flac, b1_level_label,
    reference_flac,
};
use crate::entropy::corpus;
use crate::error::{Error, Result};
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::hash::sha256::{Sha256, hex};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash: the court fails if a change silently alters the
/// conventional-baseline results. Empty means "not yet frozen" (the observed
/// value is printed); re-freeze only with a documented reason.
pub const CONVENTIONAL_RESULT_SHA256: &str =
    "11f8683f6ab5df3641ea95fe8349fcbdd4e10e410d4d7d83f328e2d2c23fdd0f";

/// The u1 default nominal rate. The corpus fixtures are generated and
/// rate-agnostic (no fixture carries a rate field), and the H.2 courts evaluate
/// them at this rate; B1 records it in STREAMINFO without resampling a single
/// sample value.
pub const BASELINE_RATE_HZ: u32 = crate::limits::DEFAULT_SAMPLE_RATE_HZ;

/// Static identity of one fixture's conventional rows (no measured quantity).
fn static_projection(cells: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for c in cells {
        out.extend_from_slice(c["fixture"].as_str().unwrap_or("").as_bytes());
        out.push(0);
        for key in [
            "b0_bytes",
            "u1_literal_bytes",
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
        {"id": "B5", "baseline": "VOLE scalar", "status": "MEASURED_ELSEWHERE",
         "where": "court semantic / court facts / court inverse"},
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
            .result_detail(format!("conventional baselines failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court conventional: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let fixtures = corpus::all();
    if fixtures.is_empty() {
        return fail("the frozen corpus is empty");
    }

    let sw = Stopwatch::start();
    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut unsupported: Vec<serde_json::Value> = Vec::new();
    let mut b0_total = 0u64;
    let mut u1_total = 0u64;
    let mut b1_primary_total = 0u64;
    let mut b1_control_totals = [0u64; 2];
    let mut encode_ns_total = 0u64;
    let mut decode_ns_total = 0u64;
    // Non-authoritative reference oracle (system `flac`, when installed).
    let mut reference_bytes_total = 0u64;
    let mut reference_available = 0u64;
    let mut reference_exact = 0u64;
    let mut reference_version: Option<String> = None;

    for fx in &fixtures {
        let frames = fx.frames();
        let b0 = b0_raw_pcm_bytes(&fx.samples);
        let u1 =
            crate::courts::entropy_common::canonical_u1_literal_bytes(frames as u64, fx.channels);

        // B1 is defined on 1..=8 channels (FLAC's own ceiling). A wider object
        // is reported visibly rather than silently dropped or approximated.
        let primary = match b1_flac(&fx.samples, fx.channels, BASELINE_RATE_HZ, B1_LEVEL_PRIMARY) {
            Ok(e) => e,
            Err(e) => {
                unsupported.push(serde_json::json!({
                    "fixture": fx.name,
                    "baseline": "B1",
                    "reason": format!("{e}"),
                }));
                continue;
            }
        };
        // The round trip is the baseline's only correctness requirement: an
        // inexact row is a correctness failure, not a smaller number.
        if !primary.exact_roundtrip || primary.source_sha256 != primary.decoded_sha256 {
            return fail(&format!("{}: B1 round trip is not exact", fx.name));
        }
        if !primary.md5_ok {
            return fail(&format!(
                "{}: B1 STREAMINFO audio MD5 did not verify",
                fx.name
            ));
        }

        let mut controls: Vec<FlacEncoding> = Vec::with_capacity(B1_LEVEL_CONTROLS.len());
        for level in B1_LEVEL_CONTROLS {
            let e = b1_flac(&fx.samples, fx.channels, BASELINE_RATE_HZ, level).map_err(|err| {
                Error::internal(format!("{}: B1 control level {level}: {err}", fx.name))
            })?;
            if !e.exact_roundtrip || e.source_sha256 != e.decoded_sha256 {
                return fail(&format!(
                    "{}: B1 control level {level} round trip is not exact",
                    fx.name
                ));
            }
            controls.push(e);
        }

        b0_total += b0;
        u1_total += u1;
        b1_primary_total += primary.encoded_bytes;
        for (i, c) in controls.iter().enumerate() {
            b1_control_totals[i] += c.encoded_bytes;
        }
        encode_ns_total += primary.encode_ns;
        decode_ns_total += primary.decode_ns;

        // Reference oracle: identical exact i32 domain, identical level, no
        // padding; never authoritative and never part of the frozen vector.
        let reference =
            reference_flac(&fx.samples, fx.channels, BASELINE_RATE_HZ, B1_LEVEL_PRIMARY)?;
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

        cells.push(serde_json::json!({
            "fixture": fx.name,
            "kind": fx.kind,
            "channels": fx.channels,
            "frames": frames,
            "sample_rate_hz": BASELINE_RATE_HZ,
            "source_sha256": hex(&primary.source_sha256),
            "b0_bytes": b0,
            "u1_literal_bytes": u1,
            "b1_primary_bytes": primary.encoded_bytes,
            "b1_control_0_bytes": controls[0].encoded_bytes,
            "b1_control_8_bytes": controls[1].encoded_bytes,
            "b1_primary_ratio_vs_b0": primary.encoded_bytes as f64 / b0.max(1) as f64,
            "b1_primary_ratio_vs_u1_literal": primary.encoded_bytes as f64 / u1.max(1) as f64,
            "b1_primary": b1_cell(&primary),
            "b1_controls": controls.iter().map(b1_cell).collect::<Vec<_>>(),
            "reference_flac": reference_cell,
        }));
    }

    let total_ns = sw.elapsed_ns().max(0) as u64;
    let result_hash = Sha256::digest(&static_projection(&cells));
    let result_hex = hex(&result_hash);
    let fixture_count = cells.len();
    if CONVENTIONAL_RESULT_SHA256.is_empty() {
        eprintln!("court conventional: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != CONVENTIONAL_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {CONVENTIONAL_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    // Every measured row is exact by construction (an inexact row fails the
    // court above); objects outside FLAC's channel domain are excluded
    // explicitly, never approximated.
    let verdict = if fixture_count == 0 {
        Verdict::UnsupportedByHardware
    } else {
        Verdict::Supported
    };

    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1 + conventional.b1/v1".into()),
        backend: Some("flac-in-process-pure-rust".into()),
        sample_rate_hz: Some(BASELINE_RATE_HZ),
        quantum_frames: Some(crate::limits::DEFAULT_QUANTUM_FRAMES),
        content_kind: Some("entropy-corpus-v1 (conventional baseline)".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("conventional");
    builder
        .result(verdict)
        .result_detail(format!(
            "conventional baselines over {} fixtures; B0 {b0_total} B, B1(level {}) {} B, \
             FLAC/u1-literal {:.3}; {} exact round trips verified; result sha256 {result_hex}",
            fixture_count,
            B1_LEVEL_PRIMARY,
            b1_primary_total,
            b1_primary_total as f64 / u1_total.max(1) as f64,
            fixture_count * (1 + B1_LEVEL_CONTROLS.len()),
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
            ..Default::default()
        })
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
                "fixtures": cells.len(),
                "b0_bytes": b0_total,
                "u1_literal_bytes": u1_total,
                "b1_primary_bytes": b1_primary_total,
                "b1_control_0_bytes": b1_control_totals[0],
                "b1_control_8_bytes": b1_control_totals[1],
                "b1_primary_ratio_vs_b0": b1_primary_total as f64 / b0_total.max(1) as f64,
                "b1_primary_ratio_vs_u1_literal": b1_primary_total as f64 / u1_total.max(1) as f64,
                "b1_encode_ns_total": encode_ns_total,
                "b1_decode_ns_total": decode_ns_total,
                "exact_round_trips_verified": fixture_count * (1 + B1_LEVEL_CONTROLS.len()),
                "reference_flac_available_fixtures": reference_available,
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
        .extra("unsupported_objects", serde_json::Value::Array(unsupported))
        .extra("cells", serde_json::Value::Array(cells))
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
            "the corpus fixtures are generated and rate-agnostic; they are evaluated at the u1 \
             default rate (48 kHz) exactly as the H.2 courts do, and no sample value is \
             resampled",
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
            "this corpus is the frozen H.2 entropy corpus, not the Phase-M flagship corpus: \
             its negative controls are built for 8-bit symbolization, so at 32 bits/sample some \
             of them are not incompressible (see `reference_ratio_vs_b1` and the per-fixture \
             cells). The Phase-M flagship corpus is the next increment and is stratified by \
             representation and by amplitude class for exactly this reason",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court conventional: {verdict}");
    println!(
        "  fixtures: {} | B0 {} B | B1(level {}) {} B | u1 literal {} B",
        fixture_count, b0_total, B1_LEVEL_PRIMARY, b1_primary_total, u1_total
    );
    println!(
        "  FLAC/u1-literal: {:.3} | exact round trips: {}",
        b1_primary_total as f64 / u1_total.max(1) as f64,
        fixture_count * (1 + B1_LEVEL_CONTROLS.len())
    );
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
