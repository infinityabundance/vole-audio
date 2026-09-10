//! `court runtime` — the common runtime substrate mechanism (Phase M, Seal 9).
//!
//! This court freezes the **runtime protocol and correctness vector** across the
//! four source architectures before any comparative timing is reported:
//!
//! ```text
//! B2  PCM-resident sampler
//! B3  raw PCM disk streaming   (cold-verified and warm-verified separately)
//! B4  compressed preload        (the exact B1 FLAC-5 artifact)
//! B5  bounded VOLE materialization (verified full-object container)
//! ```
//!
//! Every source receives the same frozen sequential `[start, 512)` trace at each
//! object's native rate and channel count, fills the same caller-owned
//! destination, and must reproduce the canonical window exactly. The court's
//! frozen static result covers the protocol, the artifact identities and the
//! correctness vector; wall-clock latencies are measured evidence and are
//! deliberately excluded from the hash.

use crate::baseline::{B1_LEVEL_PRIMARY, b1_flac_artifact};
use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::{Error, Result};
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::fullobj;
use crate::fullobj::FullSemantics;
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::runtime::{
    DiskPcmSource, FlacPreloadSource, QUANTUM_FRAMES, ReadEvidence, ResidentPcmSource,
    RuntimeSource, VoleBoundedSource, frozen_trace,
};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

/// Frozen static-result hash: the court fails if a change silently alters the
/// runtime mechanism or its correctness vector. Empty means "not yet frozen".
pub const RUNTIME_RESULT_SHA256: &str =
    "a66774555f11bd9506ccfbf1f14e9e4404e6fdc8aa8478518cdee3c705d5ea84";

/// The frozen search budget used to build the B5 artifacts (same as the
/// flagship result).
pub fn budget() -> SearchBudget {
    SearchBudget::default()
}

fn semantics_of(spec: &Spec) -> FullSemantics {
    match spec.semantics.period_frames() {
        Some(period_frames) => FullSemantics::Loop { period_frames },
        None => FullSemantics::OneShot,
    }
}

/// Accumulated counters of one trace run over one source.
#[derive(Debug, Clone, Default)]
struct RunTotals {
    exact: bool,
    windows: usize,
    requested_frames: u64,
    returned_frames: u64,
    output_bytes: u64,
    logical_source_bytes_read: u64,
    storage_bytes_read: u64,
    encoded_bytes_examined: u64,
    encoded_bytes_parsed: u64,
    sample_domain_bytes_materialized: u64,
    segments_touched: u64,
    pages_touched: u64,
    scratch_peak_bytes: u64,
    working_state_bytes: u64,
    deadline_misses: u64,
    total_latency_ns: u64,
    max_latency_ns: u64,
}

impl RunTotals {
    fn absorb(&mut self, e: &ReadEvidence) {
        self.windows += 1;
        self.requested_frames += u64::from(e.requested_frames);
        self.returned_frames += u64::from(e.returned_frames);
        self.output_bytes += e.output_bytes;
        self.logical_source_bytes_read += e.logical_source_bytes_read;
        self.storage_bytes_read += e.storage_bytes_read;
        self.encoded_bytes_examined += e.encoded_bytes_examined;
        self.encoded_bytes_parsed += e.encoded_bytes_parsed;
        self.sample_domain_bytes_materialized += e.sample_domain_bytes_materialized;
        self.segments_touched += u64::from(e.segments_touched);
        self.pages_touched += u64::from(e.pages_touched);
        self.scratch_peak_bytes = self.scratch_peak_bytes.max(e.scratch_peak_bytes);
        self.working_state_bytes = self.working_state_bytes.max(e.working_state_bytes);
        if e.deadline_missed {
            self.deadline_misses += 1;
        }
        self.total_latency_ns += e.latency_ns;
        self.max_latency_ns = self.max_latency_ns.max(e.latency_ns);
    }
}

/// Run the frozen trace over one source, requiring every window to be exact.
fn run_trace(
    source: &mut dyn RuntimeSource,
    expected: &[i32],
    ch: usize,
    trace: &[(u64, u32)],
) -> Result<RunTotals> {
    let info = source.info();
    let mut totals = RunTotals {
        exact: true,
        ..Default::default()
    };
    let mut dst = vec![0i32; QUANTUM_FRAMES as usize * ch];
    for &(start, frames) in trace {
        let n = frames as usize * ch;
        let e = source.read(start, frames, &mut dst[..n])?;
        let lo = start as usize * ch;
        let hi = lo + n;
        if dst[..n] != expected[lo..hi] {
            return Err(Error::internal(format!(
                "{}: window [{start}, +{frames}) is not exact",
                info.name
            )));
        }
        totals.absorb(&e);
    }
    Ok(totals)
}

fn source_cell(name: &str, totals: &RunTotals, extra: serde_json::Value) -> serde_json::Value {
    let mut obj = serde_json::json!({
        "source": name,
        "status": "MEASURED",
        "exact": totals.exact,
        "windows": totals.windows,
        "requested_frames": totals.requested_frames,
        "returned_frames": totals.returned_frames,
        "output_bytes": totals.output_bytes,
        "logical_source_bytes_read": totals.logical_source_bytes_read,
        "storage_bytes_read": totals.storage_bytes_read,
        "encoded_bytes_examined": totals.encoded_bytes_examined,
        "encoded_bytes_parsed": totals.encoded_bytes_parsed,
        "sample_domain_bytes_materialized": totals.sample_domain_bytes_materialized,
        "segments_touched": totals.segments_touched,
        "pages_touched": totals.pages_touched,
        "scratch_peak_bytes": totals.scratch_peak_bytes,
        "working_state_bytes": totals.working_state_bytes,
        "deadline_misses": totals.deadline_misses,
        "total_latency_ns": totals.total_latency_ns,
        "max_latency_ns": totals.max_latency_ns,
    });
    if let (Some(dst), Some(src)) = (obj.as_object_mut(), extra.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    obj
}

/// Deterministic static projection: protocol, artifact identities, correctness
/// vector and counters. No measured timing.
fn static_projection(manifest_sha: &str, corpus_sha: &str, cells: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for head in [manifest_sha, corpus_sha] {
        out.extend_from_slice(head.as_bytes());
        out.push(0);
    }
    out.extend_from_slice(&QUANTUM_FRAMES.to_le_bytes());
    for c in cells {
        out.extend_from_slice(c["id"].as_str().unwrap_or("").as_bytes());
        out.push(0);
        out.extend_from_slice(c["canonical_i32_sha256"].as_str().unwrap_or("").as_bytes());
        out.push(0);
        out.extend_from_slice(&c["channels"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["frames"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["sample_rate_hz"].as_u64().unwrap_or(0).to_le_bytes());
        if let Some(sources) = c["sources"].as_object() {
            for (name, s) in sources {
                out.extend_from_slice(name.as_bytes());
                out.push(0);
                out.push(s["exact"].as_bool().unwrap_or(false) as u8);
                out.push(matches!(s["status"].as_str(), Some("MEASURED")) as u8);
                for key in [
                    "requested_frames",
                    "returned_frames",
                    "output_bytes",
                    "logical_source_bytes_read",
                    "encoded_bytes_examined",
                    "encoded_bytes_parsed",
                    "sample_domain_bytes_materialized",
                    "segments_touched",
                    "pages_touched",
                    "scratch_peak_bytes",
                    "working_state_bytes",
                    "persistent_encoded_bytes",
                    "persistent_sample_domain_bytes",
                ] {
                    out.extend_from_slice(&s[key].as_u64().unwrap_or(0).to_le_bytes());
                }
                if let Some(a) = s["artifact_sha256"].as_str() {
                    out.extend_from_slice(a.as_bytes());
                }
                out.push(0);
            }
        }
        out.push(0xEE);
    }
    out
}

/// Run the court; writes an immutable receipt under `receipts/runtime/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("runtime");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("runtime mechanism failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court runtime: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    // The corpus gate runs first.
    let manifest = corpus::manifest()?;
    let report = corpus::verify_manifest(&manifest)?;
    if !report.ok() {
        return fail("the frozen flagship corpus does not verify");
    }
    let specs = corpus::specs::specs();
    let spec_by_id: BTreeMap<&str, &Spec> = specs.iter().map(|s| (s.id.as_str(), s)).collect();

    // B3 artifacts must live on a block-backed filesystem: `fadvise(DONTNEED)`
    // cannot evict tmpfs pages, so cold verification would be impossible there.
    // `target/` is gitignored and outside the seal subject.
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("runtime-b3-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;

    let sw = Stopwatch::start();
    let mut cells: Vec<serde_json::Value> = Vec::with_capacity(manifest.objects.len());
    let mut kind_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut cache_states: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut total_windows = 0u64;
    let mut exact_windows = 0u64;
    let mut all_exact = true;

    for o in &manifest.objects {
        let spec = spec_by_id
            .get(o.id.as_str())
            .ok_or_else(|| Error::internal(format!("{}: not in the frozen membership", o.id)))?;
        let samples = generate::generate(spec)?;
        let canonical = hex(&generate::canonical_sha256(&samples));
        if canonical != o.canonical_i32_sha256 {
            return fail(&format!("{}: regenerated content changed", o.id));
        }
        let ch = usize::from(o.channels);
        let trace = frozen_trace(o.frames, QUANTUM_FRAMES)?;
        let mut sources = serde_json::Map::new();

        // B2 — resident PCM.
        let mut b2 =
            ResidentPcmSource::prepare("B2", o.channels, o.sample_rate_hz, samples.clone())?;
        let info2 = b2.info();
        let t2 = run_trace(&mut b2, &samples, ch, &trace)?;
        all_exact &= t2.exact;
        sources.insert(
            "B2".into(),
            source_cell(
                "B2",
                &t2,
                serde_json::json!({
                    "persistent_encoded_bytes": info2.persistent_encoded_bytes,
                    "persistent_sample_domain_bytes": info2.persistent_sample_domain_bytes,
                    "setup_ns": info2.setup_ns,
                }),
            ),
        );

        // B3 — raw PCM on disk, cold-verified then warm-verified.
        let path = dir.join(format!("{}.pcm", o.id));
        let mut b3 = DiskPcmSource::prepare("B3", o.channels, o.sample_rate_hz, &samples, &path)?;
        let info3 = b3.info();
        let cold = b3.evict_and_verify();
        *cache_states.entry(cold.state_name()).or_insert(0) += 1;
        let t3c = run_trace(&mut b3, &samples, ch, &trace)?;
        all_exact &= t3c.exact;
        sources.insert(
            "B3-cold".into(),
            source_cell(
                "B3-cold",
                &t3c,
                serde_json::json!({
                    "cache_state": cold.state_name(),
                    "resident_bytes": cold.resident_bytes,
                    "range_bytes": cold.range_bytes,
                    "method": cold.method,
                    "detail": cold.detail,
                    "persistent_encoded_bytes": info3.persistent_encoded_bytes,
                    "persistent_sample_domain_bytes": info3.persistent_sample_domain_bytes,
                    "setup_ns": info3.setup_ns,
                }),
            ),
        );
        let warm = b3.prime_and_verify()?;
        *cache_states.entry(warm.state_name()).or_insert(0) += 1;
        let t3w = run_trace(&mut b3, &samples, ch, &trace)?;
        all_exact &= t3w.exact;
        sources.insert(
            "B3-warm".into(),
            source_cell(
                "B3-warm",
                &t3w,
                serde_json::json!({
                    "cache_state": warm.state_name(),
                    "resident_bytes": warm.resident_bytes,
                    "range_bytes": warm.range_bytes,
                    "method": warm.method,
                    "detail": warm.detail,
                    "persistent_encoded_bytes": info3.persistent_encoded_bytes,
                    "persistent_sample_domain_bytes": info3.persistent_sample_domain_bytes,
                }),
            ),
        );
        let _ = std::fs::remove_file(&path);

        // B4 — the exact B1 FLAC artifact, decoded once.
        let mut b1_artifact_sha: Option<String> = None;
        if generate::b1_comparable(o.channels) {
            let artifact =
                b1_flac_artifact(&samples, o.channels, o.sample_rate_hz, B1_LEVEL_PRIMARY)
                    .map_err(|e| Error::internal(format!("{}: B1 artifact: {e}", o.id)))?;
            b1_artifact_sha = Some(hex(&artifact.sha256));
            let mut b4 = FlacPreloadSource::prepare("B4", o.channels, o.sample_rate_hz, &artifact)?;
            let info4 = b4.info();
            let t4 = run_trace(&mut b4, &samples, ch, &trace)?;
            all_exact &= t4.exact;
            sources.insert(
                "B4".into(),
                source_cell(
                    "B4",
                    &t4,
                    serde_json::json!({
                        "artifact_sha256": hex(&artifact.sha256),
                        "persistent_encoded_bytes": info4.persistent_encoded_bytes,
                        "persistent_sample_domain_bytes": info4.persistent_sample_domain_bytes,
                        "setup_ns": info4.setup_ns,
                        "setup_detail": info4.setup_detail,
                    }),
                ),
            );
        } else {
            sources.insert(
                "B4".into(),
                serde_json::json!({
                    "source": "B4",
                    "status": "NOT_APPLICABLE_BY_FORMAT_DOMAIN",
                    "exact": serde_json::Value::Null,
                }),
            );
        }

        // B5 — bounded VOLE materialization over the verified container.
        let semantics = semantics_of(spec);
        let obj = fullobj::compile_full_object(
            &o.id,
            o.sample_rate_hz,
            o.channels,
            o.frames,
            semantics,
            &samples,
            budget(),
        )?;
        let container_sha = obj.sha256();
        let segment_count = obj.segment_count();
        for s in &obj.segments {
            *kind_counts.entry(s.kind.name()).or_insert(0) += 1;
        }
        let mut b5 = VoleBoundedSource::prepare("B5", obj.bytes.clone())?;
        let info5 = b5.info();
        let t5 = run_trace(&mut b5, &samples, ch, &trace)?;
        all_exact &= t5.exact;
        sources.insert(
            "B5".into(),
            source_cell(
                "B5",
                &t5,
                serde_json::json!({
                    "artifact_sha256": hex(&container_sha),
                    "container_bytes": obj.complete_bytes(),
                    "segments": segment_count,
                    "persistent_encoded_bytes": info5.persistent_encoded_bytes,
                    "persistent_sample_domain_bytes": info5.persistent_sample_domain_bytes,
                    "setup_ns": info5.setup_ns,
                    "setup_detail": info5.setup_detail,
                }),
            ),
        );

        total_windows += t2.windows as u64;
        exact_windows += t2.windows as u64;
        cells.push(serde_json::json!({
            "id": o.id,
            "source_structure_class": o.source_structure_class,
            "amplitude_class": o.amplitude_class,
            "channel_structure": o.channel_structure,
            "temporal_class": o.temporal_class,
            "entropy_class": o.entropy_class,
            "sample_rate_hz": o.sample_rate_hz,
            "channels": o.channels,
            "frames": o.frames,
            "canonical_i32_sha256": canonical,
            "b1_comparable": generate::b1_comparable(o.channels),
            "b1_artifact_sha256": b1_artifact_sha,
            "vole_container_sha256": hex(&container_sha),
            "vole_segments": segment_count,
            "trace_windows": trace.len(),
            "sources": serde_json::Value::Object(sources),
        }));
    }

    let _ = std::fs::remove_dir_all(&dir);
    let total_ns = sw.elapsed_ns().max(0) as u64;
    let object_count = cells.len();

    if !all_exact {
        return fail("a runtime source did not reproduce every requested window exactly");
    }

    let result_hex = hex(&Sha256::digest(&static_projection(
        &report.manifest_sha256,
        &report.corpus_sha256,
        &cells,
    )));
    if RUNTIME_RESULT_SHA256.is_empty() {
        eprintln!("court runtime: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != RUNTIME_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {RUNTIME_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    let verdict = Verdict::Supported;
    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!("{} + runtime.v1", manifest.profile)),
        backend: Some("common runtime substrate: B2/B3/B4/B5".into()),
        sample_rate_hz: None, // per object, native
        quantum_frames: Some(QUANTUM_FRAMES),
        content_kind: Some("flagship corpus (frozen before results): runtime mechanism".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("runtime");
    builder
        .result(verdict)
        .result_detail(format!(
            "runtime mechanism over {} frozen objects; frozen sequential {QUANTUM_FRAMES}-frame trace at \
             each object's native rate/channels; B2/B3-cold/B3-warm/B5 for every object and B4 for the \
             B1-comparable objects; every requested window reproduced exactly on every source; result \
             sha256 {result_hex}",
            cells.len(),
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
            corpus_hash: Some(report.corpus_sha256.clone()),
            ..Default::default()
        })
        .extra(
            "protocol",
            serde_json::json!({
                "quantum_frames": QUANTUM_FRAMES,
                "trace": "sequential windows [0,512), [512,1024), ... plus a final partial window",
                "rate": "each object's native rate (no resampling)",
                "channels": "each object's native channel count",
                "transforms": "none: no gain, pan, filter or random seek",
                "output": "exact canonical interleaved i32 into a caller-owned destination",
                "deadline": "computed independently by each source from the native rate",
                "scope": "source materialization, not endpoint/resampling performance",
                "outside_the_interface": "artifact construction (inverse search, FLAC encode, container \
                                           compile) is authoring/compile work, never a playback operation",
            }),
        )
        .extra(
            "sources",
            serde_json::json!({
                "B2": "PCM resident (canonical i32 in memory)",
                "B3-cold": "raw PCM disk streaming, cold-verified (fadvise + mincore)",
                "B3-warm": "raw PCM disk streaming, warm-verified (primed + mincore)",
                "B4": "the exact B1 FLAC-5 artifact, one-time full decode -> resident PCM (no seektable)",
                "B5": "bounded VOLE materialization over the verified full-object container",
            }),
        )
        .extra(
            "aggregate",
            serde_json::json!({
                "objects": cells.len(),
                "trace_windows_per_source": total_windows,
                "exact_windows": exact_windows,
                "all_sources_exact": all_exact,
                "cache_states": cache_states,
                "selected_candidate_kinds": kind_counts,
                "total_ns": total_ns,
            }),
        )
        .extra(
            "measured_latency_note",
            serde_json::json!({
                "status": "MEASURED_EXCLUDED_FROM_FROZEN_HASH",
                "detail": "per-source latency totals and maxima are recorded as measured evidence; the \
                           frozen static result covers the protocol, artifact identities, counters and \
                           the correctness vector only",
            }),
        )
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "this court freezes the runtime mechanism and proves exactness; it reports no comparative \
             timing headline. B2-B5 measurement is the next increment. Measured latency and physical \
             storage-read traffic are recorded as evidence but are deliberately excluded from the \
             frozen static result, which covers the protocol, artifact identities, deterministic \
             counters and the correctness vector",
        )
        .limitation(
            "B3 cold/warm states are VERIFIED (posix_fadvise is only an attempt, so mincore confirms \
             residency); when residency cannot be established the state is CACHE_STATE_NOT_CONFIRMED, \
             never assumed. Global drop_caches is never used, and the B3 artifacts live on a \
             block-backed filesystem because fadvise cannot evict tmpfs pages",
        )
        .limitation(
            "B4 is a compressed-storage / decoded-resident sampler: the sealed B1 stream is decoded \
             once in full and never modified to gain a seektable",
        )
        .limitation(
            "B5 verifies the container once outside the timed path and then reads bounded windows; the \
             reader retains encoded segment state only, never a decoded waveform",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court runtime: {verdict}");
    println!(
        "  objects: {} | trace windows per source: {total_windows} | all sources exact: {all_exact}",
        object_count
    );
    println!("  cache states: {cache_states:?}");
    println!("  selected kinds: {kind_counts:?}");
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
