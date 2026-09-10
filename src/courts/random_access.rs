//! `court random-access` — bounded random access cost (Phase M).
//!
//! The runtime court measures the frozen *sequential* trace. This court measures
//! the other half of the access question: deterministic **random** windows, at
//! the same frozen quantum, over the same architectures, plus the honest
//! conventional compressed-seek row.
//!
//! ```text
//! B2        PCM resident                        (random access = bounded copy)
//! B3-warm   raw PCM on disk                     (positioned reads, verified warm)
//! B4        the exact B1 FLAC-5 artifact, decoded-resident
//! B4-seek   the same artifact, stateless decode_seek (no SEEKTABLE)
//! B5        bounded VOLE materialization        (pages/segments touched per seek)
//! ```
//!
//! The random request sequence is derived from each object's canonical hash, so
//! it is frozen and never chosen to favour a backend. It always includes the
//! first frame, the last frame, a window straddling the 65,536-frame container
//! boundary, and a full quantum at the end.
//!
//! Latency remains harness-owned (see [`super::measure`]); residency is
//! split into storage / resident-sample / resident-encoded as in the runtime
//! court.

use crate::baseline::{B1_LEVEL_PRIMARY, FlacArtifact, b1_flac_artifact};
use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::{Error, Result};
use crate::evidence::TailSummary;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::fullobj::{self, FullSemantics, VerifiedFullObject};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::runtime::cache::CacheState;
use crate::runtime::{
    DiskPcmArtifact, DiskPcmSource, FlacPreloadSource, FlacSeekSource, QUANTUM_FRAMES, REPEATS,
    ResidentPcmSource, RuntimeSource, VoleBoundedSource, WindowPlan, frozen_random_trace,
    rotate_order,
};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use super::measure::{Traversal, run_traversal, timer_overhead_ns};

/// Frozen static-result hash (empty means "not yet frozen").
pub const RANDOM_ACCESS_RESULT_SHA256: &str =
    "3a76aef7b18fa981a5e5bf019acb6ac6b2f70ee1e355e191659416c2f9a09b48";

/// The frozen random-access protocol identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.random_access.protocol.v1";

/// Pseudo-random windows per object (plus the four forced edge cases).
pub const RANDOM_WINDOWS: usize = 64;

fn budget() -> SearchBudget {
    SearchBudget::default()
}

fn semantics_of(spec: &Spec) -> FullSemantics {
    match spec.semantics.period_frames() {
        Some(period_frames) => FullSemantics::Loop { period_frames },
        None => FullSemantics::OneShot,
    }
}

const SOURCES: [&str; 5] = ["B2", "B3-warm", "B4", "B4-seek", "B5"];

/// Deterministic counters plus pooled measured samples for one source.
///
/// Constructed only through [`Acc::measured`] / [`Acc::not_applicable`] so the
/// correctness accumulator starts `true` (a `Default` would start `false` and
/// never recover under `&=`) and a format-domain row is never serialized as a
/// measured one.
#[derive(Clone)]
struct Acc {
    status: &'static str,
    exact: bool,
    started: bool,
    det: Traversal,
    latencies: Vec<u64>,
    sequential: Vec<u64>,
    physical_storage_bytes_read: u64,
    deadline_misses: u64,
    artifact_storage_bytes: u64,
    resident_sample_domain_bytes: u64,
    resident_encoded_bytes: u64,
    runtime_setup_ns: u64,
    artifact_sha256: Option<String>,
    setup_detail: String,
}

impl Acc {
    fn new(status: &'static str) -> Self {
        Acc {
            status,
            exact: true,
            started: false,
            det: Traversal::default(),
            latencies: Vec::new(),
            sequential: Vec::new(),
            physical_storage_bytes_read: 0,
            deadline_misses: 0,
            artifact_storage_bytes: 0,
            resident_sample_domain_bytes: 0,
            resident_encoded_bytes: 0,
            runtime_setup_ns: 0,
            artifact_sha256: None,
            setup_detail: String::new(),
        }
    }

    fn measured() -> Self {
        Acc::new("MEASURED")
    }

    fn not_applicable() -> Self {
        let mut a = Acc::new("NOT_APPLICABLE_BY_FORMAT_DOMAIN");
        a.setup_detail = "outside FLAC's format domain (>8 channels)".into();
        a
    }

    fn is_measured(&self) -> bool {
        self.status == "MEASURED"
    }

    fn record(&mut self, t: &Traversal) {
        if !self.started {
            self.started = true;
            self.det = t.clone();
            self.det.latencies = Vec::new();
        }
        self.exact &= t.exact;
        self.latencies.extend_from_slice(&t.latencies);
        self.physical_storage_bytes_read += t.physical_storage_bytes_read;
        self.deadline_misses += t.deadline_misses;
    }

    fn cell(&self, source: &str) -> serde_json::Value {
        if !self.is_measured() {
            return serde_json::json!({
                "source": source,
                "status": self.status,
                "exact": serde_json::Value::Null,
                "setup_detail": self.setup_detail,
            });
        }
        serde_json::json!({
            "source": source,
            "status": self.status,
            "exact": self.exact,
            "windows_per_traversal": self.det.windows,
            "requested_frames": self.det.requested_frames,
            "returned_frames": self.det.returned_frames,
            "output_bytes": self.det.output_bytes,
            "logical_source_bytes_read": self.det.logical_source_bytes_read,
            "encoded_bytes_examined": self.det.encoded_bytes_examined,
            "encoded_bytes_parsed": self.det.encoded_bytes_parsed,
            "sample_domain_bytes_materialized": self.det.sample_domain_bytes_materialized,
            "segments_touched": self.det.segments_touched,
            "pages_touched": self.det.pages_touched,
            "scratch_peak_bytes": self.det.scratch_peak_bytes,
            "artifact_storage_bytes": self.artifact_storage_bytes,
            "resident_sample_domain_bytes": self.resident_sample_domain_bytes,
            "resident_encoded_bytes": self.resident_encoded_bytes,
            "artifact_sha256": self.artifact_sha256,
            "setup_detail": self.setup_detail,
            "measured": {
                "random_latency": TailSummary::summarize(&self.latencies),
                "sequential_latency": TailSummary::summarize(&self.sequential),
                "random_total_latency_ns": self.latencies.iter().sum::<u64>(),
                "physical_storage_bytes_read": self.physical_storage_bytes_read,
                "deadline_misses": self.deadline_misses,
                "runtime_setup_ns": self.runtime_setup_ns,
            },
        })
    }
}

fn static_projection(manifest_sha: &str, corpus_sha: &str, cells: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for head in [PROTOCOL_SCHEMA, manifest_sha, corpus_sha] {
        out.extend_from_slice(head.as_bytes());
        out.push(0);
    }
    out.extend_from_slice(&QUANTUM_FRAMES.to_le_bytes());
    out.extend_from_slice(&(RANDOM_WINDOWS as u64).to_le_bytes());
    out.extend_from_slice(&REPEATS.to_le_bytes());
    for c in cells {
        for key in ["id", "canonical_i32_sha256"] {
            out.extend_from_slice(c[key].as_str().unwrap_or("").as_bytes());
            out.push(0);
        }
        out.extend_from_slice(&c["channels"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["frames"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["sample_rate_hz"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["random_windows"].as_u64().unwrap_or(0).to_le_bytes());
        if let Some(sources) = c["sources"].as_object() {
            for (name, s) in sources {
                out.extend_from_slice(name.as_bytes());
                out.push(0);
                out.push(matches!(s["status"].as_str(), Some("MEASURED")) as u8);
                out.push(s["exact"].as_bool().unwrap_or(false) as u8);
                for key in [
                    "windows_per_traversal",
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
                    "artifact_storage_bytes",
                    "resident_sample_domain_bytes",
                    "resident_encoded_bytes",
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

/// Run the court; writes an immutable receipt under `receipts/random-access/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("random-access");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("random-access failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court random-access: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let manifest = corpus::manifest()?;
    let report = corpus::verify_manifest(&manifest)?;
    if !report.ok() {
        return fail("the frozen flagship corpus does not verify");
    }
    let specs = corpus::specs::specs();
    let spec_by_id: BTreeMap<&str, &Spec> = specs.iter().map(|s| (s.id.as_str(), s)).collect();

    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("random-access-b3-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;

    let mut cells: Vec<serde_json::Value> = Vec::with_capacity(manifest.objects.len());
    let mut all_exact = true;
    let mut pool: BTreeMap<&'static str, Acc> =
        SOURCES.iter().map(|s| (*s, Acc::new("POOLED"))).collect();
    // Per-source random windows, summed only over the objects where the source
    // actually exists (B4/B4-seek cover the 110 B1-domain objects, not all 115).
    let mut pool_windows: BTreeMap<&'static str, u64> =
        SOURCES.iter().map(|s| (*s, 0u64)).collect();

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
        let seed = seed_from_hex(&canonical);
        let random = WindowPlan::from_spans(
            &samples,
            ch,
            frozen_random_trace(o.frames, QUANTUM_FRAMES, seed, RANDOM_WINDOWS)?,
        )?;
        let sequential = WindowPlan::build(&samples, ch)?;

        // ---- authoring ----
        let mut b2 = ResidentPcmSource::open("B2", o.channels, o.sample_rate_hz, samples.clone())?;
        let path = dir.join(format!("{}.pcm", o.id));
        let b3_art = DiskPcmArtifact::create(&path, o.channels, o.sample_rate_hz, &samples)?;
        let mut b3 = DiskPcmSource::open("B3-warm", &b3_art)?;
        let b4_artifact: Option<FlacArtifact> = if generate::b1_comparable(o.channels) {
            Some(
                b1_flac_artifact(&samples, o.channels, o.sample_rate_hz, B1_LEVEL_PRIMARY)
                    .map_err(|e| Error::internal(format!("{}: B1 artifact: {e}", o.id)))?,
            )
        } else {
            None
        };
        let mut b4 = match &b4_artifact {
            Some(a) => Some(FlacPreloadSource::open(
                "B4",
                o.channels,
                o.sample_rate_hz,
                a,
            )?),
            None => None,
        };
        let mut b4s = match &b4_artifact {
            Some(a) => Some(FlacSeekSource::open(
                "B4-seek",
                o.channels,
                o.sample_rate_hz,
                a,
            )?),
            None => None,
        };
        let obj = fullobj::compile_full_object(
            &o.id,
            o.sample_rate_hz,
            o.channels,
            o.frames,
            semantics_of(spec),
            &samples,
            budget(),
        )?;
        let container_sha = obj.sha256();
        let verified = Arc::new(VerifiedFullObject::verify(obj.bytes.clone())?);
        let mut b5 = VoleBoundedSource::open("B5", verified.clone(), false)?;

        let mut accs: BTreeMap<&'static str, Acc> =
            SOURCES.iter().map(|s| (*s, Acc::measured())).collect();
        if b4.is_none() {
            accs.insert("B4", Acc::not_applicable());
            accs.insert("B4-seek", Acc::not_applicable());
        }
        accs.get_mut("B2").unwrap().artifact_storage_bytes = b2.info().artifact_storage_bytes;
        accs.get_mut("B2").unwrap().resident_sample_domain_bytes =
            b2.info().resident_sample_domain_bytes;
        accs.get_mut("B2").unwrap().runtime_setup_ns = b2.info().runtime_setup_ns;
        accs.get_mut("B2").unwrap().setup_detail = b2.info().setup_detail;
        {
            let i = b3.info();
            let a = accs.get_mut("B3-warm").unwrap();
            a.artifact_storage_bytes = i.artifact_storage_bytes;
            a.resident_sample_domain_bytes = i.resident_sample_domain_bytes;
            a.resident_encoded_bytes = i.resident_encoded_bytes;
            a.runtime_setup_ns = i.runtime_setup_ns;
            a.setup_detail = i.setup_detail;
        }
        if let Some(a) = b4.as_ref() {
            let i = a.info();
            let acc = accs.get_mut("B4").unwrap();
            acc.artifact_storage_bytes = i.artifact_storage_bytes;
            acc.resident_sample_domain_bytes = i.resident_sample_domain_bytes;
            acc.resident_encoded_bytes = i.resident_encoded_bytes;
            acc.runtime_setup_ns = i.runtime_setup_ns;
            acc.setup_detail = i.setup_detail;
            acc.artifact_sha256 = Some(hex(&b4_artifact.as_ref().unwrap().sha256));
        }
        if let Some(a) = b4s.as_ref() {
            let i = a.info();
            let acc = accs.get_mut("B4-seek").unwrap();
            acc.artifact_storage_bytes = i.artifact_storage_bytes;
            acc.resident_sample_domain_bytes = i.resident_sample_domain_bytes;
            acc.resident_encoded_bytes = i.resident_encoded_bytes;
            acc.runtime_setup_ns = i.runtime_setup_ns;
            acc.setup_detail = i.setup_detail;
            acc.artifact_sha256 = Some(hex(&b4_artifact.as_ref().unwrap().sha256));
        }
        {
            let i = b5.info();
            let acc = accs.get_mut("B5").unwrap();
            acc.artifact_storage_bytes = i.artifact_storage_bytes;
            acc.resident_sample_domain_bytes = i.resident_sample_domain_bytes;
            acc.resident_encoded_bytes = i.resident_encoded_bytes;
            acc.runtime_setup_ns = i.runtime_setup_ns;
            acc.setup_detail = i.setup_detail;
            acc.artifact_sha256 = Some(hex(&container_sha));
        }

        let mut dst = vec![0i32; QUANTUM_FRAMES as usize * ch];
        let mut scratch: Vec<u8> = Vec::new();
        let mut slots: Vec<&'static str> = vec!["B2", "B3-warm"];
        if b4.is_some() {
            slots.push("B4");
            slots.push("B4-seek");
        }
        slots.push("B5");

        // One sequential pass per source, for the random-vs-sequential penalty.
        // B4-seek is skipped: its only honest row is random access (sequential
        // access is B4's decoded-resident path), and decoding forward from the
        // first frame for every sequential window would be quadratic and
        // meaningless.
        for name in &slots {
            if *name == "B4-seek" {
                continue;
            }
            let t = run_named(
                name,
                &mut b2,
                &mut b3,
                b4.as_mut(),
                b4s.as_mut(),
                &mut b5,
                &sequential,
                ch,
                o.sample_rate_hz,
                &mut dst,
                &mut scratch,
            )?;
            all_exact &= t.exact;
            accs.get_mut(*name).unwrap().sequential = t.latencies;
        }

        // Frozen random trace, repeated with rotated source order.
        for repeat in 0..REPEATS {
            for &idx in &rotate_order(slots.len(), repeat) {
                let name = slots[idx];
                if name == "B3-warm" {
                    let ev = b3.prime_and_verify()?;
                    if ev.state != CacheState::Warm {
                        return fail(&format!("{}: B3 warm state not verified", o.id));
                    }
                }
                let t = run_named(
                    name,
                    &mut b2,
                    &mut b3,
                    b4.as_mut(),
                    b4s.as_mut(),
                    &mut b5,
                    &random,
                    ch,
                    o.sample_rate_hz,
                    &mut dst,
                    &mut scratch,
                )?;
                all_exact &= t.exact;
                accs.get_mut(name).unwrap().record(&t);
            }
        }

        for (name, acc) in &accs {
            if !acc.is_measured() {
                continue;
            }
            let p = pool.get_mut(name).unwrap();
            p.exact &= acc.exact;
            p.latencies.extend_from_slice(&acc.latencies);
            p.sequential.extend_from_slice(&acc.sequential);
            p.deadline_misses += acc.deadline_misses;
            p.physical_storage_bytes_read += acc.physical_storage_bytes_read;
            p.artifact_storage_bytes += acc.artifact_storage_bytes;
            p.resident_sample_domain_bytes += acc.resident_sample_domain_bytes;
            p.resident_encoded_bytes += acc.resident_encoded_bytes;
            p.runtime_setup_ns += acc.runtime_setup_ns;
            *pool_windows.get_mut(name).unwrap() += acc.det.windows as u64;
        }

        let _ = std::fs::remove_file(&path);

        let sources: serde_json::Map<String, serde_json::Value> = accs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.cell(k)))
            .collect();
        cells.push(serde_json::json!({
            "id": o.id,
            "sample_rate_hz": o.sample_rate_hz,
            "channels": o.channels,
            "frames": o.frames,
            "canonical_i32_sha256": canonical,
            "b1_comparable": generate::b1_comparable(o.channels),
            "random_windows": random.len(),
            "sources": serde_json::Value::Object(sources),
        }));
    }

    let _ = std::fs::remove_dir_all(&dir);
    if !all_exact {
        return fail("a random-access window was not reproduced exactly");
    }

    let result_hex = hex(&Sha256::digest(&static_projection(
        &report.manifest_sha256,
        &report.corpus_sha256,
        &cells,
    )));
    if RANDOM_ACCESS_RESULT_SHA256.is_empty() {
        eprintln!("court random-access: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != RANDOM_ACCESS_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {RANDOM_ACCESS_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    let (timer_min_ns, timer_median_ns) = timer_overhead_ns();
    let surface = random_surface(&pool, &pool_windows, timer_min_ns, timer_median_ns);
    let object_count = cells.len();
    let total_random_windows: u64 = cells
        .iter()
        .map(|c| c["random_windows"].as_u64().unwrap_or(0))
        .sum();
    let verdict = Verdict::Supported;
    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!("{} + random-access.v1", manifest.profile)),
        backend: Some("B2 / B3-warm / B4 / B4-seek / B5".into()),
        sample_rate_hz: None,
        quantum_frames: Some(QUANTUM_FRAMES),
        content_kind: Some("flagship corpus (frozen before results): random access".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("random-access");
    builder
        .result(verdict)
        .result_detail(format!(
            "bounded random access over {} frozen objects; {RANDOM_WINDOWS} pseudo-random windows per \
             object plus the first frame, last frame, a 65,536-frame boundary straddle and a final \
             full quantum, repeated {REPEATS} times with rotated source order; B2/B3-warm/B4/B4-seek/B5 \
             (B4 rows for the {} B1-comparable objects); every window reproduced exactly; result sha256 \
             {result_hex}",
            cells.len(),
            cells
                .iter()
                .filter(|c| c["b1_comparable"].as_bool().unwrap_or(false))
                .count(),
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
                "schema": PROTOCOL_SCHEMA,
                "quantum_frames": QUANTUM_FRAMES,
                "random_windows_per_object": RANDOM_WINDOWS,
                "repeats": REPEATS,
                "seed": "derived from each object's canonical i32 SHA-256 (never from measurement)",
                "forced_edges": "first frame, last frame, 65,536-frame boundary straddle, final full quantum",
                "widths": "1, 64, 256 and 512 frames",
                "latency_boundary": "harness-owned, wrapping the whole RuntimeSource::read call",
                "sequential_reference": "one sequential pass per source gives the random-vs-sequential penalty",
                "populations": "all 115 objects (B2/B3-warm/B5) and the 110 B1-domain objects (adding B4/B4-seek)",
            }),
        )
        .extra("measurement", surface)
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "B4 is decoded-resident, so its random access is a bounded copy; B4-seek is the honest \
             stateless compressed-seek row and, because the sealed B1 stream has no SEEKTABLE, each \
             B4-seek window decodes forward from the first frame. The two are never averaged",
        )
        .limitation(
            "latency is harness-owned and sub-100 ns rows are near the apparatus floor \
             (`timer_overhead_ns`); physical storage traffic is sampled at traversal boundaries",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court random-access: {verdict}");
    println!(
        "  objects: {object_count} | random windows total: {total_random_windows} | all exact: {all_exact}"
    );
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

/// Run one named source over a plan.
#[allow(clippy::too_many_arguments)]
fn run_named(
    name: &str,
    b2: &mut ResidentPcmSource,
    b3: &mut DiskPcmSource,
    b4: Option<&mut FlacPreloadSource>,
    b4s: Option<&mut FlacSeekSource>,
    b5: &mut VoleBoundedSource,
    plan: &WindowPlan,
    ch: usize,
    rate: u32,
    dst: &mut [i32],
    scratch: &mut Vec<u8>,
) -> Result<Traversal> {
    match name {
        "B2" => {
            b2.reset()?;
            run_traversal(b2, plan, ch, rate, dst, scratch)
        }
        "B3-warm" => run_traversal(b3, plan, ch, rate, dst, scratch),
        "B4" => {
            let s = b4.ok_or_else(|| Error::internal("B4 not available"))?;
            run_traversal(s, plan, ch, rate, dst, scratch)
        }
        "B4-seek" => {
            let s = b4s.ok_or_else(|| Error::internal("B4-seek not available"))?;
            run_traversal(s, plan, ch, rate, dst, scratch)
        }
        "B5" => {
            b5.reset()?;
            run_traversal(b5, plan, ch, rate, dst, scratch)
        }
        other => Err(Error::internal(format!("unknown source '{other}'"))),
    }
}

fn seed_from_hex(canonical: &str) -> u64 {
    let mut seed = 0u64;
    for b in canonical.bytes().take(16) {
        seed = seed.wrapping_mul(16).wrapping_add(u64::from(b));
    }
    seed
}

fn random_surface(
    pool: &BTreeMap<&'static str, Acc>,
    pool_windows: &BTreeMap<&'static str, u64>,
    timer_min_ns: u64,
    timer_median_ns: u64,
) -> serde_json::Value {
    let mut per = serde_json::Map::new();
    for (name, acc) in pool {
        let random = TailSummary::summarize(&acc.latencies);
        let sequential = TailSummary::summarize(&acc.sequential);
        let penalty = match (random.p50_ns, sequential.p50_ns) {
            (Some(r), Some(s)) if s > 0 => Some(r as f64 / s as f64),
            _ => None,
        };
        per.insert(
            (*name).to_string(),
            serde_json::json!({
                "exact": acc.exact,
                "random_windows": pool_windows.get(name).copied().unwrap_or(0),
                "random_latency": random,
                "sequential_latency": sequential,
                "random_over_sequential_p50": penalty,
                "artifact_storage_bytes": acc.artifact_storage_bytes,
                "resident_sample_domain_bytes": acc.resident_sample_domain_bytes,
                "resident_encoded_bytes": acc.resident_encoded_bytes,
                "runtime_setup_ns": acc.runtime_setup_ns,
                "physical_storage_bytes_read": acc.physical_storage_bytes_read,
                "deadline_misses": acc.deadline_misses,
            }),
        );
    }
    serde_json::json!({
        "timer_overhead_ns": { "min": timer_min_ns, "median": timer_median_ns },
        "note": "random and sequential latency are pooled over the objects where the source exists \
                 (B4/B4-seek cover the 110 B1-domain objects only); random_windows is that \
                 source's own window total. random_over_sequential_p50 is the random-access \
                 penalty. Per-object detail is in cells",
        "sources": per,
    })
}
