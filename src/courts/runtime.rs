//! `court runtime` — the common runtime substrate (Phase M, Seals 9–10).
//!
//! One contract, four source architectures, measured under one protocol:
//!
//! ```text
//! B2  PCM-resident sampler          canonical i32 held in memory
//! B3  raw PCM disk streaming        canonical LE i32 on disk, cold/warm VERIFIED
//! B4  compressed preload            the exact B1 FLAC-5 artifact, decoded once
//! B5  bounded VOLE materialization  verified full-object container, page-bounded
//! ```
//!
//! The **measurement boundary**, **source state** and **population** are frozen
//! here, not left to each adapter:
//!
//! * latency is the harness's: a stopwatch wraps the entire
//!   [`RuntimeSource::read`] call, so B3 cannot report "kernel read" while B2/B4
//!   time their copy to `dst`;
//! * `/proc/self/io` is sampled at traversal boundaries, never inside a timer;
//! * residency is split into storage / resident-sample / resident-encoded, so a
//!   disk source is not credited with resident PCM it does not hold;
//! * B3 artifact creation is authoring, not runtime setup;
//! * the trace is the frozen sequential [`QUANTUM_FRAMES`]-frame run, repeated
//!   [`REPEATS`] times with deterministic source-order rotation, each repeat
//!   starting from equivalent source state (B3 re-verifies cold/warm; B5
//!   first-play gets a fresh bounded reader);
//! * every aggregate is computed inside an explicit population: all 115 objects
//!   (where B4 does not exist) and the 110 B1-domain objects (where it does).
//!
//! The frozen static result covers the protocol, artifact identities,
//! deterministic counters and the correctness vector. Latency, physical storage
//! traffic and setup times are measured evidence and are **excluded** from it.

use crate::baseline::{B1_LEVEL_PRIMARY, FlacArtifact, b1_flac_artifact};
use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::{Error, Result};
use crate::evidence::TailSummary;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder, TraceInfo};
use crate::evidence::timing::Stopwatch;
use crate::evidence::trace::TraceFile;
use crate::fullobj::{self, FullSemantics, VerifiedFullObject};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::runtime::cache::CacheState;
use crate::runtime::{
    DiskPcmArtifact, DiskPcmSource, FlacPreloadSource, QUANTUM_FRAMES, REPEATS, ResidentPcmSource,
    RuntimeSource, VoleBoundedSource, WindowPlan, rotate_order,
};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use super::measure::{Traversal, run_traversal, timer_overhead_ns};

/// Frozen static-result hash: the court fails if a change silently alters the
/// runtime protocol, artifact identities, counters or correctness vector.
/// Empty means "not yet frozen".
pub const RUNTIME_RESULT_SHA256: &str =
    "d7d11681141f5f92baf1a22d572b40d5cb7e1813f902d2f5067a2e1dbd54ba79";

/// The frozen external measurement protocol identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.runtime.protocol.v2";

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

/// The source slots of the runtime ladder. B4 is absent for objects outside
/// FLAC's format domain.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    B2,
    B3Cold,
    B3Warm,
    B4,
    B5,
    B5Prepared,
}

impl Slot {
    fn name(self) -> &'static str {
        match self {
            Slot::B2 => "B2",
            Slot::B3Cold => "B3-cold",
            Slot::B3Warm => "B3-warm",
            Slot::B4 => "B4",
            Slot::B5 => "B5",
            Slot::B5Prepared => "B5-prepared",
        }
    }
}

/// Deterministic counters and measured samples from one traversal.
// (shared: `super::measure::Traversal`)
#[derive(Clone)]
struct SourceAgg {
    status: &'static str,
    repeats_total: u32,
    repeats_eligible: u32,
    exact: bool,
    windows_per_traversal: usize,
    /// Deterministic counters, identical across repeats (sources reset).
    det: Traversal,
    /// Measured samples pooled over eligible repeats.
    latencies: Vec<u64>,
    physical_storage_bytes_read: u64,
    deadline_misses: u64,
    worst_deadline_margin_ns: i64,
    cache_states: BTreeMap<&'static str, u32>,
    ineligible_reasons: Vec<String>,
    artifact_build_ns: u64,
    runtime_setup_ns: u64,
    artifact_storage_bytes: u64,
    resident_sample_domain_bytes: u64,
    resident_encoded_bytes: u64,
    artifact_sha256: Option<String>,
    setup_detail: String,
}

impl SourceAgg {
    fn new(status: &'static str, info: &crate::runtime::SourceInfo) -> Self {
        SourceAgg {
            status,
            repeats_total: REPEATS,
            repeats_eligible: 0,
            exact: true,
            windows_per_traversal: 0,
            det: Traversal::default(),
            latencies: Vec::new(),
            physical_storage_bytes_read: 0,
            deadline_misses: 0,
            worst_deadline_margin_ns: i64::MAX,
            cache_states: BTreeMap::new(),
            ineligible_reasons: Vec::new(),
            artifact_build_ns: 0,
            runtime_setup_ns: info.runtime_setup_ns,
            artifact_storage_bytes: info.artifact_storage_bytes,
            resident_sample_domain_bytes: info.resident_sample_domain_bytes,
            resident_encoded_bytes: info.resident_encoded_bytes,
            artifact_sha256: None,
            setup_detail: info.setup_detail.clone(),
        }
    }

    /// A row that exists but is outside the format domain: no counters at all.
    fn not_applicable() -> Self {
        SourceAgg {
            status: "NOT_APPLICABLE_BY_FORMAT_DOMAIN",
            repeats_total: 0,
            repeats_eligible: 0,
            exact: true,
            windows_per_traversal: 0,
            det: Traversal::default(),
            latencies: Vec::new(),
            physical_storage_bytes_read: 0,
            deadline_misses: 0,
            worst_deadline_margin_ns: i64::MAX,
            cache_states: BTreeMap::new(),
            ineligible_reasons: Vec::new(),
            artifact_build_ns: 0,
            runtime_setup_ns: 0,
            artifact_storage_bytes: 0,
            resident_sample_domain_bytes: 0,
            resident_encoded_bytes: 0,
            artifact_sha256: None,
            setup_detail: "outside FLAC's format domain (>8 channels)".into(),
        }
    }

    fn record(&mut self, t: &Traversal) {
        if self.repeats_eligible == 0 {
            self.windows_per_traversal = t.windows;
            self.det = t.clone();
            self.det.latencies = Vec::new();
        } else {
            debug_assert_eq!(
                self.det.encoded_bytes_examined, t.encoded_bytes_examined,
                "deterministic counters must not vary across repeats"
            );
        }
        self.repeats_eligible += 1;
        self.exact &= t.exact;
        self.latencies.extend_from_slice(&t.latencies);
        self.physical_storage_bytes_read += t.physical_storage_bytes_read;
        self.deadline_misses += t.deadline_misses;
        self.worst_deadline_margin_ns = self
            .worst_deadline_margin_ns
            .min(t.worst_deadline_margin_ns);
    }

    fn mark_ineligible(&mut self, why: String) {
        self.ineligible_reasons.push(why);
    }

    fn cell(&self, source_name: &str) -> serde_json::Value {
        let margin = if self.repeats_eligible == 0 {
            0
        } else {
            self.worst_deadline_margin_ns
        };
        serde_json::json!({
            "source": source_name,
            "status": self.status,
            "repeats_total": self.repeats_total,
            "repeats_eligible": self.repeats_eligible,
            "exact": self.exact,
            "windows_per_traversal": self.windows_per_traversal,
            "requested_frames": self.det.requested_frames,
            "returned_frames": self.det.returned_frames,
            "output_bytes": self.det.output_bytes,
            "logical_source_bytes_read": self.det.logical_source_bytes_read,
            "encoded_bytes_examined": self.det.encoded_bytes_examined,
            "encoded_bytes_parsed": self.det.encoded_bytes_parsed,
            "sample_domain_bytes_materialized": self.det.sample_domain_bytes_materialized,
            "segments_touched": self.det.segments_touched,
            "pages_touched": self.det.pages_touched,
            "working_state_bytes": self.det.working_state_bytes,
            "scratch_peak_bytes": self.det.scratch_peak_bytes,
            "artifact_storage_bytes": self.artifact_storage_bytes,
            "resident_sample_domain_bytes": self.resident_sample_domain_bytes,
            "resident_encoded_bytes": self.resident_encoded_bytes,
            "artifact_sha256": self.artifact_sha256,
            "setup_detail": self.setup_detail,
            "cache_states": self.cache_states,
            "ineligible_reasons": self.ineligible_reasons,
            "measured": {
                "latency": TailSummary::summarize(&self.latencies),
                "total_latency_ns": self.latencies.iter().sum::<u64>(),
                "physical_storage_bytes_read": self.physical_storage_bytes_read,
                "deadline_misses": self.deadline_misses,
                "worst_deadline_margin_ns": margin,
                "runtime_setup_ns": self.runtime_setup_ns,
                "artifact_build_ns": self.artifact_build_ns,
            },
        })
    }
}

/// Deterministic projection: protocol, artifact identities, counters and the
/// correctness vector. No measured timing and no physical I/O.
fn static_projection(manifest_sha: &str, corpus_sha: &str, cells: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for head in [PROTOCOL_SCHEMA, manifest_sha, corpus_sha] {
        out.extend_from_slice(head.as_bytes());
        out.push(0);
    }
    out.extend_from_slice(&QUANTUM_FRAMES.to_le_bytes());
    out.extend_from_slice(&REPEATS.to_le_bytes());
    for c in cells {
        for key in [
            "id",
            "canonical_i32_sha256",
            "source_structure_class",
            "amplitude_class",
            "channel_structure",
            "temporal_class",
            "entropy_class",
        ] {
            out.extend_from_slice(c[key].as_str().unwrap_or("").as_bytes());
            out.push(0);
        }
        out.extend_from_slice(&c["channels"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["frames"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["sample_rate_hz"].as_u64().unwrap_or(0).to_le_bytes());
        out.push(c["b1_comparable"].as_bool().unwrap_or(false) as u8);
        if let Some(sources) = c["sources"].as_object() {
            for (name, s) in sources {
                out.extend_from_slice(name.as_bytes());
                out.push(0);
                out.push(matches!(s["status"].as_str(), Some("MEASURED")) as u8);
                out.push(s["exact"].as_bool().unwrap_or(false) as u8);
                for key in [
                    "repeats_total",
                    "repeats_eligible",
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
                    "working_state_bytes",
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

/// Run the court; writes an immutable receipt under `receipts/runtime/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("runtime");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("runtime measurement failed: {why}"));
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

    // Raw latency vectors are bulk artifacts: they live under `receipts/traces/`
    // (gitignored) and are bound to the receipt by their SHA-256.
    let trace_dir = receipts_root.join("traces").join("runtime");
    std::fs::create_dir_all(&trace_dir)?;
    let trace_name = format!(
        "runtime-{}-{}.jsonl",
        std::process::id(),
        crate::evidence::timing::monotonic_raw_ns() / 1_000_000
    );
    let mut trace = TraceFile::create_in(&trace_dir, &trace_name)?;

    let sw = Stopwatch::start();
    let mut cells: Vec<serde_json::Value> = Vec::with_capacity(manifest.objects.len());
    let mut all_exact = true;
    let mut total_windows_b2: u64 = 0;
    let mut total_traversals: u64 = 0;
    let mut pool_all = Pool::default();
    let mut pool_b1 = Pool::default();
    let mut strata: BTreeMap<String, Pool> = BTreeMap::new();

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
        let plan = WindowPlan::build(&samples, ch)?;

        // ---- authoring (outside the runtime measurement) ----
        let mut b2 = ResidentPcmSource::open("B2", o.channels, o.sample_rate_hz, samples.clone())?;

        let path = dir.join(format!("{}.pcm", o.id));
        let b3_artifact = DiskPcmArtifact::create(&path, o.channels, o.sample_rate_hz, &samples)?;
        let b3_build_ns = b3_artifact.build_ns();
        let mut b3 = DiskPcmSource::open("B3", &b3_artifact)?;

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

        let author_sw = Stopwatch::start();
        let obj = fullobj::compile_full_object(
            &o.id,
            o.sample_rate_hz,
            o.channels,
            o.frames,
            semantics_of(spec),
            &samples,
            budget(),
        )?;
        let container_build_ns = author_sw.elapsed_ns().max(0) as u64;
        let container_sha = obj.sha256();
        let verified = Arc::new(VerifiedFullObject::verify(obj.bytes.clone())?);
        let mut b5 = VoleBoundedSource::open("B5", verified.clone(), false)?;
        let mut b5p = VoleBoundedSource::open("B5-prepared", verified.clone(), true)?;

        // ---- the frozen trace, repeated with rotated source order ----
        let mut slots = vec![Slot::B2, Slot::B3Cold, Slot::B3Warm];
        if b4.is_some() {
            slots.push(Slot::B4);
        }
        slots.push(Slot::B5);
        slots.push(Slot::B5Prepared);

        let mut aggs: BTreeMap<&'static str, SourceAgg> = BTreeMap::new();
        let b2_info = b2.info();
        let b3_info = b3.info();
        let b5_info = b5.info();
        let b5p_info = b5p.info();
        aggs.insert(Slot::B2.name(), SourceAgg::new("MEASURED", &b2_info));
        aggs.insert(Slot::B3Cold.name(), SourceAgg::new("MEASURED", &b3_info));
        aggs.insert(Slot::B3Warm.name(), SourceAgg::new("MEASURED", &b3_info));
        if let Some(b4s) = b4.as_ref() {
            aggs.insert(Slot::B4.name(), SourceAgg::new("MEASURED", &b4s.info()));
        } else {
            aggs.insert(Slot::B4.name(), SourceAgg::not_applicable());
        }
        aggs.insert(Slot::B5.name(), SourceAgg::new("MEASURED", &b5_info));
        aggs.insert(
            Slot::B5Prepared.name(),
            SourceAgg::new("MEASURED", &b5p_info),
        );

        aggs.get_mut(Slot::B3Cold.name()).unwrap().artifact_build_ns = b3_build_ns;
        aggs.get_mut(Slot::B3Warm.name()).unwrap().artifact_build_ns = b3_build_ns;
        if let Some(a) = b4.as_ref() {
            // Authoring time is the FLAC *encode*; `runtime_setup_ns` is the later
            // decode/preload and must not be duplicated into the build field.
            let build_ns = b4_artifact.as_ref().unwrap().encoding.encode_ns;
            aggs.get_mut(Slot::B4.name()).unwrap().artifact_build_ns = build_ns;
            aggs.get_mut(Slot::B4.name()).unwrap().artifact_sha256 =
                Some(hex(&b4_artifact.as_ref().unwrap().sha256));
            let _ = a;
        }
        aggs.get_mut(Slot::B5.name()).unwrap().artifact_build_ns = container_build_ns;
        aggs.get_mut(Slot::B5.name()).unwrap().artifact_sha256 = Some(hex(&container_sha));
        aggs.get_mut(Slot::B5Prepared.name())
            .unwrap()
            .artifact_build_ns = container_build_ns;
        aggs.get_mut(Slot::B5Prepared.name())
            .unwrap()
            .artifact_sha256 = Some(hex(&container_sha));

        let max_window = QUANTUM_FRAMES as usize * ch;
        let mut dst = vec![0i32; max_window];
        let mut scratch: Vec<u8> = Vec::new();

        for repeat in 0..REPEATS {
            for &idx in &rotate_order(slots.len(), repeat) {
                let slot = slots[idx];
                match slot {
                    Slot::B2 => {
                        b2.reset()?;
                        let t = run_traversal(
                            &mut b2,
                            &plan,
                            ch,
                            o.sample_rate_hz,
                            &mut dst,
                            &mut scratch,
                        )?;
                        all_exact &= t.exact;
                        total_traversals += 1;
                        if repeat == 0 {
                            total_windows_b2 += t.windows as u64;
                        }
                        aggs.get_mut(Slot::B2.name()).unwrap().record(&t);
                        write_trace(&mut trace, &o.id, Slot::B2.name(), repeat, &t)?;
                    }
                    Slot::B3Cold => {
                        let ev = b3.evict_and_verify();
                        let name = ev.state_name();
                        *aggs
                            .get_mut(Slot::B3Cold.name())
                            .unwrap()
                            .cache_states
                            .entry(name)
                            .or_insert(0) += 1;
                        if ev.state == CacheState::Cold {
                            b3.reset()?;
                            let t = run_traversal(
                                &mut b3,
                                &plan,
                                ch,
                                o.sample_rate_hz,
                                &mut dst,
                                &mut scratch,
                            )?;
                            all_exact &= t.exact;
                            total_traversals += 1;
                            aggs.get_mut(Slot::B3Cold.name()).unwrap().record(&t);
                            write_trace(&mut trace, &o.id, Slot::B3Cold.name(), repeat, &t)?;
                        } else {
                            aggs.get_mut(Slot::B3Cold.name())
                                .unwrap()
                                .mark_ineligible(format!("repeat {repeat}: cache state {name}"));
                        }
                    }
                    Slot::B3Warm => {
                        let ev = b3.prime_and_verify()?;
                        let name = ev.state_name();
                        *aggs
                            .get_mut(Slot::B3Warm.name())
                            .unwrap()
                            .cache_states
                            .entry(name)
                            .or_insert(0) += 1;
                        if ev.state == CacheState::Warm {
                            b3.reset()?;
                            let t = run_traversal(
                                &mut b3,
                                &plan,
                                ch,
                                o.sample_rate_hz,
                                &mut dst,
                                &mut scratch,
                            )?;
                            all_exact &= t.exact;
                            total_traversals += 1;
                            aggs.get_mut(Slot::B3Warm.name()).unwrap().record(&t);
                            write_trace(&mut trace, &o.id, Slot::B3Warm.name(), repeat, &t)?;
                        } else {
                            aggs.get_mut(Slot::B3Warm.name())
                                .unwrap()
                                .mark_ineligible(format!("repeat {repeat}: cache state {name}"));
                        }
                    }
                    Slot::B4 => {
                        if let Some(s) = b4.as_mut() {
                            s.reset()?;
                            let t = run_traversal(
                                s,
                                &plan,
                                ch,
                                o.sample_rate_hz,
                                &mut dst,
                                &mut scratch,
                            )?;
                            all_exact &= t.exact;
                            total_traversals += 1;
                            aggs.get_mut(Slot::B4.name()).unwrap().record(&t);
                            write_trace(&mut trace, &o.id, Slot::B4.name(), repeat, &t)?;
                        }
                    }
                    Slot::B5 => {
                        b5.reset()?;
                        let t = run_traversal(
                            &mut b5,
                            &plan,
                            ch,
                            o.sample_rate_hz,
                            &mut dst,
                            &mut scratch,
                        )?;
                        all_exact &= t.exact;
                        total_traversals += 1;
                        aggs.get_mut(Slot::B5.name()).unwrap().record(&t);
                        write_trace(&mut trace, &o.id, Slot::B5.name(), repeat, &t)?;
                    }
                    Slot::B5Prepared => {
                        b5p.reset()?;
                        let t = run_traversal(
                            &mut b5p,
                            &plan,
                            ch,
                            o.sample_rate_hz,
                            &mut dst,
                            &mut scratch,
                        )?;
                        all_exact &= t.exact;
                        total_traversals += 1;
                        aggs.get_mut(Slot::B5Prepared.name()).unwrap().record(&t);
                        write_trace(&mut trace, &o.id, Slot::B5Prepared.name(), repeat, &t)?;
                    }
                }
            }
        }

        let _ = std::fs::remove_file(&path);

        for (name, agg) in &aggs {
            pool_all.absorb(name, agg);
            if generate::b1_comparable(o.channels) {
                pool_b1.absorb(name, agg);
            }
            for key in [
                format!("source_structure_class={}", o.source_structure_class),
                format!("amplitude_class={}", o.amplitude_class),
                format!("channel_structure={}", o.channel_structure),
                format!("temporal_class={}", o.temporal_class),
                format!("entropy_class={}", o.entropy_class),
                format!("sample_rate_hz={}", o.sample_rate_hz),
            ] {
                strata.entry(key).or_default().absorb(name, agg);
            }
        }

        let sources: serde_json::Map<String, serde_json::Value> = aggs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.cell(k)))
            .collect();
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
            "trace_windows": plan.len(),
            "sources": serde_json::Value::Object(sources),
        }));
    }

    let _ = std::fs::remove_dir_all(&dir);
    let total_ns = sw.elapsed_ns().max(0) as u64;

    if !all_exact {
        return fail("a runtime source did not reproduce every requested window exactly");
    }
    if cells.iter().any(|c| {
        c["sources"]
            .as_object()
            .map(|s| {
                s.values()
                    .any(|v| v["status"] == "MEASURED" && v["repeats_eligible"] == 0)
            })
            .unwrap_or(false)
    }) {
        return fail("a measured source had no eligible repeat");
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

    let trace_digest = trace.finalize()?;
    let trace_bytes = std::fs::metadata(trace_dir.join(&trace_name))
        .map(|m| m.len())
        .unwrap_or(0);
    let trace_info = TraceInfo {
        rel_path: format!("../traces/runtime/{trace_name}"),
        sha256: hex(&trace_digest),
        bytes: trace_bytes,
    };

    let measurement = measurement_surface(&pool_all, &pool_b1, &strata);
    let (timer_min_ns, timer_median_ns) = timer_overhead_ns();
    let mut measurement = measurement;
    if let Some(obj) = measurement.as_object_mut() {
        obj.insert(
            "timer_overhead_ns".into(),
            serde_json::json!({ "min": timer_min_ns, "median": timer_median_ns }),
        );
    }
    let object_count = cells.len();
    let verdict = Verdict::Supported;
    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!("{} + runtime.v1", manifest.profile)),
        backend: Some("common runtime substrate: B2/B3/B4/B5".into()),
        sample_rate_hz: None, // per object, native
        quantum_frames: Some(QUANTUM_FRAMES),
        content_kind: Some("flagship corpus (frozen before results): runtime measurement".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("runtime");
    builder
        .result(verdict)
        .result_detail(format!(
            "runtime measurement over {} frozen objects; frozen sequential {QUANTUM_FRAMES}-frame trace, \
             {REPEATS} repeats with rotated source order; B2/B3-cold/B3-warm/B5/B5-prepared for every \
             object and B4 for the {} B1-comparable objects; every requested window reproduced exactly; \
             result sha256 {result_hex}",
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
        .trace(trace_info)
        .extra(
            "protocol",
            serde_json::json!({
                "schema": PROTOCOL_SCHEMA,
                "quantum_frames": QUANTUM_FRAMES,
                "repeats": REPEATS,
                "source_order": "rotated deterministically per repeat (repeat r starts one source later)",
                "trace": "sequential windows [0,512), [512,1024), ... plus a final partial window",
                "rate": "each object's native rate (no resampling)",
                "channels": "each object's native channel count",
                "transforms": "none: no gain, pan, filter or random seek",
                "output": "exact canonical interleaved i32 into a caller-owned destination",
                "latency_boundary": "the harness times the entire RuntimeSource::read call; no source \
                                     times itself",
                "deadline": "computed by the harness from the native rate",
                "correctness": "each timed window is checked against a digest precomputed during \
                                preparation, so verification never walks the whole canonical vector",
                "physical_io": "sampled from /proc/self/io at traversal boundaries, outside every timer",
                "scope": "source materialization, not endpoint/resampling performance",
                "outside_the_interface": "artifact construction (raw PCM write, FLAC encode, container \
                                           compile) is authoring/compile work, never a playback operation",
                "populations": "all 115 objects (B2/B3/B5/B5-prepared) and the 110 B1-domain objects \
                                (adding B4); no ratio mixes the two",
            }),
        )
        .extra(
            "sources",
            serde_json::json!({
                "B2": "PCM resident (canonical i32 in memory)",
                "B3-cold": "raw PCM disk streaming, cold-verified (fadvise + mincore) each repeat",
                "B3-warm": "raw PCM disk streaming, warm-verified (primed + mincore) each repeat",
                "B4": "the exact B1 FLAC-5 artifact, one-time full decode -> resident PCM (no seektable)",
                "B5": "bounded VOLE materialization, first-play (fresh reader per repeat)",
                "B5-prepared": "bounded VOLE materialization, prepared control (all segments parsed \
                                before measurement; never averaged with B5)",
            }),
        )
        .extra(
            "aggregate",
            serde_json::json!({
                "objects": cells.len(),
                "b1_comparable_objects": cells
                    .iter()
                    .filter(|c| c["b1_comparable"].as_bool().unwrap_or(false))
                    .count(),
                "trace_windows_per_traversal": total_windows_b2,
                "traversals": total_traversals,
                "all_sources_exact": all_exact,
                "total_ns": total_ns,
            }),
        )
        .extra("measurement", measurement)
        .extra(
            "measured_latency_note",
            serde_json::json!({
                "status": "MEASURED_EXCLUDED_FROM_FROZEN_HASH",
                "detail": "per-source latency, physical storage traffic and setup times are measured \
                           evidence; the frozen static result covers the protocol, artifact \
                           identities, deterministic counters and the correctness vector only",
            }),
        )
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "this court measures the four source architectures under one frozen protocol. Latency is \
             the harness's: the stopwatch wraps the whole read() call, so B3 is not timed as \
             'kernel read' while B2/B4 time their copy to dst. Physical storage traffic is sampled \
             at traversal boundaries, outside every timer",
        )
        .limitation(
            "storage and residency are separated: a disk source is not credited with resident PCM, \
             and a decoded-PCM source does not retain its compressed artifact. Artifact construction \
             is authoring, not runtime setup",
        )
        .limitation(
            "B3 eligibility is per repeat: a cold repeat counts only with a COLD_VERIFIED state and a \
             warm repeat only with WARM_VERIFIED; an unconfirmed state makes that repeat ineligible \
             rather than 'close enough'. B5 first-play and the prepared control are reported \
             separately and never averaged",
        )
        .limitation(
            "sub-100 ns p50 latencies are at or near the cost of the measurement apparatus itself \
             (two clock reads per window, reported as `timer_overhead_ns`); they bound the \
             operation but do not resolve it more finely than that",
        )
        .limitation(
            "raw per-quantum latency vectors are written to the receipt's trace (see `trace`), a bulk \
             artifact under receipts/traces/ (gitignored) bound by SHA-256; the receipt carries the \
             derived distributions",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court runtime: {verdict}");
    println!(
        "  objects: {} | trace windows per traversal: {total_windows_b2} | traversals: {total_traversals} | all exact: {all_exact}",
        object_count
    );
    println!("  result sha256: {result_hex}");
    println!("  trace: {}", path.display());
    Ok(verdict)
}

/// Write one traversal's raw latency vector as a JSON line.
fn write_trace(
    trace: &mut TraceFile,
    object_id: &str,
    source: &str,
    repeat: u32,
    t: &Traversal,
) -> Result<()> {
    let rec = serde_json::json!({
        "object": object_id,
        "source": source,
        "repeat": repeat,
        "exact": t.exact,
        "latency_ns": t.latencies,
    });
    let mut line = serde_json::to_vec(&rec)?;
    line.push(b'\n');
    trace.write(&line)?;
    Ok(())
}

/// Accumulated evidence for one source over one object.
#[derive(Default)]
struct Pool {
    latencies: BTreeMap<String, Vec<u64>>,
    artifact_storage: BTreeMap<String, u64>,
    resident_sample: BTreeMap<String, u64>,
    resident_encoded: BTreeMap<String, u64>,
    setup_ns: BTreeMap<String, u64>,
    storage_reads: BTreeMap<String, u64>,
    deadline_misses: BTreeMap<String, u64>,
    scratch_peak: BTreeMap<String, u64>,
    objects: BTreeMap<String, u64>,
    exact_objects: BTreeMap<String, u64>,
}

impl Pool {
    fn absorb(&mut self, source: &str, agg: &SourceAgg) {
        if agg.status != "MEASURED" {
            return;
        }
        let s = source.to_string();
        self.latencies
            .entry(s.clone())
            .or_default()
            .extend_from_slice(&agg.latencies);
        *self.artifact_storage.entry(s.clone()).or_default() += agg.artifact_storage_bytes;
        *self.resident_sample.entry(s.clone()).or_default() += agg.resident_sample_domain_bytes;
        *self.resident_encoded.entry(s.clone()).or_default() += agg.resident_encoded_bytes;
        *self.setup_ns.entry(s.clone()).or_default() += agg.runtime_setup_ns;
        *self.storage_reads.entry(s.clone()).or_default() += agg.physical_storage_bytes_read;
        *self.deadline_misses.entry(s.clone()).or_default() += agg.deadline_misses;
        let peak = self.scratch_peak.entry(s.clone()).or_default();
        *peak = (*peak).max(agg.det.scratch_peak_bytes);
        *self.objects.entry(s.clone()).or_default() += 1;
        if agg.exact {
            *self.exact_objects.entry(s).or_default() += 1;
        }
    }

    fn p50(&self, source: &str) -> Option<u64> {
        self.latencies
            .get(source)
            .map(|l| TailSummary::summarize(l).p50_ns)
            .unwrap_or(None)
    }

    fn ratio(a: Option<u64>, b: Option<u64>) -> Option<f64> {
        match (a, b) {
            (Some(a), Some(b)) if b > 0 => Some(a as f64 / b as f64),
            _ => None,
        }
    }

    fn json(&self) -> serde_json::Value {
        let mut per = serde_json::Map::new();
        for (src, lats) in &self.latencies {
            per.insert(
                src.clone(),
                serde_json::json!({
                    "objects": self.objects.get(src).copied().unwrap_or(0),
                    "exact_objects": self.exact_objects.get(src).copied().unwrap_or(0),
                    "runtime_setup_ns": self.setup_ns.get(src).copied().unwrap_or(0),
                    "physical_storage_bytes_read": self.storage_reads.get(src).copied().unwrap_or(0),
                    "artifact_storage_bytes": self.artifact_storage.get(src).copied().unwrap_or(0),
                    "resident_sample_domain_bytes": self.resident_sample.get(src).copied().unwrap_or(0),
                    "resident_encoded_bytes": self.resident_encoded.get(src).copied().unwrap_or(0),
                    "scratch_peak_bytes": self.scratch_peak.get(src).copied().unwrap_or(0),
                    "deadline_misses": self.deadline_misses.get(src).copied().unwrap_or(0),
                    "latency": TailSummary::summarize(lats),
                }),
            );
        }
        serde_json::Value::Object(per)
    }

    /// Crossover ratios for this population/stratum: bounded VOLE against the
    /// conventional alternatives, at p50 and in stored bytes.
    fn crossover(&self) -> serde_json::Value {
        let b5 = self.p50("B5");
        let bytes = |s: &str| self.artifact_storage.get(s).copied().unwrap_or(0);
        let b5b = bytes("B5");
        serde_json::json!({
            "latency_p50_b5_over_b2": Self::ratio(b5, self.p50("B2")),
            "latency_p50_b5_over_b3_warm": Self::ratio(b5, self.p50("B3-warm")),
            "latency_p50_b5_over_b4": Self::ratio(b5, self.p50("B4")),
            "latency_p50_b5_over_b5_prepared": Self::ratio(b5, self.p50("B5-prepared")),
            "bytes_b5_over_b3": if bytes("B3-warm") > 0 { Some(b5b as f64 / bytes("B3-warm") as f64) } else { None },
            "bytes_b5_over_b4": if bytes("B4") > 0 { Some(b5b as f64 / bytes("B4") as f64) } else { None },
        })
    }
}

/// Build the per-source measurement surface over explicit populations and the
/// frozen descriptive axes.
fn measurement_surface(
    all: &Pool,
    b1: &Pool,
    strata: &BTreeMap<String, Pool>,
) -> serde_json::Value {
    let stratum_json: serde_json::Map<String, serde_json::Value> = strata
        .iter()
        .map(|(k, p)| {
            (
                k.clone(),
                serde_json::json!({ "sources": p.json(), "crossover": p.crossover() }),
            )
        })
        .collect();
    serde_json::json!({
        "note": "latency distributions are pooled per source over the whole population \
                 (per-object distributions are in cells, and raw per-quantum vectors are in the \
                 trace). Percentiles follow the evidence tail policy (n >= ceil(20/(1-p))); an \
                 unreportable percentile is null, never fabricated",
        "populations": {
            "all_115": { "sources": all.json(), "crossover": all.crossover() },
            "b1_domain_110": { "sources": b1.json(), "crossover": b1.crossover() },
        },
        "strata": stratum_json,
    })
}
