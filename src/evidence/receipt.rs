//! `vole.audio.evidence.v1` receipts.
//!
//! A receipt is an immutable, self-describing JSON document that binds one
//! court/probe result to the environment, hardware, method, counters, timing,
//! provenance hashes, and verdict that produced it.
//!
//! Canonicalism: the `Receipt` struct serializes with stable field order
//! (serde struct order) and BTreeMap extras, so the same logical receipt
//! always produces the same compact JSON bytes. `receipt_sha256` is the
//! SHA-256 of exactly those canonical bytes, enabling `verify()`.
//!
//! Immutability: `write_atomic` creates a new file with `create_new` semantics
//! under a unique run id; an existing receipt is never overwritten or mutated.

use crate::error::{Error, Kind, Result};
use crate::evidence::counters::Counters;
use crate::evidence::energy::EnergyReport;
use crate::evidence::environment::Environment;
use crate::evidence::hardware::Hardware;
use crate::evidence::timing::TailSummary;
use crate::hash::sha256::Sha256;
use crate::status::Verdict;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const EVIDENCE_SCHEMA: &str = "vole.audio.evidence.v1";
pub const EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Time/timing facts about the run (host domain).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunTiming {
    /// Total wall duration of the measured section, ns.
    pub total_ns: Option<i64>,
    pub deadline_ns: Option<i64>,
    pub warmup_runs: Option<u64>,
    pub run_count: Option<u64>,
    pub observation_count: Option<u64>,
    /// Submission/tail latency over the run, if measured.
    pub tail: Option<TailSummary>,
}

/// Provenance hashes that let a receipt be re-bound to exact artifacts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// SHA-256 of the reference (scalar) observation, hex.
    pub reference_hash: Option<String>,
    /// SHA-256 of this backend's observation, hex.
    pub backend_hash: Option<String>,
    /// SHA-256 of the actual endpoint-region bytes read back (D1/D2 courts),
    /// hex — independently accumulated where the endpoint mapping is
    /// verification-readable; equal to `backend_hash` when the backend wrote
    /// the endpoint region directly and the readback was byte-exact.
    pub endpoint_hash: Option<String>,
    /// SHA-256 of the GPU artifact (PTX/cubin/code object) used, hex.
    pub gpu_artifact_hash: Option<String>,
    /// Source-tree hash at run time (git identity + dirty, see Environment).
    pub source_hash: Option<String>,
    /// Corpus content hash, hex.
    pub corpus_hash: Option<String>,
    /// Resampler/frozen table hashes in play, hex.
    pub table_hashes: Vec<String>,
    /// Ordering of benchmarks within the run, e.g. ["scalar","simd","cuda"].
    pub benchmark_order: Vec<String>,
    /// Exact equal-hash check outcome (reference vs backend).
    pub exact_equality: Option<bool>,
}

/// Endpoint/directness evidence for D1/D2 courts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointEvidence {
    pub directness: Option<String>,
    pub topology: Option<String>,
    /// Exact host address + length registered, e.g. "0x7f..+4096".
    pub registered_range: Option<String>,
    pub registration_result: Option<String>,
    /// Pointer attributes / device pointer observed.
    pub device_pointer: Option<String>,
    pub synchronization_mechanism: Option<String>,
    pub fence_sync_evidence: Option<String>,
    pub coherency_assumptions: Option<String>,
    /// Explicit note about any hidden host staging investigated.
    pub hidden_staging_investigation: Option<String>,
    pub endpoint_clock: Option<String>,
}

/// Court-specific parameters that define the workload.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CourtParams {
    pub universe: Option<String>,
    pub profile: Option<String>,
    pub backend: Option<String>,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u32>,
    /// Voice count or per-representation distribution.
    pub voices: Option<serde_json::Value>,
    pub quantum_frames: Option<u32>,
    pub duration_secs: Option<f64>,
    pub content_kind: Option<String>,
}

/// The immutable receipt body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    pub schema: String,
    pub schema_version: u32,
    pub run_id: String,
    pub court: String,
    pub created_unix_ms: i64,
    pub result: Verdict,
    pub result_detail: Option<String>,
    pub environment: Environment,
    pub hardware: Hardware,
    pub params: CourtParams,
    pub counters: Counters,
    pub timing: RunTiming,
    pub provenance: Provenance,
    pub endpoint: EndpointEvidence,
    pub energy: Option<EnergyReport>,
    pub trace: Option<TraceInfo>,
    pub limitations: Vec<String>,
    /// Forward-compatible extension point (BTreeMap keeps JSON canonical).
    pub extras: BTreeMap<String, serde_json::Value>,
}

/// File-level binding for a raw trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceInfo {
    /// File name relative to the receipt's directory.
    pub rel_path: String,
    /// SHA-256 of the trace bytes, hex.
    pub sha256: String,
    pub bytes: u64,
}

/// On-disk envelope: canonical receipt + its self-hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiptEnvelope {
    pub schema: String,
    pub schema_version: u32,
    pub receipt: Receipt,
    /// SHA-256 (hex) of the canonical compact JSON of `receipt`.
    pub receipt_sha256: String,
}

impl ReceiptEnvelope {
    /// Verify the embedded self-hash over the canonical receipt JSON of the
    /// **current** schema (typed round-trip). File-level verification is
    /// [`ReceiptEnvelope::from_json_bytes`], which hashes the receipt exactly
    /// as the file carries it and therefore also verifies receipts produced
    /// before an additive schema field existed.
    pub fn verify(&self) -> Result<()> {
        if self.schema != EVIDENCE_SCHEMA || self.schema_version != EVIDENCE_SCHEMA_VERSION {
            return Err(Error::new(
                Kind::Unsupported,
                format!(
                    "receipt schema {}/v{} != expected {}/v{}",
                    self.schema, self.schema_version, EVIDENCE_SCHEMA, EVIDENCE_SCHEMA_VERSION
                ),
            ));
        }
        let canonical = canonical_json(&self.receipt)?;
        let digest = Sha256::digest(&canonical);
        let expect = crate::hash::sha256::hex(&digest);
        if expect != self.receipt_sha256 {
            return Err(Error::integrity(format!(
                "receipt self-hash mismatch: stored {} computed {}",
                self.receipt_sha256, expect
            )));
        }
        Ok(())
    }

    /// Parse and verify an envelope from canonical (compact or pretty) JSON.
    ///
    /// The self-hash is checked against the canonical compact encoding of the
    /// `receipt` value **exactly as the file carries it** (object key order
    /// preserved). That makes the hash a pure function of the stored bytes, so
    /// a receipt remains verifiable after the schema grows additively: a
    /// receipt written before a field existed still hashes its own field set.
    /// (Hashing the typed struct instead would silently invalidate every older
    /// receipt whenever a field is added.)
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        Self::verify_value(&value)?;
        let env: Self = serde_json::from_slice(bytes)?;
        Ok(env)
    }

    /// Verify a parsed envelope value (order-preserving) against its embedded
    /// self-hash.
    fn verify_value(value: &serde_json::Value) -> Result<()> {
        let obj = value
            .as_object()
            .ok_or_else(|| Error::integrity("receipt envelope is not a JSON object"))?;
        let schema = obj
            .get("schema")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::integrity("receipt envelope has no schema string"))?;
        let version = obj
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::integrity("receipt envelope has no schema_version"))?;
        if schema != EVIDENCE_SCHEMA || version != u64::from(EVIDENCE_SCHEMA_VERSION) {
            return Err(Error::new(
                Kind::Unsupported,
                format!(
                    "receipt schema {schema}/v{version} != expected {EVIDENCE_SCHEMA}/v{EVIDENCE_SCHEMA_VERSION}"
                ),
            ));
        }
        let receipt = obj
            .get("receipt")
            .ok_or_else(|| Error::integrity("receipt envelope has no receipt body"))?;
        let stored = obj
            .get("receipt_sha256")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::integrity("receipt envelope has no receipt_sha256"))?;
        let canonical = serde_json::to_vec(receipt)?;
        let digest = Sha256::digest(&canonical);
        let expect = crate::hash::sha256::hex(&digest);
        if expect != stored {
            return Err(Error::integrity(format!(
                "receipt self-hash mismatch: stored {stored} computed {expect}"
            )));
        }
        Ok(())
    }
}

/// Deterministic compact JSON of the receipt body.
fn canonical_json(receipt: &Receipt) -> Result<Vec<u8>> {
    serde_json::to_vec(receipt).map_err(Into::into)
}

impl Receipt {
    /// Produce a new receipt; caller sets fields via the builder.
    pub fn builder(court: &str) -> ReceiptBuilder {
        ReceiptBuilder::new(court)
    }

    /// Canonical compact bytes of the receipt body.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        canonical_json(self)
    }
}

/// Incremental builder; receipts are assembled then frozen.
#[derive(Debug, Clone)]
pub struct ReceiptBuilder {
    court: String,
    run_id: String,
    created_unix_ms: i64,
    result: Verdict,
    result_detail: Option<String>,
    environment: Option<Environment>,
    hardware: Option<Hardware>,
    params: CourtParams,
    counters: Counters,
    timing: RunTiming,
    provenance: Provenance,
    endpoint: EndpointEvidence,
    energy: Option<EnergyReport>,
    trace: Option<TraceInfo>,
    limitations: Vec<String>,
    extras: BTreeMap<String, serde_json::Value>,
}

impl ReceiptBuilder {
    pub fn new(court: &str) -> Self {
        // Wall-clock Unix milliseconds. This must be comparable across boots:
        // a boot-relative monotonic clock resets on reboot, which would let an
        // older receipt outrank a newer one in the "newest per court" seal
        // selection. Wall clock is the correct ordering key here; fine-grained
        // durations elsewhere continue to use the monotonic clock.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let run_id = format!("{}-{}-{:04x}", court, now_ms, std::process::id() as u32);
        Self {
            court: court.to_string(),
            run_id,
            created_unix_ms: now_ms,
            result: Verdict::Inconclusive,
            result_detail: None,
            environment: None,
            hardware: None,
            params: CourtParams::default(),
            counters: Counters::new(),
            timing: RunTiming::default(),
            provenance: Provenance::default(),
            endpoint: EndpointEvidence::default(),
            energy: None,
            trace: None,
            limitations: Vec::new(),
            extras: BTreeMap::new(),
        }
    }

    pub fn result(&mut self, v: Verdict) -> &mut Self {
        self.result = v;
        self
    }

    pub fn result_detail(&mut self, d: impl Into<String>) -> &mut Self {
        self.result_detail = Some(d.into());
        self
    }

    pub fn environment(&mut self, e: Environment) -> &mut Self {
        self.environment = Some(e);
        self
    }

    pub fn hardware(&mut self, h: Hardware) -> &mut Self {
        self.hardware = Some(h);
        self
    }

    pub fn params(&mut self, p: CourtParams) -> &mut Self {
        self.params = p;
        self
    }

    pub fn counters(&mut self, c: Counters) -> &mut Self {
        self.counters = c;
        self
    }

    pub fn timing(&mut self, t: RunTiming) -> &mut Self {
        self.timing = t;
        self
    }

    pub fn provenance(&mut self, p: Provenance) -> &mut Self {
        self.provenance = p;
        self
    }

    pub fn endpoint(&mut self, e: EndpointEvidence) -> &mut Self {
        self.endpoint = e;
        self
    }

    pub fn energy(&mut self, e: EnergyReport) -> &mut Self {
        self.energy = Some(e);
        self
    }

    pub fn trace(&mut self, t: TraceInfo) -> &mut Self {
        self.trace = Some(t);
        self
    }

    pub fn limitation(&mut self, l: impl Into<String>) -> &mut Self {
        self.limitations.push(l.into());
        self
    }

    pub fn extra(&mut self, k: impl Into<String>, v: serde_json::Value) -> &mut Self {
        self.extras.insert(k.into(), v);
        self
    }

    /// Freeze into an envelope with self-hash.
    pub fn finish(&mut self) -> Result<ReceiptEnvelope> {
        let receipt = Receipt {
            schema: EVIDENCE_SCHEMA.to_string(),
            schema_version: EVIDENCE_SCHEMA_VERSION,
            run_id: self.run_id.clone(),
            court: self.court.clone(),
            created_unix_ms: self.created_unix_ms,
            result: self.result,
            result_detail: self.result_detail.clone(),
            environment: self
                .environment
                .clone()
                .unwrap_or_else(Environment::capture),
            hardware: self.hardware.clone().unwrap_or_else(Hardware::capture),
            params: self.params.clone(),
            counters: self.counters.clone(),
            timing: self.timing.clone(),
            provenance: self.provenance.clone(),
            endpoint: self.endpoint.clone(),
            energy: self.energy.clone(),
            trace: self.trace.clone(),
            limitations: self.limitations.clone(),
            extras: self.extras.clone(),
        };
        let canonical = canonical_json(&receipt)?;
        let digest = Sha256::digest(&canonical);
        Ok(ReceiptEnvelope {
            schema: EVIDENCE_SCHEMA.to_string(),
            schema_version: EVIDENCE_SCHEMA_VERSION,
            receipt,
            receipt_sha256: crate::hash::sha256::hex(&digest),
        })
    }

    /// Finish and atomically persist under `receipts/<court>/`.
    ///
    /// The file name embeds the run id and is created with `create_new`, so an
    /// existing receipt is never overwritten (receipts are immutable outputs).
    pub fn finish_write(&mut self, receipts_root: &Path) -> Result<(ReceiptEnvelope, PathBuf)> {
        let env = self.finish()?;
        let dir = receipts_root.join(&self.court);
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::new(Kind::Io, format!("mkdir {}: {e}", dir.display())))?;
        let file_name = format!("{}.json", env.receipt.run_id);
        let path = dir.join(&file_name);
        let json = serde_json::to_string_pretty(&env)?;
        // create_new: never overwrite.
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        let mut f = opts
            .open(&path)
            .map_err(|e| Error::new(Kind::Io, format!("create {}: {e}", path.display())))?;
        use std::io::Write;
        f.write_all(json.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
        Ok((env, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::timing::DurationNs;
    use std::time::Duration;

    fn sample_builder() -> ReceiptBuilder {
        let mut b = ReceiptBuilder::new("court-unit-test");
        b.result(Verdict::Supported)
            .result_detail("unit test receipt")
            .params(CourtParams {
                universe: Some("vole.audio.u1".into()),
                profile: Some("u1/v1".into()),
                backend: Some("scalar".into()),
                sample_rate_hz: Some(48_000),
                channels: Some(2),
                quantum_frames: Some(1024),
                ..Default::default()
            });
        b
    }

    #[test]
    fn receipt_roundtrip_and_verify() {
        let mut b = sample_builder();
        b.counters(Counters {
            host_pcm_staging_bytes: 1024,
            ..Counters::new()
        });
        let env = b.finish().expect("finish");
        assert_eq!(env.receipt.result, Verdict::Supported);
        env.verify().expect("verify passes");

        // Serialize -> parse -> verify roundtrip.
        let bytes = serde_json::to_vec(&env).unwrap();
        let parsed = ReceiptEnvelope::from_json_bytes(&bytes).expect("parse+verify");
        assert_eq!(parsed.receipt.run_id, env.receipt.run_id);
        assert_eq!(parsed.receipt.counters.host_pcm_staging_bytes, 1024);
    }

    #[test]
    fn canonical_bytes_are_stable() {
        let env = sample_builder().finish().unwrap();
        let a = env.receipt.canonical_bytes().unwrap();
        let env2: ReceiptEnvelope =
            serde_json::from_slice(&serde_json::to_vec(&env).unwrap()).unwrap();
        let b = env2.receipt.canonical_bytes().unwrap();
        assert_eq!(a, b, "same logical receipt must serialize identically");
    }

    #[test]
    fn write_is_immutable_and_unique() {
        let dir = std::env::temp_dir().join(format!(
            "vole-receipt-{}-{}",
            std::process::id(),
            Duration::from_micros(1).as_nanos()
        ));
        let (_, p1) = sample_builder().finish_write(&dir).expect("first write");
        let (_, p2) = sample_builder().finish_write(&dir).expect("second write");
        assert_ne!(p1, p2, "run ids must differ");
        let bytes = std::fs::read(&p1).unwrap();
        let parsed = ReceiptEnvelope::from_json_bytes(&bytes).expect("file verifies");
        assert_eq!(parsed.receipt.court, "court-unit-test");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verdict_serializes_as_snake() {
        let env = sample_builder().finish().unwrap();
        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains("\"result\":\"SUPPORTED\""));
    }

    #[test]
    fn timing_duration_display() {
        let d = DurationNs(2_500_000_000);
        let s = d.to_string();
        assert!(s.contains("s"), "{s}");
    }

    #[test]
    fn canonical_forms_agree_for_the_pointer_style() {
        // The producer hashes `to_vec(receipt)`; verification hashes
        // `to_vec(to_value(receipt))` from the stored bytes. These must be the
        // same encoding, or a receipt could never verify itself.
        let env = sample_builder().finish().unwrap();
        let typed = env.receipt.canonical_bytes().unwrap();
        let value = serde_json::to_value(&env.receipt).unwrap();
        let via_value = serde_json::to_vec(&value).unwrap();
        assert_eq!(typed, via_value, "producer and verifier encodings differ");
    }

    #[test]
    fn float_heavy_extras_roundtrip_byte_exactly() {
        // A float whose shortest decimal form is 1 ULP-sensitive under a
        // non-round-tripping parser: it must still verify from disk.
        let mut b = sample_builder();
        b.extra("ratio", serde_json::json!(32506.0f64 / 32839.0f64));
        b.extra("integral", serde_json::json!(0.1f64 + 0.2f64));
        let env = b.finish().unwrap();
        let bytes = serde_json::to_vec_pretty(&env).unwrap();
        let parsed = ReceiptEnvelope::from_json_bytes(&bytes).expect("verifies from disk");
        assert_eq!(parsed.receipt_sha256, env.receipt_sha256);
    }

    #[test]
    fn receipts_written_before_an_additive_field_still_verify() {
        // Simulate schema growth: drop a field from the stored receipt body,
        // rehash the body as stored, and require file-level verification to
        // accept it (a typed re-serialization would have added the field back
        // and rejected a byte-honest old receipt).
        let env = sample_builder().finish().unwrap();
        let mut value = serde_json::to_value(&env).unwrap();
        let digest = {
            let receipt = value.get_mut("receipt").unwrap();
            let obj = receipt.as_object_mut().unwrap();
            let provenance = obj.get_mut("provenance").unwrap().as_object_mut().unwrap();
            assert!(provenance.remove("endpoint_hash").is_some());
            let canonical = serde_json::to_vec(receipt).unwrap();
            Sha256::digest(&canonical)
        };
        value.as_object_mut().unwrap().insert(
            "receipt_sha256".into(),
            serde_json::Value::String(crate::hash::sha256::hex(&digest)),
        );
        let bytes = serde_json::to_vec_pretty(&value).unwrap();
        ReceiptEnvelope::from_json_bytes(&bytes).expect("older-shape receipt verifies");
    }

    #[test]
    fn tampered_receipt_body_is_rejected() {
        let env = sample_builder().finish().unwrap();
        let mut value = serde_json::to_value(&env).unwrap();
        value
            .get_mut("receipt")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("result".into(), serde_json::json!("FAILED_CORRECTNESS"));
        let bytes = serde_json::to_vec_pretty(&value).unwrap();
        assert!(ReceiptEnvelope::from_json_bytes(&bytes).is_err());
    }
}
