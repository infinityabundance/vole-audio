//! Complete representation cost and sample-exposure ledger (H.2.11/H.2.15).
//!
//! Owner of the cost structs; the *rules* live in `docs/ENTROPY_ACCOUNTING.md`
//! and the receipt surface in `evidence`. Every candidate representation
//! reports [`CompleteCost`], never a bare rANS body size. The fields are
//! additive and serde-friendly so courts can fold them into receipt extras
//! without hand transcription.

use serde::{Deserialize, Serialize};

/// Complete byte cost of one representation candidate.
///
/// `complete_bytes == metadata + hypothesis + model + payload + index +
/// dependency + integrity` (see `CompleteCost::compute`). The three reporting
/// baselines (`raw_sample_bytes`, `canonical_literal_bytes`,
/// `source_wav_bytes`) are never part of the sum — they are comparison
/// baselines.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompleteCost {
    /// Container/descriptor/version metadata bytes.
    pub metadata_bytes: u64,
    /// Deterministic hypothesis bytes (procedural state / semantic residual
    /// model). Zero for pure literal representations.
    pub hypothesis_bytes: u64,
    /// Entropy model bytes: inline models count in full; shared models count
    /// once (standalone attribution policy — see [`SharedCost`] for the
    /// marginal view).
    pub model_bytes: u64,
    /// Entropy payload bytes (rANS/RAW bodies).
    pub payload_bytes: u64,
    /// Page-index bytes.
    pub index_bytes: u64,
    /// Dependency bytes (referenced content id prefixes etc.).
    pub dependency_bytes: u64,
    /// Integrity digest bytes (32 per protected record; 0 when disabled).
    pub integrity_bytes: u64,
    /// Computed sum of the seven fields above.
    pub complete_bytes: u64,
    // --- comparison baselines (never part of the sum) ---
    /// Canonical i32 LE sample codes, `frames * channels * 4`.
    pub raw_sample_bytes: u64,
    /// Canonical U1 literal bytes (descriptor header + sample count + codes).
    pub canonical_literal_bytes: u64,
    /// Source WAV payload bytes where applicable (else 0).
    pub source_wav_bytes: u64,
}

impl CompleteCost {
    /// Recompute `complete_bytes` from the parts.
    pub fn compute(&mut self) {
        self.complete_bytes = self
            .metadata_bytes
            .saturating_add(self.hypothesis_bytes)
            .saturating_add(self.model_bytes)
            .saturating_add(self.payload_bytes)
            .saturating_add(self.index_bytes)
            .saturating_add(self.dependency_bytes)
            .saturating_add(self.integrity_bytes);
    }
}

/// Declared / unique / physical storage accounting (H.2.24). Always reported
/// as three distinct quantities; a shared dependency is never zero bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageBytes {
    /// Complete bytes attributed to each logical object if stored standalone.
    pub declared_bytes: u64,
    /// Content-unique canonical payload bytes across the store.
    pub unique_bytes: u64,
    /// Actual backing bytes of the store engine (incl. its metadata).
    pub physical_bytes: u64,
}

/// Standalone vs marginal cost of a shared dependency (H.2.8).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedCost {
    /// The dependency's full byte cost attributed to one object.
    pub standalone_bytes: u64,
    /// Marginal bytes once the dependency is already resident.
    pub marginal_bytes: u64,
    /// Number of objects sharing the dependency.
    pub sharers: u64,
}

/// Sample-domain exposure ledger (H.2.15): surfaces measured separately;
/// materialization is never conflated with verification instrumentation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExposureLedger {
    /// Persistent baked sample-domain bytes (literal/cycle payloads etc.).
    pub persistent_sample_domain_bytes: u64,
    /// Entropy state bytes (rANS/RAW payloads).
    pub entropy_state_bytes: u64,
    pub entropy_model_bytes: u64,
    pub entropy_payload_bytes: u64,
    pub entropy_index_bytes: u64,
    /// Host materialization peak/integral (sample bytes, byte·s).
    pub host_materialization_sample_peak: u64,
    pub host_materialization_sample_integral: f64,
    /// Host verification peak/integral (court readback).
    pub host_verification_sample_peak: u64,
    pub host_verification_sample_integral: f64,
    /// GPU global sample intermediates (bounded window scratch etc.).
    pub gpu_global_sample_intermediate_bytes: u64,
    /// GPU transient sample words (registers/shared/local).
    pub gpu_transient_sample_words: u64,
    pub gpu_to_host_sample_bytes: u64,
    pub host_sample_copy_bytes: u64,
    pub endpoint_observation_bytes: u64,
}

impl ExposureLedger {
    pub fn record_host_materialization(&mut self, bytes: u64, dt_secs: f64) {
        self.host_materialization_sample_peak = self.host_materialization_sample_peak.max(bytes);
        self.host_materialization_sample_integral += bytes as f64 * dt_secs;
    }

    pub fn record_host_verification(&mut self, bytes: u64, dt_secs: f64) {
        self.host_verification_sample_peak = self.host_verification_sample_peak.max(bytes);
        self.host_verification_sample_integral += bytes as f64 * dt_secs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_cost_sums_parts() {
        let mut c = CompleteCost {
            metadata_bytes: 40,
            hypothesis_bytes: 8,
            model_bytes: 512,
            payload_bytes: 100,
            index_bytes: 24,
            dependency_bytes: 0,
            integrity_bytes: 0,
            complete_bytes: 0,
            raw_sample_bytes: 8192,
            canonical_literal_bytes: 8200,
            source_wav_bytes: 0,
        };
        c.compute();
        assert_eq!(c.complete_bytes, 40 + 8 + 512 + 100 + 24);
        assert_eq!(c.raw_sample_bytes, 8192);
    }

    #[test]
    fn exposure_ledger_tracks_extremes() {
        let mut l = ExposureLedger::default();
        l.record_host_materialization(100, 0.5);
        l.record_host_materialization(200, 0.5);
        assert_eq!(l.host_materialization_sample_peak, 200);
        assert!((l.host_materialization_sample_integral - 150.0).abs() < 1e-9);
        l.record_host_verification(300, 1.0);
        assert_eq!(l.host_verification_sample_peak, 300);
    }
}
