//! Deterministic named counters for sample-domain and traffic accounting.
//!
//! Measurement boundary (paper §"PCM/sample-domain exposure", §41 of the
//! implementation contract):
//!
//! * `host_pcm_resident_peak` / `host_pcm_resident_integral`: application-level
//!   PCM that *persists* on the host across quanta. Aliases of one physical
//!   shared region are not double counted unless reporting virtual exposure.
//! * `host_pcm_staging_bytes`: writes into host PCM staging buffers (cumulative
//!   over the run).
//! * `host_pcm_copy_bytes`: PCM bytes copied between host buffers.
//! * `gpu_to_host_pcm_bytes`: device->host sample traffic.
//! * `endpoint_observation_bytes`: bytes written into the endpoint-visible
//!   observation region.
//! * `endpoint_depth_frames`: current transient depth held by the endpoint
//!   (DMA/FIFO/register elasticity) at the measurement point.
//! * `transient_compute_words`: register/shared-memory reduction accumulators —
//!   compute state, counted separately from independently addressable storage.
//!
//! Every receipt must name its measurement boundary; see `docs/EVIDENCE.md`.

use serde::{Deserialize, Serialize};

/// Cumulative and peak byte counters for one run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Counters {
    // --- cumulative traffic (bytes) ---
    pub host_pcm_staging_bytes: u64,
    pub host_pcm_copy_bytes: u64,
    pub gpu_to_host_pcm_bytes: u64,
    pub endpoint_observation_bytes: u64,
    /// Bytes rendered into any *diagnostic* VRAM sample block (D0 block).
    pub device_sample_block_bytes: u64,
    /// Transient compute accumulation state: register/shared words touched.
    pub transient_compute_words: u64,
    // --- residency (bytes) ---
    pub host_pcm_resident_peak_bytes: u64,
    /// byte·seconds integral of host PCM residency.
    pub host_pcm_resident_integral: f64,
    /// Endpoint observation depth in frames at the last commit.
    pub endpoint_depth_frames: u64,
    pub endpoint_depth_min_frames: u64,
    pub endpoint_depth_max_frames: u64,
    // --- events ---
    pub quanta_submitted: u64,
    pub events_consumed: u64,
    pub xruns: u64,
    pub kernel_launches: u64,
    pub missed_deadlines: u64,
}

impl Counters {
    pub const fn new() -> Self {
        Self {
            host_pcm_staging_bytes: 0,
            host_pcm_copy_bytes: 0,
            gpu_to_host_pcm_bytes: 0,
            endpoint_observation_bytes: 0,
            device_sample_block_bytes: 0,
            transient_compute_words: 0,
            host_pcm_resident_peak_bytes: 0,
            host_pcm_resident_integral: 0.0,
            endpoint_depth_frames: 0,
            endpoint_depth_min_frames: u64::MAX,
            endpoint_depth_max_frames: 0,
            quanta_submitted: 0,
            events_consumed: 0,
            xruns: 0,
            kernel_launches: 0,
            missed_deadlines: 0,
        }
    }

    /// Record a residency observation; maintains peak and byte·second integral.
    /// `dt` is the wall time the residency level was held, in seconds.
    pub fn observe_residency(&mut self, bytes: u64, dt_secs: f64) {
        self.host_pcm_resident_peak_bytes = self.host_pcm_resident_peak_bytes.max(bytes);
        self.host_pcm_resident_integral += bytes as f64 * dt_secs;
    }

    pub fn observe_endpoint_depth(&mut self, frames: u64) {
        self.endpoint_depth_frames = frames;
        self.endpoint_depth_min_frames = self.endpoint_depth_min_frames.min(frames);
        self.endpoint_depth_max_frames = self.endpoint_depth_max_frames.max(frames);
    }

    pub fn add_staging(&mut self, bytes: u64) {
        self.host_pcm_staging_bytes = self.host_pcm_staging_bytes.saturating_add(bytes);
    }

    pub fn add_copy(&mut self, bytes: u64) {
        self.host_pcm_copy_bytes = self.host_pcm_copy_bytes.saturating_add(bytes);
    }

    pub fn add_gpu_to_host(&mut self, bytes: u64) {
        self.gpu_to_host_pcm_bytes = self.gpu_to_host_pcm_bytes.saturating_add(bytes);
    }

    pub fn add_endpoint_observation(&mut self, bytes: u64) {
        self.endpoint_observation_bytes = self.endpoint_observation_bytes.saturating_add(bytes);
    }

    pub fn add_device_sample_block(&mut self, bytes: u64) {
        self.device_sample_block_bytes = self.device_sample_block_bytes.saturating_add(bytes);
    }

    pub fn add_transient_compute_words(&mut self, words: u64) {
        self.transient_compute_words = self.transient_compute_words.saturating_add(words);
    }

    /// Fold another counter set into this one (cumulative traffic/events;
    /// residency fields keep this set's own extremes). Used by courts to
    /// aggregate per-world device counters into the receipt.
    pub fn add_from(&mut self, other: &Counters) {
        self.host_pcm_staging_bytes = self
            .host_pcm_staging_bytes
            .saturating_add(other.host_pcm_staging_bytes);
        self.host_pcm_copy_bytes = self
            .host_pcm_copy_bytes
            .saturating_add(other.host_pcm_copy_bytes);
        self.gpu_to_host_pcm_bytes = self
            .gpu_to_host_pcm_bytes
            .saturating_add(other.gpu_to_host_pcm_bytes);
        self.endpoint_observation_bytes = self
            .endpoint_observation_bytes
            .saturating_add(other.endpoint_observation_bytes);
        self.device_sample_block_bytes = self
            .device_sample_block_bytes
            .saturating_add(other.device_sample_block_bytes);
        self.transient_compute_words = self
            .transient_compute_words
            .saturating_add(other.transient_compute_words);
        self.host_pcm_resident_peak_bytes = self
            .host_pcm_resident_peak_bytes
            .max(other.host_pcm_resident_peak_bytes);
        self.quanta_submitted += other.quanta_submitted;
        self.events_consumed += other.events_consumed;
        self.xruns += other.xruns;
        self.kernel_launches += other.kernel_launches;
        self.missed_deadlines += other.missed_deadlines;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residency_tracking() {
        let mut c = Counters::new();
        c.observe_residency(100, 0.5);
        c.observe_residency(200, 0.5);
        assert_eq!(c.host_pcm_resident_peak_bytes, 200);
        assert!((c.host_pcm_resident_integral - 150.0).abs() < 1e-9);
    }

    #[test]
    fn endpoint_depth_tracking() {
        let mut c = Counters::new();
        c.observe_endpoint_depth(256);
        c.observe_endpoint_depth(128);
        c.observe_endpoint_depth(512);
        assert_eq!(c.endpoint_depth_min_frames, 128);
        assert_eq!(c.endpoint_depth_max_frames, 512);
        assert_eq!(c.endpoint_depth_frames, 512);
    }

    #[test]
    fn add_from_folds_cumulative_but_keeps_residency_extremes() {
        let mut a = Counters::new();
        a.gpu_to_host_pcm_bytes = 100;
        a.kernel_launches = 3;
        a.quanta_submitted = 2;
        a.host_pcm_resident_peak_bytes = 4096;
        let mut b = Counters::new();
        b.gpu_to_host_pcm_bytes = 40;
        b.kernel_launches = 1;
        b.quanta_submitted = 1;
        b.host_pcm_resident_peak_bytes = 8192;
        a.add_from(&b);
        assert_eq!(a.gpu_to_host_pcm_bytes, 140);
        assert_eq!(a.kernel_launches, 4);
        assert_eq!(a.quanta_submitted, 3);
        assert_eq!(
            a.host_pcm_resident_peak_bytes, 8192,
            "residency keeps the max"
        );
    }
}
