//! The frozen external measurement protocol (Phase M, Seal 10).
//!
//! This module exists so the *measurement contract* is stated once and shared by
//! every runtime court, rather than re-derived inside each court:
//!
//! * the **frozen trace** (`[0, 512), [512, 1024), ...` plus a final partial
//!   window) is the only request sequence a source ever sees here;
//! * the **latency boundary** is the harness's, wrapping the entire
//!   `RuntimeSource::read` call;
//! * **output correctness** is checked against window digests computed during
//!   preparation, so verifying a timed window never walks the whole canonical
//!   source vector;
//! * **source order** is rotated deterministically per repeat so no architecture
//!   permanently holds the coolest cache state;
//! * **repeats** start from equivalent initial source state.

use crate::error::Result;
use crate::evidence::TailSummary;
use crate::hash::sha256::Sha256;

use super::{QUANTUM_FRAMES, frozen_trace};

/// Expected content of every frozen window of one object.
#[derive(Debug, Clone)]
pub struct WindowPlan {
    spans: Vec<(u64, u32)>,
    digests: Vec<[u8; 32]>,
}

impl WindowPlan {
    /// Build the frozen window plan for one canonical interleaved object.
    pub fn build(samples: &[i32], channels: usize) -> Result<WindowPlan> {
        let total_frames = (samples.len() / channels) as u64;
        let spans = frozen_trace(total_frames, QUANTUM_FRAMES)?;
        let mut scratch = Vec::new();
        let mut digests = Vec::with_capacity(spans.len());
        for &(start, frames) in &spans {
            let lo = start as usize * channels;
            let hi = lo + frames as usize * channels;
            digests.push(digest_interleaved(&samples[lo..hi], &mut scratch));
        }
        Ok(WindowPlan { spans, digests })
    }

    pub fn spans(&self) -> &[(u64, u32)] {
        &self.spans
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Does `dst` equal frozen window `i`, byte-for-byte?
    pub fn window_matches(&self, i: usize, dst: &[i32], scratch: &mut Vec<u8>) -> Option<bool> {
        let expected = self.digests.get(i)?;
        Some(digest_interleaved(dst, scratch) == *expected)
    }
}

/// Canonical little-endian digest of interleaved `i32` samples.
///
/// The scratch buffer is reused across windows so correctness checking never
/// allocates inside the measurement loop.
pub fn digest_interleaved(samples: &[i32], scratch: &mut Vec<u8>) -> [u8; 32] {
    scratch.clear();
    scratch.reserve(samples.len() * 4);
    for s in samples {
        scratch.extend_from_slice(&s.to_le_bytes());
    }
    Sha256::digest(scratch)
}

/// Deterministic source-order rotation for repeat `r` over `n` sources.
///
/// Repeat 0 keeps the declared order; each later repeat starts one source later.
pub fn rotate_order(n: usize, repeat: u32) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let shift = (repeat as usize) % n;
    (0..n).map(|i| (i + shift) % n).collect()
}

/// One traversal of the frozen trace over one source for one object.
///
/// Counters are deterministic; `latency` is measured evidence.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TraversalRecord {
    pub source: String,
    pub object_id: String,
    pub repeat: u32,
    /// Was this repeat's precondition established (e.g. verified cache state)?
    pub eligible: bool,
    pub ineligible_reason: Option<String>,
    pub windows: usize,
    pub exact: bool,
    pub requested_frames: u64,
    pub returned_frames: u64,
    pub output_bytes: u64,
    pub logical_source_bytes_read: u64,
    pub physical_storage_bytes_read: u64,
    pub encoded_bytes_examined: u64,
    pub encoded_bytes_parsed: u64,
    pub sample_domain_bytes_materialized: u64,
    pub segments_touched: u64,
    pub pages_touched: u64,
    pub working_state_bytes: u64,
    pub scratch_peak_bytes: u64,
    pub deadline_misses: u64,
    /// Tightest deadline margin over the traversal (ns; negative = overrun).
    pub worst_deadline_margin_ns: i64,
    pub total_latency_ns: u64,
    pub latency: TailSummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_plan_is_sequential_and_exact() {
        let samples: Vec<i32> = (0..1300).collect();
        let plan = WindowPlan::build(&samples, 1).unwrap();
        assert_eq!(plan.len(), 3);
        assert_eq!(plan.spans(), &[(0, 512), (512, 512), (1024, 276)]);
        let mut scratch = Vec::new();
        assert_eq!(
            plan.window_matches(0, &samples[0..512], &mut scratch),
            Some(true)
        );
        assert_eq!(
            plan.window_matches(2, &samples[1024..1300], &mut scratch),
            Some(true)
        );
        let mut wrong = samples[0..512].to_vec();
        wrong[7] += 1;
        assert_eq!(plan.window_matches(0, &wrong, &mut scratch), Some(false));
        assert_eq!(plan.window_matches(3, &samples[0..512], &mut scratch), None);
    }

    #[test]
    fn rotation_is_deterministic_and_complete() {
        assert_eq!(rotate_order(4, 0), vec![0, 1, 2, 3]);
        assert_eq!(rotate_order(4, 1), vec![1, 2, 3, 0]);
        assert_eq!(rotate_order(4, 5), vec![1, 2, 3, 0]);
        let mut seen = rotate_order(5, 3);
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        assert!(rotate_order(0, 2).is_empty());
    }
}
