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

use crate::error::{Error, Result};
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
        WindowPlan::from_spans(samples, channels, spans)
    }

    /// Build a plan over an explicit request sequence (e.g. the frozen
    /// random-access trace), precomputing each window's digest.
    pub fn from_spans(
        samples: &[i32],
        channels: usize,
        spans: Vec<(u64, u32)>,
    ) -> Result<WindowPlan> {
        let total_frames = (samples.len() / channels) as u64;
        let mut scratch = Vec::new();
        let mut digests = Vec::with_capacity(spans.len());
        for &(start, frames) in &spans {
            let end = start
                .checked_add(u64::from(frames))
                .ok_or_else(|| crate::error::Error::limit("window overflow"))?;
            if frames == 0 || end > total_frames {
                return Err(crate::error::Error::malformed(
                    "window plan request outside the finite extent",
                ));
            }
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

/// Frozen random-access trace for one object.
///
/// Deterministic from a seed (the object's canonical hash) — never from
/// measurement. It always includes the awkward edges a bounded reader must
/// handle: the first frame, the last frame, a window straddling the
/// [`crate::fullobj::MAX_SEGMENT_FRAMES`] boundary, and a full quantum at the
/// end; then `count` pseudo-random windows of varied width.
pub fn frozen_random_trace(
    total_frames: u64,
    quantum: u32,
    seed: u64,
    count: usize,
) -> Result<Vec<(u64, u32)>> {
    if total_frames == 0 {
        return Err(Error::malformed("empty random trace"));
    }
    if quantum == 0 {
        return Err(Error::malformed("zero quantum"));
    }
    let mut out: Vec<(u64, u32)> = Vec::with_capacity(count + 4);
    let push = |out: &mut Vec<(u64, u32)>, start: u64, frames: u64| {
        if start < total_frames && frames > 0 {
            let frames = frames.min(total_frames - start) as u32;
            out.push((start, frames));
        }
    };
    push(&mut out, 0, 1);
    push(&mut out, total_frames - 1, 1);
    let seg = crate::fullobj::MAX_SEGMENT_FRAMES;
    if total_frames > seg + 1 {
        push(&mut out, seg - 3, 7);
    }
    push(
        &mut out,
        total_frames.saturating_sub(u64::from(quantum)),
        u64::from(quantum),
    );

    let mut x = seed | 1;
    let mut next = || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        x >> 33
    };
    for _ in 0..count {
        let start = next() % total_frames;
        let width = match next() % 4 {
            0 => 1,
            1 => 64,
            2 => u64::from(quantum / 2).max(1),
            _ => u64::from(quantum),
        };
        push(&mut out, start, width);
    }
    Ok(out)
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

    #[test]
    fn random_trace_is_deterministic_bounded_and_includes_edges() {
        let a = frozen_random_trace(70_000, 512, 0xABCDEF, 32).unwrap();
        let b = frozen_random_trace(70_000, 512, 0xABCDEF, 32).unwrap();
        assert_eq!(a, b);
        assert!(a.contains(&(0, 1)));
        assert!(a.contains(&(69_999, 1)));
        assert!(a.contains(&(65_533, 7)), "segment-boundary straddle");
        assert!(a.contains(&(69_488, 512)));
        for &(start, frames) in &a {
            assert!(frames >= 1);
            assert!(start + u64::from(frames) <= 70_000);
        }
        // A different seed gives a different tail of pseudo-random windows.
        let c = frozen_random_trace(70_000, 512, 0x123456, 32).unwrap();
        assert_ne!(a, c);
        assert!(frozen_random_trace(0, 512, 1, 4).is_err());
    }
}
