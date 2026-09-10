//! Shared measured traversal (Phase M).
//!
//! Every Phase-M measurement court runs the same operation: a sequence of
//! bounded `[start, frames)` reads over a [`RuntimeSource`], timed by the
//! harness, checked against window digests precomputed during preparation, and
//! counted for physical I/O at traversal boundaries.
//!
//! Keeping this in one place is what makes B2/B3/B4/B5 comparable across
//! courts: no source ever times itself, and no court gets to choose a
//! different boundary.

use crate::error::Result;
use crate::evidence::timing::Stopwatch;
use crate::runtime::{RuntimeSource, WindowPlan, deadline_ns, proc_self_io};

/// Deterministic counters and measured samples from one traversal.
#[derive(Clone, Default)]
pub struct Traversal {
    pub exact: bool,
    pub windows: usize,
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
    pub latencies: Vec<u64>,
}

/// Run a prepared window plan once over one source.
///
/// The stopwatch wraps the **entire** `read` call: this is the single latency
/// boundary every source is measured against. Physical storage traffic is
/// sampled around the whole traversal, outside every timer.
pub fn run_traversal(
    source: &mut dyn RuntimeSource,
    plan: &WindowPlan,
    channels: usize,
    sample_rate_hz: u32,
    dst: &mut [i32],
    scratch: &mut Vec<u8>,
) -> Result<Traversal> {
    let mut t = Traversal {
        exact: true,
        windows: plan.len(),
        worst_deadline_margin_ns: i64::MAX,
        ..Default::default()
    };
    let io_before = proc_self_io();
    for (i, &(start, frames)) in plan.spans().iter().enumerate() {
        let n = frames as usize * channels;
        let sw = Stopwatch::start();
        let e = source.read(start, frames, &mut dst[..n])?;
        let latency_ns = sw.elapsed_ns().max(0) as u64;

        if plan.window_matches(i, &dst[..n], scratch) != Some(true) {
            t.exact = false;
        }
        let deadline = deadline_ns(frames, sample_rate_hz);
        if latency_ns > deadline {
            t.deadline_misses += 1;
        }
        t.worst_deadline_margin_ns = t
            .worst_deadline_margin_ns
            .min(deadline as i64 - latency_ns as i64);
        t.latencies.push(latency_ns);

        t.requested_frames += u64::from(e.requested_frames);
        t.returned_frames += u64::from(e.returned_frames);
        t.output_bytes += e.output_bytes;
        t.logical_source_bytes_read += e.logical_source_bytes_read;
        t.encoded_bytes_examined += e.encoded_bytes_examined;
        t.encoded_bytes_parsed += e.encoded_bytes_parsed;
        t.sample_domain_bytes_materialized += e.sample_domain_bytes_materialized;
        t.segments_touched += u64::from(e.segments_touched);
        t.pages_touched += u64::from(e.pages_touched);
        t.working_state_bytes = t.working_state_bytes.max(e.working_state_bytes);
        t.scratch_peak_bytes = t.scratch_peak_bytes.max(e.scratch_peak_bytes);
    }
    let io_after = proc_self_io();
    t.physical_storage_bytes_read = match (io_before, io_after) {
        (Some((_, a)), Some((_, b))) => b.saturating_sub(a),
        _ => 0,
    };
    if t.windows == 0 {
        t.worst_deadline_margin_ns = 0;
    }
    Ok(t)
}

/// Cost of the measurement apparatus itself (two clock reads per window).
///
/// Sub-100 ns latencies are near this floor and must be read that way rather
/// than as a precise operation time.
pub fn timer_overhead_ns() -> (u64, u64) {
    let mut v: Vec<u64> = Vec::with_capacity(10_001);
    for _ in 0..10_001 {
        let sw = Stopwatch::start();
        v.push(sw.elapsed_ns().max(0) as u64);
    }
    v.sort_unstable();
    (v[0], v[v.len() / 2])
}

/// Minimum prefill depth (in quanta) at which a producer with these per-window
/// latencies never underruns a consumer that needs one quantum per `deadline_ns`.
///
/// ```text
/// completion[i] = latency_0 + ... + latency_i
/// underrun  <=>  completion[i] - i*deadline > completion[k-1]
/// min depth  =  smallest k with completion[k-1] >= max_i(completion[i] - i*deadline)
/// ```
///
/// `None` means no finite depth within the traversal keeps up (the producer is
/// slower than realtime in aggregate).
pub fn min_stable_depth_quanta(latencies: &[u64], deadline_ns: u64) -> Option<u64> {
    let n = latencies.len();
    if n == 0 || deadline_ns == 0 {
        return None;
    }
    let mut completion: Vec<u64> = Vec::with_capacity(n);
    let mut acc = 0u64;
    for &l in latencies {
        acc = acc.saturating_add(l);
        completion.push(acc);
    }
    let need = completion
        .iter()
        .enumerate()
        .map(|(i, &c)| c as i64 - (i as i64) * deadline_ns as i64)
        .max()
        .unwrap_or(0);
    for k in 1..=n {
        if completion[k - 1] as i64 >= need {
            return Some(k as u64);
        }
    }
    None
}

/// Sustained-streaming stability for a repeated/infinite workload.
///
/// [`min_stable_depth_quanta`] is a *finite-object* quantity: at `k = n` the
/// whole object has been prefetched, so it always resolves. That is an escape,
/// not a streaming guarantee. For a repeated workload the producer keeps up iff
/// total production time does not exceed total consumption time:
///
/// ```text
/// sum(latency) <= n * deadline   =>  stable, needs `initial_depth` buffered
/// otherwise                      =>  the buffer grows without bound
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingStability {
    /// Average production exceeds consumption: no bounded buffer keeps up.
    UnstableStreaming,
    /// Stable; the minimum initial buffer that absorbs the measured jitter.
    Stable { initial_depth_quanta: u64 },
}

/// Decide sustained-streaming stability from measured per-window latencies.
pub fn streaming_stability(latencies: &[u64], deadline_ns: u64) -> StreamingStability {
    let n = latencies.len() as u128;
    if n == 0 || deadline_ns == 0 {
        return StreamingStability::UnstableStreaming;
    }
    let total: u128 = latencies.iter().map(|&x| x as u128).sum();
    if total > n * deadline_ns as u128 {
        return StreamingStability::UnstableStreaming;
    }
    match min_stable_depth_quanta(latencies, deadline_ns) {
        Some(initial_depth_quanta) => StreamingStability::Stable {
            initial_depth_quanta,
        },
        None => StreamingStability::UnstableStreaming,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_is_one_when_every_window_meets_its_deadline() {
        assert_eq!(min_stable_depth_quanta(&[10, 20, 30], 100), Some(1));
    }

    #[test]
    fn a_slow_early_window_raises_the_required_depth() {
        // window 1 takes 25 against a 10 deadline: need = 20, so prefill 2.
        assert_eq!(min_stable_depth_quanta(&[5, 25], 10), Some(2));
    }

    #[test]
    fn degenerate_inputs_have_no_finite_depth() {
        assert_eq!(min_stable_depth_quanta(&[], 100), None);
        assert_eq!(min_stable_depth_quanta(&[1, 2], 0), None);
    }

    #[test]
    fn streaming_stability_separates_slow_producers_from_jitter() {
        // Every window at half its deadline: stable, one quantum of headroom.
        assert_eq!(
            streaming_stability(&[5, 5, 5], 10),
            StreamingStability::Stable {
                initial_depth_quanta: 1
            }
        );
        // Average production above the deadline: no bounded buffer keeps up.
        assert_eq!(
            streaming_stability(&[15, 15, 15], 10),
            StreamingStability::UnstableStreaming
        );
        // A finite object can still be partially prefetched even when the average
        // is too slow, which is precisely why the two quantities are distinct.
        assert_eq!(min_stable_depth_quanta(&[15, 15, 15], 10), Some(2));
        assert!(streaming_stability(&[], 10) == StreamingStability::UnstableStreaming);
    }
}
