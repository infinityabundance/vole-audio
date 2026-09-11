//! Common runtime substrate (Phase M): one contract, four source architectures.
//!
//! Every backend receives the **same** `[start, frames)` request and the **same**
//! caller-owned destination, and must return the exact canonical interleaved
//! `i32` window. Artifact construction (inverse search, FLAC encoding, raw PCM
//! file writing) is deliberately *outside* this interface: those are
//! authoring/compile operations, not playback.
//!
//! ```text
//! B2  PCM-resident sampler         canonical i32 in memory
//! B3  raw PCM disk streaming       canonical LE i32 on disk (verified cold/warm)
//! B4  compressed-file preload      the exact B1 FLAC-5 artifact -> resident PCM
//! B5  bounded VOLE materialization verified full-object container, page-bounded
//! ```
//!
//! **The latency boundary belongs to the harness, not the source.** A source
//! cannot decide which part of its work counts: the measurement layer wraps the
//! whole `read()` call, so every architecture is timed over the same operation.
//! For the same reason a source never performs `/proc/self/io` instrumentation
//! inside `read()`; physical read traffic is sampled by the harness at traversal
//! boundaries (and separately, in a dedicated instrumentation mode, if wanted).
//!
//! The frozen trace is sequential [`QUANTUM_FRAMES`]-frame windows at each
//! object's native rate and channel count: this court is about **source
//! materialization**, not endpoint resampling, so there are no gains, pans,
//! filters or random seeks.

pub mod advanced;
pub mod cache;
pub mod energy;
pub mod load;
pub mod protocol;
pub mod sources;

pub use cache::{CacheEvidence, CacheState, proc_self_io};
pub use energy::{EnergyCounter, PowerSource, probe_energy_counter, probe_power};
pub use load::{LoadGuard, LoadKind};
pub use protocol::{WindowPlan, frozen_random_trace, rotate_order};
pub use sources::{
    DiskPcmArtifact, DiskPcmSource, FlacPreloadSource, FlacSeekSource, ResidentPcmSource,
    VoleBoundedSource,
};

use crate::error::{Error, Result};

/// Frozen measurement quantum (frames per read).
pub const QUANTUM_FRAMES: u32 = 512;

/// Frozen number of trace traversals per source *and* per object. Each repeat
/// starts from equivalent initial source state ([`RuntimeSource::reset`]) and
/// the traversal order of sources is rotated deterministically per repeat.
pub const REPEATS: u32 = 3;

/// Source identity and persistent residency. A property of the prepared source,
/// not of any one read.
///
/// Storage and residency are **separate** quantities: bytes that live in a file
/// are not resident sample-domain bytes, and a decoded-PCM source does not keep
/// its compressed artifact resident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInfo {
    pub name: &'static str,
    pub channels: u8,
    pub total_frames: u64,
    pub sample_rate_hz: u32,
    /// Bytes the artifact occupies **in storage** (0 for a purely in-memory
    /// source; the compressed/decompressed file size otherwise).
    pub artifact_storage_bytes: u64,
    /// Sample-domain bytes **persistently resident in the process**.
    pub resident_sample_domain_bytes: u64,
    /// Encoded bytes **persistently resident in the process**.
    pub resident_encoded_bytes: u64,
    /// Runtime setup: open/load/decode/preload (ns). Artifact construction is
    /// not part of this and is reported separately by the court.
    pub runtime_setup_ns: u64,
    pub setup_detail: String,
}

/// Architecture-explicit evidence for one bounded read.
///
/// Deliberately contains **no timing**: latency, the frame deadline and the
/// deadline verdict are computed by the harness around the whole `read()` call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadEvidence {
    pub requested_frames: u32,
    pub returned_frames: u32,
    pub output_bytes: u64,

    /// Bytes of the logical source stream corresponding to the window.
    pub logical_source_bytes_read: u64,

    /// Encoded payload bytes examined to produce the window.
    pub encoded_bytes_examined: u64,
    /// Encoded payload bytes pulled into memory by a first parse this read.
    pub encoded_bytes_parsed: u64,
    /// Sample-domain bytes materialized for the window.
    pub sample_domain_bytes_materialized: u64,

    /// Encoded bytes persistently retained by the source.
    pub resident_encoded_bytes: u64,
    /// Sample-domain bytes persistently retained by the source.
    pub resident_sample_domain_bytes: u64,
    /// Parsed encoded state retained by the source (cumulative).
    pub working_state_bytes: u64,
    /// Peak transient buffer for this read.
    pub scratch_peak_bytes: u64,

    pub segments_touched: u32,
    pub pages_touched: u32,
}

/// The common runtime contract.
pub trait RuntimeSource {
    /// Source identity and persistent residency.
    fn info(&self) -> SourceInfo;

    /// Read `[start_frame, start_frame + frames)` into `dst`
    /// (`dst.len() == frames * channels`), exactly.
    fn read(&mut self, start_frame: u64, frames: u32, dst: &mut [i32]) -> Result<ReadEvidence>;

    /// Restore initial source state before a repeated traversal.
    ///
    /// B5's primary architecture is *first-play*: a fresh bounded reader per
    /// repeat, so the lazy per-segment parse cost occurs naturally inside the
    /// measured traversal. The default is a no-op (B2/B3/B4 are stateless
    /// readers or are re-controlled externally for B3's cold/warm repeats).
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Frame deadline for a window at a native rate (ns), using integer math.
pub const fn deadline_ns(frames: u32, sample_rate_hz: u32) -> u64 {
    if sample_rate_hz == 0 {
        return 0;
    }
    frames as u64 * 1_000_000_000 / sample_rate_hz as u64
}

/// Validate a bounded read request against a source identity.
pub(crate) fn check_request(
    info: &SourceInfo,
    start_frame: u64,
    frames: u32,
    dst: &[i32],
) -> Result<()> {
    let ch = usize::from(info.channels);
    if ch == 0 {
        return Err(Error::malformed("source has zero channels"));
    }
    if frames == 0 {
        return Err(Error::malformed("zero-frame read"));
    }
    if dst.len() != frames as usize * ch {
        return Err(Error::malformed(
            "read destination length != frames x channels",
        ));
    }
    let end = start_frame
        .checked_add(u64::from(frames))
        .ok_or_else(|| Error::limit("read overflow"))?;
    if end > info.total_frames {
        return Err(Error::malformed("read runs past the finite extent"));
    }
    Ok(())
}

/// The frozen sequential trace: [`QUANTUM_FRAMES`]-frame windows with a final
/// partial window, starting at frame 0.
pub fn frozen_trace(total_frames: u64, quantum: u32) -> Result<Vec<(u64, u32)>> {
    if quantum == 0 {
        return Err(Error::malformed("zero quantum"));
    }
    if total_frames == 0 {
        return Err(Error::malformed("empty trace"));
    }
    let mut out = Vec::new();
    let mut start = 0u64;
    while start < total_frames {
        let frames = quantum.min((total_frames - start) as u32);
        out.push((start, frames));
        start += u64::from(frames);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_trace_is_sequential_with_a_final_partial_window() {
        let t = frozen_trace(1300, 512).unwrap();
        assert_eq!(t, vec![(0, 512), (512, 512), (1024, 276)]);
        assert_eq!(frozen_trace(512, 512).unwrap(), vec![(0, 512)]);
        assert!(frozen_trace(0, 512).is_err());
        assert!(frozen_trace(10, 0).is_err());
    }

    #[test]
    fn deadlines_follow_the_native_rate() {
        assert_eq!(deadline_ns(512, 48_000), 10_666_666);
        assert_eq!(deadline_ns(512, 44_100), 11_609_977);
        assert_eq!(deadline_ns(512, 96_000), 5_333_333);
        assert_eq!(deadline_ns(512, 192_000), 2_666_666);
    }
}
