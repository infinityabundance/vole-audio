//! The four runtime source architectures (Phase M).
//!
//! Each implements the same [`RuntimeSource`] contract: exact bounded output
//! into a caller-owned destination, with architecture-explicit counters.

use crate::baseline::FlacArtifact;
use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::fullobj::{FullObjectReader, VerifiedFullObject};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::cache::{self, CacheEvidence};
use super::{ReadEvidence, RuntimeSource, SourceInfo, check_request, deadline_ns};

/// B2 — PCM-resident sampler: the canonical `i32` object held in memory.
///
/// The runtime operation is a bounded copy into `dst`. It deliberately does
/// *not* return a borrowed slice: every source must satisfy the same output
/// contract.
pub struct ResidentPcmSource {
    info: SourceInfo,
    samples: Vec<i32>,
}

impl ResidentPcmSource {
    pub fn prepare(
        name: &'static str,
        channels: u8,
        sample_rate_hz: u32,
        samples: Vec<i32>,
    ) -> Result<ResidentPcmSource> {
        let ch = usize::from(channels);
        if ch == 0 || samples.is_empty() || !samples.len().is_multiple_of(ch) {
            return Err(Error::malformed("B2 source samples are not frame-aligned"));
        }
        let total_frames = (samples.len() / ch) as u64;
        Ok(ResidentPcmSource {
            info: SourceInfo {
                name,
                channels,
                total_frames,
                sample_rate_hz,
                persistent_encoded_bytes: 0,
                persistent_sample_domain_bytes: samples.len() as u64 * 4,
                setup_ns: 0,
                setup_detail: "canonical i32 resident in memory (prepared once)".into(),
            },
            samples,
        })
    }
}

impl RuntimeSource for ResidentPcmSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn read(&mut self, start_frame: u64, frames: u32, dst: &mut [i32]) -> Result<ReadEvidence> {
        check_request(&self.info, start_frame, frames, dst)?;
        let ch = usize::from(self.info.channels);
        let sw = Stopwatch::start();
        let lo = start_frame as usize * ch;
        let hi = lo + frames as usize * ch;
        dst.copy_from_slice(&self.samples[lo..hi]);
        let latency_ns = sw.elapsed_ns().max(0) as u64;
        let bytes = u64::from(frames) * ch as u64 * 4;
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: bytes,
            logical_source_bytes_read: bytes,
            sample_domain_bytes_materialized: bytes,
            persistent_encoded_bytes: self.info.persistent_encoded_bytes,
            persistent_sample_domain_bytes: self.info.persistent_sample_domain_bytes,
            latency_ns,
            deadline_ns: deadline_ns(frames, self.info.sample_rate_hz),
            deadline_missed: latency_ns > deadline_ns(frames, self.info.sample_rate_hz),
            ..Default::default()
        })
    }
}

/// B3 — raw PCM disk streaming: the exact canonical little-endian `i32` stream
/// written to a file, serviced by positioned reads.
///
/// The stored representation is exactly B0's raw canonical PCM (no container or
/// header), so the comparison is like-for-like.
pub struct DiskPcmSource {
    info: SourceInfo,
    file: File,
    path: PathBuf,
    byte_len: u64,
}

impl DiskPcmSource {
    /// Write the canonical LE stream to `path` and open it for reading.
    pub fn prepare(
        name: &'static str,
        channels: u8,
        sample_rate_hz: u32,
        samples: &[i32],
        path: &Path,
    ) -> Result<DiskPcmSource> {
        let ch = usize::from(channels);
        if ch == 0 || samples.is_empty() || !samples.len().is_multiple_of(ch) {
            return Err(Error::malformed("B3 source samples are not frame-aligned"));
        }
        let sw = Stopwatch::start();
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        {
            use std::io::Write;
            let mut w = File::create(path)?;
            w.write_all(&bytes)?;
            // Flush so a later cold-eviction attempt can actually drop the pages
            // (a dirty page cannot be evicted). The write time includes this.
            w.sync_all()?;
        }
        let file = File::open(path)?;
        let setup_ns = sw.elapsed_ns().max(0) as u64;
        let byte_len = bytes.len() as u64;
        Ok(DiskPcmSource {
            info: SourceInfo {
                name,
                channels,
                total_frames: (samples.len() / ch) as u64,
                sample_rate_hz,
                persistent_encoded_bytes: 0,
                persistent_sample_domain_bytes: byte_len,
                setup_ns,
                setup_detail: format!(
                    "canonical LE i32 written to {} ({} bytes)",
                    path.display(),
                    byte_len
                ),
            },
            file,
            path: path.to_path_buf(),
            byte_len,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Verified cache state of `[start_frame, start_frame + frames)`.
    pub fn cache_evidence(&self, start_frame: u64, frames: u32) -> CacheEvidence {
        let ch = u64::from(self.info.channels);
        let off = start_frame * ch * 4;
        let len = u64::from(frames) * ch * 4;
        cache::verify_cache(&self.file, off, len)
    }

    /// Attempt to evict the whole object, then report the verified state.
    pub fn evict_and_verify(&self) -> CacheEvidence {
        cache::evict_and_verify(&self.file, 0, self.byte_len)
    }

    /// Prime the whole object, then report the verified state.
    pub fn prime_and_verify(&self) -> Result<CacheEvidence> {
        Ok(cache::prime_and_verify(&self.file, 0, self.byte_len)?)
    }
}

impl RuntimeSource for DiskPcmSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn read(&mut self, start_frame: u64, frames: u32, dst: &mut [i32]) -> Result<ReadEvidence> {
        check_request(&self.info, start_frame, frames, dst)?;
        let ch = usize::from(self.info.channels);
        let byte_len = frames as usize * ch * 4;
        let offset = start_frame * ch as u64 * 4;
        let mut buf = vec![0u8; byte_len];
        let io_before = cache::proc_self_io();
        let sw = Stopwatch::start();
        self.file.read_exact_at(&mut buf, offset)?;
        let latency_ns = sw.elapsed_ns().max(0) as u64;
        let io_after = cache::proc_self_io();
        let storage_bytes_read = match (io_before, io_after) {
            (Some((_, a)), Some((_, b))) => b.saturating_sub(a),
            _ => 0,
        };
        for (i, w) in buf.as_chunks::<4>().0.iter().enumerate() {
            dst[i] = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        }
        let dl = deadline_ns(frames, self.info.sample_rate_hz);
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: byte_len as u64,
            logical_source_bytes_read: byte_len as u64,
            storage_bytes_read,
            sample_domain_bytes_materialized: byte_len as u64,
            persistent_encoded_bytes: self.info.persistent_encoded_bytes,
            persistent_sample_domain_bytes: self.info.persistent_sample_domain_bytes,
            scratch_peak_bytes: byte_len as u64,
            latency_ns,
            deadline_ns: dl,
            deadline_missed: latency_ns > dl,
            ..Default::default()
        })
    }
}

/// B4 — conventional compressed-file preload: the exact B1 FLAC-5 artifact,
/// decoded once into resident PCM, then bounded playback reads.
///
/// This is truthfully a *compressed-storage / decoded-resident* sampler: the
/// sealed B1 stream has no SEEKTABLE and is not modified to gain one.
pub struct FlacPreloadSource {
    info: SourceInfo,
    pcm: Vec<i32>,
    artifact_sha256: [u8; 32],
}

impl FlacPreloadSource {
    /// Decode the artifact once (measured setup) and hold the PCM.
    pub fn prepare(
        name: &'static str,
        channels: u8,
        sample_rate_hz: u32,
        artifact: &FlacArtifact,
    ) -> Result<FlacPreloadSource> {
        let sw = Stopwatch::start();
        let decoded = libflac_rs::decode(&artifact.bytes).ok_or_else(|| {
            Error::internal("B4: the artifact the encoder produced is not decodable")
        })?;
        if decoded.channels != u32::from(channels)
            || decoded.sample_rate != sample_rate_hz
            || decoded.bits_per_sample != 32
        {
            return Err(Error::internal(
                "B4: artifact stream format disagrees with the object",
            ));
        }
        let setup_ns = sw.elapsed_ns().max(0) as u64;
        let total_frames = (decoded.interleaved.len() / usize::from(channels.max(1))) as u64;
        Ok(FlacPreloadSource {
            info: SourceInfo {
                name,
                channels,
                total_frames,
                sample_rate_hz,
                persistent_encoded_bytes: artifact.bytes.len() as u64,
                persistent_sample_domain_bytes: decoded.interleaved.len() as u64 * 4,
                setup_ns,
                setup_detail: "one-time exact full FLAC decode -> resident PCM \
                               (no seektable; the sealed B1 stream is unmodified)"
                    .into(),
            },
            pcm: decoded.interleaved,
            artifact_sha256: artifact.sha256,
        })
    }

    pub fn artifact_sha256(&self) -> [u8; 32] {
        self.artifact_sha256
    }
}

impl RuntimeSource for FlacPreloadSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn read(&mut self, start_frame: u64, frames: u32, dst: &mut [i32]) -> Result<ReadEvidence> {
        check_request(&self.info, start_frame, frames, dst)?;
        let ch = usize::from(self.info.channels);
        let sw = Stopwatch::start();
        let lo = start_frame as usize * ch;
        let hi = lo + frames as usize * ch;
        dst.copy_from_slice(&self.pcm[lo..hi]);
        let latency_ns = sw.elapsed_ns().max(0) as u64;
        let bytes = u64::from(frames) * ch as u64 * 4;
        let dl = deadline_ns(frames, self.info.sample_rate_hz);
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: bytes,
            logical_source_bytes_read: bytes,
            sample_domain_bytes_materialized: bytes,
            persistent_encoded_bytes: self.info.persistent_encoded_bytes,
            persistent_sample_domain_bytes: self.info.persistent_sample_domain_bytes,
            latency_ns,
            deadline_ns: dl,
            deadline_missed: latency_ns > dl,
            ..Default::default()
        })
    }
}

/// B5 — bounded VOLE materialization over a verified full-object container.
///
/// The container is verified once (outside timed reads); bounded reads decode
/// only the pages their window touches and never retain a decoded waveform.
pub struct VoleBoundedSource {
    info: SourceInfo,
    reader: FullObjectReader,
    container_sha256: [u8; 32],
    container_bytes: u64,
}

impl VoleBoundedSource {
    pub fn prepare(name: &'static str, bytes: Vec<u8>) -> Result<VoleBoundedSource> {
        let verified = VerifiedFullObject::verify(bytes)?;
        let channels = verified.channels();
        let total_frames = verified.total_frames();
        let sample_rate_hz = verified.sample_rate_hz();
        let setup_ns = verified.verify_ns();
        let container_bytes = verified.container_bytes();
        let container_sha256 = verified.sha256();
        let segments = verified.segment_count();
        Ok(VoleBoundedSource {
            info: SourceInfo {
                name,
                channels,
                total_frames,
                sample_rate_hz,
                persistent_encoded_bytes: container_bytes,
                persistent_sample_domain_bytes: 0,
                setup_ns,
                setup_detail: format!(
                    "full-object container verified once ({segments} segments); bounded reader \
                     retains encoded state only"
                ),
            },
            reader: FullObjectReader::new(verified),
            container_sha256,
            container_bytes,
        })
    }

    pub fn container_sha256(&self) -> [u8; 32] {
        self.container_sha256
    }

    pub fn container_bytes(&self) -> u64 {
        self.container_bytes
    }
}

impl RuntimeSource for VoleBoundedSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn read(&mut self, start_frame: u64, frames: u32, dst: &mut [i32]) -> Result<ReadEvidence> {
        check_request(&self.info, start_frame, frames, dst)?;
        let ch = usize::from(self.info.channels);
        let sw = Stopwatch::start();
        let stats = self.reader.read(start_frame, frames, dst)?;
        let latency_ns = sw.elapsed_ns().max(0) as u64;
        let bytes = u64::from(frames) * ch as u64 * 4;
        let dl = deadline_ns(frames, self.info.sample_rate_hz);
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: bytes,
            logical_source_bytes_read: stats.encoded_bytes_examined,
            storage_bytes_read: 0,
            encoded_bytes_examined: stats.encoded_bytes_examined,
            encoded_bytes_parsed: stats.encoded_bytes_parsed,
            sample_domain_bytes_materialized: stats.sample_domain_bytes_materialized,
            persistent_encoded_bytes: self.info.persistent_encoded_bytes,
            persistent_sample_domain_bytes: self.info.persistent_sample_domain_bytes,
            working_state_bytes: stats.working_state_bytes,
            scratch_peak_bytes: stats.scratch_peak_bytes,
            segments_touched: stats.segments_touched,
            pages_touched: stats.pages_touched,
            latency_ns,
            deadline_ns: dl,
            deadline_missed: latency_ns > dl,
        })
    }
}
