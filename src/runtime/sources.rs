//! The four runtime source architectures (Phase M).
//!
//! Each implements the same [`RuntimeSource`] contract: exact bounded output
//! into a caller-owned destination, with architecture-explicit counters and a
//! truthfully separated storage/residency model.
//!
//! **Nothing here times itself.** The harness owns the latency boundary; these
//! adapters only report deterministic counters and residency. Physical storage
//! traffic is the harness's business too (`/proc/self/io` is sampled at
//! traversal boundaries, never inside a timed read).

use crate::baseline::FlacArtifact;
use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::fullobj::{FullObjectReader, VerifiedFullObject};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::cache::{self, CacheEvidence};
use super::{ReadEvidence, RuntimeSource, SourceInfo, check_request};

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
    /// Take ownership of the canonical interleaved object (already built).
    pub fn open(
        name: &'static str,
        channels: u8,
        sample_rate_hz: u32,
        samples: Vec<i32>,
    ) -> Result<ResidentPcmSource> {
        let sw = Stopwatch::start();
        let ch = usize::from(channels);
        if ch == 0 || samples.is_empty() || !samples.len().is_multiple_of(ch) {
            return Err(Error::malformed("B2 source samples are not frame-aligned"));
        }
        let total_frames = (samples.len() / ch) as u64;
        let sample_bytes = samples.len() as u64 * 4;
        let runtime_setup_ns = sw.elapsed_ns().max(0) as u64;
        Ok(ResidentPcmSource {
            info: SourceInfo {
                name,
                channels,
                total_frames,
                sample_rate_hz,
                // Nothing is stored: the canonical PCM is the residency.
                artifact_storage_bytes: 0,
                resident_sample_domain_bytes: sample_bytes,
                resident_encoded_bytes: 0,
                runtime_setup_ns,
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
        let lo = start_frame as usize * ch;
        let hi = lo + frames as usize * ch;
        dst.copy_from_slice(&self.samples[lo..hi]);
        let bytes = u64::from(frames) * ch as u64 * 4;
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: bytes,
            logical_source_bytes_read: bytes,
            sample_domain_bytes_materialized: bytes,
            resident_encoded_bytes: self.info.resident_encoded_bytes,
            resident_sample_domain_bytes: self.info.resident_sample_domain_bytes,
            ..Default::default()
        })
    }
}

/// B3's authoring artifact: the canonical little-endian `i32` stream written to
/// disk. Creation (serialize + write + flush) is **authoring**, not runtime; the
/// runtime source only opens it.
pub struct DiskPcmArtifact {
    path: PathBuf,
    byte_len: u64,
    channels: u8,
    sample_rate_hz: u32,
    total_frames: u64,
    build_ns: u64,
}

impl DiskPcmArtifact {
    /// Create the raw canonical PCM file (outside the runtime measurement).
    pub fn create(
        path: &Path,
        channels: u8,
        sample_rate_hz: u32,
        samples: &[i32],
    ) -> Result<DiskPcmArtifact> {
        let ch = usize::from(channels);
        if ch == 0 || samples.is_empty() || !samples.len().is_multiple_of(ch) {
            return Err(Error::malformed(
                "B3 artifact samples are not frame-aligned",
            ));
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
            // (a dirty page cannot be evicted). This is authoring time.
            w.sync_all()?;
        }
        let build_ns = sw.elapsed_ns().max(0) as u64;
        Ok(DiskPcmArtifact {
            path: path.to_path_buf(),
            byte_len: bytes.len() as u64,
            channels,
            sample_rate_hz,
            total_frames: (samples.len() / ch) as u64,
            build_ns,
        })
    }

    /// Authoring time (ns): serialize, write and flush. Never runtime setup.
    pub fn build_ns(&self) -> u64 {
        self.build_ns
    }

    pub fn storage_bytes(&self) -> u64 {
        self.byte_len
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// B3 — raw PCM disk streaming: the exact canonical little-endian `i32` stream
/// serviced by positioned reads.
///
/// The stored representation is exactly B0's raw canonical PCM (no container or
/// header), so the comparison is like-for-like. Storage is on disk; **resident
/// sample-domain bytes are zero**.
pub struct DiskPcmSource {
    info: SourceInfo,
    file: File,
    path: PathBuf,
    byte_len: u64,
    scratch: Vec<u8>,
}

impl DiskPcmSource {
    /// Open the artifact for reading (runtime setup = open only).
    pub fn open(name: &'static str, artifact: &DiskPcmArtifact) -> Result<DiskPcmSource> {
        let sw = Stopwatch::start();
        let file = File::open(artifact.path())?;
        let runtime_setup_ns = sw.elapsed_ns().max(0) as u64;
        Ok(DiskPcmSource {
            info: SourceInfo {
                name,
                channels: artifact.channels,
                total_frames: artifact.total_frames,
                sample_rate_hz: artifact.sample_rate_hz,
                artifact_storage_bytes: artifact.byte_len,
                resident_sample_domain_bytes: 0,
                resident_encoded_bytes: 0,
                runtime_setup_ns,
                setup_detail: format!(
                    "canonical LE i32 on disk ({} bytes); positioned reads",
                    artifact.byte_len
                ),
            },
            file,
            path: artifact.path().to_path_buf(),
            byte_len: artifact.byte_len,
            scratch: Vec::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
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
        if self.scratch.len() < byte_len {
            self.scratch.resize(byte_len, 0);
        }
        let buf = &mut self.scratch[..byte_len];
        self.file.read_exact_at(buf, offset)?;
        for (i, w) in buf.as_chunks::<4>().0.iter().enumerate() {
            dst[i] = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        }
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: byte_len as u64,
            logical_source_bytes_read: byte_len as u64,
            sample_domain_bytes_materialized: byte_len as u64,
            resident_encoded_bytes: self.info.resident_encoded_bytes,
            resident_sample_domain_bytes: self.info.resident_sample_domain_bytes,
            scratch_peak_bytes: byte_len as u64,
            ..Default::default()
        })
    }
}

/// B4 — conventional compressed-file preload: the exact B1 FLAC-5 artifact,
/// decoded once into resident PCM, then bounded playback reads.
///
/// This is truthfully a *compressed-storage / decoded-resident* sampler: the
/// sealed B1 stream has no SEEKTABLE and is not modified to gain one. The
/// compressed artifact is **storage**, not resident — only the decoded PCM is
/// retained.
pub struct FlacPreloadSource {
    info: SourceInfo,
    pcm: Vec<i32>,
    artifact_sha256: [u8; 32],
}

impl FlacPreloadSource {
    /// Decode the artifact once (runtime setup) and hold the PCM.
    pub fn open(
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
        let decode_ns = sw.elapsed_ns().max(0) as u64;
        let ch = usize::from(channels.max(1));
        let total_frames = (decoded.interleaved.len() / ch) as u64;
        Ok(FlacPreloadSource {
            info: SourceInfo {
                name,
                channels,
                total_frames,
                sample_rate_hz,
                artifact_storage_bytes: artifact.bytes.len() as u64,
                resident_sample_domain_bytes: decoded.interleaved.len() as u64 * 4,
                resident_encoded_bytes: 0,
                runtime_setup_ns: decode_ns,
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
        let lo = start_frame as usize * ch;
        let hi = lo + frames as usize * ch;
        dst.copy_from_slice(&self.pcm[lo..hi]);
        let bytes = u64::from(frames) * ch as u64 * 4;
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: bytes,
            logical_source_bytes_read: bytes,
            sample_domain_bytes_materialized: bytes,
            resident_encoded_bytes: self.info.resident_encoded_bytes,
            resident_sample_domain_bytes: self.info.resident_sample_domain_bytes,
            ..Default::default()
        })
    }
}

/// B5 — bounded VOLE materialization over a verified full-object container.
///
/// The container is verified once (outside timed reads); bounded reads decode
/// only the pages their window touches and never retain a decoded waveform.
///
/// Two architectures are selectable and **never averaged together**:
///
/// * **first-play** (`prepared == false`): a fresh bounded reader per repeat, so
///   the lazy per-segment parse cost occurs inside the measured traversal;
/// * **prepared** (`prepared == true`): every segment representation is parsed
///   before measurement, so the timed reads are steady-state materialization.
pub struct VoleBoundedSource {
    info: SourceInfo,
    verified: Arc<VerifiedFullObject>,
    reader: FullObjectReader,
    prepared: bool,
    container_sha256: [u8; 32],
    container_bytes: u64,
}

impl VoleBoundedSource {
    pub fn open(
        name: &'static str,
        verified: Arc<VerifiedFullObject>,
        prepared: bool,
    ) -> Result<VoleBoundedSource> {
        let sw = Stopwatch::start();
        let mut reader = FullObjectReader::new(verified.clone());
        if prepared {
            reader.prepare_all()?;
        }
        let runtime_setup_ns = sw.elapsed_ns().max(0) as u64;
        let container_bytes = verified.container_bytes();
        let container_sha256 = verified.sha256();
        let segments = verified.segment_count();
        let mode = if prepared { "prepared" } else { "first-play" };
        Ok(VoleBoundedSource {
            info: SourceInfo {
                name,
                channels: verified.channels(),
                total_frames: verified.total_frames(),
                sample_rate_hz: verified.sample_rate_hz(),
                artifact_storage_bytes: container_bytes,
                resident_sample_domain_bytes: 0,
                // The verified container bytes are retained by the reader.
                resident_encoded_bytes: container_bytes,
                runtime_setup_ns,
                setup_detail: format!(
                    "full-object container ({segments} segments) {mode}; bounded reader retains \
                     encoded state only"
                ),
            },
            verified,
            reader,
            prepared,
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
        let stats = self.reader.read(start_frame, frames, dst)?;
        let bytes = u64::from(frames) * ch as u64 * 4;
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: bytes,
            logical_source_bytes_read: stats.encoded_bytes_examined,
            encoded_bytes_examined: stats.encoded_bytes_examined,
            encoded_bytes_parsed: stats.encoded_bytes_parsed,
            sample_domain_bytes_materialized: stats.sample_domain_bytes_materialized,
            resident_encoded_bytes: self.info.resident_encoded_bytes,
            resident_sample_domain_bytes: self.info.resident_sample_domain_bytes,
            working_state_bytes: stats.working_state_bytes,
            scratch_peak_bytes: stats.scratch_peak_bytes,
            segments_touched: stats.segments_touched,
            pages_touched: stats.pages_touched,
        })
    }

    fn reset(&mut self) -> Result<()> {
        // Prepared: the reader keeps its parsed state, so repeats are
        // steady-state. First-play: a fresh reader per repeat, so the lazy
        // parse occurs inside the measured traversal every time.
        if self.prepared {
            return Ok(());
        }
        self.reader = FullObjectReader::new(self.verified.clone());
        Ok(())
    }
}

/// B4-seek — conventional compressed random access: `decode_seek` on the exact
/// B1 artifact, which has **no SEEKTABLE**, so every seek decodes forward from
/// the first frame. This is the honest cost of stateless compressed seeking and
/// is deliberately a separate row from the decoded-resident [`FlacPreloadSource`].
pub struct FlacSeekSource {
    info: SourceInfo,
    bytes: Vec<u8>,
    artifact_sha256: [u8; 32],
}

impl FlacSeekSource {
    pub fn open(
        name: &'static str,
        channels: u8,
        sample_rate_hz: u32,
        artifact: &FlacArtifact,
    ) -> Result<FlacSeekSource> {
        let sw = Stopwatch::start();
        // Validate decodability once (setup); the read path is what is measured.
        let decoded = libflac_rs::decode(&artifact.bytes)
            .ok_or_else(|| Error::internal("B4-seek: artifact is not decodable"))?;
        if decoded.channels != u32::from(channels)
            || decoded.sample_rate != sample_rate_hz
            || decoded.bits_per_sample != 32
        {
            return Err(Error::internal(
                "B4-seek: artifact stream format disagrees with the object",
            ));
        }
        let ch = usize::from(channels.max(1));
        let total_frames = (decoded.interleaved.len() / ch) as u64;
        let runtime_setup_ns = sw.elapsed_ns().max(0) as u64;
        Ok(FlacSeekSource {
            info: SourceInfo {
                name,
                channels,
                total_frames,
                sample_rate_hz,
                artifact_storage_bytes: artifact.bytes.len() as u64,
                resident_sample_domain_bytes: 0,
                resident_encoded_bytes: artifact.bytes.len() as u64,
                runtime_setup_ns,
                setup_detail: "exact B1 FLAC-5 artifact held compressed; stateless decode_seek \
                               (no SEEKTABLE: decodes forward from the first frame)"
                    .into(),
            },
            bytes: artifact.bytes.clone(),
            artifact_sha256: artifact.sha256,
        })
    }

    pub fn artifact_sha256(&self) -> [u8; 32] {
        self.artifact_sha256
    }
}

impl RuntimeSource for FlacSeekSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn read(&mut self, start_frame: u64, frames: u32, dst: &mut [i32]) -> Result<ReadEvidence> {
        check_request(&self.info, start_frame, frames, dst)?;
        let ch = usize::from(self.info.channels);
        let seek = libflac_rs::decode_seek(&self.bytes, start_frame)
            .ok_or_else(|| Error::internal("B4-seek: decode_seek failed"))?;
        if seek.first_sample != start_frame || seek.channels as usize != ch {
            return Err(Error::internal(
                "B4-seek: decode_seek returned an unexpected origin/geometry",
            ));
        }
        let n = frames as usize * ch;
        if seek.interleaved.len() < n {
            return Err(Error::internal(
                "B4-seek: decode_seek returned too few samples",
            ));
        }
        dst.copy_from_slice(&seek.interleaved[..n]);
        Ok(ReadEvidence {
            requested_frames: frames,
            returned_frames: frames,
            output_bytes: n as u64 * 4,
            // No seektable: the decoder reads the whole compressed stream.
            logical_source_bytes_read: self.bytes.len() as u64,
            sample_domain_bytes_materialized: seek.interleaved.len() as u64 * 4,
            resident_encoded_bytes: self.info.resident_encoded_bytes,
            resident_sample_domain_bytes: self.info.resident_sample_domain_bytes,
            scratch_peak_bytes: seek.interleaved.len() as u64 * 4,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("runtime-src-test-{tag}-{}", std::process::id()))
    }

    #[test]
    fn storage_and_residency_are_separated() {
        let samples: Vec<i32> = (0..2048).collect();

        // B2: no stored artifact, the full canonical PCM is resident.
        let b2 = ResidentPcmSource::open("B2", 2, 48_000, samples.clone()).unwrap();
        let i2 = b2.info();
        assert_eq!(i2.artifact_storage_bytes, 0);
        assert_eq!(i2.resident_sample_domain_bytes, samples.len() as u64 * 4);
        assert_eq!(i2.resident_encoded_bytes, 0);

        // B3: the file is storage, not residency.
        let d = dir("b3");
        std::fs::create_dir_all(&d).unwrap();
        let path = d.join("a.pcm");
        let art = DiskPcmArtifact::create(&path, 2, 48_000, &samples).unwrap();
        let b3 = DiskPcmSource::open("B3", &art).unwrap();
        let i3 = b3.info();
        assert_eq!(i3.artifact_storage_bytes, art.storage_bytes());
        assert_eq!(i3.artifact_storage_bytes, samples.len() as u64 * 4);
        assert_eq!(
            i3.resident_sample_domain_bytes, 0,
            "disk PCM is not resident"
        );
        assert_eq!(i3.resident_encoded_bytes, 0);
        // The artifact path is a real file of exactly the canonical size.
        assert_eq!(std::fs::metadata(&path).unwrap().len(), art.storage_bytes());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn b3_reads_are_exact_and_positioned() {
        let samples: Vec<i32> = (0..4096i32).map(|i| i.wrapping_mul(7919)).collect();
        let d = dir("b3read");
        std::fs::create_dir_all(&d).unwrap();
        let path = d.join("b.pcm");
        let art = DiskPcmArtifact::create(&path, 1, 48_000, &samples).unwrap();
        let mut b3 = DiskPcmSource::open("B3", &art).unwrap();
        let mut dst = vec![0i32; 512];
        // Out-of-order positioned reads must match the canonical slice.
        for &start in &[1024u64, 0, 3584] {
            let frames = 512u32.min((samples.len() as u64 - start) as u32);
            let e = b3.read(start, frames, &mut dst[..frames as usize]).unwrap();
            assert_eq!(
                dst[..frames as usize],
                samples[start as usize..(start as usize + frames as usize)]
            );
            assert_eq!(e.logical_source_bytes_read, u64::from(frames) * 4);
            assert_eq!(e.scratch_peak_bytes, u64::from(frames) * 4);
        }
        // Reads past the finite extent are rejected, never clamped silently.
        assert!(b3.read(4096, 1, &mut dst[..1]).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
