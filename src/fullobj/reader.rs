//! Bounded full-object reader (Phase M runtime).
//!
//! Verification and runtime observation are deliberately separated:
//!
//! ```text
//! container bytes
//!       │
//!       ▼
//! VerifiedFullObject::verify   ← once, outside any timed path. Checks the outer
//!       │                        digest, the frozen layout, every segment's
//!       │                        index content_id binding and the root semantics.
//!       ▼
//! FullObjectReader            ← bounded [start, frames) reads. Parses each
//!       │                        segment's encoded representation once (lazily)
//!       │                        and never retains a decoded waveform.
//!       ▼
//! dst: exact canonical interleaved i32
//! ```
//!
//! This is what makes B5 a fair runtime source: a bounded read decodes only the
//! pages its window touches, using the frozen page-bounded
//! `RepresentedLiteral::materialize` / `RepresentedResidual::materialize_closure`
//! paths, and windows crossing the 65,536-frame segment boundary concatenate
//! seamlessly.

use super::{
    DecodedFullObject, FullSemantics, decode_full_object, parse_canonical_segment, segment_payload,
    verify_segment_bindings,
};
use crate::entropy::represent::{RepresentedLiteral, RepresentedResidual};
use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::hash::sha256::Sha256;
use crate::object::ObjectData;
use crate::object::descriptor::ObjectDescriptor;

/// A container verified once, bound to its exact outer SHA-256.
#[derive(Debug, Clone)]
pub struct VerifiedFullObject {
    bytes: Vec<u8>,
    decoded: DecodedFullObject,
    sha256: [u8; 32],
    verify_ns: u64,
}

impl VerifiedFullObject {
    /// Full verification: parse, integrity, frozen layout, per-segment
    /// `content_id` binding and root semantics. Never in a timed read path.
    pub fn verify(bytes: Vec<u8>) -> Result<VerifiedFullObject> {
        let sw = Stopwatch::start();
        let decoded = decode_full_object(&bytes)?;
        verify_segment_bindings(&bytes, &decoded)?;
        let sha256 = Sha256::digest(&bytes);
        Ok(VerifiedFullObject {
            bytes,
            decoded,
            sha256,
            verify_ns: sw.elapsed_ns().max(0) as u64,
        })
    }

    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub fn decoded(&self) -> &DecodedFullObject {
        &self.decoded
    }

    pub fn channels(&self) -> u8 {
        self.decoded.channels
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.decoded.sample_rate_hz
    }

    pub fn total_frames(&self) -> u64 {
        self.decoded.total_frames
    }

    pub fn semantics(&self) -> FullSemantics {
        self.decoded.semantics
    }

    pub fn container_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Verification time (measured once, outside playback).
    pub fn verify_ns(&self) -> u64 {
        self.verify_ns
    }

    pub fn segment_count(&self) -> usize {
        self.decoded.segments.len()
    }
}

/// A parsed segment representation (encoded form only, never a waveform).
enum Prepared {
    Literal(RepresentedLiteral),
    Residual(RepresentedResidual),
    Canonical(ObjectDescriptor, ObjectData),
}

/// Per-read bounded-materialization counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadStats {
    pub segments_touched: u32,
    pub pages_touched: u32,
    /// Encoded payload bytes of the pages actually decoded for this window.
    pub encoded_bytes_examined: u64,
    /// Encoded payload bytes pulled into memory by a first parse this read.
    pub encoded_bytes_parsed: u64,
    pub sample_domain_bytes_materialized: u64,
    pub scratch_peak_bytes: u64,
    /// Bytes of parsed-segment state retained so far (cumulative).
    pub working_state_bytes: u64,
}

/// Bounded reader over a [`VerifiedFullObject`].
pub struct FullObjectReader {
    verified: VerifiedFullObject,
    prepared: Vec<Option<Prepared>>,
    working_state_bytes: u64,
}

impl FullObjectReader {
    pub fn new(verified: VerifiedFullObject) -> Self {
        let n = verified.decoded.segments.len();
        FullObjectReader {
            verified,
            prepared: (0..n).map(|_| None).collect(),
            working_state_bytes: 0,
        }
    }

    pub fn verified(&self) -> &VerifiedFullObject {
        &self.verified
    }

    pub fn channels(&self) -> usize {
        usize::from(self.verified.decoded.channels)
    }

    pub fn total_frames(&self) -> u64 {
        self.verified.decoded.total_frames
    }

    /// Parse (once) the encoded representation of segment `i`. Returns `true`
    /// when this call performed the parse.
    fn prepare(&mut self, i: usize) -> Result<bool> {
        if self.prepared[i].is_some() {
            return Ok(false);
        }
        let seg = self.verified.decoded.segments[i];
        let ch = self.verified.decoded.channels;
        let payload = segment_payload(&self.verified.bytes, &seg)?;
        let frames = seg.plan.frame_count;
        let prepared = match seg.representation {
            crate::object::descriptor::Representation::Literal => {
                let rl = crate::entropy::represent::parse_literal_container(payload)?;
                if rl.descriptor.extent_frames != frames || rl.descriptor.layout.count() != ch {
                    return Err(Error::malformed(
                        "stored literal segment disagrees with the index geometry",
                    ));
                }
                Prepared::Literal(rl)
            }
            crate::object::descriptor::Representation::PredictorResidual => {
                let rr = crate::entropy::represent::parse_residual_container(payload)?;
                if rr.descriptor.extent_frames != frames || rr.descriptor.layout.count() != ch {
                    return Err(Error::malformed(
                        "stored residual segment disagrees with the index geometry",
                    ));
                }
                Prepared::Residual(rr)
            }
            _ => {
                let (descriptor, data) = parse_canonical_segment(payload)?;
                if descriptor.layout.count() != ch {
                    return Err(Error::malformed(
                        "stored canonical segment channel count disagrees with the index",
                    ));
                }
                Prepared::Canonical(descriptor, data)
            }
        };
        // Parsed state is bounded by the encoded payload, never a waveform.
        self.working_state_bytes += seg.payload_length;
        self.prepared[i] = Some(prepared);
        Ok(true)
    }

    /// Materialize `[local_start, local_start + count)` within segment `i`.
    #[allow(clippy::type_complexity)]
    fn materialize_segment(
        &self,
        i: usize,
        local_start: u64,
        count: u32,
        ch: usize,
    ) -> Result<(Vec<i32>, u32, u64, u64)> {
        match self.prepared[i]
            .as_ref()
            .ok_or_else(|| Error::internal("segment not prepared"))?
        {
            Prepared::Literal(rl) => {
                let samples = rl.materialize(local_start, count)?;
                let mut pages = 0u32;
                let mut examined = 0u64;
                let mut peak = 0u64;
                let touching: std::collections::BTreeSet<u64> =
                    rl.pages_touching(local_start, count).into_iter().collect();
                for p in &rl.pages {
                    if touching.contains(&p.start_frame) {
                        pages += 1;
                        examined += rl.page_payload_bytes(p)?;
                        peak = peak.max(u64::from(p.frames) * ch as u64 * 4);
                    }
                }
                Ok((samples, pages, examined, peak))
            }
            Prepared::Residual(rr) => {
                let samples = rr.materialize_closure(local_start, count)?;
                let mut pages = 0u32;
                let mut examined = 0u64;
                let mut peak = 0u64;
                let touching: std::collections::BTreeSet<u64> =
                    rr.pages_touching(local_start, count).into_iter().collect();
                for p in &rr.pages {
                    if touching.contains(&p.start_frame) {
                        pages += 1;
                        examined += rr.page_payload_bytes(p)?;
                        peak = peak.max(u64::from(p.frames) * ch as u64 * 4);
                    }
                }
                Ok((samples, pages, examined, peak))
            }
            Prepared::Canonical(descriptor, data) => {
                let samples = canonical_window(descriptor, data, local_start, count, ch)?;
                Ok((samples, 0, 0, 0))
            }
        }
    }

    /// Read `[start_frame, start_frame + frames)` into `dst`.
    pub fn read(&mut self, start_frame: u64, frames: u32, dst: &mut [i32]) -> Result<ReadStats> {
        let ch = self.channels();
        if frames == 0 {
            return Err(Error::malformed("zero-frame full-object read"));
        }
        if dst.len() != frames as usize * ch {
            return Err(Error::malformed(
                "full-object read destination length != frames x channels",
            ));
        }
        let end = start_frame
            .checked_add(u64::from(frames))
            .ok_or_else(|| Error::limit("full-object read overflow"))?;
        if end > self.verified.decoded.total_frames {
            return Err(Error::malformed(
                "full-object read runs past the finite extent",
            ));
        }
        let mut stats = ReadStats::default();
        for i in 0..self.verified.decoded.segments.len() {
            let seg = self.verified.decoded.segments[i];
            let seg_start = seg.plan.start_frame;
            let seg_end = seg_start + seg.plan.frame_count;
            if seg_end <= start_frame || seg_start >= end {
                continue;
            }
            let lo = start_frame.max(seg_start);
            let hi = end.min(seg_end);
            let count = (hi - lo) as u32;
            let local_start = lo - seg_start;
            let freshly_parsed = self.prepare(i)?;
            if freshly_parsed {
                stats.encoded_bytes_parsed += seg.payload_length;
            }
            let (chunk, pages, examined, page_peak) =
                self.materialize_segment(i, local_start, count, ch)?;
            let dst_off = (lo - start_frame) as usize * ch;
            dst[dst_off..dst_off + count as usize * ch].copy_from_slice(&chunk);
            stats.segments_touched += 1;
            stats.pages_touched += pages;
            stats.encoded_bytes_examined += examined;
            stats.sample_domain_bytes_materialized += u64::from(count) * ch as u64 * 4;
            stats.scratch_peak_bytes = stats
                .scratch_peak_bytes
                .max(u64::from(count) * ch as u64 * 4 + page_peak);
        }
        stats.working_state_bytes = self.working_state_bytes;
        Ok(stats)
    }
}

/// Bounded window of a canonical segment (`Silence`, `Constant`, `ExactRepeat`).
fn canonical_window(
    descriptor: &ObjectDescriptor,
    data: &ObjectData,
    start: u64,
    frames: u32,
    ch: usize,
) -> Result<Vec<i32>> {
    if descriptor.layout.count() as usize != ch || ch == 0 {
        return Err(Error::malformed("canonical segment layout mismatch"));
    }
    let mut out = vec![0i32; frames as usize * ch];
    match data {
        ObjectData::Silence => {}
        ObjectData::Constant(c) => out.fill(c.level),
        ObjectData::ExactRepeat(cycle) => {
            if cycle.samples.len() % ch != 0 {
                return Err(Error::malformed("cycle payload is not frame-aligned"));
            }
            let period = cycle.samples.len() / ch;
            if period == 0 {
                return Err(Error::malformed("empty cycle payload"));
            }
            for f in 0..frames as usize {
                let src = ((start as usize + f) % period) * ch;
                out[f * ch..f * ch + ch].copy_from_slice(&cycle.samples[src..src + ch]);
            }
        }
        other => {
            return Err(Error::new(
                crate::error::Kind::Unsupported,
                format!("bounded runtime read of canonical {other:?} is not supported"),
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fullobj, inverse::SearchBudget};

    fn tone(frames: usize, channels: usize) -> Vec<i32> {
        (0..frames)
            .flat_map(|f| {
                let s = (((f % 64) as i32) << 14) - (1 << 12);
                (0..channels).map(move |c| s.wrapping_sub(c as i32 * 31))
            })
            .collect()
    }

    fn budget() -> SearchBudget {
        SearchBudget {
            max_period_scan: 128,
            max_residual_period_candidates: 2,
            max_candidates: 12,
            ..SearchBudget::default()
        }
    }

    #[test]
    fn bounded_reads_equal_the_full_materialization_and_nothing_is_retained() {
        // 65,537 frames means a second segment that a window can straddle.
        let total = 65_537u64;
        let channels = 2u8;
        let samples = tone(total as usize, usize::from(channels));
        let obj = fullobj::compile_full_object(
            "reader",
            48_000,
            channels,
            total,
            FullSemantics::OneShot,
            &samples,
            budget(),
        )
        .unwrap();
        let verified = VerifiedFullObject::verify(obj.bytes.clone()).unwrap();
        assert_eq!(verified.total_frames(), total);
        assert_eq!(verified.channels(), channels);
        let mut reader = FullObjectReader::new(verified);
        let ch = usize::from(channels);
        let mut dst = vec![0i32; 512 * ch];
        // Sequential 512-frame windows, including the straddling and final ones.
        let mut start = 0u64;
        let mut reads = 0;
        let mut segments_touched = 0u32;
        let mut pages_touched = 0u32;
        while start < total {
            let frames = 512u32.min((total - start) as u32);
            let mut dst = vec![0i32; frames as usize * ch];
            let stats = reader.read(start, frames, &mut dst).unwrap();
            let lo = start as usize * ch;
            let hi = (start + u64::from(frames)) as usize * ch;
            assert_eq!(dst, samples[lo..hi], "window {start} +{frames}");
            assert!(stats.segments_touched >= 1);
            segments_touched += stats.segments_touched;
            pages_touched += stats.pages_touched;
            reads += 1;
            start += u64::from(frames);
        }
        assert!(reads > 128);
        assert!(segments_touched >= reads as u32);
        // A literal/residual window touches at least one page; a canonical one
        // may touch none. The counters must be plausible, not zero everywhere.
        assert!(pages_touched > 0);
        // The reader retained encoded state only, never a waveform.
        let working = reader.working_state_bytes;
        assert!(working <= obj.complete_bytes());
        let _ = &mut dst;
    }

    #[test]
    fn loop_semantics_are_preserved_by_the_bounded_reader() {
        // Within the finite extent the bounded reader reproduces the stored
        // samples exactly (a one-shot root has no continuation).
        let period = 64usize;
        let total = 4096u64;
        let base = tone(period, 1);
        let samples: Vec<i32> = (0..total as usize).map(|f| base[f % period]).collect();
        let obj = fullobj::compile_full_object(
            "loop-reader",
            48_000,
            1,
            total,
            FullSemantics::Loop {
                period_frames: period as u32,
            },
            &samples,
            budget(),
        )
        .unwrap();
        let mut reader = FullObjectReader::new(VerifiedFullObject::verify(obj.bytes).unwrap());
        assert_eq!(
            reader.verified().semantics(),
            FullSemantics::Loop { period_frames: 64 }
        );
        let mut dst = vec![0i32; 96];
        let stats = reader.read(4000, 96, &mut dst).unwrap();
        assert_eq!(dst, samples[4000..4096]);
        assert!(stats.segments_touched >= 1);
    }
}
