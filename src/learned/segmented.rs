//! Segmented learned object (Exp2, priority `2`).
//!
//! A stationary model is often punished by one non-stationary region. The
//! segment model is a first-class representation: the extent is partitioned
//! into independently materializable regions, each with its own learned
//! hypothesis and its own exact residual. The canonical residual of the object
//! is the **concatenation** of the segment residuals, so accounting stays
//! exactly the sum of the physical bytes.
//!
//! Segment independence is declared: an inner model resets at its segment
//! boundary (block-local semantics), so reconstruction is a plain
//! concatenation and random access touches one segment.
//!
//! The unsplit path (one segment covering the extent) is always a candidate, so
//! segmentation can never enlarge the portfolio (`O` no-regression).

use crate::error::{Error, Result};
use crate::learned::model::LearnedModel;

/// One segment: a frame count and the hypothesis that explains it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Frames covered by this segment (`>= 1`).
    pub frames: u32,
    /// The learned hypothesis for this segment (never itself segmented).
    pub model: Box<LearnedModel>,
}

/// A sequence of independently materializable learned segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentedModel {
    pub channels: u8,
    pub segments: Vec<Segment>,
}

impl SegmentedModel {
    /// Total frames across all segments.
    pub fn frames(&self) -> u64 {
        self.segments.iter().map(|s| u64::from(s.frames)).sum()
    }

    /// Validate the segmentation, its nesting bound and its ceilings.
    pub fn validate(&self) -> Result<()> {
        let c = usize::from(self.channels);
        if c == 0 || c > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed(
                "segmented model channel count out of range",
            ));
        }
        if self.segments.is_empty() {
            return Err(Error::malformed("segmented model has no segments"));
        }
        if self.segments.len() as u32 > crate::limits::MAX_LEARNED_SEGMENTS {
            return Err(Error::limit("segmented model has too many segments"));
        }
        for s in &self.segments {
            if s.frames == 0 {
                return Err(Error::malformed("segmented model has an empty segment"));
            }
            if matches!(&*s.model, LearnedModel::Segmented(_)) {
                return Err(Error::malformed(
                    "segmented models may not nest another segmented model",
                ));
            }
            s.model.validate()?;
            if s.model.channels() != self.channels {
                return Err(Error::malformed(
                    "segment channel count disagrees with the segmented model",
                ));
            }
        }
        Ok(())
    }

    /// Global frame index to `(segment_index, local_frame)`.
    pub fn locate(&self, global_frame: usize) -> Result<(usize, usize)> {
        let mut remaining = global_frame;
        for (i, s) in self.segments.iter().enumerate() {
            let f = s.frames as usize;
            if remaining < f {
                return Ok((i, remaining));
            }
            remaining -= f;
        }
        Err(Error::malformed("segmented frame index exceeds the extent"))
    }

    /// Declared receptive field (maximum over segments).
    pub fn receptive_field(&self) -> u64 {
        self.segments
            .iter()
            .map(|s| s.model.receptive_field())
            .max()
            .unwrap_or(0)
    }

    /// Abstract operations per output sample (maximum over segments).
    pub fn ops_per_sample(&self) -> u64 {
        self.segments
            .iter()
            .map(|s| s.model.ops_per_sample())
            .max()
            .unwrap_or(0)
    }

    /// Persistent state bytes (sum over segments).
    pub fn state_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.model.state_bytes()).sum()
    }

    /// Declared checkpoints (sum over segments).
    pub fn checkpoint_count(&self) -> u32 {
        self.segments
            .iter()
            .map(|s| s.model.checkpoint_count())
            .sum()
    }

    /// Frames replayed to serve a seek: the local offset inside the segment.
    pub fn replay_frames(&self, start: usize) -> usize {
        match self.locate(start) {
            Ok((_, local)) => local,
            Err(_) => start,
        }
    }

    /// Canonical model bytes:
    /// `kind(4) || channels || seg_count(u32) || [frames(u32) || model_len(u64) || model]`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(4); // model kind 4 = segmented
        out.push(self.channels);
        out.extend_from_slice(&(self.segments.len() as u32).to_le_bytes());
        for s in &self.segments {
            out.extend_from_slice(&s.frames.to_le_bytes());
            let mb = s.model.canonical_bytes();
            out.extend_from_slice(&(mb.len() as u64).to_le_bytes());
            out.extend_from_slice(&mb);
        }
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<SegmentedModel> {
        if bytes.len() < 6 || bytes[0] != 4 {
            return Err(Error::malformed("segmented model header mismatch"));
        }
        let channels = bytes[1];
        let count = u32::from_le_bytes(bytes[2..6].try_into().unwrap());
        if count == 0 || count > crate::limits::MAX_LEARNED_SEGMENTS {
            return Err(Error::limit(
                "segmented model segment count exceeds the bound",
            ));
        }
        let mut at = 6usize;
        let mut segments = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let frames = u32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .ok_or_else(|| Error::malformed("segmented model is truncated"))?
                    .try_into()
                    .unwrap(),
            );
            at += 4;
            let mlen = u64::from_le_bytes(
                bytes
                    .get(at..at + 8)
                    .ok_or_else(|| Error::malformed("segmented model is truncated"))?
                    .try_into()
                    .unwrap(),
            );
            at += 8;
            let mlen = usize::try_from(mlen)
                .map_err(|_| Error::limit("segment model length exceeds host usize"))?;
            if mlen > crate::limits::MAX_LEARNED_WEIGHT_BYTES.saturating_mul(8) as usize {
                return Err(Error::limit("segment model length exceeds the bound"));
            }
            let mb = bytes
                .get(at..at + mlen)
                .ok_or_else(|| Error::malformed("segmented model is truncated"))?;
            at += mlen;
            let model = LearnedModel::from_canonical_bytes(mb)?;
            segments.push(Segment {
                frames,
                model: Box::new(model),
            });
        }
        if at != bytes.len() {
            return Err(Error::malformed("segmented model has trailing bytes"));
        }
        let m = SegmentedModel { channels, segments };
        m.validate()?;
        Ok(m)
    }

    /// Compute the hypothesis over the whole extent, open-loop from a source.
    /// `source` is the concatenated source (segment-major).
    pub fn hypothesis_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if source.len() != frames * c {
            return Err(Error::malformed("segmented source length mismatch"));
        }
        if self.frames() != frames as u64 {
            return Err(Error::malformed(
                "segmented frame count disagrees with the extent",
            ));
        }
        let mut out = Vec::with_capacity(frames * c);
        let mut base = 0usize;
        for s in &self.segments {
            let f = s.frames as usize;
            let h = s
                .model
                .hypothesis_from_source(&source[base * c..(base + f) * c], f)?;
            out.extend_from_slice(&h);
            base += f;
        }
        Ok(out)
    }

    /// Reconstruct `[start, start+len)` exactly from the concatenated residual.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if residual.len() != frames * c {
            return Err(Error::malformed("segmented residual length mismatch"));
        }
        if self.frames() != frames as u64 {
            return Err(Error::malformed(
                "segmented frame count disagrees with the extent",
            ));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("segmented range overflows"))?;
        if end > frames {
            return Err(Error::malformed("segmented range exceeds the extent"));
        }
        let mut out = vec![0i32; len * c];
        if len == 0 {
            return Ok(out);
        }
        let mut base = 0usize;
        for s in &self.segments {
            let f = s.frames as usize;
            let seg_start = base;
            let seg_end = base + f;
            let lo = start.max(seg_start);
            let hi = end.min(seg_end);
            if lo < hi {
                let local_start = lo - seg_start;
                let local_len = hi - lo;
                let seg_residual = &residual[seg_start * c..seg_end * c];
                let vals = s
                    .model
                    .evaluate_range(seg_residual, f, local_start, local_len)?;
                out[(lo - start) * c..(hi - start) * c].copy_from_slice(&vals);
            }
            base = seg_end;
            if base >= end {
                break;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::quantize_weight;
    use crate::learned::finite_field::LinearPredictor;
    use crate::learned::object::LearnedObject;

    fn simple_segment(frames: u32, samples: &[i32]) -> (Segment, Vec<i32>) {
        // A previous-sample predictor per segment.
        let p = LinearPredictor {
            channels: 1,
            taps: 1,
            weights: vec![quantize_weight(1.0)],
            bias: vec![0],
            block_frames: Some(frames),
        };
        let model = LearnedModel::Linear(p);
        let h = model
            .hypothesis_from_source(samples, frames as usize)
            .unwrap();
        let residual: Vec<i32> = samples
            .iter()
            .zip(h.iter())
            .map(|(&x, &hh)| (i64::from(x) - i64::from(hh)) as i32)
            .collect();
        (
            Segment {
                frames,
                model: Box::new(model),
            },
            residual,
        )
    }

    #[test]
    fn segmentation_reconstructs_exactly() {
        let a: Vec<i32> = (0..100).map(|i| i * 3 - 50).collect();
        let b: Vec<i32> = (0..100).map(|i| 40_000 - i * 7).collect();
        let (sa, ra) = simple_segment(100, &a);
        let (sb, rb) = simple_segment(100, &b);
        let mut residual = ra.clone();
        residual.extend_from_slice(&rb);
        let m = SegmentedModel {
            channels: 1,
            segments: vec![sa, sb],
        };
        m.validate().unwrap();
        let source = [a.clone(), b.clone()].concat();
        assert_eq!(m.hypothesis_from_source(&source, 200).unwrap().len(), 200);
        let full = m.evaluate_range(&residual, 200, 0, 200).unwrap();
        assert_eq!(full, source);
        // Partial range across the boundary.
        let part = m.evaluate_range(&residual, 200, 80, 60).unwrap();
        assert_eq!(part, source[80..140].to_vec());
        // Seek into the second segment replays only within it.
        assert_eq!(m.replay_frames(150), 50);
        assert_eq!(m.replay_frames(10), 10);
    }

    #[test]
    fn canonical_round_trip_is_exact_and_rejects_nesting() {
        let a: Vec<i32> = (0..64).map(|i| i * 5 - 30).collect();
        let (sa, _) = simple_segment(64, &a);
        let m = SegmentedModel {
            channels: 1,
            segments: vec![sa.clone()],
        };
        let bytes = m.canonical_bytes();
        let back = SegmentedModel::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(back, m);
        // Nested segmentation is rejected.
        let nested = SegmentedModel {
            channels: 1,
            segments: vec![Segment {
                frames: 64,
                model: Box::new(LearnedModel::Segmented(m.clone())),
            }],
        };
        assert!(nested.validate().is_err());
    }

    #[test]
    fn segmented_object_round_trips_through_the_container() {
        let a: Vec<i32> = (0..120).map(|i| (i * 97) % 401 - 200).collect();
        let b: Vec<i32> = (0..120).map(|i| (i * 13) % 97 + 1000).collect();
        let mut source = a.clone();
        source.extend_from_slice(&b);
        let (sa, _) = simple_segment(120, &a);
        let (sb, _) = simple_segment(120, &b);
        let m = SegmentedModel {
            channels: 1,
            segments: vec![sa, sb],
        };
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::Segmented(m),
            1,
            240,
            48_000,
            Vec::new(),
            &source,
        )
        .unwrap();
        assert!(o.verify(&source));
        let bytes = o.canonical_bytes();
        let back = LearnedObject::parse(&bytes).unwrap();
        assert_eq!(back.materialize().unwrap(), source);
        assert_eq!(back.profile, crate::learned::profile::LearnedProfile::Exp2);
    }
}
