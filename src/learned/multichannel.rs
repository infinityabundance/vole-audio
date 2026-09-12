//! Multichannel prediction with reversible integer lifting (Exp2, priority `5`).
//!
//! A dense `K·C·C` learned matrix is general but expensive. Audio channel
//! redundancy is usually much simpler than an arbitrary all-to-all FIR. This
//! family applies an **exactly reversible integer channel transform** (lifting)
//! and then predicts each transformed component with its own mono sparse
//! predictor. Every transform here is bijective on `i32` (wrapping arithmetic),
//! so reconstruction is exact over the full canonical domain.
//!
//! Model kind tag `7`. The hypothesis stays in the **channel domain**:
//!
//! ```text
//! Z_c[t]      = predictor_c(Z history)[t]          // one mono predictor per component
//! H[t]        = T^{-1}(Z_0[t], …, Z_{C-1}[t])      // channel-domain hypothesis
//! X_hat[t]    = sat_i32(H[t] + R[t])
//! ```
//!
//! The component history is the forward transform of the already reconstructed
//! `X_hat` history, so decoding is closed-loop and exact.

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, sat_i32};
use crate::learned::sparse::SparseLinearPredictor;

/// Channel transforms (frozen ids).
pub const TRANSFORM_NONE: u8 = 0;
/// Stereo reversible mid/side lifting: `S = R - L`, `M = L + (S >> 1)`.
pub const TRANSFORM_MID_SIDE: u8 = 1;
/// Stereo left/difference lifting: `(L, R - L)`.
pub const TRANSFORM_LEFT_DIFF: u8 = 2;
/// Stereo right/difference lifting: `(R, L - R)`.
pub const TRANSFORM_RIGHT_DIFF: u8 = 3;

/// The frozen reversible two-channel transform ladder, in canonical tie order.
pub const STEREO_TRANSFORMS: [u8; 4] = [
    TRANSFORM_NONE,
    TRANSFORM_MID_SIDE,
    TRANSFORM_LEFT_DIFF,
    TRANSFORM_RIGHT_DIFF,
];

/// Downstream run/zero topology of a reversible channel transform.
///
/// `RleAwareChannelTransform` proposes transforms by the degeneracy of their
/// component streams — zero density, zero runs, the longest zero run and equal
/// neighbour pairs — rather than by residual variance. The measured complete
/// bytes still decide; this only orders and explains the proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelTopology {
    /// Component samples that are exactly zero.
    pub zeros: u64,
    /// Maximal runs of zeros across all components.
    pub zero_runs: u64,
    /// Longest zero run across all components.
    pub longest_zero_run: u64,
    /// Adjacent equal component samples (`x[t] == x[t-1]`).
    pub equal_pairs: u64,
}

impl ChannelTopology {
    /// A deterministic degeneracy score (larger is more run-length friendly).
    pub const fn rle_score(self) -> u64 {
        self.zero_runs
            .saturating_mul(8)
            .saturating_add(self.zeros.saturating_mul(4))
            .saturating_add(self.longest_zero_run.saturating_mul(16))
            .saturating_add(self.equal_pairs)
    }
}

/// Forward reversible transform of one interleaved frame.
pub fn transform_frame(transform: u8, frame: &[i32], out: &mut [i32]) {
    match transform {
        TRANSFORM_MID_SIDE => {
            let l = frame[0];
            let s = frame[1].wrapping_sub(l);
            out[0] = l.wrapping_add(s >> 1);
            out[1] = s;
        }
        TRANSFORM_LEFT_DIFF => {
            let l = frame[0];
            out[0] = l;
            out[1] = frame[1].wrapping_sub(l);
        }
        TRANSFORM_RIGHT_DIFF => {
            let r = frame[1];
            out[0] = r;
            out[1] = frame[0].wrapping_sub(r);
        }
        _ => out.copy_from_slice(frame),
    }
}

/// Inverse reversible transform of one component frame.
pub fn inverse_transform_frame(transform: u8, comp: &[i32], out: &mut [i32]) {
    match transform {
        TRANSFORM_MID_SIDE => {
            let l = comp[0].wrapping_sub(comp[1] >> 1);
            out[0] = l;
            out[1] = l.wrapping_add(comp[1]);
        }
        TRANSFORM_LEFT_DIFF => {
            let l = comp[0];
            out[0] = l;
            out[1] = l.wrapping_add(comp[1]);
        }
        TRANSFORM_RIGHT_DIFF => {
            let r = comp[0];
            out[0] = r.wrapping_add(comp[1]);
            out[1] = r;
        }
        _ => out.copy_from_slice(comp),
    }
}

/// Measure the downstream run/zero topology of `transform` on an interleaved
/// multichannel signal, without materializing the transformed signal.
pub fn channel_topology(
    interleaved: &[i32],
    channels: u8,
    frames: usize,
    transform: u8,
) -> Result<ChannelTopology> {
    let c = usize::from(channels);
    if c == 0 || interleaved.len() != frames * c {
        return Err(Error::malformed("channel topology geometry mismatch"));
    }
    if transform != TRANSFORM_NONE && c != 2 {
        return Err(Error::malformed(
            "channel topology stereo transform requires two channels",
        ));
    }
    if transform > TRANSFORM_RIGHT_DIFF {
        return Err(Error::malformed("channel topology transform out of range"));
    }
    let mut topo = ChannelTopology {
        zeros: 0,
        zero_runs: 0,
        longest_zero_run: 0,
        equal_pairs: 0,
    };
    let mut in_zero = vec![false; c];
    let mut run_len = vec![0u64; c];
    let mut prev = vec![0i32; c];
    let mut seen = vec![false; c];
    let mut frame = vec![0i32; c];
    for t in 0..frames {
        transform_frame(transform, &interleaved[t * c..t * c + c], &mut frame);
        for cc in 0..c {
            let v = frame[cc];
            if v == 0 {
                topo.zeros += 1;
                if in_zero[cc] {
                    run_len[cc] += 1;
                } else {
                    in_zero[cc] = true;
                    run_len[cc] = 1;
                    topo.zero_runs += 1;
                }
                topo.longest_zero_run = topo.longest_zero_run.max(run_len[cc]);
            } else {
                in_zero[cc] = false;
                run_len[cc] = 0;
            }
            if seen[cc] && v == prev[cc] {
                topo.equal_pairs += 1;
            }
            prev[cc] = v;
            seen[cc] = true;
        }
    }
    Ok(topo)
}

/// Decoder work state: component history, component predictions, channel
/// hypothesis, reconstructed frame, and the forward-transformed frame.
type State = (Vec<Vec<i32>>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>);

/// A reversible channel transform plus per-component mono predictors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultichannelPredictor {
    pub channels: u8,
    /// `TRANSFORM_NONE` or `TRANSFORM_MID_SIDE`.
    pub transform: u8,
    /// One mono predictor per component (`channels` entries).
    pub predictors: Vec<SparseLinearPredictor>,
}

impl MultichannelPredictor {
    pub fn validate(&self) -> Result<()> {
        let c = usize::from(self.channels);
        if c == 0 || c > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed(
                "multichannel predictor channel count out of range",
            ));
        }
        if self.predictors.len() != c {
            return Err(Error::malformed(
                "multichannel predictor component count mismatch",
            ));
        }
        for p in &self.predictors {
            if p.channels != 1 {
                return Err(Error::malformed(
                    "multichannel component predictors must be mono",
                ));
            }
            p.validate()?;
        }
        match self.transform {
            TRANSFORM_NONE => {}
            TRANSFORM_MID_SIDE | TRANSFORM_LEFT_DIFF | TRANSFORM_RIGHT_DIFF => {
                if c != 2 {
                    return Err(Error::malformed(
                        "stereo channel transform requires exactly two channels",
                    ));
                }
            }
            other => {
                return Err(Error::malformed(format!(
                    "unknown channel transform {other}"
                )));
            }
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        self.predictors
            .iter()
            .map(|p| p.receptive_field())
            .max()
            .unwrap_or(0)
            .max(1)
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.predictors.iter().map(|p| p.ops_per_sample()).sum()
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        start
    }

    /// Forward reversible transform of one frame.
    fn forward_frame(&self, frame: &[i32], out: &mut [i32]) {
        transform_frame(self.transform, frame, out);
    }

    /// Inverse reversible transform of one frame.
    fn inverse_frame(&self, comp: &[i32], out: &mut [i32]) {
        inverse_transform_frame(self.transform, comp, out);
    }

    fn max_lag(&self) -> usize {
        self.predictors
            .iter()
            .map(|p| p.max_lag())
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// Predict each component from the component history (`chist[k][cc]` is
    /// component `cc` at frame `t-1-k`).
    fn predict_components(&self, chist: &[Vec<i32>], z: &mut [i32]) {
        let c = usize::from(self.channels);
        let mut hist = vec![0i32; self.max_lag()];
        for cc in 0..c {
            let p = &self.predictors[cc];
            let m = p.max_lag();
            for (k, slot) in hist.iter_mut().enumerate().take(m) {
                *slot = chist[k][cc];
            }
            let mut o = [0i32; 1];
            p.hypothesis(&hist[..m], &mut o);
            z[cc] = o[0];
        }
    }

    fn make_state(&self) -> State {
        let c = usize::from(self.channels);
        (
            vec![vec![0i32; c]; self.max_lag()],
            vec![0i32; c],
            vec![0i32; c],
            vec![0i32; c],
            vec![0i32; c],
        )
    }

    /// Hypothesis over a whole extent, open-loop from a source.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if source.len() != frames * c {
            return Err(Error::malformed("multichannel source length mismatch"));
        }
        let mut out = vec![0i32; frames * c];
        let (mut chist, mut z, mut hframe, _comp_frame, mut cframe) = self.make_state();
        for t in 0..frames {
            self.predict_components(&chist, &mut z);
            self.inverse_frame(&z, &mut hframe);
            out[t * c..t * c + c].copy_from_slice(&hframe);
            let frame = &source[t * c..t * c + c];
            self.forward_frame(frame, &mut cframe);
            chist.rotate_right(1);
            chist[0].copy_from_slice(&cframe);
        }
        Ok(out)
    }

    /// Reconstruct `[start, start+len)` exactly from a dense interleaved residual.
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
            return Err(Error::malformed("multichannel residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("multichannel range overflows"))?;
        if end > frames {
            return Err(Error::malformed("multichannel range exceeds the extent"));
        }
        let mut out = vec![0i32; len * c];
        if len == 0 {
            return Ok(out);
        }
        let (mut chist, mut z, mut hframe, mut x_hat, mut cframe) = self.make_state();
        for t in 0..frames {
            self.predict_components(&chist, &mut z);
            self.inverse_frame(&z, &mut hframe);
            for cc in 0..c {
                x_hat[cc] = sat_i32(Acc::from(hframe[cc]) + Acc::from(residual[t * c + cc]));
            }
            if t >= start && t < end {
                out[(t - start) * c..(t - start) * c + c].copy_from_slice(&x_hat);
            }
            self.forward_frame(&x_hat, &mut cframe);
            chist.rotate_right(1);
            chist[0].copy_from_slice(&cframe);
        }
        Ok(out)
    }

    /// Canonical bytes: `kind(7) || channels || transform || [u32 pred_len || pred]*`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(7);
        out.push(self.channels);
        out.push(self.transform);
        for p in &self.predictors {
            let pb = p.canonical_bytes();
            out.extend_from_slice(&(pb.len() as u32).to_le_bytes());
            out.extend_from_slice(&pb);
        }
        out
    }

    /// Parse canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<MultichannelPredictor> {
        if bytes.len() < 3 || bytes[0] != 7 {
            return Err(Error::malformed("multichannel model header mismatch"));
        }
        let channels = bytes[1];
        let transform = bytes[2];
        let c = usize::from(channels);
        if c == 0 || c > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed("multichannel channel count out of range"));
        }
        let mut at = 3usize;
        let mut predictors = Vec::with_capacity(c);
        for _ in 0..c {
            let len = u32::from_le_bytes(
                bytes
                    .get(at..at + 4)
                    .ok_or_else(|| Error::malformed("multichannel model is truncated"))?
                    .try_into()
                    .unwrap(),
            ) as usize;
            at += 4;
            let pb = bytes
                .get(at..at + len)
                .ok_or_else(|| Error::malformed("multichannel component is truncated"))?;
            at += len;
            predictors.push(SparseLinearPredictor::from_canonical_bytes(pb)?);
        }
        if at != bytes.len() {
            return Err(Error::malformed("multichannel model has trailing bytes"));
        }
        let p = MultichannelPredictor {
            channels,
            transform,
            predictors,
        };
        p.validate()?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::quantize_weight;
    use crate::learned::model::LearnedModel;
    use crate::learned::object::LearnedObject;
    use crate::learned::sparse::COEFF_DELTA_VARINT;

    fn mono_pred() -> SparseLinearPredictor {
        SparseLinearPredictor {
            channels: 1,
            lags: vec![1],
            weights: vec![quantize_weight(0.9)],
            bias: vec![0],
            block_frames: None,
            coeff_encoding: COEFF_DELTA_VARINT,
        }
    }

    fn model(transform: u8) -> MultichannelPredictor {
        MultichannelPredictor {
            channels: 2,
            transform,
            predictors: vec![mono_pred(), mono_pred()],
        }
    }

    fn stereo_source() -> Vec<i32> {
        let mut v = Vec::new();
        for t in 0..2000i64 {
            let l = ((t * 97) % 20011) as i32 - 10000;
            let r = l / 2 + ((t * 13) % 401) as i32 - 200;
            v.push(l);
            v.push(r);
        }
        v
    }

    #[test]
    fn stereo_lifting_is_exactly_reversible_over_the_full_i32_domain() {
        for transform in STEREO_TRANSFORMS {
            let m = model(transform);
            for (l, r) in [
                (i32::MIN, i32::MAX),
                (i32::MAX, i32::MIN),
                (0, 0),
                (-1, 1),
                (123456789, -987654321),
            ] {
                let frame = [l, r];
                let mut comp = [0i32; 2];
                let mut back = [0i32; 2];
                m.forward_frame(&frame, &mut comp);
                m.inverse_frame(&comp, &mut back);
                assert_eq!(back, frame, "transform {transform}");
            }
        }
    }

    #[test]
    fn multichannel_objects_close_exactly_for_every_stereo_transform() {
        let source = stereo_source();
        for transform in STEREO_TRANSFORMS {
            let m = model(transform);
            let o = LearnedObject::from_intrinsic_exp2(
                LearnedModel::Multichannel(m),
                2,
                2000,
                48_000,
                Vec::new(),
                &source,
            )
            .unwrap();
            assert!(o.verify(&source), "transform {transform}");
            assert_eq!(o.materialize_range(500, 40).unwrap(), &source[1000..1080]);
        }
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        for transform in STEREO_TRANSFORMS {
            let m = model(transform);
            let b = m.canonical_bytes();
            let back = MultichannelPredictor::from_canonical_bytes(&b).unwrap();
            assert_eq!(back, m);
        }
    }

    #[test]
    fn topology_rewards_the_transform_that_creates_zeros() {
        // R equals L except for sparse events: `R - L` is a long zero run, so
        // the difference transforms must out-score `NONE` on zero topology.
        let mut source = Vec::new();
        for t in 0..4096i64 {
            let l = ((t * 89) % 15013) as i32 - 7000;
            let r = if t % 512 == 7 { l + 1000 } else { l };
            source.push(l);
            source.push(r);
        }
        let none = channel_topology(&source, 2, 4096, TRANSFORM_NONE).unwrap();
        assert_eq!(none.zeros, 0);
        for transform in [
            TRANSFORM_MID_SIDE,
            TRANSFORM_LEFT_DIFF,
            TRANSFORM_RIGHT_DIFF,
        ] {
            let topo = channel_topology(&source, 2, 4096, transform).unwrap();
            assert!(topo.zeros > 0, "transform {transform}");
            assert!(
                topo.rle_score() > none.rle_score(),
                "transform {transform}: {} vs {}",
                topo.rle_score(),
                none.rle_score()
            );
        }
    }
}
