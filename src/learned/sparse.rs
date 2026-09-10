//! Sparse / high-order learned linear prediction (Exp2, priority `3`).
//!
//! The dense Exp1 FIR stores `K·C·C` consecutive coefficients. High-order
//! prediction is better served by *selected* lags: only the taps that pay for
//! themselves are stored (Ghido & Tabús, *Sparse Modeling for Lossless Audio
//! Compression*). This family stores an ascending lag set and the coefficients
//! for those lags only, with a compact canonical coefficient syntax.
//!
//! Model kind tag `5`. The family is mono-first (`channels == 1`); multichannel
//! structure is handled by the dedicated multichannel family.
//!
//! ```text
//! H[t] = bias + Σ_i w_i · X_hat[t - lag_i]   (Q12)
//! X_hat[t] = sat_i32(H[t] + R[t])
//! ```

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, round_shift_half_away, sat_i32};

/// How the coefficient sequence is canonically encoded.
pub const COEFF_RAW_I16: u8 = 0;
/// Delta + zigzag varint coefficient encoding.
pub const COEFF_DELTA_VARINT: u8 = 1;

/// A mono causal predictor over a selected set of lags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseLinearPredictor {
    /// Channel count (this family is mono-only: always `1`).
    pub channels: u8,
    /// Selected lags `>= 1`, strictly ascending and distinct.
    pub lags: Vec<u16>,
    /// `K` Q12 weights in lag order.
    pub weights: Vec<i16>,
    /// `1` Q12 bias.
    pub bias: Vec<i32>,
    /// Optional block-local reset size.
    pub block_frames: Option<u32>,
    /// Canonical coefficient encoding (`COEFF_RAW_I16` or `COEFF_DELTA_VARINT`).
    pub coeff_encoding: u8,
}

impl SparseLinearPredictor {
    /// Validate dimensions, ordering, bounds and the accumulator proof.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::malformed(
                "sparse linear predictor is mono-only in this build",
            ));
        }
        let k = self.lags.len();
        if k == 0 || k as u32 > crate::limits::MAX_LEARNED_SPARSE_LAGS {
            return Err(Error::limit("sparse lag count out of range"));
        }
        if self.weights.len() != k || self.bias.len() != 1 {
            return Err(Error::malformed("sparse coefficient count mismatch"));
        }
        let mut prev = 0u16;
        for &l in &self.lags {
            if l == 0 || l <= prev {
                return Err(Error::malformed(
                    "sparse lags must be positive and strictly ascending",
                ));
            }
            prev = l;
        }
        if u32::from(*self.lags.last().unwrap_or(&1)) > crate::limits::MAX_LEARNED_TAPS {
            return Err(Error::limit("sparse lag exceeds the bound"));
        }
        // The accumulator reads the same number of terms as taps.
        if !crate::learned::arithmetic::accumulator_is_safe(k as u32) {
            return Err(Error::limit("sparse accumulator bound exceeded"));
        }
        if self.coeff_encoding > COEFF_DELTA_VARINT {
            return Err(Error::malformed("unknown sparse coefficient encoding"));
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("sparse block size exceeds the bound"));
        }
        let bytes = (self.weights.len() as u64) * 2
            + (self.lags.len() as u64) * 2
            + (self.bias.len() as u64) * 4;
        if bytes > crate::limits::MAX_LEARNED_WEIGHT_BYTES {
            return Err(Error::limit("sparse weight bytes exceed the bound"));
        }
        Ok(())
    }

    /// Maximum lag (ring-buffer length).
    pub fn max_lag(&self) -> usize {
        usize::from(self.lags.last().copied().unwrap_or(1))
    }

    pub fn receptive_field(&self) -> u64 {
        self.max_lag() as u64
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.lags.len() as u64 + 1
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    /// One hypothesis evaluation. `history[j]` is `X_hat[t-1-j]`.
    #[inline]
    pub fn hypothesis(&self, history: &[i32], out: &mut [i32]) {
        let mut acc = Acc::from(self.bias[0]);
        for (i, &lag) in self.lags.iter().enumerate() {
            acc += Acc::from(self.weights[i]) * Acc::from(history[usize::from(lag) - 1]);
        }
        out[0] = sat_i32(round_shift_half_away(acc, crate::limits::LEARNED_WEIGHT_Q));
    }

    fn ranges(&self, frames: usize) -> Vec<(usize, usize)> {
        match self.block_frames {
            None => vec![(0, frames)],
            Some(b) => {
                let b = b as usize;
                (0..frames)
                    .step_by(b)
                    .map(|blk| (blk, (blk + b).min(frames)))
                    .collect()
            }
        }
    }

    /// Reconstruct `[start, start+len)` exactly from a dense residual.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        if residual.len() != frames {
            return Err(Error::malformed("sparse residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("sparse range overflows"))?;
        if end > frames {
            return Err(Error::malformed("sparse range exceeds the extent"));
        }
        let mut out = vec![0i32; len];
        if len == 0 {
            return Ok(out);
        }
        let m = self.max_lag();
        let mut history = vec![0i32; m];
        let mut h = [0i32; 1];
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            for t in from..to {
                self.hypothesis(&history, &mut h);
                let x = sat_i32(Acc::from(h[0]) + Acc::from(residual[t]));
                if t >= start && t < end {
                    out[t - start] = x;
                }
                if m > 1 {
                    history.copy_within(0..m - 1, 1);
                }
                history[0] = x;
            }
        }
        Ok(out)
    }

    /// Hypothesis over a whole extent, open-loop from a source.
    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        if source.len() != frames {
            return Err(Error::malformed("sparse source length mismatch"));
        }
        let mut h_all = vec![0i32; frames];
        let m = self.max_lag();
        let mut history = vec![0i32; m];
        let mut h = [0i32; 1];
        for (from, to) in self.ranges(frames) {
            history.fill(0);
            for t in from..to {
                self.hypothesis(&history, &mut h);
                h_all[t] = h[0];
                if m > 1 {
                    history.copy_within(0..m - 1, 1);
                }
                history[0] = source[t];
            }
        }
        Ok(h_all)
    }

    pub fn replay_frames(&self, start: usize) -> usize {
        match self.block_frames {
            None => start,
            Some(b) => start % (b as usize),
        }
    }

    /// Canonical model bytes:
    /// `kind(5) || channels || count(u16) || block(u32) || coeff_enc || lag_payload
    ///  || coeff_payload`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(5);
        out.push(self.channels);
        out.extend_from_slice(&(self.lags.len() as u16).to_le_bytes());
        out.extend_from_slice(&self.block_frames.unwrap_or(0).to_le_bytes());
        out.push(self.coeff_encoding);
        // Lag payload: first lag, then ascending deltas (varint).
        put_uvarint(&mut out, u64::from(self.lags[0]));
        for i in 1..self.lags.len() {
            put_uvarint(&mut out, u64::from(self.lags[i] - self.lags[i - 1]));
        }
        // Coefficient payload.
        match self.coeff_encoding {
            COEFF_DELTA_VARINT => {
                let mut prev = 0i64;
                for &w in &self.weights {
                    let d = i64::from(w) - prev;
                    prev = i64::from(w);
                    put_uvarint(&mut out, zigzag(d));
                }
            }
            _ => {
                for &w in &self.weights {
                    out.extend_from_slice(&w.to_le_bytes());
                }
            }
        }
        for b in &self.bias {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    /// Parse canonical model bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<SparseLinearPredictor> {
        if bytes.len() < 10 || bytes[0] != 5 {
            return Err(Error::malformed("sparse linear model header mismatch"));
        }
        let channels = bytes[1];
        let count = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
        if count == 0 || count as u32 > crate::limits::MAX_LEARNED_SPARSE_LAGS {
            return Err(Error::limit("sparse lag count out of range"));
        }
        let block = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let coeff_encoding = bytes[8];
        if coeff_encoding > COEFF_DELTA_VARINT {
            return Err(Error::malformed("unknown sparse coefficient encoding"));
        }
        let mut r = SliceReader::new(&bytes[9..]);
        let mut lags = Vec::with_capacity(count);
        let first = r.uvarint()?;
        if first == 0 {
            return Err(Error::malformed("sparse lag must be positive"));
        }
        lags.push(u16::try_from(first).map_err(|_| Error::limit("sparse lag overflows"))?);
        for _ in 1..count {
            let d = r.uvarint()?;
            let next = u64::from(*lags.last().unwrap())
                .checked_add(d)
                .ok_or_else(|| Error::limit("sparse lag overflows"))?;
            lags.push(u16::try_from(next).map_err(|_| Error::limit("sparse lag overflows"))?);
        }
        // Re-serialize the lag payload length so the coefficient payload
        // boundary is exact.
        let lag_len = {
            let mut probe = Vec::new();
            put_uvarint(&mut probe, u64::from(lags[0]));
            for i in 1..lags.len() {
                put_uvarint(&mut probe, u64::from(lags[i] - lags[i - 1]));
            }
            probe.len()
        };
        let mut cr = SliceReader::new(&bytes[9 + lag_len..]);
        let mut weights = Vec::with_capacity(count);
        if coeff_encoding == COEFF_DELTA_VARINT {
            let mut prev = 0i64;
            for _ in 0..count {
                let d = unzigzag(cr.uvarint()?);
                let w = prev
                    .checked_add(d)
                    .ok_or_else(|| Error::malformed("sparse coefficient overflows"))?;
                if !(i64::from(i16::MIN)..=i64::from(i16::MAX)).contains(&w) {
                    return Err(Error::malformed("sparse coefficient out of i16 domain"));
                }
                prev = w;
                weights.push(w as i16);
            }
        } else {
            for _ in 0..count {
                let b = cr.take(2)?;
                weights.push(i16::from_le_bytes(b.try_into().unwrap()));
            }
        }
        let bias = i32::from_le_bytes(cr.take(4)?.try_into().unwrap());
        cr.finish()?;
        let p = SparseLinearPredictor {
            channels,
            lags,
            weights,
            bias: vec![bias],
            block_frames: if block == 0 { None } else { Some(block) },
            coeff_encoding,
        };
        p.validate()?;
        Ok(p)
    }
}

// --- local varint/zigzag helpers (kept private to this module) ---

fn put_uvarint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

struct SliceReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SliceReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        SliceReader { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("sparse read overflows"))?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::malformed("sparse model is truncated"))?;
        self.pos = end;
        Ok(s)
    }
    fn uvarint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0u32;
        loop {
            let b = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| Error::malformed("sparse varint is truncated"))?;
            self.pos += 1;
            if shift >= 64 {
                return Err(Error::malformed("sparse varint overflows"));
            }
            v |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
        }
    }
    fn finish(&self) -> Result<()> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::malformed("sparse model has trailing bytes"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::quantize_weight;
    use crate::learned::model::LearnedModel;
    use crate::learned::object::LearnedObject;

    fn ar2_predictor() -> SparseLinearPredictor {
        SparseLinearPredictor {
            channels: 1,
            lags: vec![1, 2],
            weights: vec![quantize_weight(0.5), quantize_weight(0.25)],
            bias: vec![0],
            block_frames: None,
            coeff_encoding: COEFF_DELTA_VARINT,
        }
    }

    #[test]
    fn sparse_predictor_closes_exactly_and_matches_dense() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut x = vec![0i32, 0];
        for t in 2..2048 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let drive = (s >> 40) as i64 % 2001 - 1000;
            let v = i64::from(x[t - 1]) / 2 + i64::from(x[t - 2]) / 4 + drive;
            x.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
        let p = ar2_predictor();
        let o = LearnedObject::from_intrinsic_exp2(
            LearnedModel::SparseLinear(p.clone()),
            1,
            2048,
            48_000,
            Vec::new(),
            &x,
        )
        .unwrap();
        assert!(o.verify(&x));
        assert_eq!(o.materialize_range(1000, 50).unwrap(), &x[1000..1050]);
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        for enc in [COEFF_RAW_I16, COEFF_DELTA_VARINT] {
            let mut p = ar2_predictor();
            p.coeff_encoding = enc;
            p.lags = vec![1, 2, 5, 9, 100, 1000];
            p.weights = vec![100, -200, 300, -400, 500, -600];
            let b = p.canonical_bytes();
            let back = SparseLinearPredictor::from_canonical_bytes(&b).unwrap();
            assert_eq!(back, p);
        }
    }

    #[test]
    fn validation_rejects_bad_lag_sets() {
        let mut p = ar2_predictor();
        p.lags = vec![2, 1];
        assert!(p.validate().is_err());
        p.lags = vec![1, 1];
        assert!(p.validate().is_err());
        p.lags = vec![1, 2];
        p.channels = 2;
        assert!(p.validate().is_err());
    }
}
