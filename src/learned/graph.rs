//! Bounded nonlinear finite-field vocabulary (`O.8`, `O.19`, `O.29`).
//!
//! A deliberately tiny integer graph: a causal tap window, one or two dense
//! layers with a frozen integer activation, and a per-channel output. It exists
//! only after the linear family has an evidence record (`O.62`), and it never
//! uses floating point, transcendental functions, or a generic tensor runtime.
//!
//! ```text
//! input(t) = X_hat[t-1 .. t-K]  flattened        (K · C values, oldest first)
//! hidden   = act(W1 · input + b1)                (Q12, i64 accumulators)
//! H(t)     = sat_i32(round_shift(W2 · hidden + b2, 12))
//! X_hat(t) = sat_i32(H(t) + R(t))
//! ```

use crate::error::{Error, Result};
use crate::learned::arithmetic::{Acc, Activation, round_shift_half_away, sat_i32};

/// One dense layer with Q12 weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLayer {
    pub in_dim: u32,
    pub out_dim: u32,
    /// `out_dim × in_dim` Q12 weights, row-major.
    pub weights: Vec<i16>,
    /// `out_dim` Q12 biases.
    pub bias: Vec<i32>,
    /// Activation applied to this layer's output.
    pub activation: Activation,
}

impl DenseLayer {
    fn validate(&self) -> Result<()> {
        let i = self.in_dim as usize;
        let o = self.out_dim as usize;
        if i == 0 || o == 0 {
            return Err(Error::malformed("learned dense layer has a zero dimension"));
        }
        if self.in_dim > crate::limits::MAX_LEARNED_TENSOR_ELEMENTS as u32
            || self.out_dim > crate::limits::MAX_LEARNED_TENSOR_ELEMENTS as u32
        {
            return Err(Error::limit(
                "learned dense layer dimension exceeds the bound",
            ));
        }
        let expect = i
            .checked_mul(o)
            .ok_or_else(|| Error::limit("learned dense layer size overflows"))?;
        if self.weights.len() != expect || self.bias.len() != o {
            return Err(Error::malformed("learned dense layer shape mismatch"));
        }
        let bytes = (expect as u64) * 2 + (o as u64) * 4;
        if bytes > crate::limits::MAX_LEARNED_WEIGHT_BYTES {
            return Err(Error::limit(
                "learned dense layer weight bytes exceed the bound",
            ));
        }
        self.activation.validate()
    }

    fn ops(&self) -> u64 {
        (self.in_dim as u64) * (self.out_dim as u64) + self.out_dim as u64
    }

    fn eval(&self, input: &[i64]) -> Vec<i64> {
        let i = self.in_dim as usize;
        let o = self.out_dim as usize;
        debug_assert_eq!(input.len(), i);
        let mut out = Vec::with_capacity(o);
        for row in 0..o {
            let mut acc: Acc = Acc::from(self.bias[row]);
            for (col, &x) in input.iter().enumerate() {
                acc += Acc::from(self.weights[row * i + col]) * x;
            }
            // Hidden layers keep Q12; the final layer is shifted by the caller.
            let activated = self.activation.apply(acc);
            let scaled = round_shift_half_away(activated, crate::limits::LEARNED_WEIGHT_Q);
            out.push(scaled);
        }
        out
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.in_dim.to_le_bytes());
        out.extend_from_slice(&self.out_dim.to_le_bytes());
        for w in &self.weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for b in &self.bias {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out.extend_from_slice(&self.activation.canonical_bytes());
        out
    }
}

/// A bounded nonlinear finite-field predictor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonlinearGraph {
    pub channels: u8,
    pub taps: u16,
    pub layers: Vec<DenseLayer>,
    /// `Some(B)` selects block-local reset semantics.
    pub block_frames: Option<u32>,
}

impl NonlinearGraph {
    pub fn validate(&self) -> Result<()> {
        let c = usize::from(self.channels);
        if c == 0 || c > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed("learned graph channel count out of range"));
        }
        if self.taps == 0 || u32::from(self.taps) > crate::limits::MAX_LEARNED_TAPS {
            return Err(Error::limit("learned graph tap count exceeds the bound"));
        }
        if self.layers.is_empty()
            || self.layers.len() as u32 > crate::limits::MAX_LEARNED_GRAPH_DEPTH
        {
            return Err(Error::limit("learned graph depth exceeds the bound"));
        }
        let expect_in = u32::from(self.taps) * self.channels as u32;
        if self.layers[0].in_dim != expect_in {
            return Err(Error::malformed("learned graph input dimension mismatch"));
        }
        for w in self.layers.windows(2) {
            if w[0].out_dim != w[1].in_dim {
                return Err(Error::malformed("learned graph layer shapes do not chain"));
            }
        }
        if self.layers[self.layers.len() - 1].out_dim != self.channels as u32 {
            return Err(Error::malformed("learned graph output dimension mismatch"));
        }
        for l in &self.layers {
            l.validate()?;
        }
        if let Some(b) = self.block_frames
            && (b == 0 || b > crate::limits::MAX_LEARNED_BLOCK_FRAMES)
        {
            return Err(Error::limit("learned graph block size exceeds the bound"));
        }
        let ops = self.ops_per_sample();
        if ops > crate::limits::MAX_LEARNED_OPS_PER_SAMPLE {
            return Err(Error::limit(
                "learned graph operations per sample exceed the bound",
            ));
        }
        Ok(())
    }

    pub fn receptive_field(&self) -> u64 {
        u64::from(self.taps)
    }

    pub fn ops_per_sample(&self) -> u64 {
        self.layers.iter().map(DenseLayer::ops).sum()
    }

    pub fn state_bytes(&self) -> u64 {
        0
    }

    pub fn checkpoint_count(&self) -> u32 {
        0
    }

    fn tap_count(&self) -> usize {
        usize::from(self.taps)
    }

    fn run_layer_chain(&self, input: &[i64]) -> Vec<i64> {
        let mut v = input.to_vec();
        for l in &self.layers {
            v = l.eval(&v);
        }
        v
    }

    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if frames * c != residual.len() {
            return Err(Error::malformed("learned residual length mismatch"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("learned range overflows"))?;
        if end > frames {
            return Err(Error::malformed("learned range exceeds the extent"));
        }
        let mut out = vec![0i32; len * c];
        if len == 0 {
            return Ok(out);
        }
        let mut history = vec![0i32; self.tap_count() * c];
        let mut input = vec![0i64; self.tap_count() * c];
        let ranges: Vec<(usize, usize)> = match self.block_frames {
            None => vec![(0, end)],
            Some(b) => {
                let b = b as usize;
                let first = start / b;
                let last = (end - 1) / b;
                (first..=last)
                    .map(|blk| (blk * b, ((blk + 1) * b).min(frames)))
                    .collect()
            }
        };
        for (from, to) in ranges {
            history.fill(0);
            for t in from..to {
                for (i, &h) in history.iter().enumerate() {
                    input[i] = i64::from(h);
                }
                let h_out = self.run_layer_chain(&input);
                if self.tap_count() > 1 {
                    history.copy_within(c.., 0);
                }
                for i in 0..c {
                    let x = sat_i32(Acc::from(h_out[i]) + Acc::from(residual[t * c + i]));
                    if t >= start && t < end {
                        out[(t - start) * c + i] = x;
                    }
                    history[(self.tap_count() - 1) * c + i] = x;
                }
            }
        }
        Ok(out)
    }

    pub fn hypothesis_all_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if source.len() != frames * c {
            return Err(Error::malformed("learned source length mismatch"));
        }
        let mut h_all = vec![0i32; frames * c];
        let mut history = vec![0i32; self.tap_count() * c];
        let mut input = vec![0i64; self.tap_count() * c];
        let ranges: Vec<(usize, usize)> = match self.block_frames {
            None => vec![(0, frames)],
            Some(b) => {
                let b = b as usize;
                (0..frames)
                    .step_by(b)
                    .map(|blk| (blk, (blk + b).min(frames)))
                    .collect()
            }
        };
        for (from, to) in ranges {
            history.fill(0);
            for t in from..to {
                for (i, &h) in history.iter().enumerate() {
                    input[i] = i64::from(h);
                }
                let h_out = self.run_layer_chain(&input);
                if self.tap_count() > 1 {
                    history.copy_within(c.., 0);
                }
                for i in 0..c {
                    h_all[t * c + i] = h_out[i] as i32;
                    history[(self.tap_count() - 1) * c + i] = source[t * c + i];
                }
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

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(1); // kind 1 = nonlinear finite-field
        out.push(self.channels);
        out.extend_from_slice(&self.taps.to_le_bytes());
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        match self.block_frames {
            None => out.extend_from_slice(&0u32.to_le_bytes()),
            Some(b) => out.extend_from_slice(&b.to_le_bytes()),
        }
        for l in &self.layers {
            let lb = l.canonical_bytes();
            out.extend_from_slice(&(lb.len() as u32).to_le_bytes());
            out.extend_from_slice(&lb);
        }
        out
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<NonlinearGraph> {
        let mut r = crate::learned::serialization::Reader::new(bytes);
        let kind = r.u8()?;
        if kind != 1 {
            return Err(Error::malformed("learned graph kind mismatch"));
        }
        let channels = r.u8()?;
        let taps = r.u16()?;
        let layer_count = r.u32()? as usize;
        if layer_count == 0 || layer_count > crate::limits::MAX_LEARNED_GRAPH_DEPTH as usize {
            return Err(Error::limit("learned graph layer count exceeds the bound"));
        }
        let block = r.u32()?;
        let mut layers = Vec::with_capacity(layer_count.min(64));
        for _ in 0..layer_count {
            let lb_len = r.u32()? as usize;
            if lb_len as u64 > crate::limits::MAX_LEARNED_WEIGHT_BYTES {
                return Err(Error::limit("learned layer bytes exceed the bound"));
            }
            let lb = r.take(lb_len)?;
            layers.push(parse_dense_layer(lb)?);
        }
        r.finish()?;
        let g = NonlinearGraph {
            channels,
            taps,
            layers,
            block_frames: if block == 0 { None } else { Some(block) },
        };
        g.validate()?;
        Ok(g)
    }
}

fn parse_dense_layer(bytes: &[u8]) -> Result<DenseLayer> {
    let mut r = crate::learned::serialization::Reader::new(bytes);
    let in_dim = r.u32()?;
    let out_dim = r.u32()?;
    let w_count = (in_dim as usize)
        .checked_mul(out_dim as usize)
        .ok_or_else(|| Error::limit("learned layer size overflows"))?;
    let mut weights = Vec::with_capacity(w_count.min(1 << 20));
    for _ in 0..w_count {
        weights.push(r.i16()?);
    }
    let mut bias = Vec::with_capacity((out_dim as usize).min(1 << 20));
    for _ in 0..out_dim {
        bias.push(r.i32()?);
    }
    let activation = Activation::from_canonical_bytes(&mut r)?;
    r.finish()?;
    Ok(DenseLayer {
        in_dim,
        out_dim,
        weights,
        bias,
        activation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::{quantize_bias, quantize_weight};

    /// A 1-hidden-layer graph whose hidden activation is identity and whose
    /// second layer is a passthrough of the first tap, so it must reproduce a
    /// one-tap linear predictor exactly.
    fn passthrough_graph(block: Option<u32>) -> NonlinearGraph {
        let c = 1u32;
        let k = 1u32;
        // hidden: identity over the input, dimension K*C = 1
        let hidden = DenseLayer {
            in_dim: k * c,
            out_dim: 1,
            weights: vec![quantize_weight(1.0)],
            bias: vec![quantize_bias(0.0)],
            activation: Activation::Identity,
        };
        // output: identity
        let out = DenseLayer {
            in_dim: 1,
            out_dim: c,
            weights: vec![quantize_weight(1.0)],
            bias: vec![quantize_bias(0.0)],
            activation: Activation::Identity,
        };
        NonlinearGraph {
            channels: 1,
            taps: 1,
            layers: vec![hidden, out],
            block_frames: block,
        }
    }

    #[test]
    fn identity_graph_reproduces_a_one_tap_predictor() {
        let g = passthrough_graph(None);
        g.validate().unwrap();
        let x: Vec<i32> = (0..80).map(|i| (i * 977) ^ -1234).collect();
        let h = g.hypothesis_all_from_source(&x, 80).unwrap();
        let residual: Vec<i32> = (0..80).map(|t| x[t] - h[t]).collect();
        let recon = g.evaluate_range(&residual, 80, 0, 80).unwrap();
        assert_eq!(recon, x);
        // Chunked == contiguous.
        assert_eq!(g.evaluate_range(&residual, 80, 33, 20).unwrap(), &x[33..53]);
    }

    #[test]
    fn block_local_graph_is_independent_per_block() {
        let g = passthrough_graph(Some(32));
        let x: Vec<i32> = (0..128).map(|i| (i * 31) - 1000).collect();
        let h = g.hypothesis_all_from_source(&x, 128).unwrap();
        let residual: Vec<i32> = (0..128).map(|t| x[t] - h[t]).collect();
        assert_eq!(g.evaluate_range(&residual, 128, 0, 128).unwrap(), x);
        assert_eq!(g.replay_frames(70), 6);
    }

    #[test]
    fn canonical_round_trip_and_hostile_bounds() {
        let g = passthrough_graph(Some(16));
        let bytes = g.canonical_bytes();
        assert_eq!(NonlinearGraph::from_canonical_bytes(&bytes).unwrap(), g);
        assert!(NonlinearGraph::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut bomb = bytes.clone();
        // layer count near u32::MAX
        bomb[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(NonlinearGraph::from_canonical_bytes(&bomb).is_err());
    }

    #[test]
    fn a_graph_with_broken_shapes_is_rejected() {
        let mut g = passthrough_graph(None);
        g.layers[1].in_dim = 7;
        assert!(g.validate().is_err());
    }
}
