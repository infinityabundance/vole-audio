//! Quantization compiler (`O.22`, `O.23`).
//!
//! Training produces floating-point parameters; the **canonical object is
//! integer**. This module is the one place that converts a fitted real
//! parameter vector into the frozen fixed-point form, and it is deliberately
//! explicit about saturation and rounding so a trained model and its archived
//! form are never confused.

use crate::learned::arithmetic::{quantize_bias, quantize_weight};
use crate::learned::finite_field::LinearPredictor;

/// Compile real-valued FIR weights and biases into a canonical predictor.
///
/// `weights` is indexed `((k-1)·C + out)·C + in` (matching the evaluator);
/// `bias` has `C` entries. Any value outside the Q12 `i16`/`i32` domain is
/// saturated by the quantizer, never silently wrapped.
pub fn compile_linear(
    channels: u8,
    taps: u16,
    block_frames: Option<u32>,
    weights: &[f64],
    bias: &[f64],
) -> LinearPredictor {
    LinearPredictor {
        channels,
        taps,
        weights: weights.iter().map(|&w| quantize_weight(w)).collect(),
        bias: bias.iter().map(|&b| quantize_bias(b)).collect(),
        block_frames,
    }
}

/// The real-valued parameters a quantized predictor came from (for diagnostics
/// and quantization-aware training).
pub fn dequantize(p: &LinearPredictor) -> (Vec<f64>, Vec<f64>) {
    (
        p.weights
            .iter()
            .map(|&w| crate::learned::arithmetic::dequantize_weight(w))
            .collect(),
        p.bias
            .iter()
            .map(|&b| crate::learned::arithmetic::dequantize_bias(b))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compilation_is_exact_and_saturating() {
        let p = compile_linear(1, 2, None, &[1.0, -1.0], &[0.5]);
        assert_eq!(p.weights, vec![4096, -4096]);
        assert_eq!(p.bias, vec![2048]);
        // Out-of-domain values saturate rather than wrap.
        let big = compile_linear(1, 1, None, &[1e9], &[1e9]);
        assert_eq!(big.weights, vec![i16::MAX]);
        assert_eq!(big.bias, vec![i32::MAX]);
        let (w, b) = dequantize(&p);
        assert!((w[0] - 1.0).abs() < 1e-9);
        assert!((b[0] - 0.5).abs() < 1e-9);
    }
}
