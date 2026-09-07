//! Shared evaluator vocabulary: execution surfaces and transform classes.
//!
//! A transform's class may differ per backend; courts record the class that
//! actually ran. The classification exists so that no recursive/stateful DSP
//! feature silently forces the whole architecture into a conventional global
//! PCM block (paper §"DSP").

/// Execution surface producing an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Surface {
    /// Scalar reference (semantic authority).
    ScalarReference,
    /// Runtime-dispatched SIMD host backend.
    Simd,
    /// CUDA device.
    Cuda,
    /// ROCm/HIP device.
    Rocm,
    /// Host CPU executing the exact scalar semantics (alias used by courts).
    Cpu,
}

impl Surface {
    pub const fn label(self) -> &'static str {
        match self {
            Surface::ScalarReference => "scalar",
            Surface::Simd => "simd",
            Surface::Cuda => "cuda",
            Surface::Rocm => "rocm",
            Surface::Cpu => "cpu",
        }
    }
}

/// How a transform fits the direct/fused path on a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformClass {
    /// Can be fused into the stateless direct observation path.
    DirectFusable,
    /// Recursive/stateful; requires explicit per-voice state and cannot sit on
    /// the stateless fused path as-is.
    DirectStateful,
    /// Only available as a buffered transform.
    BufferedOnly,
    /// Not supported on this backend.
    Unsupported,
}

/// Identities of the narrow transform set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformId {
    NearestInterp,
    LinearInterp,
    PolyphaseResampler,
    Envelope,
    GainPan,
    Mix,
    Biquad,
    OnePole,
}

impl TransformId {
    pub const fn label(self) -> &'static str {
        match self {
            TransformId::NearestInterp => "nearest_interp",
            TransformId::LinearInterp => "linear_interp",
            TransformId::PolyphaseResampler => "polyphase_resampler",
            TransformId::Envelope => "envelope",
            TransformId::GainPan => "gain_pan",
            TransformId::Mix => "mix",
            TransformId::Biquad => "biquad",
            TransformId::OnePole => "one_pole",
        }
    }
}

/// Default classification of the transform set on a fused-capable backend
/// (scalar/SIMD/GPU). Stateful transforms carry their state explicitly.
pub fn default_class(t: TransformId) -> TransformClass {
    match t {
        TransformId::NearestInterp
        | TransformId::LinearInterp
        | TransformId::PolyphaseResampler
        | TransformId::Envelope
        | TransformId::GainPan
        | TransformId::Mix => TransformClass::DirectFusable,
        TransformId::Biquad | TransformId::OnePole => TransformClass::DirectStateful,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_stable() {
        assert_eq!(
            default_class(TransformId::Mix),
            TransformClass::DirectFusable
        );
        assert_eq!(
            default_class(TransformId::Biquad),
            TransformClass::DirectStateful
        );
        assert_eq!(Surface::ScalarReference.label(), "scalar");
    }
}
