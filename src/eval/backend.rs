//! Backend selection and runtime dispatch (Phase F).
//!
//! The u1 semantics are identical on every execution surface; backends differ
//! only in *how* they compute the frozen per-(voice, frame) contribution.
//! Selection is therefore never semantic — the same observation must result
//! from any exact backend (enforced by differential courts and parity tests).
//!
//! Vocabulary (evidence-facing labels are frozen):
//!
//! ```text
//! Backend::Scalar  -> concrete Scalar        (semantic authority; scalar oracle)
//! Backend::Simd    -> concrete Simd{isa}     (planned engine, vector kernel)
//! Backend::Auto    -> Scalar on non-x86-64, otherwise the best detected ISA
//! ```
//!
//! ISA detection is a *runtime probe* (`is_x86_feature_detected`), never a
//! compile-time assumption: this crate builds for a generic x86-64 baseline
//! and dispatch happens per process. Non-x86-64 hosts (e.g. aarch64) get the
//! exact scalar floor until a NEON kernel exists (recorded, not silently
//! claimed).

use crate::eval::common::Surface;

/// User-visible backend selection (CLI/evidence vocabulary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Scalar reference evaluator (semantic authority).
    Scalar,
    /// Runtime-dispatched host SIMD engine.
    Simd,
    /// Best exact backend for this host (currently CPU-only: SIMD where the
    /// ISA exists, else scalar; CUDA/ROCm join when those phases land).
    Auto,
}

impl Backend {
    pub const fn label(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Simd => "simd",
            Backend::Auto => "auto",
        }
    }

    /// Parse a CLI/evidence backend name. Unknown names are rejected (the CLI
    /// never implies support that is absent).
    pub fn parse(s: &str) -> Option<Backend> {
        match s {
            "scalar" => Some(Backend::Scalar),
            "simd" => Some(Backend::Simd),
            "auto" => Some(Backend::Auto),
            _ => None,
        }
    }
}

/// Concrete instruction-set floor for the host SIMD engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Exact scalar semantics (reference floor; used when no vector ISA is
    /// available and as the differential anchor for kernel tests).
    Scalar,
    /// 256-bit x86-64 kernel (4× i64 lanes; i64 multiply emulated — AVX2 has
    /// no native 64-bit vector multiply).
    Avx2,
    /// 512-bit x86-64 kernel (8× i64 lanes; requires AVX-512F + DQ + VL for
    /// `vpmullq`).
    Avx512,
}

impl Isa {
    pub const fn label(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
        }
    }
}

/// Resolve a requested backend to a concrete execution surface for this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConcreteBackend {
    Scalar,
    Simd { isa: Isa },
}

impl ConcreteBackend {
    pub fn label(self) -> String {
        match self {
            ConcreteBackend::Scalar => "scalar".into(),
            ConcreteBackend::Simd { isa } => format!("simd/{}", isa.label()),
        }
    }

    pub fn surface(self) -> Surface {
        match self {
            ConcreteBackend::Scalar => Surface::ScalarReference,
            ConcreteBackend::Simd { .. } => Surface::Simd,
        }
    }
}

/// Detect the best ISA available on this host (runtime probe).
pub fn detect_isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
        {
            return Isa::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return Isa::Avx2;
        }
        Isa::Scalar
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // NEON and other floors arrive with their kernels; until then the
        // exact scalar floor runs everywhere (recorded, never claimed).
        Isa::Scalar
    }
}

/// Resolve a `Backend` to its concrete execution surface on this host.
pub fn resolve(requested: Backend) -> ConcreteBackend {
    match requested {
        Backend::Scalar => ConcreteBackend::Scalar,
        Backend::Simd => ConcreteBackend::Simd { isa: detect_isa() },
        Backend::Auto => match detect_isa() {
            Isa::Scalar => ConcreteBackend::Scalar,
            isa => ConcreteBackend::Simd { isa },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_frozen() {
        assert_eq!(Backend::Scalar.label(), "scalar");
        assert_eq!(Backend::Simd.label(), "simd");
        assert_eq!(Backend::Auto.label(), "auto");
        assert_eq!(Isa::Avx512.label(), "avx512");
        assert_eq!(Isa::Avx2.label(), "avx2");
    }

    #[test]
    fn parse_rejects_unknown_and_accepts_frozen_names() {
        assert_eq!(Backend::parse("scalar"), Some(Backend::Scalar));
        assert_eq!(Backend::parse("simd"), Some(Backend::Simd));
        assert_eq!(Backend::parse("auto"), Some(Backend::Auto));
        assert_eq!(Backend::parse("cuda"), None); // not yet a CPU surface
        assert_eq!(Backend::parse("SIMD"), None); // case-sensitive frozen names
    }

    #[test]
    fn concrete_surface_maps_to_common_vocabulary() {
        assert_eq!(ConcreteBackend::Scalar.surface(), Surface::ScalarReference);
        assert_eq!(
            ConcreteBackend::Simd { isa: Isa::Avx512 }.surface(),
            Surface::Simd
        );
        assert_eq!(
            ConcreteBackend::Simd { isa: Isa::Scalar }.label(),
            "simd/scalar"
        );
    }

    #[test]
    fn detection_never_panics_and_is_consistent() {
        let isa = detect_isa();
        // On x86-64 with any vector support the ISA must be one of the three
        // frozen floors; the probe itself must be callable repeatedly.
        assert!(matches!(isa, Isa::Scalar | Isa::Avx2 | Isa::Avx512));
        assert_eq!(detect_isa(), isa);
    }
}
