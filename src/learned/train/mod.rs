//! Training and fitting (`O.20`, `O.21`, `O.25`, `O.43`, `O.58`).
//!
//! Training is **disposable and non-normative**. The canonical object format
//! never depends on a trainer, optimizer, floating-point library, automatic
//! differentiation engine, random order, GPU vendor, or framework. Only the
//! compiled integer hypothesis participates in exact closure and selection.
//!
//! No generic autodiff framework is implemented (`O.20`): specialized fitting
//! (ridge/least squares, coordinate descent, bounded search, hand-written
//! passes) is used instead, and only for the vocabulary that already has courts.

pub mod adaptive;
pub mod context_mixture;
pub mod finite_field;
pub mod hierarchy;
pub mod linear;
pub mod ltp;
pub mod multichannel;
pub mod objective;
pub mod optimizer;
pub mod optimizer2;
pub mod quant_aware;
pub mod sparse;

/// Training cost, kept separate from playback cost but never hidden (`O.41`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainStats {
    /// Candidate models attempted.
    pub candidates: u64,
    /// Candidates rejected by a court or by non-closure.
    pub rejected: u64,
    /// Optimizer iterations actually executed.
    pub iterations: u64,
    /// Quantization compilations attempted.
    pub quantization_attempts: u64,
    /// Wall time spent fitting (ns).
    pub fit_ns: u64,
    /// GPU time spent fitting (ns); zero when training is CPU-only.
    pub gpu_ns: u64,
    /// Peak host RAM observed during fitting (bytes, best effort).
    pub peak_ram_bytes: u64,
    /// Peak device VRAM observed during fitting (bytes, best effort).
    pub peak_vram_bytes: u64,
}

impl TrainStats {
    /// Merge a sub-fit's statistics.
    pub fn merge(&mut self, other: &TrainStats) {
        self.candidates += other.candidates;
        self.rejected += other.rejected;
        self.iterations += other.iterations;
        self.quantization_attempts += other.quantization_attempts;
        self.fit_ns += other.fit_ns;
        self.gpu_ns += other.gpu_ns;
        self.peak_ram_bytes = self.peak_ram_bytes.max(other.peak_ram_bytes);
        self.peak_vram_bytes = self.peak_vram_bytes.max(other.peak_vram_bytes);
    }
}

/// A bounded training/search budget. Search budgets are evidence (`O.43`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainBudget {
    /// Maximum candidate models to fit.
    pub max_candidates: u64,
    /// Maximum causal taps considered.
    pub max_taps: u32,
    /// Maximum hidden units for the nonlinear family.
    pub max_hidden: u32,
    /// Maximum optimizer iterations per candidate.
    pub max_iterations: u64,
    /// Maximum canonical model bytes accepted.
    pub max_model_bytes: u64,
    /// Ridge regularization strength (training only).
    pub ridge_lambda: f64,
}

impl Default for TrainBudget {
    fn default() -> Self {
        TrainBudget {
            max_candidates: 64,
            max_taps: 256,
            max_hidden: 8,
            max_iterations: 4096,
            max_model_bytes: 1 << 20,
            ridge_lambda: 1e-6,
        }
    }
}

impl TrainBudget {
    /// Enforce the ceilings before a search begins (never a silent no-op).
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.max_candidates == 0 {
            return Err(crate::error::Error::malformed(
                "training budget must allow at least one candidate",
            ));
        }
        if self.max_taps as u64 > u64::from(crate::limits::MAX_LEARNED_TAPS) {
            return Err(crate::error::Error::limit(
                "training budget tap count exceeds the bound",
            ));
        }
        if self.max_model_bytes > crate::limits::MAX_LEARNED_WEIGHT_BYTES.saturating_mul(4) {
            return Err(crate::error::Error::limit(
                "training budget model bytes exceed the bound",
            ));
        }
        if !self.ridge_lambda.is_finite() || self.ridge_lambda < 0.0 {
            return Err(crate::error::Error::malformed(
                "training ridge lambda must be finite and non-negative",
            ));
        }
        Ok(())
    }
}
