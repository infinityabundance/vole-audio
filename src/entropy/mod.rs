//! Entropy-native audio core (Phase H.2).
//!
//! H.2 corrects an architectural omission: VOLE-Audio must not become merely
//! `procedural state -> PCM -> endpoint`. Its deeper architecture stores the
//! deterministic explanation, entropy-codes what the explanation cannot
//! reproduce, and materializes sample-domain observations only when required.
//!
//! Representation contract (H.2.1, owner: `docs/ENTROPY_NATIVE.md`):
//!
//! * [`crate::object::SampleObject`] semantics are **not** redefined. Entropy
//!   coding is a *physical/canonical representation* of a SampleObject's
//!   information, orthogonal to the hypothesis family. "rANS object" is never
//!   itself an audio semantics.
//! * The rANS/entropy layer never changes U1 arithmetic, the frozen
//!   representation tags, or the exact residual closure (Phase E).
//! * Scalar decode is the semantic authority. SIMD and CUDA decode reproduce
//!   scalar symbols exactly and have zero semantic authority.
//! * EntropyFS (persistence) and DSFB (search governance) are optional,
//!   default-off, never required by materialization, never in the decoder.
//!
//! Layout:
//!
//! * [`rans`] — native deterministic rANS primitives (`no_std`-clean, scalar
//!   authority; frozen parameters in [`crate::limits`]; spec in
//!   `docs/RANS.md`).
//! * [`model`] — canonical deterministic frequency normalization
//!   (largest-remainder; frozen tie-breaks).
//! * `symbol` (std) — reversible symbolizations over canonical sample codes.
//! * `block` (std) — canonical self-describing coded blocks/pages
//!   (`vole.entropy.p1`) with mandatory RAW fallback.
//! * `accounting` (std) — complete representation cost and byte ledger.
//! * `represent` (std) — physical SampleObject representations with partial
//!   (page-bounded) materialization.
//! * `store` (std) — content-addressed persistence abstraction
//!   (EmbeddedStore; EntropyFS behind the default-off `entropyfs-store` feature).
//! * `search` (std) — zero-authority candidate-search governance for the
//!   frozen entropy universe (H.2.26–H.2.29; DSFB behind the default-off
//!   `dsfb` feature).
//! * `hostile` (std) — deterministic hostile-input corpus for parser/decoder
//!   hardening.
//! * `corpus` (std) — the frozen entropy development corpus (H.2.30).
//!
//! Complete-cost rules: `docs/ENTROPY_ACCOUNTING.md`. Model bytes, page-index
//! bytes, and every dependency are always counted.

pub mod model;
pub mod rans;
pub mod transform;

#[cfg(feature = "std")]
pub mod accounting;
#[cfg(feature = "std")]
pub mod block;
#[cfg(feature = "std")]
pub mod corpus;
#[cfg(feature = "entropyfs-store")]
pub mod entropyfs_store;
#[cfg(feature = "std")]
pub mod hostile;
#[cfg(feature = "std")]
pub mod page_batch;
#[cfg(feature = "std")]
pub mod recoil;
#[cfg(feature = "std")]
pub mod represent;
#[cfg(feature = "std")]
pub mod search;
#[cfg(feature = "std")]
pub mod store;
#[cfg(feature = "std")]
pub mod symbol;
