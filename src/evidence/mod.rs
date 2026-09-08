//! Evidence subsystem: `vole.audio.evidence.v1`.
//!
//! Versioned, immutable, JSON-serialized receipts plus the counters, timing,
//! environment, hardware, trace, and energy primitives they bind together.
//! Everything in this module is host (`std`) only; device kernels emit plain
//! status bytes from the shared vocabulary in `crate::status`.

pub mod artifact;
pub mod counters;
pub mod energy;
pub mod environment;
pub mod hardware;
pub mod receipt;
pub mod timing;
pub mod trace;

pub use counters::Counters;
pub use receipt::{
    CourtParams, EVIDENCE_SCHEMA, EVIDENCE_SCHEMA_VERSION, EndpointEvidence, Provenance, Receipt,
    ReceiptBuilder, ReceiptEnvelope, RunTiming, TraceInfo,
};
pub use timing::{DurationNs, InstantNs, Stopwatch, TailSummary};
