//! Hard, universe-wide resource bounds.
//!
//! These numbers are part of the u1 profile contract (see `docs/U1_SPEC.md`).
//! Parsers, schedulers, decoders, kernels, and evidence accounting all enforce
//! the same ceilings so that hostile input cannot produce unbounded allocation,
//! unbounded expansion, pathological recursion, or accumulator overflow.
//!
//! This module is `no_std`-clean and compiles unchanged for the GPU device
//! targets; kernels share the identical constants.
//!
//! Freezing policy: changing any bound below is a universe/profile change and
//! requires a new profile id plus fresh reference vectors.

#![allow(dead_code)] // Constants are consumed progressively as phases land.

/// Nominal channel ceiling for observation views and ingest.
pub const MAX_CHANNELS: u32 = 32;

/// Maximum object intrinsic extent, in frames.
///
/// (2^39 - 1) frames ≈ 3.2 hours at 44.1 kHz, and the bound exists so that
/// `extent << 24` fits a positive i64 (Q24 positions can address any valid
/// frame). This is the *ceiling*; real objects are far smaller.
pub const MAX_OBJECT_FRAMES: u64 = (1 << 39) - 1;

/// Maximum single observation window, in frames. Bounded so that
/// `rate_max * window` cannot overflow i64 position math
/// (2^16 frames/frame * 2^32 frames = 2^48).
pub const MAX_OBSERVATION_FRAMES: u64 = 1 << 32;

/// Maximum concurrent active voices in the sampler world.
///
/// Overflow proof (U1_SPEC.md §"Mixing bound"): each voice's contribution to
/// the per-channel mix accumulator is saturated to |c| <= 2^31 - 1 before the
/// i64 accumulation, so |mix| <= 4096 * (2^31 - 1) < 2^43, far inside i64.
pub const MAX_ACTIVE_VOICES: u32 = 4096;

/// Maximum total scheduled events in a world timeline.
pub const MAX_SCHEDULED_EVENTS: u32 = 1 << 20;

/// Maximum events consumed within a single quantum (host submit slice).
pub const MAX_EVENTS_PER_QUANTUM: u32 = 1 << 16;

/// Maximum dependencies declared by one SampleObject.
pub const MAX_DEPENDENCIES_PER_OBJECT: u32 = 256;

/// Maximum nodes in one dependency/reference graph (reachable set).
pub const MAX_GRAPH_NODES: u32 = 1 << 16;

/// Maximum reference indirection depth before a dependency is declared cyclic.
pub const MAX_REFERENCE_DEPTH: u32 = 64;

/// Maximum residual records in one residual-governed object.
pub const MAX_RESIDUAL_RECORDS: u32 = 1 << 24;

/// Maximum checkpoints in one checkpointed object.
pub const MAX_CHECKPOINTS_PER_OBJECT: u32 = 1 << 16;

/// Maximum SampleObjects in one corpus / archive.
pub const MAX_OBJECTS_PER_CORPUS: u32 = 1 << 20;

/// Maximum nominal sample rate accepted by ingest (Hz).
pub const MAX_SAMPLE_RATE_HZ: u32 = 1_000_000;

/// Maximum nominal sample rate accepted by observation (Hz); endpoints and
/// clock hardware must report rates in `[MIN, MAX]`.
pub const MIN_SAMPLE_RATE_HZ: u32 = 1;

/// Maximum observation quantum (frames) for one submit slice.
pub const MAX_QUANTUM_FRAMES: u32 = 1 << 16;

/// Absolute file-size ceiling for any ingest path (archive or WAV).
pub const MAX_FILE_BYTES: u64 = 1 << 40;

/// Maximum WAV data payload bytes accepted by the parser.
pub const MAX_WAV_DATA_BYTES: u64 = 1 << 34;

/// Maximum single archive chunk length.
pub const MAX_CHUNK_BYTES: u32 = 1 << 30;

/// Maximum length of any bounded string field (ids, names, paths in media).
pub const MAX_STRING_BYTES: u32 = 1 << 12;

/// Maximum frozen lookup-table size the loader will accept into any memory
/// class (the u1 polyphase resampler table is 128 KiB; this is a generous cap).
pub const MAX_TABLE_BYTES: u32 = 1 << 20;

/// Maximum partials in a partial-bank SampleObject.
pub const MAX_PARTIALS: u32 = 4096;

/// Maximum taps of the frozen polyphase resampler.
pub const MAX_INTERPOLATION_TAPS: u32 = 64;

/// Default u1 nominal sample rate (Hz).
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 48_000;

/// Default observation quantum for the sampler scheduler (frames).
pub const DEFAULT_QUANTUM_FRAMES: u32 = 1024;

/// Fixed-point fraction bits for source position and rate (u1 semantics).
pub const FIXED_Q: u32 = 24;

/// Fixed-point fraction bits for gain/pan/envelope multipliers (u1 semantics).
/// Unity (exactly 1.0) is raw `1 << GAIN_Q`.
pub const GAIN_Q: u32 = 16;

/// Oscillator phase width: u64 modulo 2^64.
pub const PHASE_BITS: u32 = 64;

// ---------------------------------------------------------------------------
// Entropy layer bounds (Phase H.2; part of the `vole.entropy.p1` profile
// contract — see docs/RANS.md and docs/ENTROPY_NATIVE.md). Parsers and
// decoders enforce these ceilings so hostile input cannot produce unbounded
// allocation/expansion/CPU (H.2.34).
// ---------------------------------------------------------------------------

/// rANS scale bits of the audio profile (frozen, see docs/RANS.md).
pub const RANS_SCALE_BITS: u32 = 14;

/// rANS total normalized frequency per model: `1 << RANS_SCALE_BITS`.
pub const RANS_MODEL_TOTAL: u32 = 1 << RANS_SCALE_BITS;

/// rANS lower bound of the normalized state interval (frozen).
pub const RANS_STATE_L: u32 = 1 << 23;

/// Maximum distinct present symbols in one entropy model (u8-valued
/// alphabets; canonical order = ascending symbol value).
pub const MAX_MODEL_ALPHABET: usize = 256;

/// Maximum intrinsic frames declared by one entropy page.
pub const MAX_ENTROPY_PAGE_FRAMES: u32 = 1 << 16;

/// Maximum pages in one entropy-coded object.
pub const MAX_ENTROPY_PAGES_PER_OBJECT: u32 = 1 << 20;

/// Maximum encoded bytes per entropy page payload (rANS/RAW body).
pub const MAX_ENTROPY_PAGE_BODY_BYTES: u32 = 1 << 26;

/// Maximum decoded bytes a single page may reconstruct (symbol count cap).
pub const MAX_ENTROPY_PAGE_DECODED_BYTES: u32 = 1 << 26;

/// Maximum inline model bytes in one page/block.
pub const MAX_ENTROPY_INLINE_MODEL_BYTES: u32 = 1 << 16;

/// Maximum shared models referenced by one object.
pub const MAX_ENTROPY_SHARED_MODELS: u32 = 1 << 16;

/// Maximum model bytes across an object's shared model pool.
pub const MAX_ENTROPY_MODEL_POOL_BYTES: u32 = 1 << 22;

/// Maximum page-index bytes in one object index.
pub const MAX_ENTROPY_INDEX_BYTES: u32 = 1 << 28;

/// Maximum entropy dependency depth (model/page references).
pub const MAX_ENTROPY_DEPENDENCY_DEPTH: u32 = 8;

/// Maximum transient sample-domain scratch bytes for one observation.
pub const MAX_ENTROPY_SCRATCH_BYTES: u32 = 1 << 26;

/// Worst-case rANS encoded bytes per symbol (renorm bytes + slack).
pub const RANS_MAX_BYTES_PER_SYMBOL: u32 = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixing_bound_has_headroom() {
        // The documented proof: 4096 voices each saturating at 2^31-1 before
        // i64 accumulation stays far below i64::MAX.
        let worst: i128 = MAX_ACTIVE_VOICES as i128 * ((1i64 << 31) - 1) as i128;
        assert!(worst < (1i128 << 62));
        // Half of i64 magnitude is 2^62; we demand an order of magnitude margin.
        assert!(worst < (1i128 << 53));
    }

    #[test]
    fn entropy_profile_constants_are_consistent() {
        const {
            assert!(RANS_MODEL_TOTAL == (1u32 << RANS_SCALE_BITS));
            assert!(RANS_MODEL_TOTAL == 16_384);
            assert!(RANS_STATE_L == (1 << 23));
            assert!(MAX_MODEL_ALPHABET <= RANS_MODEL_TOTAL as usize);
            assert!(MAX_ENTROPY_PAGE_DECODED_BYTES >= 4 * MAX_ENTROPY_PAGE_FRAMES);
        }
    }

    #[test]
    fn limits_respect_repr_widths() {
        // Frame coordinates are i64; extents stay below the Q24 i64 ceiling
        // (extent<<24 < 2^63) and the quantum stays small.
        const {
            assert!(MAX_OBJECT_FRAMES < (1u64 << 39));
            assert!(MAX_QUANTUM_FRAMES < (1 << 20));
            assert!(MAX_OBSERVATION_FRAMES <= 1u64 << 32);
        };
    }
}
