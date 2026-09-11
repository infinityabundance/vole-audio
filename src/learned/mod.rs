//! Phase O — learned deterministic prediction (experimental profile
//! `vole.audio.learned.exp1`).
//!
//! A learned hypothesis is one more bounded, falsifiable candidate family
//! inside VOLE-Audio's exact residual-closure architecture. It is never
//! semantic authority: `SampleObject` stays authoritative, the literal fallback
//! stays mandatory, and a learned candidate participates in selection only
//! after canonical quantization, exact residual closure, and complete
//! dependency accounting (`O.65`).
//!
//! The module is `std`-gated: canonical evaluation is integer-only, but the
//! training/fitting path uses floating point and host facilities.

pub mod accounting;
pub mod analytical;
pub mod arithmetic;
pub mod corpus;
pub mod finite_field;
pub mod graph;
pub mod hierarchy;
pub mod ltp;
pub mod model;
pub mod multichannel;
pub mod object;
pub mod profile;
pub mod quantize;
pub mod residual_codec;
pub mod residual_codec2;
pub mod segmentation;
pub mod segmented;
pub mod serialization;
pub mod sparse;
pub mod stateful;
pub mod train;
pub mod transfer;

pub use arithmetic::{
    Acc, Activation, Bias, Weight, accumulator_is_safe, round_shift_half_away, sat_i16, sat_i32,
};
pub use residual_codec::{
    ResidualCodec, ResidualEncoding, decode_encoding, encode_all, encode_best,
};
pub use residual_codec2::{
    ResidualCodecV2, ResidualEncodingV2, decode_encoding_v2, encode_all_v2, encode_best_v1,
    encode_best_v2,
};
