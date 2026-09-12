//! Media format work: exact WAV ingest (Phase E); the canonical `.voleaudio`
//! archive lands in Phase N. All parsers are length-checked and hostile-input
//! tested; the media format is explicit binary, never serde/bincode.

pub mod archive;
pub mod manifest;
pub mod solid;
pub mod wav;

pub use wav::{DecodedWav, PcmFormat};
