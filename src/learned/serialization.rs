//! Canonical learned-object serialization (`O.28`).
//!
//! Explicit little-endian binary, written and read by hand. serde/bincode
//! layout, Rust enum memory layout, and any implementation-language object
//! model are **never** normative media representation.
//!
//! ```text
//! LearnedObject :=
//!     MAGIC          12   b"vole.learned"
//!     VERSION        u8   = 1
//!     PROFILE_LEN    u8
//!     PROFILE        bytes  b"vole.audio.u1/vole.audio.learned.exp1"
//!     CHANNELS       u8
//!     FRAMES         u64
//!     SAMPLE_RATE    u32
//!     MODEL_LEN      u64
//!     MODEL          bytes  (kind tag first; see learned::model)
//!     RESIDUAL_CODEC u8
//!     RESIDUAL_LEN   u64
//!     RESIDUAL       bytes  (payload only, no codec id)
//!     DEP_COUNT      u32
//!     DEP            DEP_COUNT × 32-byte content id
//!     DIGEST         32     SHA-256 over everything before it
//! ```
//!
//! Every read is bounds-checked; every length is converted with `try_from`
//! (never a cast) and bounded before allocation.

use crate::error::{Error, Kind, Result};
use crate::hash::sha256::Sha256;
use crate::learned::model::LearnedModel;
use crate::learned::object::LearnedObject;
use crate::learned::profile::{LEARNED_FORMAT_VERSION, LEARNED_MAGIC, LEARNED_PROFILE_TAG};
use crate::learned::residual_codec::ResidualCodec;
use crate::object::id::ContentId;

/// A checked little-endian reader over a byte slice.
pub struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("learned read offset overflows"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::malformed("learned object is truncated"))?;
        self.pos = end;
        Ok(slice)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn array32(&mut self) -> Result<[u8; 32]> {
        Ok(self.take(32)?.try_into().unwrap())
    }

    pub fn finish(&self) -> Result<()> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(Error::malformed("learned object has trailing bytes"))
        }
    }
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Canonical bytes of a learned object, including its trailing identity digest.
pub fn encode(o: &LearnedObject) -> Vec<u8> {
    let model_bytes = o.model.canonical_bytes();
    let mut out = Vec::with_capacity(
        LEARNED_MAGIC.len()
            + 1
            + 1
            + LEARNED_PROFILE_TAG.len()
            + 1
            + 8
            + 4
            + 8
            + model_bytes.len()
            + 1
            + 8
            + o.residual_bytes.len()
            + 4
            + o.dependencies.len() * 32
            + 32,
    );
    out.extend_from_slice(LEARNED_MAGIC);
    out.push(LEARNED_FORMAT_VERSION);
    out.push(LEARNED_PROFILE_TAG.len() as u8);
    out.extend_from_slice(LEARNED_PROFILE_TAG);
    out.push(o.channels);
    put_u64(&mut out, o.frames);
    out.extend_from_slice(&o.sample_rate_hz.to_le_bytes());
    put_u64(&mut out, model_bytes.len() as u64);
    out.extend_from_slice(&model_bytes);
    out.push(o.residual_codec.id());
    put_u64(&mut out, o.residual_bytes.len() as u64);
    out.extend_from_slice(&o.residual_bytes);
    out.extend_from_slice(&(o.dependencies.len() as u32).to_le_bytes());
    for d in &o.dependencies {
        out.extend_from_slice(&d.to_bytes());
    }
    let digest = Sha256::digest(&out);
    out.extend_from_slice(&digest);
    out
}

/// Parse and fully validate a canonical learned object.
pub fn decode(bytes: &[u8]) -> Result<LearnedObject> {
    const MIN: usize = 12 + 1 + 1 + 1 + 8 + 4 + 8 + 1 + 8 + 4 + 32;
    if bytes.len() < MIN {
        return Err(Error::malformed("learned object is too short"));
    }
    if &bytes[..LEARNED_MAGIC.len()] != LEARNED_MAGIC {
        return Err(Error::malformed("learned object magic mismatch"));
    }
    let mut r = Reader::new(bytes);
    let _ = r.take(LEARNED_MAGIC.len())?;
    let version = r.u8()?;
    if version != LEARNED_FORMAT_VERSION {
        return Err(Error::new(
            Kind::Unsupported,
            format!("unsupported learned object version {version}"),
        ));
    }
    let profile_len = r.u8()? as usize;
    let profile = r.take(profile_len)?;
    if profile != LEARNED_PROFILE_TAG {
        return Err(Error::malformed("learned object profile tag mismatch"));
    }
    let channels = r.u8()?;
    if channels == 0 || channels > crate::limits::MAX_CHANNELS as u8 {
        return Err(Error::malformed(
            "learned object channel count out of range",
        ));
    }
    let frames = r.u64()?;
    if frames == 0 || frames > crate::limits::MAX_OBJECT_FRAMES {
        return Err(Error::limit("learned object frame count exceeds the bound"));
    }
    let sample_rate_hz = r.u32()?;
    if sample_rate_hz == 0 || sample_rate_hz > crate::limits::MAX_SAMPLE_RATE_HZ {
        return Err(Error::malformed("learned object sample rate out of domain"));
    }
    let model_len = r.u64()?;
    let model_len_usize = usize::try_from(model_len)
        .map_err(|_| Error::limit("learned model length exceeds host usize"))?;
    if model_len > crate::limits::MAX_LEARNED_WEIGHT_BYTES.saturating_mul(4) {
        return Err(Error::limit("learned model length exceeds the bound"));
    }
    let model_bytes = r.take(model_len_usize)?;
    let model = LearnedModel::from_canonical_bytes(model_bytes)?;
    let codec_id = r.u8()?;
    let codec = ResidualCodec::from_id(codec_id).ok_or_else(|| {
        Error::new(
            Kind::Unsupported,
            format!("unknown residual codec {codec_id}"),
        )
    })?;
    let residual_len = r.u64()?;
    let residual_len_usize = usize::try_from(residual_len)
        .map_err(|_| Error::limit("learned residual length exceeds host usize"))?;
    if residual_len > crate::limits::MAX_LEARNED_RESIDUAL_BYTES {
        return Err(Error::limit("learned residual length exceeds the bound"));
    }
    let residual_bytes = r.take(residual_len_usize)?.to_vec();
    let dep_count = r.u32()?;
    if dep_count > crate::limits::MAX_LEARNED_DEPENDENCIES {
        return Err(Error::limit("learned dependency count exceeds the bound"));
    }
    let mut dependencies = Vec::with_capacity(dep_count as usize);
    for _ in 0..dep_count {
        dependencies.push(ContentId::from_bytes(r.array32()?));
    }
    // The trailing digest is the final 32 bytes of the canonical object.
    if r.remaining() != 32 {
        return Err(Error::malformed(
            "learned object has trailing or missing integrity bytes",
        ));
    }
    let stated = r.array32()?;
    r.finish()?;
    let body_end = bytes.len() - 32;
    if Sha256::digest(&bytes[..body_end]) != stated {
        return Err(Error::new(
            Kind::Integrity,
            "learned object digest mismatch",
        ));
    }
    let o = LearnedObject {
        channels,
        frames,
        sample_rate_hz,
        model,
        residual_codec: codec,
        residual_bytes,
        dependencies,
    };
    o.validate()?;
    Ok(o)
}
