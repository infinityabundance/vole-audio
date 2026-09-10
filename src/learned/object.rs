//! Canonical learned object (`O.4`, `O.12`, `O.28`).
//!
//! A [`LearnedObject`] is a complete, self-describing executable hypothesis
//! plus its exact residual and its dependency closure. It is **not** a
//! `u1/v1` `SampleObject`: it lives in the experimental learned profile, and it
//! must prove exact closure to the canonical intrinsic sample domain before it
//! can participate in any comparison.
//!
//! ```text
//! X_hat[t] = sat_i32(H_Theta(X_hat history)[t] + R[t])     and   X_hat == X
//! ```

use crate::error::{Error, Kind, Result};
use crate::hash::sha256::Sha256;
use crate::learned::model::LearnedModel;
use crate::learned::profile::LearnedProfile;
use crate::learned::residual_codec2::{
    ResidualCodecV2, ResidualEncodingV2, decode_encoding_v2, encode_best_v1, encode_best_v2,
};
use crate::learned::serialization;
use crate::object::id::ContentId;

/// A complete learned representation of one intrinsic object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnedObject {
    /// The experimental profile this object belongs to (`exp1` is frozen).
    pub profile: LearnedProfile,
    pub channels: u8,
    pub frames: u64,
    pub sample_rate_hz: u32,
    pub model: LearnedModel,
    pub residual_codec: ResidualCodecV2,
    /// Residual payload only (no codec id byte).
    pub residual_bytes: Vec<u8>,
    /// Transitive dependency content ids (models, sources, shared tables).
    pub dependencies: Vec<ContentId>,
}

impl LearnedObject {
    /// Decode the full dense residual (`frames × channels` values).
    pub fn residual(&self) -> Result<Vec<i32>> {
        let len = self.residual_len()?;
        self.residual_codec.decode(&self.residual_bytes, len)
    }

    fn residual_len(&self) -> Result<usize> {
        let c = usize::from(self.channels);
        usize::try_from(self.frames)
            .ok()
            .and_then(|f| f.checked_mul(c))
            .ok_or_else(|| Error::limit("learned residual length overflows host usize"))
    }

    /// Validate geometry, model, residual and complexity ceilings.
    pub fn validate(&self) -> Result<()> {
        if self.channels == 0 || self.channels > crate::limits::MAX_CHANNELS as u8 {
            return Err(Error::malformed(
                "learned object channel count out of range",
            ));
        }
        if self.frames == 0 || self.frames > crate::limits::MAX_OBJECT_FRAMES {
            return Err(Error::limit("learned object frame count exceeds the bound"));
        }
        if self.sample_rate_hz == 0 || self.sample_rate_hz > crate::limits::MAX_SAMPLE_RATE_HZ {
            return Err(Error::malformed("learned object sample rate out of domain"));
        }
        self.model.validate()?;
        if self.model.channels() != self.channels {
            return Err(Error::malformed(
                "learned model channels disagree with the object",
            ));
        }
        if self.dependencies.len() as u32 > crate::limits::MAX_LEARNED_DEPENDENCIES {
            return Err(Error::limit("learned dependency count exceeds the bound"));
        }
        if self.residual_bytes.len() as u64 > crate::limits::MAX_LEARNED_RESIDUAL_BYTES {
            return Err(Error::limit("learned residual bytes exceed the bound"));
        }
        // The residual must decode to exactly the declared geometry.
        let len = self.residual_len()?;
        let _ = self
            .residual_codec
            .decode(&self.residual_bytes, len)
            .map_err(|_| Error::malformed("learned residual does not match its geometry"))?;
        // Decode-complexity ceiling (`O.30`): declared worst case for the extent.
        let ops = self
            .model
            .ops_per_sample()
            .saturating_mul(self.frames)
            .saturating_add(self.model.receptive_field().saturating_mul(self.frames));
        if ops > crate::limits::MAX_LEARNED_DECODE_OPS {
            return Err(Error::limit("learned decode complexity exceeds the bound"));
        }
        Ok(())
    }

    /// The exact source that a transfer object depends on, if any.
    pub fn requires_source(&self) -> bool {
        self.model.requires_source()
    }

    /// Reconstruct the whole canonical intrinsic extent exactly.
    pub fn materialize(&self) -> Result<Vec<i32>> {
        self.materialize_with_source(None)
    }

    /// Reconstruct the whole canonical intrinsic extent exactly, supplying the
    /// source samples for a transfer object.
    pub fn materialize_with_source(&self, source: Option<&[i32]>) -> Result<Vec<i32>> {
        let frames = self.frames as usize;
        self.materialize_range_with_source(0, frames, source)
    }

    /// Reconstruct `[start, start+len)` exactly.
    pub fn materialize_range(&self, start: usize, len: usize) -> Result<Vec<i32>> {
        self.materialize_range_with_source(start, len, None)
    }

    /// Reconstruct `[start, start+len)` exactly, supplying a transfer source.
    pub fn materialize_range_with_source(
        &self,
        start: usize,
        len: usize,
        source: Option<&[i32]>,
    ) -> Result<Vec<i32>> {
        let frames = self.frames as usize;
        let residual = self.residual()?;
        self.model
            .evaluate_range_with_source(&residual, source, frames, start, len)
    }

    /// Frames replayed to serve a seek (0 = truly random access).
    pub fn replay_frames(&self, start: usize) -> usize {
        self.model.replay_frames(start)
    }

    /// Build the canonical learned object that closes `source` exactly under
    /// `model`. Fails rather than degrading if closure is not exact.
    pub fn from_intrinsic(
        model: LearnedModel,
        channels: u8,
        frames: u64,
        sample_rate_hz: u32,
        dependencies: Vec<ContentId>,
        source: &[i32],
    ) -> Result<LearnedObject> {
        Self::build(
            LearnedProfile::Exp1,
            model,
            channels,
            frames,
            sample_rate_hz,
            dependencies,
            source,
            source,
        )
    }

    /// Build the canonical **Exp2** learned object that closes `source` exactly
    /// under `model`. Exp2 imports every Exp1 candidate and adds the v2 residual
    /// codec family; the Exp1 constructor above is byte-for-byte unchanged.
    pub fn from_intrinsic_exp2(
        model: LearnedModel,
        channels: u8,
        frames: u64,
        sample_rate_hz: u32,
        dependencies: Vec<ContentId>,
        source: &[i32],
    ) -> Result<LearnedObject> {
        Self::build(
            LearnedProfile::Exp2,
            model,
            channels,
            frames,
            sample_rate_hz,
            dependencies,
            source,
            source,
        )
    }

    /// Build the canonical transfer object whose operator maps `source` onto
    /// `target`, with exact closure `X_hat == target`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_transfer_operator(
        model: LearnedModel,
        channels: u8,
        frames: u64,
        sample_rate_hz: u32,
        dependencies: Vec<ContentId>,
        source: &[i32],
        target: &[i32],
    ) -> Result<LearnedObject> {
        if !model.requires_source() {
            return Err(Error::malformed(
                "transfer construction requires a source-dependent model",
            ));
        }
        Self::build(
            LearnedProfile::Exp1,
            model,
            channels,
            frames,
            sample_rate_hz,
            dependencies,
            source,
            target,
        )
    }

    /// Exp2 transfer construction (v2 residual family).
    #[allow(clippy::too_many_arguments)]
    pub fn from_transfer_operator_exp2(
        model: LearnedModel,
        channels: u8,
        frames: u64,
        sample_rate_hz: u32,
        dependencies: Vec<ContentId>,
        source: &[i32],
        target: &[i32],
    ) -> Result<LearnedObject> {
        if !model.requires_source() {
            return Err(Error::malformed(
                "transfer construction requires a source-dependent model",
            ));
        }
        Self::build(
            LearnedProfile::Exp2,
            model,
            channels,
            frames,
            sample_rate_hz,
            dependencies,
            source,
            target,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        profile: LearnedProfile,
        model: LearnedModel,
        channels: u8,
        frames: u64,
        sample_rate_hz: u32,
        dependencies: Vec<ContentId>,
        hypothesis_source: &[i32],
        target: &[i32],
    ) -> Result<LearnedObject> {
        model.validate()?;
        if model.channels() != channels {
            return Err(Error::malformed(
                "learned model channels disagree with the intrinsic",
            ));
        }
        let n = usize::try_from(frames)
            .map_err(|_| Error::limit("learned frame count exceeds host usize"))?
            .checked_mul(usize::from(channels))
            .ok_or_else(|| Error::limit("learned sample count overflows"))?;
        if target.len() != n {
            return Err(Error::malformed(
                "learned intrinsic length does not match the geometry",
            ));
        }
        let h = model.hypothesis_from_source(hypothesis_source, frames as usize)?;
        let mut residual = vec![0i32; n];
        for i in 0..n {
            let d = i64::from(target[i]) - i64::from(h[i]);
            if d < i64::from(i32::MIN) || d > i64::from(i32::MAX) {
                return Err(Error::new(
                    Kind::Integrity,
                    "learned residual does not fit i32 (uncloseable gap)",
                ));
            }
            residual[i] = d as i32;
        }
        let enc = match profile {
            LearnedProfile::Exp1 => encode_best_v1(&residual),
            LearnedProfile::Exp2 => encode_best_v2(&residual),
        };
        let object = LearnedObject {
            profile,
            channels,
            frames,
            sample_rate_hz,
            model,
            residual_codec: enc.codec,
            residual_bytes: enc.bytes[1..].to_vec(),
            dependencies,
        };
        object.validate()?;
        // Prove exact closure before accepting the object.
        let recon = object.materialize_with_source(Some(hypothesis_source))?;
        if recon != target {
            return Err(Error::new(
                Kind::Integrity,
                "learned candidate does not close exactly to the intrinsic",
            ));
        }
        Ok(object)
    }

    /// Canonical bytes (including the trailing identity digest).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serialization::encode(self)
    }

    /// Content identity: SHA-256 of the canonical bytes.
    pub fn content_id(&self) -> ContentId {
        ContentId::from_bytes(Sha256::digest(&self.canonical_bytes()))
    }

    /// Parse and fully validate a canonical learned object.
    pub fn parse(bytes: &[u8]) -> Result<LearnedObject> {
        serialization::decode(bytes)
    }

    /// Re-verify exact closure against a canonical source.
    pub fn verify(&self, source: &[i32]) -> bool {
        match self.materialize_with_source(Some(source)) {
            Ok(v) => v == source,
            Err(_) => false,
        }
    }

    /// Re-verify exact closure of a transfer object against its target.
    pub fn verify_with_source(&self, source: &[i32], target: &[i32]) -> bool {
        match self.materialize_with_source(Some(source)) {
            Ok(v) => v == target,
            Err(_) => false,
        }
    }

    /// Decode one residual encoding candidate for accounting/courts.
    pub fn residual_encoding(&self) -> Result<ResidualEncodingV2> {
        let mut bytes = Vec::with_capacity(self.residual_bytes.len() + 1);
        bytes.push(self.residual_codec.id());
        bytes.extend_from_slice(&self.residual_bytes);
        decode_encoding_v2(&bytes, self.residual_len()?)
    }
}
