//! Checkpoint payloads and resync (Phase N, contract §36/§37).
//!
//! A checkpoint is the (epoch, logical frame, opaque deterministic state) tuple
//! a receiver resynchronises to after a gap or an epoch change. The state bytes
//! are opaque here: the sampler layer owns their semantics, and recovery must be
//! deterministic under the chosen [`crate::universe::clock::XrunPolicy`].
//!
//! ```text
//! CheckpointPayload :=
//!     EPOCH         u32
//!     LOGICAL_FRAME i64
//!     STATE_LEN     u32     0..=MAX_CHECKPOINT_STATE_BYTES
//!     STATE         bytes
//! ```

use crate::error::{Error, Result};
use crate::limits::MAX_CHECKPOINT_STATE_BYTES;

/// Fixed header size preceding the state bytes.
pub const CHECKPOINT_HEADER_BYTES: usize = 4 + 8 + 4;

/// A checkpoint: the recovery anchor for one epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub epoch: u32,
    pub logical_frame: i64,
    pub state: Vec<u8>,
}

impl Checkpoint {
    pub fn new(epoch: u32, logical_frame: i64, state: Vec<u8>) -> Self {
        Checkpoint {
            epoch,
            logical_frame,
            state,
        }
    }
}

/// Encode a checkpoint payload.
pub fn encode_checkpoint(c: &Checkpoint) -> Result<Vec<u8>> {
    let state_len = u32::try_from(c.state.len())
        .map_err(|_| Error::limit("checkpoint state exceeds the bound"))?;
    if state_len > MAX_CHECKPOINT_STATE_BYTES {
        return Err(Error::limit("checkpoint state exceeds the bound"));
    }
    let mut out = Vec::with_capacity(CHECKPOINT_HEADER_BYTES + c.state.len());
    out.extend_from_slice(&c.epoch.to_le_bytes());
    out.extend_from_slice(&c.logical_frame.to_le_bytes());
    out.extend_from_slice(&state_len.to_le_bytes());
    out.extend_from_slice(&c.state);
    Ok(out)
}

/// Parse a checkpoint payload (no trailing bytes).
pub fn decode_checkpoint(bytes: &[u8]) -> Result<Checkpoint> {
    if bytes.len() < CHECKPOINT_HEADER_BYTES {
        return Err(Error::malformed("checkpoint payload is truncated"));
    }
    let epoch = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let logical_frame = i64::from_le_bytes(bytes[4..12].try_into().unwrap());
    let state_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if state_len > MAX_CHECKPOINT_STATE_BYTES {
        return Err(Error::limit("checkpoint state exceeds the bound"));
    }
    let state_len = usize::try_from(state_len)
        .map_err(|_| Error::limit("checkpoint state length exceeds host usize"))?;
    let end = CHECKPOINT_HEADER_BYTES
        .checked_add(state_len)
        .ok_or_else(|| Error::limit("checkpoint length overflows"))?;
    if end != bytes.len() {
        return Err(Error::malformed("checkpoint payload has trailing bytes"));
    }
    Ok(Checkpoint {
        epoch,
        logical_frame,
        state: bytes[CHECKPOINT_HEADER_BYTES..end].to_vec(),
    })
}

/// Digest of a checkpoint's state, for identity/binding in receipts.
pub fn checkpoint_state_digest(c: &Checkpoint) -> [u8; 32] {
    crate::hash::sha256::Sha256::digest(&c.state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trip_is_exact() {
        let c = Checkpoint::new(3, 1_234_567, vec![0xAA, 0xBB, 0xCC]);
        let bytes = encode_checkpoint(&c).unwrap();
        assert_eq!(bytes.len(), CHECKPOINT_HEADER_BYTES + 3);
        assert_eq!(decode_checkpoint(&bytes).unwrap(), c);
        // An empty state is legal (a pure position anchor).
        let empty = Checkpoint::new(0, 0, Vec::new());
        assert_eq!(
            decode_checkpoint(&encode_checkpoint(&empty).unwrap()).unwrap(),
            empty
        );
    }

    #[test]
    fn malformed_checkpoints_are_rejected() {
        assert!(decode_checkpoint(&[0u8; 4]).is_err());
        let c = Checkpoint::new(1, 2, vec![1, 2, 3]);
        let mut bytes = encode_checkpoint(&c).unwrap();
        bytes.truncate(CHECKPOINT_HEADER_BYTES + 2);
        assert!(decode_checkpoint(&bytes).is_err());
        let mut trailing = encode_checkpoint(&c).unwrap();
        trailing.push(0);
        assert!(decode_checkpoint(&trailing).is_err());
    }
}
