//! Content-addressed canonical persistence (H.2.21–H.2.25).
//!
//! VOLE-Audio stays materializable with **no** external store: canonical
//! entropy/procedural records are byte-identical whether fetched through the
//! embedded store or (feature `entropyfs`, default-off) through the EntropyFS
//! adapter. Identity is content-derived: a [`StoreId`] is the SHA-256 of the
//! canonical payload bytes, so deduplication is exact and retrieval always
//! re-verifiable.
//!
//! Accounting (H.2.24) reports **declared / unique / physical** separately —
//! never a shared model as zero bytes, never physical bytes conflated with
//! logical bytes (owner: `docs/ENTROPY_ACCOUNTING.md`).

use crate::entropy::accounting::StorageBytes;
use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;
use std::collections::HashMap;

/// Content identity of one canonical payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StoreId(pub [u8; 32]);

impl StoreId {
    pub const fn from_bytes(b: [u8; 32]) -> Self {
        Self(b)
    }
    pub const fn to_bytes(self) -> [u8; 32] {
        self.0
    }
    /// Content id of canonical bytes.
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes))
    }
}

impl core::fmt::Display for StoreId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Bounded store operations shared by every backend.
pub trait StoreBackend {
    /// Store canonical bytes; returns the content id. Exact duplicates are
    /// deduplicated (same id, one physical copy).
    fn put(&mut self, canonical: &[u8]) -> Result<StoreId>;
    /// Fetch a payload; `None` when absent. Fails typed when `max_bytes` is
    /// exceeded (bounded retrieval; no heap bombs).
    fn get(&self, id: &StoreId, max_bytes: u64) -> Result<Option<Vec<u8>>>;
    fn contains(&self, id: &StoreId) -> bool;
    /// Flush durability (no-op where the backend is synchronous in-memory).
    fn sync(&mut self) -> Result<()>;
    /// Declared / unique / physical byte accounting.
    fn accounting(&self) -> StorageBytes;
}

/// In-crate embedded store: canonical payloads held by content id.
///
/// `physical == unique` here by construction (no engine metadata); the
/// distinction becomes meaningful for the EntropyFS adapter, whose physical
/// bytes include engine framing. Declared bytes accumulate over `put` calls
/// (duplicates count toward declared, never toward unique/physical).
#[derive(Debug, Clone, Default)]
pub struct EmbeddedStore {
    objects: HashMap<StoreId, Vec<u8>>,
    declared_bytes: u64,
    put_count: u64,
}

impl EmbeddedStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl StoreBackend for EmbeddedStore {
    fn put(&mut self, canonical: &[u8]) -> Result<StoreId> {
        if canonical.len() as u64 > crate::limits::MAX_CHUNK_BYTES as u64 {
            return Err(Error::limit("store payload above chunk ceiling"));
        }
        let id = StoreId::of(canonical);
        self.declared_bytes = self.declared_bytes.saturating_add(canonical.len() as u64);
        self.put_count += 1;
        self.objects.entry(id).or_insert_with(|| canonical.to_vec());
        Ok(id)
    }

    fn get(&self, id: &StoreId, max_bytes: u64) -> Result<Option<Vec<u8>>> {
        let Some(bytes) = self.objects.get(id) else {
            return Ok(None);
        };
        if bytes.len() as u64 > max_bytes {
            return Err(Error::limit("store payload exceeds retrieval bound"));
        }
        // Re-verify content identity on every retrieval.
        if StoreId::of(bytes) != *id {
            return Err(Error::integrity("store payload identity mismatch"));
        }
        Ok(Some(bytes.clone()))
    }

    fn contains(&self, id: &StoreId) -> bool {
        self.objects.contains_key(id)
    }

    fn sync(&mut self) -> Result<()> {
        Ok(()) // in-memory: already coherent
    }

    fn accounting(&self) -> StorageBytes {
        let unique: u64 = self.objects.values().map(|b| b.len() as u64).sum();
        StorageBytes {
            declared_bytes: self.declared_bytes,
            unique_bytes: unique,
            physical_bytes: unique,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_store_dedupes_and_accounts() {
        let mut s = EmbeddedStore::new();
        let a_payload = b"canonical-payload-a".as_slice();
        let b_payload = b"canonical-payload-b".as_slice();
        let a = s.put(a_payload).unwrap();
        let a2 = s.put(a_payload).unwrap();
        let b = s.put(b_payload).unwrap();
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(s.len(), 2);
        let acc = s.accounting();
        assert_eq!(acc.unique_bytes, (a_payload.len() + b_payload.len()) as u64);
        assert_eq!(
            acc.declared_bytes,
            (a_payload.len() * 2 + b_payload.len()) as u64
        );
        assert_eq!(acc.physical_bytes, acc.unique_bytes);
        assert_eq!(s.get(&a, 1024).unwrap().as_deref(), Some(a_payload));
        assert!(s.get(&a, 4).is_err(), "bounded retrieval enforced");
        assert!(s.contains(&b));
        assert!(!s.contains(&StoreId::of(b"missing")));
    }

    #[test]
    fn retrieval_revalidates_identity() {
        let mut s = EmbeddedStore::new();
        let _id = s.put(b"payload").unwrap();
        let acc = s.accounting();
        assert_eq!(acc.declared_bytes, 7);
        assert_eq!(acc.physical_bytes, 7);
        // Tamper the map directly (simulating corruption); retrieval fails.
        let corrupted = StoreId([0u8; 32]);
        s.objects.insert(corrupted, b"tampered!".to_vec());
        assert!(s.get(&corrupted, 1024).is_err());
    }
}
