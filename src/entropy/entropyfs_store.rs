//! EntropyFS persistence adapter (H.2.21–H.2.25; feature `entropyfs-store`,
//! default-off).
//!
//! A thin adapter implementing [`StoreBackend`] over the **real published
//! EntropyFS engine** (`entropyfs` 0.7.17, `default-features = false`: the
//! fuse/ublk/uring/tracing frontends are not needed — the embeddable
//! kernel-free `entropyfs::engine::Engine` facade runs on the sync IO
//! backend). EntropyFS is **not** reimplemented in vole-audio.
//!
//! Identity model (the boundary in `docs/ENTROPYFS.md`):
//!
//! * VOLE-Audio content identity is [`StoreId`] — SHA-256 of the canonical
//!   payload bytes.
//! * EntropyFS content identity is `entropyfs::engine::BlobId` — BLAKE3 of
//!   the blob's materialized logical bytes.
//!
//! The adapter keeps an **explicit mapping** `StoreId <-> BlobId` and
//! re-verifies canonical payloads on retrieval: `get` re-hashes the fetched
//! bytes with SHA-256 and requires the VOLE id to match, and the engine's own
//! `get_blob` exactness gate re-hashes with BLAKE3 first. VOLE identity is
//! never silently replaced by EntropyFS content identity.
//!
//! Accounting (H.2.24): **declared** bytes accumulate over every `put`
//! (duplicates count toward declared only); **unique** bytes count distinct
//! content ids once; **physical** bytes come from the engine's own metrics
//! (sum of segment-file lengths — the engine's actual backing store,
//! including its metadata/framing). The three quantities are always reported
//! separately.
//!
//! This module is encoder/store governance only: it never enters the decoder
//! or the playback path.

use crate::entropy::accounting::StorageBytes;
use crate::entropy::store::{StoreBackend, StoreId};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// Embeddable engine facade of the published EntropyFS crate (kernel-free;
/// content-addressed blobs over the persistent store).
use entropyfs::engine::{BlobId, Engine, EngineOpenOptions};

/// EntropyFS-backed content store.
///
/// Canonical payloads are stored as engine blobs, content-addressed by the
/// engine (BLAKE3); the adapter maps each VOLE [`StoreId`] (SHA-256) to the
/// engine blob id that holds its preimage. Retrieval re-verifies both hashes
/// before any byte leaves the adapter.
pub struct EntropyFsStore {
    /// The engine facade (dedups identical blobs; own durability barrier).
    engine: Engine,
    /// Explicit VOLE content identity -> engine blob id mapping.
    blobs: HashMap<StoreId, BlobId>,
    /// Declared bytes: accumulated on every put (duplicates count).
    declared_bytes: u64,
    /// Unique bytes: content-unique payload bytes (first put of an id only).
    unique_bytes: u64,
    /// Payload put operations observed (cumulative).
    put_count: u64,
}

impl EntropyFsStore {
    /// Create a fresh engine-backed store (mkfs) inside an existing, empty
    /// directory `dir`. The caller owns `dir`'s lifecycle.
    pub fn create(dir: &Path) -> Result<Self> {
        let engine =
            Engine::create(dir, &EngineOpenOptions::default()).map_err(map_engine_error)?;
        Ok(Self {
            engine,
            blobs: HashMap::new(),
            declared_bytes: 0,
            unique_bytes: 0,
            put_count: 0,
        })
    }

    /// Distinct content ids stored.
    pub fn len(&self) -> usize {
        self.blobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blobs.is_empty()
    }

    /// Close the engine handle (drains in-flight operations; the store
    /// remains on disk under the directory it was created in).
    pub fn close(&self) -> Result<()> {
        self.engine.close().map_err(map_engine_error)
    }

    /// Override the VOLE-id -> engine-blob binding for one id.
    ///
    /// Court/hostile instrumentation only (the integrity court simulates
    /// store-side corruption this way): normal operation always binds a
    /// `StoreId` to the blob the engine returns for its exact preimage.
    /// Retrieving through a wrong binding must fail the SHA-256
    /// re-verification with a typed integrity error.
    pub fn bind_blob_id(&mut self, sid: StoreId, blob: BlobId) {
        self.blobs.insert(sid, blob);
    }

    /// Engine accounting snapshot (physical bytes as reported by the engine;
    /// O(store) namespace scan).
    pub fn engine_accounting(&self) -> Result<entropyfs::engine::AccountingMetrics> {
        Ok(self.engine.metrics().map_err(map_engine_error)?.accounting)
    }

    /// Put raw bytes through the engine facade only (no VOLE accounting),
    /// returning the engine blob id. Used by the integrity court to obtain a
    /// blob whose preimage differs from a mapped VOLE id.
    pub fn engine_put(&self, bytes: &[u8]) -> Result<BlobId> {
        self.engine.put_blob(bytes).map_err(map_engine_error)
    }
}

/// Map a typed engine error onto the vole error taxonomy.
fn map_engine_error(e: entropyfs::engine::EngineError) -> Error {
    use entropyfs::engine::ErrorCode;
    match e.code() {
        ErrorCode::CorruptStore => Error::integrity(format!("entropyfs: {e}")),
        ErrorCode::NotFound => {
            Error::new(crate::error::Kind::Dependency, format!("entropyfs: {e}"))
        }
        ErrorCode::Closed => Error::unavailable(format!("entropyfs: {e}")),
        _ => Error::external(format!("entropyfs: {e}")),
    }
}

impl StoreBackend for EntropyFsStore {
    fn put(&mut self, canonical: &[u8]) -> Result<StoreId> {
        if canonical.len() as u64 > crate::limits::MAX_CHUNK_BYTES as u64 {
            return Err(Error::limit("store payload above chunk ceiling"));
        }
        let sid = StoreId::of(canonical);
        // Engine dedup: identical bytes are a no-op returning the same blob
        // id (content-addressed by BLAKE3 of the logical bytes).
        let blob = self.engine.put_blob(canonical).map_err(map_engine_error)?;
        self.declared_bytes = self.declared_bytes.saturating_add(canonical.len() as u64);
        self.put_count += 1;
        if self.blobs.insert(sid, blob).is_none() {
            self.unique_bytes = self.unique_bytes.saturating_add(canonical.len() as u64);
        }
        Ok(sid)
    }

    fn get(&self, id: &StoreId, max_bytes: u64) -> Result<Option<Vec<u8>>> {
        let Some(blob) = self.blobs.get(id) else {
            return Ok(None);
        };
        // Engine exactness gate first (BLAKE3), then the VOLE re-verification
        // (SHA-256): a storage-side mismatch is a typed integrity error, never
        // silently returned bytes.
        let bytes = match self.engine.get_blob(*blob) {
            Ok(b) => b,
            Err(e) if e.code() == entropyfs::engine::ErrorCode::NotFound => return Ok(None),
            Err(e) => return Err(map_engine_error(e)),
        };
        if bytes.len() as u64 > max_bytes {
            return Err(Error::limit("store payload exceeds retrieval bound"));
        }
        if StoreId::of(&bytes) != *id {
            return Err(Error::integrity("entropyfs payload identity mismatch"));
        }
        Ok(Some(bytes))
    }

    fn contains(&self, id: &StoreId) -> bool {
        self.blobs.contains_key(id)
    }

    fn sync(&mut self) -> Result<()> {
        // Power-durability barrier of the engine (everything acknowledged so
        // far becomes power-durable).
        self.engine.sync().map_err(map_engine_error)
    }

    fn accounting(&self) -> StorageBytes {
        // Physical bytes: the engine's own actual backing store (segment-file
        // lengths incl. engine metadata/framing). Collected lazily. A metrics
        // failure reports `None` (physical measurement unavailable) — it is
        // never substituted with logical unique bytes, which would understate
        // real backing storage.
        let physical = self.engine_accounting().ok().map(|a| a.physical_used_bytes);
        StorageBytes {
            declared_bytes: self.declared_bytes,
            unique_bytes: self.unique_bytes,
            physical_bytes: physical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_engine_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vole-entropyfs-test-{}-{}-{}",
            tag,
            std::process::id(),
            crate::evidence::timing::monotonic_raw_ns()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn engine_store_roundtrips_and_dedupes() {
        let dir = temp_engine_dir("roundtrip");
        let mut s = EntropyFsStore::create(&dir).expect("engine mkfs");
        let payload = b"canonical-literal-container".as_slice();
        let dup = b"canonical-literal-container".as_slice();
        let other = b"canonical-residual-container".as_slice();
        let id = s.put(payload).unwrap();
        assert_eq!(s.put(dup).unwrap(), id, "exact duplicates dedupe");
        let id2 = s.put(other).unwrap();
        assert_ne!(id, id2);
        assert_eq!(s.len(), 2);
        assert!(s.contains(&id));
        assert!(!s.contains(&StoreId::of(b"missing")));
        assert_eq!(s.get(&id, 1024).unwrap().as_deref(), Some(payload));
        assert_eq!(s.get(&id2, 1024).unwrap().as_deref(), Some(other));
        // Bounded retrieval enforced.
        assert!(s.get(&id, 4).is_err());
        s.sync().expect("durability barrier");
        // Accounting: declared grows on duplicates; unique does not; physical
        // is the engine's own backing store (>= unique).
        let acc = s.accounting();
        assert_eq!(acc.declared_bytes, (payload.len() * 2 + other.len()) as u64);
        assert_eq!(acc.unique_bytes, (payload.len() + other.len()) as u64);
        assert!(acc.physical_bytes.is_some() && acc.physical_bytes.unwrap() >= acc.unique_bytes);
        s.close().unwrap();
        cleanup(&dir);
    }

    #[test]
    fn engine_absent_id_returns_none() {
        let dir = temp_engine_dir("absent");
        let s = EntropyFsStore::create(&dir).expect("engine mkfs");
        assert_eq!(s.get(&StoreId::of(b"never-put"), 1024).unwrap(), None);
        s.close().unwrap();
        cleanup(&dir);
    }

    #[test]
    fn retrieval_detects_wrong_binding() {
        // Simulated store-side corruption: bind a VOLE id to a blob that does
        // NOT contain the id's preimage. Retrieval must fail the SHA-256
        // re-verification with a typed integrity error.
        let dir = temp_engine_dir("tamper");
        let mut s = EntropyFsStore::create(&dir).expect("engine mkfs");
        let sid = s.put(b"canonical-payload").unwrap();
        // The engine holds a *different* payload under its own blob id; bind
        // the VOLE id to it (simulated store-side corruption).
        let wrong = s.engine.put_blob(b"different-payload").expect("engine put");
        assert_ne!(StoreId::of(b"different-payload"), sid);
        assert_eq!(
            s.get(&sid, 1024).unwrap().as_deref(),
            Some(&b"canonical-payload"[..])
        );
        s.bind_blob_id(sid, wrong);
        let err = s.get(&sid, 1024).unwrap_err();
        assert_eq!(
            err.kind(),
            crate::error::Kind::Integrity,
            "wrong binding must be detected: {err}"
        );
        s.close().unwrap();
        cleanup(&dir);
    }

    #[test]
    fn engine_dedup_appears_once_physically() {
        // An exact shared payload occupies physical storage once: re-putting
        // identical bytes grows declared (never unique), the engine's blob
        // namespace holds one blob, and the dedup hit writes no physical
        // bytes. Absolute physical bytes are the engine's own backing store
        // (segment-file lengths incl. its metadata/framing, H.2.24) and are
        // never asserted equal to logical bytes — only the dedup-hit growth
        // of an exact duplicate is zero.
        let dir = temp_engine_dir("physical-once");
        let mut s = EntropyFsStore::create(&dir).expect("engine mkfs");
        let payload = vec![0x5au8; 4096];
        let before = s.engine_accounting().unwrap();
        let id = s.put(&payload).unwrap();
        let after_first = s.engine_accounting().unwrap();
        assert_eq!(s.put(&payload).unwrap(), id);
        let after = s.engine_accounting().unwrap();
        assert_eq!(after.blob_count - before.blob_count, 1, "one blob");
        assert!(after_first.physical_used_bytes >= before.physical_used_bytes);
        assert_eq!(
            after.physical_used_bytes - after_first.physical_used_bytes,
            0,
            "dedup hit writes no physical copy"
        );
        s.close().unwrap();
        cleanup(&dir);
    }
}
