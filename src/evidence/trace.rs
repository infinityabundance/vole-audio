//! Raw trace recording with incremental hashing.
//!
//! Traces are raw observation/measurement byte streams (not summary JSON).
//! Each trace is hashed with SHA-256 as it is written so receipts can bind a
//! trace to a digest without holding the whole stream in memory.

use crate::hash::sha256::Sha256;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// A trace sink that writes bytes to a file and hashes them incrementally.
pub struct TraceFile {
    file: Option<File>,
    path: PathBuf,
    hasher: Sha256,
    bytes_written: u64,
    finished: bool,
}

impl TraceFile {
    /// Open a trace file for append-style sequential writes.
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            file: Some(file),
            path: path.to_path_buf(),
            hasher: Sha256::new(),
            bytes_written: 0,
            finished: false,
        })
    }

    /// Create under `dir` with the given file name.
    pub fn create_in(dir: &Path, name: &str) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Self::create(&dir.join(name))
    }

    pub fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
        debug_assert!(!self.finished, "trace already finalized");
        self.file.as_mut().expect("file open").write_all(bytes)?;
        self.hasher.update(bytes);
        self.bytes_written += bytes.len() as u64;
        Ok(())
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush, close, and return the SHA-256 of everything written.
    pub fn finalize(mut self) -> io::Result<[u8; 32]> {
        if let Some(mut f) = self.file.take() {
            f.flush()?;
            f.sync_all()?;
        }
        self.finished = true;
        Ok(self.hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256::Sha256;

    #[test]
    fn trace_hash_matches_plain_hash() {
        let dir = std::env::temp_dir().join(format!("vole-trace-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("trace.bin");
        let mut t = TraceFile::create(&path).expect("create");
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 256) as u8).collect();
        for chunk in data.chunks(1000) {
            t.write(chunk).expect("write");
        }
        let digest = t.finalize().expect("finalize");
        let expect = Sha256::digest(&data);
        assert_eq!(digest, expect);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
