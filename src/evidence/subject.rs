//! Seal-subject identity: a filtered hash of the source tree that evidence
//! outputs cannot disturb.
//!
//! The self-reference problem: receipts record the tree hash they attest,
//! but committing those receipts into the same tree changes the tree — so a
//! strict `verifier.git_tree == receipt.git_tree` invariant can never hold
//! once evidence is committed. The fix is to separate **what was measured**
//! from **where the measurement record was subsequently committed**:
//!
//! * `git_commit` / `git_tree_sha` stay in every receipt as exact historical
//!   provenance;
//! * `seal_subject_hash` is the identity of everything *capable of affecting
//!   execution*, deliberately excluding the trees that only record or govern
//!   it:
//!
//! ```text
//! receipts/**     evidence outputs (committed after every battery)
//! target/**       build outputs
//! scripts/out/**  generated device artifacts + sidecars
//! docs/**         governance: specs, charters, seal ledgers — they guide
//!                 but never execute (semantic authority is code, which is
//!                 included)
//! ```
//!
//! The seal invariant therefore becomes
//!
//! ```text
//! verifier.seal_subject_hash == receipt.seal_subject_hash
//! ```
//!
//! which survives the receipts commit and the ledger append, and changes
//! only when a file that can affect execution changes (source, manifest,
//! build script, assets) — which is exactly when a new seal is required.
//! Version bumps and release metadata live in the subject (`Cargo.toml`/
//! `Cargo.lock` are included), so they must be made **before** the seal
//! battery runs; then the release head still verifies the sealed subject
//! without `--historical`.

use crate::error::{Error, Kind, Result};
use crate::hash::sha256::Sha256;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Root-relative prefixes excluded from the seal subject (see module docs).
/// `docs/` is excluded as non-executing governance; every *normative*
/// statement it makes is enforced by code under `src/`, which is included.
pub const SEAL_SUBJECT_EXCLUDES: &[&str] =
    &["receipts/", "target/", "scripts/out/", "docs/", ".git/"];

/// True when a root-relative path (posix separators) is excluded.
pub fn is_excluded(rel: &str) -> bool {
    SEAL_SUBJECT_EXCLUDES.iter().any(|p| rel.starts_with(p))
}

/// The sorted list of tracked, non-excluded file paths (root-relative,
/// posix separators), from `git ls-files`. `None` outside a git work tree.
pub fn tracked_source_files(root: &Path) -> Result<Option<Vec<String>>> {
    let out = Command::new("git")
        .args(["-C", root.to_str().unwrap_or("."), "ls-files", "-z"])
        .output()
        .map_err(|e| Error::new(Kind::Io, format!("git ls-files: {e}")))?;
    if !out.status.success() {
        // Not a git work tree (e.g. the crates.io tarball): no subject.
        return Ok(None);
    }
    let mut files: Vec<String> = out
        .stdout
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
        .filter(|p| !is_excluded(p))
        .collect();
    files.sort();
    Ok(Some(files))
}

/// SHA-256 of the seal subject: each non-excluded tracked file contributes
/// its root-relative path followed by its content bytes, in sorted path
/// order. `None` outside a git work tree.
pub fn seal_subject_hash(root: &Path) -> Result<Option<String>> {
    let Some(files) = tracked_source_files(root)? else {
        return Ok(None);
    };
    let mut hasher = Sha256::new();
    for rel in &files {
        let full = root.join(rel);
        let bytes = std::fs::read(&full)
            .map_err(|e| Error::new(Kind::Io, format!("read {}: {e}", full.display())))?;
        hasher.update(rel.as_bytes());
        hasher.update(&[0]);
        hasher.update(&bytes);
    }
    Ok(Some(crate::hash::sha256::hex(&hasher.finalize())))
}

/// Resolve the repository root of the current directory (best effort).
pub fn repo_root() -> Option<PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusion_roots_are_exact_prefixes() {
        assert!(is_excluded("receipts/rocm/rocm-1.json"));
        assert!(is_excluded("target/amdgcn/debug/x.rlib"));
        assert!(is_excluded("scripts/out/vole_audio.ptx"));
        assert!(is_excluded("docs/U1_SPEC.md"));
        assert!(is_excluded("docs/PHASE_I.md"));
        assert!(!is_excluded("src/lib.rs"));
        assert!(!is_excluded("Cargo.toml"));
        assert!(!is_excluded("assets/u1/resampler.bin"));
        assert!(!is_excluded("receipts-note.md")); // prefix must include '/'
        assert!(!is_excluded("scripts/build-rocm-device.sh"));
    }

    #[test]
    fn subject_is_deterministic_inside_the_repo() {
        let Some(root) = repo_root() else {
            eprintln!("skipping: not a git work tree");
            return;
        };
        let a = seal_subject_hash(&root).unwrap().expect("subject in repo");
        let b = seal_subject_hash(&root).unwrap().expect("subject in repo");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        // Code is in the subject; evidence docs are not.
        let files = tracked_source_files(&root).unwrap().unwrap();
        assert!(files.iter().any(|f| f.starts_with("src/")));
        assert!(!files.iter().any(|f| f.starts_with("receipts/")));
        assert!(!files.iter().any(|f| f.starts_with("docs/")));
    }
}
