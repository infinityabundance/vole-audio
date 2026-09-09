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
//!
//! Subject derivation (review-6 micro-hardening): entries come from the Git
//! **index** (`git ls-files -s`), not bare path listing, and each
//! non-excluded entry contributes its Git **mode** as well as its path and
//! content:
//!
//! ```text
//! mode || 0x00 || path || 0x00 || content
//! ```
//!
//! in sorted path order. Content is the worktree bytes for regular files,
//! the link-target bytes for symlinks (`120000` — the Git blob content), and
//! the pinned object id for gitlinks/submodules (`160000`, which have no
//! worktree content). A mode-only change (`100644` → `100755` on an
//! executable script) therefore changes the subject even when the bytes do
//! not — the subject identifies what git would commit, not merely what the
//! files currently contain.

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

/// One tracked index entry relevant to the seal subject.
///
/// `mode` is the 6-digit octal Git mode (`100644`, `100755`, `120000`,
/// `160000`, …) exactly as the index records it; `oid` is the index object
/// id (used as the content identity of gitlinks, which have no worktree
/// content); `path` is root-relative with posix separators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub mode: String,
    pub oid: String,
    pub path: String,
}

impl IndexEntry {
    /// True when the mode is the regular-file kind (`100xxx`); symlinks are
    /// `120000`, gitlinks `160000`.
    pub fn is_regular(&self) -> bool {
        self.mode.starts_with("100")
    }

    pub fn is_symlink(&self) -> bool {
        self.mode.starts_with("120")
    }

    pub fn is_gitlink(&self) -> bool {
        self.mode.starts_with("160")
    }
}

/// Parse one `git ls-files -s -z` record: `<mode> <oid> <stage>\t<path>`.
///
/// Fail-closed semantics: a structurally malformed record is an error, and
/// so is a non-zero stage (an unmerged index — merge/rebase/cherry-pick in
/// progress). Neither class is ever silently dropped from the subject: the
/// subject must either cover the whole index or not exist.
fn parse_entry(record: &[u8]) -> Result<IndexEntry> {
    let tab = record.iter().position(|&b| b == b'\t').ok_or_else(|| {
        Error::malformed("seal subject: malformed git index record (no path tab)")
    })?;
    let head = &record[..tab];
    let path_bytes = &record[tab + 1..];
    if path_bytes.is_empty() {
        return Err(Error::malformed(
            "seal subject: malformed git index record (empty path)",
        ));
    }
    let mut fields = head.split(|&b| b == b' ');
    let mode = String::from_utf8_lossy(fields.next().ok_or_else(|| {
        Error::malformed("seal subject: malformed git index record (missing mode)")
    })?)
    .into_owned();
    let oid = String::from_utf8_lossy(fields.next().ok_or_else(|| {
        Error::malformed("seal subject: malformed git index record (missing oid)")
    })?)
    .into_owned();
    let stage = String::from_utf8_lossy(fields.next().ok_or_else(|| {
        Error::malformed("seal subject: malformed git index record (missing stage)")
    })?)
    .into_owned();
    if stage != "0" {
        return Err(Error::malformed(format!(
            "seal subject: unmerged git index entry (stage {stage}) — an in-progress \
             merge/rebase cannot be sealed"
        )));
    }
    if fields.next().is_some() {
        return Err(Error::malformed(
            "seal subject: malformed git index record (trailing fields)",
        ));
    }
    if mode.len() != 6 || !mode.bytes().all(|b| b.is_ascii_digit()) || oid.is_empty() {
        return Err(Error::malformed(
            "seal subject: malformed git index record (bad mode/oid)",
        ));
    }
    let path = String::from_utf8_lossy(path_bytes).replace('\\', "/");
    Ok(IndexEntry { mode, oid, path })
}

/// Parse the full NUL-terminated `git ls-files -s -z` stream. Fails (never
/// silently skips) on the first malformed or unmerged record, so a subject
/// can never be computed over a partial index.
fn parse_ls_files_stream(stdout: &[u8]) -> Result<Vec<IndexEntry>> {
    let mut entries = Vec::new();
    for record in stdout.split(|&b| b == 0).filter(|s| !s.is_empty()) {
        entries.push(parse_entry(record)?);
    }
    Ok(entries)
}

/// The tracked, non-excluded index entries (root-relative posix paths) of a
/// git work tree, from `git ls-files -s`. `None` outside a git work tree.
/// Fails (never silently drops) on a malformed or unmerged index record: the
/// subject must cover the whole index or not exist.
pub fn tracked_entries(root: &Path) -> Result<Option<Vec<IndexEntry>>> {
    let out = Command::new("git")
        .args(["-C", root.to_str().unwrap_or("."), "ls-files", "-s", "-z"])
        .output()
        .map_err(|e| Error::new(Kind::Io, format!("git ls-files: {e}")))?;
    if !out.status.success() {
        // Not a git work tree (e.g. the crates.io tarball): no subject.
        return Ok(None);
    }
    let mut entries: Vec<IndexEntry> = parse_ls_files_stream(&out.stdout)?
        .into_iter()
        .filter(|e| !is_excluded(&e.path))
        .collect();
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Some(entries))
}

/// Content bytes a subject entry contributes: worktree bytes for regular
/// files, the link target for symlinks (Git stores the target as the blob
/// content), and the pinned object id for gitlinks (no worktree content).
fn entry_content(root: &Path, e: &IndexEntry) -> Result<Vec<u8>> {
    if e.is_gitlink() {
        // A submodule commit is pinned by its index oid; that IS its content
        // identity from git's perspective.
        return Ok(e.oid.as_bytes().to_vec());
    }
    let full = root.join(&e.path);
    if e.is_symlink() {
        let target = std::fs::read_link(&full)
            .map_err(|er| Error::new(Kind::Io, format!("read_link {}: {er}", full.display())))?;
        return Ok(symlink_target_bytes(&target));
    }
    std::fs::read(&full)
        .map_err(|er| Error::new(Kind::Io, format!("read {}: {er}", full.display())))
}

#[cfg(unix)]
fn symlink_target_bytes(target: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    target.as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn symlink_target_bytes(target: &Path) -> Vec<u8> {
    target.to_string_lossy().into_owned().into_bytes()
}

/// SHA-256 of the seal subject over the given index entries (see module
/// docs for the byte formula). Exposed separately so the mode/path/content
/// contribution rule is unit-testable without a git work tree.
pub fn hash_entries(root: &Path, entries: &[IndexEntry]) -> Result<String> {
    let mut hasher = Sha256::new();
    for e in entries {
        let content = entry_content(root, e)?;
        hasher.update(e.mode.as_bytes());
        hasher.update(&[0]);
        hasher.update(e.path.as_bytes());
        hasher.update(&[0]);
        hasher.update(&content);
    }
    Ok(crate::hash::sha256::hex(&hasher.finalize()))
}

/// SHA-256 of the seal subject of a git work tree: every tracked,
/// non-excluded index entry contributes `mode || NUL || path || NUL ||
/// content` in sorted path order. `None` outside a git work tree.
pub fn seal_subject_hash(root: &Path) -> Result<Option<String>> {
    let Some(entries) = tracked_entries(root)? else {
        return Ok(None);
    };
    hash_entries(root, &entries).map(Some)
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
    fn ls_files_records_parse_into_entries() {
        let rec = b"100755 0123456789abcdef0123456789abcdef01234567 0\tscripts/run.sh";
        let e = parse_entry(rec).expect("regular entry");
        assert_eq!(e.mode, "100755");
        assert_eq!(e.path, "scripts/run.sh");
        assert!(e.is_regular());
        assert!(!e.is_symlink());
        let rec = b"120000 0123456789abcdef0123456789abcdef01234567 0\tlink";
        let e = parse_entry(rec).expect("symlink entry");
        assert!(e.is_symlink());
        let rec = b"160000 0123456789abcdef0123456789abcdef01234567 0\tsub";
        let e = parse_entry(rec).expect("gitlink entry");
        assert!(e.is_gitlink());
        // Unmerged (stage != 0) and malformed records are distinct errors —
        // never silent skips.
        let e = parse_entry(b"100644 abc 1\tconflict").unwrap_err();
        assert!(e.to_string().contains("unmerged"), "{e}");
        for bad in [
            &b"bogus\tpath"[..],
            b"100644 oid\tpath",         // missing stage
            b"100644 oid 0 extra\tpath", // trailing fields
            b"100644 0123456789abcdef0123456789abcdef01234567 0\t", // empty path
            b"",
        ] {
            assert!(parse_entry(bad).is_err(), "{bad:?} must fail");
        }
    }

    #[test]
    fn unmerged_index_fails_closed() {
        // The security property: a subject cannot be produced from an
        // incomplete (unmerged/malformed) index — derivation must fail, not
        // silently drop the bad record and seal over whatever remains.
        let good = b"100644 0123456789abcdef0123456789abcdef01234567 0\tfile.rs";
        let unmerged = b"100644 0123456789abcdef0123456789abcdef01234567 2\tconflict.rs";
        let mut stream = good.to_vec();
        stream.push(0);
        stream.extend_from_slice(unmerged);
        stream.push(0);
        let err = parse_ls_files_stream(&stream).unwrap_err();
        assert!(err.to_string().contains("unmerged"), "{err}");
        // Malformed records fail the whole stream the same way.
        let mut stream = good.to_vec();
        stream.push(0);
        stream.extend_from_slice(b"not-an-index-record");
        stream.push(0);
        assert!(parse_ls_files_stream(&stream).is_err());
        // A clean stream parses fully (every record present, in order).
        let mut stream = good.to_vec();
        stream.push(0);
        stream.extend_from_slice(b"100755 abcdef0123456789abcdef0123456789abcdef01 0\tbin/run");
        stream.push(0);
        let entries = parse_ls_files_stream(&stream).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].path, "file.rs");
        assert_eq!(entries[1].path, "bin/run");
    }

    #[test]
    fn mode_changes_alter_the_subject_even_when_bytes_do_not() {
        // Review-6 micro-hardening: 100644 vs 100755 on identical bytes must
        // produce different subjects (the second is executable).
        let dir = std::env::temp_dir().join(format!(
            "vole-subject-mode-{}-{}",
            std::process::id(),
            crate::evidence::timing::monotonic_raw_ns()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("run.sh"), b"#!/bin/sh\nexit 0\n").unwrap();
        let e = |mode: &str| IndexEntry {
            mode: mode.to_string(),
            oid: "0123456789abcdef0123456789abcdef01234567".to_string(),
            path: "run.sh".to_string(),
        };
        let plain = hash_entries(&dir, &[e("100644")]).unwrap();
        let exec = hash_entries(&dir, &[e("100755")]).unwrap();
        assert_ne!(plain, exec, "a mode change must change the subject");
        assert_eq!(plain.len(), 64);
        // Identical entries hash identically.
        let again = hash_entries(&dir, &[e("100755")]).unwrap();
        assert_eq!(exec, again);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_content_is_the_link_target_not_the_target_file() {
        let dir = std::env::temp_dir().join(format!(
            "vole-subject-link-{}-{}",
            std::process::id(),
            crate::evidence::timing::monotonic_raw_ns()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("real.txt"), b"target bytes").unwrap();
        std::os::unix::fs::symlink("real.txt", dir.join("alias")).unwrap();
        let link = IndexEntry {
            mode: "120000".to_string(),
            oid: "0123456789abcdef0123456789abcdef01234567".to_string(),
            path: "alias".to_string(),
        };
        // Git stores the link TARGET as the blob content — never the bytes
        // of the pointed-at file.
        let content = entry_content(&dir, &link).unwrap();
        assert_eq!(content, b"real.txt");
        let pointed = IndexEntry {
            mode: "100644".to_string(),
            oid: "x".to_string(),
            path: "real.txt".to_string(),
        };
        assert_eq!(entry_content(&dir, &pointed).unwrap(), b"target bytes");
        assert_ne!(
            content, b"target bytes",
            "symlink subject != pointed-at content"
        );
        let _ = std::fs::remove_dir_all(&dir);
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
        let entries = tracked_entries(&root).unwrap().unwrap();
        assert!(entries.iter().any(|e| e.path.starts_with("src/")));
        assert!(!entries.iter().any(|e| e.path.starts_with("receipts/")));
        assert!(!entries.iter().any(|e| e.path.starts_with("docs/")));
        // Regular tracked files carry modes (100644/100755 here).
        assert!(entries.iter().any(|e| e.is_regular()));
    }
}
