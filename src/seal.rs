//! `vole-audio seal verify` — the executable phase-seal gate.
//!
//! `court-all.sh` collects evidence (every court exits 0 after writing a
//! receipt, whatever its verdict — correct for a collector, insufficient for
//! a seal). This module turns a seal into executable evidence: it validates
//! an explicit expected-verdict matrix over the newest receipt of each named
//! court and fails the seal on any deviation.
//!
//! For every receipt it requires:
//!
//! * `source_binding == bound` (compiled-from == executed-in-worktree, both
//!   explicitly clean);
//! * every receipt from the same seal tree (one git commit / tree sha);
//! * the court's verdict matches its expectation (`SUPPORTED`, `NEGATIVE` —
//!   any honest non-SUPPORTED label — or `ANY`).
//!
//! Phase rules beyond the matrix:
//!
//! * `semantic` reference hash must equal the frozen constant;
//! * `authored` reference hash must equal the frozen constant;
//! * `rocm` with a non-SUPPORTED result must still carry a *satisfied*
//!   compile surface (Phase I evidence = hardware-unavailable runtime row +
//!   a clean, bound amdgcn build — the build is not optional).

use crate::error::{Error, Kind, Result};
use crate::evidence::receipt::ReceiptEnvelope;
use serde_json::Value;
use std::path::Path;

/// Frozen semantic-court reference hash (must never change; a profile change
/// is a new universe, never a silent edit).
pub const FROZEN_SEMANTIC: &str =
    "1791816f4b938375cc4298b2587ce19eef260d063d9ed88c597e04d31837f6d0";
/// Frozen authored-court reference hash (Phase F correctness re-freeze).
pub const FROZEN_AUTHORED: &str =
    "f7e103f3a97d5fafd6988e3bb6551b3c3af62a2e65c6a82898d6c2d4aff9d6db";

/// Verdict expectation for one court row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    Supported,
    /// Any honest non-SUPPORTED label (typed causes recorded in the receipt).
    Negative,
    /// Any verdict.
    Any,
}

#[derive(Debug, Clone)]
pub struct Row {
    pub court: String,
    pub expected: String,
    pub got: String,
    pub pass: bool,
    pub note: Option<String>,
}

/// Parse a `court=EXPECT,...` matrix string. Recognized expectations:
/// `SUPPORTED`, `NEGATIVE`, `ANY`.
pub fn parse_expectations(s: &str) -> Result<Vec<(String, Expect)>> {
    let mut out = Vec::new();
    for tok in s.split(',').filter(|t| !t.is_empty()) {
        let (court, exp) = tok.split_once('=').ok_or_else(|| {
            Error::malformed(format!("seal expectation '{tok}': expected court=EXPECT"))
        })?;
        let e = match exp.to_ascii_uppercase().as_str() {
            "SUPPORTED" => Expect::Supported,
            "NEGATIVE" => Expect::Negative,
            "ANY" => Expect::Any,
            other => {
                return Err(Error::malformed(format!(
                    "seal expectation '{court}={other}': expected SUPPORTED | NEGATIVE | ANY"
                )));
            }
        };
        out.push((court.trim().to_string(), e));
    }
    Ok(out)
}

fn newest_receipt(root: &Path, court: &str) -> Result<Option<(std::path::PathBuf, Value)>> {
    let dir = root.join(court);
    let mut best: Option<(std::path::PathBuf, String)> = None; // (path, name)
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with(&format!("{court}-")) || !name.ends_with(".json") {
            continue;
        }
        match &best {
            None => best = Some((e.path(), name.clone())),
            Some((_, bn)) if name > *bn => best = Some((e.path(), name)),
            _ => {}
        }
    }
    match best {
        Some((path, _)) => {
            let bytes = std::fs::read(&path)
                .map_err(|e| Error::new(Kind::Io, format!("read {}: {e}", path.display())))?;
            let env: ReceiptEnvelope = ReceiptEnvelope::from_json_bytes(&bytes)?;
            let value = serde_json::to_value(env.receipt)
                .map_err(|e| Error::new(Kind::Io, format!("serialize {}: {e}", path.display())))?;
            Ok(Some((path, value)))
        }
        None => Ok(None),
    }
}

/// Verify a seal over the newest receipts in `root` against `expectations`.
/// Returns the pass/fail rows; `ok` is true only when every row passed.
pub fn verify_seal(root: &Path, expectations: &[(String, Expect)]) -> Result<(bool, Vec<Row>)> {
    let mut rows: Vec<Row> = Vec::new();
    let mut ok = true;

    // Single seal tree + per-receipt binding are cross-cutting: validate on
    // the first receipt found and check every other receipt agrees.
    let mut seal_tree: Option<(String, String)> = None; // (git_commit, git_tree_sha)

    for (court, expect) in expectations {
        let label = |e: &Expect| match e {
            Expect::Supported => "SUPPORTED",
            Expect::Negative => "NEGATIVE",
            Expect::Any => "ANY",
        };
        let Some((path, r)) = newest_receipt(root, court)? else {
            ok = false;
            rows.push(Row {
                court: court.clone(),
                expected: label(expect).to_string(),
                got: "NO_RECEIPT".into(),
                pass: false,
                note: Some(format!("no receipt under {}", root.join(court).display())),
            });
            continue;
        };
        let _ = path;
        let env = r.get("environment").cloned().unwrap_or_default();
        let get = |k: &str| env.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let binding = get("source_binding");
        let git_commit = get("git_commit");
        let git_tree = get("git_tree_sha");
        let build_commit = get("build_git_commit");
        let result = r
            .get("result")
            .and_then(|v| v.as_str())
            .unwrap_or("MISSING")
            .to_string();

        let mut notes: Vec<String> = Vec::new();
        let mut pass = true;

        // Cross-cutting binding/tree checks.
        if binding.as_deref() != Some("bound") {
            pass = false;
            notes.push(format!("source_binding={}", binding.unwrap_or_default()));
        }
        if build_commit != git_commit {
            pass = false;
            notes.push("compiled-from != executed-in-worktree".into());
        }
        if let (Some(c), Some(t)) = (&git_commit, &git_tree) {
            match &seal_tree {
                None => seal_tree = Some((c.clone(), t.clone())),
                Some((c0, t0)) => {
                    if c0 != c || t0 != t {
                        pass = false;
                        notes.push("receipts from different seal trees".into());
                    }
                }
            }
        } else {
            pass = false;
            notes.push("missing git identity".into());
        }

        // Verdict expectation.
        let verdict_ok = match expect {
            Expect::Supported => result == "SUPPORTED",
            Expect::Negative => result != "SUPPORTED",
            Expect::Any => true,
        };
        if !verdict_ok {
            pass = false;
            notes.push(format!("verdict {result} != {}", label(expect)));
        }

        // Phase rules.
        if court == "semantic" {
            let h = r
                .pointer("/provenance/reference_hash")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if h != FROZEN_SEMANTIC {
                pass = false;
                notes.push("semantic reference hash != frozen constant".into());
            }
        }
        if court == "authored" {
            let h = r
                .pointer("/provenance/reference_hash")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if h != FROZEN_AUTHORED {
                pass = false;
                notes.push("authored reference hash != frozen constant".into());
            }
        }
        if court == "rocm" && result != "SUPPORTED" {
            let satisfied = r
                .pointer("/extras/compile_surface/satisfied")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !satisfied {
                pass = false;
                notes.push("rocm compile surface not satisfied".into());
            }
            if r.pointer("/provenance/gpu_artifact_hash").is_none() {
                pass = false;
                notes.push("rocm artifact hash missing".into());
            }
        }

        ok &= pass;
        rows.push(Row {
            court: court.clone(),
            expected: label(expect).to_string(),
            got: result,
            pass,
            note: if notes.is_empty() {
                None
            } else {
                Some(notes.join("; "))
            },
        });
    }
    Ok((ok, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expectation_parsing() {
        let e = parse_expectations("semantic=SUPPORTED,rocm=NEGATIVE,h2=ANY").unwrap();
        assert_eq!(e.len(), 3);
        assert_eq!(e[0], ("semantic".to_string(), Expect::Supported));
        assert_eq!(e[1], ("rocm".to_string(), Expect::Negative));
        assert_eq!(e[2], ("h2".to_string(), Expect::Any));
        assert!(parse_expectations("cuda=MAYBE").is_err());
        assert!(parse_expectations("naked").is_err());
    }

    #[test]
    fn frozen_hashes_match_the_courts() {
        // Guard: the seal gate's frozen constants must equal the constants
        // the courts enforce, or the gate would check the wrong thing.
        assert_eq!(
            FROZEN_SEMANTIC,
            crate::courts::semantic::SEMANTIC_COURT_REFERENCE_SHA256
        );
        assert_eq!(
            FROZEN_AUTHORED,
            crate::courts::authored::AUTHORED_COURT_REFERENCE_SHA256
        );
    }
}
