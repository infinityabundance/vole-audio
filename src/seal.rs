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
//! * the court's verdict matches its expectation — `SUPPORTED`, `ANY`, or an
//!   **explicit allowed-verdict set** (e.g. for the Phase-I host
//!   `rocm=UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE`). A
//!   categorical "anything but SUPPORTED" is not a seal: a court that
//!   executes and produces corrupt samples must never satisfy one;
//! * the **verifying executable is part of the seal**: in the default mode
//!   its build tree == its worktree == the receipt seal tree, all bound
//!   (`--historical` relaxes only the verifier-equality requirement).
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

/// Verdict expectation for one court row. A seal must state what it will
/// accept; categorical "anything but SUPPORTED" is not a seal (a court that
/// executes and produces corrupt samples must never satisfy a seal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expect {
    Supported,
    /// Any verdict.
    Any,
    /// Exactly these verdict labels (e.g. rocm =
    /// UNSUPPORTED_BY_HARDWARE | UNSUPPORTED_BY_API | INCONCLUSIVE for the
    /// Phase-I host) — never FAILED_CORRECTNESS / FAILED_DEADLINE /
    /// FELL_BACK_TO_D0 / NOT_IMPLEMENTED unless a phase explicitly allows
    /// one of them.
    Allowed(Vec<String>),
}

/// The verdict vocabulary a seal expectation may name (must match
/// `status::Verdict::label` exactly — typos fail parsing).
pub const VERDICT_VOCABULARY: &[&str] = &[
    "SUPPORTED",
    "UNSUPPORTED_BY_API",
    "UNSUPPORTED_BY_HARDWARE",
    "UNSUPPORTED_BY_TOPOLOGY",
    "FAILED_CORRECTNESS",
    "FAILED_DEADLINE",
    "FELL_BACK_TO_D0",
    "INCONCLUSIVE",
    "NOT_APPLICABLE",
    "NOT_IMPLEMENTED",
];

impl Expect {
    /// Human-readable expectation, e.g. "SUPPORTED" or
    /// "UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE".
    pub fn display(&self) -> String {
        match self {
            Expect::Supported => "SUPPORTED".to_string(),
            Expect::Any => "ANY".to_string(),
            Expect::Allowed(v) => v.join("|"),
        }
    }

    fn accepts(&self, got: &str) -> bool {
        match self {
            Expect::Supported => got == "SUPPORTED",
            Expect::Any => true,
            Expect::Allowed(v) => v.iter().any(|a| a == got),
        }
    }
}

/// Parse a `court=EXPECT,...` matrix string. Expectations: `SUPPORTED`,
/// `ANY`, or an explicit pipe-joined allowed-verdict list whose members must
/// be in [`VERDICT_VOCABULARY`] (e.g.
/// `rocm=UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE`).
pub fn parse_expectations(s: &str) -> Result<Vec<(String, Expect)>> {
    let mut out = Vec::new();
    for tok in s.split(',').filter(|t| !t.is_empty()) {
        let (court, exp) = tok.split_once('=').ok_or_else(|| {
            Error::malformed(format!("seal expectation '{tok}': expected court=EXPECT"))
        })?;
        let upper = exp.to_ascii_uppercase();
        let e = match upper.as_str() {
            "SUPPORTED" => Expect::Supported,
            "ANY" => Expect::Any,
            _ => {
                let labels = upper.split('|').collect::<Vec<_>>();
                for l in &labels {
                    if !VERDICT_VOCABULARY.contains(l) {
                        return Err(Error::malformed(format!(
                            "seal expectation '{court}={exp}': '{l}' is not a verdict label \
                             (allowed: {} | SUPPORTED | ANY)",
                            VERDICT_VOCABULARY.join(" | ")
                        )));
                    }
                }
                Expect::Allowed(labels.iter().map(|l| (*l).to_string()).collect())
            }
        };
        out.push((court.trim().to_string(), e));
    }
    Ok(out)
}

/// One validated seal row.
#[derive(Debug, Clone)]
pub struct Row {
    pub court: String,
    pub expected: String,
    pub got: String,
    pub pass: bool,
    pub note: Option<String>,
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
/// `verifier` is the running executable's environment; the default phase-seal
/// invariant is `verifier build tree == verifier worktree == receipt seal
/// tree` (all bound). `historical` relaxes only the verifier-equality
/// requirement (for re-verifying an older seal with a newer binary). Returns
/// the pass/fail rows; `ok` is true only when every row passed.
pub fn verify_seal(
    root: &Path,
    expectations: &[(String, Expect)],
    verifier: &crate::evidence::environment::Environment,
    historical: bool,
) -> Result<(bool, Vec<Row>)> {
    let mut rows: Vec<Row> = Vec::new();
    let mut ok = true;

    // Cross-cutting facts: the verifier itself must be bound to its worktree
    // (compiled-from == executed, both clean) for any seal; in the default
    // (non-historical) mode the verifier tree must ALSO equal the receipt
    // seal tree — a materially different binary cannot verify a seal as if
    // it produced it.
    let mut verifier_note = String::new();
    if !verifier.source_bound() {
        verifier_note = format!("verifier not bound ({})", verifier.source_binding().label());
        // Historical mode intentionally relaxes the verifier requirement
        // (verifying an older seal with a newer or dirty binary); only the
        // default phase-seal mode fails on it.
        if !historical {
            ok = false;
        }
    }

    // Single seal tree + per-receipt binding are cross-cutting: validate on
    // the first receipt found and check every other receipt agrees.
    let mut seal_tree: Option<(String, String)> = None; // (git_commit, git_tree_sha)

    for (court, expect) in expectations {
        let Some((path, r)) = newest_receipt(root, court)? else {
            ok = false;
            rows.push(Row {
                court: court.clone(),
                expected: expect.display(),
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
            // Default phase-seal invariant: verifier == receipt seal tree.
            if !historical
                && (verifier.git_commit.as_deref() != Some(c.as_str())
                    || verifier.git_tree_sha.as_deref() != Some(t.as_str()))
            {
                pass = false;
                notes.push(
                    "verifier tree != receipt seal tree (default mode; use --historical \
                     to verify an older seal with a newer binary)"
                        .into(),
                );
            }
        } else {
            pass = false;
            notes.push("missing git identity".into());
        }
        if !verifier_note.is_empty() && !historical {
            pass = false;
            notes.push(verifier_note.clone());
        }

        // Verdict expectation (explicit allowed set; never a categorical
        // "anything but SUPPORTED").
        if !expect.accepts(&result) {
            pass = false;
            notes.push(format!("verdict {result} not in {{{}}}", expect.display()));
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
            expected: expect.display(),
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
        let e = parse_expectations(
            "semantic=SUPPORTED,rocm=UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE,\
             h2=ANY",
        )
        .unwrap();
        assert_eq!(e.len(), 3);
        assert_eq!(e[0], ("semantic".to_string(), Expect::Supported));
        assert_eq!(
            e[1],
            (
                "rocm".to_string(),
                Expect::Allowed(vec![
                    "UNSUPPORTED_BY_HARDWARE".into(),
                    "UNSUPPORTED_BY_API".into(),
                    "INCONCLUSIVE".into()
                ])
            )
        );
        assert_eq!(e[2], ("h2".to_string(), Expect::Any));
        assert!(parse_expectations("cuda=MAYBE").is_err()); // not a verdict
        assert!(parse_expectations("cuda=FAILED_CORRECTNESS").is_ok()); // explicit allowed
        assert!(parse_expectations("naked").is_err());
    }

    #[test]
    fn allowed_sets_never_accept_corruption_verdicts_by_default() {
        // The Phase-I default rocm set must never contain the corruption/
        // execution-failure classes: a court that ran and produced corrupt
        // samples must not satisfy the seal.
        let e = parse_expectations("rocm=UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE")
            .unwrap()[0]
            .1
            .clone();
        assert!(e.accepts("UNSUPPORTED_BY_HARDWARE"));
        assert!(e.accepts("INCONCLUSIVE"));
        assert!(!e.accepts("FAILED_CORRECTNESS"));
        assert!(!e.accepts("FAILED_DEADLINE"));
        assert!(!e.accepts("FELL_BACK_TO_D0"));
        assert!(!e.accepts("NOT_IMPLEMENTED"));
        assert!(!e.accepts("SUPPORTED"));
    }

    #[test]
    fn allowed_sets_members_must_be_real_verdicts() {
        assert!(parse_expectations("rocm=UNSUPPORTED_BY_HARDWARE|BOGUS").is_err());
        assert!(parse_expectations("rocm=INCONCLUSIVE").is_ok());
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
