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
//! * every receipt from the same seal — one **seal subject** (a filtered
//!   source-tree hash that excludes the evidence/governance trees
//!   `receipts/`, `target/`, `scripts/out/`, `docs/`, `.git/`, so committing
//!   receipts and ledgers can never invalidate a seal) and one battery tree
//!   (git commit / tree sha, kept as exact historical provenance);
//! * the court's verdict matches its expectation — `SUPPORTED`, `ANY`, or an
//!   **explicit allowed-verdict set** (e.g. for the Phase-I host
//!   `rocm=UNSUPPORTED_BY_HARDWARE|UNSUPPORTED_BY_API|INCONCLUSIVE`). A
//!   categorical "anything but SUPPORTED" is not a seal: a court that
//!   executes and produces corrupt samples must never satisfy one;
//! * the **verifying executable is part of the seal**: in the default mode
//!   its seal subject == the receipts' seal subject, all bound
//!   (`--historical` relaxes only the verifier requirement, and accepts
//!   pre-amendment receipts that carry no seal subject).
//!
//! Why a subject and not the git tree: receipts record the tree hash they
//! attest, but committing those receipts into that same tree changes it — a
//! strict `verifier.git_tree == receipt.git_tree` invariant can never hold
//! once evidence is committed. The subject separates **what was measured**
//! (everything capable of affecting execution) from **where the measurement
//! record was subsequently committed** (receipts/ and docs/).
//!
//! Phase rules beyond the matrix:
//!
//! * `semantic` reference hash must equal the frozen constant;
//! * `authored` reference hash must equal the frozen constant;
//! * `rocm` with a non-SUPPORTED result must still carry a *satisfied*
//!   compile surface (Phase I evidence = hardware-unavailable runtime row +
//!   a clean, bound amdgcn build — the build is not optional).

use crate::error::{Error, Result};
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
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };
    let prefix = format!("{court}-");
    // Select by the receipt's recorded creation time, not by filename order: a
    // filename sort silently depends on the run-id format and mis-orders across
    // changes to it (and across clocks). Ties break lexicographically on name.
    let mut best: Option<(std::path::PathBuf, String, i64, Value)> = None;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with(&prefix) || !name.ends_with(".json") {
            continue;
        }
        let Ok(bytes) = std::fs::read(e.path()) else {
            continue;
        };
        let Ok(env) = ReceiptEnvelope::from_json_bytes(&bytes) else {
            continue;
        };
        let Ok(value) = serde_json::to_value(env.receipt) else {
            continue;
        };
        let created = value
            .get("created_unix_ms")
            .and_then(Value::as_i64)
            .unwrap_or(i64::MIN);
        let better = match &best {
            None => true,
            Some((_, bn, bt, _)) => created > *bt || (created == *bt && name > *bn),
        };
        if better {
            best = Some((e.path(), name, created, value));
        }
    }
    Ok(best.map(|(path, _, _, value)| (path, value)))
}

/// Verify a seal over the newest receipts in `root` against `expectations`.
/// `verifier` is the running executable's environment; the default phase-seal
/// invariant is `verifier seal subject == receipt seal subject` (all bound) —
/// a materially different *source* cannot verify a seal as if it produced it,
/// while committing the receipts themselves (an excluded tree) never breaks
/// the seal. `historical` relaxes the verifier requirement (re-verifying an
/// older seal with a newer binary) and accepts pre-amendment receipts that
/// carry no seal subject. Returns the pass/fail rows; `ok` is true only when
/// every row passed.
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
    // (non-historical) mode the verifier's seal SUBJECT must ALSO equal the
    // receipt seal subject — a materially different source cannot verify a
    // seal as if it produced it. (Git-tree equality is deliberately NOT the
    // invariant: committing the receipts into the attested tree changes that
    // tree, so it could never hold once evidence is committed.)
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

    // Single-seal anchors, validated on the first receipt found and checked
    // on every other receipt:
    //  * seal_subject_hash — the identity a seal compares (excludes the
    //    evidence/governance trees, so committing receipts/docs never breaks
    //    a seal; changes only when code changes);
    //  * (git_commit, git_tree_sha) — exact historical provenance of the
    //    single clean-tree battery that produced the receipts.
    let mut seal_subject: Option<String> = None;
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
        let subject = get("seal_subject_hash");
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

        // Cross-cutting binding checks.
        if binding.as_deref() != Some("bound") {
            pass = false;
            notes.push(format!("source_binding={}", binding.unwrap_or_default()));
        }
        if build_commit != git_commit {
            pass = false;
            notes.push("compiled-from != executed-in-worktree".into());
        }

        // Seal-subject identity: one subject across every receipt of the
        // seal. Default mode also requires the verifier's subject to equal
        // it. Pre-amendment receipts (no subject) are historical-only.
        if let Some(s) = &subject {
            match &seal_subject {
                None => seal_subject = Some(s.clone()),
                Some(s0) if s0 != s => {
                    pass = false;
                    notes.push("receipts from different seal subjects".into());
                }
                _ => {}
            }
        } else if !historical {
            pass = false;
            notes.push(
                "receipt lacks seal_subject_hash (pre-amendment evidence); verify with \
                 --historical"
                    .into(),
            );
        }
        if !historical {
            match (&verifier.seal_subject_hash, &subject) {
                (Some(v), Some(s)) if v == s => {}
                (Some(_), Some(_)) => {
                    pass = false;
                    notes.push("verifier seal subject != receipt seal subject".into());
                }
                (None, Some(_)) => {
                    pass = false;
                    notes
                        .push("verifier seal subject unavailable (not built in a git tree)".into());
                }
                _ => {} // subjectless receipt already failed above.
            }
        }

        // Historical anchor: every receipt of a seal comes from one battery
        // tree (kept as exact provenance; NOT the default-mode invariant).
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

    // ---- seal-subject gate tests ----------------------------------------
    //
    // These exercise the *verifier logic* with synthetic receipts (they do
    // not run courts). The receipts are built through ReceiptBuilder so they
    // carry valid self-hashes, exactly as real evidence does.

    use crate::evidence::environment::{Environment, SourceBinding};
    use crate::evidence::receipt::{Provenance, ReceiptBuilder};
    use crate::status::Verdict;
    use std::path::{Path, PathBuf};

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vole-seal-test-{tag}-{}-{}",
            std::process::id(),
            crate::evidence::timing::monotonic_raw_ns()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A bound environment (compiled-from == executed, both clean) with an
    /// optional seal subject. Commit/tree name the battery tree; the subject
    /// is the filtered source identity.
    fn bound_env(subject: Option<&str>, commit: &str, tree: &str) -> Environment {
        Environment {
            git_commit: Some(commit.to_string()),
            git_dirty: Some(false),
            git_tree_sha: Some(tree.to_string()),
            build_git_commit: Some(commit.to_string()),
            build_dirty: Some(false),
            source_binding: SourceBinding::Bound,
            seal_subject_hash: subject.map(str::to_string),
            ..Environment::default()
        }
    }

    fn write_supported_receipt(root: &Path, court: &str, env: Environment) {
        let mut b = ReceiptBuilder::new(court);
        b.result(Verdict::Supported).environment(env);
        // Courts with frozen-hash phase rules need their provenance set.
        match court {
            "semantic" => {
                b.provenance(Provenance {
                    reference_hash: Some(FROZEN_SEMANTIC.to_string()),
                    ..Default::default()
                });
            }
            "authored" => {
                b.provenance(Provenance {
                    reference_hash: Some(FROZEN_AUTHORED.to_string()),
                    ..Default::default()
                });
            }
            _ => {}
        }
        b.finish_write(root).expect("write test receipt");
    }

    fn matrix() -> Vec<(String, Expect)> {
        vec![
            ("semantic".to_string(), Expect::Supported),
            ("authored".to_string(), Expect::Supported),
        ]
    }

    #[test]
    fn same_subject_across_commits_verifies_default_mode() {
        // The self-reference fix: receipts attest commit C / tree T, then
        // committing them moves the worktree to a later commit D with a
        // different tree — but the seal SUBJECT (filtered source identity)
        // is unchanged. A binary built at D must verify the seal in default
        // mode; only a changed subject requires --historical.
        let root = temp_root("subject-same");
        write_supported_receipt(&root, "semantic", bound_env(Some("S"), "C", "T"));
        write_supported_receipt(&root, "authored", bound_env(Some("S"), "C", "T"));
        let verifier = bound_env(Some("S"), "D", "U"); // later commit, same source
        let (ok, rows) = verify_seal(&root, &matrix(), &verifier, false).unwrap();
        assert!(ok, "same subject must verify: {rows:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn changed_subject_fails_default_mode() {
        // A materially different source (different subject) cannot verify the
        // seal in default mode — even at the same commit/tree.
        let root = temp_root("subject-diff");
        write_supported_receipt(&root, "semantic", bound_env(Some("S"), "C", "T"));
        write_supported_receipt(&root, "authored", bound_env(Some("S"), "C", "T"));
        let verifier = bound_env(Some("S2"), "D", "U");
        let (ok, rows) = verify_seal(&root, &matrix(), &verifier, false).unwrap();
        assert!(!ok, "different subject must fail: {rows:?}");
        assert!(rows.iter().all(|r| !r.pass));
        assert!(rows[0].note.as_deref().unwrap().contains("seal subject"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn receipts_with_different_subjects_are_not_one_seal() {
        let root = temp_root("subject-mixed");
        write_supported_receipt(&root, "semantic", bound_env(Some("S"), "C", "T"));
        write_supported_receipt(&root, "authored", bound_env(Some("S2"), "C", "T"));
        let verifier = bound_env(Some("S"), "C", "T");
        let (ok, rows) = verify_seal(&root, &matrix(), &verifier, false).unwrap();
        assert!(!ok);
        assert!(
            rows.iter().any(|r| r
                .note
                .as_deref()
                .is_some_and(|n| n.contains("different seal subjects"))),
            "rows: {rows:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn subjectless_receipts_are_historical_only() {
        // Pre-amendment receipts carry no seal subject: a default seal must
        // refuse them (nothing to compare), while --historical still verifies
        // them by their exact battery-tree provenance.
        let root = temp_root("subject-legacy");
        write_supported_receipt(&root, "semantic", bound_env(None, "C", "T"));
        write_supported_receipt(&root, "authored", bound_env(None, "C", "T"));
        let verifier = bound_env(Some("S"), "D", "U");
        let (ok_default, rows) = verify_seal(&root, &matrix(), &verifier, false).unwrap();
        assert!(
            !ok_default,
            "subjectless receipts must not default-verify: {rows:?}"
        );
        assert!(rows[0].note.as_deref().unwrap().contains("--historical"));
        let (ok_hist, _) = verify_seal(&root, &matrix(), &verifier, true).unwrap();
        assert!(ok_hist, "--historical must accept pre-amendment receipts");
        let _ = std::fs::remove_dir_all(&root);
    }
}
