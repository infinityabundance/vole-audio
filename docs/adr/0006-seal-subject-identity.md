# ADR 0006 — Seal subject identity is the filtered source tree, not the git tree

- Status: accepted (Phase I review-5 closure)
- Owner: `src/evidence/subject.rs`, `src/evidence/environment.rs`,
  `src/seal.rs`, `docs/PHASE_I.md` (Seal 6), `docs/EVIDENCE.md`

## Context

Receipts record the git commit and tree hash of the tree they attest. But
committing those receipts — and the phase ledger describing them — into that
same tree changes the tree the receipts name. A strict default invariant
`verifier.git_tree == receipt.git_tree` can therefore never hold at the
release head: the evidence commit, the ledger append, and the version bump
all move the tree past the attested one, forcing every verification to run
`--historical` even when nothing about the source changed.

That is a self-reference problem in the evidence architecture, not a property
of the audio work. It recurs after every phase unless the *identity being
sealed* is separated from the *location where the measurement record is
subsequently committed*.

## Decision

Each receipt's `environment` carries a new optional field,
`seal_subject_hash`:

```text
SHA-256 over, in sorted path order, every tracked non-excluded index entry
(git ls-files -s), of:

    git mode || 0x00 || path || 0x00 || content

mode      the 6-digit octal Git mode (100644 / 100755 / 120000 / 160000 …)
path      root-relative posix path
content   worktree bytes for regular files; the link-target bytes for
          symlinks (120000 — Git stores the target as the blob content);
          the pinned object id for gitlinks/submodules (160000, which have
          no worktree content)
```

where the excluded trees are the ones that only record or govern evidence
and cannot affect execution:

```text
receipts/**     evidence outputs (committed after every battery)
target/**       build outputs
scripts/out/**  generated device artifacts + sidecars
docs/**         governance: specs, charters, seal ledgers, ADRs
.git/**         git internals
```

`git_commit` and `git_tree_sha` remain in every receipt as exact historical
provenance of the battery tree, but they are no longer the default-mode seal
invariant.

The phase-seal invariant becomes:

```text
verifier.seal_subject_hash == receipt.seal_subject_hash
```

(all receipts also share one subject and one battery tree). `--historical`
still relaxes only the verifier requirement and additionally accepts
pre-amendment receipts that carry no subject.

Because `Cargo.toml`/`Cargo.lock` are part of the subject, the release
version is bumped **before** the clean-tree seal battery runs; the release
head then verifies the sealed subject without `--historical`.

## Consequences

- Committing receipts, ledger entries, and other docs can no longer
  invalidate a seal; only a change to a file that can affect execution does.
- A seal now means "this exact source content produced these receipts",
  which is the property that actually matters for reproducibility.
- Review-6 micro-hardening: the subject is derived from Git **index**
  entries and includes the Git **mode**, so a mode-only change (`100644` →
  `100755`) alters the subject even when bytes do not, and symlinks/gitlinks
  have well-defined identity (link-target blob / pinned oid).
- Pre-amendment archived seals (Seal 5 and earlier, subjectless) remain
  verifiable with `--historical`; default mode refuses them with an explicit
  message rather than silently falling back to tree equality.
- `vole-audio seal subject` prints the current subject; `vole-audio version`
  shows it alongside the binding state.
