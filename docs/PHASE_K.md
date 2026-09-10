# PHASE K — inverse compiler

## Mission

Implement the paper's **bounded inverse proceduralization**: given an observed
bounded sample window, find the cheapest **exact** deterministic `SampleObject`
explanation. Deliverables (implementation contract §33/§35/§46):

* bounded candidates;
* a deterministic Pareto frontier;
* exact acceptance;
* complete dependency accounting.

The forward architecture answers *"how do I materialize this SampleObject
exactly?"*; Phase K answers the dual question *"what is the cheapest exact
SampleObject that explains these observed samples?"*. The owner of the
candidate vocabulary, accounting rules and frontier semantics is
[`INVERSE.md`](INVERSE.md); this file is the phase ledger.

## Claim boundary (what Phase K does NOT claim)

* **No optimality.** The compiler reports the best of a bounded, documented
  candidate set — never "the" explanation.
* **No lossy/perceptual acceptance.** A candidate is accepted only when two
  independent reconstructions are exactly equal to the window.
* **No semantic authority.** The scalar evaluator remains authority; the
  inverse compiler proposes and the evaluator verifies.
* **No decoder dependency.** Decoding never invokes the search.
* **No GPU search.** Parallel candidate sweeps are Phase L; the Phase-K
  compiler is host-only and deterministic.
* **No invented candidate families.** Delta/linear-predictor and
  partial/harmonic hypotheses require a residual-model vocabulary extension
  beyond the frozen u1 v1 models; they are explicitly deferred (see
  `INVERSE.md` §2) rather than approximated.

## Work items (executed in order)

1. `src/inverse/observe.rs` — identity-voice scalar observation, bounded window
   observation, and intrinsic reconstruction for every candidate class.
2. `src/inverse/cost.rs` — complete cost via the H.2 complete-cost oracle
   (`entropy_literal` / `entropy_residual`) and canonical object bytes;
   abstract universe work.
3. `src/inverse/propose.rs` — bounded deterministic proposals: literal,
   silence, constant, exact-repeat (minimal frame period), residual
   zero/constant/periodic (bounded scan), shared reference; reference library.
4. `src/inverse/frontier.rs` — deterministic Pareto frontier over static
   objectives with a validity self-check.
5. `src/inverse/mod.rs` — `Intrinsic`, `Candidate`, `Acceptance`,
   `SearchReport`, `SearchBudget`, `compile()` (proposal → exact acceptance →
   pricing → frontier).
6. `src/courts/inverse.rs` — `court inverse`: explanation search over the frozen
   H.2 corpus window, exactness/frontier/negative-control/determinism gates,
   archive-dedup measurement, procedural-library reference demonstration, and a
   frozen static-result hash.
7. `src/courts/flattening.rs` — `court flattening`: host flat-evaluator parity
   (`flat == scalar` bit-for-bit) over the frozen fixtures and an adversarial
   battery, with honest residual-closure materialization and upload accounting.
8. `docs/INVERSE.md` — the inverse-compiler contract (candidate families,
   accounting rules, frontier semantics, non-claims).
9. Registry/CLI/seal wiring: both courts registered in `courts::COURT_NAMES`,
   dispatched from `main.rs`, added to the default seal expectation matrix and
   to `scripts/court-all.sh`.
10. **Evidence-integrity fix (found while sealing Phase K).** `court inverse`
    is the first float-bearing receipt in the seal matrix, which exposed two
    latent receipt self-hash defects:
    * `serde_json`'s default `f64` parser is not correctly rounded
      (`0.9898596181369713` parsed 1 ULP low), so a receipt containing such a
      float could not reproduce its own self-hash — the feature
      `float_roundtrip` is now required (it is load-bearing, not an
      optimization);
    * the self-hash was computed over the *typed* struct, so every receipt
      written before an additive schema field stopped re-verifying the moment
      that field was added (180 of 348 archived receipts failed
      `receipt show`). `ReceiptEnvelope::from_json_bytes` now hashes the
      `receipt` value **exactly as the file carries it**
      (`preserve_order`), so the hash is a pure function of the stored bytes
      and is stable under additive schema growth.
    With both fixes, all 348 archived receipts verify (`receipt show`),
    including the 180 that previously failed. Documented in
    `docs/EVIDENCE.md`;
    regression tests: `float_heavy_extras_roundtrip_byte_exactly`,
    `receipts_written_before_an_additive_field_still_verify`,
    `tampered_receipt_body_is_rejected`, `canonical_forms_agree_for_the_pointer_style`.

## Seal history

### Seal 1 — Phase K implementation + clean-tree battery (2026-09-10)

Seal run (release, `--all-features`, clean tree at the implementation commit,
version 0.8.0):

- `court inverse` SUPPORTED: 14 fixtures (the frozen H.2 corpus truncated to a
  4096-frame inverse window), every fixture's `Literal` fallback accepted,
  **11/14** fixtures explained more cheaply by a non-literal candidate
  (`silence` 46 B, `dc`/`dc-negative` 50 B, `single-sine` 574 B at an exact
  period of 64, `impulse-train` 224 B, `transient-heavy` 601 B, `am-signal`
  6228 B at an exact period of 128, `quasi-periodic` 18334 B):
  - negative controls (`white-noise`, `random-control`, `scrambled-control`)
    are never compressed by a hypothesis (0.954–0.990 of the literal floor);
  - `harmonic-tone`, `fm-signal`, `stereo-correlated` are cheapest as
    entropy-coded literals — an honest negative;
  - the frontier is a valid Pareto set for every fixture and every accepted
    candidate is either a member or dominated;
  - a second compile per fixture is statically identical;
  - archive dedup accepts an exact shared reference (32 dependency bytes, 0
    persistent bytes) for every fixture; a procedural (`ExactRepeat`) library
    object is referenceable;
  - frozen static-result hash
    `9effb3c31b0b0fb0aa224084b3aa82820981ae3c2e89cbac769b0463f3de0a3a`.
- `court flattening` SUPPORTED: 254 worlds (frozen semantic fixture, frozen
  authored fixture, 252 adversarial battery worlds), 764 windows compared,
  `flat == scalar` bit-for-bit on every window; 84 336 bytes of residual
  closure materialized host-side and reported (never hidden).
- All pre-existing courts SUPPORTED with frozen hashes unchanged (semantic
  `1791816f4b93…`, authored `f7e103f3a97d…`); no device artifact changes (the
  Phase-K work is host-only), so the PTX (`8b23325d…`) and AMDGPU
  (`5092e129…`) artifacts are rebuilt only to re-bind their provenance
  sidecars to the new tree.
- Host tests: **363 passed, 5 ignored** all-features (353 passed, 5 ignored
  default-features; +17 over Phase J Seal 3: the inverse observe/cost/propose/
  frontier unit battery, the inverse-court gates, the flattening battery, and
  the receipt canonicalization regression tests); clippy `-D warnings` and
  `cargo fmt --check` clean.
- `vole-audio seal verify` PASSes on the 12-row matrix in **default mode** at
  the battery tree and (after rebuilding from the release head) at the head
  itself; every receipt `source_binding: bound` and carries one seal subject.

## Execution record (implementation summary)

- The inverse compiler is `std`-gated host code over the shared `no_std`
  semantic core; it adds no device code and no new semantic vocabulary.
- Candidate pricing reuses the frozen H.2 complete-cost API, so a
  representation can never cost two different things in two places.
- Both courts run without any GPU or audio hardware, so the Phase-K seal is
  fully reproducible on a CPU-only machine.
- Evidence integrity: all 348 archived receipts verify after the
  canonicalization fix (work item 10); the fix is additive for existing
  receipts (the producer already hashed the same encoding) and makes the
  README's `receipt show <file>` claim true for the whole archive.
