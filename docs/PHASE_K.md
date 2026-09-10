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

## Review amendment 1 (corrected H.2 cost adaptation)

The first external review of Phase K found that `CandidateCost` was folding
three different measurements into one number, so the *economic ranking* (not
the exactness evidence) was wrong:

* the literal candidate's storage cost summed the H.2 metadata + entropy
  payload **plus the raw samples it replaced**, while dropping the H.2
  `model_bytes` / `index_bytes` / `integrity_bytes` — roughly a 2× distortion
  on RAW-fallback content (white-noise literal reported 32 839 B instead of
  the H.2 16 519 B);
* residual candidates were charged for both their entropy-coded body and the
  unencoded delta bytes.

Fixed at the type level. `CandidateCost` now preserves the H.2 decomposition
verbatim (`metadata`, `hypothesis`, `model`, `payload`, `index`, `checkpoint`,
`dependency`, `integrity`, `complete_bytes`) and separates:

* **storage cost** — the eight components, with `complete_bytes` taken
  directly from the H.2 `CompleteCost` for entropy-carrying representations
  (no re-summing under a different interpretation);
* **state/exposure** — `persistent_sample_domain_bytes`, `state_bytes`
  (sample-domain content that may exist while evaluating; never summed);
* **baseline** — `raw_sample_bytes`, `canonical_literal_bytes` (never summed).

For representations with no entropy body the decomposition is over the
canonical object bytes and sums to exactly that length.

Three bounded-search contract bugs are fixed with it:

1. `SearchBudget::max_candidates == 0` is now **rejected** (`Literal` is
   always a candidate; truncating to zero would have dropped the universal
   fallback silently);
2. `max_period_scan == 0` (and `max_residual_period_candidates == 0`) now
   mean the periodic residual family is **disabled literally** — no period is
   scanned, so a caller asking for zero periods no longer silently gets
   period 1;
3. `ReferenceLibrary::register_literal` validates the channel count before
   dividing by it, so `channels == 0` is malformed input rather than a
   division-by-zero panic.

Hostile tests: `zero_max_candidates_is_rejected_not_silently_no_literal`,
`minimum_budget_still_yields_the_literal_fallback`,
`zero_period_scan_disables_the_periodic_family_literally`,
`register_literal_with_zero_channels_is_malformed_not_a_panic`,
`cost_decomposition_is_consistent_for_every_accepted_candidate`, plus the
cost-unit invariants (`literal_cost_equals_the_h2_complete_cost`,
`residual_cost_does_not_double_charge_the_deltas`,
`cycle_table_is_storage_payload_and_also_reported_as_state`).

## Seal history

### Seal 1 — Phase K implementation + clean-tree battery (2026-09-10)

> **Superseded for economics (not for exactness).** The exactness evidence in
> this seal stands, but its cost numbers folded baselines and sample-domain
> state into the storage sum; the candidate ranking, the
> `non_literal_wins` count and the frozen result hash were therefore produced
> under the wrong cost transformation. See “Review amendment 1” and Seal 2.

Seal run (release, `--all-features`, clean tree `7652c6b`, version 0.8.0):

- 22 receipts (eleven standalone courts — semantic, authored, simd, facts,
  inverse, flattening, cuda, d1, rocm, rocm-d0, rocm-d1 — plus `court h2`
  with its ten sub-courts), every receipt `source_binding: bound` and
  carrying `seal_subject_hash =
  1e1cd8e11766fd3268d1d2d638d158a08e4b40ef0f66b99dde646f72c2efaf08`;
  `vole-audio seal verify` PASSes the 12-row matrix in **default mode** at
  the battery tree and (after rebuilding from the release head) at the head
  itself; every receipt also verifies individually with `receipt show` (all
  348 archived receipts do — see work item 10).
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
    `9effb3c31b0b0fb0aa224084b3aa82820981ae3c2e89cbac769b0463f3de0a3a`
    (superseded by Seal 2).
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

### Seal 2 — review-1 closure: corrected H.2 cost adaptation (2026-09-10)

Seal run (release, `--all-features`, clean tree at the corrected implementation
commit, version 0.8.1):

Re-runs the whole battery after the cost fix. The corrected economics are
worse and more honest, and the court was **not** tuned to keep the old result:

- `court inverse`: **6/14** fixtures now have a non-literal exact explanation
  (down from 11/14 under the wrong accounting): `silence` 46 B vs literal
  183 B, `dc`/`dc-negative` 50 B vs 327 B, `single-sine` 574 B (exact period
  64) vs 6331 B, `quasi-periodic` 15937 B vs 16169 B, `am-signal` 1420 B
  (residual period 128) vs 8185 B. The literal floor now matches the H.2 cost
  exactly (the 4096-frame mono white-noise literal is 16 519 B = 71 metadata +
  16 384 payload + 64 index, not 32 839 B);
- honest reversals: `impulse-train` (literal 219 B) and `transient-heavy`
  (literal 633 B) are now cheapest as entropy-coded literals — the earlier
  “residual wins” were an artefact of double-charging the deltas — and all
  three negative controls (`white-noise`, `random-control`,
  `scrambled-control`) now correctly fall back to the literal, instead of the
  residual appearing 1–5 % cheaper;
- frozen static-result hash re-frozen at
  `e0b2e35ca31f64b84e232d79e3f348a252e6b408195a33fc1173acc593e95a23`;
  archive dedup still accepted for every fixture (32 dependency bytes, 0
  persistent bytes).
- Exactness evidence unchanged: every accepted candidate still reproduces the
  window through both reconstructions plus a bounded seek window, and
  `court flattening` still reports `flat == scalar`.
- Host tests: **371 passed, 5 ignored** all-features (361 passed, 5 ignored
  default-features; +8 over Seal 1: the cost decomposition invariants and the
  budget/channel hostile tests); clippy `-D warnings` and
  `cargo fmt --check` clean.

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
