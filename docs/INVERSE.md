# INVERSE — bounded inverse proceduralization (Phase K)

`src/inverse/`  implements the **inverse compiler**: given an observed bounded
sample window it searches for the cheapest **exact** deterministic
`SampleObject` explanation. It is the dual of the forward architecture.

```
forward:  SampleObject ──> exact observation (PCM is a view)
inverse:  observation  ──> cheapest exact SampleObject explanation
```

The inverse compiler is **not** a semantic authority. Its output is a
*proposal*; every accepted proposal has been re-verified by the scalar
evaluator (semantic authority) and by the representation's own intrinsic
closure. It has **zero decoder authority**.

## 1. What "exact acceptance" means

A candidate is accepted only when **both** independent reconstructions equal
the observed window:

1. **intrinsic closure** — the representation's own reconstruction, computed
   without the sampler (`Literal` bytes, a cycle's wrapping expansion,
   `sat_i32(H + R)` residual closure); and
2. **scalar-oracle observation** — the object is inserted into a store and
   observed through `World` with an *identity voice* (unity rate/gain/pan,
   instant full-sustain envelope, `Interp::Nearest` at integer frames, start 0,
   no loop). `sampler::gain` proves the unity chain is the identity.

A bounded **seek** check is added: one window of `SEEK_FRAMES` frames at a
content-derived start must equal the same slice of the window.

If (1) holds but (2) does not, the court fails with `FAILED_CORRECTNESS`: that
is an inconsistency between the representation's closure and the evaluator, not
a hypothesis that merely did not fit. "Close" is never accepted.

## 2. Candidate families (implemented)

| family | hypothesis | acceptance condition |
| ------ | ---------- | -------------------- |
| `Literal` | the canonical samples (universal fallback, **always** proposed) | identity |
| `Silence` | zero everywhere (endless, extent 0) | window is all zeros |
| `Constant` | one level = deterministic mode of channel 0 | window is constant |
| `ExactRepeat` | the minimal exact frame period (KMP over frame blocks) | window is exactly periodic |
| `ResidualZero` | `Zero` hypothesis + exact sparse residual | closure is exact |
| `ResidualConstant` | mode-level hypothesis + exact sparse residual | closure is exact |
| `ResidualPeriodic` | bounded period scan, best K by residual record count | closure is exact |
| `SharedReference` | an object already in the reference library whose observation equals the window | library observation is exact |

`ExactRepeat` uses the *minimal* exact frame period, because a longer period is
strictly more bytes for the same content. The cycle-family tags
(`SingleCycle`, `Wavetable`) are equal-cost tag alternatives of the same
payload; the compiler proposes the semantically exact one (`ExactRepeat`) and
does not inflate the frontier with tag duplicates.

### Deferred families (explicit, not silently approximated)

The contract also lists candidates that require a **residual-model vocabulary
extension** beyond the frozen u1 v1 `ResidualModel` (`Zero`, `Constant`,
`Periodic`):

* delta predictor / bounded integer linear predictor — a predictor whose
  hypothesis depends on previously reconstructed samples is *not* a function of
  the frame index alone, so it cannot be a v1 `X = H(f,ch) + R(f,ch)` model;
* simple partial/harmonic hypothesis — likewise needs a generator-model
  vocabulary in the closure.

Freezing v1 residual models is a universe-level decision (`docs/U1_SPEC.md`
§14), so these families are **deferred to a documented universe amendment**
rather than approximated by a v1 model that does not mean what the name says.
Their status is recorded in the Phase-K ledger; nothing claims them.

## 3. Complete dependency accounting

Four quantities are kept strictly apart, because they are different
measurements (H.2 memory-path doctrine: a persistent representation is not a
transient materialization and is not verification instrumentation):

```text
STORAGE COST       physical representation bytes
                     complete_bytes = metadata + hypothesis + model
                                    + payload + index + checkpoint
                                    + dependency + integrity
REPRESENTATION     does the CHOSEN representation persist baked samples?
PERSISTENCE          persistent_sample_domain_bytes
TRANSIENT          what must be decoded to observe it?
MATERIALIZATION      decoded_sample_state_bytes
                     decoded_residual_state_bytes
                     decoded_window_state_bytes
BASELINE           the original/raw/canonical sample bytes
                     raw_sample_bytes, canonical_literal_bytes
                     (NEVER part of the sum)
```

Only the eight storage components are summed. Critically, an **entropy-coded**
literal or residual persists *no* baked sample-domain content — its
persistence **is** the entropy state — so its
`persistent_sample_domain_bytes` is 0 and the decoded samples are reported as
transient materialization. A canonical cycle *does* persist its table, so that
counts as persistent sample-domain bytes (and also as storage `payload`; a
resident table is never free).

Where an entropy body exists the storage cost **is** the frozen H.2 complete
cost: `CandidateCost` carries the eight H.2 components through unchanged and
`complete_bytes` is the H.2 `complete_bytes` — the inverse compiler and the
entropy encoder cannot disagree:

* `Literal` is priced by the best frozen literal representation
  (`entropy_literal`): symbolization × page size, minimum `complete_bytes`;
* residual candidates are priced by the best frozen residual encoding
  (`entropy_residual`): page sizes 256/512/1024, minimum `complete_bytes`.

For representations with no entropy body (silence, constant, cycle, shared
reference) the decomposition is over the canonical object bytes, sums to
exactly that length (`canonical_object`), and keeps **truthful component
names**: a constant's level and a cycle's framing are `hypothesis_bytes`
(deterministic model state), a stored table is `payload_bytes`, and a
reference's target content id is `dependency_bytes` (its transpose/loop
parameters are `hypothesis_bytes`). `debug_assert!`s check the invariant, and
`CandidateCost::decomposition_is_consistent()` exposes it to tests.

Worked example — the 4096-frame mono white-noise window:

```text
raw samples        16 384 B   (baseline)
literal storage       71 metadata + 16 384 payload + 64 index = 16 519 B
persistence            0 B    (an entropy-coded literal bakes no samples)
decoded (transient)  16 384 B (materialization, reported, never summed)
```

## 3a. Search-time allocations (separate from cost and persistence)

The compiler also reports, per candidate, its own working set while
proposing/verifying (accounted, not allocator-instrumented):

| field | meaning |
| ----- | ------- |
| `search_input_bytes` | the intrinsic window held constant for the compile |
| `candidate_semantic_state_bytes` | candidate state instantiated while proposing/accepting (record vectors, sample vectors, stored tables) |
| `intrinsic_reconstruction_peak` | peak sample-domain buffer during intrinsic closure |
| `oracle_observation_peak` | peak sample-domain buffer during the full scalar-oracle observation |
| `seek_observation_peak` | peak sample-domain buffer during the bounded seek observation |

This is the inverse compiler's instrumentation, not a property of the chosen
representation — which is why it does not share the word *persistent*.

## 4. Abstract universe work

Static counts derived from the representation structure (not measurements):

| class | `generator_ops` | `residual_ops` | `lookup_ops` |
| ----- | --------------- | -------------- | ------------ |
| silence / constant | one per output code | 0 | 0 |
| literal / cycle / reference | 0 | 0 | one per output code |
| residual | one per frame | one per residual record | one per output code |

`filter_ops` is 0 across the frozen u1 v1 vocabulary (no filters exist yet).
`seek_ops` is the same work scaled to one bounded seek window.

## 5. Pareto frontier

The frontier is a genuine Pareto set over **static** objective vectors:

```
(complete_bytes, total_ops, seek_ops)      all minimized
```

`a` dominates `b` when `a <= b` componentwise and `a != b`. Costs are never
collapsed into a weighted "score".

Measured wall times (`materialize_ns`, `seek_latency_ns`, `intrinsic_ns`,
`proposal_ns`) and accounted memory are reported per candidate but are **not**
frontier objectives: a frontier whose membership depended on wall-clock noise
would not be reproducible, and the contract requires a deterministic frontier.
`accounted_peak_bytes` is the accounted sample-domain transient over the
buffers the evaluator allocates — it is **not** an allocator-instrumented peak.

## 6. Explanation vs deduplication

Two different questions are kept separate:

* **explanation** — the cheapest exact deterministic explanation *from
  scratch*. `court inverse` runs this against an **empty** library, so its
  frontier is the hypothesis frontier.
* **deduplication** — once content is already in the archive the cheapest exact
  representation is a shared reference (32 dependency bytes, 0 persistent
  bytes). The court measures this against a library containing the fixtures and
  reports the saved bytes, and separately demonstrates referencing a
  *procedural* (`ExactRepeat`) library object.

A shared reference is therefore excluded from the "can a hypothesis explain
incompressible content away?" gate: it does not explain the content, it points
at stored content.

## 7. Determinism and bounds

* No randomness anywhere; proposals are closed-form or bounded scans.
* `SearchBudget { max_period_scan: 512, max_residual_period_candidates: 4,
  max_candidates: 64 }`, with deterministic truncation.
* `court inverse` re-compiles every fixture and requires identical static
  results, and freezes a static-result hash over every report.

## 8. Running it

```
vole-audio court inverse --receipts receipts
```

writes `receipts/inverse/`. The receipt contains, per fixture, every accepted
candidate with its complete cost, abstract work, measured times, frontier
membership, and the per-fixture dedup row.

## 9. Non-claims

* The inverse compiler does **not** claim to find the global optimum; it
  reports the best of a bounded, documented candidate set.
* It does **not** claim perceptual or lossy quality — acceptance is exactness.
* A negative result (literal wins) is a result and is reported as one: on the
  frozen corpus, `harmonic-tone`, `fm-signal`, `stereo-correlated`,
  `impulse-train`, `transient-heavy` and all three negative controls are
  cheapest as entropy-coded literals — the procedural hypothesis is available
  and exact, but it is not cheaper.
* GPU inverse search is Phase L; nothing here claims parallel search.
