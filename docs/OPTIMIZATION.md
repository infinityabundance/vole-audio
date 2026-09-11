# Whole-repository optimization program

A campaign over the parts of VOLE-Audio **outside** the Phase O Exp2 learned
work. Every item keeps the evidence constitution: exact outputs, real byte
accounting, no corpus-name special cases, and the previous implementation kept
as a control.

## Track A — native procedural composition (`compound`)

**Spec audit first.** `docs/U1_SPEC.md` is the normative authority for `u1/v1`
and contains **no** Compound payload syntax and **no** Compound observation
semantics; `Representation::Compound = 0x0A` exists only as a code tag with a
one-line descriptor comment. Implementing a payload behind `0x0A` would be a
silent semantic break, so the capability is a **separate experimental profile**:

| item | value |
| ---- | ----- |
| universe | `vole.audio.u1` (unchanged) |
| profile | `vole.audio.compound.exp1` |
| magic | `vole.compound` |

Bounded deterministic graph over already-exact primitives: silence, constant,
the frozen DDS oscillator (frozen sine table + `eff_incr`), Q16 gain, integer
delay, the frozen analytic ADSR law, and an exact i64 `Add` saturating once.

Court `compound` (known constructions, exact materialization, priced against the
literal floor, FLAC and the bounded VOLE inverse compiler):

| case | compound | literal | FLAC | u1 best |
| ---- | -------- | ------- | ---- | ------- |
| `polyphony-3` | **168 B** | 192 054 B | 70 062 B | 192 830 B |
| `percussion-adsr` | **159 B** | 192 054 B | 22 446 B | 42 709 B |
| `layered-pad` | **207 B** | 192 054 B | 57 575 B | 192 830 B |

Result sha256 `31ddcb66…`.

## Track B — B5 hot path + entropy decode table

`FullObjectReader` now uses a binary search plus a monotonic sequential cursor,
direct-to-destination materialization (`RepresentedLiteral::materialize_into`,
`RepresentedResidual::materialize_closure_into`) with page accounting returned as
a traversal by-product, and no `BTreeSet` second pass.

The measured hot cost turned out to be the per-symbol rANS model lookup, not the
scan/copy. `SymbolModel::slot_table()` derives a `slot -> entry` table over the
frozen 16 384-slot domain from the canonical model (no stored byte changes;
decoded symbols are identical), used above a 4096-symbol threshold.

Measured on the frozen Phase-M runtime court, B5 all-115 population, identical
bytes:

| metric | before | after | factor |
| ------ | ------ | ----- | ------ |
| p50 | 882 ns | 901 ns | — |
| p90 | 37 741 ns | 34 404 ns | 1.10× |
| p99 | 303 310 ns | 117 501 ns | **2.58×** |
| p99.9 | 840 850 ns | 392 618 ns | **2.14×** |
| max | 1 299 311 ns | 591 782 ns | **2.20×** |
| mean | 20 769 ns | 10 087 ns | **2.06×** |

## Report 3 — real-speech campaign (Seal S0 …)

FLAC-5 beats the wired three-family Exp2 real-speech portfolio by ~3.6–4.6 %.
This campaign attributes that gap with evidence before changing any codec.

### Seal S0 — diagnostic (no codec change)

Two new courts, both diagnostic:

* `learned-speech-trace` parses the **actual** frozen B1 FLAC-5 artifact bit for
  bit (`baseline::flac_trace`): frame-header CRC-8, frame-footer CRC-16, and a
  full sample reconstruction that must equal the encoder's round trip. It
  classifies every subframe and records its order, coefficient precision,
  prediction shift, warmup/coefficient bits, residual partition order and
  residual payload bits, beside the VOLE complete-byte waterfall.
* `learned-real-corpus-u1` replays the portfolio under the **frozen U1 s16
  ingest** (`i32 = i16 << 16`) with new identities/hashes in
  `corpus/real_manifest_u1.json`.

**What FLAC-5 actually does on the 16 clips.** All frames are LPC (order 4–8,
dominant 8), coefficient precision 15, prediction shift 12–14, partitioned Rice
with per-block partition orders up to 5. Constant and verbatim subframes are
essentially never selected; fixed predictors appear only occasionally. FLAC's
residual payload is ~98 % of its bytes; framing/metadata is ~130 B/clip.

**Where VOLE's bytes go.** The wired portfolio's winning family is sparse-10 or
hierarchy-2; the model costs 16–70 B and metadata ~110 B, so VOLE's own framing
is *comparable to or smaller than FLAC's*. The entire gap is residual bytes:
over `dev-clean`, FLAC's residual payload is 128 305 B against VOLE's 133 881 B
(+4.3 %), and the totals are 130 331 B vs 135 124 B.

**Conclusion.** The loss is a *prediction/residual-quality* loss, not a
container-overhead loss. FLAC fits a local dense all-pole predictor every
4096 samples; the wired VOLE portfolio fits whole-clip sparse/hierarchical
models. This is exactly the model-class mismatch the campaign targets.

**U1-domain surprise.** Under the true U1 mapping every sample has 16 low zero
bits. FLAC exploits them through wasted bits (its size is unchanged), but the
learned portfolio does not: VOLE's U1 bytes balloon to ~3× FLAC (397 283 B vs
130 006 B over `dev-clean`). This is a real, exact, bounded gap — a
common-factor/wasted-bits mechanism is required before the U1 domain is
competitive on integer-scaled content.

Frozen result hashes: `learned-speech-trace` `0661a292…`,
`learned-real-corpus-u1` `d07cb3e7…`.

Seal S0 also lands the **dense-LPC model vocabulary** (`learned::lpc`, model kind
`12`, coefficients stored exactly as `i32` with a declared precision and
arithmetic shift), with its own exact-closure and accumulator-proof unit tests.
The fitting and portfolio wiring arrive in Seal S2; landing the vocabulary first
keeps S2 a pure add-candidate change.

### Seal S1 — coefficient-free fixed finite-difference predictors

New family `learned::fixed` (model kind `13`): orders 0–4 with the frozen
finite-difference coefficients, **no stored coefficient bytes**, optional
block-local reset over a frozen ladder `{whole-clip, 4096, 2048, 1024, 512}`.
The accumulating court `learned-speech` runs the baseline plus this family and
records per-family standalone bytes for every clip.

Measured (court `learned-speech`, result `c92c64ed…`): the fixed family wins on
**1 of 16** clips (`1221-135766`, 21 810 B vs baseline 21 815 B) and is dominated
elsewhere; the portfolio totals are unchanged on effectiveness (135 124 B vs FLAC
130 331 B) and improve by 5 B on Mode C (147 532 B vs 147 537 B). This is the
honest, small shape the literature predicts: fixed differences are nearly free
but rarely beat a fitted model on speech. FLAC's own LPC selection confirms it —
its fixed subframes appear only occasionally (Seal S0 trace).

### Seal S2 — dense local all-pole LPC

New fitter `learned::train::lpc`: per 4096-frame block a Tukey(0.5)-windowed
autocorrelation and a Levinson–Durbin recursion produce predictor coefficients
`H[n] = Σ_j c_j·x[n-j]`; the canonical integer form is the kind-`12`
`LpcPredictor` (Q12 `i16` coefficients, declared precision/shift). Per-block
coefficients are realised through the existing segmented container, so the
canonical format is unchanged. The fitter competes a single shared order (1..=8)
with **per-block order selection** and keeps the smallest complete artifact.

Measured (court `learned-speech`, result `5d492cff…`): the LPC family wins 5/8
effectiveness clips and the portfolio beats the baseline on 6/8 (and 3/8 on
held-out Mode C). Totals move from 135 124 → **134 174 B** (effectiveness) and
147 537 → **147 425 B** (Mode C), i.e. the FLAC gap falls from ~4.6 % to ~2.95 %
(effectiveness) and ~2.3 % (Mode C). FLAC still wins every clip.

> **Correction (Seal S3).** The S2 Levinson–Durbin recursion updated its
> coefficient array **in place**, so `a[i-j]` aliased values already rewritten in
> the same iteration; the coefficients diverged for order ≥ 4 on real speech and
the S2 numbers understate the family. The S2 receipt and hash remain as the
> honest measurement of that code; Seal S3 fixes the recursion and supersedes
> the result below.

The S2 diagnostic (measured directly from the frozen B1 artifact) shows why the
remaining gap is prediction, not container: FLAC's per-subframe residual
magnitude is ~6–18 % smaller than the S2 predictor's (e.g. clip `1272`: 210.7 vs
223.6; clip `1462`: 99.4 vs 119.1), while VOLE's residual *coding* is already at
least as good as FLAC's partitioned Rice. The next levers are the encoder-side
predictor economics Seal S3 adds (variable coefficient precision and right
shift, error-feedback quantisation) and the estimator/residual work in S4–S6.

### Seal S3 — variable precision/shift, error-feedback quantisation, recursion fix

Three changes, all encoder-side except the model syntax:

* `LpcPredictor` canonical coefficients are now **packed at the declared
  precision** (1–16 bits, MSB-first) instead of a fixed width, so a low-precision
  predictor pays only for the bits it uses.
* the fitter searches `precision ∈ {10,12,14,16}`, `shift ∈ {10,12,14}` and both
  an independent-rounding and an **error-feedback** quantiser
  (`err += c·2^shift; q = round(err); err -= q`), per block, choosing
  `order × precision × shift × quantiser` by `model bits + optimal-Rice residual
  bits`, then measuring actual complete bytes;
* **the Levinson–Durbin recursion is corrected**: the S2 code updated its
  coefficient array in place, so `a[i-j]` read values already overwritten in the
  same iteration and the coefficients diverged for order ≥ 4. With the fix the
  order-8 residual magnitude matches FLAC's (e.g. clip `1462`: 98.7 vs FLAC
  99.4).

Measured (court `learned-speech`, result `ba12fd66…`): effectiveness
134 174 → **131 770 B** against FLAC 130 331 (gap **1.1 %**); Mode C
147 425 → **143 989 B** against FLAC 144 145, i.e. **VOLE now wins the Mode-C
aggregate and 4/8 held-out clips**. The LPC family wins 6/8 effectiveness clips
and 7/8 Mode-C clips; `2035-147960` flips to a VOLE win (20 947 vs 21 243 B).

### Seal S4 — general Golomb and centered Golomb residual coding

A new **Exp3** profile (`vole.audio.learned.exp3`) imports every Exp2 candidate
and adds two residual codecs, leaving Exp2 and the frozen Seal-K evidence
byte-for-byte untouched:

* `Golomb` (id 12): partitioned general Golomb coding with an arbitrary,
  **non-power-of-two** divisor `M` per 512-sample partition, a quotient in unary
  and a truncated-binary remainder. Rice is the `M = 2^k` special case; searching
  arbitrary `M` removes the power-of-two quantisation of the Rice parameter.
* `CenteredGolomb` (id 13): the median residual is stored once and the centered
  values are Golomb-coded, which pays on asymmetric residual tails.

The `learned-speech` portfolio now re-encodes every candidate under Exp3, so the
new codecs are selected by actual complete bytes.

Measured (court `learned-speech`, result `3e8d99f3…`): effectiveness
131 770 → **130 395 B** against FLAC 130 331 — a **0.05 %** gap, with VOLE winning
5/8 clips; Mode C 143 989 → **142 684 B** against FLAC 144 145, i.e. VOLE is
**1.0 % smaller and wins 5/8 held-out clips**. General Golomb is selected on
almost every clip (`centered_golomb` on the strongly skewed `1988-147956`).

### Seal S5 — estimator diversity (Burg and covariance/least-squares)

Three estimators now propose coefficients for the *same* canonical integer
decoder syntax: Tukey-windowed autocorrelation + Levinson, **Burg**, and the
**covariance/least-squares** normal equations (Gaussian elimination with partial
pivoting). For each block and order every proposal is quantised across the S3
precision/shift/quantiser sweep, and the proposal minimising the estimated
`model + Rice` cost is kept. The estimator has no semantic authority.

Measured (court `learned-speech`, result `2ea09b96…`): effectiveness
130 395 → **130 320 B** against FLAC 130 331 — VOLE is now **ahead**, winning
5/8 clips; Mode C 142 684 → **142 570 B** against FLAC 144 145 (1.1 % smaller,
6/8 held-out wins). The paired exact Wilcoxon is 945313 ppm (effectiveness) and
78125 ppm (Mode C); the deterministic bootstrap median `FLAC/VOLE` ratio is
approximately 1.0 on effectiveness and below 1.0 on Mode C. This is statistical
parity-to-ahead, not yet a decisive margin; S6–S8 target that margin.

### Seal S6 — dense LPC orders above FLAC-5's ceiling

The per-block order search ceiling is raised to the frozen ladder top of **16**
(FLAC-5 stops at 8), and the court now reports **FLAC-8** (`-l 12`) as a
secondary control beside the primary FLAC-5.

Measured (court `learned-speech`, result `f8cb4fec…`): effectiveness
130 320 → **129 652 B** against FLAC-5 130 331 (**0.52 % smaller**) and FLAC-8
129 713 (edging it, 4/4); Mode C 142 570 → **141 706 B** against FLAC-5 144 145
(1.7 % smaller, **7/8 wins**, Wilcoxon 39 063 ppm) and FLAC-8 142 714 (0.7 %
smaller). Higher order still repays its coefficient bytes on this corpus.

### Seal S7 — lattice/PARCOR realisation (honest near-negative)

New family `learned::lattice` (model kind `14`): the predictor is stored as
quantised **reflection (PARCOR) coefficients** (`|k| < 1`, Q14) and the direct
predictor is reconstructed by a deterministic fixed-point
Levinson-from-reflection recurrence before the usual integer prediction. The
reflections come from Levinson and Burg (both already expose the reflection
sequence).

Measured (court `learned-speech`, result `d155764d…`): the lattice family is
essentially never selected on this speech corpus — effectiveness is unchanged
at 129 652 B and Mode C improves by 2 B (141 706 → 141 704). This is the honest
shape the report anticipated for a coefficient-geometry change: at Q14 the
lattice does not beat the direct-form predictor's quantisation on these blocks.
The family stays in the portfolio as a non-regressing candidate.

### Seal S8 — pole-zero (ARMA) exact residual closure

New family `learned::polezero` (model kind `15`): an exact causal pole-zero
predictor using only decoder-visible history — already reconstructed samples and
already decoded exact residuals — over the bounded `(p,q)` ladder
`{(4,1),(6,1),(8,1),(6,2),(8,2)}`. The AR part comes from Levinson; the MA part
is a Levinson fit of the AR residual, and the MA history is the true decoded
residual, so closure stays exact.

Measured (court `learned-speech`, result `83c3e06e…`): effectiveness
129 652 → **129 612 B** against FLAC-5 130 331 and FLAC-8 129 713; Mode C
unchanged at 141 704 against FLAC-5 144 145 and FLAC-8 142 714. A small but real
positive on effectiveness.

### Speech campaign result (S0–S8)

| split | VOLE | FLAC-5 | FLAC-8 | wins vs FLAC-5 |
| ----- | ---- | ------ | ------ | -------------- |
| effectiveness (dev-clean) | **129 612 B** | 130 331 B | 129 713 B | 5/8 |
| Mode C held-out (test-clean) | **141 704 B** | 144 145 B | 142 714 B | 7/8 |

The wired three-family Exp2 portfolio started ~4.6 % behind FLAC-5 on
effectiveness; the campaign now places VOLE **ahead of FLAC-5 on both splits and
ahead of FLAC-8 in aggregate**, with every candidate exact and every byte
accounted. Progression (share of the original FLAC-5 effectiveness gap closed):
S2 ≈ 0.5 %, S3 ≈ 1.9 %, S4 ≈ 2.8 %, S5 ≈ 2.9 %, S6 ≈ 3.1 %, S7 ≈ 3.1 %,
S8 ≈ 3.1 %.

### Mechanism 14 — exact common-factor / wasted-bits exploitation

Seal S0's U1 replay showed VOLE losing ~3× to FLAC under the frozen U1 s16
ingest because FLAC strips the 16 low zero bits through wasted bits while the
learned model did not. Two pieces fix it, both exact:

* a **`Wasted` model wrapper** (kind `16`, Exp3) models the quotients `X >> 16`
  with any inner hypothesis and scales the prediction back — a pure integer
transform;
* a **`FactorShift` residual codec** (id `14`) stores the exact common integer
  factor `g` of the residual as a varint and encodes the quotients `R / g` with
  the v2 family — so non-power-of-two factors such as `3`, `10` or `100` are
  captured, not only shifts of two.

The wrapper is only accepted when the scaled prediction does not saturate, so
closure stays exact.

Measured (court `learned-u1-wasted`, result `91c7d7f6…`): over the 8 U1-mapped
effectiveness clips the S0 baseline portfolio is **397 283 B** and the
wasted-wrapped LPC portfolio is **133 416 B** — a **2.98×** reduction, winning
8/8 clips, against FLAC-5's 130 006 B. `factor_shift` is selected on every clip.
The U1-domain 3× gap is closed to ~2.6 %.

### Remaining speech mechanisms (forward/reverse, lattice precision, Elias–Fano)

Three more bounded, exact mechanisms land as non-regressing portfolio candidates:

* **per-block forward/reverse direction** (`learned::reverse`, model kind `17`, court
  family `lpc_bidir`): a block may be predicted right-to-left from its terminal
  state; the fitter estimates both directions and keeps the cheaper one.
* **lattice precision search**: `fit_lattice_object` now searches reflection
  shifts `{8,10,12,13,14,15}` instead of a fixed Q14.
* **Elias–Fano sparse residual positions** (`ResidualCodecV2::EliasFano`, id `15`):
  the monotone nonzero-position sequence is Elias–Fano coded and the magnitudes
  are carried separately, beside the factor codec (id `14`).

Measured (court `learned-speech`, result `55409530…`): effectiveness unchanged at
129 612 B; Mode C 141 704 → **141 700 B** (7/8 wins). The direction choice and
lattice are honest near-negatives on dense speech (bidir 15 722 B vs 15 700 B for
direct LPC on `1272`); Elias–Fano targets sparse material rather than dense
speech residuals. **Compressed warmup** is already covered: warmup samples appear
as residual values and are adaptively coded by the partitioned-Rice/Golomb
partitions, so a separate state coder is not warranted on this corpus.

## Status

Implemented and pushed: Track A, Track B (including the entropy decode table),
and the Report 3 Seal S0 diagnostic courts (`learned-speech-trace`,
`learned-real-corpus-u1`), Seal S1 fixed differences, Seal S2 dense local LPC,
Seal S3 precision/shift/error-feedback, Seal S4 general Golomb residual
coding, Seal S5 estimator diversity, Seal S6 higher dense-LPC orders, Seal S7
lattice/PARCOR realisation and Seal S8 pole-zero/ARMA closure (court
`learned-speech`). The campaign places VOLE ahead of FLAC-5 on both splits,
and the U1 common-factor fix (`learned-u1-wasted`) closes the S0 U1-domain 3×
gap to ~2.6 %. The remaining speech mechanisms (forward/reverse direction,
lattice precision search, Elias–Fano positions) are implemented as non-regressing
candidates.
The remaining tracks from the two whole-repository
optimization reports
(CPU frame-tile multicore + PartialBank vectorization, GPU work decomposition,
CUDA/HIP graphs, entropy model p2 / compatible-model reuse, integer packing,
lifting, page-local LZ, reciprocal rANS encode, PreparedWorld, voice coalescing,
SHA acceleration, borrowed/mapped views, GPUDirect Storage, ALSA hardware-clock
scheduling, radix score assembly, PGO/BOLT, energy counters) are **not yet
implemented** and remain the declared remainder of the campaign.
