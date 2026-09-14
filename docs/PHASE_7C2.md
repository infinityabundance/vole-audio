# Phase 7C.2 — `vole.audio.stream.voice.exp2` (HPE-PR)

Charter and **frozen constitution** for Phase 7C.2, *Hierarchical Predictive
Excitation and Procedural Refinement*. Frozen before implementation, in the same
discipline as [`PHASE_7C.md`](PHASE_7C.md) §13. `voice.exp1` remains the
**control**: its wire format is constitutional and is not mutated. Any gain that
requires hidden lookahead, unbounded decoder state, target-rate overshoot, or a
model that breaks the latency constitution is not a gain.

Controlling principle:

> Transmit only what the decoder cannot reliably infer. Predict, proceduralize
> or synthesize everything else; transmit correction only when the prediction
> fails a measured rate–distortion test.

## 1. Profile identity

| item | value |
| --- | --- |
| profile id | `vole.audio.stream.voice.exp2` |
| profile tag bytes | `b"vole.audio.stream.voice.exp2"` |
| control profile | `vole.audio.stream.voice.exp1` (unchanged) |
| frame | 20 ms / 320 samples at 16 kHz (10 ms / 160 permitted where declared) |
| latency constitution | inherited from 7C §5, unchanged |
| packet independence | inherited from 7C §4.4, unchanged |

## 2. Why `exp1` cannot reach the target (the finding that motivates 7C.2)

`exp1`'s generic frame envelope duplicates CELP-specific information and pads to
bytes. All figures below are for a 320-sample, pitched, VQ-spectrum CELP frame;
the per-packet header (`flags` + base gain, 16 b) is common to both profiles and
is excluded, so the comparison is frame-for-frame:

```text
exp1 frame, no pulses (bits)
  mode                              8
  generic gain delta                8
  generic pitch lag + LTP gain     24
  VQ spectrum                      32
  residual length                  16
  residual = codec id 8 + CELP 4×23 = 100  → padded to 104
                                   --
  information                      188   → padded 192

exp2 frame, no pulses (bits)
  spectral tier 2 + index 32       34
  excitation family 2 + CELP 4×23  94
                                   --
  information                      128   → padded 128
```

The generic pitch/gain fields duplicate the per-subframe CELP fields, and the
residual length is derivable from the pulse counts. With four pulses the same
relation holds: `exp1` 316 b information / 320 b padded versus `exp2` 228 b.
This is a **representation floor**, not a tuned constant: at 3.2 kbps a 20 ms
frame is 64 bits, so `exp1` cannot express a 3.2 kbps CELP frame at all. The
floor is removed by syntax (`src/voice/exp2.rs`), before any codec-quality work.
Both figures are asserted by test, not estimated.

## 3. Wire format (normative)

The frame is a **bitstream**, not a sequence of byte-aligned fields, MSB-first.
There is no byte padding inside a frame, and no generic field is serialised on a
frame whose mode does not use it.

```text
frame := spectral excitation
spectral := tier(2)
              tier 0  index(28)                       # frozen split-MSVQ
              tier 1  order(3) width(2) codes(order×width)   # scalar reflection
              tier 2,3 reserved
excitation := family(2)
              family 0  scalar: gain(8) payload_bits(12) payload(payload_bits)
              family 1  celp:   nsub × { lag(9) pitch_gain(5) gain(6)
                                         count(3) rank(pb) signs(count) }
              family 2,3 reserved (tcx / procedural, 7C.2-G/H)
```

* A CELP frame **carries no residual length**: its length follows from the
  pulse counts.
* A CELP frame **carries no generic pitch or gain field**.
* `pb = ceil(log2 C(80, count))`; pulse sets are canonical (ascending positions).
* The exact information bit count of a `Frame2` is `Frame2::bits()` and must
  equal the writer's count (`exp2.rs` asserts this in test).

## 4. Excitation cores (one exact rate–distortion selector)

| core | regime | principle |
| --- | --- | --- |
| HPE-ACELP | voiced / quasi-periodic | fractional multi-tap adaptive excitation + track-structured algebraic innovation |
| HPE-NOISE | unvoiced, breath, fricatives | deterministic shaped stochastic excitation + sparse correction |
| HPE-TCX | transients, mixed | LPC-conditioned MDCT/PVQ excitation (reuse existing machinery) |
| HPE-PR | 3.2–9.2 kbps | transmitted structural base + zero-bit predicted/procedural refinement + optional correction |

Classification proposes candidates. **The selector is exact serialised bits
versus measured reconstruction distortion**, with the hard rate ceiling as a
constraint:

```text
c* = argmin_{c : B(c) ≤ B_frame} D_perceptual(x, x̂_c),   B(c) = exact serialised bits
```

Screening may use `J(c) = D(c) + λB(c)`, but final selection is subject to the
hard ceiling.

## 5. Hard rate conformance

`exp2` removes the structural need for `OVERSHOOT_LIMIT = 4`. Every declared
rate must have at least one legal representation that fits its frame allowance;
when the richest core does not fit, the codec switches to a lower-information
core. Frame allowances (20 ms):

```text
3.2 kbps 64 b · 6 kbps 120 b · 8 kbps 160 b · 9.2 kbps 184 b
12 kbps 240 b · 16 kbps 320 b
```

## 6. Packet independence (non-negotiable)

Losing packet *n* must not corrupt interpretation of packet *n+1*. Predicted or
procedural refinement is therefore **packet-local** or anchored by an explicit
absolute state field in the packet. Cross-packet hidden state is forbidden, and
so is mandatory cross-packet LSF prediction without an absolute safety-net
representation.

## 7. Evaluation constitution

**Frozen challenger corpus (7C.2-A).** The challenger material is the held-out
LibriSpeech `test-clean` split already frozen in `corpus/real_manifest.json`:
8 clips, 8 speakers, 131 072 frames (8.19 s), speaker-disjoint from both the
regression court's `dev-clean` material and the LSF-VQ training split.

```text
challenger corpus tag    vole.audio.stream.voice.exp2.challenger.v1
challenger identity      7de2453d2b249130b2edff2b457fcc29a37eb200215387aac82debec4200c6fa
speakers                 1089 1188 121 1221 1284 1320 1580 1995
exposed via              corpus_real::challenger_clips() / challenger_corpus_sha256()
```

The corpus is frozen **before** `voice.exp2` codec work, so the challenger court
cannot be tuned to it.

**Coverage limitation (disclosed, not waived).** The available frozen material
is read speech only. It does not contain whisper/breath, realistic background
noise, or non-English speech, so the challenger court cannot currently support
the full coverage the constitution asks for (gender, pitch range, style,
fricatives, plosives, onsets, realistic conditions). Any broader claim is
therefore blocked until additional rights-clean, speaker-disjoint material is
acquired and frozen under the same manifest discipline. This is recorded as a
gap, not assumed away.

* The frozen 7C court (`learned-voice-stream`) remains the **regression court**;
  the challenger court is `learned-voice-stream-exp2`, and its codec under test
  is `voice.exp2` once 7C.2-C lands (`voice.exp1` is the labelled control before
  then).
* Comparison uses **complete actual codec payload bitrate**, never requested
  bitrate. Competitor curves are interpolated only between measured operating
  points; Lyra is compared at its native 3.2 / 6.0 / 9.2 kbps; no extrapolation.
* The headline metric is an integrated **bitrate-at-equal-quality** measure
  (BD-rate or equivalent) per competitor, with bootstrap confidence intervals at
  the case level — not a count of individual cells.
* Any perceptual superiority claim requires a preregistered blinded listening
  experiment (MUSHRA / P.808-class).
* Frozen competitor tool versions are recorded in the receipt's `harness`
  extras, as in the regression court.

## 8. Control baseline (measured, 7C.2-A)

The frozen challenger court, run on the held-out corpus against the `voice.exp1`
control (result identity `23eaf361…`, SUPPORTED):

```text
quality-at-equal-rate   vs Opus  −4.54 dB SNR  (95% CI −5.35 … −3.72) · −2.61 MOS ViSQOL
                        vs EVS   −4.39 dB SNR  (CI −4.66 … −4.12)      · −2.59 MOS
                        vs Lyra   n/a (no overlapping rate range)
bitrate-at-equal-quality vs Opus +2.72 dB (CI +2.53 … +2.91)   positive = VOLE needs MORE bits
                         vs EVS  +3.39 dB (CI +3.31 … +3.47)
matched-bitrate SNR      vs Opus  −6.35 dB (CI −9.15 … −3.56)
encode p99 5811 µs (budget 5 ms; the control already exceeds it at some cells)
```

This is the number Phase 7C.2 has to move, and it is a **loss**. It is recorded
now, before any `exp2` codec work, so the challenger court cannot be tuned.

## 9. Implementation order (seals)

| seal | work | promotion condition |
| --- | --- | --- |
| 7C.2-A | freeze this constitution + challenger corpus | **complete**: corpus identity `7de2453d…` frozen in `corpus/real_manifest.json` and exposed as `corpus_real::challenger_clips()`; court `learned-voice-stream-exp2` frozen at result `23eaf361…`, SUPPORTED on the `voice.exp1` control |
| 7C.2-B | bit-accurate mode-specific serializer | **implemented** (`src/voice/exp2.rs`): bit writer/reader, tiered spectrum, CELP and scalar families, exact round-trip and exact bit accounting, and a **decode path proven faithful** to the shared synthesis loop for both families; floor removed. Not yet a live profile |
| 7C.2-C | hard-rate fallback architecture | **in progress**: a low-information noise core, `Frame2::minimal` legal at 64 bits, and `Exp2Codec::encode_frame` — a hard-rate encoder that never exceeds the frame allowance (no `OVERSHOOT_LIMIT`). Measured on a synthetic speech-like signal over 40 frames per rate: 3.2 kbps 0.00 dB, 6 kbps 0.07, 8 kbps 0.07, 9.2 kbps 3.90, 12 kbps 4.89, 16 kbps 5.45 dB SNR. The 6–8 kbps cells collapse to the noise core because the per-subframe **23-bit overhead** does not fit 120–160 bits: this is the measured motivation for 7C.2-D |
| 7C.2-D | frame pitch anchor + contours, gain prediction, implicit pulse count | materially lower side information at identical reconstruction |
| 7C.2-E | fractional multi-tap LTP + track ACELP + joint R–D search | CELP beats the scalar path in complete bits *and* perceptual quality |
| 7C.2-F | voicing-dependent allocation + noise excitation | held-out RD gain; no dev-only promotion |
| 7C.2-G | TCX/PVQ escape mode | selected naturally by exact RD; improves mixed/transient cells |
| 7C.2-H | procedural excitation + zero-bit predicted refinement | valid 6–9.2 kbps points improve perceptual metrics |
| 7C.2-I | packet-reset entropy coding of remaining uncertainty | artifact shrinks without quality or recovery regression |
| 7C.2-J | frozen causal learned refinement experiment | promote only if a gain survives model size, decode p99, memory and the held-out court |
| 7C.2-K | final listening + impairment seal | superiority claim permitted **only** here |

Order is intentional: remove transmitted redundancy first, then predict
parameters, then change temporal rate, then vectorise, then entropy-code.

## 10. Exit criteria

Phase 7C.2 is complete only if `exp2` has genuine operating points inside Lyra's
range and the physical artifacts reproduce under the frozen challenger court.
The central victory condition is a **negative bitrate delta at equal perceptual
quality** against Opus, EVS and Lyra on the preregistered held-out court, with
uncertainty reported. A weaker but honest outcome (beating Opus/EVS while
measuring that Lyra 3.2 remains ahead) is acceptable and preferred to a nominal
three-way "win" obtained by manipulating operating points.

## 10. Non-claims

* `exp1` remains the control; its wire format is untouched.
* The serializer in `src/voice/exp2.rs` is not yet a live profile: no court
  result, no quality claim, and no bitrate claim attaches to it until it is
  wired and measured. Only its **exact bit accounting** is asserted, by test.
* The rate envelopes in this charter are budgets, not measurements.
* No learned profile asset exists yet; if one is built it must be frozen,
  hashed, reproducibly trained and inside the evidence chain.
* Mechanisms that lose a controlled A/B stay in-tree but disabled, as
  established by 7C.1.
