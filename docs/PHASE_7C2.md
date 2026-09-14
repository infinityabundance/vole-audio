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
`exp1` frame, no pulses (bits)
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
  excitation family 2 + CELP, 7C.2-B/7C.2-D
    lag anchor 9 + 3×contour 12    21
    pitch gain 5 + 3×delta 9       14
    innovation gain 6 + 3×delta 12 18
    frame pulse count               3
                                   --
  information                       92   → padded 96
```

The generic pitch/gain fields duplicate the per-subframe CELP fields, and the
residual length is derivable from the pulse counts. With four pulses the same
relation holds: `exp1` 316 b information / 320 b padded versus `exp2`
`92 + 4 × (21 + 4) = 192 b`. This is a **representation floor**, not a tuned
constant: at 3.2 kbps a 20 ms frame is 64 bits, so `exp1` cannot express a
3.2 kbps CELP frame at all. The floor is removed by syntax
(`src/voice/exp2.rs`), before any codec-quality work. Both figures are asserted
by test, not estimated.

7C.2-D then removes the per-subframe *repetition* of slowly varying side
information. Before it, a CELP frame paid 4 × 23 = 92 b of side information and
4 × count(3) pulse counts; after it the frame pays one lag anchor plus contour,
one pitch gain plus deltas, one innovation gain plus deltas, and one pulse count:
that is `2 + 9 + 5 + 6 + 3 + 4·(4 + 3 + 4)` bits of side information instead of
`4 · (9 + 5 + 6 + 3)` — a drop of **60 b per 20 ms frame** at the four-subframe
geometry, before any pulse is sent.

## 3. Wire format (normative)

The frame is a **bitstream**, not a sequence of byte-aligned fields, MSB-first.
There is no byte padding inside a frame, and no generic field is serialised on a
frame whose mode does not use it.

```text
frame := spectral excitation
excitation := family(2)
spectral := tier(2)
              tier 0  index(32)                       # full split-MSVQ (28 used)
              tier 1  order(3) width(2) codes(order×width)   # scalar reflection
              tier 2  stage0(8) stage0(8)             # nested: stage 0 only, 16 b
              tier 3  reserved
excitation := family(2)
              family 0  scalar: gain(8) payload_bits(12) payload(payload_bits)
              family 1  celp (7C.2-D):
                            lag_anchor(9)
                            (nsub−1) × lag_delta(4)          # ±8, clamped
                            pitch_gain(5)
                            (nsub−1) × pitch_gain_delta(3)   # ±4, clamped
                            gain(6)
                            (nsub−1) × gain_delta(4)         # ±8, clamped
                            count(3)
                            nsub × { rank(ceil(log2 C(80,count))) signs(count) }
              family 2  fallback, sub-mode(1):
                          0  noise:  gain(6) seed(3)
                          1  escape: k(3)
                                     anchor_gain(6) + (blocks−1) × delta(3)
                                     blocks × pvq_index(ceil(log2 V(80,k)))
              family 3  acelp (7C.2-E):
                            class(2)                          # pulses per track − 1
                            lag_q_anchor(11)                  # quarter-sample lag
                            (nsub−1) × lag_q_delta(4)         # ±2 samples, clamped
                            pitch_gain(5)
                            (nsub−1) × pitch_gain_delta(3)
                            gain(6)
                            (nsub−1) × gain_delta(4)
                            nsub × TRACKS × rank(ceil(log2 C(20,k)·2^k))
```

* A CELP frame **carries no residual length**: its length follows from the
  pulse counts.
* A CELP frame **carries no generic pitch or gain field**.
* A CELP frame **does not repeat per-subframe side information** (7C.2-D): one
  frame lag anchor plus an 4-bit contour, one pitch gain plus 3-bit deltas, one
  innovation gain plus 4-bit deltas, and one frame pulse count. The deltas clamp,
  so the wire is a *canonicalising* map: writing a frame and reading it back
  yields a fixed point, asserted by the `celp_wire_is_idempotent…` test.
* The **fractional-track core** (family 3, 7C.2-E) carries its pulse count as the
  codebook class, not as a field, and places pulses on four interleaved tracks
  (positions ≡ track mod 4), at most `class + 1` per track, each with a sign
  carried in the track index. The position rank is therefore over
  `C(20,k)·2^k` for each track rather than over `C(80,count)` for the whole
  subframe. Independent tracks are what stop a single shared gain from
  *coarsening* the innovation as pulses are added (measured below), and an
  implicit class is what stops one subframe's pulse count from truncating
  another's.
* `pb = ceil(log2 C(80, count))`; pulse sets are canonical (ascending positions).
* The exact information bit count of a `Frame2` is `Frame2::bits()` and must
  equal the writer's count (`exp2.rs` asserts this in test).

## 4. Excitation cores (one exact rate–distortion selector)

| core | regime | principle |
| --- | --- | --- |
| HPE-ACELP | voiced / quasi-periodic | fractional multi-tap adaptive excitation + track-structured algebraic innovation |
| HPE-NOISE | unvoiced, breath, fricatives | deterministic shaped stochastic excitation + sparse correction |
| HPE-TCX | transients, mixed | LPC-conditioned MDCT/PVQ excitation (reuse existing machinery) — **7C.2-G, implemented and disabled; see §13** |
| HPE-PR | 3.2–9.2 kbps | transmitted structural base + zero-bit predicted/procedural refinement + optional correction — 7C.2-H |

Implemented excitation **families** in the wire (`src/voice/exp2.rs`): family 0
scalar, family 1 free-combinatorial CELP (7C.2-B/D), family 2 noise, family 3
fractional-track ACELP (7C.2-E). The remaining cores become further families
without disturbing the selector.

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
| 7C.2-C | hard-rate fallback architecture | **complete**: a low-information noise core, `Frame2::minimal` legal at 64 bits, and `Exp2Codec::encode_frame` — a hard-rate encoder that never exceeds the frame allowance (no `OVERSHOOT_LIMIT`). Measured on a synthetic speech-like signal over 40 frames per rate: 3.2 kbps 0.00 dB, 6 kbps 0.07, 8 kbps 0.07, 9.2 kbps 3.90, 12 kbps 4.89, 16 kbps 5.45 dB SNR. The 6–8 kbps cells collapse to the noise core because the per-subframe **23-bit overhead** does not fit 120–160 bits: this is the measured motivation for 7C.2-D |
| 7C.2-D | frame pitch anchor + contours, gain prediction, implicit pulse count | **implemented** (`src/voice/exp2.rs`): CELP side information is one frame lag anchor (9 b) + 4-bit contour, one pitch gain (5 b) + 3-bit deltas, one innovation gain (6 b) + 4-bit deltas, and one frame pulse count (3 b). Per-subframe overhead fell from 23 b to ~6 b, a **60 b/frame** reduction at 320 samples. The wire is now a canonicalising map and the property asserted is idempotence plus decode fidelity to the read-back shot (not equality to pre-wire search parameters). Ladder over a synthetic speech-like signal, 40 frames/rate: 3.2 kbps 0.00 dB, 6 kbps 0.09, 8 kbps 0.09, 9.2 kbps 3.60, 12 kbps 4.61, 16 kbps 4.36. 6–8 kbps still collapse to the noise core, and 16 kbps regressed (5.45 → 4.36) because a single shared pulse count cannot satisfy subframes that want different counts: this is precisely the tension 7C.2-E removes by spreading side information at its intrinsic rate rather than at the subframe rate |
| 7C.2-E | fractional multi-tap LTP + track ACELP + joint R–D search | **implemented** (`src/voice/fcelp.rs`, wire family 3). Quarter-sample lag via a fixed 4-tap Lagrange interpolation of the reconstructed excitation; four interleaved tracks of `k` signed pulses, with the pulse count carried by the codebook class instead of a field; joint per-subframe selection where the pitch is scored through the real synthesis loop and the innovation gain is closed-loop. **Measured A/B on the frozen effectiveness corpus (600 frames, `voice_bench exp2`):** vs the 7C.2-B/D CELP core, +1.77 dB at 12 kbps and +2.23 dB at 16 kbps; vs the scalar+noise baseline +3.27 dB and +3.26 dB. The core wins 271/600 frames at 12 kbps and 279/600 at 16 kbps. **It is never selected at ≤ 9.2 kbps**, because its side information does not fit those allowances: that is the measured motivation for 7C.2-F/G/H. Synthetic hard-rate ladder moved from 4.61/4.36 dB to **6.63/8.75 dB** at 12/16 kbps. Encode p99 was 11 477 µs when first written; after three measured cost fixes it is **4805 µs**, inside the 5 ms constitution. Two findings are retained as tests: the long-term gain is *recursive* (the output is a polynomial in the gain, not linear), and the class field removes the shared-count truncation flaw |
| 7C.2-F | voicing-dependent allocation + noise excitation | **implemented, not promoted; two negative results recorded.** (a) The **nested spectral operating point** (`Spectral::Vq0`, tier 2, 18 bits vs 34, saving 16 bits/frame) is legal because the MSVQ is embedded, but it is selected in **zero frames at every declared rate** — the coarse spectrum's distortion penalty exceeds the excitation the freed bits buy. Kept in-tree at zero cost when unused; it is a prerequisite for 7C.2-H. (b) **Diagnosis: at 3.2 kbps the codec outputs near-silence** (SNR 0.00 dB), because for an *uncorrelated* excitation — exactly what the stochastic fallback produces — MSE is minimised by gain → 0, so the selector prefers silence to correctly-levelled shaped noise. A short-time envelope term (`envelope_penalty`, weight `ENV_WEIGHT`) was implemented as §10 requires and **measured and disabled**: at weight 2.0 waveform SNR fell at 6/8/9.2/16 kbps and rose only at 12 kbps. Promotion on dev-only evidence is forbidden, so the weight ships at 0.0 |
| 7C.2-G | TCX/PVQ escape mode | **implemented, measured, disabled; negative result recorded.** A new voice-track primitive `src/voice/pvq.rs` (orthonormal DCT-IV + PVQ, no dependency on the general-audio `src/lossy/` track) backs an escape sub-mode of the fallback family, so ACELP frames pay zero bits for it. On the frozen development corpus it is selected in **zero frames at every declared rate** — bits and SNR are identical with and without it — and it costs up to **8864 µs** encode p99 at 16 kbps against the 5 ms constitution. The mechanism is verified *faithful* by test (`the_escape_core_reconstructs_its_own_quantisation`), so the negative is a real measurement. The diagnosis in §13 ties it to 7C.2-F: with MSE as the objective, any reconstruction with correlation ρ < 0.5 scores worse than silence |
| 7C.2-H | procedural excitation + zero-bit predicted refinement | **five steps landed.** (1) The perceptual arbiter (§14) and (2) the **fitted selector weight** (§15) — mean MOS-LQO 1.250 → **1.416 (+13 %)**, with **SNR and MOS disagreeing in sign** at 6 and 9.2 kbps, the phase's most important result, which also reverses 7C.2-G's verdict. (3) **Procedural excitation** (§16): implemented, tested, packet-local by construction, selected in **zero frames**; shipped disabled. (4) **The prediction split** (§17): the proposed re-balance of short-term against long-term prediction order is **refuted** — total prediction gain is maximized at the highest short-term order (19.62 dB). (5) **Scalar rate resolution** (§18): the idle channel is **not** a defect fixable by a finer scalar gain ladder — the fine ladder used 41 more bits per frame at 16 kbps and scored 0.082 MOS worse. That closes the under-spend hypothesis and points at entropy coding as the cheap-bits lever. Zero-bit predicted refinement still to come |
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

## 11. Seal 7C.2-E — measured record

**Mechanism 1: fractional long-term prediction.** The control core resolves the
pitch lag to one 16 kHz sample (62.5 µs). At a 200 Hz voice the period is 80
samples, so a one-sample lag error is a 1.25 % period error and the predictor
phase drifts across the 5 ms subframe. `voice_bench pg` measures the
consequence: the integer long-term predictor adds only **0.66 dB** over the LPC
residual on real speech. Family 3 resolves the lag at **quarter-sample**
resolution by interpolating the reconstructed excitation with a fixed 4-tap
Lagrange filter (`INTERP`), whose coefficients are a partition of unity so the
integer fraction reduces exactly to the integer tap (asserted by test).

**Mechanism 2: interleaved tracks instead of a free pulse codebook.** A free
`count`-subset of 80 positions with a single shared gain has a structural
flaw: every pulse has unit magnitude times that one gain, so adding a pulse
always adds energy. It cannot refine, it can only coarsen. `voice_bench cmp`
measures it directly — the control core reaches **10.66 dB** with one pulse per
subframe and *falls* to **3.83 dB** with two. Family 3 spreads the innovation
over four interleaved tracks by construction, carries the sign inside each track
index, and takes its pulse count from the codebook class, so it spends no count
field and cannot truncate a subframe.

**Mechanism 3: joint selection.** Each subframe's pitch candidate is scored with
the adaptive term synthesised through the real loop, and the winning innovation
set's gain is re-selected by closed-loop synthesis rather than by the
correlation proxy. The selector remains exact bits vs measured distortion: the
core is offered alongside the control core, never forced.

**A finding that changed the design.** The synthesis loop is *recursive* in the
long-term gain: the gain multiplies the interpolated reconstructed excitation, so
it sits inside the predictor's own feedback. `out(g)` is therefore a polynomial
in `g`, not `zir + g·q`, and the cheap one-probe-per-lag shortcut is invalid. The
analysis therefore synthesises each gain candidate explicitly, which is the
dominant cost and is why the lag set is kept focused. Asserted by
`the_long_term_gain_is_recursive_so_scoring_must_resynthesise`.

**A repair found en route.** `exp2`'s family-1 pulse budget was still derived
from the pre-7C.2-D per-subframe overhead model, so the encoder believed CELP did
not fit at 6–8 kbps and fell back to the noise core. Deriving `maxp` from the
exact serialised size lifted the synthetic 8 kbps cell from 0.09 dB to 3.45 dB.
The `celp::max_pulses_for_bits` helper is unchanged, because `exp1` still uses
it.

**Cost, disclosed.** The first working version breached the constitution: encode
p99 11 477 µs at 16 kbps. Three measured fixes brought it to 4805 µs: offering
the core once per frame (best-residual candidate × cheapest spectral tier)
instead of six times, hoisting the perceptual weighting filter into a
precomputed `vp::Weighting`, and a weights-passing `synthesize_w` so the search
does not rebuild the short-term filter per candidate. Quality was unaffected. All
declared rates are within the 5 ms budget.

**What it does not do.** Family 3 needs 189 bits (VQ spectrum + one pulse per
track) and 253 bits (two per track) per 20 ms frame. It therefore cannot appear at
3.2, 6, 8 or 9.2 kbps, and the measured A/B shows it is selected in **zero**
frames there. The 12–16 kbps win is real and measured; the low-rate target is
still open and belongs to the cheaper-side-information seals.

## 12. Seal 7C.2-F — measured record (no promotable gain)

Seal 7C.2-F was scoped as rate-scaled spectral precision plus voicing-dependent
allocation. Both halves were implemented. **Neither is promoted**, and the reason
is measured rather than assumed.

**The nested spectral operating point.** `vq::stage0` / `vq::pack_stage0` extract
and rebuild a stage-0-only index; the encoder offers it as `Spectral::Vq0`
(tier 2) for both ACELP cores and every excitation. It is exactly 16 bits cheaper
than the full index (18 vs 34), and it is a *nested* point rather than a second
codebook: the MSVQ is embedded, so stage 0 is the coarse envelope the full index
refines. The controlled A/B is unchanged to the bit at every declared rate, and
the family histogram shows the tier is selected in **zero frames**. The reason is
visible in the budget: at 9.2 kbps the full spectrum (34 b) plus a coarse ACELP
frame already fit inside 184 bits, so the freed 16 bits buy fewer additional
pulses than the spectral precision they cost. The tier stays in-tree — it is a
legal operating point that costs nothing when unused, and it is the precondition
for 7C.2-H — but a mechanism that loses its A/B does not get promoted.

**The finding that matters more than the seal: MSE selects silence.** At 3.2 kbps
(64 bits/frame) the codec's measured SNR is **0.00 dB**, which is not "poor
quality" but "the output is uncorrelated with the input". Tracing it gives a
structural result:

> For a *stochastic* excitation the MSE-optimal gain is zero. If the
> reconstruction is uncorrelated with the target, `min_g ‖t − g·h‖²` is attained
> at `g = (t·h)/(h·h) ≈ 0`. Silence therefore scores better than
> correctly-levelled shaped noise, and the lowest-rate fallback degenerates to
> near-silence.

This is why the charter's §10 warning — "raw SNR cannot remain the search north
star" — is not stylistic: MSE is not merely a weak proxy at the bottom of the
rate range, it is the *wrong* objective for the fallback core. A short-time
temporal-envelope term (`envelope_penalty`) and a combined proxy
(`mse + W·penalty`) were implemented accordingly, and the property is pinned by
`envelope_penalty_rejects_silence_that_raw_mse_accepts`.

**Why the fix ships disabled.** At `W = 2.0` the controlled A/B
(`voice_bench exp2`, 600 frames of the frozen development corpus) moved waveform
SNR **down**: 6 kbps +0.24 → −0.69, 8 kbps +1.04 → +0.73, 9.2 kbps +1.77 → +1.22,
16 kbps +5.04 → +4.80, with an increase only at 12 kbps (+4.23 → +4.77) as
selection shifted toward level-matched but uncorrelated noise frames. Whether that
trade is perceptually better is exactly what SNR cannot decide, and `voice.exp2`
is not a live profile, so the external court cannot arbitrate. §7 and §10 forbid
promotion on dev-only evidence, so `ENV_WEIGHT = 0.0` and the term is retained
in-tree, following the `SUBFRAME_GAIN` precedent from 7C.1.

**Conclusion.** 7C.2-F produced no promotable gain, and the honest reading is
that the low-rate deficit is **not** a parameter-allocation problem that
frame-local tuning can solve. The 64-bit frame cannot carry a coarse spectrum,
a pitch trajectory, gains and a useful innovation at once, which confirms §18:
3.2 kbps has to be a synthesis-and-refinement mode (7C.2-H), not "ACELP with
fewer pulses".

## 13. Seal 7C.2-G — measured record (negative result)

**A new track, not the general-audio one.** The charter suggested reusing the
existing transform machinery. That machinery lives in `src/lossy/`
(`mdct.rs`, `transform.rs`, `psy.rs`), which is the **general-audio lossy
profile**, and it cannot be used here. `lossy::mdct::Mdct` is a *framing*
transform: `n` coefficients from a `2n`-sample windowed input, reconstructed by
windowed overlap-add across frames. Its synthesis returns `w[m]·x[m]`, so
recovering the input within one frame means dividing by the sine window, whose
smallest value is `sin(π/4n) ≈ π/4n` at the frame edges — about a 200×
amplification of edge quantisation noise at `n = 160`. A per-frame escape
excitation has no overlap-add and no second half to cancel that error against.

So 7C.2-G adds a **new voice-track primitive**, `src/voice/pvq.rs`, with zero
dependency on the lossy track:

* an **orthonormal, self-inverse DCT-IV** (`dct4` *is* its inverse), so the
  coefficient norm equals the residual norm and a transmitted gain means
  exactly what it says;
* **PVQ** with the analytic cardinality `V(n,k)` the charter asked for —
  `count`, `rank`, `unrank`, and a greedy shape search — so a pulse count fixes
  the transmitted rate with no learned probability table;
* fixed **80-sample blocks**, deliberately the codec's own 5 ms subframe. The
  block size sets the index cost per pulse: at 40 samples a *single* pulse
already costs 7 bits, which is why the escape mode can only reach 6 kbps at
  the larger block. The cap `MAX_PULSES = 5` follows from `V(80,6) > 2^32`, i.e.
  from the wire's index width, not from the quantiser's quality.

**Wire placement.** The escape mode is a **sub-mode of the fallback family**
(one discriminator bit on family 2), not a fifth family. A fifth family would
need three family bits and would cost one bit on *every* frame — including
3.2 kbps, where the escape mode cannot fit at all. This way an ACELP frame pays
nothing for the escape core's existence.

**Measured, and negative.** `voice_bench exp2` over the frozen development
corpus (600 frames), A/B with the escape offered and withheld:

```text
rate      SNR with      SNR without   frames choosing the escape
3.2 kbps    -0.00         -0.00        0 / 600
6 kbps       0.24          0.24        0 / 600
8 kbps       1.04          1.04        0 / 600
9.2 kbps     1.77          1.77        0 / 600
12 kbps      4.23          4.23        0 / 600
16 kbps      5.04          5.04        0 / 600
```

Bit counts are identical too, so the selected frames are literally the same
frames. The escape core is offered at every pulse density that fits (from
`k = 1`, which fits at 6 kbps) and loses every comparison.

**Cost, disclosed.** Offering it breached the constitution: encode p99 reached
**11 258 µs** at 16 kbps. The transform depends on the spectral tier but not on
the pulse density, so hoisting it out of the `k` loop brought that to
**8864 µs** — better, still far outside 5 ms.

**The mechanism is faithful, so this is a real negative.** Before accepting the
result, the encode path was pinned by test: given a residual built from a few
transform components, the frame the encoder actually emits decodes to a block
whose shape correlates better than 0.5 with what it was given
(`the_escape_core_reconstructs_its_own_quantisation`). The failure is therefore
in the *selection*, not in the wiring.

**The diagnosis, and it generalises.** Combining this with §12 gives a single
structural result:

> With MSE as the objective, a reconstruction whose correlation with the target
> is `ρ` has error `2E(1−ρ)`, while silence has error `E`. Silence therefore
> scores **better than any reconstruction with `ρ < 0.5`** — and a sparse PVQ
> transform shape at these rates sits below that threshold. So does the
> stochastic fallback. MSE does not merely prefer weak reconstructions; at the
> bottom of the rate range it prefers *nothing at all*.

This is why 7C.2-F's envelope proxy existed, and why it was not promotable: the
natural fix lowers waveform SNR, and only an external perceptual court can
arbitrate. The consequence for the phase is important and is recorded as such:
**every non-CELP core at low and mid rate is currently being judged by an
objective that is structurally biased against it.** The escape core is retained
in-tree but **disabled** (`Options::default().tcx = false`), following the
`SUBFRAME_GAIN` precedent, so the experiment stays reproducible and can be
re-judged once the selector objective is fixed.

## 14. Perceptual arbiter — 7C.2-H, first step

§7 requires a perceptual measure and §10 refuses to let waveform SNR be the
search objective. §12 and §13 then showed why that is not a stylistic
preference: under MSE, silence outscores every reconstruction with correlation
`ρ < 0.5`, so the objective is *structurally* biased against every non-CELP core
at low and mid rate. A probe was therefore added: `voice_bench visqol` runs the
**court's frozen ViSQOL protocol** (same binary, same speech-mode lattice model,
same `moslqo` column) on the **development** corpus. The held-out challenger
corpus is deliberately untouched, so anything fitted here can still be judged
there. The protocol is duplicated into the instrument rather than shared, so a
change to a dev probe can never silently change what the court measures.

Calibration first, because an instrument that reports 5.0 for a perfect match is
lying: on these three development cases ViSQOL scores a **self-comparison at
4.402 MOS-LQO**, which is its ceiling here. One call takes ≈0.42 s.

Measured, MOS-LQO mean over 3 development cases:

```text
rate     noise only   +celp      +acelp    +tcx      bits/frame (noise → acelp)
3.2 kbps    1.000      1.000      1.000     1.000       46.0 →  46.0
6 kbps      1.205      1.205      1.205     1.205       87.8 →  87.8
8 kbps      1.151      1.180      1.180     1.180       99.1 → 122.1
9.2 kbps    1.151      1.156      1.156     1.156      108.4 → 143.3
12 kbps     1.190      1.221      1.342     1.342      123.7 → 209.1
16 kbps     1.110      1.144      1.369     1.369      149.7 → 246.1
ceiling (self-comparison) 4.402
```

Four results, in order of importance.

**1. The 7C.2-E win survives a perceptual judge.** The fractional-track core beats
the control core at 12 kbps (1.342 vs 1.221) and 16 kbps (1.369 vs 1.144) — the
same two rates and the same direction the waveform measurement reported. For the
CELP-family cores the proxy was therefore adequate, and 7C.2-E's claim does not
depend on SNR being a good metric. This is the seal's strongest evidence.

**2. The 7C.2-G negative is confirmed perceptually.** `+tcx` is identical to
`+acelp` at every rate, exactly as the bit counts and family histograms predicted.
The escape core is not merely losing a biased comparison; it is genuinely not
better here.

**3. The absolute position is far from competitive.** 1.0–1.37 MOS-LQO against a
4.40 ceiling. This corroborates the regression court's −2.61 ViSQOL MOS against
Opus and −2.59 against EVS, now measured inside this phase's own instrument.

**4. New defect, and it is a rate defect, not only a quality one: the encoder
leaves the channel idle.** With only fallback cores available the codec spends
**99.1 of 160 bits** at 8 kbps and **149.7 of 320 bits** at 16 kbps — up to 53 %
of the allowance unspent — because the MSE objective prefers a 12-bit stochastic
frame to a several-hundred-bit scalar frame whose reconstruction is weakly but
positively correlated. Combined with §13 this is the sharpest statement of the
blocker yet:

> The MSE objective costs the codec a large fraction of its *bandwidth* as well
> as its quality. No amount of additional excitation machinery can help while the
> selector is free to choose near-silence and leave the frame half-empty.

At 3.2–9.2 kbps the spread between cores is within ±0.03 MOS, so the low-rate
range is uniformly non-competitive *regardless of core*, which is 7C.2-F's
conclusion reached independently by a perceptual measure.

## 15. Fitted selector weight — 7C.2-H, second step

§14 established that the MSE objective is the binding blocker. §7 prescribes the
remedy: fit the proxy's free parameter on **development** material, freeze it, and
let the held-out court decide whether the proxy correlates with a perceptual
judge. `voice_bench weights` does the fitting.

**The fit.** Mean MOS-LQO over the five rates, by envelope weight:

```text
w = 0.0   1.250      w = 1.0   1.344      w = 4.0   1.390
w = 0.25  1.293      w = 2.0   1.416      w = 8.0   1.345
```

The response is unimodal and shallow. `2.0` peaks the mean and is best or
near-best at *every* rate, while `4.0` collapses at 8 kbps (1.248 against 1.383).
That shallow, monotone-in-the-middle shape is what makes the choice defensible
rather than a knife edge. It is frozen as [`exp2::DEFAULT_ENV_WEIGHT`], recorded
as a **dev fit** — three development cases — and **no quality claim attaches to
it** until the held-out court measures it.

**The result that matters: SNR and perceptual quality now disagree in sign.**

```text
rate      SNR (w=0 → w=2)        ViSQOL MOS (w=0 → w=2)
6 kbps     +0.24 → −0.69          1.205 → 1.445
8 kbps     +1.04 → +0.73          1.180 → 1.383
9.2 kbps   +1.77 → 1.22           1.156 → 1.408
12 kbps    +4.23 → +4.77          1.342 → 1.414
16 kbps    +5.04 → +4.80          1.369 → 1.428
```

At 6 kbps and 9.2 kbps the two measures move in **opposite directions**, and the
perceptual judge prefers the configuration the waveform measure rejects. This is
the concrete, measured vindication of the charter's §10 refusal to make SNR the
north star: for three seals the objective was not merely imprecise at the bottom
of the rate range, it was pointing the wrong way. Mean MOS rises from 1.250 to
1.416, a 13 % relative improvement, with the largest gains exactly where the
codec was weakest.

**A re-judgement: 7C.2-G was partly an artifact of the objective.** Under MSE the
escape core was selected in zero frames at every rate. Under the fitted proxy it
is now selected at 8 kbps and 9.2 kbps, and is *better* there (1.406 vs 1.383 and
1.481 vs 1.408) while using the same or fewer bits. Its quality judgement is
therefore reversed. It still does **not** ship enabled, because the cost stands:
its encode p99 is 4464 µs at 8 kbps and 5379 µs at 9.2 kbps, and §13's 8.8 ms at
16 kbps. The switch is retained so the reversal is reproducible.

**Cost, and it is a pre-existing gap.** With the fitted proxy the encode p99 at
16 kbps is 5527 µs against the 5 ms constitution. This is not a new regression:
the *control* `voice.exp1` baseline is already **5811 µs** in the regression
court. The voice profile's encode deadline is an open problem inherited from 7C,
not introduced here, and it is recorded as such rather than attributed to the
selector change.

## 16. Procedural excitation — 7C.2-H, third step (negative, with a diagnosis)

The charter's §13 proposes the decoder *materialise* the excitation: transmit the
structure, generate the detail. `src/voice/proc.rs` implements exactly that — a
pitch-synchronous glottal excitation blended with shaped noise, every sample a
pure function of the transmitted fields and the sample index, so packet
independence holds by construction rather than by review. It carries `lag_q(11)`
· `gain(6)` · `voicing(4)` · `phase(6)` · `seed(3)` = 34 bits, as a sub-mode of
the fallback family, so ACELP frames pay nothing for it.

Three properties are pinned by test: the output RMS is exactly the transmitted
level at every voicing value (including the blend's cross term), the pulse train
lands on the transmitted fractional period, and the generator is deterministic and
packet-local.

**Measured: selected in zero frames.** Forced to be the *only* core on offer, its
output is bit-identical to the stochastic fallback at every rate:

```text
rate      noise only   proc only    +acelp      +proc
3.2 kbps     1.000       1.000       1.000      1.000
6 kbps       1.445       1.445       1.445      1.445
8 kbps       1.453       1.453       1.383      1.383
9.2 kbps     1.423       1.423       1.408      1.408
12 kbps      1.158       1.158       1.414      1.414
16 kbps      1.211       1.211       1.428      1.428
```

The same holds in waveform SNR and in bits/frame. Offering the core costs
0.9–1.5 ms of encode time, so it ships **disabled**.

**Two hypotheses were tested and rejected before the real cause was found.**

1. *Phase resolution.* A comb carries no memory, so its phase must be transmitted,
   and the first version spent 3 bits on it — four positions inside a `period/8`
   span. The diagnostic `phase_resolution_is_what_limits_a_memoryless_pulse_train`
   confirms the coarse grid is a limiter in isolation (a 64-step grid reaches
   ≈0.70 correlation against a pure comb at period 81, an 8-step grid far less).
   So the phase became a **6-bit field solved for analytically** (`proc::best_phase`,
   `O(P·N)` per parameter combination) rather than searched jointly. Measured
   effect on the codec: **none**.
2. *Lag resolution.* The lag field is already quarter-sample, but only integer
   lags were searched. Searching its full transmitted resolution costs no wire
   bits. Measured effect: **none**.

**The actual cause, and it generalises.** `voice_bench pg` reports that order-16
LPC already extracts **18.31 dB** and that a pitch predictor on top of that
residual adds only **0.66 dB**:

```text
signal energy          1.1337e12
LPC residual energy    1.6748e10   gain 18.31 dB
+pitch residual energy 1.4381e10   gain 18.97 dB
```

Order-16 short-term prediction over a 20 ms frame has already absorbed most of the
pitch periodicity. The residual the excitation has to code is therefore *very
nearly pitch-free*, and no long-term or procedural model has more than a fraction
of a dB to win from it. That single number explains all three of this phase's
excitation negatives — 7C.1's weak adaptive codebook, 7C.2-G's escape core, and
this procedural core — and it also explains why 7C.2-E's fractional LTP only
started paying at 12–16 kbps, where the spectrum is good enough for the residual
to retain structure.

> **The binding constraint is the short-term predictor's order, not the excitation
> model.** A high-order LPC that has already eaten the pitch leaves every long-term
> mechanism with under 1 dB of headroom, and leaves the excitation quantiser with
> an almost-white residual — which the MSE objective then declines to code.

The standards do not do this: SILK pairs a *low-order* short-term filter with a
fifth-order long-term predictor, deliberately splitting the prediction so that the
long-term stage has structure to exploit. The next work on this path is therefore
to re-balance short-term and long-term prediction order (§16's spectral work is the
same lever from the other side), not to add further excitation machinery.

## 17. The prediction split — 7C.2-H, fourth step (hypothesis refuted)

§16 ended by proposing the next lever: re-balance short-term against long-term
prediction, following SILK's low-order short-term filter plus high-order long-term
predictor. That proposal was a *hypothesis about why* the residual is pitch-poor,
and `voice_bench split` tests it directly before any architecture changes. It is
refuted.

The diagnostic reports the **total** prediction gain per short-term order, with the
long-term gain measured both frame-wide and per 5 ms subframe — the latter is what
the codec actually carries, and the former understates it by roughly 2×:

```text
 order |  frames | short-term | LTP frame | gain  | LTP 5 ms | gain
     8 |     600 |   17.15 dB |  17.95 dB | +0.80 |  18.71 dB | +1.56
    10 |     600 |   17.59 dB |  18.34 dB | +0.75 |  19.05 dB | +1.47
    12 |     600 |   17.86 dB |  18.62 dB | +0.76 |  19.33 dB | +1.47
    14 |     600 |   18.11 dB |  18.82 dB | +0.71 |  19.49 dB | +1.38
    16 |     600 |   18.27 dB |  18.96 dB | +0.69 |  19.62 dB | +1.35

 order-16 width sweep
 width |  frames | short-term | LTP frame | gain  | LTP 5 ms | gain
     6 |     600 |   18.27 dB |  18.96 dB | +0.69 |  19.62 dB | +1.35
     8 |     600 |   18.43 dB |  18.98 dB | +0.55 |  19.62 dB | +1.19
```

**1. There is no beneficial re-balance.** Lowering the short-term order raises the
long-term gain (+0.69 → +1.56 dB) but by *less* than the short-term gain falls
(18.27 → 17.15 dB). Total prediction gain is maximized at the **highest** order,
monotonically. SILK's split does not transfer to this predictor.

**2. Quantisation is not hiding pitch either.** A *finer* spectrum at the top order
(width 8 vs 6) raises the short-term gain (18.43 vs 18.27) and *lowers* the
long-term gain (1.19 vs 1.35), leaving the total unchanged at 19.62 dB. A better
short-term filter simply absorbs more of the pitch — the same trade seen along the
order axis, from the other direction.

**3. The frame-wide figure understates the long-term gain by about 2×.** This
matters for honesty: §16 recorded 0.66–0.69 dB, which is the frame-wide number,
while the codec carries one lag and gain per 5 ms subframe. The diagnostic now
reports both so the figure cannot be misread. The corrected value is 1.35–1.56 dB —
still small, and still shrinking as the short-term filter improves, but not
negligible.

**What this closes off, and what it leaves.** Total prediction gain is 19.62 dB,
so the residual carries 1/92 of the signal energy: the *predictor is good*. The
codec's loss is therefore not prediction, which confirms `VOICE_RD.md`'s
attribution to the residual quantiser and closes the prediction-order direction.
The remaining levers are **quantisation efficiency** — the spectral envelope (§16)
and packet-reset entropy coding (7C.2-I) — not more excitation structure and not a
re-balanced predictor.

## 18. Scalar rate resolution — 7C.2-H, fifth step (hypothesis refuted)

§14 measured that the encoder leaves up to **53 %** of the frame allowance idle at
6–9.2 kbps, and that the scalar residual is selected in almost no frames. The
evident reading was a defect: `residual::encode_best` emits the *smallest* artifact
for whatever symbols the quantiser produced, and the scalar candidate swept only
seven gain codes, so the path had almost no rate resolution and could not spend the
budget it was given. The test was to extend the sweep to `GAIN_MIN..=60` in steps
of four (17 values, ≈2.4× the candidates).

**It is refuted, and the refutation is unusually clean.**

```text
rate      SNR before → after     ViSQOL before → after
8 kbps    +0.73 → +0.79          1.383 → 1.357
9.2 kbps  +1.22 → +1.56          1.408 → 1.242
12 kbps   +4.77 → +3.83          1.414 → 1.328
16 kbps   +4.80 → +5.47          1.428 → 1.333
```

Waveform SNR *rises* at 9.2 and 16 kbps while ViSQOL *falls* by 0.03–0.17 MOS
across 8–16 kbps, and encode p99 rises to 6315 µs at 16 kbps. Another SNR/MOS
divergence, in the opposite direction from §15's — and decisive, because ViSQOL is
the arbiter.

The decisive detail is the **scalar-only** configuration, which has no other core
to hide behind:

```text
16 kbps, scalar+noise only:  147.3 bits, 1.211 MOS  (seven gains)
                             188.4 bits, 1.129 MOS  (fine ladder)
```

Given more rate resolution, the scalar path spends 41 more bits per frame and
scores **0.082 MOS worse**. So the idle channel is not a defect that scalar rate
resolution can fix: the encoder is *right* to decline those bits. The under-spend
is a symptom of the scalar quantiser's quality, not of its search.

**Verdict: rejected and reverted**, and the v0.85.0 baseline reproduces exactly
(8/9.2/12/16 kbps → 1.383/1.408/1.414/1.428 MOS on 121.2/139.9/214.6/251.8 bits).
The lever it points at is the *opposite* of spending more bits: make the existing
bits cheaper, which is 7C.2-I's packet-reset entropy coding.

## 19. Non-claims

* `exp1` remains the control; its wire format is untouched.
* The serializer in `src/voice/exp2.rs` is not yet a live profile: no court
  result, no quality claim, and no bitrate claim attaches to it until it is
  wired and measured. Only its **exact bit accounting** is asserted, by test.
* The rate envelopes in this charter are budgets, not measurements.
* No learned profile asset exists yet; if one is built it must be frozen,
  hashed, reproducibly trained and inside the evidence chain.
* Mechanisms that lose a controlled A/B stay in-tree but disabled, as
  established by 7C.1.
