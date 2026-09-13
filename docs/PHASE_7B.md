# Phase 7B — `vole.audio.lossy.exp1`

Phase 7 has exactly three objectives:

```text
7A — make entropy-seed proceduralization work for sampled audio   (sealed at v0.74.0)
7B — implement vole.audio.lossy.exp1 and beat Opus + Lyra on lossy (this document)
7C — implement vole.audio.stream.voice.exp1                        (not started)
```

This document is the 7B engineering record. It states what was built, why each
mechanism is the one it is, what was measured, and — the part that matters most
for the next iteration — **which cells VOLE still loses and what mechanism is
responsible**.

---

## 1. What 7B is

`vole.audio.lossy.exp1` is a real encoder/decoder profile, not an estimator. It
is built directly on the Phase-7A principle:

```text
PCM ──► explanation search H ──► quantised residual ──► VOLE-native entropy code
```

The *explanation* `H` is the representation, not a fixed algorithm. Two
engines propose explanations and the encoder emits whichever one leaves the
cheapest representation at the requested size:

```text
                 ┌────────────────────────────────────────────┐
                 │  predictive H: short-term LPC (reflection   │
                 │  coefficients) + long-term pitch predictor  │
                 │  driven by a decoder-synchronised DPCM loop │
                 └────────────────────────────────────────────┘
PCM ──► search ──┤
                 └────────────────────────────────────────────┐
                 │  transform H: MDCT + Bark masking model +   │
                 │  per-band dead-zone scalar quantiser        │
                 └────────────────────────────────────────────┘
                                   │
                     per-frame residual symbols / coefficients
                                   │
                     VOLE-native entropy family (ResidualCodecV2)
```

Nothing competitor-shaped is inside VOLE: no neural network, no external codec,
no wrapped payload.

### Module layout

| file | role |
| ---- | ---- |
| `src/lossy/mod.rs` | profile identity, container, framing, engine search, rate control |
| `src/lossy/predict.rs` | short-term LPC + long-term pitch prediction, reflection-coefficient quantisation, closed DPCM loop |
| `src/lossy/transform.rs` | MDCT frame coding, 2-D exponent prediction, band-shape/gain split |
| `src/lossy/psy.rs` | Bark band layout and the Schroeder simultaneous-masking model |
| `src/lossy/mdct.rs` | critically sampled MDCT with a sine window and TDAC |

---

## 2. The predictive engine

Short-term linear prediction with the parameters **transmitted per frame**, not
adapted backward. This is a deliberate reversal of the obvious first design:

* a *backward-adaptive* (decoder-derived) predictor needs no side information,
  which is attractive, but in a **lossy** loop its regressor is the
  reconstruction, which carries the quantisation noise. The adaptation is then
  driven by that noise and the loop misadjusts badly: measured on real speech it
  collapsed to silence at low rates and never exceeded the LPC prediction limit
  at high rates.
* a *forward* predictor is robust because the synthesis filter `1/A(z)` is
  guaranteed minimum phase: every quantised reflection coefficient is held
  strictly inside `(-1, 1)`, and the coefficient is quantised in the
  `asin` domain.

The long-term (pitch) predictor is the second half and is what actually closes
the gap to the competitors: short-term LPC alone removes the formant envelope
(8–22 dB on speech) while the periodic excitation survives, and the pitch
predictor removes that.

Per frame the transmitted parameters are:

```text
16 quantised reflection coefficients   (5 bits each, delta-coded vs previous frame)
 1 pitch lag                            (integer samples, delta-coded, 0 = unvoiced)
 1 long-term gain                       (5 bits, delta-coded)
 1 residual step                        (half-bit log2 units, delta-coded)
+ n residual symbols
```

all entropy-coded together by the VOLE residual-codec family.

### Coefficient interpolation

The synthesis filter is interpolated in `LPC_SUBBLOCKS = 4` sub-blocks per frame
between the previous frame's quantised coefficients and this frame's. LPC
parameters quantised once per frame and applied abruptly produce a filter
discontinuity every frame, which is audible even when the residual error is
small. Interpolation is decoder-visible (both endpoints are in the bitstream),
so it costs no bits.

---

## 3. The transform engine

MDCT with a sine window and 50 % overlap (TDAC), a Bark band layout, the
standard Schroeder spreading function with tonal/noise-like masker offsets, and
an absolute threshold of hearing calibrated from dB SPL into transform-
coefficient units.

The exponent stream is split into

```text
shape[b]  = round(log2(step0[b]))      the band step profile   (transmitted as a residual)
gain      = round(g / 6.0206)          the single rate scalar   (transmitted separately)
exps[b]   = shape[b] + gain
```

and the shape residual uses **2-D prediction** — temporal (previous frame's
shape) followed by spectral (previous band's residual). Splitting the gain out
is what makes the temporal term small: the rate control moves the gain sharply
between frames, and folding it into the shape would carry the whole gain change
into every band's delta.

Coefficient symbols are quantised with a dead-zone scalar quantiser whose
reconstruction point sits toward the cell's lower edge (`c = 0.33`, where a
Laplacian coefficient density concentrates).

### Distortion units

The two engines must be compared in the same units or the search is meaningless.
The transform's per-frame distortion is the **time-domain** error energy the
frame contributes after overlap-add:

```text
distortion = (2 / N) · Σ_k e[k]²
```

derived from `Σ_m w²[m] ≈ N` and `Σ_k cos²[m,k] = N/2`, and verified against a
true overlap-add reconstruction by the
`distortion_matches_time_domain_error` unit test. Getting this constant wrong by
a factor of `N` (an early bug) silently biased every engine decision toward the
predictive engine.

---

## 4. Rate control

Both engines expose a single integer rate scalar; the encoder bisects it to land
on the target size. Two details are load-bearing.

**An absolute step, not a per-frame relative one.** The first implementation set
each frame's step from that frame's own residual RMS and a global
bits-per-sample. That is the wrong objective: uniform bits per sample does *not*
minimise total error. Reverse water-filling — one absolute step across all
frames, so a frame with more residual energy automatically spends more symbols —
is the MSE-optimal allocation subject to the rate, and it moved the codec from
~8 dB to ~22 dB SNR at 24 kbps on real speech. This was the single largest
quality change of the phase.

**Minimise distortion subject to the rate.** The candidate search evaluates a
bisection over the rate scalar plus a refinement neighbourhood, then selects the
**lowest-distortion candidate that meets the byte target** (falling back to the
fewest bytes if nothing fits). Selecting "the finest gain that fits" instead
lets a frame fall into a *silence trap*: on a pure tone, a step larger than the
signal dead-zones every symbol, the prediction history never leaves zero, and
the frame stays silent permanently even though a coarser-but-fitting candidate
exists.

### Search acceleration

The rate search evaluates each frame with a 6-codec subset
(`residual_codec2::SEARCH_CODECS`); once a gain is chosen the whole channel is
re-encoded with the full 28-codec family. The subset only has to *order* the
candidates, so it does not define the final size — and it takes the court from
~30 s to ~1 s per encoded second.

---

## 5. Defects found and fixed during 7B

Recorded because each one changed the measured result, and each one is a class
of bug that can recur:

| defect | symptom | fix |
| ------ | ------- | --- |
| backward-adaptive predictor in a lossy loop | silence collapse at low rates, saturation at high rates | forward per-frame LPC + pitch |
| lag window far too aggressive (`1 - lag/(P+1)`, squared) | LPC prediction gain ~7 dB instead of ~20 dB | Gaussian 60 Hz lag window |
| gain folded into the band shape | shape deltas ~10 bits/band | separate gain, 2-D shape prediction |
| transform distortion scale off by `N` | every engine decision biased; transform always lost | `2/N · Σe²`, with a test |
| per-frame RMS-relative step | uniform bits/sample, ~8 dB at 24 kbps | absolute step (reverse water-filling) |
| "finest fitting gain" selection | permanent silence trap on tones | lowest-distortion-among-fitting |
| predictive stream written twice | encoder/decoder desync, ~0 dB everywhere | single write (caught by a multi-frame agreement test) |

The last one is worth emphasising: the encoder and decoder were each
self-consistent at the frame level, and the frame-level unit test passed. Only a
*multi-frame* encoder/decoder agreement test caught it. The lossy module now
carries both.

---

## 6. Court `learned-lossy`

`src/courts/learned_lossy.rs` measures VOLE against **external** Opus and Lyra
at **matched actual bitrate**:

* VOLE emits its real bytes in process;
* Opus (`opusenc`/`opusdec`) is driven over a real VBR ladder of requested
  bitrates;
* Lyra (`encoder_main`/`decoder_main`, 16 kHz mono) is driven over its three
  bitrate options;
* ViSQOL (MOS-LQO, speech mode) is the external perceptual metric where the
  pinned binary and model run.

Because the competitors overshoot or undershoot their requested bitrate,
"VOLE at 24 kbps" is never compared with "Opus asked for 24 kbps". Each
competitor's measured `(actual bitrate, quality)` curve is **interpolated at
VOLE's actual bitrate**; cells outside the competitor's measured range are
`null`, never extrapolated.

Outputs are delay-aligned by a bounded cross-correlation (`ALIGN_LAG = 512`
samples — wide enough for every codec's algorithmic delay, narrow enough that it
cannot silently realign different content) before distortion is measured, and
ViSQOL is always fed a length-matched aligned pair with a freshly emptied
results CSV (ViSQOL appends, so a stale row reads back as a fresh measurement).

External code is never imported, linked, wrapped, or used as a fallback, and no
competitor payload is ever placed in a VOLE object.

---

## 7. Measured position

Numbers are from `court learned-lossy` (receipt `receipts/learned-lossy/`,
frozen projection `426a6fab…`). "Matched-bitrate delta" is VOLE minus the
competitor's interpolated quality at VOLE's actual bitrate; a positive number is
a VOLE win.

**Headline.** Over 17 cases and 8 target rates:

```text
VOLE beats external Opus on delay-aligned SNR      76 / 89 cells   mean +22.64 dB
VOLE beats external Opus on ViSQOL MOS-LQO          0 / 18 cells   mean  -0.729 MOS
VOLE beats external Lyra on delay-aligned SNR      11 / 21 cells   mean  +3.77 dB
```

**By target rate** (median VOLE SNR over the 17 cases, and the mean matched-
bitrate Opus SNR delta where a comparable Opus cell exists):

| target | comparable cells | wins | mean Opus SNR delta | median VOLE SNR |
| ------ | ---------------- | ---- | ------------------- | --------------- |
| 8 kbps | 4 | 2 | +0.33 dB | 0.0 dB |
| 12 kbps | 4 | 2 | +0.33 dB | 0.0 dB |
| 16 kbps | 12 | 6 | +9.27 dB | 2.2 dB |
| 24 kbps | 11 | 8 | +16.71 dB | 10.2 dB |
| 32 kbps | 14 | 14 | +18.84 dB | 26.2 dB |
| 48 kbps | 15 | 15 | +27.14 dB | 32.7 dB |
| 64 kbps | 14 | 14 | +31.00 dB | 44.2 dB |
| 96 kbps | 15 | 15 | +40.85 dB | 59.5 dB |

VOLE wins **every** comparable cell at 32 kbps and above and loses the low-rate
end outright. That split is the whole story of this increment.

**Speech, with the external perceptual metric** (ViSQOL MOS-LQO, speech mode;
`--` means the competitor has no measured cell at that achieved bitrate):

| clip | target | VOLE bps | VOLE SNR | VOLE MOS | Opus MOS | ΔMOS |
| ---- | ------ | -------- | -------- | -------- | -------- | ---- |
| 1272-128104-0000 | 8 kbps | 9 224 | 0.00 dB | 1.19 | -- | -- |
| 1272-128104-0000 | 12 kbps | 11 992 | 7.71 dB | 2.86 | -- | -- |
| 1272-128104-0000 | 16 kbps | 14 608 | 12.40 dB | 2.22 | 2.92 | −0.70 |
| 1272-128104-0000 | 24 kbps | 22 224 | 25.43 dB | 2.78 | 3.21 | −0.43 |
| 1272-128104-0000 | 32 kbps | 30 712 | 32.95 dB | 2.96 | 3.20 | −0.24 |
| 1272-128104-0000 | 48 kbps | 45 392 | 40.05 dB | 2.79 | 3.28 | −0.49 |
| 1272-128104-0000 | 64 kbps | 62 272 | 48.59 dB | 3.35 | 3.56 | −0.21 |
| 1272-128104-0000 | 96 kbps | 87 864 | 59.94 dB | 4.26 | 4.33 | −0.07 |
| 1462-170138-0000 | 16 kbps | 15 760 | 8.61 dB | 2.80 | 3.69 | −0.88 |
| 1462-170138-0000 | 24 kbps | 22 840 | 29.26 dB | 3.58 | 4.43 | −0.86 |
| 1462-170138-0000 | 32 kbps | 29 240 | 36.78 dB | 3.82 | 4.45 | −0.63 |
| 1462-170138-0000 | 48 kbps | 47 184 | 47.49 dB | 3.83 | 4.45 | −0.62 |
| 1462-170138-0000 | 64 kbps | 59 176 | 53.01 dB | 3.82 | 4.52 | −0.70 |
| 1462-170138-0000 | 96 kbps | 95 320 | 67.01 dB | 4.54 | 4.55 | −0.01 |
| 1673-143396-0000 | 16 kbps | 15 968 | 4.30 dB | 1.00 | 3.06 | −2.06 |
| 1673-143396-0000 | 24 kbps | 22 704 | 17.41 dB | 1.40 | 3.45 | −2.06 |
| 1673-143396-0000 | 32 kbps | 30 912 | 34.53 dB | 2.13 | 3.57 | −1.44 |
| 1673-143396-0000 | 48 kbps | 41 608 | 42.29 dB | 2.74 | 3.52 | −0.78 |
| 1673-143396-0000 | 64 kbps | 62 888 | 50.06 dB | 3.00 | 3.92 | −0.92 |
| 1673-143396-0000 | 96 kbps | 89 488 | 61.76 dB | 4.42 | 4.46 | −0.04 |

At the top of the range VOLE is within 0.07 MOS of Opus while carrying 20–36 dB
less noise. Below about 32 kbps the perceptual gap opens to 0.4–2.1 MOS.

---

## 8. Attribution of the losing cells

Every losing cell was attributed to a mechanism, and the mechanisms are
distinct:

### 8.1 The low-rate floor is model-description cost, not residual coding

At 8 and 12 kbps VOLE does not reach the target at all: the median VOLE SNR is
**0.0 dB**, meaning the emitted stream is essentially the model description with
no coded residual left over. The predictive engine emits, per 20 ms frame,
16 quantised reflection coefficients, a pitch lag, a long-term gain, a residual
step and the residual symbols. Measured on the fixtures, the parameter vector
alone costs 13–22 bytes per frame, i.e. **roughly 10 kbps before a single
residual symbol is coded**. That is the entire low-rate failure.

No amount of residual-coder work can move that ceiling. The fix is a cheaper
model description: vector-quantised spectral parameters (7B.4) instead of
scalar-quantised, delta-coded reflection coefficients.

### 8.2 The perceptual gap is noise *character*, not noise *level*

The result that matters for the next increment is in the table above: at
30.7 kbps VOLE's SNR is **+20 dB better than Opus's** while its ViSQOL is
**0.24 MOS worse**. VOLE's error is smaller and worse-sounding at the same
time. That is the signature of non-stationary artefacts rather than broadband
noise — the quantiser minimises plain MSE, so nothing in the loop knows where
the ear can and cannot hear. The large amplitude of the SNR win rules out
transform-coefficient accuracy as the problem; the mechanism to add is
perceptual weighting (analysis-by-synthesis or masking-weighted distortion) and
the 7B.6 residual context model, not a finer quantiser.

### 8.3 Tonal material collapses into the dead zone

The structural controls show a second, independent defect. A pure-tone fixture
(`harmonic-tone`) is reconstructed at ~0 dB SNR at 32 kbps, and the codec's own
instrumentation localises it precisely: the transform's allocation is a single
**global** gain applied to a masking-derived band shape, and on a narrow band
holding one strong coefficient the resulting step can exceed the coefficient
itself, so the band is quantised to zero. The constraint is
`exps[b] ≤ log2(peak[b]) − 1`. It is recorded in `transform::gain_cap`, but a
single global gain cannot satisfy it for every band at once: one quiet band pins
the gain for the whole channel. Enforcing it needs **per-band rate allocation**
(transmitted per-band exponents rather than shape + one global gain), which is
why it is listed as the first item of the remainder rather than shipped as a
clamp.

### 8.4 Lyra is a different kind of competitor

Lyra operates at 3.2–9.2 kbps, where VOLE cannot currently reach at all
(§8.1). Where the ranges do overlap the SNR comparison is already positive
(11/21 cells, mean +3.77 dB), but the honest reading is that **this increment
does not make a low-rate claim against a neural codec** — it needs the model-
description work before that comparison is meaningful.

### 8.5 What the SNR wins do and do not mean

A +22.64 dB mean SNR margin over Opus is a real property of the emitted
waveform and is measured on identical, delay-aligned material. It is **not** a
perceptual-quality claim, and the court deliberately reports both so the two
cannot be confused.

---

## 9. What was not done

7B's spec lists mechanisms that are **not** in this increment. They are named
here rather than silently omitted, and they are the declared remainder of the
objective:

* **7B.4 native residual VQ.** The residual is scalar-quantised and
  entropy-coded. A VOLE-native excitation codebook (gain/shape, split VQ,
  multistage) is the standard next step for the low-rate cells.
* **7B.5 procedural `H` front end.** The Phase-7A blind `Compound` proposer is
  not yet subtracted before quantisation in the lossy path, so tonal and
  synthetic material pays transform cost for structure `H` already explains.
* **7B.6 residual probability model.** The residual is coded by the existing
  `ResidualCodecV2` family; a context model over residual magnitude/sign, pitch
  phase, voicing and transient state is not yet in the loop.
* **Perceptual noise shaping.** The predictive engine minimises plain MSE, so
  its quantisation noise sits under the LPC envelope but is not explicitly
  shaped below a masking threshold.

---

## 10. The 7B report

### 7B mechanisms

```text
built on 7A          explanation search H, then quantised residual, then VOLE-native entropy
engines              forward-LPC + pitch prediction; MDCT + Bark masking
rate control         absolute step (reverse water-filling), lowest-distortion-among-fitting
entropy              ResidualCodecV2 family (6-codec search subset, 28-codec final pass)
container            v2, per-channel engine tag, per-frame length-prefixed records
VQ                   not implemented (7B.4)
procedural H front   not implemented (7B.5)
residual context     not implemented (7B.6)
```

### 7B competitive result

```text
Opus  SNR     76 / 89 cells won     mean +22.64 dB
Opus  ViSQOL   0 / 18 cells won     mean  -0.729 MOS
Lyra  SNR     11 / 21 cells won     mean  +3.77 dB
verdict       SUPPORTED (every case encoded, decoded and measured)
```

### 7B remaining losing cells

```text
8-16 kbps             model-description floor (~10 kbps of parameters); §8.1
speech ViSQOL         noise character, not level; §8.2
tonal / synthetic     single global gain cannot respect per-band peak; §8.3
Lyra 3.2-9.2 kbps     unreachable today; §8.4
```
