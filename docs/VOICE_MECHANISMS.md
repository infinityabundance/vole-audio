# Voice codec mechanisms: what the competitors do, in numbers

Grounded reference for the mechanisms AMR-WB, EVS, Opus (SILK) and Lyra use to
reach their rate–distortion efficiency, with the concrete numbers needed to
re-implement the *structure* (not the trained tables) in this repository. Every
figure is sourced; where a figure comes from the reference C rather than the
prose spec, that is stated. Companion notes: [`VOICE_RD.md`](VOICE_RD.md)
(measured attribution), [`LOWRATE_SPEECH_RD.md`](LOWRATE_SPEECH_RD.md)
(rate–distortion theory and the closed-loop overload derivation).

Sources fetched: 3GPP TS 26.190 (AMR-WB transcoding, ETSI rendering) and TS
26.173 (AMR-WB ANSI-C reference); RFC 4867; ITU-T G.729 (2012); 3GPP TS 26.445
(EVS); RFC 6716 (Opus); `draft-vos-silk-01` (in this repo's session transcript);
SoundStream (arXiv:2107.03312); Kleijn et al. (arXiv:2102.09660); Lyra V2
release notes and the `research/lyra-main` source.

## 1. AMR-WB (G.722.2 / TS 26.190) — 20 ms, order 16, 4×64-sample subframes at 12.8 kHz

AMR-WB runs internally at **12.8 kHz** (`L_FRAME 256`, `L_SUBFR 64`,
`NB_SUBFR 4`), so a 16 kHz design must re-derive the tables; only the structure
transfers.

### 1.1 Per-mode bit allocation (TS 26.190 Table 1)

| Mode | ISP | LTP-filter flag | Pitch delay (4 sf) | Algebraic code (per sf ×4) | Gains (per sf ×4) | HB | Total |
|---|---|---|---|---|---|---|---|
| 23.85 | 46 | 4 | 9/6/9/6 = 30 | 88×4 = 352 | 7×4 = 28 | 16 | 477 |
| 23.05 | 46 | 4 | 30 | 352 | 28 | – | 461 |
| 19.85 | 46 | 4 | 30 | 288 | 28 | – | 397 |
| 18.25 | 46 | 4 | 30 | 256 | 28 | – | 365 |
| 15.85 | 46 | 4 | 30 | 208 | 28 | – | 317 |
| 14.25 | 46 | 4 | 30 | 176 | 28 | – | 285 |
| 12.65 | 46 | 4 | 30 | 144 | 28 | – | 253 |
| 8.85 | 46 | – | 8/5/8/5 = 26 | 80 | 24 | – | 177 |
| 6.60 | 36 | – | 8/5/5/5 = 23 | 48 | 24 | – | 132 |

The mode is carried out-of-band; the only non-parameter bit in the frame is a
1-bit VAD flag. **The whole non-excitation side is ≈106 bits at 12.65 kbps**
(ISP 46 + pitch 30 + gains 28 + flags), leaving 144 bits for 4 subframes of
excitation.

### 1.2 ACELP algebraic codebook (ISPP, TS 26.190 §5.8)

- 64 positions per subframe split into **4 interleaved tracks of 16**, spacing 4:
  track *t* = positions ≡ *t* (mod 4). (6.60 kbps uses 2 tracks of 32, spacing 2.)
- Pulses per track and bits per track (M = 4 position bits; amplitude ±1):

| Mode | Pulses | Bits/track | Bits/subframe |
|---|---|---|---|
| 6.60 | 2 (1 each, 2 tracks) | 6 = 5 + 1 sign | 12 |
| 8.85 | 4 (1/track) | 5 = 4 + 1 sign | 20 |
| 12.65 | 8 (2/track) | 9 | 36 |
| 14.25 | 10 (3,3,2,2) | 13,13,9,9 | 44 |
| 15.85 | 12 (3/track) | 13 | 52 |
| 18.25 | 16 (4/track) | 16 | 64 |
| 19.85 | 18 (5,5,4,4) | 20,20,16,16 | 72 |
| 23.05/23.85 | 24 (6/track) | 22 = 6M−2 | 88 |

- **Sign coding is the lever:** 2 pulses per track use only **one** sign bit, the
  second deduced from pulse ordering; 3+ pulses/track embed signs in a sectioned
  index, so the marginal cost falls to ≈4 bits/pulse at high counts. There is no
  codebook memory — the excitation is reconstructed algebraically.
- An adaptive prefilter `1/(1 − 0.85 z^−T)` × tilt is folded into `h(n)`.

### 1.3 Fractional pitch (TS 26.190 §5.7)

- **1/4-sample** resolution for lags in [34, 127¾], 1/2 for [128, 159¾], integer
  [160, 231]; subframes 2 and 4 use 1/4 around the previous lag ±8. Search ±7
  around the open-loop lag, testing fractions −¾…¾ in ¼ steps.
- Two FIR Hamming-windowed-sinc interpolators. Prose says sinc truncated at ±17
  and ±63; the **reference C uses 8-tap (correlation) and 32-tap (excitation)**
  4-phase filters (`pitch_f4.c`, `pred_lt4.c`) — prefer the C numbers.
- The low-pass character means even integer lags are filtered, not copied.

### 1.4 Gains (TS 26.190 §5.9)

- Adaptive (pitch) gain and fixed-gain correction factor γ are **jointly VQ'd**:
  6-bit codebook at 6.60/8.85, **7-bit** otherwise, **per subframe**.
- The fixed gain is *predicted*: 4th-order MA on innovation energy,
  `b = [0.5,0.4,0.3,0.2]`, mean 30 dB; only γ is transmitted.

### 1.5 ISP quantisation (TS 26.190 §5.2.5)

- Order 16 in the ISP/ISF domain, **1st-order MA prediction with coefficient
  1/3**, **split-MSVQ** with split 9+7.
- Stage 1: 8 + 8 = 16 bits. Stage 2: 6+7+7 (r1) + 5+5 (r2) = **46 bits** total
  (36 at 6.60). Interpolated across subframes as
  `q̂1 = 0.55 q̂4(n−1) + 0.45 q̂4(n)`, `q̂2 = 0.2/0.8`, `q̂3 = 0.04/0.96`.
- The standard publishes **no** spectral-distortion figure; do not attribute one.

### 1.6 Encoder search (TS 26.190 §5.3, §5.8.3)

- Weighting `W(z) = A(z/γ1)·H_de-emph(z)`, **γ1 = 0.92**, de-emphasis
  **β = 0.68** (no γ2 denominator; pre-emphasis supplies the tilt).
- Criterion `max (dᵗc)²/(cᵗΦc)` with `d = Hᵗx2`, `Φ = HᵗH`.
- **Sign pre-selection** from a likelihood vector `b(n) = √(Ed/Er)·r_LTP + α·d(n)`,
  with sign-folded `d'`, `φ'`; α from 2 (low rates) down to 0.5.
- **Depth-first tree search, 2 pulses at a time in consecutive tracks**; only
  consecutive-track φ blocks + the diagonal are stored (1088 words vs 4096).

### 1.7 G.729 (8 kbps, 10 ms, 2×40-sample subframes)

- LSF: switched 4th-order MA, **18 bits** = 1 select + 7 + 5 + 5.
- Pitch: **1/3-sample**, P1 8 bits, P2 5 bits, plus a parity bit.
- Fixed codebook: 4 pulses (ISPP), **13 position bits + 4 sign bits = 17/subframe**.
- Gains: 7 bits/subframe, two-stage conjugate VQ (3+4); fixed gain predicted with
  `b = [0.68,0.58,0.34,0.19]`.
- Weighting `A(z/γ1)/A(z/γ2)`, γ1=0.94, γ2=0.6 (tilted); reduced form γ=0.75.
- "Focused search": the 4th pulse loop runs only above a correlation threshold
  (K3 = 0.4), ≤ ~90 entries/subframe.

### 1.8 EVS (TS 26.445 / TS 26.443 reference)

- Subframe is **64 samples at any internal rate**: 4×5 ms at 12.8 kHz, **5×4 ms
  at 16 kHz**; order 16; native rates 5.9–24.4 kbps (WB) plus ACELP@32/48/64.
- Pitch resolution **1/4**, range [34, 91½] fractional / [92, 231] integer at
  12.8 kHz ([36, 289] at 16 kHz).
- Algebraic codebooks: 7-bit = 1 pulse/64; 12-bit = 2 pulses in 2 tracks of 32;
  20-bit = 4 pulses in 4 tracks of 16; larger = multi-track joint indexing.
- Gains: per-subframe joint VQ (6/7 bits), algebraic-gain energy predicted once
  per frame with 3–5 bits, **memory-less**.
- LSF: frame-end MSVQ (1–4 stages) + MSLVQ lattice stage, with
  safety-net / MA-predictive / switched AR predictors; 22–41 bits by mode.
- Adds content-adaptive core switching (ACELP ↔ TCX/MDCT + HQ), explicit
  TBE/FD-BWE/IGF extension layers, and channel-aware 13.2 kbps redundancy.

## 2. Opus — RFC 6716, the SILK (LP) layer

At 8–16 kbps for 16 kHz speech, **Opus in practice is SILK** (RFC §2: the MDCT
layer is not used for speech at WB or below; "sweet spots" 8–12 kbps NB,
16–20 kbps WB).

### 2.1 LSF quantisation (§4.2.7.5)

- Order 16 for WB. **Two-stage VQ**: stage 1 is a **32-entry codebook** (4–5 bits,
  Appendix-computed entropy 4.06–4.74); stage 2 is **predictive scalar
  quantisation per coefficient** against one of 16 PDFs selected by the stage-1
  index, with intra-frame **backward prediction** of the *next-higher*
  coefficient and reverse coefficient order.
- Measured entropy from the RFC's own CDF tables: stage 2 ≈ **15.5–20.4 bits**
  (NB/MB) and **20.0–28.9 bits** (WB), so **LSF ≈ 19–33 bits per 20 ms frame** —
  the largest fixed side-info field.
- Indices are in −4…+4 with an escape symbol to ±10.
- Search is rate–distortion with **Laroia IHMW weights** and survivor-based
  multi-stage search; stage 2 is **delayed-decision (Viterbi)** scalar
  quantisation — a training-free R–D gain at fixed rate.

### 2.2 Pitch / LTP (§4.2.7.6)

- Primary lag absolute = 4.34 bits (high) + 2–3 bits (low, uniform); relative
  lag = **3.73 bits**. A subframe **pitch contour VQ** (3–34 entries,
  1.4–4.5 bits) refines it.
- **LTP is a 5-tap filter per 5 ms subframe**, not a scalar gain. Three codebooks
  of **8/16/32** vectors at **1.61 / 3.68 / 4.85 bits per subframe**; **all
  subframes share one codebook per frame** (a periodicity index, 1.58 bits).
- Subframe **gains are 6-bit log**, ≈1.369 dB resolution, **delta-coded**
  (2.99 bits) after the first (≈2.1–2.6 bits).

### 2.3 Noise-shaping quantiser (§5.2.3.3, §5.2.3.8)

The scalar quantiser is embedded in an analysis-by-synthesis loop with a shaping
filter

```text
H(z) = G · (1 − c_tilt z^-1) · W_ana(z)/W_syn(z)
a_ana(k) = a(k)·g_ana^k,  g_ana = 0.95 − 0.01C
a_syn(k) = a(k)·g_syn^k,  g_syn = 0.95 + 0.01C
```

i.e. **more bandwidth expansion on the analysis side than the synthesis side**,
so spectral valleys are de-emphasised relative to formants/harmonics — the RFC's
own argument is that this *lowers excitation entropy*, and it is explicitly
"without substantially changing the bitrate". An optional delayed-decision mode
runs a Viterbi over rounding choices with a 32-sample delay.

### 2.4 Range coder (§4.1, §5.1)

- Integer-exact FIFO arithmetic coder, renormalising at `rng > 2²³`, with the
  truncation error deliberately accumulated on symbol 0.
- **Every quantised parameter has its own CDF from a training histogram**,
  context-selected by already-decoded quantities (signal type, bandwidth, frame
  size, stage-1 index, rate level, periodicity index).
- The encoder also *chooses* the excitation rate level (9 models, ~3 bits) and
  the LCG seed (2 bits) inside the R–D loop — "costless modelling gain".

### 2.5 Worked side-info budget (WB voiced, 20 ms)

Sum of non-excitation fields ≈ **79 bits** (`[C]`), leaving ≈241 bits at 16 kbps
and ≈161 at 12 kbps for 20 shell blocks of excitation. **The efficiency at
8–16 kbps is the efficiency of that side-info layer**, not of the excitation
quantiser.

## 3. CELT / PVQ — the training-free structured quantiser

- CELT is the MDCT layer (48 kHz internal, 2.5–20 ms, 21 bands). Bands are coded
  **gain/shape separated**: coarse energy is 6 dB-log with a 2-D
  time/frequency predictor and Laplace residual coding; the unit-norm shape is
  **PVQ**.
- **PVQ** index space = integer vectors with `Σ|y_j| = K` in `N` dims,
  `V(N,K) = V(N−1,K) + V(N,K−1) + V(N−1,K−1)`. The codeword is a **uniform
  integer in `0..V(N,K)−1`**, so its rate is exactly `log₂ V(N,K)` — **no trained
  table, no learned PDF, exact rate accounting**. Codebooks cap at 32 bits and
  split recursively with entropy-coded relative gains.
- This is the strongest training-free primitive available for excitation/shape
  coding (Fischer, *A Pyramid Vector Quantizer*, IEEE Trans. IT, 1986).

## 4. Lyra — why it is a different information budget

- Lyra V2 is SoundStream-based: encoder → **residual VQ** → LyraGAN decoder;
  bitrates 3.2/6/9.2 kbps = **64/120/184 bits per 20 ms frame**
  (`lyra_config.cc`), 16 kHz internal.
- 184 bits/320 samples is **0.575 bit/sample**: >96% of the output waveform is
  supplied by the **trained decoder prior**, not the bitstream. A deterministic,
  training-free decoder has no such prior, so matching Lyra at 3.2 kbps is not a
  tuning problem — it is a different information budget. Documented neural
  failure modes (phoneme hallucination; worse noise robustness than classical
  codecs) are the honest counterpoints, not licence to claim parity.
- A 2023–2026 literature sweep found **no** training-free deterministic codec
  that beats or approaches Opus/EVS at 8–16 kbps. The levers that remain open to
  a training-free design are the *structural* ones: PVQ + combinatorial
  enumeration, predictive/uniform scalar quantisation with parametric Laplace
  models, delayed-decision quantisation, noise shaping, and content-adaptive bit
  allocation.

## 5. What this implies for VOLE (mapped to the ordered plan)

| Plan | Mechanism | Concrete target from the numbers above |
|---|---|---|
| P1 | weighted error inside AbS | `A(z/γ1)/A(z/γ2)` with γ1∈[0.9,0.94], γ2=0.6; already in `celp.rs` |
| P2 | efficient ACELP | 4-track ISPP, sign-deduced indices, ≈4–6 bits/pulse; lag once/frame + relative; joint gain VQ |
| P3 | fractional pitch | 1/4 sample, 8-tap correlation + 32-tap excitation 4-phase filters |
| P4 | ISP + MA prediction | order-16 split MSVQ on top of the frozen LSF table, 1/3 MA prediction |
| P5 | index entropy coding | range/ANS over *transmitted* indices with parametric models (≈10–25% on gains/lags/LSF; ≈0% on ACELP positions) |
| P6 | subframe allocation / mode | per-5 ms gains (SILK) or joint gain VQ (AMR-WB); voicing-driven mode switch |

## 6. Non-claims

This document reports mechanisms and numbers from public specifications and
papers. It imports no competitor code, and none of the figures above is a claim
about VOLE's current performance — that is in [`VOICE_RD.md`](VOICE_RD.md).
