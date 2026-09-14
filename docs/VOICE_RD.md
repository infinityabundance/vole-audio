# Voice rate–distortion: measured attribution and the mechanism plan

This note is the working research record for the voice-call profile
(`vole.audio.stream.voice.exp1`, Phase 7C). It records **what the codec actually
does**, attributes the measured loss against the external competitors to
specific mechanisms, and specifies — with bit-budget arithmetic — the ordered
set of mechanisms that must be built to close it. It is an engineering note, not
a claim: every number below is a measurement from this tree, and every
prediction is labelled as one.

Everything measured here is reproducible with the committed development
instruments:

```text
cargo run --release --example voice_bench            # clean RD ladder (VOLE only)
cargo run --release --example voice_bench -- pg      # prediction gain (open loop)
cargo run --release --example voice_bench -- diag    # closed-loop scalar RD curve
cargo run --release --example voice_bench -- cmp     # scalar vs CELP head-to-head
./target/release/vole-audio court learned-voice-stream
```

`voice_bench` is a development instrument; the authoritative measurement is
always `court learned-voice-stream`.

## 1. The measured position

Court `learned-voice-stream`, frozen corpus (12 dev-clean clips → 3 cases of
≈4.1 s at 16 kHz mono), matched **actual** bitrate, SNR after global best-lag
alignment, ViSQOL in speech mode:

| increment | vs Opus (mean ΔSNR) | wins | vs EVS (mean ΔSNR) | wins | vs Lyra |
| --- | --- | --- | --- | --- | --- |
| v0.76.0 (scalar reflection codes) | −8.36 dB | 0/3 | −5.75 dB | 0/3 | no cell in range |
| LSF MSVQ + CELP excitation | −7.40 dB | 0/3 | −4.77 dB | 0/3 | no cell in range |
| + combinatorial pulse-position coding | **−7.15 dB** | 0/3 | **−4.32 dB** | 0/3 | no cell in range |

Result identity: `cc2addfa019a96b3c722ddaabe9fd0dff7eee4ed0cc47039eb435acecc2dba50`
(frozen in `src/courts/learned_voice_stream.rs`). Encode p99 is 4.57 ms
against the 5 ms constitution; impaired-cell mean SNR 3.42 dB.

Three further mechanisms were implemented this session and **rejected on
measurement** (each retained in the tree, defaulted off, and re-openable — the
project's win / near-negative / reject discipline):

```text
per-subframe residual gain (SUBFRAME_GAIN)   matched means −7.42 / −5.15 dB
                                             encode p99 6053 µs, max 8276 µs
per-frame encoder state advance              matched means unchanged;
                                             impaired mean 3.42 → 2.94 dB;
                                             encode p99 4.57 → 5.69 ms
fit-preferring winner selection              matched means −7.42 / −5.15 dB
```

None beat the state above, so each is off. The measured attribution and the
grounded mechanism reference are in
[`VOICE_MECHANISMS.md`](VOICE_MECHANISMS.md).

The per-cell detail shows the shape of the problem. On case `1272-128104-0000`,
in the four matched cells that exist:

```text
VOLE actual rate 17.4–24.8 kbps, VOLE SNR ≈ 6.7–10.9 dB, VOLE ViSQOL ≈ 1.33–2.15
Opus at the same rate   SNR ≈ 17.2–19.9 dB   ViSQOL ≈ 3.79–3.90
EVS  at the same rate   SNR ≈ 14.1–14.3 dB   ViSQOL ≈ 3.68–3.95
```

So at an identical actual bitrate VOLE is **10–11 dB** behind Opus on SNR and
**1.7–2.5 MOS** behind on ViSQOL. The gap is not a small tuning deficit.

## 2. Attribution: where the bits go, and where the decibels are lost

### 2.1 The predictor is not the problem

Measured open-loop over 600 frames of 320 samples (true signal as history):

```text
short-term (LPC order ≤ 16, width-6 codes) : 18.31 dB prediction gain
+ long-term (integer pitch, single tap)    : 18.97 dB
```

18.3 dB of formant prediction is healthy; this is not where the loss is.

### 2.2 The residual quantiser is the problem

At 320 samples (20 ms) and a 6 kbps target the encoder emits **17.4 kbps**
(≈43.5 B/frame; ≈9.7 B model, ≈33 B residual, i.e. ≈0.83 bits/sample of
residual), and the aligned SNR is **6.63 dB**.

The closed-loop scalar RD curve (encoder-side, `diag`) is:

```text
budget  used      rate        SNR
  40 B   34.2 B   13.7 kbps   2.79 dB
  48 B   41.4 B   16.6 kbps   4.16 dB
  64 B   58.5 B   23.4 kbps   6.92 dB
  96 B   91.7 B   36.7 kbps  19.53 dB
```

Two facts matter:

1. **The slope is ~6 dB per rate doubling** — the signature of *scalar*
   quantisation. Vector quantisation is what buys the extra decibels at low
   rate, and the codec is not getting them.
2. **There is a cliff between ≈59 and ≈92 B/frame**: +12.6 dB for 2.75× the
   rate. That is not a normal RD trade. It is the coarse-step collapse of
   *closed-loop* scalar DPCM: once the dead-zone step is coarse relative to the
   residual, the reconstruction error fed back through the synthesis filter
   overloads, and the loop operates far from its fine-quantisation potential.

For the same frame the arithmetic is stark. With 18.3 dB of prediction and
0.83 bits/sample of residual, a fine-regime scalar DPCM would sit near 25–30 dB.
It measures 6.6 dB. The gap is the closed-loop overload, and it is the single
largest identified loss.

### 2.3 The CELP excitation coder is currently a net negative

The excitation path (`src/voice/celp.rs`) is genuine analysis-by-synthesis: it
minimises the perceptually weighted *synthesised output* error (not the residual
error), which is the correct principle. But its wire cost is too high to win:

```text
per 80-sample subframe:
  lag 9 b + pitch gain 5 b + pulse count 3 b + gain 6 b  = 23 b overhead
  plus 8 b per pulse (7 b position over 128 values for 80 positions + 1 b sign)
4 subframes × (23 + 4×8) = 220 b ≈ 27.5 B + 9.7 B model ≈ 37 B/frame
```

A direct end-to-end A/B with the CELP selector disabled (same tree, one
constant) makes the effect unambiguous:

```text
320×1, target 12 kbps   CELP on : 23 153 bps, SNR  8.82 dB
                        CELP off: 23 660 bps, SNR 10.40 dB
```

CELP used **fewer** bytes and delivered **1.6 dB less**. At 6/8 kbps the cells
are identical (CELP never fits). At 24/32 kbps it helps slightly. So the current
ACELP is a correct mechanism with a crippling cost model: 23 bits/subframe of
overhead and 8 bits/pulse, against AMR-WB-class figures of ≈4 bits/pulse and a
lag transmitted once per frame with differential/gain sharing.

The first fix to that cost model is in: the pulse *positions* are now coded
combinatorially (`ceil(log2 C(80, n))` bits rather than `7n` — 21 bits for four
pulses instead of 28, 29 instead of 42 for six). Measured effect: the 24 kbps
cell gains **+0.43 dB at 742 bps less**, the court's matched means move
−7.40 → −7.15 dB (Opus) and −4.77 → −4.32 dB (EVS). The A/B still shows CELP
losing at 12 kbps (23 344 bps / 8.92 dB with it, 23 660 bps / 10.40 dB
without), so the remaining cost — the 23 bits/subframe of overhead — is the
next thing to cut.

### 2.4 The encoder does not honour its rate target

At a 6 kbps target the codec emits 17.4 kbps (2.9×), by design:
`choose_gain` prefers a fitting step and otherwise accepts up to
`OVERSHOOT_LIMIT = 4` × the frame allowance. The overshoot is disclosed and
honest, but it means the matched-bitrate comparison is always against a much
higher-rate competitor point. Rate control is therefore also a quality lever:
every byte not spent on the model or on wasted excitation is a byte that buys
distortion.

## 3. The governing mechanism

Modern low-rate speech codecs are not scalar DPCM. They are **analysis-by-
synthesis vector quantisers** for two reasons that this codec must adopt:

1. **Excitation VQ.** At 0.5–1 bit/sample, scalar quantisation of the residual
   is 3–6 dB worse than vector quantisation of the excitation at the same rate
   (rate–distortion theory: the space-filling advantage plus codebook shaping).
   This is the mechanism behind ACELP, and it is why the current codec cliffs.
2. **Noise shaping / weighted synthesis-domain error.** The decoder is
   `x̂ = 1/A(z) · u`. Error in the excitation is coloured by `1/A(z)` (the
   formant gain). Both CELP's weighted search and SILK's noise-shaping quantiser
   (NSQ) place the quantiser error in the *synthesised* domain, where the ear —
   and the SNR metric — actually sit. VOLE's scalar path does not.

The competitors use exactly these, plus efficient index coding:

* **Opus (RFC 6716)** — SILK for speech: LSF **split VQ with MA prediction**,
    LTP with **VQ'd** pitch-lag/gain, an **NSQ** stage, and a **range coder**
    with trained probability tables; CELT (PVQ/MDCT) for the rest, with a
    redundancy/PLC layer.
* **EVS (TS 26.445)** — ACELP with fractional pitch and a larger algebraic
    codebook, plus bandwidth extension at the low modes.
* **Lyra** — a learned generative codec; it operates at 3.2–9.2 kbps and
    VOLE currently has no operating point that low, so the cell is empty rather
    than lost.

VOLE cannot copy their codecs, and must not: it needs deterministic,
training-free, bounded-decoder mechanisms. The good news is that the decisive
ones are deterministic and codebook-free (algebraic codebooks, NSQ, arithmetic
coding on transmitted indices).

## 4. Bit-budget arithmetic for Opus-class speech at 8–16 kbps

Frame: 20 ms @ 16 kHz = 320 samples. Rate → bits/frame:

```text
 8 kbps = 160 b ·  12 kbps = 240 b ·  16 kbps = 320 b
```

Target allocation (AMR-WB-class, order-16, 4 subframes of 80):

| field | bits | mechanism |
| --- | --- | --- |
| spectrum (order-16) | 30–46 | ISF/ISP, split VQ, MA prediction |
| pitch lag | 8–10 | once per frame + differential per subframe |
| pitch gains | 16–20 | 4 subframes, VQ or differential |
| innovation gains | 12–20 | 4 subframes, joint VQ |
| fixed codebook | 80–112 | ACELP, ≈4 pulses/subframe, interleaved tracks |
| mode / frametype | ≈5 | mode switching |
| **total** | **151–213 b** | **≈7.5–10.7 kbps** |

This is reachable. VOLE today spends ≈72–100 bits on the model and ≈256 bits on
a **scalar** residual, which is both more expensive and less effective than the
table above.

## 5. Ordered mechanism plan

The order is the plan the profile already declared; each step is implemented,
tested, measured by the court, and sealed, with the previous state kept as the
control. No step is skipped.

**P1 — Perceptual weighting inside the analysis-by-synthesis search.**
Status: implemented in `celp.rs` (`weighted_error_energy`, `weight_gammas`);
the scalar path does not use it. The earlier measurement that plain weighting
lowered *SNR* and breached the 5 ms encode budget was taken when weighting was a
*selection* criterion over a broken quantiser; it must be re-judged with the
court's **ViSQOL** column once P2 is efficient. Acceptance: ViSQOL improves at
equal bytes without a deadline miss.

**P2 — Efficient ACELP excitation (the priority).**
The current coder's cost model is the defect, not its search. Required:
interleaved-track algebraic codebook (T tracks, one pulse per track, position
bits `log2(L/T)`), sign-deduced indices so the marginal cost approaches ~4
bits/pulse (AMR-WB: 2 pulses/track use one sign bit; 3+ embed signs in a
sectioned index), lag transmitted once per frame plus a small per-subframe
correction, joint/differential gain quantisation, and a beam search bounded to
the 5 ms deadline. Acceptance: CELP beats the scalar path on **both** distortion
and bytes at 8–16 kbps, and removes the §2.2 cliff. **Status: combinatorial
position coding is in (a real ≈0.4 dB win at 24 kbps); the track structure and
overhead reduction are not.**

**P3 — Fractional pitch (1/4 sample) with interpolation and a small tap set.**
The predictor's long-term term is currently integer-lag, single-tap and adds
only 0.66 dB open-loop; AMR-WB/EVS get 1–2 dB from fractional interpolation.

**P4 — ISP domain with MA prediction, on top of the frozen MSVQ.** The LSF MSVQ
(`vq.rs`) already cut the model from 40–128 bits to 28; MA prediction over the
transmitted indices (decoder-synchronised, loss-safe because the indices are
absolute in every packet) is the next step.

**P5 — Entropy coding of the transmitted indices.** SILK range-codes every
index against trained tables. VOLE can use an adaptive coder over
*transmitted* indices (safe: both sides see them) or analytically-derived
models, keeping the training-free posture.

**P6 — Subframe bit allocation and mode switching.** Per-subframe gains and
voicing-dependent mode selection (voiced ACELP / unvoiced noise excitation /
transient), which is how AMR-WB and EVS fit the same coder to very different
material.

## 6. Non-claims

* This note claims no victory. It records a **loss** and attributes it.
* The matched-bitrate numbers are objective (aligned SNR, ViSQOL predictions);
  they are not listening-test results.
* The bit-budget table is a **design target**, not a measurement.
* No competitor code is imported, linked, wrapped or used as a fallback; the
  competitor descriptions above are from their public specifications.
* Implemented-but-unmeasured mechanisms (fractional pitch, ISP prediction,
  index entropy coding) are listed as *not yet implemented*, never as wins.
