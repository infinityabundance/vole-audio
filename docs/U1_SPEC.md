# U1_SPEC — `vole.audio.u1` (normative)

Status: **FROZEN for the code in this tree** (Phase B). Every statement below
has an executable counterpart in `src/universe/`; the unit tests enforce it.
Changing any frozen item is a **profile change** (`u1/v2` at minimum): new
profile id, new reference vectors, never a silent edit.

Source of truth: VOLE-Audio v1.0 paper (DOI 10.5281/zenodo.22649073 — the
disclosure that normatively defines U1; the current architecture revision
v1.1, DOI 10.5281/zenodo.22666746, makes the entropy layer explicit and does
not contradict U1) + this spec + the code. Where this spec and the code
disagree, the code is wrong.

---

## 1. Identity

| constant            | value            |
| ------------------- | ---------------- |
| universe id         | `vole.audio.u1`  |
| profile id          | `u1/v1`          |
| profile version     | `1`              |
| default rate        | 48 000 Hz        |
| default quantum     | 1024 frames      |

## 2. Canonical sample domain

`SampleCode = i32`; the **entire** i32 range is valid. Integer PCM ingest is
exact:

| source          | mapping                          |
| --------------- | -------------------------------- |
| u8              | `((x as i32) - 128) << 24`       |
| s16             | `x << 16`                        |
| s24             | `sign_extend_24(x) << 8`         |
| s32             | `x` (identity)                   |

Consequences (documented, not bugs): u8 maps to `[-2^31, 2^31 - 2^24]`;
`from_u8(128) == 0`.

Exactness claim: **audio-sample reconstruction**, not WAV container-byte
reconstruction. WAV metadata/chunk layout is outside the u1 equality claim.
Floating WAV input is rejected by the exact profile (no silent quantization).
A lossy profile, if added, records the transformed target hash separately.

## 3. Time model

Separate axes, never conflated:

* media time — integer frames since media epoch (`MediaFrame` i64);
* object intrinsic coordinate — integer frames since object start
  (`ObjectFrame`);
* endpoint frames consumed by the physical endpoint (`EndpointFrame`);
* nominal sample rate — integer Hz (`NominalRate`);
* endpoint physical clock — measured host-side, never media-authoritative;
* wall-clock instrumentation — monotonic only.

`f64` seconds are never authoritative media time. Epoch id is `u32`
(`EpochId`), incremented only by explicit discontinuity policy
(`XrunPolicy::{PreserveTimeline, Discontinuity, RestartEpoch}`), recorded in
receipts.

## 4. Fixed-point arithmetic (exact)

| quantity            | type | Q   | unity                       |
| ------------------- | ---- | --- | --------------------------- |
| source position     | i64  | 24  | —                           |
| rate increment      | i64  | 24  | `1 << 24`                   |
| gain/pan/env/amplitude | i32 | 16 | `1 << 16`                   |
| sine table entries  | i32  | 30  | peak `±(2^30 - 1)`          |

Rules:

1. **Rounding**: round half away from zero, applied once per defined reduce
   point (`rnd_shift(v, q)` = `(v + 2^(q-1)) >> q` for `v >= 0`, mirrored for
   `v < 0`; input bound `|v| < 2^62`).
2. **Saturation** (`sat_i32`) only at defined boundaries: voice bus (after
   gain/pan/envelope), generator amplitude entry, and the final output.
3. **Mix accumulation**: pure i64 addition, order-independent, never
   saturated until the output boundary.
4. Q16 multiply-reduce `mul_q16(a, b) = sat(rnd(a·b, 16))`; unity multiplier
   is the identity (`mul_q16(x, 1<<16) == x`).
5. Linear interpolation `lerp(a, b, frac24) = sat(a + rnd((b-a)·frac24, 24))`,
   used for both sample interpolation and sine-table interpolation.
6. Positions split as `idx = pos >> 24`, `frac = pos & 0xFF_FFFF`.

**Mixing bound proof.** Each voice's contribution is saturated to
`|c| <= 2^31 - 1` before entering the per-channel i64 mix. With
`MAX_ACTIVE_VOICES = 4096`: `|mix| <= 4096·(2^31 - 1) < 2^43`, leaving a
2^20 margin below `i64::MAX`. i64 addition is associative, so scalar, SIMD,
and GPU reduction trees may differ without changing the result — provided the
per-(voice, frame) contribution is identical and gain/envelope are applied
inside it identically.

## 5. Oscillator phase (free-running DDS)

* Phase is `u64`, advancing modulo 2^64.
* Increment: exact `round(2^64 · f / r)` computed in 64-bit arithmetic only
  (`freq_to_incr`; clamp `f <= r/2` Nyquist). No 128-bit division anywhere in
  the device-compiled core.
* Addressing: `index = phase >> 52` (12 bits, 0..4095 by construction),
  `frac24 = (phase >> 28) & 0xFF_FFFF`. Interpolation between adjacent table
  entries with the exact linear rule; the table is periodic (wrap via
  `(index + 1) & 4095`).
* Amplitude reduce: `sine_amp_to_code(t30, amp16) = sat(rnd(t30·amp16, 15))`;
  peak at unity amplitude is `±(2^31 - 2)` (never saturates at unity).

### Frozen table: `assets/u1/sine_q30_4096.bin`

* 4096 × i32 little-endian, Q30, values in `[-(2^30-1), 2^30-1]`.
* SHA-256: `455d4647044595871d5f07789581abc28a6499b2bf622f4f3b62ed85297fdf21`
* Structure is bit-exact: odd-symmetric about index 0
  (`S[i] == -S[(4096-i) & 4095]`), anti-symmetric half-turn
  (`S[i] == -S[(i+2048) & 4095]`), even-symmetric about index 1024
  (`S[1024+d] == S[1024-d]`), `S[0] = S[2048] = 0`, `S[1024] = 2^30 - 1`,
  `S[3072] = -(2^30-1)`.
* Regeneration (from a computed quadrant + integer mirror) is **non-normative**
  after freeze; the committed bytes are authoritative.

## 6. Event model

Total order over `(MediaFrame, EventClass::priority, sequence)` where
`sequence` is the unique monotonic arrival id. Class priorities:

| class        | priority |
| ------------ | -------- |
| Start        | 0        |
| Stop         | 1        |
| Param        | 2        |
| World        | 3        |
| Clock        | 4        |
| Diagnostic   | 5        |

Ordering never depends on hashmap iteration, thread scheduling, GPU lane
order, or allocation order.

## 7. PRNG (frozen)

* `VOLE-SPLITMIX64-STREAM`: noise is a pure function of
  `(stream_key: u64, frame: i64)` — `splitmix64_finalize(scramble(key) ^
  mix(frame))`, cast to the full i32 domain. Random access holds by
  construction (`seek == sequential`); no hidden per-voice noise state.
* `VOLE-XOSHIRO256STARSTAR-1`: xoshiro256** with 256-bit state, splitmix64
  seeding, canonical `jump`.
* `rand::thread_rng()` is never used in normative media semantics.
* SplitMix64 reference vector (seed 0, first output):
  `0xE220A8397B1DCDAF`.

## 8. Layout and canonical observation form

* Channel layouts: `Mono`, `Stereo`, `Channels(1..=32)` (`MAX_CHANNELS`).
* Canonical observation byte form: frame-interleaved,
  `[ch0 f0, ch1 f0, ..., chC-1 f0, ch0 f1, ...]`, each code as **little-endian
  i32**.
* Canonical observation hash: SHA-256 over exactly those bytes
  (`observation_sha256`). This is the reference-equality primitive for
  differential courts and reference vectors.

## 9. Interpolation and resampling

* Nearest and linear are integer-exact as specified in §4; read-continuation
  rules: one-shot holds the final sample (`b = a`); loop `[a, b)` wraps the
  interpolation neighbor at the region end back to `a` (periodic
  continuation). Reads at the wrapped coordinate are half-open `[a, b)`.
* **Frozen polyphase resampler** (`sampler::resampler`): 64 taps × 1024
  phases, i16 Q15 coefficients, Blackman–Harris (4-term) windowed sinc with
  cutoff at source Nyquist, phase-major canonical bytes in
  `assets/u1/resampler_bh64_p1024_q15.bin` (128 KiB).
  * Evaluation: `y = sat(rnd(Σ_j c_j·x_j, 15))`, tap `j` reads source frame
    `m + j - 31` (`m = floor(pos)`); one rounding, one saturation.
  * Every phase row sums to exactly `2^15` (largest-remainder quantization),
    so DC gain is exactly 1.0 in Q15 arithmetic: constant input reproduces
    itself through every phase.
  * SHA-256: `3b3015a81b9da1298b532127e72309f7dd1afdbf817628dbcabd8fe958438750`
  * Measured (host tests `measure_response` / `quantized_table_response_bounds`;
    the two figures are deliberately separate):
    - **Continuous design (analog prototype only)**: passband ripple
      `max |H-1| = 8.3e-5` (0.0007 dB) for F ≤ 0.45; stopband attenuation
      ≥ 210 dB for F ∈ [0.5625, 1] — window-sidelobe numerics of the
      *pre-quantization* kernel. This is a prototype property (relevant to
      image rejection when the kernel is used for oversampling), not a
      property of the quantized table.
    - **Frozen Q15 table (the arithmetic the evaluator runs)**: DTFT of all
      1024 quantized rows; worst passband deviation `max |H-1| = 1.1e-3`
      (0.0095 dB) for F ≤ 0.45 (edge-dominated at F = 0.45), and the same
      worst deviation in the beyond-Nyquist image band F ∈ [0.55, 0.95]
      (exact mirror of the passband). The rows are critically sampled
      (cutoff == source Nyquist), so **no digital stopband exists**: the
      beyond-Nyquist response is the periodic passband image, and
      coefficient quantization is therefore quoted as amplitude error
      (≤ 1.1e-3), never as a stopband figure.
  * Table footprint 128 KiB; cost 64 multiply-adds per output sample.
  * The resampler is a **fractional-coordinate interpolator** (not a
    ratio-adaptive decimator); object reads default to the linear path and
    the polyphase path is an explicit quality transform.

## 12. Sampler transforms (frozen, Phase C)

Position/loop semantics (see also `sampler::rate`, `sampler::voice`):

* Signed Q24 rate; domain `0` or `[±2^8, ±2^40]`; position
  `u(t) = p0 + rate·(t − t0)` — analytic, no per-frame state.
* Loop mapping for both directions is the single Euclidean wrap
  `w = A + ((u − A) mod L)` (reverse tape looping = negative rate).
* One-shot ends exactly at the first out-of-range frame (`end_frame`); reads
  are half-open `[0, extent)`.
* Reference transpose composes with rate via one Q24 rounding
  (`compose_transpose`, saturated at `±2^47`); effective rate revalidated.

Envelope: analytic piecewise-linear ADSR as in `sampler::envelope` (Q16,
round-half-up segment law, zero-length segments jump instantly, release
starts at the note-off frame's level). Gain chain and pan law as in
`sampler::gain`/`pan` (unity-identity chain, `L + R == unity` exact). Mix:
per-(frame, channel) i64 accumulators, one final saturation.

Filter set: `Biquad` (RBJ Butterworth lowpass, integer coefficients from the
frozen sine table, Direct Form I, `DirectStateful`) and `OnePole`
(`y += (x−y)>>k`). State is explicit and checkpointable; neither sits on the
stateless fused path without declared state.

Observation: canonical interleaved output; voices are immutable spawn
records, so `chunked == contiguous` and `seek == sequential` hold by
construction (verified by `court semantic`).

## 10. Host clock (non-media)

Endpoint scheduling math uses integer frames; host frame<->microsecond
conversions round half-up at the documented rate (`HostClock`). xrun policy
is recorded per receipt.

## 11. Reference vectors (Phase B freeze)

Covered by unit tests: FIPS 180-4 SHA-256 vectors; u8/s16/s24/s32 mappings;
rounding ties (`rnd_shift` half-away at Q1/Q2/Q24); Q16 unity identity;
saturation edges; lerp oracle sweep (i128 reference) incl. extreme spans;
`freq_to_incr` vs a u128 oracle across rates; sine table symmetry + frozen
SHA-256; SplitMix64 first-output vector; event-order determinism; interleave/
deinterleave roundtrip; empty-observation hash
`e3b0c442...52b855`. Object-level vectors (small SampleObjects and their
expected observation hashes) freeze in Phase C.

### Phase C vectors (frozen)

* `court semantic` fixture observation hash (2400 frames, stereo, 4 voices:
  one-shot ramp w/ ADSR, looped voice w/ note-off, referenced-object reverse
  at half transpose, panned stereo object):
  `1791816f4b938375cc4298b2587ce19eef260d063d9ed88c597e04d31837f6d0`
  (enforced by `courts::semantic::tests::fixture_reference_hash_is_frozen`).
* Resampler table SHA-256 and sine table SHA-256 (see §§5, 9).

### Phase D vectors (frozen)

* `court authored` fixture observation hash (4000 frames, mono; silence,
  constant, deterministic noise, oscillator, 8-partial bank, wavetable,
  single-cycle, exact-repeat, and a half-transpose reference to a wavetable):
  `f91b5b4228022d46a609a2fe7fc862e6e72c4058405c4b72316b202feb14fe87`
  (enforced by `courts::authored::tests::fixture_reference_hash_is_frozen`).

## 13. Procedural generators (frozen, Phase D)

Object payloads and observation semantics:

* `Silence`/`Constant`/`Noise`/`Oscillator`/`PartialBank` are **endless**
  (extent 0, no natural end; observation never depends on stored samples —
  zero resident sample bytes by construction).
* `Wavetable`/`SingleCycle`/`ExactRepeat` hold one stored **cycle** (a
  resident table; bytes counted as table/dependency) and always wrap the
  cycle; voice loop regions are rejected for them.
* Oscillator/partial-bank phase advance modulo 2^64 with per-output-frame
  increment `eff_incr = clamp(round(2^64·f0·rate/2^24/fs), 2^63)` computed
  host-side (device receives the increment). Fundamentals clamp at Nyquist
  (no alias fold); harmonic partials alias-fold through their modulo-2^64
  increment arithmetic.
* Oscillator output `sat(rnd(T30·amp, 15))` from the frozen sine table;
  partial banks accumulate in i64 and saturate once at the generator output
  boundary (order-independent).
* Noise: `VOLE-SPLITMIX64-STREAM` keyed by object seed and the **media
  frame** (identical objects at the same media frame sound identical).
* Reference transpose composes with rate; endless sources require start
  position 0 and no voice loop region (malformed otherwise).

## 14. Residual closure (frozen, Phase E)

* Sampled-origin intrinsic closure `X_O(f,ch) = H(f,ch) + R(f,ch)` — exact,
  in the code domain — precedes observation. No `T(H)+T(R)` commutation is
  ever assumed.
* v1 models: `Zero`, `Constant(level)`, `Periodic(cycle)` (mono cycle
  inline). Model output is code domain; channels beyond the model domain
  model 0.
* Sparse residual records (frame, channel, i32 delta), unique per
  (channel, frame), sorted, count `<= MAX_RESIDUAL_RECORDS`; closure adds
  one i64 sum + one saturation. Exact closure of an intrinsic requires
  `|X_O − H| < 2^31`; uncloseable hypotheses are rejected at construction.
* Residual-governed reads interpolate reconstructed neighbors, so
  integer-frame observation equals the literal observation of the same
  content (verified: `residual_closure_equals_literal_through_the_world`).
* WAV ingest (`format::wav`): integer PCM only (u8/s16/s24/s32), explicit
  rejection of float/other tags; hostile-input tested; maps to canonical
  codes per §2. Container metadata is outside the u1 equality claim.
