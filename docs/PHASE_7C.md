# Phase 7C — `vole.audio.stream.voice.exp1`

> **Constitutional document.** Written and frozen **before** the codec exists so
> that the court cannot be tuned to its own results. Everything below is a
> commitment: the profile identity, the frame constitution, the packet layout,
> the latency accounting, the impairment engine, the measurement set and the
> competitor pins. Where an implementation afterwards departs from this text,
> the departure is a defect in the implementation, not an amendment to the
> constitution.

Phase 7 has exactly three objectives. This document governs the third:

```text
7C — implement vole.audio.stream.voice.exp1 and beat the leading voice-call
     codecs on latency + loss + jitter while preserving strong speech quality
```

7C is **not** an archival codec and **not** a static listening score. The thing
being optimised is a *joint* surface:

```text
quality  ×  bitrate  ×  one-way latency  ×  packet loss  ×  jitter
```

A voice codec that compresses beautifully and misses deadlines has failed 7C. A
voice codec that obtains quality by hiding behind a 200 ms jitter buffer has
failed 7C. A voice codec that wins on clean speech and collapses at 3 % loss has
failed 7C.

---

## 1. Profile identity

| item | value |
| --- | --- |
| profile id | `vole.audio.stream.voice.exp1` |
| profile tag bytes | `b"vole.audio.stream.voice.exp1"` |
| container magic | `b"vole.voice"` |
| container version | `1` |
| court | `learned-voice-stream` |
| universe | `vole.audio.u1` |

The profile is an **experimental streaming profile** in the same sense as
`vole.audio.learned.exp*` and `vole.audio.lossy.exp1`: it does not modify `u1`,
does not claim exactness, and the literal fallback remains mandatory elsewhere
in the crate. It is a *transport* representation, not a `SampleObject`.

The **transport** framing (sequence number, timestamp, payload) is defined here
but is deliberately minimal and is **not** a network protocol. No RTP, no ICE,
no STUN, no TURN, no peer discovery, no sockets. The court drives packets
through an in-process deterministic impairment engine (§6).

---

## 2. Frame constitution (7C.1)

```text
sample rate   : 16 000 Hz
channels      : 1
frame lengths : 160 samples (10 ms), 320 samples (20 ms)
```

* **No whole-clip predictor mode exists in this profile.** Every algorithm
  operates causally on bounded frames and bounded state. A window wider than one
  frame is permitted only for *encoder-side analysis* and is disclosed as
  lookahead (§5.3).
* Frame duration is never hidden from the latency number (§5).
* The encoder never reads a sample that the decoder has not already been given
  the means to reconstruct, except through the declared analysis window.

### 2.1 Packet shapes (7C.4)

```text
shape A : 1 × 160  (10 ms)  — minimum accumulation, maximum header rate
shape B : 2 × 160  (10 ms)  — 20 ms of audio, two coded frames in one packet
shape C : 1 × 320  (20 ms)  — 20 ms of audio, one coded frame
```

Shapes B and C both carry 20 ms of audio and both cost roughly one packet
header. They are **not** the same codec: B pays for two sets of frame
parameters, C pays for one but must model 20 ms of evolution with a single set.
The court measures all three; **no shape is selected by default**. Coalescing is
an encoder decision evaluated against the measured trade between header/state
savings, added accumulation latency and loss impact.

---

## 3. Packet layout (normative)

All multi-byte fields are little-endian. Every packet is self-contained: it can
be decoded with no reference to any other packet (§4.4).

```text
VoicePacket
  u8   flags
         bit 0    NO_DATA        1 = inactive/DTX packet (no coded frames)
         bit 1    HAS_CAPSULE    1 = state capsule present
         bit 2    HAS_REDUNDANCY 1 = selective redundancy present
         bit 3    CONCEALED      0 on every transmitted packet (decoder-internal)
         bits 4-5 N_FRAMES − 1  ∈ {0,1}  (1 or 2 coded frames)
         bits 6-7 reserved, must be 0
  u8   rate               residual step index for frame 0
  [u8  sid_level]         only when NO_DATA: comfort-noise level class
  [u8  sid_shape]         only when NO_DATA: comfort-noise spectral class
  [capsule]               only when HAS_CAPSULE
  [redundancy]            only when HAS_REDUNDANCY
  frame record × N_FRAMES

frame record
  u8   mode
         bits 0-1  estimator  0=autocorrelation-Levinson 1=Burg 2=covariance-LSQ
         bit  2    PITCH      1 = long-term predictor active
         bits 3-4  ORDER      0..4  →  8,10,12,14,16
         bits 5-6  WIDTH      0..3  →  5,6,7,8 bits per reflection coefficient
         bit  7    reserved
  u8   gain_delta        residual step index relative to `rate`
  [u8  pitch_hi] [u8 pitch_lo]   only when PITCH: lag in samples
  [u8 ltpg]                      only when PITCH: long-term gain code 0..31
  u8   k_bytes[]         ceil(order × width / 8) packed signed reflection codes
  u32  residual_len
  u8   residual[]        codec id byte + payload (native residual family)

capsule
  u8   capsule_kind      bit-coded: which state fields follow
  ...  state fields, each bounded and canonically ordered
  u8   capsule_hash_lo   low byte of the capsule's content hash (see §4.5)

redundancy
  u8   redundancy_mask   which fields of the *previous* frame are repeated
  ...  those fields, canonically ordered
```

The `mode` byte is the whole per-frame model description. There is deliberately
**no** cross-packet dependence in the parameter fields: the pitch lag, the
long-term gain, the residual step and the coefficient vector are all coded
against a *decoder-visible* predictor (§4.4), never against another packet.

---

## 4. Codec structure

### 4.1 Predictor (7C.2)

Per frame the encoder competes a **bounded candidate set**:

| axis | candidates |
| --- | --- |
| estimator | Tukey-window autocorrelation + Levinson–Durbin; Burg; covariance least-squares |
| order | 8, 10, 12, 14, 16 (bound: 16 for this profile) |
| coefficient quantiser | uniform Q quantiser in the **reflection** domain (`quantize_reflections`) |
| width / shift | `width ∈ {5,6,7,8}` bits, `shift = width − 1`, packed `pack_signed` |

The reflection domain is chosen deliberately: the decoder performs no stability
check because every quantised reflection coefficient is clamped to
`|k| ≤ 1 − 2^−shift`, so the synthesis filter is minimum phase **by
construction**, for every bit pattern. A direct-form quantiser would require the
encoder to validate stability and the decoder to trust or reject — a worse
contract for a lossy profile.

Existing machinery is reused, not reinvented:

* `learned::train::lpc::{tukey_window, autocorrelation, levinson_durbin,
  burg_reflections, lsq_coefficients, quantize_reflections}` for the estimators
  and the coefficient quantiser;
* `lossy::predict::reflection_to_weights` for the synthesis filter;
* `learned::lpc::{pack_signed, unpack_signed}` for bit-exact coefficient packing.

The search is bounded so that the encode deadline in §5.2 holds. If the
measurement shows the search overruns, the search is *constrained*, never the
deadline relaxed.

### 4.2 Long-term predictor (7C.2)

```text
lag  ∈ [32, 288] samples   (55 Hz … 500 Hz at 16 kHz)
gain ∈ [0, 1.5], 32 codes
prediction: pred[n] += ltpg · x̂[n − lag]
```

The long-term reference is the **reconstructed output history**, not a private
excitation buffer. Consequence: the entire decoder state is one bounded ring of
reconstructed samples of length `max_lag + order` (≤ 304 samples), which is what
makes §4.4 and §7 tractable.

### 4.3 Residual path (7C.3)

```text
r[n]  = x[n] − pred[n]
ŝ[n]  = dead-zone scalar quantiser at step = 2^(gain/2), centroid reconstruction
x̂[n]  = pred[n] + Q⁻¹(ŝ[n])
```

Residual symbols are entropy coded with the **existing native residual family**
(`learned::residual_codec2`), searched over the declared subset. `FactorShift`,
general Golomb, `Bgmc`, `EmaRans`, `ZeroMaskRice` and `RunLengthRice` are all
already members; no parallel entropy subsystem is created.

The native rANS primitive in `entropy::rans` currently carries a
division-based `enc_put`. 7C.3 requires a **reciprocal rANS encode** path, proven
byte-identical to the division path. `PreparedWorld` is assessed separately in
§10 and is *not* assumed to be required.

### 4.4 Loss resilience: absolute parameters, bounded state

Because a packet may be lost and a later one may still arrive, the parameter
fields must be decodable without the lost one. The constitution therefore
requires:

1. **Parameters are absolute within the packet.** The first frame of every
   packet carries a complete model description.
2. **Only the reconstructed-sample history is stateful.** Loss corrupts that
   history; the contamination decays with the short-term filter memory (order ≤
   16 samples, < 1 ms) plus the pitch reference ring (≤ 288 samples, 18 ms).
3. **The decoder must never desynchronise a later packet.** A packet received
   after a gap decodes with the concealed history in the ring; it does not
   require any repair to be *correct* — only to be *accurate*.

Claim discipline: concealed audio is **never** described as exact
reconstruction, and no lossless claim is made for this profile anywhere.

### 4.5 State capsules and resynchronisation (7C.6)

The capsule carries only what a decoder cannot cheaply re-derive:

```text
bit 0  pitch lag + long-term gain
bit 1  residual step history (short run)
bit 2  low-order spectral state (first 4 reflection coefficients)
bit 3  comfort-noise state
```

Capsules are emitted:

* **periodically** at a fixed cadence (a run-length-coded count of frames);
* **triggers**: the encoder may also emit one at a mode change or after a long
  voiced run, priced by the byte oracle;
* **loss-triggered repair** and **late-packet repair** are *decoder* behaviours
  (§7): a capsule that arrives with a later packet repairs future state without
  rewriting audio that has already been played.

A capsule's `capsule_hash_lo` lets the decoder reject a capsule that does not
match the state it is being applied to, so a stale duplicate cannot corrupt the
service.

### 4.6 Selective redundancy (7C.7)

Redundancy is spent, not sprayed. The mask orders the fields by *future damage
per byte*:

```text
restart state  >  pitch/voicing  >  spectral state  >  coarse residual
                                                       >  fine residual
```

Redundancy overhead is measured **separately** from primary codec bitrate and
reported as such.

**Resolution (measured, §13.3 of the results):** backward redundancy is a net
negative for this profile and is defaulted **off**. The mechanism is retained
and priced rather than deleted, so a future change to the playout depth can
re-open the question on evidence.

### 4.7 VAD / DTX / comfort noise (7C.8)

```text
VAD       : deterministic energy + spectral-stationarity decision on the frame
DTX       : inactive frames send NO_DATA packets with a SID update at a
            bounded cadence, never stored comfort-noise PCM
comfort   : procedural noise shaped by the last transmitted spectral envelope
```

Reported **separately**: active-speech bitrate, inactive bitrate, whole-call
average bitrate. Long silence is never allowed to make the active codec look
better than it is.

---

## 5. Latency constitution (7C.12)

### 5.1 The accounting

Each contribution is reported in microseconds, never merged blindly:

```text
1  frame accumulation        (160 or 320 samples)
2  analysis / lookahead      (encoder window beyond the frame)
3  encode                    (the whole frame's search)
4  packetisation             (framing + entropy flush)
5  network                   (the simulated delay of §6; a test parameter)
6  jitter buffer             (the depth actually required, §7.3)
7  decode                    (frame decode)
8  playout scheduling        (deadline entry precision, §9)
```

### 5.2 The envelope

```text
encode budget              ≤ 5 ms
network test                = 10 ms   (one-way, ideal; a parameter, not a claim)
decode budget              ≤ 5 ms
```

That is a **20 ms processing + network envelope**. It is explicitly *not* a
total one-way latency, because frame accumulation has not been added yet.

### 5.3 One-way targets

```text
10 ms frames  :  ≤ 30 ms one-way before extra jitter-buffer delay
20 ms frames  :  ≤ 40 ms one-way before extra jitter-buffer delay
```

Any lookahead adds to these numbers and **must be disclosed**. A codec whose
lookahead is hidden has failed the constitution.

### 5.4 Reporting

```text
p50, p90, p95, p99, p99.9 where useful, maximum, deadline misses
```

The tail matters more than the mean. A codec that is fast on average and misses
1 % of deadlines in a call has failed 7C.

---

## 6. Deterministic impairment engine (7C.11)

No networking stack. Canonical VOLE voice packets are driven through a seeded,
purely deterministic model. Every result is replayable exactly from the seed.

| impairment | model | parameters |
| --- | --- | --- |
| random loss | i.i.d. Bernoulli per packet | 0, 1, 3, 5, 10 % (mandatory points) |
| burst loss | Gilbert–Elliott two-state chain | mean burst 1, 2, 4, 8 packets |
| duplication | i.i.d. Bernoulli | 0.5 %, 2 % |
| reordering | adjacent transposition | 0.5 %, 2 % |
| jitter | arrival offset distribution | 0, ±1, ±2.5, ±5, ±10 ms uniform; plus a two-mode bursty distribution |
| late delivery | arrival later than the jitter-buffer deadline | derived from the above, not a separate knob |
| clock drift | linear sample-rate offset | ±0, ±20, ±50, ±100 ppm |

The engine converts a jittered arrival time into exactly one decoder-visible
event: **played**, **concealed** (missing/late past deadline), or **discarded**
(duplicate/stale). The decision rule is part of the constitution:

```text
play_at(packet) = deadline   when arrival ≤ deadline
                  concealed  otherwise
deadline(packet) = expected_playout(packet) + jitter_buffer_depth
```

Fixed seeds are recorded in the receipt.

---

## 7. Decode-side recovery

### 7.1 Packet loss concealment (7C.5)

On a concealed frame the decoder synthesises a causal continuation from
decoder-visible state only:

| regime | mechanism |
| --- | --- |
| voiced | pitch-period repetition at the last good lag and long-term gain, through the last good short-term filter |
| unvoiced | shaped noise through the last good short-term filter |
| transition | cross-fade between the two by long-term-gain stability |
| silence/noise | continuation of the comfort-noise state |

Gain decays monotonically with concealment length and is bounded below.

### 7.2 Recovery

**Recovery time** is defined, frozen, as:

```text
the number of milliseconds after the last concealed sample until the decoded
signal re-enters 20 dB segmental SNR against the no-loss reference and stays
there for 20 ms
```

This is an arbitrary but *pre-declared* definition. It is not chosen after
seeing which definition flatters the codec. Both the definition and the
measured recovery time go in the receipt.

### 7.3 Jitter constitution (7C.13)

The court answers, per operating point:

> How much arrival-time variance can the decoder tolerate before audible
> degradation, underrun, or deadline failure?

using the ladders of §6 and reporting required jitter-buffer depth, added
latency, deadline misses, PLC activations, state-repair events and quality
degradation. A codec that only obtains quality behind a large jitter buffer has
not beaten a low-latency competitor, and the receipt says so.

---

## 8. Competitors (7C.14)

Competitors run as **external processes**, exactly as in 7B. They are never
imported, linked, made a dependency, used as a fallback, and no competitor
payload is ever copied into a VOLE object.

| role | tool | pinned operating points |
| --- | --- | --- |
| speech | Opus (`opusenc`/`opusdec`) | 6, 8, 12, 16, 24 kbps, `--vbr`, fullband and speech-targeted |
| voice | EVS (`EVS_cod`/`EVS_dec`) | 5.9, 7.2, 8.0, 9.6, 13.2, 16.4, 24.4 kbps, 16 kHz in/out |
| very low rate | Lyra | 3.2, 6.0, 9.2 kbps |

Each receipt pins the implementation, its version, its sha256, the frame size,
the requested bitrate, the **actual emitted** bitrate and the algorithmic delay.
An absent metric is `null` / `NOT_AVAILABLE`, never invented.

---

## 9. Runtime surface (7C.9, 7C.10)

7C.9 brings the declared **ALSA hardware-clock scheduling** remainder into
scope, because it directly serves a voice call: long-call drift is a real
failure mode.

```text
measure: capture/playout clock drift, scheduler wake error, buffer depth,
         underrun/overrun, deadline misses
reuse:   hybrid sleep/spin deadline entry, SCHED_FIFO entry, huge-page arena
```

The court must run identically with and without real-time scheduling; it never
assumes a policy the user did not grant.

7C.10 is **measure-first**: CPU frame-tile multicore, PartialBank/vector paths,
radix score assembly and PGO/BOLT are implemented only where the voice court
demonstrates a need, and the acceptance criterion is the voice-call latency
figure, not the presence of the mechanism.

---

## 10. Explicit non-goals and assessments

* `PreparedWorld` (declared remainder of an earlier campaign) is **assessed** in
  the receipt: if the voice hot path does not need a prepared evaluation world,
  that is recorded as a determination rather than silently skipped or
  mechanically implemented.
* No GPU path, no neural vocoder, no pretrained network, no competitor code,
  no networking stack, no new storage architecture.
* No claim that concealed audio is reconstruction. No lossless claim for this
  profile. No claim of perceptual transparency.
* Where VOLE loses a cell, the loss is attributed (§11) rather than dropped.

---

## 11. Evidence discipline and the competitive gate

Every subphase that changes behaviour: implement → test → run the court → record
actual physical output and runtime → preserve the old control → emit a receipt →
seal the increment. The existing release/seal policy is used unchanged; no second
evidence system is created.

The gate is the **measured Pareto surface**, not a plan:

```text
VOLE quality > Opus quality  at matched actual bitrate
VOLE quality > Lyra quality  at matched actual bitrate
VOLE beats the appropriate EVS operating points on the combined
             latency/loss/jitter Pareto surface
```

with `VOLE 8 kbps` targeted at the quality of the appropriate Opus speech
operating point around 12 kbps **while meeting the latency constitution**, and a
very-low-rate mode attacking Lyra-class quality with deterministic bounded
decode.

On a loss the response is attribution, not a new track:

```text
frame predictor · pitch model · residual coder · rANS encode speed ·
state overhead · packet overhead · PLC · resynchronisation · jitter buffering ·
scheduler timing · rate allocation · DTX
→ improve the responsible component → rerun `learned-voice-stream`
→ 7C.17.1, 7C.17.2, …
```

---

## 12. Measurement set (7C.16)

Per clean and per impaired operating point:

```text
actual bitrate
ViSQOL (speech mode) where the pinned binary and model run
POLQA only where legitimately available  (otherwise NOT_AVAILABLE)
STOI / ESTOI where useful
packet loss · jitter distribution
PLC duration · recovery duration (definition of §7.2)
one-way codec latency (the full accounting of §5.1)
encode p99 · decode p99 · deadline misses
```

Predicted-quality metrics are reported as predictions, not as listening-test
results. Blinded listening material may supplement the numbers once objective
results justify it; fabricated subjective results are forbidden.

---

## 13. Status

Frozen at the start of Phase 7C, before implementation.

### 13.1 Resolution

The constitution's mechanisms are implemented and sealed by the court
`learned-voice-stream`:

```text
7C.1  frame constitution   16 kHz mono, 160/320 sample frames, no whole-clip mode
7C.2  voice predictor      autocorrelation+Levinson / Burg / covariance-LSQ ×
                           orders 8..16 × reflection widths 5..8 bits, plus
                           long-term prediction on the reconstructed excitation
7C.3  residual path        dead-zone scalar quantiser + a compact voice residual
                           coder (zero-run/Rice/varint) that keeps the general
                           learned family reachable for large frames
7C.4  coalescing           all three shapes measured, none selected by default
7C.5  PLC                  excitation-ring repetition (voiced) / shaped noise
                           (unvoiced), monotone decay, fade to comfort noise
7C.6  capsules             periodic + transition-triggered, checksummed,
                           applied to the concealment model only
7C.7  redundancy           priced on/off; measured a net negative and defaulted
                           OFF (see 13.3)
7C.8  VAD/DTX/comfort      procedural comfort noise; active, inactive and
                           whole-call rates reported separately
7C.11 impairment engine    seeded, deterministic, no network stack
7C.12 latency accounting   all eight contributions kept separate
7C.13 jitter ladder        measured required depth, not an assumed buffer
7C.14 competitors          Opus, EVS, Lyra as external processes; EVS also driven
                           through its own G.192 erasure flag
7C.16 measurement set      SNR, ViSQOL, spectral, transient, PLC, recovery,
                           latency tails, deadline misses
```

### 13.2 Result: latency and robustness pass, clean quality loses

Measured on the three conversation-scale cases the court builds from the frozen
`effectiveness` corpus, at the frozen projection hash
`989e3c3b98a304da615e1880a5ec0a4fb285aae50b612f720359fd9300f7c331`.

**The latency constitution passes with margin.**

```text
320 sample frames    accumulation 20.0 ms · encode p99 1.46 ms · decode p99 6 µs
                     one-way p50 31.1 ms   (target ≤ 40 ms)
160 sample frames    accumulation 10.0 ms · encode p99 0.90 ms · decode p99 4 µs
                     one-way p50 20.7 ms   (target ≤ 30 ms)
encode deadline misses 0        decode deadline misses 0
```

The encode budget is 5 ms and the worst measured frame is under 2 ms, which is
why 7C.10's hot-path items are *not* implemented: the court demonstrates no need,
and 7C.10 forbids implementing them to tick a box.

**Loss and jitter behave.**

```text
random loss 0/1/3/5/10 %   0 · 5 · 16 · 23 · 40 concealed frames of 400
                           30.3 · 11.9 · 11.6 · 10.0 dB against the no-loss run
                           recovery 10 ms at 1 %; not reached at ≥ 3 %
burst 2/4/8 at 3 %         longest concealment run 4 · 9 · 23 frames
jitter 0…±10 ms            zero concealment at every rung once the *measured*
                           depth (1.7 → 10.6 ms) is provided
drift ±20/±100 ppm         no concealment — not detectable over 4 s, disclosed
mixed all-on               9 concealed, 14.5 dB, depth 5.5 ms
```

EVS, driven through its own erasure flag at 13.2 kbps over the same ladder,
degrades from 12.8 dB to 5.5 dB; VOLE on 20 ms frames goes from clean to 10.0 dB
at 10 % loss but at roughly 1.6× the rate. Per-decibel robustness is not a
victory, and the receipt does not claim one.

**Clean quality loses, and the loss is attributed.**

```text
                       mean ΔSNR vs VOLE at matched actual bitrate
Opus   0 of 3 cells    −8.36 dB
EVS    0 of 3 cells    −5.75 dB
Lyra   0 cells         VOLE has no operating point inside 3.2–9.2 kbps
```

Only three matched cells exist per competitor because VOLE's *achievable* range
(below) barely overlaps theirs. The cause is not the predictor, the entropy
coder, the concealment or the perceptual model: it is that **the model
description is transmitted as scalar reflection codes, and it alone costs more
than the entire frame allowance at the rates a voice call wants.**

```text
                   target   achievable
160 sample frames  6 kbps   ≈ 31 kbps
320 sample frames  6 kbps   ≈ 17.7 kbps
                   8 kbps   ≈ 20.5 kbps
```

The court reports the target and the achieved rate side by side for exactly this
reason, and the encoder's overshoot policy is bounded and visible rather than
hidden behind silence.

### 13.3 A measured rejection: selective redundancy

The constitution required redundancy to be *priced*, not assumed. It is:

```text
320 sample frames, 8 kbps target
redundancy off    20 506 bps   clean 8.88 dB   at 3 % loss 10.42 dB
redundancy on     21 392 bps   clean 3.25 dB   at 3 % loss  2.04 dB
```

Backward redundancy spends frame bytes the residual needs, and it can only
repair a frame a jitter buffer has not yet released — a window this codec's
measured depth does not open. It is therefore defaulted **off**, retained and
priced, and re-openable on evidence. This is the win / near-negative / reject
discipline applied honestly rather than a mechanism kept because it was planned.

### 13.4 Declared remainder of 7C

These are 7C's own items, not a new track, and each is recorded with the
measurement that sets its priority:

1. **Vector-quantise the spectral parameters (7B.4 pulled forward).** This is the
   single cause of §13.2's loss. Scalar reflection codes set a ~17.7 kbps floor
   on 20 ms frames before any residual symbol exists. This is the first thing to
   build.
2. **7C.9 ALSA hardware-clock scheduling.** Not implemented. The court measures
   tolerance and drift *within* the impairment engine; the hardware-clock
   remainder would extend the same measurements to a real device clock.
3. **7C.3 reciprocal rANS encode.** Not implemented, and *assessed as not on this
   path*: the voice residual uses the compact Rice codecs, and the measured
   encode p99 (1.46 ms against a 5 ms budget, zero misses) shows no need for the
   division-free path. Recorded as a determination, not an omission.
4. **`PreparedWorld`.** Assessed irrelevant to the voice hot path for the same
   reason; the encoder's per-frame work is a bounded closed-loop search that
   mutates one small state, and it is not re-preparing a world per frame.
5. **Coefficient-domain inter-frame prediction inside a packet.** The reserved
   room to cut the floor without full VQ.

### 13.5 Non-claims

* The profile is lossy. `decode(encode(x)) == x` is **not** claimed, and the
  court reports no exactness metric for it.
* Concealed audio is **never** reconstruction. Every concealment cell is reported
  as a degradation against the no-loss run, never as a recovery of the source.
* No perceptual-transparency claim is made. ViSQOL values here are objective
  predictions, not listening-test results.
* No networking, RTP, ICE/STUN/TURN, peer discovery or real transport is
  implemented, and the impairment engine is a model rather than a claim about
  any real channel.
* Opus and Lyra expose no packet-loss simulation through their pinned CLIs, so
  their loss behaviour is `NOT_AVAILABLE` rather than guessed.

### 13.6 Historical note

Resolution and any attributed losing cells are recorded in the 7C receipts and
summarised in `docs/OPTIMIZATION.md`.
