# Voice LSF split multi-stage vector quantiser (`src/voice/vq.rs`)

Normative note for the voice codec's line-spectral-frequency (LSF) split
multi-stage vector quantiser (MSVQ) and its frozen codebook asset
`assets/voice/lsf_msvq_v1.bin`. The implementation is `src/voice/vq.rs`; the
reproducible trainer is `examples/voice_train.rs`.

## 1. Domain

The quantised quantity is the voice frame's short-term spectrum expressed as
order-16 **line spectral frequencies**: radians in `(0, pi)`, strictly
increasing. The codec's existing LSF representation layer
(`crate::voice::lsf`) is authoritative for the forward/inverse conversions and
for stability; the VQ only interpolates inside that domain.

Reconstruction always ends in `lsf::stabilize_lsf`, so every decoded vector is
finite, inside `[MIN_LSF, PI - MIN_LSF]`, and separated by at least
`MIN_SEPARATION` radians.

## 2. Codebook structure

* A single global **mean** LSF vector, plus two independent **splits**
  (`LSF[0..8]` and `LSF[8..16]`).
* Each split has two **stages** with sizes `[256, 64]`: 256 stage-0 centroids
  and 64 stage-1 centroids of width 8.
* Index bits are `8 + 6` per split = **28 bits**, packed little-endian into
  exactly 4 wire bytes (`VQ_INDEX_BYTES`); the top 4 bits are always zero.
  * split 0 stage 0: bits `0..8`
  * split 0 stage 1: bits `8..14`
  * split 1 stage 0: bits `14..22`
  * split 1 stage 1: bits `22..28`
* Reconstruction per split:
  `lsf_hat = mean + stage0[idx0] + stage1[idx1]`, then
  `stabilize_lsf(&mut lsf_hat)` over the concatenated order-16 vector.
* Encoding is a **joint** search over `(idx0, idx1)` per split (256 x 64 =
  16384 candidates) minimising squared error in the LSF domain against the
  mean-removed target; ties go to the lowest index, so the search is
  deterministic.

### Quantisation convention

The mean and every centroid are stored as `i16` with `Q = 32767 / pi`
(`q = round(x * 32767 / pi)`, `x = q * pi / 32767`). **All encoding and
decoding uses the quantised `i16` values** — the encoder searches against the
dequantised table and the decoder reconstructs from it — so the two sides agree
bit-for-bit and the table is self-consistent by construction.

### Internal reflection codes

`decode_index` hands the decoded spectrum to the synthesiser as reflection
codes: `decode_lsf -> lsf::lsf_to_weights -> predict::predictor_to_reflections
-> predict::quantise_k(k, VQ_INTERNAL_WIDTH)` with `VQ_INTERNAL_WIDTH = 14`.
`lsf_to_weights` returns the synthesis weights `w = -A` (the negation of the AR
polynomial), while `predictor_to_reflections` inverts the AR step-up and
therefore returns `-k`; the recovered reflection vector is negated back to the
lossy-engine convention that `reflection_to_weights` / `weights_of` use. If
`predictor_to_reflections` returns `None` (only reachable for pathological
out-of-distribution centroid combinations), the decoder falls back
deterministically to an all-zero reflection vector.

## 3. Asset format (little-endian)

```
magic        : b"VOLEVQ01"                (8 bytes)
u16          order                        (= 16)
u16          splits                       (= 2)
u8           stages                       (= 2)
u8           reserved                     (= 0)
u32[4]       stage_sizes                  (= 256, 64, 256, 64)
i16[16]      mean                         (Q = 32767/pi)
i16[256*8]   split0 stage0
i16[64*8]    split0 stage1
i16[256*8]   split1 stage0
i16[64*8]    split1 stage1
u8[32]       sha256 of every preceding byte (raw hash bytes, not hex)
```

Total size 10334 bytes (10302-byte hashed payload + 32-byte SHA-256). The asset
is loaded with `include_bytes!` and parsed/verified exactly once in a
`OnceLock`; a mismatch panics (the bytes are compiled in, so a failure is a
build defect, not runtime input). `vq::codebook_sha256()` returns the **hex**
SHA-256 of the payload (everything before the trailing hash field).

## 4. Training procedure (`examples/voice_train.rs`)

```
cargo run --release --example voice_train -- \
    --root target/real-corpus/LibriSpeech/dev-clean \
    --out assets/voice/lsf_msvq_v1.bin \
    --files 400 [--report <path>]
```

Fully deterministic; no randomness beyond a fixed-seed xorshift in the test
harness. Steps:

1. Walk the speaker directories under `--root`, **skip the twelve
   `effectiveness` speakers as whole directories** (see §5), sort the remaining
   `.flac` paths, and take the first `--files` (default 400).
2. Decode each file to 16 kHz mono PCM with the `flac` CLI
   (`flac -d -c -s`), parsing the stdout WAV.
3. Extract order-16 LSF vectors with the codec's analysis geometry: for frame
   starts `t = 0, 160, 320, ...`, `hi = t + 320`,
   `lo = t.saturating_sub(304)`, `block = source[lo..hi]`. For each block run
   the three estimators the codec competes
   (`autocorrelation`+`levinson_reflections`, `burg_reflections`, and
   `lsq_coefficients` -> `predictor_to_reflections`), negate each reflection
   vector (the codec's convention), and convert with `reflections_to_lsf`;
   frames whose conversion fails are skipped. The vector count is capped at
   300000 (deterministic early stop).
4. Subtract the global mean per split. Train each split's stage-0 (size 256)
   and stage-1 (size 64) codebooks with LBG: start from the mean, repeatedly
   split each centroid by `+/- eps` (`eps = 0.01` rad), then run 25 Lloyd
   iterations (assign by squared error with lowest-index tie-breaking; an empty
   cell keeps its previous centroid). Stage 1 is trained on the stage-0
   residuals.
5. Quantise the mean and centroids to `i16` (`Q = 32767/pi`), then refine in
   the **quantised** domain for 5 iterations: assign against the dequantised
   quantised table and re-quantise each cell's mean, so the frozen table is
   self-consistent.
6. Write the asset plus its trailing SHA-256; print the training-vector count,
   per-split stage sizes, the final mean squared LSF error on the training
   vectors, and the payload SHA-256. With `--report`, the same facts are
   written to a text file.

Observed result for the committed asset (`--files 400`: 300000 vectors):
training MSE `4.733e-4 rad^2`, MAE `1.690e-2 rad`, payload SHA-256
`02fee7309cb16b9bd7e5ee95b943c358f9b6e31f37653a18ca6b6d8eec9f5139`.

## 5. Disjointness rule (mandatory)

The frozen `effectiveness` evaluation corpus for this codec is the 12 dev-clean
utterances whose speakers are:

```
1272 1462 1673 174 1919 1988 1993 2035 2078 2086 2277 2412
```

Training excludes these **whole speaker directories**, not merely those
utterances, so no evaluation speaker contributes any frame to the codebook.
Training files are taken from the remaining speakers in sorted path order. The
trainer implements this rule directly (`EXCLUDED` in `examples/voice_train.rs`);
it is not a convention that can be waived by a flag.

## 6. Accounting statement: the codebook is a shared decoder table

The 10334-byte codebook is a **fixed, shared decoder resource**, not per-stream
side information. It is compiled into every decoder (and encoder) as a constant
and is therefore:

* charged **once** to the decoder's static footprint, not to any packet or
  utterance;
* **never transmitted**, so it consumes zero bytes of the voice bitstream;
* identical for encoder and decoder by construction (both read the same frozen
  asset), which is what makes the 28-bit wire index fully self-describing.

The per-frame wire cost of this quantiser is exactly `VQ_INDEX_BYTES = 4` bytes
for the entire order-16 short-term spectrum (28 information bits). The frozen
asset's identity is `codebook_sha256()`; any change to the asset is a
codec-version-visible change to the shared table.
