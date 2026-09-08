# PHASE H.2 — ENTROPY-NATIVE AUDIO CORE

> Phase charter and seal ledger. H.2 is inserted **between** the sealed
> Phase H (CUDA D1) and Phase I (ROCm), without renumbering I–N. It exists to
> correct an architectural omission in the original implementation sequence:
> VOLE-Audio must not become merely `procedural state -> PCM -> endpoint`; its
> deeper architecture is
>
> ```
> deterministic explanation
>   + entropy/configuration state
>   + entropy-coded irreducible residual
>     -> bounded observation
>       -> endpoint sample codes
> ```
>
> Companion architecture (must be studied before touching code): the VOLE
> Video native rANS floor / DSFB search courts / feature-gated EntropyFS store
> adapter (`infinityabundance/vole`), `infinityabundance/entropyfs`,
> `infinityabundance/dsfb`; paper DOI 10.5281/zenodo.22649073 (v1.1 revision
> makes the entropy layer explicit; this document does not contradict v1.0).

## The defining principle

> **Store the deterministic explanation. Entropy-code what the explanation
> cannot reproduce. Materialize sample-domain observations only when actually
> required.**

PCM remains a legitimate observation surface and a literal representation is
always the mandatory universal fallback; it is simply **not** universally
privileged as authoritative durable state.

## Phase position

A–H exist and remain intact. H.2 is inserted between H and I. Phases are not
renumbered:

| Phase | Name |
| --- | --- |
| I | ROCm |
| J | ROCm D1 |
| K | inverse compiler (storage-cost oracle = H.2) |
| L | GPU inverse search |
| M | production corpus / depth courts |
| N | transport/archive (embeds H.2 canonical records) |
| O | learned deterministic prediction addendum (judged by H.2 cost API) |

Constraints honored across the phase: exactly **one Cargo package** (no
workspace, no sub-crate); existing frozen reference hashes unchanged;
existing semantic/SIMD/CUDA/D1 receipts remain valid; entropy is
*representation*, never semantic authority; EntropyFS and DSFB stay optional
and never enter the decoder.

## Entry state (H.2.0 freeze — captured before H.2 code)

Captured on a clean tree at the Phase-H entry-gate seal:

- Commit: `39a349b5d1ee9cadcc8d2ab1865224a486f42958`
  ("Receipts: Phase-H entry-gate seal — all six courts SUPPORTED on clean
  tree 0b65bfdd (git_dirty: false)").
- Source tree (git): `b923b4e8020ff8937e8b6b01fe08d8079828f253`.
- Prior implementation commit: `0b65bfdd` (entry-gate fixes: xrun-terminate
  policy, base-equality proof, evidence promotion).
- Test count: 200 host tests green (debug + release) at entry-gate seal.
- CUDA PTX artifact (scripts/out/vole_audio.ptx, release-equivalent):
  SHA-256 `c33592be73edeb1601d3d44fcc9019f3d965347f7d7f66724bd7faf09b5f8b14`;
  rustc `1.99.0-nightly`, LLVM 22.1.8, `sm_70` baseline PTX, entry
  `vole_render_d0`.
- Existing courts and receipts (all `git_dirty: false` seals):
  `semantic`, `authored`, `simd`, `facts`, `cuda`, `d1`.
- Existing semantic hashes (frozen; must not change):
  semantic court `1791816f…`, authored court `f7e103f3…` (current), facts
  F01–F15 stable ids, CUDA/D1 scalar == SIMD == CUDA equality anchors.

## H.2 normative documents (one owner per decision)

| Document | Owns |
| --- | --- |
| `docs/ENTROPY_NATIVE.md` | the corrected architecture; representation contract; sample-domain exposure surfaces |
| `docs/RANS.md` | the frozen native rANS codec (parameters, state machine, byte layout, model normalization) |
| `docs/ENTROPY_ACCOUNTING.md` | complete-cost rules; declared/unique/physical; receipts fields; performance methodology |
| `docs/ENTROPYFS.md` | optional persistence adapter boundaries; EntropyFS claim boundary |
| `docs/DSFB_SEARCH.md` | zero-authority search governance; strategy court semantics; claim boundary |
| this file | phase charter, position, order of work, seal ledger |

## Order of work (executed in sequence, no skipping)

1. **H.2.0** — freeze entry state (above); write the six docs. (This file.)
2. **H.2.1** — representation contract: semantic `SampleObject` vs physical
   canonical representation; entropy orthogonal to hypothesis family.
3. **H.2.2** — native deterministic rANS core (`src/entropy/…`): 32-bit state,
   `scale_bits = 14`, `MODEL_TOTAL = 16384`, `STATE_L = 2^23`, deterministic
   largest-remainder model normalization, canonical self-describing blocks,
   mandatory RAW fallback, hostile-input typed errors, no heap bombs; checked
   integer arithmetic; scalar is semantic authority; independent oracle
   parity (ryg-rans-rs, dev-only) where model/layout semantics match.
4. **H.2.3** — canonical model normalization frozen (zero handling, minimum
   count, total, scaling, tie-breaks, overflow behavior). Model bytes counted.
5. **H.2.4** — literal entropy floor: `LiteralRaw` + `LiteralRans`
   (reconstructs canonical U1 sample codes exactly), symbolization
   comparison (canonical i32 LE bytes, byte-position-separated lanes,
   modular first-difference, ZigZag signed mapping, per-channel streams).
6. **H.2.5** — RAW fallback: rANS chosen only when complete bytes win;
   uniform/incompressible data converges to RAW.
7. **H.2.6** — block-addressable entropy pages, independently decodable;
   page index; random access bounded.
8. **H.2.7** — page-size Pareto court (64…4096).
9. **H.2.8** — shared entropy models: standalone vs marginal vs shared cost;
   content-addressable model identity.
10. **H.2.9** — entropy-code the *existing* exact residual (Phase-E closure
    unchanged); `R -> Psi(R) -> rANS_M`; reconstruction returns the identical
    semantic residual.
11. **H.2.10** — residual symbolization (mask / runs / magnitudes), simple
    canonical baseline, complex candidates must defeat it.
12. **H.2.11** — complete representation cost `L_complete`.
13. **H.2.12** — conventional baselines (RAW, canonical U1 literal, WAV,
    native rANS literal; FLAC via pinned external executable when present,
    else NOT_AVAILABLE).
14. **H.2.13** — partial entropy materialization API; partial == full slice.
15. **H.2.14** — reverse / loop / non-unit rate / random seek interaction.
16. **H.2.15** — sample-domain exposure instrumentation (additive counters).
17. **H.2.16** — SIMD decode (exact; parallel surfaces = independent
    pages/channels/states; measured, not fabricated).
18. **H.2.17** — CUDA entropy decoder (`src/device/entropy_shared.rs`),
    parallel over pages/channels/states; scalar == CUDA symbols.
19. **H.2.18** — fused CUDA entropy + procedural + exact closure; no
    full-object global waveform; bounded page scratch; receipt all global
    sample intermediates.
20. **H.2.19** — flagship: entropy → CUDA → **D1 registered ALSA endpoint**;
    scalar == entropy-scalar == entropy-CUDA == entropy-CUDA-D1 byte-for-byte.
21. **H.2.20** — D0 entropy baseline, equal-work comparison.
22. **H.2.21–25** — ObjectStore abstraction; EmbeddedStore; `entropyfs`
    optional feature adapter; object types; declared/unique/physical.
23. **H.2.26–29** — DSFB zero-authority adapter; H.2 role; strategies over
    one candidate universe; success criterion.
24. **H.2.30–31** — frozen entropy corpus + negative controls.
25. **H.2.32** — courts: `entropy-rans`, `entropy-literal`, `entropy-residual`,
    `entropy-pages`, `entropy-partial`, `entropy-simd`, `entropy-cuda`,
    `entropy-d1`, `entropyfs`, `dsfb-entropy`, `h2` (aggregate).
26. **H.2.33** — hostile-input court (typed errors; never panic/UB/blowup).
27. **H.2.34** — bounded decode complexity hard limits.
28. **H.2.35** — evidence receipts (additive fields; Markdown from receipts).
29. **H.2.36–40** — performance methodology; GPU/memory-path/D1/EntropyFS/
    DSFB claim boundaries.
30. **H.2.41–44** — Phase O / K / N integration preparation (cost API).
31. **H.2.45–46** — documentation + README central language.
32. **H.2.47** — test matrix + fuzz targets.
33. **H.2.48** — unsafe policy.
34. **H.2.49** — build/release matrix.
35. **H.2.50** — receipt sealing (clean tree; immutable receipts; separate
    receipts commit; negative receipts forever).
36. **H.2.51–55** — exit criteria (all 43 conditions); forbidden shortcuts;
    flagship demonstration; research standard; final architectural invariant.

## Seal ledger

Seals are appended in order as they are produced. Every seal is a clean-tree
run with immutable receipts committed separately; negative receipts are kept.

<!-- ledger entries are appended chronologically below this line -->
