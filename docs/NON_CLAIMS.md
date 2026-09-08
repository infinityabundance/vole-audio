# Non-claims

Statements this project explicitly refuses to make without evidence. When a
non-claim looks like a headline, the repository's answer is a receipt, not an
assertion.

1. **PCM may never exist.** False by design: literal representation is a
   mandatory universal fallback, and physical endpoints require bounded
   elasticity (DMA regions, FIFOs, registers). Zero physical sample storage is
   not claimed; the *minimum transient observation state* is what is measured.
2. **Everything must compress procedurally.** No: noise, encrypted/random-like
   content, and high-entropy material legitimately stay literal or
   residual-heavy. If literal wins the Pareto frontier, that is the result.
3. **A seed contains arbitrary information.** No: seeds/parameters only carry
   what their model and residual provably carry. No magical compression
   claims.
4. **GPU audio is new.** Not claimed. GPU audio synthesis/DSP has decades of
   prior art (see `PRIOR_ART.md`).
5. **Procedural audio is new.** Not claimed. MUSIC-N/Csound and MPEG-4
   Structured Audio predate this work by decades.
6. **Zero physical buffering is possible.** Not claimed. Endpoints need
   bounded elasticity; the architecture measures how much transient state is
   actually required.
7. **GPU always beats CPU.** Not claimed. The courts measure the crossover;
   at low polyphony/small quanta the optimized CPU may win.
8. **D1/D2 must work on commodity hardware.** Not claimed. Registration may
   fail; topology may forbid; results are recorded as
   `UNSUPPORTED_BY_API` / `_HARDWARE` / `_TOPOLOGY` / `INCONCLUSIVE`.
9. **Mean latency is real-time.** No: real time requires deadline and tail
   evidence (p50/p90/p99/p99.9 with sufficient N, xruns, missed deadlines).
10. **Sound output implies exactness.** No: exactness is byte/hash equality
    of the sample domain under the u1 semantics, differentially tested across
    backends.
11. **"Compression ratio" from model size.** No: complete-dependency
    accounting (residuals, tables, dictionaries, checkpoints, indexes,
    metadata, integrity, universe/decoder dependencies, amortization) is
    required before any size claim.
12. **Energy figures from TDP.** No: energy is only reported from real
    measurement sources with method and uncertainty.
13. **FPGA/DSP/ASIC endpoint evaluation is implemented.** Not implemented and
    not scaffolded; D3 is future conceptual work only
    (`FUTURE_ENDPOINT_EVALUATORS.md`).
14. **Performance expectations.** No expected-number claims appear in
    `PERFORMANCE.md`; it says NOT YET MEASURED until receipts exist.
15. **Phase G CUDA results imply direct endpoint materialization.** No. Phase
    G is the **D0 buffered diagnostic** only (`GpuBufferedDiagnostic`): the
    GPU renders a final VRAM observation block that is copied back to host
    PCM. No ALSA endpoint region is written by the GPU, no `cuMemHostRegister`
    of an endpoint mapping exists, and no D1/D2 path exists until Phase H —
    receipts carry the D0 label and the sample-traffic counters.
16. **The DAC consumes compressed audio / the endpoint understands rANS.** No.
    The H.2 D1 path decodes entropy pages into exact final sample codes on
    the GPU and writes those codes into the endpoint region; the endpoint
    always receives plain S32_LE codes. rANS never leaves the representation
    layer.
17. **rANS is a procedural generator.** No (ADR 0001): entropy payloads are
    representations of exact semantic content; a short coded payload never
    contains more than its model + residual provably carry.
18. **EntropyFS or DSFB are required for playback.** No (ADRs 0002/0003): both
    are default-off optional features — persistence and encoder-side search
    governance only; they never enter the decoder or the playback path.
19. **Shared models / deduplication are free.** No: storage accounting always
    reports declared, unique, and physical bytes separately; shared
    dependencies are never reported as zero (ENTROPY_ACCOUNTING.md).
20. **GPU entropy decode is faster than CPU.** Not claimed without a court:
    the single-thread serial rANS decode is latency-chain bound (named wall
    regimes on the seal GPU: ~20 ms/window idle-first-launch, ~2–3 ms
    court-warmup, ~0.4–0.5 ms aggregate-hot — see PERFORMANCE.md); the
    courts report
    whichever surface wins (page counts, clocks, workload) and the D1 court
    records that its value is directness/traffic, not latency.
21. **D1 entropy results are a universal zero-copy claim.** No: the H.2 D1
    receipt is per-device/per-driver; transient page-local sample state,
    endpoint ring samples, and FIFO/DMA state always exist and are counted
    under their own surfaces.
22. **Phase I (ROCm) executes kernels / the amdgcn artifact is loadable.**
    No: this host has no AMD GPU, no KFD, and no ROCm userspace. The Phase I
    `amdgcn` code object (`scripts/out/vole_audio.amdgcn.elf`) is compile
    evidence only; it is per-ISA (baseline gfx906) and its loadability on a
    real device is Phase J evidence, never assumed. `court rocm` never
    reports device `SUPPORTED`; the scalar == ROCm differential battery is
    Phase J on ROCm hardware.
23. **Missing ROCm userspace is a hardware deficiency.** No: when the AMD
    GPU and KFD are present but the HIP/HSA compute runtime is not loadable,
    the probe classifies `UNSUPPORTED_BY_API` (runtime/library level), never
    `UNSUPPORTED_BY_HARDWARE`. `librocm_smi64` is telemetry, not evidence of
    a compute runtime.
24. **A receipt can attest a tree its binary was not built from.** No:
    receipts record both compiled-from (build.rs stamp) and
    executed-in-worktree (runtime capture); a seal requires them to match
    and both to be clean (`Environment::source_bound`). GPU artifacts are
    bound back to their source tree through their provenance sidecars.

Anything in this list that later gains evidence moves into a claims document
with its receipt. Until then: **not claimed.**
