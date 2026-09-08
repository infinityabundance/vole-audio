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

Anything in this list that later gains evidence moves into a claims document
with its receipt. Until then: **not claimed.**
