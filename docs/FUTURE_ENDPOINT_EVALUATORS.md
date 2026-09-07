# Future endpoint evaluators (FPGA/DSP/ASIC) — conceptual preservation only

This document exists solely to preserve the already-disclosed future
architecture (paper §"Future conceptual exploration"). It is **not** an
implementation plan for this repository.

## Status

- **D3 (`Directness::D3EndpointNative`) is future conceptual work.** In code
  it exists only in the vocabulary (`src/audio/directness.rs`) and always
  returns `NOT_IMPLEMENTED`.
- This repository will **not** contain HDL, FPGA toolchain scaffolding, DSP
  device code, or any pretending that D3 exists.
- ROCm/HIP and CUDA remain the current execution targets; a copied GPU→host
  PCM path is diagnostic only (D0).

## What is preserved for future evaluation

- Endpoint-native evaluation would move the deterministic evaluator itself
  into the endpoint device (FPGA/DSP/ASIC), so that sample codes are produced
  by endpoint logic rather than written across an interconnect.
- The architecture keeps the *semantic core* (universe u1 exact arithmetic,
  SampleObject model, residual closure, checkpoint semantics) independent of
  execution surface precisely so an endpoint-native port would reuse the same
  frozen semantics rather than re-deriving them.
- Any future endpoint work must satisfy the same evidence rules: exact
  differential parity with the scalar oracle, directness proven by memory-path
  evidence, receipts, and honest negative results.
- Candidate future mechanism classes to evaluate (not implement): device-BAR
  resident observation state, endpoint DMA descriptors written by the GPU,
  peer-to-peer export/import with explicit fences, and endpoint-internal
  evaluation of frozen procedural kernels.

No D3 work is scheduled in the current implementation phases (A–N).
