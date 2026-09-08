# DIRECTNESS — D0, D1, D2, D3

What "direct" means in VOLE-Audio, how it is measured, and how it is *not*
claimed.

## Path classes

```rust
enum Directness {
    D0Buffered,        // host PCM staging and/or device->host copy; diagnostic
    D1EndpointMapped,  // GPU writes final codes into the endpoint's mapped region
    D2PeerDevice,      // peer-DMA/export-import materialization
    D3EndpointNative,  // endpoint-native evaluation — FUTURE CONCEPTUAL ONLY
}
```

- **D0** is the conventional, fully supported diagnostic. It isolates
  evaluator throughput, transfer cost, and endpoint cost. Every D0 receipt is
  labeled `D0` / `GpuBufferedDiagnostic`.
- **D1** is the first falsification court: take the *actual* ALSA mmap region,
  register *that exact existing memory* (CUDA `cuMemHostRegister` /
  HIP `hipHostRegister` equivalents), obtain a device-visible pointer, have the
  GPU write final endpoint codes there, synchronize, commit to the endpoint.
  If registration fails, the exact error is recorded and classified; the run
  continues on D0 **and says so** (`FELL_BACK_TO_D0`). Allocating a fresh
  pinned CUDA buffer and calling it "the ALSA region" answers a different
  question and is forbidden.
- **D2** requires peer-capable memory/export/import mechanisms *and* fence /
  ordering / visibility / endpoint-consumption proof. A shared address without
  those is not D2. If no qualifying endpoint exists, the result is
  `UNSUPPORTED_BY_HARDWARE` / `UNSUPPORTED_BY_TOPOLOGY`.
- **D3** exists in the vocabulary and spec only. It always returns
  `NOT_IMPLEMENTED` (future conceptual: FPGA/DSP/ASIC endpoint evaluators).
  See `docs/FUTURE_ENDPOINT_EVALUATORS.md`.

## Topology is a separate axis

```rust
enum Topology { Uma, HostMapped, PciePeer, DeviceBar, CustomEndpoint, NetworkEndpoint, Unknown }
```

An mmap'd ALSA region does **not** imply D1 (registration may fail; pages may
be I/O memory; coherency may be unprovable). A PCIe link does **not** imply
D2. Receipts always carry both `directness` and `topology`, independently
measured.

## Why mapped host memory is not the default home of state

Host memory mapped into the GPU address space crosses the CPU–GPU
interconnect: higher latency, lower bandwidth than device memory. VOLE-Audio
therefore keeps SampleObjects, voice state, generator tables, residual
indexes, filter state, and checkpoint state resident in VRAM/GPU-local
memory, and uses host-mapped/endpoint-mapped memory only where it materially
helps — compact control traffic (if measured useful) and **final endpoint
observation writes** — plus explicit evidence/control words.

## ALSA discipline (first physical endpoint)

The first endpoint court uses the Linux `hw:` PCM with direct mmap access —
not `snd_pcm_mmap_writei/n`, not plug conversions, not PipeWire/JACK/PulseAudio
(D1 measurement). The loop records: device identity, access mode, format,
rate, channels, period/buffer size, channel area base/step, offsets,
contiguous frame counts, hw pointer/avail/delay, timestamps, and xrun state.
The endpoint format preference is a hardware-native format (S32_LE first,
then S16_LE) — no plug-mediated "direct" claims.

## Evidence requirements per path

For every D1/D2 attempt a receipt must record: exact base/length/alignment of
the registered range; registration result (exact API error on failure);
pointer attributes; device pointer; synchronization mechanism and fence/order
evidence; coherency declarations; commit sequence; underrun evidence;
unregister/shutdown ordering; and the verdict.

## Phase H measured outcome (CUDA D1, `court d1`)

`court d1` runs the falsification directly: for each candidate endpoint it
opens the frozen `hw:` shape (MMAP_INTERLEAVED + S32_LE + 48 kHz + stereo,
512-frame period / 1024-frame buffer), registers the **exact mapped ring**
with `cuMemHostRegister(DEVICEMAP)` + `cuMemHostGetDevicePointer`, and — on
the first successful registration — runs a paced session where every
contiguous mmap chunk's final codes are written by the kernel into the
registered region, the stream is synchronized, the codes are shadow-verified
**in place** against the scalar oracle, and only then committed. A D0-mmap
baseline on the same endpoint shape measures the bytes D1 removes. Default
content is silence-safe; `--emit-audio` opts into audible content.

Measured on the seal machine (RTX 4080 SUPER, driver 610.57.04, on-board
`ALC897` analog via `snd_hda_intel`):

```text
D0-mmap baseline (24 000 frames):  gpu->host 192 000 B | host copy 192 000 B | endpoint obs 192 000 B
D1-direct        (48 000 frames):  gpu->host       0 B | host copy       0 B | endpoint obs 384 000 B
```

D1 verdict **SUPPORTED** (`D1_ENDPOINT_MAPPED` / `HOST_MAPPED`): the GPU
wrote 94 chunks byte-exact vs the scalar oracle with zero xruns and a clean
drain; endpoint depth stayed 512–1024 frames. Registration of the actual HDA
DMA ring succeeded — an important data point, and still one that must be
re-measured per device: the receipt records every candidate's own row
(open/mmap/format/registration), and failures are classified exactly
(`UNSUPPORTED_BY_*` / busy `INCONCLUSIVE`), never replaced by a substitute
pinned buffer.

Interpretation discipline: this proves the memory path on one
hardware/driver combination, not a universal property. A device whose ring
the CUDA driver cannot register, or that lacks hw:mmap, produces an explicit
negative row — which is exactly the first-class evidence the paper demands.
