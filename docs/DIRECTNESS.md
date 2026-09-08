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
`ALC897` analog via `snd_hda_intel`). Both paths run the SAME 48 000-frame
window; materialization traffic and verification reads are separate named
surfaces. The D0 baseline uses the stronger `render_into` form (the DtoH
transfer lands directly in one host buffer — no internal staging copy — then
one buffer→region copy):

```text
D0-mmap baseline (48 000 frames):  dtoh 384 000 B | host copy 384 000 B | endpoint obs 384 000 B
D1-direct        (48 000 frames):  dtoh       0 B | host copy       0 B | endpoint obs 384 000 B
```

Verification reads (comparing already-written samples against the oracle) are
counted separately and are 0-shadow-copy: the D1 verifier compares the
mapped region in place, never building a shadow sample buffer. Measured
per-chunk wall across seals: D1 mean 0.20–0.55 ms vs D0 mean 0.14–0.19 ms
(run-to-run variance; the gap shrinks when the D1 path is re-measured warm) —
D1 eliminates intermediate sample movement but mapped-host GPU stores do not
beat VRAM render + DtoH here; this court is therefore a
directness/residency/traffic result, not a latency optimization. Phase M
owns the workload crossover question.

D1 verdict **SUPPORTED** (`D1_ENDPOINT_MAPPED` / `HOST_MAPPED`): the GPU
wrote 94 chunks byte-exact vs the scalar oracle with zero xruns, an exact
`snd_pcm_mmap_commit` transfer check on every chunk, and a clean drain;
endpoint depth stayed 512–1024 frames. `cuPointerGetAttribute` returned rc 1
(invalid argument) for every attribute and both address targets on this
driver — recorded per query; registration + device pointer + the working
direct kernel writes are the evidence, no pointer-attribute value is claimed.
The endpoint must grant the exact 48 kHz rate and the full per-channel area
geometry is validated at open (the sealed ALC897 ring: `ch0 first=0 step=64;
ch1 first=32 step=64`).

Every candidate device gets its own trial row: after the D1 session the
remaining endpoints are probed for open/mmap/format/registration
(`playback_attempted: false`). In the sealed receipt the other HDA rings
(NVIDIA HDMI 1,3/7/8/9 and ALC897 Digital 2,1) registered but were not
played, and the PipeWire-held USB interface is `INCONCLUSIVE` (busy).

Interpretation discipline: this proves the memory path on one
hardware/driver combination, not a universal property. A device whose ring
the CUDA driver cannot register, or that lacks hw:mmap, produces an explicit
negative row — which is exactly the first-class evidence the paper demands.
