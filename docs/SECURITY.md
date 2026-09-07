# Security — hostile-input posture

Media input is treated as hostile. This file states the posture; the numeric
ceilings live in `src/limits.rs` (compiled into device builds unchanged), and
the enforcement lives in every parser, decoder, scheduler, and kernel.

## Hard bounds (see `src/limits.rs`)

Objects, voices, dependencies, recursion depth, graph nodes, event rate,
channel count, sample rate, object frames, checkpoints, residual records,
allocations, file sizes, table sizes, string lengths.

## Rejections

- dependency cycles where not explicitly legal;
- zero-delay algebraic cycles (recursive filter topologies);
- integer overflow anywhere in normative arithmetic (explicit
  checked/wrapping/saturating operations only);
- invalid offsets, overlapping malformed chunks, impossible layouts;
- unbounded expansion (a small file must not expand without limit);
- hash mismatches (integrity failure is an error, not a warning);
- no arbitrary native code in `.voleaudio` files — period. If a future
  procedural bytecode exists it must be bounded and sandboxed; that is not
  built first.

## Parser rules

Every parser operation is length-checked. No recursive allocation bombs, no
unbounded object graphs, no unchecked `usize` conversions from file lengths,
no hidden platform endian assumptions (wire format is explicit little-endian).
Malformed WAV chunk lengths, odd padding, truncated files, duplicate chunks,
unsupported formats, absurd rates, and overflow are tested (`tests`, fuzz
targets in `fuzz/`).

## Unsafe policy

`unsafe` lives only at narrow platform boundaries (ALSA/CUDA/HIP FFI, mmap
views, SIMD intrinsics). Every unsafe block carries a `// SAFETY:` comment
covering ownership, lifetime, alignment, initialized extent, concurrent
access, synchronization, device visibility, and teardown ordering. RAII
wrappers own PCM handles, CUDA context/module/stream/event, HIP
stream/module/event, host registrations, mmap registrations, and device
allocations. Shutdown order is: stop submissions → synchronize GPU → stop/
prepare/drop PCM → unregister mapped memory → free GPU state → destroy runtime
objects → close endpoint. Error paths are tested.
