#!/usr/bin/env python3
"""Bootstrap-only generator for assets/u1/resampler_bh64_p1024_q15.bin.

The Rust regenerator (cargo test -- --ignored regenerate_resampler_asset) is
the authoritative non-normative generator; this script exists only to break
the include_bytes bootstrapping cycle and MUST produce identical bytes.
"""
import math
import struct

TAPS, PHASES = 64, 1024
A0, A1, A2, A3 = 0.35875, 0.48829, 0.14128, 0.01168


def bh(x):
    return A0 - A1 * math.cos(2 * math.pi * x) + A2 * math.cos(4 * math.pi * x) - A3 * math.cos(6 * math.pi * x)


def sinc(t):
    if abs(t) < 1e-12:
        return 1.0
    p = math.pi * t
    return math.sin(p) / p


def kernel(t):
    if abs(t) > 32.0:
        return 0.0
    return sinc(t) * bh((t + 32.0) / 64.0)


out = bytearray()
for p in range(PHASES):
    f = (p + 0.5) / PHASES
    raw = [kernel(31.0 + f - j) for j in range(TAPS)]
    s = sum(raw)
    scaled = [min(k * (1 << 15) / s, 32767.0) for k in raw]
    q = [int(math.floor(v)) for v in scaled]
    deficit = (1 << 15) - sum(q)
    while deficit > 0:
        best, best_frac = None, -1.0
        for j, v in enumerate(scaled):
            if q[j] >= 32767:
                continue
            frac = v - q[j]
            if frac > best_frac:
                best_frac, best = frac, j
        assert best is not None
        q[best] += 1
        deficit -= 1
    assert all(-32768 <= v <= 32767 for v in q)
    assert sum(q) == 1 << 15
    for v in q:
        out += struct.pack("<h", v)

path = "assets/u1/resampler_bh64_p1024_q15.bin"
with open(path, "wb") as fh:
    fh.write(bytes(out))
print("wrote", len(out), "bytes to", path)
