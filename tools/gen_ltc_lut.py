#!/usr/bin/env python3
"""
Generate the LTC LUT binary asset consumed by the renderer.

Source: Heitz et al. "Real-Time Polygonal-Light Shading with Linearly Transformed
Cosines" (2016) — the canonical fitted tables published at
https://github.com/selfshadow/ltc_code (`fit/results/ltc.inc`).

Run once when the LUT changes:
    curl -sSL -o /tmp/ltc.inc \\
        https://raw.githubusercontent.com/selfshadow/ltc_code/master/fit/results/ltc.inc
    python3 tools/gen_ltc_lut.py /tmp/ltc.inc scenes/assets/ltc_lut.bin

Output layout (little-endian f32, 131072 bytes total):
    [0,        65536) — LUT0: 64x64 RGBA32F, GGX-fitted Minv components.
                        Row-major; row = theta-parameter (sqrt(1-NdotV)),
                        column = roughness. Each texel packs the four
                        non-trivial elements of the inverse matrix
                        (which has the form [[a,0,b],[0,1,0],[c,0,d]]
                        after normalising by m11):
                            R = a   G = b   B = c   A = d
    [65536, 131072)   — LUT1: 64x64 RGBA32F, magnitude scaling.
                        R = magnitude, G/B/A = 0 (reserved for fresnel/etc).

The renderer's GLSL reconstructs Minv from LUT0 as:
    mat3 Minv = mat3(vec3(t.x, 0, t.y), vec3(0, 1, 0), vec3(t.z, 0, t.w));
"""

import re
import struct
import sys

LUT_SIZE = 64
ENTRIES = LUT_SIZE * LUT_SIZE


def parse_brace_table(text: str, header: str) -> list[list[float]]:
    """Extract a sequence of `{a, b, c, ...}` inner rows that follow `header`.

    The C source has an outer `{ ... };` wrapping the inner `{...}` rows, so we
    skip past the first `{` (the outer opener) before scanning.
    """
    start = text.index(header)
    end = text.index("};", start)
    body = text[start + text[start:].index("{") + 1 : end]
    rows = re.findall(r"\{([^}]*)\}", body)
    return [[float(x) for x in row.split(",")] for row in rows]


def parse_float_table(text: str, header: str) -> list[float]:
    """Extract a sequence of bare floats that follow `header`."""
    start = text.index(header)
    end = text.index("};", start)
    body = text[start:end]
    return [
        float(m.group(0).rstrip("f"))
        for m in re.finditer(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?f?", body[len(header):])
    ]


def main(in_path: str, out_path: str) -> None:
    text = open(in_path).read()

    minv = parse_brace_table(text, "tabMinv[size*size]")
    mags = parse_float_table(text, "tabMagnitude[size*size]")

    if len(minv) != ENTRIES:
        raise SystemExit(f"expected {ENTRIES} Minv entries, got {len(minv)}")
    if len(mags) != ENTRIES:
        raise SystemExit(f"expected {ENTRIES} magnitude entries, got {len(mags)}")

    out = bytearray()

    # LUT0: normalised Minv components.
    # The fitter normalises so the matrix has the form
    #   [a 0 b]
    #   [0 c 0]
    #   [d 0 e]
    # We renormalise by c (= m[1][1]) so the middle row becomes identity, then
    # store (a, b, d, e) in RGBA.
    for row in minv:
        m = row  # 9 elements, row-major
        c = m[4]
        if c == 0.0:
            c = 1.0
        a = m[0] / c
        b = m[2] / c
        d = m[6] / c
        e = m[8] / c
        out += struct.pack("<ffff", a, b, d, e)

    # LUT1: magnitude in R, G/B/A reserved.
    for mag in mags:
        out += struct.pack("<ffff", mag, 0.0, 0.0, 0.0)

    with open(out_path, "wb") as f:
        f.write(out)
    print(f"wrote {len(out)} bytes to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: gen_ltc_lut.py <ltc.inc> <out.bin>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
