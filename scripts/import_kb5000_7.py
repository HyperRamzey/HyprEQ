#!/usr/bin/env python3
"""Convert a KB5000_7 export into the two-column CSV the target recipe expects.

The KB5000_7 curve (Olive & Clark, Express Paper 394, AES 159th Convention, 2025) is
published as an attachment the authors distribute themselves. Exports of it come out of
measurement tools in the tool's own format: quoted title rows, an ``X,Y`` marker,
``Hz,dBSPL`` header row, and quoted summary rows at the end. This script keeps only the
numeric frequency/gain pairs — sorted, de-duplicated, nothing else — and writes the
``frequency_hz,gain_db`` file ``targets/harman_kb5000_7.json`` reads beside it.

No curve data lives in this repository: the numbers come from your own copy of the
attachment, and this script parses whatever it is given without embedding any of it.

Usage:
    python3 scripts/import_kb5000_7.py <export.txt> [output.csv]

The default output is ``targets/kb5000_7.csv`` relative to the repository root.
"""

import math
import sys
from pathlib import Path


def parse_pairs(path: Path) -> list[tuple[float, float]]:
    """Return the numeric (frequency_hz, gain_db) pairs in an ARTA-style export."""
    pairs: dict[float, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [f.strip().strip('"').strip() for f in line.split(",")]
        try:
            nums = [float(f) for f in fields if f != ""]
        except ValueError:
            continue
        # Pairs are read positionally so a stray trailing field cannot shift the columns
        # into each other; zip() drops an odd last field rather than pairing it onward.
        for f, g in zip(nums[0::2], nums[1::2], strict=False):
            if f > 0.0 and math.isfinite(f) and math.isfinite(g):
                pairs[f] = g
    return sorted(pairs.items())


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(argv[1])
    out = (
        Path(argv[2])
        if len(argv) == 3
        else Path(__file__).resolve().parent.parent / "targets" / "kb5000_7.csv"
    )
    pairs = parse_pairs(src)
    if len(pairs) < 2:
        print(
            f"error: no usable frequency/gain pairs found in '{src}'", file=sys.stderr
        )
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{f:.6g},{g:.3f}\n" for f, g in pairs), encoding="utf-8")
    near_1k = min(pairs, key=lambda p: abs(math.log(p[0] / 1000.0)))
    print(
        f"wrote {len(pairs)} points ({pairs[0][0]:.6g}..{pairs[-1][0]:.6g} Hz) to "
        f"'{out}'; level at {near_1k[0]:.6g} Hz = {near_1k[1]:.3f} dB",
        file=sys.stderr,
    )
    if pairs[-1][0] < 20000.0:
        print(
            "warning: curve ends below 20 kHz; the fit will hold the last value flat above it",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
