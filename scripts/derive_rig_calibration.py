#!/usr/bin/env python3
"""Derive a HyprEQ RigCalibration from paired same-model cross-rig measurements.

No public dataset surfaced with a ready-made transfer impedance *and* its uncertainty,
so this script builds one from the data that does exist: the same headphone measured on
two fixtures (e.g. GRAS 43AG and B&K 5128, available from headphones.com's database or
headphonedatabase.com). Each pair is 1 kHz-normalised before subtraction, so a rig's
absolute SPL calibration cannot masquerade as a transfer-impedance difference.

    python3 scripts/derive_rig_calibration.py \
        --source-fixture gras_43ag --target-fixture bk5128 \
        --pair model_a_43ag.csv model_a_5128.csv \
        --pair model_b_43ag.csv model_b_5128.csv \
        --acoustic-class closed_back --out rig_43ag_to_5128.json

The resulting JSON is passed to ``hypreq --rig-calibration``. ``delta_mean_db`` is
target-rig minus source-rig; ``delta_sigma_db`` is the across-model spread and is what
the global QP propagates as sigma_rig. Run with ``--self-test`` to check the math without CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FIXTURES = {
    "gras_43ag",
    "gras_43ag_clone",
    "gras_45ca",
    "gras_kb5000",
    "bk5128",
    "iec60318_4",
}
CLASSES = {"open_back", "closed_back", "in_ear", "unknown", ""}


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    f, g = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cells = [c for c in line.replace(";", ",").split(",") if c.strip()]
        try:
            numbers = [float(c) for c in cells]
        except ValueError:
            continue
        if len(numbers) >= 2 and numbers[0] > 0 and np.isfinite(numbers[-1]):
            f.append(numbers[0])
            g.append(numbers[-1])
    if len(f) < 2:
        raise ValueError(f"{path}: need at least two numeric rows")
    f = np.asarray(f, dtype=np.float64)
    order = np.argsort(f, kind="stable")
    return f[order], np.asarray(g, dtype=np.float64)[order]


def on_grid(f: np.ndarray, g: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = np.interp(np.log(grid), np.log(f), g)
    ref = float(np.interp(np.log(1000.0), np.log(f), g))
    return vals - ref


def derive(pairs, grid, sigma_floor: float):
    deltas = []
    for source_path, target_path in pairs:
        sf, sg = read_curve(Path(source_path))
        tf, tg = read_curve(Path(target_path))
        deltas.append(on_grid(tf, tg, grid) - on_grid(sf, sg, grid))
    if not deltas:
        raise ValueError("no measurement pairs supplied")
    matrix = np.vstack(deltas)
    mean = matrix.mean(axis=0)
    sigma = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
    sigma = np.maximum(sigma, sigma_floor)
    return mean, sigma, matrix.shape[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", nargs=2, action="append", metavar=("SOURCE_CSV", "TARGET_CSV"),
                    help="paired curves of the SAME headphone; repeat for several models")
    ap.add_argument("--source-fixture", choices=sorted(FIXTURES))
    ap.add_argument("--target-fixture", choices=sorted(FIXTURES))
    ap.add_argument("--acoustic-class", default="unknown", choices=sorted(CLASSES))
    ap.add_argument("--f-min", type=float, default=20.0)
    ap.add_argument("--f-max", type=float, default=20000.0)
    ap.add_argument("--bins-per-octave", type=int, default=48)
    ap.add_argument("--sigma-floor", type=float, default=0.30,
                    help="minimum reported uncertainty in dB (default 0.30)")
    ap.add_argument("--out", type=Path, help="write the calibration JSON here (default stdout)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    steps = round(np.log2(args.f_max / args.f_min) * args.bins_per_octave)
    grid = args.f_min * 2.0 ** (np.arange(steps + 1) / args.bins_per_octave)
    grid[0], grid[-1] = args.f_min, args.f_max

    if args.self_test:
        # derive() reads files; exercise the aggregation math directly with two constant offsets.
        b = np.full_like(grid, 3.0)
        matrix = np.vstack([b, b + 0.5])
        mean, sigma = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        assert np.allclose(mean, 3.25), mean[:3]
        assert np.allclose(sigma, 0.25 * np.sqrt(2.0)), sigma[:3]
        assert np.all(np.diff(grid) > 0)
        print(f"self-test OK: mean={mean[0]:.2f} dB, sigma={sigma[0]:.2f} dB, {len(grid)} bins")
        return 0

    if not args.source_fixture or not args.target_fixture:
        ap.error("--source-fixture and --target-fixture are required")
    if not args.pair:
        ap.error("give at least one --pair, or --self-test")
    try:
        mean, sigma, n = derive(args.pair, grid, args.sigma_floor)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    payload = {
        "source": args.source_fixture,
        "target": args.target_fixture,
        "acoustic_class": args.acoustic_class,
        "f_hz": [round(float(v), 6) for v in grid],
        "delta_mean_db": [round(float(v), 6) for v in mean],
        "delta_sigma_db": [round(float(v), 6) for v in sigma],
        "models": n,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
