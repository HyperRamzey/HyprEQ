#!/usr/bin/env python3
"""Convert SOFA headphone/HRTF measurements into HyprEQ ensemble CSVs.

PHOnA (Princeton Headphone Open Archive) stores repeated headphone placements as
SOFA files. HyprEQ's ``--ensemble-csv`` takes plain ``frequency_hz,gain_db``
curves, one per placement. This script bridges the two: it reads the IR (or TF)
for each measurement × receiver, converts it to a 1 kHz-normalised magnitude
response on a common log-frequency grid, and writes one CSV per curve.

The project venv is required because AUR has no ``python-sofar`` package:

    python3 -m venv .venv
    .venv/bin/python -m pip install sofar
    .venv/bin/python scripts/sofa_to_ensemble.py PHOnA_0.3/.../*.sofa --out-dir ensemble/

Self-test (no SOFA file needed): ``--self-test`` synthesises a known IR, checks
the conversion numerically, and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def log_grid(f_min: float, f_max: float, bins_per_octave: int) -> np.ndarray:
    steps = round(np.log2(f_max / f_min) * bins_per_octave)
    grid = f_min * 2.0 ** (np.arange(steps + 1) / bins_per_octave)
    # Match hypreq::global_fit::uniform_log_grid: exact endpoints, clamped at the top.
    grid[0] = f_min
    grid[-1] = f_max
    return grid


def magnitude_db_from_ir(ir: np.ndarray, fs: float, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rfftfreq, 20log10|H|) for a real impulse response."""
    n = max(int(n_fft), 2 * len(ir))
    spec = np.fft.rfft(np.asarray(ir, dtype=np.float64), n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(spec), 1e-12))
    return freqs, mag


def magnitude_db_from_tf(real: np.ndarray, imag: np.ndarray, fs: float, n_fft: int):
    n = max(int(n_fft), 2 * (len(real) - 1))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.asarray(real, dtype=np.float64) + 1j * np.asarray(imag, dtype=np.float64)
    return freqs, 20.0 * np.log10(np.maximum(np.abs(spec), 1e-12))


def resample_normalised(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    grid: np.ndarray,
    normalise_hz: float,
) -> np.ndarray:
    """Log-frequency linear interpolation, then a 1 kHz reference subtraction."""
    valid = (freqs > 0) & np.isfinite(mag_db)
    if valid.sum() < 2:
        raise ValueError("spectrum has fewer than two usable frequency bins")
    log_f = np.log(freqs[valid])
    vals = np.interp(np.log(grid), log_f, mag_db[valid])
    ref = float(np.interp(np.log(normalise_hz), log_f, mag_db[valid]))
    return vals - ref


def curves_from_sofa(path: Path, grid: np.ndarray, n_fft: int, ears: str, normalise_hz: float):
    """Yield (label, curve_db) for each usable measurement × receiver in a SOFA file."""
    try:
        import sofar
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "sofar is not installed in this interpreter; create .venv and install it (see module docstring)"
        ) from exc

    data = sofar.read_sofa(str(path))
    fs = float(np.asarray(data.get("Data.SamplingRate")).ravel()[0])

    receivers: list[int]
    if "Data.IR" in data:
        ir = np.asarray(data["Data.IR"], dtype=np.float64)
        if ir.ndim == 2:
            ir = ir[np.newaxis, ...]
        if ir.ndim == 1:
            ir = ir[np.newaxis, np.newaxis, ...]
        n_m, n_r, _ = ir.shape
        get_spectrum = lambda m, r: magnitude_db_from_ir(ir[m, r], fs, n_fft)
    elif "Data.Real" in data and "Data.Imag" in data:
        real = np.asarray(data["Data.Real"], dtype=np.float64)
        imag = np.asarray(data["Data.Imag"], dtype=np.float64)
        if real.ndim == 2:
            real, imag = real[np.newaxis, ...], imag[np.newaxis, ...]
        if real.ndim == 1:
            real, imag = real[np.newaxis, np.newaxis, ...], imag[np.newaxis, np.newaxis, ...]
        n_m, n_r, _ = real.shape
        get_spectrum = lambda m, r: magnitude_db_from_tf(real[m, r], imag[m, r], fs, n_fft)
    else:
        raise ValueError(f"{path}: SOFA file has neither Data.IR nor Data.Real/Data.Imag")

    if ears == "left":
        receivers = [0]
    elif ears == "right":
        receivers = [min(1, n_r - 1)]
    else:
        receivers = list(range(n_r))

    for m in range(n_m):
        for r in receivers:
            freqs, mag = get_spectrum(m, r)
            curve = resample_normalised(freqs, mag, grid, normalise_hz)
            yield f"{path.stem}_m{m:03d}_r{r}", curve


def write_csv(path: Path, grid: np.ndarray, curve: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# frequency_hz,gain_db  (1 kHz reference removed)\n")
        for f, g in zip(grid, curve):
            fh.write(f"{f:.6f},{g:.6f}\n")


def self_test(grid: np.ndarray, n_fft: int, normalise_hz: float) -> int:
    """A one-bin IR is flat; a one-sample delay must NOT change its magnitude."""
    fs = 48000.0
    ir = np.zeros(128)
    ir[0] = 1.0
    f0, m0 = magnitude_db_from_ir(ir, fs, n_fft)
    delayed = np.zeros(128)
    delayed[17] = 1.0
    f1, m1 = magnitude_db_from_ir(delayed, fs, n_fft)
    assert np.allclose(f0, f1)
    assert np.allclose(m0, m1, atol=1e-9), "delay must not alter magnitude"
    flat = resample_normalised(f0, m0, grid, normalise_hz)
    assert np.allclose(flat, 0.0, atol=1e-9), f"flat IR gave {flat.min():.3g}..{flat.max():.3g} dB"
    assert np.all(np.diff(grid) > 0)

    # Exercise the SOFA extraction shape with an in-memory fake reader: no real SOFA file (or
    # writable SOFA library) needed, and the rest of the path is the same as a PHOnA file.
    import sys
    import types

    fake = types.ModuleType("sofar")
    fake.read_sofa = lambda _path: {
        "Data.SamplingRate": np.array([fs]),
        "Data.IR": np.tile(ir, (2, 2, 1)),  # M=2 placements, R=2 receivers
    }
    saved = sys.modules.get("sofar")
    sys.modules["sofar"] = fake
    try:
        curves = list(curves_from_sofa(Path("fake.sofa"), grid, n_fft, "all", normalise_hz))
    finally:
        if saved is None:
            sys.modules.pop("sofar", None)
        else:
            sys.modules["sofar"] = saved
    assert len(curves) == 4, f"expected 2 placements × 2 ears, got {len(curves)}"
    for _, curve in curves:
        assert np.allclose(curve, 0.0, atol=1e-9), "fake flat SOFA must yield flat curves"
    print(
        f"self-test OK: {len(grid)} bins, {grid[0]:.1f}-{grid[-1]:.1f} Hz, "
        f"{len(curves)} flat curves from fake SOFA"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sofa", nargs="*", type=Path, help="SOFA files (PHOnA uses one per headphone/placement set)")
    ap.add_argument("--out-dir", type=Path, default=Path("ensemble"))
    ap.add_argument("--bins-per-octave", type=int, default=48)
    ap.add_argument("--f-min", type=float, default=20.0)
    ap.add_argument("--f-max", type=float, default=20000.0)
    ap.add_argument("--n-fft", type=int, default=32768)
    ap.add_argument("--normalise-hz", type=float, default=1000.0)
    ap.add_argument("--ear", choices=["all", "left", "right"], default="all")
    ap.add_argument("--prefix", default="hp")
    ap.add_argument("--self-test", action="store_true", help="verify the conversion math without SOFA input")
    args = ap.parse_args(argv)

    if args.bins_per_octave < 1 or args.f_min <= 0 or args.f_max <= args.f_min:
        ap.error("--bins-per-octave must be >= 1 and 0 < --f-min < --f-max")
    grid = log_grid(args.f_min, args.f_max, args.bins_per_octave)
    if args.self_test:
        return self_test(grid, args.n_fft, args.normalise_hz)
    if not args.sofa:
        ap.error("give at least one SOFA file, or --self-test")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in args.sofa:
        for label, curve in curves_from_sofa(path, grid, args.n_fft, args.ear, args.normalise_hz):
            out = args.out_dir / f"{args.prefix}_{label}.csv"
            write_csv(out, grid, curve)
            written += 1
            print(out)
    print(f"wrote {written} ensemble curves to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
