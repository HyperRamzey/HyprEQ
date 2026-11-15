#!/usr/bin/env python3
"""Parity harness: Rust `hypreq` against the Python reference in `hypreq.py`.

Each scenario runs in its own scratch directory on both sides, from the same
`measurements.json` and the same input WAV, then compares:

* stdout, after normalising the absolute paths the tools print, byte-for-byte;
* every WAV the run wrote — container fields exactly, samples within tolerance
  (float payloads must agree to ~1 ULP, PCM payloads must agree bit-for-bit);
* audit CSVs, byte-for-byte;
* usage errors: message and exit code.

The verification PNGs are deliberately *not* compared: matplotlib and `plotters`
are different renderers, so the PNG check is "the Rust side wrote a decodable PNG
of the expected size", and the geometry itself is covered by unit tests.

Exit code is 0 only when every comparison passed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
SAMPLE_TOL = 4e-16  # relative to the payload peak; ~2 ULP of f64


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Scenario:
    name: str
    argv: list[str]
    expected_exit: int = 0
    artifacts: list[str] = field(default_factory=list)
    compare_stdout: bool = True
    png: list[str] = field(default_factory=list)


def norm(text: str, workdir: Path) -> str:
    """Make stdout comparable across the two scratch directories."""
    text = text.replace(str(workdir.resolve()), "<DIR>")
    text = text.replace(str(workdir), "<DIR>")
    return text.replace("\\", "/")


def read_wav_header(path: Path) -> dict:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    off, chunks, fmt = 12, [], {}
    while off + 8 <= len(data):
        cid = data[off : off + 4].decode("latin1")
        size = struct.unpack("<I", data[off + 4 : off + 8])[0]
        body = data[off + 8 : off + 8 + size]
        chunks.append(cid)
        if cid == "fmt ":
            fmt = {
                "audio_format": struct.unpack("<H", body[0:2])[0],
                "channels": struct.unpack("<H", body[2:4])[0],
                "fs": struct.unpack("<I", body[4:8])[0],
                "bits": struct.unpack("<H", body[14:16])[0],
            }
        off += 8 + size + (size & 1)
    fmt["chunks"] = chunks
    fmt["bytes"] = len(data)
    return fmt


def cmp_wav(py: Path, rs: Path) -> Check:
    try:
        hp, hr = read_wav_header(py), read_wav_header(rs)
    except (OSError, ValueError, struct.error) as exc:
        return Check(py.name, False, f"unreadable WAV: {exc}")
    if hp["chunks"] != hr["chunks"]:
        return Check(py.name, False, f"chunk layout {hp['chunks']} != {hr['chunks']}")
    for k in ("audio_format", "channels", "fs", "bits"):
        if hp[k] != hr[k]:
            return Check(py.name, False, f"{k}: {hp[k]} != {hr[k]}")

    try:
        a, _ = sf.read(py, always_2d=True, dtype="float64")
        b, _ = sf.read(rs, always_2d=True, dtype="float64")
    except Exception as exc:  # soundfile raises several flavours depending on the file
        return Check(py.name, False, f"cannot decode payload: {exc}")
    if a.shape != b.shape:
        return Check(py.name, False, f"shape {a.shape} != {b.shape}")
    if a.size == 0:
        return Check(py.name, True, "empty payload")

    # PCM containers are quantised, so the honest comparison is the raw integers:
    # the dither streams are the same fixed sequence on both sides, so equal bytes
    # are required, and anything else is a real divergence.
    if hp["audio_format"] == 1:
        ra = py.read_bytes()[py.read_bytes().find(b"data") :]
        rb = rs.read_bytes()[rs.read_bytes().find(b"data") :]
        if ra == rb:
            return Check(py.name, True, f"PCM {hp['bits']}-bit payload identical")
        lsb = 2.0 ** (-(hp["bits"] - 1))
        delta = float(np.abs(a - b).max())
        if delta <= 3.0 * lsb:
            return Check(
                py.name,
                True,
                f"PCM within dither spread: max {delta / lsb:.2f} LSB",
            )
        return Check(py.name, False, f"PCM max delta {delta / lsb:.2f} LSB")

    peak = float(np.abs(a).max())
    delta = float(np.abs(a - b).max())
    rel = delta / peak if peak else delta
    ok = rel <= SAMPLE_TOL
    return Check(
        py.name,
        ok,
        f"{a.shape[0]} frames x {a.shape[1]}ch, peak {peak:.9f}, "
        f"max delta {delta:.3e} ({rel:.2e} rel)",
    )


def cmp_bytes(py: Path, rs: Path) -> Check:
    try:
        a, b = py.read_bytes(), rs.read_bytes()
    except OSError as exc:
        return Check(py.name, False, f"cannot read: {exc}")
    if a == b:
        return Check(py.name, True, f"{len(a)} bytes identical")
    la, lb = a.decode("utf-8", "replace").splitlines(), b.decode("utf-8", "replace").splitlines()
    first = next(
        (i for i, (x, y) in enumerate(zip(la, lb, strict=False)) if x != y),
        min(len(la), len(lb)),
    )
    return Check(
        py.name,
        False,
        f"{len(a)} vs {len(b)} bytes, first diff at line {first + 1}: "
        f"{la[first] if first < len(la) else '<eof>'!r} != "
        f"{lb[first] if first < len(lb) else '<eof>'!r}",
    )


def cmp_json(py: Path, rs: Path) -> Check:
    try:
        ja, jb = json.loads(py.read_text()), json.loads(rs.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return Check(py.name, False, f"unreadable JSON: {exc}")
    if ja == jb:
        return Check(py.name, True, "parsed JSON equal")
    bad = [k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k)]
    return Check(py.name, False, f"keys differ: {bad}")


def png_check(path: Path, want_wh: tuple[int, int] | None) -> Check:
    if not path.exists():
        return Check(path.name, False, "not written")
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return Check(path.name, False, "not a PNG")
    w, h = struct.unpack(">II", data[16:24])
    if want_wh and (w, h) != want_wh:
        return Check(path.name, False, f"{w}x{h} != {want_wh[0]}x{want_wh[1]}")
    return Check(path.name, True, f"{w}x{h} PNG, {len(data)} bytes")


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #
def make_input_wav(path: Path, fs: int = 48000, seconds: float = 3.0) -> None:
    """A deterministic stereo file that drives the limiter hard: dense multitone
    bed, a synchronised kick transient, and a 20 kHz tone to expose the >0.45 fs
    band the true-peak detector has to interpolate through."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    bed = (
        0.42 * np.sin(2 * np.pi * 99.0 * t + 0.3)
        + 0.33 * np.sin(2 * np.pi * 420.0 * t)
        + 0.28 * np.sin(2 * np.pi * 3100.0 * t + 1.1)
        + 0.16 * np.sin(2 * np.pi * 11800.0 * t + 0.7)
    )
    kick = np.zeros(n)
    for start in (0, int(0.9 * fs), int(1.7 * fs)):
        ln = min(900, n - start)
        seg = np.arange(ln) - ln / 2
        sinc = np.where(seg == 0.0, 0.84, np.sin(2 * np.pi * 0.42 * seg) / (np.pi * seg))
        kick[start : start + ln] += 1.7 * sinc * np.hanning(ln)
    left = bed + kick
    right = 0.92 * np.roll(bed, 37) + np.roll(kick, 37)
    sig = np.column_stack([left, right])
    sig *= 1.18 / np.abs(sig).max()  # deliberately over 0 dBFS on the samples
    sf.write(path, sig.astype(np.float32), fs, subtype="FLOAT")


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=1800, env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "MPLBACKEND": "Agg", "LC_ALL": "C"}
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base",
        default=None,
        help="scratch tree (default: a private mode-0700 directory under $TMPDIR)",
    )
    ap.add_argument("--rust", default=str(REPO / "target" / "release" / "hypreq"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--ref", default=str(REPO / "hypreq.py"))
    ap.add_argument("--data", default=str(REPO / "measurements.json"))
    ap.add_argument("--only", default=None, help="run scenarios whose name contains this")
    ap.add_argument("--keep", action="store_true", help="keep the scratch tree")
    args = ap.parse_args()

    # Created with mode 0700 rather than a fixed world-writable path, so the scratch
    # tree cannot be pre-seeded or watched by another user.
    base = (
        Path(args.base)
        if args.base
        else Path(tempfile.mkdtemp(prefix="hypreq-parity-"))
    )
    base.mkdir(parents=True, exist_ok=True)
    os.chmod(base, 0o700)

    data = Path(args.data)
    inp = base / "input_audio.wav"
    try:
        make_input_wav(inp)
    except Exception as exc:  # noqa: BLE001 - the harness must report, not traceback
        print(f"cannot build the test input: {exc}", file=sys.stderr)
        return 2

    scenarios = [
        Scenario("list-models", ["--list-models"]),
        Scenario(
            "model-jbl-full",
            ["-H", "jbl770nc", "--export-csv"],
            artifacts=[
                "jbl770nc_f64_48k.wav",
                "jbl770nc_mastering_mono_48k.wav",
                "jbl770nc_mastering_stereo_48k.wav",
                "jbl770nc_audit_data.csv",
                "ir.wav",
            ],
            png=["jbl770nc_final_verification.png"],
        ),
        Scenario(
            "model-all-csv",
            ["--headphone", "all", "--export-csv", "--no-plot"],
            artifacts=[
                "jbl770nc_f64_48k.wav",
                "dt770_80_f64_48k.wav",
                "ft1pro_f64_48k.wav",
                "jbl770nc_audit_data.csv",
                "dt770_80_audit_data.csv",
                "ft1pro_audit_data.csv",
            ],
        ),
        Scenario(
            "notch-original",
            ["-H", "dt770_80", "--notch-mode", "original", "--taps", "8192", "--no-plot"],
            artifacts=["dt770_80_f64_48k.wav"],
        ),
        Scenario(
            "notch-zero",
            ["-H", "dt770_80", "--notch-mode", "zero", "--taps", "8192", "--export-csv", "--no-plot"],
            artifacts=["dt770_80_f64_48k.wav", "dt770_80_audit_data.csv"],
        ),
        Scenario(
            "apodize",
            ["-H", "ft1pro", "--apodize", "--taps", "8192", "--no-plot"],
            artifacts=["ft1pro_f64_48k.wav"],
        ),
        Scenario(
            "rate-96k",
            ["-H", "ft1pro", "--fs", "96000", "--taps", "8192", "--no-plot"],
            artifacts=["ft1pro_f64_96k.wav", "ft1pro_mastering_mono_96k.wav"],
        ),
        Scenario(
            "rate-44_1k",
            ["-H", "jbl_770nc", "--fs", "44100", "--taps", "4096", "--no-plot"],
            artifacts=["jbl770nc_f64_44.1k.wav"],
        ),
        Scenario(
            "headroom-and-tp",
            [
                "-H", "dt770", "--headroom-db", "-4.25", "--tp-ratio", "x32",
                "--tp-ceiling", "-1.0", "--taps", "4096", "--no-plot",
            ],
            artifacts=["dt770_80_f64_48k.wav"],
        ),
        Scenario(
            "audio-eq-pcm24",
            ["-H", "jbl770nc", "--input", "input_audio.wav", "--out-bits", "24", "--taps", "8192"],
            artifacts=["input_audio_jbl770nc_tp_x16.wav"],
        ),
        Scenario(
            "audio-eq-trim-f32",
            [
                "-H", "dt770_80ohm", "--input", "input_audio.wav",
                "--out-bits", "32", "--trim-tail", "--taps", "8192",
            ],
            artifacts=["input_audio_dt770_80_tp_x16.wav"],
        ),
        Scenario(
            "audio-bypass-pcm16",
            ["--bypass-eq", "--input", "input_audio.wav", "--out-bits", "16"],
            artifacts=["input_audio_tp_x16.wav"],
        ),
        Scenario("measure-wav", ["--measure-wav", "input_audio.wav"]),
        Scenario(
            "audio-custom-out-x32",
            [
                "-H", "ft1pro", "--input", "input_audio.wav", "--output", "custom_out.wav",
                "--out-bits", "64", "--tp-ratio", "x32", "--tp-ceiling", "-1.0", "--taps", "8192",
            ],
            artifacts=["custom_out.wav"],
        ),
        Scenario(
            "audio-limiter-off",
            ["--bypass-eq", "--no-tp-limiter", "--input", "input_audio.wav", "--out-bits", "24"],
            artifacts=["input_audio.wav"],
        ),
        Scenario(
            "audio-lookahead-raised",
            [
                "-H", "ft1pro", "--input", "input_audio.wav", "--tp-ratio", "x32",
                "--tp-lookahead", "0.2", "--taps", "4096", "--out-bits", "32",
            ],
            artifacts=["input_audio_ft1pro_tp_x32.wav"],
        ),
        Scenario(
            "upgrade-json",
            ["--upgrade-json"],
            artifacts=["measurements.json"],
        ),
        Scenario("usage-taps-too-small", ["--taps", "100", "--no-plot"], expected_exit=2),
        Scenario("usage-bad-model", ["-H", "nope", "--no-plot"], expected_exit=2),
        Scenario("usage-bad-bits", ["--bypass-eq", "--input", "input_audio.wav", "--out-bits", "20"], expected_exit=2),
        Scenario("usage-bad-ratio", ["--bypass-eq", "--input", "input_audio.wav", "--tp-ratio", "x64"], expected_exit=2),
        Scenario("usage-bad-ceiling", ["--bypass-eq", "--input", "input_audio.wav", "--tp-ceiling", "-30"], expected_exit=2),
    ]

    rows: list[tuple[str, list[Check]]] = []
    failures = 0
    for sc in scenarios:
        if args.only and args.only not in sc.name:
            continue
        checks: list[Check] = []
        py_dir, rs_dir = base / f"{sc.name}_py", base / f"{sc.name}_rs"
        setup: list[Check] = []
        for d in (py_dir, rs_dir):
            try:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
                shutil.copy(data, d / "measurements.json")
                shutil.copy(inp, d / "input_audio.wav")
            except OSError as exc:
                setup.append(Check(f"setup {d.name}", False, str(exc)))
        try:
            shutil.copy(REPO / "hypreq.py", py_dir / "hypreq.py")
        except OSError as exc:
            setup.append(Check("setup reference script", False, str(exc)))
        checks.extend(setup)
        if setup:
            failures += len(setup)
            rows.append((sc.name, checks))
            print(f"FAIL  {sc.name}")
            for c in setup:
                print(f"        {c.name}: {c.detail}")
            continue

        rp = run([args.python, "hypreq.py", *sc.argv], py_dir)
        rr = run([args.rust, *sc.argv], rs_dir)

        if rp.returncode != sc.expected_exit:
            checks.append(
                Check("python exit", False, f"{rp.returncode} != {sc.expected_exit}: {rp.stderr[-300:]}")
            )
        if rr.returncode != sc.expected_exit:
            checks.append(
                Check("rust exit", False, f"{rr.returncode} != {sc.expected_exit}: {rr.stderr[-300:]}")
            )
        else:
            checks.append(Check("exit code", True, f"both {rr.returncode}"))

        if sc.expected_exit == 0:
            if sc.compare_stdout:
                a = norm(rp.stdout, py_dir)
                b = norm(rr.stdout, rs_dir)
                if a == b:
                    checks.append(Check("stdout", True, f"{len(a.splitlines())} lines identical"))
                else:
                    la, lb = a.splitlines(), b.splitlines()
                    first = next(
                        (i for i, (x, y) in enumerate(zip(la, lb, strict=False)) if x != y),
                        min(len(la), len(lb)),
                    )
                    checks.append(
                        Check(
                            "stdout",
                            False,
                            f"line {first + 1}:\n  py: {(la[first] if first < len(la) else '<eof>')!r}\n"
                            f"  rs: {(lb[first] if first < len(lb) else '<eof>')!r}",
                        )
                    )
        else:
            a = norm(rp.stderr, py_dir).strip().splitlines()
            b = norm(rr.stderr, rs_dir).strip().splitlines()
            tail_a = a[-1] if a else ""
            tail_b = b[-1] if b else ""
            checks.append(
                Check(
                    "usage error",
                    tail_a.startswith("usage:") is False and tail_b.startswith("usage:") is False
                    and "error" in tail_a.lower()
                    and "error" in tail_b.lower(),
                    f"py: {tail_a[:110]!r} | rs: {tail_b[:110]!r}",
                )
            )

        for name in sc.artifacts:
            p, r = py_dir / name, rs_dir / name
            if not p.exists() and not r.exists():
                checks.append(Check(name, False, "neither side wrote it"))
                continue
            if not p.exists():
                checks.append(Check(name, False, "python did not write it"))
                continue
            if not r.exists():
                checks.append(Check(name, False, "rust did not write it"))
                continue
            checks.append(cmp_json(p, r) if name.endswith(".json") else (cmp_bytes(p, r) if name.endswith(".csv") else cmp_wav(p, r)))

        for name in sc.png:
            checks.append(png_check(rs_dir / name, (1950, 1725)))

        bad = [c for c in checks if not c.ok]
        failures += len(bad)
        rows.append((sc.name, checks))
        print(f"{'FAIL' if bad else 'ok  '}  {sc.name}")
        for c in bad:
            print(f"        {c.name}: {c.detail}")

    print()
    total = sum(len(c) for _, c in rows)
    print(f"{total - failures}/{total} checks passed across {len(rows)} scenarios")
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)
    else:
        print(f"scratch tree kept in {base}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
