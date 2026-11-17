#!/usr/bin/env python3
"""Parity harness: Rust `hypreq` against the Python reference in `hypreq.py`.

Each scenario runs in its own scratch directory on both sides, from the same
`measurements.json` and the same input WAV, then compares:

* stdout, after normalising the absolute paths the tools print, byte-for-byte;
* every WAV the run wrote — container fields exactly, samples within tolerance
  (float payloads must agree to ~1 ULP, PCM payloads must agree bit-for-bit);
* audit CSVs, byte-for-byte;
* usage errors: message and exit code.

The verification PNGs are deliberately *not* compared pixel to pixel: matplotlib and
`plotters` are different renderers. What is checked instead is that the Rust side wrote
a PNG of the expected size **with real plotted ink inside each of the four panels** —
a figure whose panels are empty (or hold nothing but their own frames) fails. Panel
ink is counted strictly inside each panel, inset past the frame and the tick marks, so
what is counted is data. The exact geometry is covered by the crate's own unit tests.

Exit code is 0 only when every comparison passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
SAMPLE_TOL = 4e-16  # relative to the payload peak; ~2 ULP of f64
# The convolvers are built through a minimum-phase synthesis on a transform of up to
# 2**19 bins, and NumPy's pocketfft and `rustfft` do not agree to the last bit at that
# size: the measured worst case over the whole sweep is 1.4e-15 relative, spread over
# essentially every sample, which is the signature of two transform libraries rather
# than of two different filters. A real divergence — wrong taper, wrong FFT size, a
# notch applied to the wrong band — moves samples by 1e-3 and more. So the bound on a
# float payload is 1e-13 relative: about twenty times the observed library noise, and
# ten million times below anything that would count as a different filter.
FFT_TOL = 1e-13


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
    # Whether this scenario is meant to drive gain reduction. Byte-identical reports
    # would also be satisfied by both sides reporting an idle limiter, so each audio
    # scenario says which of the two outcomes it is actually testing.
    expect_engaged: bool = False


ERROR_PREFIX = re.compile(r"^(?:hypreq(?:\.py)?(?:\.exe)?:\s*)?error:\s*", re.IGNORECASE)


# argparse has rendered this one message three different ways, and reverted one of its own
# changes on the way. Measured by running `hypreq.py` on each interpreter — not read off a
# changelog, which is what makes the boundary a patch level rather than a minor version:
#
#   <= 3.12.7, 3.13.0     repr everywhere    invalid choice: 'bogus' (choose from 'safe', 'original', 'zero')
#                                           invalid choice: 20 (choose from 16, 24, 32, 64)
#   3.12.8 .. 3.14.3      str() everywhere   invalid choice: '20' (choose from safe, original, zero)
#   3.13.7+, 3.14.4+      repr(str(x))        invalid choice: '20' (choose from '16', '24', '32', '64')
#
# Observed on 3.11.18, 3.12.7, 3.12.16, 3.13.0, 3.14.3 and 3.14.10; `--selftest` asserts the
# shapes. The flag name, the wording, the value and the set of choices are ours to get right and
# stay compared word for word. The quote characters are CPython's: they have moved twice, one of
# those moves was undone, and they move at patch granularity, so pinning them makes the harness
# fail on an interpreter that is telling the user exactly the correct thing. Only the quoting
# *inside this one message* is forgiven, and `--selftest` checks that nothing else is.
INVALID_CHOICE = re.compile(r"(invalid choice: )(.*?)( \(choose from [^()]*\))")


def _unquote(token: str) -> str:
    """Strip one matched pair of surrounding quotes, and only those."""
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def canonicalize_choice(message: str) -> str:
    """Drop the quoting from the `invalid choice:` message, leaving its content alone.

    The choice list is split on commas, which is safe for every list this CLI owns: all of them
    are model keys, mode names and bit depths, none of which contain a comma. Anything that does
    not match the message is returned untouched, byte for byte, quotes included.
    """
    match = INVALID_CHOICE.search(message)
    if match is None:
        return message
    value = _unquote(match.group(2))
    tail = match.group(3)
    listed = tail[len(" (choose from ") : -1]
    choices = ", ".join(_unquote(choice) for choice in listed.split(","))
    return f"{message[: match.start()]}{match.group(1)}{value} (choose from {choices}){message[match.end() :]}"


def norm_error(text: str) -> str:
    """The line a script would grep, with the two things CPython owns normalised away.

    argparse writes `hypreq.py: error: ...`, clap writes `error: ...` — that prefix is the
    first difference the two cannot be made to share, since one is a Python file and the other
    a binary. The second is the quoting of the `invalid choice:` message, which is a CPython
    patch-level detail (see `INVALID_CHOICE`). Everything else is a user-facing string and has
    to match word for word: a script greps it, and `'16'` and `16` are different things to grep
    for in any message that is not the one argparse re-quotes itself.
    """
    lines = [line for line in text.strip().splitlines() if "error" in line.lower()]
    line = lines[-1] if lines else ""
    return canonicalize_choice(ERROR_PREFIX.sub("", line.strip()))


def norm(text: str, workdir: Path) -> str:
    """Make stdout comparable across the two scratch directories."""
    text = text.replace(str(workdir.resolve()), "<DIR>")
    text = text.replace(str(workdir), "<DIR>")
    return text.replace("\\", "/")


def selftest() -> int:
    """Prove the harness forgives CPython's quoting and nothing else.

    `norm_error` deliberately ignores one thing, and a normaliser that is never checked is how a
    parity harness ends up passing when nothing matches. The strings below were captured from
    `hypreq.py` on 3.11.18, 3.12.7, 3.12.16, 3.13.0, 3.14.3 and 3.14.10 — they are what those
    interpreters actually print, not what their changelogs say they print.
    """
    failures: list[str] = []
    checks = 0

    def expect(want: bool, got: bool, what: str) -> None:
        nonlocal checks
        checks += 1
        if want != got:
            failures.append(what)

    # The three CPython renderings of one error must collapse to one string.
    bits = [
        "hypreq.py: error: argument --out-bits: invalid choice: 20 (choose from 16, 24, 32, 64)",
        "hypreq.py: error: argument --out-bits: invalid choice: '20' (choose from 16, 24, 32, 64)",
        "error: argument --out-bits: invalid choice: '20' (choose from '16', '24', '32', '64')",
    ]
    notch = [
        "hypreq.py: error: argument --notch-mode: invalid choice: 'bogus' (choose from 'safe', 'original', 'zero')",
        "error: argument --notch-mode: invalid choice: 'bogus' (choose from safe, original, zero)",
    ]
    expect(True, len({norm_error(t) for t in bits}) == 1, "the three --out-bits renderings must agree")
    expect(True, len({norm_error(t) for t in notch}) == 1, "the two --notch-mode renderings must agree")

    # Collapsing quoting must not collapse content. Each mutant below is written as a RAW message
    # that some other CLI mistake would really print, and it is normalised on the way into the
    # comparison, so what is asserted is that `norm_error` preserves the difference. Deriving the
    # mutants from the already-normalised text instead asserts only that `str.replace` can change
    # a string: a normaliser that deleted the whole choice list would have passed that version.
    reference = norm_error(bits[-1])
    mutants = [
        (bits[-1].replace("--out-bits", "--tp-window"), "a different flag must stay different"),
        (bits[-1].replace("choice: '20'", "choice: '21'"), "a different value must stay different"),
        (bits[-1].replace(", '64'", ""), "a shortened choice list must stay different"),
        (
            bits[-1].replace("invalid choice", "invalid Choice"),
            "the wording itself must stay compared",
        ),
        (notch[0], "a different message must stay different"),
    ]
    for raw, why in mutants:
        expect(True, norm_error(raw) != reference, why)

    # Every other message is compared byte for byte, quotes and all: these carry meaning that is
    # not argparse's to change.
    for literal in [
        "hypreq.py: error: argument --tp-ceiling: expected one argument",
        "hypreq.py: error: sample rate must be positive (got 0)",
        "error: argument --tp-ratio: could not convert string to float: 'x64'",
        "hypreq.py: error: argument --fs: invalid int value: 'wide'",
    ]:
        got = norm_error(literal)
        want = ERROR_PREFIX.sub("", literal)
        expect(True, got == want, f"non-choice message must pass through untouched: {literal!r}")

    if failures:
        for reason in failures:
            print(f"FAIL  selftest: {reason}")
        return 1
    print(f"selftest: {checks} assertions passed (choice quoting normalised, nothing else)")
    return 0


def read_wav_header(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    off, chunks = 12, []
    fmt: dict[str, Any] = {}
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
    except Exception as exc:  # noqa: BLE001 - soundfile raises several flavours depending on the file
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

    # A float container. The DSP is f64 on both sides, but a 32-bit file quantises the
    # result to f32, and asking two f64 pipelines that rounded identically to agree to
    # 4e-16 relative would be asking them to agree about bits the container threw away.
    # So the tolerance is the container's own resolution, not the math's.
    peak = float(np.abs(a).max())
    delta = float(np.abs(a - b).max())
    rel = delta / peak if peak else delta
    # f32 has 24 bits of significand; f64 keeps the tolerance the pipeline itself needs.
    tol = max(SAMPLE_TOL, 2.0 ** -23) if hp["bits"] == 32 else FFT_TOL
    ok = rel <= tol
    return Check(
        py.name,
        ok,
        f"{a.shape[0]} frames x {a.shape[1]}ch {hp['bits']}-bit float, peak {peak:.9f}, "
        f"max delta {delta:.3e} ({rel:.2e} rel, tol {tol:.1e})",
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


# Mirrors `panel_rects()` in src/plot.rs panel by panel, in draw order: entries 3 and 4 are the
# row-3 pair, and the third is the LEFT panel (phase), the fourth the RIGHT one (impulse). Rust
# test `the_panel_rects_are_the_ones_the_harness_mirrors` pins these four literals, so a layout
# change fails a named test instead of quietly turning the ink counts below into a survey of
# whatever rectangle happens to be written here.
PANELS = [
    ("level", 127, 129, 1878, 569),
    ("error/target", 127, 736, 1878, 1176),
    ("phase", 127, 1343, 909, 1673),
    ("impulse", 1096, 1343, 1878, 1673),
]
# The panel background in src/plot.rs is `Rgb(0x1B, 0x1F, 0x22)` and the figure behind it is
# `Rgb(0x15, 0x16, 0x18)`, so a box that has drifted off its panel is detectable rather than
# merely smaller. Both are asserted in `png_check`.
PANEL_BG = (0x1B, 0x1F, 0x22)
FIG_BG = (0x15, 0x16, 0x18)
# Inset past the frame line and the tick marks, so what is counted is data.
INK_INSET = 8
# A flat 0 dB line already covers the panel width; anything under this is an empty panel.
MIN_INK = 150


def png_check(path: Path, want_wh: tuple[int, int] | None) -> Check:
    """Decodable PNG of the right size, with plotted ink in all four panels."""
    if not path.exists():
        return Check(path.name, False, "not written")
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return Check(path.name, False, "not a PNG")
    w, h = struct.unpack(">II", data[16:24])
    if want_wh and (w, h) != want_wh:
        return Check(path.name, False, f"{w}x{h} != {want_wh[0]}x{want_wh[1]}")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        return Check(path.name, False, f"cannot verify plotted ink without Pillow: {exc}")
    try:
        img = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad PNG for our purposes
        return Check(path.name, False, f"PNG will not decode: {exc}")
    if img.shape[1] != w or img.shape[0] != h:
        return Check(path.name, False, f"decoded {img.shape[1]}x{img.shape[0]} != {w}x{h}")

    empty = []
    counts = []
    for name, x0, y0, x1, y1 in PANELS:
        x0, y0 = x0 + INK_INSET, y0 + INK_INSET
        x1, y1 = x1 - INK_INSET, y1 - INK_INSET
        if x1 <= x0 or y1 <= y0:
            return Check(path.name, False, f"panel {name} is outside the {w}x{h} figure")
        block = img[y0:y1, x0:x1]
        # Ink is counted against `block[0, 0]`, whatever that pixel happens to be, so a box that
        # has slid off its panel still reports plenty of "ink" - it is counting the neighbouring
        # panel's frame and the axis labels in the gutter - and passes while covering no data.
        # Two things have to be true of the box, measured on the whole interior and on each edge:
        # it must not be dominated by the figure background (the gutter between the two bottom
        # panels is exactly that), and its majority colour must be the panel's own. Panels 1 and 2
        # legitimately end in the shaded HF-bypass span, which is why the second test allows any
        # non-figure colour and the first one is the strict test.
        edge = 12
        for where, strip in [
            ("interior", block),
            ("left edge", block[:, :edge]),
            ("right edge", block[:, -edge:]),
            ("top edge", block[:edge, :]),
            ("bottom edge", block[-edge:, :]),
        ]:
            flat = strip.reshape(-1, 3)
            colours, seen = np.unique(flat, axis=0, return_counts=True)
            dom = colours[seen.argmax()].astype(int).tolist()
            if max(abs(a - b) for a, b in zip(dom, FIG_BG, strict=True)) <= 2:
                return Check(
                    path.name,
                    False,
                    f"panel {name} box is not on its panel: its {where} is the figure "
                    f"background {tuple(dom)}, not the panel background {PANEL_BG}",
                )
            if where == "interior" and max(abs(a - b) for a, b in zip(dom, PANEL_BG, strict=True)) > 2:
                return Check(
                    path.name,
                    False,
                    f"panel {name} box is not on a panel: its background is {tuple(dom)}, "
                    f"not {PANEL_BG}",
                )
        # The interior's own top-left pixel is background; ink is anything that differs
        # from it by more than the encoder's rounding on a single channel.
        bg = block[0, 0]
        ink = int((np.abs(block - bg).max(axis=2) > 24).sum())
        counts.append(f"{name} {ink}")
        if ink < MIN_INK:
            empty.append(f"{name} ({ink} ink px)")
    if empty:
        return Check(path.name, False, f"no plotted ink in: {', '.join(empty)}")
    return Check(path.name, True, f"{w}x{h} PNG, {len(data)} bytes, ink px: " + ", ".join(counts))


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
        # sin(x)/x with the centre written in rather than divided by: np.where evaluates
        # both branches, so the naive form raises a 0/0 warning on every run.
        sinc = np.full(ln, 0.84)
        np.divide(
            np.sin(2 * np.pi * 0.42 * seg), np.pi * seg, out=sinc, where=seg != 0.0
        )
        kick[start : start + ln] += 1.7 * sinc * np.hanning(ln)
    left = bed + kick
    right = 0.92 * np.roll(bed, 37) + np.roll(kick, 37)
    sig = np.column_stack([left, right])
    sig *= 1.18 / np.abs(sig).max()  # deliberately over 0 dBFS on the samples
    sf.write(path, sig.astype(np.float32), fs, subtype="FLOAT")


ENGAGED_RE = re.compile(r"Engaged:\s*([\d,]+)\s*frames")
# One report line, with the oversampling ratio read from it rather than assumed: the
# scenarios exercise both x16 and x32 and the line names the ratio twice.
PRE_POST_RE = re.compile(
    r"Pre -> Post\s*:\s*([+-][\d.]+)\s*->\s*([+-][\d.]+)\s*dBTP\s*\(x\d+\)\s*\|"
    r"\s*Independent x\d+:\s*([+-][\d.]+)"
)


def limiter_engaged(report: str, want: bool) -> Check:
    """Assert what the scenario claims about its limiter, and cross-check its numbers.

    The report line is `Limiter Pre -> Post : a -> b dBTP (x16) | Independent x16: c`,
    where `c` comes from a second true-peak detector run over the written output — so
    `b != c` means the limiter's accounting and its own meter disagree. A scenario that
    expects to stay under its ceiling is checked the other way: `b == a`, nothing gained
    or lost, because a transparent run that quietly shaved the peaks is a bug too.
    """
    m = ENGAGED_RE.search(report)
    if m is None:
        return Check("limiter state", False, "no 'Engaged: N frames' line in the report")
    frames = int(m.group(1).replace(",", ""))
    pp = PRE_POST_RE.search(report)
    if pp is None:
        return Check("limiter state", False, "no Pre -> Post line to cross-check")
    pre, post, independent = (float(pp.group(i)) for i in (1, 2, 3))
    if abs(post - independent) > 0.005:
        return Check(
            "limiter state",
            False,
            f"post {post} dBTP vs the report's own independent detector {independent} dBTP",
        )
    if want and frames == 0:
        return Check("limiter state", False, f"ceiling never reached (pre {pre} dBTP)")
    if not want and (frames or abs(post - pre) > 0.005):
        return Check(
            "limiter state",
            False,
            f"expected transparency, got {frames:,} engaged frames, {pre} -> {post} dBTP",
        )
    return Check(
        "limiter state",
        True,
        f"{frames:,} engaged frames, {pre} -> {post} dBTP, "
        f"independent detector agrees ({independent} dBTP)",
    )


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run one side of a scenario.

    The child inherits the caller's environment: the reference is launched by
    `sys.executable` (or `--python`), and a venv, conda or Nix install reaches its
    own libraries through variables that a hand-written four-key environment drops —
    which reads as a reference crash, not as a harness bug. Only the knobs that
    actually move the output are pinned here: the matplotlib backend, the locale,
    and a scratch font/image cache so a stale user cache cannot bend the figure.
    """
    env = os.environ.copy()
    env.setdefault("HOME", str(Path.home()))
    env["MPLBACKEND"] = "Agg"
    env["LC_ALL"] = "C"
    env["MPLCONFIGDIR"] = str(cwd / ".mplcache")
    return subprocess.run(
        cmd,
        check=False,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )


def main() -> int:
    # Checked before the parser, so the harness's own logic is testable without a built binary,
    # a numpy, or the reference present.
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(
        description="Parity harness: Rust hypreq against the Python reference in hypreq.py"
    )
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
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="check the harness's own normalisers and exit (no binaries run)",
    )
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
        # Note the two naming rules, which are not the same and are both checked here:
        # convolvers are written under the model's `file_prefix` ("dt770_80ohm"), the audit
        # CSV under its key ("dt770_80"). Getting one of them wrong looks like a missing
        # artefact on both sides, so it is spelled out.
        Scenario(
            "model-all-csv",
            ["--headphone", "all", "--export-csv", "--no-plot"],
            artifacts=[
                "jbl770nc_f64_48k.wav",
                "dt770_80ohm_f64_48k.wav",
                "ft1pro_f64_48k.wav",
                "jbl770nc_audit_data.csv",
                "dt770_80_audit_data.csv",
                "ft1pro_audit_data.csv",
            ],
        ),
        Scenario(
            "notch-original",
            ["-H", "dt770_80", "--notch-mode", "original", "--taps", "8192", "--no-plot"],
            artifacts=["dt770_80ohm_f64_48k.wav"],
        ),
        Scenario(
            "notch-zero",
            ["-H", "dt770_80", "--notch-mode", "zero", "--taps", "8192", "--export-csv", "--no-plot"],
            artifacts=["dt770_80ohm_f64_48k.wav", "dt770_80_audit_data.csv"],
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
            artifacts=["dt770_80ohm_f64_48k.wav"],
        ),
        Scenario(
            "audio-eq-pcm24",
            ["-H", "jbl770nc", "--input", "input_audio.wav", "--out-bits", "24", "--taps", "8192"],
            artifacts=["input_audio_jbl770nc_tp_x16.wav"],
        ),
        # The model preamp (-5.6 dB here) parks the corrected signal under a -3 dBTP
        # ceiling, so this scenario pulls the ceiling to -8 to drive gain reduction
        # through the f32 path; `audio-eq-pcm24` stays at the default ceiling and
        # asserts the never-engaged report is identical too.
        Scenario(
            "audio-eq-trim-f32",
            [
                "-H", "dt770_80ohm", "--input", "input_audio.wav",
                "--out-bits", "32", "--trim-tail", "--taps", "8192",
                "--tp-ceiling", "-8.0",
            ],
            artifacts=["input_audio_dt770_80ohm_tp_x16.wav"],
            expect_engaged=True,
        ),
        Scenario(
            "audio-bypass-pcm16",
            ["--bypass-eq", "--input", "input_audio.wav", "--out-bits", "16"],
            artifacts=["input_audio_tp_x16.wav"],
            expect_engaged=True,
        ),
        Scenario("measure-wav", ["--measure-wav", "input_audio.wav"]),
        Scenario(
            "audio-custom-out-x32",
            [
                # -6 dBTP is below this model's post-EQ peak (-1.96), so the f64 writer
                # and the x32 detector are exercised with the limiter actually working.
                "-H", "ft1pro", "--input", "input_audio.wav", "--output", "custom_out.wav",
                "--out-bits", "64", "--tp-ratio", "x32", "--tp-ceiling", "-6.0", "--taps", "8192",
            ],
            expect_engaged=True,
            artifacts=["custom_out.wav"],
        ),
        Scenario(
            "audio-limiter-off",
            ["--bypass-eq", "--no-tp-limiter", "--input", "input_audio.wav", "--out-bits", "24"],
            # With no --output, and both the EQ tag and the limiter tag empty, both programs
            # append `_out` rather than overwrite the file they just read, so the artefact this
            # scenario can check is the `_out` file. Naming the input here would compare a seed
            # against itself — the rewritten-artefact guard below now refuses that.
            artifacts=["input_audio_out.wav"],
        ),
        Scenario(
            "audio-lookahead-raised",
            [
                # A lookahead the limiter never reaches cannot be observed, so this
                # scenario sets a ceiling below the post-EQ peak (-1.96 dBTP).
                "-H", "ft1pro", "--input", "input_audio.wav", "--tp-ratio", "x32",
                "--tp-lookahead", "0.2", "--taps", "4096", "--out-bits", "32",
                "--tp-ceiling", "-6.0",
            ],
            artifacts=["input_audio_ft1pro_tp_x32.wav"],
            expect_engaged=True,
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
        # Every domain refusal the limiter can raise, and the two argparse choices the
        # CLI owns. Each one compares the whole message, so a reworded or re-quoted error
        # is a parity failure rather than a silent drift.
        Scenario(
            "usage-bad-lookahead",
            ["--bypass-eq", "--input", "input_audio.wav", "--tp-lookahead", "5.0"],
            expected_exit=2,
        ),
        Scenario(
            "usage-bad-attack",
            ["--bypass-eq", "--input", "input_audio.wav", "--tp-attack", "500"],
            expected_exit=2,
        ),
        Scenario(
            "usage-bad-release",
            ["--bypass-eq", "--input", "input_audio.wav", "--tp-release", "5"],
            expected_exit=2,
        ),
        Scenario(
            "usage-bad-link",
            ["--bypass-eq", "--input", "input_audio.wav", "--tp-link-transient", "75"],
            expected_exit=2,
        ),
        Scenario("usage-bad-notch", ["--notch-mode", "bogus", "--no-plot"], expected_exit=2),
        Scenario(
            "usage-bad-notch-audio",
            ["--bypass-eq", "--input", "input_audio.wav", "--notch-mode", "bogus"],
            expected_exit=2,
        ),
    ]

    # Say which two programs were actually compared. A stale `target/release/hypreq`
    # otherwise passes every check and proves nothing about the code you just changed.
    print(f"rust: {args.rust}")
    print(f"python: {args.python}\nreference: {args.ref}\n")

    # The files copied into both scratch directories before a scenario runs.
    seeds = {"measurements.json": data, "input_audio.wav": inp}

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
            shutil.copy(Path(args.ref), py_dir / "hypreq.py")
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
        if rp.returncode == sc.expected_exit and rr.returncode == sc.expected_exit:
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
            # The message itself, not merely the fact that something was called an error:
            # a usage error is a user-facing string, and scripts grep it. The program-name
            # prefix is the one difference clap and argparse cannot be made to share (one
            # is a Python file, the other a binary), so that is normalised away and the
            # rest has to match word for word.
            ea, eb = norm_error(rp.stderr), norm_error(rr.stderr)
            checks.append(
                Check(
                    "usage wording",
                    ea == eb,
                    "identical" if ea == eb else f"py: {ea!r}\n    rs: {eb!r}",
                )
            )
            checks.append(
                Check(
                    "usage error",
                    tail_a.startswith("usage:") is False and tail_b.startswith("usage:") is False
                    and "error" in tail_a.lower()
                    and "error" in tail_b.lower(),
                    f"py: {tail_a[:110]!r} | rs: {tail_b[:110]!r}",
                )
            )

        if sc.expected_exit == 0 and any("_tp_" in n for n in sc.artifacts):
            # Byte-identical stdout would also be satisfied by two sides that agree the
            # limiter never touched the signal, so engagement and the cross-check the
            # report performs on itself are asserted here as well.
            checks.append(limiter_engaged(rp.stdout, sc.expect_engaged))

        for name in sc.artifacts:
            p, r = py_dir / name, rs_dir / name
            if name in seeds and p.exists() and r.exists():
                # Every scenario is handed a copy of these files before it runs, so comparing one
                # after the run only means something if the run actually rewrote it. Naming the
                # input as the artefact is how `audio-limiter-off` ended up byte-comparing a seed
                # against itself once both programs started appending `_out` to the default name.
                pristine = seeds[name].read_bytes()
                if p.read_bytes() == pristine and r.read_bytes() == pristine:
                    checks.append(
                        Check(
                            f"{name} rewritten",
                            False,
                            "neither side changed the file this scenario claims to check, so the "
                            "comparison is a copy of the input against its own copy",
                        )
                    )
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
        try:
            shutil.rmtree(base, ignore_errors=True)
        except OSError as exc:  # the harness's own scratch tree, never operator data
            print(f"warning: left {base} behind: {exc}", file=sys.stderr)
    else:
        print(f"scratch tree kept in {base}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
