#!/usr/bin/env python3
"""
HyprEQ — Minimum-Phase FIR Headphone EQ & Verification Suite
============================================================
Synthesizes 64-bit IEEE float (pcm_f64le) minimum-phase impulse responses via
real-cepstrum Hilbert folding, plus 32-bit float (pcm_f32le) mastering IRs,
audit CSVs, and 4-panel acoustic verification plots.

All target and headphone measurement data lives outside this script in
`measurements.json` (and optionally individual `.json` / `.csv` files inside
`models/`) so users can add new headphones without editing Python code.

Usage:
  python3 hypreq.py --headphone all
  python3 hypreq.py --headphone ft1pro
  python3 hypreq.py --headphone dt770_80 --notch-mode safe
  python3 hypreq.py --headphone jbl770nc --fs 96000 --taps 131072
  python3 hypreq.py --headphone all --export-csv
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import struct
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


# =============================================================================
# 1. EXTERNAL MEASUREMENT DATA LOADER & AUTO-EQ CURVE DERIVATION
# =============================================================================

DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "measurements.json"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"

MODEL_ALIASES = {
    "fiio_ft1_pro": "ft1pro",
    "ft1_pro": "ft1pro",
    "dt770": "dt770_80",
    "dt770_80ohm": "dt770_80",
    "jbl_770nc": "jbl770nc",
}


def format_rate_tag(fs: int) -> str:
    return f"{fs // 1000}k" if fs % 1000 == 0 else f"{fs / 1000:.1f}k"


def pchip_log_interp(
    f_nodes: np.ndarray, g_nodes: np.ndarray, f_query: np.ndarray
) -> np.ndarray:
    """Fritsch-Carlson monotonic cubic Hermite spline (PCHIP) in log10(f)."""
    x = np.log10(np.asarray(f_nodes, dtype=np.float64))
    y = np.asarray(g_nodes, dtype=np.float64)
    xq = np.log10(
        np.clip(np.asarray(f_query, dtype=np.float64), f_nodes[0], f_nodes[-1])
    )

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)

    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            d[k] = 0.0

    d[0] = delta[0]
    d[-1] = delta[-1]

    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    dx = xq - x[idx]
    hi = h[idx]
    t = dx / hi

    h00 = (2.0 * t**3) - (3.0 * t**2) + 1.0
    h10 = (t**3) - (2.0 * t**2) + t
    h01 = (-2.0 * t**3) + (3.0 * t**2)
    h11 = (t**3) - (t**2)

    return (
        h00 * y[idx]
        + h10 * hi * d[idx]
        + h01 * y[idx + 1]
        + h11 * hi * d[idx + 1]
    )


def derive_ctrl_pts_from_raw(
    raw_pts: List[Tuple[float, ...]],
    target_pts: List[Tuple[float, float]],
    harm_offset: float = 0.0,
    hf_taper_start_hz: float = 8500.0,
    hf_bypass_hz: float = 9500.0,
    subbass_leak_discount_db: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Automatically derives a rig-compensated HyprEQ control curve from `raw_pts`
    when a user adds a custom headphone model with only raw measurements.
    """
    f_raw = np.array([p[0] for p in raw_pts], dtype=np.float64)
    g_raw = np.array([p[-1] for p in raw_pts], dtype=np.float64)
    f_tar = np.array([p[0] for p in target_pts], dtype=np.float64)
    g_tar = np.array([p[1] for p in target_pts], dtype=np.float64) + harm_offset

    g_tar_on_raw = pchip_log_interp(f_tar, g_tar, f_raw)
    err = g_tar_on_raw - g_raw

    ctrl: List[Tuple[float, float]] = []
    for f, e in zip(f_raw, err):
        if f < 65.0 and subbass_leak_discount_db > 0.0:
            disc = subbass_leak_discount_db * np.exp(
                -((np.log2(max(f, 20.0) / 20.0)) / 0.75) ** 2
            )
            val = e - disc
        elif f <= hf_taper_start_hz:
            val = e
        elif f < hf_bypass_hz:
            w = 0.5 * (
                1.0
                + np.cos(
                    np.pi
                    * np.log2(f / hf_taper_start_hz)
                    / np.log2(hf_bypass_hz / hf_taper_start_hz)
                )
            )
            val = e * w
        else:
            val = 0.0
        ctrl.append((float(f), float(np.round(val, 2))))
    return ctrl


def load_csv_points(csv_path: Path) -> List[Tuple[float, ...]]:
    """Loads a 2-column (freq, avg) or 4-column (freq, L, R, avg) CSV file."""
    pts: List[Tuple[float, ...]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            try:
                vals = tuple(float(x.strip()) for x in row if x.strip())
                if len(vals) >= 2:
                    pts.append(vals)
            except ValueError:
                continue
    return pts


def normalize_model_entry(
    key: str,
    cfg: dict,
    targets: Dict[str, List[Tuple[float, float]]],
    base_dir: Path,
) -> dict:
    """Validates and normalizes a single headphone model configuration."""
    target_name = cfg.get("target", "harman_2018")
    if target_name not in targets:
        raise ValueError(f"Model '{key}' references unknown target '{target_name}'")
    target_pts = targets[target_name]

    if "raw_csv" in cfg and not cfg.get("raw_pts"):
        raw_pts = load_csv_points(base_dir / cfg["raw_csv"])
    else:
        raw_pts = [tuple(float(x) for x in pt) for pt in cfg["raw_pts"]]

    harm_offset = float(cfg.get("harm_offset", 0.0))
    hf_bypass_hz = float(cfg.get("hf_bypass_hz", 9500.0))
    hf_taper_start_hz = float(cfg.get("hf_taper_start_hz", hf_bypass_hz * 0.88))

    if "ctrl_csv" in cfg and not cfg.get("ctrl_pts"):
        ctrl_pts = [(p[0], p[-1]) for p in load_csv_points(base_dir / cfg["ctrl_csv"])]
    elif cfg.get("ctrl_pts"):
        ctrl_pts = [(float(p[0]), float(p[1])) for p in cfg["ctrl_pts"]]
    else:
        ctrl_pts = derive_ctrl_pts_from_raw(
            raw_pts=raw_pts,
            target_pts=target_pts,
            harm_offset=harm_offset,
            hf_taper_start_hz=hf_taper_start_hz,
            hf_bypass_hz=hf_bypass_hz,
            subbass_leak_discount_db=float(cfg.get("subbass_leak_discount_db", 0.0)),
        )

    notch_range = cfg.get("notch_range_hz")
    if notch_range is not None:
        notch_range = (float(notch_range[0]), float(notch_range[1]))

    return {
        "key": key,
        "name": cfg.get("name", key),
        "file_prefix": cfg.get("file_prefix", key),
        "rig_info": cfg.get("rig_info", "IEC 60318-4 (711) Coupler"),
        "target_name": target_name,
        "target_pts": target_pts,
        "harm_offset": harm_offset,
        "gauss_apodize": bool(cfg.get("gauss_apodize", True)),
        "hf_bypass_hz": hf_bypass_hz,
        "notch_range_hz": notch_range,
        "safe_cap_db": float(cfg.get("safe_cap_db", 1.50)),
        "autoeq_geq_str": cfg.get("autoeq_geq_str"),
        "annotations": cfg.get("annotations", []),
        "raw_pts": raw_pts,
        "ctrl_pts": ctrl_pts,
    }


def load_measurement_database(
    data_file: Path = DEFAULT_DATA_FILE,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, dict]]:
    """
    Loads targets and headphone models from `measurements.json` and any
    standalone `.json` or `.csv` files placed inside `models/`.
    """
    if not data_file.exists():
        raise FileNotFoundError(
            f"Measurement database not found at '{data_file}'. "
            "Place 'measurements.json' next to hypreq.py or pass --data-file."
        )

    with data_file.open("r", encoding="utf-8") as fh:
        raw_db = json.load(fh)

    targets: Dict[str, List[Tuple[float, float]]] = {}
    for t_key, t_pts in raw_db.get("targets", {}).items():
        targets[t_key] = [(float(p[0]), float(p[1])) for p in t_pts]

    models: Dict[str, dict] = {}
    for m_key, m_cfg in raw_db.get("models", {}).items():
        models[m_key] = normalize_model_entry(
            m_key, m_cfg, targets, data_file.parent
        )

    if models_dir.exists() and models_dir.is_dir():
        for p in sorted(models_dir.glob("*.json")):
            with p.open("r", encoding="utf-8") as fh:
                m_cfg = json.load(fh)
            m_key = m_cfg.get("key", p.stem)
            models[m_key] = normalize_model_entry(m_key, m_cfg, targets, p.parent)
        for p in sorted(models_dir.glob("*.csv")):
            m_key = p.stem
            if m_key not in models:
                m_cfg = {
                    "name": f"{m_key} (Custom CSV Model)",
                    "raw_csv": p.name,
                }
                models[m_key] = normalize_model_entry(
                    m_key, m_cfg, targets, p.parent
                )

    return targets, models


# =============================================================================
# 2. MINIMUM-PHASE CEPSTRAL DSP & WAV/CSV EXPORTERS
# =============================================================================

def bessel_i0(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 9.8.1 / 9.8.2 polynomial approximation of I0(x)."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    y = np.zeros_like(ax)
    idx1 = ax < 3.75
    t1 = (ax[idx1] / 3.75) ** 2
    y[idx1] = 1.0 + t1 * (
        3.5156229
        + t1
        * (
            3.0899424
            + t1
            * (1.2067492 + t1 * (0.2659732 + t1 * (0.0360768 + t1 * 0.0045813)))
        )
    )
    idx2 = ~idx1
    t2 = 3.75 / ax[idx2]
    y[idx2] = (np.exp(ax[idx2]) / np.sqrt(ax[idx2])) * (
        0.39894228
        + t2
        * (
            0.01328592
            + t2
            * (
                0.00225319
                + t2
                * (
                    -0.00157565
                    + t2
                    * (
                        0.00916281
                        + t2
                        * (
                            -0.02057706
                            + t2
                            * (
                                0.02635537
                                + t2 * (-0.01647633 + t2 * 0.00392377)
                            )
                        )
                    )
                )
            )
        )
    )
    return y


def write_wav_ieee_float(
    filename: str, data: np.ndarray, fs: int = 48000, bits: int = 64
) -> None:
    """Zero-dependency RIFF/WAVE writer for IEEE Float (Format Tag 3: 32/64-bit)."""
    arr = np.asarray(data, dtype=np.float64 if bits == 64 else np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_frames, n_channels = arr.shape
    bytes_per_sample = bits // 8
    block_align = n_channels * bytes_per_sample
    byte_rate = fs * block_align
    raw_bytes = np.ascontiguousarray(arr).tobytes()
    data_size = len(raw_bytes)

    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        3,  # WAVE_FORMAT_IEEE_FLOAT
        n_channels,
        fs,
        byte_rate,
        block_align,
        bits,
    )
    fact_chunk = struct.pack("<4sII", b"fact", 4, n_frames)
    data_header = struct.pack("<4sI", b"data", data_size)
    riff_size = 4 + len(fmt_chunk) + len(fact_chunk) + len(data_header) + data_size
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")

    with open(filename, "wb") as f:
        f.write(riff_header)
        f.write(fmt_chunk)
        f.write(fact_chunk)
        f.write(data_header)
        f.write(raw_bytes)


def apply_notch_mode(
    ctrl_pts: List[Tuple[float, float]],
    mode: str = "safe",
    safe_cap_db: float = 1.50,
    notch_range_hz: Optional[Tuple[float, float]] = (2800.0, 4400.0),
) -> List[Tuple[float, float]]:
    """
    Handles non-minimum-phase concha/driver reflection notches in the control curve:
      - 'original': Unmodified control curve
      - 'safe': Smoothly soft-caps positive boosts in notch_range_hz via tanh
      - 'zero': Clamps positive boosts in notch_range_hz to 0.0 dB
    """
    if mode == "original" or notch_range_hz is None:
        return list(ctrl_pts)
    f_lo, f_hi = notch_range_hz
    out: List[Tuple[float, float]] = []
    for f, g in ctrl_pts:
        if f_lo <= f <= f_hi and g > 0.0:
            if mode == "zero":
                g_new = 0.0
            elif mode == "safe":
                g_new = float(np.round(safe_cap_db * np.tanh(g / safe_cap_db), 2))
            else:
                raise ValueError(f"Unknown notch mode: {mode}")
            out.append((f, g_new))
        else:
            out.append((f, g))
    return out


def synthesize_minimum_phase_ir(
    ctrl_pts: List[Tuple[float, float]],
    fs: int = 48000,
    n_fft: int = 262144,
    n_taps: int = 65536,
    n_tail: int = 8192,
    kaiser_beta: float = 14.0,
    gauss_apodize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Synthesizes a 64-bit minimum-phase FIR filter via real-cepstrum Hilbert folding."""
    f_nodes = np.array([p[0] for p in ctrl_pts], dtype=np.float64)
    g_nodes = np.array([p[1] for p in ctrl_pts], dtype=np.float64)

    nyq = fs / 2.0
    freqs = np.linspace(0.0, nyq, n_fft // 2 + 1, dtype=np.float64)
    mag_db = np.zeros_like(freqs)

    pos = freqs > 0.0
    f_pos = freqs[pos]

    low_mask = f_pos < 20.0
    mid_mask = (f_pos >= 20.0) & (f_pos <= f_nodes[-1])
    high_mask = f_pos > f_nodes[-1]

    g_pos = np.zeros_like(f_pos)
    g_pos[low_mask] = g_nodes[0]
    g_pos[mid_mask] = pchip_log_interp(f_nodes, g_nodes, f_pos[mid_mask])
    g_pos[high_mask] = g_nodes[-1]

    if gauss_apodize and nyq > 20200.0:
        u_mask = f_pos > 20200.0
        g_pos[u_mask] *= np.exp(-0.5 * ((f_pos[u_mask] - 20200.0) / 850.0) ** 2)

    mag_db[pos] = g_pos
    mag_db[0] = g_nodes[0]

    peak_boost_db = float(np.max(mag_db))
    mag_db_norm = mag_db - peak_boost_db
    mag_lin = np.maximum(10.0 ** (mag_db_norm / 20.0), 1e-12)

    log_mag_full = np.concatenate([np.log(mag_lin), np.log(mag_lin[-2:0:-1])])
    cepstrum = np.fft.ifft(log_mag_full).real

    win = np.zeros(n_fft, dtype=np.float64)
    win[0] = 1.0
    win[1 : n_fft // 2] = 2.0
    win[n_fft // 2] = 1.0

    analytic_spec = np.fft.fft(cepstrum * win)
    H_min = np.exp(analytic_spec)
    h_full = np.fft.ifft(H_min).real

    ir = h_full[:n_taps].copy()
    if n_tail > 1 and n_tail <= n_taps:
        n_idx = np.arange(n_tail, dtype=np.float64)
        arg = kaiser_beta * np.sqrt(
            np.clip(1.0 - (n_idx / (n_tail - 1)) ** 2, 0.0, 1.0)
        )
        tail_taper = bessel_i0(arg) / bessel_i0(np.array([kaiser_beta]))[0]
        ir[-n_tail:] *= tail_taper

    return ir, freqs, mag_db, mag_db_norm, peak_boost_db, H_min[: n_fft // 2 + 1]


def export_audit_csv(
    model_key: str,
    raw_pts: List[Tuple[float, ...]],
    ctrl_pts: List[Tuple[float, float]],
    target_pts: List[Tuple[float, float]],
    harm_offset: float,
) -> str:
    """Exports a 500-point log-spaced CSV audit table for external verification."""
    f_grid = np.logspace(np.log10(20.0), np.log10(20000.0), 500)
    f_raw = np.array([p[0] for p in raw_pts], dtype=np.float64)
    g_raw = np.array([p[-1] for p in raw_pts], dtype=np.float64)
    f_ctrl = np.array([p[0] for p in ctrl_pts], dtype=np.float64)
    g_ctrl = np.array([p[1] for p in ctrl_pts], dtype=np.float64)
    f_harm = np.array([p[0] for p in target_pts], dtype=np.float64)
    g_harm = np.array([p[1] for p in target_pts], dtype=np.float64) + harm_offset

    raw_i = pchip_log_interp(f_raw, g_raw, f_grid)
    eq_i = pchip_log_interp(f_ctrl, g_ctrl, f_grid)
    harm_i = pchip_log_interp(f_harm, g_harm, f_grid)
    post_i = raw_i + eq_i

    csv_name = f"{model_key}_audit_data.csv"
    with open(csv_name, "w", encoding="utf-8") as f:
        f.write(
            "freq_hz,raw_dBr,target_harman2018_dBr,eq_filter_dB,predicted_post_eq_dBr,residual_error_dB\n"
        )
        for fr, rw, hm, eq, po in zip(f_grid, raw_i, harm_i, eq_i, post_i):
            f.write(
                f"{fr:.2f},{rw:.3f},{hm:.3f},{eq:.3f},{po:.3f},{po - hm:.3f}\n"
            )
    return csv_name


# =============================================================================
# 3. FOUR-PANEL ACOUSTIC VERIFICATION PLOTTER
# =============================================================================

def render_verification_plot(
    model_cfg: dict,
    ctrl_pts: List[Tuple[float, float]],
    ir: np.ndarray,
    freqs: np.ndarray,
    mag_db: np.ndarray,
    mag_db_norm: np.ndarray,
    peak_boost_db: float,
    H_pos: np.ndarray,
    fs: int,
    notch_mode: str,
    out_png: str,
) -> None:
    """Renders the 4-panel dark-themed acoustic verification plot."""
    raw_pts = model_cfg["raw_pts"]
    target_pts = model_cfg["target_pts"]
    harm_offset = model_cfg["harm_offset"]
    hf_bypass_hz = model_cfg["hf_bypass_hz"]

    f_plot = np.logspace(np.log10(20.0), np.log10(20000.0), 1200)
    f_raw = np.array([p[0] for p in raw_pts], dtype=np.float64)
    g_raw_avg = np.array([p[-1] for p in raw_pts], dtype=np.float64)
    f_tar = np.array([p[0] for p in target_pts], dtype=np.float64)
    g_tar = np.array([p[1] for p in target_pts], dtype=np.float64) + harm_offset
    f_ctrl = np.array([p[0] for p in ctrl_pts], dtype=np.float64)
    g_ctrl = np.array([p[1] for p in ctrl_pts], dtype=np.float64)

    raw_avg_i = pchip_log_interp(f_raw, g_raw_avg, f_plot)
    tar_i = pchip_log_interp(f_tar, g_tar, f_plot)
    eq_i = pchip_log_interp(f_ctrl, g_ctrl, f_plot)
    post_i = raw_avg_i + eq_i

    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "text.color": "#e6edf3",
            "grid.color": "#21262d",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), dpi=160)
    fig.suptitle(
        f"HyprEQ Minimum-Phase FIR Verification — {model_cfg['name']}\n"
        f"Rig: {model_cfg['rig_info']} | fs={fs} Hz | Taps={len(ir):,} | "
        f"Preamp=-{peak_boost_db:.2f} dB | Notch Mode={notch_mode}",
        fontsize=13,
        fontweight="bold",
        color="#f0f6fc",
        y=0.97,
    )

    def style_freq_axis(ax: plt.Axes) -> None:
        ax.set_xscale("log")
        ax.set_xlim(20.0, 20000.0)
        ticks = [20, 30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000, 20000]
        labels = [
            "20",
            "30",
            "50",
            "100",
            "200",
            "500",
            "1k",
            "2k",
            "4k",
            "8k",
            "10k",
            "20k",
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.8)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)

    # Panel 1: Raw vs Target vs Post-EQ
    ax1 = axes[0, 0]
    style_freq_axis(ax1)
    if len(raw_pts[0]) >= 4:
        g_left = np.array([p[1] for p in raw_pts], dtype=np.float64)
        g_right = np.array([p[2] for p in raw_pts], dtype=np.float64)
        ax1.plot(
            f_plot,
            pchip_log_interp(f_raw, g_left, f_plot),
            color="#58a6ff",
            alpha=0.28,
            linewidth=1.0,
            label="Raw Left (dBr)",
        )
        ax1.plot(
            f_plot,
            pchip_log_interp(f_raw, g_right, f_plot),
            color="#f78166",
            alpha=0.28,
            linewidth=1.0,
            label="Raw Right (dBr)",
        )
    ax1.plot(
        f_plot,
        raw_avg_i,
        color="#ff7b72",
        linewidth=1.8,
        linestyle="--",
        label="Measured Raw Avg (dBr)",
    )
    ax1.plot(
        f_plot,
        tar_i,
        color="#8b949e",
        linewidth=2.0,
        linestyle=":",
        label=f"Harman 2018 Target ({harm_offset:+.2f} dB)",
    )
    ax1.plot(
        f_plot,
        post_i,
        color="#3fb950",
        linewidth=2.3,
        label="Predicted Post-EQ Response",
    )
    ax1.axvspan(
        hf_bypass_hz,
        20000.0,
        color="#d29922",
        alpha=0.10,
        label=f">{hf_bypass_hz/1000:.1f} kHz Coupler Null Bypass",
    )
    ax1.set_title("1. Acoustic Response: Raw vs. Harman 2018 vs. Post-EQ", fontsize=11)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Relative SPL (dBr)")
    ax1.set_ylim(-20.0, 14.0)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.85)

    # Panel 2: Filter Gain & Headroom Normalization
    ax2 = axes[0, 1]
    style_freq_axis(ax2)
    ax2.axhline(0.0, color="#8b949e", linewidth=0.9, linestyle="--")
    if model_cfg.get("autoeq_geq_str"):
        geq_pairs = [
            x.strip().split()
            for x in model_cfg["autoeq_geq_str"].split(";")
            if x.strip()
        ]
        f_geq = np.array([float(p[0]) for p in geq_pairs], dtype=np.float64)
        g_geq = np.array([float(p[1]) for p in geq_pairs], dtype=np.float64)
        g_geq_1k = pchip_log_interp(f_geq, g_geq, np.array([1000.0]))[0]
        g_ctrl_1k = pchip_log_interp(f_ctrl, g_ctrl, np.array([1000.0]))[0]
        ax2.plot(
            f_plot,
            pchip_log_interp(f_geq, g_geq - g_geq_1k + g_ctrl_1k, f_plot),
            color="#8b949e",
            linewidth=1.3,
            linestyle=":",
            alpha=0.75,
            label="Uncompensated AutoEq GEQ (Ref)",
        )
    ax2.plot(
        f_plot,
        eq_i,
        color="#58a6ff",
        linewidth=2.2,
        label="HyprEQ Target Curve (dB)",
    )
    ax2.plot(
        f_plot,
        eq_i - peak_boost_db,
        color="#d2a8ff",
        linewidth=1.5,
        linestyle="-.",
        label=f"Normalized FIR Gain (-{peak_boost_db:.2f} dB Preamp)",
    )
    if model_cfg.get("notch_range_hz"):
        n_lo, n_hi = model_cfg["notch_range_hz"]
        ax2.axvspan(
            n_lo,
            n_hi,
            color="#a371f7",
            alpha=0.12,
            label=f"Concha Notch Zone ({n_lo/1000:.1f}–{n_hi/1000:.1f} kHz)",
        )
    ax2.axvspan(hf_bypass_hz, 20000.0, color="#d29922", alpha=0.10)
    ax2.set_title("2. Minimum-Phase FIR Transfer Function", fontsize=11)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Filter Gain (dB)")
    ax2.set_ylim(-12.0, 8.0)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.85)

    # Panel 3: Minimum-Phase Angle & Group Delay
    ax3 = axes[1, 0]
    style_freq_axis(ax3)
    valid = (freqs >= 20.0) & (freqs <= 20000.0)
    f_v = freqs[valid]
    phase_deg = np.degrees(np.unwrap(np.angle(H_pos[valid])))
    ax3.axhline(0.0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax3.plot(
        f_v,
        phase_deg,
        color="#ffa657",
        linewidth=1.9,
        label="Minimum-Phase Angle (deg)",
    )
    ax3.set_title("3. Hilbert-Derived Minimum-Phase Response", fontsize=11)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Phase Angle (degrees)")
    ax3.set_ylim(-90.0, 90.0)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

    # Panel 4: Time-Domain Impulse Decay (dBFS)
    ax4 = axes[1, 1]
    n_show = min(len(ir), int(0.080 * fs))
    t_ms = (np.arange(n_show, dtype=np.float64) / fs) * 1000.0
    ir_db = 20.0 * np.log10(np.maximum(np.abs(ir[:n_show]), 1e-16))
    ax4.plot(
        t_ms,
        ir_db,
        color="#79c0ff",
        linewidth=1.1,
        label="|h[n]| Envelope (20*log10)",
    )
    ax4.set_title("4. Causal Minimum-Phase Impulse Response Decay (First 80 ms)", fontsize=11)
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Amplitude (dBFS)")
    ax4.set_xlim(0.0, 80.0)
    ax4.set_ylim(-160.0, 5.0)
    ax4.grid(True, linestyle="-", linewidth=0.7, alpha=0.8)
    ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


# =============================================================================
# 4. PIPELINE RUNNER & CLI ENTRYPOINT
# =============================================================================

def run_model(
    model_cfg: dict,
    fs: int,
    n_taps: int,
    notch_mode: str,
    export_csv: bool,
    copy_default_ir: bool,
) -> None:
    model_key = model_cfg["key"]
    prefix = model_cfg["file_prefix"]
    rate_tag = format_rate_tag(fs)

    ctrl_pts = apply_notch_mode(
        model_cfg["ctrl_pts"],
        mode=notch_mode,
        safe_cap_db=model_cfg["safe_cap_db"],
        notch_range_hz=model_cfg["notch_range_hz"],
    )

    n_fft = max(262144, 1 << (n_taps * 4 - 1).bit_length())
    n_tail = min(8192, n_taps // 8)

    ir, freqs, mag_db, mag_db_norm, peak_boost_db, H_pos = (
        synthesize_minimum_phase_ir(
            ctrl_pts=ctrl_pts,
            fs=fs,
            n_fft=n_fft,
            n_taps=n_taps,
            n_tail=n_tail,
            kaiser_beta=14.0,
            gauss_apodize=model_cfg["gauss_apodize"],
        )
    )

    f64_wav = f"{prefix}_f64_{rate_tag}.wav"
    write_wav_ieee_float(f64_wav, ir, fs=fs, bits=64)
    if copy_default_ir:
        shutil.copyfile(f64_wav, "ir.wav")

    master_taps = min(16384, n_taps)
    ir_master, *_ = synthesize_minimum_phase_ir(
        ctrl_pts=ctrl_pts,
        fs=fs,
        n_fft=n_fft,
        n_taps=master_taps,
        n_tail=min(2048, master_taps // 8),
        kaiser_beta=14.0,
        gauss_apodize=model_cfg["gauss_apodize"],
    )
    mono_f32_wav = f"{prefix}_mastering_mono_{rate_tag}.wav"
    stereo_f32_wav = f"{prefix}_mastering_stereo_{rate_tag}.wav"
    write_wav_ieee_float(mono_f32_wav, ir_master, fs=fs, bits=32)
    write_wav_ieee_float(
        stereo_f32_wav, np.column_stack([ir_master, ir_master]), fs=fs, bits=32
    )

    out_png = f"{prefix}_final_verification.png"
    render_verification_plot(
        model_cfg=model_cfg,
        ctrl_pts=ctrl_pts,
        ir=ir,
        freqs=freqs,
        mag_db=mag_db,
        mag_db_norm=mag_db_norm,
        peak_boost_db=peak_boost_db,
        H_pos=H_pos,
        fs=fs,
        notch_mode=notch_mode,
        out_png=out_png,
    )

    csv_info = ""
    if export_csv:
        csv_name = export_audit_csv(
            model_key=model_key,
            raw_pts=model_cfg["raw_pts"],
            ctrl_pts=ctrl_pts,
            target_pts=model_cfg["target_pts"],
            harm_offset=model_cfg["harm_offset"],
        )
        csv_info = f"\n  Audit CSV        : {csv_name}"

    print("=" * 78)
    print(f"Model              : {model_cfg['name']} [{model_key}]")
    print(f"Measurement Rig    : {model_cfg['rig_info']}")
    print(f"Sample Rate / Taps : {fs} Hz / {n_taps:,} taps (FFT N={n_fft:,})")
    print(f"Peak Boost / Preamp: +{peak_boost_db:.2f} dB (Normalized by -{peak_boost_db:.2f} dB)")
    print(f"64-bit Float WAV   : {f64_wav}" + (" (+ copied to ir.wav)" if copy_default_ir else ""))
    print(f"32-bit Mastering   : {mono_f32_wav}, {stereo_f32_wav}")
    print(f"Verification Plot  : {out_png}{csv_info}")
    print("-" * 78)
    print("# Suggested mpv.conf block:")
    print(f"audio-samplerate={fs}")
    print(
        f"af=lavfi=[aformat=sample_fmts=dblp,"
        f"aresample={fs}:filter_size=256:phase_shift=12:cutoff=0.98:filter_type=kaiser:kaiser_beta=16:osf=dblp[in];"
        f"amovie=/etc/mpv/{f64_wav},aformat=sample_fmts=dblp[ir];"
        f"[in][ir]afir=irfmt=mono:irnorm=-1:wet=1:minp=1024:maxp=8192:precision=double]"
    )
    print("=" * 78)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    pre_parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    pre_args, _ = pre_parser.parse_known_args()

    _, models = load_measurement_database(
        data_file=pre_args.data_file, models_dir=pre_args.models_dir
    )
    valid_choices = ["all"] + list(models.keys()) + list(MODEL_ALIASES.keys())

    parser = argparse.ArgumentParser(
        description="HyprEQ — Minimum-Phase FIR Headphone EQ & Verification Suite"
    )
    parser.add_argument(
        "--headphone",
        type=str,
        default="all",
        choices=valid_choices,
        help=f"Headphone model key to synthesize (available: {', '.join(['all'] + list(models.keys()))})",
    )
    parser.add_argument(
        "--notch-mode",
        type=str,
        default="safe",
        choices=["safe", "original", "zero"],
        help="Concha reflection notch handling mode (default: safe)",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=48000,
        help="Target sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=65536,
        help="Number of FIR taps for the 64-bit IR (default: 65536)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export 500-point log-spaced CSV audit tables",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Path to external JSON measurement database (default: measurements.json)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Optional directory containing additional per-model .json or .csv files",
    )

    args = parser.parse_args()
    selected = MODEL_ALIASES.get(args.headphone, args.headphone)

    if selected == "all":
        for m_cfg in models.values():
            run_model(
                model_cfg=m_cfg,
                fs=args.fs,
                n_taps=args.taps,
                notch_mode=args.notch_mode,
                export_csv=args.export_csv,
                copy_default_ir=False,
            )
    else:
        run_model(
            model_cfg=models[selected],
            fs=args.fs,
            n_taps=args.taps,
            notch_mode=args.notch_mode,
            export_csv=args.export_csv,
            copy_default_ir=True,
        )


if __name__ == "__main__":
    main()
