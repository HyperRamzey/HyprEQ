#!/usr/bin/env python3
"""
hypreq.py
Unified Minimum-Phase Convolver Generator & 4-Panel Verification Suite

All target and headphone measurement data lives outside this script in
`measurements.json` (and optionally individual `.json` / `.csv` files inside
`models/`) so users can add new headphones without editing Python code.

Supported Headphones (in measurements.json):
  1. jbl770nc : JBL Live 770NC (Passive / Wired Mode)
                - Raw Measurement : Filk (Clone GRAS 43AG / KB006x pinna + IEC 60318-4 711 coupler), via AutoEq.
                - Target Curve    : Harman Over-Ear 2018 (Harman International).
  2. dt770_80 : Beyerdynamic DT 770 Pro (80 Ohm)
                - Raw Measurement : Crinacle (Genuine GRAS 43AG-7: RA0402 High-Frequency 711 Coupler +
                                    KB500X Anthropometric Pinna), via AutoEq.
                - Target Curve    : Harman Over-Ear 2018 (Harman International).
  3. ft1pro   : FiiO FT1 Pro (Open-Back Planar Magnetic)
                - Raw Measurement : oratory1990 / Kuulokenurkka (Genuine GRAS 43AG: KB50xx Anthropometric Pinna +
                                    IEC 60318-4 711 Coupler), via AutoEq.
                - Target Curve    : Harman Over-Ear 2018 (Harman International).

Usage Examples:
  python3 hypreq.py --headphone all --export-csv
  python3 hypreq.py --headphone ft1pro --notch-mode safe
  python3 hypreq.py --headphone dt770_80 --notch-mode safe --taps 16384
  python3 hypreq.py --headphone jbl770nc --fs 44100
  python3 hypreq.py --headphone jbl770nc --fs 96000 --taps 131072
  python3 hypreq.py --upgrade-json
  python3 hypreq.py --headphone dt770_80 --input song.wav --output song_eq_limited.wav
  python3 hypreq.py --measure-wav song.wav
"""

import argparse
import csv
from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ==============================================================================
# 1. EXTERNAL MEASUREMENT DATA LOADER & BUILT-IN PLOT METADATA DEFAULTS
# ==============================================================================

DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "measurements.json"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"

MODEL_ALIASES = {
    "fiio_ft1_pro": "ft1pro",
    "ft1_pro": "ft1pro",
    "dt770": "dt770_80",
    "dt770_80ohm": "dt770_80",
    "jbl_770nc": "jbl770nc",
}

DEFAULT_MODEL_METADATA: Dict[str, dict] = {
    "jbl770nc": {
        "short_title": "JBL Live 770NC (Passive)",
        "notch_ref_hz": None,
        "autoeq_offset_db": 0.0,
        "gauss_apodize": False,
        "plot_meta": {
            "title_template": "VERIFICATION: JBL Live 770NC (Passive) Convolver (Preamp: {pre_rt:.2f} dB | Unwindowed Floor: {unwin_rt:.1f} dBFS)",
            "bypass_label": "Uncorrected >9.1 kHz Coupler Cylinder Null Zone (Clamped to 0.0 dB)",
            "raw_label": "Raw JBL Live 770NC (Passive on Filk KB006x + IEC 711 Coupler)",
            "autoeq_ax1_label": "Standard AutoEq (+1.5dB accidental boost on 6.88k/8.48kHz spikes & +2.7dB 20Hz boost)",
            "wav_ax1_label": "Final WAV Convolver (128-Node PCHIP + Breakup Pole Cuts)",
            "raw_err_ax2_label": "Raw Unsmoothed Error (+8.75dB non-min-phase concha dip @ 5.37kHz)",
            "autoeq_ax2_label": "Standard AutoEq Curve (2.0 oct treble blur boosts 6.88k/8.48kHz peaks)",
            "ax1_ylim": [-15.0, 9.0],
            "ax1_yticks": [-15, -10, -5, 0, 5],
            "ax2_ylim": [-11.0, 10.0],
            "ax2_yticks": [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0],
            "annotations": [
                {"text": "1. KB006x Stiff-Pinna Leak Fix\n(+0.12dB @ 20Hz vs AutoEq +2.7dB)", "xy": [20.5, 0.12], "xytext": [22.0, 5.8], "color": "cyan"},
                {"text": "2. Coupled Passive ANC Cup Dipole\n(-7.04dB @ 160Hz, +2.60dB @ 455Hz)", "xy": [156.9, -7.03], "xytext": [95.0, 5.2], "color": "cyan"},
                {"text": "", "xy": [454.9, 2.60], "xytext": [275.0, 5.2], "color": "cyan"},
                {"text": "3. 1.05kHz Nasal Cut & Ear-Gain\n(-2.45dB @ 1.05kHz; +5.00dB @ 4.06kHz)", "xy": [1049.4, -2.45], "xytext": [480.0, 3.2], "color": "cyan"},
                {"text": "4. 5.37kHz Concha Notch Shelf Bridge\n(+3.70dB flat bridge across 4.8-5.5kHz;\nignores narrow +8.75dB null spike)", "xy": [5365.9, 3.70], "xytext": [1450.0, 7.3], "color": "green"},
                {"text": "5. 6.88kHz & 8.48kHz Breakup Poles\n(-2.31dB & -3.16dB Surgical Cuts)", "xy": [6881.4, -2.31], "xytext": [3200.0, -8.5], "color": "green"},
                {"text": "", "xy": [8480.6, -3.16], "xytext": [6200.0, -6.8], "color": "green"},
                {"text": "6. >9.1kHz Uncorrected Coupler Null Zone\n(0.0dB wire: ignores 9.75k & 11.2kHz nulls)", "xy": [10347.9, 0.0], "xytext": [5900.0, 4.8], "color": "cyan"},
            ],
        },
    },
    "dt770_80": {
        "short_title": "Beyerdynamic DT 770 Pro (80 Ω)",
        "notch_ref_hz": 3703.0,
        "autoeq_offset_db": 4.55,
        "gauss_apodize": False,
        "plot_meta": {
            "title_template": "VERIFICATION: Beyerdynamic DT 770 Pro (80 Ω) Convolver (Notch Mode: {notch_mode} | Preamp: {pre_rt:.2f} dB)",
            "bypass_label": "Uncorrected >8.8 kHz Coupler Null & >15.8 kHz 0.0 dB Wire Zone (Left Unmodified)",
            "lr_label": "Raw L / R Channels (L vs R: -2.5dB 20Hz Pad Leak & 6.6dB 3.84kHz Concha Null Shift)",
            "raw_label": "Raw Beyerdynamic DT 770 Pro (80 Ω on Crinacle GRAS 43AG-7: RA0402 + KB500X)",
            "autoeq_ax1_label": "Standard AutoEq (GraphicEQ.txt: +3.7dB un-cut 8.2kHz Beyer spike & +1.5dB 20Hz boost)",
            "wav_ax1_label": "Final WAV Convolver (Notch Mode: {notch_mode} | Unwindowed Floor: {unwin_rt:.1f} dBFS @ 16k)",
            "raw_err_ax2_label": "Raw Unsmoothed Error (+14.3dB non-min-phase concha null @ 3.78kHz)",
            "autoeq_ax2_label": "Standard AutoEq Curve (2.0 oct treble blur erases 8.2kHz cut + +1.5dB 20Hz boost)",
            "ax1_ylim": [-14.5, 11.5],
            "ax1_yticks": [-10, -5, 0, 5, 10],
            "ax2_ylim": [-10.8, 10.5],
            "ax2_yticks": [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0],
            "annotations": [
                {"text": "1. EDT 770 V Velour Pad-Leak Fix\n(+0.20dB @ 20Hz vs AutoEq +1.5dB)", "xy": [20.5, 0.12], "xytext": [21.5, 7.5], "color": "cyan"},
                {"text": "2. Coupled Port & Cup Suckout\n(-4.05dB @ 59Hz, +4.10dB @ 205Hz)", "xy": [59.0, -4.05], "xytext": [82.0, 5.2], "color": "cyan"},
                {"text": "", "xy": [204.9, 4.10], "xytext": [168.0, 5.5], "color": "cyan"},
                {"text": "3. 940Hz Cup & 2.35kHz Trim\n(-1.38dB @ 940Hz & -1.43dB @ 2.35kHz)", "xy": [940.0, -1.38], "xytext": [420.0, 5.8], "color": "cyan"},
                {"text": "", "xy": [2347.0, -1.43], "xytext": [1120.0, 5.4], "color": "cyan"},
                {"text": "4. 3.76kHz Non-Min-Phase Concha Zero\n({notch_db:+.2f}dB in '{notch_mode}' mode;\navoids L/R & head-to-head glare)", "xy": [3703.0, "notch_db"], "xytext": [1500.0, 7.3], "color": "green"},
                {"text": "5. 5.3k, 6.6k & 8.16kHz Beyer Breakup Poles\n(-3.59dB, -4.45dB & -5.04dB Surgical Cuts)", "xy": [8155.0, -5.04], "xytext": [2250.0, -9.2], "color": "green"},
                {"text": "", "xy": [6644.0, -4.45], "xytext": [4500.0, -7.2], "color": "green"},
                {"text": "", "xy": [5337.0, -3.59], "xytext": [3900.0, -7.2], "color": "green"},
                {"text": "6. >8.8kHz Uncorrected Null & HF Wire\n(Bypasses 9.2kHz null; 0.0dB >15.8kHz\nleave coupler response unmodified)", "xy": [11490.0, -1.55], "xytext": [5300.0, 4.0], "color": "cyan"},
            ],
        },
    },
    "ft1pro": {
        "short_title": "FiiO FT1 Pro (Open-Back Planar)",
        "notch_ref_hz": 3839.0,
        "autoeq_offset_db": 4.30,
        "gauss_apodize": False,
        "plot_meta": {
            "title_template": "VERIFICATION: FiiO FT1 Pro (Open-Back Planar) Convolver (Notch Mode: {notch_mode} | Preamp: {pre_rt:.2f} dB)",
            "bypass_label": "Uncorrected >9.75 kHz Coupler Null & 0.0 dB Wire Zone (Left Unmodified)",
            "lr_label": "Raw L / R Channels (L vs R: 35Hz Front-Volume Leak Resonance & 4.05dB 3.84kHz Concha Split)",
            "raw_label": "Raw FiiO FT1 Pro (Open-Back Planar on GRAS 43AG: KB50xx Pinna + IEC 711 Coupler)",
            "autoeq_ax1_label": "Standard AutoEq (GraphicEQ.txt: +3.7dB 20Hz sub-bass boost & un-damped 35Hz planar fs bump)",
            "wav_ax1_label": "Final WAV Convolver (Notch Mode: {notch_mode} | Unwindowed Floor: {unwin_rt:.1f} dBFS @ 16k)",
            "raw_err_ax2_label": "Raw Unsmoothed Error (+4.1dB 1.82kHz ear-gain scoop & +6.2dB 9.75kHz coupler null)",
            "autoeq_ax2_label": "Standard AutoEq Curve (+3.7dB 20Hz boost & high-frequency shelf tail)",
            "ax1_ylim": [-14.5, 12.5],
            "ax1_yticks": [-10, -5, 0, 5, 10],
            "ax2_ylim": [-10.8, 10.5],
            "ax2_yticks": [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0],
            "annotations": [
                {"text": "1. 1μm Planar Excursion & 35Hz fs Fix\n(+2.37dB @ 20Hz, +1.77dB @ 35Hz vs AutoEq +3.7dB)", "xy": [35.0, 1.77], "xytext": [21.5, 7.3], "color": "cyan"},
                {"text": "", "xy": [20.5, 2.37], "xytext": [24.5, 7.1], "color": "cyan"},
                {"text": "2. Planar Bass-to-Mid Transition\n(+2.70dB @ 55Hz shelf; -1.31dB @ 292Hz trim)", "xy": [55.2, 2.70], "xytext": [75.0, 4.6], "color": "cyan"},
                {"text": "", "xy": [292.0, -1.31], "xytext": [175.0, 4.4], "color": "cyan"},
                {"text": "3. 1.82kHz Ear-Gain & 2.82kHz Glare\n(+4.07dB @ 1.82kHz; -2.64dB @ 2.82kHz)", "xy": [1818.0, 4.07], "xytext": [410.0, 6.0], "color": "cyan"},
                {"text": "", "xy": [2815.0, -2.64], "xytext": [1080.0, 5.6], "color": "cyan"},
                {"text": "4. 3.84kHz L/R-Asymmetric Concha Notch\n({notch_db:+.2f}dB in '{notch_mode}' mode;\navoids 4.05dB L/R split glare)", "xy": [3839.0, "notch_db"], "xytext": [1650.0, 7.3], "color": "green"},
                {"text": "5. 5.75kHz Planar Treble Mode Cut\n(-2.25dB Surgical Cut)", "xy": [5750.0, -2.25], "xytext": [2800.0, -8.5], "color": "green"},
                {"text": "6. >9.75kHz Uncorrected Coupler Null Zone\n(0.0dB wire: ignores 9.75k & 12.8kHz nulls)", "xy": [11300.0, 0.0], "xytext": [5500.0, 4.4], "color": "cyan"},
            ],
        },
    },
}


def format_rate_tag(fs: int) -> str:
    """Formats 48000 -> '48k', 44100 -> '44.1k', 96000 -> '96k' (never truncates 44.1k to 44k)."""
    return f"{fs / 1000.0:g}k"


def pchip_log_interp(
    f_nodes: np.ndarray, g_nodes: np.ndarray, f_query: np.ndarray
) -> np.ndarray:
    """Pure-NumPy Fritsch-Carlson Monotonic Cubic Hermite (PCHIP) interpolator in log10(f) space."""
    f_arr = np.asarray(f_nodes, dtype=np.float64)
    y = np.asarray(g_nodes, dtype=np.float64)
    if len(f_arr) == 1:
        return np.full_like(np.asarray(f_query, dtype=np.float64), y[0], dtype=np.float64)

    x = np.log10(f_arr)
    xq = np.log10(np.clip(np.asarray(f_query, dtype=np.float64), f_arr[0], f_arr[-1]))

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
    return h00 * y[idx] + h10 * hi * d[idx] + h01 * y[idx + 1] + h11 * hi * d[idx + 1]


def sort_and_dedup_points(pts: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    """Sorts measurement or control points by ascending frequency and deduplicates identical frequencies."""
    seen: Dict[float, Tuple[float, ...]] = {}
    for pt in pts:
        f = float(pt[0])
        if f > 0.0:
            seen[f] = tuple(float(v) for v in pt)
    return [seen[f] for f in sorted(seen.keys())]


def normalize_raw_near_1khz(
    raw_pts: List[Tuple[float, ...]], ref_hz: float = 1000.0
) -> List[Tuple[float, ...]]:
    """Shifts raw measurement curve(s) so average SPL at 1 kHz equals 0.00 dBr (preserves L/R delta)."""
    f_raw = np.array([p[0] for p in raw_pts], dtype=np.float64)
    g_avg = np.array([p[-1] for p in raw_pts], dtype=np.float64)
    offset = float(pchip_log_interp(f_raw, g_avg, np.array([ref_hz]))[0])
    if abs(offset) < 1e-9:
        return raw_pts
    return [(pt[0], *(float(v - offset) for v in pt[1:])) for pt in raw_pts]


def derive_ctrl_pts_from_raw(
    raw_pts: List[Tuple[float, ...]],
    target_pts: List[Tuple[float, float]],
    harm_offset: float = 0.0,
    hf_taper_start_hz: float = 8500.0,
    hf_bypass_hz: float = 9500.0,
    subbass_leak_discount_db: float = 0.0,
) -> List[Tuple[float, float]]:
    """Automatically derives a rig-compensated HyprEQ control curve from `raw_pts`."""
    f_raw = np.array([p[0] for p in raw_pts], dtype=np.float64)
    g_raw = np.array([p[-1] for p in raw_pts], dtype=np.float64)
    f_tar = np.array([p[0] for p in target_pts], dtype=np.float64)
    g_tar = np.array([p[1] for p in target_pts], dtype=np.float64) + harm_offset

    g_tar_on_raw = pchip_log_interp(f_tar, g_tar, f_raw)
    err = g_tar_on_raw - g_raw

    ctrl: List[Tuple[float, float]] = []
    for f, e in zip(f_raw, err):
        if f < 65.0 and subbass_leak_discount_db > 0.0:
            disc = subbass_leak_discount_db * np.exp(-((np.log2(max(f, 20.0) / 20.0)) / 0.75) ** 2)
            val = e - disc
        elif f <= hf_taper_start_hz:
            val = e
        elif f < hf_bypass_hz:
            w = 0.5 * (1.0 + np.cos(np.pi * np.log2(f / hf_taper_start_hz) / np.log2(hf_bypass_hz / hf_taper_start_hz)))
            val = e * w
        else:
            val = 0.0
        ctrl.append((float(f), float(np.round(val, 2))))
    return ctrl


def load_csv_points(csv_path: Path) -> List[Tuple[float, ...]]:
    """Loads a 2-column (freq, avg) or 4-column (freq, L, R, avg) CSV file."""
    pts: List[Tuple[float, ...]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
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
    defaults = DEFAULT_MODEL_METADATA.get(key, {})
    target_name = cfg.get("target", "harman_2018")
    if target_name not in targets:
        raise ValueError(f"Model '{key}' references unknown target '{target_name}'")
    target_pts = sort_and_dedup_points(targets[target_name])  # type: ignore[assignment]

    if "raw_csv" in cfg and not cfg.get("raw_pts"):
        raw_pts = load_csv_points(base_dir / cfg["raw_csv"])
    else:
        raw_pts = [tuple(float(x) for x in pt) for pt in cfg["raw_pts"]]
    raw_pts = sort_and_dedup_points(raw_pts)

    has_manual_ctrl = bool(("ctrl_csv" in cfg and not cfg.get("ctrl_pts")) or cfg.get("ctrl_pts"))
    should_norm_1k = cfg.get("normalize_1khz", not has_manual_ctrl)
    if should_norm_1k:
        raw_pts = normalize_raw_near_1khz(raw_pts, ref_hz=1000.0)

    harm_offset = float(cfg.get("harm_offset", 0.0))
    hf_bypass_hz = float(cfg.get("hf_bypass_hz", 9500.0))
    hf_taper_start_hz = float(cfg.get("hf_taper_start_hz", hf_bypass_hz * 0.88))

    if "ctrl_csv" in cfg and not cfg.get("ctrl_pts"):
        ctrl_pts = [(p[0], p[-1]) for p in sort_and_dedup_points(load_csv_points(base_dir / cfg["ctrl_csv"]))]
    elif cfg.get("ctrl_pts"):
        ctrl_pts = [(p[0], p[1]) for p in sort_and_dedup_points([(float(p[0]), float(p[1])) for p in cfg["ctrl_pts"]])]
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

    notch_ref_hz = cfg.get("notch_ref_hz", defaults.get("notch_ref_hz"))
    if notch_ref_hz is not None:
        notch_ref_hz = float(notch_ref_hz)

    plot_meta = dict(defaults.get("plot_meta", {}))
    plot_meta.update(cfg.get("plot_meta", {}))
    if cfg.get("annotations") and "annotations" not in cfg.get("plot_meta", {}):
        plot_meta["annotations"] = cfg["annotations"]

    return {
        "key": key,
        "name": cfg.get("name", key),
        "short_title": cfg.get("short_title", defaults.get("short_title", cfg.get("name", key))),
        "file_prefix": cfg.get("file_prefix", key),
        "rig_info": cfg.get("rig_info", "IEC 60318-4 (711) Coupler"),
        "target_name": target_name,
        "target_pts": target_pts,
        "harm_offset": harm_offset,
        "gauss_apodize": bool(cfg.get("gauss_apodize", defaults.get("gauss_apodize", False))),
        "hf_bypass_hz": hf_bypass_hz,
        "notch_range_hz": notch_range,
        "notch_ref_hz": notch_ref_hz,
        "safe_cap_db": float(cfg.get("safe_cap_db", 1.50)),
        "autoeq_geq_str": cfg.get("autoeq_geq_str"),
        "autoeq_offset_db": cfg.get("autoeq_offset_db", defaults.get("autoeq_offset_db")),
        "plot_meta": plot_meta,
        "raw_pts": raw_pts,
        "ctrl_pts": ctrl_pts,
    }


def load_measurement_database(
    data_file: Path = DEFAULT_DATA_FILE,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, dict]]:
    """Loads targets and headphone models from `measurements.json` and `models/`."""
    if not data_file.exists():
        raise FileNotFoundError(
            f"Measurement database not found at '{data_file}'. "
            "Place 'measurements.json' next to hypreq.py or pass --data-file."
        )

    with data_file.open("r", encoding="utf-8") as fh:
        raw_db = json.load(fh)

    targets: Dict[str, List[Tuple[float, float]]] = {
        t_key: [(float(p[0]), float(p[1])) for p in t_pts]
        for t_key, t_pts in raw_db.get("targets", {}).items()
    }

    models: Dict[str, dict] = {
        m_key: normalize_model_entry(m_key, m_cfg, targets, data_file.parent)
        for m_key, m_cfg in raw_db.get("models", {}).items()
    }

    if models_dir.exists() and models_dir.is_dir():
        for p in sorted(models_dir.glob("*.json")):
            with p.open("r", encoding="utf-8") as fh:
                m_cfg = json.load(fh)
            m_key = m_cfg.get("key", p.stem)
            models[m_key] = normalize_model_entry(m_key, m_cfg, targets, p.parent)
        for p in sorted(models_dir.glob("*.csv")):
            m_key = p.stem
            if m_key not in models:
                m_cfg = {"name": f"{m_key} (Custom CSV Model)", "raw_csv": p.name}
                models[m_key] = normalize_model_entry(m_key, m_cfg, targets, p.parent)

    return targets, models


def upgrade_measurements_json(data_file: Path = DEFAULT_DATA_FILE) -> None:
    """Writes built-in `plot_meta` and `annotations` directly into `measurements.json`."""
    if not data_file.exists():
        raise FileNotFoundError(f"Measurement database not found at '{data_file}'.")
    with data_file.open("r", encoding="utf-8") as fh:
        raw_db = json.load(fh)
    for m_key, m_cfg in raw_db.get("models", {}).items():
        if m_key in DEFAULT_MODEL_METADATA:
            defs = DEFAULT_MODEL_METADATA[m_key]
            m_cfg.setdefault("short_title", defs["short_title"])
            m_cfg.setdefault("notch_ref_hz", defs["notch_ref_hz"])
            m_cfg.setdefault("autoeq_offset_db", defs["autoeq_offset_db"])
            m_cfg.setdefault("gauss_apodize", defs.get("gauss_apodize", False))
            pm = dict(defs["plot_meta"])
            pm.update(m_cfg.get("plot_meta", {}))
            m_cfg["plot_meta"] = pm
    with data_file.open("w", encoding="utf-8") as fh:
        json.dump(raw_db, fh, indent=2, ensure_ascii=False)
    print(f"Upgraded '{data_file}' with plot_meta and callout annotations.")


# ==============================================================================
# 2. SHARED DSP ENGINE (BESSEL I0, WAV I/O, HILBERT-CEPSTRAL, CONVOLVER)
# ==============================================================================

def bessel_i0(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Exact 64-bit float power series for Modified Bessel Function I0(x) (zero scipy dependency)."""
    if np.ndim(x) == 0:
        xf = float(x)
        q = xf * xf / 4.0
        term = 1.0
        s = 1.0
        k = 1.0
        while term > 1e-18 * s:
            term *= q / (k * k)
            s += term
            k += 1.0
        return s

    arr = np.asarray(x, dtype=np.float64)
    res = np.ones_like(arr)
    term = np.ones_like(arr)
    half_x_sq = (0.5 * arr) ** 2
    for k in range(1, 65):
        term *= half_x_sq / (k * k)
        res += term
        if np.all(term <= 1e-18 * res):
            break
    return res


def write_wav_ieee_float(
    filename: str, data: np.ndarray, fs: int = 48000, bits: int = 64
) -> None:
    """Writes an IEEE 754 64-bit float (`pcm_f64le`) or 32-bit float (`pcm_f32le`) WAV file."""
    dtype_str = "<f8" if bits == 64 else "<f4"
    bytes_per_sample = 8 if bits == 64 else 4
    if data.ndim == 1:
        channels = 1
        raw_bytes = data.astype(dtype_str).tobytes()
    else:
        channels = data.shape[1]
        raw_bytes = np.ascontiguousarray(data, dtype_str).tobytes()

    byte_rate = fs * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    data_size = len(raw_bytes)
    riff_size = 36 + data_size
    if riff_size > 0xFFFFFFFF:
        raise ValueError(f"WAV output '{filename}' exceeds 4 GiB RIFF limit ({riff_size} bytes)")

    with open(filename, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, channels, fs, byte_rate, block_align, bits))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(raw_bytes)


def read_wav(filename: str) -> Tuple[int, List[np.ndarray], str]:
    """
    Reads a RIFF/WAVE file into normalized float64 planar channels in [-1.0, 1.0].
    Supports PCM (8, 16, 24, 32-bit), IEEE float (32, 64-bit), and WAVE_FORMAT_EXTENSIBLE.
    """
    with open(filename, "rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"'{filename}' is not a valid RIFF/WAVE file")

        fmt_tag = None
        channels = 0
        fs = 0
        bits = 0
        subformat_tag = None
        data_bytes = None

        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            chunk_id, chunk_size = hdr[:4], struct.unpack("<I", hdr[4:8])[0]
            payload = f.read(chunk_size)
            if len(payload) < chunk_size:
                raise ValueError(f"Truncated '{chunk_id.decode('ascii', 'replace')}' chunk in '{filename}'")
            if chunk_size & 1:
                f.read(1)

            if chunk_id == b"fmt ":
                if chunk_size < 16:
                    raise ValueError(f"Invalid fmt chunk size ({chunk_size}) in '{filename}'")
                fmt_tag, channels, fs, _, _, bits = struct.unpack("<HHIIHH", payload[:16])
                if fmt_tag == 0xFFFE and chunk_size >= 40:
                    subformat_tag = struct.unpack("<H", payload[24:26])[0]
            elif chunk_id == b"data":
                data_bytes = payload

        if fmt_tag is None or data_bytes is None:
            raise ValueError(f"Missing fmt or data chunk in '{filename}'")
        eff_tag = subformat_tag if fmt_tag == 0xFFFE and subformat_tag is not None else fmt_tag
        if channels <= 0 or fs <= 0:
            raise ValueError(f"Invalid WAV channel/rate metadata in '{filename}'")

        if eff_tag == 3:
            if bits == 32:
                inter = np.frombuffer(data_bytes, dtype="<f4").astype(np.float64)
                fmt_desc = f"pcm_f32le | {fs:,} Hz | {channels}ch"
            elif bits == 64:
                inter = np.frombuffer(data_bytes, dtype="<f8").copy()
                fmt_desc = f"pcm_f64le | {fs:,} Hz | {channels}ch"
            else:
                raise ValueError(f"Unsupported IEEE float bit depth {bits} in '{filename}'")
        elif eff_tag == 1:
            if bits == 8:
                raw = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float64)
                inter = (raw - 128.0) / 128.0
                fmt_desc = f"pcm_u8 | {fs:,} Hz | {channels}ch"
            elif bits == 16:
                inter = np.frombuffer(data_bytes, dtype="<i2").astype(np.float64) / 32768.0
                fmt_desc = f"pcm_s16le | {fs:,} Hz | {channels}ch"
            elif bits == 24:
                raw_u8 = np.frombuffer(data_bytes, dtype=np.uint8)
                n_samp = raw_u8.size // 3
                b = raw_u8[: n_samp * 3].reshape(-1, 3)
                vals = (
                    b[:, 0].astype(np.int32)
                    | (b[:, 1].astype(np.int32) << 8)
                    | (b[:, 2].astype(np.int32) << 16)
                )
                vals = np.where(vals >= 0x800000, vals - 0x1000000, vals)
                inter = vals.astype(np.float64) / 8388608.0
                fmt_desc = f"pcm_s24le | {fs:,} Hz | {channels}ch"
            elif bits == 32:
                inter = np.frombuffer(data_bytes, dtype="<i4").astype(np.float64) / 2147483648.0
                fmt_desc = f"pcm_s32le | {fs:,} Hz | {channels}ch"
            else:
                raise ValueError(f"Unsupported PCM bit depth {bits} in '{filename}'")
        else:
            raise ValueError(f"Unsupported WAV format tag 0x{eff_tag:04x} in '{filename}'")

        frames = inter.size // channels
        if frames == 0:
            return fs, [np.empty(0, dtype=np.float64) for _ in range(channels)], fmt_desc
        mat = inter[: frames * channels].reshape(frames, channels)
        return fs, [np.ascontiguousarray(mat[:, c], dtype=np.float64) for c in range(channels)], fmt_desc


def write_wav_pcm_or_float(
    filename: str, planar: List[np.ndarray], fs: int = 48000, bits: int = 32
) -> None:
    """Writes planar float64 audio to WAV in 16/24-bit TPDF-dithered PCM or 32/64-bit IEEE float."""
    if not planar:
        raise ValueError("Cannot write empty channel list")
    data = planar[0] if len(planar) == 1 else np.column_stack(planar)
    if bits in (32, 64):
        write_wav_ieee_float(filename, data, fs=fs, bits=bits)
        return

    channels = len(planar)
    frames = len(planar[0])
    mat = np.nan_to_num(data[:, None] if channels == 1 else data, nan=0.0, posinf=1.0, neginf=-1.0)
    rng = np.random.default_rng(0x66726D65)
    tpdf = rng.uniform(-0.5, 0.5, size=mat.shape) + rng.uniform(-0.5, 0.5, size=mat.shape)

    if bits == 16:
        scale = 32768.0
        q = np.clip(np.rint(mat * scale + tpdf), -32768.0, 32767.0).astype("<i2")
        raw_bytes = np.ascontiguousarray(q).tobytes()
        bytes_per_sample = 2
    elif bits == 24:
        scale = 8388608.0
        q = np.clip(np.rint(mat * scale + tpdf), -8388608.0, 8388607.0).astype(np.int32).ravel()
        packed = np.empty((q.size, 3), dtype=np.uint8)
        packed[:, 0] = q & 0xFF
        packed[:, 1] = (q >> 8) & 0xFF
        packed[:, 2] = (q >> 16) & 0xFF
        raw_bytes = packed.tobytes()
        bytes_per_sample = 3
    else:
        raise ValueError(f"Unsupported output WAV bit depth: {bits} (choose 16, 24, 32, or 64)")

    byte_rate = fs * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    data_size = frames * block_align
    riff_size = 36 + data_size
    if riff_size > 0xFFFFFFFF:
        raise ValueError(f"WAV output '{filename}' exceeds 4 GiB RIFF limit ({riff_size} bytes)")

    with open(filename, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, channels, fs, byte_rate, block_align, bits))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(raw_bytes)


def convolve_planar_fft(
    planar: List[np.ndarray], ir: np.ndarray, trim_to_input: bool = False
) -> List[np.ndarray]:
    """Exact 64-bit overlap-add FFT linear convolution of planar audio against a mono FIR."""
    if not planar:
        return []
    ir_f64 = np.asarray(ir, dtype=np.float64)
    n_ir = ir_f64.shape[0]
    n_in = planar[0].shape[0]
    if n_in == 0:
        return [np.empty(0, dtype=np.float64) for _ in planar]

    block_len = max(16384, min(65536, n_in))
    n_fft = 1 << (block_len + n_ir - 2).bit_length()
    H = np.fft.rfft(ir_f64, n=n_fft)
    out_len = n_in if trim_to_input else (n_in + n_ir - 1)
    results: List[np.ndarray] = []

    for ch in planar:
        acc = np.zeros(n_in + n_ir - 1, dtype=np.float64)
        for start in range(0, n_in, block_len):
            blk = ch[start : start + block_len]
            seg_len = blk.shape[0] + n_ir - 1
            y_blk = np.fft.irfft(np.fft.rfft(blk, n=n_fft) * H, n=n_fft)[:seg_len]
            acc[start : start + seg_len] += y_blk
        results.append(acc[:out_len].copy() if trim_to_input else acc)
    return results


def apply_notch_mode(
    ctrl_pts: List[Tuple[float, float]],
    mode: str = "safe",
    safe_cap_db: float = 1.50,
    notch_range_hz: Optional[Tuple[float, float]] = (2800.0, 4400.0),
) -> List[Tuple[float, float]]:
    """
    Adjusts non-minimum-phase concha/driver reflection notch regions:
      - 'original': keeps the regularized fill as-is.
      - 'safe'    : smoothly caps positive boosts inside notch_range_hz to safe_cap_db via tanh.
      - 'zero'    : clamps positive boosts inside notch_range_hz to 0.00 dB.
    """
    if mode == "original" or notch_range_hz is None:
        return list(ctrl_pts)
    f_lo, f_hi = notch_range_hz
    out: List[Tuple[float, float]] = []
    for f, g in ctrl_pts:
        if f_lo <= f <= f_hi and g > 0.0:
            if mode == "zero":
                out.append((float(f), 0.0))
            elif mode == "safe":
                g_soft = safe_cap_db * np.tanh(g / safe_cap_db)
                out.append((float(f), float(np.round(g_soft, 4))))
            else:
                raise ValueError(f"Unknown notch mode: {mode}")
        else:
            out.append((float(f), float(g)))
    return out


def synthesize_minimum_phase_ir(
    ctrl_pts: List[Tuple[float, float]],
    fs: int = 48000,
    n_fft: int = 262144,
    n_taps: int = 65536,
    n_tail: int = 8192,
    headroom_db: float = -1.50,
    gauss_apodize: bool = False,
    kaiser_beta: float = 18.0,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    """Synthesizes a 64-bit minimum-phase FIR filter via Hilbert-Cepstral folding."""
    f_nodes = np.array([p[0] for p in ctrl_pts], dtype=np.float64)
    g_nodes = np.array([p[1] for p in ctrl_pts], dtype=np.float64)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    f_safe = np.clip(freqs, f_nodes[0], f_nodes[-1])
    target_db = pchip_log_interp(f_nodes, g_nodes, f_safe)
    target_db[0] = g_nodes[0]

    if gauss_apodize:
        apod_mask = freqs > 20200.0
        target_db[apod_mask] += -3.0103 * ((freqs[apod_mask] - 20200.0) / 1650.0) ** 2

    max_boost_db = float(np.max(target_db))
    preamp_db = headroom_db - max_boost_db
    gain_db_norm = target_db + preamp_db

    mag_lin = 10.0 ** (gain_db_norm / 20.0)
    log_mag = np.log(np.maximum(mag_lin, 1e-18))
    log_spec_full = np.concatenate([log_mag, log_mag[-2:0:-1]])
    cepstrum = np.fft.ifft(log_spec_full).real

    win_cep = np.zeros(n_fft, dtype=np.float64)
    win_cep[0] = 1.0
    win_cep[1 : n_fft // 2] = 2.0
    win_cep[n_fft // 2] = 1.0

    min_phase_spec = np.exp(np.fft.fft(cepstrum * win_cep))
    ir_full_f64 = np.fft.ifft(min_phase_spec).real

    ir_f64 = ir_full_f64[:n_taps].copy()
    unwindowed_floor_dbfs = float(20.0 * np.log10(np.max(np.abs(ir_f64[-min(128, n_taps):])) + 1e-25))

    if n_tail > 0 and n_tail <= n_taps:
        t_norm = np.arange(n_tail, dtype=np.float64) / n_tail
        kaiser_tail = bessel_i0(kaiser_beta * np.sqrt(np.maximum(0.0, 1.0 - t_norm**2))) / bessel_i0(kaiser_beta)
        ir_f64[-n_tail:] *= kaiser_tail

    windowed_floor_dbfs = float(20.0 * np.log10(np.max(np.abs(ir_f64[-min(100, n_taps):])) + 1e-25))
    energy_pct = float(np.sum(ir_f64**2) / np.sum(ir_full_f64**2) * 100.0)
    return ir_f64, mag_lin, max_boost_db, preamp_db, energy_pct, unwindowed_floor_dbfs, windowed_floor_dbfs


def export_audit_csv(
    model_key: str,
    raw_pts: List[Tuple[float, ...]],
    harm_pts: List[Tuple[float, float]],
    harm_offset: float,
    ctrl_pts: List[Tuple[float, float]],
) -> str:
    """Exports an auditable CSV table containing Frequency, Raw Measurement, Harman Target, Raw Error, and Applied EQ."""
    csv_file = f"{model_key}_audit_data.csv"
    f_ctrl = np.array([p[0] for p in ctrl_pts], dtype=np.float64)
    g_ctrl = np.array([p[1] for p in ctrl_pts], dtype=np.float64)
    f_raw = np.array([p[0] for p in raw_pts], dtype=np.float64)
    g_raw = np.array([p[-1] for p in raw_pts], dtype=np.float64)
    raw_at_ctrl = pchip_log_interp(f_raw, g_raw, f_ctrl)
    f_h = np.array([p[0] for p in harm_pts], dtype=np.float64)
    g_h = np.array([p[1] for p in harm_pts], dtype=np.float64) + harm_offset
    harm_at_ctrl = pchip_log_interp(f_h, g_h, f_ctrl)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "raw_dBr", "harman_target_dBr", "raw_error_dB", "applied_eq_dB"])
        for freq, r_val, h_val, eq_val in zip(f_ctrl, raw_at_ctrl, harm_at_ctrl, g_ctrl):
            writer.writerow([f"{freq:.1f}", f"{r_val:.2f}", f"{h_val:.2f}", f"{h_val - r_val:.2f}", f"{eq_val:.2f}"])
    return csv_file


# ==============================================================================
# 3. TRUE-PEAK MEASUREMENT ENGINE (ITU-R BS.1770-5 / x2..x32) & METERS
# ==============================================================================

DBL_MIN_POS: float = math.ldexp(1.0, -1022)
F32_MAX: float = float(np.finfo(np.float32).max)

ITU_TAPS: int = 12
ITU_PHASES: np.ndarray = np.array(
    [
        # Phase 0
        [
            0.001_708_984_375_00,
            0.010_986_328_125_00,
            -0.019_653_320_312_50,
            0.033_203_125_000_00,
            -0.059_448_242_187_50,
            0.137_329_101_562_50,
            0.972_167_968_750_00,
            -0.102_294_921_875_00,
            0.047_607_421_875_00,
            -0.026_611_328_125_00,
            0.014_892_578_125_00,
            -0.008_300_781_250_00,
        ],
        # Phase 1
        [
            -0.029_174_804_687_50,
            0.029_296_875_000_00,
            -0.051_757_812_500_00,
            0.089_111_328_125_00,
            -0.166_503_906_250_00,
            0.465_087_890_625_00,
            0.779_785_156_250_00,
            -0.200_317_382_812_50,
            0.101_562_500_000_00,
            -0.058_227_539_062_50,
            0.033_081_054_687_50,
            -0.018_920_898_437_50,
        ],
        # Phase 2
        [
            -0.018_920_898_437_50,
            0.033_081_054_687_50,
            -0.058_227_539_062_50,
            0.101_562_500_000_00,
            -0.200_317_382_812_50,
            0.779_785_156_250_00,
            0.465_087_890_625_00,
            -0.166_503_906_250_00,
            0.089_111_328_125_00,
            -0.051_757_812_500_00,
            0.029_296_875_000_00,
            -0.029_174_804_687_50,
        ],
        # Phase 3
        [
            -0.008_300_781_250_00,
            0.014_892_578_125_00,
            -0.026_611_328_125_00,
            0.047_607_421_875_00,
            -0.102_294_921_875_00,
            0.972_167_968_750_00,
            0.137_329_101_562_50,
            -0.059_448_242_187_50,
            0.033_203_125_000_00,
            -0.019_653_320_312_50,
            0.010_986_328_125_00,
            0.001_708_984_375_00,
        ],
    ],
    dtype=np.float64,
)

BAND_EDGE: float = 0.45
STOPBAND_DB: float = 80.0


def kaiser_beta(att_db: float) -> float:
    """Kaiser window shape parameter for `att_db` stop-band attenuation."""
    if att_db > 50.0:
        return 0.1102 * (att_db - 8.7)
    if att_db >= 21.0:
        return 0.5842 * ((att_db - 21.0) ** 0.4) + 0.07886 * (att_db - 21.0)
    return 0.0


def derived_taps(_upsample: int) -> int:
    """Taps per phase of a derived Kaiser polyphase bank (51 for 80 dB / 0.45 fs)."""
    return int(math.ceil((STOPBAND_DB - 7.95) / (14.36 * (1.0 - 2.0 * BAND_EDGE))))


def derived_branches(upsample: int, taps: int) -> np.ndarray:
    """Builds a centred, symmetric Kaiser-windowed sinc polyphase bank `(upsample, taps)`."""
    l_f = float(upsample)
    centre = (taps - 1) / 2.0
    half = centre + (l_f - 1.0) / (2.0 * l_f)
    beta = kaiser_beta(STOPBAND_DB)
    i0_beta = float(bessel_i0(beta))
    branches = np.empty((upsample, taps), dtype=np.float64)
    for p in range(upsample):
        offset = (p - (l_f - 1.0) / 2.0) / l_f
        for j in range(taps):
            d = j - centre + offset
            s = 1.0 if d == 0.0 else math.sin(math.pi * d) / (math.pi * d)
            r = d / half
            w = float(bessel_i0(beta * math.sqrt(max(0.0, 1.0 - r * r)))) / i0_beta
            branches[p, j] = s * w
    return branches


def db_from_peak(p: float) -> float:
    """True peak in dBTP (`-inf` for silence)."""
    return 20.0 * math.log10(p) if p > 0.0 else float("-inf")


class Ratio(Enum):
    """Over-sampling ratios supported by the true-peak detector & limiter."""

    X2 = "x2"
    X4 = "x4"
    X8 = "x8"
    X16 = "x16"
    X32 = "x32"

    @classmethod
    def all(cls) -> List["Ratio"]:
        return [cls.X2, cls.X4, cls.X8, cls.X16, cls.X32]

    @classmethod
    def from_str(cls, s: str) -> "Ratio":
        key = s.strip().lower()
        mapping = {
            "2": cls.X2,
            "x2": cls.X2,
            "4": cls.X4,
            "x4": cls.X4,
            "8": cls.X8,
            "x8": cls.X8,
            "16": cls.X16,
            "x16": cls.X16,
            "32": cls.X32,
            "x32": cls.X32,
        }
        if key not in mapping:
            raise ValueError(f"over-sampling ratio must be 2, 4, 8, 16 or 32 (got {s!r})")
        return mapping[key]

    @property
    def n(self) -> int:
        return {"x2": 2, "x4": 4, "x8": 8, "x16": 16, "x32": 32}[self.value]

    @property
    def label(self) -> str:
        return self.value

    @property
    def taps(self) -> int:
        return ITU_TAPS if self == Ratio.X4 else derived_taps(self.n)

    def mac_per_sample(self) -> int:
        return self.n * self.taps

    def branches(self) -> np.ndarray:
        if self == Ratio.X4:
            return ITU_PHASES.copy()
        return derived_branches(self.n, self.taps)

    def group_delay(self) -> float:
        t = float(self.taps)
        n = float(self.n)
        if self == Ratio.X4:
            return (t - 1.0) / 2.0 + (n - 1.0) / (2.0 * n)
        return (t - 1.0) / 2.0

    def worst_underread_db(self) -> float:
        if self == Ratio.X2:
            return 0.30
        if self == Ratio.X4:
            return 0.60
        return 0.02

    def fullband_underread_db(self) -> float:
        if self == Ratio.X2:
            return 1.00
        if self == Ratio.X4:
            return 1.50
        return 0.90

    def worst_overread_db(self) -> float:
        return 0.25 if self == Ratio.X4 else 0.10

    def refines(self) -> bool:
        return self != Ratio.X4

    def report_delay(self) -> int:
        return int(math.ceil(self.group_delay())) + (1 if self.refines() else 0)

    def min_lookahead_samples(self) -> int:
        return self.report_delay()

    def bound_db(self, f_norm: float) -> float:
        return 20.0 * math.log10(math.cos(math.pi * f_norm / float(self.n)))

    def bound_table(self) -> Tuple[float, float]:
        return (self.bound_db(0.45), self.bound_db(0.5))

    def chain(self) -> str:
        return {
            Ratio.X2: "2-phase Kaiser polyphase (51 taps/phase) + peak refinement",
            Ratio.X4: "ITU-R BS.1770-5 4-phase, printed coefficients",
            Ratio.X8: "8-phase Kaiser polyphase (51 taps/phase) + peak refinement",
            Ratio.X16: "16-phase Kaiser polyphase (51 taps/phase) + peak refinement",
            Ratio.X32: "32-phase Kaiser polyphase (51 taps/phase) + peak refinement",
        }[self]

    def __str__(self) -> str:
        return self.label


_TP_BRANCH_CACHE: Dict[Ratio, np.ndarray] = {}


def _get_tp_branches(ratio: Ratio) -> np.ndarray:
    cached = _TP_BRANCH_CACHE.get(ratio)
    if cached is None:
        cached = ratio.branches()
        cached.setflags(write=False)
        _TP_BRANCH_CACHE[ratio] = cached
    return cached


def vertex(a: float, b: float, c: float) -> float:
    """Parabolic vertex through three adjacent magnitudes when the middle point is a local max."""
    if b >= a and b >= c:
        den = a - 2.0 * b + c
        if den < 0.0:
            return b - (c - a) * (c - a) / (8.0 * den)
    return b


class Stage:
    """One polyphase stage: `phases[p, k]` oldest tap first over a matching shift register."""

    def __init__(self, ratio: Ratio) -> None:
        self.phases = _get_tp_branches(ratio)
        self.n_phases, self.taps = self.phases.shape
        self.hist = np.zeros(self.taps, dtype=np.float64)

    def push(self, x: float) -> Tuple[np.ndarray, float]:
        self.hist[:-1] = self.hist[1:]
        self.hist[-1] = x
        out = self.phases @ self.hist
        return out, float(np.max(np.abs(out)))


class Detector:
    """Streaming true-peak detector for one channel (scalar and chunk-invariant block paths)."""

    def __init__(self, ratio: Ratio = Ratio.X16) -> None:
        self.stage = Stage(ratio)
        self._peak = 0.0
        self._ratio = ratio
        self.lag = int(math.floor(ratio.group_delay() + 0.5))
        self.refine = ratio.refines()
        self.older = 0.0
        self.newest = 0.0

    def peak(self) -> float:
        return max(self._peak, self.newest)

    def dbtp(self) -> float:
        return db_from_peak(self.peak())

    def clear(self) -> None:
        self.stage.hist.fill(0.0)
        self._peak = 0.0
        self.older = 0.0
        self.newest = 0.0

    def reset_peak(self) -> None:
        self._peak = 0.0

    def push(self, x: float) -> float:
        xf = float(x)
        out, raw = self.stage.push(xf)
        if self.refine:
            peak = 0.0
            for c_val in np.abs(out[::-1]):
                c = float(c_val)
                v = vertex(self.older, self.newest, c)
                if v > peak:
                    peak = v
                self.older = self.newest
                self.newest = c
        else:
            peak = raw

        centre = self.stage.taps - 1 - self.lag
        xs = abs(float(self.stage.hist[centre]))
        if xs > peak:
            peak = xs
        if peak > self._peak:
            self._peak = peak
        ax = abs(xf)
        if ax > self._peak:
            self._peak = ax
        return peak

    def push_block_frames(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float64)
        n = x.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float64)

        taps = self.stage.taps
        n_phases = self.stage.n_phases
        padded = np.concatenate([self.stage.hist[1:], x])
        windows = sliding_window_view(padded, taps)
        out = windows @ self.stage.phases.T

        if self.refine:
            c_seq = np.abs(out[:, ::-1]).ravel()
            s = np.empty(c_seq.shape[0] + 2, dtype=np.float64)
            s[0] = self.older
            s[1] = self.newest
            s[2:] = c_seq
            a = s[:-2]
            b = s[1:-1]
            c = s[2:]
            den = a - 2.0 * b + c
            mask = (b >= a) & (b >= c) & (den < 0.0)
            v = np.where(mask, b - (c - a) ** 2 / (8.0 * np.where(mask, den, -1.0)), b)
            frame_peaks = np.max(v.reshape(n, n_phases), axis=1)
            self.older = float(s[-2])
            self.newest = float(s[-1])
        else:
            frame_peaks = np.max(np.abs(out), axis=1)

        centre_offset = taps - 1 - self.lag
        xs = np.abs(padded[centre_offset : centre_offset + n])
        frame_peaks = np.maximum(frame_peaks, xs)
        self.stage.hist[:] = padded[-taps:]
        self._peak = max(self._peak, float(np.max(frame_peaks)), float(np.max(np.abs(x))))
        return frame_peaks

    def push_block(self, block: np.ndarray) -> float:
        self.push_block_frames(block)
        return self.peak()

    def ratio(self) -> Ratio:
        return self._ratio


class TruePeak:
    """Multichannel true-peak meter (reports max across channels)."""

    def __init__(self, ratio: Ratio = Ratio.X16, channels: int = 2) -> None:
        self._ratio = ratio
        self.chans = [Detector(ratio) for _ in range(channels)]

    def push_ch(self, ch: int, block: np.ndarray) -> None:
        if 0 <= ch < len(self.chans):
            self.chans[ch].push_block(block)

    def push_interleaved(self, block: np.ndarray) -> None:
        nch = len(self.chans)
        if nch == 0:
            return
        arr = np.asarray(block, dtype=np.float64)
        usable = (arr.size // nch) * nch
        if usable == 0:
            return
        mat = arr[:usable].reshape(-1, nch)
        for c in range(nch):
            self.chans[c].push_block(mat[:, c])

    def drain(self) -> None:
        n = int(math.ceil(self._ratio.group_delay())) + 1
        zeros = np.zeros(n, dtype=np.float64)
        for d in self.chans:
            d.push_block(zeros)

    def peak(self) -> float:
        return max((d.peak() for d in self.chans), default=0.0)

    def dbtp(self) -> float:
        return db_from_peak(self.peak())

    def per_channel_dbtp(self) -> List[float]:
        return [d.dbtp() for d in self.chans]

    def reset(self) -> None:
        for d in self.chans:
            d.reset_peak()

    def ratio(self) -> Ratio:
        return self._ratio

    def min_lookahead_samples(self) -> int:
        return self._ratio.min_lookahead_samples()


# --- Peak Meters (ported from g6me meter.rs) ---
METER_FALL_S: float = 0.30
IN_SLOTS: List[int] = [0, 1, 2, 3, 4, 5, 9, 10]
IN_CHANNELS: int = 8
OUT_CHANNELS: int = 2
TOTAL_METERS: int = IN_CHANNELS + OUT_CHANNELS
IN_LABELS: List[str] = ["In1 FL", "In2 FR", "In3 FC", "In4 LFE", "In5 BL", "In6 BR", "In7 SL", "In8 SR"]
OUT_LABELS: List[str] = ["OutL", "OutR"]


_FLOOR_F32: float = float(np.float32(0.001))


def level_to_param(v: float) -> float:
    """Maps linear peak `v` onto normalized `-60..0 dB` parameter `[0.0, 1.0]` (exact -60 dB floor & NaN-safe)."""
    vf = float(v)
    if math.isnan(vf):
        return 0.0
    v32 = float(np.float32(vf))
    if v32 <= _FLOOR_F32:
        return 0.0
    if v32 >= 1.0:
        return 1.0
    db = 20.0 * math.log10(v32)
    return float(np.float32((db + 60.0) / 60.0))


def param_to_db(p: float) -> float:
    """Inverse of `level_to_param` (`0..1` -> `-60..0 dB`)."""
    return float(p) * 60.0 - 60.0


class Meters:
    """
    Instant-attack, 0.30s exponential-release peak-hold meters on the 0.30 s
    exponential-release scale, in the 10-channel slot layout (`In1..In8, OutL, OutR`) or sized
    to any channel count via `Meters(channels=N)` for `observe_planar`.
    """

    def __init__(self, channels: int = TOTAL_METERS) -> None:
        self.values: List[float] = [0.0] * channels
        self._rate: float = 0.0
        self._block: int = 0
        self._frames: int = 0
        self._xruns: int = 0

    def set_format(self, sample_rate: float, block: int) -> None:
        self._rate = float(sample_rate)
        self._block = int(block)

    def decay(self) -> float:
        sr = float(self._rate) if self._rate > 0.0 else 48000.0
        n = float(max(1, self._block))
        return float(np.float32(math.exp(-(n / (METER_FALL_S * sr)))))

    def hold(self, index: int, peak: float) -> None:
        prev = self.values[index]
        fall = float(np.float32(prev * self.decay())) if math.isfinite(prev) else 0.0
        pk = float(np.float32(peak)) if math.isfinite(peak) else F32_MAX
        self.values[index] = pk if pk > fall else fall

    @staticmethod
    def peak_of(buf: np.ndarray, n: int) -> float:
        arr = np.asarray(buf[:n], dtype=np.float64)
        if arr.size == 0:
            return 0.0
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0
        mx = float(np.max(np.abs(finite)))
        return float(np.float32(min(mx, F32_MAX)))

    def observe_block(
        self,
        planar_slots: List[Optional[np.ndarray]],
        l_buf: np.ndarray,
        r_buf: np.ndarray,
        n: int,
    ) -> None:
        for i, slot in enumerate(IN_SLOTS):
            if i < len(self.values) and slot < len(planar_slots) and planar_slots[slot] is not None:
                self.hold(i, self.peak_of(planar_slots[slot], n))  # type: ignore[arg-type]
        if IN_CHANNELS < len(self.values):
            self.hold(IN_CHANNELS, self.peak_of(l_buf, n))
        if IN_CHANNELS + 1 < len(self.values):
            self.hold(IN_CHANNELS + 1, self.peak_of(r_buf, n))
        self._frames += n

    def observe_planar(self, planar: List[np.ndarray], n: Optional[int] = None) -> None:
        if not planar:
            return
        frames = len(planar[0]) if n is None else int(n)
        for c in range(min(len(planar), len(self.values))):
            self.hold(c, self.peak_of(planar[c], frames))
        self._frames += frames

    def decay_once(self) -> None:
        for i in range(len(self.values)):
            self.hold(i, 0.0)

    def peak(self, index: int) -> float:
        return self.values[index] if 0 <= index < len(self.values) else 0.0

    def peaks(self) -> List[float]:
        return list(self.values)

    def param(self, index: int) -> float:
        return level_to_param(self.peak(index))

    def frames(self) -> int:
        return self._frames

    def xruns(self) -> int:
        return self._xruns

    def note_xrun(self, n: int) -> None:
        self._xruns += int(n)

    def reset(self) -> None:
        self.values = [0.0] * len(self.values)
        self._frames = 0
        self._xruns = 0


class MeterTap:
    """Sample tap wrapper around `Meters`, feeding one slot per channel."""

    def __init__(self, meters: Optional[Meters] = None) -> None:
        self._meters = meters if meters is not None else Meters()
        self._rate: int = 48_000

    def meters(self) -> Meters:
        return self._meters

    def rate(self) -> int:
        return self._rate

    def will_render(self, sample_rate: int) -> None:
        if sample_rate > 0:
            self._rate = int(sample_rate)

    def tap(
        self,
        planar_slots: List[Optional[np.ndarray]],
        l_buf: np.ndarray,
        r_buf: np.ndarray,
        n: int,
    ) -> None:
        self._meters.set_format(float(self._rate), n)
        self._meters.observe_block(planar_slots, l_buf, r_buf, n)


@dataclass
class SignalMeasurements:
    channels: int
    frames: int
    fs: int
    duration_s: float
    ratio: Ratio
    sample_peak_lin: float
    sample_peak_dbfs: float
    per_channel_sample_dbfs: List[float]
    true_peak_lin: float
    true_peak_dbtp: float
    per_channel_dbtp: List[float]
    itu_x4_dbtp: float
    itu_x4_per_channel_dbtp: List[float]
    rms_dbfs: float
    per_channel_rms_dbfs: List[float]
    crest_tp_db: float
    crest_sample_db: float
    meter_peak_hold_lin: float
    meter_peak_hold_dbfs: float
    meter_param: float


def measure_signal(
    planar: List[np.ndarray],
    fs: int,
    ratio: Ratio = Ratio.X16,
    block_size: int = 1024,
) -> SignalMeasurements:
    """Runs true-peak (`ratio` + ITU x4) and 0.30s peak-hold meter measurements on planar audio."""
    nch = len(planar)
    frames = len(planar[0]) if nch > 0 else 0
    if nch == 0 or frames == 0:
        return SignalMeasurements(
            channels=nch,
            frames=0,
            fs=fs,
            duration_s=0.0,
            ratio=ratio,
            sample_peak_lin=0.0,
            sample_peak_dbfs=float("-inf"),
            per_channel_sample_dbfs=[],
            true_peak_lin=0.0,
            true_peak_dbtp=float("-inf"),
            per_channel_dbtp=[],
            itu_x4_dbtp=float("-inf"),
            itu_x4_per_channel_dbtp=[],
            rms_dbfs=float("-inf"),
            per_channel_rms_dbfs=[],
            crest_tp_db=0.0,
            crest_sample_db=0.0,
            meter_peak_hold_lin=0.0,
            meter_peak_hold_dbfs=float("-inf"),
            meter_param=0.0,
        )

    tp_main = TruePeak(ratio, nch)
    tp_itu = TruePeak(Ratio.X4, nch)
    meters = Meters(channels=nch)

    sp_ch: List[float] = []
    rms_ch_lin: List[float] = []
    for c, ch in enumerate(planar):
        arr = np.asarray(ch, dtype=np.float64)
        sp_ch.append(float(np.max(np.abs(arr))))
        rms_ch_lin.append(float(np.sqrt(np.mean(arr * arr))))
        tp_main.push_ch(c, arr)
        tp_itu.push_ch(c, arr)

    tp_main.drain()
    tp_itu.drain()

    for start in range(0, frames, block_size):
        end = min(frames, start + block_size)
        blk_len = end - start
        meters.set_format(float(fs), blk_len)
        meters.observe_planar([ch[start:end] for ch in planar], blk_len)

    sp_lin = max(sp_ch)
    sp_db = db_from_peak(sp_lin)
    tp_lin = tp_main.peak()
    tp_db = tp_main.dbtp()
    itu_db = tp_itu.dbtp()
    rms_all = math.sqrt(sum(r * r for r in rms_ch_lin) / nch)
    rms_db = db_from_peak(rms_all)
    hold_lin = max(meters.peaks(), default=0.0)
    hold_db = db_from_peak(hold_lin)
    hold_param = level_to_param(hold_lin)

    return SignalMeasurements(
        channels=nch,
        frames=frames,
        fs=fs,
        duration_s=frames / float(fs),
        ratio=ratio,
        sample_peak_lin=sp_lin,
        sample_peak_dbfs=sp_db,
        per_channel_sample_dbfs=[db_from_peak(v) for v in sp_ch],
        true_peak_lin=tp_lin,
        true_peak_dbtp=tp_db,
        per_channel_dbtp=tp_main.per_channel_dbtp(),
        itu_x4_dbtp=itu_db,
        itu_x4_per_channel_dbtp=tp_itu.per_channel_dbtp(),
        rms_dbfs=rms_db,
        per_channel_rms_dbfs=[db_from_peak(v) for v in rms_ch_lin],
        crest_tp_db=(tp_db - rms_db) if (math.isfinite(tp_db) and math.isfinite(rms_db)) else 0.0,
        crest_sample_db=(sp_db - rms_db) if (math.isfinite(sp_db) and math.isfinite(rms_db)) else 0.0,
        meter_peak_hold_lin=hold_lin,
        meter_peak_hold_dbfs=hold_db,
        meter_param=hold_param,
    )


# ==============================================================================
# 4. TWO-STAGE TRUE-PEAK CEILING LIMITER (DEFAULT X16)
# ==============================================================================

ATTACK_MS: float = 0.5
GAIN_SLEW_GUARD_DB: float = 0.02
MIN_CEILING_DB: float = -24.0
MAX_CEILING_DB: float = 0.0
MIN_RAMP_FRAMES: int = 2
GAIN_DEADBAND: float = 1e-9
ENGAGED_EPS: float = 1e-4
MIN_LOOKAHEAD_MS: float = 0.2
MAX_LOOKAHEAD_MS: float = 2.0
DEFAULT_LOOKAHEAD_MS: float = 2.0
DEFAULT_CEILING_DB: float = -3.0
DEFAULT_LIMITER_RATIO: Ratio = Ratio.X16
MIN_ATTACK_MS: float = 1.0
MAX_ATTACK_MS: float = 200.0
DEFAULT_ATTACK_MS: float = 60.0
MIN_RELEASE_MS: float = 10.0
MAX_RELEASE_MS: float = 5000.0
DEFAULT_RELEASE_MS: float = 900.0
DEFAULT_LINK_TRANSIENT: float = 1.0
DEFAULT_LINK_RELEASE: float = 0.75
RELEASE_WINDOW_MS: float = 5.0
TRANSIENT_RELEASE_MS: float = 25.0


@dataclass
class LimiterCfg:
    """Configuration for the two-stage true-peak ceiling limiter."""

    enable: bool = True
    ceiling_db: float = DEFAULT_CEILING_DB
    ratio: Ratio = DEFAULT_LIMITER_RATIO
    lookahead_ms: float = DEFAULT_LOOKAHEAD_MS
    attack_ms: float = DEFAULT_ATTACK_MS
    release_ms: float = DEFAULT_RELEASE_MS
    link_transient: float = DEFAULT_LINK_TRANSIENT
    link_release: float = DEFAULT_LINK_RELEASE

    def validate(self) -> None:
        if not self.enable:
            return
        if not (math.isfinite(self.ceiling_db) and MIN_CEILING_DB <= self.ceiling_db <= MAX_CEILING_DB):
            raise ValueError(
                f"true-peak ceiling must be between {MIN_CEILING_DB} and {MAX_CEILING_DB} dBTP (got {self.ceiling_db})"
            )
        if not (math.isfinite(self.lookahead_ms) and MIN_LOOKAHEAD_MS <= self.lookahead_ms <= MAX_LOOKAHEAD_MS):
            raise ValueError(
                f"lookahead must be between {MIN_LOOKAHEAD_MS} and {MAX_LOOKAHEAD_MS} ms (got {self.lookahead_ms})"
            )
        for name, v, lo, hi in [
            ("attack", self.attack_ms, MIN_ATTACK_MS, MAX_ATTACK_MS),
            ("release", self.release_ms, MIN_RELEASE_MS, MAX_RELEASE_MS),
        ]:
            if not (math.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi} ms (got {v})")
        for name, v in [("transient link", self.link_transient), ("release link", self.link_release)]:
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(
                    f"{name} must be between 0 and 1 (got {v}; it is a fraction, so 75 % is 0.75)"
                )

    def coef(self, ms: float, rate: int) -> float:
        return math.exp(-1.0 / (ms * 1e-3 * float(rate)))

    def release_window(self, rate: int) -> int:
        return int(round(RELEASE_WINDOW_MS * 1e-3 * float(rate)))

    def requested_lookahead_samples(self, rate: int) -> int:
        return int(round(self.lookahead_ms * 1e-3 * float(rate)))

    def min_lookahead_samples(self) -> int:
        return self.ratio.report_delay() + MIN_RAMP_FRAMES + 1

    def lookahead_samples(self, rate: int) -> int:
        return max(self.requested_lookahead_samples(rate), self.min_lookahead_samples(), 1)

    def guard_db(self) -> float:
        return self.ratio.worst_underread_db() + GAIN_SLEW_GUARD_DB

    def band_caveat_db(self) -> float:
        return max(0.0, self.ratio.fullband_underread_db() - self.ratio.worst_underread_db())

    def ramp_frames(self, rate: int) -> int:
        la = self.lookahead_samples(rate)
        room = max(0, la - self.ratio.report_delay() - 1)
        preferred = int(round(ATTACK_MS * 1e-3 * float(rate)))
        return min(max(room, MIN_RAMP_FRAMES), max(preferred, MIN_RAMP_FRAMES))

    def effective_lookahead_ms(self, rate: int) -> float:
        return float(self.lookahead_samples(rate)) / float(rate) * 1e3

    def lookahead_was_raised(self, rate: int) -> bool:
        return self.lookahead_samples(rate) > self.requested_lookahead_samples(rate)

    def latency_samples(self, rate: int) -> int:
        return self.lookahead_samples(rate) + self.ratio.report_delay()


@dataclass
class LimiterReport:
    pre_dbtp: float = float("-inf")
    post_dbtp: float = float("-inf")
    max_gain_reduction_db: float = 0.0
    ceiling_db: float = 0.0
    guard_db: float = 0.0
    effective_lookahead_ms: float = 0.0
    requested_lookahead_ms: float = 0.0
    latency_ms: float = 0.0
    latency_samples: int = 0
    ramp_frames: int = 0
    attack_ms: float = 0.0
    release_ms: float = 0.0
    link_transient: float = 0.0
    link_release: float = 0.0
    transient_reduction_db: float = 0.0
    release_reduction_db: float = 0.0
    release_frames: int = 0
    ratio: Ratio = DEFAULT_LIMITER_RATIO
    engaged_frames: int = 0
    band_caveat_db: float = 0.0

    def within_ceiling(self) -> bool:
        return self.post_dbtp <= self.ceiling_db


def _blend_gain(a: float, b: float, w: float) -> float:
    la = math.log(a if a > DBL_MIN_POS else DBL_MIN_POS)
    lb = math.log(b if b > DBL_MIN_POS else DBL_MIN_POS)
    return math.exp(la + (lb - la) * w)


class Limiter:
    """
    Streaming two-stage true-peak ceiling limiter (default x16).
    Uses vectorized NumPy Pass 1/2/3 over steady-state sub-blocks while maintaining exact
    chunk-size invariance and bit-level equivalence with the per-frame ring algorithm.
    """

    _SUB_CHUNK = 16384

    def __init__(self, cfg: LimiterCfg, rate: int, channels: int, _enabled: bool = True) -> None:
        self.cfg = cfg
        self.rate = int(rate)
        if not _enabled or not cfg.enable:
            self.cfg = LimiterCfg(
                enable=False,
                ceiling_db=cfg.ceiling_db,
                ratio=cfg.ratio,
                lookahead_ms=cfg.lookahead_ms,
                attack_ms=cfg.attack_ms,
                release_ms=cfg.release_ms,
                link_transient=cfg.link_transient,
                link_release=cfg.link_release,
            )
            self.latency = 0
            self.delta = 0
            self.ramp = 0
            self.side: List[Detector] = []
            self.out_meter = TruePeak(Ratio.X4, 0)
            self._report = LimiterReport()
            return

        if rate <= 0:
            raise ValueError(f"sample rate must be positive (got {rate})")
        cfg.validate()
        nch = max(1, channels)
        delta = cfg.ratio.report_delay()
        latency = cfg.latency_samples(rate)
        ramp = cfg.ramp_frames(rate)
        ring_len = max(16, latency + 2 * delta + ramp + 8)

        self.side = [Detector(cfg.ratio) for _ in range(channels)]
        self.out_meter = TruePeak(cfg.ratio, channels)
        self.lines: List[List[float]] = [[0.0] * max(1, latency) for _ in range(nch)]
        self.pos: int = 0
        self.latency: int = latency
        self.delta: int = delta
        self.ring_len: int = ring_len
        self.req_ring: List[List[float]] = [[1.0] * ring_len for _ in range(nch)]
        self.sched_ring: List[List[float]] = [[1.0] * ring_len for _ in range(nch)]
        self.gr_ring: List[List[float]] = [[1.0] * ring_len for _ in range(nch)]
        self.fed: int = 0
        self.scheduled: int = 0
        self.gain: List[float] = [1.0] * nch
        self.g_rel: List[float] = [1.0] * nch
        self.g_rel_group: float = 1.0
        self.hold: List[float] = [0.0] * nch
        self.hold_left: int = max(1, cfg.release_window(rate))
        self.g_target: List[float] = [1.0] * nch
        self.g_target_group: float = 1.0
        self.gr_now: List[float] = [1.0] * nch
        self.peaks: List[float] = [0.0] * nch
        self.atk_coef: float = cfg.coef(cfg.attack_ms, rate)
        self.slow_rel_coef: float = cfg.coef(cfg.release_ms, rate)
        self.release_window: int = max(1, cfg.release_window(rate))
        self.release_coef: float = cfg.coef(TRANSIENT_RELEASE_MS, rate)
        self.slope: float = 1.0 / float(ramp)
        self.ramp: int = ramp
        self.ceiling_lin: float = 10.0 ** ((cfg.ceiling_db - cfg.guard_db()) / 20.0)
        self._report = LimiterReport(
            pre_dbtp=float("-inf"),
            post_dbtp=float("-inf"),
            max_gain_reduction_db=0.0,
            ceiling_db=cfg.ceiling_db,
            guard_db=cfg.guard_db(),
            effective_lookahead_ms=cfg.effective_lookahead_ms(rate),
            requested_lookahead_ms=cfg.lookahead_ms,
            latency_ms=float(latency) / float(rate) * 1e3,
            latency_samples=latency,
            ramp_frames=ramp,
            attack_ms=cfg.attack_ms,
            release_ms=cfg.release_ms,
            link_transient=cfg.link_transient,
            link_release=cfg.link_release,
            transient_reduction_db=0.0,
            release_reduction_db=0.0,
            release_frames=0,
            ratio=cfg.ratio,
            engaged_frames=0,
            band_caveat_db=cfg.band_caveat_db(),
        )

    @classmethod
    def disabled(cls) -> "Limiter":
        return cls(LimiterCfg(enable=False), 48000, 0, _enabled=False)

    @classmethod
    def enabled(cls, cfg: LimiterCfg, rate: int, channels: int) -> "Limiter":
        return cls(cfg, rate, channels, _enabled=True)

    def is_enabled(self) -> bool:
        return self.cfg.enable

    def latency_samples(self) -> int:
        return self.latency

    def per_channel_post_dbtp(self) -> List[float]:
        return self.out_meter.per_channel_dbtp()

    def report(self) -> LimiterReport:
        r = LimiterReport(**self._report.__dict__)
        r.pre_dbtp = db_from_peak(max((d.peak() for d in self.side), default=0.0))
        r.post_dbtp = self.out_meter.dbtp()
        return r

    def _schedule_to(self, limit: int) -> None:
        if limit <= self.scheduled:
            return
        delta = self.delta
        ring_len = self.ring_len
        for c in range(len(self.req_ring)):
            req_c = self.req_ring[c]
            sched_c = self.sched_ring[c]
            gr_c = self.gr_ring[c]
            for m in range(limit - 1, self.scheduled - 1, -1):
                lo = max(0, m - delta)
                gr_m = max(gr_c[m % ring_len], DBL_MIN_POS)
                ga = 1.0
                for k in range(lo, m + delta + 1):
                    kslot = k % ring_len
                    scale = gr_c[kslot] / gr_m
                    if scale > 1.0:
                        scale = 1.0
                    v = req_c[kslot] * scale
                    if v < ga:
                        ga = v
                sched_c[m % ring_len] = ga
                if ga < 1.0 - ENGAGED_EPS:
                    gr = -20.0 * math.log10(max(ga, DBL_MIN_POS))
                    if gr > self._report.transient_reduction_db:
                        self._report.transient_reduction_db = gr
        self.scheduled = limit

    def _step_with_peaks(self, frame_vals: List[float], frame_peaks: List[float]) -> List[float]:
        nch = len(frame_vals)
        latency = self.latency
        ring_len = self.ring_len
        delta = self.delta

        for c in range(nch):
            self.peaks[c] = frame_peaks[c]
        self.fed += 1

        for c in range(nch):
            if self.peaks[c] > self.hold[c]:
                self.hold[c] = self.peaks[c]
        self.hold_left = max(0, self.hold_left - 1)
        if self.hold_left == 0:
            self.hold_left = max(1, self.release_window)
            aim_db = 20.0 * math.log10(max(self.ceiling_lin, DBL_MIN_POS))
            group = float("-inf")
            for c in range(nch):
                lvl = 20.0 * math.log10(max(self.hold[c], DBL_MIN_POS))
                if lvl > group:
                    group = lvl
                self.g_target[c] = 10.0 ** (min(0.0, aim_db - lvl) / 20.0)
                self.hold[c] = 0.0
            self.g_target_group = 10.0 ** (min(0.0, aim_db - group) / 20.0)

        atk = 1.0 - self.atk_coef
        rel = 1.0 - self.slow_rel_coef
        for c in range(nch):
            t = self.g_target[c]
            g = self.g_rel[c]
            nxt = g + (t - g) * (atk if t < g else rel)
            self.g_rel[c] = t if abs(t - nxt) < GAIN_DEADBAND else nxt

        tg = self.g_target_group
        gg = self.g_rel_group
        nxt = gg + (tg - gg) * (atk if tg < gg else rel)
        self.g_rel_group = tg if abs(tg - nxt) < GAIN_DEADBAND else nxt

        slow_min = 1.0
        link_rel = self.cfg.link_release
        for c in range(nch):
            gr = _blend_gain(self.g_rel[c], self.g_rel_group, link_rel)
            self.gr_now[c] = gr
            if gr < slow_min:
                slow_min = gr
        if slow_min < 1.0 - ENGAGED_EPS:
            self._report.release_frames += 1
            db = -20.0 * math.log10(max(slow_min, DBL_MIN_POS))
            if db > self._report.release_reduction_db:
                self._report.release_reduction_db = db

        m = max(0, self.fed - (1 + delta))
        slot = m % ring_len
        group = 1.0
        reqs = [1.0] * nch
        ceil_lin = self.ceiling_lin
        for c in range(nch):
            after = max(self.peaks[c] * self.gr_now[c], DBL_MIN_POS)
            req = ceil_lin / after if after > ceil_lin else 1.0
            reqs[c] = req
            if req < group:
                group = req
        link_tr = self.cfg.link_transient
        for c in range(nch):
            req = _blend_gain(reqs[c], group, link_tr) if link_tr > 0.0 else reqs[c]
            if m == 0:
                req = min(req, self.req_ring[c][slot])
            self.req_ring[c][slot] = req
            self.gr_ring[c][slot] = self.gr_now[c]

        limit = max(0, self.fed - delta * 2)
        self._schedule_to(limit)

        j = max(0, self.fed - (latency + 1))
        jslot = j % ring_len
        live = self.fed > latency

        ramp = self.ramp
        slope = self.slope
        scheduled = self.scheduled
        for c in range(nch):
            target = 1.0
            if live:
                sched_c = self.sched_ring[c]
                for r in range(ramp + 1):
                    idx = j + r
                    demand = sched_c[idx % ring_len] if idx < scheduled else 1.0
                    v = demand + float(r) * slope
                    if v < target:
                        target = v
            if target < 0.0:
                target = 0.0
            elif target > 1.0:
                target = 1.0
            g = self.gain[c]
            if target < g:
                self.gain[c] = target
            else:
                nxt_g = min(target, g + (1.0 - g) * (1.0 - self.release_coef))
                self.gain[c] = target if (target - nxt_g) < GAIN_DEADBAND else nxt_g

        pos = self.pos
        deepest = 1.0
        out_frame = [0.0] * nch
        for c in range(nch):
            g = self.gain[c] * self.gr_ring[c][jslot]
            line = self.lines[c]
            old = line[pos]
            line[pos] = frame_vals[c]
            out_frame[c] = old * g
            if g < deepest:
                deepest = g

        if deepest < 1.0 - ENGAGED_EPS:
            self._report.engaged_frames += 1
            db = -20.0 * math.log10(max(deepest, DBL_MIN_POS))
            if db > self._report.max_gain_reduction_db:
                self._report.max_gain_reduction_db = db

        self.pos = 0 if pos + 1 >= max(1, latency) else pos + 1
        return out_frame

    def _process_chunk(self, planar: List[np.ndarray]) -> None:
        n = len(planar[0])
        if n == 0:
            return
        nch = min(len(self.side), len(planar))
        if nch == 0:
            return

        peaks_mat = [self.side[c].push_block_frames(planar[c]) for c in range(nch)]
        startup_end = 2 * self.delta + self.latency + self.ramp + 4

        if self.fed < startup_end and n > 128:
            need = min(n, startup_end - self.fed)
            out_cols = [np.empty(need, dtype=np.float64) for _ in range(nch)]
            for i in range(need):
                fvals = [float(planar[c][i]) for c in range(nch)]
                fpeaks = [float(peaks_mat[c][i]) for c in range(nch)]
                out_f = self._step_with_peaks(fvals, fpeaks)
                for c in range(nch):
                    out_cols[c][i] = out_f[c]
            for c in range(nch):
                planar[c][:need] = out_cols[c]
                self.out_meter.push_ch(c, planar[c][:need])
            if need < n:
                self._process_steady_chunk([p[need:] for p in planar], [pk[need:] for pk in peaks_mat])
            return

        if n <= 128 or self.fed < startup_end:
            out_cols = [np.empty(n, dtype=np.float64) for _ in range(nch)]
            for i in range(n):
                fvals = [float(planar[c][i]) for c in range(nch)]
                fpeaks = [float(peaks_mat[c][i]) for c in range(nch)]
                out_f = self._step_with_peaks(fvals, fpeaks)
                for c in range(nch):
                    out_cols[c][i] = out_f[c]
            for c in range(nch):
                planar[c][:] = out_cols[c]
                self.out_meter.push_ch(c, planar[c])
            return

        self._process_steady_chunk(planar, peaks_mat)

    def _process_steady_chunk(self, planar: List[np.ndarray], peaks_mat: List[np.ndarray]) -> None:
        n = len(planar[0])
        if n == 0:
            return
        nch = min(len(self.side), len(planar))
        if n <= 128:
            out_cols = [np.empty(n, dtype=np.float64) for _ in range(nch)]
            for i in range(n):
                fvals = [float(planar[c][i]) for c in range(nch)]
                fpeaks = [float(peaks_mat[c][i]) for c in range(nch)]
                out_f = self._step_with_peaks(fvals, fpeaks)
                for c in range(nch):
                    out_cols[c][i] = out_f[c]
            for c in range(nch):
                planar[c][:] = out_cols[c]
                self.out_meter.push_ch(c, planar[c])
            return

        fed0 = self.fed
        delta = self.delta
        latency = self.latency
        ramp = self.ramp
        ring_len = self.ring_len
        scheduled0 = self.scheduled

        req_base = scheduled0 - delta
        pre_req_len = (fed0 - delta) - req_base

        req_arr = [np.empty(pre_req_len + n, dtype=np.float64) for _ in range(nch)]
        gr_arr = [np.empty(pre_req_len + n, dtype=np.float64) for _ in range(nch)]
        for c in range(nch):
            rc = self.req_ring[c]
            gc = self.gr_ring[c]
            for t in range(pre_req_len):
                slot = (req_base + t) % ring_len
                req_arr[c][t] = rc[slot]
                gr_arr[c][t] = gc[slot]

        atk = 1.0 - self.atk_coef
        rel = 1.0 - self.slow_rel_coef
        aim_db = 20.0 * math.log10(max(self.ceiling_lin, DBL_MIN_POS))
        ceil_lin = self.ceiling_lin
        link_rel = self.cfg.link_release
        link_tr = self.cfg.link_transient
        rel_win = max(1, self.release_window)

        hold = self.hold
        hold_left = self.hold_left
        g_target = self.g_target
        g_target_group = self.g_target_group
        g_rel = self.g_rel
        g_rel_group = self.g_rel_group
        gr_now = self.gr_now
        rel_frames = self._report.release_frames
        rel_db_max = self._report.release_reduction_db

        peaks_lists = [peaks_mat[c].tolist() for c in range(nch)]
        req_lists = [[0.0] * n for _ in range(nch)]
        gr_lists = [[0.0] * n for _ in range(nch)]

        if nch == 1:
            p0 = peaks_lists[0]
            r0 = req_lists[0]
            g0_out = gr_lists[0]
            h0 = hold[0]
            gt0 = g_target[0]
            grl0 = g_rel[0]
            for i in range(n):
                pk = p0[i]
                if pk > h0:
                    h0 = pk
                hold_left = hold_left - 1 if hold_left > 0 else 0
                if hold_left == 0:
                    hold_left = rel_win
                    lvl = 20.0 * math.log10(h0 if h0 > DBL_MIN_POS else DBL_MIN_POS)
                    d = aim_db - lvl
                    gt0 = 10.0 ** ((0.0 if d > 0.0 else d) / 20.0)
                    g_target_group = gt0
                    h0 = 0.0
                nxt = grl0 + (gt0 - grl0) * (atk if gt0 < grl0 else rel)
                grl0 = gt0 if abs(gt0 - nxt) < GAIN_DEADBAND else nxt
                nxtg = g_rel_group + (g_target_group - g_rel_group) * (atk if g_target_group < g_rel_group else rel)
                g_rel_group = g_target_group if abs(g_target_group - nxtg) < GAIN_DEADBAND else nxtg
                gr = _blend_gain(grl0, g_rel_group, link_rel)
                if gr < 1.0 - ENGAGED_EPS:
                    rel_frames += 1
                    db = -20.0 * math.log10(gr if gr > DBL_MIN_POS else DBL_MIN_POS)
                    if db > rel_db_max:
                        rel_db_max = db
                after = pk * gr
                if after < DBL_MIN_POS:
                    after = DBL_MIN_POS
                req = ceil_lin / after if after > ceil_lin else 1.0
                r0[i] = req
                g0_out[i] = gr
            hold[0] = h0
            g_target[0] = gt0
            g_rel[0] = grl0
            gr_now[0] = g0_out[-1]
            self.peaks[0] = p0[-1]
        elif nch == 2:
            p0, p1 = peaks_lists[0], peaks_lists[1]
            r0, r1 = req_lists[0], req_lists[1]
            g0_out, g1_out = gr_lists[0], gr_lists[1]
            h0, h1 = hold[0], hold[1]
            gt0, gt1 = g_target[0], g_target[1]
            grl0, grl1 = g_rel[0], g_rel[1]
            for i in range(n):
                pk0, pk1 = p0[i], p1[i]
                if pk0 > h0:
                    h0 = pk0
                if pk1 > h1:
                    h1 = pk1
                hold_left = hold_left - 1 if hold_left > 0 else 0
                if hold_left == 0:
                    hold_left = rel_win
                    lvl0 = 20.0 * math.log10(h0 if h0 > DBL_MIN_POS else DBL_MIN_POS)
                    lvl1 = 20.0 * math.log10(h1 if h1 > DBL_MIN_POS else DBL_MIN_POS)
                    grp = lvl0 if lvl0 > lvl1 else lvl1
                    d0 = aim_db - lvl0
                    gt0 = 10.0 ** ((0.0 if d0 > 0.0 else d0) / 20.0)
                    d1 = aim_db - lvl1
                    gt1 = 10.0 ** ((0.0 if d1 > 0.0 else d1) / 20.0)
                    dg = aim_db - grp
                    g_target_group = 10.0 ** ((0.0 if dg > 0.0 else dg) / 20.0)
                    h0 = 0.0
                    h1 = 0.0
                nxt0 = grl0 + (gt0 - grl0) * (atk if gt0 < grl0 else rel)
                grl0 = gt0 if abs(gt0 - nxt0) < GAIN_DEADBAND else nxt0
                nxt1 = grl1 + (gt1 - grl1) * (atk if gt1 < grl1 else rel)
                grl1 = gt1 if abs(gt1 - nxt1) < GAIN_DEADBAND else nxt1
                nxtg = g_rel_group + (g_target_group - g_rel_group) * (atk if g_target_group < g_rel_group else rel)
                g_rel_group = g_target_group if abs(g_target_group - nxtg) < GAIN_DEADBAND else nxtg
                gr0 = _blend_gain(grl0, g_rel_group, link_rel)
                gr1 = _blend_gain(grl1, g_rel_group, link_rel)
                smin = gr0 if gr0 < gr1 else gr1
                if smin < 1.0 - ENGAGED_EPS:
                    rel_frames += 1
                    db = -20.0 * math.log10(smin if smin > DBL_MIN_POS else DBL_MIN_POS)
                    if db > rel_db_max:
                        rel_db_max = db
                a0 = pk0 * gr0
                req0 = ceil_lin / (a0 if a0 > DBL_MIN_POS else DBL_MIN_POS) if a0 > ceil_lin else 1.0
                a1 = pk1 * gr1
                req1 = ceil_lin / (a1 if a1 > DBL_MIN_POS else DBL_MIN_POS) if a1 > ceil_lin else 1.0
                grp_req = req0 if req0 < req1 else req1
                if link_tr > 0.0:
                    req0 = _blend_gain(req0, grp_req, link_tr)
                    req1 = _blend_gain(req1, grp_req, link_tr)
                r0[i], r1[i] = req0, req1
                g0_out[i], g1_out[i] = gr0, gr1
            hold[0], hold[1] = h0, h1
            g_target[0], g_target[1] = gt0, gt1
            g_rel[0], g_rel[1] = grl0, grl1
            gr_now[0], gr_now[1] = g0_out[-1], g1_out[-1]
            self.peaks[0], self.peaks[1] = p0[-1], p1[-1]
        else:
            for i in range(n):
                for c in range(nch):
                    pk = peaks_lists[c][i]
                    if pk > hold[c]:
                        hold[c] = pk
                hold_left = hold_left - 1 if hold_left > 0 else 0
                if hold_left == 0:
                    hold_left = rel_win
                    grp = float("-inf")
                    for c in range(nch):
                        lvl = 20.0 * math.log10(max(hold[c], DBL_MIN_POS))
                        if lvl > grp:
                            grp = lvl
                        g_target[c] = 10.0 ** (min(0.0, aim_db - lvl) / 20.0)
                        hold[c] = 0.0
                    g_target_group = 10.0 ** (min(0.0, aim_db - grp) / 20.0)
                for c in range(nch):
                    t = g_target[c]
                    g = g_rel[c]
                    nxt = g + (t - g) * (atk if t < g else rel)
                    g_rel[c] = t if abs(t - nxt) < GAIN_DEADBAND else nxt
                nxtg = g_rel_group + (g_target_group - g_rel_group) * (atk if g_target_group < g_rel_group else rel)
                g_rel_group = g_target_group if abs(g_target_group - nxtg) < GAIN_DEADBAND else nxtg
                smin = 1.0
                for c in range(nch):
                    gr = _blend_gain(g_rel[c], g_rel_group, link_rel)
                    gr_now[c] = gr
                    if gr < smin:
                        smin = gr
                if smin < 1.0 - ENGAGED_EPS:
                    rel_frames += 1
                    db = -20.0 * math.log10(max(smin, DBL_MIN_POS))
                    if db > rel_db_max:
                        rel_db_max = db
                grp_req = 1.0
                tmp_req = [1.0] * nch
                for c in range(nch):
                    a = max(peaks_lists[c][i] * gr_now[c], DBL_MIN_POS)
                    rq = ceil_lin / a if a > ceil_lin else 1.0
                    tmp_req[c] = rq
                    if rq < grp_req:
                        grp_req = rq
                for c in range(nch):
                    rq = _blend_gain(tmp_req[c], grp_req, link_tr) if link_tr > 0.0 else tmp_req[c]
                    req_lists[c][i] = rq
                    gr_lists[c][i] = gr_now[c]
            for c in range(nch):
                self.peaks[c] = peaks_lists[c][-1]

        self.hold_left = hold_left
        self.g_target_group = g_target_group
        self.g_rel_group = g_rel_group
        self._report.release_frames = rel_frames
        self._report.release_reduction_db = rel_db_max

        for c in range(nch):
            req_arr[c][pre_req_len:] = req_lists[c]
            gr_arr[c][pre_req_len:] = gr_lists[c]

        limit_end = fed0 + n - 2 * delta
        j_start = fed0 - latency
        pre_sched_len = scheduled0 - j_start
        sched_arr = [np.empty(pre_sched_len + n, dtype=np.float64) for _ in range(nch)]

        gr_emit = [np.empty(n, dtype=np.float64) for _ in range(nch)]
        m_new_start = fed0 - delta
        old_gr_count = min(n, max(0, m_new_start - j_start))

        slope_vec = np.arange(ramp + 1, dtype=np.float64) * self.slope
        rel_coef_inv = 1.0 - self.release_coef
        deepest_arr: Optional[np.ndarray] = None

        for c in range(nch):
            gc_ring = self.gr_ring[c]
            for t in range(old_gr_count):
                gr_emit[c][t] = gc_ring[(j_start + t) % ring_len]
            if old_gr_count < n:
                gr_emit[c][old_gr_count:] = gr_arr[c][pre_req_len : pre_req_len + (n - old_gr_count)]

            req_win = sliding_window_view(req_arr[c], 2 * delta + 1)
            gr_win = sliding_window_view(gr_arr[c], 2 * delta + 1)
            gr_center = np.maximum(gr_arr[c][delta : delta + n], DBL_MIN_POS)[:, None]
            ga = np.min(req_win * np.minimum(1.0, gr_win / gr_center), axis=1)

            mask_tr = ga < (1.0 - ENGAGED_EPS)
            if np.any(mask_tr):
                min_ga = float(np.min(ga[mask_tr]))
                tr_db = -20.0 * math.log10(max(min_ga, DBL_MIN_POS))
                if tr_db > self._report.transient_reduction_db:
                    self._report.transient_reduction_db = tr_db

            sc_ring = self.sched_ring[c]
            for t in range(pre_sched_len):
                sched_arr[c][t] = sc_ring[(j_start + t) % ring_len]
            sched_arr[c][pre_sched_len:] = ga

            sched_win = sliding_window_view(sched_arr[c], ramp + 1)[:n]
            targets = np.clip(np.min(sched_win + slope_vec[None, :], axis=1), 0.0, 1.0)

            g_cur = self.gain[c]
            t_list = targets.tolist()
            g_out = [0.0] * n
            for i in range(n):
                tgt = t_list[i]
                if tgt < g_cur:
                    g_cur = tgt
                else:
                    nxt_g = g_cur + (1.0 - g_cur) * rel_coef_inv
                    if nxt_g > tgt:
                        nxt_g = tgt
                    g_cur = tgt if (tgt - nxt_g) < GAIN_DEADBAND else nxt_g
                g_out[i] = g_cur
            self.gain[c] = g_cur

            g_total = np.asarray(g_out, dtype=np.float64) * gr_emit[c]
            x_in = np.asarray(planar[c], dtype=np.float64)
            line = np.asarray(self.lines[c], dtype=np.float64)
            ordered_delay = np.concatenate([line[self.pos :], line[: self.pos]]) if self.pos > 0 else line
            combined = np.concatenate([ordered_delay, x_in])
            delayed = combined[:n]
            new_ordered = combined[-latency:]
            new_pos = (self.pos + n) % latency
            if new_pos == 0:
                self.lines[c] = new_ordered.tolist()
            else:
                buf = np.empty(latency, dtype=np.float64)
                buf[new_pos:] = new_ordered[: latency - new_pos]
                buf[:new_pos] = new_ordered[latency - new_pos :]
                self.lines[c] = buf.tolist()

            planar[c][:] = delayed * g_total
            self.out_meter.push_ch(c, planar[c])

            req_tail_count = min(ring_len, len(req_arr[c]))
            req_start_idx = len(req_arr[c]) - req_tail_count
            for k in range(req_tail_count):
                slot = (req_base + req_start_idx + k) % ring_len
                self.req_ring[c][slot] = float(req_arr[c][req_start_idx + k])
                self.gr_ring[c][slot] = float(gr_arr[c][req_start_idx + k])

            sched_tail_count = min(ring_len, len(sched_arr[c]))
            sched_start_idx = len(sched_arr[c]) - sched_tail_count
            for k in range(sched_tail_count):
                slot = (j_start + sched_start_idx + k) % ring_len
                self.sched_ring[c][slot] = float(sched_arr[c][sched_start_idx + k])

            deepest_arr = g_total if deepest_arr is None else np.minimum(deepest_arr, g_total)

        if deepest_arr is not None:
            mask_eng = deepest_arr < (1.0 - ENGAGED_EPS)
            cnt = int(np.count_nonzero(mask_eng))
            if cnt > 0:
                self._report.engaged_frames += cnt
                min_g = float(np.min(deepest_arr[mask_eng]))
                db = -20.0 * math.log10(max(min_g, DBL_MIN_POS))
                if db > self._report.max_gain_reduction_db:
                    self._report.max_gain_reduction_db = db

        self.pos = (self.pos + n) % latency
        self.fed = fed0 + n
        self.scheduled = limit_end

    def process(self, planar: List[np.ndarray]) -> None:
        if not self.cfg.enable or not planar:
            return
        n = len(planar[0])
        if n == 0:
            return
        if not all(len(p) == n for p in planar):
            raise ValueError("ragged block into the limiter")
        for start in range(0, n, self._SUB_CHUNK):
            end = min(n, start + self._SUB_CHUNK)
            self._process_chunk([p[start:end] for p in planar])

    def flush(self) -> List[np.ndarray]:
        if not self.cfg.enable:
            return []
        nch = min(len(self.lines), len(self.side))
        if nch == 0:
            return []
        zeros = np.zeros(self.latency, dtype=np.float64)
        out = [np.empty(self.latency, dtype=np.float64) for _ in range(nch)]
        peaks_mat = [self.side[c].push_block_frames(zeros) for c in range(nch)]
        zframe = [0.0] * nch
        for i in range(self.latency):
            fpeaks = [float(peaks_mat[c][i]) for c in range(nch)]
            out_f = self._step_with_peaks(zframe, fpeaks)
            for c in range(nch):
                out[c][i] = out_f[c]
        for c in range(nch):
            self.out_meter.push_ch(c, out[c])
        self.out_meter.drain()
        return out


# ==============================================================================
# 5. UNIFIED 4-PANEL VERIFICATION PLOTTER (3-ROW GRIDSPEC + CALLOUT ANNOTATIONS)
# ==============================================================================

def compute_gaussian_autoeq(f_plot: np.ndarray, raw_eq_needed: np.ndarray) -> np.ndarray:
    """Variable-bandwidth Gaussian smoother matching standard AutoEq's 2.0-octave treble blur."""
    log_grid = np.log2(f_plot)
    autoeq_plot = np.zeros_like(f_plot)
    for i, f_val in enumerate(f_plot):
        bw = 0.08 if f_val <= 4000.0 else (2.00 if f_val >= 6000.0 else 0.08 + 1.92 * ((f_val - 4000.0) / 2000.0))
        w = np.exp(-0.5 * ((log_grid - log_grid[i]) / (bw / 2.355)) ** 2)
        autoeq_plot[i] = np.sum(raw_eq_needed * w) / np.sum(w)
        if f_val >= 6000.0:
            autoeq_plot[i] *= 0.50
        elif f_val > 4000.0:
            autoeq_plot[i] *= 1.0 - 0.50 * ((f_val - 4000.0) / 2000.0)
    return autoeq_plot


def build_auto_annotations(f_plot: np.ndarray, wav_eq_plot: np.ndarray, hf_bypass_hz: float) -> List[dict]:
    """Auto-generates callout annotations for custom user-added CSV/JSON models."""
    g_20 = float(wav_eq_plot[0])
    bass_mask = (f_plot >= 35.0) & (f_plot <= 350.0)
    idx_bass = int(np.argmax(np.abs(wav_eq_plot[bass_mask])))
    f_bass, g_bass = float(f_plot[bass_mask][idx_bass]), float(wav_eq_plot[bass_mask][idx_bass])

    mid_mask = (f_plot > 350.0) & (f_plot <= 3000.0)
    idx_mid = int(np.argmax(np.abs(wav_eq_plot[mid_mask])))
    f_mid, g_mid = float(f_plot[mid_mask][idx_mid]), float(wav_eq_plot[mid_mask][idx_mid])

    treb_hi = max(4800.0, min(hf_bypass_hz, 16000.0))
    treb_mask = (f_plot > 4500.0) & (f_plot <= treb_hi)
    idx_treb = int(np.argmin(wav_eq_plot[treb_mask]))
    f_treb, g_treb = float(f_plot[treb_mask][idx_treb]), float(wav_eq_plot[treb_mask][idx_treb])

    return [
        {"text": f"1. Sub-Bass Target Shelf\n({g_20:+.2f}dB @ 20Hz)", "xy": [20.5, g_20], "xytext": [22.0, max(5.5, g_20 + 2.0)], "color": "cyan"},
        {"text": f"2. Bass / Low-Mid Correction\n({g_bass:+.2f}dB @ {f_bass:.0f}Hz)", "xy": [f_bass, g_bass], "xytext": [max(55.0, f_bass * 0.55), 5.4 if g_bass < 3.5 else -7.5], "color": "cyan"},
        {"text": f"3. Midrange / Ear-Gain\n({g_mid:+.2f}dB @ {f_mid/1000:.2f}kHz)", "xy": [f_mid, g_mid], "xytext": [max(450.0, f_mid * 0.45), 6.2 if g_mid >= 0 else -7.5], "color": "cyan"},
        {"text": f"4. Treble Peak Control\n({g_treb:+.2f}dB @ {f_treb/1000:.2f}kHz)", "xy": [f_treb, g_treb], "xytext": [2400.0, -8.8], "color": "green"},
        {"text": f"5. >{hf_bypass_hz/1000:.1f}kHz Coupler Null Bypass\n(0.0dB ultrasonic wire)", "xy": [min(14000.0, max(hf_bypass_hz * 1.18, hf_bypass_hz + 500.0)), 0.0], "xytext": [5400.0, 4.5], "color": "cyan"},
    ]


def render_verification_plot(
    model_cfg: dict,
    fs: int,
    n_fft_rt: int,
    ir_rt_f32: np.ndarray,
    mag_lin_rt: np.ndarray,
    boost_rt: float,
    pre_rt: float,
    unwin_rt: float,
    active_ctrl_pts: List[Tuple[float, float]],
    notch_mode: str,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FixedLocator, NullFormatter

    prefix = model_cfg["file_prefix"]
    plot_file = f"{prefix}_final_verification.png"
    raw_pts = model_cfg["raw_pts"]
    target_pts = model_cfg["target_pts"]
    harm_offset = model_cfg["harm_offset"]
    hf_bypass_hz = model_cfg["hf_bypass_hz"]
    meta = model_cfg.get("plot_meta", {})

    H_wav = np.fft.rfft(ir_rt_f32.astype(np.float64), n=n_fft_rt)
    spec_f32 = np.abs(H_wav)
    freqs_fft = np.fft.rfftfreq(n_fft_rt, d=1.0 / fs)
    f_plot = np.geomspace(20.0, 20000.0, 1200)

    f_h = np.array([p[0] for p in target_pts], dtype=np.float64)
    g_h = np.array([p[1] for p in target_pts], dtype=np.float64)
    harm_plot = pchip_log_interp(f_h, g_h, f_plot) + harm_offset

    wav_mag_db_unnorm = 20.0 * np.log10(np.maximum(spec_f32, 1e-15)) - pre_rt
    wav_phase_deg = np.degrees(np.angle(H_wav))
    wav_eq_plot = np.interp(f_plot, freqs_fft, wav_mag_db_unnorm)
    wav_phase_plot = np.interp(f_plot, freqs_fft, wav_phase_deg)
    ir_lin_centered = np.fft.fftshift(np.fft.irfft(mag_lin_rt, n=n_fft_rt))

    f_raw = np.array([p[0] for p in raw_pts], dtype=np.float64)
    raw_avg = pchip_log_interp(f_raw, np.array([p[-1] for p in raw_pts], dtype=np.float64), f_plot)
    raw_eq_needed = harm_plot - raw_avg

    if model_cfg.get("autoeq_geq_str"):
        geq_pts = [
            tuple(map(float, item.strip().split()))
            for item in model_cfg["autoeq_geq_str"].split(";")
            if item.strip()
        ]
        f_geq = np.array([p[0] for p in geq_pts], dtype=np.float64)
        g_geq = np.array([p[1] for p in geq_pts], dtype=np.float64)
        if model_cfg.get("autoeq_offset_db") is not None:
            g_geq = g_geq + float(model_cfg["autoeq_offset_db"])
        else:
            f_c = np.array([p[0] for p in active_ctrl_pts], dtype=np.float64)
            g_c = np.array([p[1] for p in active_ctrl_pts], dtype=np.float64)
            g_geq = g_geq - np.interp(1000.0, f_geq, g_geq) + np.interp(1000.0, f_c, g_c)
        autoeq_plot = np.interp(np.log10(f_plot), np.log10(f_geq), g_geq)
    else:
        autoeq_plot = compute_gaussian_autoeq(f_plot, raw_eq_needed)

    notch_ref_hz = model_cfg.get("notch_ref_hz")
    if notch_ref_hz is not None:
        ctrl_dict = dict(active_ctrl_pts)
        if notch_ref_hz in ctrl_dict:
            notch_db = ctrl_dict[notch_ref_hz]
        else:
            f_c = np.array([p[0] for p in active_ctrl_pts], dtype=np.float64)
            g_c = np.array([p[1] for p in active_ctrl_pts], dtype=np.float64)
            notch_db = float(pchip_log_interp(f_c, g_c, np.array([notch_ref_hz]))[0])
    else:
        notch_db = 0.0

    fmt_ctx = {
        "notch_mode": notch_mode,
        "pre_rt": pre_rt,
        "unwin_rt": unwin_rt,
        "boost_rt": boost_rt,
        "notch_db": notch_db,
    }

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#343b42"
    plt.rcParams["axes.linewidth"] = 0.9

    fig = plt.figure(figsize=(13.0, 11.5), dpi=150, facecolor="#151618")
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[1.15, 1.15, 0.85], hspace=0.26, wspace=0.18,
        left=0.065, right=0.965, top=0.945, bottom=0.055,
    )
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    major_freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    major_labels = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
    minor_freqs = [30, 40, 60, 70, 80, 90, 300, 400, 600, 700, 800, 900, 3000, 4000, 6000, 7000, 8000, 9000, 15000]

    def style_freq_axis(ax):
        ax.set_facecolor("#1b1f22")
        ax.set_xscale("log")
        ax.set_xlim(20, 20000)
        ax.xaxis.set_major_locator(FixedLocator(major_freqs))
        ax.set_xticklabels(major_labels, color="#c9d1d9", fontsize=9.5)
        ax.xaxis.set_minor_locator(FixedLocator(minor_freqs))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", colors="#c9d1d9", labelsize=9.5)
        ax.tick_params(axis="x", which="both", colors="#6e7681", direction="out", length=3)
        ax.grid(True, which="major", color="#2d333b", linestyle="-", linewidth=0.75, alpha=0.9)
        ax.grid(True, which="minor", color="#24292f", linestyle="-", linewidth=0.45, alpha=0.7)
        ax.axhline(0.0, color="#6e7681", linewidth=0.8, alpha=0.6, zorder=1)

    arrow_kw = dict(arrowstyle="->", color="#58d5e0", lw=1.15, shrinkA=2, shrinkB=3)
    arrow_green = dict(arrowstyle="->", color="#26e683", lw=1.2, shrinkA=2, shrinkB=3)

    # Panel 1: Acoustic Level (Raw vs Target vs Standard AutoEq vs Final WAV Convolver)
    style_freq_axis(ax1)
    bypass_label = meta.get(
        "bypass_label",
        f"Uncorrected >{hf_bypass_hz/1000:.1f} kHz Coupler Null Zone (Clamped to 0.0 dB)",
    )
    ax1.axvspan(hf_bypass_hz, 20000.0, color="#30363d", alpha=0.35, zorder=1, label=bypass_label)

    if model_cfg.get("notch_range_hz") is not None:
        default_title = f"VERIFICATION: {model_cfg['short_title']} Convolver (Notch Mode: {{notch_mode}} | Preamp: {{pre_rt:.2f}} dB)"
        default_wav_lbl = "Final WAV Convolver (Notch Mode: {notch_mode} | Unwindowed Floor: {unwin_rt:.1f} dBFS @ 16k)"
    else:
        default_title = f"VERIFICATION: {model_cfg['short_title']} Convolver (Preamp: {{pre_rt:.2f}} dB | Unwindowed Floor: {{unwin_rt:.1f}} dBFS)"
        default_wav_lbl = "Final WAV Convolver (Minimum-Phase PCHIP FIR | Unwindowed Floor: {unwin_rt:.1f} dBFS @ 16k)"

    title_tmpl = meta.get("title_template", default_title)
    ax1.set_title(title_tmpl.format(**fmt_ctx), color="#ffffff", fontsize=12.0, fontweight="bold", pad=10)

    if len(raw_pts[0]) >= 4:
        raw_L = pchip_log_interp(f_raw, np.array([p[1] for p in raw_pts], dtype=np.float64), f_plot)
        raw_R = pchip_log_interp(f_raw, np.array([p[2] for p in raw_pts], dtype=np.float64), f_plot)
        lr_label = meta.get("lr_label", "Raw L / R Channels")
        ax1.plot(f_plot, raw_L, color="#58a6ff", linestyle=":", linewidth=0.85, alpha=0.38, zorder=2, label=lr_label)
        ax1.plot(f_plot, raw_R, color="#58a6ff", linestyle=":", linewidth=0.85, alpha=0.38, zorder=2)

    raw_label = meta.get("raw_label", f"Raw {model_cfg['name']}")
    default_target_label = (
        "Harman Over-Ear 2018 Target"
        if model_cfg["target_name"] == "harman_2018"
        else f"Target Curve ({model_cfg['target_name']})"
    )
    target_label = meta.get("target_label", default_target_label)
    autoeq_ax1_label = meta.get("autoeq_ax1_label", "Standard AutoEq Response")
    wav_ax1_label = meta.get("wav_ax1_label", default_wav_lbl).format(**fmt_ctx)

    ax1.plot(f_plot, raw_avg, color="#9ca3af", linestyle="--", linewidth=1.35, alpha=0.9, zorder=3, label=raw_label)
    ax1.plot(f_plot, harm_plot, color="#1f6feb", linestyle="-", linewidth=3.2, alpha=0.88, zorder=4, label=target_label)
    ax1.plot(f_plot, raw_avg + autoeq_plot, color="#f85149", linestyle="-", linewidth=1.55, alpha=0.92, zorder=5, label=autoeq_ax1_label)
    ax1.plot(f_plot, raw_avg + wav_eq_plot, color="#00e676", linestyle="-", linewidth=2.5, alpha=0.98, zorder=6, label=wav_ax1_label)
    ax1.set_ylabel("Acoustic Level (dBr)", color="#e6edf3", fontsize=10.5)

    if "ax1_ylim" in meta:
        ax1.set_ylim(*meta["ax1_ylim"])
    else:
        y1_lo = min(-14.5, float(np.min(raw_avg + wav_eq_plot)) - 2.0)
        y1_hi = max(11.5, float(np.max(harm_plot)) + 2.5, float(np.max(raw_avg)) + 2.0)
        ax1.set_ylim(y1_lo, y1_hi)
    ax1.set_yticks(meta.get("ax1_yticks", [-10, -5, 0, 5, 10]))

    # Panel 2: Filter Gain (Raw Error vs Standard AutoEq vs Final WAV Convolver + Callouts)
    style_freq_axis(ax2)
    ax2.axvspan(hf_bypass_hz, 20000.0, color="#30363d", alpha=0.35, zorder=1)
    raw_err_ax2_label = meta.get("raw_err_ax2_label", "Raw Unsmoothed Error")
    autoeq_ax2_label = meta.get("autoeq_ax2_label", "Standard AutoEq Curve")

    ax2.plot(f_plot, raw_eq_needed, color="#6e7681", linestyle=":", linewidth=1.0, alpha=0.65, zorder=2, label=raw_err_ax2_label)
    ax2.plot(f_plot, autoeq_plot, color="#f5a623", linestyle="--", linewidth=2.0, alpha=0.95, zorder=4, label=autoeq_ax2_label)
    ax2.plot(
        f_plot, wav_eq_plot, color="#00e676", linestyle="-", linewidth=2.6, alpha=0.98, zorder=5,
        label=f"Final WAV Convolver EQ Curve (Peak +{boost_rt:.2f} dB -> Normalized w/ {pre_rt:.2f} dB Preamp)",
    )
    ax2.set_ylabel("Filter Gain (dBr)", color="#e6edf3", fontsize=10.5)

    if "ax2_ylim" in meta:
        ax2.set_ylim(*meta["ax2_ylim"])
    else:
        y2_lo = min(-10.8, float(np.min(wav_eq_plot)) - 2.5)
        y2_hi = max(10.5, float(np.max(wav_eq_plot)) + 3.2)
        ax2.set_ylim(y2_lo, y2_hi)
    ax2.set_yticks(meta.get("ax2_yticks", [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]))

    annotations = meta.get("annotations") or build_auto_annotations(f_plot, wav_eq_plot, hf_bypass_hz)
    for ann in annotations:
        txt = ann.get("text", "").format(**fmt_ctx)
        xy_x = float(ann["xy"][0])
        xy_y = notch_db if ann["xy"][1] == "notch_db" else float(ann["xy"][1])
        xytext = (float(ann["xytext"][0]), float(ann["xytext"][1]))
        is_green = ann.get("color", "cyan") == "green"
        akw = arrow_green if is_green else arrow_kw
        tcol = "#26e683" if is_green else "#58d5e0"
        if txt:
            ax2.annotate(txt, xy=(xy_x, xy_y), xytext=xytext, color=tcol, fontsize=8.3, arrowprops=akw)
        else:
            ax2.annotate("", xy=(xy_x, xy_y), xytext=xytext, arrowprops=akw)

    leg1 = ax1.legend(loc="lower left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.5)
    for text in leg1.get_texts():
        text.set_color("#d0d7de")
    leg2 = ax2.legend(loc="lower left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.5)
    for text in leg2.get_texts():
        text.set_color("#d0d7de")

    # Panel 3: Hilbert-Cepstral Minimum-Phase Angle
    style_freq_axis(ax3)
    ax3.set_title("Filter Minimum-Phase Response (Hilbert-Cepstral Phase Angle)", color="#ffffff", fontsize=10.8, fontweight="bold", pad=8)
    ax3.plot(f_plot, wav_phase_plot, color="#c961de", linestyle="-", linewidth=2.0, alpha=0.95, label="Filter Minimum-Phase Response (deg)")
    ax3.set_xlabel("Frequency (Hz)", color="#e6edf3", fontsize=10.0)
    ax3.set_ylabel("Phase (deg)", color="#e6edf3", fontsize=10.0)
    ax3.set_ylim(-90, 90)
    ax3.set_yticks([-90, -45, 0, 45, 90])
    leg3 = ax3.legend(loc="upper left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.6)
    for text in leg3.get_texts():
        text.set_color("#d0d7de")

    # Panel 4: Impulse Response: Minimum-Phase vs. Linear-Phase Pre-Ringing
    ax4.set_facecolor("#1b1f22")
    ax4.set_title("Impulse Response: Minimum-Phase vs. Linear-Phase Pre-Ringing", color="#ffffff", fontsize=10.8, fontweight="bold", pad=8)
    peak_mp = float(np.max(np.abs(ir_rt_f32)))
    t_mp_ms = np.arange(len(ir_rt_f32)) / fs * 1000.0
    t_lp_ms = (np.arange(n_fft_rt) - n_fft_rt // 2) / fs * 1000.0
    t_mp_plot = np.concatenate([np.linspace(-1.0, -1.0 / fs * 1000.0, 48), t_mp_ms[:250]])
    ir_mp_plot = np.concatenate([np.zeros(48), ir_rt_f32[:250] / peak_mp])
    lp_mask = (t_lp_ms >= -1.0) & (t_lp_ms <= 4.0)
    ax4.plot(
        t_lp_ms[lp_mask], ir_lin_centered[lp_mask] / peak_mp, color="#f85149",
        linestyle="--", linewidth=1.4, alpha=0.88, label="Linear-Phase IR (Pre-Ringing before 0.0 ms)",
    )
    ax4.plot(
        t_mp_plot, ir_mp_plot, color="#00e676", linestyle="-", linewidth=2.0, alpha=0.96,
        label="Minimum-Phase IR (0.00 ms IR Peak Delay, Zero Pre-Ringing)",
    )
    ax4.axvline(0.0, color="#388bfd", linestyle=":", linewidth=1.1, alpha=0.85)
    ax4.axhline(0.0, color="#6e7681", linewidth=0.8, alpha=0.6)
    ax4.set_xlim(-1.0, 4.0)
    ax4.set_ylim(-0.45, 1.08)
    ax4.set_xlabel("Time Relative to Transient Peak (ms)", color="#e6edf3", fontsize=10.0)
    ax4.set_ylabel("Normalized Amplitude", color="#e6edf3", fontsize=10.0)
    ax4.tick_params(axis="both", colors="#c9d1d9", labelsize=9.5)
    ax4.grid(True, which="major", color="#2d333b", linestyle="-", linewidth=0.75, alpha=0.9)
    leg4 = ax4.legend(loc="upper right", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.4)
    for text in leg4.get_texts():
        text.set_color("#d0d7de")

    plt.savefig(plot_file, dpi=150, facecolor="#151618")
    plt.close(fig)
    return plot_file


# ==============================================================================
# 6. PER-MODEL RUNNER, AUDIO CONVOLVER/LIMITER & DYNAMIC MPV CONFIG GENERATOR
# ==============================================================================

def build_limiter_cfg(args) -> LimiterCfg:
    """Builds and validates a LimiterCfg from CLI arguments (default: x16 @ -3.0 dBTP)."""
    ratio = Ratio.from_str(args.tp_ratio)
    cfg = LimiterCfg(
        enable=not getattr(args, "no_tp_limiter", False),
        ceiling_db=float(args.tp_ceiling),
        ratio=ratio,
        lookahead_ms=float(args.tp_lookahead),
        attack_ms=float(args.tp_attack),
        release_ms=float(args.tp_release),
        link_transient=float(args.tp_link_transient),
        link_release=float(args.tp_link_release),
    )
    cfg.validate()
    return cfg


def build_convolver_probe(fs: int, duration_s: float = 0.5) -> np.ndarray:
    """
    Constructs a deterministic 0.00 dBTP (x16) band-limited (<= 0.45 fs) multi-tone +
    transient probe signal to verify EQ convolution + true-peak limiting headroom.
    """
    n = max(4096, int(round(fs * duration_s)))
    t = np.arange(n, dtype=np.float64) / float(fs)
    nyq_safe = 0.43 * fs
    f_hi = min(12000.0, nyq_safe)
    f_mid = min(4060.0, 0.35 * fs)
    probe = (
        0.55 * np.sin(2.0 * math.pi * f_mid * t + 0.30)
        + 0.45 * np.sin(2.0 * math.pi * f_hi * t + math.pi / 4.0)
        + 0.30 * np.sin(2.0 * math.pi * 150.0 * t)
    )
    burst_len = min(1024, n // 4)
    burst_start = n // 2
    b_t = np.arange(burst_len, dtype=np.float64) - burst_len / 2.0
    fc = 0.42
    sinc_b = np.where(b_t == 0.0, 2.0 * fc, np.sin(2.0 * math.pi * fc * b_t) / (math.pi * b_t))
    win_b = np.hanning(burst_len)
    probe[burst_start : burst_start + burst_len] += (
        1.25 * (sinc_b * win_b) / max(float(np.max(np.abs(sinc_b * win_b))), 1e-12)
    )

    d = Detector(Ratio.X16)
    d.push_block(probe)
    d.push_block(np.zeros(int(math.ceil(Ratio.X16.group_delay())) + 1, dtype=np.float64))
    pk = max(d.peak(), 1e-12)
    return probe / pk


def print_signal_measurements(label: str, m: SignalMeasurements) -> None:
    """Prints true-peak (ratio + ITU x4) and 0.30s peak-hold meter measurements."""
    ch_tp = ", ".join(f"{v:+.2f}" for v in m.per_channel_dbtp)
    ch_itu = ", ".join(f"{v:+.2f}" for v in m.itu_x4_per_channel_dbtp)
    print(f"  {label}:")
    print(
        f"    True Peak ({m.ratio.label})  : {m.true_peak_dbtp:+.3f} dBTP"
        + (f" [{ch_tp}]" if m.channels > 1 else "")
        + f" | ITU BS.1770-5 (x4): {m.itu_x4_dbtp:+.3f} dBTP"
        + (f" [{ch_itu}]" if m.channels > 1 else "")
    )
    print(
        f"    Sample Peak / RMS : {m.sample_peak_dbfs:+.3f} dBFS | {m.rms_dbfs:+.2f} dBFS RMS"
        f" | Crest: {m.crest_tp_db:.2f} dB (TP) / {m.crest_sample_db:.2f} dB (sample)"
    )
    print(
        f"    Meter (0.30s Hold): {m.meter_peak_hold_dbfs:+.3f} dBFS"
        f" (param {m.meter_param:.4f} on -60..0 dB scale | {m.frames:,} frames, {m.duration_s:.3f} s)"
    )


def print_limiter_report(rep: LimiterReport, post_meas: SignalMeasurements, cfg: LimiterCfg, fs: int) -> None:
    """Prints the two-stage true-peak limiter report and independent verification."""
    raised = " [RAISED to filter floor]" if cfg.lookahead_was_raised(fs) else ""
    print(
        f"  Side-Chain / Ratio  : {rep.ratio.label} ({rep.ratio.chain()})"
    )
    print(
        f"  Ceiling / Guard     : {rep.ceiling_db:+.2f} dBTP"
        f" (aim guard -{rep.guard_db:.2f} dB | >0.45 fs band caveat +{rep.band_caveat_db:.2f} dB)"
    )
    print(
        f"  Lookahead / Latency : {rep.effective_lookahead_ms:.2f} ms eff ({rep.requested_lookahead_ms:.2f} ms req){raised}"
        f" | {rep.latency_samples} samples ({rep.latency_ms:.2f} ms), ramp {rep.ramp_frames} frames"
    )
    print(
        f"  Two-Stage Envelope  : Slow {rep.attack_ms:.1f}/{rep.release_ms:.1f} ms (link {rep.link_release*100:.0f}%)"
        f" | Transient {TRANSIENT_RELEASE_MS:.1f} ms rel (link {rep.link_transient*100:.0f}%)"
    )
    print(
        f"  Limiter Pre -> Post : {rep.pre_dbtp:+.3f} -> {rep.post_dbtp:+.3f} dBTP ({rep.ratio.label})"
        f" | Independent {post_meas.ratio.label}: {post_meas.true_peak_dbtp:+.3f} dBTP"
        f" | ITU x4: {post_meas.itu_x4_dbtp:+.3f} dBTP | Within ceiling: {rep.within_ceiling()}"
    )
    print(
        f"  Gain Reduction      : Max {rep.max_gain_reduction_db:.3f} dB"
        f" (Transient {rep.transient_reduction_db:.3f} dB, Slow {rep.release_reduction_db:.3f} dB)"
        f" | Engaged: {rep.engaged_frames:,} frames (Slow hold: {rep.release_frames:,})"
    )


def run_audio_wav_mode(args, model_cfg: Optional[dict], lim_cfg: LimiterCfg) -> None:
    """
    Processes an input WAV file: measures input, optionally convolves with the selected
    headphone EQ IR, applies the x16 true-peak limiter, verifies output, and writes WAV.
    """
    wav_fs, planar_in, fmt_desc = read_wav(args.input)
    fs = wav_fs if args.fs is None else int(args.fs)
    if args.fs is not None and args.fs != wav_fs:
        print(f"  [Warning] --fs={fs} differs from input WAV sample rate {wav_fs}; synthesizing EQ IR at {wav_fs} Hz.")
        fs = wav_fs

    print("\n================================================================================")
    if args.measure_only and args.bypass_eq:
        print(f"AUDIO MEASUREMENT: {args.input} ({fmt_desc})")
    elif args.bypass_eq:
        print(f"TRUE-PEAK LIMITER ({lim_cfg.ratio.label}): {args.input} ({fmt_desc})")
    else:
        assert model_cfg is not None
        print(f"CONVOLVER + {lim_cfg.ratio.label.upper()} LIMITER: {args.input} ({fmt_desc}) -> {model_cfg['name']}")
    print("================================================================================")

    m_in = measure_signal(planar_in, fs=fs, ratio=lim_cfg.ratio)
    print_signal_measurements("Input Audio", m_in)

    work = [ch.copy() for ch in planar_in]
    orig_n = len(planar_in[0]) if planar_in else 0
    ir_tag = "bypass"
    if not args.bypass_eq:
        assert model_cfg is not None
        use_apodize = model_cfg["gauss_apodize"] if args.apodize is None else bool(args.apodize)
        active_ctrl_pts = apply_notch_mode(
            model_cfg["ctrl_pts"],
            mode=args.notch_mode,
            safe_cap_db=model_cfg["safe_cap_db"],
            notch_range_hz=model_cfg["notch_range_hz"],
        )
        n_fft_mpv = max(65536, 1 << (args.taps * 4 - 1).bit_length())
        ir_f64, _, boost_db, pre_db, _, _, _ = synthesize_minimum_phase_ir(
            active_ctrl_pts,
            fs=fs,
            n_fft=n_fft_mpv,
            n_taps=args.taps,
            n_tail=args.taps // 8,
            headroom_db=float(args.headroom_db),
            gauss_apodize=use_apodize,
            kaiser_beta=18.0,
        )
        l1_gain_db = db_from_peak(float(np.sum(np.abs(ir_f64))))
        print(
            f"  EQ FIR ({model_cfg['file_prefix']})   : {args.taps:,} taps @ {fs:,} Hz"
            f" | Preamp {pre_db:+.2f} dB (Max boost +{boost_db:.2f} dB, ||h||_1 {l1_gain_db:+.2f} dB)"
        )
        work = convolve_planar_fft(work, ir_f64, trim_to_input=bool(args.trim_tail))
        m_eq = measure_signal(work, fs=fs, ratio=lim_cfg.ratio)
        print_signal_measurements("Post-EQ (Pre-Limiter)", m_eq)
        ir_tag = model_cfg["file_prefix"]

    if args.measure_only:
        return

    if lim_cfg.enable:
        lim = Limiter.enabled(lim_cfg, fs, len(work))
        lim.process(work)
        tail = lim.flush()
        if tail:
            if args.trim_tail:
                lat = lim.latency_samples()
                work = [np.concatenate([w, t])[lat : lat + orig_n] for w, t in zip(work, tail)]
            else:
                work = [np.concatenate([w, t]) for w, t in zip(work, tail)]
        rep = lim.report()
        m_out = measure_signal(work, fs=fs, ratio=lim_cfg.ratio)
        print("\n=== True-Peak Limiter Report ===")
        print_limiter_report(rep, m_out, lim_cfg, fs)
        print_signal_measurements("Output Audio (Verified)", m_out)
    else:
        print("\n  [Note] True-peak limiter disabled (--no-tp-limiter).")
        m_out = measure_signal(work, fs=fs, ratio=lim_cfg.ratio)
        print_signal_measurements("Output Audio", m_out)

    in_path = Path(args.input)
    if args.output:
        out_path = str(args.output)
    else:
        suffix = f"_{ir_tag}" if ir_tag != "bypass" else ""
        lim_tag = f"_tp_{lim_cfg.ratio.label}" if lim_cfg.enable else ""
        out_path = str(in_path.with_name(f"{in_path.stem}{suffix}{lim_tag}.wav"))

    write_wav_pcm_or_float(out_path, work, fs=fs, bits=int(args.out_bits))
    print(f"  Written Output WAV  : {out_path} ({args.out_bits}-bit @ {fs:,} Hz, {len(work)}ch, {len(work[0]):,} frames)")


def run_model(model_cfg: dict, args, copy_default_ir: bool, lim_cfg: LimiterCfg) -> None:
    fs = 48000 if args.fs is None else int(args.fs)
    rate_tag = format_rate_tag(fs)
    model_key = model_cfg["key"]
    prefix = model_cfg["file_prefix"]
    headroom_db = float(args.headroom_db)

    use_apodize = model_cfg["gauss_apodize"] if args.apodize is None else bool(args.apodize)

    title = model_cfg["name"]
    if model_cfg.get("notch_range_hz") is not None:
        title = f"{title} | Notch: {args.notch_mode}"

    active_ctrl_pts = apply_notch_mode(
        model_cfg["ctrl_pts"],
        mode=args.notch_mode,
        safe_cap_db=model_cfg["safe_cap_db"],
        notch_range_hz=model_cfg["notch_range_hz"],
    )
    raw_pts = model_cfg["raw_pts"]
    target_pts = model_cfg["target_pts"]
    harm_offset = model_cfg["harm_offset"]

    f64_wav = f"{prefix}_f64_{rate_tag}.wav"
    mono_rt = f"{prefix}_mastering_mono_{rate_tag}.wav"
    stereo_rt = f"{prefix}_mastering_stereo_{rate_tag}.wav"

    n_fft_mpv = max(65536, 1 << (args.taps * 4 - 1).bit_length())
    n_tail_mpv = args.taps // 8
    kaiser_beta = 18.0
    kaiser_form_db = kaiser_beta / 0.1102 + 8.7

    ir_mpv_f64, _, boost_mpv, pre_mpv, en_mpv, unwin_mpv, win_mpv = synthesize_minimum_phase_ir(
        active_ctrl_pts,
        fs=fs,
        n_fft=n_fft_mpv,
        n_taps=args.taps,
        n_tail=n_tail_mpv,
        headroom_db=headroom_db,
        gauss_apodize=use_apodize,
        kaiser_beta=kaiser_beta,
    )
    write_wav_ieee_float(f64_wav, ir_mpv_f64, fs=fs, bits=64)
    if copy_default_ir:
        shutil.copyfile(f64_wav, "ir.wav")

    ir_mpv_meas = measure_signal([ir_mpv_f64], fs=fs, ratio=lim_cfg.ratio)
    ir_l1_db = db_from_peak(float(np.sum(np.abs(ir_mpv_f64))))

    apod_desc = "Gaussian Apodizer (>20.2 kHz)" if use_apodize else "Non-Apodizing 0.00 dB HF Wire"
    print("\n================================================================================")
    print(f"MODEL: {title}")
    print("================================================================================")
    print(f"=== [1] Offline / mpv 64-Bit Double Convolver ({f64_wav}) ===")
    print(f"  Format            : 64-bit IEEE Float (pcm_f64le) | {fs:,} Hz | Mono")
    print(f"  FFT / Taps        : {n_fft_mpv:,} FFT ({fs/n_fft_mpv:.4f} Hz bin) -> {args.taps:,} taps ({args.taps/fs*1000:.1f} ms)")
    print(f"  HF Mode / Taper   : {apod_desc} + Half-Kaiser Fade (beta={kaiser_beta:.1f}, {kaiser_form_db:.1f} dB Kaiser formula)")
    print(f"  Total Preamp      : {pre_mpv:.2f} dB (Max Boost +{boost_mpv:.2f} dB, {headroom_db:.2f} dB steady-state headroom)")
    print(f"  Causal Energy     : {en_mpv:.10f}%")
    print(f"  Decay @ Truncation: {unwin_mpv:.2f} dBFS (unwindowed) | {win_mpv:.2f} dBFS (after half-Kaiser fade)")
    print(
        f"  IR Peak / Meter   : {ir_mpv_meas.sample_peak_dbfs:+.3f} dBFS sample"
        f" | {ir_mpv_meas.true_peak_dbtp:+.3f} dBTP ({lim_cfg.ratio.label})"
        f" | {ir_mpv_meas.itu_x4_dbtp:+.3f} dBTP (ITU x4)"
        f" | ||h||_1 bound: {ir_l1_db:+.2f} dB"
    )
    print(
        f"  IR Hold (0.30s)   : {ir_mpv_meas.meter_peak_hold_dbfs:+.3f} dBFS"
        f" (param {ir_mpv_meas.meter_param:.4f} on -60..0 dB scale)"
    )

    n_fft_rt = 32768
    ir_rt_f64, mag_lin_rt, boost_rt, pre_rt, _, unwin_rt, win_rt = synthesize_minimum_phase_ir(
        active_ctrl_pts,
        fs=fs,
        n_fft=n_fft_rt,
        n_taps=16384,
        n_tail=2048,
        headroom_db=headroom_db,
        gauss_apodize=use_apodize,
        kaiser_beta=16.0,
    )
    ir_rt_f32 = ir_rt_f64.astype(np.float32)
    write_wav_ieee_float(mono_rt, ir_rt_f32, fs=fs, bits=32)
    write_wav_ieee_float(stereo_rt, np.column_stack([ir_rt_f32, ir_rt_f32]), fs=fs, bits=32)
    err_rms_dbfs = 20.0 * np.log10(np.sqrt(np.mean((ir_rt_f64 - ir_rt_f32.astype(np.float64)) ** 2)) + 1e-30)
    ir_rt_meas = measure_signal([ir_rt_f32.astype(np.float64)], fs=fs, ratio=lim_cfg.ratio)

    print("\n=== [2] Real-Time PipeWire / DAW 32-Bit Float Convolvers ===")
    print(f"  FFT / Taps        : {n_fft_rt:,} FFT ({fs/n_fft_rt:.4f} Hz bin) -> 16,384 taps ({16384/fs*1000:.1f} ms, Half-Kaiser beta=16.0)")
    print(f"  Decay @ Truncation: {unwin_rt:.2f} dBFS (unwindowed) | {win_rt:.2f} dBFS (after half-Kaiser fade)")
    print(
        f"  IR Peak / Meter   : {ir_rt_meas.sample_peak_dbfs:+.3f} dBFS sample"
        f" | {ir_rt_meas.true_peak_dbtp:+.3f} dBTP ({lim_cfg.ratio.label})"
        f" | {ir_rt_meas.itu_x4_dbtp:+.3f} dBTP (ITU x4)"
    )
    print(f"  f32 vs f64 Error  : {err_rms_dbfs:.2f} dBFS RMS")
    print(f"  Written WAVs      : {mono_rt}, {stereo_rt}")

    if args.export_csv:
        csv_path = export_audit_csv(model_key, raw_pts, target_pts, harm_offset, active_ctrl_pts)
        print(f"  Written Audit CSV : {csv_path}")

    if not args.no_plot:
        try:
            plot_file = render_verification_plot(
                model_cfg, fs, n_fft_rt, ir_rt_f32, mag_lin_rt,
                boost_rt, pre_rt, unwin_rt, active_ctrl_pts, args.notch_mode,
            )
            print(f"  Written Plot      : {plot_file}")
        except ImportError:
            print("  [Note] matplotlib not installed; skipped PNG plot generation.")

    probe = build_convolver_probe(fs, duration_s=0.5)
    convolved_probe = convolve_planar_fft([probe], ir_mpv_f64, trim_to_input=False)
    pre_probe_meas = measure_signal(convolved_probe, fs=fs, ratio=lim_cfg.ratio)

    print(f"\n=== [3] True-Peak {lim_cfg.ratio.label} Limiter & Convolver Verification (0.00 dBTP Probe -> IR) ===")
    print_signal_measurements("Convolved Probe (Pre-Limiter)", pre_probe_meas)
    if lim_cfg.enable:
        probe_work = [convolved_probe[0].copy()]
        lim = Limiter.enabled(lim_cfg, fs, 1)
        lim.process(probe_work)
        tail = lim.flush()
        if tail:
            probe_work = [np.concatenate([probe_work[0], tail[0]])]
        rep = lim.report()
        post_probe_meas = measure_signal(probe_work, fs=fs, ratio=lim_cfg.ratio)
        print_limiter_report(rep, post_probe_meas, lim_cfg, fs)
        print_signal_measurements("Convolved Probe (Post-Limiter)", post_probe_meas)
    else:
        print("  [Note] True-peak limiter disabled (--no-tp-limiter).")

    abs_ir_path = os.path.abspath(f64_wav).replace("\\", "/")
    if ":" in abs_ir_path:
        abs_ir_path = abs_ir_path.replace(":", "\\:")
    print(f"\n=== [4] Matching mpv.conf Snippet ({fs} Hz, minp=1024:maxp=8192 -> {1024/fs*1000:.1f} ms block latency) ===")
    print(f"audio-samplerate={fs}")
    print(
        f"af=lavfi=[aformat=sample_fmts=dblp,aresample={fs}:filter_size=256:phase_shift=12:cutoff=0.98:"
        f"filter_type=kaiser:kaiser_beta=16:osf=dblp[in];amovie={abs_ir_path},aformat=sample_fmts=dblp[ir];"
        f"[in][ir]afir=irfmt=mono:irnorm=-1:wet=1:minp=1024:maxp=8192:precision=double]"
    )


def main() -> None:
    help_requested = any(flag in sys.argv[1:] for flag in ("-h", "--help"))
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    pre_parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    pre_parser.add_argument("--upgrade-json", action="store_true")
    pre_parser.add_argument("--list-models", action="store_true")
    pre_parser.add_argument("--input", type=str, default=None)
    pre_parser.add_argument("--measure-wav", type=str, default=None)
    pre_parser.add_argument("--bypass-eq", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.upgrade_json and not help_requested:
        upgrade_measurements_json(pre_args.data_file)
        return

    needs_models = not (pre_args.measure_wav or (pre_args.input and pre_args.bypass_eq))
    models: Dict[str, dict] = {}
    if pre_args.data_file.exists() or (not help_requested and needs_models):
        _, models = load_measurement_database(data_file=pre_args.data_file, models_dir=pre_args.models_dir)

    active_aliases = {a: t for a, t in MODEL_ALIASES.items() if t in models}
    valid_choices = ["all"] + list(models.keys()) + list(active_aliases.keys())

    parser = argparse.ArgumentParser(
        description="HyprEQ: Unified Minimum-Phase Convolver Generator, x16 True-Peak Limiter & Measurements Suite."
    )
    parser.add_argument(
        "-H", "--headphone", choices=valid_choices if models else None, default="all",
        help=f"Headphone profile to generate/apply (available: {', '.join(['all'] + list(models.keys()))}; default: all)",
    )
    parser.add_argument("--fs", type=int, default=None, help="Sample rate in Hz (default: 48000 for IR gen, or input WAV rate)")
    parser.add_argument(
        "--taps", type=int, default=65536,
        help="Tap count for 64-bit float WAV (default: 65536; 16384 already reaches <-134 dBFS unwindowed)",
    )
    parser.add_argument(
        "--headroom-db", type=float, default=-1.50,
        help="Steady-state convolver preamp headroom in dB (default: -1.50)",
    )
    parser.add_argument(
        "--notch-mode", choices=["safe", "original", "zero"], default="safe",
        help="Non-minimum-phase concha notch handling: 'safe' (default, caps narrow fill to +1.50 dB), 'original', or 'zero' (0.0 dB)",
    )
    apod_grp = parser.add_mutually_exclusive_group()
    apod_grp.add_argument(
        "--apodize", dest="apodize", action="store_true", default=None,
        help="Force Gaussian Nyquist roll-off (>20.2 kHz) for legacy brickwall-ringing sources",
    )
    apod_grp.add_argument(
        "--no-apodize", dest="apodize", action="store_false",
        help="Force non-apodizing 0.00 dB ultrasonic wire (overrides model JSON gauss_apodize)",
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export auditable CSV tables of raw, target, and EQ data",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib 4-panel verification plot generation",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all loaded headphone models from measurements.json and models/ and exit",
    )
    parser.add_argument(
        "--upgrade-json", action="store_true",
        help="Enrich measurements.json in-place with built-in plot_meta and callout annotations",
    )
    parser.add_argument(
        "--data-file", type=Path, default=DEFAULT_DATA_FILE,
        help="Path to external JSON measurement database (default: measurements.json)",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
        help="Optional directory containing additional per-model .json or .csv files",
    )

    tp_grp = parser.add_argument_group("True-Peak Limiter & Measurements")
    tp_grp.add_argument(
        "--tp-ratio", type=str, default="x16",
        help="True-peak oversampling ratio: 2/x2, 4/x4, 8/x8, 16/x16, 32/x32 (default: x16)",
    )
    tp_grp.add_argument(
        "--tp-ceiling", type=float, default=DEFAULT_CEILING_DB,
        help=f"True-peak limiter ceiling in dBTP, [{MIN_CEILING_DB}, {MAX_CEILING_DB}] (default: {DEFAULT_CEILING_DB})",
    )
    tp_grp.add_argument(
        "--tp-lookahead", type=float, default=DEFAULT_LOOKAHEAD_MS,
        help=f"Transient lookahead in ms, [{MIN_LOOKAHEAD_MS}, {MAX_LOOKAHEAD_MS}] (default: {DEFAULT_LOOKAHEAD_MS})",
    )
    tp_grp.add_argument(
        "--tp-attack", type=float, default=DEFAULT_ATTACK_MS,
        help=f"Slow-stage attack in ms, [{MIN_ATTACK_MS}, {MAX_ATTACK_MS}] (default: {DEFAULT_ATTACK_MS})",
    )
    tp_grp.add_argument(
        "--tp-release", type=float, default=DEFAULT_RELEASE_MS,
        help=f"Slow-stage release in ms, [{MIN_RELEASE_MS}, {MAX_RELEASE_MS}] (default: {DEFAULT_RELEASE_MS})",
    )
    tp_grp.add_argument(
        "--tp-link-transient", type=float, default=DEFAULT_LINK_TRANSIENT,
        help=f"Transient stage channel linking [0..1] (default: {DEFAULT_LINK_TRANSIENT})",
    )
    tp_grp.add_argument(
        "--tp-link-release", type=float, default=DEFAULT_LINK_RELEASE,
        help=f"Slow stage channel linking [0..1] (default: {DEFAULT_LINK_RELEASE})",
    )
    tp_grp.add_argument(
        "--no-tp-limiter", action="store_true",
        help="Disable the true-peak ceiling limiter (measurements still run)",
    )
    tp_grp.add_argument(
        "--input", type=str, default=None,
        help="Input WAV file to measure, convolve with headphone EQ, and true-peak limit",
    )
    tp_grp.add_argument(
        "--output", type=str, default=None,
        help="Output WAV path for --input mode (default: <input>_<model>_tp_<ratio>.wav)",
    )
    tp_grp.add_argument(
        "--out-bits", type=int, choices=[16, 24, 32, 64], default=32,
        help="Output WAV bit depth for --input mode: 16/24 TPDF PCM or 32/64 float (default: 32)",
    )
    tp_grp.add_argument(
        "--bypass-eq", action="store_true",
        help="Skip EQ convolution in --input mode (run true-peak limiter + measurements only)",
    )
    tp_grp.add_argument(
        "--trim-tail", action="store_true",
        help="Trim convolver/limiter trailing latency/tail to match input frame count in --input mode",
    )
    tp_grp.add_argument(
        "--measure-only", action="store_true",
        help="Measure audio file / convolved signal and print the true-peak & meter report without writing WAV",
    )
    tp_grp.add_argument(
        "--measure-wav", type=str, default=None,
        help="Shorthand for --input <wav> --bypass-eq --measure-only",
    )

    args = parser.parse_args()

    if args.taps < 256:
        parser.error("--taps must be >= 256")

    if args.measure_wav:
        args.input = args.measure_wav
        args.bypass_eq = True
        args.measure_only = True

    try:
        lim_cfg = build_limiter_cfg(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_models:
        for k, m in models.items():
            print(f"{k:12s} | {m['name']} | target={m['target_name']} | gauss_apodize={m['gauss_apodize']}")
        return

    if args.input:
        selected = active_aliases.get(args.headphone, args.headphone)
        model_cfg = None
        if not args.bypass_eq:
            if selected == "all":
                if not models:
                    parser.error("No headphone models loaded; pass --bypass-eq or --data-file.")
                selected = next(iter(models.keys()))
                print(f"  [Note] --headphone=all with --input selects first model '{selected}'. Pass -H <model> to choose.")
            model_cfg = models[selected]
        run_audio_wav_mode(args, model_cfg, lim_cfg)
        return

    selected = active_aliases.get(args.headphone, args.headphone)
    if selected == "all":
        for m_cfg in models.values():
            run_model(m_cfg, args, copy_default_ir=False, lim_cfg=lim_cfg)
    else:
        run_model(models[selected], args, copy_default_ir=True, lim_cfg=lim_cfg)


if __name__ == "__main__":
    main()
