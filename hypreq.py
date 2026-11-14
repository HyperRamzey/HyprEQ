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
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np


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
# 2. SHARED DSP ENGINE (PCHIP, BESSEL I0, WAV I/O, HILBERT-CEPSTRAL SYNTHESIS)
# ==============================================================================

def bessel_i0(x: np.ndarray) -> np.ndarray:
    """Exact 64-bit float 45-term power series for Modified Bessel Function I0(x) (zero scipy dependency)."""
    x = np.asarray(x, dtype=np.float64)
    res = np.ones_like(x)
    term = np.ones_like(x)
    half_x_sq = (0.5 * x) ** 2
    for k in range(1, 45):
        term *= half_x_sq / (k * k)
        res += term
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
        raw_bytes = np.ascontiguousarray(data, dtype=dtype_str).tobytes()

    byte_rate = fs * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    data_size = len(raw_bytes)
    riff_size = 36 + data_size

    with open(filename, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, channels, fs, byte_rate, block_align, bits))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(raw_bytes)


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
# 3. UNIFIED 4-PANEL VERIFICATION PLOTTER (3-ROW GRIDSPEC + CALLOUT ANNOTATIONS)
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
# 4. PER-MODEL RUNNER & DYNAMIC MPV CONFIG GENERATOR
# ==============================================================================

def run_model(model_cfg: dict, args, copy_default_ir: bool) -> None:
    fs = args.fs
    rate_tag = format_rate_tag(fs)
    model_key = model_cfg["key"]
    prefix = model_cfg["file_prefix"]

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
        headroom_db=-1.50,
        gauss_apodize=use_apodize,
        kaiser_beta=kaiser_beta,
    )
    write_wav_ieee_float(f64_wav, ir_mpv_f64, fs=fs, bits=64)
    if copy_default_ir:
        shutil.copyfile(f64_wav, "ir.wav")

    apod_desc = "Gaussian Apodizer (>20.2 kHz)" if use_apodize else "Non-Apodizing 0.00 dB HF Wire"
    print("\n================================================================================")
    print(f"MODEL: {title}")
    print("================================================================================")
    print(f"=== [1] Offline / mpv 64-Bit Double Convolver ({f64_wav}) ===")
    print(f"  Format            : 64-bit IEEE Float (pcm_f64le) | {fs:,} Hz | Mono")
    print(f"  FFT / Taps        : {n_fft_mpv:,} FFT ({fs/n_fft_mpv:.4f} Hz bin) -> {args.taps:,} taps ({args.taps/fs*1000:.1f} ms)")
    print(f"  HF Mode / Taper   : {apod_desc} + Half-Kaiser Fade (beta={kaiser_beta:.1f}, {kaiser_form_db:.1f} dB Kaiser formula)")
    print(f"  Total Preamp      : {pre_mpv:.2f} dB (Max Boost +{boost_mpv:.2f} dB, -1.50 dB true-peak headroom)")
    print(f"  Causal Energy     : {en_mpv:.10f}%")
    print(f"  Decay @ Truncation: {unwin_mpv:.2f} dBFS (unwindowed) | {win_mpv:.2f} dBFS (after half-Kaiser fade)")

    n_fft_rt = 32768
    ir_rt_f64, mag_lin_rt, boost_rt, pre_rt, _, unwin_rt, win_rt = synthesize_minimum_phase_ir(
        active_ctrl_pts,
        fs=fs,
        n_fft=n_fft_rt,
        n_taps=16384,
        n_tail=2048,
        headroom_db=-1.50,
        gauss_apodize=use_apodize,
        kaiser_beta=16.0,
    )
    ir_rt_f32 = ir_rt_f64.astype(np.float32)
    write_wav_ieee_float(mono_rt, ir_rt_f32, fs=fs, bits=32)
    write_wav_ieee_float(stereo_rt, np.column_stack([ir_rt_f32, ir_rt_f32]), fs=fs, bits=32)
    err_rms_dbfs = 20.0 * np.log10(np.sqrt(np.mean((ir_rt_f64 - ir_rt_f32.astype(np.float64)) ** 2)) + 1e-30)

    print("\n=== [2] Real-Time PipeWire / DAW 32-Bit Float Convolvers ===")
    print(f"  FFT / Taps        : {n_fft_rt:,} FFT ({fs/n_fft_rt:.4f} Hz bin) -> 16,384 taps ({16384/fs*1000:.1f} ms, Half-Kaiser beta=16.0)")
    print(f"  Decay @ Truncation: {unwin_rt:.2f} dBFS (unwindowed) | {win_rt:.2f} dBFS (after half-Kaiser fade)")
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

    abs_ir_path = os.path.abspath(f64_wav).replace("\\", "/")
    if ":" in abs_ir_path:
        abs_ir_path = abs_ir_path.replace(":", "\\:")
    print(f"\n=== [3] Matching mpv.conf Snippet ({fs} Hz, minp=1024:maxp=8192 -> {1024/fs*1000:.1f} ms block latency) ===")
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
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.upgrade_json and not help_requested:
        upgrade_measurements_json(pre_args.data_file)
        return

    models: Dict[str, dict] = {}
    if pre_args.data_file.exists() or not help_requested:
        _, models = load_measurement_database(data_file=pre_args.data_file, models_dir=pre_args.models_dir)

    active_aliases = {a: t for a, t in MODEL_ALIASES.items() if t in models}
    valid_choices = ["all"] + list(models.keys()) + list(active_aliases.keys())

    parser = argparse.ArgumentParser(
        description="HyprEQ: Unified Minimum-Phase Convolver Generator & 4-Panel Verification Suite."
    )
    parser.add_argument(
        "-H", "--headphone", choices=valid_choices if models else None, default="all",
        help=f"Headphone profile to generate (available: {', '.join(['all'] + list(models.keys()))}; default: all)",
    )
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate in Hz (default: 48000)")
    parser.add_argument(
        "--taps", type=int, default=65536,
        help="Tap count for 64-bit float WAV (default: 65536; 16384 already reaches <-134 dBFS unwindowed)",
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
    args = parser.parse_args()

    if args.taps < 256:
        parser.error("--taps must be >= 256")

    if args.list_models:
        for k, m in models.items():
            print(f"{k:12s} | {m['name']} | target={m['target_name']} | gauss_apodize={m['gauss_apodize']}")
        return

    selected = active_aliases.get(args.headphone, args.headphone)
    if selected == "all":
        for m_cfg in models.values():
            run_model(m_cfg, args, copy_default_ir=False)
    else:
        run_model(models[selected], args, copy_default_ir=True)


if __name__ == "__main__":
    main()
