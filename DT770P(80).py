#!/usr/bin/env python3
"""
DT770P(80).py
Dual-Mode Mastering Convolver & 4-Panel Verification Suite for Beyerdynamic DT 770 Pro (80 Ohm)

Data Sources & Attribution:
  - Raw Measurement : Crinacle (Beyerdynamic DT 770 Pro 80 Ohm measured on Genuine GRAS 43AG-7:
                      RA0402 High-Frequency 711 Coupler + KB500X Anthropometric Pinna),
                      sourced via the AutoEq database (Jaakko Pasanen, MIT License).
  - Target Curve    : Harman Over-Ear 2018 Target (Harman International).

Key Acoustic Design Choices (& Limitations):
  - Single-coupler measurements have inherent seat-to-seat and head-to-head variance (+/- 2-4 dB in sub-bass
    velour pad seal and >3 kHz pinna interference notches). Even between the Left and Right channels of this
    measured unit, the ~3.76 kHz non-minimum-phase concha reflection null shifts by 135 Hz and 6.6 dB
    (at 3,838 Hz: L = -2.16 dBr vs. R = -8.78 dBr).
  - Use `--notch-mode safe` (default, caps narrow 3.0-4.3 kHz notch fill to +1.50 dB), `--notch-mode original`
    (+4.26 dB regularized fill), or `--notch-mode zero` (0.00 dB across the notch).
"""

import argparse
import struct
import numpy as np

# ==============================================================================
# 1. EMBEDDED RAW MEASUREMENT (Crinacle GRAS 43AG-7: RA0402 + KB500X, L & R dBr)
# ==============================================================================
DT770_80OHM_RAW_LR = [
    (20.0,+3.82,+1.27,+2.54), (21.5,+4.28,+1.71,+3.00), (23.2,+4.78,+2.24,+3.51),
    (25.1,+5.32,+2.78,+4.05), (27.2,+5.84,+3.23,+4.54), (29.5,+6.27,+3.68,+4.98),
    (32.0,+6.69,+4.08,+5.38), (35.0,+7.09,+4.52,+5.80), (38.2,+7.36,+4.94,+6.15),
    (41.5,+7.69,+5.29,+6.49), (45.0,+8.04,+5.61,+6.82), (48.5,+8.20,+5.79,+6.99),
    (51.8,+8.27,+5.88,+7.07), (55.2,+8.26,+5.89,+7.07), (59.0,+8.14,+5.82,+6.98),
    (63.0,+7.88,+5.68,+6.78), (66.5,+7.52,+5.36,+6.44), (70.0,+7.04,+4.84,+5.94),
    (74.0,+6.39,+4.14,+5.27), (78.2,+5.72,+3.58,+4.65), (83.0,+4.95,+3.30,+4.13),
    (87.0,+4.49,+3.20,+3.85), (91.0,+4.51,+3.26,+3.88), (95.0,+4.61,+3.23,+3.92),
    (98.5,+4.36,+2.92,+3.64), (102.0,+3.77,+2.25,+3.01), (105.5,+3.15,+1.64,+2.40),
    (109.0,+2.80,+1.36,+2.08), (113.0,+2.75,+1.34,+2.05), (117.6,+2.88,+1.43,+2.16),
    (121.5,+2.99,+1.50,+2.24), (125.5,+2.99,+1.49,+2.24), (131.2,+2.82,+1.30,+2.06),
    (136.5,+2.46,+1.00,+1.73), (143.0,+1.78,+0.53,+1.16), (151.0,+0.57,-0.20,+0.18),
    (160.0,-0.96,-1.35,-1.15), (165.0,-2.12,-2.03,-2.07), (169.0,-3.04,-2.60,-2.82),
    (174.5,-4.02,-3.26,-3.64), (180.3,-4.91,-3.92,-4.42), (190.0,-6.09,-4.85,-5.47),
    (199.0,-6.48,-5.48,-5.98), (204.9,-6.45,-5.71,-6.08), (210.5,-6.22,-5.80,-6.01),
    (216.0,-5.84,-5.77,-5.81), (222.0,-5.39,-5.65,-5.52), (235.0,-4.56,-5.14,-4.85),
    (248.0,-3.76,-4.47,-4.11), (262.0,-3.02,-3.74,-3.38), (277.0,-2.36,-2.94,-2.65),
    (292.0,-1.89,-2.29,-2.09), (309.0,-1.50,-1.77,-1.63), (327.6,-1.17,-1.34,-1.25),
    (351.2,-1.10,-0.95,-1.03), (385.0,-0.81,-0.58,-0.69), (429.0,-0.65,-0.35,-0.50),
    (478.1,-0.65,-0.25,-0.45), (534.0,-0.68,-0.26,-0.47), (596.0,-0.66,-0.27,-0.47),
    (665.0,-0.48,-0.23,-0.36), (743.0,+0.04,+0.25,+0.15), (826.5,+0.60,+0.92,+0.76),
    (880.0,+0.98,+1.24,+1.11), (940.0,+1.24,+1.43,+1.33), (985.0,+1.15,+1.36,+1.26),
    (1032.0,+0.89,+1.24,+1.06), (1090.0,+0.54,+1.16,+0.85), (1151.0,+0.52,+1.27,+0.90),
    (1255.2,+1.14,+1.66,+1.40), (1335.0,+1.83,+1.72,+1.77), (1386.6,+2.16,+1.70,+1.93),
    (1501.4,+2.51,+1.88,+2.20), (1599.0,+3.00,+2.49,+2.74), (1660.0,+3.44,+2.99,+3.21),
    (1725.9,+3.89,+3.51,+3.70), (1885.0,+5.23,+4.90,+5.06), (2043.9,+6.76,+6.27,+6.52),
    (2221.0,+8.34,+7.28,+7.81), (2347.0,+7.87,+7.72,+8.29), (2479.0,+8.91,+7.87,+8.39),
    (2618.0,+8.78,+7.89,+8.34), (2766.0,+8.46,+7.73,+8.10), (2921.0,+7.91,+7.33,+7.62),
    (3086.0,+6.99,+6.56,+6.77), (3260.0,+5.34,+5.33,+5.34), (3443.0,+2.00,+2.98,+2.49),
    (3580.0,-1.68,-0.12,-0.90), (3703.0,-4.87,-4.87,-4.87), (3780.0,-3.70,-8.15,-5.93),
    (3838.4,-2.16,-8.78,-5.47), (4058.0,+4.41,+0.40,+2.41), (4287.0,+8.64,+5.44,+7.04),
    (4410.0,+9.88,+6.79,+8.34), (4528.0,+10.61,+7.70,+9.15), (4650.0,+10.89,+8.28,+9.59),
    (4783.0,+10.94,+8.72,+9.83), (5052.0,+10.75,+9.00,+9.87), (5337.0,+10.62,+9.07,+9.85),
    (5490.0,+10.30,+8.88,+9.59), (5637.0,+9.74,+8.46,+9.10), (5800.0,+8.49,+7.52,+8.00),
    (5955.0,+7.39,+6.78,+7.08), (6120.0,+7.96,+8.09,+8.03), (6290.0,+8.62,+9.41,+9.01),
    (6460.0,+8.82,+9.96,+9.39), (6644.0,+8.86,+10.10,+9.48), (6830.0,+8.58,+9.72,+9.15),
    (7018.0,+8.21,+9.21,+8.71), (7210.0,+7.90,+8.74,+8.32), (7414.0,+7.78,+8.45,+8.11),
    (7620.0,+7.98,+8.44,+8.21), (7831.0,+8.23,+8.49,+8.36), (8000.0,+8.41,+8.66,+8.54),
    (8155.0,+8.44,+8.74,+8.59), (8300.0,+7.70,+8.50,+8.10), (8500.0,+5.60,+7.38,+6.49),
    (8738.0,+1.78,+4.69,+3.24), (8980.0,-3.55,+0.12,-1.72), (9230.0,-6.91,-7.28,-7.10),
    (9550.0,-4.22,-7.10,-5.66), (10298.0,-0.84,-2.61,-1.72), (11490.0,+2.06,+1.13,+1.60),
    (12821.0,-2.79,-4.97,-3.88), (13672.6,-8.01,-7.05,-7.53), (15110.0,-14.59,-8.14,-11.36),
    (16860.0,+0.69,-1.55,-0.43), (18246.1,-4.65,-7.31,-5.98), (19955.5,-7.16,-6.62,-6.89)
]

HARMAN_2018_PTS = [
    (20.0,3.86),(25.1,4.02),(30.1,3.96),(35.3,3.84),(40.1,3.70),(48.0,3.44),(55.2,3.09),
    (60.4,2.88),(66.0,2.61),(72.2,2.33),(78.2,2.06),(85.5,1.75),(94.4,1.33),(100.3,1.04),
    (110.7,0.55),(121.1,0.07),(131.2,-0.34),(140.6,-0.66),(153.8,-1.00),(164.9,-1.27),(180.3,-1.62),
    (201.2,-1.96),(231.3,-2.08),(260.6,-1.95),(302.5,-1.65),(351.2,-1.32),(403.7,-1.12),
    (454.9,-0.98),(528.2,-0.75),(650.9,-0.40),(826.5,-0.16),(1008.4,0.02),(1136.3,0.35),
    (1255.2,0.85),(1386.6,1.53),(1501.4,2.11),(1658.5,3.04),(1850.4,4.29),(2003.7,5.29),
    (2235.4,6.44),(2494.0,7.39),(2782.5,8.04),(3104.3,8.44),(3361.5,8.58),(3568.3,8.56),
    (3825.7,8.36),(4061.1,8.10),(4310.9,7.73),(4621.9,7.17),(5004.9,6.44),(5365.9,5.89),
    (5753.0,5.39),(6046.4,5.06),(6482.6,4.47),(6881.4,3.96),(7304.7,3.37),(7677.4,2.84),
    (8069.0,2.33),(8480.6,1.77),(9002.3,1.00),(9556.1,0.10),(10043.6,-0.77),(11094.4,-2.59),
    (12013.6,-4.26),(13537.2,-6.46),(15103.1,-8.08),(16518.0,-10.07),(18065.5,-13.91),(19955.5,-22.92)
]

AUTOEQ_GEQ_STR = (
    "20 -3.1; 21 -3.4; 22 -3.6; 23 -3.9; 24 -4.2; 26 -4.8; 27 -5.1; 29 -5.4; 30 -5.7; 32 -6.1; "
    "34 -6.4; 36 -6.6; 38 -6.9; 40 -7.2; 43 -7.6; 45 -7.8; 48 -8.1; 50 -8.3; 53 -8.5; 56 -8.6; "
    "59 -8.6; 63 -8.7; 66 -8.5; 70 -8.2; 74 -7.7; 78 -7.2; 83 -6.9; 87 -6.6; 92 -7.0; 97 -7.4; "
    "103 -6.5; 109 -5.9; 115 -6.3; 121 -6.7; 128 -6.9; 136 -6.8; 143 -6.5; 151 -5.7; 160 -4.5; "
    "169 -3.2; 178 -2.1; 188 -1.1; 199 -0.5; 210 -0.5; 222 -1.0; 235 -1.7; 248 -2.4; 262 -3.0; "
    "277 -3.6; 292 -4.1; 309 -4.4; 326 -4.6; 345 -4.8; 364 -4.9; 385 -5.0; 406 -5.1; 429 -5.1; "
    "453 -5.1; 479 -5.0; 506 -4.9; 534 -4.8; 565 -4.7; 596 -4.6; 630 -4.6; 665 -4.6; 703 -4.7; "
    "743 -4.9; 784 -5.2; 829 -5.5; 875 -5.7; 924 -6.0; 977 -5.9; 1032 -5.6; 1090 -5.2; 1151 -5.0; "
    "1216 -5.0; 1284 -5.1; 1357 -5.0; 1433 -4.8; 1514 -4.6; 1599 -4.6; 1689 -4.7; 1784 -4.8; "
    "1885 -5.0; 1991 -5.3; 2103 -5.7; 2221 -5.9; 2347 -5.9; 2479 -5.6; 2618 -5.2; 2766 -4.7; "
    "2921 -4.0; 3086 -3.0; 3260 -1.6; 3443 -0.2; 3637 -0.5; 3842 -2.0; 4058 -3.4; 4287 -4.8; "
    "4528 -6.3; 4783 -7.5; 5052 -8.1; 5337 -8.5; 5637 -8.1; 5955 -7.4; 6290 -8.7; 6644 -9.4; "
    "7018 -9.1; 7414 -8.7; 7831 -8.1; 8272 -7.2; 8738 -6.3; 9230 -5.7; 9749 -5.3; 10298 -5.1; "
    "10878 -5.4; 11490 -5.7; 12137 -6.2; 12821 -6.8; 13543 -7.4; 14305 -8.2; 15110 -9.1; "
    "15961 -10.1; 16860 -11.2; 17809 -12.4; 18812 -13.8; 19871 -15.2"
)

# ==============================================================================
# 2. 128-NODE RIG-COMPENSATED CONTROL TABLE (DT 770 Pro 80 Ohm on GRAS 43AG-7)
# ==============================================================================
CTRL_PTS_DT770_80OHM = [
    (20.0,+0.20), (21.5,-0.13), (23.2,-0.45), (25.1,-0.84), (27.2,-1.25),
    (29.5,-1.58), (32.0,-1.93), (35.0,-2.15), (38.2,-2.39), (41.5,-2.87),
    (45.0,-3.29), (48.5,-3.58), (51.8,-3.83), (55.2,-3.98), (59.0,-4.05),
    (63.0,-4.03), (66.5,-3.86), (70.0,-3.52), (74.0,-3.02), (78.2,-2.59),
    (83.0,-2.27), (87.0,-2.17), (91.0,-2.39), (95.0,-2.62), (98.5,-2.46),
    (102.0,-2.03), (105.5,-1.62), (109.0,-1.45), (113.0,-1.59), (117.6,-1.94),
    (121.5,-2.19), (125.5,-2.35), (131.2,-2.39), (136.5,-2.26), (143.0,-1.88),
    (151.0,-1.11), (160.0,+0.00), (165.0,+0.93), (169.0,+1.46), (174.5,+2.12),
    (180.3,+2.80), (190.0,+3.69), (199.0,+4.05), (204.9,+4.10), (210.5,+4.01),
    (216.0,+3.87), (222.0,+3.47), (235.0,+2.79), (248.0,+2.11), (262.0,+1.44),
    (277.0,+0.83), (292.0,+0.37), (309.0,+0.03), (327.6,-0.23), (351.2,-0.30),
    (385.0,-0.50), (429.0,-0.55), (478.1,-0.45), (534.0,-0.26), (596.0,-0.09),
    (665.0,-0.03), (743.0,-0.41), (826.5,-0.92), (880.0,-1.21), (940.0,-1.38),
    (985.0,-1.26), (1032.0,-0.98), (1090.0,-0.62), (1151.0,-0.48), (1255.2,-0.55),
    (1335.0,-0.50), (1386.6,-0.40), (1501.4,-0.09), (1599.0,-0.04), (1660.0,-0.14),
    (1725.9,-0.21), (1885.0,-0.55), (2043.9,-1.02), (2221.0,-1.43), (2347.0,-1.43),
    (2479.0,-1.05), (2618.0,-0.66), (2766.0,-0.09), (2921.0,+0.61), (3086.0,+1.57),
    (3260.0,+2.78), (3443.0,+3.71), (3580.0,+4.08), (3703.0,+4.26), (3838.4,+4.02),
    (4058.0,+2.75), (4287.0,+0.68), (4410.0,-0.74), (4528.0,-1.72), (4650.0,-2.38),
    (4783.0,-2.83), (5052.0,-3.28), (5337.0,-3.59), (5490.0,-3.55), (5637.0,-3.21),
    (5800.0,-2.36), (5955.0,-1.70), (6120.0,-2.54), (6290.0,-3.72), (6460.0,-4.33),
    (6644.0,-4.45), (6830.0,-4.32), (7018.0,-4.13), (7210.0,-4.00), (7414.0,-4.01),
    (7620.0,-4.23), (7831.0,-4.58), (8000.0,-4.90), (8155.0,-5.04), (8300.0,-4.79),
    (8500.0,-3.74), (8738.0,-1.85), (8980.0,-1.35), (9230.0,-1.20), (9550.0,-1.15),
    (10298.0,-1.18), (11490.0,-1.55), (12821.0,-1.03), (13672.6,-0.55), (15110.0,-0.05),
    (16860.0,+0.00), (18246.1,+0.00), (19955.5,+0.00)
]


def apply_dt770_notch_mode(ctrl_pts, mode: str = "safe", safe_cap_db: float = 1.50):
    """
    Adjusts the 2.9 kHz - 4.3 kHz non-minimum-phase concha reflection notch region:
      - 'original': keeps the +4.26 dB @ 3,703 Hz regularized fill as-is.
      - 'safe'    : smoothly caps positive boosts inside 2.8-4.4 kHz to +1.50 dB so the
                    L/R-asymmetric concha null is bridged without creating a +4.3 dB peak.
      - 'zero'    : clamps positive boosts inside 2.8-4.4 kHz to 0.00 dB.
    """
    if mode == "original":
        return list(ctrl_pts)

    out = []
    for f, g in ctrl_pts:
        if 2800.0 <= f <= 4400.0 and g > 0.0:
            if mode == "zero":
                out.append((f, 0.0))
            else:
                g_soft = safe_cap_db * np.tanh(g / safe_cap_db)
                out.append((f, float(np.round(g_soft, 4))))
        else:
            out.append((f, float(g)))
    return out


def pchip_log_interp(f_nodes: np.ndarray, g_nodes: np.ndarray, f_query: np.ndarray) -> np.ndarray:
    """Pure-NumPy Fritsch-Carlson Monotonic Cubic Hermite (PCHIP) interpolator in log10(f) space."""
    x = np.log10(np.asarray(f_nodes, dtype=np.float64))
    y = np.asarray(g_nodes, dtype=np.float64)
    xq = np.log10(np.clip(np.asarray(f_query, dtype=np.float64), f_nodes[0], f_nodes[-1]))

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


def write_wav_ieee_float(filename: str, data: np.ndarray, fs: int = 48000, bits: int = 64) -> None:
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


def synthesize_minimum_phase_ir(
    ctrl_pts,
    fs: int = 48000,
    n_fft: int = 262144,
    n_taps: int = 65536,
    n_tail: int = 8192,
    headroom_db: float = -1.50,
    gauss_apodize: bool = False,
    kaiser_beta: float = 18.0,
):
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

    # 64-bit Kolmogorov / Hilbert-Cepstral Minimum-Phase Reconstruction
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
    unwindowed_floor_dbfs = float(20.0 * np.log10(np.max(np.abs(ir_f64[-128:])) + 1e-25))

    t_norm = np.arange(n_tail, dtype=np.float64) / n_tail
    kaiser_tail = bessel_i0(kaiser_beta * np.sqrt(np.maximum(0.0, 1.0 - t_norm**2))) / bessel_i0(kaiser_beta)
    ir_f64[-n_tail:] *= kaiser_tail

    windowed_floor_dbfs = float(20.0 * np.log10(np.max(np.abs(ir_f64[-100:])) + 1e-25))
    energy_pct = float(np.sum(ir_f64**2) / np.sum(ir_full_f64**2) * 100.0)
    return ir_f64, mag_lin, max_boost_db, preamp_db, energy_pct, unwindowed_floor_dbfs, windowed_floor_dbfs


def main():
    parser = argparse.ArgumentParser(
        description="Generate Beyerdynamic DT 770 Pro (80 Ohm) Minimum-Phase Convolver IRs."
    )
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate in Hz (default: 48000)")
    parser.add_argument(
        "--taps", type=int, default=65536, choices=[8192, 16384, 32768, 65536],
        help="Tap count for ir.wav (default: 65536; 16384 already reaches -147 dBFS unwindowed)"
    )
    parser.add_argument(
        "--notch-mode", choices=["safe", "original", "zero"], default="safe",
        help="How to handle the ~3.76 kHz non-minimum-phase concha notch: "
             "'safe' (default, caps narrow fill to +1.50 dB), 'original' (+4.26 dB fill), or 'zero' (0.0 dB)"
    )
    parser.add_argument(
        "--apodize", action="store_true",
        help="Enable Gaussian Nyquist roll-off (>20.2 kHz) for legacy brickwall-ringing sources"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib PNG verification plot")
    args = parser.parse_args()

    fs = args.fs
    active_ctrl_pts = apply_dt770_notch_mode(CTRL_PTS_DT770_80OHM, mode=args.notch_mode)
    notch_3703_db = dict(active_ctrl_pts).get(3703.0, 0.0)

    n_fft_mpv = max(65536, args.taps * 4)
    n_tail_mpv = args.taps // 8
    kaiser_beta = 18.0
    kaiser_form_db = kaiser_beta / 0.1102 + 8.7

    # 1. Generate 64-Bit Float (pcm_f64le) Offline / mpv Master Convolver (`ir.wav`)
    ir_mpv_f64, _, boost_mpv, pre_mpv, en_mpv, unwin_mpv, win_mpv = synthesize_minimum_phase_ir(
        active_ctrl_pts, fs=fs, n_fft=n_fft_mpv, n_taps=args.taps, n_tail=n_tail_mpv,
        headroom_db=-1.50, gauss_apodize=args.apodize, kaiser_beta=kaiser_beta
    )
    write_wav_ieee_float("ir.wav", ir_mpv_f64, fs=fs, bits=64)

    apod_desc = "Gaussian Apodizer (>20.2 kHz)" if args.apodize else "Non-Apodizing 0.00 dB Wire >15.8 kHz"
    print("=== [1] Offline / mpv 64-Bit Double Convolver (ir.wav) ===")
    print(f"  Headphone / Mode  : Beyerdynamic DT 770 Pro (80 Ohm) | 3.76 kHz Notch Mode: {args.notch_mode} ({notch_3703_db:+.2f} dB)")
    print(f"  Format            : 64-bit IEEE Float (pcm_f64le) | {fs:,} Hz | Mono")
    print(f"  FFT / Taps        : {n_fft_mpv:,} FFT ({fs/n_fft_mpv:.4f} Hz bin) -> {args.taps:,} taps ({args.taps/fs*1000:.1f} ms)")
    print(f"  HF Mode / Taper   : {apod_desc} + Half-Kaiser Fade (beta={kaiser_beta:.1f}, {kaiser_form_db:.1f} dB Kaiser formula)")
    print(f"  Total Preamp      : {pre_mpv:.2f} dB (Max Boost +{boost_mpv:.2f} dB, -1.50 dB true-peak headroom)")
    print(f"  Causal Energy     : {en_mpv:.10f}%")
    print(f"  Decay @ Truncation: {unwin_mpv:.2f} dBFS (unwindowed) | {win_mpv:.2f} dBFS (after half-Kaiser fade)")

    # 2. Generate 32-Bit Float (pcm_f32le) Real-Time PipeWire / Nuendo Convolvers
    n_fft_rt = 32768
    ir_rt_f64, mag_lin_rt, boost_rt, pre_rt, en_rt, unwin_rt, win_rt = synthesize_minimum_phase_ir(
        active_ctrl_pts, fs=fs, n_fft=n_fft_rt, n_taps=16384, n_tail=2048,
        headroom_db=-1.50, gauss_apodize=args.apodize, kaiser_beta=16.0
    )
    ir_rt_f32 = ir_rt_f64.astype(np.float32)
    mono_rt = f"dt770_80ohm_gras_surgical_mono_{fs//1000}k.wav"
    stereo_rt = f"dt770_80ohm_gras_surgical_stereo_{fs//1000}k.wav"
    write_wav_ieee_float(mono_rt, ir_rt_f32, fs=fs, bits=32)
    write_wav_ieee_float(stereo_rt, np.column_stack([ir_rt_f32, ir_rt_f32]), fs=fs, bits=32)

    err_rms_dbfs = 20.0 * np.log10(np.sqrt(np.mean((ir_rt_f64 - ir_rt_f32.astype(np.float64)) ** 2)))
    H_wav = np.fft.rfft(ir_rt_f32.astype(np.float64), n=n_fft_rt)
    spec_f32 = np.abs(H_wav)

    print("\n=== [2] Real-Time PipeWire / Nuendo 32-Bit Float Convolvers ===")
    print(f"  FFT / Taps        : {n_fft_rt:,} FFT ({fs/n_fft_rt:.4f} Hz bin) -> 16,384 taps ({16384/fs*1000:.1f} ms, Half-Kaiser beta=16.0)")
    print(f"  Decay @ Truncation: {unwin_rt:.2f} dBFS (unwindowed) | {win_rt:.2f} dBFS (after half-Kaiser fade)")
    print(f"  f32 vs f64 Error  : {err_rms_dbfs:.2f} dBFS RMS")
    print(f"  Written WAVs      : {mono_rt}, {stereo_rt}")

    if args.no_plot:
        return

    # 3. Render 4-Panel Verification Plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.ticker import FixedLocator, NullFormatter

        freqs_fft = np.fft.rfftfreq(n_fft_rt, d=1.0 / fs)
        f_plot = np.geomspace(20.0, 20000.0, 1200)
        f_raw = np.array([p[0] for p in DT770_80OHM_RAW_LR])
        raw_L = pchip_log_interp(f_raw, np.array([p[1] for p in DT770_80OHM_RAW_LR]), f_plot)
        raw_R = pchip_log_interp(f_raw, np.array([p[2] for p in DT770_80OHM_RAW_LR]), f_plot)
        raw_avg = pchip_log_interp(f_raw, np.array([p[3] for p in DT770_80OHM_RAW_LR]), f_plot)

        f_h = np.array([p[0] for p in HARMAN_2018_PTS])
        g_h = np.array([p[1] for p in HARMAN_2018_PTS])
        harm_plot = pchip_log_interp(f_h, g_h, f_plot)

        geq_pts = [tuple(map(float, item.strip().split())) for item in AUTOEQ_GEQ_STR.split(";")]
        f_geq = np.array([p[0] for p in geq_pts])
        g_geq = np.array([p[1] for p in geq_pts]) + 4.55
        autoeq_plot = np.interp(np.log10(f_plot), np.log10(f_geq), g_geq)

        wav_mag_db_unnorm = 20.0 * np.log10(np.maximum(spec_f32, 1e-15)) - pre_rt
        wav_phase_deg = np.degrees(np.angle(H_wav))
        wav_eq_plot = np.interp(f_plot, freqs_fft, wav_mag_db_unnorm)
        wav_phase_plot = np.interp(f_plot, freqs_fft, wav_phase_deg)
        ir_lin_centered = np.fft.fftshift(np.fft.irfft(mag_lin_rt, n=n_fft_rt))

        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#343b42"
        plt.rcParams["axes.linewidth"] = 0.9

        fig = plt.figure(figsize=(13.0, 11.5), dpi=150, facecolor="#151618")
        gs = gridspec.GridSpec(
            3, 2, height_ratios=[1.15, 1.15, 0.85], hspace=0.26, wspace=0.18,
            left=0.065, right=0.965, top=0.945, bottom=0.055
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

        style_freq_axis(ax1)
        ax1.set_title(
            f"FINAL VERIFICATION: Beyerdynamic DT 770 Pro (80 Ω) Convolver (Notch Mode: {args.notch_mode} | Preamp: {pre_rt:.2f} dB)",
            color="#ffffff", fontsize=12.0, fontweight="bold", pad=10
        )
        ax1.plot(f_plot, raw_L, color="#58a6ff", linestyle=":", linewidth=0.85, alpha=0.38, zorder=2,
                 label="Raw L / R Channels (L vs R: -2.5dB 20Hz Pad Leak & 6.6dB 3.84kHz Concha Null Shift)")
        ax1.plot(f_plot, raw_R, color="#58a6ff", linestyle=":", linewidth=0.85, alpha=0.38, zorder=2)
        ax1.plot(f_plot, raw_avg, color="#9ca3af", linestyle="--", linewidth=1.35, alpha=0.9, zorder=3,
                 label="Raw Beyerdynamic DT 770 Pro (80 Ω on Crinacle GRAS 43AG-7: RA0402 + KB500X)")
        ax1.plot(f_plot, harm_plot, color="#1f6feb", linestyle="-", linewidth=3.2, alpha=0.88, zorder=4,
                 label="Harman Over-Ear 2018 Target")
        ax1.plot(f_plot, raw_avg + autoeq_plot, color="#f85149", linestyle="-", linewidth=1.55, alpha=0.92, zorder=5,
                 label="Standard AutoEq (GraphicEQ.txt: +3.7dB un-cut 8.2kHz Beyer spike & +1.5dB 20Hz boost)")
        ax1.plot(f_plot, raw_avg + wav_eq_plot, color="#00e676", linestyle="-", linewidth=2.5, alpha=0.98, zorder=6,
                 label=f"Final WAV Convolver (Notch Mode: {args.notch_mode} | Unwindowed Floor: {unwin_rt:.1f} dBFS @ 16k)")
        ax1.set_ylabel("Acoustic Level (dBr)", color="#e6edf3", fontsize=10.5)
        ax1.set_ylim(-14.5, 11.5)
        ax1.set_yticks([-10, -5, 0, 5, 10])
        leg1 = ax1.legend(loc="lower left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.8)
        for text in leg1.get_texts():
            text.set_color("#d0d7de")

        style_freq_axis(ax2)
        ax2.plot(f_plot, -(raw_avg - harm_plot), color="#6e7681", linestyle=":", linewidth=1.0, alpha=0.65, zorder=2,
                 label="Raw Unsmoothed Error (-Error: +14.3dB non-min-phase concha null @ 3.78kHz)")
        ax2.plot(f_plot, autoeq_plot, color="#f5a623", linestyle="--", linewidth=2.0, alpha=0.95, zorder=4,
                 label="Standard AutoEq Curve (2.0 oct treble blur erases 8.2kHz cut + +1.5dB 20Hz boost)")
        ax2.plot(f_plot, wav_eq_plot, color="#00e676", linestyle="-", linewidth=2.6, alpha=0.98, zorder=5,
                 label=f"Final WAV Convolver EQ Curve (Peak +{boost_rt:.2f} dB -> Normalized w/ {pre_rt:.2f} dB Preamp)")
        ax2.set_ylabel("Filter Gain (dBr)", color="#e6edf3", fontsize=10.5)
        ax2.set_ylim(-10.8, 10.5)
        ax2.set_yticks([-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0])

        arrow_kw = dict(arrowstyle="->", color="#58d5e0", lw=1.15, shrinkA=2, shrinkB=3)
        arrow_green = dict(arrowstyle="->", color="#26e683", lw=1.2, shrinkA=2, shrinkB=3)

        ax2.annotate("1. EDT 770 V Velour Pad-Leak Fix\n(+0.2dB @ 20Hz vs AutoEq +1.5dB)",
                     xy=(20.5, 0.12), xytext=(21.5, 7.5), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("2. Coupled Port & Cup Suckout\n(-4.05dB @ 59Hz, +4.10dB @ 205Hz)",
                     xy=(59.0, -4.05), xytext=(82.0, 5.2), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("", xy=(204.9, 4.10), xytext=(168.0, 5.5), arrowprops=arrow_kw)
        ax2.annotate("3. 940Hz Cup & 2.35kHz Trim\n(-1.38dB @ 940Hz & -1.43dB @ 2.35kHz)",
                     xy=(940.0, -1.38), xytext=(420.0, 5.8), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)
        ax2.annotate("", xy=(2347.0, -1.43), xytext=(1120.0, 5.4), arrowprops=arrow_kw)
        ax2.annotate(f"4. 3.76kHz Non-Min-Phase Concha Zero\n({notch_3703_db:+.2f}dB in '{args.notch_mode}' mode;\navoids L/R & head-to-head glare)",
                     xy=(3703.0, notch_3703_db), xytext=(1500.0, 7.3), color="#26e683", fontsize=8.3, arrowprops=arrow_green)
        ax2.annotate("5. 5.3k, 6.6k & 8.16kHz Beyer Breakup Poles\n(-3.59dB, -4.45dB & -5.04dB Surgical Cuts)",
                     xy=(8155.0, -5.04), xytext=(2250.0, -9.2), color="#26e683", fontsize=8.3, arrowprops=arrow_green)
        ax2.annotate("", xy=(6644.0, -4.45), xytext=(4500.0, -7.2), arrowprops=arrow_green)
        ax2.annotate("", xy=(5337.0, -3.59), xytext=(3900.0, -7.2), arrowprops=arrow_green)
        ax2.annotate("6. >8.8kHz RA0402 Null Bypass & Wire\n(Ignores 9.2kHz null; 0.0dB >15.8kHz\nnon-apodizing zero-phase-lag wire)",
                     xy=(11490.0, -1.55), xytext=(5300.0, 4.0), color="#58d5e0", fontsize=8.3, arrowprops=arrow_kw)

        leg2 = ax2.legend(loc="lower left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.6)
        for text in leg2.get_texts():
            text.set_color("#d0d7de")

        style_freq_axis(ax3)
        ax3.set_title("Minimum-Phase Angle (Inverts Driver & Cup Phase)", color="#ffffff", fontsize=11.0, fontweight="bold", pad=8)
        ax3.plot(f_plot, wav_phase_plot, color="#c961de", linestyle="-", linewidth=2.0, alpha=0.95,
                 label="Minimum-Phase Response (deg)")
        ax3.set_xlabel("Frequency (Hz)", color="#e6edf3", fontsize=10.0)
        ax3.set_ylabel("Phase (deg)", color="#e6edf3", fontsize=10.0)
        ax3.set_ylim(-90, 90)
        ax3.set_yticks([-90, -45, 0, 45, 90])
        leg3 = ax3.legend(loc="upper left", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.6)
        for text in leg3.get_texts():
            text.set_color("#d0d7de")

        ax4.set_facecolor("#1b1f22")
        ax4.set_title("Impulse Response: Minimum-Phase vs. Linear-Phase Pre-Ringing", color="#ffffff", fontsize=11.0, fontweight="bold", pad=8)
        peak_mp = float(np.max(np.abs(ir_rt_f32)))
        t_mp_ms = np.arange(len(ir_rt_f32)) / fs * 1000.0
        t_lp_ms = (np.arange(n_fft_rt) - n_fft_rt // 2) / fs * 1000.0
        t_mp_plot = np.concatenate([np.linspace(-1.0, -1.0 / fs * 1000.0, 48), t_mp_ms[:250]])
        ir_mp_plot = np.concatenate([np.zeros(48), ir_rt_f32[:250] / peak_mp])
        lp_mask = (t_lp_ms >= -1.0) & (t_lp_ms <= 4.0)

        ax4.plot(t_lp_ms[lp_mask], ir_lin_centered[lp_mask] / peak_mp, color="#f85149", linestyle="--", linewidth=1.4, alpha=0.88,
                 label="Linear-Phase IR (Pre-Ringing before 0.0 ms)")
        ax4.plot(t_mp_plot, ir_mp_plot, color="#00e676", linestyle="-", linewidth=2.0, alpha=0.96,
                 label="Minimum-Phase IR (0.00 ms IR Peak Delay, Zero Pre-Ringing)")
        ax4.axvline(0.0, color="#388bfd", linestyle=":", linewidth=1.1, alpha=0.85)
        ax4.axhline(0.0, color="#6e7681", linewidth=0.8, alpha=0.6)
        ax4.set_xlim(-1.0, 4.0)
        ax4.set_ylim(-0.45, 1.08)
        ax4.set_xlabel("Time Relative to Transient Peak (ms)", color="#e6edf3", fontsize=10.0)
        ax4.set_ylabel("Normalized Amplitude", color="#e6edf3", fontsize=10.0)
        ax4.tick_params(axis="both", colors="#c9d1d9", labelsize=9.5)
        ax4.grid(True, which="major", color="#2d333b", linestyle="-", linewidth=0.75, alpha=0.9)
        leg4 = ax4.legend(loc="upper right", facecolor="#16191d", edgecolor="#343b42", framealpha=0.92, fontsize=8.6)
        for text in leg4.get_texts():
            text.set_color("#d0d7de")

        plt.savefig("dt770_80ohm_final_verification.png", dpi=150, facecolor="#151618")
        plt.close(fig)
        print("  Written Plot      : dt770_80ohm_final_verification.png")
    except ImportError:
        print("  [Note] matplotlib not installed; skipped PNG plot generation.")


if __name__ == "__main__":
    main()
