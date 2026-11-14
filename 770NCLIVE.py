#!/usr/bin/env python3
"""
770NCLIVE.py
Rig-Compensated Minimum-Phase Mastering Convolver Generator for JBL Live 770NC (Passive Mode)

Data Sources & Attribution:
  - Raw Measurement : Filk (JBL Live 770NC passive on Clone GRAS 43AG / KB006x pinna + IEC 60318-4 711 coupler),
                      sourced via the AutoEq database (Jaakko Pasanen, MIT License; measurement data per Filk terms).
  - Target Curve    : Harman Over-Ear 2018 Target (Harman International).

Key Acoustic Design Choices (& Limitations):
  - Single-coupler measurements have inherent seat-to-seat and head-to-head variance (+/- 2-4 dB in sub-bass
    and >3 kHz pinna modes). This curve discounts sub-40 Hz stiff-pinna leakage on the KB006x plate, inverts
    the lumped-parameter 160 Hz / 455 Hz passive cup dipole and 6.88 kHz / 8.48 kHz diaphragm breakup peaks
    (preventing standard AutoEq's 2.0-octave treble smoother from bleeding adjacent dip errors into boosting
    those sibilance peaks), bridges smoothly across the 5.37 kHz concha notch without narrow peak boosting,
    and clamps >9.1 kHz coupler cylinder nulls to 0.0 dB.
"""

import argparse
import struct
import numpy as np

CTRL_PTS_JBL770NC = [
    (20.0,+0.12), (21.4,+0.09), (23.2,+0.03), (25.1,-0.05), (27.5,-0.16),
    (30.1,-0.30), (32.9,-0.44), (38.2,-0.71), (40.1,-0.82), (43.9,-1.02),
    (48.0,-1.21), (55.2,-1.38), (60.4,-1.43), (66.0,-1.57), (70.1,-1.83),
    (73.6,-2.16), (78.2,-2.65), (82.2,-3.04), (85.5,-3.27), (89.9,-3.38),
    (94.4,-3.46), (100.3,-3.77), (105.4,-3.93), (110.7,-4.09), (121.1,-4.63),
    (126.0,-4.96), (131.2,-5.36), (136.5,-5.85), (140.6,-6.22), (146.3,-6.63),
    (150.8,-6.87), (153.8,-6.97), (160.0,-7.04), (164.9,-7.02), (171.6,-6.95),
    (180.3,-6.76), (189.5,-6.45), (201.2,-5.95), (215.7,-5.29), (231.3,-4.56),
    (247.9,-3.70), (260.6,-3.06), (279.4,-2.19), (302.5,-1.21), (327.6,-0.25),
    (351.2,+0.50), (376.6,+1.11), (403.7,+1.67), (428.6,+2.24), (446.0,+2.54),
    (454.9,+2.60), (468.7,+2.54), (482.9,+2.36), (502.5,+2.08), (528.2,+1.76),
    (555.1,+1.46), (583.4,+1.17), (650.9,+0.36), (690.9,-0.05), (733.4,-0.48),
    (778.6,-0.94), (826.5,-1.40), (877.3,-1.80), (931.3,-2.15), (1008.4,-2.44),
    (1049.4,-2.45), (1092.0,-2.39), (1136.3,-2.27), (1194.3,-2.01), (1255.2,-1.66),
    (1319.3,-1.26), (1386.5,-0.85), (1442.8,-0.53), (1593.8,+0.46), (1658.5,+0.81),
    (1725.9,+1.15), (1795.9,+1.55), (1887.5,+2.09), (1964.2,+2.46), (2043.9,+2.75),
    (2148.2,+3.00), (2235.4,+3.18), (2373.0,+3.44), (2444.9,+3.47), (2518.9,+3.42),
    (2700.6,+3.25), (2810.3,+3.13), (2953.7,+3.06), (3104.3,+3.11), (3230.4,+3.22),
    (3361.5,+3.65), (3498.0,+4.44), (3640.1,+4.87), (3825.7,+4.84), (3941.7,+4.93),
    (4061.1,+5.00), (4184.1,+4.87), (4268.3,+4.81), (4397.6,+4.83), (4486.0,+4.51),
    (4621.9,+4.15), (4762.0,+3.78), (4906.2,+3.58), (5054.9,+3.61), (5208.1,+3.72),
    (5365.9,+3.70), (5528.5,+3.46), (5753.0,+2.88), (6046.4,+2.01), (6291.9,+1.61),
    (6547.4,-0.18), (6745.8,-1.76), (6881.4,-2.31), (7089.9,-1.92), (7304.7,-0.61),
    (7601.3,+0.00), (7910.0,-0.68), (8149.7,-1.91), (8313.5,-2.70), (8480.6,-3.16),
    (8651.0,-2.69), (8824.9,-1.11), (9092.3,-0.01), (9461.5,+0.00), (10347.9,+0.00),
    (12013.6,+0.00), (15103.0,+0.00), (19955.5,+0.00)
]


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

    # 64-bit Hilbert-Cepstral Minimum-Phase Reconstruction
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
    return ir_f64, max_boost_db, preamp_db, energy_pct, unwindowed_floor_dbfs, windowed_floor_dbfs


def main():
    parser = argparse.ArgumentParser(description="Generate JBL Live 770NC (Passive) Minimum-Phase Convolver IRs.")
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate in Hz (default: 48000)")
    parser.add_argument("--taps", type=int, default=65536, choices=[8192, 16384, 32768, 65536],
                        help="Tap count for ir.wav (default: 65536; 16384 already reaches -152.6 dBFS unwindowed)")
    parser.add_argument("--apodize", action="store_true",
                        help="Enable Gaussian Nyquist roll-off (>20.2 kHz) for legacy brickwall-ringing sources")
    args = parser.parse_args()

    n_fft_mpv = max(65536, args.taps * 4)
    n_tail_mpv = args.taps // 8
    kaiser_beta = 18.0
    kaiser_form_db = kaiser_beta / 0.1102 + 8.7

    ir_mpv_f64, boost_mpv, pre_mpv, en_mpv, unwin_db, win_db = synthesize_minimum_phase_ir(
        CTRL_PTS_JBL770NC, fs=args.fs, n_fft=n_fft_mpv, n_taps=args.taps, n_tail=n_tail_mpv,
        headroom_db=-1.50, gauss_apodize=args.apodize, kaiser_beta=kaiser_beta
    )
    write_wav_ieee_float("ir.wav", ir_mpv_f64, fs=args.fs, bits=64)

    apod_desc = "Gaussian Apodizer (>20.2 kHz)" if args.apodize else "Non-Apodizing 0.00 dB HF Wire (>9.1 kHz)"
    print("=== [1] Offline / mpv 64-Bit Double Convolver (ir.wav) ===")
    print(f"  Format            : 64-bit IEEE Float (pcm_f64le) | {args.fs:,} Hz | Mono")
    print(f"  FFT / Taps        : {n_fft_mpv:,} FFT ({args.fs/n_fft_mpv:.4f} Hz bin) -> {args.taps:,} taps ({args.taps/args.fs*1000:.1f} ms)")
    print(f"  HF Mode / Taper   : {apod_desc} + Half-Kaiser Fade (beta={kaiser_beta:.1f}, {kaiser_form_db:.1f} dB Kaiser formula)")
    print(f"  Total Preamp      : {pre_mpv:.2f} dB (Max Boost +{boost_mpv:.2f} dB, -1.50 dB true-peak headroom)")
    print(f"  Causal Energy     : {en_mpv:.10f}%")
    print(f"  Decay @ Truncation: {unwin_db:.2f} dBFS (unwindowed) | {win_db:.2f} dBFS (after half-Kaiser fade)")

    ir_rt_f64, _, _, _, unwin_rt_db, _ = synthesize_minimum_phase_ir(
        CTRL_PTS_JBL770NC, fs=args.fs, n_fft=32768, n_taps=16384, n_tail=2048,
        headroom_db=-1.50, gauss_apodize=args.apodize, kaiser_beta=16.0
    )
    ir_rt_f32 = ir_rt_f64.astype(np.float32)
    mono_rt = f"jbl_770nc_mastering_mono_{args.fs//1000}k.wav"
    stereo_rt = f"jbl_770nc_mastering_stereo_{args.fs//1000}k.wav"
    write_wav_ieee_float(mono_rt, ir_rt_f32, fs=args.fs, bits=32)
    write_wav_ieee_float(stereo_rt, np.column_stack([ir_rt_f32, ir_rt_f32]), fs=args.fs, bits=32)
    print("\n=== [2] Real-Time PipeWire / Nuendo 32-Bit Float Convolvers ===")
    print(f"  Written           : {mono_rt}, {stereo_rt} (16,384 taps, {unwin_rt_db:.2f} dBFS unwindowed floor)")


if __name__ == "__main__":
    main()
