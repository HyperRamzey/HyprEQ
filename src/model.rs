//! Measurement database: targets, headphone models, and the derivation of a
//! control curve from a raw coupler trace.
//!
//! Everything about *which* headphones exist and *what* their measurements say lives
//! outside the code, in `measurements.json` (plus any `.json`/`.csv` dropped into
//! `models/`). This module is the loader and the validator; the two things it
//! computes rather than reads are:
//!
//! * [`normalize_raw_near_1khz`] — shift a raw trace so its 1 kHz average is `0.00 dBr`,
//!   preserving the L/R delta rather than folding it away.
//! * [`derive_ctrl_pts_from_raw`] — for a model that ships no hand-tuned knots: target
//!   minus raw, with the sub-bass pinna/pad-leak discount below 65 Hz and a
//!   log-cosine taper to `0.00 dB` into the HF bypass zone, rounded to 0.01 dB so the
//!   curve a user reads off a plot is the curve that was filtered.
//!
//! The `json!` blob in [`built_in_metadata`] is the built-in plot metadata for the three
//! shipped models — callout positions, axis limits, panel labels. It is a data table,
//! not logic: `--upgrade-json` writes it into `measurements.json` so a model can be
//! edited without touching Rust, and a model with no entry here gets
//! [`crate::plot::auto_annotations`] instead.

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// A measurement sample: `freq_hz` followed by one or more levels. Two columns are
/// `(freq, avg)`; four are `(freq, left, right, avg)`. The **last** column is the
/// average, which is the only one the DSP uses — the L/R pair is plot-only.
pub type Point = Vec<f64>;

pub const DEFAULT_DATA_FILE: &str = "measurements.json";
pub const DEFAULT_MODELS_DIR: &str = "models";

/// Alternate spellings of model keys. Applied only when the target exists.
pub const MODEL_ALIASES: [(&str, &str); 5] = [
    ("fiio_ft1_pro", "ft1pro"),
    ("ft1_pro", "ft1pro"),
    ("dt770", "dt770_80"),
    ("dt770_80ohm", "dt770_80"),
    ("jbl_770nc", "jbl770nc"),
];

/// Built-in per-model metadata (`short_title`, `notch_ref_hz`, `autoeq_offset_db`,
/// `gauss_apodize`, `plot_meta`), keyed by model. Mirrors
/// `hypreq.py::DEFAULT_MODEL_METADATA`.
#[must_use]
pub fn built_in_metadata(key: &str) -> Option<Value> {
    let all = json!({
      "jbl770nc": {
        "short_title": "JBL Live 770NC (Passive)",
        "notch_ref_hz": null,
        "autoeq_offset_db": 0.0,
        "gauss_apodize": false,
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
            {"text": "6. >9.1kHz Uncorrected Coupler Null Zone\n(0.0dB wire: ignores 9.75k & 11.2kHz nulls)", "xy": [10347.9, 0.0], "xytext": [5900.0, 4.8], "color": "cyan"}
          ]
        }
      },
      "dt770_80": {
        "short_title": "Beyerdynamic DT 770 Pro (80 Ω)",
        "notch_ref_hz": 3703.0,
        "autoeq_offset_db": 4.55,
        "gauss_apodize": false,
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
            {"text": "6. >8.8kHz Uncorrected Null & HF Wire\n(Bypasses 9.2kHz null; 0.0dB >15.8kHz\nleave coupler response unmodified)", "xy": [11490.0, -1.55], "xytext": [5300.0, 4.0], "color": "cyan"}
          ]
        }
      },
      "ft1pro": {
        "short_title": "FiiO FT1 Pro (Open-Back Planar)",
        "notch_ref_hz": 3839.0,
        "autoeq_offset_db": 4.30,
        "gauss_apodize": false,
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
            {"text": "6. >9.75kHz Uncorrected Coupler Null Zone\n(0.0dB wire: ignores 9.75k & 12.8kHz nulls)", "xy": [11300.0, 0.0], "xytext": [5500.0, 4.4], "color": "cyan"}
          ]
        }
      }
    });
    all.get(key).cloned()
}

/// One normalised headphone model, ready to synthesise.
#[derive(Debug, Clone)]
pub struct Model {
    pub key: String,
    pub name: String,
    pub short_title: String,
    pub file_prefix: String,
    pub rig_info: String,
    pub target_name: String,
    pub target_pts: Vec<(f64, f64)>,
    pub harm_offset: f64,
    pub gauss_apodize: bool,
    pub hf_bypass_hz: f64,
    pub notch_range_hz: Option<(f64, f64)>,
    pub notch_ref_hz: Option<f64>,
    pub safe_cap_db: f64,
    pub autoeq_geq_str: Option<String>,
    pub autoeq_offset_db: Option<f64>,
    pub plot_meta: Map<String, Value>,
    pub raw_pts: Vec<Point>,
    pub ctrl_pts: Vec<(f64, f64)>,
}

/// Raised for a bad database: unknown target, missing `raw_pts`, unreadable CSV, and
/// so on. The message says what to fix, in the terms the user typed.
#[derive(Debug, Clone)]
pub struct ModelError(pub String);

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ModelError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ModelError> {
    Err(ModelError(msg.into()))
}

/// `str(float(x))` for the numbers parsed out of JSON and CSV.
fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn f64_or(cfg: &Map<String, Value>, key: &str, default: f64) -> f64 {
    cfg.get(key).and_then(as_f64).unwrap_or(default)
}

fn str_opt(cfg: &Map<String, Value>, key: &str) -> Option<String> {
    cfg.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Sort measurement points by ascending frequency, last write winning on a repeated
/// frequency — `sort_and_dedup_points`.
#[must_use]
pub fn sort_and_dedup_points(pts: &[Point]) -> Vec<Point> {
    // A `BTreeMap<f64, _>` would order by total order and collapse NaN, and
    // `Ord` is not even implemented for `f64`; sorting keeps NumPy's behaviour,
    // where the sort is stable and the dict overwrite leaves the last write at a
    // repeated frequency standing.
    // A frequency must be a positive *finite* number. The reference's `f > 0.0` test lets
    // `inf` through, and one infinite knot turns every interpolated gain into NaN; a JSON
    // `1e400` or a hand-edited `inf` row is exactly the input that must not reach PCHIP.
    let mut kept: Vec<Point> = pts
        .iter()
        .filter(|pt| pt.first().is_some_and(|f| f.is_finite() && *f > 0.0))
        .cloned()
        .collect();
    kept.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let mut out: Vec<Point> = Vec::with_capacity(kept.len());
    for pt in kept {
        if out.last().is_some_and(|prev: &Point| prev[0] == pt[0]) {
            let n = out.len();
            out[n - 1] = pt;
        } else {
            out.push(pt);
        }
    }
    out
}

/// Sort `(freq, gain)` control/target knots the same way.
#[must_use]
pub fn sort_and_dedup_pairs(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let as_points: Vec<Point> = pts.iter().map(|&(f, g)| vec![f, g]).collect();
    sort_and_dedup_points(&as_points)
        .into_iter()
        .map(|p| (p[0], p[1]))
        .collect()
}

/// Shift a raw curve so its average level at `ref_hz` is `0.00 dBr`, moving every
/// column by the same amount so the L/R delta survives.
#[must_use]
pub fn normalize_raw_near_1khz(raw_pts: &[Point], ref_hz: f64) -> Vec<Point> {
    if raw_pts.is_empty() {
        return Vec::new();
    }
    let f_raw: Vec<f64> = raw_pts.iter().map(|p| p[0]).collect();
    let g_avg: Vec<f64> = raw_pts.iter().map(|p| *p.last().unwrap()).collect();
    let offset = crate::pchip::pchip_eval1(&f_raw, &g_avg, ref_hz);
    if offset.abs() < 1e-9 {
        return raw_pts.to_vec();
    }
    raw_pts
        .iter()
        .map(|p| {
            let mut q = Vec::with_capacity(p.len());
            q.push(p[0]);
            q.extend(p[1..].iter().map(|&v| v - offset));
            q
        })
        .collect()
}

/// Derive a rig-compensated control curve from raw measurement + target curve.
///
/// * below 65 Hz, subtract a Gaussian-in-`log2` leak discount (stiff-pinna or
///   asymmetric pad leakage is a fixture artefact, not a driver error);
/// * between `hf_taper_start_hz` and `hf_bypass_hz`, cosine-taper in `log2(f)` to
///   zero, so the coupler's half-wave cylinder null is approached, not attacked;
/// * above `hf_bypass_hz`, exactly `0.00 dB`;
/// * everything rounded to 0.01 dB, which is finer than a measurement and coarser
///   than a float, and keeps a hand-edited CSV and a derived curve in the same units.
#[must_use]
pub fn derive_ctrl_pts_from_raw(
    raw_pts: &[Point],
    target_pts: &[(f64, f64)],
    harm_offset: f64,
    hf_taper_start_hz: f64,
    hf_bypass_hz: f64,
    subbass_leak_discount_db: f64,
) -> Vec<(f64, f64)> {
    let f_raw: Vec<f64> = raw_pts.iter().map(|p| p[0]).collect();
    let f_tar: Vec<f64> = target_pts.iter().map(|p| p.0).collect();
    let g_tar: Vec<f64> = target_pts.iter().map(|p| p.1 + harm_offset).collect();
    let g_tar_on_raw = crate::pchip::pchip_log_interp(&f_tar, &g_tar, &f_raw);
    let log_ratio = (hf_bypass_hz / hf_taper_start_hz).log2();

    raw_pts
        .iter()
        .zip(&g_tar_on_raw)
        .map(|(p, &tar)| {
            let f = p[0];
            let e = tar - *p.last().unwrap();
            let val = if f < 65.0 && subbass_leak_discount_db > 0.0 {
                let z = (f.max(20.0) / 20.0).log2() / 0.75;
                let disc = subbass_leak_discount_db * (-z * z).exp();
                e - disc
            } else if f <= hf_taper_start_hz {
                e
            } else if f < hf_bypass_hz {
                let w = 0.5
                    * (1.0
                        + (std::f64::consts::PI * (f / hf_taper_start_hz).log2() / log_ratio)
                            .cos());
                e * w
            } else {
                0.0
            };
            (f, round2(val))
        })
        .collect()
}

/// Python's `round(x, 2)` on the 0.01 dB grid the control curves are written in.
///
/// `round()` and `np.round()` break ties to even, so 220.5 becomes 220 and 2.675 lands on
/// 2.67 rather than 2.68. Half-up rounding here moved a control knot by 0.01 dB relative
/// to the reference whenever a scaled value landed exactly on a half.
#[must_use]
pub fn round2(x: f64) -> f64 {
    (x * 100.0).round_ties_even() / 100.0
}

/// Round to 4 decimals, as `apply_notch_mode` does to a `tanh`-capped boost, with the
/// same ties-to-even rule as [`round2`].
#[must_use]
pub fn round4(x: f64) -> f64 {
    (x * 1e4).round_ties_even() / 1e4
}

/// `hypreq.py::load_csv_points` — 2- or 4-column CSV, `#` comments and blank rows
/// skipped, unparseable rows dropped.
///
/// Rows come from [`csv_records`], which follows the same quoting rules as Python's
/// `csv.reader`: a quoted field may contain commas, and `""` is a literal quote.
/// Splitting on `,` instead would drop every row of a quoted table, and a quoted
/// `"1,2"` cell would be read as two columns.
pub fn load_csv_points(csv_path: &Path) -> Result<Vec<Point>, ModelError> {
    let text = fs::read_to_string(csv_path)
        .map_err(|e| ModelError(format!("cannot read '{}': {e}", csv_path.display())))?;
    let mut pts = Vec::new();
    for row in csv_records(&text) {
        let Some(first) = row.first().map(|c| c.trim()) else {
            continue;
        };
        if first.is_empty() || first.starts_with('#') {
            continue;
        }
        // Only non-empty cells are converted, and a single non-numeric cell drops the
        // whole row — the reference's `try: tuple(float(x) ...) except ValueError: continue`.
        let vals: Option<Vec<f64>> = row
            .iter()
            .map(|c| c.trim())
            .filter(|c| !c.is_empty())
            .map(|c| c.parse::<f64>().ok())
            .collect();
        if vals.as_ref().is_some_and(|v| v.len() >= 2) {
            pts.push(vals.unwrap());
        }
    }
    Ok(pts)
}

/// Split CSV text into records with Python's `csv.reader` quoting rules: a field may be
/// wrapped in `"…"`, `""` inside a quoted field is a literal quote, and a quoted field may
/// contain the delimiter and line breaks.
#[must_use]
pub fn csv_records(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    // End a record only at a line break outside quotes; a record with no fields at all is
    // what a blank line looks like to `csv.reader`, and it is not a row.
    let end_row = |row: &mut Vec<String>, field: String| {
        row.push(field);
        row.len() > 1 || !row[0].is_empty()
    };
    while let Some(c) = chars.next() {
        if quoted {
            match c {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        field.push('"');
                    } else {
                        quoted = false;
                    }
                }
                other => field.push(other),
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            ',' => row.push(std::mem::take(&mut field)),
            '\n' | '\r' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                if end_row(&mut row, std::mem::take(&mut field)) {
                    rows.push(std::mem::take(&mut row));
                }
                // A blank line is not a record, but its empty field still has to go: keeping
                // it would prepend a blank cell to the next row and make `load_csv_points`
                // drop that row for starting with an empty field.
                row.clear();
            }
            other => field.push(other),
        }
    }
    // The final record counts even when the file has no trailing newline.
    if quoted || !field.is_empty() || row.len() > 1 {
        end_row(&mut row, field);
        rows.push(row);
    }
    rows
}

/// Validate and normalise one model entry. `cfg` is the raw JSON object; `base_dir`
/// resolves a relative `raw_csv` / `ctrl_csv`.
pub fn normalize_model_entry(
    key: &str,
    cfg: &Map<String, Value>,
    targets: &BTreeMap<String, Vec<(f64, f64)>>,
    base_dir: &Path,
) -> Result<Model, ModelError> {
    let defaults = built_in_metadata(key)
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();

    let target_name = cfg
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("harman_2018")
        .to_string();
    let Some(target_pts) = targets.get(&target_name) else {
        return err(format!(
            "Model '{key}' references unknown target '{target_name}'"
        ));
    };
    let target_pts = sort_and_dedup_pairs(target_pts);
    // PCHIP needs a segment, so a target with one knot (or none) is not a curve.
    if target_pts.len() < 2 {
        return err(format!(
            "Model '{key}': target '{target_name}' needs at least two points, got {}.",
            target_pts.len()
        ));
    }

    // raw_pts: an inline array, or a CSV named by `raw_csv` when no inline array is present.
    let raw_inline = cfg.get("raw_pts").and_then(Value::as_array);
    let raw_pts: Vec<Point> = if let Some(arr) = raw_inline.filter(|a| !a.is_empty()) {
        arr.iter()
            .map(|row| {
                row.as_array()
                    .and_then(|r| r.iter().map(as_f64).collect::<Option<Vec<f64>>>())
                    .ok_or_else(|| {
                        ModelError(format!("Model '{key}': raw_pts row is not numeric: {row}"))
                    })
            })
            .collect::<Result<Vec<_>, ModelError>>()?
    } else if let Some(rel) = cfg.get("raw_csv").and_then(Value::as_str) {
        load_csv_points(&base_dir.join(rel))?
    } else {
        return err(format!(
            "Model '{key}' has neither 'raw_pts' nor 'raw_csv'; a model needs one of them."
        ));
    };
    let raw_pts = sort_and_dedup_points(&raw_pts);
    if raw_pts.is_empty() {
        return err(format!(
            "Model '{key}' resolved to zero raw measurement points."
        ));
    }

    let ctrl_inline = cfg
        .get("ctrl_pts")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty());
    // `ctrl_csv` only counts when it names a file: the reference's `"ctrl_csv" in cfg`
    // test treated `"ctrl_csv": null` and `"ctrl_csv": ""` as a manual control curve, which
    // then switched off 1 kHz normalisation and pointed at a path that does not exist.
    let ctrl_csv = cfg
        .get("ctrl_csv")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string);
    let has_manual_ctrl = (ctrl_csv.is_some() && ctrl_inline.is_none()) || ctrl_inline.is_some();
    let should_norm_1k = cfg
        .get("normalize_1khz")
        .and_then(Value::as_bool)
        .unwrap_or(!has_manual_ctrl);
    let raw_pts = if should_norm_1k {
        normalize_raw_near_1khz(&raw_pts, 1000.0)
    } else {
        raw_pts
    };

    let harm_offset = f64_or(cfg, "harm_offset", 0.0);
    let hf_bypass_hz = f64_or(cfg, "hf_bypass_hz", 9500.0);
    let hf_taper_start_hz = f64_or(cfg, "hf_taper_start_hz", hf_bypass_hz * 0.88);

    let ctrl_pts: Vec<(f64, f64)> = if let Some(arr) = ctrl_inline {
        let pairs: Vec<(f64, f64)> = arr
            .iter()
            .map(|row| {
                let r = row.as_array().ok_or_else(|| {
                    ModelError(format!(
                        "Model '{key}': ctrl_pts row is not an array: {row}"
                    ))
                })?;
                let f = r.first().and_then(as_f64).ok_or_else(|| {
                    ModelError(format!(
                        "Model '{key}': ctrl_pts row needs a numeric frequency"
                    ))
                })?;
                let g = r.get(1).and_then(as_f64).ok_or_else(|| {
                    ModelError(format!("Model '{key}': ctrl_pts row needs a numeric gain"))
                })?;
                Ok((f, g))
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        sort_and_dedup_pairs(&pairs)
    } else if let Some(rel) = ctrl_csv {
        sort_and_dedup_points(&load_csv_points(&base_dir.join(rel))?)
            .into_iter()
            .map(|p| (p[0], *p.last().unwrap()))
            .collect()
    } else {
        derive_ctrl_pts_from_raw(
            &raw_pts,
            &target_pts,
            harm_offset,
            hf_taper_start_hz,
            hf_bypass_hz,
            f64_or(cfg, "subbass_leak_discount_db", 0.0),
        )
    };

    // Every control knot must be a finite pair, and PCHIP needs at least two of them;
    // without this check a malformed curve survives until it panics deep in the
    // interpolation or silently paints NaN across the whole filter.
    let ctrl_pts: Vec<(f64, f64)> = ctrl_pts
        .into_iter()
        .filter(|(f, g)| f.is_finite() && g.is_finite() && *f > 0.0)
        .collect();
    if ctrl_pts.len() < 2 {
        return err(format!(
            "Model '{key}': control curve needs at least two finite points above 0 Hz, got {}.",
            ctrl_pts.len()
        ));
    }

    let notch_range_hz = cfg
        .get("notch_range_hz")
        .filter(|v| !v.is_null())
        .and_then(Value::as_array)
        .and_then(|a| {
            if a.len() == 2 {
                Some((as_f64(&a[0])?, as_f64(&a[1])?))
            } else {
                None
            }
        });

    let notch_ref_hz = match cfg.get("notch_ref_hz") {
        Some(v) if !v.is_null() => as_f64(v),
        Some(_) => None,
        None => defaults
            .get("notch_ref_hz")
            .filter(|v| !v.is_null())
            .and_then(as_f64),
    };

    let mut plot_meta = defaults
        .get("plot_meta")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(pm) = cfg.get("plot_meta").and_then(Value::as_object) {
        for (k, v) in pm {
            plot_meta.insert(k.clone(), v.clone());
        }
    }
    // A model may hang `annotations` off the top level instead of inside `plot_meta`, and
    // that promotion is independent of whether `plot_meta` exists at all. The reference
    // tests truthiness, so `null` and `[]` leave the built-in callouts in place, and an
    // `annotations` key already inside the model's own `plot_meta` always wins.
    if let Some(ann) = cfg.get("annotations") {
        // Python truthiness, which is what the reference's `if cfg.get("annotations")` is.
        let truthy = match ann {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64().is_some_and(|v| v != 0.0),
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        };
        let owned_by_plot_meta = cfg
            .get("plot_meta")
            .and_then(Value::as_object)
            .is_some_and(|pm| pm.contains_key("annotations"));
        if truthy && !owned_by_plot_meta {
            plot_meta.insert("annotations".to_string(), ann.clone());
        }
    }

    let autoeq_offset_db = match cfg.get("autoeq_offset_db") {
        Some(v) if !v.is_null() => as_f64(v),
        Some(_) => None,
        None => defaults
            .get("autoeq_offset_db")
            .filter(|v| !v.is_null())
            .and_then(as_f64),
    };

    let name = str_opt(cfg, "name").unwrap_or_else(|| key.to_string());
    let short_title = str_opt(cfg, "short_title")
        .or_else(|| {
            defaults
                .get("short_title")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| name.clone());

    Ok(Model {
        key: key.to_string(),
        plot_meta,
        name,
        short_title,
        file_prefix: str_opt(cfg, "file_prefix").unwrap_or_else(|| key.to_string()),
        rig_info: str_opt(cfg, "rig_info")
            .unwrap_or_else(|| "IEC 60318-4 (711) Coupler".to_string()),
        target_name,
        target_pts,
        harm_offset,
        gauss_apodize: match cfg.get("gauss_apodize") {
            Some(v) if v.is_boolean() => v.as_bool().unwrap_or(false),
            Some(_) => defaults
                .get("gauss_apodize")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            None => defaults
                .get("gauss_apodize")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        hf_bypass_hz,
        notch_range_hz,
        notch_ref_hz,
        safe_cap_db: f64_or(cfg, "safe_cap_db", 1.50),
        autoeq_geq_str: str_opt(cfg, "autoeq_geq_str"),
        autoeq_offset_db,
        raw_pts,
        ctrl_pts,
    })
}

/// The loaded database: target curves and models, in file order for the models.
#[derive(Default)]
pub struct Database {
    pub targets: BTreeMap<String, Vec<(f64, f64)>>,
    /// Model key → model, kept in insertion order so `--headphone all` is
    /// reproducible and matches the Python's dict order.
    pub order: Vec<String>,
    pub models: Vec<Model>,
}

impl Database {
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.key == key)
    }

    /// Model keys in load order.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        self.order.iter().map(|s| s.as_str()).collect()
    }

    /// Resolve `-H` value (or alias) to a model key; `None` if unknown.
    #[must_use]
    pub fn resolve<'a>(&self, requested: &'a str) -> Option<&'a str> {
        if requested == "all" {
            return Some("all");
        }
        if self.get(requested).is_some() {
            return Some(requested);
        }
        let target = MODEL_ALIASES
            .iter()
            .find(|(a, _)| *a == requested)
            .map(|&(_, t)| t)?;
        self.get(target).map(move |_| target)
    }
}

fn targets_from(raw: &Value) -> BTreeMap<String, Vec<(f64, f64)>> {
    let mut out = BTreeMap::new();
    if let Some(map) = raw.get("targets").and_then(Value::as_object) {
        for (k, pts) in map {
            let pairs: Vec<(f64, f64)> = pts
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|row| {
                            let r = row.as_array()?;
                            Some((as_f64(r.first()?)?, as_f64(r.get(1)?)?))
                        })
                        .collect()
                })
                .unwrap_or_default();
            out.insert(k.clone(), pairs);
        }
    }
    out
}

/// Load `measurements.json` and then `models/` (JSON first, then bare CSVs whose
/// stem is not already taken).
pub fn load_measurement_database(
    data_file: &Path,
    models_dir: &Path,
) -> Result<Database, ModelError> {
    if !data_file.exists() {
        return err(format!(
            "Measurement database not found at '{}'. Place 'measurements.json' next to the \
             binary or pass --data-file.",
            data_file.display()
        ));
    }
    let text = fs::read_to_string(data_file)
        .map_err(|e| ModelError(format!("cannot read '{}': {e}", data_file.display())))?;
    let raw_db: Value = serde_json::from_str(&text)
        .map_err(|e| ModelError(format!("'{}' is not valid JSON: {e}", data_file.display())))?;

    let targets = targets_from(&raw_db);
    let base_dir = data_file.parent().unwrap_or(Path::new(".")).to_path_buf();

    let mut db = Database {
        targets: targets.clone(),
        order: Vec::new(),
        models: Vec::new(),
    };

    if let Some(map) = raw_db.get("models").and_then(Value::as_object) {
        for (key, cfg) in map {
            let cfg = cfg
                .as_object()
                .ok_or_else(|| ModelError(format!("Model '{key}' is not a JSON object")))?;
            let m = normalize_model_entry(key, cfg, &targets, &base_dir)?;
            db.order.push(m.key.clone());
            db.models.push(m);
        }
    }

    if models_dir.is_dir() {
        let mut jsons = Vec::new();
        let mut csvs = Vec::new();
        for entry in fs::read_dir(models_dir)
            .map_err(|e| ModelError(format!("cannot read '{}': {e}", models_dir.display())))?
        {
            let path = entry
                .map_err(|e| ModelError(format!("cannot read '{}': {e}", models_dir.display())))?
                .path();
            match path.extension().and_then(|e| e.to_str()) {
                Some("json") => jsons.push(path),
                Some("csv") => csvs.push(path),
                _ => {}
            }
        }
        jsons.sort();
        csvs.sort();

        for path in jsons {
            let text = fs::read_to_string(&path)
                .map_err(|e| ModelError(format!("cannot read '{}': {e}", path.display())))?;
            let cfg: Value = serde_json::from_str(&text)
                .map_err(|e| ModelError(format!("'{}' is not valid JSON: {e}", path.display())))?;
            let mut cfg = cfg
                .as_object()
                .ok_or_else(|| ModelError(format!("'{}' is not a JSON object", path.display())))?
                .clone();
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let key = cfg
                .get("key")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or(stem);
            if !cfg.contains_key("name") {
                cfg.entry("name".to_string())
                    .or_insert_with(|| Value::String(key.clone()));
            }
            let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            let m = normalize_model_entry(&key, &cfg, &targets, &parent)?;
            db.set_model(m);
        }

        for path in csvs {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if db.get(&stem).is_some() {
                continue;
            }
            let mut cfg = Map::new();
            cfg.insert(
                "name".to_string(),
                Value::String(format!("{stem} (Custom CSV Model)")),
            );
            cfg.insert(
                "raw_csv".to_string(),
                Value::String(
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string(),
                ),
            );
            let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            let m = normalize_model_entry(&stem, &cfg, &targets, &parent)?;
            db.set_model(m);
        }
    }

    Ok(db)
}

impl Database {
    fn set_model(&mut self, m: Model) {
        match self.models.iter().position(|x| x.key == m.key) {
            Some(i) => self.models[i] = m,
            None => {
                self.order.push(m.key.clone());
                self.models.push(m);
            }
        }
    }
}

/// Write the built-in plot metadata and callout annotations into `measurements.json`,
/// in place, so a user can edit them as data.
pub fn upgrade_measurements_json(data_file: &Path) -> Result<PathBuf, ModelError> {
    if !data_file.exists() {
        return err(format!(
            "Measurement database not found at '{}'.",
            data_file.display()
        ));
    }
    let text = fs::read_to_string(data_file)
        .map_err(|e| ModelError(format!("cannot read '{}': {e}", data_file.display())))?;
    let mut raw_db: Value = serde_json::from_str(&text)
        .map_err(|e| ModelError(format!("'{}' is not valid JSON: {e}", data_file.display())))?;

    if let Some(map) = raw_db.get_mut("models").and_then(Value::as_object_mut) {
        let keys: Vec<String> = map.keys().cloned().collect();
        for key in keys {
            let Some(defs) = built_in_metadata(&key) else {
                continue;
            };
            let Some(cfg) = map.get_mut(&key).and_then(Value::as_object_mut) else {
                continue;
            };
            let defs = defs.as_object().cloned().unwrap_or_default();
            for field in [
                "short_title",
                "notch_ref_hz",
                "autoeq_offset_db",
                "gauss_apodize",
            ] {
                if let Some(v) = defs.get(field) {
                    cfg.entry(field.to_string()).or_insert_with(|| v.clone());
                }
            }
            let mut pm = defs
                .get("plot_meta")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(user_pm) = cfg.get("plot_meta").and_then(Value::as_object) {
                for (k, v) in user_pm {
                    pm.insert(k.clone(), v.clone());
                }
            }
            cfg.insert("plot_meta".to_string(), Value::Object(pm));
        }
    }

    let out = serde_json::to_string_pretty(&raw_db)
        .map_err(|e| ModelError(format!("cannot serialize database: {e}")))?;
    fs::write(data_file, format!("{out}\n"))
        .map_err(|e| ModelError(format!("cannot write '{}': {e}", data_file.display())))?;
    Ok(data_file.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(v: &[f64]) -> Point {
        v.to_vec()
    }

    #[test]
    fn sort_and_dedup_keeps_the_last_write_per_frequency() {
        let pts = vec![
            pt(&[1000.0, 1.0]),
            pt(&[20.0, 3.0]),
            pt(&[1000.0, -2.0]),
            pt(&[0.0, 9.0]),
        ];
        let got = sort_and_dedup_points(&pts);
        assert_eq!(got.len(), 2, "freq 0 is dropped, 1000 is deduped");
        assert_eq!(got[0], pt(&[20.0, 3.0]));
        assert_eq!(
            got[1],
            pt(&[1000.0, -2.0]),
            "later row wins, like the Python dict"
        );
    }

    #[test]
    fn normalize_1khz_shifts_every_column_together() {
        let raw = vec![
            pt(&[500.0, 2.0, -1.0, 0.5]),
            pt(&[1000.0, 3.0, 0.0, 1.5]),
            pt(&[2000.0, 1.0, -2.0, -0.5]),
        ];
        let n = normalize_raw_near_1khz(&raw, 1000.0);
        assert!(
            (n[1][3] - 0.0).abs() < 1e-12,
            "average column is 0 at 1 kHz"
        );
        assert!(
            (n[1][1] - n[1][2] - 3.0).abs() < 1e-12,
            "L-R delta preserved verbatim"
        );
        assert_eq!(n[0][0], 500.0, "frequency column untouched");
    }

    #[test]
    fn derive_ctrl_tapers_to_zero_at_the_bypass_hz() {
        // A flat raw curve against a target that lifts 6 dB by 20 kHz, so the error
        // is real and rising through the taper window: without the taper the knots
        // above 8.8 kHz would keep growing instead of being walked to zero.
        let raw: Vec<Point> = (0..40)
            .map(|i| pt(&[20.0 * 10f64.powf(i as f64 / 39.0 * 3.0), 0.0]))
            .collect();
        let target = vec![(20.0, 0.0), (1000.0, 0.0), (20000.0, 6.0)];
        let ctrl = derive_ctrl_pts_from_raw(&raw, &target, 0.0, 8800.0, 10000.0, 0.0);

        // The knot frequencies are the measurement frequencies: the derivation adds
        // no grid of its own, it only decides the gain at each measurement point.
        assert_eq!(ctrl.len(), raw.len());

        // Reference values from the NumPy derivation on the same inputs.
        let at = |f: f64| -> f64 {
            ctrl.iter()
                .find(|(cf, _)| (cf - f).abs() < 1.0)
                .unwrap_or_else(|| panic!("no knot near {f}"))
                .1
        };
        assert_eq!(
            at(8249.252765802705),
            3.86,
            "still full error below the taper"
        );
        assert_eq!(at(9847.765263413483), 0.15, "inside the log-cosine taper");

        // Everything at or past the bypass frequency is a flat wire.
        let above: Vec<&(f64, f64)> = ctrl.iter().filter(|(f, _)| *f >= 10000.0).collect();
        assert!(above.len() >= 4, "grid must reach past the bypass");
        assert!(
            above.iter().all(|(_, g)| *g == 0.0),
            "0.00 dB above the bypass: {above:?}"
        );

        // The taper is monotonic through the window and every gain sits on the
        // 0.01 dB grid the audit is printed with.
        let in_taper = ctrl
            .iter()
            .filter(|(f, _)| *f > 8800.0 && *f < 10000.0)
            .count();
        assert!(in_taper >= 1, "taper window must contain a knot");
        assert!(
            ctrl.iter()
                .all(|(_, g)| (g * 100.0).round() - g * 100.0 < 1e-6),
            "0.01 dB grid"
        );
    }

    /// The sub-bass leak discount is a Gaussian in log2(f/20), so it bites hardest
    /// at 20 Hz and is gone by a few hundred Hz. Reference values from NumPy.
    #[test]
    fn derive_ctrl_subbass_leak_discount_is_a_log_gaussian() {
        let raw: Vec<Point> = [20.0, 25.0, 32.0, 50.0, 100.0, 1000.0, 20000.0]
            .iter()
            .map(|&f| pt(&[f, 0.0, 0.0, 0.0]))
            .collect();
        let target = vec![(20.0, 0.0), (1000.0, 0.0), (20000.0, 6.0)];
        let ctrl = derive_ctrl_pts_from_raw(&raw, &target, 1.5, 8800.0, 10000.0, 2.5);
        let got: Vec<f64> = ctrl.iter().map(|&(_, g)| g).collect();
        assert_eq!(got, vec![-1.0, -0.58, 0.4, 1.39, 1.5, 1.5, 0.0], "{got:?}");

        // With the discount switched off the same curve starts at the harm offset.
        let off = derive_ctrl_pts_from_raw(&raw, &target, 1.5, 8800.0, 10000.0, 0.0);
        assert_eq!(off[0].1, 1.5, "no discount, no dip at 20 Hz");
    }

    #[test]
    fn subbass_discount_is_gaussian_in_log2_and_zero_at_65hz() {
        let raw = vec![
            pt(&[20.0, 0.0]),
            pt(&[40.0, 0.0]),
            pt(&[64.0, 0.0]),
            pt(&[70.0, 0.0]),
        ];
        let target = vec![(20.0, 3.0), (1000.0, 3.0)];
        let ctrl = derive_ctrl_pts_from_raw(&raw, &target, 0.0, 8800.0, 9500.0, 2.5);
        assert!(
            ctrl[0].1 < ctrl[1].1 && ctrl[1].1 < ctrl[2].1,
            "monotone recovery toward the target"
        );
        assert!(
            ctrl[0].1 < 3.0 - 2.0,
            "20 Hz is discounted by most of 2.5 dB: {}",
            ctrl[0].1
        );
        assert_eq!(
            ctrl[3].1, 3.0,
            "above 65 Hz the discount is not applied at all"
        );
    }

    #[test]
    fn csv_loader_skips_junk_and_accepts_two_or_four_columns() {
        let dir = std::env::temp_dir().join(format!("hypreq_csv_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("m.csv");
        fs::write(
            &f,
            "# comment\n\n20, -1.5\n1000, 2.0, 0.5, 1.25\nbad, row\n20000, 3.0, 1.0, 2.0\n",
        )
        .unwrap();
        let pts = load_csv_points(&f).unwrap();
        // The reference keeps every row whose first field is not blank or `#` and
        // whose fields all parse as numbers with at least two of them — so 2-column,
        // 4-column and 3-column rows are all accepted, and "bad, row" is dropped.
        assert_eq!(pts.len(), 3);
        assert_eq!(pts[0], vec![20.0, -1.5]);
        assert_eq!(pts[1], vec![1000.0, 2.0, 0.5, 1.25]);
        assert_eq!(pts[2], vec![20000.0, 3.0, 1.0, 2.0]);
        fs::remove_dir_all(&dir).ok();
    }

    /// A scratch directory unique to this test: `cargo test` runs tests in parallel in one
    /// process, so a name keyed only on the pid would collide across tests.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hypreq_{}_{:?}",
            name,
            std::thread::current().name().unwrap_or("t")
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn targets(one: Vec<(f64, f64)>) -> BTreeMap<String, Vec<(f64, f64)>> {
        let mut t = BTreeMap::new();
        t.insert("harman_2018".to_string(), one);
        t
    }

    fn cfg(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    /// A raw curve that sits +6 dB above the target at 1 kHz, so a run with
    /// normalisation on is distinguishable from one with it off.
    fn offset_raw() -> Value {
        json!([[20.0, 6.0], [1000.0, 6.0], [10000.0, 6.0]])
    }

    fn flat_target() -> Vec<(f64, f64)> {
        vec![(20.0, 0.0), (1000.0, 0.0), (10000.0, 0.0)]
    }

    fn level_at_1k(m: &Model) -> f64 {
        m.raw_pts
            .iter()
            .find(|p| p[0] == 1000.0)
            .map(|p| *p.last().unwrap())
            .unwrap_or(f64::NAN)
    }

    #[test]
    fn a_quoted_csv_cell_stays_one_cell() {
        // The reference reads these files with csv.reader. Splitting on `,` instead tears
        // every quoted field at its commas, so a table written by Excel or a spreadsheet
        // export loses rows.
        let dir = scratch("quoted");
        let f = dir.join("q.csv");
        fs::write(
            &f,
            "# header\n20,\"-1.5\"\n1000,2.0,0.5,1.25\n20000,\"3.0\",\"1.0\",2.0\n",
        )
        .unwrap();
        let pts = load_csv_points(&f).unwrap();
        assert_eq!(pts.len(), 3, "every quoted row must survive: {pts:?}");
        assert_eq!(pts[0], vec![20.0, -1.5]);
        assert_eq!(pts[2], vec![20000.0, 3.0, 1.0, 2.0]);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn csv_records_follow_csv_reader_through_quotes_and_blank_lines() {
        // Every expectation here was taken from python's csv.reader on the same input.
        let rows = csv_records("a,\"b,2\",\"c\"\"x\"\nd,2\n");
        assert_eq!(rows[0], vec!["a", "b,2", "c\"x"]);
        assert_eq!(rows[1], vec!["d", "2"]);

        // A line break inside a quoted field belongs to the field, not to the record.
        let rows = csv_records("a,\"multi\nline\",3\n");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec!["a", "multi\nline", "3"]);

        // An unterminated quote swallows the remainder, just as the reader does.
        let rows = csv_records("a,\"b,2\",\"c\"\"\"\"\nd,2\n");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][2], "c\"\"\nd,2\n");

        // No trailing newline: the last record still counts. CRLF ends a record too.
        assert_eq!(csv_records("1,2\n3,4").len(), 2);
        assert_eq!(
            csv_records("1,1\r\n2,2\r\n"),
            vec![vec!["1", "1"], vec!["2", "2"]]
        );
        // A blank line is no record at all - and must not prepend an empty cell to the
        // row that follows it, which is what makes load_csv_points drop that row.
        assert_eq!(
            csv_records("1,1\n\n2,2\n"),
            vec![vec!["1", "1"], vec!["2", "2"]]
        );
    }

    #[test]
    fn null_or_blank_ctrl_csv_is_not_a_manual_curve() {
        // `"ctrl_csv": null` / `""` names no file, so it must not switch 1 kHz
        // normalisation off the way the reference's `"ctrl_csv" in cfg` test did.
        for blank in [Value::Null, json!(""), json!("   ")] {
            let mut c = cfg(json!({ "raw_pts": offset_raw(), "ctrl_csv": blank }));
            c.remove("ctrl_pts");
            let m =
                normalize_model_entry("t", &c, &targets(flat_target()), Path::new(".")).unwrap();
            assert!(
                level_at_1k(&m).abs() < 1e-9,
                "{blank} must not disable normalisation: {:?}",
                m.raw_pts
            );
        }
        // A real control CSV is a manual curve, and then normalisation stays off.
        let dir = scratch("ctrlcsv");
        fs::write(dir.join("ctrl.csv"), "20,1.0\n1000,1.0\n10000,1.0\n").unwrap();
        let c = cfg(json!({ "raw_pts": offset_raw(), "ctrl_csv": "ctrl.csv" }));
        let m = normalize_model_entry("t", &c, &targets(flat_target()), &dir).unwrap();
        assert!(
            (level_at_1k(&m) - 6.0).abs() < 1e-9,
            "a manual control curve keeps the raw measurement as measured"
        );
        assert_eq!(m.ctrl_pts[1], (1000.0, 1.0));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn top_level_annotations_promote_without_plot_meta() {
        let ann = json!([{"text": "note", "xy": [100.0, 1.0], "xytext": [120.0, 2.0]}]);
        let c = cfg(json!({ "raw_pts": offset_raw(), "annotations": ann }));
        let m = normalize_model_entry("t", &c, &targets(flat_target()), Path::new(".")).unwrap();
        assert_eq!(
            m.plot_meta.get("annotations"),
            Some(&ann),
            "a model with no plot_meta still gets its own callouts"
        );

        // plot_meta's own copy wins, and an empty or null list leaves the built-ins alone.
        let c = cfg(json!({
            "raw_pts": offset_raw(),
            "annotations": ann,
            "plot_meta": { "annotations": [] }
        }));
        let m = normalize_model_entry("t", &c, &targets(flat_target()), Path::new(".")).unwrap();
        assert_eq!(m.plot_meta.get("annotations"), Some(&json!([])));

        for empty in [json!([]), Value::Null] {
            let c = cfg(json!({ "raw_pts": offset_raw(), "annotations": empty }));
            let m =
                normalize_model_entry("t", &c, &targets(flat_target()), Path::new(".")).unwrap();
            assert!(
                !m.plot_meta.contains_key("annotations"),
                "{empty} is falsy in Python, so nothing is promoted"
            );
        }
    }

    #[test]
    fn a_model_that_cannot_make_a_curve_is_rejected_not_panicking() {
        let t1 = targets(vec![(1000.0, 0.0)]);
        let e = normalize_model_entry(
            "t",
            &cfg(json!({ "raw_pts": offset_raw() })),
            &t1,
            Path::new("."),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("at least two points"), "{e}");

        // One usable control knot cannot span the band.
        let e = normalize_model_entry(
            "t",
            &cfg(json!({ "raw_pts": offset_raw(), "ctrl_pts": [[1000.0, 1.0]] })),
            &targets(flat_target()),
            Path::new("."),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("control curve"), "{e}");

        // A control row that is not a number is refused, not skipped.
        let e = normalize_model_entry(
            "t",
            &cfg(json!({ "raw_pts": offset_raw(), "ctrl_pts": [[1000.0, "loud"]] })),
            &targets(flat_target()),
            Path::new("."),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("numeric gain"), "{e}");
    }

    #[test]
    fn an_infinite_frequency_never_reaches_the_interpolator() {
        // JSON cannot carry an infinity: an out-of-range literal is refused before it can
        // become a knot. A CSV can, because "inf" and "NaN" parse as floats in both the
        // reference and here — and one such knot turns every interpolated gain, and so the
        // whole filter, into NaN.
        let text = r#"{"raw_pts": [[20.0, 6.0], [1e999, 6.0]]}"#;
        assert!(serde_json::from_str::<Value>(text).is_err());

        let dir = scratch("inffreq");
        fs::write(
            &dir.join("raw.csv"),
            "20,6.0\ninf,6.0\nNaN,6.0\n-5,6.0\n1000,6.0\n10000,6.0\n",
        )
        .unwrap();
        let c = cfg(json!({
            "raw_csv": "raw.csv",
            "ctrl_pts": [[20.0, 0.0], [1000.0, 0.0], [10000.0, 0.0]],
        }));
        let m = normalize_model_entry("t", &c, &targets(flat_target()), &dir).unwrap();
        assert_eq!(m.raw_pts.len(), 3, "only the three usable rows remain");
        assert!(
            m.raw_pts.iter().all(|p| p.iter().all(|&v| v.is_finite())),
            "{:?}",
            m.raw_pts
        );

        // A control CSV can carry the same poison in the gain column; a knot that cannot
        // be represented cannot stay in the curve.
        fs::write(
            &dir.join("ctrl.csv"),
            "20,0.0\n1000,inf\n10000,NaN\n20000,1.0\n",
        )
        .unwrap();
        let c = cfg(json!({ "raw_csv": "raw.csv", "ctrl_csv": "ctrl.csv" }));
        let m = normalize_model_entry("t", &c, &targets(flat_target()), &dir).unwrap();
        assert!(
            m.ctrl_pts
                .iter()
                .all(|&(f, g)| f.is_finite() && g.is_finite()),
            "{:?}",
            m.ctrl_pts
        );
        assert_eq!(
            m.ctrl_pts.len(),
            2,
            "the two finite knots survive: {:?}",
            m.ctrl_pts
        );
        fs::remove_dir_all(&dir).ok();

        // Directly too: neither NaN nor +/-inf survives the sort.
        let junk = vec![
            pt(&[1000.0, 1.0]),
            vec![f64::INFINITY, 0.0],
            vec![f64::NAN, 0.0],
            vec![-1.0, 0.0],
            pt(&[20.0, 0.0]),
        ];
        let kept = sort_and_dedup_points(&junk);
        assert_eq!(kept.len(), 2, "only the two usable knots remain");
        assert!(kept.iter().all(|p| p[0].is_finite()));
    }

    #[test]
    fn the_ft1pro_callout_says_micrometre() {
        // "1μ Planar Excursion" is a truncated unit; the reference says 1μm.
        let meta = built_in_metadata("ft1pro").unwrap().to_string();
        assert!(meta.contains("1\u{3bc}m Planar Excursion"), "unit is wrong");
        assert!(
            !meta.contains("1\u{3bc} Planar"),
            "truncated unit came back"
        );
    }

    #[test]
    fn aliases_and_resolution() {
        let mut db = Database::default();
        for k in ["jbl770nc", "dt770_80", "ft1pro"] {
            let m = Model {
                key: k.to_string(),
                name: k.to_string(),
                short_title: k.to_string(),
                file_prefix: k.to_string(),
                rig_info: String::new(),
                target_name: "harman_2018".to_string(),
                target_pts: vec![(20.0, 0.0)],
                harm_offset: 0.0,
                gauss_apodize: false,
                hf_bypass_hz: 9500.0,
                notch_range_hz: None,
                notch_ref_hz: None,
                safe_cap_db: 1.5,
                autoeq_geq_str: None,
                autoeq_offset_db: None,
                plot_meta: Map::new(),
                raw_pts: vec![pt(&[20.0, 0.0])],
                ctrl_pts: vec![(20.0, 0.0)],
            };
            db.set_model(m);
        }
        assert_eq!(db.resolve("all"), Some("all"));
        assert_eq!(db.resolve("ft1pro"), Some("ft1pro"));
        assert_eq!(db.resolve("fiio_ft1_pro"), Some("ft1pro"));
        assert_eq!(db.resolve("dt770"), Some("dt770_80"));
        assert_eq!(db.resolve("nope"), None);
        assert_eq!(db.keys(), vec!["jbl770nc", "dt770_80", "ft1pro"]);
    }

    #[test]
    fn built_in_metadata_covers_the_shipped_models() {
        for k in ["jbl770nc", "dt770_80", "ft1pro"] {
            let m =
                built_in_metadata(k).unwrap_or_else(|| panic!("missing built-in metadata for {k}"));
            let pm = m
                .get("plot_meta")
                .and_then(Value::as_object)
                .expect("plot_meta");
            assert!(pm.contains_key("title_template"), "{k}");
            assert!(
                pm.get("annotations")
                    .and_then(Value::as_array)
                    .map_or(false, |a| !a.is_empty()),
                "{k} annotations"
            );
        }
        assert!(
            built_in_metadata("hd600").is_none(),
            "custom models have no built-in metadata"
        );
        // The dt770 concha callout anchors on the notch knot as a *string* token.
        let dt = built_in_metadata("dt770_80").unwrap();
        assert_eq!(dt.get("notch_ref_hz").and_then(as_f64), Some(3703.0));
        assert_eq!(dt.get("autoeq_offset_db").and_then(as_f64), Some(4.55));
    }

    #[test]
    fn rounding_helpers() {
        assert_eq!(round2(1.005), 1.0); // f64 1.005 is below the midpoint
        assert_eq!(round2(2.675), 2.68);
        assert_eq!(round4(-0.000_049_9), -0.0);
        assert_eq!(round4(1.234_56), 1.234_6);
        // Python's round()/np.round() break ties to even, and these two are exact binary
        // halves: 0.125 -> 0.12 and 0.375 -> 0.38. Half-up gave 0.13 and 0.38.
        assert_eq!(round2(0.125), 0.12, "a tie must go to the even cent");
        assert_eq!(
            round2(0.375),
            0.38,
            "the other side of the tie goes up to even"
        );
        assert_eq!(round2(-0.125), -0.12);
        assert_eq!(round4(0.0625), 0.0625);
    }

    #[test]
    fn the_shipped_database_keeps_its_insertion_order() {
        // serde_json's default map is a BTreeMap, which would sort the models
        // alphabetically. `preserve_order` is what keeps `--list-models`, the default
        // `-H all` sweep and the built-in metadata telling the same story as the file.
        let db =
            load_measurement_database(Path::new(DEFAULT_DATA_FILE), Path::new(DEFAULT_MODELS_DIR))
                .expect("the shipped measurements.json must load from the crate root");
        assert_eq!(
            db.keys(),
            vec!["jbl770nc", "dt770_80", "ft1pro"],
            "model order must be the file's key order, not alphabetical"
        );
        assert_eq!(db.resolve("all"), Some("all"));
        // The first key is what `--input` with `-H all` selects.
        assert_eq!(db.keys().first().copied(), Some("jbl770nc"));
    }

    #[test]
    fn a_json_object_keeps_the_order_it_was_written_in() {
        let text = r#"{"zebra": 1, "mango": 2, "apple": 3}"#;
        let v: Value = serde_json::from_str(text).unwrap();
        let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["zebra", "mango", "apple"]);

        // And a round trip through the upgrader must not reorder the file either.
        let dir = scratch("order");
        let f = dir.join("measurements.json");
        fs::write(
            &f,
            r#"{"targets": {"harman_2018": [[20.0, 1.0], [1000.0, 0.0]]},
                "models": {"zz_last": {"raw_pts": [[20.0, 0.0], [1000.0, 0.0]]},
                            "aa_first": {"raw_pts": [[20.0, 0.0], [1000.0, 0.0]]}}}"#,
        )
        .unwrap();
        upgrade_measurements_json(&f).unwrap();
        let after: Value = serde_json::from_str(&fs::read_to_string(&f).unwrap()).unwrap();
        let keys: Vec<String> = after["models"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            keys,
            vec!["zz_last", "aa_first"],
            "upgrade reordered the file"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
