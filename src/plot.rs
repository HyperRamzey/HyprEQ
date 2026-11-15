//! The 4-panel verification figure: `<file_prefix>_final_verification.png`.
//!
//! Four panels over a dark `#151618` canvas, in a 1.15 / 1.15 / 0.85 row layout:
//!
//! 1. **Acoustic level (dBr)** — raw L/R, raw average, the target plus the model's
//!    harmonic offset, a standard-AutoEq response, and the final convolver's response
//!    as it will actually measure, with the coupler-null bypass zone shaded.
//! 2. **Filter gain (dBr)** — the unsmoothed raw error, the AutoEq curve, and the
//!    magnitude the generated FIR really has, re-read off its own spectrum, so the plot
//!    shows what was built rather than what was asked for; plus the callouts.
//! 3. **Minimum-phase phase angle** — the Hilbert-cepstral phase of that same spectrum.
//! 4. **Impulse response** — minimum-phase against the linear-phase IR of the same
//!    magnitude, so the pre-ringing that minimum-phase removes is visible.
//!
//! The curves are computed in [`PlotData::build`] from the same inputs the report
//! printed; [`PlotData::render`] only draws. The split is deliberate: the numbers behind
//! the figure are unit-testable (geometry, the AutoEq blur, the callout selection),
//! while the PNG is checked by eye — `plotters` has no dash patterns, so line *style*
//! differences of the reference figure become line *width* differences here.

use crate::error::{Error, Result};
use crate::fft::{self, Complex64};
use crate::model::Model;
use crate::pchip;

/// Figure size in pixels: the reference figure's 13.0 x 11.5 in at 150 dpi.
pub const FIG_W: u32 = 1950;
/// Figure height in pixels.
pub const FIG_H: u32 = 1725;

/// Space above the first panel, for nothing but margin.
const M_TOP: u32 = 95;
/// Plot-area left edge; a panel's y labels occupy the 62 px to its left.
const M_LEFT: u32 = 127;
/// Plot-area right edge.
const M_RIGHT: u32 = 1882;
/// Pixels reserved at the top of a band for its caption.
const TITLE_H: u32 = 34;
/// Pixels reserved at the bottom of a band for x ticks and the x label.
const X_LABEL_H: u32 = 52;
/// Pixels reserved left of a plot area for y ticks and the y label.
const Y_LABEL_W: u32 = 62;
/// Height of the panel-1 plot area.
const ROW1_H: u32 = 440;
/// Height of the panel-2 plot area.
const ROW2_H: u32 = 440;
/// Height of the two bottom plot areas.
const ROW3_H: u32 = 330;
/// Vertical gap between the bands.
const ROW_GAP: u32 = 81;
/// Width of one bottom-row plot area.
const COL_W: u32 = 786;
/// Horizontal gap between the bottom-row panels.
const COL_GAP: u32 = 121;

/// Figure background.
const FIG_BG: Rgb = Rgb(0x15, 0x16, 0x18);
/// Panel background (`axes.facecolor`).
const PANEL_BG: Rgb = Rgb(0x1B, 0x1F, 0x22);
/// Tick and axis-label text.
const TEXT: Rgb = Rgb(0xC9, 0xD1, 0xD9);
/// Panel titles.
const TITLE: Rgb = Rgb(0xFF, 0xFF, 0xFF);
/// Bypass-zone shading.
const BYPASS: Rgb = Rgb(0x30, 0x36, 0x3D);
/// Raw average.
const RAW: Rgb = Rgb(0x9C, 0xA3, 0xAF);
/// Raw L/R traces.
const RAW_LR: Rgb = Rgb(0x58, 0xA6, 0xFF);
/// Target curve.
const TARGET: Rgb = Rgb(0x1F, 0x6F, 0xEB);
/// Standard-AutoEq response.
const AUTOEQ_RED: Rgb = Rgb(0xF8, 0x51, 0x49);
/// Standard-AutoEq curve on the gain panel.
const AUTOEQ_AMBER: Rgb = Rgb(0xF5, 0xA6, 0x23);
/// The generated convolver's own response.
const WAV_GREEN: Rgb = Rgb(0x00, 0xE6, 0x76);
/// Minimum-phase phase trace.
const PHASE: Rgb = Rgb(0xC9, 0x61, 0xDE);
/// Raw unsmoothed error.
const RAW_ERR: Rgb = Rgb(0x6E, 0x76, 0x81);
/// Callout arrows and text, cyan variant.
const ARROW_CYAN: Rgb = Rgb(0x58, 0xD5, 0xE0);
/// Callout arrows and text, green variant.
const ARROW_GREEN: Rgb = Rgb(0x26, 0xE6, 0x83);
/// The 0 dBr reference line.
const ZERO_LINE: Rgb = Rgb(0x6E, 0x76, 0x81);
/// Major grid lines.
const GRID: Rgb = Rgb(0x2D, 0x33, 0x3B);
/// The 0 ms reference line in the impulse panel.
const ZERO_MS: Rgb = Rgb(0x38, 0x8B, 0xFD);

/// A colour triple, kept out of plotters' types so the palette stays readable.
#[derive(Clone, Copy, Debug)]
pub struct Rgb(u8, u8, u8);

impl Rgb {
    /// The same colour as a `plotters` colour.
    #[must_use]
    pub fn color(self) -> plotters::prelude::RGBColor {
        plotters::prelude::RGBColor(self.0, self.1, self.2)
    }
}

/// Major x ticks shared by the three frequency-axis panels.
pub const MAJOR_FREQS: [f64; 10] = [20.0, 50.0, 100.0, 200.0, 500.0, 1e3, 2e3, 5e3, 1e4, 2e4];
/// Default y ticks for panel 1.
pub const AX1_YTICKS: [f64; 5] = [-10.0, -5.0, 0.0, 5.0, 10.0];
/// Default y ticks for panel 2.
pub const AX2_YTICKS: [f64; 9] = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0];
/// Y ticks for the phase panel, fixed by the reference.
pub const PHASE_YTICKS: [f64; 5] = [-90.0, -45.0, 0.0, 45.0, 90.0];

/// One callout on panel 2: text, the point it points at, where the text sits.
#[derive(Clone, Debug)]
pub struct Annotation {
    /// Callout body.
    pub text: String,
    /// Frequency of the arrow head, in Hz.
    pub xy_x: f64,
    /// Level of the arrow head, in dBr.
    pub xy_y: f64,
    /// Text anchor, `(Hz, dBr)`, at the text's upper-left corner.
    pub xytext: (f64, f64),
    /// `true` for the green arrow style.
    pub green: bool,
}

/// Everything the figure draws, computed once from the model and the real-time IR.
#[derive(Clone, Debug)]
pub struct PlotData {
    /// Panel 1 caption.
    pub title: String,
    /// Legend entries of panel 1, in drawing order.
    pub bypass_label: String,
    /// Raw L/R legend (only used when the measurement has both columns).
    pub lr_label: String,
    /// Raw average legend.
    pub raw_label: String,
    /// Target legend.
    pub target_label: String,
    /// AutoEq legend.
    pub autoeq_label: String,
    /// Generated-convolver legend.
    pub wav_label: String,
    /// Panel 2 caption.
    pub gain_title: String,
    /// Panel 2 legends.
    pub raw_err_label: String,
    /// AutoEq legend on panel 2.
    pub autoeq_gain_label: String,
    /// Convolver legend on panel 2.
    pub wav_gain_label: String,
    /// Panel 3 caption.
    pub phase_title: String,
    /// Panel 3 legend.
    pub phase_label: String,
    /// Panel 4 caption.
    pub ir_title: String,
    /// Linear-phase IR legend.
    pub lin_ir_label: String,
    /// Minimum-phase IR legend.
    pub mp_ir_label: String,
    /// Common frequency grid: 1200 points geometrically spaced over 20 Hz .. 20 kHz.
    pub f_plot: Vec<f64>,
    /// Raw L and R interpolated on [`PlotData::f_plot`]; empty for 2-column raw data.
    pub raw_l: Vec<f64>,
    /// Raw R trace.
    pub raw_r: Vec<f64>,
    /// Raw average (last column of the measurement, which is the L/R average).
    pub raw_avg: Vec<f64>,
    /// Target plus the model's harmonic offset.
    pub harm_plot: Vec<f64>,
    /// Standard-AutoEq gain.
    pub autoeq: Vec<f64>,
    /// Gain the generated FIR actually has, read back off its own spectrum.
    pub wav_eq: Vec<f64>,
    /// Its minimum-phase phase, in degrees.
    pub wav_phase: Vec<f64>,
    /// Panel 4: linear-phase IR, `(t_ms, amplitude)`.
    pub lin_t_ms: Vec<f64>,
    /// Linear-phase IR amplitudes.
    pub lin_ir: Vec<f64>,
    /// Panel 4: minimum-phase IR trace, starting left of 0 ms with zeros.
    pub mp_t_ms: Vec<f64>,
    /// Minimum-phase IR amplitudes, normalised to a unit peak.
    pub mp_ir: Vec<f64>,
    /// Frequency above which the model is deliberately left uncorrected.
    pub hf_bypass_hz: f64,
    /// Right edge of the shaded bypass zone.
    pub bypass_to_hz: f64,
    /// Panel 1 y limits.
    pub ax1_ylim: (f64, f64),
    /// Panel 2 y limits.
    pub ax2_ylim: (f64, f64),
    /// Panel 1 y ticks (from `plot_meta.ax1_yticks`, or [`AX1_YTICKS`]).
    pub ax1_yticks: Vec<f64>,
    /// Panel 2 y ticks.
    pub ax2_yticks: Vec<f64>,
    /// Callouts on panel 2.
    pub annotations: Vec<Annotation>,
}

/// Inputs of [`PlotData::build`]: the real-time IR and the scalars its report printed.
pub struct PlotInputs<'a> {
    /// The 32-bit real-time IR as written, widened to `f64`.
    pub ir_rt: &'a [f64],
    /// Linear-phase magnitude of the same control curve, for panel 4's reference IR.
    pub mag_lin: &'a [f64],
    /// FFT length the spectrum is read back at.
    pub n_fft_rt: usize,
    /// Sample rate of the IR.
    pub fs: u32,
    /// Peak gain of the control curve before normalisation, in dB.
    pub boost_rt: f64,
    /// Preamp applied to the IR, in dB.
    pub pre_rt: f64,
    /// Unwindowed tail decay floor, in dBFS.
    pub unwin_rt: f64,
    /// Control knots after the notch policy was applied.
    pub active_ctrl: &'a [(f64, f64)],
    /// Name of the notch policy in effect, for labels.
    pub notch_mode: &'a str,
}

/// A solid stroke style.
fn stroke(c: Rgb, width: u32) -> plotters::style::ShapeStyle {
    use plotters::style::Color;
    plotters::style::ShapeStyle {
        color: c.color().to_rgba(),
        filled: false,
        stroke_width: width,
    }
}

/// A filled style with explicit coverage, used for the bypass zone and legend boxes.
fn fill(c: Rgb, alpha: f64) -> plotters::style::ShapeStyle {
    plotters::style::ShapeStyle {
        color: plotters::prelude::RGBAColor(c.0, c.1, c.2, alpha),
        filled: true,
        stroke_width: 0,
    }
}

/// An invisible style, to switch the mesh's minor grid off.
fn clear() -> plotters::style::ShapeStyle {
    fill(Rgb(0, 0, 0), 0.0)
}

/// Common log-frequency axis decoration: fixed tick count, labels, grid.
macro_rules! freq_mesh {
    ($chart:expr, $y_desc:expr, $yticks:expr) => {{
        use plotters::prelude::*;
        let ticks: Vec<f64> = $yticks.clone();
        $chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(ticks.len().max(2))
            .x_desc("Frequency (Hz)")
            .y_desc($y_desc)
            .axis_desc_style(("sans-serif", DESC_FONT).into_font().color(&TEXT.color()))
            .label_style(("sans-serif", TICK_FONT).into_font().color(&TEXT.color()))
            .axis_style(stroke(EDGE, 1))
            .bold_line_style(stroke(GRID, 1))
            .light_line_style(clear())
            .x_label_formatter(&|f| format_hz(*f as f64))
            .y_label_formatter(&|v| fmt_tick(*v))
            .draw()
            .map_err(dv)
    }};
}

/// The shaded coupler-null bypass zone, drawn before any trace.
macro_rules! bypass_span {
    ($chart:expr, $self:expr) => {{
        use plotters::prelude::*;
        let lo = $self.ax1_ylim.0.min($self.ax2_ylim.0);
        let hi = $self.ax1_ylim.1.max($self.ax2_ylim.1);
        $chart
            .draw_series(vec![Rectangle::new(
                [($self.hf_bypass_hz, lo), ($self.bypass_to_hz, hi)],
                fill(BYPASS, 0.35),
            )])
            .map_err(dv)?;
        Ok::<(), crate::error::Error>(())
    }};
}

/// One polyline in `(freq, level)` data coordinates, optionally registered in the legend.
macro_rules! trace {
    ($chart:expr, $label:expr, $xs:expr, $ys:expr, $color:expr, $w:expr) => {{
        use plotters::prelude::*;
        let xs: &[f64] = $xs;
        let ys: &[f64] = $ys;
        if xs.is_empty() || ys.is_empty() {
            Ok(())
        } else {
            let pts: Vec<(f64, f64)> = xs.iter().zip(ys).map(|(&x, &y)| (x, y)).collect();
            let anno = $chart
                .draw_series(LineSeries::new(pts, stroke($color, $w as u32)))
                .map_err(dv)?;
            if let Some(l) = $label {
                anno.label(l.to_string());
            }
            Ok::<(), crate::error::Error>(())
        }
    }};
}

/// A reference line that must not appear in the legend.
macro_rules! guide {
    ($chart:expr, $pts:expr, $color:expr, $w:expr) => {{
        use plotters::prelude::*;
        $chart
            .draw_series(LineSeries::new($pts.to_vec(), stroke($color, $w as u32)))
            .map_err(dv)?;
    }};
}

/// The legend of everything registered with `.label(..)`.
macro_rules! legend {
    ($chart:expr, $pos:expr) => {{
        use plotters::prelude::*;
        $chart
            .configure_series_labels()
            .position($pos)
            .background_style(fill(LEGEND_BG, 0.92))
            .border_style(stroke(EDGE, 1))
            .label_font(("sans-serif", TICK_FONT).into_font().color(&TEXT.color()))
            .draw()
            .map_err(dv)
    }};
}

/// The panel-2 callouts: an arrow from each text block to its point on the curve, in
/// data coordinates so the model's `plot_meta` positions are honoured exactly.
macro_rules! annotations {
    ($chart:expr, $self:expr) => {{
        use plotters::prelude::*;
        let (y_lo, y_hi) = $self.ax2_ylim;
        let dy = (y_hi - y_lo) * 0.045;
        for a in &$self.annotations {
            let color = if a.green { ARROW_GREEN } else { ARROW_CYAN };
            let head_x =
                a.xy_x - dx_toward(a.xy_x, a.xytext.0) * (1.0 + (a.xy_x - a.xytext.0).abs() * 0.02);
            $chart
                .draw_series(vec![PathElement::new(
                    vec![(a.xytext.0, a.xytext.1 - dy * 0.4), (a.xy_x, a.xy_y)],
                    stroke(color, 2),
                )])
                .map_err(dv)?;
            $chart
                .draw_series(vec![PathElement::new(
                    vec![
                        (a.xy_x, a.xy_y),
                        (head_x, a.xy_y + dy * 0.55),
                        (head_x, a.xy_y - dy * 0.55),
                        (a.xy_x, a.xy_y),
                    ],
                    fill(color, 1.0),
                )])
                .map_err(dv)?;
            for (i, line) in a.text.split('\n').enumerate() {
                $chart
                    .draw_series(vec![Text::new(
                        line.to_string(),
                        (a.xytext.0, a.xytext.1 - dy * (i as f64 + 0.8)),
                        ("sans-serif", ANNOT_FONT).into_font().color(&color.color()),
                    )])
                    .map_err(dv)?;
            }
        }
        Ok::<(), crate::error::Error>(())
    }};
}

/// Unit step from `from` towards `to`, so an arrowhead always points back at the text.
fn dx_toward(from: f64, to: f64) -> f64 {
    (to - from).signum()
}

impl PlotData {
    /// Compute every curve and label of the figure.
    pub fn build(model: &Model, in_: &PlotInputs) -> Self {
        let fs = f64::from(in_.fs);
        let f_plot = fft::geomspace(20.0, 20000.0, 1200);

        // The spectrum of the IR as written, on the reference's rfft grid, with the
        // preamp taken back out so panel 1/2 show the acoustic response, not the gain
        // the file carries.
        let spec: Vec<Complex64> = fft::rfft(in_.ir_rt, in_.n_fft_rt);
        let freqs = fft::rfftfreq(in_.n_fft_rt, fs);
        let mag_db: Vec<f64> = spec
            .iter()
            .map(|c| 20.0 * c.norm().max(1e-15).log10() - in_.pre_rt)
            .collect();
        let phase_deg: Vec<f64> = spec.iter().map(|c| c.im.atan2(c.re).to_degrees()).collect();
        let wav_eq = fft::interp_grid(&f_plot, &freqs, &mag_db);
        let wav_phase = fft::interp_grid(&f_plot, &freqs, &phase_deg);

        let f_tar: Vec<f64> = model.target_pts.iter().map(|p| p.0).collect();
        let g_tar: Vec<f64> = model.target_pts.iter().map(|p| p.1).collect();
        let harm_plot: Vec<f64> = pchip::pchip_log_interp(&f_tar, &g_tar, &f_plot)
            .iter()
            .map(|v| v + model.harm_offset)
            .collect();

        let f_raw: Vec<f64> = model.raw_pts.iter().map(|p| p[0]).collect();
        let g_raw: Vec<f64> = model.raw_pts.iter().map(|p| *p.last().unwrap()).collect();
        let raw_avg = pchip::pchip_log_interp(&f_raw, &g_raw, &f_plot);
        let (raw_l, raw_r) = if model.raw_pts.first().is_some_and(|p| p.len() >= 4) {
            let gl: Vec<f64> = model.raw_pts.iter().map(|p| p[1]).collect();
            let gr: Vec<f64> = model.raw_pts.iter().map(|p| p[2]).collect();
            (
                pchip::pchip_log_interp(&f_raw, &gl, &f_plot),
                pchip::pchip_log_interp(&f_raw, &gr, &f_plot),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let raw_eq_needed: Vec<f64> = harm_plot
            .iter()
            .zip(&raw_avg)
            .map(|(&h, &r)| h - r)
            .collect();
        let autoeq = match model.autoeq_geq_str.as_deref() {
            Some(s) => autoeq_from_geq(s, model.autoeq_offset_db, in_.active_ctrl),
            None => gaussian_autoeq(&f_plot, &raw_eq_needed),
        };

        // Panel 4: the linear-phase IR of the same magnitude (centred, so its energy
        // straddles 0 ms and pre-rings) against the minimum-phase IR, which starts at
        // 0 ms. Both normalised to the minimum-phase peak.
        let ir_lin_centered = fftshift(&crate::ir::linear_phase_ir(in_.mag_lin, in_.n_fft_rt));
        let peak_mp = in_
            .ir_rt
            .iter()
            .fold(0.0f64, |a, &v| a.max(v.abs()))
            .max(f64::from(f32::EPSILON));
        let (lin_t_ms, lin_ir): (Vec<f64>, Vec<f64>) = (0..in_.n_fft_rt)
            .map(|i| ((i as f64 - (in_.n_fft_rt / 2) as f64) / fs * 1000.0, i))
            .filter(|&(t, _)| (-1.0..=4.0).contains(&t))
            .map(|(t, i)| (t, ir_lin_centered[i] / peak_mp))
            .unzip();
        // A run of zeros left of 0 ms, so the minimum-phase trace visibly begins at the
        // transient instead of being interpolated across the empty half of the panel.
        let lead_n = 48usize;
        let mp_t_ms: Vec<f64> = (0..lead_n)
            .map(|i| -1.0 + (i as f64 + 1.0) / lead_n as f64)
            .chain((0..250.min(in_.ir_rt.len())).map(|i| i as f64 / fs * 1000.0))
            .collect();
        let mp_ir: Vec<f64> = vec![0.0; lead_n]
            .into_iter()
            .chain(in_.ir_rt.iter().take(250).map(|&v| v / peak_mp))
            .collect();

        let meta = &model.plot_meta;
        let str_meta = |k: &str| -> Option<String> {
            meta.get(k).and_then(|v| v.as_str()).map(str::to_string)
        };
        let notch_db = crate::ir::notch_db_at(in_.active_ctrl, model.notch_ref_hz);
        // The reference's labels are Python format templates; only these five fields
        // ever appear in them.
        let subst = |s: String| {
            s.replace("{notch_mode}", in_.notch_mode)
                .replace("{pre_rt:.2f}", &format!("{:.2}", in_.pre_rt))
                .replace("{unwin_rt:.1f}", &format!("{:.1}", in_.unwin_rt))
                .replace("{boost_rt:.2f}", &format!("{:.2}", in_.boost_rt))
                .replace("{notch_db:.2f}", &format!("{:.2}", notch_db))
        };

        let has_notch = model.notch_range_hz.is_some();
        let default_title = if has_notch {
            format!(
                "VERIFICATION: {} Convolver (Notch Mode: {{notch_mode}} | Preamp: {:.2} dB)",
                model.short_title, in_.pre_rt
            )
        } else {
            format!(
                "VERIFICATION: {} Convolver (Preamp: {:.2} dB | Unwindowed Floor: {:.1} dBFS)",
                model.short_title, in_.pre_rt, in_.unwin_rt
            )
        };
        let default_wav = if has_notch {
            format!(
                "Final WAV Convolver (Notch Mode: {{notch_mode}} | Unwindowed Floor: {:.1} dBFS @ 16k)",
                in_.unwin_rt
            )
        } else {
            format!(
                "Final WAV Convolver (Minimum-Phase PCHIP FIR | Unwindowed Floor: {:.1} dBFS @ 16k)",
                in_.unwin_rt
            )
        };

        let ax1_ylim = meta_pair(meta, "ax1_ylim").unwrap_or_else(|| {
            let summed = add(&raw_avg, &wav_eq);
            (
                (min_of(&summed) - 2.0).min(-14.5),
                (max_of(&harm_plot) + 2.5)
                    .max(max_of(&raw_avg) + 2.0)
                    .max(11.5),
            )
        });
        let ax2_ylim = meta_pair(meta, "ax2_ylim").unwrap_or_else(|| {
            (
                (min_of(&wav_eq) - 2.5).min(-10.8),
                (max_of(&wav_eq) + 3.2).max(10.5),
            )
        });

        let annotations = meta_annotations(meta, notch_db)
            .unwrap_or_else(|| build_auto_annotations(&f_plot, &wav_eq, model.hf_bypass_hz));

        Self {
            title: subst(str_meta("title_template").unwrap_or(default_title)),
            bypass_label: str_meta("bypass_label").unwrap_or_else(|| {
                format!(
                    "Uncorrected >{:.1} kHz Coupler Null Zone (Clamped to 0.0 dB)",
                    model.hf_bypass_hz / 1000.0
                )
            }),
            lr_label: str_meta("lr_label").unwrap_or_else(|| "Raw L / R Channels".to_string()),
            raw_label: str_meta("raw_label").unwrap_or_else(|| format!("Raw {}", model.name)),
            target_label: str_meta("target_label").unwrap_or_else(|| {
                if model.target_name == "harman_2018" {
                    "Harman Over-Ear 2018 Target".to_string()
                } else {
                    format!("Target Curve ({})", model.target_name)
                }
            }),
            autoeq_label: str_meta("autoeq_ax1_label")
                .unwrap_or_else(|| "Standard AutoEq Response".to_string()),
            wav_label: subst(str_meta("wav_ax1_label").unwrap_or(default_wav)),
            gain_title: "Filter Gain: Raw Error vs. Standard AutoEq vs. Final WAV Convolver"
                .to_string(),
            raw_err_label: str_meta("raw_err_ax2_label")
                .unwrap_or_else(|| "Raw Unsmoothed Error".to_string()),
            autoeq_gain_label: str_meta("autoeq_ax2_label")
                .unwrap_or_else(|| "Standard AutoEq Curve".to_string()),
            wav_gain_label: format!(
                "Final WAV Convolver EQ Curve (Peak +{:.2} dB -> Normalized w/ {:.2} dB Preamp)",
                in_.boost_rt, in_.pre_rt
            ),
            phase_title: "Filter Minimum-Phase Response (Hilbert-Cepstral Phase Angle)".into(),
            phase_label: "Filter Minimum-Phase Response (deg)".to_string(),
            ir_title: "Impulse Response: Minimum-Phase vs. Linear-Phase Pre-Ringing".into(),
            lin_ir_label: "Linear-Phase IR (Pre-Ringing before 0.0 ms)".to_string(),
            mp_ir_label: "Minimum-Phase IR (0.00 ms IR Peak Delay, Zero Pre-Ringing)".to_string(),
            f_plot,
            raw_l,
            raw_r,
            raw_avg,
            harm_plot,
            autoeq,
            wav_eq,
            wav_phase,
            lin_t_ms,
            lin_ir,
            mp_t_ms,
            mp_ir,
            hf_bypass_hz: model.hf_bypass_hz,
            bypass_to_hz: 20000.0,
            ax1_ylim,
            ax2_ylim,
            ax1_yticks: meta_ticks(meta, "ax1_yticks", &AX1_YTICKS),
            ax2_yticks: meta_ticks(meta, "ax2_yticks", &AX2_YTICKS),
            annotations,
        }
    }

    /// Write `<file_prefix>_final_verification.png`; returns the path written.
    pub fn render(&self, prefix: &str) -> Result<std::path::PathBuf> {
        use plotters::prelude::*;

        let path = std::path::PathBuf::from(format!("{prefix}_final_verification.png"));
        // The bitmap backend writes the file when it is dropped, i.e. when `root` and
        // everything split from it go out of scope at the end of this function.
        let written = path.clone();
        let root = BitMapBackend::new(&path, (FIG_W, FIG_H)).into_drawing_area();
        root.fill(&FIG_BG.color())
            .map_err(|e| Error::msg(format!("figure background failed: {e}")))?;

        // Bands peel off the top; the inter-row gaps are drawn as throwaway bands so
        // the figure background shows through them.
        let (_, rest) = root.split_vertically(M_TOP);
        let (band1, rest) = rest.split_vertically(TITLE_H + ROW1_H + X_LABEL_H);
        let (_, rest) = rest.split_vertically(ROW_GAP);
        let (band2, rest) = rest.split_vertically(TITLE_H + ROW2_H + X_LABEL_H);
        let (_, rest) = rest.split_vertically(ROW_GAP);
        let (band3, _) = rest.split_vertically(TITLE_H + ROW3_H + X_LABEL_H);

        // Peel the left margin off first: with `y_label_area_size(Y_LABEL_W)` the plot
        // area then begins exactly at M_LEFT and ends exactly at M_RIGHT.
        let (row1, _) = band1.split_horizontally(M_LEFT - Y_LABEL_W);
        let (band1, _) = row1.split_horizontally(Y_LABEL_W + (M_RIGHT - M_LEFT));
        let (row2, _) = band2.split_horizontally(M_LEFT - Y_LABEL_W);
        let (band2, _) = row2.split_horizontally(Y_LABEL_W + (M_RIGHT - M_LEFT));

        self.draw_level(&band1).map_err(io_err(&path, "panel 1"))?;
        self.draw_gain(&band2).map_err(io_err(&path, "panel 2"))?;

        // Bottom row: left margin, [ y labels | plot ], gap, [ y labels | plot ].
        let (row3, _) = band3.split_horizontally(M_LEFT - Y_LABEL_W);
        let (col1, rest) = row3.split_horizontally(Y_LABEL_W + COL_W);
        let (_, rest) = rest.split_horizontally(COL_GAP);
        let (col2, _) = rest.split_horizontally(Y_LABEL_W + COL_W);
        self.draw_phase(&col1).map_err(io_err(&path, "panel 3"))?;
        self.draw_impulse(&col2).map_err(io_err(&path, "panel 4"))?;

        Ok(written)
    }

    /// Panel 1: acoustic level — raw against target, AutoEq, and the built FIR.
    fn draw_level(&self, band: &BitmapArea<'_>) -> PlotResult {
        use plotters::prelude::*;
        let (y0, y1) = self.ax1_ylim;
        let mut chart = ChartBuilder::on(band)
            .margin(0u32)
            .margin_top(TITLE_H)
            .x_label_area_size(X_LABEL_H)
            .y_label_area_size(Y_LABEL_W)
            .caption(
                self.title.as_str(),
                ("sans-serif", TITLE_FONT).into_font().color(&TITLE.color()),
            )
            .build_cartesian_2d((20.0f64..20000.0f64).log_scale(), y0..y1)
            .map_err(dv)?;
        chart.plotting_area().fill(&PANEL_BG.color()).map_err(dv)?;
        let _ = freq_mesh!(chart, "Acoustic Level (dBr)", self.ax1_yticks);

        let _ = bypass_span!(chart, self);
        let _ = trace!(chart, None::<String>, &self.f_plot, &self.raw_l, RAW_LR, 2);
        let _ = trace!(chart, None::<String>, &self.f_plot, &self.raw_r, RAW_LR, 2);
        let _ = trace!(
            chart,
            Some(&self.raw_label),
            &self.f_plot,
            &self.raw_avg,
            RAW,
            3
        );
        let _ = trace!(
            chart,
            Some(&self.target_label),
            &self.f_plot,
            &self.harm_plot,
            TARGET,
            5
        );
        let autoeq_resp = add(&self.raw_avg, &self.autoeq);
        let _ = trace!(
            chart,
            Some(&self.autoeq_label),
            &self.f_plot,
            &autoeq_resp,
            AUTOEQ_RED,
            4
        );
        let wav_resp = add(&self.raw_avg, &self.wav_eq);
        let _ = trace!(
            chart,
            Some(&self.wav_label),
            &self.f_plot,
            &wav_resp,
            WAV_GREEN,
            5
        );
        legend!(&mut chart, SeriesLabelPosition::LowerLeft)
    }

    /// Panel 2: filter gain, plus the callouts.
    fn draw_gain(&self, band: &BitmapArea<'_>) -> PlotResult {
        use plotters::prelude::*;
        let (y0, y1) = self.ax2_ylim;
        let mut chart = ChartBuilder::on(band)
            .margin(0u32)
            .margin_top(TITLE_H)
            .x_label_area_size(X_LABEL_H)
            .y_label_area_size(Y_LABEL_W)
            .caption(
                self.gain_title.as_str(),
                ("sans-serif", GAIN_TITLE_FONT)
                    .into_font()
                    .color(&TITLE.color()),
            )
            .build_cartesian_2d((20.0f64..20000.0f64).log_scale(), y0..y1)
            .map_err(dv)?;
        chart.plotting_area().fill(&PANEL_BG.color()).map_err(dv)?;
        let _ = freq_mesh!(chart, "Filter Gain (dBr)", self.ax2_yticks);
        let _ = bypass_span!(chart, self);

        let err: Vec<f64> = self
            .harm_plot
            .iter()
            .zip(&self.raw_avg)
            .map(|(&h, &r)| h - r)
            .collect();
        let _ = trace!(
            chart,
            Some(&self.raw_err_label),
            &self.f_plot,
            &err,
            RAW_ERR,
            2
        );
        let _ = trace!(
            chart,
            Some(&self.autoeq_gain_label),
            &self.f_plot,
            &self.autoeq,
            AUTOEQ_AMBER,
            4
        );
        let _ = trace!(
            chart,
            Some(&self.wav_gain_label),
            &self.f_plot,
            &self.wav_eq,
            WAV_GREEN,
            5
        );
        legend!(&mut chart, SeriesLabelPosition::LowerLeft)?;
        annotations!(&mut chart, self)
    }

    /// Panel 3: the minimum-phase phase angle of the same spectrum.
    fn draw_phase(&self, band: &BitmapArea<'_>) -> PlotResult {
        use plotters::prelude::*;
        let mut chart = ChartBuilder::on(band)
            .margin(0u32)
            .margin_top(TITLE_H)
            .x_label_area_size(X_LABEL_H)
            .y_label_area_size(Y_LABEL_W)
            .caption(
                self.phase_title.as_str(),
                ("sans-serif", SMALL_TITLE_FONT)
                    .into_font()
                    .color(&TITLE.color()),
            )
            .build_cartesian_2d((20.0f64..20000.0f64).log_scale(), -90.0f64..90.0)
            .map_err(dv)?;
        chart.plotting_area().fill(&PANEL_BG.color()).map_err(dv)?;
        let _ = freq_mesh!(chart, "Phase (deg)", PHASE_YTICKS.to_vec());
        let _ = trace!(
            chart,
            Some(&self.phase_label),
            &self.f_plot,
            &self.wav_phase,
            PHASE,
            4
        );
        legend!(&mut chart, SeriesLabelPosition::UpperLeft)
    }

    /// Panel 4: minimum-phase against the linear-phase IR of the same magnitude.
    fn draw_impulse(&self, band: &BitmapArea<'_>) -> PlotResult {
        use plotters::prelude::*;
        let mut chart = ChartBuilder::on(band)
            .margin(0u32)
            .margin_top(TITLE_H)
            .x_label_area_size(X_LABEL_H)
            .y_label_area_size(Y_LABEL_W)
            .caption(
                self.ir_title.as_str(),
                ("sans-serif", SMALL_TITLE_FONT)
                    .into_font()
                    .color(&TITLE.color()),
            )
            .build_cartesian_2d(-1.0f64..4.0f64, -0.45f64..1.08)
            .map_err(dv)?;
        chart.plotting_area().fill(&PANEL_BG.color()).map_err(dv)?;
        chart
            .configure_mesh()
            .x_labels(6)
            .y_labels(6)
            .x_desc("Time Relative to Transient Peak (ms)")
            .y_desc("Normalized Amplitude")
            .axis_desc_style(("sans-serif", DESC_FONT).into_font().color(&TEXT.color()))
            .label_style(("sans-serif", TICK_FONT).into_font().color(&TEXT.color()))
            .axis_style(ShapeStyle::from(EDGE.color()).stroke_width(1))
            .bold_line_style(ShapeStyle::from(GRID.color()).stroke_width(1))
            .light_line_style(clear())
            .draw()
            .map_err(dv)?;

        // The 0 ms line is what the panel is about: anything the linear-phase trace has
        // to its left is pre-ringing the minimum-phase design does not have.
        guide!(chart, &[(-1.0, 0.0), (4.0, 0.0)], ZERO_LINE, 1);
        guide!(chart, &[(0.0, -0.45), (0.0, 1.08)], ZERO_MS, 2);
        let _ = trace!(
            chart,
            Some(&self.lin_ir_label),
            &self.lin_t_ms,
            &self.lin_ir,
            AUTOEQ_RED,
            3
        );
        let _ = trace!(
            chart,
            Some(&self.mp_ir_label),
            &self.mp_t_ms,
            &self.mp_ir,
            WAV_GREEN,
            4
        );
        legend!(&mut chart, SeriesLabelPosition::UpperRight)
    }
}

/// Font sizes, in pixels, for a 1950 x 1725 figure.
const TITLE_FONT: u32 = 26;
/// Panel-2 caption.
const GAIN_TITLE_FONT: u32 = 24;
/// Bottom-row captions.
const SMALL_TITLE_FONT: u32 = 20;
/// Tick label font.
const TICK_FONT: u32 = 16;
/// Axis description font.
const DESC_FONT: u32 = 17;
/// Callout text font.
const ANNOT_FONT: u32 = 15;
/// Axis border colour.
const EDGE: Rgb = Rgb(0x34, 0x3B, 0x42);
/// Legend box fill.
const LEGEND_BG: Rgb = Rgb(0x16, 0x19, 0x1D);

/// Drawing errors from `plotters` carry a backend-specific type, so every call site
/// funnels them through these aliases instead of naming the type.
type PlotResult = std::result::Result<(), crate::error::Error>;
/// The area type every panel is drawn into: a bitmap backend at the figure origin.
type BitmapArea<'a> =
    plotters::drawing::DrawingArea<plotters::prelude::BitMapBackend<'a>, plotters::coord::Shift>;
/// A drawing-area error (the backend error type behind this crate's backend).
fn dv<E: std::fmt::Debug>(e: E) -> crate::error::Error {
    Error::msg(format!("draw failed: {e:?}"))
}
fn io_err<'a>(
    path: &'a std::path::Path,
    what: &'static str,
) -> impl Fn(crate::error::Error) -> crate::error::Error + 'a {
    move |e| Error::msg(format!("{what} in '{}': {e}", path.display()))
}

/// Tick label: whole numbers without a decimal point, halves with one.
#[must_use]
pub fn fmt_tick(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{:.0}", v)
    } else {
        format!("{v}")
    }
}

/// 20 -> "20", 1000 -> "1k", 20000 -> "20k".
#[must_use]
pub fn format_hz(f: f64) -> String {
    if f >= 1000.0 {
        let k = f / 1000.0;
        if (k - k.round()).abs() < 1e-9 {
            format!("{:.0}k", k.round())
        } else {
            format!("{k:.1}k")
        }
    } else if (f - f.round()).abs() < 1e-9 {
        format!("{:.0}", f.round())
    } else {
        format!("{f:.1}")
    }
}

/// `numpy.fft.fftshift` for an even-length real array: swap the two halves.
#[must_use]
pub fn fftshift(x: &[f64]) -> Vec<f64> {
    let h = x.len() / 2;
    let mut out = Vec::with_capacity(x.len());
    out.extend_from_slice(&x[h..]);
    out.extend_from_slice(&x[..h]);
    out
}

/// The variable-bandwidth Gaussian blur standing in for AutoEq's smoothing.
///
/// Below 4 kHz the blur is narrow (0.08 octave); above 6 kHz it is 2.0 octaves wide
/// *and* halved in level, which is why a standard-AutoEq curve visibly flattens in the
/// treble while the FIR keeps its shape. Between the two, width ramps and level scales.
#[must_use]
pub fn gaussian_autoeq(f_plot: &[f64], raw_eq_needed: &[f64]) -> Vec<f64> {
    let log_grid: Vec<f64> = f_plot.iter().map(|f| f.log2()).collect();
    let mut out = vec![0.0f64; f_plot.len()];
    for (i, &f) in f_plot.iter().enumerate() {
        let bw = if f <= 4000.0 {
            0.08
        } else if f >= 6000.0 {
            2.0
        } else {
            0.08 + 1.92 * ((f - 4000.0) / 2000.0)
        };
        let sigma = bw / 2.355;
        let mut num = 0.0;
        let mut den = 0.0;
        for (j, &lg) in log_grid.iter().enumerate() {
            let w = (-0.5 * ((lg - log_grid[i]) / sigma).powi(2)).exp();
            num += raw_eq_needed[j] * w;
            den += w;
        }
        let mut v = num / den;
        if f >= 6000.0 {
            v *= 0.5;
        } else if f > 4000.0 {
            v *= 1.0 - 0.5 * ((f - 4000.0) / 2000.0);
        }
        out[i] = v;
    }
    out
}

/// A model's stored AutoEq PEQ list, as a curve on the plot grid.
///
/// `autoeq_geq_str` holds `f g q gain;…` records. With an explicit `autoeq_offset_db`
/// the curve shifts by that amount; otherwise it is re-levelled so its 1 kHz gain equals
/// the control curve's at 1 kHz — which is what makes the overlay comparable to the FIR
/// instead of an arbitrary offset.
#[must_use]
pub fn autoeq_from_geq(geq_str: &str, offset_db: Option<f64>, ctrl: &[(f64, f64)]) -> Vec<f64> {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for item in geq_str.split(';') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let nums: Vec<f64> = item
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if nums.len() >= 2 {
            pts.push((nums[0], nums[1]));
        }
    }
    let f_plot = fft::geomspace(20.0, 20000.0, 1200);
    if pts.len() < 2 {
        return vec![0.0; f_plot.len()];
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let f_geq: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let mut g_geq: Vec<f64> = pts.iter().map(|p| p.1).collect();
    match offset_db {
        Some(off) => g_geq.iter_mut().for_each(|g| *g += off),
        None => {
            if !ctrl.is_empty() {
                let f_c: Vec<f64> = ctrl.iter().map(|p| p.0).collect();
                let g_c: Vec<f64> = ctrl.iter().map(|p| p.1).collect();
                let shift = fft::interp(1000.0, &f_geq, &g_geq) - fft::interp(1000.0, &f_c, &g_c);
                g_geq.iter_mut().for_each(|g| *g -= shift);
            }
        }
    }
    let lf: Vec<f64> = f_geq.iter().map(|f| f.log10()).collect();
    f_plot
        .iter()
        .map(|&f| fft::interp(f.log10(), &lf, &g_geq))
        .collect()
}

/// Callouts for a model with no hand-written annotations: the extrema of the correction
/// per band, plus the bypass-zone note.
#[must_use]
pub fn build_auto_annotations(
    f_plot: &[f64],
    wav_eq: &[f64],
    hf_bypass_hz: f64,
) -> Vec<Annotation> {
    let g_20 = wav_eq[0];
    // Select by |gain| (or signed min for the treble dip), report the signed gain
    // at the selected grid point.
    let pick = |lo: f64, hi: f64, mode: Pick| -> (f64, f64) {
        let mut best: Option<(usize, f64)> = None;
        for (i, (&f, &g)) in f_plot.iter().zip(wav_eq).enumerate() {
            if f < lo || f > hi {
                continue;
            }
            let key = match mode {
                Pick::Abs => g.abs(),
                Pick::SignedMin => g,
            };
            let better = match best {
                None => true,
                Some((_, bk)) => match mode {
                    Pick::Abs => key > bk,
                    Pick::SignedMin => key < bk,
                },
            };
            if better {
                best = Some((i, key));
            }
        }
        best.map(|(i, _)| (f_plot[i], wav_eq[i]))
            .unwrap_or((lo, 0.0))
    };
    let (f_bass, g_bass) = pick(35.0, 350.0, Pick::Abs);
    let (f_mid, g_mid) = pick(350.0, 3000.0, Pick::Abs);
    let treb_hi = hf_bypass_hz.min(16000.0).max(4800.0);
    let (f_treb, g_treb) = pick(4500.0, treb_hi, Pick::SignedMin);

    vec![
        Annotation {
            text: format!("1. Sub-Bass Target Shelf\n({g_20:+.2}dB @ 20Hz)"),
            xy_x: 20.5,
            xy_y: g_20,
            xytext: (22.0, (g_20 + 2.0).max(5.5)),
            green: false,
        },
        Annotation {
            text: format!("2. Bass / Low-Mid Correction\n({g_bass:+.2}dB @ {f_bass:.0}Hz)"),
            xy_x: f_bass,
            xy_y: g_bass,
            xytext: (
                (f_bass * 0.55).max(55.0),
                if g_bass < 3.5 { 5.4 } else { -7.5 },
            ),
            green: false,
        },
        Annotation {
            text: format!(
                "3. Midrange / Ear-Gain\n({g_mid:+.2}dB @ {:.2}kHz)",
                f_mid / 1000.0
            ),
            xy_x: f_mid,
            xy_y: g_mid,
            xytext: (
                (f_mid * 0.45).max(450.0),
                if g_mid >= 0.0 { 6.2 } else { -7.5 },
            ),
            green: false,
        },
        Annotation {
            text: format!(
                "4. Treble Peak Control\n({g_treb:+.2}dB @ {:.2}kHz)",
                f_treb / 1000.0
            ),
            xy_x: f_treb,
            xy_y: g_treb,
            xytext: (2400.0, -8.8),
            green: true,
        },
        Annotation {
            text: format!(
                "5. >{:.1}kHz Coupler Null Bypass\n(0.0dB ultrasonic wire)",
                hf_bypass_hz / 1000.0
            ),
            xy_x: (hf_bypass_hz * 1.18).min(14000.0).max(hf_bypass_hz + 500.0),
            xy_y: 0.0,
            xytext: (5400.0, 4.5),
            green: false,
        },
    ]
}

#[derive(Clone, Copy)]
enum Pick {
    Abs,
    SignedMin,
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
fn min_of(v: &[f64]) -> f64 {
    v.iter().fold(f64::INFINITY, |a, &x| a.min(x))
}
fn max_of(v: &[f64]) -> f64 {
    v.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x))
}
fn meta_pair(meta: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<(f64, f64)> {
    let arr = meta.get(key)?.as_array()?;
    Some((arr.first()?.as_f64()?, arr.get(1)?.as_f64()?))
}
fn meta_ticks(
    meta: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    dflt: &[f64],
) -> Vec<f64> {
    match meta.get(key).and_then(|v| v.as_array()) {
        Some(arr) if !arr.is_empty() => arr.iter().filter_map(|v| v.as_f64()).collect(),
        _ => dflt.to_vec(),
    }
}
/// Callouts supplied through `plot_meta.annotations`, if the model has any.
fn meta_annotations(
    meta: &serde_json::Map<String, serde_json::Value>,
    notch_db: f64,
) -> Option<Vec<Annotation>> {
    let arr = meta.get("annotations")?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = item.as_object()?;
        let xy = obj.get("xy")?.as_array()?;
        let xytext = obj.get("xytext")?.as_array()?;
        let y = if xy.get(1)?.as_str() == Some("notch_db") {
            notch_db
        } else {
            xy.get(1)?.as_f64()?
        };
        out.push(Annotation {
            text: obj
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            xy_x: xy.first()?.as_f64()?,
            xy_y: y,
            xytext: (xytext.first()?.as_f64()?, xytext.get(1)?.as_f64()?),
            green: obj.get("color").and_then(|v| v.as_str()) == Some("green"),
        });
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        Model {
            key: "t".into(),
            name: "Test Set".into(),
            short_title: "TEST".into(),
            file_prefix: "test".into(),
            rig_info: String::new(),
            target_name: "harman_2018".into(),
            target_pts: vec![(20.0, 4.0), (1000.0, 0.0), (20000.0, -2.0)],
            harm_offset: 1.0,
            gauss_apodize: false,
            hf_bypass_hz: 10000.0,
            notch_range_hz: None,
            notch_ref_hz: None,
            safe_cap_db: 1.5,
            autoeq_geq_str: None,
            autoeq_offset_db: None,
            plot_meta: serde_json::Map::new(),
            raw_pts: vec![
                vec![20.0, 2.0, 2.0, 2.0],
                vec![1000.0, 0.0, 0.0, 0.0],
                vec![20000.0, -6.0, -6.0, -6.0],
            ],
            ctrl_pts: vec![(20.0, 2.0), (1000.0, 0.0), (20000.0, 4.0)],
        }
    }

    fn inputs<'a>(ir: &'a [f64], mag: &'a [f64]) -> PlotInputs<'a> {
        PlotInputs {
            ir_rt: ir,
            mag_lin: mag,
            n_fft_rt: 8,
            fs: 48000,
            boost_rt: 4.0,
            pre_rt: -1.5,
            unwin_rt: -120.0,
            active_ctrl: &[(20.0, 2.0), (1000.0, 0.0), (20000.0, 4.0)],
            notch_mode: "safe",
        }
    }

    #[test]
    fn bands_tile_the_figure_exactly() {
        // Every panel's band is peeled off the figure with split_vertically, so if the
        // bands and gaps do not sum to the figure height the last panel is cut off.
        let b = TITLE_H + ROW1_H + X_LABEL_H;
        let total = M_TOP
            + b
            + ROW_GAP
            + (TITLE_H + ROW2_H + X_LABEL_H)
            + ROW_GAP
            + (TITLE_H + ROW3_H + X_LABEL_H);
        assert_eq!(total, FIG_H, "vertical layout must tile the figure");
        let cols = (M_LEFT - Y_LABEL_W) + Y_LABEL_W + COL_W + COL_GAP + Y_LABEL_W + COL_W;
        assert_eq!(cols, M_RIGHT, "bottom row must fit between the margins");
        // The last band's plot area must clear the figure bottom by at least its own
        // x-label area, or panel 4 is drawn off the canvas.
        let last_plot_bottom =
            M_TOP + b + ROW_GAP + (TITLE_H + ROW2_H + X_LABEL_H) + ROW_GAP + TITLE_H + ROW3_H;
        assert!(
            last_plot_bottom + X_LABEL_H <= FIG_H,
            "bottom margin: {last_plot_bottom}"
        );
    }

    #[test]
    fn hz_labels_are_hundreds_then_kilo() {
        assert_eq!(format_hz(20.0), "20");
        assert_eq!(format_hz(500.0), "500");
        assert_eq!(format_hz(1000.0), "1k");
        assert_eq!(format_hz(15000.0), "15k");
        assert_eq!(format_hz(2500.0), "2.5k");
    }

    #[test]
    fn fftshift_swaps_the_two_halves() {
        let x: Vec<f64> = (0..8).map(|i| f64::from(i)).collect();
        assert_eq!(fftshift(&x), vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn autoeq_blur_stays_sharp_low_and_halves_high() {
        // A step in the requested correction must stay sharp at 1 kHz and both spread
        // and halve by 8 kHz — that asymmetry is the whole point of this helper.
        let f = fft::geomspace(20.0, 20000.0, 1200);
        let needed: Vec<f64> = f
            .iter()
            .map(|&x| if x > 3000.0 { 6.0 } else { 0.0 })
            .collect();
        let out = gaussian_autoeq(&f, &needed);
        let at = |q: f64| out[f.iter().position(|&x| x >= q).unwrap()];
        assert!(out[0].abs() < 0.05, "flat below the step: {}", out[0]);
        assert!(
            at(8000.0) > 2.5 && at(8000.0) < 3.5,
            "halved above: {}",
            at(8000.0)
        );
        assert!(at(4500.0) > at(8000.0), "monotonic through 4k..6k");
    }

    #[test]
    fn geq_curve_is_levelled_to_the_control_curve_at_1khz() {
        // Without an explicit offset the overlay is shifted so its 1 kHz gain matches
        // the control curve's — otherwise the comparison in panel 1 is meaningless.
        let ctrl = [(100.0, 0.0), (1000.0, 3.0), (10000.0, 1.0)];
        let geq = "20 5 0.7 0; 1000 -2 0.7 0; 8000 4 0.7 0";
        let lvl = autoeq_from_geq(geq, None, &ctrl);
        let f = fft::geomspace(20.0, 20000.0, 1200);
        let i = f.iter().position(|&x| x >= 1000.0).unwrap();
        assert!(
            (lvl[i] - 3.0).abs() < 0.35,
            "1 kHz should land on the control curve's 3 dB, got {}",
            lvl[i]
        );
        // With an explicit offset the shift is arithmetic instead: the raw geq curve
        // moved by exactly that number, with no levelling to the control curve.
        let raw = autoeq_from_geq(geq, Some(0.0), &ctrl);
        let off = autoeq_from_geq(geq, Some(-2.0), &ctrl);
        for (a, b) in raw.iter().zip(&off) {
            assert!(
                (b - a + 2.0).abs() < 1e-9,
                "offset must be a pure -2 dB shift"
            );
        }
        assert!(
            (lvl[i] - raw[i]).abs() > 1.0,
            "levelling must shift the raw curve"
        );
    }

    #[test]
    fn curves_carry_the_harmonic_offset_and_the_normalised_ir() {
        let ir = vec![1.0f64, 0.0, 0.0, 0.0];
        let mag = vec![1.0f64; 5];
        let d = PlotData::build(&model(), &inputs(&ir, &mag));
        assert_eq!(d.f_plot.len(), 1200);
        assert_eq!(d.f_plot[0], 20.0);
        assert!((d.f_plot[1199] - 20000.0).abs() < 1e-9);
        // Target at 1 kHz is 0 dB, plus the model's +1 dB harmonic offset.
        let i1k = d.f_plot.iter().position(|&x| x >= 1000.0).unwrap();
        assert!(
            (d.harm_plot[i1k] - 1.0).abs() < 0.05,
            "harm offset missing: {}",
            d.harm_plot[i1k]
        );
        assert_eq!(d.raw_l.len(), 1200, "4-column raw data has L/R traces");
        assert!(d.lin_t_ms.iter().all(|&t| (-1.0..=4.0).contains(&t)));
        assert_eq!(&d.mp_ir[..48], &[0.0; 48][..], "nothing drawn left of 0 ms");
        assert_eq!(d.mp_ir[48], 1.0, "the minimum-phase peak normalises to 1");
        assert_eq!(d.mp_t_ms[48], 0.0);
    }

    #[test]
    fn titles_substitute_the_report_figures() {
        let ir = vec![1.0f64, 0.0, 0.0, 0.0];
        let mag = vec![1.0f64; 5];
        let d = PlotData::build(&model(), &inputs(&ir, &mag));
        assert!(d.title.contains("Preamp: -1.50 dB"), "caption: {}", d.title);
        assert!(
            d.wav_label.contains("-120.0 dBFS"),
            "legend: {}",
            d.wav_label
        );
        assert!(
            d.wav_gain_label.contains("+4.00 dB") && d.wav_gain_label.contains("-1.50 dB"),
            "{}",
            d.wav_gain_label
        );
        assert!(
            !d.title.contains('{'),
            "unsubstituted template: {}",
            d.title
        );

        // A model with a notch range gets the notch-mode caption instead.
        let mut m = model();
        m.notch_range_hz = Some((3000.0, 4500.0));
        let mut meta = serde_json::Map::new();
        meta.insert(
            "title_template".into(),
            serde_json::json!("VERIFICATION: {notch_mode} | {pre_rt:.2f} | {notch_db:.2f}"),
        );
        m.plot_meta = meta;
        m.notch_ref_hz = Some(3703.0);
        let d2 = PlotData::build(&m, &inputs(&ir, &mag));
        assert_eq!(
            d2.title,
            format!(
                "VERIFICATION: safe | -1.50 | {:.2}",
                crate::ir::notch_db_at(&[(20.0, 2.0), (1000.0, 0.0), (20000.0, 4.0)], Some(3703.0))
            ),
            "{}",
            d2.title
        );
    }

    #[test]
    fn annotations_pick_the_per_band_extrema() {
        let f = fft::geomspace(20.0, 20000.0, 1200);
        let g: Vec<f64> = f
            .iter()
            .map(|&x| match x {
                v if v < 100.0 => -6.0,
                v if v < 1000.0 => 3.0,
                v if v < 5000.0 => -1.0,
                _ => 0.0,
            })
            .collect();
        let ann = build_auto_annotations(&f, &g, 10000.0);
        assert_eq!(ann.len(), 5);
        assert!(ann[1].text.contains("-6.00dB"), "{}", ann[1].text);
        assert!(ann[2].text.contains("+3.00dB"), "{}", ann[2].text);
        assert!(
            ann[3].text.contains("4. Treble Peak Control") && ann[3].text.contains("kHz"),
            "{}",
            ann[3].text
        );
        assert!(ann[4].text.contains(">10.0kHz"), "{}", ann[4].text);
        assert!(ann[3].green, "the treble callout is the green arrow");
        // The arrow head sits on the curve, not on the |.| key used to find it.
        assert_eq!(ann[1].xy_y, -6.0);
    }

    #[test]
    fn metadata_overrides_win_and_defaults_back_stop() {
        let mut m = model();
        let mut meta = serde_json::Map::new();
        meta.insert("ax1_ylim".into(), serde_json::json!([-12.0, 14.0]));
        meta.insert("ax1_yticks".into(), serde_json::json!([-10.0, 0.0, 10.0]));
        meta.insert(
            "annotations".into(),
            serde_json::json!([
                {"text": "notch {notch_db:.2f}", "xy": [3703.0, "notch_db"], "xytext": [1000.0, 5.0], "color": "green"}
            ]),
        );
        m.plot_meta = meta;
        m.notch_ref_hz = Some(3703.0);
        let ir = vec![1.0f64, 0.0, 0.0, 0.0];
        let mag = vec![1.0f64; 5];
        let d = PlotData::build(&m, &inputs(&ir, &mag));
        assert_eq!(d.ax1_ylim, (-12.0, 14.0));
        assert_eq!(d.ax1_yticks, vec![-10.0, 0.0, 10.0]);
        assert_eq!(
            d.ax2_yticks,
            AX2_YTICKS.to_vec(),
            "other panels keep defaults"
        );
        assert_eq!(d.annotations.len(), 1);
        // The reference's "notch_db" placeholder is the control curve read at the notch
        // reference frequency, interpolated when the frequency is not itself a knot.
        let want = crate::ir::notch_db_at(&m.ctrl_pts, m.notch_ref_hz);
        assert!(
            want > 0.0 && want < 4.0,
            "expected an interpolated gain, got {want}"
        );
        assert!(
            (d.annotations[0].xy_y - want).abs() < 1e-12,
            "{} vs {}",
            d.annotations[0].xy_y,
            want
        );
        assert!(d.annotations[0].green);

        // An empty annotations list falls back to the automatic callouts rather than
        // silently drawing none.
        let mut m2 = model();
        let mut meta2 = serde_json::Map::new();
        meta2.insert("annotations".into(), serde_json::json!([]));
        m2.plot_meta = meta2;
        let d2 = PlotData::build(&m2, &inputs(&ir, &mag));
        assert_eq!(d2.annotations.len(), 5);
    }

    #[test]
    fn render_writes_a_png_of_the_expected_size() {
        // Small enough to run in the suite; the geometry assertions above are what
        // keep the full-size figure honest.
        let dir = std::env::temp_dir().join(format!("hypreq-plot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ir: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin() * 0.5).collect();
        let mag: Vec<f64> = (0..33).map(|_| 1.0).collect();
        let d = PlotData::build(&model(), &inputs(&ir, &mag));
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let out = d.render("test");
        let path = out.expect("render");
        // The path is relative to the working directory render() ran in, so read it
        // before that directory is restored.
        let bytes = std::fs::read(&path).unwrap();
        std::env::set_current_dir(&prev).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "not a PNG");
        assert!(
            bytes.len() > 20_000,
            "suspiciously small figure: {}",
            bytes.len()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
