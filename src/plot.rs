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
//! The axes are drawn by hand rather than through a `plotters` mesh: the reference fixes
//! its own major and minor frequency ticks and its own y-tick values, and every polyline
//! is clipped to its panel because `draw_series` does not clip.
//!
//! The curves are computed in [`PlotData::build`] from the same inputs the report
//! printed; [`PlotData::render`] only draws. The split is deliberate: the numbers behind
//! the figure are unit-testable (geometry, the AutoEq blur, the callout selection, the
//! clipping), while the PNG is checked by a test that decodes it and measures the ink in
//! each panel.

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
/// Pixels kept free at a band's right edge.
///
/// A drawing area clips at its own last column, so a frame line or an edge tick drawn at
/// `x == width` silently disappears. Reserving a few pixels puts the right frame and the
/// 20 kHz / 4 ms ticks inside the clip rectangle instead of on its boundary.
const RIGHT_PAD: u32 = 4;
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
/// The phase panel's y window (`ax3.set_ylim(-90, 90)`), wide enough for every tick above.
pub const PHASE_YLIM: (f64, f64) = (-90.0, 90.0);

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
    /// The two ears' difference, `left − right`, on [`PlotData::f_plot`].
    ///
    /// Empty unless the model was measured per ear *and* `--stereo` was asked for. Drawn on panel 2
    /// rather than panel 1: panel 1 shows what the measurement looks like, and this is a statement
    /// about what one shared filter cannot fix — the difference it leaves standing.
    pub raw_lr_diff: Vec<f64>,
    /// Panel 2 legend for [`PlotData::raw_lr_diff`]; empty when there is no trace.
    pub lr_diff_label: String,
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
    /// `true` when `plot_meta` pinned panel 1's window, which the reference then treats
    /// as exact rather than as a floor.
    pub ax1_ylim_fixed: bool,
    /// Panel 2 y limits.
    pub ax2_ylim: (f64, f64),
    /// `true` when `plot_meta` pinned panel 2's window.
    pub ax2_ylim_fixed: bool,
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
    /// Draw the two ears' difference on panel 2.
    ///
    /// Opt-in because the reference figure has no such trace and the default figure is the one
    /// people compare against their own captures: this is set from `--stereo` being anything other
    /// than `off`, so nobody's unchanged run gains a curve.
    pub show_ears: bool,
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

/// The reference's minor x ticks: the in-between decades that get a light grid line and
/// no label (`FixedLocator(minor_freqs)` with a `NullFormatter`).
pub const MINOR_FREQS: [f64; 19] = [
    30.0, 40.0, 60.0, 70.0, 80.0, 90.0, 300.0, 400.0, 600.0, 700.0, 800.0, 900.0, 3000.0, 4000.0,
    6000.0, 7000.0, 8000.0, 9000.0, 15000.0,
];

/// Lowest plotted frequency (`ax.set_xlim(20, 20000)`).
const F_MIN: f64 = 20.0;
/// Highest plotted frequency.
const F_MAX: f64 = 20000.0;
/// Light grid colour, `grid(which="minor")`.
const GRID_MINOR: Rgb = Rgb(0x24, 0x29, 0x2F);
/// Legend text size in pixels (the reference's 8.5 pt at 150 dpi).
const LEGEND_FONT: u32 = 18;
/// Legend text colour.
const LEGEND_TEXT: Rgb = Rgb(0xD0, 0xD7, 0xDE);
/// Dash length and gap for the reference's `linestyle="--"`, in pixels.
const DASH: (u32, u32) = (9, 7);
/// Dash length and gap for the reference's `linestyle=":"`.
const DOT: (u32, u32) = (1, 4);
/// x ticks of the impulse panel (milliseconds relative to the transient).
const IR_XTICKS: [f64; 6] = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
/// y ticks of the impulse panel.
const IR_YTICKS: [f64; 8] = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
/// Impulse panel x range, in ms.
const IR_XMIN: f64 = -1.0;
/// Impulse panel x upper limit, in ms.
const IR_XMAX: f64 = 4.0;
/// Impulse panel y range.
const IR_YMIN: f64 = -0.45;
/// Impulse panel y upper limit.
const IR_YMAX: f64 = 1.08;

/// The pixel rectangle a panel's data occupies.
///
/// `ChartBuilder` is configured with `margin(0)`, `margin_top(TITLE_H)`,
/// `x_label_area_size(X_LABEL_H)` and `y_label_area_size(Y_LABEL_W)`, so the plotting
/// area is exactly this rectangle. The grid, tick labels, frame and legend are drawn
/// against it by hand: `plotters` derives tick positions from the coordinate type, and a
/// logarithmic axis can only produce decades plus fractional light points, never the
/// fixed 1-2-5 octave list the reference pins with `FixedLocator`.
fn plot_rect(band: &BitmapArea<'_>) -> (i32, i32, i32, i32) {
    let (w, h) = band.dim_in_pixel();
    let w = i32::try_from(w).unwrap_or(i32::MAX);
    let h = i32::try_from(h).unwrap_or(i32::MAX);
    (
        Y_LABEL_W as i32,
        TITLE_H as i32,
        (w - RIGHT_PAD as i32).max(Y_LABEL_W as i32 + 1),
        (h - X_LABEL_H as i32).max(TITLE_H as i32 + 1),
    )
}

/// The x pixel of a data value, unrounded. `log` matches the panel's log-frequency
/// mapping and the same normalisation `plotters`' log coordinate applies.
fn x_pix_f(v: f64, min: f64, max: f64, log: bool, rect: (i32, i32, i32, i32)) -> f64 {
    let t = if log {
        (v / min).ln() / (max / min).ln()
    } else {
        (v - min) / (max - min)
    };
    f64::from(rect.0) + f64::from(rect.2 - rect.0) * t
}

/// The y pixel of a data value, unrounded; pixel rows grow downwards, data grows up.
fn y_pix_f(v: f64, min: f64, max: f64, rect: (i32, i32, i32, i32)) -> f64 {
    let t = (v - min) / (max - min);
    f64::from(rect.3) - f64::from(rect.3 - rect.1) * t
}

/// The x pixel of a data value; `log` matches the panel's log-frequency mapping.
fn x_pix(v: f64, min: f64, max: f64, log: bool, rect: (i32, i32, i32, i32)) -> i32 {
    x_pix_f(v, min, max, log, rect).round() as i32
}

/// The y pixel of a data value; pixel rows grow downwards, data grows up.
fn y_pix(v: f64, min: f64, max: f64, rect: (i32, i32, i32, i32)) -> i32 {
    y_pix_f(v, min, max, rect).round() as i32
}

/// A pixel-space text element: what to draw, how big, in which colour, anchored where.
struct TextSpec<'a> {
    text: &'a str,
    size: u32,
    color: Rgb,
    anchor: plotters::style::text_anchor::Pos,
    rotate: bool,
}

/// Inverse of [`x_pix`]: the data value at a pixel column.
fn x_from_pix(px: f64, min: f64, max: f64, log: bool, rect: (i32, i32, i32, i32)) -> f64 {
    let t = (px - f64::from(rect.0)) / f64::from((rect.2 - rect.0).max(1));
    if log {
        min * (max / min).powf(t)
    } else {
        min + t * (max - min)
    }
}

/// Inverse of [`y_pix`]: the data value at a pixel row.
fn y_from_pix(py: f64, min: f64, max: f64, rect: (i32, i32, i32, i32)) -> f64 {
    let t = (f64::from(rect.3) - py) / f64::from((rect.3 - rect.1).max(1));
    min + t * (max - min)
}

/// The three corners of a filled arrow head sitting on `tip` and pointing along
/// `tail -> tip`, given in data coordinates.
///
/// The head is specified in *pixels* and converted back to data space, so it keeps a
/// constant size on a three-decade logarithmic axis instead of stretching with frequency
/// the way a data-coordinate triangle would.
#[must_use]
pub fn arrow_head_points(
    tail: (f64, f64),
    tip: (f64, f64),
    w: Window,
    rect: (i32, i32, i32, i32),
    head_px: (f64, f64),
) -> [(f64, f64); 3] {
    let (len_px, half_width_px) = head_px;
    let (x_range, y_range) = (w.xr, w.yr);
    let log_x = w.log_x;
    // Unrounded pixel positions: snapping to whole pixels here would change the head's
    // length by up to a pixel depending on where it lands.
    let tip_px = (
        x_pix_f(tip.0, x_range.0, x_range.1, log_x, rect),
        y_pix_f(tip.1, y_range.0, y_range.1, rect),
    );
    let tail_px = (
        x_pix_f(tail.0, x_range.0, x_range.1, log_x, rect),
        y_pix_f(tail.1, y_range.0, y_range.1, rect),
    );
    let dx = tip_px.0 - tail_px.0;
    let dy = tip_px.1 - tail_px.1;
    let norm = (dx * dx + dy * dy).sqrt();
    if norm < 1e-9 {
        return [(tip.0, tip.1), (tip.0, tip.1), (tip.0, tip.1)];
    }
    let (ux, uy) = (dx / norm, dy / norm);
    // Base centre behind the tip, corners either side of the shaft.
    let bx = tip_px.0 - ux * len_px;
    let by = tip_px.1 - uy * len_px;
    let corners_px = [
        tip_px,
        (bx - uy * half_width_px, by + ux * half_width_px),
        (bx + uy * half_width_px, by - ux * half_width_px),
    ];
    corners_px.map(|(px, py)| {
        (
            x_from_pix(px, x_range.0, x_range.1, log_x, rect),
            y_from_pix(py, y_range.0, y_range.1, rect),
        )
    })
}

/// Draw text in pixel space with an explicit anchor point, optionally rotated for a y
/// unit label. `FontTransform::Rotate90` turns 90 degrees *clockwise*; matplotlib reads a
/// y label bottom-to-top, which is the 270-degree variant.
fn label(band: &BitmapArea<'_>, x: i32, y: i32, spec: &TextSpec<'_>) -> PlotResult {
    use plotters::prelude::*;
    let style = ("sans-serif", f64::from(spec.size))
        .into_font()
        .color(&spec.color.color())
        .pos(spec.anchor);
    let style = if spec.rotate {
        style.transform(plotters::style::FontTransform::Rotate270)
    } else {
        style
    };
    band.draw_text(spec.text, &style, (x, y)).map_err(dv)
}

/// A stroked shape style with the reference's per-trace alpha.
pub fn stroke_a(c: Rgb, width: u32, alpha: f64) -> plotters::style::ShapeStyle {
    use plotters::style::Color as _;
    plotters::style::ShapeStyle {
        color: c.color().mix(alpha),
        filled: false,
        stroke_width: width,
    }
}

/// A filled shape style. `Color -> ShapeStyle` sets `filled: false`, so handing a plain
/// colour to `Rectangle::new` draws a hollow outline rather than a filled box.
fn solid(c: Rgb) -> plotters::style::ShapeStyle {
    use plotters::style::Color as _;
    c.color().filled()
}

/// The rendered width of a string, measured on the active backend.
fn text_w(band: &BitmapArea<'_>, text: &str, size: u32) -> Plot<i32> {
    use plotters::prelude::*;
    let style = ("sans-serif", f64::from(size))
        .into_font()
        .color(&TEXT.color());
    let (w, _) = band.estimate_text_size(text, &style).map_err(dv)?;
    Ok(i32::try_from(w).unwrap_or(i32::MAX))
}

/// A pixel-space horizontal segment.
///
/// `plotters` has no pixel-coordinate line element (only the data-coordinate
/// `PathElement`), so a segment is drawn as a filled box `w` pixels thick.
fn hline(band: &BitmapArea<'_>, y: i32, x0: i32, x1: i32, c: Rgb, w: u32) -> PlotResult {
    if x1 <= x0 {
        return Ok(());
    }
    band.draw(&plotters::element::Rectangle::new(
        [(x0, y), (x1, y + i32::try_from(w).unwrap_or(1))],
        solid(c),
    ))
    .map_err(dv)
}

/// A pixel-space vertical segment.
fn vline(band: &BitmapArea<'_>, x: i32, y0: i32, y1: i32, c: Rgb, w: u32) -> PlotResult {
    if y1 <= y0 {
        return Ok(());
    }
    band.draw(&plotters::element::Rectangle::new(
        [(x, y0), (x + i32::try_from(w).unwrap_or(1), y1)],
        solid(c),
    ))
    .map_err(dv)
}

/// A dotted pixel-space vertical segment, for the reference's dotted `axvline`.
fn dotted_vline(band: &BitmapArea<'_>, x: i32, y0: i32, y1: i32, c: Rgb, w: u32) -> PlotResult {
    let mut y = y0;
    while y < y1 {
        vline(band, x, y, (y + 2).min(y1), c, w)?;
        y += 6;
    }
    Ok(())
}

/// Everything one panel's hand-drawn axes need: the data window, which tick values to
/// honour, and what to call the y axis.
struct AxesSpec<'a> {
    x_range: (f64, f64),
    y_range: (f64, f64),
    log_x: bool,
    xticks: &'a [f64],
    yticks: &'a [f64],
    y_desc: &'a str,
    /// The reference labels only the bottom row, so the upper panels pass `None`.
    x_desc: Option<&'a str>,
    zero_line: bool,
}

/// Tick values inside the axis window, **endpoints included**: matplotlib draws the tick
/// mark and its label for a value sitting exactly on a limit and only omits the grid
/// line there, which is the difference between `20`/`20k` appearing and vanishing.
fn ticks_in(ticks: &[f64], range: (f64, f64)) -> impl Iterator<Item = &f64> {
    ticks
        .iter()
        .filter(move |v| **v >= range.0 && **v <= range.1)
}

/// Whether a tick earns a grid line, i.e. is strictly inside the window.
fn grid_line_inside(v: f64, range: (f64, f64)) -> bool {
    v > range.0 && v < range.1
}

/// Panel background, grid, fixed tick labels, axis frame and unit label.
///
/// This is the hand-drawn replacement for `style_freq_axis`: major grid at `xticks`,
/// light grid at [`MINOR_FREQS`] for a log axis, horizontal grid at `yticks`, the 0 dBr
/// reference line, tick marks pointing outwards, and the tick labels at the same fixed
/// values, plus the rotated y unit.
fn draw_axes(band: &BitmapArea<'_>, config: &AxesSpec<'_>) -> PlotResult {
    use plotters::style::text_anchor::{HPos, Pos, VPos};
    let &AxesSpec {
        x_range: xr,
        y_range: yr,
        log_x,
        xticks,
        yticks,
        y_desc,
        x_desc,
        zero_line,
    } = config;
    let rect = plot_rect(band);

    band.draw(&plotters::element::Rectangle::new(
        [(rect.0, rect.1), (rect.2, rect.3)],
        solid(PANEL_BG),
    ))
    .map_err(dv)?;

    if log_x {
        for &f in MINOR_FREQS.iter().filter(|f| **f > xr.0 && **f < xr.1) {
            vline(
                band,
                x_pix(f, xr.0, xr.1, true, rect),
                rect.1,
                rect.3,
                GRID_MINOR,
                1,
            )?;
        }
    }
    for &f in ticks_in(xticks, xr) {
        let x = x_pix(f, xr.0, xr.1, log_x, rect);
        if grid_line_inside(f, xr) {
            vline(band, x, rect.1, rect.3, GRID, 1)?;
        }
        vline(band, x, rect.3, rect.3 + 3, EDGE, 1)?;
        label(
            band,
            x,
            rect.3 + 5,
            &TextSpec {
                text: &format_hz(f),
                size: TICK_FONT,
                color: TEXT,
                anchor: Pos {
                    h_pos: HPos::Center,
                    v_pos: VPos::Top,
                },
                rotate: false,
            },
        )?;
    }
    for &t in ticks_in(yticks, yr) {
        let y = y_pix(t, yr.0, yr.1, rect);
        if grid_line_inside(t, yr) {
            hline(band, y, rect.0, rect.2, GRID, 1)?;
        }
        hline(band, y, rect.0 - 3, rect.0, EDGE, 1)?;
        label(
            band,
            rect.0 - 6,
            y,
            &TextSpec {
                text: &fmt_tick(t),
                size: TICK_FONT,
                color: TEXT,
                anchor: Pos {
                    h_pos: HPos::Right,
                    v_pos: VPos::Center,
                },
                rotate: false,
            },
        )?;
    }
    // The loop above already labels the limits; only their grid line is skipped, which
    // is exactly matplotlib's behaviour on a FixedLocator axis.
    if zero_line && 0.0 > yr.0 && 0.0 < yr.1 {
        hline(
            band,
            y_pix(0.0, yr.0, yr.1, rect),
            rect.0,
            rect.2,
            ZERO_LINE,
            1,
        )?;
    }

    // Frame.
    hline(band, rect.1, rect.0, rect.2, EDGE, 1)?;
    hline(band, rect.3, rect.0, rect.2, EDGE, 1)?;
    vline(band, rect.0, rect.1, rect.3, EDGE, 1)?;
    vline(band, rect.2, rect.1, rect.3, EDGE, 1)?;

    // The unit label, rotated beside the y tick column.
    label(
        band,
        i32::try_from(DESC_FONT).unwrap_or(16) / 2 + 2,
        (rect.1 + rect.3) / 2,
        &TextSpec {
            text: y_desc,
            size: DESC_FONT,
            color: TEXT,
            anchor: Pos {
                h_pos: HPos::Center,
                v_pos: VPos::Center,
            },
            rotate: true,
        },
    )?;

    // The x unit, centred under the tick-label row (`set_xlabel` on the bottom row).
    if let Some(text) = x_desc {
        label(
            band,
            (rect.0 + rect.2) / 2,
            rect.3 + i32::try_from(X_LABEL_H).unwrap_or(52) - 4,
            &TextSpec {
                text,
                size: DESC_FONT,
                color: TEXT,
                anchor: Pos {
                    h_pos: HPos::Center,
                    v_pos: VPos::Bottom,
                },
                rotate: false,
            },
        )?;
    }
    Ok(())
}

/// Where a legend box sits inside its panel.
#[derive(Clone, Copy)]
enum LegendPos {
    LowerLeft,
    UpperLeft,
    UpperRight,
}

/// A legend swatch: a line in the trace's style, or a filled patch.
struct LegendItem<'a> {
    text: &'a str,
    swatch: Swatch,
}

/// The two kinds of swatch the reference's legend contains.
enum Swatch {
    Line {
        color: Rgb,
        width: u32,
        dash: Option<(u32, u32)>,
    },
    Patch(Rgb),
}

/// The legend box, drawn by hand: `plotters` attaches swatches only through
/// `.legend(closure)` on a series segment, and that closure receives data coordinates,
/// which on a three-decade logarithmic axis cannot express a 25 px swatch.
fn draw_legend(band: &BitmapArea<'_>, items: &[LegendItem<'_>], pos: LegendPos) -> PlotResult {
    use plotters::style::text_anchor::{HPos, Pos, VPos};
    if items.is_empty() {
        return Ok(());
    }
    let rect = plot_rect(band);
    let line_h = (f64::from(LEGEND_FONT) * 1.45).round() as i32;
    let swatch_w = 26;
    let pad = 6;
    let widest = items
        .iter()
        .map(|i| text_w(band, i.text, LEGEND_FONT))
        .collect::<Plot<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let box_w = (swatch_w + 8 + widest + 2 * pad).min((rect.2 - rect.0) - 8);
    let box_h = line_h * items.len() as i32 + 2 * pad;
    let x = match pos {
        LegendPos::LowerLeft | LegendPos::UpperLeft => rect.0 + 8,
        LegendPos::UpperRight => rect.2 - 8 - box_w,
    };
    let y = match pos {
        LegendPos::LowerLeft => rect.3 - 8 - box_h,
        LegendPos::UpperLeft | LegendPos::UpperRight => rect.1 + 8,
    };

    band.draw(&plotters::element::Rectangle::new(
        [(x, y), (x + box_w, y + box_h)],
        fill(LEGEND_BG, 0.92),
    ))
    .map_err(dv)?;
    band.draw(&plotters::element::Rectangle::new(
        [(x, y), (x + box_w, y + box_h)],
        stroke(EDGE, 1),
    ))
    .map_err(dv)?;

    for (i, item) in items.iter().enumerate() {
        let cy = y + pad + line_h / 2 + line_h * i as i32;
        match item.swatch {
            Swatch::Patch(c) => band
                .draw(&plotters::element::Rectangle::new(
                    [
                        (x + pad, cy - line_h / 3),
                        (x + pad + swatch_w, cy + line_h / 3),
                    ],
                    fill(c, 0.55),
                ))
                .map_err(dv)?,
            Swatch::Line { color, width, dash } => match dash {
                None => hline(band, cy, x + pad, x + pad + swatch_w, color, width)?,
                Some((d, g)) => {
                    let mut sx = x + pad;
                    while sx < x + pad + swatch_w {
                        let e = (sx + d as i32).min(x + pad + swatch_w);
                        hline(band, cy, sx, e, color, width)?;
                        sx = e + g as i32;
                    }
                }
            },
        }
        label(
            band,
            x + pad + swatch_w + 8,
            cy,
            &TextSpec {
                text: item.text,
                size: LEGEND_FONT,
                color: LEGEND_TEXT,
                anchor: Pos {
                    h_pos: HPos::Left,
                    v_pos: VPos::Center,
                },
                rotate: false,
            },
        )?;
    }
    Ok(())
}

/// The panel title, centred over the data area like matplotlib's `set_title`, wrapped
/// with measured glyph widths when the line would run past the panel.
fn draw_title(band: &BitmapArea<'_>, text: &str, size: u32) -> PlotResult {
    use plotters::style::text_anchor::{HPos, Pos, VPos};
    if text.is_empty() {
        return Ok(());
    }
    let rect = plot_rect(band);
    let avail = rect.2 - rect.0;
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let trial = if cur.is_empty() {
            word.to_string()
        } else {
            format!("{cur} {word}")
        };
        if cur.is_empty() || text_w(band, &trial, size)? <= avail {
            cur = trial;
        } else {
            lines.push(std::mem::take(&mut cur));
            cur = word.to_string();
        }
    }
    if !cur.is_empty() || lines.is_empty() {
        lines.push(cur);
    }

    let line_h = (f64::from(size) * 1.25).round() as i32;
    let block = line_h * lines.len() as i32;
    // Titles grow upwards from the top of the plot area, into the reserved caption band.
    let top = (TITLE_H as i32 - block).max(2);
    for (i, line) in lines.iter().enumerate() {
        label(
            band,
            (rect.0 + rect.2) / 2,
            top + line_h * i as i32,
            &TextSpec {
                text: line,
                size,
                color: TITLE,
                anchor: Pos {
                    h_pos: HPos::Center,
                    v_pos: VPos::Top,
                },
                rotate: false,
            },
        )?;
    }
    Ok(())
}

/// The data window one panel draws into. Polylines are clipped to it, because
/// `ChartContext::draw_series` is not: an unclipped trace runs across the margin and off
/// the canvas.
#[derive(Clone, Copy)]
pub struct Window {
    pub xr: (f64, f64),
    pub yr: (f64, f64),
    pub log_x: bool,
}

/// Clip the segment `a -> b` against the axis-aligned box `[lo, hi]` (Liang–Barsky),
/// returning the visible sub-segment or `None` when the segment misses the box.
fn liang_barsky(
    a: (f64, f64),
    b: (f64, f64),
    lo: (f64, f64),
    hi: (f64, f64),
) -> Option<((f64, f64), (f64, f64))> {
    let d = (b.0 - a.0, b.1 - a.1);
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    // (p, q) per slab edge: entering/exiting x, then entering/exiting y.
    for (p, q) in [
        (-d.0, a.0 - lo.0),
        (d.0, hi.0 - a.0),
        (-d.1, a.1 - lo.1),
        (d.1, hi.1 - a.1),
    ] {
        if p == 0.0 {
            // Parallel to this edge: visible only if already inside the slab.
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                if r > t1 {
                    return None;
                }
                t0 = t0.max(r);
            } else {
                if r < t0 {
                    return None;
                }
                t1 = t1.min(r);
            }
        }
    }
    Some((
        (a.0 + t0 * d.0, a.1 + t0 * d.1),
        (a.0 + t1 * d.0, a.1 + t1 * d.1),
    ))
}

fn flush_run(cur: &mut Option<(Vec<f64>, Vec<f64>)>, runs: &mut Vec<(Vec<f64>, Vec<f64>)>) {
    if let Some(run) = cur.take() {
        if run.0.len() >= 2 {
            runs.push(run);
        }
    }
}

/// Split a polyline into the runs that lie inside `w`, with the boundary crossings
/// interpolated.
///
/// Clipping happens in the space the axis actually maps (`ln x` on a log-frequency panel),
/// so a segment that is straight on screen stays straight and its endpoint lands exactly
/// on the border rather than somewhere along an unclipped overshoot.
fn clip_polyline(xs: &[f64], ys: &[f64], w: Window) -> Vec<(Vec<f64>, Vec<f64>)> {
    let fwd = |x: f64| if w.log_x { x.ln() } else { x };
    let inv = |u: f64| if w.log_x { u.exp() } else { u };
    // A non-positive frequency has no log image, so it breaks the polyline instead of
    // dragging -inf through the transform.
    let usable = |x: f64, y: f64| x.is_finite() && y.is_finite() && (!w.log_x || x > 0.0);
    let lo = (fwd(w.xr.0.max(f64::MIN_POSITIVE)), w.yr.0);
    let hi = (fwd(w.xr.1), w.yr.1);
    let mut runs = Vec::new();
    let mut cur: Option<(Vec<f64>, Vec<f64>)> = None;
    for i in 0..xs.len().saturating_sub(1) {
        let (Some((&x0, &y0)), Some((&x1, &y1))) =
            (xs.get(i).zip(ys.get(i)), xs.get(i + 1).zip(ys.get(i + 1)))
        else {
            flush_run(&mut cur, &mut runs);
            continue;
        };
        if !usable(x0, y0) || !usable(x1, y1) {
            flush_run(&mut cur, &mut runs);
            continue;
        }
        match liang_barsky((fwd(x0), y0), (fwd(x1), y1), lo, hi) {
            Some((a, b)) => {
                // Snap the clipped endpoints back onto the window: the round trip through
                // `ln`/`exp` and the `t` interpolation are otherwise a fraction of an ULP
                // outside, and a point "inside" the panel would still be reported as a
                // border breach.
                let point = |p: (f64, f64)| {
                    (
                        inv(p.0.clamp(lo.0, hi.0)).clamp(w.xr.0, w.xr.1),
                        p.1.clamp(w.yr.0, w.yr.1),
                    )
                };
                let run = cur.get_or_insert_with(|| (Vec::new(), Vec::new()));
                if run.0.is_empty() {
                    let a = point(a);
                    run.0.push(a.0);
                    run.1.push(a.1);
                }
                let b = point(b);
                run.0.push(b.0);
                run.1.push(b.1);
            }
            None => flush_run(&mut cur, &mut runs),
        }
    }
    flush_run(&mut cur, &mut runs);
    runs
}

/// The shaded coupler-null bypass zone, drawn before any trace and clamped to the panel's
/// own window, so a model whose bypass starts past the plot edge draws nothing instead of
/// painting over the frame.
macro_rules! bypass_span {
    ($chart:expr, $from:expr, $to:expr, $win:expr) => {{
        let w = $win;
        if let Some((from, to)) = $crate::plot::clamp_span($from, $to, w.xr) {
            $chart
                .draw_series(vec![plotters::element::Rectangle::new(
                    [(from, w.yr.0), (to, w.yr.1)],
                    $crate::plot::fill($crate::plot::BYPASS, 0.35),
                )])
                .map_err($crate::plot::dv)?;
        }
    }};
}

/// The bypass zone intersected with the axis window, or `None` when it lies wholly
/// outside.
///
/// The reference gets this for free: `axvspan` draws into an axis that clips. Here an
/// unclamped rectangle is painted over the frame and into the neighbouring panel, and a
/// model whose bypass starts past 20 kHz shades the whole plot. A span that only touches
/// the window at one edge has no width and is not drawn either.
#[must_use]
pub fn clamp_span(from: f64, to: f64, window: (f64, f64)) -> Option<(f64, f64)> {
    let (from, to) = (from.max(window.0), to.min(window.1));
    (from < to).then_some((from, to))
}

/// One polyline in data coordinates, clipped to the panel window. Solid by default;
/// `dash <len>, <gap>` draws the reference's `--` / `:` styles.
macro_rules! trace {
    ($chart:expr, $win:expr, $xs:expr, $ys:expr, $style:expr) => {{
        for (tx, ty) in $crate::plot::clip_polyline($xs, $ys, $win) {
            let pts: Vec<(f64, f64)> = tx.into_iter().zip(ty).collect();
            $chart
                .draw_series(LineSeries::new(pts, $style))
                .map_err($crate::plot::dv)?;
        }
    }};
    ($chart:expr, $win:expr, $xs:expr, $ys:expr, $style:expr, dash $len:expr, $gap:expr) => {{
        for (tx, ty) in $crate::plot::clip_polyline($xs, $ys, $win) {
            let pts: Vec<(f64, f64)> = tx.into_iter().zip(ty).collect();
            $chart
                .draw_series(plotters::prelude::DashedLineSeries::new(
                    pts, $len, $gap, $style,
                ))
                .map_err($crate::plot::dv)?;
        }
    }};
}

/// Where the lines of a callout block sit, and where its arrow leaves.
///
/// matplotlib anchors an `annotate` block at `xytext` with `va="baseline"`, so a multi-line
/// block grows **upwards** from that y. Growing it downwards made a two-line callout clamp
/// both rows onto the same line, left the secondary arrows (which have no text of their
/// own) pointing at nothing, and started the shaft above the first row so it sliced
/// through its own text.
#[must_use]
pub fn callout_layout(lines: usize, xytext_y: f64, xy_y: f64, dy: f64) -> (Vec<f64>, f64) {
    // Baseline-to-centre of one row, in units of the line pitch.
    let baseline = 0.4;
    let n = lines as f64;
    let centres = (0..lines)
        .map(|i| xytext_y + (n - f64::from(i as u32) - 1.0 + baseline) * dy)
        .collect();
    let tail_y = if xy_y > xytext_y {
        xytext_y + n * dy // top of the block
    } else {
        xytext_y // bottom of the block
    };
    (centres, tail_y)
}

/// The panel-2 callouts: a shaft from the text block to the point on the curve, a filled
/// arrow head at the tip (the reference's `arrowstyle="->"`), and the text.
///
/// The head is sized in *pixels* through [`arrow_head_points`], because the axis is
/// logarithmic: a fixed number of hertz is a different number of pixels at 35 Hz and at
/// 5 kHz, and a linear offset pushed the sub-100 Hz callouts off the left edge.
macro_rules! annotations {
    ($chart:expr, $self:expr, $rect:expr, $win:expr) => {{
        use plotters::prelude::*;
        let w = $win;
        let (y_lo, y_hi) = w.yr;
        let dy = (y_hi - y_lo) * 0.045;
        for a in &$self.annotations {
            // A log axis cannot place a non-positive frequency at all.
            if a.xy_x <= 0.0 || a.xytext.0 <= 0.0 || !a.xy_y.is_finite() {
                continue;
            }
            let color = if a.green { ARROW_GREEN } else { ARROW_CYAN };
            let lines: Vec<&str> = if a.text.is_empty() {
                Vec::new()
            } else {
                a.text.split('\n').collect()
            };
            let (centres, block_tail) =
                $crate::plot::callout_layout(lines.len(), a.xytext.1, a.xy_y, dy);
            let tail_y = block_tail.clamp(y_lo, y_hi);
            $chart
                .draw_series(vec![PathElement::new(
                    vec![(a.xytext.0, tail_y), (a.xy_x, a.xy_y)],
                    $crate::plot::stroke_a(color, 2, 0.95),
                )])
                .map_err($crate::plot::dv)?;
            // A filled head, sized in pixels: matplotlib's `->` arrow keeps its size at
            // every frequency, which a data-coordinate triangle on a log axis cannot.
            let head = $crate::plot::arrow_head_points(
                (a.xytext.0, tail_y),
                (a.xy_x, a.xy_y),
                $crate::plot::Window {
                    xr: ($crate::plot::F_MIN, $crate::plot::F_MAX),
                    yr: (y_lo, y_hi),
                    log_x: true,
                },
                $rect,
                (12.0, 5.0),
            );
            $chart
                .draw_series(vec![plotters::element::Polygon::new(
                    head.to_vec(),
                    $crate::plot::fill(color, 0.95),
                )])
                .map_err($crate::plot::dv)?;

            for (line, &ty) in lines.iter().zip(&centres) {
                $chart
                    .draw_series(vec![Text::new(
                        line.to_string(),
                        (a.xytext.0, ty),
                        ("sans-serif", $crate::plot::ANNOT_FONT)
                            .into_font()
                            .color(&color.color()),
                    )])
                    .map_err($crate::plot::dv)?;
            }
        }
    }};
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
        // Gated on the flag, not just on the columns: the default figure is the one a reader may be
        // comparing against a capture of their own, and a curve appearing in it because a column
        // happens to exist is a change nobody asked for.
        let raw_lr_diff: Vec<f64> = if raw_l.is_empty() || !in_.show_ears {
            Vec::new()
        } else {
            raw_l.iter().zip(&raw_r).map(|(&l, &r)| l - r).collect()
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
        let ir_lin_centered = crate::ir::linear_phase_ir_centered(in_.mag_lin, in_.n_fft_rt);
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
            .map(|i| -1.0 + (1.0 - 1e3 / fs) * i as f64 / (lead_n - 1) as f64)
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
        // The reference's labels are Python `str.format` templates over the same five
        // values its `fmt_ctx` carries.
        let ctx = FmtCtx {
            notch_mode: in_.notch_mode,
            pre_rt: in_.pre_rt,
            unwin_rt: in_.unwin_rt,
            boost_rt: in_.boost_rt,
            notch_db,
        };
        let subst = |s: String| substitute(&s, &ctx);
        // The number in the legend is the figure's own claim about the pair, so it is read off the
        // trace being drawn rather than quoted from a report somewhere else.
        let lr_diff_label = if raw_lr_diff.is_empty() {
            String::new()
        } else {
            let apart = raw_lr_diff.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            subst(
                str_meta("raw_lr_diff_label")
                    .unwrap_or_else(|| format!("Left − Right (ears, up to {apart:.1} dB apart)")),
            )
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
            .unwrap_or_else(|| build_auto_annotations(&f_plot, &wav_eq, model.hf_bypass_hz))
            .into_iter()
            .map(|mut a| {
                a.text = substitute(&a.text, &ctx);
                a
            })
            .collect();

        Self {
            title: subst(str_meta("title_template").unwrap_or(default_title)),
            bypass_label: subst(str_meta("bypass_label").unwrap_or_else(|| {
                format!(
                    "Uncorrected >{:.1} kHz Coupler Null Zone (Clamped to 0.0 dB)",
                    model.hf_bypass_hz / 1000.0
                )
            })),
            lr_label: subst(
                str_meta("lr_label").unwrap_or_else(|| "Raw L / R Channels".to_string()),
            ),
            raw_label: subst(
                str_meta("raw_label").unwrap_or_else(|| format!("Raw {}", model.name)),
            ),
            target_label: subst(str_meta("target_label").unwrap_or_else(|| {
                if model.target_name == "harman_2018" {
                    "Harman Over-Ear 2018 Target".to_string()
                } else {
                    format!("Target Curve ({})", model.target_name)
                }
            })),
            autoeq_label: subst(
                str_meta("autoeq_ax1_label")
                    .unwrap_or_else(|| "Standard AutoEq Response".to_string()),
            ),
            wav_label: subst(str_meta("wav_ax1_label").unwrap_or(default_wav)),
            raw_err_label: subst(
                str_meta("raw_err_ax2_label").unwrap_or_else(|| "Raw Unsmoothed Error".to_string()),
            ),
            autoeq_gain_label: subst(
                str_meta("autoeq_ax2_label").unwrap_or_else(|| "Standard AutoEq Curve".to_string()),
            ),
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
            raw_lr_diff,
            lr_diff_label,
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
            ax1_ylim_fixed: meta_pair(meta, "ax1_ylim").is_some(),
            ax2_ylim,
            ax2_ylim_fixed: meta_pair(meta, "ax2_ylim").is_some(),
            ax1_yticks: meta_ticks(meta, "ax1_yticks", &AX1_YTICKS),
            ax2_yticks: meta_ticks(meta, "ax2_yticks", &AX2_YTICKS),
            annotations,
        }
    }

    /// The four plot-area rectangles `(x0, y0, x1, y1)` in figure pixels.
    ///
    /// These mirror the band splits `render_to` performs; `plotters` keeps `Rect`'s own
    /// splitting private, so the geometry is spelled out here and the rendered figure is
    /// checked against it by a test (the axis frame has to land exactly on these edges).
    #[must_use]
    pub fn panel_rects() -> [(i32, i32, i32, i32); 4] {
        let row_h = |h: u32| TITLE_H + h + X_LABEL_H;
        let band1_y = M_TOP;
        let band2_y = band1_y + row_h(ROW1_H) + ROW_GAP;
        let band3_y = band2_y + row_h(ROW2_H) + ROW_GAP;
        let left_band = M_LEFT - Y_LABEL_W;
        let wide = M_RIGHT - M_LEFT;
        let plot = |band_x: u32, band_y: u32, w: u32, h: u32| {
            let x0 = band_x as i32 + Y_LABEL_W as i32;
            let y0 = band_y as i32 + TITLE_H as i32;
            (x0, y0, x0 + (w - RIGHT_PAD) as i32, y0 + h as i32)
        };
        // Each band is `Y_LABEL_W` wider than its plot area, so the second column's band
        // starts a full label-area plus panel past the first one.
        [
            plot(left_band, band1_y, wide, ROW1_H),
            plot(left_band, band2_y, wide, ROW2_H),
            plot(left_band, band3_y, COL_W, ROW3_H),
            plot(
                left_band + Y_LABEL_W + COL_W + COL_GAP,
                band3_y,
                COL_W,
                ROW3_H,
            ),
        ]
    }

    /// Write `<file_prefix>_final_verification.png`; returns the path written.
    ///
    /// The bitmap backend would also flush when the areas are dropped, but that error is
    /// unreachable, so `present()` is called explicitly: a failed write must be reported
    /// rather than leave a truncated file behind an `Ok`.
    pub fn render(&self, prefix: &str) -> Result<std::path::PathBuf> {
        self.render_to(&std::path::PathBuf::from(format!(
            "{prefix}_final_verification.png"
        )))
    }

    /// Render to an explicit path. `render` is this with the reference's file naming;
    /// tests use this directly so they never have to move the process's working directory.
    pub fn render_to(&self, path: &std::path::Path) -> Result<std::path::PathBuf> {
        use plotters::prelude::*;

        let written = path.to_path_buf();
        let root = BitMapBackend::new(path, (FIG_W, FIG_H)).into_drawing_area();
        root.fill(&FIG_BG.color())
            .map_err(dv)
            .map_err(io_err(path, "figure background failed"))?;

        // Rows: `split_vertically(n)` returns the top `n` pixels and the remainder, so
        // the inter-row gaps are the peeled bands that get dropped and let the figure
        // background show through.
        let (_, rest) = root.split_vertically(M_TOP);
        let (band1, rest) = rest.split_vertically(TITLE_H + ROW1_H + X_LABEL_H);
        let (_, rest) = rest.split_vertically(ROW_GAP);
        let (band2, rest) = rest.split_vertically(TITLE_H + ROW2_H + X_LABEL_H);
        let (_, rest) = rest.split_vertically(ROW_GAP);
        let (band3, _) = rest.split_vertically(TITLE_H + ROW3_H + X_LABEL_H);

        // Columns: `split_horizontally(n)` returns the left `n` pixels *first* and the
        // remainder second, so the left margin is the discarded first element and the
        // panel is what remains. Peeling `M_LEFT - Y_LABEL_W` and then keeping
        // `Y_LABEL_W + (M_RIGHT - M_LEFT)` puts every plot area exactly in
        // `[M_LEFT, M_RIGHT)`, matching the reference's `left`/`right` figure margins.
        let (_, row) = band1.split_horizontally(M_LEFT - Y_LABEL_W);
        let (band1, _) = row.split_horizontally(Y_LABEL_W + (M_RIGHT - M_LEFT));
        let (_, row) = band2.split_horizontally(M_LEFT - Y_LABEL_W);
        let (band2, _) = row.split_horizontally(Y_LABEL_W + (M_RIGHT - M_LEFT));

        // Bottom row: [ y labels | plot ] , gap , [ y labels | plot ].
        let (_, row) = band3.split_horizontally(M_LEFT - Y_LABEL_W);
        let (band3, rest) = row.split_horizontally(Y_LABEL_W + COL_W);
        let (_, rest) = rest.split_horizontally(COL_GAP);
        let (band4, _) = rest.split_horizontally(Y_LABEL_W + COL_W);

        self.draw_level(&band1).map_err(io_err(path, "panel 1"))?;
        self.draw_gain(&band2).map_err(io_err(path, "panel 2"))?;
        self.draw_phase(&band3).map_err(io_err(path, "panel 3"))?;
        self.draw_impulse(&band4).map_err(io_err(path, "panel 4"))?;

        root.present()
            .map_err(dv)
            .map_err(io_err(path, "PNG write failed"))?;
        Ok(written)
    }

    /// Panel 1: acoustic level — raw against target, AutoEq, and the built FIR.
    fn draw_level(&self, band: &BitmapArea<'_>) -> PlotResult {
        use plotters::prelude::*;
        let autoeq_resp = add(&self.raw_avg, &self.autoeq);
        let wav_resp = add(&self.raw_avg, &self.wav_eq);
        // `hypreq.py:2415` uses a metadata window verbatim (`set_ylim(*meta["ax1_ylim"])`)
        // and lets matplotlib clip the traces; only when the model omits one does it fall
        // back to padding around the curves. Widening a configured window would defeat it.
        let (y0, y1) = self.level_window();
        let win = Window {
            xr: (F_MIN, F_MAX),
            yr: (y0, y1),
            log_x: true,
        };
        draw_title(band, &self.title, TITLE_FONT)?;
        draw_axes(
            band,
            &AxesSpec {
                x_range: (F_MIN, F_MAX),
                y_range: (y0, y1),
                log_x: true,
                xticks: &MAJOR_FREQS,
                yticks: &self.ax1_yticks,
                y_desc: "Acoustic Level (dBr)",
                x_desc: None,
                zero_line: true,
            },
        )?;
        let mut chart = ChartBuilder::on(band)
            .margin(0u32)
            .margin_top(TITLE_H)
            .margin_right(RIGHT_PAD)
            .x_label_area_size(X_LABEL_H)
            .y_label_area_size(Y_LABEL_W)
            .build_cartesian_2d((F_MIN..F_MAX).log_scale(), y0..y1)
            .map_err(dv)?;

        bypass_span!(&mut chart, self.hf_bypass_hz, self.bypass_to_hz, win);
        // The two raw channels sit under everything else, at the reference's alpha.
        trace!(chart, win, &self.f_plot, &self.raw_l, stroke_a(RAW_LR, 2, 0.38), dash DOT.0, DOT.1);
        trace!(chart, win, &self.f_plot, &self.raw_r, stroke_a(RAW_LR, 2, 0.38), dash DOT.0, DOT.1);
        trace!(chart, win, &self.f_plot, &self.raw_avg, stroke_a(RAW, 3, 0.9), dash DASH.0, DASH.1);
        trace!(
            chart,
            win,
            &self.f_plot,
            &self.harm_plot,
            stroke_a(TARGET, 7, 0.88)
        );
        trace!(
            chart,
            win,
            &self.f_plot,
            &autoeq_resp,
            stroke_a(AUTOEQ_RED, 3, 0.92)
        );
        trace!(
            chart,
            win,
            &self.f_plot,
            &wav_resp,
            stroke_a(WAV_GREEN, 5, 0.98)
        );

        draw_legend(band, &self.level_legend(), LegendPos::LowerLeft)
    }

    /// Panel 1's y window.
    ///
    /// A window the model configured through `plot_meta.ax1_ylim` is used verbatim, the
    /// way `hypreq.py:2415` hands it straight to `set_ylim`; only when the model omits one
    /// is the fallback padded around the curves that are actually drawn. Widening a
    /// configured window would defeat the model author's choice.
    #[must_use]
    pub fn level_window(&self) -> (f64, f64) {
        if self.ax1_ylim_fixed {
            return self.ax1_ylim;
        }
        let autoeq_resp = add(&self.raw_avg, &self.autoeq);
        let wav_resp = add(&self.raw_avg, &self.wav_eq);
        widen_to(
            self.ax1_ylim,
            &[
                (&self.raw_l, 1.0),
                (&self.raw_r, 1.0),
                (&self.raw_avg, 1.0),
                (&self.harm_plot, 1.0),
                (&autoeq_resp, 1.0),
                (&wav_resp, 1.0),
                (&[0.0][..], 0.0),
            ],
        )
    }

    /// Panel 2's y window, with the same configured-or-computed rule as [`Self::level_window`].
    #[must_use]
    pub fn gain_window(&self) -> (f64, f64) {
        if self.ax2_ylim_fixed {
            return self.ax2_ylim;
        }
        let err: Vec<f64> = self
            .harm_plot
            .iter()
            .zip(&self.raw_avg)
            .map(|(&h, &r)| h - r)
            .collect();
        widen_to(
            self.ax2_ylim,
            &[
                (&err, 1.0),
                (&self.autoeq, 1.0),
                (&self.wav_eq, 1.0),
                (&[0.0][..], 0.0),
            ],
        )
    }

    /// Panel 1's legend, in the reference's order.
    ///
    /// The raw L/R entry only appears when the measurement really carries both columns:
    /// `hypreq.py:2412` labels that trace inside `if len(raw_pts[0]) >= 4`, so a 2-column
    /// model must not show a swatch for a line that was never drawn.
    fn level_legend(&self) -> Vec<LegendItem<'_>> {
        let mut items = vec![LegendItem {
            text: &self.bypass_label,
            swatch: Swatch::Patch(BYPASS),
        }];
        if !self.raw_l.is_empty() {
            items.push(LegendItem {
                text: &self.lr_label,
                swatch: Swatch::Line {
                    color: RAW_LR,
                    width: 2,
                    dash: Some(DOT),
                },
            });
        }
        items.extend([
            LegendItem {
                text: &self.raw_label,
                swatch: Swatch::Line {
                    color: RAW,
                    width: 3,
                    dash: Some(DASH),
                },
            },
            LegendItem {
                text: &self.target_label,
                swatch: Swatch::Line {
                    color: TARGET,
                    width: 7,
                    dash: None,
                },
            },
            LegendItem {
                text: &self.autoeq_label,
                swatch: Swatch::Line {
                    color: AUTOEQ_RED,
                    width: 3,
                    dash: None,
                },
            },
            LegendItem {
                text: &self.wav_label,
                swatch: Swatch::Line {
                    color: WAV_GREEN,
                    width: 5,
                    dash: None,
                },
            },
        ]);
        items
    }

    /// Panel 2: filter gain, plus the callouts. The reference gives this panel no title,
    /// only a y unit, so none is drawn.
    fn draw_gain(&self, band: &BitmapArea<'_>) -> PlotResult {
        use plotters::prelude::*;
        let err: Vec<f64> = self
            .harm_plot
            .iter()
            .zip(&self.raw_avg)
            .map(|(&h, &r)| h - r)
            .collect();
        let (y0, y1) = self.gain_window();
        let win = Window {
            xr: (F_MIN, F_MAX),
            yr: (y0, y1),
            log_x: true,
        };
        draw_axes(
            band,
            &AxesSpec {
                x_range: (F_MIN, F_MAX),
                y_range: (y0, y1),
                log_x: true,
                xticks: &MAJOR_FREQS,
                yticks: &self.ax2_yticks,
                y_desc: "Filter Gain (dBr)",
                x_desc: None,
                zero_line: true,
            },
        )?;
        let mut chart = ChartBuilder::on(band)
            .margin(0u32)
            .margin_top(TITLE_H)
            .margin_right(RIGHT_PAD)
            .x_label_area_size(X_LABEL_H)
            .y_label_area_size(Y_LABEL_W)
            .build_cartesian_2d((F_MIN..F_MAX).log_scale(), y0..y1)
            .map_err(dv)?;

        bypass_span!(&mut chart, self.hf_bypass_hz, self.bypass_to_hz, win);
        trace!(chart, win, &self.f_plot, &err, stroke_a(RAW_ERR, 2, 0.65), dash DOT.0, DOT.1);
        trace!(chart, win, &self.f_plot, &self.autoeq, stroke_a(AUTOEQ_AMBER, 4, 0.95), dash DASH.0, DASH.1);
        trace!(
            chart,
            win,
            &self.f_plot,
            &self.wav_eq,
            stroke_a(WAV_GREEN, 5, 0.98)
        );

        // The callouts are drawn in this panel's own window, so their arrowheads are
        // measured against the same y-scale the chart maps with.
        annotations!(&mut chart, self, plot_rect(band), win);

        if !self.raw_lr_diff.is_empty() {
            // Drawn last so it reads on top of the error trace it explains, and in the ears' own
            // blue, which panel 2 has no other use for.
            trace!(
                chart,
                win,
                &self.f_plot,
                &self.raw_lr_diff,
                stroke_a(RAW_LR, 2, 0.95)
            );
        }

        let mut items = vec![
            LegendItem {
                text: &self.raw_err_label,
                swatch: Swatch::Line {
                    color: RAW_ERR,
                    width: 2,
                    dash: Some(DOT),
                },
            },
            LegendItem {
                text: &self.autoeq_gain_label,
                swatch: Swatch::Line {
                    color: AUTOEQ_AMBER,
                    width: 4,
                    dash: Some(DASH),
                },
            },
            LegendItem {
                text: &self.wav_gain_label,
                swatch: Swatch::Line {
                    color: WAV_GREEN,
                    width: 5,
                    dash: None,
                },
            },
        ];
        if !self.raw_lr_diff.is_empty() {
            items.push(LegendItem {
                text: &self.lr_diff_label,
                swatch: Swatch::Line {
                    color: RAW_LR,
                    width: 2,
                    dash: None,
                },
            });
        }
        draw_legend(band, &items, LegendPos::LowerLeft)
    }

    /// Panel 3: the minimum-phase phase angle of the same spectrum.
    fn draw_phase(&self, band: &BitmapArea<'_>) -> PlotResult {
        use plotters::prelude::*;
        // `hypreq.py:2473` pins this axis with `set_ylim(-90, 90)`, so the configured
        // ticks (-90, -45, 0, 45, 90) always have room; autoscaling to the data, which
        // stays inside roughly +/-35 deg, would leave 0 as the only visible tick.
        let (y0, y1) = PHASE_YLIM;
        let win = Window {
            xr: (F_MIN, F_MAX),
            yr: (y0, y1),
            log_x: true,
        };
        draw_title(band, &self.phase_title, SMALL_TITLE_FONT)?;
        draw_axes(
            band,
            &AxesSpec {
                x_range: (F_MIN, F_MAX),
                y_range: (y0, y1),
                log_x: true,
                xticks: &MAJOR_FREQS,
                yticks: &PHASE_YTICKS,
                y_desc: "Phase (deg)",
                x_desc: Some("Frequency (Hz)"),
                zero_line: true,
            },
        )?;
        let mut chart = ChartBuilder::on(band)
            .margin(0u32)
            .margin_top(TITLE_H)
            .margin_right(RIGHT_PAD)
            .x_label_area_size(X_LABEL_H)
            .y_label_area_size(Y_LABEL_W)
            .build_cartesian_2d((F_MIN..F_MAX).log_scale(), y0..y1)
            .map_err(dv)?;
        trace!(
            chart,
            win,
            &self.f_plot,
            &self.wav_phase,
            stroke_a(PHASE, 4, 0.95)
        );

        let items = [LegendItem {
            text: &self.phase_label,
            swatch: Swatch::Line {
                color: PHASE,
                width: 4,
                dash: None,
            },
        }];
        draw_legend(band, &items, LegendPos::UpperLeft)
    }

    /// Panel 4: the minimum-phase IR against the linear-phase IR of the same magnitude,
    /// which pre-rings before the transient.
    fn draw_impulse(&self, band: &BitmapArea<'_>) -> PlotResult {
        use plotters::prelude::*;
        draw_title(band, &self.ir_title, SMALL_TITLE_FONT)?;
        draw_axes(
            band,
            &AxesSpec {
                x_range: (IR_XMIN, IR_XMAX),
                y_range: (IR_YMIN, IR_YMAX),
                log_x: false,
                xticks: &IR_XTICKS,
                yticks: &IR_YTICKS,
                y_desc: "Normalized Amplitude",
                x_desc: Some("Time Relative to Transient Peak (ms)"),
                zero_line: true,
            },
        )?;
        let win = Window {
            xr: (IR_XMIN, IR_XMAX),
            yr: (IR_YMIN, IR_YMAX),
            log_x: false,
        };
        let mut chart = ChartBuilder::on(band)
            .margin(0u32)
            .margin_top(TITLE_H)
            .margin_right(RIGHT_PAD)
            .x_label_area_size(X_LABEL_H)
            .y_label_area_size(Y_LABEL_W)
            .build_cartesian_2d(IR_XMIN..IR_XMAX, IR_YMIN..IR_YMAX)
            .map_err(dv)?;

        // The 0 ms guide, drawn under both traces like the reference's `axvline`.
        let rect = plot_rect(band);
        dotted_vline(
            band,
            x_pix(0.0, IR_XMIN, IR_XMAX, false, rect),
            rect.1,
            rect.3,
            ZERO_MS,
            2,
        )?;

        trace!(chart, win, &self.lin_t_ms, &self.lin_ir, stroke_a(AUTOEQ_RED, 3, 0.88), dash DASH.0, DASH.1);
        trace!(
            chart,
            win,
            &self.mp_t_ms,
            &self.mp_ir,
            stroke_a(WAV_GREEN, 4, 0.96)
        );

        let items = [
            LegendItem {
                text: &self.lin_ir_label,
                swatch: Swatch::Line {
                    color: AUTOEQ_RED,
                    width: 3,
                    dash: Some(DASH),
                },
            },
            LegendItem {
                text: &self.mp_ir_label,
                swatch: Swatch::Line {
                    color: WAV_GREEN,
                    width: 4,
                    dash: None,
                },
            },
        ];
        draw_legend(band, &items, LegendPos::UpperRight)
    }
}

/// Font sizes, in pixels, for a 1950 x 1725 figure.
const TITLE_FONT: u32 = 26;
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
/// A drawing result carrying a value.
type Plot<T> = std::result::Result<T, crate::error::Error>;
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

/// Widen a configured axis window until every given curve (with a margin in data units)
/// and every listed point fits inside it.
fn widen_to(range: (f64, f64), series: &[(&[f64], f64)]) -> (f64, f64) {
    let mut out = range;
    for &(values, margin) in series {
        for &v in values.iter().filter(|v| v.is_finite()) {
            out.0 = out.0.min(v - margin);
            out.1 = out.1.max(v + margin);
        }
    }
    out
}

/// The values `hypreq.py` keeps in `fmt_ctx` for `str.format` on plot labels.
pub struct FmtCtx<'a> {
    notch_mode: &'a str,
    pre_rt: f64,
    unwin_rt: f64,
    boost_rt: f64,
    notch_db: f64,
}

/// Expand the reference's `str.format` fields, e.g. `{notch_mode}`, `{pre_rt:.2f}`,
/// `{notch_db:+.2f}`. Doubled braces are literal braces in either direction, as in Python.
/// An unknown field is left exactly as written, so a typo in the database stays visible in the
/// figure instead of being papered over with a number.
#[must_use]
pub fn substitute(text: &str, ctx: &FmtCtx<'_>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '}' => {
                out.push('}');
                if chars.peek() == Some(&'}') {
                    chars.next();
                }
            }
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '{' => {
                let mut field = String::new();
                let mut closed = false;
                for ch in chars.by_ref() {
                    if ch == '}' {
                        closed = true;
                        break;
                    }
                    field.push(ch);
                }
                match (closed, ctx.field(&field)) {
                    (true, Some(value)) => out.push_str(&value),
                    // Unknown field: reproduce the text as written, brace and all.
                    _ => {
                        out.push('{');
                        out.push_str(&field);
                        if closed {
                            out.push('}');
                        }
                    }
                }
            }
            other => out.push(other),
        }
    }
    out
}

impl FmtCtx<'_> {
    /// The value behind one `name:spec` field, or `None` when the name is unknown.
    fn field(&self, field: &str) -> Option<String> {
        let (name, spec) = match field.split_once(':') {
            Some((n, s)) => (n, Some(s)),
            None => (field, None),
        };
        match name {
            "notch_mode" => Some(self.notch_mode.to_string()),
            "pre_rt" => Some(format_number(self.pre_rt, spec)),
            "unwin_rt" => Some(format_number(self.unwin_rt, spec)),
            "boost_rt" => Some(format_number(self.boost_rt, spec)),
            "notch_db" => Some(format_number(self.notch_db, spec)),
            _ => None,
        }
    }
}

/// Python's `format(v, spec)` for the fixed-point specs the database uses: an optional
/// leading `+` and an optional `.Nf` precision, e.g. `+.2f`. Anything else falls back to
/// the default float rendering, which is what an empty spec does in Python.
#[must_use]
pub fn format_number(v: f64, spec: Option<&str>) -> String {
    let spec = spec.unwrap_or("");
    let plus = spec.starts_with('+');
    let digits = spec
        .trim_start_matches('+')
        .strip_suffix('f')
        .and_then(|t| t.strip_prefix('.'))
        .and_then(|d| d.parse::<usize>().ok());
    let mut out = match digits {
        Some(d) => format!("{v:.d$}"),
        None => format!("{v}"),
    };
    if plus && !out.starts_with('-') {
        out.insert(0, '+');
    }
    out
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
        // The curve is interpolated in log frequency, so a non-positive knot would put
        // -inf into the grid and poison every point below it.
        if nums.len() >= 2 && nums[0] > 0.0 && nums[1].is_finite() {
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
    // A model with nothing to plot still gets the bypass note rather than panicking here.
    let g_20 = wav_eq.first().copied().unwrap_or(0.0);
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
    let treb_hi = hf_bypass_hz.clamp(4800.0, 16000.0);
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
            // `hypreq.py:2324` is `min(14000, max(f*1.18, f+500))`: the 14 kHz cap is
            // applied last, so a bypass that starts at 13.5 kHz or higher stays inside.
            xy_x: (hf_bypass_hz * 1.18).max(hf_bypass_hz + 500.0).min(14000.0),
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
            derivation: None,
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
            show_ears: false,
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

    /// The coupler-null zone against the axis window. The reference clips it for free;
    /// here an unclamped rectangle paints over the frame and into the neighbouring panel,
    /// and a model whose bypass starts past the plot edge shades the entire plot.
    #[test]
    fn the_bypass_zone_is_intersected_with_the_axis_it_is_drawn_on() {
        let win = (20.0, 20_000.0);
        // Fully inside: untouched.
        assert_eq!(
            clamp_span(6_000.0, 20_000.0, win),
            Some((6_000.0, 20_000.0))
        );
        // Running past the right edge: the rectangle stops at the edge.
        assert_eq!(
            clamp_span(18_000.0, 96_000.0, win),
            Some((18_000.0, 20_000.0)),
            "an unclamped span paints over the frame"
        );
        // Starting before the left edge, and both ends outside on one side.
        assert_eq!(clamp_span(1.0, 40.0, win), Some((20.0, 40.0)));
        assert_eq!(
            clamp_span(30_000.0, 96_000.0, win),
            None,
            "entirely off-axis"
        );
        assert_eq!(clamp_span(1.0, 9.0, win), None, "entirely off-axis, left");
        // Touching the window at one edge only is zero width, not a hairline.
        assert_eq!(clamp_span(20_000.0, 40_000.0, win), None);
        assert_eq!(clamp_span(5.0, 20.0, win), None);
        // A window the span swallows whole still draws, clamped on both sides.
        assert_eq!(clamp_span(1.0, 1e6, win), Some((20.0, 20_000.0)));
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

    /// A scratch directory unique to this test: the suite runs tests in parallel, and no
    /// test may touch the process's working directory.
    fn scratch(name: &str) -> std::path::PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("hypreq-plot-{}-{name}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Decode a rendered figure into `(width, height, RGB8 pixels row-major)`.
    fn decode_png(path: &std::path::Path) -> (u32, u32, Vec<u8>) {
        let mut decoder = png::Decoder::new(std::fs::File::open(path).expect("open png"));
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let mut reader = decoder.read_info().expect("png header");
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("png frame");
        let px = info.width as usize * info.height as usize * 3;
        buf.truncate(px);
        (info.width, info.height, buf)
    }

    fn at(pixels: &[u8], w: u32, x: usize, y: usize) -> [u8; 3] {
        let i = (y * w as usize + x) * 3;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    }

    fn close(c: [u8; 3], want: Rgb, tol: i32) -> bool {
        let plotters::prelude::RGBColor(r, g, b) = want.color();
        let want = [r, g, b];
        (0..3)
            .map(|k| (i32::from(c[k]) - i32::from(want[k])).abs())
            .sum::<i32>()
            <= tol
    }

    /// A realistically sized input pair for the render tests: the figure reads the IR
    /// spectrum at `n_fft_rt`, so a toy FFT length would hide geometry bugs.
    fn full_inputs() -> (Vec<f64>, Vec<f64>) {
        let n = 4096usize;
        let ir: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / 48.0;
                (t * 0.7).sin() * (-t / 40.0).exp() * 0.4
            })
            .collect();
        let mag: Vec<f64> = (0..n / 2 + 1)
            .map(|k| 1.0 - 0.4 * (k as f64 / n as f64))
            .collect();
        (ir, mag)
    }

    fn full_inputs_ref<'a>(ir: &'a [f64], mag: &'a [f64]) -> PlotInputs<'a> {
        PlotInputs {
            ir_rt: ir,
            mag_lin: mag,
            n_fft_rt: 4096,
            fs: 48000,
            boost_rt: 4.25,
            pre_rt: -6.5,
            unwin_rt: -152.6,
            active_ctrl: &[(20.0, 2.0), (1000.0, 0.0), (20000.0, 4.0)],
            notch_mode: "safe",
            show_ears: false,
        }
    }

    #[test]
    fn render_writes_a_png_of_the_expected_size() {
        let dir = scratch("size");
        let (ir, mag) = full_inputs();
        let d = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        let path = d.render_to(&dir.join("figure.png")).expect("render");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "not a PNG");
        // IHDR width and height are the two big-endian u32s at offsets 16 and 20.
        let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!((w, h), (FIG_W, FIG_H), "figure must be 1950x1725");
        assert!(bytes.len() > 20_000, "suspiciously small figure");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The things that sit *on* an axis limit, which is what a window filter quietly
    /// deletes: the end tick labels (`20` and `20k` under the level and phase panels, `-1`
    /// and `4` under the impulse panel), the phase panel's `90` and `-90`, and the two
    /// x-axis titles — each title pinned to the panel that owns it, because the row shares
    /// nothing and matplotlib shares no axes here. Dropping a limit tick looks like tidying
    /// the frame, so this test is verified discriminating by tightening `ticks_in` to
    /// `range.0 < *v && *v < range.1` and watching it fail on the pixels, not just numbers.
    #[test]
    fn the_labels_on_the_axis_limits_are_drawn() {
        let dir = scratch("edge-labels");
        let (ir, mag) = full_inputs();
        let d = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        let path = d.render_to(&dir.join("figure.png")).expect("render");
        let (w, _, px) = decode_png(&path);

        // Ink pixels in a box, and how far that ink reaches horizontally.
        let box_ink = |x0: usize, y0: usize, x1: usize, y1: usize| -> usize {
            (y0..y1)
                .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                .filter(|&(x, y)| !close(at(&px, w, x, y), FIG_BG, 6))
                .count()
        };
        let box_width = |x0: usize, y0: usize, x1: usize, y1: usize| -> usize {
            let cols: Vec<usize> = (x0..x1)
                .filter(|&x| (y0..y1).any(|y| !close(at(&px, w, x, y), FIG_BG, 6)))
                .collect();
            cols.first()
                .copied()
                .map(|a| cols[cols.len() - 1] - a)
                .unwrap_or(0)
        };

        let rects = PlotData::panel_rects();
        let [level, _gain, phase, impulse] = rects;

        // End tick labels, below each panel's frame and clear of its 3 px tick marks.
        for (idx, r, ends) in [
            (1usize, level, "20 / 20k"),
            (3, phase, "20 / 20k"),
            (4, impulse, "-1 / 4"),
        ] {
            let (x0, x1, y1) = (r.0 as usize, r.2 as usize, r.3 as usize);
            let left = box_ink(x0.saturating_sub(18), y1 + 4, x0 + 22, y1 + 22);
            let right = box_ink(x1.saturating_sub(24), y1 + 4, x1 + 26, y1 + 22);
            assert!(
                left > 20,
                "panel {idx} has no label at its left limit ({ends}): {left} px"
            );
            assert!(
                right > 20,
                "panel {idx} has no label at its right limit ({ends}): {right} px"
            );
        }

        // The phase panel keeps ±90 as labelled limits, left of its frame.
        let (x0, y0, y1) = (phase.0 as usize, phase.1 as usize, phase.3 as usize);
        let top = box_ink(
            x0.saturating_sub(40),
            y0.saturating_sub(11),
            x0 - 8,
            y0 + 12,
        );
        let bottom = box_ink(x0.saturating_sub(40), y1 - 11, x0 - 8, y1 + 12);
        assert!(
            top > 12,
            "no label at the phase panel's +90 limit: {top} px"
        );
        assert!(
            bottom > 12,
            "no label at the phase panel's -90 limit: {bottom} px"
        );

        // Each x-axis title on its own panel, told apart by how far the string reaches:
        // "Frequency (Hz)" belongs to the phase panel, "Time Relative to Transient Peak
        // (ms)" — two and a half times as long — to the impulse panel.
        for (idx, r, min_w, max_w, name) in [
            (3usize, phase, 60, 200, "Frequency (Hz)"),
            (
                4usize,
                impulse,
                220,
                900,
                "Time Relative to Transient Peak (ms)",
            ),
        ] {
            let (x0, x1, y1) = (r.0 as usize, r.2 as usize, r.3 as usize);
            let mid = (x0 + x1) / 2;
            let (bx0, by0, bx1, by1) = (mid.saturating_sub(320), y1 + 22, mid + 320, y1 + 50);
            let ink = box_ink(bx0, by0, bx1, by1);
            let wide = box_width(bx0, by0, bx1, by1);
            assert!(
                ink > 60,
                "panel {idx} has no x-axis title {name:?}: {ink} px"
            );
            assert!(
                wide > min_w && wide < max_w,
                "panel {idx} title is {wide} px wide, not {name:?} ({min_w}..{max_w})"
            );
        }
    }

    #[test]
    fn rendered_figure_draws_ink_in_every_panel_at_the_expected_place() {
        let dir = scratch("ink");
        let (ir, mag) = full_inputs();
        let d = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        let path = d.render_to(&dir.join("figure.png")).expect("render");
        let (w, h, px) = decode_png(&path);
        assert_eq!((w, h), (FIG_W, FIG_H));

        // The figure margin outside every band stays the untouched figure background.
        for (x, y) in [(4usize, 4usize), (30, 900), (1930, 1710), (30, 1710)] {
            assert!(
                close(at(&px, w, x, y), FIG_BG, 4),
                "margin pixel ({x},{y}) was drawn over: {:?}",
                at(&px, w, x, y)
            );
        }

        for (idx, &(x0, y0, x1, y1)) in PlotData::panel_rects().iter().enumerate() {
            let (x0, y0, x1, y1) = (x0 as usize, y0 as usize, x1 as usize, y1 as usize);
            let rows = ((y0 + 2)..(y1 - 2)).step_by(2);
            let n = ((y1 - y0 - 4) / 2).max(1);

            // Just outside the left and right edges the figure background must show
            // through: that is what pins the panel to these exact coordinates.
            let left_gutter = rows
                .clone()
                .filter(|&y| close(at(&px, w, x0 - 4, y), FIG_BG, 6))
                .count();
            assert!(
                left_gutter * 20 > n * 19,
                "panel {} is not bounded on the left at x={x0}: {left_gutter}/{n}",
                idx + 1
            );
            let right_gutter = rows
                .clone()
                .filter(|&y| close(at(&px, w, x1 + 3, y), FIG_BG, 6))
                .count();
            assert!(
                right_gutter * 20 > n * 19,
                "panel {} is not bounded on the right at x={x1}: {right_gutter}/{n}",
                idx + 1
            );

            // Inside, the panel background plus its content must cover the area (nothing
            // of the figure background may leak in), and real ink must be present.
            let mut panelish = 0usize;
            let mut ink = 0usize;
            for y in rows.clone() {
                for x in ((x0 + 2)..(x1 - 2)).step_by(2) {
                    let p = at(&px, w, x, y);
                    if close(p, FIG_BG, 6) {
                        continue;
                    }
                    panelish += 1;
                    if !close(p, PANEL_BG, 8) {
                        ink += 1;
                    }
                }
            }
            let samples = (((y1 - y0 - 4) / 2) * ((x1 - x0 - 4) / 2)).max(1);
            assert!(
                panelish * 20 > samples * 19,
                "panel {} leaks figure background: {panelish}/{samples}",
                idx + 1
            );
            let fraction = ink as f64 / samples as f64;
            assert!(
                fraction > 0.02,
                "panel {} has only {:.4} non-background pixels",
                idx + 1,
                fraction
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn arrow_head_keeps_its_pixel_size_across_the_log_axis() {
        let rect = (0i32, 0i32, 1755i32, 440i32);
        let xr = (20.0f64, 20000.0f64);
        let yr = (-10.0f64, 10.0f64);
        let len_px = 12.0;
        let half_px = 5.0;
        for tip_hz in [25.0f64, 1000.0, 15000.0] {
            let pts = arrow_head_points(
                (tip_hz * 3.0, 5.0),
                (tip_hz, 0.0),
                Window {
                    xr,
                    yr,
                    log_x: true,
                },
                rect,
                (len_px, half_px),
            );
            let px = |p: (f64, f64)| {
                (
                    x_pix_f(p.0, xr.0, xr.1, true, rect),
                    y_pix_f(p.1, yr.0, yr.1, rect),
                )
            };
            let dist = |a: (f64, f64), b: (f64, f64)| {
                ((px(a).0 - px(b).0).powi(2) + (px(a).1 - px(b).1).powi(2)).sqrt()
            };
            // The head is a triangle: tip, and two base corners either side of the shaft.
            // Its length is tip to the base centre, its width the distance between corners.
            let base_c = (
                x_from_pix((px(pts[1]).0 + px(pts[2]).0) / 2.0, xr.0, xr.1, true, rect),
                y_from_pix((px(pts[1]).1 + px(pts[2]).1) / 2.0, yr.0, yr.1, rect),
            );
            let length = dist((tip_hz, 0.0), base_c);
            let width = dist(pts[1], pts[2]);
            assert!(
                (length - len_px).abs() < 0.05,
                "head at {tip_hz} Hz is {length:.3} px long"
            );
            assert!(
                (width - 2.0 * half_px).abs() < 0.05,
                "head at {tip_hz} Hz is {width:.3} px wide"
            );
        }
    }

    #[test]
    fn ticks_on_the_axis_limits_are_still_labelled() {
        // The reference pins its ticks with FixedLocator, and matplotlib labels a tick that
        // sits exactly on the limit while omitting only its grid line. Filtering with a
        // strict inequality deleted 20 Hz, 20 kHz, -1/4 ms and the -15/-90/90 rows.
        let xf: Vec<f64> = ticks_in(&MAJOR_FREQS, (F_MIN, F_MAX)).copied().collect();
        assert_eq!(xf.first(), Some(&20.0), "left edge tick vanished");
        assert_eq!(xf.last(), Some(&20000.0), "right edge tick vanished");
        assert_eq!(xf.len(), MAJOR_FREQS.len());

        let ir_x: Vec<f64> = ticks_in(&IR_XTICKS, (IR_XMIN, IR_XMAX)).copied().collect();
        assert_eq!(ir_x.first(), Some(&-1.0));
        assert_eq!(ir_x.last(), Some(&4.0));

        let phase: Vec<f64> = ticks_in(&PHASE_YTICKS, PHASE_YLIM).copied().collect();
        assert_eq!(
            phase,
            PHASE_YTICKS.to_vec(),
            "every phase tick must survive"
        );

        let y: Vec<f64> = ticks_in(&[-15.0, -10.0, 0.0, 14.0], (-15.0, 14.0))
            .copied()
            .collect();
        assert_eq!(y, vec![-15.0, -10.0, 0.0, 14.0]);

        // …and only the interior ones earn a grid line, so nothing is drawn over the frame.
        assert!(!grid_line_inside(20.0, (F_MIN, F_MAX)));
        assert!(!grid_line_inside(20000.0, (F_MIN, F_MAX)));
        assert!(grid_line_inside(1000.0, (F_MIN, F_MAX)));
    }

    #[test]
    fn phase_panel_spans_the_full_range_the_ticks_need() {
        // `hypreq.py:2473` pins ax3 with set_ylim(-90, 90); the window the panel actually
        // uses is asserted on the rendered tick marks (rendered_panels_tick_their_axis_limits),
        // because autoscaling to the data - which stays inside +/-35 deg - would leave 0 as
        // the only visible tick.
        assert_eq!(PHASE_YLIM, (-90.0, 90.0));
        for t in PHASE_YTICKS {
            assert!(
                t >= PHASE_YLIM.0 && t <= PHASE_YLIM.1,
                "tick {t} falls outside the window"
            );
        }
        assert_eq!(
            ticks_in(&PHASE_YTICKS, PHASE_YLIM).count(),
            PHASE_YTICKS.len()
        );
        // The window is fixed, so a phase trace that swings past +/-90 deg is clipped at
        // the border instead of rescaling the axis and hiding the outer ticks.
        let (ir, mag) = full_inputs();
        let d = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        let w = Window {
            xr: (F_MIN, F_MAX),
            yr: PHASE_YLIM,
            log_x: true,
        };
        let runs = clip_polyline(&d.f_plot, &d.wav_phase, w);
        assert!(!runs.is_empty(), "the phase trace disappeared");
        for (fx, ph) in &runs {
            assert!(
                fx.iter()
                    .zip(ph)
                    .all(|(&f, &p)| (F_MIN..=F_MAX).contains(&f) && (-90.0..=90.0).contains(&p)),
                "phase trace escaped the fixed window"
            );
        }
    }

    #[test]
    fn legend_only_offers_the_traces_that_were_drawn() {
        // The 4-column fixture carries L and R, so the legend has an entry for them; a
        // 2-column measurement draws no such trace and must not advertise one.
        let (ir, mag) = inputs_64();
        let stereo = PlotData::build(&model(), &inputs(&ir, &mag));
        assert!(!stereo.raw_l.is_empty());
        let names: Vec<&str> = stereo.level_legend().iter().map(|i| i.text).collect();
        assert_eq!(names.len(), 6, "{names:?}");
        assert!(names.contains(&stereo.lr_label.as_str()), "{names:?}");

        let mut mono = model();
        mono.raw_pts = mono.raw_pts.iter().map(|p| vec![p[0], p[1]]).collect();
        let mono = PlotData::build(&mono, &inputs(&ir, &mag));
        assert!(mono.raw_l.is_empty(), "2-column raw data has no L/R trace");
        let names: Vec<&str> = mono.level_legend().iter().map(|i| i.text).collect();
        assert_eq!(names.len(), 5, "{names:?}");
        assert!(
            !names.contains(&mono.lr_label.as_str()),
            "phantom legend entry for a trace that was never drawn: {names:?}"
        );
    }

    /// Is there tick-mark ink just outside the panel edge at this column/row?
    fn tick_ink(px: &[u8], w: u32, x: usize, y: usize) -> bool {
        // Tick marks are drawn in `EDGE`, which is clearly separated from the figure
        // background, and they are the only thing drawn outside the panel frame.
        let sep = (i32::from(EDGE.0) - i32::from(FIG_BG.0))
            + (i32::from(EDGE.1) - i32::from(FIG_BG.1))
            + (i32::from(EDGE.2) - i32::from(FIG_BG.2));
        let p = at(px, w, x, y);
        let d = |want: Rgb| {
            (i32::from(p[0]) - i32::from(want.0)).abs()
                + (i32::from(p[1]) - i32::from(want.1)).abs()
                + (i32::from(p[2]) - i32::from(want.2)).abs()
        };
        d(EDGE) <= 24 && d(FIG_BG) > sep / 2
    }

    #[test]
    fn the_panel_rects_are_the_ones_the_harness_mirrors() {
        // `scripts/parity.py` keeps a copy of these four rectangles to find plotted ink, and a
        // copy drifts silently: the harness just counts ink somewhere else. The defect this pins
        // was exactly that — the fourth box sat 155 px left of the impulse panel, inside the
        // inter-column gap, so its "ink" was two panel frames and some axis labels, and the two
        // row-3 panels were named the wrong way round. If a layout constant moves, this test
        // fails by name and the harness list is updated with it.
        assert_eq!(
            PlotData::panel_rects(),
            [
                (127, 129, 1878, 569),    // 1: level
                (127, 736, 1878, 1176),   // 2: error / target
                (127, 1343, 909, 1673),   // 3: phase, the left panel of the bottom row
                (1096, 1343, 1878, 1673)  // 4: impulse, the right one
            ]
        );
        // The bottom row is two panels, not one wide one: the right band starts past the first
        // plot, the column gutter and this panel's own y-label area.
        let [_, _, phase, impulse] = PlotData::panel_rects();
        assert_eq!(phase.0, impulse.0 - (Y_LABEL_W + COL_W + COL_GAP) as i32);
        assert_eq!(phase.1, impulse.1);
        assert_eq!(impulse.2 - phase.2, (Y_LABEL_W + COL_W + COL_GAP) as i32);
    }

    /// A fixture whose two ears disagree by about 6 dB through the midrange, so the difference
    /// made by `--stereo` has something to draw. `model()`'s own columns are identical, which is
    /// realistic for a good pair and useless for an ink test.
    fn ear_model() -> Model {
        let mut m = model();
        m.raw_pts = vec![
            vec![20.0, 2.0, 2.0, 2.0],
            vec![200.0, 1.0, -1.0, 0.0],
            vec![1000.0, 0.0, -4.0, -2.0],
            vec![4000.0, -2.0, -8.0, -5.0],
            vec![20000.0, -6.0, -6.0, -6.0],
        ];
        m
    }

    #[test]
    fn the_ear_difference_is_traced_only_when_the_flag_asks() {
        let (ir, mag) = full_inputs();
        let on = PlotInputs {
            show_ears: true,
            ..full_inputs_ref(&ir, &mag)
        };
        let off = full_inputs_ref(&ir, &mag);

        let d_on = PlotData::build(&ear_model(), &on);
        assert_eq!(d_on.raw_lr_diff.len(), d_on.f_plot.len());
        let apart = d_on
            .raw_lr_diff
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(
            apart > 5.0 && apart < 6.5,
            "worst disagreement {apart:.2} dB, expected near the fixture's 6"
        );
        assert!(
            d_on.lr_diff_label.contains("Left")
                && d_on.lr_diff_label.contains(&format!("{apart:.1}")),
            "the legend does not quote the trace's own worst band: {}",
            d_on.lr_diff_label
        );

        // The default path is the figure people compare against their own captures.
        let d_off = PlotData::build(&ear_model(), &off);
        assert!(
            d_off.raw_lr_diff.is_empty(),
            "an unchanged run gained a curve"
        );
        assert!(d_off.lr_diff_label.is_empty());

        // One curve has nothing to subtract, however the flag is set.
        let mut mono = ear_model();
        mono.raw_pts = mono
            .raw_pts
            .iter()
            .map(|p| vec![p[0], *p.last().unwrap()])
            .collect();
        let d_mono = PlotData::build(&mono, &on);
        assert!(
            d_mono.raw_lr_diff.is_empty(),
            "a single curve was subtracted from itself and reported as a difference"
        );
    }

    #[test]
    fn the_ear_trace_draws_ink_in_the_error_panel() {
        let (ir, mag) = full_inputs();
        let on = PlotInputs {
            show_ears: true,
            ..full_inputs_ref(&ir, &mag)
        };
        let dir = scratch("ears-ink");
        let on_path = PlotData::build(&ear_model(), &on)
            .render_to(&dir.join("on.png"))
            .expect("render");
        let off_path = PlotData::build(&ear_model(), &full_inputs_ref(&ir, &mag))
            .render_to(&dir.join("off.png"))
            .expect("render");
        let (w, _, px_on) = decode_png(&on_path);
        let (_, _, px_off) = decode_png(&off_path);

        // Panel 2 only: panel 1 draws raw L and R in the same blue, so counting the figure would
        // say nothing about the trace this test is about.
        let (x0, y0, x1, y1) = PlotData::panel_rects()[1];
        let (x0, y0, x1, y1) = (
            x0.max(4) as usize,
            y0.max(4) as usize,
            x1 as usize,
            y1 as usize,
        );
        let blue = |px: &[u8]| -> usize {
            (y0 + 3..y1 - 3)
                .flat_map(|y| (x0 + 3..x1 - 3).map(move |x| (x, y)))
                .filter(|&(x, y)| close(at(px, w, x, y), RAW_LR, 24))
                .count()
        };
        let (n_on, n_off) = (blue(&px_on), blue(&px_off));
        assert!(
            n_on > 400,
            "the ears trace drew only {n_on} px in the error panel"
        );
        assert_eq!(
            n_off, 0,
            "the default figure carries the ears colour in the error panel ({n_off} px)"
        );
    }

    #[test]
    fn rendered_panels_tick_their_axis_limits() {
        // The reported defect dropped the tick *and* its label wherever a fixed tick sat
        // exactly on a limit, so the panels lost 20 Hz, 20 kHz, -1/4 ms and (with the
        // autoscaled phase window) everything except 0 deg. This checks the marks are
        // really on the canvas, at the coordinates the panels claim.
        let dir = scratch("edgeticks");
        let (ir, mag) = full_inputs();
        let d = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        let path = d.render_to(&dir.join("figure.png")).expect("render");
        let (w, _, px) = decode_png(&path);
        let rects = PlotData::panel_rects();

        // Frequency panels: the 20 Hz and 20 kHz marks hang below the bottom frame.
        for panel in [0usize, 1, 2] {
            let (x0, _, x1, y1) = rects[panel];
            for (f, want) in [(F_MIN, "20 Hz"), (F_MAX, "20 kHz")] {
                let t = (f / F_MIN).ln() / (F_MAX / F_MIN).ln();
                let col = (f64::from(x0) + f64::from(x1 - x0) * t).round() as usize;
                assert!(
                    tick_ink(&px, w, col, y1 as usize + 2),
                    "panel {} has no tick at {want}",
                    panel + 1
                );
            }
        }
        // The phase panel needs its -90 and +90 marks, which only exist if the window is
        // the reference's -90..90 rather than a window autoscaled to +/-35 deg.
        let (x0, y0, _, y1) = rects[2];
        for v in [-90.0f64, 90.0] {
            let frac = (v - PHASE_YLIM.0) / (PHASE_YLIM.1 - PHASE_YLIM.0);
            let row = (f64::from(y1) - f64::from(y1 - y0) * frac).round() as usize;
            assert!(
                tick_ink(&px, w, x0 as usize - 2, row),
                "panel 3 has no tick at {v} deg (row {row})"
            );
        }
        // The impulse panel: -1 ms and 4 ms, on a linear axis.
        let (x0, _, x1, y1) = rects[3];
        for ms in [IR_XMIN, IR_XMAX] {
            let t = (ms - IR_XMIN) / (IR_XMAX - IR_XMIN);
            let col = (f64::from(x0) + f64::from(x1 - x0) * t).round() as usize;
            assert!(
                tick_ink(&px, w, col, y1 as usize + 2),
                "panel 4 has no tick at {ms} ms"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_configured_axis_window_is_used_verbatim() {
        let mut m = model();
        m.plot_meta =
            serde_json::from_str(r#"{"ax2_ylim": [-3.0, 3.0], "ax1_ylim": [-4.0, 4.0]}"#).unwrap();
        let (ir, mag) = full_inputs();
        let d = PlotData::build(&m, &full_inputs_ref(&ir, &mag));
        // The fixture's unsmoothed error reaches +5 dBr at 20 kHz, i.e. past the
        // configured ceiling: widening the window (as the reference never does) would
        // turn this into 6.0 and shift every arrowhead in the panel.
        let err_max = d
            .harm_plot
            .iter()
            .zip(&d.raw_avg)
            .map(|(&h, &r)| h - r)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(err_max > 3.0, "fixture must exceed the window: {err_max}");
        assert_eq!(d.gain_window(), (-3.0, 3.0));
        assert_eq!(d.level_window(), (-4.0, 4.0));

        // Without a configured window the fallback pads around the correction, so the same
        // model with 20 dB more correction gets a visibly taller window. Only the IR
        // changed between the two, so the difference is entirely the padding rule.
        let plain = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        assert!(!plain.ax2_ylim_fixed);
        let mut louder = ir.clone();
        louder.iter_mut().for_each(|v| *v *= 10.0);
        let loud = PlotData::build(&model(), &full_inputs_ref(&louder, &mag));
        assert!(
            loud.gain_window().1 > plain.gain_window().1,
            "window must grow with the boost: {:?} vs {:?}",
            loud.gain_window(),
            plain.gain_window()
        );
        assert_eq!(
            d.gain_window(),
            (-3.0, 3.0),
            "a configured window stays put no matter how loud the curves get"
        );
        assert!(max_of(&loud.wav_eq) > max_of(&plain.wav_eq) + 19.0);
    }

    #[test]
    fn callout_block_grows_upwards_from_its_anchor() {
        // matplotlib anchors an annotate() block at xytext with va="baseline", so a
        // two-line callout occupies [xytext.y, xytext.y + 2*dy], not the band below it.
        let dy = 0.9585;
        let (centres, tail) = callout_layout(2, -9.2, -13.0, dy);
        assert_eq!(centres.len(), 2);
        assert!(
            centres[0] > centres[1],
            "line 0 must sit above line 1: {centres:?}"
        );
        assert!(
            (centres[1] - (-9.2 + 0.4 * dy)).abs() < 1e-9,
            "bottom row must sit on the anchor: {centres:?}"
        );
        assert!(
            (centres[0] - centres[1] - dy).abs() < 1e-9,
            "rows must be one pitch apart: {centres:?}"
        );

        // An empty annotation is not a zero-height line of text: the reference calls
        // `annotate("")`, which draws the arrow and nothing else. Counting that as one row
        // would lift the shaft off its anchor by a line pitch.
        let (no_rows, tail_down) = callout_layout(0, -9.2, -13.0, dy);
        let (_, tail_up) = callout_layout(0, -9.2, -6.0, dy);
        assert!(no_rows.is_empty(), "no text, no rows");
        assert!(
            (tail_down - (-9.2)).abs() < 1e-12 && (tail_up - (-9.2)).abs() < 1e-12,
            "an empty block starts and ends on its anchor: {tail_down} / {tail_up}"
        );
        assert!(
            (centres[1] - centres[0]).abs() > 1e-6,
            "rows may never share a y (that is the line-on-line collision)"
        );
        assert!(
            (tail - (-9.2)).abs() < 1e-9,
            "pointing down, the shaft leaves the bottom of the block: {tail}"
        );

        // Pointing up, it leaves the top, which is where the text-free secondary arrows
        // of the shipped models are anchored.
        let (_, tail_up) = callout_layout(2, -9.2, -4.0, dy);
        assert!((tail_up - (-9.2 + 2.0 * dy)).abs() < 1e-9, "{tail_up}");

        // A single-row callout is the same rule with n = 1.
        let (one, tail_one) = callout_layout(1, 4.5, 0.0, dy);
        assert_eq!(one.len(), 1);
        assert!((tail_one - 4.5).abs() < 1e-9);
    }

    #[test]
    fn polylines_are_clipped_to_their_panel() {
        let w = Window {
            xr: (20.0, 20000.0),
            yr: (-10.0, 10.0),
            log_x: true,
        };
        let xs: Vec<f64> = fft::geomspace(1.0, 1e6, 60);
        let ys: Vec<f64> = (0..60).map(|i| -60.0 + i as f64 * 2.0).collect();
        let runs = clip_polyline(&xs, &ys, w);
        assert!(!runs.is_empty(), "a crossing trace must still be drawn");
        for (tx, ty) in &runs {
            assert!(tx.len() >= 2);
            for (&x, &y) in tx.iter().zip(ty) {
                assert!(
                    (20.0..=20000.0).contains(&x) && (-10.0..=10.0).contains(&y),
                    "unclipped point ({x}, {y}) escaped the panel"
                );
            }
        }

        // A trace that leaves the window and comes back must split into two runs rather
        // than bridge the gap along the border. Touching the edge and turning back is not
        // a gap, so that case stays one run.
        let split = clip_polyline(
            &[100.0, 1000.0, 2000.0, 10000.0],
            &[0.0, 100.0, 100.0, 0.0],
            w,
        );
        assert_eq!(split.len(), 2, "expected two visible runs");
        let touch = clip_polyline(&[100.0, 1000.0, 10000.0], &[0.0, 100.0, 0.0], w);
        assert_eq!(touch.len(), 1, "a turn-around at the border stays one run");

        // Non-positive x has no place on a log axis: it breaks the run instead of
        // producing a -inf coordinate.
        let bad = clip_polyline(&[-5.0, 100.0, 1000.0], &[0.0, 1.0, 2.0], w);
        assert_eq!(bad.len(), 1);
        assert!(bad[0].0.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn the_impulse_trace_stops_at_the_panel_edge() {
        // mp_t_ms runs 250 samples past 0 ms (5.19 ms at 48 kHz) while the panel stops at
        // 4.0 ms; unclipped, that drew straight over the margin and off the canvas.
        let (ir, mag) = full_inputs();
        let d = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        let w = Window {
            xr: (IR_XMIN, IR_XMAX),
            yr: (IR_YMIN, IR_YMAX),
            log_x: false,
        };
        assert!(
            d.mp_t_ms.iter().any(|&t| t > IR_XMAX),
            "fixture must overshoot to test the clip"
        );
        for (tx, ty) in clip_polyline(&d.mp_t_ms, &d.mp_ir, w) {
            for (&t, &v) in tx.iter().zip(&ty) {
                assert!(
                    (IR_XMIN..=IR_XMAX).contains(&t) && (IR_YMIN..=IR_YMAX).contains(&v),
                    "impulse trace escaped panel 4 at ({t}, {v})"
                );
            }
        }
    }

    #[test]
    fn the_bypass_callout_respects_the_14khz_cap() {
        // hypreq.py: min(14000, max(f*1.18, f+500)). Applying .min before .max put the
        // arrow past 14 kHz for a model whose bypass starts at 13.5 kHz or higher.
        for hf in [9000.0, 13000.0, 13500.0, 15000.0, 19000.0] {
            let ann = build_auto_annotations(&[20.0, 1000.0, 20000.0], &[0.0; 3], hf);
            assert!(
                ann[4].xy_x <= 14000.0,
                "hf bypass {hf} Hz put the callout at {} Hz",
                ann[4].xy_x
            );
            if hf < 14000.0 {
                assert!(
                    ann[4].xy_x > hf,
                    "the callout must sit right of the bypass edge"
                );
            }
        }
        // Nothing to plot still yields the bypass note instead of indexing an empty curve.
        let ann = build_auto_annotations(&[], &[], 9000.0);
        assert_eq!(ann.len(), 5);
    }

    #[test]
    fn geq_knots_must_be_positive_and_finite() {
        // The overlay is interpolated in log frequency, so a 0 Hz knot would put -inf in
        // the grid and poison everything below it with NaN.
        let ctrl = [(100.0, 0.0), (1000.0, 3.0), (10000.0, 1.0)];
        let out = autoeq_from_geq("0 5 0.7 0; 1000 -2 0.7 0; 8000 4 0.7 0", Some(0.0), &ctrl);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite overlay");
        assert_eq!(out.len(), 1200);
    }

    #[test]
    fn rendered_impulse_panel_draws_the_zero_reference_line() {
        // hypreq.py:2497 draws ax4.axhline(0.0, ...). Passing zero_line: false deleted it.
        let dir = scratch("zeroline");
        let (ir, mag) = full_inputs();
        let d = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        let path = d.render_to(&dir.join("figure.png")).expect("render");
        let (w, _, px) = decode_png(&path);
        let (x0, y0, x1, y1) = PlotData::panel_rects()[3];
        let frac = (0.0 - IR_YMIN) / (IR_YMAX - IR_YMIN);
        let row = (f64::from(y1) - f64::from(y1 - y0) * frac).round() as usize;
        let hits = ((x0 + 4) as usize..(x1 - 4) as usize)
            .filter(|&x| close(at(&px, w, x, row), ZERO_LINE, 12))
            .count();
        let span = (x1 - x0 - 8).max(1) as usize;
        assert!(
            hits * 3 > span,
            "panel 4 has no 0.0 reference line: {hits}/{span} pixels at row {row}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn substitution_expands_every_reference_field() {
        let ctx = FmtCtx {
            notch_mode: "zero",
            pre_rt: -6.5,
            unwin_rt: -152.63,
            boost_rt: 4.25,
            notch_db: 1.4999,
        };
        assert_eq!(substitute("Mode {notch_mode}", &ctx), "Mode zero");
        assert_eq!(substitute("Preamp {pre_rt:.2f}", &ctx), "Preamp -6.50");
        assert_eq!(
            substitute("({notch_db:+.2f}dB in '{notch_mode}' mode)", &ctx),
            "(+1.50dB in 'zero' mode)"
        );
        assert_eq!(substitute("Floor {unwin_rt:.1f}", &ctx), "Floor -152.6");
        assert_eq!(substitute("Peak {boost_rt:.2f}", &ctx), "Peak 4.25");
        assert_eq!(
            substitute("bare {notch_db} and {pre_rt}", &ctx),
            "bare 1.4999 and -6.5"
        );
        // Python escapes a literal brace by doubling it, and an unknown field name raises;
        // here it stays visible in the figure instead of being silently dropped.
        assert_eq!(substitute("{{literal}}", &ctx), "{literal}");
        assert_eq!(
            substitute("unknown {nope:.2f} stays", &ctx),
            "unknown {nope:.2f} stays"
        );
        assert_eq!(
            substitute("unterminated {brace", &ctx),
            "unterminated {brace"
        );
    }

    #[test]
    fn format_number_matches_pythons_format() {
        assert_eq!(format_number(-6.5, None), "-6.5");
        assert_eq!(
            format_number(-6.5, Some(":.2f".trim_start_matches(':'))),
            "-6.50"
        );
        assert_eq!(format_number(4.25, Some("+.2f")), "+4.25");
        assert_eq!(format_number(-4.25, Some("+.2f")), "-4.25");
        assert_eq!(format_number(-152.63, Some(".1f")), "-152.6");
    }

    #[test]
    fn metadata_annotations_substitute_their_template_fields() {
        let mut m = model();
        m.plot_meta = serde_json::from_str(
            r#"{"annotations":[{"text":"4. Concha Zero\n({notch_db:+.2f}dB in '{notch_mode}' mode)","xy":[3703.0,"notch_db"],"xytext":[1500.0,7.3],"color":"green"}]}"#,
        )
        .unwrap();
        let (ir, mag) = inputs_64();
        let d = PlotData::build(&m, &inputs(&ir, &mag));
        assert_eq!(d.annotations.len(), 1);
        assert!(
            !d.annotations[0].text.contains('{'),
            "unsubstituted template survived: {:?}",
            d.annotations[0].text
        );
        assert!(
            d.annotations[0].text.contains("in 'safe' mode"),
            "{:?}",
            d.annotations[0].text
        );
    }

    #[test]
    fn minimum_phase_time_axis_has_no_duplicated_zero() {
        let (ir, mag) = inputs_64();
        let d = PlotData::build(&model(), &inputs(&ir, &mag));
        let lead = 48;
        // The lead-in ends one sample before the transient, then the IR itself starts at
        // 0 ms: 0 ms must appear exactly once or the trace doubles back on itself.
        assert!(
            (d.mp_t_ms[lead - 1] + 1e3 / 48000.0).abs() < 1e-12,
            "lead-in ends at {} ms",
            d.mp_t_ms[lead - 1]
        );
        assert_eq!(d.mp_t_ms[lead], 0.0);
        assert_eq!(d.mp_t_ms.iter().filter(|t| **t == 0.0).count(), 1);
        assert_eq!(d.mp_t_ms.len(), d.mp_ir.len());
        assert!(d.mp_ir[..lead].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn linear_phase_ir_peaks_at_zero_ms_inside_the_window() {
        // A flat magnitude inverts to a single sample at index 0, which the centring moves
        // to the middle of the array, i.e. exactly 0 ms on the panel's time axis.
        let (ir, _) = full_inputs();
        let mag: Vec<f64> = vec![1.0; 2049];
        let d = PlotData::build(&model(), &full_inputs_ref(&ir, &mag));
        assert!(d.lin_t_ms.len() > 100, "window too small to test");
        let peak = d
            .lin_ir
            .iter()
            .enumerate()
            .fold((0usize, 0.0f64), |acc, (i, &v)| {
                if v.abs() > acc.1 {
                    (i, v.abs())
                } else {
                    acc
                }
            })
            .0;
        // `linear_phase_ir` already centres the peak; a second shift would push it to an
        // edge of the panel window instead of sitting it on the 0 ms guide.
        assert_eq!(d.lin_t_ms[peak], 0.0, "linear-phase peak must sit at 0 ms");
        assert!(
            peak > 0 && peak + 1 < d.lin_ir.len(),
            "peak pinned to a window edge"
        );
    }

    /// The small fixture used by the data-level tests.
    fn inputs_64() -> (Vec<f64>, Vec<f64>) {
        let ir: Vec<f64> = (0..64).map(|i| (f64::from(i) * 0.2).sin() * 0.5).collect();
        let mag: Vec<f64> = (0..33).map(|_| 1.0).collect();
        (ir, mag)
    }
}
