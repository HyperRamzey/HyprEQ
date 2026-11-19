//! WAV I/O for HyprEQ: read any common PCM/IEEE-float file, write the file the
//! pipeline produced.
//!
//! Two layers meet here. The *container* side is [`crate::riff`]: [`WavWriter`]
//! lays out the classic 44-byte header for mono/stereo (byte-for-byte what
//! `hypreq.py` writes, including the two sizes patched on `finish()`) and switches to
//! `WAVE_FORMAT_EXTENSIBLE` with a real channel mask above two channels, which the
//! Python never does. The *chunk walk and sample codec* below follow
//! `hypreq.py::read_wav` exactly, because the numbers downstream depend on the
//! details:
//!
//! * the **last** `data` chunk wins, the way Python's loop overwrites `data_bytes`
//!   (`crate::riff::read_info` stops at the first one, so it is not used here — it
//!   stays available for callers that want first-chunk semantics);
//! * `WAVE_FORMAT_EXTENSIBLE` takes its real format from the SubFormat GUID
//!   (`payload[24]`), so a float extensible file is still read as float;
//! * normalisation dividers are 128.0 / 32768.0 / 8388608.0 / 2147483648.0, and
//!   the 8-bit track is *unsigned* with a 128 offset;
//! * a `data` chunk whose length is not a whole number of samples is refused the
//!   way `np.frombuffer` refuses it, and a trailing partial frame is dropped;
//! * 16/24-bit output is TPDF-dithered with the fixed numpy stream in
//!   [`crate::dither`], rounded half-to-even, then clamped — in that order, which
//!   is what `np.clip(np.rint(...), vmin, vmax)` does.
//!
//! Everything stays `f64`; 32-bit writes round once, at the container.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::dither::Pcg64;
use crate::error::{Error, Result};
use crate::riff::WavWriter;

const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Largest `fmt ` body worth allocating; a real one is 16 or 40 bytes.
const MAX_FMT_CHUNK: u64 = 1024;

/// Decoded audio: sample rate plus one `f64` buffer per channel, in [-1, 1].
#[derive(Clone, Debug)]
pub struct WavAudio {
    pub fs: u32,
    pub planar: Vec<Vec<f64>>,
    /// Human-readable description of the container as read, e.g.
    /// `pcm_s24le | 48,000 Hz | 2ch`.
    pub fmt_desc: String,
}

impl WavAudio {
    pub fn channels(&self) -> usize {
        self.planar.len()
    }

    /// Shortest channel length; the pipeline treats the buffer as rectangular.
    pub fn frames(&self) -> usize {
        self.planar.iter().map(|c| c.len()).min().unwrap_or(0)
    }
}

/// `f"{n:,}"` — a thousands-separated integer, as the Python prints rates.
pub fn comma_sep(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let lead = s.len() % 3;
    // bytes(), not chars(): `n.to_string()` is ASCII digits, so the index is a
    // byte offset and the grouping is exact.
    for (i, b) in s.as_bytes().iter().enumerate() {
        // Groups are cut from the left at `lead` and then every three digits:
        // 48000 → lead 2 → "48" + ",000"; 8000 → lead 1 → "8" + ",000".
        if i > 0 && i >= lead && (i - lead).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// What the `fmt ` chunk said, after the extensible SubFormat is applied.
struct Format {
    /// Effective tag: the SubFormat's first two bytes for extensible files.
    eff_tag: u16,
    channels: u16,
    fs: u32,
    bits: u16,
}

/// Read a RIFF/WAVE file into normalised `f64` planar channels in [-1, 1].
///
/// Supports PCM 8/16/24/32, IEEE float 32/64, and `WAVE_FORMAT_EXTENSIBLE`.
pub fn read_wav(path: &Path) -> Result<WavAudio> {
    let name = path.display().to_string();
    let mut f = File::open(path)?;
    let file_len = f.metadata()?.len();

    let mut riff = [0u8; 12];
    if f.read_exact(&mut riff).is_err() || &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err(Error::msg(format!(
            "'{name}' is not a valid RIFF/WAVE file"
        )));
    }

    let mut fmt: Option<Format> = None;
    let mut data: Option<(u64, u64)> = None;

    loop {
        let mut hdr = [0u8; 8];
        if f.read_exact(&mut hdr).is_err() {
            break;
        }
        let id: [u8; 4] = hdr[0..4].try_into().expect("slice of length 4");
        let size = u32::from_le_bytes(hdr[4..8].try_into().expect("slice of length 4")) as u64;
        let pos = f.stream_position()?;
        // An ASCII chunk id, for the truncation message the Python prints.
        let id_name: String = id
            .iter()
            .map(|&b| {
                if b.is_ascii_graphic() || b == b' ' {
                    b as char
                } else {
                    '\u{fffd}'
                }
            })
            .collect();

        // The reference reads *every* chunk's payload and raises when the read comes up
        // short, so a truncated LIST, bext or junk is reported exactly like a truncated
        // data chunk. Skipping unknown chunks instead would let a file that the reference
        // rejects load here — and `size > file_len - pos` cannot overflow.
        if size > file_len.saturating_sub(pos) {
            return Err(Error::msg(format!(
                "Truncated '{id_name}' chunk in '{name}'"
            )));
        }

        if id == *b"fmt " {
            if size > MAX_FMT_CHUNK {
                return Err(Error::msg(format!(
                    "absurd WAV fmt chunk size ({size}) in '{name}'"
                )));
            }
            let mut b = vec![0u8; size as usize];
            f.read_exact(&mut b)?;
            if size < 16 {
                return Err(Error::msg(format!(
                    "Invalid fmt chunk size ({size}) in '{name}'"
                )));
            }
            let tag = u16::from_le_bytes([b[0], b[1]]);
            let channels = u16::from_le_bytes([b[2], b[3]]);
            let fs = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let bits = u16::from_le_bytes([b[14], b[15]]);
            // Extensible: validate cbSize >= 22, the full SubFormat GUID, and wValidBitsPerSample.
            const SUBFORMAT_GUID_TAIL: [u8; 14] = [
                0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
            ];
            let eff_tag = if tag == WAVE_FORMAT_EXTENSIBLE && size >= 40 {
                let cb_size = u16::from_le_bytes([b[16], b[17]]);
                let valid_bits = u16::from_le_bytes([b[18], b[19]]);
                if cb_size >= 22
                    && b[26..40] == SUBFORMAT_GUID_TAIL
                    && (valid_bits == 0 || valid_bits <= bits)
                {
                    u16::from_le_bytes([b[24], b[25]])
                } else {
                    tag
                }
            } else {
                tag
            };
            fmt = Some(Format {
                eff_tag,
                channels,
                fs,
                bits,
            });
        } else if id == *b"data" {
            // The last data chunk wins, as in Python's read loop.
            data = Some((pos, size));
        }
        // Chunks are word-aligned: an odd size carries a pad byte.
        f.seek(SeekFrom::Start(pos + size + (size & 1)))?;
    }

    let (Some(fmt), Some((data_off, data_len))) = (fmt, data) else {
        return Err(Error::msg(format!("Missing fmt or data chunk in '{name}'")));
    };
    if fmt.channels == 0 || fmt.fs == 0 {
        return Err(Error::msg(format!(
            "Invalid WAV channel/rate metadata in '{name}'"
        )));
    }
    let channels = fmt.channels as usize;

    let mut payload = vec![0u8; data_len as usize];
    f.seek(SeekFrom::Start(data_off))?;
    f.read_exact(&mut payload)?;

    let (inter, desc_kind) = decode(&payload, &fmt, &name)?;
    let fmt_desc = format!(
        "{desc_kind} | {} Hz | {channels}ch",
        comma_sep(u64::from(fmt.fs))
    );

    let frames = inter.len() / channels;
    if frames == 0 {
        return Ok(WavAudio {
            fs: fmt.fs,
            planar: vec![Vec::new(); channels],
            fmt_desc,
        });
    }
    let mut planar = vec![Vec::with_capacity(frames); channels];
    for (i, v) in inter[..frames * channels].iter().enumerate() {
        planar[i % channels].push(*v);
    }
    Ok(WavAudio {
        fs: fmt.fs,
        planar,
        fmt_desc,
    })
}

/// Decode one `data` payload into interleaved `f64` in [-1, 1].
///
/// Returns the values and the codec name used in `fmt_desc`.
fn decode(payload: &[u8], fmt: &Format, name: &str) -> Result<(Vec<f64>, &'static str)> {
    let item = match fmt.eff_tag {
        WAVE_FORMAT_IEEE_FLOAT => match fmt.bits {
            32 | 64 => fmt.bits as usize / 8,
            b => {
                return Err(Error::msg(format!(
                    "Unsupported IEEE float bit depth {b} in '{name}'"
                )))
            }
        },
        WAVE_FORMAT_PCM => match fmt.bits {
            8 | 16 | 24 | 32 => fmt.bits as usize / 8,
            b => {
                return Err(Error::msg(format!(
                    "Unsupported PCM bit depth {b} in '{name}'"
                )))
            }
        },
        tag => {
            return Err(Error::msg(format!(
                "Unsupported WAV format tag 0x{tag:04x} in '{name}'"
            )))
        }
    };
    // np.frombuffer refuses a buffer that is not a whole number of items, so a
    // ragged payload is an error rather than a silently dropped tail sample.
    if !payload.len().is_multiple_of(item) {
        return Err(Error::msg(format!(
            "Buffer size must be a multiple of {item} bytes \
             (data chunk of {} bytes in '{name}')",
            payload.len()
        )));
    }

    let mut out = Vec::with_capacity(payload.len() / item);
    match (fmt.eff_tag, fmt.bits) {
        (WAVE_FORMAT_IEEE_FLOAT, 32) => {
            for w in payload.chunks_exact(4) {
                out.push(f32::from_le_bytes([w[0], w[1], w[2], w[3]]) as f64);
            }
            Ok((out, "pcm_f32le"))
        }
        (WAVE_FORMAT_IEEE_FLOAT, 64) => {
            for w in payload.chunks_exact(8) {
                out.push(f64::from_le_bytes(w.try_into().expect("chunk of length 8")));
            }
            Ok((out, "pcm_f64le"))
        }
        (WAVE_FORMAT_PCM, 8) => {
            // Unsigned with a 128 offset — 0 is -1.0, 255 is just under +1.0.
            for &b in payload {
                out.push((f64::from(b) - 128.0) / 128.0);
            }
            Ok((out, "pcm_u8"))
        }
        (WAVE_FORMAT_PCM, 16) => {
            for w in payload.chunks_exact(2) {
                out.push(i16::from_le_bytes([w[0], w[1]]) as f64 / 32768.0);
            }
            Ok((out, "pcm_s16le"))
        }
        (WAVE_FORMAT_PCM, 24) => {
            for w in payload.chunks_exact(3) {
                // Assemble unsigned, then shift into the top of an i32 and back
                // with an arithmetic shift to sign-extend — the same result as
                // Python's `np.where(vals >= 0x800000, vals - 0x1000000, vals)`.
                // Without the second shift the value is 256× too large and the
                // decoded audio leaves the [-1, 1) range entirely.
                let v = u32::from(w[0]) | (u32::from(w[1]) << 8) | (u32::from(w[2]) << 16);
                out.push((((v << 8) as i32) >> 8) as f64 / 8_388_608.0);
            }
            Ok((out, "pcm_s24le"))
        }
        (WAVE_FORMAT_PCM, 32) => {
            for w in payload.chunks_exact(4) {
                out.push(i32::from_le_bytes([w[0], w[1], w[2], w[3]]) as f64 / 2_147_483_648.0);
            }
            Ok((out, "pcm_s32le"))
        }
        _ => unreachable!("decode() validates the tag and depth before this point"),
    }
}

/// The 4 GiB ceiling, checked before a byte is written.
///
/// Two bounds, because a WAV carries two sizes. The first is the reference's own: it
/// measures `36 + data_size` against `u32::MAX`, and that wording and that threshold are
/// kept so a file this size fails the way the reference says it fails. The second is the
/// file's: the RIFF field holds `hdr_len - 8 + data_size + (data_size & 1)`; the data
/// `ckSize` excludes that even-alignment pad byte, but the enclosing RIFF size includes
/// it. Under an 80-byte
/// `WAVE_FORMAT_EXTENSIBLE` header the payload that still fits is 36 bytes smaller than
/// the first bound allows. Without it a multichannel write slips past the check, `as u32`
/// wraps the RIFF size down to a few dozen bytes, and the result is a file that plays and
/// is silently truncated at the wrap. The reference shares that arithmetic, so this is
/// the one place the port is deliberately stricter than it (see `docs/SETUP.md`).
fn riff_guard(name: &str, data_size: usize, hdr_len: usize) -> Result<()> {
    let riff = data_size
        .checked_add(36)
        .ok_or_else(|| Error::msg(format!("WAV output '{name}' exceeds 4 GiB RIFF limit")))?;
    if u64::try_from(riff).expect("usize fits in u64") > u64::from(u32::MAX) {
        return Err(Error::msg(format!(
            "WAV output '{name}' exceeds 4 GiB RIFF limit ({riff} bytes)"
        )));
    }
    if let Err(e) = crate::riff::riff_sizes(data_size as u64, hdr_len as u64) {
        return Err(Error::msg(format!(
            "WAV output '{name}' cannot be written: {e}"
        )));
    }
    Ok(())
}

/// The header the writer will emit for these settings, without pretending to know its
/// length: asked of the builder that makes it, so the two size fields and the header actually
/// written cannot drift apart.
fn header_len_for(channels: usize, bits: u32, float: bool) -> Result<usize> {
    // Refused rather than clamped. Clamping a count the container cannot carry kept the guard on
    // the strict (extensible) side only while it decided, and the header that came out described a
    // frame stride the payload does not have.
    let channels = u16::try_from(channels)
        .map_err(|_| Error::msg(format!("cannot write {channels} channels in one WAV file")))?;
    crate::riff::header_len(channels, bits, float)
}

/// Open the writer for a mono/stereo file: [`WavWriter`] emits the plain
/// 44-byte header for these, and only reaches for `WAVE_FORMAT_EXTENSIBLE` above
/// two channels.
fn writer(path: &Path, fs: u32, channels: usize, bits: u32, float: bool) -> Result<WavWriter> {
    let channels = u16::try_from(channels)
        .map_err(|_| Error::msg(format!("cannot write {channels} channels in one WAV file")))?;
    WavWriter::create(path, fs, channels, bits, float, None)
}

/// Write planar `f64` as IEEE float (`pcm_f32le` at 32 bits, `pcm_f64le` at 64).
///
/// Values are **not** clipped or cleansed — a straight cast, as `astype('<f4')`
/// is — and the payload is interleaved frame-major.
pub fn write_wav_ieee_float(path: &Path, planar: &[Vec<f64>], fs: u32, bits: u32) -> Result<usize> {
    if fs == 0 {
        return Err(Error::msg("sample rate must be positive (got 0)"));
    }
    if planar.is_empty() {
        return Err(Error::msg("Cannot write empty channel list"));
    }
    if bits != 32 && bits != 64 {
        return Err(Error::msg(format!(
            "IEEE float output needs 32 or 64 bits, got {bits}"
        )));
    }
    let name = path.display().to_string();
    let channels = planar.len();
    let frames = planar.iter().map(|c| c.len()).min().unwrap_or(0);
    let bps = (bits / 8) as usize;
    let data_size = frames
        .checked_mul(channels)
        .and_then(|fc| fc.checked_mul(bps))
        .ok_or_else(|| Error::msg(format!("WAV output '{name}' exceeds addressable size")))?;
    riff_guard(&name, data_size, header_len_for(channels, bits, true)?)?;

    let mut w = writer(path, fs, channels, bits, true)?;
    // Written in frame-major blocks so the interleave is explicit and the
    // buffer stays bounded for long files.
    let mut buf: Vec<u8> = Vec::with_capacity(8 * channels * 1024);
    for f0 in (0..frames).step_by(1024) {
        let hi = (f0 + 1024).min(frames);
        buf.clear();
        // One frame at a time, walking the channels' windows in step.
        let windows: Vec<&[f64]> = planar.iter().take(channels).map(|c| &c[f0..hi]).collect();
        for fr in 0..hi - f0 {
            for &v in windows.iter().map(|w| &w[fr]) {
                match bits {
                    32 => buf.extend_from_slice(&(v as f32).to_le_bytes()),
                    _ => buf.extend_from_slice(&v.to_le_bytes()),
                }
            }
        }
        w.write(&buf)?;
    }
    let written = w.finish()?;
    debug_assert_eq!(written as usize, data_size);
    Ok(frames)
}

/// Write planar `f64` as TPDF-dithered 16/24-bit PCM, or delegate to
/// [`write_wav_ieee_float`] for 32/64-bit float.
///
/// The PCM path mirrors `hypreq.py::write_wav_pcm_or_float`:
/// replace non-finite samples (NaN → 0, ±Inf → ±1), add the fixed TPDF stream,
/// round half-to-even, then clamp to the container's integer range.
#[allow(clippy::needless_range_loop)]
pub fn write_wav_pcm_or_float(
    path: &Path,
    planar: &[Vec<f64>],
    fs: u32,
    bits: u32,
) -> Result<usize> {
    if fs == 0 {
        return Err(Error::msg("sample rate must be positive (got 0)"));
    }
    if planar.is_empty() {
        return Err(Error::msg("Cannot write empty channel list"));
    }
    if bits == 32 || bits == 64 {
        return write_wav_ieee_float(path, planar, fs, bits);
    }
    if bits != 16 && bits != 24 {
        return Err(Error::msg(format!(
            "Unsupported output WAV bit depth: {bits} (choose 16, 24, 32, or 64)"
        )));
    }

    let name = path.display().to_string();
    let channels = planar.len();
    let frames = planar.iter().map(|c| c.len()).min().unwrap_or(0);
    let bps = (bits / 8) as usize;
    let data_size = frames
        .checked_mul(channels)
        .and_then(|fc| fc.checked_mul(bps))
        .ok_or_else(|| Error::msg(format!("WAV output '{name}' exceeds addressable size")))?;
    riff_guard(&name, data_size, header_len_for(channels, bits, false)?)?;

    let (scale, vmin, vmax) = if bits == 16 {
        (32768.0f64, -32768.0f64, 32767.0f64)
    } else {
        (8_388_608.0f64, -8_388_608.0f64, 8_388_607.0f64)
    };
    // Stream TPDF dither without upfront buffer allocation:
    // generator 1 starts at 0, generator 2 starts at frames * channels.
    let mut rng1 = Pcg64::hypreq_dither();
    let mut rng2 = Pcg64::hypreq_dither();
    rng2.advance((frames * channels) as u128);

    let mut w = writer(path, fs, channels, bits, false)?;
    let mut buf: Vec<u8> = Vec::with_capacity(bps * channels * 1024);
    for f0 in (0..frames).step_by(1024) {
        let hi = (f0 + 1024).min(frames);
        buf.clear();
        for fr in f0..hi {
            for c in 0..channels {
                let v = cleanse(planar[c][fr]);
                let dither = rng1.next_uniform_half() + rng2.next_uniform_half();
                let q = (v * scale + dither).round_ties_even().clamp(vmin, vmax);
                let q = q as i32;
                match bits {
                    16 => buf.extend_from_slice(&(q.clamp(-32768, 32767) as i16).to_le_bytes()),
                    _ => {
                        let q = q.clamp(-8_388_608, 8_388_607) as u32;
                        buf.extend_from_slice(&[q as u8, (q >> 8) as u8, (q >> 16) as u8]);
                    }
                }
            }
        }
        w.write(&buf)?;
    }
    let written = w.finish()?;
    debug_assert_eq!(written as usize, data_size);
    Ok(frames)
}

/// `np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)`.
#[inline]
fn cleanse(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        1.0
    } else if v == f64::NEG_INFINITY {
        -1.0
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_separators_match_python() {
        assert_eq!(comma_sep(0u64), "0");
        assert_eq!(comma_sep(4), "4");
        assert_eq!(comma_sep(104), "104");
        assert_eq!(comma_sep(100), "100");
        assert_eq!(comma_sep(10_000), "10,000");
        assert_eq!(comma_sep(100_000), "100,000");
        assert_eq!(comma_sep(1_000_000_000), "1,000,000,000");
        assert_eq!(comma_sep(999), "999");
        assert_eq!(comma_sep(1000), "1,000");
        assert_eq!(comma_sep(12345), "12,345");
        assert_eq!(comma_sep(44100), "44,100");
        assert_eq!(comma_sep(48000), "48,000");
        assert_eq!(comma_sep(1000000), "1,000,000");
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hypreq-wav-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    fn raw_wav(path: &Path, tag: u16, ch: u16, fs: u32, bits: u16, payload: &[u8]) {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + payload.len() as u32).to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ch.to_le_bytes());
        b.extend_from_slice(&fs.to_le_bytes());
        let block = ch as u32 * bits as u32 / 8;
        b.extend_from_slice(&(fs * block).to_le_bytes());
        b.extend_from_slice(&(block as u16).to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        b.extend_from_slice(payload);
        std::fs::write(path, b).unwrap();
    }

    #[test]
    fn pcm_codecs_decode_to_the_same_dividers_as_numpy() {
        let p = tmp("u8.wav");
        raw_wav(&p, 1, 1, 8000, 8, &[0, 128, 255]);
        let a = read_wav(&p).unwrap();
        assert_eq!(a.fmt_desc, "pcm_u8 | 8,000 Hz | 1ch");
        assert_eq!(a.planar[0][0], -1.0);
        assert_eq!(a.planar[0][1], 0.0);
        assert!((a.planar[0][2] - (127.0 / 128.0)).abs() < 1e-15);

        // Six bytes of stereo 16-bit is one and a half frames. The reference reader
        // counts whole frames only, so the trailing two bytes are dropped:
        // frame 0 = (0x8000, 0x4000) -> (-1.0, 0.5).
        raw_wav(&p, 1, 2, 44100, 16, &[0, 128, 0x00, 0x40, 0x00, 0xC0]);
        let a = read_wav(&p).unwrap();
        assert_eq!(a.fmt_desc, "pcm_s16le | 44,100 Hz | 2ch");
        assert_eq!(a.planar[0], vec![-1.0]);
        assert_eq!(a.planar[1], vec![0.5]);

        // 24-bit: 0x7FFFFF, 0x800000 (most negative), and -1.
        let p24 = vec![0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0xFF];
        raw_wav(&p, 1, 1, 48000, 24, &p24);
        let a = read_wav(&p).unwrap();
        assert_eq!(a.fmt_desc, "pcm_s24le | 48,000 Hz | 1ch");
        assert_eq!(a.planar[0][0], 8_388_607.0 / 8_388_608.0, "0x7FFFFF");
        assert_eq!(a.planar[0][1], -1.0, "0x800000 is the most negative value");
        assert_eq!(a.planar[0][2], -1.0 / 8_388_608.0, "0xFFFFFF is -1 LSB");

        raw_wav(&p, 1, 1, 48000, 32, &0x4000_0000i32.to_le_bytes());
        let a = read_wav(&p).unwrap();
        assert_eq!(a.fmt_desc, "pcm_s32le | 48,000 Hz | 1ch");
        assert_eq!(a.planar[0][0], 0.5);
    }

    #[test]
    fn float_codecs_and_extensible_subformat() {
        let p = tmp("f32.wav");
        let mut pl = Vec::new();
        pl.extend_from_slice(&0.25f32.to_le_bytes());
        pl.extend_from_slice(&(-0.5f32).to_le_bytes());
        raw_wav(&p, 3, 1, 96000, 32, &pl);
        let a = read_wav(&p).unwrap();
        assert_eq!(a.fmt_desc, "pcm_f32le | 96,000 Hz | 1ch");
        assert_eq!(a.planar[0], vec![0.25, -0.5]);

        let mut pl = Vec::new();
        pl.extend_from_slice(&0.1f64.to_le_bytes());
        raw_wav(&p, 3, 1, 48000, 64, &pl);
        let a = read_wav(&p).unwrap();
        assert_eq!(a.fmt_desc, "pcm_f64le | 48,000 Hz | 1ch");
        assert_eq!(a.planar[0][0], 0.1);

        // WAVE_FORMAT_EXTENSIBLE carrying IEEE float in its SubFormat GUID: the
        // effective tag must come from the GUID, not the 0xFFFE container.
        let mut fmt_body: Vec<u8> = Vec::new();
        fmt_body.extend_from_slice(&0xFFFEu16.to_le_bytes());
        fmt_body.extend_from_slice(&1u16.to_le_bytes()); // channels
        fmt_body.extend_from_slice(&48000u32.to_le_bytes());
        fmt_body.extend_from_slice(&(48000u32 * 4).to_le_bytes());
        fmt_body.extend_from_slice(&4u16.to_le_bytes());
        fmt_body.extend_from_slice(&32u16.to_le_bytes());
        fmt_body.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        fmt_body.extend_from_slice(&32u16.to_le_bytes()); // valid bits
        fmt_body.extend_from_slice(&3u32.to_le_bytes()); // mask: FL|FR
        fmt_body.extend_from_slice(&[
            0x03, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71,
        ]);
        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
        b.extend_from_slice(&fmt_body);
        b.extend_from_slice(b"data");
        let pl = 0.75f32.to_le_bytes();
        b.extend_from_slice(&(pl.len() as u32).to_le_bytes());
        b.extend_from_slice(&pl);
        let len = b.len() as u32;
        b[4..8].copy_from_slice(&(len - 8).to_le_bytes());
        std::fs::write(&p, &b).unwrap();
        let a = read_wav(&p).unwrap();
        assert_eq!(a.fmt_desc, "pcm_f32le | 48,000 Hz | 1ch");
        assert_eq!(a.planar[0][0], 0.75);
    }

    #[test]
    fn last_data_chunk_wins_and_pad_bytes_are_skipped() {
        let p = tmp("two_data.wav");
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        let mut fmt_body: Vec<u8> = Vec::new();
        fmt_body.extend_from_slice(&1u16.to_le_bytes());
        fmt_body.extend_from_slice(&1u16.to_le_bytes());
        fmt_body.extend_from_slice(&8000u32.to_le_bytes());
        fmt_body.extend_from_slice(&16000u32.to_le_bytes());
        fmt_body.extend_from_slice(&2u16.to_le_bytes());
        fmt_body.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
        b.extend_from_slice(&fmt_body);
        // A LIST chunk with an odd size (pad byte must be consumed), then two
        // data chunks; the second is the audio.
        let list = b"INFOadt1!".to_vec(); // an odd chunk size, so a pad byte follows
        b.extend_from_slice(b"LIST");
        b.extend_from_slice(&(list.len() as u32).to_le_bytes());
        b.extend_from_slice(&list);
        b.push(0);
        let first = vec![0x11u8, 0x22];
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(first.len() as u32).to_le_bytes());
        b.extend_from_slice(&first);
        let second: Vec<u8> = (0x4000i16).to_le_bytes().to_vec();
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(second.len() as u32).to_le_bytes());
        b.extend_from_slice(&second);
        let len = b.len() as u32;
        b[4..8].copy_from_slice(&(len - 8).to_le_bytes());
        std::fs::write(&p, &b).unwrap();
        let a = read_wav(&p).unwrap();
        assert_eq!(a.planar[0].len(), 1, "the last data chunk wins");
        assert_eq!(a.planar[0][0], 0.5);
    }

    /// An odd `fmt ` payload carries a pad byte that belongs to the chunk, and every
    /// writer that emits an 18-byte `fmt ` with a non-`EXTENSIBLE` tag has been seen to
    /// word-align it. Skipping only the payload leaves every following header one byte
    /// early, which is not a wrong number but a file that will not parse at all.
    #[test]
    fn an_odd_sized_fmt_chunk_still_leaves_the_rest_of_the_file_aligned() {
        let p = tmp("odd_fmt.wav");
        let mut fmt_body: Vec<u8> = Vec::new();
        fmt_body.extend_from_slice(&1u16.to_le_bytes()); // PCM
        fmt_body.extend_from_slice(&1u16.to_le_bytes()); // one channel
        fmt_body.extend_from_slice(&8000u32.to_le_bytes());
        fmt_body.extend_from_slice(&16000u32.to_le_bytes()); // byte rate
        fmt_body.extend_from_slice(&2u16.to_le_bytes()); // block align
        fmt_body.extend_from_slice(&16u16.to_le_bytes()); // bits
        fmt_body.push(0xAA); // one byte of extension, so the chunk size is odd

        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
        b.extend_from_slice(&fmt_body);
        b.push(0); // the pad byte that word alignment requires
        let payload = 0x4000u16.to_le_bytes(); // one sample, +0.5
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        b.extend_from_slice(&payload);
        let len = b.len() as u32;
        b[4..8].copy_from_slice(&(len - 8).to_le_bytes());
        std::fs::write(&p, &b).unwrap();

        let a = read_wav(&p).unwrap_or_else(|e| panic!("odd fmt payload broke the parse: {e}"));
        assert_eq!(a.fmt_desc, "pcm_s16le | 8,000 Hz | 1ch");
        assert_eq!(
            a.planar[0],
            vec![0.5],
            "the data chunk was found where it is"
        );
    }

    /// The `SubFormat` of a `WAVE_FORMAT_EXTENSIBLE` header is a 16-bit tag in the low
    /// half of its GUID, not a single byte. Reading one byte is the same answer for the
    /// two formats anyone uses (`01 00` PCM, `03 00` float) and a different answer for
    /// every other value — so the case that proves the width is a GUID that is neither.
    #[test]
    fn the_extensible_subformat_is_read_as_a_16_bit_tag() {
        let extensible = |subtype_low: u16| {
            let mut fmt_body: Vec<u8> = Vec::new();
            fmt_body.extend_from_slice(&0xFFFEu16.to_le_bytes()); // WAVE_FORMAT_EXTENSIBLE
            fmt_body.extend_from_slice(&1u16.to_le_bytes()); // channels
            fmt_body.extend_from_slice(&48000u32.to_le_bytes());
            fmt_body.extend_from_slice(&(48000u32 * 4).to_le_bytes());
            fmt_body.extend_from_slice(&4u16.to_le_bytes());
            fmt_body.extend_from_slice(&32u16.to_le_bytes());
            fmt_body.extend_from_slice(&22u16.to_le_bytes()); // cbSize
            fmt_body.extend_from_slice(&32u16.to_le_bytes()); // valid bits
            fmt_body.extend_from_slice(&3u32.to_le_bytes()); // channel mask
            let mut guid = [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
                0x9B, 0x71,
            ];
            guid[..2].copy_from_slice(&subtype_low.to_le_bytes());
            fmt_body.extend_from_slice(&guid);

            let mut b: Vec<u8> = b"RIFF".to_vec();
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(b"WAVE");
            b.extend_from_slice(b"fmt ");
            b.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
            b.extend_from_slice(&fmt_body);
            b.extend_from_slice(b"data");
            let payload = 0.75f32.to_le_bytes();
            b.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            b.extend_from_slice(&payload);
            let len = b.len() as u32;
            b[4..8].copy_from_slice(&(len - 8).to_le_bytes());
            b
        };

        // Control: a real float subtype is float.
        let p = tmp("sub_float.wav");
        std::fs::write(&p, extensible(0x0003)).unwrap();
        assert_eq!(
            read_wav(&p).unwrap().fmt_desc,
            "pcm_f32le | 48,000 Hz | 1ch"
        );

        // 0x0103: the low byte is 3, so a one-byte read would call this IEEE float and
        // decode the payload as f32 in silence. It is not a subtype in the registry, and
        // the refusal has to name the full tag — possible only if both bytes were read.
        let p = tmp("sub_neither.wav");
        std::fs::write(&p, extensible(0x0103)).unwrap();
        let e =
            read_wav(&p).expect_err("an unknown SubFormat must be refused, not decoded as float");
        assert!(
            e.to_string().contains("0x0103"),
            "the message must name the 16-bit tag, got: {e}"
        );
    }

    #[test]
    fn ragged_and_unsupported_payloads_are_refused() {
        let p = tmp("ragged.wav");
        raw_wav(&p, 1, 1, 8000, 16, &[1, 2, 3]);
        let e = read_wav(&p).unwrap_err().to_string();
        assert!(e.contains("multiple of 2 bytes"), "got {e}");

        raw_wav(&p, 1, 1, 8000, 12, &[0, 0]);
        let e = read_wav(&p).unwrap_err().to_string();
        assert!(e.contains("Unsupported PCM bit depth 12"), "got {e}");

        raw_wav(&p, 3, 1, 8000, 16, &[0, 0]);
        let e = read_wav(&p).unwrap_err().to_string();
        assert!(e.contains("Unsupported IEEE float bit depth 16"), "got {e}");

        raw_wav(&p, 0xFF, 1, 8000, 16, &[0, 0]);
        let e = read_wav(&p).unwrap_err().to_string();
        assert!(e.contains("format tag 0x00ff"), "got {e}");

        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&100u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"data");
        b.extend_from_slice(&8u32.to_le_bytes());
        std::fs::write(&p, &b).unwrap();
        let e = read_wav(&p).unwrap_err().to_string();
        assert!(e.contains("Truncated 'data' chunk"), "got {e}");

        std::fs::write(&p, b"OggS not a wav").unwrap();
        let e = read_wav(&p).unwrap_err().to_string();
        assert!(e.contains("not a valid RIFF/WAVE"), "got {e}");
    }

    #[test]
    fn empty_data_chunk_reads_as_zero_frames() {
        let p = tmp("empty.wav");
        raw_wav(&p, 1, 2, 48000, 24, &[]);
        let a = read_wav(&p).unwrap();
        assert_eq!(a.frames(), 0);
        assert_eq!(a.channels(), 2);
        assert_eq!(a.fmt_desc, "pcm_s24le | 48,000 Hz | 2ch");
    }

    /// Byte-for-byte check of the dithered PCM path against the Python, on the
    /// same eight frames and two channels.
    #[test]
    fn pcm_output_matches_python_bytes() {
        let l = [0.0, 0.5, -0.5, 0.999999, -1.0, 1.0, 1e-9, -0.25];
        let r = [0.25, -0.75, 0.125, 0.0, 0.5, -0.5, 1.0, -1e-12];
        let planar = vec![l.to_vec(), r.to_vec()];

        let q16 = [
            1, 8192, 16383, -24576, -16385, 4096, 32767, 0, -32768, 16384, 32767, -16384, -1,
            32767, -8192, 0,
        ];
        let p = tmp("py16.wav");
        let n = write_wav_pcm_or_float(&p, &planar, 48000, 16).unwrap();
        assert_eq!(n, 8);
        let b = std::fs::read(&p).unwrap();
        assert_eq!(b.len(), 44 + 32);
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes([b[4], b[5], b[6], b[7]]), 36 + 32);
        assert_eq!(u16::from_le_bytes([b[20], b[21]]), 1, "PCM tag");
        assert_eq!(u16::from_le_bytes([b[34], b[35]]), 16);
        let mut got = Vec::new();
        for w in b[44..].chunks_exact(2) {
            got.push(i16::from_le_bytes([w[0], w[1]]) as i32);
        }
        assert_eq!(got, q16, "16-bit dithered samples must match numpy exactly");

        let q24 = [
            1, 2097152, 4194303, -6291456, -4194305, 1048576, 8388600, 0, -8388608, 4194304,
            8388607, -4194304, -1, 8388607, -2097152, 0,
        ];
        let p = tmp("py24.wav");
        write_wav_pcm_or_float(&p, &planar, 48000, 24).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(b.len(), 44 + 48);
        assert_eq!(u16::from_le_bytes([b[34], b[35]]), 24);
        let mut got = Vec::new();
        for w in b[44..].chunks_exact(3) {
            let mut v = u32::from(w[0]) | (u32::from(w[1]) << 8) | (u32::from(w[2]) << 16);
            if v >= 0x800_000 {
                v = v.wrapping_sub(0x100_0000);
            }
            got.push(v as i32);
        }
        assert_eq!(got, q24, "24-bit dithered samples must match numpy exactly");
    }

    #[test]
    fn float_output_is_interleaved_and_unclipped() {
        let p = tmp("f32out.wav");
        let planar = vec![vec![0.0, 0.5, 0.25], vec![1.0, -0.5, -0.25]];
        write_wav_ieee_float(&p, &planar, 48000, 32).unwrap();
        let b = std::fs::read(&p).unwrap();
        let hdr_len = header_len_for(2, 32, true).unwrap();
        assert_eq!(hdr_len, 58);
        assert_eq!(&b[38..42], b"fact");
        let got: Vec<f32> = b[hdr_len..]
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        assert_eq!(got, vec![0.0, 1.0, 0.5, -0.5, 0.25, -0.25]);
        assert_eq!(u16::from_le_bytes([b[20], b[21]]), 3, "IEEE float tag");
        assert_eq!(u32::from_le_bytes([b[28], b[29], b[30], b[31]]), 384_000);

        // 64-bit keeps the value exactly, and out-of-range values survive: the
        // float container is not a limiter.
        let p = tmp("f64out.wav");
        write_wav_ieee_float(&p, &[vec![0.1, 2.0, f64::NAN]], 48000, 64).unwrap();
        let b = std::fs::read(&p).unwrap();
        let hdr_len64 = header_len_for(1, 64, true).unwrap();
        assert_eq!(hdr_len64, 58);
        let v = f64::from_le_bytes(b[hdr_len64..hdr_len64 + 8].try_into().unwrap());
        assert_eq!(v, 0.1);
        let v = f64::from_le_bytes(b[hdr_len64 + 8..hdr_len64 + 16].try_into().unwrap());
        assert_eq!(v, 2.0);
        assert!(f64::from_le_bytes(b[hdr_len64 + 16..hdr_len64 + 24].try_into().unwrap()).is_nan());
    }

    #[test]
    fn pcm_path_cleanses_non_finite_before_quantising() {
        let p = tmp("cleanse.wav");
        let planar = vec![vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]];
        write_wav_pcm_or_float(&p, &planar, 48000, 16).unwrap();
        let b = std::fs::read(&p).unwrap();
        let mut q = Vec::new();
        for w in b[44..].chunks_exact(2) {
            q.push(i16::from_le_bytes([w[0], w[1]]));
        }
        // NaN -> 0, +Inf -> full scale, -Inf -> -full scale, each with dither.
        assert!(q[0].abs() <= 2, "nan became {}", q[0]);
        assert!(q[1] > 32700 || q[1] == 32767, "posinf became {}", q[1]);
        assert!(q[2] < -32700 || q[2] == -32768, "neginf became {}", q[2]);
    }

    #[test]
    fn round_trip_through_every_depth() {
        let sig: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin() * 0.9).collect();
        for bits in [16u32, 24, 32, 64] {
            let p = tmp(&format!("rt{bits}.wav"));
            write_wav_pcm_or_float(
                &p,
                &[sig.clone(), sig.iter().map(|v| -v).collect()],
                48000,
                bits,
            )
            .unwrap();
            let a = read_wav(&p).unwrap();
            assert_eq!(a.fs, 48000);
            assert_eq!(a.channels(), 2);
            assert_eq!(a.frames(), 64);
            // 16/24-bit carry at most a LSB of dither plus a LSB of quantisation;
            // IEEE float 32 is a plain cast that still loses the tail of an f64
            // (≈1e-7 near full scale); IEEE float 64 is bit-exact.
            let tol = match bits {
                16 => 2.0 / 32768.0,
                24 => 2.0 / 8_388_608.0,
                32 => 1e-6,
                _ => 0.0,
            };
            for (i, &v) in a.planar[0].iter().enumerate().take(64) {
                assert!(
                    v.abs() < 1.0,
                    "bits {bits} sample {i} decoded outside the unit range: {v}"
                );
                assert!(
                    (v - sig[i]).abs() <= tol + f64::EPSILON,
                    "bits {bits} sample {i}: {v} vs {}",
                    sig[i]
                );
                // The float containers are a plain cast, so the mirrored channel
                // comes back exactly negated. PCM does not: each (frame, channel)
                // gets its own TPDF draw from the fixed stream, so `-sig` is
                // quantized with different noise and lands within a couple of LSB
                // of the negation rather than on it.
                let spread = (a.planar[1][i] + a.planar[0][i]).abs();
                if bits == 64 {
                    assert_eq!(a.planar[1][i], -a.planar[0][i]);
                } else if bits == 32 {
                    assert!(spread <= 2.0 * tol, "bits {bits} sample {i}: {spread}");
                } else {
                    assert!(
                        spread <= 2.0 * tol,
                        "bits {bits} sample {i}: channels are not mirror images ({spread})"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_bad_arguments() {
        let p = tmp("args.wav");
        assert!(write_wav_pcm_or_float(&p, &[], 48000, 24)
            .unwrap_err()
            .to_string()
            .contains("empty channel list"));
        assert!(write_wav_pcm_or_float(&p, &[vec![0.0]], 48000, 8)
            .unwrap_err()
            .to_string()
            .contains("Unsupported output WAV bit depth: 8"));
        assert!(write_wav_ieee_float(&p, &[vec![0.0]], 48000, 24)
            .unwrap_err()
            .to_string()
            .contains("IEEE float output needs 32 or 64 bits"));
    }

    /// The two ceilings the writer enforces, and the wording of the one the reference
    /// owns. Nothing here writes 4 GiB: the guard runs before the first byte, so it is
    /// tested as the arithmetic it is.
    #[test]
    fn the_write_is_refused_before_it_can_pass_a_size_it_cannot_hold() {
        let max = usize::try_from(u32::MAX).expect("32 bits fit anywhere");
        // The reference's own bound and its exact sentence, on a classic 44-byte header.
        let e = riff_guard("big.wav", max - 35, 44).unwrap_err().to_string();
        assert_eq!(
            e, "WAV output 'big.wav' exceeds 4 GiB RIFF limit (4294967296 bytes)",
            "the reference's sentence, byte count included"
        );
        // A payload the data field accepts but the RIFF field of an extensible header does
        // not: the multichannel case that used to slip past and wrap.
        let e = riff_guard("multi.wav", max - 50, 68)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("cannot be written"),
            "the second ceiling must speak for itself, got: {e}"
        );
        // Ordinary files are untouched by any of this.
        riff_guard("ok.wav", 48000 * 2 * 4, 44).unwrap();
        riff_guard("ok8.wav", 48000 * 8 * 4, 68).unwrap();
    }

    #[test]
    fn a_truncated_chunk_of_any_kind_is_reported() {
        // The reference reads every chunk's payload and raises on a short read, so a file
        // cut off inside a LIST is an error, not a chunk to walk past on the way to data.
        let good = tmp("trun_good.wav");
        write_wav_pcm_or_float(&good, &[vec![0.25, -0.25]], 48_000, 32).unwrap();
        let mut bytes = std::fs::read(&good).unwrap();
        // Append a LIST chunk that promises far more than the file holds.
        bytes.extend_from_slice(b"LIST");
        bytes.extend_from_slice(&(1u32 << 20).to_le_bytes());
        bytes.extend_from_slice(b"INFO");
        let cut = tmp("trun_cut.wav");
        std::fs::write(&cut, &bytes).unwrap();
        let e = read_wav(&cut).unwrap_err().to_string();
        assert!(e.contains("Truncated 'LIST'"), "{e}");

        // An id that is not printable text is replaced rather than reformatted, and a
        // chunk that claims to be there but is not cannot be seeked past the end.
        let mut junk = std::fs::read(&good).unwrap();
        junk.extend_from_slice(&[0xf0u8, 0x9f, 0x92, 0xa9]);
        junk.extend_from_slice(&100u32.to_le_bytes());
        let weird = tmp("trun_weird.wav");
        std::fs::write(&weird, &junk).unwrap();
        let e = read_wav(&weird).unwrap_err().to_string();
        assert!(e.contains('\u{fffd}'), "{e}");
    }

    #[test]
    fn writer_refuses_zero_sample_rate() {
        let p = tmp("zero_rate.wav");
        let planar = vec![vec![0.0; 100]];
        assert!(write_wav_ieee_float(&p, &planar, 0, 32).is_err());
        assert!(write_wav_pcm_or_float(&p, &planar, 0, 16).is_err());
    }

    #[test]
    fn extensible_with_unknown_guid_is_rejected() {
        let p = tmp("bad_guid.wav");
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&72u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(&0xFFFEu16.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&48000u32.to_le_bytes());
        b.extend_from_slice(&(48000 * 4u32).to_le_bytes());
        b.extend_from_slice(&4u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(&22u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&[0xFF; 14]); // bogus GUID tail
        b.extend_from_slice(b"data");
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 8]);
        std::fs::write(&p, &b).unwrap();

        assert!(
            read_wav(&p).is_err(),
            "bogus extensible GUID must be rejected"
        );
    }
}
