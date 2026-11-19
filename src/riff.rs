//! RIFF/WAVE: header layout, streaming writer, and format reader.
//!
//! This is the container layer only — it knows chunk ids and nothing about sample
//! values, so the codec work lives in [`crate::wav`]. Two rules in here are the ones
//! that quietly corrupt files when they are missing, and both are tested from bytes
//! rather than through this module's own reader:
//!
//! * Multi-channel files are written as `WAVE_FORMAT_EXTENSIBLE` with an explicit
//!   channel mask, because that mask is the field a decoder reads to learn the
//!   channel *order*; a plain PCM header carries no order at all, which is how 7.1
//!   files end up with their surrounds in the front slots.
//! * The RIFF and `data` sizes are patched from the header length that was *actually
//!   written*, and the `data` size lives in the last four bytes of that header. A
//!   hard-coded 36 writes a RIFF size 24 bytes short onto every 68-byte extensible
//!   file, and a strict parser then drops the last 24 bytes of audio: the file plays
//!   and is quietly truncated.
//!
//! Chunk bodies are size-capped before they are allocated, and a `data` chunk before
//! any `fmt ` is refused rather than trusted — parsed as a header it claims 0
//! channels at 0 Hz, and every caller believes that until it divides.

use crate::error::{Error, Result};
#[cfg(test)]
use std::io::Read;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

const SUBTYPE_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const SUBTYPE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// Standard channel order for a mask-free file, most-common-first.
pub fn guess_mask(channels: u16) -> u32 {
    const BITS: [u32; 18] = [
        0x1,     // FL
        0x2,     // FR
        0x4,     // FC
        0x8,     // LFE
        0x10,    // BL
        0x20,    // BR
        0x200,   // SL
        0x400,   // SR
        0x40,    // FLC
        0x80,    // FRC
        0x100,   // BC
        0x800,   // TOP_CENTER
        0x1000,  // TOP_FRONT_LEFT
        0x2000,  // TOP_FRONT_CENTER
        0x4000,  // TOP_FRONT_RIGHT
        0x8000,  // TOP_BACK_LEFT
        0x10000, // TOP_BACK_CENTER
        0x20000, // TOP_BACK_RIGHT
    ];
    BITS.iter()
        .take(channels.min(BITS.len() as u16) as usize)
        .fold(0, |acc, bit| acc | bit)
}

/// `nBlockAlign`: one frame's width in bytes, which the container carries in 16 bits.
///
/// A wide, deep file can exceed that — 16384 channels of 32-bit float is 65536 bytes per frame —
/// and the honest answer is that this container cannot hold the file. The alternative was clamping
/// to `u16::MAX`, which writes a header whose stride disagrees with the payload: the file opens,
/// plays, and is wrong, which is worse than refusing.
pub fn block_align(channels: u16, bits: u32) -> Result<u16> {
    if bits == 0 || !bits.is_multiple_of(8) {
        return Err(Error::msg(format!(
            "cannot write {bits}-bit samples: a WAV frame has to be a whole number of bytes"
        )));
    }
    let block = u64::from(channels) * u64::from(bits / 8);
    u16::try_from(block).map_err(|_| {
        Error::msg(format!(
            "cannot write {channels} channels at {bits} bits in one WAV file: one frame is \
             {block} bytes and the RIFF container carries that width in 16 bits (65535 at most)"
        ))
    })
}

/// Classic 44-byte header when the file is 1–2 channels and needs no mask,
/// otherwise the extensible header.
///
/// # Errors
/// When [`block_align`] cannot represent the frame, or the sample width is not a whole number of
/// bytes.
pub fn header(
    rate: u32,
    channels: u16,
    bits: u32,
    float: bool,
    mask: Option<u32>,
) -> Result<Vec<u8>> {
    let block = u32::from(block_align(channels, bits)?);
    let extensible = channels > 2 || mask.is_some_and(|m| m != guess_mask(channels));
    let mut h = Vec::with_capacity(68);
    let tag = if float {
        WAVE_FORMAT_IEEE_FLOAT
    } else {
        WAVE_FORMAT_PCM
    };
    let fmt_tag = if extensible {
        WAVE_FORMAT_EXTENSIBLE
    } else {
        tag
    };
    let fmt_len = if extensible { 40u32 } else { 16u32 };
    // RIFF header: size is patched by the writer, so zero it here.
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&0u32.to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&fmt_len.to_le_bytes());
    h.extend_from_slice(&fmt_tag.to_le_bytes());
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate.saturating_mul(block)).to_le_bytes());
    h.extend_from_slice(&(block as u16).to_le_bytes());
    h.extend_from_slice(&(bits as u16).to_le_bytes());
    if extensible {
        // wValidBitsPerSample: the container and the payload have the same width
        // here, because ffmpeg hands us exactly the precision we asked for.
        h.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        h.extend_from_slice(&(bits as u16).to_le_bytes()); // wValidBitsPerSample
        h.extend_from_slice(&(mask.unwrap_or_else(|| guess_mask(channels))).to_le_bytes());
        h.extend_from_slice(if float { &SUBTYPE_FLOAT } else { &SUBTYPE_PCM });
    }
    h.extend_from_slice(b"data");
    h.extend_from_slice(&0u32.to_le_bytes());
    Ok(h)
}

/// The length of the header [`header`] would produce for these settings.
///
/// Both RIFF size fields are measured from this number, so it is not allowed to be a guess: a
/// 68-byte `WAVE_FORMAT_EXTENSIBLE` header measured as 44 writes a RIFF size 24 bytes short, and
/// near 4 GiB the same confusion is what lets a payload slip past the size check and wrap.
///
/// It is measured from the builder itself — [`header`], run once with a placeholder rate — rather
/// than from arithmetic that could drift away from it. That costs one allocation per output file,
/// against a header that is written once per file.
pub fn header_len(channels: u16, bits: u32, float: bool) -> Result<usize> {
    Ok(header(8000, channels, bits, float, None)?.len())
}

/// The two sizes a WAV header carries, checked before either is written.
///
/// `u32::MAX` is not one ceiling but two: the data chunk size, and the RIFF chunk size,
/// which is the payload *plus the header minus 8*. Measuring only the payload lets a
/// multichannel file pass a check it should fail, and `as u32` then wraps the RIFF size
/// down to a few dozen bytes — a file that plays, and quietly drops the audio past the
/// wrap. The reference shares that arithmetic, so this is one place where being correct
/// means disagreeing with it (documented in `docs/SETUP.md`).
pub fn riff_sizes(payload: u64, hdr_len: u64) -> Result<(u32, u32)> {
    let riff = payload
        .checked_add(hdr_len)
        .and_then(|t| t.checked_sub(8))
        .ok_or_else(|| Error::msg("WAV payload length overflows the RIFF size field"))?;
    let data = u32::try_from(payload)
        .map_err(|_| Error::msg("WAV payload is larger than the data chunk size field"))?;
    let riff = u32::try_from(riff)
        .map_err(|_| Error::msg("WAV file is larger than the RIFF size field"))?;
    Ok((riff, data))
}

/// Streaming WAV writer: header first, sizes patched on [`finish`](WavWriter::finish).
pub struct WavWriter {
    f: std::fs::File,
    /// The length of the header that was actually written. Not a constant: a classic
    /// 2-channel header is 44 bytes and a `WAVE_FORMAT_EXTENSIBLE` one is 68, and both
    /// size fields patched at the end are measured from it.
    hdr_len: u64,
    data_len: u64,
}

impl WavWriter {
    pub fn create(
        path: &Path,
        rate: u32,
        channels: u16,
        bits: u32,
        float: bool,
        mask: Option<u32>,
    ) -> Result<Self> {
        let hdr = header(rate, channels, bits, float, mask)?;
        // Recorded rather than recomputed at the end: the header that exists and the
        // header guessed from the channel count are different lengths as soon as a custom
        // mask is involved, and the two size fields live at the end of whichever one was
        // actually written.
        let hdr_len = hdr.len() as u64;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&hdr)?;
        Ok(Self {
            f,
            hdr_len,
            data_len: 0,
        })
    }

    pub fn write(&mut self, pcm: &[u8]) -> Result<()> {
        self.f.write_all(pcm)?;
        self.data_len += pcm.len() as u64;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        let total = self.data_len;
        // RIFF chunk size = file size - 8; data chunk size = payload. Both measured from
        // `hdr_len`, because a hard-coded 36 (the 44-byte classic header minus 8) writes a
        // RIFF size 24 bytes short onto every 68-byte extensible file — which is every
        // multichannel WAV this program writes. A strict parser honours that size and
        // drops the last 24 bytes of audio, so the file plays and is quietly truncated.
        let (riff, data) = riff_sizes(total, self.hdr_len)?;
        self.f.seek(SeekFrom::Start(4))?;
        self.f.write_all(&riff.to_le_bytes())?;
        // The header ends with "data" followed by its 4-byte size, so the size
        // field is the last 4 bytes of the header — writing 8 bytes back would
        // land on the "data" tag itself and ffmpeg would reject the file.
        self.f.seek(SeekFrom::Start(self.hdr_len - 4))?;
        self.f.write_all(&data.to_le_bytes())?;
        self.f.flush()?;
        Ok(total)
    }
}

/// Parse a WAV header and report where the samples start.
///
/// The whole header walk, in one call, for a caller that only wants to know what it is looking at
/// without reading samples. The program's own reader is [`crate::wav::read_wav`], which walks the
/// chunks differently on purpose — see the note at the top of that module — so this is reached by
/// the tests here and by anyone who needs the header alone.
#[cfg(test)]
pub fn read_info(path: &Path) -> Result<(WavInfo, u64)> {
    read_info_file(&mut std::fs::File::open(path)?)
}

/// What a WAV header claims.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavInfo {
    pub rate: u32,
    pub channels: u16,
    pub bits: u32,
    pub float: bool,
    pub mask: Option<u32>,
    pub data_len: u64,
    /// Raw `wFormatTag` of the `fmt ` chunk (1 PCM, 3 IEEE float, 0xFFFE extensible).
    pub tag: u16,
}

#[cfg(test)]
impl WavInfo {
    pub fn frames(&self) -> u64 {
        let b = (self.channels as u64) * (self.bits as u64 / 8);
        // A header claiming zero bytes per frame is corrupt, not infinite.
        self.data_len.checked_div(b).unwrap_or(0)
    }
}

#[cfg(test)]
pub fn read_info_file(f: &mut std::fs::File) -> Result<(WavInfo, u64)> {
    let mut riff = [0u8; 12];
    f.read_exact(&mut riff).map_err(short)?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err(Error::msg("not a RIFF/WAVE file"));
    }
    let mut info = WavInfo {
        rate: 0,
        channels: 0,
        bits: 0,
        float: false,
        mask: None,
        data_len: 0,
        tag: 0,
    };
    loop {
        let mut ch = [0u8; 8];
        match f.read_exact(&mut ch) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(Error::msg("WAV has no data chunk"))
            }
            Err(e) => return Err(Error::Io(e)),
        }
        // The chunk header is 4 bytes of id followed by 4 bytes of size.
        let id = &ch[0..4];
        let size = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]) as u64;
        let truncated = |e: std::io::Error| -> Error {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                let name: String = id
                    .iter()
                    .map(|&b| {
                        if b.is_ascii_graphic() || b == b' ' {
                            b as char
                        } else {
                            '\u{fffd}'
                        }
                    })
                    .collect();
                Error::msg(format!("Truncated '{name}' chunk"))
            } else {
                Error::Io(e)
            }
        };
        if id == b"fmt " {
            // Cap checked before the allocation, not after. A chunk header can claim any
            // 32-bit size, and `vec![0u8; size]` on the strength of it means a 44-byte
            // file can ask for 4 GiB of RAM before anything reads it and notices. The
            // largest fmt chunk that exists in the wild is 40 bytes.
            if size > 1024 {
                return Err(Error::msg("absurd WAV fmt chunk size"));
            }
            let mut b = vec![0u8; size as usize];
            f.read_exact(&mut b).map_err(truncated)?;
            // RIFF chunks are word-aligned: an odd payload carries a pad byte that belongs
            // to it, and skipping only the payload leaves every later header one byte early.
            if size & 1 == 1 {
                f.seek(SeekFrom::Current(1))?;
            }
            if b.len() < 16 {
                return Err(Error::msg("short WAV fmt chunk"));
            }
            let tag = u16::from_le_bytes([b[0], b[1]]);
            info.tag = tag;
            info.channels = u16::from_le_bytes([b[2], b[3]]);
            info.rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            info.bits = u16::from_le_bytes([b[14], b[15]]) as u32;
            info.float = tag == WAVE_FORMAT_IEEE_FLOAT;
            if tag == WAVE_FORMAT_EXTENSIBLE && b.len() >= 40 {
                // cbSize is at 16, so dwChannelMask follows the 2-byte valid-bits
                // field at 20, and the 16-byte sub-format GUID starts at 24.
                info.mask = Some(u32::from_le_bytes([b[20], b[21], b[22], b[23]]));
                // The sub-format is a 16-byte GUID whose first two bytes are the old
                // WAVE_FORMAT tag. Reading one byte would call anything starting with 0x03
                // float; the tag is the whole 16-bit field.
                info.float = u16::from_le_bytes([b[24], b[25]]) == WAVE_FORMAT_IEEE_FLOAT;
            } else if info.channels > 2 {
                info.mask = Some(guess_mask(info.channels));
            }
        } else if id == b"data" {
            // Samples with no format in front of them are not a stream. Returned as a
            // parsed header it says 0 channels at 0 Hz and 0 bits, which every caller
            // believes until it divides.
            if info.channels == 0 || info.rate == 0 || info.bits == 0 {
                return Err(Error::msg("WAV data chunk before any fmt chunk"));
            }
            let off = f.stream_position()?;
            info.data_len = size;
            return Ok((info, off));
        } else {
            f.seek(SeekFrom::Current(size as i64 + (size & 1) as i64))?;
        }
    }
}

#[cfg(test)]
fn short(e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        Error::msg("file is shorter than its WAV header")
    } else {
        Error::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hypreq-riff-wav-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    /// The declared sizes of an 8-channel file, read back as bytes rather than through
    /// this module's own reader: a reader that trusts the header cannot catch a header
    /// that lies.
    #[test]
    fn multichannel_wav_declares_its_real_sizes() {
        let p = tmp("8ch-sizes.wav");
        let pcm = vec![0x5au8; 1024 * 8 * 4];
        let mut w = WavWriter::create(&p, 48000, 8, 32, false, None).unwrap();
        w.write(&pcm).unwrap();
        w.finish().unwrap();
        let b = std::fs::read(&p).unwrap();
        let len = b.len() as u64;
        assert_eq!(len, 68 + pcm.len() as u64, "extensible header is 68 bytes");
        assert_eq!(&b[60..64], b"data", "data chunk must sit at 60");
        let riff = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as u64;
        let data = u32::from_le_bytes([b[64], b[65], b[66], b[67]]) as u64;
        assert_eq!(riff, len - 8, "RIFF size is the file minus 8");
        assert_eq!(data, pcm.len() as u64, "data size is the payload");
    }

    #[test]
    fn a_custom_mask_on_two_channels_keeps_its_guid_and_sizes() {
        let p = tmp("2ch-mask.wav");
        let pcm = vec![0x11u8; 512 * 2 * 2];
        let mut w = WavWriter::create(&p, 48000, 2, 16, false, Some(0x4)).unwrap();
        w.write(&pcm).unwrap();
        w.finish().unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[60..64], b"data", "header stayed extensible");
        assert_eq!(
            &b[44..50],
            &SUBTYPE_PCM[0..6],
            "the SubFormat GUID must survive the size patch"
        );
        let riff = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as u64;
        assert_eq!(riff, b.len() as u64 - 8);
        let data = u32::from_le_bytes([b[64], b[65], b[66], b[67]]) as u64;
        assert_eq!(data, pcm.len() as u64, "the real data size, not zero");
        let (info, off) = read_info(&p).unwrap();
        assert_eq!(info.mask, Some(0x4));
        assert_eq!(info.frames(), 512);
        assert_eq!(off as usize + pcm.len(), b.len());
    }

    #[test]
    fn data_before_fmt_is_refused_not_trusted() {
        let p = tmp("data-first.wav");
        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&36u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"data");
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 8]);
        std::fs::write(&p, &b).unwrap();
        let e = read_info(&p).unwrap_err();
        assert!(
            e.to_string().contains("fmt"),
            "the error names what is missing: {e}"
        );
    }

    /// A `fmt ` chunk whose payload is an odd number of bytes is followed by a pad byte
    /// that belongs to the chunk. Skipping only the payload leaves every following header
    /// one byte early — which does not report a wrong number, it stops the file parsing.
    #[test]
    fn an_odd_sized_fmt_chunk_leaves_the_next_header_aligned() {
        let p = tmp("odd-fmt.wav");
        let mut fmt_body: Vec<u8> = Vec::new();
        fmt_body.extend_from_slice(&1u16.to_le_bytes()); // PCM
        fmt_body.extend_from_slice(&2u16.to_le_bytes()); // two channels
        fmt_body.extend_from_slice(&44100u32.to_le_bytes());
        fmt_body.extend_from_slice(&176_400u32.to_le_bytes()); // byte rate
        fmt_body.extend_from_slice(&4u16.to_le_bytes()); // block align
        fmt_body.extend_from_slice(&16u16.to_le_bytes()); // bits
        fmt_body.push(0x5A); // one byte of extension, so the chunk size is odd

        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
        b.extend_from_slice(&fmt_body);
        b.push(0); // the alignment pad byte
        let payload = vec![0u8; 4 * 10]; // 10 frames x 2ch x 16-bit
        let payload_start = b.len();
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        b.extend_from_slice(&payload);
        let len = b.len() as u32;
        b[4..8].copy_from_slice(&(len - 8).to_le_bytes());
        std::fs::write(&p, &b).unwrap();

        let (info, off) = read_info(&p).unwrap_or_else(|e| panic!("odd fmt payload: {e}"));
        assert_eq!((info.channels, info.rate, info.bits), (2, 44100, 16));
        assert_eq!(
            off as usize,
            payload_start + 8,
            "the data payload begins where the file says it does"
        );
        assert_eq!(info.frames(), 10);
    }

    /// The same word-alignment rule on a chunk the reader does not interpret: an odd
    /// `LIST` payload is followed by a pad byte, and skipping only the payload puts the
    /// `data` header one byte early — the metadata chunks are where a reader that only
    /// ever met 44-byte headers finds out it has to count bytes.
    #[test]
    fn an_odd_metadata_chunk_is_skipped_with_its_pad_byte() {
        let p = tmp("odd-list.wav");
        let mut fmt_body: Vec<u8> = Vec::new();
        fmt_body.extend_from_slice(&1u16.to_le_bytes());
        fmt_body.extend_from_slice(&1u16.to_le_bytes());
        fmt_body.extend_from_slice(&8000u32.to_le_bytes());
        fmt_body.extend_from_slice(&8000u32.to_le_bytes());
        fmt_body.extend_from_slice(&1u16.to_le_bytes());
        fmt_body.extend_from_slice(&8u16.to_le_bytes());

        let list = b"INFOISFTHyprE".to_vec(); // 13 bytes: odd, so a pad byte follows
        let payload = vec![0x80u8; 32];
        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
        b.extend_from_slice(&fmt_body);
        let list_start = b.len();
        b.extend_from_slice(b"LIST");
        b.extend_from_slice(&(list.len() as u32).to_le_bytes());
        b.extend_from_slice(&list);
        b.push(0); // pad, because (list_start + 8 + 14) is odd
        let payload_start = b.len();
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        b.extend_from_slice(&payload);
        let len = b.len() as u32;
        b[4..8].copy_from_slice(&(len - 8).to_le_bytes());
        std::fs::write(&p, &b).unwrap();
        assert_eq!(
            (list_start + 8 + list.len()) % 2,
            1,
            "the fixture is odd-sized"
        );

        let (info, off) = read_info(&p).unwrap_or_else(|e| panic!("odd LIST chunk: {e}"));
        assert_eq!((info.channels, info.rate, info.bits), (1, 8000, 8));
        assert_eq!(off as usize, payload_start + 8, "found where it is");
        assert_eq!(info.frames(), 32);
    }

    /// The `SubFormat` of a `WAVE_FORMAT_EXTENSIBLE` header is a 16-bit tag in the low
    /// half of a 16-byte GUID. One byte is the same answer for the two formats in
    /// circulation (`01 00` PCM, `03 00` float) and the wrong answer for anything else.
    #[test]
    fn the_extensible_subformat_tag_is_sixteen_bits_wide() {
        let extensible = |sub_low: u16, bits: u16| {
            let mut body: Vec<u8> = Vec::new();
            body.extend_from_slice(&0xFFFEu16.to_le_bytes()); // WAVE_FORMAT_EXTENSIBLE
            body.extend_from_slice(&1u16.to_le_bytes());
            body.extend_from_slice(&48000u32.to_le_bytes());
            body.extend_from_slice(&(48000u32 * 4).to_le_bytes());
            body.extend_from_slice(&4u16.to_le_bytes());
            body.extend_from_slice(&bits.to_le_bytes());
            body.extend_from_slice(&22u16.to_le_bytes()); // cbSize
            body.extend_from_slice(&bits.to_le_bytes()); // valid bits
            body.extend_from_slice(&0x4000_0003u32.to_le_bytes()); // FL|FR + mask bit
            let mut guid = vec![0u8; 16];
            guid[..2].copy_from_slice(&sub_low.to_le_bytes());
            guid[6] = 0x10; // as in every KSDATAFORMAT_SUBTYPE_* GUID
            body.extend_from_slice(&guid);

            let mut b: Vec<u8> = b"RIFF".to_vec();
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(b"WAVE");
            b.extend_from_slice(b"fmt ");
            b.extend_from_slice(&(body.len() as u32).to_le_bytes());
            b.extend_from_slice(&body);
            b.extend_from_slice(b"data");
            b.extend_from_slice(&8u32.to_le_bytes());
            b.extend_from_slice(&[0u8; 8]);
            let len = b.len() as u32;
            b[4..8].copy_from_slice(&(len - 8).to_le_bytes());
            b
        };
        let write = |name: &str, bytes: Vec<u8>| {
            let p = tmp(name);
            std::fs::write(&p, &bytes).unwrap();
            p
        };

        // Controls: the two subtypes that exist.
        let p = write("sub-pcm.wav", extensible(0x0001, 32));
        let (info, _) = read_info(&p).unwrap();
        assert!(!info.float, "0x0001 is PCM");
        let p = write("sub-float.wav", extensible(0x0003, 32));
        let (info, _) = read_info(&p).unwrap();
        assert!(info.float, "0x0003 is IEEE float");

        // 0x0103: the low byte is 3, so a one-byte read calls it float. The tag is both
        // bytes, and both bytes are not 3.
        let p = write("sub-neither.wav", extensible(0x0103, 32));
        let (info, _) = read_info(&p).unwrap();
        assert!(
            !info.float,
            "a SubFormat of 0x0103 is not IEEE float; one byte was read as 0x03"
        );
        assert_eq!(info.tag, 0xFFFE, "the container tag is still extensible");
    }

    #[test]
    fn an_absurd_fmt_size_is_refused_before_it_is_allocated() {
        let p = tmp("huge-fmt.wav");
        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&400u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        b.extend_from_slice(&[0u8; 16]);
        std::fs::write(&p, &b).unwrap();
        let e = read_info(&p).unwrap_err();
        assert!(e.to_string().contains("fmt"), "got: {e}");
    }

    /// The header length both size fields are measured from, checked against the header
    /// builder rather than against arithmetic about it.
    #[test]
    fn the_measured_header_is_the_header_written() {
        for (ch, bits, float, want) in [
            (1u16, 16, false, 44usize),
            (2, 32, true, 44),
            (2, 24, false, 44),
            (3, 16, false, 68),
            (8, 32, true, 68),
        ] {
            assert_eq!(
                header_len(ch, bits, float).unwrap(),
                header(48000, ch, bits, float, None).unwrap().len(),
                "{ch}ch {bits}bit float={float}"
            );
            assert_eq!(
                header_len(ch, bits, float).unwrap(),
                want,
                "{ch}ch {bits}bit"
            );
        }
    }

    /// A payload can fit the data chunk's 32-bit size and still not fit the RIFF chunk's,
    /// because the RIFF field also counts the header. Truncating instead of refusing is
    /// what writes a file that plays and is silently cut at the wrap.
    #[test]
    fn a_payload_too_wide_for_the_riff_field_is_refused_not_wrapped() {
        let max = u64::from(u32::MAX);
        // Comfortably inside both fields.
        assert_eq!(riff_sizes(1000, 44).unwrap(), (1036, 1000));
        assert_eq!(riff_sizes(1000, 68).unwrap(), (1060, 1000));
        // Fits the data field, does not fit the RIFF field of a 68-byte header: exactly
        // the multichannel case a payload-only check lets through.
        let too_big = max - 50;
        assert!(
            u32::try_from(too_big).is_ok(),
            "the data chunk field alone would accept it"
        );
        assert_eq!(
            (too_big + 68 - 8) as u32,
            9,
            "and a truncating cast writes a 9-byte RIFF size for it"
        );
        assert!(
            riff_sizes(too_big, 68).is_err(),
            "so this must be an error, not a 9-byte header"
        );
        // The largest payload that genuinely fits a classic header, at the boundary.
        assert!(
            riff_sizes(max - 36, 44).is_ok(),
            "max - 36 is the true limit"
        );
        assert!(riff_sizes(max - 35, 44).is_err(), "one byte past it");
        // And a payload that does not fit its own field at all.
        assert!(riff_sizes(max + 1, 44).is_err());
    }

    #[test]
    fn roundtrip_8ch_s32_carries_the_7_1_mask() {
        let p = tmp("8ch.wav");
        let pcm = vec![0u8; 48000 * 8 * 4];
        let mut w = WavWriter::create(&p, 48000, 8, 32, false, None).unwrap();
        w.write(&pcm).unwrap();
        w.finish().unwrap();
        let (info, off) = read_info(&p).unwrap();
        assert_eq!(info.channels, 8);
        assert_eq!(info.rate, 48000);
        assert_eq!(info.bits, 32);
        assert_eq!(info.mask, Some(0x63f), "7.1 mask");
        assert_eq!(off as usize, 68, "extensible header is 68 bytes");
        assert_eq!(info.frames(), 48000);
    }

    #[test]
    fn stereo_stays_classic_44_bytes() {
        let p = tmp("st.wav");
        let pcm = vec![0u8; 44100 * 2 * 4];
        let mut w = WavWriter::create(&p, 44100, 2, 32, false, None).unwrap();
        w.write(&pcm).unwrap();
        w.finish().unwrap();
        let (info, off) = read_info(&p).unwrap();
        assert_eq!(off as usize, 44);
        assert_eq!(info.mask, None);
        assert_eq!(info.frames(), 44100);
    }

    #[test]
    fn patched_sizes_are_right() {
        let p = tmp("sizes.wav");
        let pcm = vec![7u8; 1000];
        let mut w = WavWriter::create(&p, 48000, 2, 16, false, None).unwrap();
        w.write(&pcm).unwrap();
        w.finish().unwrap();
        let len = std::fs::metadata(&p).unwrap().len();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(len, 44 + 1000);
        assert_eq!(u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as u64, len - 8);
        assert_eq!(
            u32::from_le_bytes([b[40], b[41], b[42], b[43]]) as u64,
            1000,
            "data chunk size"
        );
        assert_eq!(&b[44..], &pcm[..]);
    }

    #[test]
    fn rejects_non_wav() {
        let p = tmp("bad.wav");
        std::fs::write(&p, b"not a wav at all, really").unwrap();
        assert!(read_info(&p).is_err());
    }

    #[test]
    fn f64_float_header_roundtrip() {
        let p = tmp("f64.wav");
        let h = header(48000, 8, 64, true, Some(0x63f)).unwrap();
        assert_eq!(h.len(), 68);
        assert_eq!(u16::from_le_bytes([h[20], h[21]]), WAVE_FORMAT_EXTENSIBLE);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        drop(f);
        let (info, _) = read_info(&p).unwrap();
        assert!(info.float);
        assert_eq!(info.bits, 64);
        assert_eq!(info.mask, Some(0x63f));
    }

    /// The mono/stereo path HyprEQ writes is the plain 44-byte header with
    /// `wFormatTag` 3 (float) or 1 (PCM) and `cbSize` absent — byte-for-byte what
    /// `hypreq.py` puts on disk.
    /// One frame's width is carried in 16 bits. Clamping a value that does not fit wrote a header
    /// whose stride disagreed with the payload — the file opened, played, and was wrong — so the
    /// builder refuses it instead.
    #[test]
    fn a_frame_wider_than_sixteen_bits_is_refused_not_clamped() {
        assert_eq!(block_align(2, 32).unwrap(), 8);
        assert_eq!(block_align(8, 64).unwrap(), 64);
        // 65535 bytes is the widest frame the container can describe; one byte more is not.
        assert_eq!(block_align(65535, 8).unwrap(), 65535);
        for (ch, bits) in [(16384u16, 32u32), (65535, 16), (8192, 64)] {
            let e = block_align(ch, bits).err().unwrap_or_else(|| {
                panic!(
                    "{ch}ch at {bits} bits is {} bytes per frame and was accepted",
                    u64::from(ch) * u64::from(bits / 8)
                )
            });
            assert!(
                e.to_string().contains("65535"),
                "the refusal did not state the limit: {e}"
            );
            assert!(
                header(48000, ch, bits, true, None).is_err(),
                "the header builder accepted a frame it cannot write"
            );
        }
        // A sample width that is not a whole byte would give a frame of 0 bytes: the same class of
        // lie about the layout, refused by the same guard.
        assert!(block_align(2, 0).is_err(), "a 0-bit frame is not a frame");
        assert!(
            block_align(2, 20).is_err(),
            "a 20-bit frame is not whole bytes"
        );
    }

    #[test]
    fn hypreq_classic_header_matches_python_layout() {
        let h = header(48000, 2, 32, true, None).unwrap();
        assert_eq!(h.len(), 44);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes([h[4], h[5], h[6], h[7]]), 0);
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(u32::from_le_bytes([h[16], h[17], h[18], h[19]]), 16);
        assert_eq!(u16::from_le_bytes([h[20], h[21]]), WAVE_FORMAT_IEEE_FLOAT);
        assert_eq!(u16::from_le_bytes([h[22], h[23]]), 2);
        assert_eq!(u32::from_le_bytes([h[24], h[25], h[26], h[27]]), 48000);
        // byte rate = 48000 * 2 * 4
        assert_eq!(u32::from_le_bytes([h[28], h[29], h[30], h[31]]), 384_000);
        assert_eq!(u16::from_le_bytes([h[32], h[33]]), 8);
        assert_eq!(u16::from_le_bytes([h[34], h[35]]), 32);
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32::from_le_bytes([h[40], h[41], h[42], h[43]]), 0);
    }
}
