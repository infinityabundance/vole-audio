//! FLAC bitstream mechanism trace (Report 3, **Seal S0**, diagnostic only).
//!
//! The Seal-K real-speech result compares VOLE against the *actual* frozen B1
//! FLAC artifact. To attribute every remaining byte correctly we must inspect
//! that artifact — not a second file, not an external `flac -a` run. This module
//! is a **self-verifying** FLAC frame/subframe parser:
//!
//! * the frame-header CRC-8 and the frame-footer CRC-16 are checked byte for
//!   byte against the actual stream (RFC 9639 §9.1.7, §9.1.8);
//! * every frame is reconstructed from its parsed subframes and partition-Rice
//!   residual, and the parser *fails* if the total consumed bytes disagree with
//!   the stream length or the STREAMINFO sample count;
//! * the reconstructed interleaved samples are returned so a court can require
//!   them to equal the encoder's own exact round trip.
//!
//! Because the reconstruction is complete, the trace is an authority on the
//! structure of the bytes rather than a guess about them. It has **zero VOLE
//! semantic authority**: it only reads a comparator artifact.
//!
//! Scope: the format features libFLAC actually emits at levels 0–8 — constant,
//! verbatim, fixed 0–4, LPC 1–32 subframes; partitioned Rice with 4- or 5-bit
//! parameters and the raw escape; all channel assignments 0–10; fixed and
//! variable blocking strategies; 8/12/16/20/24/32-bit streams.

use crate::error::{Error, Kind, Result};

/// One parsed FLAC stream.
#[derive(Debug, Clone)]
pub struct FlacTrace {
    pub marker_ok: bool,
    pub streaminfo: StreamInfo,
    pub metadata_bytes: usize,
    pub frames: Vec<FrameTrace>,
    /// Reconstructed interleaved samples (`i64` because a 32-bit side channel
    /// is 33-bit wide).
    pub decoded: Vec<i64>,
    pub decoded_channels: u8,
}

impl FlacTrace {
    /// Aggregate subframe-kind histogram `(constant, verbatim, fixed, lpc)`.
    pub fn kind_histogram(&self) -> [u64; 4] {
        let mut h = [0u64; 4];
        for f in &self.frames {
            for s in &f.subframes {
                match s.kind {
                    SubframeKind::Constant { .. } => h[0] += 1,
                    SubframeKind::Verbatim => h[1] += 1,
                    SubframeKind::Fixed { .. } => h[2] += 1,
                    SubframeKind::Lpc { .. } => h[3] += 1,
                }
            }
        }
        h
    }
}

/// The frozen STREAMINFO fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamInfo {
    pub min_block_size: u16,
    pub max_block_size: u16,
    pub min_frame_bytes: u32,
    pub max_frame_bytes: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    pub total_samples: u64,
    pub md5: [u8; 16],
}

/// One parsed frame.
#[derive(Debug, Clone)]
pub struct FrameTrace {
    /// Frame number (fixed blocking) or first-sample number (variable blocking).
    pub coded_number: u64,
    pub variable_blocking: bool,
    pub block_size: u32,
    pub sample_rate: u32,
    pub channel_assignment: u8,
    pub bits_per_sample: u8,
    pub start_byte: usize,
    pub frame_bytes: usize,
    pub header_bits: u32,
    pub crc8_ok: bool,
    pub crc16_ok: bool,
    pub subframes: Vec<SubframeTrace>,
    /// Reconstructed interleaved samples for this frame.
    pub decoded: Vec<i64>,
}

impl FrameTrace {
    /// Total subframe bits (all channels).
    pub fn subframe_bits(&self) -> u64 {
        self.subframes.iter().map(|s| u64::from(s.total_bits)).sum()
    }
}

/// The subframe model actually selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubframeKind {
    Constant {
        value: i64,
    },
    Verbatim,
    Fixed {
        order: u32,
    },
    Lpc {
        order: u32,
        precision: u32,
        shift: i32,
        coeffs: Vec<i32>,
    },
}

impl SubframeKind {
    pub const fn label(&self) -> &'static str {
        match self {
            SubframeKind::Constant { .. } => "constant",
            SubframeKind::Verbatim => "verbatim",
            SubframeKind::Fixed { .. } => "fixed",
            SubframeKind::Lpc { .. } => "lpc",
        }
    }
}

/// One parsed subframe.
#[derive(Debug, Clone)]
pub struct SubframeTrace {
    pub kind: SubframeKind,
    pub wasted_bits: u32,
    /// Channel bits per sample before wasted-bit reduction.
    pub channel_bits: u8,
    /// Effective subframe bits per sample.
    pub subframe_bits: u8,
    pub blocksize: u32,
    /// Bits in the subframe header (type + wasted-bits unary).
    pub header_bits: u32,
    /// Bits of warmup samples (fixed/LPC).
    pub warmup_bits: u64,
    /// Bits of LPC coefficients (precision+shift+coeffs), 0 for other kinds.
    pub coeff_bits: u64,
    /// Residual description, absent for constant/verbatim.
    pub residual: Option<ResidualTrace>,
    /// Total bits consumed by this subframe.
    pub total_bits: u32,
    /// Reconstructed subframe samples (still channel-decorrelated).
    pub samples: Vec<i64>,
}

impl SubframeTrace {
    pub fn residual_bits(&self) -> u64 {
        self.residual
            .as_ref()
            .map(|r| u64::from(r.header_bits) + r.payload_bits)
            .unwrap_or(0)
    }
}

/// Partitioned-Rice residual description.
#[derive(Debug, Clone)]
pub struct ResidualTrace {
    /// 0 = 4-bit parameters, 1 = 5-bit parameters.
    pub method: u8,
    pub partition_order: u32,
    pub parts: Vec<ResidualPart>,
    pub header_bits: u32,
    pub payload_bits: u64,
}

/// One residual partition.
#[derive(Debug, Clone, Copy)]
pub struct ResidualPart {
    pub samples: u32,
    pub parameter: u32,
    /// `true` when the raw (escape) coding was used.
    pub escaped: bool,
    /// Bits per raw sample on the escape path.
    pub raw_width: u32,
    pub bits: u64,
}

// ---------------------------------------------------------------------------
// Bit reader
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        BitReader { bytes, bit: 0 }
    }

    fn bit_len(&self) -> usize {
        self.bytes.len() * 8
    }

    fn read_bits(&mut self, n: u32) -> Result<u64> {
        if n > 64 {
            return Err(Error::malformed("FLAC trace: bit read too wide"));
        }
        let mut v = 0u64;
        for _ in 0..n {
            if self.bit >= self.bit_len() {
                return Err(Error::malformed("FLAC trace: stream truncated"));
            }
            let byte = self.bytes[self.bit >> 3];
            let bit = (byte >> (7 - (self.bit & 7))) & 1;
            v = (v << 1) | u64::from(bit);
            self.bit += 1;
        }
        Ok(v)
    }

    fn read_signed(&mut self, n: u32) -> Result<i64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 {
            return Err(Error::malformed("FLAC trace: signed read too wide"));
        }
        let v = self.read_bits(n)?;
        if n == 64 {
            return Ok(v as i64);
        }
        let m = 1u64 << (n - 1);
        if v & m != 0 {
            Ok((v | !((1u64 << n) - 1)) as i64)
        } else {
            Ok(v as i64)
        }
    }

    fn read_unary(&mut self) -> Result<u64> {
        let mut n = 0u64;
        loop {
            if self.bit >= self.bit_len() {
                return Err(Error::malformed("FLAC trace: unary run truncated"));
            }
            let byte = self.bytes[self.bit >> 3];
            let bit = (byte >> (7 - (self.bit & 7))) & 1;
            self.bit += 1;
            if bit == 1 {
                return Ok(n);
            }
            n += 1;
            if n > (1 << 24) {
                return Err(Error::malformed("FLAC trace: unary run unbounded"));
            }
        }
    }

    fn align(&mut self) {
        self.bit = (self.bit + 7) & !7;
    }
}

// ---------------------------------------------------------------------------
// CRC
// ---------------------------------------------------------------------------

/// FLAC frame-header CRC-8 (`x^8 + x^2 + x + 1`, init 0).
fn crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in bytes {
        crc ^= b;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// FLAC frame-footer CRC-16 (`x^16 + x^15 + x^2 + 1`, init 0).
fn crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in bytes {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse and fully reconstruct a FLAC stream.
pub fn trace_flac(bytes: &[u8]) -> Result<FlacTrace> {
    if bytes.len() < 8 || &bytes[..4] != b"fLaC" {
        return Err(Error::malformed("FLAC trace: missing stream marker"));
    }
    let mut pos = 4usize;
    let mut streaminfo: Option<StreamInfo> = None;
    loop {
        if pos + 4 > bytes.len() {
            return Err(Error::malformed("FLAC trace: metadata truncated"));
        }
        let header = bytes[pos];
        let last = header & 0x80 != 0;
        let kind = header & 0x7F;
        let len = (usize::from(bytes[pos + 1]) << 16)
            | (usize::from(bytes[pos + 2]) << 8)
            | usize::from(bytes[pos + 3]);
        pos += 4;
        if pos + len > bytes.len() {
            return Err(Error::malformed("FLAC trace: metadata body truncated"));
        }
        if kind == 0 {
            streaminfo = Some(parse_streaminfo(&bytes[pos..pos + len])?);
        }
        pos += len;
        if last {
            break;
        }
    }
    let streaminfo =
        streaminfo.ok_or_else(|| Error::malformed("FLAC trace: stream has no STREAMINFO"))?;
    let metadata_bytes = pos;

    let mut frames = Vec::new();
    let mut decoded: Vec<i64> = Vec::new();
    let mut frame_index = 0u64;
    while pos < bytes.len() {
        let (frame, next) = parse_frame(bytes, pos, &streaminfo)?;
        decoded.extend_from_slice(&frame.decoded);
        pos = next;
        frames.push(frame);
        frame_index += 1;
        if frame_index > 1_000_000 {
            return Err(Error::limit("FLAC trace: absurd frame count"));
        }
    }
    if streaminfo.total_samples != 0
        && decoded.len() as u64 != streaminfo.total_samples * u64::from(streaminfo.channels)
    {
        return Err(Error::new(
            Kind::Integrity,
            "FLAC trace: reconstructed sample count disagrees with STREAMINFO",
        ));
    }
    Ok(FlacTrace {
        marker_ok: true,
        streaminfo,
        metadata_bytes,
        frames,
        decoded,
        decoded_channels: streaminfo.channels,
    })
}

fn parse_streaminfo(body: &[u8]) -> Result<StreamInfo> {
    if body.len() < 34 {
        return Err(Error::malformed("FLAC trace: STREAMINFO too short"));
    }
    let min_block_size = u16::from_be_bytes(body[0..2].try_into().unwrap());
    let max_block_size = u16::from_be_bytes(body[2..4].try_into().unwrap());
    let min_frame_bytes = u32::from_be_bytes([0, body[4], body[5], body[6]]);
    let max_frame_bytes = u32::from_be_bytes([0, body[7], body[8], body[9]]);
    // 20 bits sample rate | 3 bits (channels-1) | 5 bits (bps-1) | 36 bits total.
    let packed: u128 = ((u64::from(body[10]) << 56)
        | (u64::from(body[11]) << 48)
        | (u64::from(body[12]) << 40)
        | (u64::from(body[13]) << 32)
        | (u64::from(body[14]) << 24)
        | (u64::from(body[15]) << 16)
        | (u64::from(body[16]) << 8)
        | u64::from(body[17])) as u128;
    let sample_rate = ((packed >> 44) & 0x000F_FFFF) as u32;
    let channels = (((packed >> 41) & 0x7) + 1) as u8;
    let bits_per_sample = (((packed >> 36) & 0x1F) + 1) as u8;
    let total_samples = packed & 0x0000_000F_FFFF_FFFF;
    let mut md5 = [0u8; 16];
    md5.copy_from_slice(&body[18..34]);
    Ok(StreamInfo {
        min_block_size,
        max_block_size,
        min_frame_bytes,
        max_frame_bytes,
        sample_rate,
        channels,
        bits_per_sample,
        total_samples: total_samples as u64,
        md5,
    })
}

fn parse_frame(bytes: &[u8], start: usize, si: &StreamInfo) -> Result<(FrameTrace, usize)> {
    let mut r = BitReader::new(&bytes[start..]);
    let sync = r.read_bits(14)?;
    if sync != 0x3FFE {
        return Err(Error::malformed("FLAC trace: frame sync code mismatch"));
    }
    if r.read_bits(1)? != 0 {
        return Err(Error::malformed("FLAC trace: reserved frame bit set"));
    }
    let variable_blocking = r.read_bits(1)? == 1;
    let block_code = r.read_bits(4)? as u32;
    let rate_code = r.read_bits(4)? as u32;
    let channel_assignment = r.read_bits(4)? as u8;
    let size_code = r.read_bits(3)? as u32;
    if r.read_bits(1)? != 0 {
        return Err(Error::malformed("FLAC trace: reserved sample-size bit set"));
    }
    let coded_number = read_utf8_number(&mut r)?;

    let block_size = match block_code {
        0 => return Err(Error::malformed("FLAC trace: reserved block size code")),
        1 => 192,
        2..=5 => 576u32 << (block_code - 2),
        6 => (r.read_bits(8)? + 1) as u32,
        7 => (r.read_bits(16)? + 1) as u32,
        c => 256u32 << (c - 8),
    };
    let sample_rate = match rate_code {
        0 => si.sample_rate,
        1 => 88_200,
        2 => 176_400,
        3 => 192_000,
        4 => 8_000,
        5 => 16_000,
        6 => 22_050,
        7 => 24_000,
        8 => 32_000,
        9 => 44_100,
        10 => 48_000,
        11 => 96_000,
        12 => r.read_bits(8)? as u32 * 1_000,
        13 => r.read_bits(16)? as u32,
        14 => r.read_bits(16)? as u32 * 10,
        _ => return Err(Error::malformed("FLAC trace: invalid sample rate code")),
    };
    let bits_per_sample = match size_code {
        0 => si.bits_per_sample,
        1 => 8,
        2 => 12,
        4 => 16,
        5 => 20,
        6 => 24,
        7 => 32,
        _ => return Err(Error::malformed("FLAC trace: reserved sample size code")),
    };

    // The header CRC-8 covers every header byte before the CRC.
    r.align();
    let header_bytes = r.bit / 8;
    if start + header_bytes + 1 > bytes.len() {
        return Err(Error::malformed("FLAC trace: frame header truncated"));
    }
    let stored_crc8 = bytes[start + header_bytes];
    let crc8_ok = crc8(&bytes[start..start + header_bytes]) == stored_crc8;
    r.bit = (header_bytes + 1) * 8;
    let header_bits = (header_bytes + 1) as u32 * 8;

    let decoder_channels = match channel_assignment {
        0..=7 => channel_assignment + 1,
        8..=10 => 2,
        _ => return Err(Error::malformed("FLAC trace: reserved channel assignment")),
    };

    let mut subframes = Vec::with_capacity(usize::from(decoder_channels));
    for ch in 0..decoder_channels {
        let channel_bits = match channel_assignment {
            8 => {
                if ch == 0 {
                    bits_per_sample
                } else {
                    bits_per_sample + 1
                }
            }
            9 => {
                if ch == 0 {
                    bits_per_sample + 1
                } else {
                    bits_per_sample
                }
            }
            10 => {
                if ch == 0 {
                    bits_per_sample
                } else {
                    bits_per_sample + 1
                }
            }
            _ => bits_per_sample,
        };
        let sf = parse_subframe(&mut r, block_size, channel_bits)?;
        subframes.push(sf);
    }

    r.align();
    let crc16_at = start + r.bit / 8;
    if crc16_at + 2 > bytes.len() {
        return Err(Error::malformed("FLAC trace: frame footer truncated"));
    }
    let stored = u16::from_be_bytes(bytes[crc16_at..crc16_at + 2].try_into().unwrap());
    let crc16_ok = crc16(&bytes[start..crc16_at]) == stored;
    let end = crc16_at + 2;

    let decoded = interleave(channel_assignment, &subframes)?;
    let frame = FrameTrace {
        coded_number,
        variable_blocking,
        block_size,
        sample_rate,
        channel_assignment,
        bits_per_sample,
        start_byte: start,
        frame_bytes: end - start,
        header_bits,
        crc8_ok,
        crc16_ok,
        subframes,
        decoded,
    };
    Ok((frame, end))
}

fn parse_subframe(r: &mut BitReader, blocksize: u32, channel_bits: u8) -> Result<SubframeTrace> {
    let sub_start_bit = r.bit;
    if r.read_bits(1)? != 0 {
        return Err(Error::malformed("FLAC trace: subframe padding bit set"));
    }
    let ty = r.read_bits(6)? as u32;
    let wasted_flag = r.read_bits(1)? == 1;
    let wasted_bits = if wasted_flag {
        (r.read_unary()? + 1) as u32
    } else {
        0
    };
    if u32::from(channel_bits) <= wasted_bits {
        return Err(Error::malformed(
            "FLAC trace: wasted bits exceed sample width",
        ));
    }
    let subframe_bits = channel_bits - wasted_bits as u8;
    let header_bits = (r.bit - sub_start_bit) as u32;

    let (kind, warmup_bits, coeff_bits, residual, samples) = if ty == 0 {
        let v = r.read_signed(u32::from(subframe_bits))?;
        (
            SubframeKind::Constant { value: v },
            0,
            0,
            None,
            vec![v; blocksize as usize],
        )
    } else if ty == 1 {
        let mut vals = Vec::with_capacity(blocksize as usize);
        for _ in 0..blocksize {
            vals.push(r.read_signed(u32::from(subframe_bits))?);
        }
        (SubframeKind::Verbatim, 0, 0, None, vals)
    } else if (0b001_000..=0b001_100).contains(&ty) {
        let order = ty - 0b001_000;
        let mut vals = Vec::with_capacity(blocksize as usize);
        let warm_start = r.bit;
        for _ in 0..order {
            vals.push(r.read_signed(u32::from(subframe_bits))?);
        }
        let warmup_bits = (r.bit - warm_start) as u64;
        let (res, res_trace) = parse_residual(r, blocksize, order)?;
        for (i, &v) in res.iter().enumerate() {
            let n = order as usize + i;
            let pred = match order {
                0 => 0,
                1 => vals[n - 1],
                2 => 2 * vals[n - 1] - vals[n - 2],
                3 => 3 * vals[n - 1] - 3 * vals[n - 2] + vals[n - 3],
                4 => 4 * vals[n - 1] - 6 * vals[n - 2] + 4 * vals[n - 3] - vals[n - 4],
                _ => unreachable!(),
            };
            vals.push(pred + v);
        }
        (
            SubframeKind::Fixed { order },
            warmup_bits,
            0,
            Some(res_trace),
            vals,
        )
    } else if ty >= 0b100_000 {
        let order = ty - 0b100_000 + 1;
        let mut vals = Vec::with_capacity(blocksize as usize);
        let warm_start = r.bit;
        for _ in 0..order {
            vals.push(r.read_signed(u32::from(subframe_bits))?);
        }
        let warmup_bits = (r.bit - warm_start) as u64;
        let coeff_start = r.bit;
        let precision = r.read_bits(4)? as u32 + 1;
        let shift = r.read_signed(5)? as i32;
        let mut coeffs = Vec::with_capacity(order as usize);
        for _ in 0..order {
            coeffs.push(r.read_signed(precision)? as i32);
        }
        let coeff_bits = (r.bit - coeff_start) as u64;
        let (res, res_trace) = parse_residual(r, blocksize, order)?;
        for (i, &v) in res.iter().enumerate() {
            let n = order as usize + i;
            let mut sum = 0i64;
            for (j, &c) in coeffs.iter().enumerate() {
                sum += i64::from(c) * vals[n - 1 - j];
            }
            let pred = if shift >= 0 {
                sum >> shift
            } else {
                sum << (-shift)
            };
            vals.push(pred + v);
        }
        (
            SubframeKind::Lpc {
                order,
                precision,
                shift,
                coeffs,
            },
            warmup_bits,
            coeff_bits,
            Some(res_trace),
            vals,
        )
    } else {
        return Err(Error::new(
            Kind::Unsupported,
            format!("FLAC trace: reserved subframe type {ty:06b}"),
        ));
    };

    let total_bits = (r.bit - sub_start_bit) as u32;
    let samples = samples.into_iter().map(|v| v << wasted_bits).collect();
    Ok(SubframeTrace {
        kind,
        wasted_bits,
        channel_bits,
        subframe_bits,
        blocksize,
        header_bits,
        warmup_bits,
        coeff_bits,
        residual,
        total_bits,
        samples,
    })
}

fn parse_residual(
    r: &mut BitReader,
    blocksize: u32,
    predictor_order: u32,
) -> Result<(Vec<i64>, ResidualTrace)> {
    let start_bit = r.bit;
    let method = r.read_bits(2)? as u8;
    if method > 1 {
        return Err(Error::new(
            Kind::Unsupported,
            "FLAC trace: reserved residual coding method",
        ));
    }
    let param_bits = if method == 0 { 4u32 } else { 5u32 };
    let escape = (1u32 << param_bits) - 1;
    let partition_order = r.read_bits(4)? as u32;
    let n_parts = 1u32 << partition_order;
    let per_part = blocksize >> partition_order;
    if per_part == 0 {
        return Err(Error::malformed(
            "FLAC trace: partition order exceeds block size",
        ));
    }
    let mut out: Vec<i64> = Vec::with_capacity(blocksize as usize);
    let mut parts = Vec::with_capacity(n_parts as usize);
    for p in 0..n_parts {
        let first = if p == 0 { predictor_order } else { 0 };
        if first > per_part {
            return Err(Error::malformed(
                "FLAC trace: predictor order exceeds the first partition",
            ));
        }
        let samples = per_part - first;
        let part_start = r.bit;
        let param = r.read_bits(param_bits)? as u32;
        if param == escape {
            let raw_width = r.read_bits(5)? as u32;
            for _ in 0..samples {
                out.push(r.read_signed(raw_width)?);
            }
            parts.push(ResidualPart {
                samples,
                parameter: param,
                escaped: true,
                raw_width,
                bits: (r.bit - part_start) as u64,
            });
        } else {
            for _ in 0..samples {
                let q = r.read_unary()?;
                let rem = r.read_bits(param)?;
                let u = (q << param) | rem;
                out.push(if u & 1 == 1 {
                    -((u >> 1) as i64) - 1
                } else {
                    (u >> 1) as i64
                });
            }
            parts.push(ResidualPart {
                samples,
                parameter: param,
                escaped: false,
                raw_width: 0,
                bits: (r.bit - part_start) as u64,
            });
        }
    }
    if out.len() != (blocksize - predictor_order) as usize {
        return Err(Error::malformed(
            "FLAC trace: residual sample count disagrees with the block size",
        ));
    }
    let payload_bits: u64 = parts.iter().map(|p| p.bits).sum();
    let header_bits = (r.bit - start_bit) as u32 - payload_bits as u32;
    Ok((
        out,
        ResidualTrace {
            method,
            partition_order,
            parts,
            header_bits,
            payload_bits,
        },
    ))
}

/// Undo FLAC's channel decorrelation and produce frame-major interleaved samples.
fn interleave(assignment: u8, subframes: &[SubframeTrace]) -> Result<Vec<i64>> {
    match assignment {
        0..=7 => {
            let channels = subframes.len();
            let block = subframes.first().map(|s| s.samples.len()).unwrap_or(0);
            let mut out = Vec::with_capacity(block * channels);
            for t in 0..block {
                for s in subframes {
                    out.push(s.samples[t]);
                }
            }
            Ok(out)
        }
        8 => {
            // left / side
            let (l, s) = (&subframes[0].samples, &subframes[1].samples);
            let mut out = Vec::with_capacity(l.len() * 2);
            for t in 0..l.len() {
                out.push(l[t]);
                out.push(l[t] - s[t]);
            }
            Ok(out)
        }
        9 => {
            // side / right
            let (s, r) = (&subframes[0].samples, &subframes[1].samples);
            let mut out = Vec::with_capacity(r.len() * 2);
            for t in 0..r.len() {
                out.push(s[t] + r[t]);
                out.push(r[t]);
            }
            Ok(out)
        }
        10 => {
            // mid / side
            let (m, s) = (&subframes[0].samples, &subframes[1].samples);
            let mut out = Vec::with_capacity(m.len() * 2);
            for t in 0..m.len() {
                let mid = (m[t] << 1) | (s[t] & 1);
                out.push((mid + s[t]) >> 1);
                out.push((mid - s[t]) >> 1);
            }
            Ok(out)
        }
        _ => Err(Error::malformed("FLAC trace: reserved channel assignment")),
    }
}

/// Read the UTF-8-like coded frame/sample number (RFC 9639 §9.1.5).
fn read_utf8_number(r: &mut BitReader) -> Result<u64> {
    let first = r.read_bits(8)? as u8;
    let (len, mut value) = if first & 0x80 == 0 {
        (1usize, u64::from(first))
    } else {
        let mut n = 0usize;
        let mut mask = 0x80u8;
        while first & mask != 0 {
            n += 1;
            mask >>= 1;
        }
        if !(2..=7).contains(&n) {
            return Err(Error::malformed("FLAC trace: invalid coded-number prefix"));
        }
        let bits_in_first = 7 - n;
        let payload_mask = ((1u16 << bits_in_first) - 1) as u8;
        (n, u64::from(first & payload_mask))
    };
    for _ in 1..len {
        let b = r.read_bits(8)? as u8;
        if b & 0xC0 != 0x80 {
            return Err(Error::malformed(
                "FLAC trace: invalid coded-number continuation",
            ));
        }
        value = (value << 6) | u64::from(b & 0x3F);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::flac::{B1_LEVEL_CONTROLS, B1_LEVEL_PRIMARY, b1_flac_artifact};

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn check(name: &str, samples: &[i32], channels: u8, rate: u32, level: u32) {
        let art = b1_flac_artifact(samples, channels, rate, level).unwrap();
        let t = trace_flac(&art.bytes).unwrap_or_else(|e| panic!("{name}: trace failed: {e}"));
        assert!(t.marker_ok && t.streaminfo.bits_per_sample == 32, "{name}");
        assert_eq!(t.decoded_channels, channels, "{name}");
        let expect: Vec<i64> = samples.iter().map(|&v| i64::from(v)).collect();
        assert_eq!(t.decoded, expect, "{name}: reconstruction mismatch");
        for f in &t.frames {
            assert!(f.crc8_ok, "{name}: CRC-8");
            assert!(f.crc16_ok, "{name}: CRC-16");
            assert!(!f.subframes.is_empty(), "{name}: subframes");
        }
        let mut total = t.metadata_bytes;
        for f in &t.frames {
            total += f.frame_bytes;
        }
        assert_eq!(total, art.bytes.len(), "{name}: total bytes");
    }

    #[test]
    fn traces_and_reconstructs_across_levels_and_shapes() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let quiet: Vec<i32> = (0..4096).map(|i| ((i * 7) % 401) - 200).collect();
        let silence = vec![0i32; 4096];
        let noise: Vec<i32> = (0..4096)
            .map(|_| (splitmix(&mut state) as i32) >> 8)
            .collect();
        for (name, src) in [("quiet", &quiet), ("silence", &silence), ("noise", &noise)] {
            for level in [0u32, 5, 8] {
                check(name, src, 1, 16_000, level);
            }
        }
    }

    #[test]
    fn traces_stereo_including_side_channels() {
        let mut state = 0xfeed_face_dead_beefu64;
        let n = 4096;
        let mut stereo = Vec::with_capacity(n * 2);
        for _ in 0..n {
            let a = ((splitmix(&mut state) as i32) >> 10) % 3000;
            stereo.push(a);
            stereo.push(a + ((splitmix(&mut state) as i32) >> 15) % 50);
        }
        check("stereo-correlated", &stereo, 2, 44_100, B1_LEVEL_PRIMARY);
    }

    #[test]
    fn rejects_corruption() {
        let src: Vec<i32> = (0..2048).map(|i| (i * 13) % 997 - 500).collect();
        let art = b1_flac_artifact(&src, 1, 16_000, B1_LEVEL_PRIMARY).unwrap();
        // Flip a bit inside the first frame body: CRC-16 must fail.
        let mut bad = art.bytes.clone();
        let at = art.bytes.len() / 2;
        bad[at] ^= 0x10;
        if let Ok(t) = trace_flac(&bad) {
            assert!(
                t.frames.iter().any(|f| !f.crc16_ok || !f.crc8_ok),
                "corruption must be detected by a CRC"
            );
        }
    }

    #[test]
    fn level_controls_trace() {
        let src: Vec<i32> = (0..16384).map(|i| ((i * 37) % 2001) - 1000).collect();
        for level in [B1_LEVEL_CONTROLS[0], B1_LEVEL_PRIMARY, B1_LEVEL_CONTROLS[1]] {
            check("controls", &src, 1, 48_000, level);
        }
    }
}
