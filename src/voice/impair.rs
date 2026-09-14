//! Deterministic packet impairment and playout scheduling (7C.11).
//!
//! There is no networking stack here, and there is deliberately none: the
//! constitution forbids RTP/ICE/STUN/TURN/peer discovery, and a codec's
//! behaviour under loss must be *replayable exactly* rather than dependent on a
//! live network. Canonical voice packets are driven through a seeded model and
//! every result is reproducible from the seed.
//!
//! ```text
//! capture  : frame k is captured at k · frame_duration
//! send     : packet p is sent at (p+1)·nf · frame_duration + encode_time
//! arrival  : send + nominal network delay + jitter offset
//! playout  : frame k is due at (p+1)·nf · frame_duration·drift + nominal + depth
//!
//! frame k decodes  ⇔  its packet arrived, was not lost, and
//!                    arrival ≤ playout_us(k)
//! ```
//!
//! The last line is the *whole* loss/jitter/reordering model. A packet that
//! arrives after the deadline is exactly as lost as one that never arrived, a
//! reordered packet is absorbed whenever the depth allows, and a late packet
//! repairs the frame retroactively precisely when it beats the deadline. That
//! is what makes the required jitter-buffer depth a measurable quantity rather
//! than an assumption.

use crate::voice::splitmix64;

/// One packet as the transport hands it to the impairment model.
#[derive(Debug, Clone)]
pub struct TransportPacket {
    /// Sequence number.
    pub seq: u32,
    /// The encoded `VoicePacket`.
    pub payload: Vec<u8>,
}

/// The impairment configuration. Every field is frozen into the receipt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Impairment {
    /// PRNG seed. The same seed replays the same channel exactly.
    pub seed: u64,
    /// i.i.d. loss probability, per mille.
    pub loss_per_mille: u32,
    /// Mean burst length in packets; ≤ 1 selects the i.i.d. model.
    pub burst_mean: u32,
    /// Duplication probability, per mille.
    pub duplicate_per_mille: u32,
    /// Adjacent reordering probability, per mille.
    pub reorder_per_mille: u32,
    /// Uniform jitter half-width, microseconds.
    pub jitter_us: i64,
    /// Use the two-mode (heavier-tailed) jitter distribution.
    pub jitter_two_mode: bool,
    /// Playout clock offset in parts per million. **Positive means the playout
    /// clock runs fast**, so deadlines arrive early and the buffer is consumed.
    pub drift_ppm: i32,
}

impl Default for Impairment {
    fn default() -> Self {
        Impairment {
            seed: 0x7C5E_ED01,
            loss_per_mille: 0,
            burst_mean: 1,
            duplicate_per_mille: 0,
            reorder_per_mille: 0,
            jitter_us: 0,
            jitter_two_mode: false,
            drift_ppm: 0,
        }
    }
}

/// A packet's fate on the channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Arrival {
    /// Sequence number.
    pub seq: u32,
    /// The payload, when the packet survived.
    pub payload: Vec<u8>,
    /// Arrival time in microseconds.
    pub arrival_us: i64,
    /// True when this copy was dropped by the loss model.
    pub lost: bool,
    /// True when this is an additional (duplicate) copy.
    pub duplicate: bool,
}

/// The latency contributions, kept separate exactly as §5.1 of the
/// constitution requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LatencyBreakdown {
    /// Frame accumulation in microseconds.
    pub frame_accumulation_us: i64,
    /// Encoder analysis/lookahead beyond the frame, in microseconds.
    pub lookahead_us: i64,
    /// Encode time in microseconds (p50 or p99 — the label says which).
    pub encode_us: i64,
    /// Packetisation in microseconds.
    pub packetisation_us: i64,
    /// Simulated one-way network delay, a test parameter.
    pub network_us: i64,
    /// Jitter-buffer depth actually required.
    pub jitter_buffer_us: i64,
    /// Decode time in microseconds.
    pub decode_us: i64,
    /// Playout scheduler wake error.
    pub playout_scheduling_us: i64,
}

impl LatencyBreakdown {
    /// The one-way total, which is what a call experiences.
    pub fn one_way_us(&self) -> i64 {
        self.frame_accumulation_us
            + self.lookahead_us
            + self.encode_us
            + self.packetisation_us
            + self.network_us
            + self.jitter_buffer_us
            + self.decode_us
            + self.playout_scheduling_us
    }
}

/// A uniform integer in `[0, n)` from the impairment PRNG.
fn rand_below(state: &mut u64, n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    splitmix64(state) % n
}

/// A per-mille Bernoulli trial.
fn trial(state: &mut u64, per_mille: u32) -> bool {
    if per_mille == 0 {
        return false;
    }
    if per_mille >= 1000 {
        return true;
    }
    rand_below(state, 1000) < u64::from(per_mille)
}

/// The jitter offset applied to one packet's arrival time.
fn jitter_offset(state: &mut u64, imp: &Impairment) -> i64 {
    if imp.jitter_us <= 0 {
        return 0;
    }
    let j = imp.jitter_us;
    let span = if imp.jitter_two_mode {
        // 80 % of packets inside a third of the window, 20 % across the whole
        // window: a heavier tail than uniform, which is what a real access
        // network produces.
        if rand_below(state, 100) < 80 {
            j / 3
        } else {
            j
        }
    } else {
        j
    };
    if span <= 0 {
        return 0;
    }
    let u = rand_below(state, (2 * span + 1) as u64) as i64;
    u - span
}

/// The loss process: i.i.d. for `burst_mean ≤ 1`, Gilbert–Elliott otherwise.
struct LossModel {
    bad: bool,
    p: f64,
    q: f64,
    burst: bool,
    per_mille: u32,
}

impl LossModel {
    fn new(imp: &Impairment) -> LossModel {
        let loss_fraction = f64::from(imp.loss_per_mille) / 1000.0;
        let burst = imp.burst_mean > 1;
        let q = if burst {
            1.0 / f64::from(imp.burst_mean)
        } else {
            1.0
        };
        let p = if loss_fraction >= 1.0 {
            1.0
        } else {
            loss_fraction * q / (1.0 - loss_fraction)
        };
        LossModel {
            bad: false,
            p,
            q,
            burst,
            per_mille: imp.loss_per_mille,
        }
    }

    fn draw(&mut self, state: &mut u64) -> bool {
        if !self.burst {
            return trial(state, self.per_mille);
        }
        let r = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
        if self.bad {
            if r < self.q {
                self.bad = false;
            }
        } else if r < self.p {
            self.bad = true;
        }
        self.bad
    }
}

/// Drive packets through the channel and return one [`Arrival`] per *copy*.
///
/// `send_us(p)` is the time packet `p` leaves the encoder: the end of its last
/// frame plus the measured encode time for that packet. Duplicates are extra
/// copies of the same sequence number that face an independent loss trial and
/// an independent jitter draw, which is what sending a packet twice on a real
/// channel actually buys.
pub fn impair(
    packets: &[TransportPacket],
    imp: &Impairment,
    frame_duration_us: i64,
    frames_per_packet: usize,
    encode_us: &[i64],
    nominal_delay_us: i64,
) -> Vec<Arrival> {
    let mut state = imp.seed | 1;
    let mut model = LossModel::new(imp);
    let n = packets.len();
    // Arrival offsets are generated first so adjacent transposition can swap
    // them; a packet that takes a later packet's offset really does arrive
    // after it.
    let mut offsets: Vec<i64> = (0..n).map(|_| jitter_offset(&mut state, imp)).collect();
    for i in 0..n.saturating_sub(1) {
        if trial(&mut state, imp.reorder_per_mille) {
            offsets.swap(i, i + 1);
        }
    }
    let mut out: Vec<Arrival> = Vec::with_capacity(n);
    for (i, pkt) in packets.iter().enumerate() {
        let send = (i as i64 + 1) * frames_per_packet as i64 * frame_duration_us
            + encode_us.get(i).copied().unwrap_or(0)
            + nominal_delay_us;
        let arrival_us = send + offsets[i];
        let lost = model.draw(&mut state);
        if trial(&mut state, imp.duplicate_per_mille) {
            // A second, independent copy: its own loss trial and its own jitter.
            let dup_arrival = send + jitter_offset(&mut state, imp);
            let dup_lost = model.draw(&mut state);
            out.push(Arrival {
                seq: pkt.seq,
                payload: pkt.payload.clone(),
                arrival_us: dup_arrival,
                lost: dup_lost,
                duplicate: true,
            });
        }
        out.push(Arrival {
            seq: pkt.seq,
            payload: pkt.payload.clone(),
            arrival_us,
            lost,
            duplicate: false,
        });
    }
    out
}

/// One frame's outcome after scheduling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameEvent {
    /// Frame index.
    pub frame: u64,
    /// True when the frame had to be concealed.
    pub concealed: bool,
    /// True when the frame's packet arrived, but after the playout deadline.
    pub late: bool,
    /// Arrival time of the frame's packet, when it arrived.
    pub arrival_us: Option<i64>,
    /// The frame's playout deadline.
    pub deadline_us: i64,
}

/// Channel and scheduling statistics for one run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimStats {
    /// Packets offered (the primary copies only).
    pub packets: u64,
    /// Primary packets dropped by the loss model.
    pub packets_lost: u64,
    /// Packets that arrived after the deadline.
    pub packets_late: u64,
    /// Additional copies that were discarded.
    pub duplicates_discarded: u64,
    /// Frames scheduled.
    pub frames: u64,
    /// Frames concealed.
    pub frames_concealed: u64,
    /// Concealed frames whose packet had arrived but too late.
    pub frames_late: u64,
    /// The worst lateness observed, in microseconds.
    pub max_lateness_us: i64,
    /// The worst headroom observed, in microseconds (buffer depth used).
    pub max_headroom_us: i64,
}

/// Schedule playout for `frames` frames given channel arrivals.
///
/// The packet carrying frame `k` is `k / frames_per_packet`. Arrivals are
/// collapsed by sequence number taking the **earliest delivered copy**, which is
/// the decoder-side rule the constitution fixes.
pub fn schedule(
    frames: u64,
    arrivals: &[Arrival],
    frames_per_packet: usize,
    frame_duration_us: i64,
    base_latency_us: i64,
    jitter_depth_us: i64,
    drift_ppm: i32,
) -> (Vec<FrameEvent>, SimStats) {
    let per_packet = frames_per_packet.max(1) as u64;
    let packets = frames.div_ceil(per_packet) as usize;
    let mut earliest: Vec<Option<i64>> = vec![None; packets];
    let mut primary_lost = vec![false; packets];
    let mut stats = SimStats::default();
    for a in arrivals {
        let Some(slot) = earliest.get_mut(a.seq as usize) else {
            continue;
        };
        if !a.duplicate {
            stats.packets += 1;
            if a.lost {
                primary_lost[a.seq as usize] = true;
                continue;
            }
        } else {
            stats.duplicates_discarded += 1;
            if a.lost {
                continue;
            }
        }
        *slot = Some(match *slot {
            Some(t) => t.min(a.arrival_us),
            None => a.arrival_us,
        });
    }
    stats.packets_lost = primary_lost.iter().filter(|&&v| v).count() as u64;

    let drift = 1.0 - f64::from(drift_ppm) * 1.0e-6;
    let mut events = Vec::with_capacity(frames as usize);
    for k in 0..frames {
        let p = (k / per_packet) as usize;
        // A frame is due once its whole packet has been captured, the network
        // has delivered it, and the buffer has waited out its depth. Every
        // frame of a packet shares that deadline, which is why the 2×10 ms
        // shape accumulates 20 ms rather than 10+10.
        let packet_done =
            ((((p as u64 + 1) * per_packet) as f64) * frame_duration_us as f64 * drift).round()
                as i64;
        let deadline = packet_done + base_latency_us + jitter_depth_us;
        let arrival = earliest.get(p).copied().flatten();
        let concealed = match arrival {
            None => true,
            Some(t) => t > deadline,
        };
        let late = concealed && arrival.is_some();
        if concealed {
            stats.frames_concealed += 1;
            if let Some(t) = arrival {
                stats.frames_late += 1;
                stats.max_lateness_us = stats.max_lateness_us.max(t - deadline);
            }
        } else if let Some(t) = arrival {
            stats.max_headroom_us = stats.max_headroom_us.max(deadline - t);
        }
        stats.frames += 1;
        events.push(FrameEvent {
            frame: k,
            concealed,
            late,
            arrival_us: arrival,
            deadline_us: deadline,
        });
    }
    // A packet that arrived but after the deadline of its last frame is counted
    // as late once, not once per frame.
    stats.packets_late = events
        .iter()
        .filter(|e| e.late)
        .map(|e| e.frame / per_packet)
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u64;
    (events, stats)
}

/// The smallest jitter-buffer depth (in microseconds, on top of the base
/// latency) at which **no** frame is concealed, for one channel realisation.
///
/// This is the "required depth" the constitution asks for: it is measured from
/// the arrivals, not assumed.
pub fn required_depth_us(
    arrivals: &[Arrival],
    frames_per_packet: usize,
    frame_duration_us: i64,
    base_latency_us: i64,
    drift_ppm: i32,
) -> i64 {
    let per_packet = frames_per_packet.max(1) as i64;
    let drift = 1.0 - f64::from(drift_ppm) * 1.0e-6;
    let mut worst = 0i64;
    for a in arrivals {
        if a.lost {
            continue;
        }
        // Every frame of a packet is due at the packet's completion time, so
        // the depth a packet needs is its arrival minus that instant, minus the
        // nominal network delay the base latency already accounts for.
        let packet_done =
            (((i64::from(a.seq) + 1) * per_packet) as f64 * frame_duration_us as f64 * drift)
                .round() as i64;
        worst = worst.max(a.arrival_us - packet_done - base_latency_us);
    }
    worst.max(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nominal one-way network delay used by the tests, matching the
    /// constitution's 10 ms test parameter.
    const NOMINAL: i64 = 10_000;

    fn packets(n: usize) -> Vec<TransportPacket> {
        (0..n)
            .map(|i| TransportPacket {
                seq: i as u32,
                payload: vec![i as u8; 8],
            })
            .collect()
    }

    fn arrivals(p: &[TransportPacket], imp: &Impairment, dur: i64, fpp: usize) -> Vec<Arrival> {
        let enc = vec![0i64; p.len()];
        impair(p, imp, dur, fpp, &enc, NOMINAL)
    }

    #[test]
    fn a_clean_channel_loses_nothing_and_needs_no_depth() {
        let p = packets(50);
        let imp = Impairment::default();
        let arr = arrivals(&p, &imp, 20_000, 1);
        let (events, stats) = schedule(50, &arr, 1, 20_000, NOMINAL, 0, 0);
        assert_eq!(stats.frames_concealed, 0);
        assert!(events.iter().all(|e| !e.concealed));
        assert_eq!(required_depth_us(&arr, 1, 20_000, NOMINAL, 0), 0);
    }

    #[test]
    fn total_loss_conceals_every_frame() {
        let p = packets(20);
        let imp = Impairment {
            loss_per_mille: 1000,
            ..Impairment::default()
        };
        let arr = arrivals(&p, &imp, 20_000, 1);
        let (events, stats) = schedule(20, &arr, 1, 20_000, NOMINAL, 0, 0);
        assert_eq!(stats.frames_concealed, 20);
        assert!(events.iter().all(|e| e.concealed && !e.late));
    }

    #[test]
    fn the_seed_replays_the_channel_exactly() {
        let p = packets(200);
        let imp = Impairment {
            seed: 0xabc,
            loss_per_mille: 50,
            burst_mean: 3,
            duplicate_per_mille: 20,
            reorder_per_mille: 20,
            jitter_us: 5_000,
            jitter_two_mode: true,
            drift_ppm: 20,
        };
        assert_eq!(arrivals(&p, &imp, 20_000, 1), arrivals(&p, &imp, 20_000, 1));
    }

    #[test]
    fn a_different_seed_gives_a_different_channel() {
        let p = packets(200);
        let a = arrivals(
            &p,
            &Impairment {
                seed: 1,
                loss_per_mille: 100,
                ..Impairment::default()
            },
            20_000,
            1,
        );
        let b = arrivals(
            &p,
            &Impairment {
                seed: 2,
                loss_per_mille: 100,
                ..Impairment::default()
            },
            20_000,
            1,
        );
        assert_ne!(
            a.iter().map(|x| x.lost).collect::<Vec<_>>(),
            b.iter().map(|x| x.lost).collect::<Vec<_>>()
        );
    }

    #[test]
    fn burst_loss_correlates_drops_rather_than_spreading_them() {
        let p = packets(400);
        let imp = Impairment {
            seed: 7,
            loss_per_mille: 100,
            burst_mean: 8,
            ..Impairment::default()
        };
        let arr = arrivals(&p, &imp, 20_000, 1);
        let lost: Vec<bool> = arr
            .iter()
            .filter(|a| !a.duplicate)
            .map(|a| a.lost)
            .collect();
        let mut runs = 0usize;
        let mut prev = false;
        for &l in &lost {
            if l && !prev {
                runs += 1;
            }
            prev = l;
        }
        let total: usize = lost.iter().filter(|&&l| l).count();
        assert!(total > 10, "the test needs a meaningful number of drops");
        assert!(
            runs < total,
            "a bursty channel must produce runs, not isolated drops"
        );
    }

    #[test]
    fn jitter_requires_depth_and_depth_removes_the_loss() {
        let p = packets(200);
        let imp = Impairment {
            seed: 42,
            jitter_us: 5_000,
            ..Impairment::default()
        };
        let arr = arrivals(&p, &imp, 20_000, 1);
        let (_, shallow) = schedule(200, &arr, 1, 20_000, NOMINAL, 0, 0);
        assert!(
            shallow.frames_concealed > 0,
            "0 µs of buffer must not absorb ±5 ms jitter"
        );
        let depth = required_depth_us(&arr, 1, 20_000, NOMINAL, 0);
        assert!(depth > 0);
        let (_, deep) = schedule(200, &arr, 1, 20_000, NOMINAL, depth, 0);
        assert_eq!(
            deep.frames_concealed, 0,
            "the measured required depth must be sufficient"
        );
    }

    #[test]
    fn two_mode_jitter_has_a_heavier_tail_than_uniform() {
        let p = packets(2000);
        let base = Impairment {
            seed: 3,
            jitter_us: 10_000,
            ..Impairment::default()
        };
        let two = Impairment {
            jitter_two_mode: true,
            ..base
        };
        let a = arrivals(&p, &base, 20_000, 1);
        let b = arrivals(&p, &two, 20_000, 1);
        let spread = |v: &[Arrival]| -> i64 {
            let mut xs: Vec<i64> = v.iter().map(|a| a.arrival_us).collect();
            xs.sort_unstable();
            xs[xs.len() - 1] - xs[0]
        };
        assert!(spread(&a) > 0 && spread(&b) > 0);
        assert_ne!(spread(&a), spread(&b));
    }

    #[test]
    fn latency_breakdown_adds_up() {
        let b = LatencyBreakdown {
            frame_accumulation_us: 20_000,
            lookahead_us: 0,
            encode_us: 3_000,
            packetisation_us: 200,
            network_us: 10_000,
            jitter_buffer_us: 2_500,
            decode_us: 2_000,
            playout_scheduling_us: 100,
        };
        assert_eq!(b.one_way_us(), 37_800);
    }

    #[test]
    fn duplicates_are_discarded_not_double_played() {
        let p = packets(5);
        let imp = Impairment {
            seed: 9,
            duplicate_per_mille: 1000,
            ..Impairment::default()
        };
        let arr = arrivals(&p, &imp, 20_000, 1);
        let (events, stats) = schedule(5, &arr, 1, 20_000, NOMINAL, 0, 0);
        assert_eq!(stats.duplicates_discarded, 5);
        assert_eq!(events.len(), 5);
        assert!(events.iter().all(|e| !e.concealed));
    }

    #[test]
    fn drift_moves_the_deadline_and_can_starve_the_buffer() {
        let p = packets(500);
        let imp = Impairment::default();
        let arr = arrivals(&p, &imp, 20_000, 1);
        let (_, clean) = schedule(500, &arr, 1, 20_000, NOMINAL, 0, 0);
        assert_eq!(clean.frames_concealed, 0);
        // A fast playout clock (+1000 ppm) with no buffer depth eventually
        // overtakes the sender: after 10 s the deadlines are one frame early.
        let (_, drifted) = schedule(500, &arr, 1, 20_000, NOMINAL, 0, 1000);
        assert!(drifted.frames_concealed > 0);
    }

    #[test]
    fn multi_frame_packets_share_one_deadline() {
        let p = packets(100);
        let imp = Impairment::default();
        let arr = arrivals(&p, &imp, 10_000, 2);
        let (events, stats) = schedule(200, &arr, 2, 10_000, NOMINAL, 0, 0);
        assert_eq!(stats.frames_concealed, 0);
        // Frames 2p and 2p+1 carry the same deadline.
        for pair in events.as_chunks::<2>().0 {
            assert_eq!(pair[0].deadline_us, pair[1].deadline_us);
        }
    }
}
