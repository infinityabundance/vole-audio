//! Bark-scale critical-band layout and a Schroeder-style simultaneous-masking
//! model used to derive per-band allowed-noise powers.
//!
//! The model is deliberately conservative and fully deterministic: given the
//! same coefficient frame it produces the same thresholds on every host. All
//! powers are expressed in the same units as the transform coefficients, so a
//! band's step size is directly `sqrt(12 * threshold / bins)`.

/// Standard critical-band edges (Hz).
const BARK_EDGES_HZ: [f64; 26] = [
    0.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
    2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
    15500.0, 1.0e9,
];

/// Critical-band layout for one transform length and sample rate.
#[derive(Debug, Clone)]
pub struct Bands {
    /// Band index of each transform coefficient.
    pub of_bin: Vec<u8>,
    /// Number of coefficients in each band.
    pub bins_per_band: Vec<u32>,
    /// Band centre in Bark.
    pub bark: Vec<f64>,
    /// Absolute-threshold-of-hearing power per band.
    pub ath_power: Vec<f64>,
    /// Number of bands.
    pub band_count: usize,
}

impl Bands {
    /// Build the layout; `full_scale_coeff` calibrates the absolute threshold
    /// from dB SPL to transform-coefficient units (see
    /// [`crate::lossy::transform`]).
    pub fn new(sample_rate_hz: u32, n: usize, full_scale_coeff: f64) -> Bands {
        let bin_hz = sample_rate_hz as f64 / (2 * n) as f64;
        let mut of_bin = vec![0u8; n];
        let mut band_count = 0usize;
        let edges = &BARK_EDGES_HZ;
        for (k, slot) in of_bin.iter_mut().enumerate() {
            let f = (k as f64 + 0.5) * bin_hz;
            let mut b = 0usize;
            while b + 2 < edges.len() && f >= edges[b + 1] {
                b += 1;
            }
            *slot = b.min(255) as u8;
            band_count = band_count.max(usize::from(*slot) + 1);
        }
        let mut bins_per_band = vec![0u32; band_count];
        for &b in &of_bin {
            bins_per_band[b as usize] += 1;
        }
        let mut bark = vec![0.0f64; band_count];
        let mut ath_power = vec![0.0f64; band_count];
        for b in 0..band_count {
            let centre_hz = band_centre_hz(&of_bin, b, bin_hz);
            bark[b] = 26.81 * centre_hz / (1960.0 + centre_hz) - 0.53;
            let f_khz = (centre_hz / 1000.0).max(0.02);
            let ath_db = 3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
                + 1.0e-3 * f_khz.powi(4);
            let ath_amp = 10f64.powf((ath_db - 96.0) / 20.0) * full_scale_coeff;
            ath_power[b] = ath_amp * ath_amp * f64::from(bins_per_band[b].max(1));
        }
        Bands {
            of_bin,
            bins_per_band,
            bark,
            ath_power,
            band_count,
        }
    }
}

fn band_centre_hz(of_bin: &[u8], b: usize, bin_hz: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for (k, &bb) in of_bin.iter().enumerate() {
        if usize::from(bb) == b {
            sum += (k as f64 + 0.5) * bin_hz;
            count += 1.0;
        }
    }
    if count == 0.0 { 0.0 } else { sum / count }
}

/// Per-band simultaneous-masking thresholds (total noise power allowed).
#[derive(Debug, Clone)]
pub struct Masking {
    /// Allowed noise power per band.
    pub threshold_power: Vec<f64>,
}

impl Masking {
    /// Analyse one coefficient frame.
    pub fn analyse(coeffs: &[f64], bands: &Bands) -> Masking {
        let bc = bands.band_count;
        let mut energy = vec![0.0f64; bc];
        let mut abs_sum = vec![0.0f64; bc];
        let mut log_sum = vec![0.0f64; bc];
        for (k, &x) in coeffs.iter().enumerate() {
            let b = bands.of_bin[k] as usize;
            energy[b] += x * x;
            let a = x.abs();
            abs_sum[b] += a;
            log_sum[b] += (a + 1e-12).ln();
        }
        let mut masker_db = vec![0.0f64; bc];
        for (b, slot) in masker_db.iter_mut().enumerate() {
            let n_b = f64::from(bands.bins_per_band[b].max(1));
            let e = energy[b].max(1e-30);
            let mean = abs_sum[b] / n_b;
            let geo = (log_sum[b] / n_b).exp();
            let tonality = if mean > 1e-30 {
                (geo / mean).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let offset = tonality * (14.5 + bands.bark[b]) + (1.0 - tonality) * 5.5;
            *slot = 10.0 * (e / n_b).max(1e-30).log10() - offset;
        }
        let mut spread_db = vec![-300.0f64; bc];
        for (i, slot) in spread_db.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (j, &mdb) in masker_db.iter().enumerate() {
                let z = bands.bark[j] - bands.bark[i] + 0.474;
                let s = 15.81 + 7.5 * z - 17.5 * (1.0 + z * z).sqrt();
                acc += 10f64.powf((mdb + s) / 10.0);
            }
            *slot = 10.0 * acc.max(1e-30).log10();
        }
        let mut threshold_power = vec![0.0f64; bc];
        for b in 0..bc {
            let n_b = f64::from(bands.bins_per_band[b].max(1));
            let spread = n_b * 10f64.powf(spread_db[b] / 10.0);
            let own = energy[b].max(1e-30);
            threshold_power[b] = (spread.min(own * 4.0) + bands.ath_power[b]).max(1e-30);
        }
        Masking { threshold_power }
    }
}
