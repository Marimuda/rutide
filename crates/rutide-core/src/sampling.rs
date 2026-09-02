//! Sampling-quality diagnostics aligned with the colored-confidence spectrum.

use crate::AnalysisError;

/// The nine residual-noise frequency bands used by Python `UTide`, in cycles/hour.
pub const COLORED_NOISE_FREQUENCY_BANDS_CPH: [[f64; 2]; 9] = [
    [0.000_10, 0.004_17],
    [0.031_92, 0.048_59],
    [0.072_18, 0.088_84],
    [0.112_43, 0.129_10],
    [0.152_69, 0.169_36],
    [0.192_95, 0.209_61],
    [0.233_20, 0.251_00],
    [0.260_00, 0.290_00],
    [0.300_00, 0.500_00],
];

/// Residual-spectrum route selected from the source timestamp grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidualSpectrumMethod {
    /// Periodic-Hann, one-sided fast Fourier transform on an equidistant grid.
    Fft,
    /// Phase-shifted Lomb–Scargle projection on retained irregular timestamps.
    LombScargle,
}

impl ResidualSpectrumMethod {
    /// Stable machine-readable method name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fft => "fft",
            Self::LombScargle => "lomb-scargle",
        }
    }
}

/// Per-series temporal coverage relevant to fitting and colored confidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingDiagnostics {
    /// Number of finite scalar or joint-vector observations.
    pub observation_count: usize,
    /// Difference between the final and first retained observation, in days.
    pub record_span_days: f64,
    /// Mean retained interval derived from span/count, in hours.
    pub mean_sample_interval_hours: f64,
    /// Largest interval between adjacent retained observations, in hours.
    pub largest_gap_hours: f64,
    /// FFT or Lomb–Scargle route selected for a colored residual spectrum.
    pub residual_spectrum_method: ResidualSpectrumMethod,
    /// Even sample count used by the residual spectrum.
    pub residual_spectrum_time_count: usize,
    /// Frequency-bin count in each [`COLORED_NOISE_FREQUENCY_BANDS_CPH`] band.
    pub spectral_band_bin_count: [usize; 9],
    /// Per-band bins left after excluding the nearest fitted-constituent bins.
    pub spectral_band_usable_bin_count: [usize; 9],
}

/// Reusable source-time plan for per-series sampling diagnostics.
///
/// Equidistant spectrum frequencies are prepared once and shared by every
/// spatial series. Truly irregular series derive their Lomb–Scargle grid from
/// the finite observation mask, matching the confidence implementation.
#[derive(Clone, Debug)]
pub struct SamplingDiagnosticsPlan {
    time_days: Vec<f64>,
    regular_frequencies_cph: Option<Vec<f64>>,
}

impl SamplingDiagnosticsPlan {
    /// Validate a source MJD axis and prepare its reusable spectral grid.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for fewer than two, non-finite, duplicate, or
    /// decreasing timestamps.
    pub fn prepare(modified_julian_days: &[f64]) -> Result<Self, AnalysisError> {
        if modified_julian_days.is_empty() {
            return Err(AnalysisError::EmptyTime);
        }
        for (index, value) in modified_julian_days.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(AnalysisError::NonFiniteTime { index });
            }
            if index > 0 && value <= modified_julian_days[index - 1] {
                return Err(AnalysisError::NonIncreasingTime { index });
            }
        }
        if modified_julian_days.len() < 2 {
            return Err(AnalysisError::InsufficientSamplingObservations { actual: 1 });
        }
        let regular_frequencies_cph =
            equidistant_sample_interval_hours(modified_julian_days).map(|sample_interval_hours| {
                regular_spectrum_frequencies(modified_julian_days.len(), sample_interval_hours)
            });
        Ok(Self {
            time_days: modified_julian_days.to_vec(),
            regular_frequencies_cph,
        })
    }

    /// Diagnose one finite-observation mask and fitted frequency set.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::SamplingMaskShape`] for a mask with the wrong
    /// length, [`AnalysisError::InsufficientSamplingObservations`] when fewer
    /// than two values remain, or [`AnalysisError::InvalidSamplingFrequency`]
    /// for a non-finite or negative fitted frequency.
    pub fn diagnose(
        &self,
        finite_observations: &[bool],
        constituent_frequencies_cph: &[f64],
    ) -> Result<SamplingDiagnostics, AnalysisError> {
        if finite_observations.len() != self.time_days.len() {
            return Err(AnalysisError::SamplingMaskShape {
                actual: finite_observations.len(),
                expected: self.time_days.len(),
            });
        }
        self.diagnose_with(constituent_frequencies_cph, |time| {
            finite_observations[time]
        })
    }

    /// Diagnose observations selected by a position predicate without allocating a mask.
    ///
    /// This is the efficient whole-field path when finiteness already lives in a
    /// time-major scalar or vector array.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::InsufficientSamplingObservations`] when fewer
    /// than two values remain, or [`AnalysisError::InvalidSamplingFrequency`]
    /// for a non-finite or negative fitted frequency.
    pub fn diagnose_with<F>(
        &self,
        constituent_frequencies_cph: &[f64],
        mut observation_is_finite: F,
    ) -> Result<SamplingDiagnostics, AnalysisError>
    where
        F: FnMut(usize) -> bool,
    {
        for (index, frequency) in constituent_frequencies_cph.iter().copied().enumerate() {
            if !frequency.is_finite() || frequency < 0.0 {
                return Err(AnalysisError::InvalidSamplingFrequency { index });
            }
        }

        let mut observation_count = 0;
        let mut first_time = 0.0;
        let mut previous_time = 0.0;
        let mut last_time = 0.0;
        let mut largest_gap_hours = 0.0_f64;
        let mut irregular_times = self
            .regular_frequencies_cph
            .is_none()
            .then(|| Vec::with_capacity(self.time_days.len()));
        for (index, time) in self.time_days.iter().copied().enumerate() {
            if !observation_is_finite(index) {
                continue;
            }
            if observation_count == 0 {
                first_time = time;
            } else {
                largest_gap_hours = largest_gap_hours.max(24.0 * (time - previous_time));
            }
            previous_time = time;
            last_time = time;
            observation_count += 1;
            if let Some(times) = &mut irregular_times {
                times.push(time);
            }
        }
        if observation_count < 2 {
            return Err(AnalysisError::InsufficientSamplingObservations {
                actual: observation_count,
            });
        }
        let record_span_days = last_time - first_time;
        let mean_sample_interval_hours =
            24.0 * record_span_days / usize_to_f64(observation_count - 1);

        let irregular_frequencies;
        let (residual_spectrum_method, frequencies_cph, residual_spectrum_time_count) =
            if let Some(frequencies) = &self.regular_frequencies_cph {
                (
                    ResidualSpectrumMethod::Fft,
                    frequencies.as_slice(),
                    self.time_days.len() - self.time_days.len() % 2,
                )
            } else {
                let retained_times = irregular_times
                    .as_ref()
                    .ok_or(AnalysisError::InsufficientSamplingObservations { actual: 0 })?;
                let sample_count = retained_times.len() - retained_times.len() % 2;
                let time_hours = retained_times[..sample_count]
                    .iter()
                    .map(|time| time * 24.0)
                    .collect::<Vec<_>>();
                irregular_frequencies = lomb_frequencies(&time_hours);
                (
                    ResidualSpectrumMethod::LombScargle,
                    irregular_frequencies.as_slice(),
                    sample_count,
                )
            };
        let (spectral_band_bin_count, spectral_band_usable_bin_count) =
            spectral_band_counts(frequencies_cph, constituent_frequencies_cph);
        Ok(SamplingDiagnostics {
            observation_count,
            record_span_days,
            mean_sample_interval_hours,
            largest_gap_hours,
            residual_spectrum_method,
            residual_spectrum_time_count,
            spectral_band_bin_count,
            spectral_band_usable_bin_count,
        })
    }
}

fn regular_spectrum_frequencies(source_time_count: usize, sample_interval_hours: f64) -> Vec<f64> {
    let sample_count = source_time_count - source_time_count % 2;
    let frequency_count = sample_count / 2 + 1;
    let frequency_spacing = 1.0 / (usize_to_f64(sample_count) * sample_interval_hours);
    (0..frequency_count)
        .map(|index| usize_to_f64(index) * frequency_spacing)
        .collect()
}

pub(crate) fn equidistant_sample_interval_hours(time_days: &[f64]) -> Option<f64> {
    let mut unique_deltas = time_days
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    unique_deltas.sort_by(f64::total_cmp);
    unique_deltas.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let mean = unique_deltas.iter().sum::<f64>() / usize_to_f64(unique_deltas.len());
    let variance = unique_deltas
        .iter()
        .map(|delta| (delta - mean).powi(2))
        .sum::<f64>()
        / usize_to_f64(unique_deltas.len());
    (variance < f64::EPSILON).then(|| 24.0 * (time_days[1] - time_days[0]))
}

pub(crate) fn lomb_frequencies(time_hours: &[f64]) -> Vec<f64> {
    const MAX_PER_BAND: usize = 500;

    let sample_count = time_hours.len();
    let delta_time =
        (time_hours[sample_count - 1] - time_hours[0]) / usize_to_f64(sample_count - 1);
    let record_length = usize_to_f64(sample_count) * delta_time;
    let base_count = sample_count / 2 - 1;
    let base = (1..=base_count)
        .map(|index| usize_to_f64(index) / record_length)
        .collect::<Vec<_>>();
    let mut frequencies = Vec::new();
    for [lower, upper] in COLORED_NOISE_FREQUENCY_BANDS_CPH {
        let start = base.partition_point(|frequency| *frequency < lower);
        let upper_insertion = base.partition_point(|frequency| *frequency < upper);
        let stop = (upper_insertion + 1).min(base.len());
        if stop <= start {
            continue;
        }
        let count = stop - start;
        if count > MAX_PER_BAND {
            let first = base[start];
            let last = base[stop - 1];
            frequencies.extend((0..MAX_PER_BAND).map(|index| {
                first + (last - first) * usize_to_f64(index) / usize_to_f64(MAX_PER_BAND - 1)
            }));
        } else {
            frequencies.extend_from_slice(&base[start..stop]);
        }
    }
    frequencies
}

fn spectral_band_counts(
    frequencies_cph: &[f64],
    constituent_frequencies_cph: &[f64],
) -> ([usize; 9], [usize; 9]) {
    let mut excluded = constituent_frequencies_cph
        .iter()
        .filter_map(|frequency| nearest_frequency_index(frequencies_cph, *frequency))
        .collect::<Vec<_>>();
    excluded.sort_unstable();
    excluded.dedup();

    let mut total = [0; 9];
    let mut usable = [0; 9];
    for (band, [lower, upper]) in COLORED_NOISE_FREQUENCY_BANDS_CPH
        .iter()
        .copied()
        .enumerate()
    {
        let start = frequencies_cph.partition_point(|frequency| *frequency < lower);
        let upper_insertion = frequencies_cph.partition_point(|frequency| *frequency < upper);
        let stop = (upper_insertion + 1).min(frequencies_cph.len());
        total[band] = stop.saturating_sub(start);
        let excluded_in_band = excluded
            .iter()
            .filter(|index| **index >= start && **index < stop)
            .count();
        usable[band] = total[band].saturating_sub(excluded_in_band);
    }
    (total, usable)
}

fn nearest_frequency_index(frequencies: &[f64], target: f64) -> Option<usize> {
    if frequencies.is_empty() {
        return None;
    }
    let right = frequencies.partition_point(|frequency| *frequency < target);
    if right == 0 {
        return Some(0);
    }
    if right == frequencies.len() {
        return Some(right - 1);
    }
    let left = right - 1;
    let fraction = (target - frequencies[left]) / (frequencies[right] - frequencies[left]);
    Some(if fraction < 0.5 {
        left
    } else if fraction > 0.5 || right % 2 == 0 {
        right
    } else {
        left
    })
}

#[allow(
    clippy::cast_precision_loss,
    reason = "slice lengths are exactly representable in practical f64 analyses"
)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::{ResidualSpectrumMethod, SamplingDiagnosticsPlan};
    use crate::{AnalysisError, sampling::COLORED_NOISE_FREQUENCY_BANDS_CPH};

    #[test]
    fn regular_and_gappy_records_share_fft_coverage() {
        let time = (0_u32..240)
            .map(|index| 58_000.0 + f64::from(index) / 24.0)
            .collect::<Vec<_>>();
        let plan = SamplingDiagnosticsPlan::prepare(&time).expect("valid hourly grid");
        let complete = plan
            .diagnose(&vec![true; time.len()], &[0.080_511_4, 1.0 / 24.0])
            .expect("valid complete diagnostics");
        let mut finite = vec![true; time.len()];
        finite[0] = false;
        finite[51] = false;
        finite[239] = false;
        let gappy = plan
            .diagnose(&finite, &[0.080_511_4, 1.0 / 24.0])
            .expect("valid gappy diagnostics");
        assert_eq!(
            complete.residual_spectrum_method,
            ResidualSpectrumMethod::Fft
        );
        assert_eq!(gappy.residual_spectrum_method, ResidualSpectrumMethod::Fft);
        assert_eq!(
            complete.spectral_band_bin_count,
            gappy.spectral_band_bin_count
        );
        assert_eq!(
            complete.spectral_band_usable_bin_count,
            gappy.spectral_band_usable_bin_count
        );
        assert_eq!(complete.observation_count, 240);
        assert_eq!(gappy.observation_count, 237);
        // Frozen from UTide 8fabe121752bc317931472a10a42e306715106de.
        assert_eq!(
            complete.spectral_band_bin_count,
            [2, 5, 5, 5, 5, 5, 6, 8, 49]
        );
        assert_eq!(
            complete.spectral_band_usable_bin_count,
            [2, 4, 4, 5, 5, 5, 6, 8, 49]
        );
        assert!((gappy.largest_gap_hours - 2.0).abs() < 1e-9);
        assert!(gappy.record_span_days < complete.record_span_days);
    }

    #[test]
    fn irregular_coverage_uses_the_lomb_grid_and_tracks_exclusions() {
        let time = (0_u32..301)
            .map(|index| {
                58_000.0 + f64::from(index) / 24.0 + 0.004 * (f64::from(index) / 7.0).sin()
            })
            .collect::<Vec<_>>();
        let plan = SamplingDiagnosticsPlan::prepare(&time).expect("valid irregular grid");
        let diagnostics = plan
            .diagnose(&vec![true; time.len()], &[0.080_511_4, 1.0 / 24.0])
            .expect("valid irregular diagnostics");
        assert_eq!(
            diagnostics.residual_spectrum_method,
            ResidualSpectrumMethod::LombScargle
        );
        assert_eq!(diagnostics.residual_spectrum_time_count, 300);
        assert_eq!(
            diagnostics.spectral_band_bin_count,
            [2, 6, 6, 6, 6, 6, 7, 10, 60]
        );
        assert_eq!(
            diagnostics.spectral_band_usable_bin_count,
            [2, 5, 5, 6, 6, 6, 7, 10, 60]
        );
        assert!(COLORED_NOISE_FREQUENCY_BANDS_CPH[0][0] > 0.0);
    }

    #[test]
    fn rejects_invalid_masks_and_under_sampled_series() {
        let plan = SamplingDiagnosticsPlan::prepare(&[58_000.0, 58_001.0])
            .expect("valid two-point source grid");
        assert_eq!(
            plan.diagnose(&[true], &[0.1]),
            Err(AnalysisError::SamplingMaskShape {
                actual: 1,
                expected: 2,
            })
        );
        assert_eq!(
            plan.diagnose(&[true, false], &[0.1]),
            Err(AnalysisError::InsufficientSamplingObservations { actual: 1 })
        );
        assert_eq!(
            plan.diagnose(&[true, true], &[f64::NAN]),
            Err(AnalysisError::InvalidSamplingFrequency { index: 0 })
        );
    }
}
