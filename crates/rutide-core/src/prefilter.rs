//! Validated preprocessing-filter response correction.

use std::sync::Arc;

use crate::{AnalysisError, Constituent};

/// Behavior when a constituent cannot use an interpolated filter gain.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PreFilterFallback {
    /// Reject frequencies outside the response grid and gains outside the
    /// configured acceptable magnitude range.
    #[default]
    Error,
    /// Substitute a unit gain, matching MATLAB `UTide`'s legacy behavior.
    Unity,
}

/// A real-valued preprocessing-filter transfer function.
///
/// Frequencies are cycles per hour. Gains describe the filter that was applied
/// to the observations, not its inverse. Values are linearly interpolated once
/// at the fitted constituent frequencies and then folded into the prepared
/// astronomical basis.
///
/// This representation supports scalar records and vector records for which
/// the same real response was applied to both components. Component-specific
/// or phase-changing vector filters require a coupled complex formulation and
/// are intentionally outside this contract.
#[derive(Clone, Debug, PartialEq)]
pub struct PreFilterCorrection {
    frequency_cph: Arc<[f64]>,
    gain: Arc<[f64]>,
    minimum_acceptable_gain: f64,
    maximum_acceptable_gain: f64,
    fallback: PreFilterFallback,
}

impl PreFilterCorrection {
    /// Construct and validate a filter response.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] unless there are at least two finite response
    /// points, frequencies are strictly increasing and non-negative, gains are
    /// finite, and the acceptable gain bounds are finite, positive, and
    /// ordered.
    pub fn new(
        frequency_cph: Vec<f64>,
        gain: Vec<f64>,
        minimum_acceptable_gain: f64,
        maximum_acceptable_gain: f64,
    ) -> Result<Self, AnalysisError> {
        if frequency_cph.len() != gain.len() {
            return Err(AnalysisError::PreFilterShape {
                frequencies: frequency_cph.len(),
                gains: gain.len(),
            });
        }
        if frequency_cph.len() < 2 {
            return Err(AnalysisError::InsufficientPreFilterSamples {
                actual: frequency_cph.len(),
            });
        }
        for (index, frequency) in frequency_cph.iter().copied().enumerate() {
            if !frequency.is_finite() || frequency < 0.0 {
                return Err(AnalysisError::InvalidPreFilterFrequency { index });
            }
            if index > 0 && frequency <= frequency_cph[index - 1] {
                return Err(AnalysisError::NonIncreasingPreFilterFrequency { index });
            }
        }
        for (index, gain) in gain.iter().copied().enumerate() {
            if !gain.is_finite() {
                return Err(AnalysisError::InvalidPreFilterGain { index });
            }
        }
        if !minimum_acceptable_gain.is_finite()
            || !maximum_acceptable_gain.is_finite()
            || minimum_acceptable_gain <= 0.0
            || maximum_acceptable_gain < minimum_acceptable_gain
        {
            return Err(AnalysisError::InvalidPreFilterGainRange);
        }
        Ok(Self {
            frequency_cph: frequency_cph.into(),
            gain: gain.into(),
            minimum_acceptable_gain,
            maximum_acceptable_gain,
            fallback: PreFilterFallback::Error,
        })
    }

    /// Set explicit invalid/outside-range handling.
    #[must_use]
    pub const fn with_fallback(mut self, fallback: PreFilterFallback) -> Self {
        self.fallback = fallback;
        self
    }

    /// Frequencies defining the response, in cycles per hour.
    #[must_use]
    pub fn frequency_cph(&self) -> &[f64] {
        &self.frequency_cph
    }

    /// Real transfer-function gains corresponding to [`Self::frequency_cph`].
    #[must_use]
    pub fn gain(&self) -> &[f64] {
        &self.gain
    }

    /// Inclusive minimum acceptable gain magnitude.
    #[must_use]
    pub const fn minimum_acceptable_gain(&self) -> f64 {
        self.minimum_acceptable_gain
    }

    /// Inclusive maximum acceptable gain magnitude.
    #[must_use]
    pub const fn maximum_acceptable_gain(&self) -> f64 {
        self.maximum_acceptable_gain
    }

    /// Configured invalid/outside-range behavior.
    #[must_use]
    pub const fn fallback(&self) -> PreFilterFallback {
        self.fallback
    }

    /// Resolve one interpolated gain per fitted constituent.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when a constituent lies outside the response
    /// grid or its interpolated gain violates the acceptable magnitude range
    /// and [`Self::fallback`] is [`PreFilterFallback::Error`].
    pub fn resolve_constituent_gains(
        &self,
        constituents: &[Constituent],
    ) -> Result<Vec<f64>, AnalysisError> {
        constituents
            .iter()
            .enumerate()
            .map(|(constituent, metadata)| {
                self.interpolate(metadata.frequency_cph)
                    .and_then(|gain| self.validate_resolved_gain(constituent, gain))
            })
            .collect()
    }

    fn interpolate(&self, frequency_cph: f64) -> Result<f64, AnalysisError> {
        let first = self.frequency_cph[0];
        let last = self.frequency_cph[self.frequency_cph.len() - 1];
        if frequency_cph < first || frequency_cph > last {
            return match self.fallback {
                PreFilterFallback::Error => Err(AnalysisError::PreFilterFrequencyOutOfRange {
                    frequency_cph,
                    minimum_cph: first,
                    maximum_cph: last,
                }),
                PreFilterFallback::Unity => Ok(1.0),
            };
        }
        match self
            .frequency_cph
            .binary_search_by(|probe| probe.total_cmp(&frequency_cph))
        {
            Ok(index) => Ok(self.gain[index]),
            Err(upper) => {
                let lower = upper - 1;
                let fraction = (frequency_cph - self.frequency_cph[lower])
                    / (self.frequency_cph[upper] - self.frequency_cph[lower]);
                Ok(self.gain[lower] + fraction * (self.gain[upper] - self.gain[lower]))
            }
        }
    }

    fn validate_resolved_gain(&self, constituent: usize, gain: f64) -> Result<f64, AnalysisError> {
        let magnitude = gain.abs();
        if magnitude < self.minimum_acceptable_gain || magnitude > self.maximum_acceptable_gain {
            return match self.fallback {
                PreFilterFallback::Error => Err(AnalysisError::PreFilterGainOutOfRange {
                    constituent,
                    gain,
                    minimum: self.minimum_acceptable_gain,
                    maximum: self.maximum_acceptable_gain,
                }),
                PreFilterFallback::Unity => Ok(1.0),
            };
        }
        Ok(gain)
    }
}

#[cfg(test)]
mod tests {
    use super::{PreFilterCorrection, PreFilterFallback};
    use crate::{AnalysisError, Constituent};

    fn constituent(frequency_cph: f64) -> Constituent {
        Constituent {
            name: "test".to_owned(),
            frequency_cph,
        }
    }

    #[test]
    fn linearly_interpolates_and_accepts_exact_endpoints() {
        let response =
            PreFilterCorrection::new(vec![0.0, 0.05, 0.1], vec![1.0, 0.5, 0.25], 0.1, 2.0)
                .expect("valid response");
        let actual = response
            .resolve_constituent_gains(&[constituent(0.0), constituent(0.025), constituent(0.1)])
            .expect("covered gains");
        assert_eq!(actual, vec![1.0, 0.75, 0.25]);
    }

    #[test]
    fn strict_and_matlab_fallbacks_are_explicit() {
        let strict = PreFilterCorrection::new(vec![0.01, 0.1], vec![0.005, 0.5], 0.01, 2.0)
            .expect("valid response");
        assert!(matches!(
            strict.resolve_constituent_gains(&[constituent(0.01)]),
            Err(AnalysisError::PreFilterGainOutOfRange { constituent: 0, .. })
        ));
        assert!(matches!(
            strict.resolve_constituent_gains(&[constituent(0.2)]),
            Err(AnalysisError::PreFilterFrequencyOutOfRange { .. })
        ));

        let compatible = strict.with_fallback(PreFilterFallback::Unity);
        assert_eq!(
            compatible
                .resolve_constituent_gains(&[constituent(0.01), constituent(0.2)])
                .expect("unity fallbacks"),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn rejects_invalid_response_tables() {
        assert!(matches!(
            PreFilterCorrection::new(vec![0.0], vec![1.0], 0.1, 2.0),
            Err(AnalysisError::InsufficientPreFilterSamples { actual: 1 })
        ));
        assert!(matches!(
            PreFilterCorrection::new(vec![0.0, 0.1], vec![1.0], 0.1, 2.0),
            Err(AnalysisError::PreFilterShape {
                frequencies: 2,
                gains: 1
            })
        ));
        assert!(matches!(
            PreFilterCorrection::new(vec![0.1, 0.1], vec![1.0, 1.0], 0.1, 2.0),
            Err(AnalysisError::NonIncreasingPreFilterFrequency { index: 1 })
        ));
        assert!(matches!(
            PreFilterCorrection::new(vec![0.0, 0.1], vec![1.0, f64::NAN], 0.1, 2.0),
            Err(AnalysisError::InvalidPreFilterGain { index: 1 })
        ));
        assert_eq!(
            PreFilterCorrection::new(vec![0.0, 0.1], vec![1.0, 1.0], 0.0, 2.0),
            Err(AnalysisError::InvalidPreFilterGainRange)
        );
    }
}
