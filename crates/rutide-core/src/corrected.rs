//! Exact Greenwich phase and nodal corrections for catalog constituents.

use std::f64::consts::TAU;

use faer::Mat;
use rayon::prelude::*;

use crate::{
    AnalysisError, Constituent, FixedRawOls, LinearConfidence, ScalarSolution, TidalConstituent,
    astronomy::at_modified_julian_day,
    catalog::{CONSTITUENT_COUNT, Metadata},
    scalar::{equidistant_sample_interval_hours, validate_time},
};

/// A reusable fixed-constituent OLS model with exact Greenwich and nodal terms.
///
/// The astronomical basis matches the default phase and nodal behavior of the
/// pinned Python `UTide` oracle. Preparation is specific to a latitude because
/// satellite amplitude corrections depend on latitude. The resulting QR
/// factorization can be reused for any number of complete series at that same
/// latitude.
#[derive(Debug)]
pub struct GreenwichNodalOls {
    tidal_constituents: Vec<TidalConstituent>,
    latitude_degrees_north: f64,
    model: FixedRawOls,
}

impl GreenwichNodalOls {
    /// Build and factorize an exact corrected basis from Modified Julian Days.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when timestamps, latitude, or constituents are
    /// invalid, or when there are too few observations to overdetermine the
    /// requested model.
    pub fn prepare_modified_julian_days(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
    ) -> Result<Self, AnalysisError> {
        validate_latitude(latitude_degrees_north)?;
        let basis = CorrectionBasis::prepare(modified_julian_days, constituents)?;
        let model = basis.model_at_latitude(latitude_degrees_north)?;

        Ok(Self {
            tidal_constituents: constituents.to_vec(),
            latitude_degrees_north,
            model,
        })
    }

    /// Return the prepared catalog constituents in coefficient order.
    #[must_use]
    pub fn tidal_constituents(&self) -> &[TidalConstituent] {
        &self.tidal_constituents
    }

    /// Return constituent names and reference-time frequencies.
    #[must_use]
    pub fn constituents(&self) -> &[Constituent] {
        self.model.constituents()
    }

    /// Return the latitude used to construct the corrected basis.
    #[must_use]
    pub const fn latitude_degrees_north(&self) -> f64 {
        self.latitude_degrees_north
    }

    /// Return the number of observations expected in each spatial series.
    #[must_use]
    pub const fn time_count(&self) -> usize {
        self.model.time_count()
    }

    /// Fit one complete, finite scalar observation series.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when the observation shape is inconsistent or
    /// any value is non-finite.
    pub fn solve(&self, observations: &[f64]) -> Result<ScalarSolution, AnalysisError> {
        self.model.solve(observations)
    }

    /// Fit one series with linearized 95% confidence intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid observations or when colored noise
    /// is requested for non-equidistant timestamps.
    pub fn solve_with_linear_confidence(
        &self,
        observations: &[f64],
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model.solve_with_linear_confidence(observations, noise)
    }

    /// Fit complete series at the prepared latitude in time-major order.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when the flattened shape is inconsistent, no
    /// series are supplied, or any value is non-finite.
    pub fn solve_many_time_major(
        &self,
        observations: &[f64],
        series_count: usize,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.model.solve_many_time_major(observations, series_count)
    }

    /// Fit complete series with linearized 95% confidence intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid observations or when colored noise
    /// is requested for non-equidistant timestamps.
    pub fn solve_many_time_major_with_linear_confidence(
        &self,
        observations: &[f64],
        series_count: usize,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.model
            .solve_many_time_major_with_linear_confidence(observations, series_count, noise)
    }
}

/// Shared exact astronomy for fitting many series at different latitudes.
///
/// Preparation evaluates the latitude-independent astronomical arguments once.
/// [`Self::solve_time_major`] then builds and solves each latitude-specific
/// design independently using Rayon's active thread pool. Results retain input
/// series order.
#[derive(Debug)]
pub struct GreenwichNodalBatch {
    basis: CorrectionBasis,
}

impl GreenwichNodalBatch {
    /// Prepare shared astronomical terms from Modified Julian Days.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when timestamps or constituents are invalid or
    /// the model would not be overdetermined.
    pub fn prepare_modified_julian_days(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
    ) -> Result<Self, AnalysisError> {
        Ok(Self {
            basis: CorrectionBasis::prepare(modified_julian_days, constituents)?,
        })
    }

    /// Return the number of observations expected for each spatial series.
    #[must_use]
    pub const fn time_count(&self) -> usize {
        self.basis.time_terms.len()
    }

    /// Return the prepared catalog constituents in coefficient order.
    #[must_use]
    pub fn tidal_constituents(&self) -> &[TidalConstituent] {
        &self.basis.tidal_constituents
    }

    /// Return constituent names and reference-time frequencies.
    #[must_use]
    pub fn constituents(&self) -> &[Constituent] {
        &self.basis.scalar_constituents
    }

    /// Fit varying-latitude scalar series stored in time-major order.
    ///
    /// `observations[time * latitudes.len() + series]` corresponds to one
    /// timestamp and latitude. Independent fits execute on Rayon's active thread
    /// pool, while the collected result order remains deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for empty inputs, invalid latitudes, inconsistent
    /// observation shape, or non-finite observations.
    pub fn solve_time_major(
        &self,
        observations: &[f64],
        latitudes: &[f64],
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(observations, latitudes, None)
    }

    /// Fit varying-latitude series with linearized confidence intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or when colored noise is
    /// requested for non-equidistant timestamps.
    pub fn solve_time_major_with_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(observations, latitudes, Some(noise))
    }

    fn solve_time_major_impl(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        confidence: Option<LinearConfidence>,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        if latitudes.is_empty() {
            return Err(AnalysisError::EmptySeries);
        }
        for latitude in latitudes.iter().copied() {
            validate_latitude(latitude)?;
        }
        let series_count = latitudes.len();
        let expected = self.time_count().saturating_mul(series_count);
        if observations.len() != expected {
            return Err(AnalysisError::ObservationShape {
                actual: observations.len(),
                expected,
            });
        }
        for (index, value) in observations.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(AnalysisError::NonFiniteObservation {
                    series: index % series_count,
                    time: index / series_count,
                });
            }
        }

        (0..series_count)
            .into_par_iter()
            .map(|series| {
                let model = self.basis.model_at_latitude(latitudes[series])?;
                let mut series_observations = Vec::with_capacity(self.time_count());
                for time in 0..self.time_count() {
                    series_observations.push(observations[time * series_count + series]);
                }
                match confidence {
                    Some(noise) => model.solve_with_linear_confidence(&series_observations, noise),
                    None => model.solve(&series_observations),
                }
            })
            .collect()
    }
}

#[derive(Debug)]
struct CorrectionBasis {
    tidal_constituents: Vec<TidalConstituent>,
    scalar_constituents: Vec<Constituent>,
    recipes: Vec<CorrectionRecipe>,
    time_terms: Vec<TimeTerms>,
    time_span_days: f64,
    sample_interval_hours: Option<f64>,
}

#[derive(Debug)]
enum CorrectionRecipe {
    Base { base_index: usize },
    Shallow { terms: Vec<ShallowRecipeTerm> },
}

#[derive(Clone, Copy, Debug)]
struct ShallowRecipeTerm {
    base_index: usize,
    coefficient: f64,
}

impl CorrectionBasis {
    fn prepare(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
    ) -> Result<Self, AnalysisError> {
        validate_tidal_constituents(constituents)?;
        let (reference_time, time_span_days) =
            validate_time(modified_julian_days, constituents.len())?;
        let (base_constituents, recipes) = dependency_recipes(constituents);
        let reference_astronomy = at_modified_julian_day(reference_time);
        let base_frequencies = base_constituents
            .iter()
            .copied()
            .map(|constituent| {
                let metadata = constituent.metadata();
                dot6(
                    metadata
                        .doodson
                        .expect("base catalog constituent has Doodson multipliers"),
                    reference_astronomy.cycles_per_day,
                ) / 24.0
            })
            .collect::<Vec<_>>();
        let scalar_constituents = constituents
            .iter()
            .copied()
            .zip(&recipes)
            .map(|(constituent, recipe)| {
                Constituent::new(
                    constituent.name(),
                    recipe.combine_frequency(&base_frequencies),
                )
            })
            .collect::<Vec<_>>();
        validate_derived_frequencies(&scalar_constituents)?;
        let time_terms = modified_julian_days
            .iter()
            .copied()
            .map(|time| {
                let astronomy = at_modified_julian_day(time);
                let base_greenwich_phase = base_constituents
                    .iter()
                    .copied()
                    .map(|constituent| {
                        let metadata = constituent.metadata();
                        (dot6(
                            metadata
                                .doodson
                                .expect("base catalog constituent has Doodson multipliers"),
                            astronomy.cycles,
                        ) + metadata
                            .semi
                            .expect("base catalog constituent has a phase offset"))
                            % 1.0
                    })
                    .collect::<Vec<_>>();
                TimeTerms {
                    greenwich_phase: recipes
                        .iter()
                        .map(|recipe| recipe.combine_phase(&base_greenwich_phase))
                        .collect(),
                    base_nodal_terms: base_constituents
                        .iter()
                        .copied()
                        .map(|constituent| {
                            precompute_nodal_terms(constituent.metadata(), astronomy.cycles)
                        })
                        .collect(),
                    normalized_trend: (time - reference_time) / time_span_days,
                }
            })
            .collect();
        Ok(Self {
            tidal_constituents: constituents.to_vec(),
            scalar_constituents,
            recipes,
            time_terms,
            time_span_days,
            sample_interval_hours: equidistant_sample_interval_hours(modified_julian_days),
        })
    }

    fn model_at_latitude(&self, latitude: f64) -> Result<FixedRawOls, AnalysisError> {
        validate_latitude(latitude)?;
        let harmonic_columns = self.tidal_constituents.len() * 2;
        let mut design = Mat::zeros(self.time_terms.len(), harmonic_columns + 2);
        let latitude_factors = latitude_factors(latitude);
        for (time_index, terms) in self.time_terms.iter().enumerate() {
            let base_corrections = terms
                .base_nodal_terms
                .iter()
                .copied()
                .map(|terms| nodal_correction(terms, latitude_factors))
                .collect::<Vec<_>>();
            for constituent_index in 0..self.tidal_constituents.len() {
                let (nodal_amplitude, nodal_phase) =
                    self.recipes[constituent_index].combine_nodal(&base_corrections);
                let angle = TAU * (nodal_phase + terms.greenwich_phase[constituent_index]);
                design[(time_index, constituent_index * 2)] = nodal_amplitude * angle.cos();
                design[(time_index, constituent_index * 2 + 1)] = nodal_amplitude * angle.sin();
            }
            design[(time_index, harmonic_columns)] = 1.0;
            design[(time_index, harmonic_columns + 1)] = terms.normalized_trend;
        }
        Ok(FixedRawOls::from_design(
            self.scalar_constituents.clone(),
            self.time_terms.len(),
            self.time_span_days,
            self.sample_interval_hours,
            design,
        ))
    }
}

#[derive(Debug)]
struct TimeTerms {
    greenwich_phase: Vec<f64>,
    base_nodal_terms: Vec<NodalTerms>,
    normalized_trend: f64,
}

impl CorrectionRecipe {
    fn combine_frequency(&self, base_frequencies: &[f64]) -> f64 {
        match self {
            Self::Base { base_index } => base_frequencies[*base_index],
            Self::Shallow { terms } => terms
                .iter()
                .map(|term| term.coefficient * base_frequencies[term.base_index])
                .sum(),
        }
    }

    fn combine_phase(&self, base_phases: &[f64]) -> f64 {
        match self {
            Self::Base { base_index } => base_phases[*base_index],
            Self::Shallow { terms } => terms
                .iter()
                .map(|term| term.coefficient * base_phases[term.base_index])
                .sum(),
        }
    }

    fn combine_nodal(&self, base_corrections: &[(f64, f64)]) -> (f64, f64) {
        match self {
            Self::Base { base_index } => base_corrections[*base_index],
            Self::Shallow { terms } => {
                let mut amplitude = 1.0;
                let mut phase = 0.0;
                for term in terms {
                    let (parent_amplitude, parent_phase) = base_corrections[term.base_index];
                    amplitude *= parent_amplitude.powf(term.coefficient.abs());
                    phase += parent_phase * term.coefficient;
                }
                (amplitude, phase)
            }
        }
    }
}

fn dependency_recipes(
    constituents: &[TidalConstituent],
) -> (Vec<TidalConstituent>, Vec<CorrectionRecipe>) {
    let mut needed = [false; CONSTITUENT_COUNT];
    for constituent in constituents.iter().copied() {
        let metadata = constituent.metadata();
        if metadata.shallow_terms.is_empty() {
            needed[constituent.index()] = true;
        } else {
            for term in metadata.shallow_terms {
                needed[term.parent_index] = true;
            }
        }
    }

    let base_constituents = TidalConstituent::all()
        .filter(|constituent| needed[constituent.index()])
        .collect::<Vec<_>>();
    let mut base_positions = [usize::MAX; CONSTITUENT_COUNT];
    for (position, constituent) in base_constituents.iter().copied().enumerate() {
        base_positions[constituent.index()] = position;
    }
    let recipes = constituents
        .iter()
        .copied()
        .map(|constituent| {
            let metadata = constituent.metadata();
            if metadata.shallow_terms.is_empty() {
                CorrectionRecipe::Base {
                    base_index: base_positions[constituent.index()],
                }
            } else {
                CorrectionRecipe::Shallow {
                    terms: metadata
                        .shallow_terms
                        .iter()
                        .map(|term| ShallowRecipeTerm {
                            base_index: base_positions[term.parent_index],
                            coefficient: term.coefficient,
                        })
                        .collect(),
                }
            }
        })
        .collect();
    (base_constituents, recipes)
}

fn validate_derived_frequencies(constituents: &[Constituent]) -> Result<(), AnalysisError> {
    for (index, constituent) in constituents.iter().enumerate() {
        if !constituent.frequency_cph.is_finite() || constituent.frequency_cph <= 0.0 {
            return Err(AnalysisError::InvalidFrequency { index });
        }
        if constituents[..index]
            .iter()
            .any(|earlier| earlier.frequency_cph.to_bits() == constituent.frequency_cph.to_bits())
        {
            return Err(AnalysisError::DuplicateFrequency { index });
        }
    }
    Ok(())
}

fn validate_latitude(latitude: f64) -> Result<(), AnalysisError> {
    if !latitude.is_finite() || !(-90.0..=90.0).contains(&latitude) {
        return Err(AnalysisError::InvalidLatitude);
    }
    if latitude == 0.0 {
        return Err(AnalysisError::EquatorialLatitude);
    }
    Ok(())
}

fn validate_tidal_constituents(constituents: &[TidalConstituent]) -> Result<(), AnalysisError> {
    if constituents.is_empty() {
        return Err(AnalysisError::EmptyConstituents);
    }
    for (index, constituent) in constituents.iter().enumerate() {
        if constituents[..index].contains(constituent) {
            return Err(AnalysisError::DuplicateFrequency { index });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct NodalTerms {
    real: [f64; 3],
    imaginary: [f64; 3],
}

fn precompute_nodal_terms(metadata: Metadata, astronomy: [f64; 6]) -> NodalTerms {
    let mut real = [0.0; 3];
    let mut imaginary = [0.0; 3];
    for satellite in metadata.satellites {
        let class = match satellite.latitude_factor {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => unreachable!("catalog latitude factors are validated at compile time"),
        };
        let argument = f64::from(satellite.delta_doodson[0]) * astronomy[3]
            + f64::from(satellite.delta_doodson[1]) * astronomy[4]
            + f64::from(satellite.delta_doodson[2]) * astronomy[5]
            + satellite.phase_correction;
        let angle = TAU * (argument % 1.0);
        real[class] += satellite.amplitude_ratio * angle.cos();
        imaginary[class] += satellite.amplitude_ratio * angle.sin();
    }
    NodalTerms { real, imaginary }
}

fn latitude_factors(latitude: f64) -> [f64; 3] {
    let adjusted_latitude = if latitude.abs() < 5.0 {
        latitude.signum() * 5.0
    } else {
        latitude
    };
    let sine_latitude = adjusted_latitude.to_radians().sin();
    [
        1.0,
        0.363_09 * (1.0 - 5.0 * sine_latitude.powi(2)) / sine_latitude,
        2.598_08 * sine_latitude,
    ]
}

fn nodal_correction(terms: NodalTerms, latitude_factors: [f64; 3]) -> (f64, f64) {
    let mut real = 1.0;
    let mut imaginary = 0.0;
    for (class, factor) in latitude_factors.into_iter().enumerate() {
        real += factor * terms.real[class];
        imaginary += factor * terms.imaginary[class];
    }
    (real.hypot(imaginary), imaginary.atan2(real) / TAU)
}

fn dot6(left: [i8; 6], right: [f64; 6]) -> f64 {
    f64::from(left[0]) * right[0]
        + f64::from(left[1]) * right[1]
        + f64::from(left[2]) * right[2]
        + f64::from(left[3]) * right[3]
        + f64::from(left[4]) * right[4]
        + f64::from(left[5]) * right[5]
}

#[cfg(test)]
mod tests {
    use super::{GreenwichNodalBatch, GreenwichNodalOls};
    use crate::{AnalysisError, TidalConstituent};

    fn times() -> Vec<f64> {
        (0_u32..745)
            .map(|index| 58_113.0 + f64::from(index) / 24.0)
            .collect()
    }

    #[test]
    fn exposes_reference_time_frequencies() {
        let model = GreenwichNodalOls::prepare_modified_julian_days(
            &times(),
            60.957_717_895_507_81,
            &[TidalConstituent::M2, TidalConstituent::S2],
        )
        .expect("valid corrected model");
        assert!((model.constituents()[0].frequency_cph - 0.080_511_400_671_577_2).abs() < 1e-15);
        assert!((model.constituents()[1].frequency_cph - 1.0 / 12.0).abs() < 1e-15);
    }

    #[test]
    fn rejects_duplicate_catalog_constituent() {
        assert!(matches!(
            GreenwichNodalOls::prepare_modified_julian_days(
                &times(),
                60.0,
                &[TidalConstituent::M2, TidalConstituent::M2],
            ),
            Err(AnalysisError::DuplicateFrequency { index: 1 })
        ));
    }

    #[test]
    fn rejects_the_oracle_equatorial_singularity() {
        assert!(matches!(
            GreenwichNodalOls::prepare_modified_julian_days(&times(), 0.0, &[TidalConstituent::K1],),
            Err(AnalysisError::EquatorialLatitude)
        ));
    }

    #[test]
    fn varying_latitude_batch_matches_individual_models_in_input_order() {
        let time = times();
        let latitudes = [60.0, 61.0];
        let constituents = [TidalConstituent::M2, TidalConstituent::K1];
        let mut observations = Vec::with_capacity(time.len() * latitudes.len());
        for index in 0..time.len() {
            let position = f64::from(u32::try_from(index).expect("fixture index fits u32"));
            observations.push(0.2 + (position / 11.0).sin());
            observations.push(-0.1 + (position / 7.0).cos());
        }
        let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &constituents)
            .expect("valid batch");
        let actual = batch
            .solve_time_major(&observations, &latitudes)
            .expect("valid observations");

        for series in 0..latitudes.len() {
            let model = GreenwichNodalOls::prepare_modified_julian_days(
                &time,
                latitudes[series],
                &constituents,
            )
            .expect("valid individual model");
            let values = observations
                .chunks_exact(latitudes.len())
                .map(|row| row[series])
                .collect::<Vec<_>>();
            let expected = model.solve(&values).expect("valid individual series");
            assert_eq!(actual[series], expected);
        }
    }
}
