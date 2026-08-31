//! Exact Greenwich phase and nodal corrections for catalog constituents.

use std::{collections::HashMap, f64::consts::TAU};

use faer::Mat;
use rayon::prelude::*;

use crate::{
    AnalysisError, Constituent, FixedRawOls, LinearConfidence, ScalarSolution, TidalConstituent,
    VectorReconstruction, VectorSolution,
    astronomy::at_modified_julian_day,
    catalog::{CONSTITUENT_COUNT, Metadata},
    scalar::{ConfidenceSampling, equidistant_sample_interval_hours, validate_time},
    vector::from_component_solutions,
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
    reference_time_modified_julian_day: f64,
    base_constituents: Vec<TidalConstituent>,
    recipes: Vec<CorrectionRecipe>,
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
            reference_time_modified_julian_day: basis.reference_time_modified_julian_day,
            base_constituents: basis.base_constituents,
            recipes: basis.recipes,
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

    /// Return the midpoint epoch used for the fitted trend, as an MJD.
    #[must_use]
    pub const fn reference_time_modified_julian_day(&self) -> f64 {
        self.reference_time_modified_julian_day
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

    /// Jointly fit one eastward/northward current series.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid component shapes or values.
    pub fn solve_vector(
        &self,
        eastward: &[f64],
        northward: &[f64],
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(eastward, northward, None)
    }

    /// Jointly fit one current series with linearized ellipse confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid inputs or unsupported irregular
    /// colored spectra.
    pub fn solve_vector_with_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        noise: LinearConfidence,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(eastward, northward, Some(noise))
    }

    fn solve_vector_impl(
        &self,
        eastward: &[f64],
        northward: &[f64],
        confidence: Option<LinearConfidence>,
    ) -> Result<VectorSolution, AnalysisError> {
        if eastward.len() != northward.len() {
            return Err(AnalysisError::ObservationShape {
                actual: northward.len(),
                expected: eastward.len(),
            });
        }
        let mut time_major = Vec::with_capacity(eastward.len() * 2);
        for (eastward, northward) in eastward.iter().copied().zip(northward.iter().copied()) {
            time_major.extend([eastward, northward]);
        }
        let mut components = match confidence {
            // Python UTide's two-dimensional colored linear CI path leaves the
            // eastward pair white and applies the band spectrum to the
            // northward pair. Keep that asymmetric reference behavior here.
            Some(noise) => self
                .model
                .solve_many_time_major_with_linear_confidence_by_series(
                    &time_major,
                    &[LinearConfidence::White, noise],
                )?,
            None => self.model.solve_many_time_major(&time_major, 2)?,
        };
        let northward = components.pop().expect("two requested component solutions");
        let eastward = components.pop().expect("two requested component solutions");
        from_component_solutions(&eastward, &northward)
    }

    /// Reconstruct one solution at arbitrary Modified Julian Days.
    ///
    /// Exact Greenwich phase and nodal corrections are evaluated at each target
    /// timestamp. The fitted mean and trend are always retained.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid target times, coefficient shapes,
    /// latitude, thresholds, or constituent filters.
    pub fn reconstruct_modified_julian_days(
        &self,
        modified_julian_days: &[f64],
        solution: &ScalarSolution,
        filter: &ReconstructionFilter,
    ) -> Result<Vec<f64>, AnalysisError> {
        GreenwichNodalReconstructor::from_parts(
            modified_julian_days,
            self.reference_time_modified_julian_day,
            self.tidal_constituents.clone(),
            &self.base_constituents,
            self.recipes.clone(),
        )?
        .reconstruct_at_latitude(solution, self.latitude_degrees_north, filter)
    }

    /// Reconstruct one current-ellipse solution at arbitrary Modified Julian Days.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid target times, solution shapes,
    /// latitude, thresholds, or constituent filters.
    pub fn reconstruct_vector_modified_julian_days(
        &self,
        modified_julian_days: &[f64],
        solution: &VectorSolution,
        filter: &ReconstructionFilter,
    ) -> Result<VectorReconstruction, AnalysisError> {
        GreenwichNodalReconstructor::from_parts(
            modified_julian_days,
            self.reference_time_modified_julian_day,
            self.tidal_constituents.clone(),
            &self.base_constituents,
            self.recipes.clone(),
        )?
        .reconstruct_vector_at_latitude(solution, self.latitude_degrees_north, filter)
    }
}

/// Constituent selection applied during reconstruction.
///
/// Explicit names are an alternative to diagnostics, matching Python `UTide`.
/// PE-only filtering remains available without confidence intervals by setting
/// `minimum_signal_to_noise` to `None`.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ReconstructionFilter {
    /// Include every fitted constituent.
    #[default]
    All,
    /// Include exactly these fitted constituents, retaining fitted-model order.
    Constituents(Vec<TidalConstituent>),
    /// Include constituents satisfying every supplied diagnostic threshold.
    Diagnostics {
        /// Minimum percent energy, inclusive.
        minimum_percent_energy: f64,
        /// Optional minimum signal-to-noise ratio, inclusive.
        minimum_signal_to_noise: Option<f64>,
    },
}

/// A reusable exact Greenwich/nodal basis for arbitrary reconstruction times.
///
/// Astronomy is prepared once and can then reconstruct many scalar solutions at
/// different latitudes. Multi-series output is series-major: one complete target
/// time series per input solution.
#[derive(Debug)]
pub struct GreenwichNodalReconstructor {
    tidal_constituents: Vec<TidalConstituent>,
    recipes: Vec<CorrectionRecipe>,
    reference_time_modified_julian_day: f64,
    time_terms: Vec<ReconstructionTimeTerms>,
}

impl GreenwichNodalReconstructor {
    /// Prepare an exact reconstruction basis from a fit epoch and target MJDs.
    ///
    /// Target times may be unordered or repeated, but must be finite and nonempty.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for an invalid reference time, target time, or
    /// constituent list.
    pub fn prepare_modified_julian_days(
        modified_julian_days: &[f64],
        reference_time_modified_julian_day: f64,
        constituents: &[TidalConstituent],
    ) -> Result<Self, AnalysisError> {
        validate_tidal_constituents(constituents)?;
        let (base_constituents, recipes) = dependency_recipes(constituents);
        Self::from_parts(
            modified_julian_days,
            reference_time_modified_julian_day,
            constituents.to_vec(),
            &base_constituents,
            recipes,
        )
    }

    fn from_parts(
        modified_julian_days: &[f64],
        reference_time_modified_julian_day: f64,
        tidal_constituents: Vec<TidalConstituent>,
        base_constituents: &[TidalConstituent],
        recipes: Vec<CorrectionRecipe>,
    ) -> Result<Self, AnalysisError> {
        validate_reconstruction_times(modified_julian_days, reference_time_modified_julian_day)?;
        let time_terms = modified_julian_days
            .iter()
            .copied()
            .map(|time| {
                let astronomy = at_modified_julian_day(time);
                let base_greenwich_phase = base_constituents
                    .iter()
                    .copied()
                    .map(|constituent| base_greenwich_phase(constituent, astronomy.cycles))
                    .collect::<Vec<_>>();
                ReconstructionTimeTerms {
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
                    modified_julian_day: time,
                }
            })
            .collect();
        Ok(Self {
            tidal_constituents,
            recipes,
            reference_time_modified_julian_day,
            time_terms,
        })
    }

    /// Return the midpoint epoch used for the reconstructed trend, as an MJD.
    #[must_use]
    pub const fn reference_time_modified_julian_day(&self) -> f64 {
        self.reference_time_modified_julian_day
    }

    /// Return the number of prepared reconstruction timestamps.
    #[must_use]
    pub fn time_count(&self) -> usize {
        self.time_terms.len()
    }

    /// Return fitted catalog constituents in coefficient order.
    #[must_use]
    pub fn tidal_constituents(&self) -> &[TidalConstituent] {
        &self.tidal_constituents
    }

    /// Reconstruct one scalar solution at a specified latitude.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for an invalid latitude, coefficient shape, or
    /// filter.
    pub fn reconstruct_at_latitude(
        &self,
        solution: &ScalarSolution,
        latitude_degrees_north: f64,
        filter: &ReconstructionFilter,
    ) -> Result<Vec<f64>, AnalysisError> {
        validate_latitude(latitude_degrees_north)?;
        let selected = reconstruction_indices(&self.tidal_constituents, solution, filter)?;
        let latitude_factors = latitude_factors(latitude_degrees_north);
        let base_count = self
            .time_terms
            .first()
            .map_or(0, |terms| terms.base_nodal_terms.len());
        let mut base_corrections = vec![(0.0, 0.0); base_count];
        let mut reconstruction = Vec::with_capacity(self.time_terms.len());
        for terms in &self.time_terms {
            for (correction, nodal_terms) in base_corrections
                .iter_mut()
                .zip(terms.base_nodal_terms.iter().copied())
            {
                *correction = nodal_correction(nodal_terms, latitude_factors);
            }
            let harmonics = selected
                .iter()
                .copied()
                .map(|constituent| {
                    let (nodal_amplitude, nodal_phase) =
                        self.recipes[constituent].combine_nodal(&base_corrections);
                    let angle = TAU * (nodal_phase + terms.greenwich_phase[constituent])
                        - solution.phase_degrees[constituent].to_radians();
                    nodal_amplitude * solution.amplitude[constituent] * angle.cos()
                })
                .sum::<f64>();
            reconstruction.push(
                solution.mean
                    + solution.slope_per_day
                        * (terms.modified_julian_day - solution.reference_time_days)
                    + harmonics,
            );
        }
        Ok(reconstruction)
    }

    /// Reconstruct varying-latitude solutions in parallel, in series-major order.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when no series are supplied, the latitude count
    /// differs from the solution count, or any solution/filter input is invalid.
    pub fn reconstruct_many_series_major(
        &self,
        solutions: &[ScalarSolution],
        latitudes: &[f64],
        filter: &ReconstructionFilter,
    ) -> Result<Vec<Vec<f64>>, AnalysisError> {
        if solutions.is_empty() {
            return Err(AnalysisError::EmptySeries);
        }
        if solutions.len() != latitudes.len() {
            return Err(AnalysisError::ObservationShape {
                actual: latitudes.len(),
                expected: solutions.len(),
            });
        }
        solutions
            .par_iter()
            .zip(latitudes.par_iter().copied())
            .map(|(solution, latitude)| self.reconstruct_at_latitude(solution, latitude, filter))
            .collect()
    }

    /// Reconstruct varying-latitude current ellipses in parallel.
    ///
    /// Each returned pair is target-time-major within its component and retains
    /// input series order.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for inconsistent series counts, latitude,
    /// solution-shape, or filter inputs.
    pub fn reconstruct_many_vectors_series_major(
        &self,
        solutions: &[VectorSolution],
        latitudes: &[f64],
        filter: &ReconstructionFilter,
    ) -> Result<Vec<VectorReconstruction>, AnalysisError> {
        if solutions.is_empty() {
            return Err(AnalysisError::EmptySeries);
        }
        if solutions.len() != latitudes.len() {
            return Err(AnalysisError::ObservationShape {
                actual: latitudes.len(),
                expected: solutions.len(),
            });
        }
        solutions
            .par_iter()
            .zip(latitudes.par_iter().copied())
            .map(|(solution, latitude)| {
                self.reconstruct_vector_at_latitude(solution, latitude, filter)
            })
            .collect()
    }

    /// Reconstruct one current-ellipse solution at a specified latitude.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid latitude, solution shape, or filter.
    pub fn reconstruct_vector_at_latitude(
        &self,
        solution: &VectorSolution,
        latitude_degrees_north: f64,
        filter: &ReconstructionFilter,
    ) -> Result<VectorReconstruction, AnalysisError> {
        let (eastward, northward) = solution.component_solutions();
        Ok(VectorReconstruction {
            eastward: self.reconstruct_at_latitude(&eastward, latitude_degrees_north, filter)?,
            northward: self.reconstruct_at_latitude(&northward, latitude_degrees_north, filter)?,
        })
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

    /// Return constituent frequencies at an arbitrary finite reference MJD.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for a non-finite epoch or a degenerate derived
    /// frequency set.
    pub fn constituents_at_reference_modified_julian_day(
        &self,
        reference_time: f64,
    ) -> Result<Vec<Constituent>, AnalysisError> {
        if !reference_time.is_finite() {
            return Err(AnalysisError::NonFiniteReferenceTime);
        }
        let constituents = scalar_constituents_at_reference(
            &self.basis.tidal_constituents,
            &self.basis.base_constituents,
            &self.basis.recipes,
            reference_time,
        );
        validate_derived_frequencies(&constituents)?;
        Ok(constituents)
    }

    /// Return the midpoint epoch used for fitted trends, as an MJD.
    #[must_use]
    pub const fn reference_time_modified_julian_day(&self) -> f64 {
        self.basis.reference_time_modified_julian_day
    }

    /// Prepare a reusable exact basis at arbitrary reconstruction MJDs.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when target timestamps are empty or non-finite.
    pub fn reconstructor_modified_julian_days(
        &self,
        modified_julian_days: &[f64],
    ) -> Result<GreenwichNodalReconstructor, AnalysisError> {
        GreenwichNodalReconstructor::from_parts(
            modified_julian_days,
            self.basis.reference_time_modified_julian_day,
            self.basis.tidal_constituents.clone(),
            &self.basis.base_constituents,
            self.basis.recipes.clone(),
        )
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

    /// Fit scalar series while treating `NaN` observations as missing.
    ///
    /// Series sharing a valid-time mask reuse the same prepared record metadata.
    /// Infinite values remain invalid. Each series must retain enough samples to
    /// overdetermine the requested harmonic model.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid shapes, latitudes, infinities, or
    /// underdetermined masked records.
    pub fn solve_time_major_with_missing(
        &self,
        observations: &[f64],
        latitudes: &[f64],
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_with_missing_impl(observations, latitudes, None)
    }

    /// Fit possibly gappy scalar series with linearized confidence intervals.
    ///
    /// Colored intervals interpolate residuals from gappy observations onto an
    /// originally equidistant timestamp grid, matching Python `UTide`. Truly
    /// irregular input timestamps are rejected until Lomb–Scargle spectra are
    /// implemented. White intervals support either sampling pattern.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid inputs or unsupported irregular
    /// colored spectra.
    pub fn solve_time_major_with_missing_and_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_with_missing_impl(observations, latitudes, Some(noise))
    }

    /// Jointly fit possibly gappy eastward/northward current series.
    ///
    /// A sample is omitted from both components when either component is `NaN`.
    /// Series sharing that joint mask reuse prepared record metadata.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid shapes, infinities, latitudes, or
    /// underdetermined masked records.
    pub fn solve_vector_time_major_with_missing(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_with_missing_impl(eastward, northward, latitudes, None)
    }

    /// Jointly fit gappy currents with linearized ellipse confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or unsupported irregular
    /// colored spectra.
    pub fn solve_vector_time_major_with_missing_and_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_with_missing_impl(eastward, northward, latitudes, Some(noise))
    }

    fn solve_vector_time_major_with_missing_impl(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        confidence: Option<LinearConfidence>,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        validate_batch_shape_and_latitudes(self.time_count(), eastward, latitudes)?;
        if northward.len() != eastward.len() {
            return Err(AnalysisError::ObservationShape {
                actual: northward.len(),
                expected: eastward.len(),
            });
        }
        let series_count = latitudes.len();
        for (index, value) in eastward.iter().chain(northward).copied().enumerate() {
            if value.is_infinite() {
                return Err(AnalysisError::NonFiniteObservation {
                    series: index % series_count,
                    time: (index % eastward.len()) / series_count,
                });
            }
        }

        let mut records = Vec::<RecordSubset>::new();
        let mut record_by_positions = HashMap::<Vec<usize>, usize>::new();
        let mut record_for_series = Vec::with_capacity(series_count);
        for series in 0..series_count {
            let positions = (0..self.time_count())
                .filter(|time| {
                    eastward[time * series_count + series].is_finite()
                        && northward[time * series_count + series].is_finite()
                })
                .collect::<Vec<_>>();
            let record_index = if let Some(index) = record_by_positions.get(&positions) {
                *index
            } else {
                let index = records.len();
                records.push(self.basis.record_subset(positions.clone())?);
                record_by_positions.insert(positions, index);
                index
            };
            record_for_series.push(record_index);
        }

        (0..series_count)
            .into_par_iter()
            .map(|series| {
                let record = &records[record_for_series[series]];
                let model = self
                    .basis
                    .model_at_latitude_for_record(latitudes[series], record)?;
                let mut component_values = Vec::with_capacity(record.positions.len() * 2);
                for time in record.positions.iter().copied() {
                    component_values.push(eastward[time * series_count + series]);
                    component_values.push(northward[time * series_count + series]);
                }
                let mut components = match confidence {
                    // Match Python UTide's asymmetric 2-D colored linear CI:
                    // eastward remains white while northward is colored.
                    Some(noise) => model.solve_many_time_major_with_linear_confidence_by_series(
                        &component_values,
                        &[LinearConfidence::White, noise],
                    )?,
                    None => model.solve_many_time_major(&component_values, 2)?,
                };
                let northward = components.pop().expect("two requested component solutions");
                let eastward = components.pop().expect("two requested component solutions");
                from_component_solutions(&eastward, &northward)
            })
            .collect()
    }

    fn solve_time_major_with_missing_impl(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        confidence: Option<LinearConfidence>,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        validate_batch_shape_and_latitudes(self.time_count(), observations, latitudes)?;
        for (index, value) in observations.iter().copied().enumerate() {
            if value.is_infinite() {
                return Err(AnalysisError::NonFiniteObservation {
                    series: index % latitudes.len(),
                    time: index / latitudes.len(),
                });
            }
        }
        if observations.iter().all(|value| value.is_finite()) {
            return self.solve_time_major_impl(observations, latitudes, confidence);
        }

        let series_count = latitudes.len();
        let mut records = Vec::<RecordSubset>::new();
        let mut record_by_positions = HashMap::<Vec<usize>, usize>::new();
        let mut record_for_series = Vec::with_capacity(series_count);
        for series in 0..series_count {
            let positions = (0..self.time_count())
                .filter(|time| observations[time * series_count + series].is_finite())
                .collect::<Vec<_>>();
            let record_index = if let Some(index) = record_by_positions.get(&positions) {
                *index
            } else {
                let index = records.len();
                records.push(self.basis.record_subset(positions.clone())?);
                record_by_positions.insert(positions, index);
                index
            };
            record_for_series.push(record_index);
        }

        (0..series_count)
            .into_par_iter()
            .map(|series| {
                let record = &records[record_for_series[series]];
                let model = self
                    .basis
                    .model_at_latitude_for_record(latitudes[series], record)?;
                let series_observations = record
                    .positions
                    .iter()
                    .copied()
                    .map(|time| observations[time * series_count + series])
                    .collect::<Vec<_>>();
                match confidence {
                    Some(noise) => model.solve_with_linear_confidence(&series_observations, noise),
                    None => model.solve(&series_observations),
                }
            })
            .collect()
    }

    fn solve_time_major_impl(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        confidence: Option<LinearConfidence>,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        validate_batch_shape_and_latitudes(self.time_count(), observations, latitudes)?;
        let series_count = latitudes.len();
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
    base_constituents: Vec<TidalConstituent>,
    recipes: Vec<CorrectionRecipe>,
    time_terms: Vec<TimeTerms>,
    reference_time_modified_julian_day: f64,
    time_span_days: f64,
    sample_interval_hours: Option<f64>,
}

#[derive(Clone, Debug)]
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
        let scalar_constituents = scalar_constituents_at_reference(
            constituents,
            &base_constituents,
            &recipes,
            reference_time,
        );
        validate_derived_frequencies(&scalar_constituents)?;
        let time_terms = modified_julian_days
            .iter()
            .copied()
            .map(|time| {
                let astronomy = at_modified_julian_day(time);
                let base_greenwich_phase = base_constituents
                    .iter()
                    .copied()
                    .map(|constituent| base_greenwich_phase(constituent, astronomy.cycles))
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
                    modified_julian_day: time,
                }
            })
            .collect();
        Ok(Self {
            tidal_constituents: constituents.to_vec(),
            scalar_constituents,
            base_constituents,
            recipes,
            time_terms,
            reference_time_modified_julian_day: reference_time,
            time_span_days,
            sample_interval_hours: equidistant_sample_interval_hours(modified_julian_days),
        })
    }

    fn model_at_latitude(&self, latitude: f64) -> Result<FixedRawOls, AnalysisError> {
        let positions = (0..self.time_terms.len()).collect::<Vec<_>>();
        let record = self.record_subset(positions)?;
        self.model_at_latitude_for_record(latitude, &record)
    }

    fn record_subset(&self, positions: Vec<usize>) -> Result<RecordSubset, AnalysisError> {
        let modified_julian_days = positions
            .iter()
            .copied()
            .map(|position| self.time_terms[position].modified_julian_day)
            .collect::<Vec<_>>();
        let (subset_reference, subset_span) =
            validate_time(&modified_julian_days, self.tidal_constituents.len())?;
        let original_is_equidistant = self.sample_interval_hours.is_some();
        let (reference_time, time_span_days, scalar_constituents, confidence_sampling) =
            if original_is_equidistant {
                let full_count = self.time_terms.len();
                let full_count_f64 = usize_to_f64(full_count);
                (
                    self.reference_time_modified_julian_day,
                    self.time_span_days,
                    self.scalar_constituents.clone(),
                    ConfidenceSampling {
                        sample_interval_hours: self.sample_interval_hours,
                        effective_record_length_days: self.time_span_days * full_count_f64
                            / (full_count_f64 - 1.0),
                        spectrum_time_count: full_count,
                        observation_positions: (positions.len() != full_count)
                            .then_some(positions.clone()),
                    },
                )
            } else {
                let count = positions.len();
                let count_f64 = usize_to_f64(count);
                (
                    subset_reference,
                    subset_span,
                    scalar_constituents_at_reference(
                        &self.tidal_constituents,
                        &self.base_constituents,
                        &self.recipes,
                        subset_reference,
                    ),
                    ConfidenceSampling {
                        sample_interval_hours: None,
                        effective_record_length_days: subset_span * count_f64 / (count_f64 - 1.0),
                        spectrum_time_count: count,
                        observation_positions: None,
                    },
                )
            };
        validate_derived_frequencies(&scalar_constituents)?;
        Ok(RecordSubset {
            positions,
            scalar_constituents,
            reference_time,
            time_span_days,
            confidence_sampling,
        })
    }

    fn model_at_latitude_for_record(
        &self,
        latitude: f64,
        record: &RecordSubset,
    ) -> Result<FixedRawOls, AnalysisError> {
        validate_latitude(latitude)?;
        let harmonic_columns = self.tidal_constituents.len() * 2;
        let mut design = Mat::zeros(record.positions.len(), harmonic_columns + 2);
        let latitude_factors = latitude_factors(latitude);
        for (time_index, position) in record.positions.iter().copied().enumerate() {
            let terms = &self.time_terms[position];
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
            design[(time_index, harmonic_columns + 1)] =
                (terms.modified_julian_day - record.reference_time) / record.time_span_days;
        }
        Ok(FixedRawOls::from_design(
            record.scalar_constituents.clone(),
            record.positions.len(),
            record.time_span_days,
            record.reference_time,
            record.confidence_sampling.clone(),
            design,
        ))
    }
}

#[derive(Debug)]
struct TimeTerms {
    greenwich_phase: Vec<f64>,
    base_nodal_terms: Vec<NodalTerms>,
    modified_julian_day: f64,
}

#[derive(Clone, Debug)]
struct RecordSubset {
    positions: Vec<usize>,
    scalar_constituents: Vec<Constituent>,
    reference_time: f64,
    time_span_days: f64,
    confidence_sampling: ConfidenceSampling,
}

#[derive(Debug)]
struct ReconstructionTimeTerms {
    greenwich_phase: Vec<f64>,
    base_nodal_terms: Vec<NodalTerms>,
    modified_julian_day: f64,
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

fn scalar_constituents_at_reference(
    constituents: &[TidalConstituent],
    base_constituents: &[TidalConstituent],
    recipes: &[CorrectionRecipe],
    reference_time: f64,
) -> Vec<Constituent> {
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
    constituents
        .iter()
        .copied()
        .zip(recipes)
        .map(|(constituent, recipe)| {
            Constituent::new(
                constituent.name(),
                recipe.combine_frequency(&base_frequencies),
            )
        })
        .collect()
}

fn validate_reconstruction_times(
    modified_julian_days: &[f64],
    reference_time_modified_julian_day: f64,
) -> Result<(), AnalysisError> {
    if !reference_time_modified_julian_day.is_finite() {
        return Err(AnalysisError::NonFiniteReferenceTime);
    }
    if modified_julian_days.is_empty() {
        return Err(AnalysisError::EmptyTime);
    }
    for (index, time) in modified_julian_days.iter().copied().enumerate() {
        if !time.is_finite() {
            return Err(AnalysisError::NonFiniteTime { index });
        }
    }
    Ok(())
}

fn reconstruction_indices(
    constituents: &[TidalConstituent],
    solution: &ScalarSolution,
    filter: &ReconstructionFilter,
) -> Result<Vec<usize>, AnalysisError> {
    let expected = constituents.len();
    for (field, actual) in [
        ("amplitude", solution.amplitude.len()),
        ("phase_degrees", solution.phase_degrees.len()),
        ("percent_energy", solution.percent_energy.len()),
    ] {
        if actual != expected {
            return Err(AnalysisError::InvalidSolutionShape {
                field,
                actual,
                expected,
            });
        }
    }

    match filter {
        ReconstructionFilter::All => Ok((0..expected).collect()),
        ReconstructionFilter::Constituents(requested) => {
            for (index, constituent) in requested.iter().copied().enumerate() {
                if requested[..index].contains(&constituent) {
                    return Err(AnalysisError::DuplicateReconstructionConstituent { index });
                }
                if !constituents.contains(&constituent) {
                    return Err(AnalysisError::UnpreparedReconstructionConstituent {
                        name: constituent.name(),
                    });
                }
            }
            Ok(constituents
                .iter()
                .enumerate()
                .filter_map(|(index, constituent)| requested.contains(constituent).then_some(index))
                .collect())
        }
        ReconstructionFilter::Diagnostics {
            minimum_percent_energy,
            minimum_signal_to_noise,
        } => {
            validate_reconstruction_threshold("percent-energy", *minimum_percent_energy)?;
            let signal_to_noise = match minimum_signal_to_noise {
                Some(minimum) => {
                    validate_reconstruction_threshold("signal-to-noise", *minimum)?;
                    let values = solution
                        .signal_to_noise
                        .as_deref()
                        .ok_or(AnalysisError::MissingSignalToNoise)?;
                    if values.len() != expected {
                        return Err(AnalysisError::InvalidSolutionShape {
                            field: "signal_to_noise",
                            actual: values.len(),
                            expected,
                        });
                    }
                    Some((*minimum, values))
                }
                None => None,
            };
            Ok((0..expected)
                .filter(|index| {
                    solution.percent_energy[*index] >= *minimum_percent_energy
                        && signal_to_noise.is_none_or(|(minimum, values)| values[*index] >= minimum)
                })
                .collect())
        }
    }
}

fn validate_reconstruction_threshold(
    diagnostic: &'static str,
    threshold: f64,
) -> Result<(), AnalysisError> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(AnalysisError::InvalidReconstructionThreshold { diagnostic });
    }
    Ok(())
}

fn base_greenwich_phase(constituent: TidalConstituent, astronomy: [f64; 6]) -> f64 {
    let metadata = constituent.metadata();
    (dot6(
        metadata
            .doodson
            .expect("base catalog constituent has Doodson multipliers"),
        astronomy,
    ) + metadata
        .semi
        .expect("base catalog constituent has a phase offset"))
        % 1.0
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

fn validate_batch_shape_and_latitudes(
    time_count: usize,
    observations: &[f64],
    latitudes: &[f64],
) -> Result<(), AnalysisError> {
    if latitudes.is_empty() {
        return Err(AnalysisError::EmptySeries);
    }
    for latitude in latitudes.iter().copied() {
        validate_latitude(latitude)?;
    }
    let expected = time_count.saturating_mul(latitudes.len());
    if observations.len() != expected {
        return Err(AnalysisError::ObservationShape {
            actual: observations.len(),
            expected,
        });
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

#[allow(
    clippy::cast_precision_loss,
    reason = "practical record lengths are exactly representable as f64"
)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::{GreenwichNodalBatch, GreenwichNodalOls, usize_to_f64};
    use crate::{AnalysisError, LinearConfidence, TidalConstituent};

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

    #[test]
    fn irregular_gappy_records_support_white_but_reject_colored_confidence() {
        let mut time = times();
        time[20] += 0.001;
        let constituents = [TidalConstituent::M2, TidalConstituent::K1];
        let mut observations = (0_u32..745)
            .map(|index| (f64::from(index) / 13.0).sin())
            .collect::<Vec<_>>();
        observations[0] = f64::NAN;
        let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &constituents)
            .expect("valid irregular batch");
        let white = batch
            .solve_time_major_with_missing_and_linear_confidence(
                &observations,
                &[60.0],
                LinearConfidence::White,
            )
            .expect("white confidence supports irregular gappy data");
        assert!(
            (white[0].reference_time_days - time[1].midpoint(time[time.len() - 1])).abs() < 1e-12
        );
        assert_eq!(
            batch.solve_time_major_with_missing_and_linear_confidence(
                &observations,
                &[60.0],
                LinearConfidence::Colored,
            ),
            Err(AnalysisError::UnevenTimeForColoredConfidence)
        );

        let mut northward = observations
            .iter()
            .enumerate()
            .map(|(index, value)| value + (usize_to_f64(index) / 17.0).cos())
            .collect::<Vec<_>>();
        northward[2] = f64::NAN;
        assert!(
            batch
                .solve_vector_time_major_with_missing_and_linear_confidence(
                    &observations,
                    &northward,
                    &[60.0],
                    LinearConfidence::White,
                )
                .is_ok()
        );
        assert_eq!(
            batch.solve_vector_time_major_with_missing_and_linear_confidence(
                &observations,
                &northward,
                &[60.0],
                LinearConfidence::Colored,
            ),
            Err(AnalysisError::UnevenTimeForColoredConfidence)
        );
    }

    #[test]
    fn vector_missing_values_use_a_joint_component_mask() {
        let time = times();
        let constituents = [TidalConstituent::M2, TidalConstituent::K1];
        let mut eastward = (0_u32..745)
            .map(|index| 0.2 + (f64::from(index) / 13.0).sin())
            .collect::<Vec<_>>();
        let mut northward = (0_u32..745)
            .map(|index| -0.1 + (f64::from(index) / 17.0).cos())
            .collect::<Vec<_>>();
        eastward[7] = f64::NAN;
        northward[19] = f64::NAN;

        let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &constituents)
            .expect("valid batch");
        let actual = batch
            .solve_vector_time_major_with_missing(&eastward, &northward, &[60.0])
            .expect("valid gappy vector");

        let retained = (0..time.len())
            .filter(|index| eastward[*index].is_finite() && northward[*index].is_finite())
            .collect::<Vec<_>>();
        let retained_time = retained
            .iter()
            .map(|index| time[*index])
            .collect::<Vec<_>>();
        let retained_eastward = retained
            .iter()
            .map(|index| eastward[*index])
            .collect::<Vec<_>>();
        let retained_northward = retained
            .iter()
            .map(|index| northward[*index])
            .collect::<Vec<_>>();
        let individual =
            GreenwichNodalOls::prepare_modified_julian_days(&retained_time, 60.0, &constituents)
                .expect("valid retained model")
                .solve_vector(&retained_eastward, &retained_northward)
                .expect("valid retained vector");
        assert_eq!(actual, [individual]);
    }
}
