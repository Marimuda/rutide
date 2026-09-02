//! Configurable astronomical phase and exact nodal corrections for catalog constituents.

use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
    f64::consts::{PI, TAU},
};

use faer::{
    Mat, c64,
    linalg::solvers::{ColPivQr, DenseSolveCore, SolveLstsq},
};
use rayon::prelude::*;

use crate::{
    AnalysisError, Constituent, FitOptions, FixedRawOls, LinearConfidence, MonteCarloOptions,
    RobustDiagnostics, RobustOptions, ScalarSolution, TidalConstituent, VectorReconstruction,
    VectorSolution,
    astronomy::at_modified_julian_day,
    catalog::{CONSTITUENT_COUNT, Metadata},
    monte_carlo::{scalar_transformed_intervals, vector_transformed_intervals},
    robust::fit_complex_with_initial as robust_complex_fit_with_initial,
    sampling::equidistant_sample_interval_hours,
    scalar::{ConfidenceSampling, constituent_stream, validate_time_with_options},
    vector::{from_component_solutions, linearized_ellipse_sigmas},
};

// Plan construction costs enough to require a moderately reused mask, while
// bounding the group count prevents many distinct repeated masks from turning
// the speed optimization into unbounded batch memory growth.
const MIN_SHARED_LOMB_SERIES: usize = 16;
const MAX_SHARED_LOMB_PLANS_PER_BATCH: usize = 4;

/// Astronomical argument used to reference reported constituent phases.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PhaseReference {
    /// Evaluate the exact Greenwich astronomical argument at every timestamp.
    #[default]
    Greenwich,
    /// Evaluate the Greenwich argument at the fit epoch and advance it with the
    /// constituent's linearized reference-time frequency.
    LinearTime,
    /// Reference phase directly to the fit epoch, without a Greenwich argument.
    Raw,
}

impl PhaseReference {
    /// Stable machine-readable name used by reports and serialized outputs.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Greenwich => "greenwich",
            Self::LinearTime => "linear-time",
            Self::Raw => "raw",
        }
    }
}

/// Nodal/satellite amplitude and phase corrections applied to each constituent.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NodalCorrections {
    /// Evaluate time-varying nodal/satellite corrections at every timestamp.
    #[default]
    Exact,
    /// Evaluate corrections once at the fitted record midpoint and hold them
    /// constant across that record.
    LinearTime,
    /// Omit nodal/satellite corrections, equivalent to unit amplitude and zero
    /// nodal phase.
    Disabled,
}

impl NodalCorrections {
    /// Stable machine-readable name used by reports and serialized outputs.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::LinearTime => "linear-time",
            Self::Disabled => "disabled",
        }
    }
}

/// Solver configuration shared by the corrected scalar and vector APIs.
///
/// The fields are private so future solver choices can be added without
/// requiring callers to update struct literals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SolverOptions {
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
}

impl SolverOptions {
    /// Construct a corrected solver configuration.
    #[must_use]
    pub const fn new(fit_options: FitOptions, phase_reference: PhaseReference) -> Self {
        Self {
            fit_options,
            phase_reference,
            nodal_corrections: NodalCorrections::Exact,
        }
    }

    /// Return the fitted mean/trend configuration.
    #[must_use]
    pub const fn fit_options(self) -> FitOptions {
        self.fit_options
    }

    /// Return the astronomical phase-reference convention.
    #[must_use]
    pub const fn phase_reference(self) -> PhaseReference {
        self.phase_reference
    }

    /// Set the nodal/satellite correction convention.
    #[must_use]
    pub const fn with_nodal_corrections(mut self, nodal_corrections: NodalCorrections) -> Self {
        self.nodal_corrections = nodal_corrections;
        self
    }

    /// Return the nodal/satellite correction convention.
    #[must_use]
    pub const fn nodal_corrections(self) -> NodalCorrections {
        self.nodal_corrections
    }
}

impl From<FitOptions> for SolverOptions {
    fn from(fit_options: FitOptions) -> Self {
        Self::new(fit_options, PhaseReference::Greenwich)
    }
}

#[derive(Clone, Copy)]
enum BatchConfidence {
    Linear(LinearConfidence),
    MonteCarlo {
        options: MonteCarloOptions,
        noise: LinearConfidence,
    },
}

impl BatchConfidence {
    const fn noise(self) -> LinearConfidence {
        match self {
            Self::Linear(noise) | Self::MonteCarlo { noise, .. } => noise,
        }
    }
}

fn solve_scalar_with_batch_confidence(
    model: &FixedRawOls,
    observations: &[f64],
    robust: Option<RobustOptions>,
    confidence: Option<BatchConfidence>,
    stream: u64,
) -> Result<ScalarSolution, AnalysisError> {
    match (robust, confidence) {
        (Some(options), Some(BatchConfidence::Linear(noise))) => {
            model.solve_robust_with_linear_confidence(observations, options, noise)
        }
        (Some(robust_options), Some(BatchConfidence::MonteCarlo { options, noise })) => model
            .solve_robust_with_monte_carlo_confidence_from_stream(
                observations,
                robust_options,
                options,
                noise,
                stream,
            ),
        (Some(options), None) => model.solve_robust(observations, options),
        (None, Some(BatchConfidence::Linear(noise))) => {
            model.solve_with_linear_confidence(observations, noise)
        }
        (None, Some(BatchConfidence::MonteCarlo { options, noise })) => model
            .solve_many_time_major_with_monte_carlo_confidence_from_stream(
                observations,
                1,
                options,
                noise,
                stream,
            )?
            .pop()
            .ok_or(AnalysisError::EmptySeries),
        (None, None) => model.solve(observations),
    }
}

fn solve_vector_with_batch_confidence(
    model: &FixedRawOls,
    observations: &[f64],
    robust: Option<RobustOptions>,
    confidence: Option<BatchConfidence>,
    stream: u64,
) -> Result<VectorSolution, AnalysisError> {
    match (robust, confidence) {
        (Some(options), Some(BatchConfidence::Linear(noise))) => {
            let mut components = model.solve_two_component_robust_with_linear_confidence(
                observations,
                options,
                noise,
            )?;
            let northward = components.pop().expect("two requested component solutions");
            let eastward = components.pop().expect("two requested component solutions");
            from_component_solutions(&eastward, &northward)
        }
        (Some(robust_options), Some(BatchConfidence::MonteCarlo { options, noise })) => model
            .solve_vector_robust_with_monte_carlo_confidence(
                observations,
                robust_options,
                options,
                noise,
                stream,
            ),
        (Some(options), None) => {
            let mut components = model.solve_two_component_robust(observations, options)?;
            let northward = components.pop().expect("two requested component solutions");
            let eastward = components.pop().expect("two requested component solutions");
            from_component_solutions(&eastward, &northward)
        }
        (None, Some(BatchConfidence::Linear(noise))) => {
            let mut components = model.solve_many_time_major_with_linear_confidence_by_series(
                observations,
                &[LinearConfidence::White, noise],
            )?;
            let northward = components.pop().expect("two requested component solutions");
            let eastward = components.pop().expect("two requested component solutions");
            from_component_solutions(&eastward, &northward)
        }
        (None, Some(BatchConfidence::MonteCarlo { options, noise })) => {
            model.solve_vector_with_monte_carlo_confidence(observations, options, noise, stream)
        }
        (None, None) => {
            let mut components = model.solve_many_time_major(observations, 2)?;
            let northward = components.pop().expect("two requested component solutions");
            let eastward = components.pop().expect("two requested component solutions");
            from_component_solutions(&eastward, &northward)
        }
    }
}

fn solve_vector_inference_with_batch_confidence(
    model: &VectorInferenceOls,
    eastward: &[f64],
    northward: &[f64],
    robust: Option<RobustOptions>,
    confidence: Option<BatchConfidence>,
    stream: u64,
) -> Result<VectorSolution, AnalysisError> {
    match (robust, confidence) {
        (Some(options), Some(BatchConfidence::Linear(noise))) => {
            model.solve_vector_robust_with_linear_confidence(eastward, northward, options, noise)
        }
        (Some(robust_options), Some(BatchConfidence::MonteCarlo { options, noise })) => model
            .solve_vector_robust_with_monte_carlo_confidence_from_stream(
                eastward,
                northward,
                robust_options,
                options,
                noise,
                stream,
            ),
        (Some(options), None) => model.solve_vector_robust(eastward, northward, options),
        (None, Some(BatchConfidence::Linear(noise))) => {
            model.solve_vector_with_linear_confidence(eastward, northward, noise)
        }
        (None, Some(BatchConfidence::MonteCarlo { options, noise })) => model
            .solve_vector_with_monte_carlo_confidence_from_stream(
                eastward, northward, options, noise, stream,
            ),
        (None, None) => model.solve_vector(eastward, northward),
    }
}

fn shared_lomb_plan_groups(record_use_count: &[usize]) -> Vec<bool> {
    let mut candidates = record_use_count
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, use_count)| *use_count >= MIN_SHARED_LOMB_SERIES)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(index, use_count)| (Reverse(*use_count), *index));
    let mut shared = vec![false; record_use_count.len()];
    for (index, _) in candidates.into_iter().take(MAX_SHARED_LOMB_PLANS_PER_BATCH) {
        shared[index] = true;
    }
    shared
}

/// A reusable fixed-constituent OLS model with configurable phase and exact nodal terms.
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
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    model: FixedRawOls,
}

/// Basis treatment used while fitting inferred constituents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceMode {
    /// Include the inferred constituent's exact astronomical basis in the
    /// constrained reference columns.
    Exact,
    /// Reproduce Python `UTide`'s reference-only approximate fit.
    Approximate,
}

/// One scalar inferred/reference amplitude and phase relationship.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarInferenceRelation {
    /// Constituent whose coefficients are constrained by the reference.
    pub inferred: TidalConstituent,
    /// Independently fitted reference constituent.
    pub reference: TidalConstituent,
    /// Non-negative inferred/reference amplitude ratio.
    pub amplitude_ratio: f64,
    /// Inferred positive-frequency phase offset in degrees.
    pub phase_offset_degrees: f64,
}

impl ScalarInferenceRelation {
    /// Construct one scalar inference relationship.
    #[must_use]
    pub const fn new(
        inferred: TidalConstituent,
        reference: TidalConstituent,
        amplitude_ratio: f64,
        phase_offset_degrees: f64,
    ) -> Self {
        Self {
            inferred,
            reference,
            amplitude_ratio,
            phase_offset_degrees,
        }
    }
}

/// A reusable corrected scalar model with constrained inferred constituents.
///
/// Reported constituents follow Python `UTide` order: ordinary requested
/// constituents, unique references in first-use order, then inferred
/// constituents grouped by reference. The QR factorization contains only the
/// ordinary constituents and unique references.
#[derive(Debug)]
pub struct ScalarInferenceOls {
    tidal_constituents: Vec<TidalConstituent>,
    constituents: Vec<Constituent>,
    relationships: Vec<ScalarInferenceRelation>,
    mode: InferenceMode,
    output_mappings: Vec<ScalarInferenceOutput>,
    latitude_degrees_north: f64,
    reference_time_modified_julian_day: f64,
    base_constituents: Vec<TidalConstituent>,
    recipes: Vec<CorrectionRecipe>,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    model: FixedRawOls,
}

#[derive(Clone, Copy, Debug)]
struct ScalarInferenceOutput {
    source_fit_index: usize,
    ratio_real: f64,
    ratio_imaginary: f64,
    inferred: bool,
}

/// One vector inferred/reference positive- and negative-rotary relationship.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorInferenceRelation {
    /// Constituent whose ellipse is constrained by the reference.
    pub inferred: TidalConstituent,
    /// Independently fitted reference constituent.
    pub reference: TidalConstituent,
    /// Inferred/reference positive-rotary amplitude ratio.
    pub positive_amplitude_ratio: f64,
    /// Positive-rotary phase offset in degrees.
    pub positive_phase_offset_degrees: f64,
    /// Inferred/reference negative-rotary amplitude ratio.
    pub negative_amplitude_ratio: f64,
    /// Negative-rotary phase offset in degrees.
    pub negative_phase_offset_degrees: f64,
}

impl VectorInferenceRelation {
    /// Construct one vector inference relationship.
    #[must_use]
    pub const fn new(
        inferred: TidalConstituent,
        reference: TidalConstituent,
        positive_amplitude_ratio: f64,
        positive_phase_offset_degrees: f64,
        negative_amplitude_ratio: f64,
        negative_phase_offset_degrees: f64,
    ) -> Self {
        Self {
            inferred,
            reference,
            positive_amplitude_ratio,
            positive_phase_offset_degrees,
            negative_amplitude_ratio,
            negative_phase_offset_degrees,
        }
    }
}

/// A reusable coupled vector model with constrained inferred constituents.
#[derive(Debug)]
pub struct VectorInferenceOls {
    tidal_constituents: Vec<TidalConstituent>,
    constituents: Vec<Constituent>,
    relationships: Vec<VectorInferenceRelation>,
    mode: InferenceMode,
    output_mappings: Vec<VectorInferenceOutput>,
    non_reference_count: usize,
    latitude_degrees_north: f64,
    reference_time_modified_julian_day: f64,
    time_span_days: f64,
    time_count: usize,
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    base_constituents: Vec<TidalConstituent>,
    recipes: Vec<CorrectionRecipe>,
    confidence_sampling: ConfidenceSampling,
    design: Mat<c64>,
    decomposition: ColPivQr<c64>,
}

/// Shared astronomy for scalar inference across varying-latitude series.
#[derive(Debug)]
pub struct ScalarInferenceBatch {
    basis: CorrectionBasis,
    layout: ScalarInferenceLayout,
    relationships: Vec<ScalarInferenceRelation>,
    mode: InferenceMode,
}

/// Shared astronomy for coupled vector inference across varying-latitude series.
#[derive(Debug)]
pub struct VectorInferenceBatch {
    basis: CorrectionBasis,
    layout: VectorInferenceLayout,
    relationships: Vec<VectorInferenceRelation>,
    mode: InferenceMode,
}

#[derive(Clone, Copy, Debug)]
struct VectorInferenceOutput {
    source_fit_index: usize,
    positive_ratio: c64,
    negative_ratio: c64,
    inferred: bool,
}

#[derive(Debug)]
struct VectorInferenceIntervals {
    eastward_cosine_variance: Vec<f64>,
    eastward_sine_variance: Vec<f64>,
    northward_cosine_variance: Vec<f64>,
    northward_sine_variance: Vec<f64>,
    inferred_linearization: Vec<Option<(usize, f64, f64)>>,
}

struct VectorInferenceCovarianceBasis {
    fitted: Mat<c64>,
    gall: Mat<c64>,
    hall: Mat<c64>,
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
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            latitude_degrees_north,
            constituents,
            SolverOptions::default(),
        )
    }

    /// Build and factorize an exact corrected basis with explicit fit options.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when timestamps, latitude, constituents, or
    /// the requested model are invalid.
    pub fn prepare_modified_julian_days_with_options(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
        fit_options: FitOptions,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            latitude_degrees_north,
            constituents,
            fit_options.into(),
        )
    }

    /// Build and factorize a corrected basis with explicit solver options.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when timestamps, latitude, constituents, or
    /// the requested model are invalid.
    pub fn prepare_modified_julian_days_with_solver_options(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        validate_latitude(latitude_degrees_north)?;
        let basis = CorrectionBasis::prepare(modified_julian_days, constituents, solver_options)?;
        let model = basis.model_at_latitude(latitude_degrees_north)?;

        Ok(Self {
            tidal_constituents: constituents.to_vec(),
            latitude_degrees_north,
            reference_time_modified_julian_day: basis.reference_time_modified_julian_day,
            base_constituents: basis.base_constituents,
            recipes: basis.recipes,
            phase_reference: basis.phase_reference,
            nodal_corrections: basis.nodal_corrections,
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

    /// Return the configured non-harmonic fit terms.
    #[must_use]
    pub const fn fit_options(&self) -> FitOptions {
        self.model.fit_options()
    }

    /// Return the configured astronomical phase-reference convention.
    #[must_use]
    pub const fn phase_reference(&self) -> PhaseReference {
        self.phase_reference
    }

    /// Return the configured nodal/satellite correction convention.
    #[must_use]
    pub const fn nodal_corrections(&self) -> NodalCorrections {
        self.nodal_corrections
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
    /// Returns [`AnalysisError`] for invalid observations.
    pub fn solve_with_linear_confidence(
        &self,
        observations: &[f64],
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model.solve_with_linear_confidence(observations, noise)
    }

    /// Fit one series with reproducible nonlinear Monte Carlo confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid observations, options, or covariance.
    pub fn solve_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model
            .solve_with_monte_carlo_confidence(observations, options, noise)
    }

    /// Fit one scalar series with Cauchy iteratively reweighted least squares.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, degenerate robust scaling,
    /// invalid leverage, or non-convergence.
    pub fn solve_robust(
        &self,
        observations: &[f64],
        options: RobustOptions,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model.solve_robust(observations, options)
    }

    /// Robustly fit one series with linearized confidence intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or robust fitting failure.
    pub fn solve_robust_with_linear_confidence(
        &self,
        observations: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model
            .solve_robust_with_linear_confidence(observations, options, noise)
    }

    /// Robustly fit one series with nonlinear Monte Carlo confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, robust fitting failure,
    /// invalid Monte Carlo options, or an unsampleable covariance.
    pub fn solve_robust_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model.solve_robust_with_monte_carlo_confidence(
            observations,
            robust_options,
            monte_carlo_options,
            noise,
        )
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
    /// Returns [`AnalysisError`] for invalid observations.
    pub fn solve_many_time_major_with_linear_confidence(
        &self,
        observations: &[f64],
        series_count: usize,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.model
            .solve_many_time_major_with_linear_confidence(observations, series_count, noise)
    }

    /// Fit complete series with reproducible nonlinear Monte Carlo intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid shapes, values, options, or covariance.
    pub fn solve_many_time_major_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        series_count: usize,
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.model
            .solve_many_time_major_with_monte_carlo_confidence(
                observations,
                series_count,
                options,
                noise,
            )
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
    /// Returns [`AnalysisError`] for invalid inputs.
    pub fn solve_vector_with_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        noise: LinearConfidence,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(eastward, northward, Some(noise))
    }

    /// Jointly fit one current series with nonlinear Monte Carlo ellipse intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid components, options, or covariance.
    pub fn solve_vector_with_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
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
        self.model
            .solve_vector_with_monte_carlo_confidence(&time_major, options, noise, 0)
    }

    /// Jointly fit one current series with shared Cauchy robust weights.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid components or robust fitting failure.
    pub fn solve_vector_robust(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: RobustOptions,
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
        let mut components = self
            .model
            .solve_two_component_robust(&time_major, options)?
            .into_iter();
        let eastward = components.next().ok_or(AnalysisError::EmptySeries)?;
        let northward = components.next().ok_or(AnalysisError::EmptySeries)?;
        from_component_solutions(&eastward, &northward)
    }

    /// Jointly robustly fit one current series with linearized ellipse intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or robust fitting failure.
    pub fn solve_vector_robust_with_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
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
        let mut components = self
            .model
            .solve_two_component_robust_with_linear_confidence(&time_major, options, noise)?
            .into_iter();
        let eastward = components.next().ok_or(AnalysisError::EmptySeries)?;
        let northward = components.next().ok_or(AnalysisError::EmptySeries)?;
        from_component_solutions(&eastward, &northward)
    }

    /// Jointly robustly fit one current series with Monte Carlo ellipse intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, robust fitting failure,
    /// invalid Monte Carlo options, or an unsampleable covariance.
    pub fn solve_vector_robust_with_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
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
        self.model.solve_vector_robust_with_monte_carlo_confidence(
            &time_major,
            robust_options,
            monte_carlo_options,
            noise,
            0,
        )
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
    /// The configured phase convention and nodal-correction mode are evaluated
    /// at each target timestamp. The fitted mean, and the trend when enabled,
    /// are retained.
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
            self.phase_reference,
            self.nodal_corrections,
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
            self.phase_reference,
            self.nodal_corrections,
        )?
        .reconstruct_vector_at_latitude(solution, self.latitude_degrees_north, filter)
    }
}

impl ScalarInferenceOls {
    /// Build and factorize a scalar inferred-constituent model from Modified
    /// Julian Days.
    ///
    /// Constituents named anywhere in the relationships are removed from the
    /// ordinary requested list. References and inferred constituents are then
    /// appended in Python `UTide`'s stable grouping order.
    ///
    /// # Errors
    ///
    /// Returns `AnalysisError` for invalid timestamps, latitude, constituents,
    /// ratios, offsets, duplicate inferred constituents, or reference
    /// chains/cycles.
    pub fn prepare_modified_julian_days(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
        relationships: &[ScalarInferenceRelation],
        mode: InferenceMode,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            latitude_degrees_north,
            constituents,
            relationships,
            mode,
            SolverOptions::default(),
        )
    }

    /// Build a scalar inferred-constituent model with explicit fit options.
    ///
    /// # Errors
    ///
    /// Returns `AnalysisError` for invalid input, inference relationships, or
    /// an underdetermined requested model.
    pub fn prepare_modified_julian_days_with_options(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
        relationships: &[ScalarInferenceRelation],
        mode: InferenceMode,
        fit_options: FitOptions,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            latitude_degrees_north,
            constituents,
            relationships,
            mode,
            fit_options.into(),
        )
    }

    /// Build a scalar inferred-constituent model with explicit solver options.
    ///
    /// # Errors
    ///
    /// Returns `AnalysisError` for invalid input, inference relationships, or
    /// an underdetermined requested model.
    pub fn prepare_modified_julian_days_with_solver_options(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
        relationships: &[ScalarInferenceRelation],
        mode: InferenceMode,
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        validate_latitude(latitude_degrees_north)?;
        let layout = scalar_inference_layout(constituents, relationships)?;
        let basis = CorrectionBasis::prepare_with_model_count(
            modified_julian_days,
            &layout.tidal_constituents,
            layout.fit_count,
            solver_options,
        )?;
        let record = basis.record_subset((0..basis.time_terms.len()).collect(), true)?;
        Self::from_basis_record(
            &basis,
            &record,
            latitude_degrees_north,
            &layout,
            relationships,
            mode,
        )
    }

    fn from_basis_record(
        basis: &CorrectionBasis,
        record: &RecordSubset,
        latitude_degrees_north: f64,
        layout: &ScalarInferenceLayout,
        relationships: &[ScalarInferenceRelation],
        mode: InferenceMode,
    ) -> Result<Self, AnalysisError> {
        let model = basis.scalar_inference_model_at_latitude_for_record(
            latitude_degrees_north,
            layout,
            mode,
            record,
        )?;
        Ok(Self {
            tidal_constituents: layout.tidal_constituents.clone(),
            constituents: record.scalar_constituents.clone(),
            relationships: relationships.to_vec(),
            mode,
            output_mappings: layout.output_mappings.clone(),
            latitude_degrees_north,
            reference_time_modified_julian_day: record.reference_time,
            base_constituents: basis.base_constituents.clone(),
            recipes: basis.recipes.clone(),
            phase_reference: basis.phase_reference,
            nodal_corrections: basis.nodal_corrections,
            model,
        })
    }

    /// Return every reported constituent in coefficient order.
    #[must_use]
    pub fn tidal_constituents(&self) -> &[TidalConstituent] {
        &self.tidal_constituents
    }

    /// Return reported names and reference-time frequencies.
    #[must_use]
    pub fn constituents(&self) -> &[Constituent] {
        &self.constituents
    }

    /// Return inference relationships in caller-supplied order.
    #[must_use]
    pub fn relationships(&self) -> &[ScalarInferenceRelation] {
        &self.relationships
    }

    /// Return whether exact or Python-compatible approximate inference is used.
    #[must_use]
    pub const fn mode(&self) -> InferenceMode {
        self.mode
    }

    /// Return the prepared latitude.
    #[must_use]
    pub const fn latitude_degrees_north(&self) -> f64 {
        self.latitude_degrees_north
    }

    /// Return the fitted trend epoch as a Modified Julian Day.
    #[must_use]
    pub const fn reference_time_modified_julian_day(&self) -> f64 {
        self.reference_time_modified_julian_day
    }

    /// Return the expected observation count.
    #[must_use]
    pub const fn time_count(&self) -> usize {
        self.model.time_count()
    }

    /// Return the configured non-harmonic fit terms.
    #[must_use]
    pub const fn fit_options(&self) -> FitOptions {
        self.model.fit_options()
    }

    /// Return the configured astronomical phase-reference convention.
    #[must_use]
    pub const fn phase_reference(&self) -> PhaseReference {
        self.phase_reference
    }

    /// Return the configured nodal/satellite correction convention.
    #[must_use]
    pub const fn nodal_corrections(&self) -> NodalCorrections {
        self.nodal_corrections
    }

    /// Fit one finite scalar series and expand inferred coefficients.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid observation shape or value.
    pub fn solve(&self, observations: &[f64]) -> Result<ScalarSolution, AnalysisError> {
        self.model
            .solve(observations)
            .map(|solution| self.expand_solution(solution))
    }

    /// Fit one scalar series with nonlinear Monte Carlo confidence intervals,
    /// propagating every reference draw through its inference relationships.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid observations, options, or covariance.
    pub fn solve_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.solve_with_monte_carlo_confidence_impl(observations, None, options, noise, 0)
    }

    /// Fit one scalar series with linear confidence intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid observation shape or value.
    pub fn solve_with_linear_confidence(
        &self,
        observations: &[f64],
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model
            .solve_with_linear_confidence(observations, noise)
            .map(|solution| self.expand_solution(solution))
    }

    /// Fit one scalar series with Cauchy IRLS.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid observations, options, or convergence.
    pub fn solve_robust(
        &self,
        observations: &[f64],
        options: RobustOptions,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model
            .solve_robust(observations, options)
            .map(|solution| self.expand_solution(solution))
    }

    /// Robustly fit one scalar series with linear confidence intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid observations, options, or convergence.
    pub fn solve_robust_with_linear_confidence(
        &self,
        observations: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.model
            .solve_robust_with_linear_confidence(observations, options, noise)
            .map(|solution| self.expand_solution(solution))
    }

    /// Robustly fit one scalar series with inferred Monte Carlo intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid observations, robust or Monte Carlo
    /// options, convergence, or covariance.
    pub fn solve_robust_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.solve_with_monte_carlo_confidence_impl(
            observations,
            Some(robust_options),
            monte_carlo_options,
            noise,
            0,
        )
    }

    /// Fit several complete time-major scalar series.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid series count, shape, or value.
    pub fn solve_many_time_major(
        &self,
        observations: &[f64],
        series_count: usize,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.model
            .solve_many_time_major(observations, series_count)
            .map(|solutions| {
                solutions
                    .into_iter()
                    .map(|solution| self.expand_solution(solution))
                    .collect()
            })
    }

    /// Fit several complete time-major series with linear confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid series count, shape, or value.
    pub fn solve_many_time_major_with_linear_confidence(
        &self,
        observations: &[f64],
        series_count: usize,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.model
            .solve_many_time_major_with_linear_confidence(observations, series_count, noise)
            .map(|solutions| {
                solutions
                    .into_iter()
                    .map(|solution| self.expand_solution(solution))
                    .collect()
            })
    }

    /// Reconstruct an inferred solution with the fitted astronomical options.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid target times, filters, or solution shapes.
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
            self.phase_reference,
            self.nodal_corrections,
        )?
        .reconstruct_at_latitude(solution, self.latitude_degrees_north, filter)
    }

    fn solve_with_monte_carlo_confidence_impl(
        &self,
        observations: &[f64],
        robust: Option<RobustOptions>,
        options: MonteCarloOptions,
        noise: LinearConfidence,
        stream: u64,
    ) -> Result<ScalarSolution, AnalysisError> {
        options.validate()?;
        let (base, covariances) = if let Some(robust) = robust {
            self.model
                .solve_robust_with_scalar_coefficient_covariances(observations, robust, noise)?
        } else {
            self.model
                .solve_with_scalar_coefficient_covariances(observations, noise)?
        };
        let mut solution = self.expand_solution(base);
        self.apply_monte_carlo_intervals(&mut solution, &covariances, options, stream)?;
        Ok(solution)
    }

    fn apply_monte_carlo_intervals(
        &self,
        solution: &mut ScalarSolution,
        covariances: &[[[f64; 2]; 2]],
        options: MonteCarloOptions,
        stream: u64,
    ) -> Result<(), AnalysisError> {
        let mut output_intervals = vec![None; self.output_mappings.len()];
        let mut cosine_variance = vec![0.0; self.output_mappings.len()];
        let mut sine_variance = vec![0.0; self.output_mappings.len()];
        for (source, covariance) in covariances.iter().copied().enumerate() {
            let mappings = self
                .output_mappings
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, mapping)| mapping.source_fit_index == source)
                .collect::<Vec<_>>();
            let ratios = mappings
                .iter()
                .map(|(_, mapping)| [mapping.ratio_real, mapping.ratio_imaginary])
                .collect::<Vec<_>>();
            let source_output = mappings
                .iter()
                .find_map(|(output, mapping)| (!mapping.inferred).then_some(*output))
                .expect("every fitted source retains its reported reference output");
            let intervals = scalar_transformed_intervals(
                [
                    solution.cosine_coefficient[source_output],
                    solution.sine_coefficient[source_output],
                ],
                covariance,
                &ratios,
                options,
                constituent_stream(stream, source),
            )
            .ok_or(AnalysisError::InvalidConfidenceCovariance {
                constituent: source,
            })?;
            for (((output, mapping), interval), ratio) in
                mappings.into_iter().zip(intervals).zip(ratios)
            {
                output_intervals[output] = Some(interval);
                let transformed = transform_scalar_covariance(covariance, ratio);
                cosine_variance[output] = transformed[0][0];
                sine_variance[output] = transformed[1][1];
                debug_assert_eq!(mapping.source_fit_index, source);
            }
        }
        let intervals = output_intervals
            .into_iter()
            .enumerate()
            .map(|(constituent, interval)| {
                interval.ok_or(AnalysisError::InvalidConfidenceCovariance { constituent })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let amplitude_ci = intervals
            .iter()
            .map(|interval| interval.amplitude)
            .collect::<Vec<_>>();
        solution.phase_ci_degrees = Some(
            intervals
                .iter()
                .map(|interval| interval.phase_degrees)
                .collect(),
        );
        solution.signal_to_noise = Some(
            solution
                .amplitude
                .iter()
                .zip(&amplitude_ci)
                .map(|(amplitude, interval)| amplitude.powi(2) / (interval / 1.96).powi(2))
                .collect(),
        );
        solution.amplitude_ci = Some(amplitude_ci);
        solution.cosine_coefficient_variance = Some(cosine_variance);
        solution.sine_coefficient_variance = Some(sine_variance);
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "keeps coefficient, diagnostic, and confidence expansion in one audited transform"
    )]
    fn expand_solution(&self, solution: ScalarSolution) -> ScalarSolution {
        let cosine_coefficient = self
            .output_mappings
            .iter()
            .map(|mapping| {
                let cosine = solution.cosine_coefficient[mapping.source_fit_index];
                let sine = solution.sine_coefficient[mapping.source_fit_index];
                mapping.ratio_real * cosine + mapping.ratio_imaginary * sine
            })
            .collect::<Vec<_>>();
        let sine_coefficient = self
            .output_mappings
            .iter()
            .map(|mapping| {
                let cosine = solution.cosine_coefficient[mapping.source_fit_index];
                let sine = solution.sine_coefficient[mapping.source_fit_index];
                -mapping.ratio_imaginary * cosine + mapping.ratio_real * sine
            })
            .collect::<Vec<_>>();
        let amplitude = cosine_coefficient
            .iter()
            .zip(&sine_coefficient)
            .map(|(cosine, sine)| cosine.hypot(*sine))
            .collect::<Vec<_>>();
        let phase_degrees = cosine_coefficient
            .iter()
            .zip(&sine_coefficient)
            .map(|(cosine, sine)| sine.atan2(*cosine).to_degrees().rem_euclid(360.0))
            .collect::<Vec<_>>();
        let total_energy = amplitude.iter().map(|value| value * value).sum::<f64>();
        let percent_energy = amplitude
            .iter()
            .map(|value| 100.0 * value * value / total_energy)
            .collect::<Vec<_>>();

        let expanded_confidence = match (
            solution.amplitude_ci.as_deref(),
            solution.phase_ci_degrees.as_deref(),
            solution.cosine_coefficient_variance.as_deref(),
            solution.sine_coefficient_variance.as_deref(),
        ) {
            (Some(base_amplitude), Some(base_phase), Some(base_cosine), Some(base_sine)) => {
                let mut amplitude_ci = Vec::with_capacity(self.output_mappings.len());
                let mut phase_ci_degrees = Vec::with_capacity(self.output_mappings.len());
                let mut cosine_variance = Vec::with_capacity(self.output_mappings.len());
                let mut sine_variance = Vec::with_capacity(self.output_mappings.len());
                for mapping in &self.output_mappings {
                    let source = mapping.source_fit_index;
                    if mapping.inferred {
                        // This intentionally matches Python UTide's inference
                        // propagation: inferred variance is derived from the
                        // reference positive/negative complex coefficients, and
                        // the linearization point remains the reference pair.
                        let ratio_real_squared = mapping.ratio_real.powi(2);
                        let ratio_imaginary_squared = mapping.ratio_imaginary.powi(2);
                        let variance_cosine = (ratio_real_squared * base_cosine[source])
                            .midpoint(ratio_imaginary_squared * base_sine[source]);
                        let variance_sine = (ratio_real_squared * base_sine[source])
                            .midpoint(ratio_imaginary_squared * base_cosine[source]);
                        let (amplitude, phase) = scalar_linear_intervals(
                            solution.cosine_coefficient[source],
                            solution.sine_coefficient[source],
                            variance_cosine,
                            variance_sine,
                        );
                        amplitude_ci.push(amplitude);
                        phase_ci_degrees.push(phase);
                        cosine_variance.push(variance_cosine);
                        sine_variance.push(variance_sine);
                    } else {
                        amplitude_ci.push(base_amplitude[source]);
                        phase_ci_degrees.push(base_phase[source]);
                        cosine_variance.push(base_cosine[source]);
                        sine_variance.push(base_sine[source]);
                    }
                }
                Some((
                    amplitude_ci,
                    phase_ci_degrees,
                    cosine_variance,
                    sine_variance,
                ))
            }
            _ => None,
        };
        let (amplitude_ci, phase_ci_degrees, signal_to_noise, cosine_variance, sine_variance) =
            match expanded_confidence {
                Some((amplitude_ci, phase_ci, cosine_variance, sine_variance)) => {
                    let signal_to_noise = amplitude
                        .iter()
                        .zip(&amplitude_ci)
                        .map(|(amplitude, interval)| amplitude.powi(2) / (interval / 1.96).powi(2))
                        .collect();
                    (
                        Some(amplitude_ci),
                        Some(phase_ci),
                        Some(signal_to_noise),
                        Some(cosine_variance),
                        Some(sine_variance),
                    )
                }
                None => (None, None, None, None, None),
            };

        ScalarSolution {
            cosine_coefficient,
            sine_coefficient,
            amplitude,
            phase_degrees,
            percent_energy,
            amplitude_ci,
            phase_ci_degrees,
            signal_to_noise,
            cosine_coefficient_variance: cosine_variance,
            sine_coefficient_variance: sine_variance,
            mean: solution.mean,
            slope_per_day: solution.slope_per_day,
            reference_time_days: solution.reference_time_days,
            robust: solution.robust,
        }
    }
}

fn transform_scalar_covariance(
    covariance: [[f64; 2]; 2],
    [ratio_real, ratio_imaginary]: [f64; 2],
) -> [[f64; 2]; 2] {
    let transform = [
        [ratio_real, ratio_imaginary],
        [-ratio_imaginary, ratio_real],
    ];
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..2)
                .flat_map(|left| {
                    (0..2).map(move |right| {
                        transform[row][left] * covariance[left][right] * transform[column][right]
                    })
                })
                .sum()
        })
    })
}

impl VectorInferenceOls {
    /// Build and factorize a coupled vector inference model.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid timestamps, latitude, constituents,
    /// positive/negative ratios, phase offsets, or reference graphs.
    pub fn prepare_modified_julian_days(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
        relationships: &[VectorInferenceRelation],
        mode: InferenceMode,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            latitude_degrees_north,
            constituents,
            relationships,
            mode,
            SolverOptions::default(),
        )
    }

    /// Build a coupled vector inference model with explicit fit options.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inputs, inference relationships, or an
    /// underdetermined requested model.
    pub fn prepare_modified_julian_days_with_options(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
        relationships: &[VectorInferenceRelation],
        mode: InferenceMode,
        fit_options: FitOptions,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            latitude_degrees_north,
            constituents,
            relationships,
            mode,
            fit_options.into(),
        )
    }

    /// Build a coupled vector inference model with explicit solver options.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inputs, inference relationships, or an
    /// underdetermined requested model.
    pub fn prepare_modified_julian_days_with_solver_options(
        modified_julian_days: &[f64],
        latitude_degrees_north: f64,
        constituents: &[TidalConstituent],
        relationships: &[VectorInferenceRelation],
        mode: InferenceMode,
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        validate_latitude(latitude_degrees_north)?;
        let layout = vector_inference_layout(constituents, relationships)?;
        let basis = CorrectionBasis::prepare_with_model_count(
            modified_julian_days,
            &layout.tidal_constituents,
            layout.fit_count,
            solver_options,
        )?;
        let record = basis.record_subset((0..basis.time_terms.len()).collect(), true)?;
        Self::from_basis_record(
            &basis,
            &record,
            latitude_degrees_north,
            &layout,
            relationships,
            mode,
        )
    }

    fn from_basis_record(
        basis: &CorrectionBasis,
        record: &RecordSubset,
        latitude_degrees_north: f64,
        layout: &VectorInferenceLayout,
        relationships: &[VectorInferenceRelation],
        mode: InferenceMode,
    ) -> Result<Self, AnalysisError> {
        let design = basis.vector_inference_design_at_latitude_for_record(
            latitude_degrees_north,
            layout,
            mode,
            record,
        )?;
        let decomposition = design.col_piv_qr();
        Ok(Self {
            tidal_constituents: layout.tidal_constituents.clone(),
            constituents: record.scalar_constituents.clone(),
            relationships: relationships.to_vec(),
            mode,
            output_mappings: layout.output_mappings.clone(),
            non_reference_count: layout.non_reference_count,
            latitude_degrees_north,
            reference_time_modified_julian_day: record.reference_time,
            time_span_days: record.time_span_days,
            time_count: record.positions.len(),
            fit_options: basis.fit_options,
            phase_reference: basis.phase_reference,
            nodal_corrections: basis.nodal_corrections,
            base_constituents: basis.base_constituents.clone(),
            recipes: basis.recipes.clone(),
            confidence_sampling: record.confidence_sampling.clone(),
            design,
            decomposition,
        })
    }

    /// Return every reported constituent in coefficient order.
    #[must_use]
    pub fn tidal_constituents(&self) -> &[TidalConstituent] {
        &self.tidal_constituents
    }

    /// Return reported names and reference-time frequencies.
    #[must_use]
    pub fn constituents(&self) -> &[Constituent] {
        &self.constituents
    }

    /// Return vector inference relationships in caller-supplied order.
    #[must_use]
    pub fn relationships(&self) -> &[VectorInferenceRelation] {
        &self.relationships
    }

    /// Return whether exact or Python-compatible approximate inference is used.
    #[must_use]
    pub const fn mode(&self) -> InferenceMode {
        self.mode
    }

    /// Return the prepared latitude.
    #[must_use]
    pub const fn latitude_degrees_north(&self) -> f64 {
        self.latitude_degrees_north
    }

    /// Return the fitted trend epoch as a Modified Julian Day.
    #[must_use]
    pub const fn reference_time_modified_julian_day(&self) -> f64 {
        self.reference_time_modified_julian_day
    }

    /// Return the expected observation count for each component.
    #[must_use]
    pub const fn time_count(&self) -> usize {
        self.time_count
    }

    /// Return the configured non-harmonic fit terms.
    #[must_use]
    pub const fn fit_options(&self) -> FitOptions {
        self.fit_options
    }

    /// Return the configured astronomical phase-reference convention.
    #[must_use]
    pub const fn phase_reference(&self) -> PhaseReference {
        self.phase_reference
    }

    /// Return the configured nodal/satellite correction convention.
    #[must_use]
    pub const fn nodal_corrections(&self) -> NodalCorrections {
        self.nodal_corrections
    }

    /// Fit one eastward/northward current series.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent shapes or non-finite observations.
    pub fn solve_vector(
        &self,
        eastward: &[f64],
        northward: &[f64],
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(eastward, northward, None, None, 0)
    }

    /// Fit one current series with linear ellipse confidence intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent shapes or non-finite observations.
    pub fn solve_vector_with_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        noise: LinearConfidence,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(
            eastward,
            northward,
            Some(BatchConfidence::Linear(noise)),
            None,
            0,
        )
    }

    /// Fit one inferred current series with nonlinear Monte Carlo ellipse
    /// confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid observations, options, or covariance.
    pub fn solve_vector_with_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_with_monte_carlo_confidence_from_stream(
            eastward, northward, options, noise, 0,
        )
    }

    fn solve_vector_with_monte_carlo_confidence_from_stream(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
        stream: u64,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(
            eastward,
            northward,
            Some(BatchConfidence::MonteCarlo { options, noise }),
            None,
            stream,
        )
    }

    /// Robustly fit one inferred current series with shared Cauchy weights.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or robust fitting failure.
    pub fn solve_vector_robust(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: RobustOptions,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(eastward, northward, None, Some(options), 0)
    }

    /// Robustly fit inferred currents with linear ellipse intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or robust fitting failure.
    pub fn solve_vector_robust_with_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(
            eastward,
            northward,
            Some(BatchConfidence::Linear(noise)),
            Some(options),
            0,
        )
    }

    /// Robustly fit inferred currents with nonlinear Monte Carlo ellipse
    /// confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid observations or options, robust fitting
    /// failure, or an unsampleable covariance.
    pub fn solve_vector_robust_with_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_robust_with_monte_carlo_confidence_from_stream(
            eastward,
            northward,
            robust_options,
            monte_carlo_options,
            noise,
            0,
        )
    }

    fn solve_vector_robust_with_monte_carlo_confidence_from_stream(
        &self,
        eastward: &[f64],
        northward: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
        stream: u64,
    ) -> Result<VectorSolution, AnalysisError> {
        self.solve_vector_impl(
            eastward,
            northward,
            Some(BatchConfidence::MonteCarlo {
                options: monte_carlo_options,
                noise,
            }),
            Some(robust_options),
            stream,
        )
    }

    /// Reconstruct one inferred vector solution with the fitted astronomical options.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid target times, filters, or solution shapes.
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
            self.phase_reference,
            self.nodal_corrections,
        )?
        .reconstruct_vector_at_latitude(solution, self.latitude_degrees_north, filter)
    }

    fn solve_vector_impl(
        &self,
        eastward: &[f64],
        northward: &[f64],
        confidence: Option<BatchConfidence>,
        robust: Option<RobustOptions>,
        stream: u64,
    ) -> Result<VectorSolution, AnalysisError> {
        if let Some(BatchConfidence::MonteCarlo { options, .. }) = confidence {
            options.validate()?;
        }
        if eastward.len() != northward.len() {
            return Err(AnalysisError::ObservationShape {
                actual: northward.len(),
                expected: eastward.len(),
            });
        }
        if eastward.len() != self.time_count {
            return Err(AnalysisError::ObservationShape {
                actual: eastward.len(),
                expected: self.time_count,
            });
        }
        for (time, (eastward, northward)) in eastward
            .iter()
            .copied()
            .zip(northward.iter().copied())
            .enumerate()
        {
            if !eastward.is_finite() {
                return Err(AnalysisError::NonFiniteObservation { series: 0, time });
            }
            if !northward.is_finite() {
                return Err(AnalysisError::NonFiniteObservation { series: 1, time });
            }
        }
        let observations = Mat::from_fn(self.time_count, 1, |time, _| {
            c64::new(eastward[time], northward[time])
        });
        let coefficients = self.decomposition.solve_lstsq(observations.as_ref());
        if let Some(options) = robust {
            let fit = robust_complex_fit_with_initial(
                &self.design,
                &observations,
                coefficients,
                options,
            )?;
            self.solution_from_coefficients(
                observations.as_ref(),
                fit.coefficients.as_ref(),
                confidence,
                Some(fit.diagnostics),
                stream,
            )
        } else {
            self.solution_from_coefficients(
                observations.as_ref(),
                coefficients.as_ref(),
                confidence,
                None,
                stream,
            )
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "keeps rotary expansion and inferred ellipse interval overrides in one audited conversion"
    )]
    fn solution_from_coefficients(
        &self,
        observations: faer::MatRef<'_, c64>,
        coefficients: faer::MatRef<'_, c64>,
        confidence: Option<BatchConfidence>,
        robust: Option<RobustDiagnostics>,
        stream: u64,
    ) -> Result<VectorSolution, AnalysisError> {
        let fit_count = self.output_mappings[..]
            .iter()
            .filter(|mapping| !mapping.inferred)
            .count();
        let reference_count = fit_count - self.non_reference_count;
        let mut positive = Vec::with_capacity(fit_count);
        let mut negative = Vec::with_capacity(fit_count);
        for ordinary in 0..self.non_reference_count {
            positive.push(coefficients[(ordinary, 0)]);
            negative.push(coefficients[(self.non_reference_count + ordinary, 0)]);
        }
        let positive_reference_start = self.non_reference_count * 2;
        let negative_reference_start = positive_reference_start + reference_count;
        for reference in 0..reference_count {
            positive.push(coefficients[(positive_reference_start + reference, 0)]);
            negative.push(coefficients[(negative_reference_start + reference, 0)]);
        }

        let output_positive = self
            .output_mappings
            .iter()
            .map(|mapping| {
                positive[mapping.source_fit_index]
                    * if mapping.inferred {
                        mapping.positive_ratio
                    } else {
                        c64::new(1.0, 0.0)
                    }
            })
            .collect::<Vec<_>>();
        let output_negative = self
            .output_mappings
            .iter()
            .map(|mapping| {
                negative[mapping.source_fit_index]
                    * if mapping.inferred {
                        mapping.negative_ratio
                    } else {
                        c64::new(1.0, 0.0)
                    }
            })
            .collect::<Vec<_>>();
        let eastward_cosine = output_positive
            .iter()
            .zip(&output_negative)
            .map(|(positive, negative)| (*positive + *negative).re)
            .collect::<Vec<_>>();
        let eastward_sine = output_positive
            .iter()
            .zip(&output_negative)
            .map(|(positive, negative)| -(*positive - *negative).im)
            .collect::<Vec<_>>();
        let northward_cosine = output_positive
            .iter()
            .zip(&output_negative)
            .map(|(positive, negative)| (*positive + *negative).im)
            .collect::<Vec<_>>();
        let northward_sine = output_positive
            .iter()
            .zip(&output_negative)
            .map(|(positive, negative)| (*positive - *negative).re)
            .collect::<Vec<_>>();

        let intervals = confidence.and_then(|confidence| match confidence {
            BatchConfidence::Linear(noise) => Some(
                self.vector_inference_intervals(
                    observations,
                    coefficients,
                    noise,
                    robust
                        .as_ref()
                        .map(|diagnostics| diagnostics.weights.as_slice()),
                ),
            ),
            BatchConfidence::MonteCarlo { .. } => None,
        });
        let harmonic_columns = fit_count * 2;
        let mean = coefficients[(harmonic_columns, 0)];
        let slope = if self.fit_options.trend {
            coefficients[(harmonic_columns + 1, 0)] / self.time_span_days
        } else {
            c64::new(0.0, 0.0)
        };
        let eastward = vector_inference_component_solution(
            eastward_cosine.clone(),
            eastward_sine.clone(),
            mean.re,
            slope.re,
            self.reference_time_modified_julian_day,
            intervals.as_ref().map(|intervals| {
                (
                    intervals.eastward_cosine_variance.clone(),
                    intervals.eastward_sine_variance.clone(),
                )
            }),
        );
        let northward = vector_inference_component_solution(
            northward_cosine.clone(),
            northward_sine.clone(),
            mean.im,
            slope.im,
            self.reference_time_modified_julian_day,
            intervals.as_ref().map(|intervals| {
                (
                    intervals.northward_cosine_variance.clone(),
                    intervals.northward_sine_variance.clone(),
                )
            }),
        );
        let mut solution = from_component_solutions(&eastward, &northward)?;
        if let Some(intervals) = intervals {
            let major_ci = solution
                .semi_major_ci
                .as_mut()
                .expect("coefficient variances produce ellipse intervals");
            let minor_ci = solution
                .semi_minor_ci
                .as_mut()
                .expect("coefficient variances produce ellipse intervals");
            let inclination_ci = solution
                .inclination_ci_degrees
                .as_mut()
                .expect("coefficient variances produce ellipse intervals");
            let phase_ci = solution
                .phase_ci_degrees
                .as_mut()
                .expect("coefficient variances produce ellipse intervals");
            for (output, inference) in intervals.inferred_linearization.iter().enumerate() {
                let Some((source, variance_x, variance_y)) = inference else {
                    continue;
                };
                let sigmas = linearized_ellipse_sigmas(
                    eastward_cosine[*source],
                    eastward_sine[*source],
                    northward_cosine[*source],
                    northward_sine[*source],
                    variance_x.sqrt(),
                    variance_y.sqrt(),
                    variance_y.sqrt(),
                    variance_x.sqrt(),
                );
                major_ci[output] = 1.96 * sigmas.0;
                minor_ci[output] = 1.96 * sigmas.1;
                phase_ci[output] = 1.96 * sigmas.2;
                inclination_ci[output] = 1.96 * sigmas.3;
            }
            solution.signal_to_noise = Some(
                solution
                    .semi_major
                    .iter()
                    .zip(&solution.semi_minor)
                    .zip(major_ci.iter())
                    .zip(minor_ci.iter())
                    .map(|(((major, minor), major_ci), minor_ci)| {
                        (major * major + minor * minor)
                            / ((major_ci / 1.96).powi(2) + (minor_ci / 1.96).powi(2))
                    })
                    .collect(),
            );
        }
        if let Some(BatchConfidence::MonteCarlo { options, noise }) = confidence {
            let covariances = self.vector_inference_monte_carlo_covariances(
                observations,
                coefficients,
                noise,
                robust
                    .as_ref()
                    .map(|diagnostics| diagnostics.weights.as_slice()),
            );
            self.apply_vector_monte_carlo_intervals(
                &mut solution,
                [
                    &eastward_cosine,
                    &eastward_sine,
                    &northward_cosine,
                    &northward_sine,
                ],
                &covariances,
                options,
                stream,
            )?;
        }
        solution.robust = robust;
        Ok(solution)
    }

    #[allow(
        clippy::similar_names,
        clippy::too_many_lines,
        reason = "UTide notation distinguishes the four Xu/Yu/Xv/Yv coefficient variances"
    )]
    fn vector_inference_intervals(
        &self,
        observations: faer::MatRef<'_, c64>,
        coefficients: faer::MatRef<'_, c64>,
        noise: LinearConfidence,
        weights: Option<&[f64]>,
    ) -> VectorInferenceIntervals {
        let basis = self.vector_inference_covariance_basis(observations, coefficients, weights);
        let fit_count = self
            .output_mappings
            .iter()
            .filter(|mapping| !mapping.inferred)
            .count();
        let mut base_variances = Vec::with_capacity(fit_count);
        for constituent in 0..fit_count {
            let negative = constituent + fit_count;
            let gall_00 = basis.gall[(constituent, constituent)];
            let gall_11 = basis.gall[(negative, negative)];
            let gall_01 = basis.gall[(constituent, negative)];
            let hall_00 = basis.hall[(constituent, constituent)];
            let hall_11 = basis.hall[(negative, negative)];
            let hall_01 = basis.hall[(constituent, negative)];
            base_variances.push((
                (gall_00 + gall_11 + gall_01 * 2.0).re / 2.0,
                (hall_00 + hall_11 - hall_01 * 2.0).re / 2.0,
                (hall_00 + hall_11 + hall_01 * 2.0).re / 2.0,
                (gall_00 + gall_11 - gall_01 * 2.0).re / 2.0,
            ));
        }
        if noise == LinearConfidence::Colored {
            let northward_residual = (0..self.time_count)
                .map(|time| {
                    weights.map_or(1.0, |weights| weights[time])
                        * (observations[(time, 0)].im - basis.fitted[(time, 0)].im)
                })
                .collect::<Vec<_>>();
            let power = self
                .confidence_sampling
                .band_averaged_residual_power(&northward_residual, &self.constituents);
            for (constituent, variance) in base_variances.iter_mut().enumerate() {
                let denominator = variance.2 + variance.3;
                variance.2 = power[constituent] * variance.2 / denominator;
                variance.3 = power[constituent] * variance.3 / denominator;
            }
        }

        let mut output = VectorInferenceIntervals {
            eastward_cosine_variance: Vec::with_capacity(self.output_mappings.len()),
            eastward_sine_variance: Vec::with_capacity(self.output_mappings.len()),
            northward_cosine_variance: Vec::with_capacity(self.output_mappings.len()),
            northward_sine_variance: Vec::with_capacity(self.output_mappings.len()),
            inferred_linearization: Vec::with_capacity(self.output_mappings.len()),
        };
        for mapping in &self.output_mappings {
            let source = mapping.source_fit_index;
            let (var_xu, var_yu, var_xv, var_yv) = base_variances[source];
            if mapping.inferred {
                let variance_real_positive = 0.25 * (var_xu + var_yv);
                let variance_imaginary_positive = 0.25 * (var_yu + var_xv);
                let real_factor =
                    mapping.positive_ratio.re.powi(2) + mapping.negative_ratio.re.powi(2);
                let imaginary_factor =
                    mapping.positive_ratio.im.powi(2) + mapping.negative_ratio.im.powi(2);
                let variance_x = real_factor * variance_real_positive
                    + imaginary_factor * variance_imaginary_positive;
                let variance_y = real_factor * variance_imaginary_positive
                    + imaginary_factor * variance_real_positive;
                output.eastward_cosine_variance.push(variance_x);
                output.eastward_sine_variance.push(variance_y);
                output.northward_cosine_variance.push(variance_y);
                output.northward_sine_variance.push(variance_x);
                output
                    .inferred_linearization
                    .push(Some((source, variance_x, variance_y)));
            } else {
                output.eastward_cosine_variance.push(var_xu);
                output.eastward_sine_variance.push(var_yu);
                output.northward_cosine_variance.push(var_xv);
                output.northward_sine_variance.push(var_yv);
                output.inferred_linearization.push(None);
            }
        }
        output
    }

    fn vector_inference_covariance_basis(
        &self,
        observations: faer::MatRef<'_, c64>,
        coefficients: faer::MatRef<'_, c64>,
        weights: Option<&[f64]>,
    ) -> VectorInferenceCovarianceBasis {
        let fitted = Mat::from_fn(self.time_count, 1, |time, _| {
            (0..self.design.ncols())
                .map(|column| self.design[(time, column)] * coefficients[(column, 0)])
                .sum::<c64>()
        });
        let degrees_of_freedom = usize_to_f64(self.time_count - self.design.ncols());
        let covariance_misfit = (0..self.time_count)
            .map(|time| {
                let weighted_observation =
                    weights.map_or(1.0, |weights| weights[time]) * observations[(time, 0)];
                observations[(time, 0)].conj() * weighted_observation
                    - fitted[(time, 0)].conj() * weighted_observation
            })
            .sum::<c64>()
            .re
            / degrees_of_freedom;
        let pseudo_misfit = (0..self.time_count)
            .map(|time| {
                let weighted_observation =
                    weights.map_or(1.0, |weights| weights[time]) * observations[(time, 0)];
                observations[(time, 0)] * weighted_observation
                    - fitted[(time, 0)] * weighted_observation
            })
            .sum::<c64>()
            / degrees_of_freedom;
        let column_count = self.design.ncols();
        let covariance_normal = Mat::from_fn(column_count, column_count, |row, column| {
            (0..self.time_count)
                .map(|time| {
                    self.design[(time, row)].conj()
                        * weights.map_or(1.0, |weights| weights[time])
                        * self.design[(time, column)]
                })
                .sum::<c64>()
        });
        let pseudo_normal = Mat::from_fn(column_count, column_count, |row, column| {
            (0..self.time_count)
                .map(|time| {
                    self.design[(time, row)]
                        * weights.map_or(1.0, |weights| weights[time])
                        * self.design[(time, column)]
                })
                .sum::<c64>()
        });
        let covariance_inverse = covariance_normal.partial_piv_lu().inverse();
        let pseudo_inverse = pseudo_normal.partial_piv_lu().inverse();
        let gall = Mat::from_fn(column_count, column_count, |row, column| {
            covariance_inverse[(row, column)] * covariance_misfit
                + pseudo_inverse[(row, column)] * pseudo_misfit
        });
        let hall = Mat::from_fn(column_count, column_count, |row, column| {
            covariance_inverse[(row, column)] * covariance_misfit
                - pseudo_inverse[(row, column)] * pseudo_misfit
        });
        VectorInferenceCovarianceBasis { fitted, gall, hall }
    }

    #[allow(
        clippy::similar_names,
        clippy::too_many_lines,
        reason = "the audited UTide covariance equations use the Xu/Yu/Xv/Yv coefficient names"
    )]
    fn vector_inference_monte_carlo_covariances(
        &self,
        observations: faer::MatRef<'_, c64>,
        coefficients: faer::MatRef<'_, c64>,
        noise: LinearConfidence,
        weights: Option<&[f64]>,
    ) -> Vec<[[f64; 4]; 4]> {
        let basis = self.vector_inference_covariance_basis(observations, coefficients, weights);
        let fit_count = self
            .output_mappings
            .iter()
            .filter(|mapping| !mapping.inferred)
            .count();
        let reference_count = fit_count - self.non_reference_count;
        let mut covariances = Vec::with_capacity(fit_count);
        for constituent in 0..fit_count {
            let (positive, negative) = if constituent < self.non_reference_count {
                (constituent, self.non_reference_count + constituent)
            } else {
                let reference = constituent - self.non_reference_count;
                (
                    self.non_reference_count * 2 + reference,
                    self.non_reference_count * 2 + reference_count + reference,
                )
            };
            let g00 = basis.gall[(positive, positive)];
            let g01 = basis.gall[(positive, negative)];
            let g10 = basis.gall[(negative, positive)];
            let g11 = basis.gall[(negative, negative)];
            let h00 = basis.hall[(positive, positive)];
            let h01 = basis.hall[(positive, negative)];
            let h10 = basis.hall[(negative, positive)];
            let h11 = basis.hall[(negative, negative)];
            let var_xu = (g00 + g11 + g01 * 2.0).re / 2.0;
            let var_yu = (h00 + h11 - h01 * 2.0).re / 2.0;
            let var_xv = (h00 + h11 + h01 * 2.0).re / 2.0;
            let var_yv = (g00 + g11 - g01 * 2.0).re / 2.0;
            let cov_xu_yu = (h00 - h01 + h10 - h11).im / 2.0;
            let cov_xv_yv = (g00 - g01 + g10 - g11).im / 2.0;
            let cov_xu_xv = (-h00 - h01 - h10 - h11).im / 2.0;
            let cov_xu_yv = (g00 - g11).re / 2.0;
            let cov_yu_xv = (-h00 + h11).re / 2.0;
            let cov_yu_yv = (-g00 + g01 + g10 - g11).im / 2.0;
            covariances.push([
                [var_xu, cov_xu_yu, cov_xu_xv, cov_xu_yv],
                [cov_xu_yu, var_yu, cov_yu_xv, cov_yu_yv],
                [cov_xu_xv, cov_yu_xv, var_xv, cov_xv_yv],
                [cov_xu_yv, cov_yu_yv, cov_xv_yv, var_yv],
            ]);
        }
        if noise == LinearConfidence::Colored {
            let mut eastward_residual = Vec::with_capacity(self.time_count);
            let mut northward_residual = Vec::with_capacity(self.time_count);
            for time in 0..self.time_count {
                let weight = weights.map_or(1.0, |weights| weights[time]);
                eastward_residual
                    .push(weight * (observations[(time, 0)].re - basis.fitted[(time, 0)].re));
                northward_residual
                    .push(weight * (observations[(time, 0)].im - basis.fitted[(time, 0)].im));
            }
            let power = self
                .confidence_sampling
                .band_averaged_vector_residual_power(
                    &eastward_residual,
                    &northward_residual,
                    &self.constituents,
                );
            for (constituent, covariance) in covariances.iter_mut().enumerate() {
                scale_covariance_block(
                    covariance,
                    [0, 1],
                    power.eastward[constituent],
                    covariance[0][0] + covariance[1][1],
                );
                scale_covariance_block(
                    covariance,
                    [2, 3],
                    power.northward[constituent],
                    covariance[2][2] + covariance[3][3],
                );
                let mut cross_denominator = 0.0;
                for row in [0, 1] {
                    for column in [2, 3] {
                        cross_denominator += covariance[row][column].abs();
                    }
                }
                let cross_scale = if cross_denominator == 0.0 {
                    0.0
                } else {
                    power.cross[constituent] / cross_denominator
                };
                for row in [0, 1] {
                    for column in [2, 3] {
                        covariance[row][column] *= cross_scale;
                        covariance[column][row] = covariance[row][column];
                    }
                }
            }
        }
        covariances
    }

    fn apply_vector_monte_carlo_intervals(
        &self,
        solution: &mut VectorSolution,
        coefficients: [&[f64]; 4],
        covariances: &[[[f64; 4]; 4]],
        options: MonteCarloOptions,
        stream: u64,
    ) -> Result<(), AnalysisError> {
        let mut output_intervals = vec![None; self.output_mappings.len()];
        for (source, covariance) in covariances.iter().copied().enumerate() {
            let mappings = self
                .output_mappings
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, mapping)| mapping.source_fit_index == source)
                .collect::<Vec<_>>();
            let ratios = mappings
                .iter()
                .map(|(_, mapping)| {
                    [
                        mapping.positive_ratio.re,
                        mapping.positive_ratio.im,
                        mapping.negative_ratio.re,
                        mapping.negative_ratio.im,
                    ]
                })
                .collect::<Vec<_>>();
            let source_output = mappings
                .iter()
                .find_map(|(output, mapping)| (!mapping.inferred).then_some(*output))
                .expect("every fitted source retains its reported reference output");
            let intervals = vector_transformed_intervals(
                [
                    coefficients[0][source_output],
                    coefficients[1][source_output],
                    coefficients[2][source_output],
                    coefficients[3][source_output],
                ],
                covariance,
                &ratios,
                options,
                constituent_stream(stream, source),
            )
            .ok_or(AnalysisError::InvalidConfidenceCovariance {
                constituent: source,
            })?;
            for ((output, _), interval) in mappings.into_iter().zip(intervals) {
                output_intervals[output] = Some(interval);
            }
        }
        let intervals = output_intervals
            .into_iter()
            .enumerate()
            .map(|(constituent, interval)| {
                interval.ok_or(AnalysisError::InvalidConfidenceCovariance { constituent })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let semi_major = intervals
            .iter()
            .map(|interval| interval.semi_major)
            .collect::<Vec<_>>();
        let semi_minor = intervals
            .iter()
            .map(|interval| interval.semi_minor)
            .collect::<Vec<_>>();
        solution.inclination_ci_degrees = Some(
            intervals
                .iter()
                .map(|interval| interval.inclination_degrees)
                .collect(),
        );
        solution.phase_ci_degrees = Some(
            intervals
                .iter()
                .map(|interval| interval.phase_degrees)
                .collect(),
        );
        solution.signal_to_noise = Some(
            solution
                .semi_major
                .iter()
                .zip(&solution.semi_minor)
                .zip(&semi_major)
                .zip(&semi_minor)
                .map(|(((major, minor), major_ci), minor_ci)| {
                    (major.powi(2) + minor.powi(2))
                        / ((major_ci / 1.96).powi(2) + (minor_ci / 1.96).powi(2))
                })
                .collect(),
        );
        solution.semi_major_ci = Some(semi_major);
        solution.semi_minor_ci = Some(semi_minor);
        Ok(())
    }
}

fn scale_covariance_block(
    covariance: &mut [[f64; 4]; 4],
    indices: [usize; 2],
    numerator: f64,
    denominator: f64,
) {
    let scale = if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    };
    for row in indices {
        for column in indices {
            covariance[row][column] *= scale;
        }
    }
}

fn vector_inference_component_solution(
    cosine_coefficient: Vec<f64>,
    sine_coefficient: Vec<f64>,
    mean: f64,
    slope_per_day: f64,
    reference_time_days: f64,
    variances: Option<(Vec<f64>, Vec<f64>)>,
) -> ScalarSolution {
    let amplitude = cosine_coefficient
        .iter()
        .zip(&sine_coefficient)
        .map(|(cosine, sine)| cosine.hypot(*sine))
        .collect::<Vec<_>>();
    let phase_degrees = cosine_coefficient
        .iter()
        .zip(&sine_coefficient)
        .map(|(cosine, sine)| sine.atan2(*cosine).to_degrees().rem_euclid(360.0))
        .collect::<Vec<_>>();
    let total_energy = amplitude
        .iter()
        .map(|amplitude| amplitude * amplitude)
        .sum::<f64>();
    let percent_energy = amplitude
        .iter()
        .map(|amplitude| 100.0 * amplitude * amplitude / total_energy)
        .collect();
    let (cosine_coefficient_variance, sine_coefficient_variance) = match variances {
        Some((cosine, sine)) => (Some(cosine), Some(sine)),
        None => (None, None),
    };
    ScalarSolution {
        cosine_coefficient,
        sine_coefficient,
        amplitude,
        phase_degrees,
        percent_energy,
        amplitude_ci: None,
        phase_ci_degrees: None,
        signal_to_noise: None,
        cosine_coefficient_variance,
        sine_coefficient_variance,
        mean,
        slope_per_day,
        reference_time_days,
        robust: None,
    }
}

fn scalar_linear_intervals(
    cosine: f64,
    sine: f64,
    cosine_variance: f64,
    sine_variance: f64,
) -> (f64, f64) {
    let magnitude_squared = cosine * cosine + sine * sine;
    let amplitude_sigma = ((cosine * cosine * cosine_variance + sine * sine * sine_variance)
        / magnitude_squared)
        .sqrt();
    let phase_sigma = ((sine * sine * cosine_variance + cosine * cosine * sine_variance)
        / magnitude_squared.powi(2))
    .sqrt();
    (1.96 * amplitude_sigma, 1.96 * phase_sigma * 180.0 / PI)
}

fn constituents_at_reference_for_basis(
    basis: &CorrectionBasis,
    reference_time: f64,
) -> Result<Vec<Constituent>, AnalysisError> {
    if !reference_time.is_finite() {
        return Err(AnalysisError::NonFiniteReferenceTime);
    }
    let constituents = scalar_constituents_at_reference(
        &basis.tidal_constituents,
        &basis.base_constituents,
        &basis.recipes,
        reference_time,
    );
    validate_derived_frequencies(&constituents)?;
    Ok(constituents)
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

/// A reusable configurable-phase and configurable-nodal reconstruction basis.
///
/// Astronomy is prepared once and can then reconstruct many scalar solutions at
/// different latitudes. Multi-series output is series-major: one complete target
/// time series per input solution.
#[derive(Debug)]
pub struct GreenwichNodalReconstructor {
    tidal_constituents: Vec<TidalConstituent>,
    constituent_frequency_cph: Vec<f64>,
    recipes: Vec<CorrectionRecipe>,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    reference_greenwich_phase: Vec<f64>,
    reference_base_nodal_terms: Vec<NodalTerms>,
    reference_time_modified_julian_day: f64,
    time_terms: Vec<ReconstructionTimeTerms>,
}

impl GreenwichNodalReconstructor {
    /// Prepare an exact-nodal Greenwich-phase basis from a fit epoch and target MJDs.
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
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            reference_time_modified_julian_day,
            constituents,
            SolverOptions::default(),
        )
    }

    /// Prepare a reconstruction basis with an explicit phase reference.
    ///
    /// Target times may be unordered or repeated, but must be finite and nonempty.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for an invalid reference time, target time, or
    /// constituent list.
    pub fn prepare_modified_julian_days_with_phase_reference(
        modified_julian_days: &[f64],
        reference_time_modified_julian_day: f64,
        constituents: &[TidalConstituent],
        phase_reference: PhaseReference,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            reference_time_modified_julian_day,
            constituents,
            SolverOptions::new(FitOptions::default(), phase_reference),
        )
    }

    /// Prepare a reconstruction basis with explicit phase and nodal options.
    ///
    /// Target times may be unordered or repeated, but must be finite and nonempty.
    /// Fit-specific fields in `solver_options` do not affect reconstruction.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for an invalid reference time, target time, or
    /// constituent list.
    pub fn prepare_modified_julian_days_with_solver_options(
        modified_julian_days: &[f64],
        reference_time_modified_julian_day: f64,
        constituents: &[TidalConstituent],
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        validate_tidal_constituents(constituents)?;
        let (base_constituents, recipes) = dependency_recipes(constituents);
        Self::from_parts(
            modified_julian_days,
            reference_time_modified_julian_day,
            constituents.to_vec(),
            &base_constituents,
            recipes,
            solver_options.phase_reference(),
            solver_options.nodal_corrections(),
        )
    }

    fn from_parts(
        modified_julian_days: &[f64],
        reference_time_modified_julian_day: f64,
        tidal_constituents: Vec<TidalConstituent>,
        base_constituents: &[TidalConstituent],
        recipes: Vec<CorrectionRecipe>,
        phase_reference: PhaseReference,
        nodal_corrections: NodalCorrections,
    ) -> Result<Self, AnalysisError> {
        validate_reconstruction_times(modified_julian_days, reference_time_modified_julian_day)?;
        let scalar_constituents = scalar_constituents_at_reference(
            &tidal_constituents,
            base_constituents,
            &recipes,
            reference_time_modified_julian_day,
        );
        validate_derived_frequencies(&scalar_constituents)?;
        let constituent_frequency_cph = scalar_constituents
            .iter()
            .map(|constituent| constituent.frequency_cph)
            .collect();
        let reference_greenwich_phase = greenwich_phases_at_time(
            base_constituents,
            &recipes,
            reference_time_modified_julian_day,
        );
        let reference_base_nodal_terms =
            base_nodal_terms_at_time(base_constituents, reference_time_modified_julian_day);
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
                    base_nodal_terms: if nodal_corrections == NodalCorrections::Exact {
                        base_constituents
                            .iter()
                            .copied()
                            .map(|constituent| {
                                precompute_nodal_terms(constituent.metadata(), astronomy.cycles)
                            })
                            .collect()
                    } else {
                        Vec::new()
                    },
                    modified_julian_day: time,
                }
            })
            .collect();
        Ok(Self {
            tidal_constituents,
            constituent_frequency_cph,
            recipes,
            phase_reference,
            nodal_corrections,
            reference_greenwich_phase,
            reference_base_nodal_terms,
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

    /// Return the configured astronomical phase-reference convention.
    #[must_use]
    pub const fn phase_reference(&self) -> PhaseReference {
        self.phase_reference
    }

    /// Return the configured nodal-correction mode.
    #[must_use]
    pub const fn nodal_corrections(&self) -> NodalCorrections {
        self.nodal_corrections
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
        let mut base_corrections = base_nodal_corrections(
            self.nodal_corrections,
            &self.time_terms[0].base_nodal_terms,
            &self.reference_base_nodal_terms,
            latitude_factors,
        );
        let mut reconstruction = Vec::with_capacity(self.time_terms.len());
        for terms in &self.time_terms {
            if self.nodal_corrections == NodalCorrections::Exact {
                for (correction, nodal_terms) in base_corrections
                    .iter_mut()
                    .zip(terms.base_nodal_terms.iter().copied())
                {
                    *correction = nodal_correction(nodal_terms, latitude_factors);
                }
            }
            let harmonics = selected
                .iter()
                .copied()
                .map(|constituent| {
                    let (nodal_amplitude, nodal_phase) =
                        self.recipes[constituent].combine_nodal(&base_corrections);
                    let astronomical_phase = phase_cycles(
                        self.phase_reference,
                        terms.greenwich_phase[constituent],
                        self.reference_greenwich_phase[constituent],
                        terms.modified_julian_day,
                        self.reference_time_modified_julian_day,
                        self.constituent_frequency_cph[constituent],
                    );
                    let angle = TAU * (nodal_phase + astronomical_phase)
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

impl ScalarInferenceBatch {
    /// Prepare shared astronomical terms for scalar inference.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid timestamps, constituents, relationships, or
    /// an underdetermined model.
    pub fn prepare_modified_julian_days(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        relationships: &[ScalarInferenceRelation],
        mode: InferenceMode,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            constituents,
            relationships,
            mode,
            SolverOptions::default(),
        )
    }

    /// Prepare shared scalar-inference astronomy with explicit fit options.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, relationships, or an underdetermined
    /// requested model.
    pub fn prepare_modified_julian_days_with_options(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        relationships: &[ScalarInferenceRelation],
        mode: InferenceMode,
        fit_options: FitOptions,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            constituents,
            relationships,
            mode,
            fit_options.into(),
        )
    }

    /// Prepare shared scalar-inference astronomy with explicit solver options.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, relationships, or an underdetermined
    /// requested model.
    pub fn prepare_modified_julian_days_with_solver_options(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        relationships: &[ScalarInferenceRelation],
        mode: InferenceMode,
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        let layout = scalar_inference_layout(constituents, relationships)?;
        let basis = CorrectionBasis::prepare_with_model_count(
            modified_julian_days,
            &layout.tidal_constituents,
            layout.fit_count,
            solver_options,
        )?;
        Ok(Self {
            basis,
            layout,
            relationships: relationships.to_vec(),
            mode,
        })
    }

    /// Return the source timestamp count.
    #[must_use]
    pub const fn time_count(&self) -> usize {
        self.basis.time_terms.len()
    }

    /// Return the configured non-harmonic fit terms.
    #[must_use]
    pub const fn fit_options(&self) -> FitOptions {
        self.basis.fit_options
    }

    /// Return the configured astronomical phase-reference convention.
    #[must_use]
    pub const fn phase_reference(&self) -> PhaseReference {
        self.basis.phase_reference
    }

    /// Return the configured nodal/satellite correction convention.
    #[must_use]
    pub const fn nodal_corrections(&self) -> NodalCorrections {
        self.basis.nodal_corrections
    }

    /// Return every reported constituent in coefficient order.
    #[must_use]
    pub fn tidal_constituents(&self) -> &[TidalConstituent] {
        &self.layout.tidal_constituents
    }

    /// Return reported names and reference-time frequencies.
    #[must_use]
    pub fn constituents(&self) -> &[Constituent] {
        &self.basis.scalar_constituents
    }

    /// Return scalar relationships in caller-supplied order.
    #[must_use]
    pub fn relationships(&self) -> &[ScalarInferenceRelation] {
        &self.relationships
    }

    /// Return the configured inference mode.
    #[must_use]
    pub const fn mode(&self) -> InferenceMode {
        self.mode
    }

    /// Return the full-record midpoint epoch as an MJD.
    #[must_use]
    pub const fn reference_time_modified_julian_day(&self) -> f64 {
        self.basis.reference_time_modified_julian_day
    }

    /// Return output frequencies at an arbitrary finite fitted epoch.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite epoch or degenerate derived frequency.
    pub fn constituents_at_reference_modified_julian_day(
        &self,
        reference_time: f64,
    ) -> Result<Vec<Constituent>, AnalysisError> {
        constituents_at_reference_for_basis(&self.basis, reference_time)
    }

    /// Prepare matching reconstruction astronomy at arbitrary target MJDs.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid target timestamps.
    pub fn reconstructor_modified_julian_days(
        &self,
        modified_julian_days: &[f64],
    ) -> Result<GreenwichNodalReconstructor, AnalysisError> {
        GreenwichNodalReconstructor::from_parts(
            modified_julian_days,
            self.basis.reference_time_modified_julian_day,
            self.layout.tidal_constituents.clone(),
            &self.basis.base_constituents,
            self.basis.recipes.clone(),
            self.basis.phase_reference,
            self.basis.nodal_corrections,
        )
    }

    /// Fit complete time-major scalar series at varying latitudes.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, latitudes, or observations.
    pub fn solve_time_major(
        &self,
        observations: &[f64],
        latitudes: &[f64],
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(observations, latitudes, None, None, false)
    }

    /// Fit complete scalar series with linear confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, latitudes, or observations.
    pub fn solve_time_major_with_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            None,
            false,
        )
    }

    /// Fit complete inferred scalar series with deterministic Monte Carlo
    /// confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, options, or covariance.
    pub fn solve_time_major_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::MonteCarlo { options, noise }),
            None,
            false,
        )
    }

    /// Robustly fit complete scalar series.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or a robust fitting failure.
    pub fn solve_time_major_robust(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(observations, latitudes, None, Some(options), false)
    }

    /// Robustly fit complete scalar series with linear confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or a robust fitting failure.
    pub fn solve_time_major_robust_with_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            Some(options),
            false,
        )
    }

    /// Robustly fit complete inferred scalar series with deterministic Monte
    /// Carlo confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or options, robust fitting failure,
    /// or covariance failure.
    pub fn solve_time_major_robust_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::MonteCarlo {
                options: monte_carlo_options,
                noise,
            }),
            Some(robust_options),
            false,
        )
    }

    /// Fit scalar series while treating `NaN` observations as missing.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, latitudes, infinities, or
    /// underdetermined retained records.
    pub fn solve_time_major_with_missing(
        &self,
        observations: &[f64],
        latitudes: &[f64],
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(observations, latitudes, None, None, true)
    }

    /// Fit gappy scalar series with linear confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or underdetermined retained records.
    pub fn solve_time_major_with_missing_and_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            None,
            true,
        )
    }

    /// Fit gappy inferred scalar series with deterministic Monte Carlo
    /// confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, options, underdetermined retained
    /// records, or covariance failure.
    pub fn solve_time_major_with_missing_and_monte_carlo_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::MonteCarlo { options, noise }),
            None,
            true,
        )
    }

    /// Robustly fit scalar series while treating `NaN` as missing.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or a robust fitting failure.
    pub fn solve_time_major_with_missing_robust(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(observations, latitudes, None, Some(options), true)
    }

    /// Robustly fit gappy scalar series with linear confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or a robust fitting failure.
    pub fn solve_time_major_with_missing_robust_and_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            Some(options),
            true,
        )
    }

    /// Robustly fit gappy inferred scalar series with deterministic Monte Carlo
    /// confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or options, robust fitting failure,
    /// underdetermined records, or covariance failure.
    pub fn solve_time_major_with_missing_robust_and_monte_carlo_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::MonteCarlo {
                options: monte_carlo_options,
                noise,
            }),
            Some(robust_options),
            true,
        )
    }

    fn solve_time_major_impl(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        confidence: Option<BatchConfidence>,
        robust: Option<RobustOptions>,
        allow_missing: bool,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        validate_batch_shape_and_latitudes(self.time_count(), observations, latitudes)?;
        let series_count = latitudes.len();
        for (index, value) in observations.iter().copied().enumerate() {
            if value.is_infinite() || (!allow_missing && value.is_nan()) {
                return Err(AnalysisError::NonFiniteObservation {
                    series: index % series_count,
                    time: index / series_count,
                });
            }
        }
        let mut unique_positions = Vec::<Vec<usize>>::new();
        let mut record_use_count = Vec::<usize>::new();
        let mut record_by_positions = HashMap::<Vec<usize>, usize>::new();
        let mut record_for_series = Vec::with_capacity(series_count);
        for series in 0..series_count {
            let positions = (0..self.time_count())
                .filter(|time| observations[time * series_count + series].is_finite())
                .collect::<Vec<_>>();
            let record_index = if let Some(index) = record_by_positions.get(&positions) {
                record_use_count[*index] += 1;
                *index
            } else {
                let index = unique_positions.len();
                record_by_positions.insert(positions.clone(), index);
                unique_positions.push(positions);
                record_use_count.push(1);
                index
            };
            record_for_series.push(record_index);
        }
        let records = unique_positions
            .into_iter()
            .zip(shared_lomb_plan_groups(&record_use_count))
            .map(|(positions, share_plan)| self.basis.record_subset(positions, share_plan))
            .collect::<Result<Vec<_>, _>>()?;
        if confidence.map(BatchConfidence::noise) == Some(LinearConfidence::Colored) {
            for record in &records {
                record
                    .confidence_sampling
                    .precompute_shared_irregular_plan();
            }
        }

        (0..series_count)
            .into_par_iter()
            .map(|series| {
                let record = &records[record_for_series[series]];
                let model = ScalarInferenceOls::from_basis_record(
                    &self.basis,
                    record,
                    latitudes[series],
                    &self.layout,
                    &self.relationships,
                    self.mode,
                )?;
                let values = record
                    .positions
                    .iter()
                    .copied()
                    .map(|time| observations[time * series_count + series])
                    .collect::<Vec<_>>();
                let stream = u64::try_from(series).expect("series index is representable as u64");
                match (robust, confidence) {
                    (Some(options), Some(BatchConfidence::Linear(noise))) => {
                        model.solve_robust_with_linear_confidence(&values, options, noise)
                    }
                    (
                        Some(robust_options),
                        Some(BatchConfidence::MonteCarlo { options, noise }),
                    ) => model.solve_with_monte_carlo_confidence_impl(
                        &values,
                        Some(robust_options),
                        options,
                        noise,
                        stream,
                    ),
                    (Some(options), None) => model.solve_robust(&values, options),
                    (None, Some(BatchConfidence::Linear(noise))) => {
                        model.solve_with_linear_confidence(&values, noise)
                    }
                    (None, Some(BatchConfidence::MonteCarlo { options, noise })) => model
                        .solve_with_monte_carlo_confidence_impl(
                            &values, None, options, noise, stream,
                        ),
                    (None, None) => model.solve(&values),
                }
            })
            .collect()
    }
}

impl VectorInferenceBatch {
    /// Prepare shared astronomical terms for coupled vector inference.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid timestamps, constituents, relationships, or
    /// an underdetermined model.
    pub fn prepare_modified_julian_days(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        relationships: &[VectorInferenceRelation],
        mode: InferenceMode,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            constituents,
            relationships,
            mode,
            SolverOptions::default(),
        )
    }

    /// Prepare shared vector-inference astronomy with explicit fit options.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, relationships, or an underdetermined
    /// requested model.
    pub fn prepare_modified_julian_days_with_options(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        relationships: &[VectorInferenceRelation],
        mode: InferenceMode,
        fit_options: FitOptions,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            constituents,
            relationships,
            mode,
            fit_options.into(),
        )
    }

    /// Prepare shared vector-inference astronomy with explicit solver options.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, relationships, or an underdetermined
    /// requested model.
    pub fn prepare_modified_julian_days_with_solver_options(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        relationships: &[VectorInferenceRelation],
        mode: InferenceMode,
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        let layout = vector_inference_layout(constituents, relationships)?;
        let basis = CorrectionBasis::prepare_with_model_count(
            modified_julian_days,
            &layout.tidal_constituents,
            layout.fit_count,
            solver_options,
        )?;
        Ok(Self {
            basis,
            layout,
            relationships: relationships.to_vec(),
            mode,
        })
    }

    /// Return the source timestamp count.
    #[must_use]
    pub const fn time_count(&self) -> usize {
        self.basis.time_terms.len()
    }

    /// Return the configured non-harmonic fit terms.
    #[must_use]
    pub const fn fit_options(&self) -> FitOptions {
        self.basis.fit_options
    }

    /// Return the configured astronomical phase-reference convention.
    #[must_use]
    pub const fn phase_reference(&self) -> PhaseReference {
        self.basis.phase_reference
    }

    /// Return the configured nodal/satellite correction convention.
    #[must_use]
    pub const fn nodal_corrections(&self) -> NodalCorrections {
        self.basis.nodal_corrections
    }

    /// Return every reported constituent in coefficient order.
    #[must_use]
    pub fn tidal_constituents(&self) -> &[TidalConstituent] {
        &self.layout.tidal_constituents
    }

    /// Return reported names and reference-time frequencies.
    #[must_use]
    pub fn constituents(&self) -> &[Constituent] {
        &self.basis.scalar_constituents
    }

    /// Return vector relationships in caller-supplied order.
    #[must_use]
    pub fn relationships(&self) -> &[VectorInferenceRelation] {
        &self.relationships
    }

    /// Return the configured inference mode.
    #[must_use]
    pub const fn mode(&self) -> InferenceMode {
        self.mode
    }

    /// Return the full-record midpoint epoch as an MJD.
    #[must_use]
    pub const fn reference_time_modified_julian_day(&self) -> f64 {
        self.basis.reference_time_modified_julian_day
    }

    /// Return output frequencies at an arbitrary finite fitted epoch.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite epoch or degenerate derived frequency.
    pub fn constituents_at_reference_modified_julian_day(
        &self,
        reference_time: f64,
    ) -> Result<Vec<Constituent>, AnalysisError> {
        constituents_at_reference_for_basis(&self.basis, reference_time)
    }

    /// Prepare matching reconstruction astronomy at arbitrary target MJDs.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid target timestamps.
    pub fn reconstructor_modified_julian_days(
        &self,
        modified_julian_days: &[f64],
    ) -> Result<GreenwichNodalReconstructor, AnalysisError> {
        GreenwichNodalReconstructor::from_parts(
            modified_julian_days,
            self.basis.reference_time_modified_julian_day,
            self.layout.tidal_constituents.clone(),
            &self.basis.base_constituents,
            self.basis.recipes.clone(),
            self.basis.phase_reference,
            self.basis.nodal_corrections,
        )
    }

    /// Fit complete time-major current series at varying latitudes.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, latitudes, or observations.
    pub fn solve_vector_time_major(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(eastward, northward, latitudes, None, None, false)
    }

    /// Fit complete current series with linear ellipse confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, latitudes, or observations.
    pub fn solve_vector_time_major_with_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            None,
            false,
        )
    }

    /// Fit complete inferred currents with deterministic Monte Carlo ellipse
    /// confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, options, or covariance.
    pub fn solve_vector_time_major_with_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::MonteCarlo { options, noise }),
            None,
            false,
        )
    }

    /// Robustly fit complete inferred current series.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or robust fitting failure.
    pub fn solve_vector_time_major_robust(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            None,
            Some(options),
            false,
        )
    }

    /// Robustly fit complete inferred currents with linear ellipse intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or robust fitting failure.
    pub fn solve_vector_time_major_robust_with_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            Some(options),
            false,
        )
    }

    /// Robustly fit complete inferred currents with deterministic Monte Carlo
    /// ellipse confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or options, robust fitting failure,
    /// or covariance failure.
    pub fn solve_vector_time_major_robust_with_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::MonteCarlo {
                options: monte_carlo_options,
                noise,
            }),
            Some(robust_options),
            false,
        )
    }

    /// Fit currents while omitting a time from both components when either is
    /// `NaN`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, latitudes, infinities, or
    /// underdetermined joint records.
    pub fn solve_vector_time_major_with_missing(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(eastward, northward, latitudes, None, None, true)
    }

    /// Fit gappy currents with linear ellipse confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or underdetermined joint records.
    pub fn solve_vector_time_major_with_missing_and_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            None,
            true,
        )
    }

    /// Fit gappy inferred currents with deterministic Monte Carlo ellipse
    /// confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, options, underdetermined retained
    /// records, or covariance failure.
    pub fn solve_vector_time_major_with_missing_and_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::MonteCarlo { options, noise }),
            None,
            true,
        )
    }

    /// Robustly fit inferred currents while jointly omitting `NaN` samples.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or robust fitting failure.
    pub fn solve_vector_time_major_with_missing_robust(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(eastward, northward, latitudes, None, Some(options), true)
    }

    /// Robustly fit gappy inferred currents with linear ellipse intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or robust fitting failure.
    pub fn solve_vector_time_major_with_missing_robust_and_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            Some(options),
            true,
        )
    }

    /// Robustly fit gappy inferred currents with deterministic Monte Carlo
    /// ellipse confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or options, robust fitting failure,
    /// underdetermined records, or covariance failure.
    pub fn solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::MonteCarlo {
                options: monte_carlo_options,
                noise,
            }),
            Some(robust_options),
            true,
        )
    }

    fn solve_vector_time_major_impl(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        confidence: Option<BatchConfidence>,
        robust: Option<RobustOptions>,
        allow_missing: bool,
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
            if value.is_infinite() || (!allow_missing && value.is_nan()) {
                return Err(AnalysisError::NonFiniteObservation {
                    series: index % series_count,
                    time: (index % eastward.len()) / series_count,
                });
            }
        }
        let mut unique_positions = Vec::<Vec<usize>>::new();
        let mut record_use_count = Vec::<usize>::new();
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
                record_use_count[*index] += 1;
                *index
            } else {
                let index = unique_positions.len();
                record_by_positions.insert(positions.clone(), index);
                unique_positions.push(positions);
                record_use_count.push(1);
                index
            };
            record_for_series.push(record_index);
        }
        let records = unique_positions
            .into_iter()
            .zip(shared_lomb_plan_groups(&record_use_count))
            .map(|(positions, share_plan)| self.basis.record_subset(positions, share_plan))
            .collect::<Result<Vec<_>, _>>()?;
        if confidence.map(BatchConfidence::noise) == Some(LinearConfidence::Colored) {
            for record in &records {
                record
                    .confidence_sampling
                    .precompute_shared_irregular_plan();
            }
        }

        (0..series_count)
            .into_par_iter()
            .map(|series| {
                let record = &records[record_for_series[series]];
                let model = VectorInferenceOls::from_basis_record(
                    &self.basis,
                    record,
                    latitudes[series],
                    &self.layout,
                    &self.relationships,
                    self.mode,
                )?;
                let eastward_values = record
                    .positions
                    .iter()
                    .copied()
                    .map(|time| eastward[time * series_count + series])
                    .collect::<Vec<_>>();
                let northward_values = record
                    .positions
                    .iter()
                    .copied()
                    .map(|time| northward[time * series_count + series])
                    .collect::<Vec<_>>();
                let stream = u64::try_from(series).expect("series index is representable as u64");
                solve_vector_inference_with_batch_confidence(
                    &model,
                    &eastward_values,
                    &northward_values,
                    robust,
                    confidence,
                    stream,
                )
            })
            .collect()
    }
}

/// Shared astronomy for fitting many series at different latitudes.
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
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            constituents,
            SolverOptions::default(),
        )
    }

    /// Prepare shared astronomical terms with explicit fit options.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when inputs or the requested model are invalid.
    pub fn prepare_modified_julian_days_with_options(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        fit_options: FitOptions,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_modified_julian_days_with_solver_options(
            modified_julian_days,
            constituents,
            fit_options.into(),
        )
    }

    /// Prepare shared astronomical terms with explicit solver options.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when inputs or the requested model are invalid.
    pub fn prepare_modified_julian_days_with_solver_options(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        Ok(Self {
            basis: CorrectionBasis::prepare(modified_julian_days, constituents, solver_options)?,
        })
    }

    /// Return the number of observations expected for each spatial series.
    #[must_use]
    pub const fn time_count(&self) -> usize {
        self.basis.time_terms.len()
    }

    /// Return the configured non-harmonic fit terms.
    #[must_use]
    pub const fn fit_options(&self) -> FitOptions {
        self.basis.fit_options
    }

    /// Return the configured astronomical phase-reference convention.
    #[must_use]
    pub const fn phase_reference(&self) -> PhaseReference {
        self.basis.phase_reference
    }

    /// Return the configured nodal/satellite correction convention.
    #[must_use]
    pub const fn nodal_corrections(&self) -> NodalCorrections {
        self.basis.nodal_corrections
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

    /// Prepare a reusable matching basis at arbitrary reconstruction MJDs.
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
            self.basis.phase_reference,
            self.basis.nodal_corrections,
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
        self.solve_time_major_impl(observations, latitudes, None, None)
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
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            None,
        )
    }

    /// Fit varying-latitude series with nonlinear Monte Carlo intervals.
    ///
    /// The root seed is split deterministically by series index, so results do
    /// not depend on Rayon's worker count or scheduling order.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, options, or covariance.
    pub fn solve_time_major_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        options.validate()?;
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::MonteCarlo { options, noise }),
            None,
        )
    }

    /// Robustly fit varying-latitude complete scalar series in parallel.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or any non-convergent series.
    pub fn solve_time_major_robust(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(observations, latitudes, None, Some(options))
    }

    /// Robustly fit complete scalar series with linear confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or any robust fitting failure.
    pub fn solve_time_major_robust_with_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            Some(options),
        )
    }

    /// Robustly fit complete scalar series with Monte Carlo intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, robust fitting failure,
    /// invalid Monte Carlo options, or covariance failure.
    pub fn solve_time_major_robust_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        monte_carlo_options.validate()?;
        self.solve_time_major_impl(
            observations,
            latitudes,
            Some(BatchConfidence::MonteCarlo {
                options: monte_carlo_options,
                noise,
            }),
            Some(robust_options),
        )
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
        self.solve_time_major_with_missing_impl(observations, latitudes, None, None)
    }

    /// Fit possibly gappy scalar series with linearized confidence intervals.
    ///
    /// Colored intervals interpolate residuals from gappy observations onto an
    /// originally equidistant timestamp grid, matching Python `UTide`, and use
    /// Lomb–Scargle residual spectra for truly irregular timestamps.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid inputs.
    pub fn solve_time_major_with_missing_and_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_with_missing_impl(
            observations,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            None,
        )
    }

    /// Fit gappy scalar series with Monte Carlo confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, underdetermined records,
    /// invalid Monte Carlo options, or covariance failure.
    pub fn solve_time_major_with_missing_and_monte_carlo_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        options.validate()?;
        self.solve_time_major_with_missing_impl(
            observations,
            latitudes,
            Some(BatchConfidence::MonteCarlo { options, noise }),
            None,
        )
    }

    /// Robustly fit scalar series while treating `NaN` observations as missing.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or any robust fitting failure.
    pub fn solve_time_major_with_missing_robust(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_with_missing_impl(observations, latitudes, None, Some(options))
    }

    /// Robustly fit gappy scalar series with linear confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or any robust fitting failure.
    pub fn solve_time_major_with_missing_robust_and_linear_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_time_major_with_missing_impl(
            observations,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            Some(options),
        )
    }

    /// Robustly fit gappy scalar series with Monte Carlo confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, robust fitting failure,
    /// invalid Monte Carlo options, or covariance failure.
    pub fn solve_time_major_with_missing_robust_and_monte_carlo_confidence(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        monte_carlo_options.validate()?;
        self.solve_time_major_with_missing_impl(
            observations,
            latitudes,
            Some(BatchConfidence::MonteCarlo {
                options: monte_carlo_options,
                noise,
            }),
            Some(robust_options),
        )
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
        self.solve_vector_time_major_with_missing_impl(eastward, northward, latitudes, None, None)
    }

    /// Jointly fit gappy currents with linearized ellipse confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input.
    pub fn solve_vector_time_major_with_missing_and_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_with_missing_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            None,
        )
    }

    /// Jointly fit gappy currents with Monte Carlo ellipse intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, underdetermined records,
    /// invalid Monte Carlo options, or covariance failure.
    pub fn solve_vector_time_major_with_missing_and_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        options.validate()?;
        self.solve_vector_time_major_with_missing_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::MonteCarlo { options, noise }),
            None,
        )
    }

    /// Robustly fit gappy currents with one shared weight per component pair.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or any robust fitting failure.
    pub fn solve_vector_time_major_with_missing_robust(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_with_missing_impl(
            eastward,
            northward,
            latitudes,
            None,
            Some(options),
        )
    }

    /// Robustly fit gappy currents with Monte Carlo ellipse intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, robust fitting failure,
    /// invalid Monte Carlo options, or covariance failure.
    pub fn solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        monte_carlo_options.validate()?;
        self.solve_vector_time_major_with_missing_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::MonteCarlo {
                options: monte_carlo_options,
                noise,
            }),
            Some(robust_options),
        )
    }

    /// Robustly fit gappy currents with linearized ellipse confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or any robust fitting failure.
    pub fn solve_vector_time_major_with_missing_robust_and_linear_confidence(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<VectorSolution>, AnalysisError> {
        self.solve_vector_time_major_with_missing_impl(
            eastward,
            northward,
            latitudes,
            Some(BatchConfidence::Linear(noise)),
            Some(options),
        )
    }

    fn solve_vector_time_major_with_missing_impl(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        confidence: Option<BatchConfidence>,
        robust: Option<RobustOptions>,
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

        let mut unique_positions = Vec::<Vec<usize>>::new();
        let mut record_use_count = Vec::<usize>::new();
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
                record_use_count[*index] += 1;
                *index
            } else {
                let index = unique_positions.len();
                record_by_positions.insert(positions.clone(), index);
                unique_positions.push(positions);
                record_use_count.push(1);
                index
            };
            record_for_series.push(record_index);
        }
        let shared_lomb_plans = shared_lomb_plan_groups(&record_use_count);
        let records = unique_positions
            .into_iter()
            .zip(shared_lomb_plans)
            .map(|(positions, share_plan)| self.basis.record_subset(positions, share_plan))
            .collect::<Result<Vec<_>, _>>()?;
        if confidence.is_some_and(|confidence| confidence.noise() == LinearConfidence::Colored) {
            for record in &records {
                record
                    .confidence_sampling
                    .precompute_shared_irregular_plan();
            }
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
                solve_vector_with_batch_confidence(
                    &model,
                    &component_values,
                    robust,
                    confidence,
                    u64::try_from(series).expect("series index is representable as u64"),
                )
            })
            .collect()
    }

    fn solve_time_major_with_missing_impl(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        confidence: Option<BatchConfidence>,
        robust: Option<RobustOptions>,
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
            return self.solve_time_major_impl(observations, latitudes, confidence, robust);
        }

        let series_count = latitudes.len();
        let mut unique_positions = Vec::<Vec<usize>>::new();
        let mut record_use_count = Vec::<usize>::new();
        let mut record_by_positions = HashMap::<Vec<usize>, usize>::new();
        let mut record_for_series = Vec::with_capacity(series_count);
        for series in 0..series_count {
            let positions = (0..self.time_count())
                .filter(|time| observations[time * series_count + series].is_finite())
                .collect::<Vec<_>>();
            let record_index = if let Some(index) = record_by_positions.get(&positions) {
                record_use_count[*index] += 1;
                *index
            } else {
                let index = unique_positions.len();
                record_by_positions.insert(positions.clone(), index);
                unique_positions.push(positions);
                record_use_count.push(1);
                index
            };
            record_for_series.push(record_index);
        }
        let shared_lomb_plans = shared_lomb_plan_groups(&record_use_count);
        let records = unique_positions
            .into_iter()
            .zip(shared_lomb_plans)
            .map(|(positions, share_plan)| self.basis.record_subset(positions, share_plan))
            .collect::<Result<Vec<_>, _>>()?;
        if confidence.is_some_and(|confidence| confidence.noise() == LinearConfidence::Colored) {
            for record in &records {
                record
                    .confidence_sampling
                    .precompute_shared_irregular_plan();
            }
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
                solve_scalar_with_batch_confidence(
                    &model,
                    &series_observations,
                    robust,
                    confidence,
                    u64::try_from(series).expect("series index is representable as u64"),
                )
            })
            .collect()
    }

    fn solve_time_major_impl(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        confidence: Option<BatchConfidence>,
        robust: Option<RobustOptions>,
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

        let positions = (0..self.time_count()).collect::<Vec<_>>();
        let record = self
            .basis
            .record_subset(positions, series_count >= MIN_SHARED_LOMB_SERIES)?;
        if confidence.is_some_and(|confidence| confidence.noise() == LinearConfidence::Colored) {
            record
                .confidence_sampling
                .precompute_shared_irregular_plan();
        }

        (0..series_count)
            .into_par_iter()
            .map(|series| {
                let model = self
                    .basis
                    .model_at_latitude_for_record(latitudes[series], &record)?;
                let mut series_observations = Vec::with_capacity(self.time_count());
                for time in 0..self.time_count() {
                    series_observations.push(observations[time * series_count + series]);
                }
                solve_scalar_with_batch_confidence(
                    &model,
                    &series_observations,
                    robust,
                    confidence,
                    u64::try_from(series).expect("series index is representable as u64"),
                )
            })
            .collect()
    }
}

#[derive(Debug)]
struct CorrectionBasis {
    tidal_constituents: Vec<TidalConstituent>,
    model_constituent_count: usize,
    scalar_constituents: Vec<Constituent>,
    base_constituents: Vec<TidalConstituent>,
    recipes: Vec<CorrectionRecipe>,
    time_terms: Vec<TimeTerms>,
    reference_time_modified_julian_day: f64,
    time_span_days: f64,
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    sample_interval_hours: Option<f64>,
}

#[derive(Clone, Debug)]
struct ScalarInferenceLayout {
    tidal_constituents: Vec<TidalConstituent>,
    output_mappings: Vec<ScalarInferenceOutput>,
    reference_groups: Vec<ScalarInferenceReferenceGroup>,
    fit_count: usize,
}

#[derive(Clone, Debug)]
struct ScalarInferenceReferenceGroup {
    fit_index: usize,
    inferred_outputs: Vec<usize>,
}

#[derive(Clone, Debug)]
struct VectorInferenceLayout {
    tidal_constituents: Vec<TidalConstituent>,
    output_mappings: Vec<VectorInferenceOutput>,
    reference_groups: Vec<VectorInferenceReferenceGroup>,
    fit_count: usize,
    non_reference_count: usize,
}

#[derive(Clone, Debug)]
struct VectorInferenceReferenceGroup {
    fit_index: usize,
    inferred_outputs: Vec<usize>,
}

fn scalar_inference_layout(
    requested: &[TidalConstituent],
    relationships: &[ScalarInferenceRelation],
) -> Result<ScalarInferenceLayout, AnalysisError> {
    validate_tidal_constituents(requested)?;
    if relationships.is_empty() {
        return Err(AnalysisError::EmptyInference);
    }

    let mut inferred = HashSet::with_capacity(relationships.len());
    for (index, relationship) in relationships.iter().enumerate() {
        if !relationship.amplitude_ratio.is_finite() || relationship.amplitude_ratio < 0.0 {
            return Err(AnalysisError::InvalidInferenceAmplitudeRatio { index });
        }
        if !relationship.phase_offset_degrees.is_finite() {
            return Err(AnalysisError::InvalidInferencePhaseOffset { index });
        }
        if relationship.inferred == relationship.reference {
            return Err(AnalysisError::SelfInference { index });
        }
        if !inferred.insert(relationship.inferred) {
            return Err(AnalysisError::DuplicateInferredConstituent { index });
        }
    }
    for relationship in relationships {
        if inferred.contains(&relationship.reference) {
            return Err(AnalysisError::InferenceReferenceIsInferred {
                name: relationship.reference.name(),
            });
        }
    }

    let references = relationships
        .iter()
        .map(|relationship| relationship.reference)
        .fold(Vec::new(), |mut references, reference| {
            if !references.contains(&reference) {
                references.push(reference);
            }
            references
        });
    let reference_set = references.iter().copied().collect::<HashSet<_>>();
    let ordinary = requested
        .iter()
        .copied()
        .filter(|constituent| {
            !inferred.contains(constituent) && !reference_set.contains(constituent)
        })
        .collect::<Vec<_>>();
    let fit_count = ordinary.len() + references.len();
    let mut tidal_constituents = ordinary;
    tidal_constituents.extend(references.iter().copied());
    let mut output_mappings = (0..fit_count)
        .map(|source_fit_index| ScalarInferenceOutput {
            source_fit_index,
            ratio_real: 1.0,
            ratio_imaginary: 0.0,
            inferred: false,
        })
        .collect::<Vec<_>>();
    let ordinary_count = fit_count - references.len();
    let mut reference_groups = Vec::with_capacity(references.len());
    for (reference_position, reference) in references.iter().copied().enumerate() {
        let fit_index = ordinary_count + reference_position;
        let mut inferred_outputs = Vec::new();
        for relationship in relationships
            .iter()
            .filter(|relationship| relationship.reference == reference)
        {
            let phase = relationship.phase_offset_degrees.to_radians();
            let output_index = tidal_constituents.len();
            tidal_constituents.push(relationship.inferred);
            output_mappings.push(ScalarInferenceOutput {
                source_fit_index: fit_index,
                ratio_real: relationship.amplitude_ratio * phase.cos(),
                ratio_imaginary: relationship.amplitude_ratio * phase.sin(),
                inferred: true,
            });
            inferred_outputs.push(output_index);
        }
        reference_groups.push(ScalarInferenceReferenceGroup {
            fit_index,
            inferred_outputs,
        });
    }

    Ok(ScalarInferenceLayout {
        tidal_constituents,
        output_mappings,
        reference_groups,
        fit_count,
    })
}

fn vector_inference_layout(
    requested: &[TidalConstituent],
    relationships: &[VectorInferenceRelation],
) -> Result<VectorInferenceLayout, AnalysisError> {
    validate_tidal_constituents(requested)?;
    if relationships.is_empty() {
        return Err(AnalysisError::EmptyInference);
    }

    let mut inferred = HashSet::with_capacity(relationships.len());
    for (index, relationship) in relationships.iter().enumerate() {
        if !relationship.positive_amplitude_ratio.is_finite()
            || relationship.positive_amplitude_ratio < 0.0
            || !relationship.negative_amplitude_ratio.is_finite()
            || relationship.negative_amplitude_ratio < 0.0
        {
            return Err(AnalysisError::InvalidInferenceAmplitudeRatio { index });
        }
        if !relationship.positive_phase_offset_degrees.is_finite()
            || !relationship.negative_phase_offset_degrees.is_finite()
        {
            return Err(AnalysisError::InvalidInferencePhaseOffset { index });
        }
        if relationship.inferred == relationship.reference {
            return Err(AnalysisError::SelfInference { index });
        }
        if !inferred.insert(relationship.inferred) {
            return Err(AnalysisError::DuplicateInferredConstituent { index });
        }
    }
    for relationship in relationships {
        if inferred.contains(&relationship.reference) {
            return Err(AnalysisError::InferenceReferenceIsInferred {
                name: relationship.reference.name(),
            });
        }
    }

    let references = relationships
        .iter()
        .map(|relationship| relationship.reference)
        .fold(Vec::new(), |mut references, reference| {
            if !references.contains(&reference) {
                references.push(reference);
            }
            references
        });
    let reference_set = references.iter().copied().collect::<HashSet<_>>();
    let mut tidal_constituents = requested
        .iter()
        .copied()
        .filter(|constituent| {
            !inferred.contains(constituent) && !reference_set.contains(constituent)
        })
        .collect::<Vec<_>>();
    let non_reference_count = tidal_constituents.len();
    tidal_constituents.extend(references.iter().copied());
    let fit_count = tidal_constituents.len();
    let mut output_mappings = (0..fit_count)
        .map(|source_fit_index| VectorInferenceOutput {
            source_fit_index,
            positive_ratio: c64::new(1.0, 0.0),
            negative_ratio: c64::new(1.0, 0.0),
            inferred: false,
        })
        .collect::<Vec<_>>();
    let mut reference_groups = Vec::with_capacity(references.len());
    for (reference_position, reference) in references.iter().copied().enumerate() {
        let fit_index = non_reference_count + reference_position;
        let mut inferred_outputs = Vec::new();
        for relationship in relationships
            .iter()
            .filter(|relationship| relationship.reference == reference)
        {
            let positive_phase = relationship.positive_phase_offset_degrees.to_radians();
            let negative_phase = relationship.negative_phase_offset_degrees.to_radians();
            let output_index = tidal_constituents.len();
            tidal_constituents.push(relationship.inferred);
            output_mappings.push(VectorInferenceOutput {
                source_fit_index: fit_index,
                positive_ratio: c64::new(
                    relationship.positive_amplitude_ratio * positive_phase.cos(),
                    relationship.positive_amplitude_ratio * positive_phase.sin(),
                ),
                // Python UTide defines Rm with the opposite phase sign.
                negative_ratio: c64::new(
                    relationship.negative_amplitude_ratio * negative_phase.cos(),
                    -relationship.negative_amplitude_ratio * negative_phase.sin(),
                ),
                inferred: true,
            });
            inferred_outputs.push(output_index);
        }
        reference_groups.push(VectorInferenceReferenceGroup {
            fit_index,
            inferred_outputs,
        });
    }

    Ok(VectorInferenceLayout {
        tidal_constituents,
        output_mappings,
        reference_groups,
        fit_count,
        non_reference_count,
    })
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
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        Self::prepare_with_model_count(
            modified_julian_days,
            constituents,
            constituents.len(),
            solver_options,
        )
    }

    fn prepare_with_model_count(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        model_constituent_count: usize,
        solver_options: SolverOptions,
    ) -> Result<Self, AnalysisError> {
        let fit_options = solver_options.fit_options();
        let nodal_corrections = solver_options.nodal_corrections();
        validate_tidal_constituents(constituents)?;
        let (reference_time, time_span_days) =
            validate_time_with_options(modified_julian_days, model_constituent_count, fit_options)?;
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
                    base_nodal_terms: if nodal_corrections == NodalCorrections::Exact {
                        base_constituents
                            .iter()
                            .copied()
                            .map(|constituent| {
                                precompute_nodal_terms(constituent.metadata(), astronomy.cycles)
                            })
                            .collect()
                    } else {
                        Vec::new()
                    },
                    modified_julian_day: time,
                }
            })
            .collect();
        Ok(Self {
            tidal_constituents: constituents.to_vec(),
            model_constituent_count,
            scalar_constituents,
            base_constituents,
            recipes,
            time_terms,
            reference_time_modified_julian_day: reference_time,
            time_span_days,
            fit_options,
            phase_reference: solver_options.phase_reference(),
            nodal_corrections,
            sample_interval_hours: equidistant_sample_interval_hours(modified_julian_days),
        })
    }

    fn model_at_latitude(&self, latitude: f64) -> Result<FixedRawOls, AnalysisError> {
        let positions = (0..self.time_terms.len()).collect::<Vec<_>>();
        let record = self.record_subset(positions, true)?;
        self.model_at_latitude_for_record(latitude, &record)
    }

    fn scalar_inference_model_at_latitude_for_record(
        &self,
        latitude: f64,
        layout: &ScalarInferenceLayout,
        mode: InferenceMode,
        record: &RecordSubset,
    ) -> Result<FixedRawOls, AnalysisError> {
        validate_latitude(latitude)?;
        let harmonic_columns = layout.fit_count * 2;
        let column_count = harmonic_columns + 1 + usize::from(self.fit_options.trend);
        let mut design = Mat::zeros(record.positions.len(), column_count);
        let latitude_factors = latitude_factors(latitude);
        let constant_base_corrections = constant_base_nodal_corrections(
            self.nodal_corrections,
            &record.reference_base_nodal_terms,
            latitude_factors,
        );
        for (time_index, position) in record.positions.iter().copied().enumerate() {
            let terms = &self.time_terms[position];
            let exact_base_corrections;
            let base_corrections = if let Some(corrections) = &constant_base_corrections {
                corrections
            } else {
                exact_base_corrections = base_nodal_corrections(
                    self.nodal_corrections,
                    &terms.base_nodal_terms,
                    &record.reference_base_nodal_terms,
                    latitude_factors,
                );
                &exact_base_corrections
            };
            let basis_values = (0..self.tidal_constituents.len())
                .map(|constituent_index| {
                    let (nodal_amplitude, nodal_phase) =
                        self.recipes[constituent_index].combine_nodal(base_corrections);
                    let astronomical_phase = phase_cycles(
                        self.phase_reference,
                        terms.greenwich_phase[constituent_index],
                        record.reference_greenwich_phase[constituent_index],
                        terms.modified_julian_day,
                        record.reference_time,
                        record.scalar_constituents[constituent_index].frequency_cph,
                    );
                    let angle = TAU * (nodal_phase + astronomical_phase);
                    (nodal_amplitude * angle.cos(), nodal_amplitude * angle.sin())
                })
                .collect::<Vec<_>>();

            for fit_index in 0..layout.fit_count {
                design[(time_index, fit_index * 2)] = basis_values[fit_index].0;
                design[(time_index, fit_index * 2 + 1)] = basis_values[fit_index].1;
            }
            if mode == InferenceMode::Exact {
                for group in &layout.reference_groups {
                    for output_index in &group.inferred_outputs {
                        let mapping = layout.output_mappings[*output_index];
                        let (cosine, sine) = basis_values[*output_index];
                        design[(time_index, group.fit_index * 2)] +=
                            mapping.ratio_real * cosine - mapping.ratio_imaginary * sine;
                        design[(time_index, group.fit_index * 2 + 1)] +=
                            mapping.ratio_imaginary * cosine + mapping.ratio_real * sine;
                    }
                }
            }
            design[(time_index, harmonic_columns)] = 1.0;
            if self.fit_options.trend {
                design[(time_index, harmonic_columns + 1)] =
                    (terms.modified_julian_day - record.reference_time) / record.time_span_days;
            }
        }
        Ok(FixedRawOls::from_design_with_confidence_constituents(
            record.scalar_constituents[..layout.fit_count].to_vec(),
            record.scalar_constituents.clone(),
            Some(layout.fit_count - layout.reference_groups.len()),
            record.positions.len(),
            record.time_span_days,
            record.reference_time,
            record.confidence_sampling.clone(),
            self.fit_options,
            design,
        ))
    }

    fn vector_inference_design_at_latitude_for_record(
        &self,
        latitude: f64,
        layout: &VectorInferenceLayout,
        mode: InferenceMode,
        record: &RecordSubset,
    ) -> Result<Mat<c64>, AnalysisError> {
        validate_latitude(latitude)?;
        let harmonic_columns = layout.fit_count * 2;
        let column_count = harmonic_columns + 1 + usize::from(self.fit_options.trend);
        let mut design = Mat::zeros(record.positions.len(), column_count);
        let latitude_factors = latitude_factors(latitude);
        let constant_base_corrections = constant_base_nodal_corrections(
            self.nodal_corrections,
            &record.reference_base_nodal_terms,
            latitude_factors,
        );
        for (time_index, position) in record.positions.iter().copied().enumerate() {
            let terms = &self.time_terms[position];
            let exact_base_corrections;
            let base_corrections = if let Some(corrections) = &constant_base_corrections {
                corrections
            } else {
                exact_base_corrections = base_nodal_corrections(
                    self.nodal_corrections,
                    &terms.base_nodal_terms,
                    &record.reference_base_nodal_terms,
                    latitude_factors,
                );
                &exact_base_corrections
            };
            let basis_values = (0..self.tidal_constituents.len())
                .map(|constituent_index| {
                    let (nodal_amplitude, nodal_phase) =
                        self.recipes[constituent_index].combine_nodal(base_corrections);
                    let astronomical_phase = phase_cycles(
                        self.phase_reference,
                        terms.greenwich_phase[constituent_index],
                        record.reference_greenwich_phase[constituent_index],
                        terms.modified_julian_day,
                        record.reference_time,
                        record.scalar_constituents[constituent_index].frequency_cph,
                    );
                    let angle = TAU * (nodal_phase + astronomical_phase);
                    c64::new(nodal_amplitude * angle.cos(), nodal_amplitude * angle.sin())
                })
                .collect::<Vec<_>>();

            for ordinary in 0..layout.non_reference_count {
                design[(time_index, ordinary)] = basis_values[ordinary];
                design[(time_index, layout.non_reference_count + ordinary)] =
                    basis_values[ordinary].conj();
            }
            let positive_reference_start = layout.non_reference_count * 2;
            let negative_reference_start = positive_reference_start + layout.reference_groups.len();
            for (reference_position, group) in layout.reference_groups.iter().enumerate() {
                let mut positive = basis_values[group.fit_index];
                let mut negative = basis_values[group.fit_index].conj();
                if mode == InferenceMode::Exact {
                    for output_index in &group.inferred_outputs {
                        let mapping = layout.output_mappings[*output_index];
                        positive += basis_values[*output_index] * mapping.positive_ratio;
                        negative += basis_values[*output_index].conj() * mapping.negative_ratio;
                    }
                }
                design[(time_index, positive_reference_start + reference_position)] = positive;
                design[(time_index, negative_reference_start + reference_position)] = negative;
            }
            design[(time_index, harmonic_columns)] = c64::new(1.0, 0.0);
            if self.fit_options.trend {
                design[(time_index, harmonic_columns + 1)] = c64::new(
                    (terms.modified_julian_day - record.reference_time) / record.time_span_days,
                    0.0,
                );
            }
        }
        Ok(design)
    }

    fn record_subset(
        &self,
        positions: Vec<usize>,
        share_irregular_plan: bool,
    ) -> Result<RecordSubset, AnalysisError> {
        let modified_julian_days = positions
            .iter()
            .copied()
            .map(|position| self.time_terms[position].modified_julian_day)
            .collect::<Vec<_>>();
        let (subset_reference, subset_span) = validate_time_with_options(
            &modified_julian_days,
            self.model_constituent_count,
            self.fit_options,
        )?;
        let original_is_equidistant = self.sample_interval_hours.is_some();
        let (reference_time, time_span_days, scalar_constituents, confidence_sampling) =
            if original_is_equidistant {
                let full_count = self.time_terms.len();
                (
                    self.reference_time_modified_julian_day,
                    self.time_span_days,
                    self.scalar_constituents.clone(),
                    ConfidenceSampling::regular_gappy(
                        self.sample_interval_hours
                            .expect("equidistant record retains its sample interval"),
                        self.time_span_days,
                        full_count,
                        positions.clone(),
                    ),
                )
            } else {
                (
                    subset_reference,
                    subset_span,
                    scalar_constituents_at_reference(
                        &self.tidal_constituents,
                        &self.base_constituents,
                        &self.recipes,
                        subset_reference,
                    ),
                    ConfidenceSampling::irregular(
                        &modified_julian_days,
                        subset_span,
                        share_irregular_plan,
                    ),
                )
            };
        validate_derived_frequencies(&scalar_constituents)?;
        let reference_greenwich_phase =
            greenwich_phases_at_time(&self.base_constituents, &self.recipes, reference_time);
        let reference_base_nodal_terms =
            base_nodal_terms_at_time(&self.base_constituents, reference_time);
        Ok(RecordSubset {
            positions,
            scalar_constituents,
            reference_greenwich_phase,
            reference_base_nodal_terms,
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
        let column_count = harmonic_columns + 1 + usize::from(self.fit_options.trend);
        let mut design = Mat::zeros(record.positions.len(), column_count);
        let latitude_factors = latitude_factors(latitude);
        let constant_base_corrections = constant_base_nodal_corrections(
            self.nodal_corrections,
            &record.reference_base_nodal_terms,
            latitude_factors,
        );
        for (time_index, position) in record.positions.iter().copied().enumerate() {
            let terms = &self.time_terms[position];
            let exact_base_corrections;
            let base_corrections = if let Some(corrections) = &constant_base_corrections {
                corrections
            } else {
                exact_base_corrections = base_nodal_corrections(
                    self.nodal_corrections,
                    &terms.base_nodal_terms,
                    &record.reference_base_nodal_terms,
                    latitude_factors,
                );
                &exact_base_corrections
            };
            for constituent_index in 0..self.tidal_constituents.len() {
                let (nodal_amplitude, nodal_phase) =
                    self.recipes[constituent_index].combine_nodal(base_corrections);
                let astronomical_phase = phase_cycles(
                    self.phase_reference,
                    terms.greenwich_phase[constituent_index],
                    record.reference_greenwich_phase[constituent_index],
                    terms.modified_julian_day,
                    record.reference_time,
                    record.scalar_constituents[constituent_index].frequency_cph,
                );
                let angle = TAU * (nodal_phase + astronomical_phase);
                design[(time_index, constituent_index * 2)] = nodal_amplitude * angle.cos();
                design[(time_index, constituent_index * 2 + 1)] = nodal_amplitude * angle.sin();
            }
            design[(time_index, harmonic_columns)] = 1.0;
            if self.fit_options.trend {
                design[(time_index, harmonic_columns + 1)] =
                    (terms.modified_julian_day - record.reference_time) / record.time_span_days;
            }
        }
        Ok(FixedRawOls::from_design(
            record.scalar_constituents.clone(),
            record.positions.len(),
            record.time_span_days,
            record.reference_time,
            record.confidence_sampling.clone(),
            self.fit_options,
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
    reference_greenwich_phase: Vec<f64>,
    reference_base_nodal_terms: Vec<NodalTerms>,
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

fn greenwich_phases_at_time(
    base_constituents: &[TidalConstituent],
    recipes: &[CorrectionRecipe],
    modified_julian_day: f64,
) -> Vec<f64> {
    let astronomy = at_modified_julian_day(modified_julian_day);
    let base_phases = base_constituents
        .iter()
        .copied()
        .map(|constituent| base_greenwich_phase(constituent, astronomy.cycles))
        .collect::<Vec<_>>();
    recipes
        .iter()
        .map(|recipe| recipe.combine_phase(&base_phases))
        .collect()
}

fn base_nodal_terms_at_time(
    base_constituents: &[TidalConstituent],
    modified_julian_day: f64,
) -> Vec<NodalTerms> {
    let astronomy = at_modified_julian_day(modified_julian_day);
    base_constituents
        .iter()
        .copied()
        .map(|constituent| precompute_nodal_terms(constituent.metadata(), astronomy.cycles))
        .collect()
}

fn phase_cycles(
    phase_reference: PhaseReference,
    exact_greenwich_phase: f64,
    reference_greenwich_phase: f64,
    modified_julian_day: f64,
    reference_time: f64,
    frequency_cph: f64,
) -> f64 {
    let linear_advance = 24.0 * (modified_julian_day - reference_time) * frequency_cph;
    match phase_reference {
        PhaseReference::Greenwich => exact_greenwich_phase,
        PhaseReference::LinearTime => reference_greenwich_phase + linear_advance,
        PhaseReference::Raw => linear_advance,
    }
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

fn base_nodal_corrections(
    mode: NodalCorrections,
    exact_terms: &[NodalTerms],
    reference_terms: &[NodalTerms],
    latitude_factors: [f64; 3],
) -> Vec<(f64, f64)> {
    match mode {
        NodalCorrections::Exact => exact_terms
            .iter()
            .copied()
            .map(|terms| nodal_correction(terms, latitude_factors))
            .collect(),
        NodalCorrections::LinearTime => reference_terms
            .iter()
            .copied()
            .map(|terms| nodal_correction(terms, latitude_factors))
            .collect(),
        NodalCorrections::Disabled => vec![(1.0, 0.0); reference_terms.len()],
    }
}

fn constant_base_nodal_corrections(
    mode: NodalCorrections,
    reference_terms: &[NodalTerms],
    latitude_factors: [f64; 3],
) -> Option<Vec<(f64, f64)>> {
    match mode {
        NodalCorrections::Exact => None,
        NodalCorrections::LinearTime => Some(
            reference_terms
                .iter()
                .copied()
                .map(|terms| nodal_correction(terms, latitude_factors))
                .collect(),
        ),
        NodalCorrections::Disabled => Some(vec![(1.0, 0.0); reference_terms.len()]),
    }
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
    use super::{
        GreenwichNodalBatch, GreenwichNodalOls, NodalCorrections, PhaseReference, SolverOptions,
        shared_lomb_plan_groups, usize_to_f64,
    };
    use crate::{AnalysisError, FitOptions, LinearConfidence, TidalConstituent};

    fn times() -> Vec<f64> {
        (0_u32..745)
            .map(|index| 58_113.0 + f64::from(index) / 24.0)
            .collect()
    }

    #[test]
    fn shared_lomb_plans_require_amortization_and_are_batch_bounded() {
        assert_eq!(
            shared_lomb_plan_groups(&[2, 16, 30, 100, 50, 40, 20]),
            [false, false, true, true, true, true, false]
        );
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
    fn irregular_gappy_records_support_white_and_colored_confidence() {
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
        let colored = batch
            .solve_time_major_with_missing_and_linear_confidence(
                &observations,
                &[60.0],
                LinearConfidence::Colored,
            )
            .expect("colored confidence uses Lomb-Scargle for irregular data");
        assert!(
            colored[0]
                .amplitude_ci
                .as_ref()
                .expect("colored confidence intervals")
                .iter()
                .all(|value| value.is_finite())
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
        assert!(
            batch
                .solve_vector_time_major_with_missing_and_linear_confidence(
                    &observations,
                    &northward,
                    &[60.0],
                    LinearConfidence::Colored,
                )
                .is_ok()
        );
    }

    #[test]
    fn irregular_astronomy_options_use_each_retained_record_epoch() {
        let mut time = times();
        time[20] += 0.001;
        let constituents = [TidalConstituent::M2, TidalConstituent::K1];
        let mut observations = (0_u32..745)
            .map(|index| (f64::from(index) / 13.0).sin())
            .collect::<Vec<_>>();
        observations[0] = f64::NAN;
        let retained_time = time[1..].to_vec();
        let retained_observations = observations[1..].to_vec();

        for phase_reference in [PhaseReference::LinearTime, PhaseReference::Raw] {
            for nodal_corrections in [
                NodalCorrections::Exact,
                NodalCorrections::LinearTime,
                NodalCorrections::Disabled,
            ] {
                let solver_options = SolverOptions::new(FitOptions::default(), phase_reference)
                    .with_nodal_corrections(nodal_corrections);
                let batch = GreenwichNodalBatch::prepare_modified_julian_days_with_solver_options(
                    &time,
                    &constituents,
                    solver_options,
                )
                .expect("valid irregular batch");
                let actual = batch
                    .solve_time_major_with_missing(&observations, &[60.0])
                    .expect("valid irregular gappy series");
                let individual =
                    GreenwichNodalOls::prepare_modified_julian_days_with_solver_options(
                        &retained_time,
                        60.0,
                        &constituents,
                        solver_options,
                    )
                    .expect("valid retained-record model")
                    .solve(&retained_observations)
                    .expect("valid retained observations");
                assert_eq!(actual, [individual]);
                assert_eq!(batch.phase_reference(), phase_reference);
                assert_eq!(batch.nodal_corrections(), nodal_corrections);
                assert!(
                    (actual[0].reference_time_days
                        - retained_time[0].midpoint(retained_time[retained_time.len() - 1]))
                    .abs()
                        < 1e-12
                );
            }
        }
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
