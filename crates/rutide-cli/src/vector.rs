//! FVCOM depth-averaged vector-current application path.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use netcdf::{FileMut, Variable};
use rayon::ThreadPoolBuilder;
use rutide_core::{
    AnalysisError, Constituent, FitOptions, GreenwichNodalBatch, GreenwichNodalReconstructor,
    NodalCorrections, PhaseReference, ReconstructionFilter, SolverOptions, TidalConstituent,
    VectorInferenceBatch, VectorReconstruction, VectorSolution,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{
    AnalysisMethod, AnalyzeConfig, AppError, ConfidenceInterval, ConstituentOrder,
    ConstituentOrderMap, ConstituentSelection, ConstituentSelectionReport, CoreSamplingDiagnostics,
    InferenceReport, NodeSelection, ReconstructionReport, ResolvedConstituentSelection,
    RobustOptionsReport, SamplingSummary, SeriesSamplingDiagnostics, StageTimings,
    VectorInferenceConfig, constituent_order_indices, diagnose_sampling, encode_hex,
    nodal_profile_component, normalize_source_observation, order_profile_suffix,
    read_fvcom_time_axis, reconstruction_report, required_dimension_length, required_variable,
    resolve_constituent_selection, retain_time_major_rows, robust_termination_code,
    summarize_sampling, temporary_sibling, update_constituent_order_digest,
    update_inference_digest, update_reconstruction_filter_digest, update_sampling_digest,
    validate_config, validate_dimensions, validate_reconstruction_filter, validate_source_value,
    write_constituent_order_indices, write_inference_metadata, write_json_report,
    write_robust_schema_metadata, write_sampling_diagnostics, write_variable,
};

const VECTOR_OUTPUT_SCHEMA_VERSION: u32 = 10;

/// Configuration for one depth-averaged FVCOM current analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorAnalyzeConfig {
    /// Read-only source FVCOM `NetCDF` path.
    pub input: PathBuf,
    /// Destination `NetCDF` ellipse-coefficient path.
    pub output: PathBuf,
    /// Optional JSON run-report path.
    pub report: Option<PathBuf>,
    /// Element subset to analyze.
    pub elements: NodeSelection,
    /// Explicit or record-length-based constituent selection.
    pub constituent_selection: ConstituentSelection,
    /// Per-series constituent presentation ranking.
    pub constituent_order: ConstituentOrder,
    /// Optional constrained positive/negative rotary inferred constituents.
    pub inference: Option<VectorInferenceConfig>,
    /// Mean/trend terms included in the harmonic fit.
    pub fit_options: FitOptions,
    /// Astronomical argument used to reference reported phases.
    pub phase_reference: PhaseReference,
    /// Exact, midpoint-linearized, or disabled nodal/satellite corrections.
    pub nodal_corrections: NodalCorrections,
    /// Optional linearized or Monte Carlo ellipse intervals and noise model.
    pub confidence_interval: ConfidenceInterval,
    /// Ordinary or Cauchy robust least squares.
    pub analysis_method: AnalysisMethod,
    /// Optional complete-series reconstruction and constituent filter.
    pub reconstruction: Option<ReconstructionFilter>,
    /// Number of outer spatial worker threads.
    pub workers: usize,
    /// Permit replacing existing output and report files.
    pub overwrite: bool,
}

/// A small retained vector-current sample in the JSON run report.
#[derive(Clone, Debug, Serialize)]
pub struct VectorSampleResult {
    /// Original zero-based FVCOM element index.
    pub element_index: usize,
    /// Element-center latitude in degrees north.
    pub latitude_degrees_north: f64,
    /// Semi-major axes in constituent order.
    pub semi_major: Vec<f64>,
    /// Signed semi-minor axes in constituent order.
    pub semi_minor: Vec<f64>,
    /// Major-axis inclinations in degrees.
    pub inclination_degrees: Vec<f64>,
    /// Phases in degrees using the report's configured reference convention.
    pub phase_degrees: Vec<f64>,
    /// Percent ellipse energy in constituent order.
    pub percent_energy: Vec<f64>,
    /// Stable constituent indices in requested presentation-rank order.
    pub constituent_index_by_rank: Vec<usize>,
    /// Semi-major 95% confidence half-widths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semi_major_ci: Option<Vec<f64>>,
    /// Semi-minor 95% confidence half-widths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semi_minor_ci: Option<Vec<f64>>,
    /// Inclination 95% confidence half-widths in degrees.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inclination_ci_degrees: Option<Vec<f64>>,
    /// Phase 95% confidence half-widths in degrees.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_ci_degrees: Option<Vec<f64>>,
    /// Ellipse-energy signal-to-noise ratios.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_to_noise: Option<Vec<f64>>,
    /// Fitted eastward constant offset.
    pub eastward_mean: f64,
    /// Fitted northward constant offset.
    pub northward_mean: f64,
    /// Fitted eastward trend per day.
    pub eastward_slope_per_day: f64,
    /// Fitted northward trend per day.
    pub northward_slope_per_day: f64,
    /// Number of joint finite component samples used by the fit.
    pub observation_count: usize,
    /// Temporal and colored-spectrum coverage for this series.
    pub sampling: SeriesSamplingDiagnostics,
    /// Epoch at which fitted means are defined, as an MJD.
    pub reference_time_modified_julian_day: f64,
}

/// Machine-readable summary of one completed vector-current run.
#[derive(Clone, Debug, Serialize)]
pub struct VectorRunReport {
    /// Vector report schema version.
    pub schema_version: u32,
    /// Unix timestamp at report construction.
    pub created_unix_seconds: u64,
    /// `RUTide` package version.
    pub rutide_version: &'static str,
    /// Frozen solver profile name.
    pub profile: String,
    /// Source path as supplied to the application.
    pub input_path: String,
    /// Source container size, not bytes physically read.
    pub input_file_bytes: u64,
    /// Logical payload bytes requested from the five input variables.
    pub logical_input_bytes: u64,
    /// Output coefficient path.
    pub output_path: String,
    /// Completed output file size.
    pub output_file_bytes: u64,
    /// Number of finite timestamps retained for analysis.
    pub time_count: usize,
    /// Number of timestamps in the source time dimension.
    pub source_time_count: usize,
    /// Number of missing source timestamps and corresponding rows discarded.
    pub discarded_timestamp_count: usize,
    /// Number of analyzed elements.
    pub series_count: usize,
    /// Number of elements with at least one missing component sample.
    pub series_with_missing_observations: usize,
    /// Aggregate temporal and spectral coverage across fitted elements.
    pub sampling: SamplingSummary,
    /// Number of outer spatial workers.
    pub workers: usize,
    /// Least-squares method: `ols` or `robust`.
    pub analysis_method: &'static str,
    /// Whether a linear trend was included alongside the fitted mean.
    pub trend_enabled: bool,
    /// Astronomical phase reference: `greenwich`, `linear-time`, or `raw`.
    pub phase_reference: &'static str,
    /// Nodal/satellite corrections: `exact`, `linear-time`, or `disabled`.
    pub nodal_corrections: &'static str,
    /// Robust options when robust fitting is enabled.
    pub robust_options: Option<RobustOptionsReport>,
    /// Auditable constituent-selection method and threshold.
    pub constituent_selection: ConstituentSelectionReport,
    /// Requested presentation ranking: selection, PE, SNR, frequency, or explicit.
    pub constituent_order: &'static str,
    /// Complete explicit presentation permutation, when supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_constituent_order: Option<Vec<&'static str>>,
    /// Whether presentation permutations differ between fitted spatial series.
    pub constituent_order_varies_by_series: bool,
    /// Inference configuration when inferred constituents are enabled.
    pub inference: Option<InferenceReport>,
    /// Confidence method: `none`, `linear`, or `monte-carlo`.
    pub confidence_interval: &'static str,
    /// Residual-noise model when confidence intervals are enabled.
    pub confidence_noise: Option<&'static str>,
    /// Number of coefficient realizations for Monte Carlo confidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monte_carlo_realizations: Option<usize>,
    /// Root random seed for Monte Carlo confidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monte_carlo_seed: Option<u64>,
    /// Complete-series reconstruction configuration, when enabled.
    pub reconstruction: Option<ReconstructionReport>,
    /// Constituent names in coefficient order.
    pub constituents: Vec<String>,
    /// Reference-time frequencies in cycles per hour.
    pub frequency_cph: Vec<f64>,
    /// Whether reference frequencies differ across fitted series.
    pub frequency_varies_by_series: bool,
    /// SHA-256 over canonical element metadata and every numeric result.
    pub result_sha256: String,
    /// Separately measured application stages.
    pub timings: StageTimings,
    /// First three results in output order.
    pub sample_results: Vec<VectorSampleResult>,
}

struct VectorInputData {
    modified_julian_days: Vec<f64>,
    source_time_count: usize,
    discarded_timestamp_count: usize,
    element_indices: Vec<usize>,
    latitudes: Vec<f64>,
    eastward: Vec<f64>,
    northward: Vec<f64>,
    observation_counts: Vec<usize>,
    input_file_bytes: u64,
    logical_input_bytes: u64,
}

enum VectorAnalysisBatch {
    Standard(GreenwichNodalBatch),
    Inferred(VectorInferenceBatch),
}

impl VectorAnalysisBatch {
    fn prepare(
        times: &[f64],
        constituents: &[TidalConstituent],
        inference: Option<&VectorInferenceConfig>,
        fit_options: FitOptions,
        phase_reference: PhaseReference,
        nodal_corrections: NodalCorrections,
    ) -> Result<Self, AnalysisError> {
        let solver_options = SolverOptions::new(fit_options, phase_reference)
            .with_nodal_corrections(nodal_corrections);
        match inference {
            Some(inference) => {
                VectorInferenceBatch::prepare_modified_julian_days_with_solver_options(
                    times,
                    constituents,
                    &inference.relationships,
                    inference.mode,
                    solver_options,
                )
                .map(Self::Inferred)
            }
            None => GreenwichNodalBatch::prepare_modified_julian_days_with_solver_options(
                times,
                constituents,
                solver_options,
            )
            .map(Self::Standard),
        }
    }

    fn constituents(&self) -> &[Constituent] {
        match self {
            Self::Standard(batch) => batch.constituents(),
            Self::Inferred(batch) => batch.constituents(),
        }
    }

    fn tidal_constituents(&self) -> &[TidalConstituent] {
        match self {
            Self::Standard(batch) => batch.tidal_constituents(),
            Self::Inferred(batch) => batch.tidal_constituents(),
        }
    }

    fn reference_time_modified_julian_day(&self) -> f64 {
        match self {
            Self::Standard(batch) => batch.reference_time_modified_julian_day(),
            Self::Inferred(batch) => batch.reference_time_modified_julian_day(),
        }
    }

    fn constituents_at_reference_modified_julian_day(
        &self,
        reference_time: f64,
    ) -> Result<Vec<Constituent>, AnalysisError> {
        match self {
            Self::Standard(batch) => {
                batch.constituents_at_reference_modified_julian_day(reference_time)
            }
            Self::Inferred(batch) => {
                batch.constituents_at_reference_modified_julian_day(reference_time)
            }
        }
    }

    fn reconstructor_modified_julian_days(
        &self,
        times: &[f64],
    ) -> Result<GreenwichNodalReconstructor, AnalysisError> {
        match self {
            Self::Standard(batch) => batch.reconstructor_modified_julian_days(times),
            Self::Inferred(batch) => batch.reconstructor_modified_julian_days(times),
        }
    }
}

/// Analyze FVCOM `ua(time, nele)` and `va(time, nele)` currents.
///
/// A time sample is omitted from both components when either source component
/// is missing. Infinities remain hard input errors.
///
/// # Errors
///
/// Returns [`AppError`] when configuration, source schema, observations,
/// numerical analysis, or output serialization fails.
#[allow(
    clippy::too_many_lines,
    reason = "top-level orchestration keeps separately timed stages visible"
)]
pub fn analyze_vector(config: &VectorAnalyzeConfig) -> Result<VectorRunReport, AppError> {
    validate_vector_config(config)?;
    faer::set_global_parallelism(faer::Par::Seq);
    let total_start = Instant::now();

    let input_start = Instant::now();
    let input = read_fvcom_vector(&config.input, &config.elements)?;
    let input_seconds = input_start.elapsed().as_secs_f64();

    let preparation_start = Instant::now();
    let selection =
        resolve_constituent_selection(&config.constituent_selection, &input.modified_julian_days)?;
    let batch = VectorAnalysisBatch::prepare(
        &input.modified_julian_days,
        &selection.constituents,
        config.inference.as_ref(),
        config.fit_options,
        config.phase_reference,
        config.nodal_corrections,
    )?;
    super::shared_constituent_order_indices(&config.constituent_order, batch.constituents())?;
    if let Some(filter) = &config.reconstruction {
        validate_reconstruction_filter(filter, batch.tidal_constituents())?;
    }
    let preparation_seconds = preparation_start.elapsed().as_secs_f64();

    let worker_pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()?;
    let solve_start = Instant::now();
    let solutions = worker_pool.install(|| match &batch {
        VectorAnalysisBatch::Standard(batch) => {
            match (config.analysis_method, config.confidence_interval) {
                (AnalysisMethod::Ols, ConfidenceInterval::None) => batch
                    .solve_vector_time_major_with_missing(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                    ),
                (AnalysisMethod::Ols, ConfidenceInterval::Linear(noise)) => batch
                    .solve_vector_time_major_with_missing_and_linear_confidence(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                        noise,
                    ),
                (AnalysisMethod::Ols, ConfidenceInterval::MonteCarlo { options, noise }) => batch
                    .solve_vector_time_major_with_missing_and_monte_carlo_confidence(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                        options,
                        noise,
                    ),
                (AnalysisMethod::Robust(options), ConfidenceInterval::None) => batch
                    .solve_vector_time_major_with_missing_robust(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                        options,
                    ),
                (AnalysisMethod::Robust(options), ConfidenceInterval::Linear(noise)) => batch
                    .solve_vector_time_major_with_missing_robust_and_linear_confidence(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                        options,
                        noise,
                    ),
                (
                    AnalysisMethod::Robust(robust_options),
                    ConfidenceInterval::MonteCarlo {
                        options: monte_carlo_options,
                        noise,
                    },
                ) => batch.solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence(
                    &input.eastward,
                    &input.northward,
                    &input.latitudes,
                    robust_options,
                    monte_carlo_options,
                    noise,
                ),
            }
        }
        VectorAnalysisBatch::Inferred(batch) => {
            match (config.analysis_method, config.confidence_interval) {
                (AnalysisMethod::Ols, ConfidenceInterval::None) => batch
                    .solve_vector_time_major_with_missing(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                    ),
                (AnalysisMethod::Ols, ConfidenceInterval::Linear(noise)) => batch
                    .solve_vector_time_major_with_missing_and_linear_confidence(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                        noise,
                    ),
                (AnalysisMethod::Ols, ConfidenceInterval::MonteCarlo { options, noise }) => batch
                    .solve_vector_time_major_with_missing_and_monte_carlo_confidence(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                        options,
                        noise,
                    ),
                (AnalysisMethod::Robust(options), ConfidenceInterval::None) => batch
                    .solve_vector_time_major_with_missing_robust(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                        options,
                    ),
                (AnalysisMethod::Robust(options), ConfidenceInterval::Linear(noise)) => batch
                    .solve_vector_time_major_with_missing_robust_and_linear_confidence(
                        &input.eastward,
                        &input.northward,
                        &input.latitudes,
                        options,
                        noise,
                    ),
                (
                    AnalysisMethod::Robust(robust_options),
                    ConfidenceInterval::MonteCarlo {
                        options: monte_carlo_options,
                        noise,
                    },
                ) => batch.solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence(
                    &input.eastward,
                    &input.northward,
                    &input.latitudes,
                    robust_options,
                    monte_carlo_options,
                    noise,
                ),
            }
        }
    })?;
    let solve_seconds = solve_start.elapsed().as_secs_f64();

    let reconstruction_start = Instant::now();
    let reconstruction = if let Some(filter) = config.reconstruction.as_ref() {
        let reconstructor =
            batch.reconstructor_modified_julian_days(&input.modified_julian_days)?;
        Some(worker_pool.install(|| {
            reconstructor.reconstruct_many_vectors_series_major(
                &solutions,
                &input.latitudes,
                filter,
            )
        })?)
    } else {
        None
    };
    let reconstruction_seconds = reconstruction_start.elapsed().as_secs_f64();

    let result_start = Instant::now();
    let series_frequency_cph = vector_solution_frequencies(&batch, &solutions)?;
    let sampling_diagnostics = diagnose_sampling(
        &worker_pool,
        &input.modified_julian_days,
        &series_frequency_cph,
        |time, series| {
            let index = time * input.element_indices.len() + series;
            input.eastward[index].is_finite() && input.northward[index].is_finite()
        },
    )?;
    if sampling_diagnostics
        .iter()
        .zip(&input.observation_counts)
        .any(|(diagnostics, count)| diagnostics.observation_count != *count)
    {
        return Err(AppError::Invalid(
            "sampling diagnostic observation counts differ from fitted vector inputs".to_owned(),
        ));
    }
    let sampling_summary = summarize_sampling(&sampling_diagnostics)?;
    let constituent_index_by_rank = constituent_order_indices(
        &config.constituent_order,
        batch.constituents(),
        &series_frequency_cph,
        &solutions,
    )?;
    let result_sha256 = vector_result_digest(
        &input.element_indices,
        &input.latitudes,
        batch.constituents(),
        &series_frequency_cph,
        &solutions,
        config.fit_options,
        config.phase_reference,
        config.nodal_corrections,
        &config.constituent_order,
        &constituent_index_by_rank,
        &sampling_diagnostics,
        config.analysis_method,
        config.confidence_interval,
        config
            .inference
            .as_ref()
            .map(VectorInferenceConfig::report)
            .as_ref(),
        config
            .reconstruction
            .as_ref()
            .zip(reconstruction.as_deref()),
    )?;
    let sample_results = retained_vector_samples(
        &input,
        &solutions,
        &constituent_index_by_rank,
        &sampling_diagnostics,
    );
    let result_processing_seconds = result_start.elapsed().as_secs_f64();

    let output_start = Instant::now();
    write_vector_output(
        &config.output,
        config.overwrite,
        &VectorOutputData {
            input: &input,
            constituents: batch.constituents(),
            series_frequency_cph: &series_frequency_cph,
            solutions: &solutions,
            constituent_order: &config.constituent_order,
            constituent_index_by_rank: &constituent_index_by_rank,
            sampling_diagnostics: &sampling_diagnostics,
            result_sha256: &result_sha256,
            selection: &selection,
            inference: config.inference.as_ref(),
            fit_options: config.fit_options,
            phase_reference: config.phase_reference,
            nodal_corrections: config.nodal_corrections,
            analysis_method: config.analysis_method,
            confidence_interval: config.confidence_interval,
            reference_time_modified_julian_day: batch.reference_time_modified_julian_day(),
            reconstruction: config
                .reconstruction
                .as_ref()
                .zip(reconstruction.as_deref()),
        },
    )?;
    let output_seconds = output_start.elapsed().as_secs_f64();
    let total_seconds = total_start.elapsed().as_secs_f64();
    let output_file_bytes = fs::metadata(&config.output)?.len();
    let created_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(std::io::Error::other)?
        .as_secs();
    let report = VectorRunReport {
        schema_version: VECTOR_OUTPUT_SCHEMA_VERSION,
        created_unix_seconds,
        rutide_version: rutide_core::VERSION,
        profile: vector_profile(
            &selection,
            config.analysis_method,
            config.inference.is_some(),
            config.fit_options,
            config.phase_reference,
            config.nodal_corrections,
            &config.constituent_order,
        ),
        input_path: config.input.to_string_lossy().into_owned(),
        input_file_bytes: input.input_file_bytes,
        logical_input_bytes: input.logical_input_bytes,
        output_path: config.output.to_string_lossy().into_owned(),
        output_file_bytes,
        time_count: input.modified_julian_days.len(),
        source_time_count: input.source_time_count,
        discarded_timestamp_count: input.discarded_timestamp_count,
        series_count: input.element_indices.len(),
        series_with_missing_observations: input
            .observation_counts
            .iter()
            .filter(|count| **count != input.modified_julian_days.len())
            .count(),
        sampling: sampling_summary,
        workers: config.workers,
        analysis_method: config.analysis_method.name(),
        trend_enabled: config.fit_options.trend,
        phase_reference: config.phase_reference.name(),
        nodal_corrections: config.nodal_corrections.name(),
        robust_options: match config.analysis_method {
            AnalysisMethod::Ols => None,
            AnalysisMethod::Robust(options) => Some(options.into()),
        },
        constituent_selection: selection.report,
        constituent_order: config.constituent_order.name(),
        explicit_constituent_order: config.constituent_order.explicit_names(),
        constituent_order_varies_by_series: constituent_index_by_rank.varies_by_series(),
        inference: config.inference.as_ref().map(VectorInferenceConfig::report),
        confidence_interval: config.confidence_interval.method(),
        confidence_noise: config.confidence_interval.noise(),
        monte_carlo_realizations: config
            .confidence_interval
            .monte_carlo_options()
            .map(|options| options.realizations),
        monte_carlo_seed: config
            .confidence_interval
            .monte_carlo_options()
            .map(|options| options.seed),
        reconstruction: config
            .reconstruction
            .as_ref()
            .map(|filter| reconstruction_report(filter, input.modified_julian_days.len())),
        constituents: batch
            .constituents()
            .iter()
            .map(|constituent| constituent.name.clone())
            .collect(),
        frequency_cph: batch
            .constituents()
            .iter()
            .map(|constituent| constituent.frequency_cph)
            .collect(),
        frequency_varies_by_series: series_frequency_cph
            .windows(2)
            .any(|pair| pair[0] != pair[1]),
        result_sha256,
        timings: StageTimings {
            input_seconds,
            preparation_seconds,
            solve_seconds,
            reconstruction_seconds,
            result_processing_seconds,
            output_seconds,
            total_seconds,
        },
        sample_results,
    };
    if let Some(path) = &config.report {
        write_json_report(path, config.overwrite, &report)?;
    }
    Ok(report)
}

fn validate_vector_config(config: &VectorAnalyzeConfig) -> Result<(), AppError> {
    validate_config(&AnalyzeConfig {
        input: config.input.clone(),
        output: config.output.clone(),
        report: config.report.clone(),
        nodes: config.elements.clone(),
        constituent_selection: config.constituent_selection.clone(),
        constituent_order: config.constituent_order.clone(),
        inference: None,
        fit_options: config.fit_options,
        phase_reference: config.phase_reference,
        nodal_corrections: config.nodal_corrections,
        confidence_interval: config.confidence_interval,
        analysis_method: config.analysis_method,
        reconstruction: config.reconstruction.clone(),
        workers: config.workers,
        overwrite: config.overwrite,
    })?;
    if config
        .inference
        .as_ref()
        .is_some_and(|inference| inference.relationships.is_empty())
    {
        return Err(AppError::Invalid(
            "inference requires at least one relationship".to_owned(),
        ));
    }
    Ok(())
}

fn vector_profile(
    selection: &ResolvedConstituentSelection,
    analysis_method: AnalysisMethod,
    inferred: bool,
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    constituent_order: &ConstituentOrder,
) -> String {
    let selection = match selection.report.method {
        "explicit" => "fixed-constituents",
        "rayleigh" => "rayleigh-auto",
        _ => unreachable!("selection methods are constructed internally"),
    };
    let inference = if inferred { "inference-" } else { "" };
    let ordering = order_profile_suffix(constituent_order);
    let trend = if fit_options.trend { "" } else { "-no-trend" };
    format!(
        "{selection}-{}-{}-vector-{inference}{}{ordering}{trend}",
        phase_reference.name(),
        nodal_profile_component(nodal_corrections),
        analysis_method.name(),
    )
}

fn read_fvcom_vector(path: &Path, selection: &NodeSelection) -> Result<VectorInputData, AppError> {
    let input_file_bytes = fs::metadata(path)?.len();
    let dataset = netcdf::open(path)?;
    let time_count = required_dimension_length(&dataset, "time")?;
    let element_count = required_dimension_length(&dataset, "nele")?;
    let element_indices = resolve_element_selection(selection, element_count)?;
    let series_count = element_indices.len();

    let (time_axis, time_element_bytes) = read_fvcom_time_axis(&dataset, time_count)?;

    let latitude_variable = required_variable(&dataset, "latc")?;
    validate_dimensions(&latitude_variable, &[("nele", element_count)])?;
    let latitude_fill = latitude_variable.fill_value::<f32>()?;
    let eastward_variable = required_vector_variable(&dataset, "ua", time_count, element_count)?;
    let northward_variable = required_vector_variable(&dataset, "va", time_count, element_count)?;
    let eastward_fill = eastward_variable.fill_value::<f32>()?;
    let northward_fill = northward_variable.fill_value::<f32>()?;

    let is_prefix = element_indices.iter().copied().eq(0..series_count);
    let (latitude_values, eastward, northward) = if is_prefix {
        (
            latitude_variable.get_values::<f64, _>(0..series_count)?,
            eastward_variable.get_values::<f64, _>((.., 0..series_count))?,
            northward_variable.get_values::<f64, _>((.., 0..series_count))?,
        )
    } else {
        let mut latitude_values = Vec::with_capacity(series_count);
        let mut eastward = vec![0.0; time_count * series_count];
        let mut northward = vec![0.0; time_count * series_count];
        for (series, element) in element_indices.iter().copied().enumerate() {
            latitude_values.push(latitude_variable.get_value::<f64, _>(element)?);
            let eastward_column = eastward_variable.get_values::<f64, _>((.., element))?;
            let northward_column = northward_variable.get_values::<f64, _>((.., element))?;
            for time in 0..time_count {
                eastward[time * series_count + series] = eastward_column[time];
                northward[time * series_count + series] = northward_column[time];
            }
        }
        (latitude_values, eastward, northward)
    };
    let mut eastward = retain_time_major_rows(
        eastward,
        time_count,
        series_count,
        time_axis.retained_indices(),
    )?;
    let mut northward = retain_time_major_rows(
        northward,
        time_count,
        series_count,
        time_axis.retained_indices(),
    )?;

    for (series, value) in latitude_values.iter().copied().enumerate() {
        validate_source_value("latc", value, latitude_fill, series, 0)?;
    }
    let mut observation_counts = vec![0; series_count];
    for index in 0..eastward.len() {
        let series = index % series_count;
        let time = index / series_count;
        eastward[index] =
            normalize_source_observation("ua", eastward[index], eastward_fill, series, time)?;
        northward[index] =
            normalize_source_observation("va", northward[index], northward_fill, series, time)?;
        if eastward[index].is_finite() && northward[index].is_finite() {
            observation_counts[series] += 1;
        } else {
            eastward[index] = f64::NAN;
            northward[index] = f64::NAN;
        }
    }

    let value_count = time_count
        .checked_mul(series_count)
        .ok_or_else(|| AppError::Invalid("logical input size exceeds usize".to_owned()))?;
    let logical_input_bytes = logical_input_bytes(&[
        (time_count, time_element_bytes[0]),
        (time_count, time_element_bytes[1]),
        (series_count, latitude_variable.vartype().size()),
        (value_count, eastward_variable.vartype().size()),
        (value_count, northward_variable.vartype().size()),
    ])?;
    let discarded_timestamp_count = time_axis.discarded_count();
    let source_time_count = time_axis.source_count();
    let (modified_julian_days, _) = time_axis.into_parts();
    Ok(VectorInputData {
        modified_julian_days,
        source_time_count,
        discarded_timestamp_count,
        element_indices,
        latitudes: latitude_values,
        eastward,
        northward,
        observation_counts,
        input_file_bytes,
        logical_input_bytes,
    })
}

fn required_vector_variable<'dataset>(
    dataset: &'dataset netcdf::File,
    name: &str,
    time_count: usize,
    element_count: usize,
) -> Result<Variable<'dataset>, AppError> {
    let variable = required_variable(dataset, name)?;
    validate_dimensions(&variable, &[("time", time_count), ("nele", element_count)])?;
    Ok(variable)
}

fn resolve_element_selection(
    selection: &NodeSelection,
    element_count: usize,
) -> Result<Vec<usize>, AppError> {
    if element_count == 0 {
        return Err(AppError::Invalid(
            "source nele dimension must not be empty".to_owned(),
        ));
    }
    match selection {
        NodeSelection::All => Ok((0..element_count).collect()),
        NodeSelection::Prefix(count) => {
            if *count == 0 || *count > element_count {
                return Err(AppError::Invalid(format!(
                    "element prefix must be between 1 and {element_count}, received {count}"
                )));
            }
            Ok((0..*count).collect())
        }
        NodeSelection::Indices(indices) => {
            if indices.is_empty() {
                return Err(AppError::Invalid(
                    "explicit element selection must not be empty".to_owned(),
                ));
            }
            let mut unique = BTreeSet::new();
            for index in indices.iter().copied() {
                if index >= element_count {
                    return Err(AppError::Invalid(format!(
                        "element index {index} is outside source element count {element_count}"
                    )));
                }
                if !unique.insert(index) {
                    return Err(AppError::Invalid(format!(
                        "element index {index} appears more than once"
                    )));
                }
            }
            Ok(indices.clone())
        }
    }
}

fn logical_input_bytes(fields: &[(usize, usize)]) -> Result<u64, AppError> {
    fields
        .iter()
        .copied()
        .try_fold(0_u64, |total, (count, width)| {
            let count = u64::try_from(count)
                .map_err(|_| AppError::Invalid("logical input size exceeds u64".to_owned()))?;
            let width = u64::try_from(width)
                .map_err(|_| AppError::Invalid("source element size exceeds u64".to_owned()))?;
            total
                .checked_add(count.checked_mul(width).ok_or_else(|| {
                    AppError::Invalid("logical input byte count overflows u64".to_owned())
                })?)
                .ok_or_else(|| {
                    AppError::Invalid("logical input byte count overflows u64".to_owned())
                })
        })
}

fn vector_solution_frequencies(
    batch: &VectorAnalysisBatch,
    solutions: &[VectorSolution],
) -> Result<Vec<Vec<f64>>, AnalysisError> {
    solutions
        .iter()
        .map(|solution| {
            batch
                .constituents_at_reference_modified_julian_day(solution.reference_time_days)
                .map(|constituents| {
                    constituents
                        .into_iter()
                        .map(|constituent| constituent.frequency_cph)
                        .collect()
                })
        })
        .collect()
}

type VectorConfidenceValues<'solution> = (
    &'solution [f64],
    &'solution [f64],
    &'solution [f64],
    &'solution [f64],
    &'solution [f64],
);

fn vector_confidence_values(
    solution: &VectorSolution,
    requested: ConfidenceInterval,
) -> Result<Option<VectorConfidenceValues<'_>>, AppError> {
    match (
        requested,
        solution.semi_major_ci.as_deref(),
        solution.semi_minor_ci.as_deref(),
        solution.inclination_ci_degrees.as_deref(),
        solution.phase_ci_degrees.as_deref(),
        solution.signal_to_noise.as_deref(),
    ) {
        (ConfidenceInterval::None, None, None, None, None, None) => Ok(None),
        (
            ConfidenceInterval::Linear(_) | ConfidenceInterval::MonteCarlo { .. },
            Some(major),
            Some(minor),
            Some(theta),
            Some(phase),
            Some(snr),
        ) => Ok(Some((major, minor, theta, phase, snr))),
        _ => Err(AppError::Invalid(
            "solver returned inconsistent vector confidence fields".to_owned(),
        )),
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the canonical digest keeps every result-shaping input explicit"
)]
fn vector_result_digest(
    element_indices: &[usize],
    latitudes: &[f64],
    constituents: &[Constituent],
    series_frequency_cph: &[Vec<f64>],
    solutions: &[VectorSolution],
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    constituent_order: &ConstituentOrder,
    constituent_index_by_rank: &ConstituentOrderMap,
    sampling_diagnostics: &[CoreSamplingDiagnostics],
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
    inference: Option<&InferenceReport>,
    reconstruction: Option<(&ReconstructionFilter, &[VectorReconstruction])>,
) -> Result<String, AppError> {
    let mut digest = Sha256::new();
    digest.update(b"rutide-vector-sampling-v10\0");
    digest.update([u8::from(fit_options.trend)]);
    digest.update(phase_reference.name().as_bytes());
    digest.update([0]);
    digest.update(nodal_corrections.name().as_bytes());
    digest.update([0]);
    update_constituent_order_digest(&mut digest, constituent_order, constituent_index_by_rank);
    update_sampling_digest(&mut digest, sampling_diagnostics)?;
    digest.update(analysis_method.name().as_bytes());
    if let AnalysisMethod::Robust(options) = analysis_method {
        digest.update(options.tuning_constant.to_bits().to_le_bytes());
        digest.update(options.tolerance.to_bits().to_le_bytes());
        digest.update(
            u64::try_from(options.max_iterations)
                .map_err(|_| AppError::Invalid("robust iteration limit exceeds u64".to_owned()))?
                .to_le_bytes(),
        );
    }
    digest.update([0]);
    digest.update(confidence_interval.method().as_bytes());
    digest.update([0]);
    if let Some(noise) = confidence_interval.noise() {
        digest.update(noise.as_bytes());
    }
    if let Some(options) = confidence_interval.monte_carlo_options() {
        digest.update(
            u64::try_from(options.realizations)
                .map_err(|_| {
                    AppError::Invalid("Monte Carlo realization count exceeds u64".to_owned())
                })?
                .to_le_bytes(),
        );
        digest.update(options.seed.to_le_bytes());
    }
    digest.update([0]);
    update_inference_digest(&mut digest, inference);
    for constituent in constituents {
        digest.update(constituent.name.as_bytes());
        digest.update([0]);
    }
    for (((element_index, latitude), frequency_cph), solution) in element_indices
        .iter()
        .copied()
        .zip(latitudes.iter().copied())
        .zip(series_frequency_cph)
        .zip(solutions)
    {
        digest.update(
            u64::try_from(element_index)
                .map_err(|_| AppError::Invalid("element index exceeds u64".to_owned()))?
                .to_le_bytes(),
        );
        digest.update(latitude.to_bits().to_le_bytes());
        digest.update(solution.reference_time_days.to_bits().to_le_bytes());
        for values in [
            frequency_cph.as_slice(),
            &solution.semi_major,
            &solution.semi_minor,
            &solution.inclination_degrees,
            &solution.phase_degrees,
            &solution.percent_energy,
        ] {
            for value in values {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
        if let Some(values) = vector_confidence_values(solution, confidence_interval)? {
            for values in [values.0, values.1, values.2, values.3, values.4] {
                for value in values {
                    digest.update(value.to_bits().to_le_bytes());
                }
            }
        }
        for value in [
            solution.eastward_mean,
            solution.northward_mean,
            solution.eastward_slope_per_day,
            solution.northward_slope_per_day,
        ] {
            digest.update(value.to_bits().to_le_bytes());
        }
        if let Some(robust) = &solution.robust {
            digest.update(
                u64::try_from(robust.iterations)
                    .map_err(|_| AppError::Invalid("robust iterations exceed u64".to_owned()))?
                    .to_le_bytes(),
            );
            digest.update(robust.residual_scale.to_bits().to_le_bytes());
            digest.update(robust.ols_rms_residual.to_bits().to_le_bytes());
            digest.update(robust.rms_residual.to_bits().to_le_bytes());
            digest.update(robust_termination_code(robust.termination).to_le_bytes());
            for values in [&robust.weights, &robust.leverage] {
                for value in values {
                    digest.update(value.to_bits().to_le_bytes());
                }
            }
        }
    }
    if let Some((filter, values)) = reconstruction {
        update_reconstruction_filter_digest(&mut digest, filter);
        for series in values {
            for component in [&series.eastward, &series.northward] {
                for value in component {
                    digest.update(value.to_bits().to_le_bytes());
                }
            }
        }
    } else {
        digest.update(b"no-reconstruction\0");
    }
    Ok(encode_hex(&digest.finalize()))
}

fn retained_vector_samples(
    input: &VectorInputData,
    solutions: &[VectorSolution],
    constituent_index_by_rank: &ConstituentOrderMap,
    sampling_diagnostics: &[CoreSamplingDiagnostics],
) -> Vec<VectorSampleResult> {
    input
        .element_indices
        .iter()
        .copied()
        .zip(input.latitudes.iter().copied())
        .zip(input.observation_counts.iter().copied())
        .zip(solutions)
        .enumerate()
        .take(3)
        .map(
            |(series, (((element_index, latitude_degrees_north), observation_count), solution))| {
                VectorSampleResult {
                    element_index,
                    latitude_degrees_north,
                    semi_major: solution.semi_major.clone(),
                    semi_minor: solution.semi_minor.clone(),
                    inclination_degrees: solution.inclination_degrees.clone(),
                    phase_degrees: solution.phase_degrees.clone(),
                    percent_energy: solution.percent_energy.clone(),
                    constituent_index_by_rank: constituent_index_by_rank
                        .row(series)
                        .iter()
                        .copied()
                        .map(usize::from)
                        .collect(),
                    semi_major_ci: solution.semi_major_ci.clone(),
                    semi_minor_ci: solution.semi_minor_ci.clone(),
                    inclination_ci_degrees: solution.inclination_ci_degrees.clone(),
                    phase_ci_degrees: solution.phase_ci_degrees.clone(),
                    signal_to_noise: solution.signal_to_noise.clone(),
                    eastward_mean: solution.eastward_mean,
                    northward_mean: solution.northward_mean,
                    eastward_slope_per_day: solution.eastward_slope_per_day,
                    northward_slope_per_day: solution.northward_slope_per_day,
                    observation_count,
                    sampling: SeriesSamplingDiagnostics::from(&sampling_diagnostics[series]),
                    reference_time_modified_julian_day: solution.reference_time_days,
                }
            },
        )
        .collect()
}

struct VectorOutputData<'data> {
    input: &'data VectorInputData,
    constituents: &'data [Constituent],
    series_frequency_cph: &'data [Vec<f64>],
    solutions: &'data [VectorSolution],
    constituent_order: &'data ConstituentOrder,
    constituent_index_by_rank: &'data ConstituentOrderMap,
    sampling_diagnostics: &'data [CoreSamplingDiagnostics],
    result_sha256: &'data str,
    selection: &'data ResolvedConstituentSelection,
    inference: Option<&'data VectorInferenceConfig>,
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
    reference_time_modified_julian_day: f64,
    reconstruction: Option<(&'data ReconstructionFilter, &'data [VectorReconstruction])>,
}

fn write_vector_output(
    path: &Path,
    overwrite: bool,
    data: &VectorOutputData<'_>,
) -> Result<(), AppError> {
    let temporary = temporary_sibling(path)?;
    if let Err(error) = write_vector_output_file(&temporary, data) {
        let _ignored = fs::remove_file(&temporary);
        return Err(error);
    }
    if path.exists() && !overwrite {
        let _ignored = fs::remove_file(&temporary);
        return Err(AppError::DestinationExists(path.to_owned()));
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "the vector NetCDF schema is one visible transaction"
)]
fn write_vector_output_file(path: &Path, data: &VectorOutputData<'_>) -> Result<(), AppError> {
    let input = data.input;
    let mut output = netcdf::create(path)?;
    output.add_dimension("series", input.element_indices.len())?;
    output.add_dimension("constituent", data.constituents.len())?;
    output.add_dimension("presentation_rank", data.constituents.len())?;
    output.add_dimension(
        "spectral_band",
        rutide_core::COLORED_NOISE_FREQUENCY_BANDS_CPH.len(),
    )?;
    if data.reconstruction.is_some() {
        output.add_dimension("time", input.modified_julian_days.len())?;
    }
    output.add_attribute("title", "RUTide depth-averaged current ellipses")?;
    output.add_attribute(
        "rutide_schema_version",
        i64::from(VECTOR_OUTPUT_SCHEMA_VERSION),
    )?;
    output.add_attribute("rutide_version", rutide_core::VERSION)?;
    output.add_attribute(
        "source_time_count",
        i64::try_from(input.source_time_count)
            .map_err(|_| AppError::Invalid("source time count exceeds i64".to_owned()))?,
    )?;
    output.add_attribute(
        "discarded_timestamp_count",
        i64::try_from(input.discarded_timestamp_count)
            .map_err(|_| AppError::Invalid("discarded timestamp count exceeds i64".to_owned()))?,
    )?;
    output.add_attribute("time_epoch", "modified-julian-day")?;
    let profile = vector_profile(
        data.selection,
        data.analysis_method,
        data.inference.is_some(),
        data.fit_options,
        data.phase_reference,
        data.nodal_corrections,
        data.constituent_order,
    );
    output.add_attribute("profile", profile.as_str())?;
    output.add_attribute("analysis_method", data.analysis_method.name())?;
    output.add_attribute("trend_enabled", i64::from(data.fit_options.trend))?;
    output.add_attribute("phase_reference", data.phase_reference.name())?;
    output.add_attribute("nodal_corrections", data.nodal_corrections.name())?;
    output.add_attribute("constituent_order", data.constituent_order.name())?;
    if let Some(names) = data.constituent_order.explicit_names() {
        output.add_attribute("explicit_constituent_order", names.join(","))?;
    }
    if let AnalysisMethod::Robust(options) = data.analysis_method {
        output.add_attribute("robust_tuning_constant", options.tuning_constant)?;
        output.add_attribute("robust_tolerance", options.tolerance)?;
        output.add_attribute(
            "robust_max_iterations",
            i64::try_from(options.max_iterations)
                .map_err(|_| AppError::Invalid("robust iteration limit exceeds i64".to_owned()))?,
        )?;
    }
    output.add_attribute("source_eastward_variable", "ua")?;
    output.add_attribute("source_northward_variable", "va")?;
    output.add_attribute("confidence_interval", data.confidence_interval.method())?;
    if let Some(noise) = data.confidence_interval.noise() {
        output.add_attribute("confidence_noise", noise)?;
    }
    if let Some(options) = data.confidence_interval.monte_carlo_options() {
        output.add_attribute(
            "monte_carlo_realizations",
            u64::try_from(options.realizations).map_err(|_| {
                AppError::Invalid("Monte Carlo realization count exceeds u64".to_owned())
            })?,
        )?;
        output.add_attribute("monte_carlo_seed", options.seed)?;
        output.add_attribute("monte_carlo_rng", "rand_chacha-0.9-ChaCha12Rng")?;
    }
    output.add_attribute("constituent_selection", data.selection.report.method)?;
    if let Some(value) = data.selection.report.rayleigh_min {
        output.add_attribute("rayleigh_min", value)?;
    }
    if let Some(value) = data.selection.report.minimum_separation_cph {
        output.add_attribute("minimum_separation_cph", value)?;
    }
    if let Some(value) = data.selection.report.record_span_days {
        output.add_attribute("record_span_days", value)?;
    }
    write_inference_metadata(
        &mut output,
        data.inference.map(VectorInferenceConfig::report).as_ref(),
    )?;
    output.add_attribute(
        "constituent_names",
        data.constituents
            .iter()
            .map(|constituent| constituent.name.as_str())
            .collect::<Vec<_>>()
            .join(","),
    )?;
    output.add_attribute(
        "reference_time_modified_julian_day",
        data.reference_time_modified_julian_day,
    )?;
    output.add_attribute("result_sha256", data.result_sha256)?;

    let element_indices = input
        .element_indices
        .iter()
        .copied()
        .map(|index| {
            i64::try_from(index)
                .map_err(|_| AppError::Invalid("element index exceeds i64".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    write_variable(
        &mut output.add_variable::<i64>("element_index", &["series"])?,
        &element_indices,
        "1",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("latitude", &["series"])?,
        &input.latitudes,
        "degrees_north",
    )?;
    let observation_counts = input
        .observation_counts
        .iter()
        .copied()
        .map(|count| {
            i64::try_from(count)
                .map_err(|_| AppError::Invalid("observation count exceeds i64".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    write_variable(
        &mut output.add_variable::<i64>("observation_count", &["series"])?,
        &observation_counts,
        "1",
    )?;
    let reference_times = data
        .solutions
        .iter()
        .map(|solution| solution.reference_time_days)
        .collect::<Vec<_>>();
    write_variable(
        &mut output.add_variable::<f64>("reference_time", &["series"])?,
        &reference_times,
        "days since 1858-11-17 00:00:00 UTC",
    )?;
    if data.series_frequency_cph.len() != data.solutions.len()
        || data
            .series_frequency_cph
            .iter()
            .any(|values| values.len() != data.constituents.len())
    {
        return Err(AppError::Invalid(
            "frequency result shape does not match output dimensions".to_owned(),
        ));
    }
    write_variable(
        &mut output.add_variable::<f64>("frequency", &["series", "constituent"])?,
        &data
            .series_frequency_cph
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        "cycles per hour",
    )?;
    write_constituent_order_indices(
        &mut output,
        data.solutions.len(),
        data.constituents.len(),
        data.constituent_index_by_rank,
    )?;
    write_sampling_diagnostics(&mut output, data.solutions.len(), data.sampling_diagnostics)?;
    write_vector_solution_variables(
        &mut output,
        data.constituents.len(),
        data.solutions,
        data.analysis_method,
        data.confidence_interval,
    )?;
    if let Some((filter, values)) = data.reconstruction {
        write_vector_reconstruction_variables(&mut output, input, filter, values)?;
    }
    output.close()?;
    Ok(())
}

fn write_vector_solution_variables(
    output: &mut FileMut,
    constituent_count: usize,
    solutions: &[VectorSolution],
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
) -> Result<(), AppError> {
    let capacity = solutions.len() * constituent_count;
    let mut semi_major = Vec::with_capacity(capacity);
    let mut semi_minor = Vec::with_capacity(capacity);
    let mut inclination = Vec::with_capacity(capacity);
    let mut phase = Vec::with_capacity(capacity);
    let mut percent_energy = Vec::with_capacity(capacity);
    let mut semi_major_ci = Vec::new();
    let mut semi_minor_ci = Vec::new();
    let mut inclination_ci = Vec::new();
    let mut phase_ci = Vec::new();
    let mut signal_to_noise = Vec::new();
    let mut eastward_mean = Vec::with_capacity(solutions.len());
    let mut northward_mean = Vec::with_capacity(solutions.len());
    let mut eastward_slope = Vec::with_capacity(solutions.len());
    let mut northward_slope = Vec::with_capacity(solutions.len());
    for solution in solutions {
        for values in [
            (&mut semi_major, &solution.semi_major),
            (&mut semi_minor, &solution.semi_minor),
            (&mut inclination, &solution.inclination_degrees),
            (&mut phase, &solution.phase_degrees),
            (&mut percent_energy, &solution.percent_energy),
        ] {
            values.0.extend_from_slice(values.1);
        }
        if let Some(values) = vector_confidence_values(solution, confidence_interval)? {
            semi_major_ci.extend_from_slice(values.0);
            semi_minor_ci.extend_from_slice(values.1);
            inclination_ci.extend_from_slice(values.2);
            phase_ci.extend_from_slice(values.3);
            signal_to_noise.extend_from_slice(values.4);
        }
        eastward_mean.push(solution.eastward_mean);
        northward_mean.push(solution.northward_mean);
        eastward_slope.push(solution.eastward_slope_per_day);
        northward_slope.push(solution.northward_slope_per_day);
    }
    for (name, values, units) in [
        ("semi_major", &semi_major, "source velocity units"),
        ("semi_minor", &semi_minor, "source velocity units"),
        ("inclination", &inclination, "degrees"),
        ("phase", &phase, "degrees"),
        ("percent_energy", &percent_energy, "percent"),
    ] {
        write_variable(
            &mut output.add_variable::<f64>(name, &["series", "constituent"])?,
            values,
            units,
        )?;
    }
    if confidence_interval != ConfidenceInterval::None {
        for (name, values, units) in [
            ("semi_major_ci", &semi_major_ci, "source velocity units"),
            ("semi_minor_ci", &semi_minor_ci, "source velocity units"),
            ("inclination_ci", &inclination_ci, "degrees"),
            ("phase_ci", &phase_ci, "degrees"),
            ("signal_to_noise", &signal_to_noise, "1"),
        ] {
            write_variable(
                &mut output.add_variable::<f64>(name, &["series", "constituent"])?,
                values,
                units,
            )?;
        }
    }
    for (name, values, units) in [
        ("eastward_mean", &eastward_mean, "source velocity units"),
        ("northward_mean", &northward_mean, "source velocity units"),
        (
            "eastward_slope",
            &eastward_slope,
            "source velocity units per day",
        ),
        (
            "northward_slope",
            &northward_slope,
            "source velocity units per day",
        ),
    ] {
        write_variable(
            &mut output.add_variable::<f64>(name, &["series"])?,
            values,
            units,
        )?;
    }
    write_vector_robust_variables(output, solutions, analysis_method)?;
    Ok(())
}

fn write_vector_robust_variables(
    output: &mut FileMut,
    solutions: &[VectorSolution],
    analysis_method: AnalysisMethod,
) -> Result<(), AppError> {
    if analysis_method == AnalysisMethod::Ols {
        if solutions.iter().any(|solution| solution.robust.is_some()) {
            return Err(AppError::Invalid(
                "OLS solver returned unexpected robust diagnostics".to_owned(),
            ));
        }
        return Ok(());
    }
    let diagnostics = solutions
        .iter()
        .map(|solution| {
            solution
                .robust
                .as_ref()
                .ok_or_else(|| AppError::Invalid("robust solver omitted diagnostics".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total_observations = diagnostics
        .iter()
        .map(|diagnostics| diagnostics.weights.len())
        .sum();
    output.add_dimension("robust_observation", total_observations)?;
    let mut row_size = Vec::with_capacity(diagnostics.len());
    let mut iterations = Vec::with_capacity(diagnostics.len());
    let mut termination = Vec::with_capacity(diagnostics.len());
    let mut residual_scale = Vec::with_capacity(diagnostics.len());
    let mut ols_rms_residual = Vec::with_capacity(diagnostics.len());
    let mut rms_residual = Vec::with_capacity(diagnostics.len());
    let mut weights = Vec::with_capacity(total_observations);
    let mut leverage = Vec::with_capacity(total_observations);
    for diagnostics in diagnostics {
        if diagnostics.weights.len() != diagnostics.leverage.len() {
            return Err(AppError::Invalid(
                "robust weight and leverage lengths differ".to_owned(),
            ));
        }
        row_size.push(
            i64::try_from(diagnostics.weights.len()).map_err(|_| {
                AppError::Invalid("robust observation count exceeds i64".to_owned())
            })?,
        );
        iterations.push(
            i64::try_from(diagnostics.iterations)
                .map_err(|_| AppError::Invalid("robust iteration count exceeds i64".to_owned()))?,
        );
        termination.push(robust_termination_code(diagnostics.termination));
        residual_scale.push(diagnostics.residual_scale);
        ols_rms_residual.push(diagnostics.ols_rms_residual);
        rms_residual.push(diagnostics.rms_residual);
        weights.extend_from_slice(&diagnostics.weights);
        leverage.extend_from_slice(&diagnostics.leverage);
    }
    for (name, values) in [
        ("robust_weight_row_size", &row_size),
        ("robust_iterations", &iterations),
    ] {
        write_variable(
            &mut output.add_variable::<i64>(name, &["series"])?,
            values,
            "1",
        )?;
    }
    write_robust_schema_metadata(output, &termination)?;
    for (name, values, units) in [
        (
            "robust_residual_scale",
            &residual_scale,
            "source velocity units",
        ),
        (
            "robust_ols_rms_residual",
            &ols_rms_residual,
            "source velocity units",
        ),
        (
            "robust_rms_residual",
            &rms_residual,
            "source velocity units",
        ),
    ] {
        write_variable(
            &mut output.add_variable::<f64>(name, &["series"])?,
            values,
            units,
        )?;
    }
    write_variable(
        &mut output.add_variable::<f64>("robust_weight", &["robust_observation"])?,
        &weights,
        "1",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("robust_leverage", &["robust_observation"])?,
        &leverage,
        "1",
    )?;
    Ok(())
}

fn write_vector_reconstruction_variables(
    output: &mut FileMut,
    input: &VectorInputData,
    filter: &ReconstructionFilter,
    values: &[VectorReconstruction],
) -> Result<(), AppError> {
    if values.len() != input.element_indices.len()
        || values.iter().any(|series| {
            series.eastward.len() != input.modified_julian_days.len()
                || series.northward.len() != input.modified_julian_days.len()
        })
    {
        return Err(AppError::Invalid(
            "vector reconstruction shape does not match output dimensions".to_owned(),
        ));
    }
    let report = reconstruction_report(filter, input.modified_julian_days.len());
    output.add_attribute("reconstruction_filter", report.filter)?;
    if let Some(constituents) = report.constituents {
        output.add_attribute("reconstruction_constituents", constituents.join(","))?;
    }
    if let Some(value) = report.minimum_percent_energy {
        output.add_attribute("reconstruction_minimum_percent_energy", value)?;
    }
    if let Some(value) = report.minimum_signal_to_noise {
        output.add_attribute("reconstruction_minimum_signal_to_noise", value)?;
    }
    write_variable(
        &mut output.add_variable::<f64>("time", &["time"])?,
        &input.modified_julian_days,
        "days since 1858-11-17 00:00:00 UTC",
    )?;
    for (name, eastward) in [
        ("eastward_reconstruction", true),
        ("northward_reconstruction", false),
    ] {
        let mut variable = output.add_variable::<f64>(name, &["time", "series"])?;
        variable.put_attribute("units", "source velocity units")?;
        for (series, reconstruction) in values.iter().enumerate() {
            let component = if eastward {
                &reconstruction.eastward
            } else {
                &reconstruction.northward
            };
            variable.put_values(component, (.., series))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use rutide_core::{
        FitOptions, InferenceMode, LinearConfidence, MonteCarloOptions, NodalCorrections,
        PhaseReference, ReconstructionFilter, RobustOptions, TidalConstituent,
        VectorInferenceRelation,
    };

    use super::{
        AnalysisMethod, ConfidenceInterval, ConstituentOrder, ConstituentSelection, NodeSelection,
        VectorAnalyzeConfig, VectorInferenceConfig, analyze_vector, read_fvcom_vector,
        temporary_sibling,
    };

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::too_many_lines,
        reason = "the f32 NetCDF integration fixture intentionally exercises source precision"
    )]
    fn vector_application_uses_joint_missing_mask_and_writes_ellipses() {
        let input_destination = std::env::temp_dir().join("rutide-vector-input-test.nc");
        let input_path = temporary_sibling(&input_destination).expect("valid input path");
        let output_destination = std::env::temp_dir().join("rutide-vector-output-test.nc");
        let output_path = temporary_sibling(&output_destination).expect("valid output path");
        let inference_output_destination =
            std::env::temp_dir().join("rutide-vector-inference-output-test.nc");
        let inference_output_path =
            temporary_sibling(&inference_output_destination).expect("valid inference output path");
        let time_count = 49_usize;
        let fill = -999.0_f32;
        let time_fill = -999_i32;
        let mut dataset = netcdf::create(&input_path).expect("create vector fixture");
        dataset.add_dimension("time", time_count).expect("add time");
        dataset.add_dimension("nele", 2).expect("add elements");
        {
            let mut variable = dataset
                .add_variable::<i32>("Itime", &["time"])
                .expect("add Itime");
            variable.set_fill_value(time_fill).expect("set time fill");
            let mut integer_days = vec![58_113; time_count];
            integer_days[10] = time_fill;
            variable.put_values(&integer_days, ..).expect("write Itime");
        }
        dataset
            .add_variable::<i32>("Itime2", &["time"])
            .expect("add Itime2")
            .put_values(
                &(0..time_count)
                    .map(|index| i32::try_from(index).expect("small index") * 3_600_000)
                    .collect::<Vec<_>>(),
                ..,
            )
            .expect("write Itime2");
        dataset
            .add_variable::<f32>("latc", &["nele"])
            .expect("add latc")
            .put_values(&[60.0, 61.0], ..)
            .expect("write latc");
        let mut eastward = Vec::with_capacity(time_count * 2);
        let mut northward = Vec::with_capacity(time_count * 2);
        for index in 0..time_count {
            let position = f64::from(u32::try_from(index).expect("small fixture index"));
            eastward.extend([
                (0.2 + (position / 4.0).sin()) as f32,
                (-0.1 + (position / 7.0).cos()) as f32,
            ]);
            northward.extend([
                (0.3 + (position / 5.0).cos()) as f32,
                (0.05 + (position / 9.0).sin()) as f32,
            ]);
        }
        eastward[3 * 2] = fill;
        northward[4 * 2 + 1] = fill;
        {
            let mut variable = dataset
                .add_variable::<f32>("ua", &["time", "nele"])
                .expect("add ua");
            variable.set_fill_value(fill).expect("set ua fill");
            variable.put_values(&eastward, ..).expect("write ua");
        }
        {
            let mut variable = dataset
                .add_variable::<f32>("va", &["time", "nele"])
                .expect("add va");
            variable.set_fill_value(fill).expect("set va fill");
            variable.put_values(&northward, ..).expect("write va");
        }
        dataset.close().expect("close vector fixture");

        let input = read_fvcom_vector(&input_path, &NodeSelection::Indices(vec![1, 0]))
            .expect("read vector fixture");
        assert_eq!(input.element_indices, [1, 0]);
        assert_eq!(input.source_time_count, 49);
        assert_eq!(input.discarded_timestamp_count, 1);
        assert_eq!(input.modified_julian_days.len(), 48);
        assert_eq!(input.observation_counts, [47, 47]);
        assert!(input.eastward[4 * 2].is_nan());
        assert!(input.northward[4 * 2].is_nan());
        assert!(input.eastward[3 * 2 + 1].is_nan());
        assert!(input.northward[3 * 2 + 1].is_nan());

        let report = analyze_vector(&VectorAnalyzeConfig {
            input: input_path.clone(),
            output: output_path.clone(),
            report: None,
            elements: NodeSelection::Indices(vec![1, 0]),
            constituent_selection: ConstituentSelection::Explicit(vec![
                TidalConstituent::M2,
                TidalConstituent::K1,
            ]),
            constituent_order: ConstituentOrder::Frequency,
            inference: None,
            fit_options: FitOptions::default(),
            phase_reference: PhaseReference::Greenwich,
            nodal_corrections: NodalCorrections::Exact,
            confidence_interval: ConfidenceInterval::MonteCarlo {
                options: MonteCarloOptions {
                    realizations: 64,
                    seed: 42,
                },
                noise: LinearConfidence::White,
            },
            analysis_method: AnalysisMethod::Robust(RobustOptions {
                tolerance: 0.01,
                ..RobustOptions::default()
            }),
            reconstruction: Some(ReconstructionFilter::All),
            workers: 2,
            overwrite: false,
        })
        .expect("analyze vector fixture");
        assert_eq!(report.series_count, 2);
        assert_eq!(report.source_time_count, 49);
        assert_eq!(report.discarded_timestamp_count, 1);
        assert_eq!(report.time_count, 48);
        assert_eq!(report.series_with_missing_observations, 2);
        assert_eq!(report.sampling.fft_series_count, 0);
        assert_eq!(report.sampling.lomb_scargle_series_count, 2);
        assert_eq!(report.sampling.minimum_observation_count, 47);
        assert!((report.sampling.maximum_gap_hours - 2.0).abs() < 1e-9);
        assert_eq!(
            report.sample_results[0].sampling.residual_spectrum_method,
            "lomb-scargle"
        );
        assert_eq!(report.analysis_method, "robust");
        assert_eq!(report.constituent_order, "frequency");
        assert!(!report.constituent_order_varies_by_series);
        assert_eq!(report.sample_results[0].constituent_index_by_rank, [1, 0]);
        assert_eq!(report.confidence_interval, "monte-carlo");
        assert_eq!(report.monte_carlo_realizations, Some(64));
        assert_eq!(report.monte_carlo_seed, Some(42));
        let output = netcdf::open(&output_path).expect("open vector output");
        assert_eq!(
            output
                .attribute("monte_carlo_realizations")
                .expect("Monte Carlo realization metadata")
                .value()
                .expect("read realization metadata"),
            netcdf::AttributeValue::Ulonglong(64)
        );
        assert_eq!(
            output
                .attribute("monte_carlo_seed")
                .expect("Monte Carlo seed metadata")
                .value()
                .expect("read seed metadata"),
            netcdf::AttributeValue::Ulonglong(42)
        );
        assert_eq!(
            output
                .variable("constituent_index_by_rank")
                .expect("frequency presentation order")
                .get_values::<i64, _>(..)
                .expect("read frequency presentation order"),
            [1, 0, 1, 0]
        );
        assert_eq!(
            output
                .attribute("discarded_timestamp_count")
                .expect("discarded timestamp metadata")
                .value()
                .expect("read discarded timestamp metadata"),
            netcdf::AttributeValue::Longlong(1)
        );
        assert_eq!(
            output
                .variable("observation_count")
                .expect("observation count")
                .get_values::<i64, _>(..)
                .expect("read observation count"),
            [47, 47]
        );
        assert_eq!(
            output
                .variable("residual_spectrum_method")
                .expect("spectrum method")
                .get_values::<i64, _>(..)
                .expect("read spectrum method"),
            [1, 1]
        );
        assert_eq!(
            output
                .variable("spectral_band_usable_bin_count")
                .expect("spectral band coverage")
                .len(),
            18
        );
        assert_eq!(output.variable("semi_major").expect("semi-major").len(), 4);
        assert_eq!(
            output
                .variable("robust_weight_row_size")
                .expect("robust row sizes")
                .get_values::<i64, _>(..)
                .expect("read robust row sizes"),
            [47, 47]
        );
        assert_eq!(
            output
                .variable("robust_weight")
                .expect("robust weights")
                .len(),
            94
        );
        assert_eq!(
            output
                .variable("robust_termination")
                .expect("robust termination")
                .attribute("flag_meanings")
                .expect("termination flag meanings")
                .value()
                .expect("read termination flag meanings"),
            netcdf::AttributeValue::Str("tolerance objective_increase exact_fit".to_owned())
        );
        assert_eq!(
            output
                .variable("eastward_reconstruction")
                .expect("eastward reconstruction")
                .len(),
            (time_count - 1) * 2
        );
        drop(output);

        let inference_report = analyze_vector(&VectorAnalyzeConfig {
            input: input_path.clone(),
            output: inference_output_path.clone(),
            report: None,
            elements: NodeSelection::Indices(vec![1, 0]),
            constituent_selection: ConstituentSelection::Explicit(vec![
                TidalConstituent::M2,
                TidalConstituent::K1,
            ]),
            constituent_order: ConstituentOrder::SignalToNoise,
            inference: Some(VectorInferenceConfig {
                mode: InferenceMode::Exact,
                relationships: vec![
                    VectorInferenceRelation::new(
                        TidalConstituent::S2,
                        TidalConstituent::M2,
                        0.35,
                        20.0,
                        0.25,
                        -10.0,
                    ),
                    VectorInferenceRelation::new(
                        TidalConstituent::O1,
                        TidalConstituent::K1,
                        0.5,
                        45.0,
                        0.4,
                        30.0,
                    ),
                ],
            }),
            fit_options: FitOptions { trend: false },
            phase_reference: PhaseReference::Raw,
            nodal_corrections: NodalCorrections::LinearTime,
            confidence_interval: ConfidenceInterval::MonteCarlo {
                options: MonteCarloOptions {
                    realizations: 64,
                    seed: 99,
                },
                noise: LinearConfidence::Colored,
            },
            analysis_method: AnalysisMethod::Robust(RobustOptions {
                tolerance: 0.01,
                ..RobustOptions::default()
            }),
            reconstruction: Some(ReconstructionFilter::All),
            workers: 2,
            overwrite: false,
        })
        .expect("analyze inferred vector fixture");
        assert_eq!(inference_report.constituents, ["M2", "K1", "S2", "O1"]);
        assert_eq!(
            inference_report
                .inference
                .as_ref()
                .expect("inference report")
                .mode,
            "exact"
        );
        assert_eq!(inference_report.analysis_method, "robust");
        assert!(!inference_report.trend_enabled);
        assert_eq!(inference_report.phase_reference, "raw");
        assert_eq!(inference_report.nodal_corrections, "linear-time");
        assert_eq!(inference_report.constituent_order, "signal-to-noise");
        assert_eq!(inference_report.confidence_interval, "monte-carlo");
        assert_eq!(inference_report.monte_carlo_realizations, Some(64));
        assert_eq!(inference_report.monte_carlo_seed, Some(99));
        assert_eq!(
            inference_report.profile,
            "fixed-constituents-raw-nodal-linear-time-vector-inference-robust-order-snr-no-trend"
        );
        let inference_output =
            netcdf::open(&inference_output_path).expect("open inferred vector output");
        assert_eq!(
            inference_output
                .attribute("inference_convention")
                .expect("inference convention")
                .value()
                .expect("read inference convention"),
            netcdf::AttributeValue::Str("vector_rotary".to_owned())
        );
        assert_eq!(
            inference_output
                .attribute("inferred_constituent_names")
                .expect("inferred constituent names")
                .value()
                .expect("read inferred constituent names"),
            netcdf::AttributeValue::Str("S2,O1".to_owned())
        );
        assert_eq!(
            inference_output
                .attribute("trend_enabled")
                .expect("trend metadata")
                .value()
                .expect("read trend metadata"),
            netcdf::AttributeValue::Longlong(0)
        );
        assert_eq!(
            inference_output
                .attribute("phase_reference")
                .expect("phase-reference metadata")
                .value()
                .expect("read phase-reference metadata"),
            netcdf::AttributeValue::Str("raw".to_owned())
        );
        assert_eq!(
            inference_output
                .attribute("nodal_corrections")
                .expect("nodal-correction metadata")
                .value()
                .expect("read nodal-correction metadata"),
            netcdf::AttributeValue::Str("linear-time".to_owned())
        );
        assert_eq!(
            inference_output
                .attribute("constituent_order")
                .expect("constituent-order metadata")
                .value()
                .expect("read constituent-order metadata"),
            netcdf::AttributeValue::Str("signal-to-noise".to_owned())
        );
        let constituent_index_by_rank = inference_output
            .variable("constituent_index_by_rank")
            .expect("SNR presentation order")
            .get_values::<i64, _>(..)
            .expect("read SNR presentation order");
        let signal_to_noise = inference_output
            .variable("signal_to_noise")
            .expect("SNR values")
            .get_values::<f64, _>(..)
            .expect("read SNR values");
        for series in 0..2 {
            let order = &constituent_index_by_rank[series * 4..(series + 1) * 4];
            let snr = &signal_to_noise[series * 4..(series + 1) * 4];
            assert!(order.windows(2).all(|pair| {
                snr[usize::try_from(pair[0]).expect("non-negative index")]
                    >= snr[usize::try_from(pair[1]).expect("non-negative index")]
            }));
        }
        assert_eq!(
            inference_output
                .attribute("confidence_interval")
                .expect("confidence method")
                .value()
                .expect("read confidence method"),
            netcdf::AttributeValue::Str("monte-carlo".to_owned())
        );
        assert_eq!(
            inference_output
                .attribute("monte_carlo_seed")
                .expect("Monte Carlo seed")
                .value()
                .expect("read Monte Carlo seed"),
            netcdf::AttributeValue::Ulonglong(99)
        );
        for name in ["eastward_slope", "northward_slope"] {
            assert!(
                inference_output
                    .variable(name)
                    .expect("slope variable")
                    .get_values::<f64, _>(..)
                    .expect("read slopes")
                    .iter()
                    .all(|value| value.to_bits() == 0.0_f64.to_bits())
            );
        }
        assert_eq!(
            inference_output
                .variable("semi_major")
                .expect("inferred semi-major")
                .len(),
            8
        );
        assert_eq!(
            inference_output
                .variable("robust_weight_row_size")
                .expect("inferred robust row sizes")
                .get_values::<i64, _>(..)
                .expect("read inferred robust row sizes"),
            [47, 47]
        );
        assert_eq!(
            inference_output
                .variable("robust_weight")
                .expect("inferred robust weights")
                .len(),
            94
        );
        drop(inference_output);
        fs::remove_file(input_path).expect("remove vector fixture");
        fs::remove_file(output_path).expect("remove vector output");
        fs::remove_file(inference_output_path).expect("remove inferred vector output");
    }
}
