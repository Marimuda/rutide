//! `NetCDF` application layer for `RUTide`.

use std::{
    collections::BTreeSet,
    error::Error,
    fmt,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use netcdf::{FileMut, Variable, VariableMut};
use rayon::ThreadPoolBuilder;
use rutide_core::{
    AnalysisError, Constituent, GreenwichNodalBatch, LinearConfidence, RayleighSelection,
    ReconstructionFilter, ScalarSolution, TidalConstituent, select_constituents_by_rayleigh,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

mod vector;

pub use vector::{VectorAnalyzeConfig, VectorRunReport, VectorSampleResult, analyze_vector};

const MILLISECONDS_PER_DAY: f64 = 86_400_000.0;
const OUTPUT_SCHEMA_VERSION: u32 = 5;
/// Backward-compatible benchmark constituent set used when none is specified.
pub const DEFAULT_CONSTITUENTS: [TidalConstituent; 5] = [
    TidalConstituent::M2,
    TidalConstituent::S2,
    TidalConstituent::N2,
    TidalConstituent::K1,
    TidalConstituent::O1,
];

/// Node subset to read from an FVCOM scalar field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NodeSelection {
    /// Every node in source order.
    All,
    /// A contiguous prefix beginning at node zero.
    Prefix(usize),
    /// Explicit node indices in the requested output order.
    Indices(Vec<usize>),
}

/// Strategy used to choose constituents for one analysis run.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstituentSelection {
    /// Use exactly these catalog constituents in the supplied order.
    Explicit(Vec<TidalConstituent>),
    /// Select all catalog entries resolved by the record span.
    Rayleigh {
        /// Dimensionless minimum Rayleigh criterion.
        minimum: f64,
    },
}

/// Confidence-interval calculation requested for an analysis run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfidenceInterval {
    /// Do not calculate confidence intervals or SNR.
    None,
    /// Calculate linearized 95% intervals using the selected noise model.
    Linear(LinearConfidence),
}

impl ConfidenceInterval {
    const fn method(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear(_) => "linear",
        }
    }

    const fn noise(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Linear(LinearConfidence::White) => Some("white"),
            Self::Linear(LinearConfidence::Colored) => Some("colored"),
        }
    }
}

/// Configuration for one scalar FVCOM analysis run.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalyzeConfig {
    /// Read-only source FVCOM `NetCDF` path.
    pub input: PathBuf,
    /// Destination `NetCDF` coefficient path.
    pub output: PathBuf,
    /// Optional JSON run-report path.
    pub report: Option<PathBuf>,
    /// Spatial subset to analyze.
    pub nodes: NodeSelection,
    /// Explicit or record-length-based constituent selection.
    pub constituent_selection: ConstituentSelection,
    /// Optional linearized confidence intervals and their noise model.
    pub confidence_interval: ConfidenceInterval,
    /// Optional complete-series reconstruction and its constituent filter.
    pub reconstruction: Option<ReconstructionFilter>,
    /// Number of outer spatial worker threads.
    pub workers: usize,
    /// Permit replacing existing output and report files.
    pub overwrite: bool,
}

/// Timings for the separately measured application stages.
#[derive(Clone, Debug, Serialize)]
pub struct StageTimings {
    /// Open, validate, and read selected `NetCDF` variables.
    pub input_seconds: f64,
    /// Prepare shared astronomical and satellite terms.
    pub preparation_seconds: f64,
    /// Construct latitude-specific designs and solve all series.
    pub solve_seconds: f64,
    /// Reconstruct every requested series; zero when reconstruction is disabled.
    pub reconstruction_seconds: f64,
    /// Canonicalize results and compute their SHA-256 identity.
    pub result_processing_seconds: f64,
    /// Create, populate, close, and atomically install the output `NetCDF` file.
    pub output_seconds: f64,
    /// Total through completed `NetCDF` output, excluding optional report writing.
    pub total_seconds: f64,
}

/// A small retained coefficient sample in the JSON run report.
#[derive(Clone, Debug, Serialize)]
pub struct SampleResult {
    /// Original zero-based FVCOM node index.
    pub node_index: usize,
    /// Node latitude in degrees north.
    pub latitude_degrees_north: f64,
    /// Amplitudes in the report's constituent order.
    pub amplitude: Vec<f64>,
    /// Greenwich phases in degrees in the report's constituent order.
    pub phase_degrees: Vec<f64>,
    /// Percent energy in the report's constituent order.
    pub percent_energy: Vec<f64>,
    /// Linearized 95% amplitude CI half-widths, when enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amplitude_ci: Option<Vec<f64>>,
    /// Linearized 95% phase CI half-widths in degrees, when enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_ci_degrees: Option<Vec<f64>>,
    /// Signal-to-noise ratio derived from amplitude CI, when enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_to_noise: Option<Vec<f64>>,
    /// Fitted constant offset.
    pub mean: f64,
    /// Fitted trend per day.
    pub slope_per_day: f64,
    /// Number of finite observations used by this fit.
    pub observation_count: usize,
    /// Epoch at which the fitted mean is defined, as an MJD.
    pub reference_time_modified_julian_day: f64,
}

/// Constituent-selection details retained in a completed run report.
#[derive(Clone, Debug, Serialize)]
pub struct ConstituentSelectionReport {
    /// Selection algorithm: `explicit` or `rayleigh`.
    pub method: &'static str,
    /// Dimensionless Rayleigh criterion for automatic selection.
    pub rayleigh_min: Option<f64>,
    /// Derived acceptance threshold in cycles per hour.
    pub minimum_separation_cph: Option<f64>,
    /// Timestamp span used by automatic selection, in days.
    pub record_span_days: Option<f64>,
}

/// Reconstruction selection retained in a completed run report.
#[derive(Clone, Debug, Serialize)]
pub struct ReconstructionReport {
    /// Selection algorithm: `all`, `constituents`, or `diagnostics`.
    pub filter: &'static str,
    /// Explicit names when constituent filtering is selected.
    pub constituents: Option<Vec<&'static str>>,
    /// Inclusive PE threshold for diagnostic filtering.
    pub minimum_percent_energy: Option<f64>,
    /// Inclusive SNR threshold for diagnostic filtering.
    pub minimum_signal_to_noise: Option<f64>,
    /// Number of reconstructed timestamps per series.
    pub time_count: usize,
}

/// Machine-readable summary of one completed application run.
#[derive(Clone, Debug, Serialize)]
pub struct RunReport {
    /// Report schema version.
    pub schema_version: u32,
    /// Unix timestamp at report construction.
    pub created_unix_seconds: u64,
    /// `RUTide` package version.
    pub rutide_version: &'static str,
    /// Frozen solver profile name.
    pub profile: &'static str,
    /// Source path as supplied to the application.
    pub input_path: String,
    /// Source container size, not bytes physically read.
    pub input_file_bytes: u64,
    /// Logical payload bytes requested from the four input variables.
    pub logical_input_bytes: u64,
    /// Output coefficient path.
    pub output_path: String,
    /// Completed output file size.
    pub output_file_bytes: u64,
    /// Number of timestamps.
    pub time_count: usize,
    /// Number of analyzed nodes.
    pub series_count: usize,
    /// Number of series containing at least one missing observation.
    pub series_with_missing_observations: usize,
    /// Number of outer spatial workers.
    pub workers: usize,
    /// Auditable constituent-selection method and threshold.
    pub constituent_selection: ConstituentSelectionReport,
    /// Confidence method: `none` or `linear`.
    pub confidence_interval: &'static str,
    /// Residual-noise model when confidence intervals are enabled.
    pub confidence_noise: Option<&'static str>,
    /// Complete-series reconstruction configuration, when enabled.
    pub reconstruction: Option<ReconstructionReport>,
    /// Constituent names in coefficient order.
    pub constituents: Vec<String>,
    /// Reference-time frequencies in cycles per hour.
    pub frequency_cph: Vec<f64>,
    /// Whether reference frequencies differ across fitted series.
    pub frequency_varies_by_series: bool,
    /// SHA-256 over canonical node metadata and every numeric result.
    pub result_sha256: String,
    /// Separately measured application stages.
    pub timings: StageTimings,
    /// First three results in output order.
    pub sample_results: Vec<SampleResult>,
}

/// Errors from FVCOM input, analysis, or result serialization.
#[derive(Debug)]
pub enum AppError {
    /// Filesystem operation failed.
    Io(std::io::Error),
    /// `NetCDF` operation failed.
    Netcdf(netcdf::Error),
    /// Harmonic analysis rejected an input.
    Analysis(AnalysisError),
    /// JSON report serialization failed.
    Json(serde_json::Error),
    /// Rayon worker-pool construction failed.
    ThreadPool(rayon::ThreadPoolBuildError),
    /// Source schema or command configuration is invalid.
    Invalid(String),
    /// A destination exists and replacement was not authorized.
    DestinationExists(PathBuf),
}

impl fmt::Display for AppError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "filesystem error: {error}"),
            Self::Netcdf(error) => write!(formatter, "NetCDF error: {error}"),
            Self::Analysis(error) => write!(formatter, "analysis error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::ThreadPool(error) => write!(formatter, "worker-pool error: {error}"),
            Self::Invalid(message) => formatter.write_str(message),
            Self::DestinationExists(path) => write!(
                formatter,
                "destination already exists; pass --overwrite to replace it: {}",
                path.display()
            ),
        }
    }
}

impl Error for AppError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Netcdf(error) => Some(error),
            Self::Analysis(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::ThreadPool(error) => Some(error),
            Self::Invalid(_) | Self::DestinationExists(_) => None,
        }
    }
}

impl From<std::io::Error> for AppError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<netcdf::Error> for AppError {
    fn from(error: netcdf::Error) -> Self {
        Self::Netcdf(error)
    }
}

impl From<AnalysisError> for AppError {
    fn from(error: AnalysisError) -> Self {
        Self::Analysis(error)
    }
}

impl From<serde_json::Error> for AppError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<rayon::ThreadPoolBuildError> for AppError {
    fn from(error: rayon::ThreadPoolBuildError) -> Self {
        Self::ThreadPool(error)
    }
}

struct InputData {
    modified_julian_days: Vec<f64>,
    node_indices: Vec<usize>,
    latitudes: Vec<f64>,
    observations: Vec<f64>,
    observation_counts: Vec<usize>,
    input_file_bytes: u64,
    logical_input_bytes: u64,
}

struct ResolvedConstituentSelection {
    constituents: Vec<TidalConstituent>,
    report: ConstituentSelectionReport,
}

impl ResolvedConstituentSelection {
    fn profile(&self) -> &'static str {
        match self.report.method {
            "explicit" => "fixed-constituents-greenwich-nodal-ols",
            "rayleigh" => "rayleigh-auto-greenwich-nodal-ols",
            _ => unreachable!("selection methods are constructed internally"),
        }
    }
}

/// Analyze an FVCOM `zeta(time, node)` field and write every coefficient.
///
/// # Errors
///
/// Returns [`AppError`] when configuration, source schema, observations,
/// numerical analysis, or output serialization fails.
#[allow(
    clippy::too_many_lines,
    reason = "the top-level orchestration keeps all separately timed application stages visible"
)]
pub fn analyze_scalar(config: &AnalyzeConfig) -> Result<RunReport, AppError> {
    validate_config(config)?;
    faer::set_global_parallelism(faer::Par::Seq);
    let total_start = Instant::now();

    let input_start = Instant::now();
    let input = read_fvcom_scalar(&config.input, &config.nodes)?;
    let input_seconds = input_start.elapsed().as_secs_f64();

    let preparation_start = Instant::now();
    let selection =
        resolve_constituent_selection(&config.constituent_selection, &input.modified_julian_days)?;
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(
        &input.modified_julian_days,
        &selection.constituents,
    )?;
    if let Some(filter) = &config.reconstruction {
        validate_reconstruction_filter(filter, &selection.constituents)?;
    }
    let preparation_seconds = preparation_start.elapsed().as_secs_f64();

    let worker_pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()?;
    let solve_start = Instant::now();
    let solutions = solve_input(&worker_pool, &batch, &input, config.confidence_interval)?;
    let solve_seconds = solve_start.elapsed().as_secs_f64();

    let reconstruction_start = Instant::now();
    let reconstruction = reconstruct_input(
        &worker_pool,
        &batch,
        &input,
        &solutions,
        config.reconstruction.as_ref(),
    )?;
    let reconstruction_seconds = reconstruction_start.elapsed().as_secs_f64();

    let result_start = Instant::now();
    let series_frequency_cph = solution_frequencies(&batch, &solutions)?;
    let result_sha256 = result_digest(
        &input.node_indices,
        &input.latitudes,
        batch.constituents(),
        &series_frequency_cph,
        &solutions,
        config.confidence_interval,
        config
            .reconstruction
            .as_ref()
            .zip(reconstruction.as_deref()),
    )?;
    let sample_results = retained_samples(
        &input.node_indices,
        &input.latitudes,
        &input.observation_counts,
        &solutions,
    );
    let result_processing_seconds = result_start.elapsed().as_secs_f64();

    let output_start = Instant::now();
    write_output(
        &config.output,
        config.overwrite,
        &OutputData {
            node_indices: &input.node_indices,
            latitudes: &input.latitudes,
            observation_counts: &input.observation_counts,
            constituents: batch.constituents(),
            series_frequency_cph: &series_frequency_cph,
            solutions: &solutions,
            result_sha256: &result_sha256,
            selection: &selection,
            confidence_interval: config.confidence_interval,
            modified_julian_days: &input.modified_julian_days,
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

    let report = RunReport {
        schema_version: OUTPUT_SCHEMA_VERSION,
        created_unix_seconds,
        rutide_version: rutide_core::VERSION,
        profile: selection.profile(),
        input_path: config.input.to_string_lossy().into_owned(),
        input_file_bytes: input.input_file_bytes,
        logical_input_bytes: input.logical_input_bytes,
        output_path: config.output.to_string_lossy().into_owned(),
        output_file_bytes,
        time_count: input.modified_julian_days.len(),
        series_count: input.node_indices.len(),
        series_with_missing_observations: input
            .observation_counts
            .iter()
            .filter(|count| **count != input.modified_julian_days.len())
            .count(),
        workers: config.workers,
        constituent_selection: selection.report,
        confidence_interval: config.confidence_interval.method(),
        confidence_noise: config.confidence_interval.noise(),
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

fn solve_input(
    worker_pool: &rayon::ThreadPool,
    batch: &GreenwichNodalBatch,
    input: &InputData,
    confidence_interval: ConfidenceInterval,
) -> Result<Vec<ScalarSolution>, AnalysisError> {
    worker_pool.install(|| match confidence_interval {
        ConfidenceInterval::None => {
            batch.solve_time_major_with_missing(&input.observations, &input.latitudes)
        }
        ConfidenceInterval::Linear(noise) => batch
            .solve_time_major_with_missing_and_linear_confidence(
                &input.observations,
                &input.latitudes,
                noise,
            ),
    })
}

fn solution_frequencies(
    batch: &GreenwichNodalBatch,
    solutions: &[ScalarSolution],
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

fn reconstruct_input(
    worker_pool: &rayon::ThreadPool,
    batch: &GreenwichNodalBatch,
    input: &InputData,
    solutions: &[ScalarSolution],
    filter: Option<&ReconstructionFilter>,
) -> Result<Option<Vec<Vec<f64>>>, AnalysisError> {
    let Some(filter) = filter else {
        return Ok(None);
    };
    let reconstructor = batch.reconstructor_modified_julian_days(&input.modified_julian_days)?;
    worker_pool
        .install(|| {
            reconstructor.reconstruct_many_series_major(solutions, &input.latitudes, filter)
        })
        .map(Some)
}

fn validate_config(config: &AnalyzeConfig) -> Result<(), AppError> {
    if config.workers == 0 {
        return Err(AppError::Invalid(
            "worker count must be greater than zero".to_owned(),
        ));
    }
    if config.input == config.output {
        return Err(AppError::Invalid(
            "input and output paths must differ".to_owned(),
        ));
    }
    if let ConstituentSelection::Explicit(constituents) = &config.constituent_selection {
        if constituents.is_empty() {
            return Err(AppError::Invalid(
                "constituent list must not be empty".to_owned(),
            ));
        }
        let mut unique_constituents = BTreeSet::new();
        for constituent in constituents.iter().copied() {
            if !unique_constituents.insert(constituent) {
                return Err(AppError::Invalid(format!(
                    "constituent {constituent} appears more than once"
                )));
            }
        }
    }
    if let Some(filter) = &config.reconstruction {
        validate_reconstruction_filter_thresholds(filter)?;
    }
    if config.output.exists() && !config.overwrite {
        return Err(AppError::DestinationExists(config.output.clone()));
    }
    if let Some(report) = &config.report {
        if report == &config.input || report == &config.output {
            return Err(AppError::Invalid(
                "report path must differ from input and output paths".to_owned(),
            ));
        }
        if report.exists() && !config.overwrite {
            return Err(AppError::DestinationExists(report.clone()));
        }
    }
    Ok(())
}

fn validate_reconstruction_filter_thresholds(
    filter: &ReconstructionFilter,
) -> Result<(), AppError> {
    if let ReconstructionFilter::Diagnostics {
        minimum_percent_energy,
        minimum_signal_to_noise,
    } = filter
    {
        for (name, threshold) in [
            ("percent-energy", Some(*minimum_percent_energy)),
            ("signal-to-noise", *minimum_signal_to_noise),
        ] {
            if threshold.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(AppError::Invalid(format!(
                    "{name} reconstruction threshold must be finite and non-negative"
                )));
            }
        }
    }
    Ok(())
}

fn validate_reconstruction_filter(
    filter: &ReconstructionFilter,
    prepared: &[TidalConstituent],
) -> Result<(), AppError> {
    validate_reconstruction_filter_thresholds(filter)?;
    if let ReconstructionFilter::Constituents(requested) = filter {
        if requested.is_empty() {
            return Err(AppError::Invalid(
                "reconstruction constituent list must not be empty".to_owned(),
            ));
        }
        let mut unique = BTreeSet::new();
        for constituent in requested.iter().copied() {
            if !unique.insert(constituent) {
                return Err(AppError::Invalid(format!(
                    "reconstruction constituent {constituent} appears more than once"
                )));
            }
            if !prepared.contains(&constituent) {
                return Err(AppError::Invalid(format!(
                    "reconstruction constituent {constituent} was not selected for analysis"
                )));
            }
        }
    }
    Ok(())
}

fn reconstruction_report(filter: &ReconstructionFilter, time_count: usize) -> ReconstructionReport {
    match filter {
        ReconstructionFilter::All => ReconstructionReport {
            filter: "all",
            constituents: None,
            minimum_percent_energy: None,
            minimum_signal_to_noise: None,
            time_count,
        },
        ReconstructionFilter::Constituents(constituents) => ReconstructionReport {
            filter: "constituents",
            constituents: Some(
                constituents
                    .iter()
                    .copied()
                    .map(TidalConstituent::name)
                    .collect(),
            ),
            minimum_percent_energy: None,
            minimum_signal_to_noise: None,
            time_count,
        },
        ReconstructionFilter::Diagnostics {
            minimum_percent_energy,
            minimum_signal_to_noise,
        } => ReconstructionReport {
            filter: "diagnostics",
            constituents: None,
            minimum_percent_energy: Some(*minimum_percent_energy),
            minimum_signal_to_noise: *minimum_signal_to_noise,
            time_count,
        },
    }
}

fn resolve_constituent_selection(
    selection: &ConstituentSelection,
    modified_julian_days: &[f64],
) -> Result<ResolvedConstituentSelection, AnalysisError> {
    match selection {
        ConstituentSelection::Explicit(constituents) => Ok(ResolvedConstituentSelection {
            constituents: constituents.clone(),
            report: ConstituentSelectionReport {
                method: "explicit",
                rayleigh_min: None,
                minimum_separation_cph: None,
                record_span_days: None,
            },
        }),
        ConstituentSelection::Rayleigh { minimum } => {
            let RayleighSelection {
                rayleigh_min,
                record_span_days,
                minimum_separation_cph,
                constituents,
            } = select_constituents_by_rayleigh(modified_julian_days, *minimum)?;
            Ok(ResolvedConstituentSelection {
                constituents,
                report: ConstituentSelectionReport {
                    method: "rayleigh",
                    rayleigh_min: Some(rayleigh_min),
                    minimum_separation_cph: Some(minimum_separation_cph),
                    record_span_days: Some(record_span_days),
                },
            })
        }
    }
}

fn read_fvcom_scalar(path: &Path, selection: &NodeSelection) -> Result<InputData, AppError> {
    let input_file_bytes = fs::metadata(path)?.len();
    let dataset = netcdf::open(path)?;
    let time_count = required_dimension_length(&dataset, "time")?;
    let node_count = required_dimension_length(&dataset, "node")?;
    let node_indices = resolve_node_selection(selection, node_count)?;
    let series_count = node_indices.len();

    let integer_day_variable = required_variable(&dataset, "Itime")?;
    validate_dimensions(&integer_day_variable, &[("time", time_count)])?;
    let integer_days = integer_day_variable.get_values::<i32, _>(..)?;

    let millisecond_variable = required_variable(&dataset, "Itime2")?;
    validate_dimensions(&millisecond_variable, &[("time", time_count)])?;
    let integer_milliseconds = millisecond_variable.get_values::<i32, _>(..)?;
    let modified_julian_days = integer_days
        .into_iter()
        .zip(integer_milliseconds)
        .map(|(day, milliseconds)| f64::from(day) + f64::from(milliseconds) / MILLISECONDS_PER_DAY)
        .collect::<Vec<_>>();

    let latitude_variable = required_variable(&dataset, "lat")?;
    validate_dimensions(&latitude_variable, &[("node", node_count)])?;
    let latitude_fill = latitude_variable.fill_value::<f32>()?;

    let zeta_variable = required_variable(&dataset, "zeta")?;
    validate_dimensions(
        &zeta_variable,
        &[("time", time_count), ("node", node_count)],
    )?;
    let zeta_fill = zeta_variable.fill_value::<f32>()?;

    let is_prefix = node_indices.iter().copied().eq(0..node_indices.len());
    let (latitude_values, mut observation_values) = if is_prefix {
        (
            latitude_variable.get_values::<f64, _>(0..series_count)?,
            zeta_variable.get_values::<f64, _>((.., 0..series_count))?,
        )
    } else {
        let mut latitude_values = Vec::with_capacity(series_count);
        let mut observation_values = vec![0.0_f64; time_count * series_count];
        for (series, node) in node_indices.iter().copied().enumerate() {
            latitude_values.push(latitude_variable.get_value::<f64, _>(node)?);
            let column = zeta_variable.get_values::<f64, _>((.., node))?;
            for (time, value) in column.into_iter().enumerate() {
                observation_values[time * series_count + series] = value;
            }
        }
        (latitude_values, observation_values)
    };

    for (series, value) in latitude_values.iter().copied().enumerate() {
        validate_source_value("lat", value, latitude_fill, series, 0)?;
    }
    for (index, value) in observation_values.iter_mut().enumerate() {
        *value = normalize_source_observation(
            "zeta",
            *value,
            zeta_fill,
            index % series_count,
            index / series_count,
        )?;
    }
    let observation_counts = (0..series_count)
        .map(|series| {
            observation_values
                .iter()
                .skip(series)
                .step_by(series_count)
                .filter(|value| value.is_finite())
                .count()
        })
        .collect();

    let observation_count = time_count
        .checked_mul(series_count)
        .ok_or_else(|| AppError::Invalid("logical input size exceeds usize".to_owned()))?;
    let logical_input_bytes = [
        (time_count, integer_day_variable.vartype().size()),
        (time_count, millisecond_variable.vartype().size()),
        (series_count, latitude_variable.vartype().size()),
        (observation_count, zeta_variable.vartype().size()),
    ]
    .into_iter()
    .try_fold(0_u64, |total, (count, element_bytes)| {
        let count = u64::try_from(count)
            .map_err(|_| AppError::Invalid("logical input size exceeds u64".to_owned()))?;
        let element_bytes = u64::try_from(element_bytes)
            .map_err(|_| AppError::Invalid("source element size exceeds u64".to_owned()))?;
        total
            .checked_add(count.checked_mul(element_bytes).ok_or_else(|| {
                AppError::Invalid("logical input byte count overflows u64".to_owned())
            })?)
            .ok_or_else(|| AppError::Invalid("logical input byte count overflows u64".to_owned()))
    })?;

    Ok(InputData {
        modified_julian_days,
        node_indices,
        latitudes: latitude_values,
        observations: observation_values,
        observation_counts,
        input_file_bytes,
        logical_input_bytes,
    })
}

fn required_dimension_length(dataset: &netcdf::File, name: &str) -> Result<usize, AppError> {
    dataset
        .dimension_len(name)
        .ok_or_else(|| AppError::Invalid(format!("source NetCDF is missing dimension {name:?}")))
}

fn required_variable<'dataset>(
    dataset: &'dataset netcdf::File,
    name: &str,
) -> Result<Variable<'dataset>, AppError> {
    dataset
        .variable(name)
        .ok_or_else(|| AppError::Invalid(format!("source NetCDF is missing variable {name:?}")))
}

fn validate_dimensions(
    variable: &Variable<'_>,
    expected: &[(&str, usize)],
) -> Result<(), AppError> {
    let actual = variable
        .dimensions()
        .iter()
        .map(|dimension| (dimension.name(), dimension.len()))
        .collect::<Vec<_>>();
    let matches = actual.len() == expected.len()
        && actual.iter().zip(expected).all(
            |((actual_name, actual_len), (expected_name, expected_len))| {
                actual_name == expected_name && actual_len == expected_len
            },
        );
    if !matches {
        return Err(AppError::Invalid(format!(
            "variable {:?} has dimensions {actual:?}; expected {expected:?}",
            variable.name()
        )));
    }
    Ok(())
}

fn resolve_node_selection(
    selection: &NodeSelection,
    node_count: usize,
) -> Result<Vec<usize>, AppError> {
    if node_count == 0 {
        return Err(AppError::Invalid(
            "source node dimension must not be empty".to_owned(),
        ));
    }
    match selection {
        NodeSelection::All => Ok((0..node_count).collect()),
        NodeSelection::Prefix(count) => {
            if *count == 0 || *count > node_count {
                return Err(AppError::Invalid(format!(
                    "node prefix must be between 1 and {node_count}, received {count}"
                )));
            }
            Ok((0..*count).collect())
        }
        NodeSelection::Indices(indices) => {
            if indices.is_empty() {
                return Err(AppError::Invalid(
                    "explicit node selection must not be empty".to_owned(),
                ));
            }
            let mut unique = BTreeSet::new();
            for index in indices.iter().copied() {
                if index >= node_count {
                    return Err(AppError::Invalid(format!(
                        "node index {index} is outside source node count {node_count}"
                    )));
                }
                if !unique.insert(index) {
                    return Err(AppError::Invalid(format!(
                        "node index {index} appears more than once"
                    )));
                }
            }
            Ok(indices.clone())
        }
    }
}

fn validate_source_value(
    variable: &str,
    value: f64,
    fill_value: Option<f32>,
    series: usize,
    time: usize,
) -> Result<(), AppError> {
    let is_fill = fill_value.is_some_and(|fill| value.to_bits() == f64::from(fill).to_bits());
    if !value.is_finite() || is_fill {
        return Err(AppError::Invalid(format!(
            "{variable} contains an unsupported missing value at series {series}, time {time}"
        )));
    }
    Ok(())
}

fn normalize_source_observation(
    variable: &str,
    value: f64,
    fill_value: Option<f32>,
    series: usize,
    time: usize,
) -> Result<f64, AppError> {
    let is_fill = fill_value.is_some_and(|fill| value.to_bits() == f64::from(fill).to_bits());
    if value.is_nan() || is_fill {
        return Ok(f64::NAN);
    }
    if value.is_infinite() {
        return Err(AppError::Invalid(format!(
            "{variable} contains an infinite value at series {series}, time {time}"
        )));
    }
    Ok(value)
}

fn result_digest(
    node_indices: &[usize],
    latitudes: &[f64],
    constituents: &[Constituent],
    series_frequency_cph: &[Vec<f64>],
    solutions: &[ScalarSolution],
    confidence_interval: ConfidenceInterval,
    reconstruction: Option<(&ReconstructionFilter, &[Vec<f64>])>,
) -> Result<String, AppError> {
    let mut digest = Sha256::new();
    digest.update(b"rutide-scalar-greenwich-nodal-v5\0");
    digest.update(confidence_interval.method().as_bytes());
    digest.update([0]);
    if let Some(noise) = confidence_interval.noise() {
        digest.update(noise.as_bytes());
    }
    digest.update([0]);
    for constituent in constituents {
        digest.update(constituent.name.as_bytes());
        digest.update([0]);
    }
    for (((node_index, latitude), frequency_cph), solution) in node_indices
        .iter()
        .copied()
        .zip(latitudes.iter().copied())
        .zip(series_frequency_cph)
        .zip(solutions)
    {
        let node_index = u64::try_from(node_index)
            .map_err(|_| AppError::Invalid("node index exceeds u64".to_owned()))?;
        digest.update(node_index.to_le_bytes());
        digest.update(latitude.to_bits().to_le_bytes());
        digest.update(solution.reference_time_days.to_bits().to_le_bytes());
        for frequency in frequency_cph {
            digest.update(frequency.to_bits().to_le_bytes());
        }
        for value in &solution.amplitude {
            digest.update(value.to_bits().to_le_bytes());
        }
        for value in &solution.phase_degrees {
            digest.update(value.to_bits().to_le_bytes());
        }
        for value in &solution.percent_energy {
            digest.update(value.to_bits().to_le_bytes());
        }
        if let Some((amplitude_ci, phase_ci_degrees, signal_to_noise)) =
            confidence_values(solution, confidence_interval)?
        {
            for values in [amplitude_ci, phase_ci_degrees, signal_to_noise] {
                for value in values {
                    digest.update(value.to_bits().to_le_bytes());
                }
            }
        }
        digest.update(solution.mean.to_bits().to_le_bytes());
        digest.update(solution.slope_per_day.to_bits().to_le_bytes());
    }
    match reconstruction {
        Some((filter, values)) => {
            update_reconstruction_filter_digest(&mut digest, filter);
            for series in values {
                for value in series {
                    digest.update(value.to_bits().to_le_bytes());
                }
            }
        }
        None => digest.update(b"no-reconstruction\0"),
    }
    Ok(encode_hex(&digest.finalize()))
}

fn update_reconstruction_filter_digest(digest: &mut Sha256, filter: &ReconstructionFilter) {
    match filter {
        ReconstructionFilter::All => digest.update(b"reconstruction:all\0"),
        ReconstructionFilter::Constituents(constituents) => {
            digest.update(b"reconstruction:constituents\0");
            for constituent in constituents {
                digest.update(constituent.name().as_bytes());
                digest.update([0]);
            }
        }
        ReconstructionFilter::Diagnostics {
            minimum_percent_energy,
            minimum_signal_to_noise,
        } => {
            digest.update(b"reconstruction:diagnostics\0");
            digest.update(minimum_percent_energy.to_bits().to_le_bytes());
            match minimum_signal_to_noise {
                Some(value) => digest.update(value.to_bits().to_le_bytes()),
                None => digest.update(b"no-snr\0"),
            }
        }
    }
}

type ConfidenceValues<'solution> = (&'solution [f64], &'solution [f64], &'solution [f64]);

fn confidence_values(
    solution: &ScalarSolution,
    requested: ConfidenceInterval,
) -> Result<Option<ConfidenceValues<'_>>, AppError> {
    match (
        requested,
        solution.amplitude_ci.as_deref(),
        solution.phase_ci_degrees.as_deref(),
        solution.signal_to_noise.as_deref(),
    ) {
        (ConfidenceInterval::None, None, None, None) => Ok(None),
        (ConfidenceInterval::Linear(_), Some(amplitude), Some(phase), Some(snr)) => {
            Ok(Some((amplitude, phase, snr)))
        }
        _ => Err(AppError::Invalid(
            "solver returned inconsistent confidence-interval fields".to_owned(),
        )),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn retained_samples(
    node_indices: &[usize],
    latitudes: &[f64],
    observation_counts: &[usize],
    solutions: &[ScalarSolution],
) -> Vec<SampleResult> {
    node_indices
        .iter()
        .copied()
        .zip(latitudes.iter().copied())
        .zip(observation_counts.iter().copied())
        .zip(solutions)
        .take(3)
        .map(
            |(((node_index, latitude_degrees_north), observation_count), solution)| SampleResult {
                node_index,
                latitude_degrees_north,
                amplitude: solution.amplitude.clone(),
                phase_degrees: solution.phase_degrees.clone(),
                percent_energy: solution.percent_energy.clone(),
                amplitude_ci: solution.amplitude_ci.clone(),
                phase_ci_degrees: solution.phase_ci_degrees.clone(),
                signal_to_noise: solution.signal_to_noise.clone(),
                mean: solution.mean,
                slope_per_day: solution.slope_per_day,
                observation_count,
                reference_time_modified_julian_day: solution.reference_time_days,
            },
        )
        .collect()
}

struct OutputData<'data> {
    node_indices: &'data [usize],
    latitudes: &'data [f64],
    observation_counts: &'data [usize],
    constituents: &'data [Constituent],
    series_frequency_cph: &'data [Vec<f64>],
    solutions: &'data [ScalarSolution],
    result_sha256: &'data str,
    selection: &'data ResolvedConstituentSelection,
    confidence_interval: ConfidenceInterval,
    modified_julian_days: &'data [f64],
    reference_time_modified_julian_day: f64,
    reconstruction: Option<(&'data ReconstructionFilter, &'data [Vec<f64>])>,
}

fn write_output(path: &Path, overwrite: bool, data: &OutputData<'_>) -> Result<(), AppError> {
    let temporary = temporary_sibling(path)?;
    let write_result = write_output_file(&temporary, data);
    if let Err(error) = write_result {
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
    reason = "the NetCDF schema is written in one visible, auditable transaction"
)]
fn write_output_file(path: &Path, data: &OutputData<'_>) -> Result<(), AppError> {
    let OutputData {
        node_indices,
        latitudes,
        observation_counts,
        constituents,
        series_frequency_cph,
        solutions,
        result_sha256,
        selection,
        confidence_interval,
        modified_julian_days,
        reference_time_modified_julian_day,
        reconstruction,
    } = *data;
    let mut output = netcdf::create(path)?;
    output.add_dimension("series", node_indices.len())?;
    output.add_dimension("constituent", constituents.len())?;
    if reconstruction.is_some() {
        output.add_dimension("time", modified_julian_days.len())?;
    }
    output.add_attribute("title", "RUTide scalar harmonic coefficients")?;
    output.add_attribute("rutide_schema_version", i64::from(OUTPUT_SCHEMA_VERSION))?;
    output.add_attribute("rutide_version", rutide_core::VERSION)?;
    output.add_attribute("profile", selection.profile())?;
    output.add_attribute("confidence_interval", confidence_interval.method())?;
    if let Some(noise) = confidence_interval.noise() {
        output.add_attribute("confidence_noise", noise)?;
    }
    output.add_attribute("constituent_selection", selection.report.method)?;
    if let Some(rayleigh_min) = selection.report.rayleigh_min {
        output.add_attribute("rayleigh_min", rayleigh_min)?;
    }
    if let Some(minimum_separation_cph) = selection.report.minimum_separation_cph {
        output.add_attribute("minimum_separation_cph", minimum_separation_cph)?;
    }
    if let Some(record_span_days) = selection.report.record_span_days {
        output.add_attribute("record_span_days", record_span_days)?;
    }
    let constituent_names = constituents
        .iter()
        .map(|constituent| constituent.name.as_str())
        .collect::<Vec<_>>()
        .join(",");
    output.add_attribute("constituent_names", constituent_names)?;
    output.add_attribute(
        "reference_time_modified_julian_day",
        reference_time_modified_julian_day,
    )?;
    output.add_attribute("result_sha256", result_sha256)?;

    let node_indices = node_indices
        .iter()
        .copied()
        .map(|index| {
            i64::try_from(index).map_err(|_| AppError::Invalid("node index exceeds i64".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    write_variable(
        &mut output.add_variable::<i64>("node_index", &["series"])?,
        &node_indices,
        "1",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("latitude", &["series"])?,
        latitudes,
        "degrees_north",
    )?;
    let observation_counts = observation_counts
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
    let reference_times = solutions
        .iter()
        .map(|solution| solution.reference_time_days)
        .collect::<Vec<_>>();
    write_variable(
        &mut output.add_variable::<f64>("reference_time", &["series"])?,
        &reference_times,
        "days since 1858-11-17 00:00:00 UTC",
    )?;
    if series_frequency_cph.len() != solutions.len()
        || series_frequency_cph
            .iter()
            .any(|values| values.len() != constituents.len())
    {
        return Err(AppError::Invalid(
            "frequency result shape does not match output dimensions".to_owned(),
        ));
    }
    let frequency = series_frequency_cph
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    write_variable(
        &mut output.add_variable::<f64>("frequency", &["series", "constituent"])?,
        &frequency,
        "cycles per hour",
    )?;

    write_solution_variables(&mut output, constituents, solutions, confidence_interval)?;
    if let Some((filter, values)) = reconstruction {
        write_reconstruction_variables(
            &mut output,
            modified_julian_days,
            node_indices.len(),
            filter,
            values,
        )?;
    }
    output.close()?;
    Ok(())
}

fn write_reconstruction_variables(
    output: &mut FileMut,
    modified_julian_days: &[f64],
    series_count: usize,
    filter: &ReconstructionFilter,
    values: &[Vec<f64>],
) -> Result<(), AppError> {
    if values.len() != series_count
        || values
            .iter()
            .any(|series| series.len() != modified_julian_days.len())
    {
        return Err(AppError::Invalid(
            "reconstruction result shape does not match output dimensions".to_owned(),
        ));
    }
    let report = reconstruction_report(filter, modified_julian_days.len());
    output.add_attribute("reconstruction_filter", report.filter)?;
    if let Some(constituents) = report.constituents {
        output.add_attribute("reconstruction_constituents", constituents.join(","))?;
    }
    if let Some(minimum) = report.minimum_percent_energy {
        output.add_attribute("reconstruction_minimum_percent_energy", minimum)?;
    }
    if let Some(minimum) = report.minimum_signal_to_noise {
        output.add_attribute("reconstruction_minimum_signal_to_noise", minimum)?;
    }
    write_variable(
        &mut output.add_variable::<f64>("time", &["time"])?,
        modified_julian_days,
        "days since 1858-11-17 00:00:00 UTC",
    )?;
    let mut reconstruction = output.add_variable::<f64>("reconstruction", &["time", "series"])?;
    reconstruction.put_attribute("units", "source variable units")?;
    reconstruction.put_attribute("long_name", "reconstructed scalar tidal signal")?;
    for (series, series_values) in values.iter().enumerate() {
        reconstruction.put_values(series_values, (.., series))?;
    }
    Ok(())
}

fn write_solution_variables(
    output: &mut FileMut,
    constituents: &[Constituent],
    solutions: &[ScalarSolution],
    confidence_interval: ConfidenceInterval,
) -> Result<(), AppError> {
    let mut amplitude = Vec::with_capacity(solutions.len() * constituents.len());
    let mut phase = Vec::with_capacity(solutions.len() * constituents.len());
    let mut percent_energy = Vec::with_capacity(solutions.len() * constituents.len());
    let mut amplitude_ci = Vec::new();
    let mut phase_ci_degrees = Vec::new();
    let mut signal_to_noise = Vec::new();
    let mut mean = Vec::with_capacity(solutions.len());
    let mut slope = Vec::with_capacity(solutions.len());
    for solution in solutions {
        amplitude.extend_from_slice(&solution.amplitude);
        phase.extend_from_slice(&solution.phase_degrees);
        percent_energy.extend_from_slice(&solution.percent_energy);
        if let Some((solution_amplitude_ci, solution_phase_ci, solution_snr)) =
            confidence_values(solution, confidence_interval)?
        {
            amplitude_ci.extend_from_slice(solution_amplitude_ci);
            phase_ci_degrees.extend_from_slice(solution_phase_ci);
            signal_to_noise.extend_from_slice(solution_snr);
        }
        mean.push(solution.mean);
        slope.push(solution.slope_per_day);
    }
    write_variable(
        &mut output.add_variable::<f64>("amplitude", &["series", "constituent"])?,
        &amplitude,
        "source variable units",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("phase", &["series", "constituent"])?,
        &phase,
        "degrees",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("percent_energy", &["series", "constituent"])?,
        &percent_energy,
        "percent",
    )?;
    if confidence_interval != ConfidenceInterval::None {
        write_variable(
            &mut output.add_variable::<f64>("amplitude_ci", &["series", "constituent"])?,
            &amplitude_ci,
            "source variable units",
        )?;
        write_variable(
            &mut output.add_variable::<f64>("phase_ci", &["series", "constituent"])?,
            &phase_ci_degrees,
            "degrees",
        )?;
        write_variable(
            &mut output.add_variable::<f64>("signal_to_noise", &["series", "constituent"])?,
            &signal_to_noise,
            "1",
        )?;
    }
    write_variable(
        &mut output.add_variable::<f64>("mean", &["series"])?,
        &mean,
        "source variable units",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("slope", &["series"])?,
        &slope,
        "source variable units per day",
    )?;
    Ok(())
}

fn write_variable<T>(
    variable: &mut VariableMut<'_>,
    values: &[T],
    units: &str,
) -> Result<(), AppError>
where
    T: netcdf::NcTypeDescriptor + Copy,
{
    variable.put_attribute("units", units)?;
    variable.put_values(values, ..)?;
    Ok(())
}

fn write_json_report<T: Serialize>(
    path: &Path,
    overwrite: bool,
    report: &T,
) -> Result<(), AppError> {
    let temporary = temporary_sibling(path)?;
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    if let Err(error) = serde_json::to_writer_pretty(&mut writer, report) {
        let _ignored = fs::remove_file(&temporary);
        return Err(error.into());
    }
    writer.write_all(b"\n")?;
    writer.flush()?;
    if path.exists() && !overwrite {
        let _ignored = fs::remove_file(&temporary);
        return Err(AppError::DestinationExists(path.to_owned()));
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

fn temporary_sibling(path: &Path) -> Result<PathBuf, AppError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| AppError::Invalid(format!("invalid destination path: {}", path.display())))?
        .to_string_lossy();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(std::io::Error::other)?
        .as_nanos();
    Ok(parent.join(format!(
        ".{file_name}.rutide-{}-{nonce}.tmp",
        std::process::id()
    )))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        NodeSelection, encode_hex, normalize_source_observation, read_fvcom_scalar,
        resolve_node_selection, temporary_sibling, write_reconstruction_variables,
    };
    use rutide_core::{ReconstructionFilter, TidalConstituent};

    #[test]
    fn selection_preserves_explicit_order() {
        assert_eq!(
            resolve_node_selection(&NodeSelection::Indices(vec![4, 1, 3]), 5)
                .expect("valid selection"),
            [4, 1, 3]
        );
    }

    #[test]
    fn selection_rejects_duplicates() {
        assert!(resolve_node_selection(&NodeSelection::Indices(vec![1, 1]), 5).is_err());
    }

    #[test]
    fn selection_rejects_empty_source_dimension() {
        assert!(resolve_node_selection(&NodeSelection::All, 0).is_err());
    }

    #[test]
    fn hex_encoding_is_lowercase_and_zero_padded() {
        assert_eq!(encode_hex(&[0, 15, 16, 255]), "000f10ff");
    }

    #[test]
    fn fvcom_f32_values_are_promoted_directly_and_reordered_exactly() {
        let destination = std::env::temp_dir().join("rutide-read-input-test.nc");
        let path = temporary_sibling(&destination).expect("valid temporary path");
        let mut dataset = netcdf::create(&path).expect("create test NetCDF");
        dataset.add_dimension("time", 2).expect("add time");
        dataset.add_dimension("node", 2).expect("add node");
        dataset
            .add_variable::<i32>("Itime", &["time"])
            .expect("add Itime")
            .put_values(&[58_113, 58_113], ..)
            .expect("write Itime");
        dataset
            .add_variable::<i32>("Itime2", &["time"])
            .expect("add Itime2")
            .put_values(&[0, 3_600_000], ..)
            .expect("write Itime2");
        let latitudes = [60.1_f32, 61.2_f32];
        dataset
            .add_variable::<f32>("lat", &["node"])
            .expect("add lat")
            .put_values(&latitudes, ..)
            .expect("write lat");
        let observations = [0.1_f32, -2.5_f32, 3.25_f32, 4.5_f32];
        dataset
            .add_variable::<f32>("zeta", &["time", "node"])
            .expect("add zeta")
            .put_values(&observations, ..)
            .expect("write zeta");
        dataset.close().expect("close test NetCDF");

        let input = read_fvcom_scalar(&path, &NodeSelection::Indices(vec![1, 0]))
            .expect("read valid FVCOM input");
        assert_eq!(
            input.modified_julian_days,
            [58_113.0, 58_113.0 + 1.0 / 24.0]
        );
        assert_eq!(input.node_indices, [1, 0]);
        assert_eq!(
            input.latitudes,
            [f64::from(latitudes[1]), f64::from(latitudes[0])]
        );
        assert_eq!(
            input.observations,
            [
                f64::from(observations[1]),
                f64::from(observations[0]),
                f64::from(observations[3]),
                f64::from(observations[2]),
            ]
        );
        assert_eq!(input.logical_input_bytes, 40);
        assert_eq!(input.observation_counts, [2, 2]);
        fs::remove_file(path).expect("remove test NetCDF");
    }

    #[test]
    fn source_fill_and_nan_observations_are_normalized_as_missing() {
        let fill = -999.0_f32;
        for value in [f64::NAN, f64::from(fill)] {
            assert!(
                normalize_source_observation("zeta", value, Some(fill), 2, 3)
                    .expect("supported missing value")
                    .is_nan()
            );
        }
        assert!(normalize_source_observation("zeta", f64::INFINITY, Some(fill), 2, 3).is_err());
    }

    #[test]
    fn reconstruction_is_written_time_major_with_filter_metadata() {
        let destination = std::env::temp_dir().join("rutide-reconstruction-output-test.nc");
        let path = temporary_sibling(&destination).expect("valid temporary path");
        let mut dataset = netcdf::create(&path).expect("create test NetCDF");
        dataset.add_dimension("time", 2).expect("add time");
        dataset.add_dimension("series", 2).expect("add series");
        write_reconstruction_variables(
            &mut dataset,
            &[58_113.0, 58_113.5],
            2,
            &ReconstructionFilter::Constituents(vec![TidalConstituent::K1, TidalConstituent::M2]),
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
        )
        .expect("write valid reconstruction");
        dataset.close().expect("close test NetCDF");

        let dataset = netcdf::open(&path).expect("open test NetCDF");
        assert_eq!(
            dataset
                .variable("reconstruction")
                .expect("reconstruction variable")
                .get_values::<f64, _>(..)
                .expect("read reconstruction"),
            [1.0, 3.0, 2.0, 4.0]
        );
        assert_eq!(
            dataset
                .attribute("reconstruction_constituents")
                .expect("constituent metadata")
                .value()
                .expect("read constituent metadata"),
            netcdf::AttributeValue::Str("K1,M2".to_owned())
        );
        drop(dataset);
        fs::remove_file(path).expect("remove test NetCDF");
    }
}
