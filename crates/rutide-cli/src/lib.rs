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
    AnalysisError, Constituent, FitOptions, GreenwichNodalBatch, GreenwichNodalReconstructor,
    InferenceMode, LinearConfidence, MonteCarloOptions, NodalCorrections, PhaseReference,
    RayleighSelection, ReconstructionFilter, RobustOptions, RobustTermination,
    ScalarInferenceBatch, ScalarInferenceRelation, ScalarSolution, SolverOptions, TidalConstituent,
    VectorInferenceRelation, VectorSolution, select_constituents_by_rayleigh,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

mod vector;

pub use vector::{VectorAnalyzeConfig, VectorRunReport, VectorSampleResult, analyze_vector};

const MILLISECONDS_PER_DAY: f64 = 86_400_000.0;
const OUTPUT_SCHEMA_VERSION: u32 = 12;
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

/// Presentation order for fitted constituents.
///
/// Bulk coefficient arrays always retain stable fitted-model order. This option
/// controls a per-series rank-to-model-index mapping carried beside those arrays.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ConstituentOrder {
    /// Preserve the selected fitted-model order.
    #[default]
    Selection,
    /// Rank each series by descending percent energy.
    PercentEnergy,
    /// Rank each series by descending signal-to-noise ratio.
    SignalToNoise,
    /// Rank each series by ascending fitted reference-time frequency.
    Frequency,
    /// Use this complete permutation of the fitted constituent names.
    Explicit(Vec<TidalConstituent>),
}

impl ConstituentOrder {
    /// Stable machine-readable name used in reports and `NetCDF` metadata.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Selection => "selection",
            Self::PercentEnergy => "percent-energy",
            Self::SignalToNoise => "signal-to-noise",
            Self::Frequency => "frequency",
            Self::Explicit(_) => "explicit",
        }
    }

    fn explicit_names(&self) -> Option<Vec<&'static str>> {
        match self {
            Self::Explicit(constituents) => Some(
                constituents
                    .iter()
                    .map(|constituent| constituent.name())
                    .collect(),
            ),
            Self::Selection | Self::PercentEnergy | Self::SignalToNoise | Self::Frequency => None,
        }
    }
}

/// Scalar inferred-constituent relationships for one application run.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarInferenceConfig {
    /// Exact composite basis or Python-compatible approximate inference.
    pub mode: InferenceMode,
    /// Inferred/reference relationships in caller-supplied order.
    pub relationships: Vec<ScalarInferenceRelation>,
}

/// Vector positive/negative rotary inference relationships.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorInferenceConfig {
    /// Exact composite basis or Python-compatible approximate inference.
    pub mode: InferenceMode,
    /// Inferred/reference rotary relationships in caller-supplied order.
    pub relationships: Vec<VectorInferenceRelation>,
}

/// Confidence-interval calculation requested for an analysis run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfidenceInterval {
    /// Do not calculate confidence intervals or SNR.
    None,
    /// Calculate linearized 95% intervals using the selected noise model.
    Linear(LinearConfidence),
    /// Calculate nonlinear intervals from seeded coefficient realizations.
    MonteCarlo {
        /// Reproducible realization count and root seed.
        options: MonteCarloOptions,
        /// White or band-averaged colored residual noise.
        noise: LinearConfidence,
    },
}

/// Least-squares method requested for an application run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnalysisMethod {
    /// Ordinary least squares.
    Ols,
    /// Cauchy iteratively reweighted least squares.
    Robust(RobustOptions),
}

impl AnalysisMethod {
    const fn name(self) -> &'static str {
        match self {
            Self::Ols => "ols",
            Self::Robust(_) => "robust",
        }
    }
}

impl ConfidenceInterval {
    const fn method(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear(_) => "linear",
            Self::MonteCarlo { .. } => "monte-carlo",
        }
    }

    const fn noise(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Linear(LinearConfidence::White)
            | Self::MonteCarlo {
                noise: LinearConfidence::White,
                ..
            } => Some("white"),
            Self::Linear(LinearConfidence::Colored)
            | Self::MonteCarlo {
                noise: LinearConfidence::Colored,
                ..
            } => Some("colored"),
        }
    }

    const fn monte_carlo_options(self) -> Option<MonteCarloOptions> {
        match self {
            Self::MonteCarlo { options, .. } => Some(options),
            Self::None | Self::Linear(_) => None,
        }
    }
}

trait PresentationDiagnostics {
    fn percent_energy(&self) -> &[f64];
    fn signal_to_noise(&self) -> Option<&[f64]>;
}

#[derive(Debug, Eq, PartialEq)]
struct ConstituentOrderMap {
    indices: Vec<u16>,
    series_count: usize,
    constituent_count: usize,
    shared: bool,
}

impl ConstituentOrderMap {
    fn shared(indices: Vec<usize>, series_count: usize) -> Result<Self, AppError> {
        let constituent_count = indices.len();
        Ok(Self {
            indices: encode_constituent_indices(indices)?,
            series_count,
            constituent_count,
            shared: true,
        })
    }

    const fn per_series(indices: Vec<u16>, series_count: usize, constituent_count: usize) -> Self {
        Self {
            indices,
            series_count,
            constituent_count,
            shared: false,
        }
    }

    fn row(&self, series: usize) -> &[u16] {
        assert!(series < self.series_count, "series index is in range");
        if self.shared {
            &self.indices
        } else {
            let start = series * self.constituent_count;
            &self.indices[start..start + self.constituent_count]
        }
    }

    fn varies_by_series(&self) -> bool {
        if self.shared || self.series_count < 2 || self.constituent_count == 0 {
            return false;
        }
        let first = &self.indices[..self.constituent_count];
        self.indices
            .chunks_exact(self.constituent_count)
            .skip(1)
            .any(|indices| indices != first)
    }

    fn is_valid_for(&self, series_count: usize, constituent_count: usize) -> bool {
        let expected_len = if self.shared {
            constituent_count
        } else {
            let Some(expected_len) = series_count.checked_mul(constituent_count) else {
                return false;
            };
            expected_len
        };
        if self.series_count != series_count
            || self.constituent_count != constituent_count
            || self.indices.len() != expected_len
        {
            return false;
        }
        if constituent_count == 0 {
            return true;
        }

        let mut seen = vec![false; constituent_count];
        let rows_to_validate = if self.shared { 1 } else { series_count };
        for indices in self
            .indices
            .chunks(constituent_count)
            .take(rows_to_validate)
        {
            seen.fill(false);
            for &index in indices {
                let index = usize::from(index);
                if index >= constituent_count || std::mem::replace(&mut seen[index], true) {
                    return false;
                }
            }
        }
        true
    }
}

fn encode_constituent_indices(
    indices: impl IntoIterator<Item = usize>,
) -> Result<Vec<u16>, AppError> {
    indices
        .into_iter()
        .map(|index| {
            u16::try_from(index)
                .map_err(|_| AppError::Invalid("constituent index exceeds u16".to_owned()))
        })
        .collect()
}

impl PresentationDiagnostics for ScalarSolution {
    fn percent_energy(&self) -> &[f64] {
        &self.percent_energy
    }

    fn signal_to_noise(&self) -> Option<&[f64]> {
        self.signal_to_noise.as_deref()
    }
}

impl PresentationDiagnostics for VectorSolution {
    fn percent_energy(&self) -> &[f64] {
        &self.percent_energy
    }

    fn signal_to_noise(&self) -> Option<&[f64]> {
        self.signal_to_noise.as_deref()
    }
}

fn shared_constituent_order_indices(
    order: &ConstituentOrder,
    constituents: &[Constituent],
) -> Result<Option<Vec<usize>>, AppError> {
    let constituent_count = constituents.len();
    match order {
        ConstituentOrder::Selection => Ok(Some((0..constituent_count).collect())),
        ConstituentOrder::Explicit(requested) => {
            if requested.len() != constituent_count {
                return Err(AppError::Invalid(format!(
                    "explicit constituent order contains {} names; expected a complete {constituent_count}-name permutation",
                    requested.len()
                )));
            }
            let mut indices = Vec::with_capacity(constituent_count);
            for (rank, requested) in requested.iter().enumerate() {
                let Some(index) = constituents
                    .iter()
                    .position(|constituent| constituent.name == requested.name())
                else {
                    return Err(AppError::Invalid(format!(
                        "explicit constituent order name {} at rank {rank} was not fitted",
                        requested.name()
                    )));
                };
                if indices.contains(&index) {
                    return Err(AppError::Invalid(format!(
                        "explicit constituent order name {} at rank {rank} is duplicated",
                        requested.name()
                    )));
                }
                indices.push(index);
            }
            Ok(Some(indices))
        }
        ConstituentOrder::PercentEnergy
        | ConstituentOrder::SignalToNoise
        | ConstituentOrder::Frequency => Ok(None),
    }
}

fn constituent_order_indices<S: PresentationDiagnostics>(
    order: &ConstituentOrder,
    constituents: &[Constituent],
    series_frequency_cph: &[Vec<f64>],
    solutions: &[S],
) -> Result<ConstituentOrderMap, AppError> {
    if series_frequency_cph.len() != solutions.len() {
        return Err(AppError::Invalid(
            "frequency and solution series counts differ".to_owned(),
        ));
    }
    let constituent_count = constituents.len();
    let shared = shared_constituent_order_indices(order, constituents)?;

    for (series, frequency) in series_frequency_cph.iter().enumerate() {
        if frequency.len() != constituent_count {
            return Err(AppError::Invalid(format!(
                "frequency series {series} contains {} values; expected {constituent_count}",
                frequency.len()
            )));
        }
    }
    if let Some(indices) = shared {
        return ConstituentOrderMap::shared(indices, solutions.len());
    }

    let capacity = solutions
        .len()
        .checked_mul(constituent_count)
        .ok_or_else(|| AppError::Invalid("constituent order size exceeds usize".to_owned()))?;
    let mut ordered = Vec::with_capacity(capacity);
    for (series, (solution, frequency)) in solutions.iter().zip(series_frequency_cph).enumerate() {
        let values = match order {
            ConstituentOrder::PercentEnergy => solution.percent_energy(),
            ConstituentOrder::SignalToNoise => solution.signal_to_noise().ok_or_else(|| {
                AppError::Invalid(
                    "SNR constituent ordering requires confidence intervals".to_owned(),
                )
            })?,
            ConstituentOrder::Frequency => frequency,
            ConstituentOrder::Selection | ConstituentOrder::Explicit(_) => {
                unreachable!("shared presentation orders returned above")
            }
        };
        if values.len() != constituent_count {
            return Err(AppError::Invalid(format!(
                "ordering diagnostic for series {series} contains {} values; expected {constituent_count}",
                values.len()
            )));
        }
        let mut indices = (0..constituent_count).collect::<Vec<_>>();
        match order {
            ConstituentOrder::Frequency => indices.sort_by(|left, right| {
                values[*left]
                    .total_cmp(&values[*right])
                    .then_with(|| left.cmp(right))
            }),
            ConstituentOrder::PercentEnergy | ConstituentOrder::SignalToNoise => {
                indices.sort_by(|left, right| {
                    values[*right]
                        .total_cmp(&values[*left])
                        .then_with(|| left.cmp(right))
                });
            }
            ConstituentOrder::Selection | ConstituentOrder::Explicit(_) => {
                unreachable!("shared presentation orders returned above")
            }
        }
        ordered.extend(encode_constituent_indices(indices)?);
    }
    Ok(ConstituentOrderMap::per_series(
        ordered,
        solutions.len(),
        constituent_count,
    ))
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
    /// Per-series constituent presentation ranking.
    pub constituent_order: ConstituentOrder,
    /// Optional constrained inferred constituents.
    pub inference: Option<ScalarInferenceConfig>,
    /// Mean/trend terms included in the harmonic fit.
    pub fit_options: FitOptions,
    /// Astronomical argument used to reference reported phases.
    pub phase_reference: PhaseReference,
    /// Exact, midpoint-linearized, or disabled nodal/satellite corrections.
    pub nodal_corrections: NodalCorrections,
    /// Optional linearized or Monte Carlo confidence and its noise model.
    pub confidence_interval: ConfidenceInterval,
    /// Ordinary or Cauchy robust least squares.
    pub analysis_method: AnalysisMethod,
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
    /// Phases in degrees using the report's configured reference convention.
    pub phase_degrees: Vec<f64>,
    /// Percent energy in the report's constituent order.
    pub percent_energy: Vec<f64>,
    /// Stable constituent indices in requested presentation-rank order.
    pub constituent_index_by_rank: Vec<usize>,
    /// 95% amplitude CI half-widths, when enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amplitude_ci: Option<Vec<f64>>,
    /// 95% phase CI half-widths in degrees, when enabled.
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

/// Machine-readable inferred-constituent configuration.
#[derive(Clone, Debug, Serialize)]
pub struct InferenceReport {
    /// Basis treatment: `exact` or `approximate`.
    pub mode: &'static str,
    /// Scalar or vector relationship convention.
    pub convention: &'static str,
    /// Relationships in caller-supplied order.
    pub relationships: Vec<InferenceRelationReport>,
}

/// One serialized inferred/reference relationship.
#[derive(Clone, Debug, Serialize)]
pub struct InferenceRelationReport {
    /// Inferred constituent name.
    pub inferred: &'static str,
    /// Reference constituent name.
    pub reference: &'static str,
    /// Scalar or positive-rotary amplitude ratio.
    pub positive_amplitude_ratio: f64,
    /// Scalar or positive-rotary phase offset.
    pub positive_phase_offset_degrees: f64,
    /// Negative-rotary amplitude ratio for vectors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_amplitude_ratio: Option<f64>,
    /// Negative-rotary phase offset for vectors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_phase_offset_degrees: Option<f64>,
}

impl ScalarInferenceConfig {
    fn report(&self) -> InferenceReport {
        InferenceReport {
            mode: inference_mode_name(self.mode),
            convention: "scalar",
            relationships: self
                .relationships
                .iter()
                .map(|relationship| InferenceRelationReport {
                    inferred: relationship.inferred.name(),
                    reference: relationship.reference.name(),
                    positive_amplitude_ratio: relationship.amplitude_ratio,
                    positive_phase_offset_degrees: relationship.phase_offset_degrees,
                    negative_amplitude_ratio: None,
                    negative_phase_offset_degrees: None,
                })
                .collect(),
        }
    }
}

impl VectorInferenceConfig {
    pub(crate) fn report(&self) -> InferenceReport {
        InferenceReport {
            mode: inference_mode_name(self.mode),
            convention: "vector_rotary",
            relationships: self
                .relationships
                .iter()
                .map(|relationship| InferenceRelationReport {
                    inferred: relationship.inferred.name(),
                    reference: relationship.reference.name(),
                    positive_amplitude_ratio: relationship.positive_amplitude_ratio,
                    positive_phase_offset_degrees: relationship.positive_phase_offset_degrees,
                    negative_amplitude_ratio: Some(relationship.negative_amplitude_ratio),
                    negative_phase_offset_degrees: Some(relationship.negative_phase_offset_degrees),
                })
                .collect(),
        }
    }
}

const fn inference_mode_name(mode: InferenceMode) -> &'static str {
    match mode {
        InferenceMode::Exact => "exact",
        InferenceMode::Approximate => "approximate",
    }
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
    pub profile: String,
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
    /// SHA-256 over canonical node metadata and every numeric result.
    pub result_sha256: String,
    /// Separately measured application stages.
    pub timings: StageTimings,
    /// First three results in output order.
    pub sample_results: Vec<SampleResult>,
}

/// Serializable robust fitting options retained in application reports.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RobustOptionsReport {
    /// Cauchy tuning constant.
    pub tuning_constant: f64,
    /// Fractional objective-improvement tolerance.
    pub tolerance: f64,
    /// Maximum IRLS iterations.
    pub max_iterations: usize,
}

impl From<RobustOptions> for RobustOptionsReport {
    fn from(options: RobustOptions) -> Self {
        Self {
            tuning_constant: options.tuning_constant,
            tolerance: options.tolerance,
            max_iterations: options.max_iterations,
        }
    }
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

enum ScalarAnalysisBatch {
    Standard(GreenwichNodalBatch),
    Inferred(ScalarInferenceBatch),
}

impl ScalarAnalysisBatch {
    fn prepare(
        modified_julian_days: &[f64],
        constituents: &[TidalConstituent],
        inference: Option<&ScalarInferenceConfig>,
        fit_options: FitOptions,
        phase_reference: PhaseReference,
        nodal_corrections: NodalCorrections,
    ) -> Result<Self, AnalysisError> {
        let solver_options = SolverOptions::new(fit_options, phase_reference)
            .with_nodal_corrections(nodal_corrections);
        match inference {
            Some(inference) => {
                ScalarInferenceBatch::prepare_modified_julian_days_with_solver_options(
                    modified_julian_days,
                    constituents,
                    &inference.relationships,
                    inference.mode,
                    solver_options,
                )
                .map(Self::Inferred)
            }
            None => GreenwichNodalBatch::prepare_modified_julian_days_with_solver_options(
                modified_julian_days,
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

struct ResolvedConstituentSelection {
    constituents: Vec<TidalConstituent>,
    report: ConstituentSelectionReport,
}

impl ResolvedConstituentSelection {
    fn profile(
        &self,
        analysis_method: AnalysisMethod,
        inferred: bool,
        fit_options: FitOptions,
        phase_reference: PhaseReference,
        nodal_corrections: NodalCorrections,
        constituent_order: &ConstituentOrder,
    ) -> String {
        let selection = match self.report.method {
            "explicit" => "fixed-constituents",
            "rayleigh" => "rayleigh-auto",
            _ => unreachable!("selection methods are constructed internally"),
        };
        let inference = if inferred { "inference-" } else { "" };
        let ordering = order_profile_suffix(constituent_order);
        let trend = if fit_options.trend { "" } else { "-no-trend" };
        format!(
            "{selection}-{}-{}-{inference}{}{ordering}{trend}",
            phase_reference.name(),
            nodal_profile_component(nodal_corrections),
            analysis_method.name(),
        )
    }
}

const fn order_profile_suffix(order: &ConstituentOrder) -> &'static str {
    match order {
        ConstituentOrder::Selection => "",
        ConstituentOrder::PercentEnergy => "-order-pe",
        ConstituentOrder::SignalToNoise => "-order-snr",
        ConstituentOrder::Frequency => "-order-frequency",
        ConstituentOrder::Explicit(_) => "-order-explicit",
    }
}

const fn nodal_profile_component(mode: NodalCorrections) -> &'static str {
    match mode {
        NodalCorrections::Exact => "nodal",
        NodalCorrections::LinearTime => "nodal-linear-time",
        NodalCorrections::Disabled => "no-nodal",
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
    let batch = ScalarAnalysisBatch::prepare(
        &input.modified_julian_days,
        &selection.constituents,
        config.inference.as_ref(),
        config.fit_options,
        config.phase_reference,
        config.nodal_corrections,
    )?;
    shared_constituent_order_indices(&config.constituent_order, batch.constituents())?;
    if let Some(filter) = &config.reconstruction {
        validate_reconstruction_filter(filter, batch.tidal_constituents())?;
    }
    let preparation_seconds = preparation_start.elapsed().as_secs_f64();

    let worker_pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()?;
    let solve_start = Instant::now();
    let solutions = solve_input(
        &worker_pool,
        &batch,
        &input,
        config.analysis_method,
        config.confidence_interval,
    )?;
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
    let constituent_index_by_rank = constituent_order_indices(
        &config.constituent_order,
        batch.constituents(),
        &series_frequency_cph,
        &solutions,
    )?;
    let result_sha256 = result_digest(
        &input.node_indices,
        &input.latitudes,
        batch.constituents(),
        &series_frequency_cph,
        &solutions,
        config.fit_options,
        config.phase_reference,
        config.nodal_corrections,
        &config.constituent_order,
        &constituent_index_by_rank,
        config.analysis_method,
        config.confidence_interval,
        config
            .inference
            .as_ref()
            .map(ScalarInferenceConfig::report)
            .as_ref(),
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
        &constituent_index_by_rank,
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
            constituent_order: &config.constituent_order,
            constituent_index_by_rank: &constituent_index_by_rank,
            result_sha256: &result_sha256,
            selection: &selection,
            inference: config.inference.as_ref(),
            fit_options: config.fit_options,
            phase_reference: config.phase_reference,
            nodal_corrections: config.nodal_corrections,
            analysis_method: config.analysis_method,
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
        profile: selection.profile(
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
        series_count: input.node_indices.len(),
        series_with_missing_observations: input
            .observation_counts
            .iter()
            .filter(|count| **count != input.modified_julian_days.len())
            .count(),
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
        inference: config.inference.as_ref().map(ScalarInferenceConfig::report),
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

fn solve_input(
    worker_pool: &rayon::ThreadPool,
    batch: &ScalarAnalysisBatch,
    input: &InputData,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
) -> Result<Vec<ScalarSolution>, AnalysisError> {
    worker_pool.install(|| match batch {
        ScalarAnalysisBatch::Standard(batch) => match (analysis_method, confidence_interval) {
            (AnalysisMethod::Ols, ConfidenceInterval::None) => {
                batch.solve_time_major_with_missing(&input.observations, &input.latitudes)
            }
            (AnalysisMethod::Ols, ConfidenceInterval::Linear(noise)) => batch
                .solve_time_major_with_missing_and_linear_confidence(
                    &input.observations,
                    &input.latitudes,
                    noise,
                ),
            (AnalysisMethod::Ols, ConfidenceInterval::MonteCarlo { options, noise }) => batch
                .solve_time_major_with_missing_and_monte_carlo_confidence(
                    &input.observations,
                    &input.latitudes,
                    options,
                    noise,
                ),
            (AnalysisMethod::Robust(options), ConfidenceInterval::None) => batch
                .solve_time_major_with_missing_robust(
                    &input.observations,
                    &input.latitudes,
                    options,
                ),
            (AnalysisMethod::Robust(options), ConfidenceInterval::Linear(noise)) => batch
                .solve_time_major_with_missing_robust_and_linear_confidence(
                    &input.observations,
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
            ) => batch.solve_time_major_with_missing_robust_and_monte_carlo_confidence(
                &input.observations,
                &input.latitudes,
                robust_options,
                monte_carlo_options,
                noise,
            ),
        },
        ScalarAnalysisBatch::Inferred(batch) => match (analysis_method, confidence_interval) {
            (AnalysisMethod::Ols, ConfidenceInterval::None) => {
                batch.solve_time_major_with_missing(&input.observations, &input.latitudes)
            }
            (AnalysisMethod::Ols, ConfidenceInterval::Linear(noise)) => batch
                .solve_time_major_with_missing_and_linear_confidence(
                    &input.observations,
                    &input.latitudes,
                    noise,
                ),
            (AnalysisMethod::Ols, ConfidenceInterval::MonteCarlo { options, noise }) => batch
                .solve_time_major_with_missing_and_monte_carlo_confidence(
                    &input.observations,
                    &input.latitudes,
                    options,
                    noise,
                ),
            (AnalysisMethod::Robust(options), ConfidenceInterval::None) => batch
                .solve_time_major_with_missing_robust(
                    &input.observations,
                    &input.latitudes,
                    options,
                ),
            (AnalysisMethod::Robust(options), ConfidenceInterval::Linear(noise)) => batch
                .solve_time_major_with_missing_robust_and_linear_confidence(
                    &input.observations,
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
            ) => batch.solve_time_major_with_missing_robust_and_monte_carlo_confidence(
                &input.observations,
                &input.latitudes,
                robust_options,
                monte_carlo_options,
                noise,
            ),
        },
    })
}

fn solution_frequencies(
    batch: &ScalarAnalysisBatch,
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
    batch: &ScalarAnalysisBatch,
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
    if config.constituent_order == ConstituentOrder::SignalToNoise
        && config.confidence_interval == ConfidenceInterval::None
    {
        return Err(AppError::Invalid(
            "SNR constituent ordering requires confidence intervals".to_owned(),
        ));
    }
    if let ConstituentOrder::Explicit(constituents) = &config.constituent_order {
        if constituents.is_empty() {
            return Err(AppError::Invalid(
                "explicit constituent order must not be empty".to_owned(),
            ));
        }
        let mut unique = BTreeSet::new();
        for constituent in constituents.iter().copied() {
            if !unique.insert(constituent) {
                return Err(AppError::Invalid(format!(
                    "explicit constituent order contains {constituent} more than once"
                )));
            }
        }
    }
    if let ConfidenceInterval::MonteCarlo { options, .. } = config.confidence_interval
        && options.realizations < 2
    {
        return Err(AppError::Invalid(
            "Monte Carlo confidence requires at least two realizations".to_owned(),
        ));
    }
    if let AnalysisMethod::Robust(options) = config.analysis_method {
        if !options.tuning_constant.is_finite() || options.tuning_constant <= 0.0 {
            return Err(AppError::Invalid(
                "robust tuning constant must be finite and greater than zero".to_owned(),
            ));
        }
        if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
            return Err(AppError::Invalid(
                "robust tolerance must be finite and greater than zero".to_owned(),
            ));
        }
        if options.max_iterations == 0 {
            return Err(AppError::Invalid(
                "robust iteration limit must be greater than zero".to_owned(),
            ));
        }
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
    if config
        .inference
        .as_ref()
        .is_some_and(|inference| inference.relationships.is_empty())
    {
        return Err(AppError::Invalid(
            "inference requires at least one relationship".to_owned(),
        ));
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

fn write_inference_metadata(
    output: &mut FileMut,
    inference: Option<&InferenceReport>,
) -> Result<(), AppError> {
    let Some(inference) = inference else {
        output.add_attribute("inference_mode", "none")?;
        return Ok(());
    };
    output.add_attribute("inference_mode", inference.mode)?;
    output.add_attribute("inference_convention", inference.convention)?;
    output.add_attribute(
        "inference_relationship_count",
        i64::try_from(inference.relationships.len()).map_err(|_| {
            AppError::Invalid("inference relationship count exceeds i64".to_owned())
        })?,
    )?;
    output.add_attribute(
        "inferred_constituent_names",
        inference
            .relationships
            .iter()
            .map(|relationship| relationship.inferred)
            .collect::<Vec<_>>()
            .join(","),
    )?;
    output.add_attribute(
        "reference_constituent_names",
        inference
            .relationships
            .iter()
            .map(|relationship| relationship.reference)
            .collect::<Vec<_>>()
            .join(","),
    )?;
    output.add_attribute(
        "inference_positive_amplitude_ratios",
        inference
            .relationships
            .iter()
            .map(|relationship| relationship.positive_amplitude_ratio.to_string())
            .collect::<Vec<_>>()
            .join(","),
    )?;
    output.add_attribute(
        "inference_positive_phase_offsets_degrees",
        inference
            .relationships
            .iter()
            .map(|relationship| relationship.positive_phase_offset_degrees.to_string())
            .collect::<Vec<_>>()
            .join(","),
    )?;
    if inference.convention == "vector_rotary" {
        output.add_attribute(
            "inference_negative_amplitude_ratios",
            inference
                .relationships
                .iter()
                .map(|relationship| {
                    relationship
                        .negative_amplitude_ratio
                        .expect("vector report contains negative amplitude ratios")
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(","),
        )?;
        output.add_attribute(
            "inference_negative_phase_offsets_degrees",
            inference
                .relationships
                .iter()
                .map(|relationship| {
                    relationship
                        .negative_phase_offset_degrees
                        .expect("vector report contains negative phase offsets")
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(","),
        )?;
    }
    Ok(())
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

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the canonical digest keeps every result-shaping input explicit"
)]
fn result_digest(
    node_indices: &[usize],
    latitudes: &[f64],
    constituents: &[Constituent],
    series_frequency_cph: &[Vec<f64>],
    solutions: &[ScalarSolution],
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    constituent_order: &ConstituentOrder,
    constituent_index_by_rank: &ConstituentOrderMap,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
    inference: Option<&InferenceReport>,
    reconstruction: Option<(&ReconstructionFilter, &[Vec<f64>])>,
) -> Result<String, AppError> {
    let mut digest = Sha256::new();
    digest.update(b"rutide-scalar-nodal-v12\0");
    digest.update([u8::from(fit_options.trend)]);
    digest.update(phase_reference.name().as_bytes());
    digest.update([0]);
    digest.update(nodal_corrections.name().as_bytes());
    digest.update([0]);
    update_constituent_order_digest(&mut digest, constituent_order, constituent_index_by_rank);
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

fn update_constituent_order_digest(
    digest: &mut Sha256,
    order: &ConstituentOrder,
    constituent_index_by_rank: &ConstituentOrderMap,
) {
    digest.update(order.name().as_bytes());
    digest.update([0]);
    if let Some(names) = order.explicit_names() {
        for name in names {
            digest.update(name.as_bytes());
            digest.update([0]);
        }
    }
    for series in 0..constituent_index_by_rank.series_count {
        let indices = constituent_index_by_rank.row(series);
        for &index in indices {
            digest.update(u64::from(index).to_le_bytes());
        }
    }
}

fn update_inference_digest(digest: &mut Sha256, inference: Option<&InferenceReport>) {
    let Some(inference) = inference else {
        digest.update(b"no-inference\0");
        return;
    };
    digest.update(inference.mode.as_bytes());
    digest.update([0]);
    digest.update(inference.convention.as_bytes());
    digest.update([0]);
    for relationship in &inference.relationships {
        digest.update(relationship.inferred.as_bytes());
        digest.update([0]);
        digest.update(relationship.reference.as_bytes());
        digest.update([0]);
        digest.update(
            relationship
                .positive_amplitude_ratio
                .to_bits()
                .to_le_bytes(),
        );
        digest.update(
            relationship
                .positive_phase_offset_degrees
                .to_bits()
                .to_le_bytes(),
        );
        match (
            relationship.negative_amplitude_ratio,
            relationship.negative_phase_offset_degrees,
        ) {
            (Some(amplitude), Some(phase)) => {
                digest.update([1]);
                digest.update(amplitude.to_bits().to_le_bytes());
                digest.update(phase.to_bits().to_le_bytes());
            }
            (None, None) => digest.update([0]),
            _ => unreachable!("inference reports are constructed internally"),
        }
    }
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
        (
            ConfidenceInterval::Linear(_) | ConfidenceInterval::MonteCarlo { .. },
            Some(amplitude),
            Some(phase),
            Some(snr),
        ) => Ok(Some((amplitude, phase, snr))),
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
    constituent_index_by_rank: &ConstituentOrderMap,
) -> Vec<SampleResult> {
    node_indices
        .iter()
        .copied()
        .zip(latitudes.iter().copied())
        .zip(observation_counts.iter().copied())
        .zip(solutions)
        .enumerate()
        .take(3)
        .map(
            |(series, (((node_index, latitude_degrees_north), observation_count), solution))| {
                SampleResult {
                    node_index,
                    latitude_degrees_north,
                    amplitude: solution.amplitude.clone(),
                    phase_degrees: solution.phase_degrees.clone(),
                    percent_energy: solution.percent_energy.clone(),
                    constituent_index_by_rank: constituent_index_by_rank
                        .row(series)
                        .iter()
                        .copied()
                        .map(usize::from)
                        .collect(),
                    amplitude_ci: solution.amplitude_ci.clone(),
                    phase_ci_degrees: solution.phase_ci_degrees.clone(),
                    signal_to_noise: solution.signal_to_noise.clone(),
                    mean: solution.mean,
                    slope_per_day: solution.slope_per_day,
                    observation_count,
                    reference_time_modified_julian_day: solution.reference_time_days,
                }
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
    constituent_order: &'data ConstituentOrder,
    constituent_index_by_rank: &'data ConstituentOrderMap,
    result_sha256: &'data str,
    selection: &'data ResolvedConstituentSelection,
    inference: Option<&'data ScalarInferenceConfig>,
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    analysis_method: AnalysisMethod,
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
        constituent_order,
        constituent_index_by_rank,
        result_sha256,
        selection,
        inference,
        fit_options,
        phase_reference,
        nodal_corrections,
        analysis_method,
        confidence_interval,
        modified_julian_days,
        reference_time_modified_julian_day,
        reconstruction,
    } = *data;
    let mut output = netcdf::create(path)?;
    output.add_dimension("series", node_indices.len())?;
    output.add_dimension("constituent", constituents.len())?;
    output.add_dimension("presentation_rank", constituents.len())?;
    if reconstruction.is_some() {
        output.add_dimension("time", modified_julian_days.len())?;
    }
    output.add_attribute("title", "RUTide scalar harmonic coefficients")?;
    output.add_attribute("rutide_schema_version", i64::from(OUTPUT_SCHEMA_VERSION))?;
    output.add_attribute("rutide_version", rutide_core::VERSION)?;
    let profile = selection.profile(
        analysis_method,
        inference.is_some(),
        fit_options,
        phase_reference,
        nodal_corrections,
        constituent_order,
    );
    output.add_attribute("profile", profile.as_str())?;
    output.add_attribute("analysis_method", analysis_method.name())?;
    output.add_attribute("trend_enabled", i64::from(fit_options.trend))?;
    output.add_attribute("phase_reference", phase_reference.name())?;
    output.add_attribute("nodal_corrections", nodal_corrections.name())?;
    output.add_attribute("constituent_order", constituent_order.name())?;
    if let Some(names) = constituent_order.explicit_names() {
        output.add_attribute("explicit_constituent_order", names.join(","))?;
    }
    if let AnalysisMethod::Robust(options) = analysis_method {
        output.add_attribute("robust_tuning_constant", options.tuning_constant)?;
        output.add_attribute("robust_tolerance", options.tolerance)?;
        output.add_attribute(
            "robust_max_iterations",
            i64::try_from(options.max_iterations)
                .map_err(|_| AppError::Invalid("robust iteration limit exceeds i64".to_owned()))?,
        )?;
    }
    output.add_attribute("confidence_interval", confidence_interval.method())?;
    if let Some(noise) = confidence_interval.noise() {
        output.add_attribute("confidence_noise", noise)?;
    }
    if let Some(options) = confidence_interval.monte_carlo_options() {
        output.add_attribute(
            "monte_carlo_realizations",
            u64::try_from(options.realizations).map_err(|_| {
                AppError::Invalid("Monte Carlo realization count exceeds u64".to_owned())
            })?,
        )?;
        output.add_attribute("monte_carlo_seed", options.seed)?;
        output.add_attribute("monte_carlo_rng", "rand_chacha-0.9-ChaCha12Rng")?;
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
    write_inference_metadata(
        &mut output,
        inference.map(ScalarInferenceConfig::report).as_ref(),
    )?;
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
    write_constituent_order_indices(
        &mut output,
        solutions.len(),
        constituents.len(),
        constituent_index_by_rank,
    )?;

    write_solution_variables(
        &mut output,
        constituents,
        solutions,
        analysis_method,
        confidence_interval,
    )?;
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

fn write_constituent_order_indices(
    output: &mut FileMut,
    series_count: usize,
    constituent_count: usize,
    constituent_index_by_rank: &ConstituentOrderMap,
) -> Result<(), AppError> {
    const TARGET_CHUNK_INDICES: usize = 1 << 20;

    if !constituent_index_by_rank.is_valid_for(series_count, constituent_count) {
        return Err(AppError::Invalid(
            "constituent presentation order is not a complete per-series permutation".to_owned(),
        ));
    }
    let mut variable = output.add_variable::<i64>(
        "constituent_index_by_rank",
        &["series", "presentation_rank"],
    )?;
    variable.put_attribute(
        "long_name",
        "stable constituent index at each requested presentation rank",
    )?;
    variable.put_attribute("start_index", 0_i64)?;
    let rows_per_chunk = (TARGET_CHUNK_INDICES / constituent_count.max(1)).max(1);
    let mut indices = Vec::with_capacity(rows_per_chunk.saturating_mul(constituent_count));
    for first_series in (0..series_count).step_by(rows_per_chunk) {
        let end_series = (first_series + rows_per_chunk).min(series_count);
        indices.clear();
        for series in first_series..end_series {
            for &index in constituent_index_by_rank.row(series) {
                indices.push(i64::from(index));
            }
        }
        variable.put_values(&indices, (first_series..end_series, ..))?;
    }
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
    analysis_method: AnalysisMethod,
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
    write_robust_variables(output, solutions, analysis_method)?;
    Ok(())
}

fn write_robust_variables(
    output: &mut FileMut,
    solutions: &[ScalarSolution],
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
            "source variable units",
        ),
        (
            "robust_ols_rms_residual",
            &ols_rms_residual,
            "source variable units",
        ),
        (
            "robust_rms_residual",
            &rms_residual,
            "source variable units",
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

const fn robust_termination_code(termination: RobustTermination) -> i64 {
    match termination {
        RobustTermination::Tolerance => 0,
        RobustTermination::ObjectiveIncrease => 1,
        RobustTermination::ExactFit => 2,
    }
}

fn write_robust_schema_metadata(output: &mut FileMut, termination: &[i64]) -> Result<(), AppError> {
    output
        .variable_mut("robust_weight_row_size")
        .ok_or_else(|| AppError::Invalid("robust row-size variable was not created".to_owned()))?
        .put_attribute("sample_dimension", "robust_observation")?;
    let mut variable = output.add_variable::<i64>("robust_termination", &["series"])?;
    write_variable(&mut variable, termination, "1")?;
    variable.put_attribute("flag_values", vec![0_i64, 1, 2])?;
    variable.put_attribute("flag_meanings", "tolerance objective_increase exact_fit")?;
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
        ConstituentOrder, NodeSelection, ScalarInferenceConfig, constituent_order_indices,
        encode_hex, normalize_source_observation, read_fvcom_scalar, resolve_node_selection,
        temporary_sibling, write_inference_metadata, write_reconstruction_variables,
    };
    use rutide_core::{
        Constituent, GreenwichNodalOls, InferenceMode, LinearConfidence, ReconstructionFilter,
        ScalarInferenceRelation, ScalarSolution, TidalConstituent,
    };

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
    fn bulk_presentation_orders_preserve_stable_constituent_indices() {
        let constituents = [
            Constituent::new("M2", 0.3),
            Constituent::new("K1", 0.1),
            Constituent::new("S2", 0.2),
        ];
        let frequencies = vec![vec![0.3, 0.1, 0.2], vec![0.2, 0.3, 0.1]];
        let solution =
            |percent_energy: [f64; 3], signal_to_noise: Option<[f64; 3]>| ScalarSolution {
                cosine_coefficient: vec![0.0; 3],
                sine_coefficient: vec![0.0; 3],
                amplitude: vec![0.0; 3],
                phase_degrees: vec![0.0; 3],
                percent_energy: percent_energy.to_vec(),
                amplitude_ci: signal_to_noise.map(|_| vec![0.0; 3]),
                phase_ci_degrees: signal_to_noise.map(|_| vec![0.0; 3]),
                signal_to_noise: signal_to_noise.map(|values| values.to_vec()),
                cosine_coefficient_variance: None,
                sine_coefficient_variance: None,
                mean: 0.0,
                slope_per_day: 0.0,
                reference_time_days: 0.0,
                robust: None,
            };
        let solutions = [
            solution([20.0, 70.0, 10.0], Some([2.0, 1.0, 3.0])),
            solution([50.0, 10.0, 40.0], Some([4.0, 6.0, 5.0])),
        ];
        for (order, expected) in [
            (
                ConstituentOrder::Selection,
                vec![vec![0, 1, 2], vec![0, 1, 2]],
            ),
            (
                ConstituentOrder::PercentEnergy,
                vec![vec![1, 0, 2], vec![0, 2, 1]],
            ),
            (
                ConstituentOrder::SignalToNoise,
                vec![vec![2, 0, 1], vec![1, 2, 0]],
            ),
            (
                ConstituentOrder::Frequency,
                vec![vec![1, 2, 0], vec![2, 0, 1]],
            ),
            (
                ConstituentOrder::Explicit(vec![
                    TidalConstituent::S2,
                    TidalConstituent::M2,
                    TidalConstituent::K1,
                ]),
                vec![vec![2, 0, 1], vec![2, 0, 1]],
            ),
        ] {
            let order_map =
                constituent_order_indices(&order, &constituents, &frequencies, &solutions)
                    .expect("valid presentation order");
            assert_eq!(
                (0..solutions.len())
                    .map(|series| order_map.row(series).to_vec())
                    .collect::<Vec<_>>(),
                expected
            );
        }

        assert!(
            constituent_order_indices(
                &ConstituentOrder::Explicit(vec![TidalConstituent::M2]),
                &constituents,
                &frequencies,
                &solutions,
            )
            .is_err()
        );
        let no_confidence = [solution([20.0, 70.0, 10.0], None)];
        assert!(
            constituent_order_indices(
                &ConstituentOrder::SignalToNoise,
                &constituents,
                &frequencies[..1],
                &no_confidence,
            )
            .is_err()
        );
    }

    #[test]
    fn presentation_orders_match_the_pinned_python_utide_oracle() {
        // Frozen from UTide 8fabe121752bc317931472a10a42e306715106de.
        let tidal_constituents = [
            TidalConstituent::M2,
            TidalConstituent::S2,
            TidalConstituent::N2,
            TidalConstituent::K1,
            TidalConstituent::O1,
        ];
        let time = (0_u32..745)
            .map(|index| 58_113.0 + f64::from(index) / 24.0)
            .collect::<Vec<_>>();
        let observations = include_str!("../../rutide-core/tests/data/fvcom_node_0_zeta_f32.hex")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| {
                f64::from(f32::from_bits(
                    u32::from_str_radix(line, 16).expect("fixture contains hexadecimal f32 bits"),
                ))
            })
            .collect::<Vec<_>>();
        let model = GreenwichNodalOls::prepare_modified_julian_days(
            &time,
            60.957_717_895_507_81,
            &tidal_constituents,
        )
        .expect("valid pinned-oracle model");
        let solution = model
            .solve_with_linear_confidence(&observations, LinearConfidence::Colored)
            .expect("valid pinned-oracle confidence solution");
        let frequencies = vec![
            model
                .constituents()
                .iter()
                .map(|constituent| constituent.frequency_cph)
                .collect(),
        ];
        for (order, expected) in [
            (ConstituentOrder::PercentEnergy, vec![0, 1, 2, 3, 4]),
            (ConstituentOrder::SignalToNoise, vec![0, 1, 3, 2, 4]),
            (ConstituentOrder::Frequency, vec![4, 3, 2, 0, 1]),
            (
                ConstituentOrder::Explicit(vec![
                    TidalConstituent::K1,
                    TidalConstituent::M2,
                    TidalConstituent::O1,
                    TidalConstituent::S2,
                    TidalConstituent::N2,
                ]),
                vec![3, 0, 4, 1, 2],
            ),
        ] {
            let order_map = constituent_order_indices(
                &order,
                model.constituents(),
                &frequencies,
                std::slice::from_ref(&solution),
            )
            .expect("valid Python-compatible presentation order");
            assert_eq!(order_map.row(0), expected);
        }
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

    #[test]
    fn scalar_inference_metadata_round_trips_through_netcdf() {
        let destination = std::env::temp_dir().join("rutide-inference-metadata-test.nc");
        let path = temporary_sibling(&destination).expect("valid temporary path");
        let inference = ScalarInferenceConfig {
            mode: InferenceMode::Approximate,
            relationships: vec![ScalarInferenceRelation::new(
                TidalConstituent::S2,
                TidalConstituent::M2,
                0.35,
                20.0,
            )],
        };
        let mut dataset = netcdf::create(&path).expect("create test NetCDF");
        write_inference_metadata(&mut dataset, Some(&inference.report()))
            .expect("write inference metadata");
        dataset.close().expect("close test NetCDF");

        let dataset = netcdf::open(&path).expect("open test NetCDF");
        for (name, expected) in [
            ("inference_mode", "approximate"),
            ("inference_convention", "scalar"),
            ("inferred_constituent_names", "S2"),
            ("reference_constituent_names", "M2"),
            ("inference_positive_amplitude_ratios", "0.35"),
            ("inference_positive_phase_offsets_degrees", "20"),
        ] {
            assert_eq!(
                dataset
                    .attribute(name)
                    .unwrap_or_else(|| panic!("missing attribute {name}"))
                    .value()
                    .unwrap_or_else(|error| panic!("cannot read attribute {name}: {error}")),
                netcdf::AttributeValue::Str(expected.to_owned())
            );
        }
        drop(dataset);
        fs::remove_file(path).expect("remove test NetCDF");
    }
}
