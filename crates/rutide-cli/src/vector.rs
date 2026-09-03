//! FVCOM depth-averaged, sigma-layer, and fixed-depth vector-current path.

use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use netcdf::{
    FileMut, Variable,
    types::{FloatType, IntType, NcVariableType},
};
use rayon::{ThreadPoolBuilder, prelude::*};
use rutide_core::{
    AnalysisError, COLORED_NOISE_FREQUENCY_BANDS_CPH, Constituent, ConstituentDiagnosticsOptions,
    ConstituentSelectionDiagnostics, FitOptions, GreenwichNodalBatch, GreenwichNodalReconstructor,
    NodalCorrections, PhaseReference, ReconstructionFilter, ResidualSpectrumMethod,
    RobustDiagnostics, RobustTermination, SolverOptions, TidalConstituent, VectorInferenceBatch,
    VectorReconstruction, VectorSolution,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{
    AnalysisMethod, AnalyzeConfig, AppError, BoundedInputPipeline, ConfidenceInterval,
    ConstituentDiagnosticsReport, ConstituentOrder, ConstituentOrderMap, ConstituentSelection,
    ConstituentSelectionReport, CoreSamplingDiagnostics, InferenceReport, NodeSelection,
    ReconstructionReport, ResolvedConstituentSelection, RobustOptionsReport, SamplingSummary,
    SeriesSamplingDiagnostics, SpectralBandSummary, StageTimings, VectorInferenceConfig,
    constituent_order_indices, define_constituent_diagnostic_variables, diagnose_sampling,
    diagnostic_neighbor_value, diagnostic_scalar_value, encode_hex, nodal_profile_component,
    normalize_source_observation, order_profile_suffix, read_fvcom_time_axis, read_selected_1d,
    read_selected_time_major, reconstruction_report, required_dimension_length, required_variable,
    resolve_constituent_selection, retain_time_major_rows, robust_termination_code,
    spatial_chunk_plan, summarize_sampling, temporary_sibling,
    update_constituent_diagnostics_digest, update_constituent_order_digest,
    update_inference_digest, update_reconstruction_filter_digest, update_robust_options_digest,
    update_sampling_digest, validate_config, validate_constituent_diagnostics_shape,
    validate_dimensions, validate_reconstruction_filter, validate_source_value,
    write_constituent_diagnostics, write_constituent_diagnostics_attributes,
    write_constituent_order_indices, write_inference_metadata, write_json_report,
    write_robust_schema_metadata, write_sampling_diagnostics, write_variable,
};

/// `NetCDF` and JSON report schema emitted by vector-current analyses.
pub const VECTOR_OUTPUT_SCHEMA_VERSION: u32 = 15;
const FIXED_DEPTH_ZETA_SPAN_AMPLIFICATION: usize = 6;

/// Configuration for one FVCOM current analysis.
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
    /// Native FVCOM sigma layers to analyze, or `None` for depth-averaged `ua`/`va`.
    pub layers: Option<NodeSelection>,
    /// Fixed physical depths in metres below the instantaneous free surface.
    ///
    /// This is mutually exclusive with [`Self::layers`]. `None` selects either
    /// native layers or depth-averaged currents according to [`Self::layers`].
    pub fixed_depths_meters: Option<Vec<f64>>,
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
    /// Ordinary or configured robust least squares.
    pub analysis_method: AnalysisMethod,
    /// Optional extended constituent-identifiability diagnostics.
    pub constituent_diagnostics: Option<ConstituentDiagnosticsOptions>,
    /// Optional complete-series reconstruction and constituent filter.
    pub reconstruction: Option<ReconstructionFilter>,
    /// Number of outer spatial worker threads.
    pub workers: usize,
    /// Maximum spatial series read and solved together, or automatic when absent.
    pub chunk_series: Option<usize>,
    /// Permit replacing existing output and report files.
    pub overwrite: bool,
}

/// A small retained vector-current sample in the JSON run report.
#[derive(Clone, Debug, Serialize)]
pub struct VectorSampleResult {
    /// Original zero-based FVCOM element index.
    pub element_index: usize,
    /// Original zero-based FVCOM sigma-layer index for depth-resolved currents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<usize>,
    /// Requested metres below the instantaneous free surface.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_meters_below_surface: Option<f64>,
    /// Whether this coordinate was fitted or lacked enough physical samples.
    pub analysis_status: &'static str,
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
    /// Logical payload bytes represented by the selected input coordinates.
    pub logical_input_bytes: u64,
    /// Output coefficient path.
    pub output_path: String,
    /// Completed output file size.
    pub output_file_bytes: u64,
    /// Result storage strategy: `buffered` or `incremental`.
    pub result_output: &'static str,
    /// Number of finite timestamps retained for analysis.
    pub time_count: usize,
    /// Number of timestamps in the source time dimension.
    pub source_time_count: usize,
    /// Number of missing source timestamps and corresponding rows discarded.
    pub discarded_timestamp_count: usize,
    /// Number of analyzed current series (`elements × selected vertical coordinates`).
    pub series_count: usize,
    /// Number of distinct selected FVCOM elements.
    pub element_count: usize,
    /// Vertical source mode: `depth-averaged`, `sigma-layer`, or `fixed-depth`.
    pub vertical_mode: &'static str,
    /// Selected zero-based native sigma-layer indices for depth-resolved currents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_indices: Option<Vec<usize>>,
    /// Requested positive metres below the instantaneous free surface.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_depths_meters: Option<Vec<f64>>,
    /// Number of vertical-element series fitted successfully.
    pub fitted_series_count: usize,
    /// Number of rectangular output rows without enough observations for the model.
    pub unavailable_series_count: usize,
    /// Number of current series with at least one missing component sample.
    pub series_with_missing_observations: usize,
    /// Aggregate temporal and spectral coverage across fitted current series.
    pub sampling: SamplingSummary,
    /// Number of outer spatial workers.
    pub workers: usize,
    /// Actual maximum number of spatial series held in one input chunk.
    pub chunk_series: usize,
    /// Number of spatial chunks processed.
    pub chunk_count: usize,
    /// Whether automatic input used a bounded reader/solver overlap.
    pub input_pipeline: &'static str,
    /// Maximum logical bytes occupied by all concurrently resident promoted
    /// component arrays.
    pub maximum_observation_buffer_bytes: u64,
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
    /// Extended constituent-identifiability diagnostics, when enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constituent_diagnostics: Option<ConstituentDiagnosticsReport>,
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
    layer_indices: Option<Vec<usize>>,
    fixed_depths_meters: Option<Vec<f64>>,
    latitudes: Vec<f64>,
    #[cfg(test)]
    eastward: Vec<f64>,
    #[cfg(test)]
    northward: Vec<f64>,
    observation_counts: Vec<usize>,
    input_file_bytes: u64,
    logical_input_bytes: u64,
}

struct VectorInputMetadata {
    modified_julian_days: Vec<f64>,
    retained_time_indices: Vec<usize>,
    source_time_count: usize,
    discarded_timestamp_count: usize,
    element_indices: Vec<usize>,
    layer_indices: Option<Vec<usize>>,
    fixed_depths_meters: Option<Vec<f64>>,
    fixed_depth_source: Option<FixedDepthSource>,
    latitudes: Vec<f64>,
    eastward_fill: Option<f32>,
    northward_fill: Option<f32>,
    input_file_bytes: u64,
    logical_input_bytes: u64,
}

struct FixedDepthSource {
    source_layer_count: usize,
    current_buffer_bytes_per_layer: usize,
    zeta_buffer_bytes_per_node: usize,
    wet_buffer_bytes_per_element: usize,
    element_nodes: Vec<[usize; 3]>,
    layer_sigma: Vec<[f64; 3]>,
    element_bathymetry: Vec<[f64; 3]>,
    zeta_fill: Option<f32>,
    wet_cells_fill: Option<i32>,
}

struct FixedDepthStaticGeometry {
    layer_sigma: Vec<[f64; 3]>,
    element_bathymetry: Vec<[f64; 3]>,
}

struct VectorInputChunk {
    eastward: Vec<f64>,
    northward: Vec<f64>,
    observation_counts: Vec<usize>,
}

struct PipelinedVectorChunk {
    first_series: usize,
    chunk: VectorInputChunk,
    input_seconds: f64,
}

/// Owns the only `NetCDF` handle used by the background input path.
///
/// The zero-capacity channel is intentional: while the caller processes chunk
/// N, the reader may construct chunk N+1, but it cannot begin N+2 until N has
/// been dropped and the caller receives N+1. That makes the pipeline a strict
/// two-buffer bound even when solving is slower than input.
fn pipelined_vector_chunk_reader(
    dataset: netcdf::File,
    metadata: Arc<VectorInputMetadata>,
    series_per_chunk: usize,
) -> Result<BoundedInputPipeline<PipelinedVectorChunk>, AppError> {
    if metadata.is_fixed_depth() {
        return Err(AppError::Invalid(
            "fixed-depth input cannot use the regular vector reader pipeline".to_owned(),
        ));
    }
    if series_per_chunk == 0 {
        return Err(AppError::Invalid(
            "pipelined vector input requires a non-zero chunk size".to_owned(),
        ));
    }
    let series_count = metadata.series_count();
    let mut first_series = 0;
    BoundedInputPipeline::spawn("rutide-netcdf-reader", move || {
        if first_series >= series_count {
            return Ok(None);
        }
        let end_series = (first_series + series_per_chunk).min(series_count);
        let input_start = Instant::now();
        let chunk = read_fvcom_vector_chunk(&dataset, &metadata, first_series..end_series)?;
        let read = PipelinedVectorChunk {
            first_series,
            chunk,
            input_seconds: input_start.elapsed().as_secs_f64(),
        };
        first_series = end_series;
        Ok(Some(read))
    })
}

enum FixedDepthFloatBlock {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl FixedDepthFloatBlock {
    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    fn retain_time_rows(
        self,
        source_time_count: usize,
        series_count: usize,
        retained_time_indices: &[usize],
    ) -> Result<Self, AppError> {
        match self {
            Self::F32(values) => retain_time_major_rows(
                values,
                source_time_count,
                series_count,
                retained_time_indices,
            )
            .map(Self::F32),
            Self::F64(values) => retain_time_major_rows(
                values,
                source_time_count,
                series_count,
                retained_time_indices,
            )
            .map(Self::F64),
        }
    }

    fn normalize(
        &mut self,
        variable: &str,
        fill_value: Option<f32>,
        series_count: usize,
    ) -> Result<(), AppError> {
        match self {
            Self::F32(values) => {
                for (index, value) in values.iter_mut().enumerate() {
                    let is_fill = fill_value.is_some_and(|fill| value.to_bits() == fill.to_bits());
                    if value.is_nan() || is_fill {
                        *value = f32::NAN;
                    } else if value.is_infinite() {
                        return Err(AppError::Invalid(format!(
                            "{variable} contains an infinite value at series {}, time {}",
                            index % series_count,
                            index / series_count,
                        )));
                    }
                }
            }
            Self::F64(values) => {
                for (index, value) in values.iter_mut().enumerate() {
                    *value = normalize_source_observation(
                        variable,
                        *value,
                        fill_value,
                        index % series_count,
                        index / series_count,
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl VectorInputData {
    fn series_count(&self) -> usize {
        self.element_indices.len() * self.vertical_count()
    }

    fn vertical_count(&self) -> usize {
        self.layer_indices.as_ref().map_or_else(
            || self.fixed_depths_meters.as_ref().map_or(1, Vec::len),
            Vec::len,
        )
    }

    fn series_coordinates(&self, series: usize) -> (Option<usize>, Option<f64>, usize, f64) {
        let element_position = series % self.element_indices.len();
        let vertical_position = series / self.element_indices.len();
        let layer_index = self
            .layer_indices
            .as_ref()
            .map(|layers| layers[vertical_position]);
        let depth = self
            .fixed_depths_meters
            .as_ref()
            .map(|depths| depths[vertical_position]);
        (
            layer_index,
            depth,
            self.element_indices[element_position],
            self.latitudes[element_position],
        )
    }

    fn is_depth_resolved(&self) -> bool {
        self.layer_indices.is_some() || self.fixed_depths_meters.is_some()
    }

    fn is_fixed_depth(&self) -> bool {
        self.fixed_depths_meters.is_some()
    }

    fn vertical_mode(&self) -> &'static str {
        if self.fixed_depths_meters.is_some() {
            "fixed-depth"
        } else if self.layer_indices.is_some() {
            "sigma-layer"
        } else {
            "depth-averaged"
        }
    }

    fn series_dimensions(&self) -> &'static [&'static str] {
        if self.fixed_depths_meters.is_some() {
            &["depth", "element"]
        } else if self.layer_indices.is_some() {
            &["siglay", "element"]
        } else {
            &["series"]
        }
    }

    fn solution_dimensions(&self) -> Vec<&'static str> {
        let mut dimensions = self.series_dimensions().to_vec();
        dimensions.push("constituent");
        dimensions
    }
}

impl VectorInputMetadata {
    fn series_count(&self) -> usize {
        self.element_indices.len() * self.vertical_count()
    }

    fn vertical_count(&self) -> usize {
        self.layer_indices.as_ref().map_or_else(
            || self.fixed_depths_meters.as_ref().map_or(1, Vec::len),
            Vec::len,
        )
    }

    fn is_fixed_depth(&self) -> bool {
        self.fixed_depths_meters.is_some()
    }

    fn chunk_latitudes(&self, series_range: std::ops::Range<usize>) -> Vec<f64> {
        let element_count = self.element_indices.len();
        series_range
            .map(|series| self.latitudes[series % element_count])
            .collect()
    }
}

fn regular_vector_chunk_plan(
    requested: Option<usize>,
    series_count: usize,
    source_time_count: usize,
    resident_component_count: usize,
    workers: usize,
) -> Result<(super::SpatialChunkPlan, bool), AppError> {
    // Explicit chunk sizes remain an exact memory/reproducibility override.
    if requested.is_some() {
        return spatial_chunk_plan(
            requested,
            series_count,
            source_time_count,
            resident_component_count,
            workers,
        )
        .map(|plan| (plan, false));
    }
    let double_buffer_components = resident_component_count
        .checked_mul(2)
        .ok_or_else(|| AppError::Invalid("vector pipeline buffer count overflows".to_owned()))?;
    let pipelined = spatial_chunk_plan(
        None,
        series_count,
        source_time_count,
        double_buffer_components,
        workers,
    )?;
    if pipelined.chunk_count > 1 {
        Ok((pipelined, true))
    } else {
        spatial_chunk_plan(
            None,
            series_count,
            source_time_count,
            resident_component_count,
            workers,
        )
        .map(|plan| (plan, false))
    }
}

struct SamplingSummaryAccumulator {
    series_count: usize,
    minimum_observation_count: usize,
    maximum_observation_count: usize,
    minimum_record_span_days: f64,
    maximum_record_span_days: f64,
    maximum_gap_hours: f64,
    fft_series_count: usize,
    lomb_scargle_series_count: usize,
    minimum_band_bin_count: [usize; 9],
    minimum_band_usable_bin_count: [usize; 9],
    series_without_usable_bins: [usize; 9],
}

impl Default for SamplingSummaryAccumulator {
    fn default() -> Self {
        Self {
            series_count: 0,
            minimum_observation_count: usize::MAX,
            maximum_observation_count: 0,
            minimum_record_span_days: f64::INFINITY,
            maximum_record_span_days: f64::NEG_INFINITY,
            maximum_gap_hours: f64::NEG_INFINITY,
            fft_series_count: 0,
            lomb_scargle_series_count: 0,
            minimum_band_bin_count: [usize::MAX; 9],
            minimum_band_usable_bin_count: [usize::MAX; 9],
            series_without_usable_bins: [0; 9],
        }
    }
}

impl SamplingSummaryAccumulator {
    fn extend(&mut self, diagnostics: &[CoreSamplingDiagnostics]) {
        for series in diagnostics {
            if series.residual_spectrum_time_count == 0 {
                continue;
            }
            self.series_count += 1;
            self.minimum_observation_count =
                self.minimum_observation_count.min(series.observation_count);
            self.maximum_observation_count =
                self.maximum_observation_count.max(series.observation_count);
            self.minimum_record_span_days =
                self.minimum_record_span_days.min(series.record_span_days);
            self.maximum_record_span_days =
                self.maximum_record_span_days.max(series.record_span_days);
            self.maximum_gap_hours = self.maximum_gap_hours.max(series.largest_gap_hours);
            match series.residual_spectrum_method {
                ResidualSpectrumMethod::Fft => self.fft_series_count += 1,
                ResidualSpectrumMethod::LombScargle => self.lomb_scargle_series_count += 1,
            }
            for band in 0..9 {
                self.minimum_band_bin_count[band] =
                    self.minimum_band_bin_count[band].min(series.spectral_band_bin_count[band]);
                self.minimum_band_usable_bin_count[band] = self.minimum_band_usable_bin_count[band]
                    .min(series.spectral_band_usable_bin_count[band]);
                if series.spectral_band_usable_bin_count[band] == 0 {
                    self.series_without_usable_bins[band] += 1;
                }
            }
        }
    }

    fn finish(self) -> SamplingSummary {
        if self.series_count == 0 {
            return SamplingSummary {
                minimum_observation_count: 0,
                maximum_observation_count: 0,
                minimum_record_span_days: 0.0,
                maximum_record_span_days: 0.0,
                maximum_gap_hours: 0.0,
                fft_series_count: 0,
                lomb_scargle_series_count: 0,
                spectral_bands: COLORED_NOISE_FREQUENCY_BANDS_CPH
                    .iter()
                    .copied()
                    .map(
                        |[lower_frequency_cph, upper_frequency_cph]| SpectralBandSummary {
                            lower_frequency_cph,
                            upper_frequency_cph,
                            minimum_bin_count: 0,
                            minimum_usable_bin_count: 0,
                            series_without_usable_bins: 0,
                        },
                    )
                    .collect(),
            };
        }
        SamplingSummary {
            minimum_observation_count: self.minimum_observation_count,
            maximum_observation_count: self.maximum_observation_count,
            minimum_record_span_days: self.minimum_record_span_days,
            maximum_record_span_days: self.maximum_record_span_days,
            maximum_gap_hours: self.maximum_gap_hours,
            fft_series_count: self.fft_series_count,
            lomb_scargle_series_count: self.lomb_scargle_series_count,
            spectral_bands: COLORED_NOISE_FREQUENCY_BANDS_CPH
                .iter()
                .copied()
                .enumerate()
                .map(
                    |(band, [lower_frequency_cph, upper_frequency_cph])| SpectralBandSummary {
                        lower_frequency_cph,
                        upper_frequency_cph,
                        minimum_bin_count: self.minimum_band_bin_count[band],
                        minimum_usable_bin_count: self.minimum_band_usable_bin_count[band],
                        series_without_usable_bins: self.series_without_usable_bins[band],
                    },
                )
                .collect(),
        }
    }
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

    fn diagnose_vector_time_major(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        solutions: &[VectorSolution],
        options: ConstituentDiagnosticsOptions,
    ) -> Result<Vec<ConstituentSelectionDiagnostics>, AnalysisError> {
        match self {
            Self::Standard(batch) => {
                batch.diagnose_vector_time_major(eastward, northward, latitudes, solutions, options)
            }
            Self::Inferred(batch) => {
                batch.diagnose_vector_time_major(eastward, northward, latitudes, solutions, options)
            }
        }
    }
}

/// Analyze depth-averaged, native sigma-layer, or fixed-depth FVCOM currents.
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
    let dataset = netcdf::open(&config.input)?;
    let metadata = Arc::new(read_fvcom_vector_metadata(
        &config.input,
        &dataset,
        &config.elements,
        config.layers.as_ref(),
        config.fixed_depths_meters.as_deref(),
    )?);
    let mut input_seconds = input_start.elapsed().as_secs_f64();

    let preparation_start = Instant::now();
    let selection = resolve_constituent_selection(
        &config.constituent_selection,
        &metadata.modified_julian_days,
    )?;
    let batch = VectorAnalysisBatch::prepare(
        &metadata.modified_julian_days,
        &selection.constituents,
        config.inference.as_ref(),
        config.fit_options,
        config.phase_reference,
        config.nodal_corrections,
    )?;
    let fitted_reference_count = batch.constituents().len().saturating_sub(
        config
            .inference
            .as_ref()
            .map_or(0, |value| value.relationships.len()),
    );
    let minimum_fit_observations = fitted_reference_count
        .checked_mul(2)
        // Mean, optional trend, and one residual degree of freedom: the core
        // deliberately requires an overdetermined model.
        .and_then(|value| value.checked_add(2 + usize::from(config.fit_options.trend)))
        .ok_or_else(|| {
            AppError::Invalid("minimum vector observation count overflows".to_owned())
        })?;
    super::shared_constituent_order_indices(&config.constituent_order, batch.constituents())?;
    if let Some(filter) = &config.reconstruction {
        validate_reconstruction_filter(filter, batch.tidal_constituents())?;
    }
    // Robust sigma-layer chunks retain two diagnostic rows per observation while
    // reconstruction can retain two more. Size automatic chunks by the largest
    // concurrently resident time-major result set, then report the actual two
    // promoted source-component buffers.
    let (mut chunk_plan, fixed_depth_elements_per_chunk, input_pipeline) = if metadata
        .is_fixed_depth()
    {
        let depth_count = metadata.vertical_count();
        let source = metadata
            .fixed_depth_source
            .as_ref()
            .ok_or_else(|| AppError::Invalid("fixed-depth geometry was not prepared".to_owned()))?;
        let source_layer_count = source.source_layer_count;
        let requested_elements = config
            .chunk_series
            .map(|series| {
                if series < depth_count {
                    return Err(AppError::Invalid(format!(
                        "fixed-depth chunk series must be at least the requested depth count {depth_count}"
                    )));
                }
                Ok(series / depth_count)
            })
            .transpose()?;
        // Source-width u/v, a bounded three-node zeta span, wet mask, and
        // interpolated u/v coexist inside the reader. Multiple depths
        // temporarily use a second interpolated buffer while converting the
        // parallel time-major result into depth-major chunks.
        let current_bytes_per_element = source_layer_count
            .checked_mul(source.current_buffer_bytes_per_layer)
            .ok_or_else(|| {
                AppError::Invalid("fixed-depth chunk budget exceeds usize".to_owned())
            })?;
        let zeta_bytes_per_element = 3_usize
            .checked_mul(FIXED_DEPTH_ZETA_SPAN_AMPLIFICATION)
            .and_then(|nodes| nodes.checked_mul(source.zeta_buffer_bytes_per_node))
            .ok_or_else(|| {
                AppError::Invalid("fixed-depth zeta chunk budget exceeds usize".to_owned())
            })?;
        let output_bytes_per_element = depth_count
            .checked_mul(if depth_count == 1 { 2 } else { 4 })
            .and_then(|components| components.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                AppError::Invalid("fixed-depth output chunk budget exceeds usize".to_owned())
            })?;
        let bytes_per_element = current_bytes_per_element
            .checked_add(zeta_bytes_per_element)
            .and_then(|bytes| bytes.checked_add(source.wet_buffer_bytes_per_element))
            .and_then(|bytes| bytes.checked_add(output_bytes_per_element))
            .ok_or_else(|| {
                AppError::Invalid("fixed-depth chunk budget exceeds usize".to_owned())
            })?;
        let component_count = bytes_per_element.div_ceil(std::mem::size_of::<f64>());
        let element_plan = spatial_chunk_plan(
            requested_elements,
            metadata.element_indices.len(),
            metadata.source_time_count,
            component_count,
            config.workers,
        )?;
        let series_per_chunk = element_plan
            .series_per_chunk
            .checked_mul(depth_count)
            .ok_or_else(|| {
                AppError::Invalid("fixed-depth chunk series exceeds usize".to_owned())
            })?;
        let plan = super::SpatialChunkPlan {
            series_per_chunk,
            chunk_count: element_plan.chunk_count,
            maximum_observation_buffer_bytes: element_plan.maximum_observation_buffer_bytes,
        };
        (plan, Some(element_plan.series_per_chunk), false)
    } else {
        let resident_component_count =
            if metadata.layer_indices.is_some() && config.analysis_method != AnalysisMethod::Ols {
                4
            } else {
                2
            };
        let (plan, pipelined) = regular_vector_chunk_plan(
            config.chunk_series,
            metadata.series_count(),
            metadata.source_time_count,
            resident_component_count,
            config.workers,
        )?;
        (plan, None, pipelined)
    };
    if !metadata.is_fixed_depth() {
        let concurrent_chunks = 1 + usize::from(input_pipeline);
        chunk_plan.maximum_observation_buffer_bytes = u64::try_from(
            chunk_plan
                .series_per_chunk
                .checked_mul(metadata.source_time_count)
                .and_then(|value| value.checked_mul(2 * std::mem::size_of::<f64>()))
                .and_then(|value| value.checked_mul(concurrent_chunks))
                .ok_or_else(|| {
                    AppError::Invalid("observation chunk size exceeds usize".to_owned())
                })?,
        )
        .map_err(|_| AppError::Invalid("observation chunk size exceeds u64".to_owned()))?;
    }
    let sampling_plan =
        rutide_core::SamplingDiagnosticsPlan::prepare(&metadata.modified_julian_days)?;
    let reconstructor = config
        .reconstruction
        .as_ref()
        .map(|_| batch.reconstructor_modified_julian_days(&metadata.modified_julian_days))
        .transpose()?;
    let preparation_seconds = preparation_start.elapsed().as_secs_f64();

    let worker_pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()?;
    let series_count = metadata.series_count();
    let mut input = VectorInputData {
        modified_julian_days: metadata.modified_julian_days.clone(),
        source_time_count: metadata.source_time_count,
        discarded_timestamp_count: metadata.discarded_timestamp_count,
        element_indices: metadata.element_indices.clone(),
        layer_indices: metadata.layer_indices.clone(),
        fixed_depths_meters: metadata.fixed_depths_meters.clone(),
        latitudes: metadata.latitudes.clone(),
        #[cfg(test)]
        eastward: Vec::new(),
        #[cfg(test)]
        northward: Vec::new(),
        observation_counts: if metadata.is_fixed_depth() {
            vec![0; series_count]
        } else {
            Vec::with_capacity(series_count)
        },
        input_file_bytes: metadata.input_file_bytes,
        logical_input_bytes: metadata.logical_input_bytes,
    };
    let incremental_results = input.is_depth_resolved();
    let mut solutions = if incremental_results {
        Vec::new()
    } else {
        Vec::with_capacity(series_count)
    };
    let mut series_frequency_cph = if incremental_results {
        Vec::new()
    } else {
        Vec::with_capacity(series_count)
    };
    let mut sampling_diagnostics = if incremental_results {
        Vec::new()
    } else {
        Vec::with_capacity(series_count)
    };
    let mut constituent_diagnostics = if incremental_results {
        None
    } else {
        config
            .constituent_diagnostics
            .map(|_| Vec::with_capacity(series_count))
    };
    let mut reconstruction = if incremental_results {
        None
    } else {
        config
            .reconstruction
            .as_ref()
            .map(|_| Vec::with_capacity(series_count))
    };
    let mut sampling_accumulator = SamplingSummaryAccumulator::default();
    let mut sample_results = Vec::<(usize, VectorSampleResult)>::with_capacity(3);
    let mut first_frequency = None::<Vec<f64>>;
    let mut frequency_varies_by_series = false;
    let mut first_constituent_order = None::<Vec<u16>>;
    let mut constituent_order_varies_by_series = false;
    let mut solve_seconds = 0.0;
    let mut reconstruction_seconds = 0.0;
    let mut result_processing_seconds = 0.0;
    let mut output_seconds = 0.0;
    let output_start = Instant::now();
    let mut incremental_output = if incremental_results {
        Some(IncrementalVectorOutput::create(
            &config.output,
            &VectorOutputDefinition {
                input: &input,
                constituents: batch.constituents(),
                constituent_order: &config.constituent_order,
                selection: &selection,
                inference: config.inference.as_ref(),
                fit_options: config.fit_options,
                phase_reference: config.phase_reference,
                nodal_corrections: config.nodal_corrections,
                analysis_method: config.analysis_method,
                confidence_interval: config.confidence_interval,
                constituent_diagnostics: config.constituent_diagnostics,
                chunk_plan,
                input_pipeline,
                reference_time_modified_julian_day: batch.reference_time_modified_julian_day(),
                reconstruction: config.reconstruction.as_ref(),
                result_output: "incremental",
            },
        )?)
    } else {
        None
    };
    output_seconds += output_start.elapsed().as_secs_f64();
    let mut dataset = Some(dataset);
    let mut pipelined_reader = if input_pipeline {
        Some(pipelined_vector_chunk_reader(
            dataset.take().ok_or_else(|| {
                AppError::Invalid("vector input dataset is unavailable".to_owned())
            })?,
            Arc::clone(&metadata),
            chunk_plan.series_per_chunk,
        )?)
    } else {
        None
    };
    let mut next_regular_series = 0;
    let mut next_fixed_element = 0;
    let mut pending_fixed_depths = VecDeque::new();
    loop {
        let next = if let Some(reader) = pipelined_reader.as_mut() {
            reader.next()?.map(|read| {
                input_seconds += read.input_seconds;
                (read.first_series, read.chunk)
            })
        } else {
            let read_start = Instant::now();
            let next = if let Some(elements_per_chunk) = fixed_depth_elements_per_chunk {
                if pending_fixed_depths.is_empty()
                    && next_fixed_element < metadata.element_indices.len()
                {
                    let first_element = next_fixed_element;
                    let end_element =
                        (first_element + elements_per_chunk).min(metadata.element_indices.len());
                    let chunks = read_fvcom_fixed_depth_element_chunk(
                        dataset.as_ref().ok_or_else(|| {
                            AppError::Invalid("vector input dataset is unavailable".to_owned())
                        })?,
                        &metadata,
                        first_element..end_element,
                        &worker_pool,
                    )?;
                    for (depth, chunk) in chunks.into_iter().enumerate() {
                        let first_series = depth * metadata.element_indices.len() + first_element;
                        pending_fixed_depths.push_back((first_series, chunk));
                    }
                    next_fixed_element = end_element;
                }
                pending_fixed_depths.pop_front()
            } else if next_regular_series < series_count {
                let first_series = next_regular_series;
                let end_series = (first_series + chunk_plan.series_per_chunk).min(series_count);
                next_regular_series = end_series;
                Some((
                    first_series,
                    read_fvcom_vector_chunk(
                        dataset.as_ref().ok_or_else(|| {
                            AppError::Invalid("vector input dataset is unavailable".to_owned())
                        })?,
                        &metadata,
                        first_series..end_series,
                    )?,
                ))
            } else {
                None
            };
            input_seconds += read_start.elapsed().as_secs_f64();
            next
        };
        let Some((first_series, chunk)) = next else {
            break;
        };
        let end_series = first_series + chunk.observation_counts.len();
        let latitudes = metadata.chunk_latitudes(first_series..end_series);

        let solve_start = Instant::now();
        let chunk_solutions = if input.is_fixed_depth() {
            solve_vector_input_with_unavailable(
                &worker_pool,
                &batch,
                &chunk.eastward,
                &chunk.northward,
                &latitudes,
                &chunk.observation_counts,
                minimum_fit_observations,
                first_series,
                config.analysis_method,
                config.confidence_interval,
            )?
        } else {
            solve_vector_input(
                &worker_pool,
                &batch,
                &chunk.eastward,
                &chunk.northward,
                &latitudes,
                first_series,
                config.analysis_method,
                config.confidence_interval,
            )?
        };
        solve_seconds += solve_start.elapsed().as_secs_f64();

        let result_start = Instant::now();
        let chunk_frequency_cph = vector_solution_frequencies(&batch, &chunk_solutions)?;
        let chunk_diagnostics = if input.is_fixed_depth() {
            diagnose_vector_sampling_with_unavailable(
                &worker_pool,
                &sampling_plan,
                &chunk_frequency_cph,
                &chunk.eastward,
                &chunk.northward,
                &chunk.observation_counts,
                minimum_fit_observations,
            )?
        } else {
            diagnose_sampling(
                &worker_pool,
                &sampling_plan,
                &chunk_frequency_cph,
                |time, series| {
                    let index = time * chunk_solutions.len() + series;
                    chunk.eastward[index].is_finite() && chunk.northward[index].is_finite()
                },
            )?
        };
        let chunk_constituent_diagnostics = config
            .constituent_diagnostics
            .map(|options| {
                if input.is_fixed_depth() {
                    diagnose_vector_with_unavailable(
                        &worker_pool,
                        &batch,
                        &chunk.eastward,
                        &chunk.northward,
                        &latitudes,
                        &chunk.observation_counts,
                        minimum_fit_observations,
                        &chunk_solutions,
                        options,
                    )
                } else {
                    worker_pool
                        .install(|| {
                            batch.diagnose_vector_time_major(
                                &chunk.eastward,
                                &chunk.northward,
                                &latitudes,
                                &chunk_solutions,
                                options,
                            )
                        })
                        .map(|values| values.into_iter().map(Some).collect())
                }
            })
            .transpose()?;
        if chunk_diagnostics
            .iter()
            .zip(&chunk.observation_counts)
            .any(|(diagnostics, count)| diagnostics.observation_count != *count)
        {
            return Err(AppError::Invalid(
                "sampling diagnostic observation counts differ from fitted vector inputs"
                    .to_owned(),
            ));
        }
        let chunk_observation_counts = chunk.observation_counts.clone();
        drop(chunk);
        result_processing_seconds += result_start.elapsed().as_secs_f64();

        let reconstruction_start = Instant::now();
        let chunk_reconstruction = if let (Some(reconstructor), Some(filter)) =
            (reconstructor.as_ref(), config.reconstruction.as_ref())
        {
            Some(if input.is_fixed_depth() {
                reconstruct_vectors_with_unavailable(
                    &worker_pool,
                    reconstructor,
                    &chunk_solutions,
                    &latitudes,
                    filter,
                    input.modified_julian_days.len(),
                )?
            } else {
                worker_pool.install(|| {
                    reconstructor.reconstruct_many_vectors_series_major(
                        &chunk_solutions,
                        &latitudes,
                        filter,
                    )
                })?
            })
        } else {
            None
        };
        reconstruction_seconds += reconstruction_start.elapsed().as_secs_f64();

        if input.is_fixed_depth() {
            input.observation_counts[first_series..end_series]
                .copy_from_slice(&chunk_observation_counts);
        } else {
            input
                .observation_counts
                .extend_from_slice(&chunk_observation_counts);
        }
        if incremental_results {
            let result_start = Instant::now();
            let chunk_order = constituent_order_indices(
                &config.constituent_order,
                batch.constituents(),
                &chunk_frequency_cph,
                &chunk_solutions,
            )?;
            sampling_accumulator.extend(&chunk_diagnostics);
            extend_retained_vector_samples(
                &mut sample_results,
                &input,
                first_series,
                &chunk_observation_counts,
                &chunk_solutions,
                &chunk_order,
                &chunk_diagnostics,
            );
            for (frequency, _) in chunk_frequency_cph
                .iter()
                .zip(&chunk_observation_counts)
                .filter(|(_, count)| **count >= minimum_fit_observations)
            {
                if let Some(first) = &first_frequency {
                    frequency_varies_by_series |= first != frequency;
                } else {
                    first_frequency = Some(frequency.clone());
                }
            }
            for series in 0..chunk_solutions.len() {
                let order = chunk_order.row(series);
                if let Some(first) = &first_constituent_order {
                    constituent_order_varies_by_series |= first.as_slice() != order;
                } else {
                    first_constituent_order = Some(order.to_vec());
                }
            }
            result_processing_seconds += result_start.elapsed().as_secs_f64();
            let output_start = Instant::now();
            incremental_output
                .as_mut()
                .ok_or_else(|| {
                    AppError::Invalid("incremental vector output was not initialized".to_owned())
                })?
                .write_chunk(
                    first_series,
                    &chunk_observation_counts,
                    &chunk_frequency_cph,
                    &chunk_solutions,
                    &chunk_order,
                    &chunk_diagnostics,
                    chunk_constituent_diagnostics.as_deref(),
                    chunk_reconstruction.as_deref(),
                )?;
            output_seconds += output_start.elapsed().as_secs_f64();
        } else {
            series_frequency_cph.extend(chunk_frequency_cph);
            sampling_diagnostics.extend(chunk_diagnostics);
            if let (Some(all), Some(chunk)) = (
                constituent_diagnostics.as_mut(),
                chunk_constituent_diagnostics,
            ) {
                all.extend(chunk);
            }
            solutions.extend(chunk_solutions);
            if let (Some(values), Some(chunk_values)) =
                (reconstruction.as_mut(), chunk_reconstruction)
            {
                values.extend(chunk_values);
            }
        }
    }
    if let Some(reader) = pipelined_reader.take() {
        reader.finish()?;
    }
    drop(dataset);

    let inference_report = config.inference.as_ref().map(VectorInferenceConfig::report);
    let (
        sampling_summary,
        constituent_order_varies_by_series,
        frequency_varies_by_series,
        result_sha256,
        sample_results,
    ) = if incremental_results {
        let output_start = Instant::now();
        let mut output = incremental_output.take().ok_or_else(|| {
            AppError::Invalid("incremental vector output was not initialized".to_owned())
        })?;
        output.close()?;
        output_seconds += output_start.elapsed().as_secs_f64();

        let result_start = Instant::now();
        let result_sha256 = vector_result_digest_from_incremental_output(
            output.temporary_path(),
            &input,
            batch.constituents(),
            config.fit_options,
            config.phase_reference,
            config.nodal_corrections,
            &config.constituent_order,
            config.analysis_method,
            config.confidence_interval,
            config.constituent_diagnostics,
            inference_report.as_ref(),
            config.reconstruction.as_ref(),
        )?;
        let sampling_summary = sampling_accumulator.finish();
        sample_results.sort_unstable_by_key(|(series, _)| *series);
        let sample_results = sample_results
            .into_iter()
            .map(|(_, sample)| sample)
            .collect::<Vec<_>>();
        result_processing_seconds += result_start.elapsed().as_secs_f64();

        let output_start = Instant::now();
        output.install(config.overwrite, &result_sha256)?;
        output_seconds += output_start.elapsed().as_secs_f64();
        (
            sampling_summary,
            constituent_order_varies_by_series,
            frequency_varies_by_series,
            result_sha256,
            sample_results,
        )
    } else {
        let result_start = Instant::now();
        let sampling_summary = summarize_sampling(&sampling_diagnostics)?;
        let constituent_index_by_rank = constituent_order_indices(
            &config.constituent_order,
            batch.constituents(),
            &series_frequency_cph,
            &solutions,
        )?;
        let result_sha256 = vector_result_digest(
            &input,
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
            constituent_diagnostics.as_deref(),
            inference_report.as_ref(),
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
        let order_varies = constituent_index_by_rank.varies_by_series();
        let frequency_varies = series_frequency_cph
            .windows(2)
            .any(|pair| pair[0] != pair[1]);
        result_processing_seconds += result_start.elapsed().as_secs_f64();

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
                constituent_diagnostics: constituent_diagnostics.as_deref(),
                constituent_diagnostics_options: config.constituent_diagnostics,
                result_sha256: &result_sha256,
                selection: &selection,
                inference: config.inference.as_ref(),
                fit_options: config.fit_options,
                phase_reference: config.phase_reference,
                nodal_corrections: config.nodal_corrections,
                analysis_method: config.analysis_method,
                confidence_interval: config.confidence_interval,
                chunk_plan,
                input_pipeline,
                reference_time_modified_julian_day: batch.reference_time_modified_julian_day(),
                reconstruction: config
                    .reconstruction
                    .as_ref()
                    .zip(reconstruction.as_deref()),
            },
        )?;
        output_seconds += output_start.elapsed().as_secs_f64();
        (
            sampling_summary,
            order_varies,
            frequency_varies,
            result_sha256,
            sample_results,
        )
    };
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
            input.vertical_mode(),
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
        result_output: if incremental_results {
            "incremental"
        } else {
            "buffered"
        },
        time_count: input.modified_julian_days.len(),
        source_time_count: input.source_time_count,
        discarded_timestamp_count: input.discarded_timestamp_count,
        series_count: input.series_count(),
        element_count: input.element_indices.len(),
        vertical_mode: input.vertical_mode(),
        layer_indices: input.layer_indices.clone(),
        fixed_depths_meters: input.fixed_depths_meters.clone(),
        fitted_series_count: input
            .observation_counts
            .iter()
            .filter(|count| **count >= minimum_fit_observations)
            .count(),
        unavailable_series_count: input
            .observation_counts
            .iter()
            .filter(|count| **count < minimum_fit_observations)
            .count(),
        series_with_missing_observations: input
            .observation_counts
            .iter()
            .filter(|count| **count != input.modified_julian_days.len())
            .count(),
        sampling: sampling_summary,
        workers: config.workers,
        chunk_series: chunk_plan.series_per_chunk,
        chunk_count: chunk_plan.chunk_count,
        input_pipeline: if input_pipeline {
            "overlapped"
        } else {
            "sequential"
        },
        maximum_observation_buffer_bytes: chunk_plan.maximum_observation_buffer_bytes,
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
        constituent_order_varies_by_series,
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
        constituent_diagnostics: config.constituent_diagnostics.map(Into::into),
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
        frequency_varies_by_series,
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

#[allow(
    clippy::too_many_arguments,
    reason = "the batch solver needs both vector components and explicit execution options"
)]
fn solve_vector_input(
    worker_pool: &rayon::ThreadPool,
    batch: &VectorAnalysisBatch,
    eastward: &[f64],
    northward: &[f64],
    latitudes: &[f64],
    series_offset: usize,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
) -> Result<Vec<VectorSolution>, AnalysisError> {
    let stream_offset = u64::try_from(series_offset).expect("usize is representable as u64");
    worker_pool.install(|| match batch {
        VectorAnalysisBatch::Standard(batch) => match (analysis_method, confidence_interval) {
            (AnalysisMethod::Ols, ConfidenceInterval::None) => {
                batch.solve_vector_time_major_with_missing(eastward, northward, latitudes)
            }
            (AnalysisMethod::Ols, ConfidenceInterval::Linear(noise)) => batch
                .solve_vector_time_major_with_missing_and_linear_confidence(
                    eastward, northward, latitudes, noise,
                ),
            (AnalysisMethod::Ols, ConfidenceInterval::MonteCarlo { options, noise }) => batch
                .solve_vector_time_major_with_missing_and_monte_carlo_confidence_with_stream_offset(
                    eastward,
                    northward,
                    latitudes,
                    options,
                    noise,
                    stream_offset,
                ),
            (AnalysisMethod::Robust(options), ConfidenceInterval::None) => batch
                .solve_vector_time_major_with_missing_robust(
                    eastward, northward, latitudes, options,
                ),
            (AnalysisMethod::Robust(options), ConfidenceInterval::Linear(noise)) => batch
                .solve_vector_time_major_with_missing_robust_and_linear_confidence(
                    eastward, northward, latitudes, options, noise,
                ),
            (
                AnalysisMethod::Robust(robust_options),
                ConfidenceInterval::MonteCarlo {
                    options: monte_carlo_options,
                    noise,
                },
            ) => batch
                .solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence_with_stream_offset(
                    eastward,
                    northward,
                    latitudes,
                    robust_options,
                    monte_carlo_options,
                    noise,
                    stream_offset,
                ),
        },
        VectorAnalysisBatch::Inferred(batch) => match (analysis_method, confidence_interval) {
            (AnalysisMethod::Ols, ConfidenceInterval::None) => {
                batch.solve_vector_time_major_with_missing(eastward, northward, latitudes)
            }
            (AnalysisMethod::Ols, ConfidenceInterval::Linear(noise)) => batch
                .solve_vector_time_major_with_missing_and_linear_confidence(
                    eastward, northward, latitudes, noise,
                ),
            (AnalysisMethod::Ols, ConfidenceInterval::MonteCarlo { options, noise }) => batch
                .solve_vector_time_major_with_missing_and_monte_carlo_confidence_with_stream_offset(
                    eastward,
                    northward,
                    latitudes,
                    options,
                    noise,
                    stream_offset,
                ),
            (AnalysisMethod::Robust(options), ConfidenceInterval::None) => batch
                .solve_vector_time_major_with_missing_robust(
                    eastward, northward, latitudes, options,
                ),
            (AnalysisMethod::Robust(options), ConfidenceInterval::Linear(noise)) => batch
                .solve_vector_time_major_with_missing_robust_and_linear_confidence(
                    eastward, northward, latitudes, options, noise,
                ),
            (
                AnalysisMethod::Robust(robust_options),
                ConfidenceInterval::MonteCarlo {
                    options: monte_carlo_options,
                    noise,
                },
            ) => batch
                .solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence_with_stream_offset(
                    eastward,
                    northward,
                    latitudes,
                    robust_options,
                    monte_carlo_options,
                    noise,
                    stream_offset,
                ),
        },
    })
}

fn unavailable_vector_solution(
    constituent_count: usize,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
) -> VectorSolution {
    let missing = vec![f64::NAN; constituent_count];
    let confidence = (confidence_interval != ConfidenceInterval::None).then(|| missing.clone());
    VectorSolution {
        semi_major: missing.clone(),
        semi_minor: missing.clone(),
        inclination_degrees: missing.clone(),
        phase_degrees: missing.clone(),
        percent_energy: missing,
        semi_major_ci: confidence.clone(),
        semi_minor_ci: confidence.clone(),
        inclination_ci_degrees: confidence.clone(),
        phase_ci_degrees: confidence.clone(),
        signal_to_noise: confidence,
        eastward_mean: f64::NAN,
        northward_mean: f64::NAN,
        eastward_slope_per_day: f64::NAN,
        northward_slope_per_day: f64::NAN,
        reference_time_days: f64::NAN,
        robust: (analysis_method != AnalysisMethod::Ols).then_some(RobustDiagnostics {
            weights: Vec::new(),
            leverage: Vec::new(),
            iterations: 0,
            termination: RobustTermination::ExactFit,
            residual_scale: f64::NAN,
            ols_rms_residual: f64::NAN,
            rms_residual: f64::NAN,
        }),
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "fixed-depth rectangular output needs explicit controls and two deterministic stream strategies"
)]
fn solve_vector_input_with_unavailable(
    worker_pool: &rayon::ThreadPool,
    batch: &VectorAnalysisBatch,
    eastward: &[f64],
    northward: &[f64],
    latitudes: &[f64],
    observation_counts: &[usize],
    minimum_observations: usize,
    series_offset: usize,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
) -> Result<Vec<VectorSolution>, AnalysisError> {
    let series_count = latitudes.len();
    if observation_counts.len() != series_count
        || eastward.len() != northward.len()
        || !eastward.len().is_multiple_of(series_count)
    {
        return Err(AnalysisError::SamplingMaskShape {
            actual: observation_counts.len(),
            expected: series_count,
        });
    }
    if observation_counts
        .iter()
        .all(|count| *count >= minimum_observations)
    {
        return solve_vector_input(
            worker_pool,
            batch,
            eastward,
            northward,
            latitudes,
            series_offset,
            analysis_method,
            confidence_interval,
        );
    }
    let time_count = eastward.len() / series_count;
    let mut solutions = observation_counts
        .iter()
        .map(|_| None)
        .collect::<Vec<Option<VectorSolution>>>();
    for (series, count) in observation_counts.iter().copied().enumerate() {
        if count < minimum_observations {
            solutions[series] = Some(unavailable_vector_solution(
                batch.constituents().len(),
                analysis_method,
                confidence_interval,
            ));
        }
    }
    if confidence_interval.monte_carlo_options().is_none() {
        let active = observation_counts
            .iter()
            .enumerate()
            .filter_map(|(series, count)| (*count >= minimum_observations).then_some(series))
            .collect::<Vec<_>>();
        if !active.is_empty() {
            let mut active_eastward = Vec::with_capacity(time_count * active.len());
            let mut active_northward = Vec::with_capacity(time_count * active.len());
            for time in 0..time_count {
                let row = time * series_count;
                active_eastward.extend(active.iter().map(|series| eastward[row + series]));
                active_northward.extend(active.iter().map(|series| northward[row + series]));
            }
            let active_latitudes = active
                .iter()
                .map(|series| latitudes[*series])
                .collect::<Vec<_>>();
            let active_solutions = solve_vector_input(
                worker_pool,
                batch,
                &active_eastward,
                &active_northward,
                &active_latitudes,
                0,
                analysis_method,
                confidence_interval,
            )?;
            for (series, solution) in active.into_iter().zip(active_solutions) {
                solutions[series] = Some(solution);
            }
        }
        return solutions
            .into_iter()
            .map(|solution| {
                solution.ok_or(AnalysisError::InvalidSolutionShape {
                    field: "fixed-depth solution",
                    actual: 0,
                    expected: 1,
                })
            })
            .collect();
    }
    let mut first = 0;
    while first < series_count {
        if observation_counts[first] < minimum_observations {
            first += 1;
            continue;
        }
        let mut end = first + 1;
        while end < series_count && observation_counts[end] >= minimum_observations {
            end += 1;
        }
        let run_count = end - first;
        let mut run_eastward = Vec::with_capacity(time_count * run_count);
        let mut run_northward = Vec::with_capacity(time_count * run_count);
        for time in 0..time_count {
            let row = time * series_count;
            run_eastward.extend_from_slice(&eastward[row + first..row + end]);
            run_northward.extend_from_slice(&northward[row + first..row + end]);
        }
        let run_solutions = solve_vector_input(
            worker_pool,
            batch,
            &run_eastward,
            &run_northward,
            &latitudes[first..end],
            series_offset + first,
            analysis_method,
            confidence_interval,
        )?;
        for (position, solution) in (first..end).zip(run_solutions) {
            solutions[position] = Some(solution);
        }
        first = end;
    }
    solutions
        .into_iter()
        .map(|solution| {
            solution.ok_or(AnalysisError::InvalidSolutionShape {
                field: "fixed-depth solution",
                actual: 0,
                expected: 1,
            })
        })
        .collect()
}

#[allow(
    clippy::too_many_arguments,
    reason = "fixed-depth diagnostic compaction mirrors the rectangular solver path"
)]
fn diagnose_vector_with_unavailable(
    worker_pool: &rayon::ThreadPool,
    batch: &VectorAnalysisBatch,
    eastward: &[f64],
    northward: &[f64],
    latitudes: &[f64],
    observation_counts: &[usize],
    minimum_observations: usize,
    solutions: &[VectorSolution],
    options: ConstituentDiagnosticsOptions,
) -> Result<Vec<Option<ConstituentSelectionDiagnostics>>, AnalysisError> {
    let series_count = latitudes.len();
    if observation_counts.len() != series_count
        || solutions.len() != series_count
        || eastward.len() != northward.len()
        || !eastward.len().is_multiple_of(series_count)
    {
        return Err(AnalysisError::SamplingMaskShape {
            actual: observation_counts.len(),
            expected: series_count,
        });
    }
    if observation_counts
        .iter()
        .all(|count| *count >= minimum_observations)
    {
        return worker_pool
            .install(|| {
                batch.diagnose_vector_time_major(eastward, northward, latitudes, solutions, options)
            })
            .map(|values| values.into_iter().map(Some).collect());
    }
    let active = observation_counts
        .iter()
        .enumerate()
        .filter_map(|(series, count)| (*count >= minimum_observations).then_some(series))
        .collect::<Vec<_>>();
    let mut diagnostics = (0..series_count).map(|_| None).collect::<Vec<_>>();
    if active.is_empty() {
        return Ok(diagnostics);
    }
    let time_count = eastward.len() / series_count;
    let mut active_eastward = Vec::with_capacity(time_count * active.len());
    let mut active_northward = Vec::with_capacity(time_count * active.len());
    for time in 0..time_count {
        let row = time * series_count;
        active_eastward.extend(active.iter().map(|series| eastward[row + series]));
        active_northward.extend(active.iter().map(|series| northward[row + series]));
    }
    let active_latitudes = active
        .iter()
        .map(|series| latitudes[*series])
        .collect::<Vec<_>>();
    let active_solutions = active
        .iter()
        .map(|series| solutions[*series].clone())
        .collect::<Vec<_>>();
    let active_diagnostics = worker_pool.install(|| {
        batch.diagnose_vector_time_major(
            &active_eastward,
            &active_northward,
            &active_latitudes,
            &active_solutions,
            options,
        )
    })?;
    for (series, series_diagnostics) in active.into_iter().zip(active_diagnostics) {
        diagnostics[series] = Some(series_diagnostics);
    }
    Ok(diagnostics)
}

fn diagnose_vector_sampling_with_unavailable(
    worker_pool: &rayon::ThreadPool,
    plan: &rutide_core::SamplingDiagnosticsPlan,
    frequencies: &[Vec<f64>],
    eastward: &[f64],
    northward: &[f64],
    observation_counts: &[usize],
    minimum_observations: usize,
) -> Result<Vec<CoreSamplingDiagnostics>, AnalysisError> {
    let series_count = frequencies.len();
    worker_pool.install(|| {
        (0..series_count)
            .into_par_iter()
            .map(|series| {
                if observation_counts[series] < minimum_observations {
                    return Ok(CoreSamplingDiagnostics {
                        observation_count: observation_counts[series],
                        record_span_days: 0.0,
                        mean_sample_interval_hours: 0.0,
                        largest_gap_hours: 0.0,
                        residual_spectrum_method: ResidualSpectrumMethod::Fft,
                        residual_spectrum_time_count: 0,
                        spectral_band_bin_count: [0; 9],
                        spectral_band_usable_bin_count: [0; 9],
                    });
                }
                plan.diagnose_with(&frequencies[series], |time| {
                    let index = time * series_count + series;
                    eastward[index].is_finite() && northward[index].is_finite()
                })
            })
            .collect()
    })
}

fn reconstruct_vectors_with_unavailable(
    worker_pool: &rayon::ThreadPool,
    reconstructor: &GreenwichNodalReconstructor,
    solutions: &[VectorSolution],
    latitudes: &[f64],
    filter: &ReconstructionFilter,
    time_count: usize,
) -> Result<Vec<VectorReconstruction>, AnalysisError> {
    if solutions
        .iter()
        .all(|solution| solution.reference_time_days.is_finite())
    {
        return worker_pool.install(|| {
            reconstructor.reconstruct_many_vectors_series_major(solutions, latitudes, filter)
        });
    }
    let mut values = vec![
        VectorReconstruction {
            eastward: vec![f64::NAN; time_count],
            northward: vec![f64::NAN; time_count],
        };
        solutions.len()
    ];
    let active = solutions
        .iter()
        .enumerate()
        .filter_map(|(series, solution)| solution.reference_time_days.is_finite().then_some(series))
        .collect::<Vec<_>>();
    if !active.is_empty() {
        let active_solutions = active
            .iter()
            .map(|series| solutions[*series].clone())
            .collect::<Vec<_>>();
        let active_latitudes = active
            .iter()
            .map(|series| latitudes[*series])
            .collect::<Vec<_>>();
        let reconstruction = worker_pool.install(|| {
            reconstructor.reconstruct_many_vectors_series_major(
                &active_solutions,
                &active_latitudes,
                filter,
            )
        })?;
        for (series, reconstruction) in active.into_iter().zip(reconstruction) {
            values[series] = reconstruction;
        }
    }
    Ok(values)
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
        constituent_diagnostics: config.constituent_diagnostics,
        reconstruction: config.reconstruction.clone(),
        workers: config.workers,
        chunk_series: config.chunk_series,
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
    if config.layers.is_some() && config.fixed_depths_meters.is_some() {
        return Err(AppError::Invalid(
            "native sigma layers and fixed physical depths are mutually exclusive".to_owned(),
        ));
    }
    if let Some(depths) = &config.fixed_depths_meters {
        if depths.is_empty() {
            return Err(AppError::Invalid(
                "fixed-depth analysis requires at least one depth".to_owned(),
            ));
        }
        let mut bits = BTreeSet::new();
        for depth in depths {
            if !depth.is_finite() || *depth <= 0.0 {
                return Err(AppError::Invalid(
                    "fixed depths must be finite positive metres below the free surface".to_owned(),
                ));
            }
            if !bits.insert(depth.to_bits()) {
                return Err(AppError::Invalid(format!(
                    "fixed depth {depth} appears more than once"
                )));
            }
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "the profile name encodes each independent public analysis option"
)]
fn vector_profile(
    selection: &ResolvedConstituentSelection,
    vertical_mode: &str,
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
    let vertical = match vertical_mode {
        "depth-averaged" => "vector",
        "sigma-layer" => "sigma-layer-vector",
        "fixed-depth" => "fixed-depth-vector",
        _ => unreachable!("vertical modes are constructed internally"),
    };
    format!(
        "{selection}-{}-{}-{vertical}-{inference}{}{ordering}{trend}",
        phase_reference.name(),
        nodal_profile_component(nodal_corrections),
        analysis_method.name(),
    )
}

#[cfg(test)]
fn read_fvcom_vector(path: &Path, selection: &NodeSelection) -> Result<VectorInputData, AppError> {
    let dataset = netcdf::open(path)?;
    let metadata = read_fvcom_vector_metadata(path, &dataset, selection, None, None)?;
    let chunk = read_fvcom_vector_chunk(&dataset, &metadata, 0..metadata.series_count())?;
    Ok(VectorInputData {
        modified_julian_days: metadata.modified_julian_days,
        source_time_count: metadata.source_time_count,
        discarded_timestamp_count: metadata.discarded_timestamp_count,
        element_indices: metadata.element_indices,
        layer_indices: metadata.layer_indices,
        fixed_depths_meters: metadata.fixed_depths_meters,
        latitudes: metadata.latitudes,
        #[cfg(test)]
        eastward: chunk.eastward,
        #[cfg(test)]
        northward: chunk.northward,
        observation_counts: chunk.observation_counts,
        input_file_bytes: metadata.input_file_bytes,
        logical_input_bytes: metadata.logical_input_bytes,
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "one metadata pass validates both native and fixed-depth FVCOM schemas"
)]
fn read_fvcom_vector_metadata(
    path: &Path,
    dataset: &netcdf::File,
    selection: &NodeSelection,
    layer_selection: Option<&NodeSelection>,
    fixed_depths_meters: Option<&[f64]>,
) -> Result<VectorInputMetadata, AppError> {
    let input_file_bytes = fs::metadata(path)?.len();
    let time_count = required_dimension_length(dataset, "time")?;
    let element_count = required_dimension_length(dataset, "nele")?;
    let element_indices = resolve_element_selection(selection, element_count)?;
    let source_layer_count = (layer_selection.is_some() || fixed_depths_meters.is_some())
        .then(|| required_dimension_length(dataset, "siglay"))
        .transpose()?;
    let layer_indices = layer_selection
        .map(|selection| {
            let layer_count = source_layer_count.ok_or_else(|| {
                AppError::Invalid("source siglay dimension was not resolved".to_owned())
            })?;
            resolve_layer_selection(selection, layer_count)
        })
        .transpose()?;
    let fixed_depths_meters = fixed_depths_meters.map(<[f64]>::to_vec);
    let series_count = element_indices
        .len()
        .checked_mul(layer_indices.as_ref().map_or_else(
            || fixed_depths_meters.as_ref().map_or(1, Vec::len),
            Vec::len,
        ))
        .ok_or_else(|| AppError::Invalid("vector series count exceeds usize".to_owned()))?;

    let (time_axis, time_element_bytes) = read_fvcom_time_axis(dataset, time_count)?;

    let latitude_variable = required_variable(dataset, "latc")?;
    validate_dimensions(&latitude_variable, &[("nele", element_count)])?;
    let latitude_fill = latitude_variable.fill_value::<f32>()?;
    let depth_resolved = layer_indices.is_some() || fixed_depths_meters.is_some();
    let (eastward_name, northward_name) = if depth_resolved {
        ("u", "v")
    } else {
        ("ua", "va")
    };
    let eastward_variable = required_vector_variable(
        dataset,
        eastward_name,
        time_count,
        source_layer_count,
        element_count,
    )?;
    let northward_variable = required_vector_variable(
        dataset,
        northward_name,
        time_count,
        source_layer_count,
        element_count,
    )?;
    let eastward_fill = eastward_variable.fill_value::<f32>()?;
    let northward_fill = northward_variable.fill_value::<f32>()?;

    let latitude_values = read_selected_1d(&latitude_variable, &element_indices)?;

    for (series, value) in latitude_values.iter().copied().enumerate() {
        validate_source_value("latc", value, latitude_fill, series, 0)?;
    }
    let source_current_series_count = if fixed_depths_meters.is_some() {
        element_indices
            .len()
            .checked_mul(source_layer_count.unwrap_or(0))
            .ok_or_else(|| {
                AppError::Invalid("source current series count exceeds usize".to_owned())
            })?
    } else {
        series_count
    };
    let value_count = time_count
        .checked_mul(source_current_series_count)
        .ok_or_else(|| AppError::Invalid("logical input size exceeds usize".to_owned()))?;
    let mut logical_inputs = vec![
        (time_count, time_element_bytes[0]),
        (time_count, time_element_bytes[1]),
        (series_count, latitude_variable.vartype().size()),
        (value_count, eastward_variable.vartype().size()),
        (value_count, northward_variable.vartype().size()),
    ];
    let fixed_depth_source = if fixed_depths_meters.is_some() {
        let layer_count = source_layer_count
            .ok_or_else(|| AppError::Invalid("fixed-depth input requires siglay".to_owned()))?;
        if layer_count < 2 {
            return Err(AppError::Invalid(
                "fixed-depth interpolation requires at least two siglay layers".to_owned(),
            ));
        }
        let node_count = required_dimension_length(dataset, "node")?;
        let three_count = required_dimension_length(dataset, "three")?;
        if three_count != 3 {
            return Err(AppError::Invalid(format!(
                "source three dimension must have length 3, received {three_count}"
            )));
        }
        let sigma_variable = required_variable(dataset, "siglay")?;
        validate_dimensions(
            &sigma_variable,
            &[("siglay", layer_count), ("node", node_count)],
        )?;
        let bathymetry_variable = required_variable(dataset, "h")?;
        validate_dimensions(&bathymetry_variable, &[("node", node_count)])?;
        let zeta_variable = required_variable(dataset, "zeta")?;
        validate_dimensions(
            &zeta_variable,
            &[("time", time_count), ("node", node_count)],
        )?;
        let connectivity_variable = required_variable(dataset, "nv")?;
        validate_dimensions(
            &connectivity_variable,
            &[("three", 3), ("nele", element_count)],
        )?;
        let wet_cells_variable = required_variable(dataset, "wet_cells")?;
        validate_dimensions(
            &wet_cells_variable,
            &[("time", time_count), ("nele", element_count)],
        )?;

        let sigma_fill = sigma_variable.fill_value::<f32>()?;
        let bathymetry_fill = bathymetry_variable.fill_value::<f32>()?;
        let zeta_fill = zeta_variable.fill_value::<f32>()?;
        let wet_cells_fill = wet_cells_variable.fill_value::<i32>()?;
        let node_sigma = sigma_variable.get_values::<f64, _>(..)?;
        let node_bathymetry = bathymetry_variable.get_values::<f64, _>(..)?;
        let connectivity = connectivity_variable.get_values::<i32, _>(..)?;
        let mut element_nodes = Vec::with_capacity(element_indices.len());
        for &element in &element_indices {
            let mut nodes = [0; 3];
            for vertex in 0..3 {
                let one_based = connectivity[vertex * element_count + element];
                if one_based <= 0
                    || usize::try_from(one_based).map_or(true, |node| node > node_count)
                {
                    return Err(AppError::Invalid(format!(
                        "nv contains invalid one-based node {one_based} at vertex {vertex}, element {element}"
                    )));
                }
                nodes[vertex] = usize::try_from(one_based - 1).map_err(|_| {
                    AppError::Invalid("FVCOM connectivity index exceeds usize".to_owned())
                })?;
            }
            element_nodes.push(nodes);
        }
        let geometry = prepare_fixed_depth_geometry(
            &element_nodes,
            &node_sigma,
            &node_bathymetry,
            sigma_fill,
            bathymetry_fill,
            layer_count,
            node_count,
        )?;
        let selected_nodes = element_nodes
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .len();
        let selected_sigma_count = layer_count.checked_mul(selected_nodes).ok_or_else(|| {
            AppError::Invalid("logical fixed-depth geometry size exceeds usize".to_owned())
        })?;
        logical_inputs.extend([
            (selected_sigma_count, sigma_variable.vartype().size()),
            (selected_nodes, bathymetry_variable.vartype().size()),
            (
                time_count.checked_mul(selected_nodes).ok_or_else(|| {
                    AppError::Invalid("logical zeta size exceeds usize".to_owned())
                })?,
                zeta_variable.vartype().size(),
            ),
            (
                element_indices.len() * 3,
                connectivity_variable.vartype().size(),
            ),
            (
                time_count
                    .checked_mul(element_indices.len())
                    .ok_or_else(|| {
                        AppError::Invalid("logical wet-cell size exceeds usize".to_owned())
                    })?,
                wet_cells_variable.vartype().size(),
            ),
        ]);
        Some(FixedDepthSource {
            source_layer_count: layer_count,
            current_buffer_bytes_per_layer: if element_indices
                .windows(2)
                .all(|pair| pair[1] == pair[0] + 1)
            {
                [eastward_variable.vartype(), northward_variable.vartype()]
                    .iter()
                    .map(|value_type| {
                        if *value_type == NcVariableType::Float(FloatType::F32) {
                            std::mem::size_of::<f32>()
                        } else {
                            std::mem::size_of::<f64>()
                        }
                    })
                    .sum()
            } else {
                2 * std::mem::size_of::<f64>()
            },
            zeta_buffer_bytes_per_node: if zeta_variable.vartype()
                == NcVariableType::Float(FloatType::F32)
            {
                std::mem::size_of::<f32>()
            } else {
                std::mem::size_of::<f64>()
            },
            wet_buffer_bytes_per_element: if element_indices
                .windows(2)
                .all(|pair| pair[1] == pair[0] + 1)
                && wet_cells_variable.vartype() == NcVariableType::Int(IntType::I32)
            {
                // The i32 source and compact u8 destination briefly overlap.
                std::mem::size_of::<i32>() + std::mem::size_of::<u8>()
            } else {
                // The generic gather promotes to f64 before compacting to u8.
                std::mem::size_of::<f64>() + std::mem::size_of::<u8>()
            },
            element_nodes,
            layer_sigma: geometry.layer_sigma,
            element_bathymetry: geometry.element_bathymetry,
            zeta_fill,
            wet_cells_fill,
        })
    } else {
        None
    };
    let logical_input_bytes = logical_input_bytes(&logical_inputs)?;
    let source_time_count = time_axis.source_count();
    let discarded_timestamp_count = time_axis.discarded_count();
    let (modified_julian_days, retained_time_indices) = time_axis.into_parts();
    Ok(VectorInputMetadata {
        modified_julian_days,
        retained_time_indices,
        source_time_count,
        discarded_timestamp_count,
        element_indices,
        layer_indices,
        fixed_depths_meters,
        fixed_depth_source,
        latitudes: latitude_values,
        eastward_fill,
        northward_fill,
        input_file_bytes,
        logical_input_bytes,
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "FVCOM static geometry inputs and their fill values remain explicit"
)]
fn prepare_fixed_depth_geometry(
    element_nodes: &[[usize; 3]],
    node_sigma: &[f64],
    node_bathymetry: &[f64],
    sigma_fill: Option<f32>,
    bathymetry_fill: Option<f32>,
    layer_count: usize,
    node_count: usize,
) -> Result<FixedDepthStaticGeometry, AppError> {
    let geometry_count = layer_count
        .checked_mul(element_nodes.len())
        .ok_or_else(|| AppError::Invalid("fixed-depth geometry exceeds usize".to_owned()))?;
    let mut layer_sigma = vec![[f64::NAN; 3]; geometry_count];
    let mut element_bathymetry = vec![[f64::NAN; 3]; element_nodes.len()];
    let source_value = |name: &str,
                        value: f64,
                        fill: Option<f32>,
                        node: usize,
                        layer: Option<usize>|
     -> Result<Option<f64>, AppError> {
        if fill.is_some_and(|fill| value.to_bits() == f64::from(fill).to_bits()) || value.is_nan() {
            return Ok(None);
        }
        if !value.is_finite() {
            return Err(AppError::Invalid(format!(
                "{name} contains an infinite value at node {node}{}",
                layer.map_or_else(String::new, |layer| format!(", layer {layer}")),
            )));
        }
        Ok(Some(value))
    };
    for layer in 0..layer_count {
        for (element, nodes) in element_nodes.iter().copied().enumerate() {
            let geometry = layer * element_nodes.len() + element;
            for (vertex, node) in nodes.into_iter().enumerate() {
                let sigma = source_value(
                    "siglay",
                    node_sigma[layer * node_count + node],
                    sigma_fill,
                    node,
                    Some(layer),
                )?;
                let bathymetry =
                    source_value("h", node_bathymetry[node], bathymetry_fill, node, None)?;
                if let Some(sigma) = sigma {
                    layer_sigma[geometry][vertex] = sigma;
                }
                if let Some(bathymetry) = bathymetry {
                    element_bathymetry[element][vertex] = bathymetry;
                }
            }
        }
    }
    Ok(FixedDepthStaticGeometry {
        layer_sigma,
        element_bathymetry,
    })
}

fn read_fvcom_vector_chunk(
    dataset: &netcdf::File,
    metadata: &VectorInputMetadata,
    series_range: std::ops::Range<usize>,
) -> Result<VectorInputChunk, AppError> {
    if metadata.is_fixed_depth() {
        return Err(AppError::Invalid(
            "fixed-depth input must be read in element blocks".to_owned(),
        ));
    }
    if series_range.end > metadata.series_count() {
        return Err(AppError::Invalid(
            "vector chunk range exceeds selected series count".to_owned(),
        ));
    }
    let series_count = series_range.len();
    if series_count == 0 {
        return Err(AppError::Invalid("vector input chunk is empty".to_owned()));
    }
    let (mut eastward, mut northward, eastward_name, northward_name) =
        if let Some(layer_indices) = &metadata.layer_indices {
            (
                read_selected_layer_element_time_major(
                    &required_variable(dataset, "u")?,
                    metadata.source_time_count,
                    layer_indices,
                    &metadata.element_indices,
                    series_range.clone(),
                )?,
                read_selected_layer_element_time_major(
                    &required_variable(dataset, "v")?,
                    metadata.source_time_count,
                    layer_indices,
                    &metadata.element_indices,
                    series_range.clone(),
                )?,
                "u",
                "v",
            )
        } else {
            let element_indices = &metadata.element_indices[series_range.clone()];
            (
                read_selected_time_major(
                    &required_variable(dataset, "ua")?,
                    metadata.source_time_count,
                    element_indices,
                )?,
                read_selected_time_major(
                    &required_variable(dataset, "va")?,
                    metadata.source_time_count,
                    element_indices,
                )?,
                "ua",
                "va",
            )
        };
    eastward = retain_time_major_rows(
        eastward,
        metadata.source_time_count,
        series_count,
        &metadata.retained_time_indices,
    )?;
    northward = retain_time_major_rows(
        northward,
        metadata.source_time_count,
        series_count,
        &metadata.retained_time_indices,
    )?;
    let mut observation_counts = vec![0; series_count];
    for index in 0..eastward.len() {
        let series = index % series_count;
        let time = index / series_count;
        eastward[index] = normalize_source_observation(
            eastward_name,
            eastward[index],
            metadata.eastward_fill,
            series_range.start + series,
            time,
        )?;
        northward[index] = normalize_source_observation(
            northward_name,
            northward[index],
            metadata.northward_fill,
            series_range.start + series,
            time,
        )?;
        if eastward[index].is_finite() && northward[index].is_finite() {
            observation_counts[series] += 1;
        } else {
            eastward[index] = f64::NAN;
            northward[index] = f64::NAN;
        }
    }
    Ok(VectorInputChunk {
        eastward,
        northward,
        observation_counts,
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "the fixed-depth reader keeps the joint geometry/current mask auditable"
)]
fn read_fvcom_fixed_depth_element_chunk(
    dataset: &netcdf::File,
    metadata: &VectorInputMetadata,
    element_range: std::ops::Range<usize>,
    worker_pool: &rayon::ThreadPool,
) -> Result<Vec<VectorInputChunk>, AppError> {
    let depths = metadata.fixed_depths_meters.as_deref().ok_or_else(|| {
        AppError::Invalid("fixed-depth reader requires requested depths".to_owned())
    })?;
    let source = metadata.fixed_depth_source.as_ref().ok_or_else(|| {
        AppError::Invalid("fixed-depth reader requires FVCOM vertical geometry".to_owned())
    })?;
    if element_range.is_empty() || element_range.end > metadata.element_indices.len() {
        return Err(AppError::Invalid(
            "fixed-depth element chunk is empty or out of bounds".to_owned(),
        ));
    }
    let selected_elements = &metadata.element_indices[element_range.clone()];
    let source_element_nodes = &source.element_nodes[element_range.clone()];
    let selected_nodes = source_element_nodes
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut local_element_nodes = Vec::with_capacity(selected_elements.len());
    for nodes in source_element_nodes {
        let mut local = [0; 3];
        for (vertex, node) in nodes.iter().copied().enumerate() {
            local[vertex] = selected_nodes.binary_search(&node).map_err(|_| {
                AppError::Invalid("fixed-depth node map is inconsistent".to_owned())
            })?;
        }
        local_element_nodes.push(local);
    }

    let all_layers = (0..source.source_layer_count).collect::<Vec<_>>();
    let native_series_count = source
        .source_layer_count
        .checked_mul(selected_elements.len())
        .ok_or_else(|| AppError::Invalid("fixed-depth source chunk exceeds usize".to_owned()))?;
    let mut eastward = read_fixed_depth_current_block(
        &required_variable(dataset, "u")?,
        metadata.source_time_count,
        &all_layers,
        selected_elements,
        native_series_count,
    )?;
    let mut northward = read_fixed_depth_current_block(
        &required_variable(dataset, "v")?,
        metadata.source_time_count,
        &all_layers,
        selected_elements,
        native_series_count,
    )?;
    eastward = eastward.retain_time_rows(
        metadata.source_time_count,
        native_series_count,
        &metadata.retained_time_indices,
    )?;
    northward = northward.retain_time_rows(
        metadata.source_time_count,
        native_series_count,
        &metadata.retained_time_indices,
    )?;
    eastward.normalize("u", metadata.eastward_fill, native_series_count)?;
    northward.normalize("v", metadata.northward_fill, native_series_count)?;

    let zeta_variable = required_variable(dataset, "zeta")?;
    let mut zeta = read_sorted_time_major_bounded_span(
        &zeta_variable,
        metadata.source_time_count,
        &selected_nodes,
    )?;
    zeta = zeta.retain_time_rows(
        metadata.source_time_count,
        selected_nodes.len(),
        &metadata.retained_time_indices,
    )?;
    zeta.normalize("zeta", source.zeta_fill, selected_nodes.len())?;

    let wet_variable = required_variable(dataset, "wet_cells")?;
    let wet_cells = read_fixed_depth_wet_block(
        &wet_variable,
        metadata.source_time_count,
        selected_elements,
        &metadata.retained_time_indices,
        source.wet_cells_fill,
    )?;

    let element_count = selected_elements.len();
    macro_rules! interpolate {
        ($eastward:expr, $northward:expr, $zeta:expr) => {
            interpolate_fixed_depth_currents(
                worker_pool,
                depths,
                source,
                element_range.start,
                &local_element_nodes,
                selected_nodes.len(),
                $zeta,
                &wet_cells,
                $eastward,
                $northward,
                metadata.modified_julian_days.len(),
                element_count,
            )
        };
    }
    macro_rules! interpolate_with_zeta {
        ($eastward:expr, $northward:expr) => {
            match &zeta {
                FixedDepthFloatBlock::F32(zeta) => interpolate!($eastward, $northward, zeta),
                FixedDepthFloatBlock::F64(zeta) => interpolate!($eastward, $northward, zeta),
            }
        };
    }
    match (&eastward, &northward) {
        (FixedDepthFloatBlock::F32(eastward), FixedDepthFloatBlock::F32(northward)) => {
            interpolate_with_zeta!(eastward, northward)
        }
        (FixedDepthFloatBlock::F32(eastward), FixedDepthFloatBlock::F64(northward)) => {
            interpolate_with_zeta!(eastward, northward)
        }
        (FixedDepthFloatBlock::F64(eastward), FixedDepthFloatBlock::F32(northward)) => {
            interpolate_with_zeta!(eastward, northward)
        }
        (FixedDepthFloatBlock::F64(eastward), FixedDepthFloatBlock::F64(northward)) => {
            interpolate_with_zeta!(eastward, northward)
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the parallel interpolation kernel keeps physical geometry and joint-current masking together"
)]
fn interpolate_fixed_depth_currents<E, N, Z>(
    worker_pool: &rayon::ThreadPool,
    depths: &[f64],
    source: &FixedDepthSource,
    first_source_element: usize,
    local_element_nodes: &[[usize; 3]],
    selected_node_count: usize,
    zeta: &[Z],
    wet_cells: &[u8],
    source_eastward: &[E],
    source_northward: &[N],
    time_count: usize,
    element_count: usize,
) -> Result<Vec<VectorInputChunk>, AppError>
where
    E: Copy + Into<f64> + Sync,
    N: Copy + Into<f64> + Sync,
    Z: Copy + Into<f64> + Sync,
{
    struct InterpolationAccumulator {
        observation_counts: Vec<usize>,
        layer_depths: Vec<f64>,
    }

    let depth_count = depths.len();
    let row_width = depth_count
        .checked_mul(element_count)
        .ok_or_else(|| AppError::Invalid("fixed-depth output row exceeds usize".to_owned()))?;
    let output_value_count = time_count
        .checked_mul(row_width)
        .ok_or_else(|| AppError::Invalid("fixed-depth output chunk exceeds usize".to_owned()))?;
    let native_series_count = source
        .source_layer_count
        .checked_mul(element_count)
        .ok_or_else(|| AppError::Invalid("fixed-depth source row exceeds usize".to_owned()))?;
    let mut eastward = vec![f64::NAN; output_value_count];
    let mut northward = vec![f64::NAN; output_value_count];
    let accumulated = worker_pool.install(|| {
        eastward
            .par_chunks_mut(row_width)
            .zip(northward.par_chunks_mut(row_width))
            .enumerate()
            .fold(
                || InterpolationAccumulator {
                    observation_counts: vec![0; row_width],
                    layer_depths: vec![0.0; source.source_layer_count],
                },
                |mut accumulated, (time, (eastward_row, northward_row))| {
                    let zeta_row =
                        &zeta[time * selected_node_count..(time + 1) * selected_node_count];
                    let wet_row = &wet_cells[time * element_count..(time + 1) * element_count];
                    let source_row = time * native_series_count;
                    for element in 0..element_count {
                        if wet_row[element] != 1 {
                            continue;
                        }
                        let local_nodes = local_element_nodes[element];
                        let surface = [
                            zeta_row[local_nodes[0]].into(),
                            zeta_row[local_nodes[1]].into(),
                            zeta_row[local_nodes[2]].into(),
                        ];
                        if surface.iter().any(|value| !value.is_finite()) {
                            continue;
                        }
                        let source_element = first_source_element + element;
                        let bathymetry = source.element_bathymetry[source_element];
                        let mut geometry_valid = true;
                        for (layer, layer_depth) in accumulated.layer_depths.iter_mut().enumerate()
                        {
                            let geometry = layer * source.element_nodes.len() + source_element;
                            let sigma = source.layer_sigma[geometry];
                            let mut total = 0.0;
                            for vertex in 0..3 {
                                total += -sigma[vertex] * (bathymetry[vertex] + surface[vertex]);
                            }
                            *layer_depth = total / 3.0;
                            geometry_valid &= layer_depth.is_finite();
                        }
                        if !geometry_valid
                            || accumulated
                                .layer_depths
                                .windows(2)
                                .any(|pair| pair[1] <= pair[0])
                        {
                            continue;
                        }
                        for (depth_position, target) in depths.iter().copied().enumerate() {
                            let Some(lower_layer) = accumulated
                                .layer_depths
                                .windows(2)
                                .position(|pair| target >= pair[0] && target <= pair[1])
                            else {
                                continue;
                            };
                            let upper_layer = lower_layer + 1;
                            let lower_depth = accumulated.layer_depths[lower_layer];
                            let upper_depth = accumulated.layer_depths[upper_layer];
                            let lower_series = source_row + lower_layer * element_count + element;
                            let upper_series = source_row + upper_layer * element_count + element;
                            let u0 = source_eastward[lower_series].into();
                            let u1 = source_eastward[upper_series].into();
                            let v0 = source_northward[lower_series].into();
                            let v1 = source_northward[upper_series].into();
                            let interpolated = if target.to_bits() == lower_depth.to_bits() {
                                (u0.is_finite() && v0.is_finite()).then_some((u0, v0))
                            } else if target.to_bits() == upper_depth.to_bits() {
                                (u1.is_finite() && v1.is_finite()).then_some((u1, v1))
                            } else if u0.is_finite()
                                && u1.is_finite()
                                && v0.is_finite()
                                && v1.is_finite()
                            {
                                let weight = (target - lower_depth) / (upper_depth - lower_depth);
                                Some((u0 + weight * (u1 - u0), v0 + weight * (v1 - v0)))
                            } else {
                                None
                            };
                            let Some((interpolated_u, interpolated_v)) = interpolated else {
                                continue;
                            };
                            let destination = depth_position * element_count + element;
                            eastward_row[destination] = interpolated_u;
                            northward_row[destination] = interpolated_v;
                            accumulated.observation_counts[destination] += 1;
                        }
                    }
                    accumulated
                },
            )
            .reduce(
                || InterpolationAccumulator {
                    observation_counts: vec![0; row_width],
                    layer_depths: Vec::new(),
                },
                |mut left, right| {
                    for (left, right) in left
                        .observation_counts
                        .iter_mut()
                        .zip(right.observation_counts)
                    {
                        *left += right;
                    }
                    left
                },
            )
    });

    if depth_count == 1 {
        return Ok(vec![VectorInputChunk {
            eastward,
            northward,
            observation_counts: accumulated.observation_counts,
        }]);
    }
    let mut chunks = depths
        .iter()
        .map(|_| VectorInputChunk {
            eastward: vec![f64::NAN; time_count * element_count],
            northward: vec![f64::NAN; time_count * element_count],
            observation_counts: Vec::new(),
        })
        .collect::<Vec<_>>();
    for (depth_position, chunk) in chunks.iter_mut().enumerate() {
        chunk.observation_counts.extend_from_slice(
            &accumulated.observation_counts
                [depth_position * element_count..(depth_position + 1) * element_count],
        );
        for time in 0..time_count {
            let source_start = time * row_width + depth_position * element_count;
            let destination_start = time * element_count;
            chunk.eastward[destination_start..destination_start + element_count]
                .copy_from_slice(&eastward[source_start..source_start + element_count]);
            chunk.northward[destination_start..destination_start + element_count]
                .copy_from_slice(&northward[source_start..source_start + element_count]);
        }
    }
    Ok(chunks)
}

fn read_fixed_depth_current_block(
    variable: &Variable<'_>,
    time_count: usize,
    all_layers: &[usize],
    selected_elements: &[usize],
    native_series_count: usize,
) -> Result<FixedDepthFloatBlock, AppError> {
    let contiguous_elements = selected_elements
        .windows(2)
        .all(|pair| pair[1] == pair[0] + 1);
    let complete_layers = all_layers.iter().copied().eq(0..all_layers.len());
    if contiguous_elements && complete_layers {
        let extents = (
            ..,
            ..,
            selected_elements[0]..selected_elements[selected_elements.len() - 1] + 1,
        );
        let values = if variable.vartype() == NcVariableType::Float(FloatType::F32) {
            FixedDepthFloatBlock::F32(variable.get_values::<f32, _>(extents)?)
        } else {
            FixedDepthFloatBlock::F64(variable.get_values::<f64, _>(extents)?)
        };
        let expected = time_count.checked_mul(native_series_count).ok_or_else(|| {
            AppError::Invalid("fixed-depth current shape exceeds usize".to_owned())
        })?;
        if values.len() != expected {
            return Err(AppError::Invalid(format!(
                "fixed-depth current read returned {} values; expected {expected}",
                values.len()
            )));
        }
        return Ok(values);
    }
    read_selected_layer_element_time_major(
        variable,
        time_count,
        all_layers,
        selected_elements,
        0..native_series_count,
    )
    .map(FixedDepthFloatBlock::F64)
}

fn read_sorted_time_major_bounded_span(
    variable: &Variable<'_>,
    time_count: usize,
    selected_indices: &[usize],
) -> Result<FixedDepthFloatBlock, AppError> {
    if selected_indices.is_empty() || selected_indices.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(AppError::Invalid(
            "bounded-span input indices must be non-empty and increasing".to_owned(),
        ));
    }
    let source_start = selected_indices[0];
    let source_end = selected_indices[selected_indices.len() - 1]
        .checked_add(1)
        .ok_or_else(|| AppError::Invalid("bounded-span input end overflows".to_owned()))?;
    let span = source_end - source_start;
    if span
        <= selected_indices
            .len()
            .saturating_mul(FIXED_DEPTH_ZETA_SPAN_AMPLIFICATION)
    {
        if variable.vartype() == NcVariableType::Float(FloatType::F32) {
            let source = variable.get_values::<f32, _>((.., source_start..source_end))?;
            return compact_bounded_span_values(
                source,
                time_count,
                span,
                source_start,
                selected_indices,
            )
            .map(FixedDepthFloatBlock::F32);
        }
        let source = variable.get_values::<f64, _>((.., source_start..source_end))?;
        return compact_bounded_span_values(
            source,
            time_count,
            span,
            source_start,
            selected_indices,
        )
        .map(FixedDepthFloatBlock::F64);
    }
    read_selected_time_major(variable, time_count, selected_indices).map(FixedDepthFloatBlock::F64)
}

fn compact_bounded_span_values<T: Copy>(
    mut source: Vec<T>,
    time_count: usize,
    span: usize,
    source_start: usize,
    selected_indices: &[usize],
) -> Result<Vec<T>, AppError> {
    let source_value_count = time_count
        .checked_mul(span)
        .ok_or_else(|| AppError::Invalid("bounded-span source shape exceeds usize".to_owned()))?;
    if source.len() != source_value_count {
        return Err(AppError::Invalid(format!(
            "bounded-span read returned {} values; expected {source_value_count}",
            source.len()
        )));
    }
    if span == selected_indices.len() {
        return Ok(source);
    }
    let value_count = time_count
        .checked_mul(selected_indices.len())
        .ok_or_else(|| AppError::Invalid("bounded-span input shape exceeds usize".to_owned()))?;
    for time in 0..time_count {
        for (destination, index) in selected_indices.iter().copied().enumerate() {
            source[time * selected_indices.len() + destination] =
                source[time * span + index - source_start];
        }
    }
    source.truncate(value_count);
    Ok(source)
}

fn read_fixed_depth_wet_block(
    variable: &Variable<'_>,
    source_time_count: usize,
    selected_elements: &[usize],
    retained_time_indices: &[usize],
    fill_value: Option<i32>,
) -> Result<Vec<u8>, AppError> {
    let contiguous = selected_elements
        .windows(2)
        .all(|pair| pair[1] == pair[0] + 1);
    if contiguous && variable.vartype() == NcVariableType::Int(IntType::I32) {
        let values = variable.get_values::<i32, _>((
            ..,
            selected_elements[0]..selected_elements[selected_elements.len() - 1] + 1,
        ))?;
        let values = retain_time_major_rows(
            values,
            source_time_count,
            selected_elements.len(),
            retained_time_indices,
        )?;
        return values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                if fill_value == Some(value) || value == 0 {
                    Ok(0)
                } else if value == 1 {
                    Ok(1)
                } else {
                    Err(AppError::Invalid(format!(
                        "wet_cells must contain only 0, 1, or its missing value; received {value} at selected element {}, retained time {}",
                        index % selected_elements.len(),
                        index / selected_elements.len(),
                    )))
                }
            })
            .collect();
    }
    let values = read_selected_time_major(variable, source_time_count, selected_elements)?;
    let values = retain_time_major_rows(
        values,
        source_time_count,
        selected_elements.len(),
        retained_time_indices,
    )?;
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if fill_value.is_some_and(|fill| value.to_bits() == f64::from(fill).to_bits())
                || value.is_nan()
                || value.to_bits() == 0.0_f64.to_bits()
            {
                Ok(0)
            } else if value.to_bits() == 1.0_f64.to_bits() {
                Ok(1)
            } else {
                Err(AppError::Invalid(format!(
                    "wet_cells must contain only 0, 1, or its missing value; received {value} at selected element {}, retained time {}",
                    index % selected_elements.len(),
                    index / selected_elements.len(),
                )))
            }
        })
        .collect()
}

fn read_selected_layer_element_time_major(
    variable: &Variable<'_>,
    time_count: usize,
    layer_indices: &[usize],
    element_indices: &[usize],
    series_range: std::ops::Range<usize>,
) -> Result<Vec<f64>, AppError> {
    const MAX_COALESCED_ELEMENT_GAP: usize = 4096;

    struct ReadRun {
        layer: usize,
        source_start: usize,
        source_end: usize,
        selections: Vec<(usize, usize)>,
    }

    let element_count = element_indices.len();
    let series_count = series_range.len();
    let total_series = element_count
        .checked_mul(layer_indices.len())
        .ok_or_else(|| AppError::Invalid("depth-resolved series count exceeds usize".to_owned()))?;
    if series_count == 0 || series_range.end > total_series {
        return Err(AppError::Invalid(
            "depth-resolved vector chunk range is empty or out of bounds".to_owned(),
        ));
    }
    let value_count = time_count
        .checked_mul(series_count)
        .ok_or_else(|| AppError::Invalid("observation chunk shape exceeds usize".to_owned()))?;
    let mut values = vec![0.0; value_count];
    let mut pieces = Vec::new();
    let mut logical_series = series_range.start;
    while logical_series < series_range.end {
        let layer_position = logical_series / element_count;
        let first_element_position = logical_series % element_count;
        let layer_remainder = element_count - first_element_position;
        let piece_length = layer_remainder.min(series_range.end - logical_series);
        let selected =
            &element_indices[first_element_position..first_element_position + piece_length];
        let destination_piece = logical_series - series_range.start;
        pieces.push((layer_indices[layer_position], selected, destination_piece));
        logical_series += piece_length;
    }
    if pieces
        .iter()
        .all(|(_, selected, _)| selected.windows(2).all(|pair| pair[1] == pair[0] + 1))
    {
        for (layer, selected, destination_piece) in pieces {
            let source = variable.get_values::<f64, _>((
                ..,
                layer..layer + 1,
                selected[0]..selected[selected.len() - 1] + 1,
            ))?;
            for time in 0..time_count {
                let source_start = time * selected.len();
                let destination_start = time * series_count + destination_piece;
                values[destination_start..destination_start + selected.len()]
                    .copy_from_slice(&source[source_start..source_start + selected.len()]);
            }
        }
        return Ok(values);
    }

    let mut runs = Vec::new();
    for (layer, selected, destination_piece) in pieces {
        let mut ordered = selected
            .iter()
            .copied()
            .enumerate()
            .map(|(position, source)| (source, destination_piece + position))
            .collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|&(source, _)| source);
        let mut first = 0;
        while first < ordered.len() {
            let mut end = first + 1;
            while end < ordered.len()
                && ordered[end].0 - ordered[end - 1].0 <= MAX_COALESCED_ELEMENT_GAP
            {
                end += 1;
            }
            let source_start = ordered[first].0;
            let source_end = ordered[end - 1].0 + 1;
            runs.push(ReadRun {
                layer,
                source_start,
                source_end,
                selections: ordered[first..end]
                    .iter()
                    .map(|&(source, destination)| (source - source_start, destination))
                    .collect(),
            });
            first = end;
        }
    }
    runs.sort_unstable_by_key(|run| (run.layer, run.source_start));
    for time in 0..time_count {
        for run in &runs {
            let source = variable.get_values::<f64, _>((
                time..time + 1,
                run.layer..run.layer + 1,
                run.source_start..run.source_end,
            ))?;
            for &(source_position, destination_series) in &run.selections {
                values[time * series_count + destination_series] = source[source_position];
            }
        }
    }
    Ok(values)
}

fn required_vector_variable<'dataset>(
    dataset: &'dataset netcdf::File,
    name: &str,
    time_count: usize,
    layer_count: Option<usize>,
    element_count: usize,
) -> Result<Variable<'dataset>, AppError> {
    let variable = required_variable(dataset, name)?;
    if let Some(layer_count) = layer_count {
        validate_dimensions(
            &variable,
            &[
                ("time", time_count),
                ("siglay", layer_count),
                ("nele", element_count),
            ],
        )?;
    } else {
        validate_dimensions(&variable, &[("time", time_count), ("nele", element_count)])?;
    }
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

fn resolve_layer_selection(
    selection: &NodeSelection,
    layer_count: usize,
) -> Result<Vec<usize>, AppError> {
    if layer_count == 0 {
        return Err(AppError::Invalid(
            "source siglay dimension must not be empty".to_owned(),
        ));
    }
    match selection {
        NodeSelection::All => Ok((0..layer_count).collect()),
        NodeSelection::Prefix(count) => {
            if *count == 0 || *count > layer_count {
                return Err(AppError::Invalid(format!(
                    "layer prefix must be between 1 and {layer_count}, received {count}"
                )));
            }
            Ok((0..*count).collect())
        }
        NodeSelection::Indices(indices) => {
            if indices.is_empty() {
                return Err(AppError::Invalid(
                    "explicit layer selection must not be empty".to_owned(),
                ));
            }
            let mut unique = BTreeSet::new();
            for index in indices.iter().copied() {
                if index >= layer_count {
                    return Err(AppError::Invalid(format!(
                        "layer index {index} is outside source layer count {layer_count}"
                    )));
                }
                if !unique.insert(index) {
                    return Err(AppError::Invalid(format!(
                        "layer index {index} appears more than once"
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
            if !solution.reference_time_days.is_finite() {
                return Ok(vec![f64::NAN; batch.constituents().len()]);
            }
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
    input: &VectorInputData,
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
    constituent_diagnostics: Option<&[Option<ConstituentSelectionDiagnostics>]>,
    inference: Option<&InferenceReport>,
    reconstruction: Option<(&ReconstructionFilter, &[VectorReconstruction])>,
) -> Result<String, AppError> {
    let mut digest = Sha256::new();
    if input.is_fixed_depth() {
        digest.update(b"rutide-vector-fixed-depth-v13\0");
    } else if input.is_depth_resolved() {
        digest.update(b"rutide-vector-sampling-v12\0");
    } else {
        digest.update(b"rutide-vector-sampling-v10\0");
    }
    digest.update([u8::from(fit_options.trend)]);
    digest.update(phase_reference.name().as_bytes());
    digest.update([0]);
    digest.update(nodal_corrections.name().as_bytes());
    digest.update([0]);
    update_constituent_order_digest(&mut digest, constituent_order, constituent_index_by_rank);
    if input.is_fixed_depth() {
        for solution in solutions {
            digest.update([u8::from(!solution.reference_time_days.is_finite())]);
        }
    }
    update_sampling_digest(&mut digest, sampling_diagnostics)?;
    digest.update(analysis_method.name().as_bytes());
    if let AnalysisMethod::Robust(options) = analysis_method {
        update_robust_options_digest(&mut digest, options)?;
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
    update_constituent_diagnostics_digest(&mut digest, constituent_diagnostics)?;
    update_inference_digest(&mut digest, inference);
    for constituent in constituents {
        digest.update(constituent.name.as_bytes());
        digest.update([0]);
    }
    for (series, (frequency_cph, solution)) in
        series_frequency_cph.iter().zip(solutions).enumerate()
    {
        let (layer_index, depth, element_index, latitude) = input.series_coordinates(series);
        if let Some(layer) = layer_index {
            digest.update(
                u64::try_from(layer)
                    .map_err(|_| AppError::Invalid("layer index exceeds u64".to_owned()))?
                    .to_le_bytes(),
            );
        }
        if let Some(depth) = depth {
            digest.update(depth.to_bits().to_le_bytes());
        }
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

fn read_layered_series_values<T: netcdf::NcTypeDescriptor + Copy>(
    variable: &Variable<'_>,
    element_count: usize,
    first_series: usize,
    row_count: usize,
    row_width: usize,
) -> Result<Vec<T>, AppError> {
    let has_trailing_dimension = variable.dimensions().len() == 3;
    let mut values = Vec::with_capacity(row_count * row_width);
    let mut local_first = 0;
    while local_first < row_count {
        let global_first = first_series + local_first;
        let layer = global_first / element_count;
        let element = global_first % element_count;
        let rows = (row_count - local_first).min(element_count - element);
        let mut extents = vec![
            netcdf::Extent::Index(layer),
            netcdf::Extent::from(element..element + rows),
        ];
        if has_trailing_dimension {
            extents.push(netcdf::Extent::from(..));
        }
        values.extend(variable.get_values::<T, _>(extents)?);
        local_first += rows;
    }
    Ok(values)
}

fn required_output_variable<'file>(
    output: &'file netcdf::File,
    name: &str,
) -> Result<Variable<'file>, AppError> {
    output
        .variable(name)
        .ok_or_else(|| AppError::Invalid(format!("incremental output omitted {name}")))
}

fn digest_nonnegative_i64(
    digest: &mut Sha256,
    value: i64,
    description: &str,
) -> Result<(), AppError> {
    digest.update(
        u64::try_from(value)
            .map_err(|_| AppError::Invalid(format!("negative {description} in output")))?
            .to_le_bytes(),
    );
    Ok(())
}

fn read_ragged_output_rows(
    variable: &Variable<'_>,
    row_starts: &[i64],
    row_sizes: &[i64],
) -> Result<Vec<f64>, AppError> {
    if row_starts.len() != row_sizes.len() {
        return Err(AppError::Invalid(
            "robust row starts and sizes differ in length".to_owned(),
        ));
    }
    let mut values = Vec::new();
    let mut first = 0;
    while first < row_starts.len() {
        let source_start = usize::try_from(row_starts[first])
            .map_err(|_| AppError::Invalid("negative robust row start in output".to_owned()))?;
        let first_size = usize::try_from(row_sizes[first])
            .map_err(|_| AppError::Invalid("negative robust row size in output".to_owned()))?;
        let mut source_end = source_start
            .checked_add(first_size)
            .ok_or_else(|| AppError::Invalid("robust row extent overflows".to_owned()))?;
        let mut end = first + 1;
        while end < row_starts.len() {
            let next_start = usize::try_from(row_starts[end])
                .map_err(|_| AppError::Invalid("negative robust row start in output".to_owned()))?;
            if next_start != source_end {
                break;
            }
            source_end = source_end
                .checked_add(usize::try_from(row_sizes[end]).map_err(|_| {
                    AppError::Invalid("negative robust row size in output".to_owned())
                })?)
                .ok_or_else(|| AppError::Invalid("robust row extent overflows".to_owned()))?;
            end += 1;
        }
        values.extend(variable.get_values::<f64, _>(source_start..source_end)?);
        first = end;
    }
    Ok(values)
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the read-back digest deliberately mirrors the established v12 byte order"
)]
fn vector_result_digest_from_incremental_output(
    path: &Path,
    input: &VectorInputData,
    constituents: &[Constituent],
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    constituent_order: &ConstituentOrder,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
    constituent_diagnostics: Option<ConstituentDiagnosticsOptions>,
    inference: Option<&InferenceReport>,
    reconstruction: Option<&ReconstructionFilter>,
) -> Result<String, AppError> {
    const DIGEST_SERIES_CHUNK: usize = 8192;
    const TARGET_RECONSTRUCTION_READ_BYTES: usize = 64 * 1024 * 1024;

    if !input.is_depth_resolved() {
        return Err(AppError::Invalid(
            "incremental vector digest requires a resolved vertical coordinate".to_owned(),
        ));
    }
    let output = netcdf::open(path)?;
    let element_count = input.element_indices.len();
    let series_count = input.series_count();
    let constituent_count = constituents.len();
    let mut digest = Sha256::new();
    if input.is_fixed_depth() {
        digest.update(b"rutide-vector-fixed-depth-v13\0");
    } else {
        digest.update(b"rutide-vector-sampling-v12\0");
    }
    digest.update([u8::from(fit_options.trend)]);
    digest.update(phase_reference.name().as_bytes());
    digest.update([0]);
    digest.update(nodal_corrections.name().as_bytes());
    digest.update([0]);
    digest.update(constituent_order.name().as_bytes());
    digest.update([0]);
    if let Some(names) = constituent_order.explicit_names() {
        for name in names {
            digest.update(name.as_bytes());
            digest.update([0]);
        }
    }
    let order_variable = required_output_variable(&output, "constituent_index_by_rank")?;
    for first_series in (0..series_count).step_by(DIGEST_SERIES_CHUNK) {
        let rows = (series_count - first_series).min(DIGEST_SERIES_CHUNK);
        let indices = read_layered_series_values::<i64>(
            &order_variable,
            element_count,
            first_series,
            rows,
            constituent_count,
        )?;
        for index in indices {
            digest_nonnegative_i64(&mut digest, index, "constituent index")?;
        }
    }

    let analysis_status = required_output_variable(&output, "analysis_status")?;
    let observation_count = required_output_variable(&output, "observation_count")?;
    let record_span = required_output_variable(&output, "sampling_record_span")?;
    let mean_interval = required_output_variable(&output, "sampling_mean_interval")?;
    let largest_gap = required_output_variable(&output, "sampling_largest_gap")?;
    let spectrum_method = required_output_variable(&output, "residual_spectrum_method")?;
    let spectrum_time_count = required_output_variable(&output, "residual_spectrum_time_count")?;
    let band_count = required_output_variable(&output, "spectral_band_frequency_bin_count")?;
    let usable_band_count = required_output_variable(&output, "spectral_band_usable_bin_count")?;
    for first_series in (0..series_count).step_by(DIGEST_SERIES_CHUNK) {
        let rows = (series_count - first_series).min(DIGEST_SERIES_CHUNK);
        let statuses = read_layered_series_values::<i64>(
            &analysis_status,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let observations = read_layered_series_values::<i64>(
            &observation_count,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let spans =
            read_layered_series_values::<f64>(&record_span, element_count, first_series, rows, 1)?;
        let intervals = read_layered_series_values::<f64>(
            &mean_interval,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let gaps =
            read_layered_series_values::<f64>(&largest_gap, element_count, first_series, rows, 1)?;
        let methods = read_layered_series_values::<i64>(
            &spectrum_method,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let spectrum_counts = read_layered_series_values::<i64>(
            &spectrum_time_count,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let bands =
            read_layered_series_values::<i64>(&band_count, element_count, first_series, rows, 9)?;
        let usable_bands = read_layered_series_values::<i64>(
            &usable_band_count,
            element_count,
            first_series,
            rows,
            9,
        )?;
        for series in 0..rows {
            if input.is_fixed_depth() {
                digest.update([match statuses[series] {
                    0 => 0,
                    1 => 1,
                    value => {
                        return Err(AppError::Invalid(format!(
                            "invalid analysis status {value} in output"
                        )));
                    }
                }]);
            }
            digest_nonnegative_i64(&mut digest, observations[series], "observation count")?;
            digest.update(spans[series].to_bits().to_le_bytes());
            digest.update(intervals[series].to_bits().to_le_bytes());
            digest.update(gaps[series].to_bits().to_le_bytes());
            digest.update([match methods[series] {
                0 => 0,
                1 => 1,
                value => {
                    return Err(AppError::Invalid(format!(
                        "invalid residual spectrum method {value} in output"
                    )));
                }
            }]);
            digest_nonnegative_i64(
                &mut digest,
                spectrum_counts[series],
                "residual spectrum time count",
            )?;
            let band_start = series * 9;
            for count in bands[band_start..band_start + 9]
                .iter()
                .chain(&usable_bands[band_start..band_start + 9])
            {
                digest_nonnegative_i64(&mut digest, *count, "spectral bin count")?;
            }
        }
    }

    digest.update(analysis_method.name().as_bytes());
    if let AnalysisMethod::Robust(options) = analysis_method {
        update_robust_options_digest(&mut digest, options)?;
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
    if let Some(options) = constituent_diagnostics {
        digest.update(b"constituent-diagnostics-v1\0");
        let scalar_variables = [
            "diagnostic_basis_condition_number",
            "diagnostic_all_constituent_signal_to_noise",
            "diagnostic_condition_adjusted_signal_to_noise",
            "diagnostic_raw_tidal_variance",
            "diagnostic_all_constituent_tidal_variance",
            "diagnostic_significant_constituent_tidal_variance",
            "diagnostic_all_constituent_percent_tidal_variance",
            "diagnostic_significant_constituent_percent_tidal_variance",
        ]
        .into_iter()
        .map(|name| required_output_variable(&output, name))
        .collect::<Result<Vec<_>, _>>()?;
        let lower_index = required_output_variable(&output, "diagnostic_lower_neighbor_index")?;
        let higher_index = required_output_variable(&output, "diagnostic_higher_neighbor_index")?;
        let lower_variables = [
            "diagnostic_lower_rayleigh_criterion",
            "diagnostic_lower_noise_modified_rayleigh_criterion",
            "diagnostic_lower_maximum_correlation",
        ]
        .into_iter()
        .map(|name| required_output_variable(&output, name))
        .collect::<Result<Vec<_>, _>>()?;
        let higher_variables = [
            "diagnostic_higher_rayleigh_criterion",
            "diagnostic_higher_noise_modified_rayleigh_criterion",
            "diagnostic_higher_maximum_correlation",
        ]
        .into_iter()
        .map(|name| required_output_variable(&output, name))
        .collect::<Result<Vec<_>, _>>()?;
        for first_series in (0..series_count).step_by(DIGEST_SERIES_CHUNK) {
            let rows = (series_count - first_series).min(DIGEST_SERIES_CHUNK);
            let statuses = read_layered_series_values::<i64>(
                &analysis_status,
                element_count,
                first_series,
                rows,
                1,
            )?;
            let scalar_values = scalar_variables
                .iter()
                .map(|variable| {
                    read_layered_series_values::<f64>(
                        variable,
                        element_count,
                        first_series,
                        rows,
                        1,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let neighbor_indices = [&lower_index, &higher_index]
                .into_iter()
                .map(|variable| {
                    read_layered_series_values::<i64>(
                        variable,
                        element_count,
                        first_series,
                        rows,
                        constituent_count,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let neighbor_values = [&lower_variables, &higher_variables]
                .into_iter()
                .map(|variables| {
                    variables
                        .iter()
                        .map(|variable| {
                            read_layered_series_values::<f64>(
                                variable,
                                element_count,
                                first_series,
                                rows,
                                constituent_count,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            for series in 0..rows {
                if input.is_fixed_depth() && statuses[series] == 1 {
                    digest.update([0]);
                    continue;
                }
                digest.update([1]);
                digest.update(options.rayleigh_minimum().to_bits().to_le_bytes());
                digest.update(options.minimum_signal_to_noise().to_bits().to_le_bytes());
                for values in &scalar_values {
                    digest.update(values[series].to_bits().to_le_bytes());
                }
                for constituent in 0..constituent_count {
                    let index = series * constituent_count + constituent;
                    for direction in 0..2 {
                        digest.update(neighbor_indices[direction][index].to_le_bytes());
                        for values in &neighbor_values[direction] {
                            digest.update(values[index].to_bits().to_le_bytes());
                        }
                    }
                }
            }
        }
    }
    update_inference_digest(&mut digest, inference);
    for constituent in constituents {
        digest.update(constituent.name.as_bytes());
        digest.update([0]);
    }

    let reference_time = required_output_variable(&output, "reference_time")?;
    let frequency = required_output_variable(&output, "frequency")?;
    let semi_major = required_output_variable(&output, "semi_major")?;
    let semi_minor = required_output_variable(&output, "semi_minor")?;
    let inclination = required_output_variable(&output, "inclination")?;
    let phase = required_output_variable(&output, "phase")?;
    let percent_energy = required_output_variable(&output, "percent_energy")?;
    let confidence_variables = if confidence_interval == ConfidenceInterval::None {
        None
    } else {
        Some([
            required_output_variable(&output, "semi_major_ci")?,
            required_output_variable(&output, "semi_minor_ci")?,
            required_output_variable(&output, "inclination_ci")?,
            required_output_variable(&output, "phase_ci")?,
            required_output_variable(&output, "signal_to_noise")?,
        ])
    };
    let eastward_mean = required_output_variable(&output, "eastward_mean")?;
    let northward_mean = required_output_variable(&output, "northward_mean")?;
    let eastward_slope = required_output_variable(&output, "eastward_slope")?;
    let northward_slope = required_output_variable(&output, "northward_slope")?;
    let robust_variables = if analysis_method == AnalysisMethod::Ols {
        None
    } else {
        Some((
            required_output_variable(&output, "robust_weight_row_start")?,
            required_output_variable(&output, "robust_weight_row_size")?,
            required_output_variable(&output, "robust_iterations")?,
            required_output_variable(&output, "robust_termination")?,
            required_output_variable(&output, "robust_residual_scale")?,
            required_output_variable(&output, "robust_ols_rms_residual")?,
            required_output_variable(&output, "robust_rms_residual")?,
            required_output_variable(&output, "robust_weight")?,
            required_output_variable(&output, "robust_leverage")?,
        ))
    };
    for first_series in (0..series_count).step_by(DIGEST_SERIES_CHUNK) {
        let rows = (series_count - first_series).min(DIGEST_SERIES_CHUNK);
        let reference_times = read_layered_series_values::<f64>(
            &reference_time,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let matrices = [
            read_layered_series_values::<f64>(
                &frequency,
                element_count,
                first_series,
                rows,
                constituent_count,
            )?,
            read_layered_series_values::<f64>(
                &semi_major,
                element_count,
                first_series,
                rows,
                constituent_count,
            )?,
            read_layered_series_values::<f64>(
                &semi_minor,
                element_count,
                first_series,
                rows,
                constituent_count,
            )?,
            read_layered_series_values::<f64>(
                &inclination,
                element_count,
                first_series,
                rows,
                constituent_count,
            )?,
            read_layered_series_values::<f64>(
                &phase,
                element_count,
                first_series,
                rows,
                constituent_count,
            )?,
            read_layered_series_values::<f64>(
                &percent_energy,
                element_count,
                first_series,
                rows,
                constituent_count,
            )?,
        ];
        let confidence_matrices = confidence_variables
            .as_ref()
            .map(|variables| {
                variables
                    .iter()
                    .map(|variable| {
                        read_layered_series_values::<f64>(
                            variable,
                            element_count,
                            first_series,
                            rows,
                            constituent_count,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let eastward_means = read_layered_series_values::<f64>(
            &eastward_mean,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let northward_means = read_layered_series_values::<f64>(
            &northward_mean,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let eastward_slopes = read_layered_series_values::<f64>(
            &eastward_slope,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let northward_slopes = read_layered_series_values::<f64>(
            &northward_slope,
            element_count,
            first_series,
            rows,
            1,
        )?;
        let robust_chunk = robust_variables
            .as_ref()
            .map(
                |(
                    row_start,
                    row_size,
                    iterations,
                    termination,
                    residual_scale,
                    ols_rms,
                    rms,
                    weights,
                    leverage,
                )| {
                    let row_starts = read_layered_series_values::<i64>(
                        row_start,
                        element_count,
                        first_series,
                        rows,
                        1,
                    )?;
                    let row_sizes = read_layered_series_values::<i64>(
                        row_size,
                        element_count,
                        first_series,
                        rows,
                        1,
                    )?;
                    let weight_values = read_ragged_output_rows(weights, &row_starts, &row_sizes)?;
                    let leverage_values =
                        read_ragged_output_rows(leverage, &row_starts, &row_sizes)?;
                    Ok::<_, AppError>((
                        row_sizes,
                        read_layered_series_values::<i64>(
                            iterations,
                            element_count,
                            first_series,
                            rows,
                            1,
                        )?,
                        read_layered_series_values::<i64>(
                            termination,
                            element_count,
                            first_series,
                            rows,
                            1,
                        )?,
                        read_layered_series_values::<f64>(
                            residual_scale,
                            element_count,
                            first_series,
                            rows,
                            1,
                        )?,
                        read_layered_series_values::<f64>(
                            ols_rms,
                            element_count,
                            first_series,
                            rows,
                            1,
                        )?,
                        read_layered_series_values::<f64>(
                            rms,
                            element_count,
                            first_series,
                            rows,
                            1,
                        )?,
                        weight_values,
                        leverage_values,
                    ))
                },
            )
            .transpose()?;
        let mut robust_local_offset = 0;
        for local_series in 0..rows {
            let series = first_series + local_series;
            let (layer_index, depth, element_index, latitude) = input.series_coordinates(series);
            if let Some(layer) = layer_index {
                digest.update(
                    u64::try_from(layer)
                        .map_err(|_| AppError::Invalid("layer index exceeds u64".to_owned()))?
                        .to_le_bytes(),
                );
            }
            if let Some(depth) = depth {
                digest.update(depth.to_bits().to_le_bytes());
            }
            digest.update(
                u64::try_from(element_index)
                    .map_err(|_| AppError::Invalid("element index exceeds u64".to_owned()))?
                    .to_le_bytes(),
            );
            digest.update(latitude.to_bits().to_le_bytes());
            digest.update(reference_times[local_series].to_bits().to_le_bytes());
            let start = local_series * constituent_count;
            let end = start + constituent_count;
            for matrix in &matrices {
                for value in &matrix[start..end] {
                    digest.update(value.to_bits().to_le_bytes());
                }
            }
            if let Some(confidence_matrices) = &confidence_matrices {
                for matrix in confidence_matrices {
                    for value in &matrix[start..end] {
                        digest.update(value.to_bits().to_le_bytes());
                    }
                }
            }
            for value in [
                eastward_means[local_series],
                northward_means[local_series],
                eastward_slopes[local_series],
                northward_slopes[local_series],
            ] {
                digest.update(value.to_bits().to_le_bytes());
            }
            if let Some((
                row_sizes,
                iterations,
                termination,
                residual_scales,
                ols_rms,
                rms,
                weights,
                leverage,
            )) = &robust_chunk
            {
                digest_nonnegative_i64(
                    &mut digest,
                    iterations[local_series],
                    "robust iteration count",
                )?;
                digest.update(residual_scales[local_series].to_bits().to_le_bytes());
                digest.update(ols_rms[local_series].to_bits().to_le_bytes());
                digest.update(rms[local_series].to_bits().to_le_bytes());
                digest.update(termination[local_series].to_le_bytes());
                let row_size = usize::try_from(row_sizes[local_series]).map_err(|_| {
                    AppError::Invalid("negative robust row size in output".to_owned())
                })?;
                let row_end = robust_local_offset + row_size;
                for values in [weights, leverage] {
                    for value in &values[robust_local_offset..row_end] {
                        digest.update(value.to_bits().to_le_bytes());
                    }
                }
                robust_local_offset = row_end;
            }
        }
    }

    if let Some(filter) = reconstruction {
        update_reconstruction_filter_digest(&mut digest, filter);
        let time_count = input.modified_julian_days.len();
        let rows_per_block = (TARGET_RECONSTRUCTION_READ_BYTES
            / (time_count.max(1) * std::mem::size_of::<f64>()))
        .max(1);
        let eastward = required_output_variable(&output, "eastward_reconstruction")?;
        let northward = required_output_variable(&output, "northward_reconstruction")?;
        let mut first_series = 0;
        while first_series < series_count {
            let layer = first_series / element_count;
            let element = first_series % element_count;
            let rows = (series_count - first_series)
                .min(element_count - element)
                .min(rows_per_block);
            let eastward_values =
                eastward.get_values::<f64, _>((.., layer, element..element + rows))?;
            let northward_values =
                northward.get_values::<f64, _>((.., layer, element..element + rows))?;
            for series in 0..rows {
                for values in [&eastward_values, &northward_values] {
                    for time in 0..time_count {
                        digest.update(values[time * rows + series].to_bits().to_le_bytes());
                    }
                }
            }
            first_series += rows;
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
        .observation_counts
        .iter()
        .copied()
        .zip(solutions)
        .enumerate()
        .take(3)
        .map(|(series, (observation_count, solution))| {
            let (layer_index, depth_meters_below_surface, element_index, latitude_degrees_north) =
                input.series_coordinates(series);
            VectorSampleResult {
                element_index,
                layer_index,
                depth_meters_below_surface,
                analysis_status: if solution.reference_time_days.is_finite() {
                    "fitted"
                } else {
                    "unavailable"
                },
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
        })
        .collect()
}

fn extend_retained_vector_samples(
    retained: &mut Vec<(usize, VectorSampleResult)>,
    input: &VectorInputData,
    first_series: usize,
    observation_counts: &[usize],
    solutions: &[VectorSolution],
    constituent_index_by_rank: &ConstituentOrderMap,
    sampling_diagnostics: &[CoreSamplingDiagnostics],
) {
    for local_series in 0..solutions.len() {
        let series = first_series + local_series;
        if series >= 3 {
            continue;
        }
        let solution = &solutions[local_series];
        let (layer_index, depth_meters_below_surface, element_index, latitude_degrees_north) =
            input.series_coordinates(series);
        retained.push((
            series,
            VectorSampleResult {
                element_index,
                layer_index,
                depth_meters_below_surface,
                analysis_status: if solution.reference_time_days.is_finite() {
                    "fitted"
                } else {
                    "unavailable"
                },
                latitude_degrees_north,
                semi_major: solution.semi_major.clone(),
                semi_minor: solution.semi_minor.clone(),
                inclination_degrees: solution.inclination_degrees.clone(),
                phase_degrees: solution.phase_degrees.clone(),
                percent_energy: solution.percent_energy.clone(),
                constituent_index_by_rank: constituent_index_by_rank
                    .row(local_series)
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
                observation_count: observation_counts[local_series],
                sampling: SeriesSamplingDiagnostics::from(&sampling_diagnostics[local_series]),
                reference_time_modified_julian_day: solution.reference_time_days,
            },
        ));
    }
}

struct VectorOutputData<'data> {
    input: &'data VectorInputData,
    constituents: &'data [Constituent],
    series_frequency_cph: &'data [Vec<f64>],
    solutions: &'data [VectorSolution],
    constituent_order: &'data ConstituentOrder,
    constituent_index_by_rank: &'data ConstituentOrderMap,
    sampling_diagnostics: &'data [CoreSamplingDiagnostics],
    constituent_diagnostics: Option<&'data [Option<ConstituentSelectionDiagnostics>]>,
    constituent_diagnostics_options: Option<ConstituentDiagnosticsOptions>,
    result_sha256: &'data str,
    selection: &'data ResolvedConstituentSelection,
    inference: Option<&'data VectorInferenceConfig>,
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
    chunk_plan: super::SpatialChunkPlan,
    input_pipeline: bool,
    reference_time_modified_julian_day: f64,
    reconstruction: Option<(&'data ReconstructionFilter, &'data [VectorReconstruction])>,
}

struct VectorOutputDefinition<'data> {
    input: &'data VectorInputData,
    constituents: &'data [Constituent],
    constituent_order: &'data ConstituentOrder,
    selection: &'data ResolvedConstituentSelection,
    inference: Option<&'data VectorInferenceConfig>,
    fit_options: FitOptions,
    phase_reference: PhaseReference,
    nodal_corrections: NodalCorrections,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
    constituent_diagnostics: Option<ConstituentDiagnosticsOptions>,
    chunk_plan: super::SpatialChunkPlan,
    input_pipeline: bool,
    reference_time_modified_julian_day: f64,
    reconstruction: Option<&'data ReconstructionFilter>,
    result_output: &'static str,
}

#[allow(
    clippy::too_many_lines,
    reason = "the vector NetCDF metadata contract is kept together"
)]
fn create_vector_output_base(
    path: &Path,
    data: &VectorOutputDefinition<'_>,
) -> Result<FileMut, AppError> {
    let input = data.input;
    let mut output = netcdf::create(path)?;
    if let Some(depths) = &input.fixed_depths_meters {
        output.add_dimension("depth", depths.len())?;
        output.add_dimension("element", input.element_indices.len())?;
    } else if let Some(layers) = &input.layer_indices {
        output.add_dimension("siglay", layers.len())?;
        output.add_dimension("element", input.element_indices.len())?;
    } else {
        output.add_dimension("series", input.element_indices.len())?;
    }
    output.add_dimension("constituent", data.constituents.len())?;
    output.add_dimension("presentation_rank", data.constituents.len())?;
    output.add_dimension(
        "spectral_band",
        rutide_core::COLORED_NOISE_FREQUENCY_BANDS_CPH.len(),
    )?;
    if data.reconstruction.is_some() {
        output.add_dimension("time", input.modified_julian_days.len())?;
    }
    output.add_attribute(
        "title",
        if input.is_fixed_depth() {
            "RUTide fixed-physical-depth current ellipses"
        } else if input.is_depth_resolved() {
            "RUTide native sigma-layer current ellipses"
        } else {
            "RUTide depth-averaged current ellipses"
        },
    )?;
    output.add_attribute(
        "rutide_schema_version",
        i64::from(VECTOR_OUTPUT_SCHEMA_VERSION),
    )?;
    output.add_attribute("rutide_version", rutide_core::VERSION)?;
    output.add_attribute("result_output", data.result_output)?;
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
    output.add_attribute(
        "chunk_series",
        u64::try_from(data.chunk_plan.series_per_chunk)
            .map_err(|_| AppError::Invalid("chunk series count exceeds u64".to_owned()))?,
    )?;
    output.add_attribute(
        "chunk_count",
        u64::try_from(data.chunk_plan.chunk_count)
            .map_err(|_| AppError::Invalid("chunk count exceeds u64".to_owned()))?,
    )?;
    output.add_attribute(
        "maximum_observation_buffer_bytes",
        data.chunk_plan.maximum_observation_buffer_bytes,
    )?;
    output.add_attribute(
        "input_pipeline",
        if data.input_pipeline {
            "overlapped"
        } else {
            "sequential"
        },
    )?;
    let profile = vector_profile(
        data.selection,
        input.vertical_mode(),
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
        output.add_attribute("robust_weight_function", options.weight_function.name())?;
        output.add_attribute("robust_tuning_constant", options.tuning_constant)?;
        output.add_attribute("robust_tolerance", options.tolerance)?;
        output.add_attribute(
            "robust_max_iterations",
            i64::try_from(options.max_iterations)
                .map_err(|_| AppError::Invalid("robust iteration limit exceeds i64".to_owned()))?,
        )?;
    }
    output.add_attribute("vertical_mode", input.vertical_mode())?;
    output.add_attribute(
        "source_eastward_variable",
        if input.is_depth_resolved() { "u" } else { "ua" },
    )?;
    output.add_attribute(
        "source_northward_variable",
        if input.is_depth_resolved() { "v" } else { "va" },
    )?;
    if input.is_fixed_depth() {
        output.add_attribute("source_vertical_dimension", "siglay")?;
        output.add_attribute("vertical_interpolation", "linear-layer-centres")?;
        output.add_attribute("vertical_extrapolation", "none")?;
        output.add_attribute("vertical_reference", "instantaneous-free-surface")?;
        output.add_attribute("wet_dry_mask", "wet_cells")?;
        output.add_attribute(
            "vertical_coordinate_note",
            "positive metres below the instantaneous free surface at FVCOM element centroids",
        )?;
    } else if input.is_depth_resolved() {
        output.add_attribute("source_vertical_dimension", "siglay")?;
        output.add_attribute(
            "vertical_coordinate_note",
            "native FVCOM layer indices; physical depth varies with bathymetry and free surface",
        )?;
    }
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
    write_constituent_diagnostics_attributes(&mut output, data.constituent_diagnostics)?;
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

    let element_indices = input
        .element_indices
        .iter()
        .copied()
        .map(|index| {
            i64::try_from(index)
                .map_err(|_| AppError::Invalid("element index exceeds i64".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let coordinate_dimension = if input.is_depth_resolved() {
        "element"
    } else {
        "series"
    };
    write_variable(
        &mut output.add_variable::<i64>("element_index", &[coordinate_dimension])?,
        &element_indices,
        "1",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("latitude", &[coordinate_dimension])?,
        &input.latitudes,
        "degrees_north",
    )?;
    if let Some(layers) = &input.layer_indices {
        let layers = layers
            .iter()
            .copied()
            .map(|index| {
                i64::try_from(index)
                    .map_err(|_| AppError::Invalid("layer index exceeds i64".to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        write_variable(
            &mut output.add_variable::<i64>("siglay_index", &["siglay"])?,
            &layers,
            "1",
        )?;
    }
    if let Some(depths) = &input.fixed_depths_meters {
        let mut variable = output.add_variable::<f64>("depth", &["depth"])?;
        variable.put_attribute("units", "m")?;
        variable.put_attribute("positive", "down")?;
        variable.put_attribute("long_name", "depth below instantaneous free surface")?;
        variable.put_values(depths, ..)?;
    }
    Ok(output)
}

struct IncrementalVectorOutput {
    destination: PathBuf,
    temporary: PathBuf,
    output: Option<FileMut>,
    element_count: usize,
    time_count: usize,
    constituent_count: usize,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
    robust_observation_offset: usize,
    installed: bool,
}

impl IncrementalVectorOutput {
    fn create(destination: &Path, data: &VectorOutputDefinition<'_>) -> Result<Self, AppError> {
        if !data.input.is_depth_resolved() {
            return Err(AppError::Invalid(
                "incremental vector output requires a resolved vertical coordinate".to_owned(),
            ));
        }
        let temporary = temporary_sibling(destination)?;
        let output = match create_vector_output_base(&temporary, data) {
            Ok(output) => output,
            Err(error) => {
                let _ignored = fs::remove_file(&temporary);
                return Err(error);
            }
        };
        let mut sink = Self {
            destination: destination.to_owned(),
            temporary,
            output: Some(output),
            element_count: data.input.element_indices.len(),
            time_count: data.input.modified_julian_days.len(),
            constituent_count: data.constituents.len(),
            analysis_method: data.analysis_method,
            confidence_interval: data.confidence_interval,
            robust_observation_offset: 0,
            installed: false,
        };
        let output = sink.output.as_mut().ok_or_else(|| {
            AppError::Invalid("incremental output closed during initialization".to_owned())
        })?;
        define_incremental_vector_variables(output, data)?;
        Ok(sink)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "chunk-local arrays remain explicit at the transactional sink boundary"
    )]
    fn write_chunk(
        &mut self,
        first_series: usize,
        observation_counts: &[usize],
        frequencies: &[Vec<f64>],
        solutions: &[VectorSolution],
        constituent_order: &ConstituentOrderMap,
        sampling: &[CoreSamplingDiagnostics],
        constituent_diagnostics: Option<&[Option<ConstituentSelectionDiagnostics>]>,
        reconstruction: Option<&[VectorReconstruction]>,
    ) -> Result<(), AppError> {
        let output = self
            .output
            .as_mut()
            .ok_or_else(|| AppError::Invalid("incremental output is already closed".to_owned()))?;
        write_incremental_vector_chunk(
            output,
            self.element_count,
            self.time_count,
            self.constituent_count,
            self.analysis_method,
            self.confidence_interval,
            first_series,
            observation_counts,
            frequencies,
            solutions,
            constituent_order,
            sampling,
            constituent_diagnostics,
            reconstruction,
            &mut self.robust_observation_offset,
        )
    }

    fn close(&mut self) -> Result<(), AppError> {
        if let Some(output) = self.output.take() {
            output.close()?;
        }
        Ok(())
    }

    fn temporary_path(&self) -> &Path {
        &self.temporary
    }

    fn install(mut self, overwrite: bool, result_sha256: &str) -> Result<(), AppError> {
        self.close()?;
        let mut output = netcdf::append(&self.temporary)?;
        output.add_attribute("result_sha256", result_sha256)?;
        output.close()?;
        if self.destination.exists() && !overwrite {
            return Err(AppError::DestinationExists(self.destination.clone()));
        }
        fs::rename(&self.temporary, &self.destination)?;
        self.installed = true;
        Ok(())
    }
}

impl Drop for IncrementalVectorOutput {
    fn drop(&mut self) {
        if !self.installed {
            let _ignored = self.output.take().and_then(|output| output.close().ok());
            let _ignored = fs::remove_file(&self.temporary);
        }
    }
}

fn add_variable_with_units<T: netcdf::NcTypeDescriptor>(
    output: &mut FileMut,
    name: &str,
    dimensions: &[&str],
    units: &str,
) -> Result<(), AppError> {
    output
        .add_variable::<T>(name, dimensions)?
        .put_attribute("units", units)?;
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "the incremental schema mirrors the buffered vector schema"
)]
fn define_incremental_vector_variables(
    output: &mut FileMut,
    data: &VectorOutputDefinition<'_>,
) -> Result<(), AppError> {
    let series_dimensions = data.input.series_dimensions();
    let solution_dimensions = data.input.solution_dimensions();
    add_variable_with_units::<i64>(output, "observation_count", series_dimensions, "1")?;
    let mut status = output.add_variable::<i64>("analysis_status", series_dimensions)?;
    status.put_attribute("units", "1")?;
    status.put_attribute("flag_values", vec![0_i64, 1])?;
    status.put_attribute("flag_meanings", "fitted unavailable")?;
    add_variable_with_units::<f64>(
        output,
        "reference_time",
        series_dimensions,
        "days since 1858-11-17 00:00:00 UTC",
    )?;
    add_variable_with_units::<f64>(output, "frequency", &solution_dimensions, "cycles per hour")?;
    let mut order_dimensions = series_dimensions.to_vec();
    order_dimensions.push("presentation_rank");
    let mut order = output.add_variable::<i64>("constituent_index_by_rank", &order_dimensions)?;
    order.put_attribute(
        "long_name",
        "stable constituent index at each requested presentation rank",
    )?;
    order.put_attribute("start_index", 0_i64)?;

    let lower = COLORED_NOISE_FREQUENCY_BANDS_CPH.map(|band| band[0]);
    let upper = COLORED_NOISE_FREQUENCY_BANDS_CPH.map(|band| band[1]);
    write_variable(
        &mut output.add_variable::<f64>("spectral_band_lower_frequency", &["spectral_band"])?,
        &lower,
        "cycles per hour",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("spectral_band_upper_frequency", &["spectral_band"])?,
        &upper,
        "cycles per hour",
    )?;
    for (name, units) in [
        ("sampling_record_span", "days"),
        ("sampling_mean_interval", "hours"),
        ("sampling_largest_gap", "hours"),
    ] {
        add_variable_with_units::<f64>(output, name, series_dimensions, units)?;
    }
    let mut spectrum_method =
        output.add_variable::<i64>("residual_spectrum_method", series_dimensions)?;
    spectrum_method.put_attribute("flag_values", vec![0_i64, 1])?;
    spectrum_method.put_attribute("flag_meanings", "fft lomb_scargle")?;
    add_variable_with_units::<i64>(
        output,
        "residual_spectrum_time_count",
        series_dimensions,
        "1",
    )?;
    let mut spectral_dimensions = series_dimensions.to_vec();
    spectral_dimensions.push("spectral_band");
    for name in [
        "spectral_band_frequency_bin_count",
        "spectral_band_usable_bin_count",
    ] {
        add_variable_with_units::<i64>(output, name, &spectral_dimensions, "1")?;
    }
    if data.constituent_diagnostics.is_some() {
        define_constituent_diagnostic_variables(
            output,
            series_dimensions,
            "source velocity units squared",
        )?;
    }

    for (name, units) in [
        ("semi_major", "source velocity units"),
        ("semi_minor", "source velocity units"),
        ("inclination", "degrees"),
        ("phase", "degrees"),
        ("percent_energy", "percent"),
    ] {
        add_variable_with_units::<f64>(output, name, &solution_dimensions, units)?;
    }
    if data.confidence_interval != ConfidenceInterval::None {
        for (name, units) in [
            ("semi_major_ci", "source velocity units"),
            ("semi_minor_ci", "source velocity units"),
            ("inclination_ci", "degrees"),
            ("phase_ci", "degrees"),
            ("signal_to_noise", "1"),
        ] {
            add_variable_with_units::<f64>(output, name, &solution_dimensions, units)?;
        }
    }
    for (name, units) in [
        ("eastward_mean", "source velocity units"),
        ("northward_mean", "source velocity units"),
        ("eastward_slope", "source velocity units per day"),
        ("northward_slope", "source velocity units per day"),
    ] {
        add_variable_with_units::<f64>(output, name, series_dimensions, units)?;
    }

    if data.analysis_method != AnalysisMethod::Ols {
        output.add_unlimited_dimension("robust_observation")?;
        for name in [
            "robust_weight_row_start",
            "robust_weight_row_size",
            "robust_iterations",
        ] {
            add_variable_with_units::<i64>(output, name, series_dimensions, "1")?;
        }
        output
            .variable_mut("robust_weight_row_size")
            .ok_or_else(|| {
                AppError::Invalid("robust row-size variable was not created".to_owned())
            })?
            .put_attribute("sample_dimension", "robust_observation")?;
        let mut termination =
            output.add_variable::<i64>("robust_termination", series_dimensions)?;
        termination.put_attribute("units", "1")?;
        termination.put_attribute("flag_values", vec![0_i64, 1, 2])?;
        termination.put_attribute("flag_meanings", "tolerance objective_increase exact_fit")?;
        for name in [
            "robust_residual_scale",
            "robust_ols_rms_residual",
            "robust_rms_residual",
        ] {
            add_variable_with_units::<f64>(
                output,
                name,
                series_dimensions,
                "source velocity units",
            )?;
        }
        for name in ["robust_weight", "robust_leverage"] {
            add_variable_with_units::<f64>(output, name, &["robust_observation"], "1")?;
        }
    }

    if let Some(filter) = data.reconstruction {
        let report = reconstruction_report(filter, data.input.modified_julian_days.len());
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
            &data.input.modified_julian_days,
            "days since 1858-11-17 00:00:00 UTC",
        )?;
        let vertical_dimension = if data.input.is_fixed_depth() {
            "depth"
        } else {
            "siglay"
        };
        for name in ["eastward_reconstruction", "northward_reconstruction"] {
            add_variable_with_units::<f64>(
                output,
                name,
                &["time", vertical_dimension, "element"],
                "source velocity units",
            )?;
        }
    }
    Ok(())
}

fn put_layered_series_values<T: netcdf::NcTypeDescriptor + Copy>(
    output: &mut FileMut,
    variable_name: &str,
    element_count: usize,
    first_series: usize,
    row_width: usize,
    values: &[T],
) -> Result<(), AppError> {
    if element_count == 0 || row_width == 0 || !values.len().is_multiple_of(row_width) {
        return Err(AppError::Invalid(format!(
            "invalid incremental shape for {variable_name}"
        )));
    }
    let has_trailing_dimension = output
        .variable(variable_name)
        .ok_or_else(|| AppError::Invalid(format!("missing output variable {variable_name}")))?
        .dimensions()
        .len()
        == 3;
    let row_count = values.len() / row_width;
    let mut local_first = 0;
    while local_first < row_count {
        let global_first = first_series + local_first;
        let layer = global_first / element_count;
        let element = global_first % element_count;
        let rows = (row_count - local_first).min(element_count - element);
        let value_first = local_first * row_width;
        let value_end = (local_first + rows) * row_width;
        let mut extents = vec![
            netcdf::Extent::Index(layer),
            netcdf::Extent::from(element..element + rows),
        ];
        if has_trailing_dimension {
            extents.push(netcdf::Extent::from(..));
        }
        output
            .variable_mut(variable_name)
            .ok_or_else(|| AppError::Invalid(format!("missing output variable {variable_name}")))?
            .put_values(&values[value_first..value_end], extents)?;
        local_first += rows;
    }
    Ok(())
}

fn write_layered_reconstruction_chunk(
    output: &mut FileMut,
    element_count: usize,
    time_count: usize,
    first_series: usize,
    values: &[VectorReconstruction],
) -> Result<(), AppError> {
    const TARGET_TRANSPOSE_BYTES: usize = 64 * 1024 * 1024;
    if values
        .iter()
        .any(|series| series.eastward.len() != time_count || series.northward.len() != time_count)
    {
        return Err(AppError::Invalid(
            "vector reconstruction shape does not match incremental output".to_owned(),
        ));
    }
    let rows_per_buffer =
        (TARGET_TRANSPOSE_BYTES / (time_count.max(1) * std::mem::size_of::<f64>())).max(1);
    for (name, eastward) in [
        ("eastward_reconstruction", true),
        ("northward_reconstruction", false),
    ] {
        let mut local_first = 0;
        while local_first < values.len() {
            let global_first = first_series + local_first;
            let layer = global_first / element_count;
            let element = global_first % element_count;
            let rows = (values.len() - local_first)
                .min(element_count - element)
                .min(rows_per_buffer);
            let block = &values[local_first..local_first + rows];
            let mut time_major = Vec::with_capacity(time_count * rows);
            for time in 0..time_count {
                time_major.extend(block.iter().map(|series| {
                    if eastward {
                        series.eastward[time]
                    } else {
                        series.northward[time]
                    }
                }));
            }
            output
                .variable_mut(name)
                .ok_or_else(|| AppError::Invalid(format!("missing output variable {name}")))?
                .put_values(&time_major, (.., layer, element..element + rows))?;
            local_first += rows;
        }
    }
    Ok(())
}

fn collect_vector_solution_field<F>(
    solutions: &[VectorSolution],
    constituent_count: usize,
    field: F,
) -> Result<Vec<f64>, AppError>
where
    F: for<'solution> Fn(&'solution VectorSolution) -> &'solution [f64],
{
    let mut values = Vec::with_capacity(solutions.len() * constituent_count);
    for (series, solution) in solutions.iter().enumerate() {
        let row = field(solution);
        if row.len() != constituent_count {
            return Err(AppError::Invalid(format!(
                "vector result series {series} contains {} constituents; expected {constituent_count}",
                row.len()
            )));
        }
        values.extend_from_slice(row);
    }
    Ok(values)
}

fn write_vector_solution_field<F>(
    output: &mut FileMut,
    name: &str,
    element_count: usize,
    first_series: usize,
    constituent_count: usize,
    solutions: &[VectorSolution],
    field: F,
) -> Result<(), AppError>
where
    F: for<'solution> Fn(&'solution VectorSolution) -> &'solution [f64],
{
    let values = collect_vector_solution_field(solutions, constituent_count, field)?;
    put_layered_series_values(
        output,
        name,
        element_count,
        first_series,
        constituent_count,
        &values,
    )
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "incremental diagnostics preserve the shared field order without whole-run buffers"
)]
fn write_incremental_constituent_diagnostics(
    output: &mut FileMut,
    element_count: usize,
    first_series: usize,
    constituent_count: usize,
    diagnostics: &[Option<ConstituentSelectionDiagnostics>],
) -> Result<(), AppError> {
    validate_constituent_diagnostics_shape(diagnostics, diagnostics.len(), constituent_count)?;
    for (name, selector) in [
        ("diagnostic_basis_condition_number", 0_u8),
        ("diagnostic_all_constituent_signal_to_noise", 1),
        ("diagnostic_condition_adjusted_signal_to_noise", 2),
        ("diagnostic_raw_tidal_variance", 3),
        ("diagnostic_all_constituent_tidal_variance", 4),
        ("diagnostic_significant_constituent_tidal_variance", 5),
        ("diagnostic_all_constituent_percent_tidal_variance", 6),
        (
            "diagnostic_significant_constituent_percent_tidal_variance",
            7,
        ),
    ] {
        let values = diagnostics
            .iter()
            .map(|diagnostics| diagnostic_scalar_value(diagnostics.as_ref(), selector))
            .collect::<Vec<_>>();
        put_layered_series_values(output, name, element_count, first_series, 1, &values)?;
    }
    for (higher, direction) in [(false, "lower"), (true, "higher")] {
        let mut indices = Vec::with_capacity(diagnostics.len().saturating_mul(constituent_count));
        for row in diagnostics {
            if let Some(row) = row {
                for constituent in &row.constituents {
                    let neighbor = if higher {
                        constituent.higher.as_ref()
                    } else {
                        constituent.lower.as_ref()
                    };
                    indices.push(match neighbor {
                        Some(neighbor) => i64::try_from(neighbor.index).map_err(|_| {
                            AppError::Invalid("diagnostic neighbor index exceeds i64".to_owned())
                        })?,
                        None => -1,
                    });
                }
            } else {
                indices.extend(std::iter::repeat_n(-1, constituent_count));
            }
        }
        put_layered_series_values(
            output,
            &format!("diagnostic_{direction}_neighbor_index"),
            element_count,
            first_series,
            constituent_count,
            &indices,
        )?;
        for (suffix, selector) in [
            ("rayleigh_criterion", 0_u8),
            ("noise_modified_rayleigh_criterion", 1),
            ("maximum_correlation", 2),
        ] {
            let mut values =
                Vec::with_capacity(diagnostics.len().saturating_mul(constituent_count));
            for row in diagnostics {
                if let Some(row) = row {
                    values.extend(row.constituents.iter().map(|constituent| {
                        let neighbor = if higher {
                            constituent.higher.as_ref()
                        } else {
                            constituent.lower.as_ref()
                        };
                        diagnostic_neighbor_value(neighbor, selector)
                    }));
                } else {
                    values.extend(std::iter::repeat_n(f64::NAN, constituent_count));
                }
            }
            put_layered_series_values(
                output,
                &format!("diagnostic_{direction}_{suffix}"),
                element_count,
                first_series,
                constituent_count,
                &values,
            )?;
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one chunk writer validates and serializes the complete vector schema"
)]
fn write_incremental_vector_chunk(
    output: &mut FileMut,
    element_count: usize,
    time_count: usize,
    constituent_count: usize,
    analysis_method: AnalysisMethod,
    confidence_interval: ConfidenceInterval,
    first_series: usize,
    observation_counts: &[usize],
    frequencies: &[Vec<f64>],
    solutions: &[VectorSolution],
    constituent_order: &ConstituentOrderMap,
    sampling: &[CoreSamplingDiagnostics],
    constituent_diagnostics: Option<&[Option<ConstituentSelectionDiagnostics>]>,
    reconstruction: Option<&[VectorReconstruction]>,
    robust_observation_offset: &mut usize,
) -> Result<(), AppError> {
    let series_count = solutions.len();
    if observation_counts.len() != series_count
        || frequencies.len() != series_count
        || sampling.len() != series_count
        || constituent_diagnostics.is_some_and(|values| values.len() != series_count)
        || !constituent_order.is_valid_for(series_count, constituent_count)
        || reconstruction.is_some_and(|values| values.len() != series_count)
    {
        return Err(AppError::Invalid(
            "incremental vector chunk shapes differ".to_owned(),
        ));
    }

    let observation_counts = observation_counts
        .iter()
        .copied()
        .map(|count| {
            i64::try_from(count)
                .map_err(|_| AppError::Invalid("observation count exceeds i64".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let analysis_status = solutions
        .iter()
        .map(|solution| i64::from(!solution.reference_time_days.is_finite()))
        .collect::<Vec<_>>();
    put_layered_series_values(
        output,
        "analysis_status",
        element_count,
        first_series,
        1,
        &analysis_status,
    )?;
    put_layered_series_values(
        output,
        "observation_count",
        element_count,
        first_series,
        1,
        &observation_counts,
    )?;
    let reference_times = solutions
        .iter()
        .map(|solution| solution.reference_time_days)
        .collect::<Vec<_>>();
    put_layered_series_values(
        output,
        "reference_time",
        element_count,
        first_series,
        1,
        &reference_times,
    )?;
    if frequencies
        .iter()
        .any(|frequency| frequency.len() != constituent_count)
    {
        return Err(AppError::Invalid(
            "frequency result shape does not match incremental output".to_owned(),
        ));
    }
    let frequency_values = frequencies.iter().flatten().copied().collect::<Vec<_>>();
    put_layered_series_values(
        output,
        "frequency",
        element_count,
        first_series,
        constituent_count,
        &frequency_values,
    )?;
    let mut order_values = Vec::with_capacity(series_count * constituent_count);
    for series in 0..series_count {
        order_values.extend(constituent_order.row(series).iter().copied().map(i64::from));
    }
    put_layered_series_values(
        output,
        "constituent_index_by_rank",
        element_count,
        first_series,
        constituent_count,
        &order_values,
    )?;

    for (name, values) in [
        (
            "sampling_record_span",
            sampling
                .iter()
                .map(|series| series.record_span_days)
                .collect::<Vec<_>>(),
        ),
        (
            "sampling_mean_interval",
            sampling
                .iter()
                .map(|series| series.mean_sample_interval_hours)
                .collect::<Vec<_>>(),
        ),
        (
            "sampling_largest_gap",
            sampling
                .iter()
                .map(|series| series.largest_gap_hours)
                .collect::<Vec<_>>(),
        ),
    ] {
        put_layered_series_values(output, name, element_count, first_series, 1, &values)?;
    }
    let spectrum_methods = sampling
        .iter()
        .map(|series| match series.residual_spectrum_method {
            ResidualSpectrumMethod::Fft => 0_i64,
            ResidualSpectrumMethod::LombScargle => 1_i64,
        })
        .collect::<Vec<_>>();
    put_layered_series_values(
        output,
        "residual_spectrum_method",
        element_count,
        first_series,
        1,
        &spectrum_methods,
    )?;
    let spectrum_time_counts = sampling
        .iter()
        .map(|series| {
            i64::try_from(series.residual_spectrum_time_count).map_err(|_| {
                AppError::Invalid("residual spectrum time count exceeds i64".to_owned())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    put_layered_series_values(
        output,
        "residual_spectrum_time_count",
        element_count,
        first_series,
        1,
        &spectrum_time_counts,
    )?;
    for (name, counts) in [
        (
            "spectral_band_frequency_bin_count",
            sampling
                .iter()
                .flat_map(|series| series.spectral_band_bin_count)
                .collect::<Vec<_>>(),
        ),
        (
            "spectral_band_usable_bin_count",
            sampling
                .iter()
                .flat_map(|series| series.spectral_band_usable_bin_count)
                .collect::<Vec<_>>(),
        ),
    ] {
        let counts = counts
            .into_iter()
            .map(|count| {
                i64::try_from(count)
                    .map_err(|_| AppError::Invalid("spectral bin count exceeds i64".to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        put_layered_series_values(output, name, element_count, first_series, 9, &counts)?;
    }
    if let Some(diagnostics) = constituent_diagnostics {
        write_incremental_constituent_diagnostics(
            output,
            element_count,
            first_series,
            constituent_count,
            diagnostics,
        )?;
    }

    write_vector_solution_field(
        output,
        "semi_major",
        element_count,
        first_series,
        constituent_count,
        solutions,
        |solution| &solution.semi_major,
    )?;
    write_vector_solution_field(
        output,
        "semi_minor",
        element_count,
        first_series,
        constituent_count,
        solutions,
        |solution| &solution.semi_minor,
    )?;
    write_vector_solution_field(
        output,
        "inclination",
        element_count,
        first_series,
        constituent_count,
        solutions,
        |solution| &solution.inclination_degrees,
    )?;
    write_vector_solution_field(
        output,
        "phase",
        element_count,
        first_series,
        constituent_count,
        solutions,
        |solution| &solution.phase_degrees,
    )?;
    write_vector_solution_field(
        output,
        "percent_energy",
        element_count,
        first_series,
        constituent_count,
        solutions,
        |solution| &solution.percent_energy,
    )?;
    if confidence_interval != ConfidenceInterval::None {
        for (name, selector) in [
            ("semi_major_ci", 0_u8),
            ("semi_minor_ci", 1),
            ("inclination_ci", 2),
            ("phase_ci", 3),
            ("signal_to_noise", 4),
        ] {
            let mut values = Vec::with_capacity(series_count * constituent_count);
            for solution in solutions {
                let confidence = vector_confidence_values(solution, confidence_interval)?
                    .ok_or_else(|| {
                        AppError::Invalid("vector confidence output was omitted".to_owned())
                    })?;
                values.extend_from_slice(match selector {
                    0 => confidence.0,
                    1 => confidence.1,
                    2 => confidence.2,
                    3 => confidence.3,
                    4 => confidence.4,
                    _ => unreachable!("confidence selector is internal"),
                });
            }
            put_layered_series_values(
                output,
                name,
                element_count,
                first_series,
                constituent_count,
                &values,
            )?;
        }
    }
    for (name, values) in [
        (
            "eastward_mean",
            solutions
                .iter()
                .map(|solution| solution.eastward_mean)
                .collect::<Vec<_>>(),
        ),
        (
            "northward_mean",
            solutions
                .iter()
                .map(|solution| solution.northward_mean)
                .collect::<Vec<_>>(),
        ),
        (
            "eastward_slope",
            solutions
                .iter()
                .map(|solution| solution.eastward_slope_per_day)
                .collect::<Vec<_>>(),
        ),
        (
            "northward_slope",
            solutions
                .iter()
                .map(|solution| solution.northward_slope_per_day)
                .collect::<Vec<_>>(),
        ),
    ] {
        put_layered_series_values(output, name, element_count, first_series, 1, &values)?;
    }

    if analysis_method == AnalysisMethod::Ols {
        if solutions.iter().any(|solution| solution.robust.is_some()) {
            return Err(AppError::Invalid(
                "OLS solver returned unexpected robust diagnostics".to_owned(),
            ));
        }
    } else {
        let diagnostics = solutions
            .iter()
            .map(|solution| {
                solution.robust.as_ref().ok_or_else(|| {
                    AppError::Invalid("robust solver omitted diagnostics".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut row_start = Vec::with_capacity(series_count);
        let mut row_size = Vec::with_capacity(series_count);
        let mut iterations = Vec::with_capacity(series_count);
        let mut termination = Vec::with_capacity(series_count);
        let mut residual_scale = Vec::with_capacity(series_count);
        let mut ols_rms_residual = Vec::with_capacity(series_count);
        let mut rms_residual = Vec::with_capacity(series_count);
        let total_observations = diagnostics
            .iter()
            .map(|diagnostics| diagnostics.weights.len())
            .sum();
        let mut next_row_start = *robust_observation_offset;
        for diagnostics in &diagnostics {
            if diagnostics.weights.len() != diagnostics.leverage.len() {
                return Err(AppError::Invalid(
                    "robust weight and leverage lengths differ".to_owned(),
                ));
            }
            row_start.push(i64::try_from(next_row_start).map_err(|_| {
                AppError::Invalid("robust observation start exceeds i64".to_owned())
            })?);
            row_size.push(i64::try_from(diagnostics.weights.len()).map_err(|_| {
                AppError::Invalid("robust observation count exceeds i64".to_owned())
            })?);
            next_row_start = next_row_start
                .checked_add(diagnostics.weights.len())
                .ok_or_else(|| {
                    AppError::Invalid("robust observation offset overflows".to_owned())
                })?;
            iterations.push(
                i64::try_from(diagnostics.iterations).map_err(|_| {
                    AppError::Invalid("robust iteration count exceeds i64".to_owned())
                })?,
            );
            termination.push(robust_termination_code(diagnostics.termination));
            residual_scale.push(diagnostics.residual_scale);
            ols_rms_residual.push(diagnostics.ols_rms_residual);
            rms_residual.push(diagnostics.rms_residual);
        }
        put_layered_series_values(
            output,
            "robust_weight_row_start",
            element_count,
            first_series,
            1,
            &row_start,
        )?;
        for (name, values) in [
            ("robust_weight_row_size", &row_size),
            ("robust_iterations", &iterations),
            ("robust_termination", &termination),
        ] {
            put_layered_series_values(output, name, element_count, first_series, 1, values)?;
        }
        for (name, values) in [
            ("robust_residual_scale", &residual_scale),
            ("robust_ols_rms_residual", &ols_rms_residual),
            ("robust_rms_residual", &rms_residual),
        ] {
            put_layered_series_values(output, name, element_count, first_series, 1, values)?;
        }
        let end = robust_observation_offset
            .checked_add(total_observations)
            .ok_or_else(|| AppError::Invalid("robust observation offset overflows".to_owned()))?;
        let weights = diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.weights.iter().copied())
            .collect::<Vec<_>>();
        output
            .variable_mut("robust_weight")
            .ok_or_else(|| AppError::Invalid("missing robust weight output".to_owned()))?
            .put_values(&weights, *robust_observation_offset..end)?;
        drop(weights);
        let leverage = diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.leverage.iter().copied())
            .collect::<Vec<_>>();
        output
            .variable_mut("robust_leverage")
            .ok_or_else(|| AppError::Invalid("missing robust leverage output".to_owned()))?
            .put_values(&leverage, *robust_observation_offset..end)?;
        *robust_observation_offset = end;
    }

    if let Some(reconstruction) = reconstruction {
        write_layered_reconstruction_chunk(
            output,
            element_count,
            time_count,
            first_series,
            reconstruction,
        )?;
    }
    Ok(())
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
    let mut output = create_vector_output_base(
        path,
        &VectorOutputDefinition {
            input,
            constituents: data.constituents,
            constituent_order: data.constituent_order,
            selection: data.selection,
            inference: data.inference,
            fit_options: data.fit_options,
            phase_reference: data.phase_reference,
            nodal_corrections: data.nodal_corrections,
            analysis_method: data.analysis_method,
            confidence_interval: data.confidence_interval,
            constituent_diagnostics: data.constituent_diagnostics_options,
            chunk_plan: data.chunk_plan,
            input_pipeline: data.input_pipeline,
            reference_time_modified_julian_day: data.reference_time_modified_julian_day,
            reconstruction: data.reconstruction.map(|(filter, _)| filter),
            result_output: "buffered",
        },
    )?;
    output.add_attribute("result_sha256", data.result_sha256)?;
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
        &mut output.add_variable::<i64>("observation_count", input.series_dimensions())?,
        &observation_counts,
        "1",
    )?;
    let analysis_status = data
        .solutions
        .iter()
        .map(|solution| i64::from(!solution.reference_time_days.is_finite()))
        .collect::<Vec<_>>();
    let mut status = output.add_variable::<i64>("analysis_status", input.series_dimensions())?;
    status.put_attribute("units", "1")?;
    status.put_attribute("flag_values", vec![0_i64, 1])?;
    status.put_attribute("flag_meanings", "fitted unavailable")?;
    status.put_values(&analysis_status, ..)?;
    let reference_times = data
        .solutions
        .iter()
        .map(|solution| solution.reference_time_days)
        .collect::<Vec<_>>();
    write_variable(
        &mut output.add_variable::<f64>("reference_time", input.series_dimensions())?,
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
        &mut output.add_variable::<f64>("frequency", &input.solution_dimensions())?,
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
        input.series_dimensions(),
    )?;
    write_sampling_diagnostics(
        &mut output,
        data.solutions.len(),
        data.sampling_diagnostics,
        input.series_dimensions(),
    )?;
    write_constituent_diagnostics(
        &mut output,
        data.solutions.len(),
        data.constituents.len(),
        data.constituent_diagnostics,
        input.series_dimensions(),
        "source velocity units squared",
    )?;
    write_vector_solution_variables(
        &mut output,
        data.constituents.len(),
        data.solutions,
        data.analysis_method,
        data.confidence_interval,
        input.series_dimensions(),
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
    series_dimensions: &[&str],
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
        let mut dimensions = series_dimensions.to_vec();
        dimensions.push("constituent");
        write_variable(
            &mut output.add_variable::<f64>(name, &dimensions)?,
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
            let mut dimensions = series_dimensions.to_vec();
            dimensions.push("constituent");
            write_variable(
                &mut output.add_variable::<f64>(name, &dimensions)?,
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
            &mut output.add_variable::<f64>(name, series_dimensions)?,
            values,
            units,
        )?;
    }
    write_vector_robust_variables(output, solutions, analysis_method, series_dimensions)?;
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "the buffered robust schema is kept together to mirror incremental output"
)]
fn write_vector_robust_variables(
    output: &mut FileMut,
    solutions: &[VectorSolution],
    analysis_method: AnalysisMethod,
    series_dimensions: &[&str],
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
    let mut row_start = Vec::with_capacity(diagnostics.len());
    let mut row_size = Vec::with_capacity(diagnostics.len());
    let mut iterations = Vec::with_capacity(diagnostics.len());
    let mut termination = Vec::with_capacity(diagnostics.len());
    let mut residual_scale = Vec::with_capacity(diagnostics.len());
    let mut ols_rms_residual = Vec::with_capacity(diagnostics.len());
    let mut rms_residual = Vec::with_capacity(diagnostics.len());
    let mut weights = Vec::with_capacity(total_observations);
    let mut leverage = Vec::with_capacity(total_observations);
    let mut next_row_start = 0_usize;
    for diagnostics in diagnostics {
        if diagnostics.weights.len() != diagnostics.leverage.len() {
            return Err(AppError::Invalid(
                "robust weight and leverage lengths differ".to_owned(),
            ));
        }
        row_start.push(
            i64::try_from(next_row_start).map_err(|_| {
                AppError::Invalid("robust observation start exceeds i64".to_owned())
            })?,
        );
        row_size.push(
            i64::try_from(diagnostics.weights.len()).map_err(|_| {
                AppError::Invalid("robust observation count exceeds i64".to_owned())
            })?,
        );
        next_row_start = next_row_start
            .checked_add(diagnostics.weights.len())
            .ok_or_else(|| AppError::Invalid("robust observation offset overflows".to_owned()))?;
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
    write_variable(
        &mut output.add_variable::<i64>("robust_weight_row_start", series_dimensions)?,
        &row_start,
        "1",
    )?;
    for (name, values) in [
        ("robust_weight_row_size", &row_size),
        ("robust_iterations", &iterations),
    ] {
        write_variable(
            &mut output.add_variable::<i64>(name, series_dimensions)?,
            values,
            "1",
        )?;
    }
    write_robust_schema_metadata(output, &termination, series_dimensions)?;
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
            &mut output.add_variable::<f64>(name, series_dimensions)?,
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
    if values.len() != input.series_count()
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
        let dimensions = if input.is_fixed_depth() {
            vec!["time", "depth", "element"]
        } else if input.is_depth_resolved() {
            vec!["time", "siglay", "element"]
        } else {
            vec!["time", "series"]
        };
        let mut variable = output.add_variable::<f64>(name, &dimensions)?;
        variable.put_attribute("units", "source velocity units")?;
        for (series, reconstruction) in values.iter().enumerate() {
            let component = if eastward {
                &reconstruction.eastward
            } else {
                &reconstruction.northward
            };
            if input.is_depth_resolved() {
                let element_count = input.element_indices.len();
                variable.put_values(
                    component,
                    (.., series / element_count, series % element_count),
                )?;
            } else {
                variable.put_values(component, (.., series))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::Arc};

    use rayon::ThreadPoolBuilder;
    use rutide_core::{
        ConstituentDiagnosticsOptions, FitOptions, InferenceMode, LinearConfidence,
        MonteCarloOptions, NodalCorrections, PhaseReference, ReconstructionFilter, RobustOptions,
        RobustWeightFunction, TidalConstituent, VectorInferenceRelation,
    };

    use super::{
        AnalysisMethod, ConfidenceInterval, ConstituentOrder, ConstituentSelection, NodeSelection,
        VectorAnalyzeConfig, VectorInferenceConfig, analyze_vector, compact_bounded_span_values,
        pipelined_vector_chunk_reader, read_fvcom_fixed_depth_element_chunk, read_fvcom_vector,
        read_fvcom_vector_metadata, regular_vector_chunk_plan, temporary_sibling,
    };

    #[test]
    fn bounded_span_compaction_preserves_later_time_rows() {
        let selected_indices = [2, 4, 7];
        let source = (0..3)
            .flat_map(|time| (2..8).map(move |index| time * 100 + index))
            .collect::<Vec<_>>();
        let compacted = compact_bounded_span_values(source, 3, 6, 2, &selected_indices)
            .expect("compact bounded time-major span");
        assert_eq!(compacted, [2, 4, 7, 102, 104, 107, 202, 204, 207]);
    }

    #[test]
    fn automatic_vector_pipeline_is_double_buffered_and_bounded() {
        let (automatic, pipelined) = regular_vector_chunk_plan(None, 144_860, 745, 2, 64)
            .expect("valid automatic vector plan");
        assert!(pipelined);
        assert!(automatic.series_per_chunk.is_multiple_of(64));
        let concurrent_source_bytes = automatic.series_per_chunk * 745 * 2 * 8 * 2;
        assert!(concurrent_source_bytes <= 512 * 1024 * 1024);
        assert!(automatic.chunk_count > 4);

        let (explicit, pipelined) = regular_vector_chunk_plan(Some(44_992), 144_860, 745, 2, 64)
            .expect("valid explicit vector plan");
        assert!(!pipelined);
        assert_eq!(explicit.series_per_chunk, 44_992);

        let (complete, pipelined) =
            regular_vector_chunk_plan(None, 100, 745, 2, 64).expect("valid complete vector plan");
        assert!(!pipelined);
        assert_eq!(complete.chunk_count, 1);
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::too_many_lines,
        reason = "the f32 FVCOM fixture freezes physical interpolation and masking semantics"
    )]
    fn fixed_depth_interpolation_is_physical_masked_and_chunk_invariant() {
        let input_path =
            temporary_sibling(&std::env::temp_dir().join("rutide-fixed-depth-input-test.nc"))
                .expect("valid fixed-depth input path");
        let output_path =
            temporary_sibling(&std::env::temp_dir().join("rutide-fixed-depth-output-test.nc"))
                .expect("valid fixed-depth output path");
        let chunked_output_path = temporary_sibling(
            &std::env::temp_dir().join("rutide-fixed-depth-chunked-output-test.nc"),
        )
        .expect("valid fixed-depth chunked output path");
        let unavailable_output_path = temporary_sibling(
            &std::env::temp_dir().join("rutide-fixed-depth-unavailable-output-test.nc"),
        )
        .expect("valid unavailable output path");
        let mixed_output_path = temporary_sibling(
            &std::env::temp_dir().join("rutide-fixed-depth-mixed-output-test.nc"),
        )
        .expect("valid mixed output path");
        let mixed_chunked_output_path = temporary_sibling(
            &std::env::temp_dir().join("rutide-fixed-depth-mixed-chunked-output-test.nc"),
        )
        .expect("valid mixed chunked output path");
        let time_count = 72_usize;
        let element_count = 2_usize;
        let node_count = 6_usize;
        let layer_count = 3_usize;
        let fill = -999.0_f32;
        let mut dataset = netcdf::create(&input_path).expect("create fixed-depth fixture");
        for (name, length) in [
            ("time", time_count),
            ("nele", element_count),
            ("node", node_count),
            ("siglay", layer_count),
            ("three", 3),
        ] {
            dataset.add_dimension(name, length).expect("add dimension");
        }
        dataset
            .add_variable::<i32>("Itime", &["time"])
            .expect("add Itime")
            .put_values(&vec![58_113_i32; time_count], ..)
            .expect("write Itime");
        dataset
            .add_variable::<i32>("Itime2", &["time"])
            .expect("add Itime2")
            .put_values(
                &(0..time_count)
                    .map(|time| i32::try_from(time).expect("small time") * 3_600_000)
                    .collect::<Vec<_>>(),
                ..,
            )
            .expect("write Itime2");
        dataset
            .add_variable::<f32>("latc", &["nele"])
            .expect("add latc")
            .put_values(&[60.0, 61.0], ..)
            .expect("write latc");
        dataset
            .add_variable::<i32>("nv", &["three", "nele"])
            .expect("add nv")
            .put_values(&[1, 4, 2, 5, 3, 6], ..)
            .expect("write nv");
        {
            let mut variable = dataset.add_variable::<f32>("h", &["node"]).expect("add h");
            variable.set_fill_value(fill).expect("set h fill");
            variable
                .put_values(&[20.0, 20.0, 20.0, 30.0, 30.0, 30.0], ..)
                .expect("write h");
        }
        {
            let mut variable = dataset
                .add_variable::<f32>("siglay", &["siglay", "node"])
                .expect("add siglay");
            variable.set_fill_value(fill).expect("set sigma fill");
            let sigma = [-0.25_f32, -0.5, -0.75]
                .into_iter()
                .flat_map(|sigma| std::iter::repeat_n(sigma, node_count))
                .collect::<Vec<_>>();
            variable.put_values(&sigma, ..).expect("write siglay");
        }
        let mut zeta = Vec::with_capacity(time_count * node_count);
        let mut wet_cells = vec![1_i32; time_count * element_count];
        let mut eastward = Vec::with_capacity(time_count * layer_count * element_count);
        let mut northward = Vec::with_capacity(time_count * layer_count * element_count);
        for time in 0..time_count {
            let position = f64::from(u32::try_from(time).expect("small time"));
            let surface = 0.4 * (position / 9.0).sin();
            zeta.extend(std::iter::repeat_n(surface as f32, node_count));
            for layer_fraction in [0.25_f64, 0.5, 0.75] {
                for element_depth in [20.0_f64, 30.0] {
                    let layer_depth = layer_fraction * (element_depth + surface);
                    let eastward_base = 0.3 + (position / 5.0).sin();
                    let northward_base = -0.2 + (position / 7.0).cos();
                    eastward.push((eastward_base + 2.0 * layer_depth) as f32);
                    northward.push((northward_base - 0.5 * layer_depth) as f32);
                }
            }
        }
        wet_cells[6 * element_count + 1] = 0;
        zeta[7 * node_count + 3] = fill;
        eastward[0] = fill;
        northward[(5 * layer_count + 1) * element_count] = fill;
        {
            let mut variable = dataset
                .add_variable::<f32>("zeta", &["time", "node"])
                .expect("add zeta");
            variable.set_fill_value(fill).expect("set zeta fill");
            variable.put_values(&zeta, ..).expect("write zeta");
        }
        dataset
            .add_variable::<i32>("wet_cells", &["time", "nele"])
            .expect("add wet cells")
            .put_values(&wet_cells, ..)
            .expect("write wet cells");
        for (name, values) in [("u", &eastward), ("v", &northward)] {
            let mut variable = dataset
                .add_variable::<f32>(name, &["time", "siglay", "nele"])
                .expect("add current component");
            variable.set_fill_value(fill).expect("set current fill");
            variable.put_values(values, ..).expect("write current");
        }
        dataset.close().expect("close fixed-depth fixture");

        let source = netcdf::open(&input_path).expect("open fixed-depth fixture");
        let interpolation_pool = ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build interpolation pool");
        let metadata = read_fvcom_vector_metadata(
            &input_path,
            &source,
            &NodeSelection::All,
            None,
            Some(&[8.0, 12.0]),
        )
        .expect("read fixed-depth metadata");
        let chunks =
            read_fvcom_fixed_depth_element_chunk(&source, &metadata, 0..2, &interpolation_pool)
                .expect("interpolate fixed depths");
        assert_eq!(chunks.len(), 2);
        for (depth_position, target) in [8.0_f64, 12.0].into_iter().enumerate() {
            let chunk = &chunks[depth_position];
            assert_eq!(
                chunk.observation_counts,
                if depth_position == 0 {
                    [70, 70]
                } else {
                    [71, 70]
                }
            );
            for time in 0..time_count {
                let position = f64::from(u32::try_from(time).expect("small time"));
                for element in 0..element_count {
                    let value = time * element_count + element;
                    let missing = (time == 5 && element == 0)
                        || (depth_position == 0 && time == 0 && element == 0)
                        || ((time == 6 || time == 7) && element == 1);
                    if missing {
                        assert!(chunk.eastward[value].is_nan());
                        assert!(chunk.northward[value].is_nan());
                    } else {
                        let expected_u = 0.3 + (position / 5.0).sin() + 2.0 * target;
                        let expected_v = -0.2 + (position / 7.0).cos() - 0.5 * target;
                        assert!(
                            (chunk.eastward[value] - expected_u).abs() < 1e-5,
                            "u mismatch at depth {target}, time {time}, element {element}: {} versus {expected_u}",
                            chunk.eastward[value],
                        );
                        assert!(
                            (chunk.northward[value] - expected_v).abs() < 3e-6,
                            "v mismatch at depth {target}, time {time}, element {element}: {} versus {expected_v}",
                            chunk.northward[value],
                        );
                    }
                }
            }
        }
        let exact_metadata = read_fvcom_vector_metadata(
            &input_path,
            &source,
            &NodeSelection::All,
            None,
            Some(&[10.0]),
        )
        .expect("read exact-layer fixed-depth metadata");
        let exact = read_fvcom_fixed_depth_element_chunk(
            &source,
            &exact_metadata,
            0..2,
            &interpolation_pool,
        )
        .expect("interpolate exact layer depth");
        assert_eq!(exact[0].observation_counts, [71, 70]);
        assert!(exact[0].eastward[0].is_finite());
        drop(source);

        let config = VectorAnalyzeConfig {
            input: input_path.clone(),
            output: output_path.clone(),
            report: None,
            elements: NodeSelection::All,
            layers: None,
            fixed_depths_meters: Some(vec![8.0, 12.0]),
            constituent_selection: ConstituentSelection::Explicit(vec![TidalConstituent::M2]),
            constituent_order: ConstituentOrder::Selection,
            inference: None,
            fit_options: FitOptions { trend: false },
            phase_reference: PhaseReference::Raw,
            nodal_corrections: NodalCorrections::Disabled,
            confidence_interval: ConfidenceInterval::MonteCarlo {
                options: MonteCarloOptions {
                    realizations: 32,
                    seed: 17,
                },
                noise: LinearConfidence::White,
            },
            analysis_method: AnalysisMethod::Robust(RobustOptions {
                weight_function: RobustWeightFunction::Welsch,
                tolerance: 0.01,
                ..RobustOptions::default()
            }),
            constituent_diagnostics: Some(ConstituentDiagnosticsOptions::default()),
            reconstruction: Some(ReconstructionFilter::All),
            workers: 2,
            chunk_series: None,
            overwrite: false,
        };
        let report = analyze_vector(&config).expect("analyze fixed-depth fixture");
        let mut chunked_config = config.clone();
        chunked_config.output = chunked_output_path.clone();
        chunked_config.chunk_series = Some(2);
        let chunked_report =
            analyze_vector(&chunked_config).expect("analyze one-element fixed-depth chunks");
        assert_eq!(report.vertical_mode, "fixed-depth");
        assert_eq!(report.fixed_depths_meters, Some(vec![8.0, 12.0]));
        assert_eq!(report.series_count, 4);
        assert_eq!(report.result_output, "incremental");
        assert_eq!(report.result_sha256, chunked_report.result_sha256);
        assert_eq!(chunked_report.chunk_series, 2);
        assert_eq!(chunked_report.chunk_count, 2);
        assert_eq!(
            report.sample_results[0].depth_meters_below_surface,
            Some(8.0)
        );

        let output = netcdf::open(&output_path).expect("open fixed-depth output");
        assert_eq!(
            output
                .variable("depth")
                .expect("depth coordinate")
                .get_values::<f64, _>(..)
                .expect("read depth coordinate"),
            [8.0, 12.0]
        );
        assert_eq!(
            output
                .variable("semi_major")
                .expect("fixed-depth semi-major")
                .dimensions()
                .iter()
                .map(netcdf::Dimension::name)
                .collect::<Vec<_>>(),
            ["depth", "element", "constituent"]
        );
        assert_eq!(
            output
                .variable("diagnostic_basis_condition_number")
                .expect("fixed-depth condition numbers")
                .dimensions()
                .iter()
                .map(netcdf::Dimension::name)
                .collect::<Vec<_>>(),
            ["depth", "element"]
        );
        assert_eq!(
            output
                .variable("eastward_reconstruction")
                .expect("fixed-depth reconstruction")
                .dimensions()
                .iter()
                .map(netcdf::Dimension::name)
                .collect::<Vec<_>>(),
            ["time", "depth", "element"]
        );
        assert_eq!(
            output
                .variable("observation_count")
                .expect("fixed-depth observation count")
                .get_values::<i64, _>(..)
                .expect("read fixed-depth observation count"),
            [70, 70, 71, 70]
        );
        let chunked_output =
            netcdf::open(&chunked_output_path).expect("open chunked fixed-depth output");
        assert_eq!(
            chunked_output
                .variable("robust_weight_row_start")
                .expect("chunked robust row starts")
                .get_values::<i64, _>(..)
                .expect("read chunked row starts"),
            [0, 141, 70, 211]
        );
        for name in ["semi_major", "semi_minor", "inclination", "phase"] {
            assert_eq!(
                output
                    .variable(name)
                    .expect("whole fixed-depth variable")
                    .get_values::<f64, _>(..)
                    .expect("read whole fixed-depth variable"),
                chunked_output
                    .variable(name)
                    .expect("chunked fixed-depth variable")
                    .get_values::<f64, _>(..)
                    .expect("read chunked fixed-depth variable"),
                "chunked fixed-depth {name} differs"
            );
        }
        assert_eq!(
            output
                .variable("diagnostic_basis_condition_number")
                .expect("whole fixed-depth condition numbers")
                .get_values::<f64, _>(..)
                .expect("read whole fixed-depth condition numbers"),
            chunked_output
                .variable("diagnostic_basis_condition_number")
                .expect("chunked fixed-depth condition numbers")
                .get_values::<f64, _>(..)
                .expect("read chunked fixed-depth condition numbers")
        );
        drop(output);
        drop(chunked_output);

        let mut unavailable_config = config.clone();
        unavailable_config.output = unavailable_output_path.clone();
        unavailable_config.fixed_depths_meters = Some(vec![100.0]);
        unavailable_config.analysis_method = AnalysisMethod::Ols;
        unavailable_config.reconstruction = None;
        let unavailable =
            analyze_vector(&unavailable_config).expect("retain unavailable fixed-depth rows");
        assert_eq!(unavailable.fitted_series_count, 0);
        assert_eq!(unavailable.unavailable_series_count, 2);
        assert_eq!(unavailable.sampling.minimum_observation_count, 0);
        let unavailable_output =
            netcdf::open(&unavailable_output_path).expect("open unavailable output");
        assert_eq!(
            unavailable_output
                .variable("analysis_status")
                .expect("analysis status")
                .get_values::<i64, _>(..)
                .expect("read analysis status"),
            [1, 1]
        );
        assert!(
            unavailable_output
                .variable("semi_major")
                .expect("unavailable semi-major")
                .get_values::<f64, _>(..)
                .expect("read unavailable semi-major")
                .iter()
                .all(|value| value.is_nan())
        );
        assert!(
            unavailable_output
                .variable("diagnostic_basis_condition_number")
                .expect("unavailable condition number")
                .get_values::<f64, _>(..)
                .expect("read unavailable condition number")
                .iter()
                .all(|value| value.is_nan())
        );
        assert!(
            unavailable_output
                .variable("diagnostic_lower_neighbor_index")
                .expect("unavailable neighbor index")
                .get_values::<i64, _>(..)
                .expect("read unavailable neighbor index")
                .iter()
                .all(|value| *value == -1)
        );
        drop(unavailable_output);

        let mut mixed_config = config.clone();
        mixed_config.output = mixed_output_path.clone();
        mixed_config.fixed_depths_meters = Some(vec![20.0]);
        mixed_config.reconstruction = None;
        let mixed = analyze_vector(&mixed_config).expect("analyze mixed fixed-depth rows");
        let mut mixed_chunked_config = mixed_config.clone();
        mixed_chunked_config.output = mixed_chunked_output_path.clone();
        mixed_chunked_config.chunk_series = Some(1);
        let mixed_chunked = analyze_vector(&mixed_chunked_config)
            .expect("analyze mixed rows one element at a time");
        assert_eq!(mixed.fitted_series_count, 1);
        assert_eq!(mixed.unavailable_series_count, 1);
        assert_eq!(mixed.result_sha256, mixed_chunked.result_sha256);
        let mixed_output = netcdf::open(&mixed_output_path).expect("open mixed output");
        assert_eq!(
            mixed_output
                .variable("analysis_status")
                .expect("mixed analysis status")
                .get_values::<i64, _>(..)
                .expect("read mixed analysis status"),
            [1, 0]
        );
        let mixed_major = mixed_output
            .variable("semi_major")
            .expect("mixed semi-major")
            .get_values::<f64, _>(..)
            .expect("read mixed semi-major");
        assert!(mixed_major[0].is_nan());
        assert!(mixed_major[1].is_finite());
        drop(mixed_output);
        fs::remove_file(input_path).expect("remove fixed-depth input");
        fs::remove_file(output_path).expect("remove fixed-depth output");
        fs::remove_file(chunked_output_path).expect("remove chunked fixed-depth output");
        fs::remove_file(unavailable_output_path).expect("remove unavailable fixed-depth output");
        fs::remove_file(mixed_output_path).expect("remove mixed fixed-depth output");
        fs::remove_file(mixed_chunked_output_path).expect("remove mixed chunked output");
    }

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
        let chunked_output_destination =
            std::env::temp_dir().join("rutide-vector-chunked-output-test.nc");
        let chunked_output_path =
            temporary_sibling(&chunked_output_destination).expect("valid chunked output path");
        let inference_output_destination =
            std::env::temp_dir().join("rutide-vector-inference-output-test.nc");
        let inference_output_path =
            temporary_sibling(&inference_output_destination).expect("valid inference output path");
        let layered_output_destination =
            std::env::temp_dir().join("rutide-vector-layered-output-test.nc");
        let layered_output_path =
            temporary_sibling(&layered_output_destination).expect("valid layered output path");
        let layered_chunked_output_destination =
            std::env::temp_dir().join("rutide-vector-layered-chunked-output-test.nc");
        let layered_chunked_output_path = temporary_sibling(&layered_chunked_output_destination)
            .expect("valid layered chunked output path");
        let time_count = 49_usize;
        let fill = -999.0_f32;
        let time_fill = -999_i32;
        let mut dataset = netcdf::create(&input_path).expect("create vector fixture");
        dataset.add_dimension("time", time_count).expect("add time");
        dataset.add_dimension("nele", 2).expect("add elements");
        dataset.add_dimension("siglay", 3).expect("add layers");
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
        let mut layered_eastward = Vec::with_capacity(time_count * 3 * 2);
        let mut layered_northward = Vec::with_capacity(time_count * 3 * 2);
        for time in 0..time_count {
            for layer in 0..3 {
                for element in 0..2 {
                    let source = time * 2 + element;
                    let eastward_value = eastward[source];
                    let northward_value = northward[source];
                    let layer_offset = [0.0_f32, 1.0, 2.0][layer];
                    layered_eastward.push(if eastward_value.to_bits() == fill.to_bits() {
                        fill
                    } else {
                        eastward_value + layer_offset
                    });
                    layered_northward.push(if northward_value.to_bits() == fill.to_bits() {
                        fill
                    } else {
                        northward_value - 0.5 * layer_offset
                    });
                }
            }
        }
        layered_northward[(8 * 3 + 2) * 2 + 1] = fill;
        {
            let mut variable = dataset
                .add_variable::<f32>("u", &["time", "siglay", "nele"])
                .expect("add u");
            variable.set_fill_value(fill).expect("set u fill");
            variable.put_values(&layered_eastward, ..).expect("write u");
        }
        {
            let mut variable = dataset
                .add_variable::<f32>("v", &["time", "siglay", "nele"])
                .expect("add v");
            variable.set_fill_value(fill).expect("set v fill");
            variable
                .put_values(&layered_northward, ..)
                .expect("write v");
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

        let pipelined_dataset = netcdf::open(&input_path).expect("open pipelined vector fixture");
        let pipelined_metadata = Arc::new(
            read_fvcom_vector_metadata(
                &input_path,
                &pipelined_dataset,
                &NodeSelection::Indices(vec![1, 0]),
                None,
                None,
            )
            .expect("read pipelined vector metadata"),
        );
        let mut pipelined =
            pipelined_vector_chunk_reader(pipelined_dataset, Arc::clone(&pipelined_metadata), 1)
                .expect("start pipelined reader");
        for series in 0..2 {
            let read = pipelined
                .next()
                .expect("read pipelined chunk")
                .expect("pipelined chunk exists");
            assert_eq!(read.first_series, series);
            assert_eq!(read.chunk.observation_counts, [47]);
            for time in 0..input.modified_julian_days.len() {
                let expected = time * 2 + series;
                for (actual, expected) in [
                    (read.chunk.eastward[time], input.eastward[expected]),
                    (read.chunk.northward[time], input.northward[expected]),
                ] {
                    assert!(
                        (actual.is_nan() && expected.is_nan())
                            || actual.to_bits() == expected.to_bits()
                    );
                }
            }
        }
        assert!(pipelined.next().expect("finish pipelined input").is_none());
        pipelined.finish().expect("join pipelined reader");

        // Dropping before receiving must cancel a reader blocked in the
        // rendezvous send rather than leaking or deadlocking its thread.
        let cancelled_dataset = netcdf::open(&input_path).expect("open cancelled vector fixture");
        let cancelled = pipelined_vector_chunk_reader(cancelled_dataset, pipelined_metadata, 1)
            .expect("start cancelled reader");
        drop(cancelled);

        let config = VectorAnalyzeConfig {
            input: input_path.clone(),
            output: output_path.clone(),
            report: None,
            elements: NodeSelection::Indices(vec![1, 0]),
            layers: None,
            fixed_depths_meters: None,
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
            constituent_diagnostics: Some(ConstituentDiagnosticsOptions::default()),
            reconstruction: Some(ReconstructionFilter::All),
            workers: 2,
            chunk_series: None,
            overwrite: false,
        };
        let report = analyze_vector(&config).expect("analyze vector fixture");
        assert_eq!(report.result_output, "buffered");
        assert_eq!(report.input_pipeline, "sequential");
        let mut chunked_config = config.clone();
        chunked_config.output = chunked_output_path.clone();
        chunked_config.chunk_series = Some(1);
        let chunked_report =
            analyze_vector(&chunked_config).expect("analyze vector fixture in one-series chunks");
        assert_eq!(chunked_report.chunk_series, 1);
        assert_eq!(chunked_report.chunk_count, 2);
        assert_eq!(chunked_report.input_pipeline, "sequential");
        assert_eq!(chunked_report.result_sha256, report.result_sha256);
        for name in [
            "semi_major",
            "semi_minor",
            "inclination",
            "phase",
            "diagnostic_basis_condition_number",
            "diagnostic_higher_maximum_correlation",
            "robust_weight",
            "eastward_reconstruction",
            "northward_reconstruction",
        ] {
            let whole = netcdf::open(&output_path)
                .expect("open whole-field vector output")
                .variable(name)
                .expect("whole-field variable")
                .get_values::<f64, _>(..)
                .expect("read whole-field variable");
            let chunked = netcdf::open(&chunked_output_path)
                .expect("open chunked vector output")
                .variable(name)
                .expect("chunked variable")
                .get_values::<f64, _>(..)
                .expect("read chunked variable");
            assert_eq!(chunked.len(), whole.len(), "chunked {name} shape differs");
            assert!(
                chunked.iter().zip(&whole).all(|(left, right)| {
                    (left.is_nan() && right.is_nan()) || left.to_bits() == right.to_bits()
                }),
                "chunked {name} differs"
            );
        }
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
        assert!(
            (report
                .constituent_diagnostics
                .as_ref()
                .expect("diagnostic report")
                .minimum_signal_to_noise
                - 2.0)
                .abs()
                < f64::EPSILON
        );
        let output = netcdf::open(&output_path).expect("open vector output");
        assert_eq!(
            output
                .attribute("input_pipeline")
                .expect("input pipeline metadata")
                .value()
                .expect("read input pipeline metadata"),
            netcdf::AttributeValue::Str("sequential".to_owned())
        );
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
                .variable("diagnostic_basis_condition_number")
                .expect("condition-number diagnostics")
                .len(),
            2
        );
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

        let mut layered_config = config.clone();
        layered_config.output = layered_output_path.clone();
        layered_config.layers = Some(NodeSelection::Indices(vec![2, 0]));
        let layered_report =
            analyze_vector(&layered_config).expect("analyze native sigma-layer currents");
        assert_eq!(
            layered_report.result_sha256,
            "0d480957ea28a127f32d81520a625a025fa160fb607a0958cd7d8b88d283fbf2"
        );
        let mut layered_chunked_config = layered_config.clone();
        layered_chunked_config.output = layered_chunked_output_path.clone();
        layered_chunked_config.chunk_series = Some(3);
        let layered_chunked_report = analyze_vector(&layered_chunked_config)
            .expect("analyze native sigma-layer currents across layer boundary");
        assert_eq!(layered_report.vertical_mode, "sigma-layer");
        assert_eq!(layered_report.element_count, 2);
        assert_eq!(layered_report.series_count, 4);
        assert_eq!(layered_report.layer_indices, Some(vec![2, 0]));
        assert_eq!(layered_report.sample_results[0].layer_index, Some(2));
        assert_eq!(layered_report.sample_results[0].element_index, 1);
        assert_eq!(layered_report.sample_results[2].layer_index, Some(0));
        assert_eq!(layered_chunked_report.chunk_series, 3);
        assert_eq!(layered_chunked_report.chunk_count, 2);
        assert_eq!(
            layered_chunked_report.result_sha256,
            layered_report.result_sha256
        );
        assert_eq!(layered_report.result_output, "incremental");
        assert_eq!(layered_chunked_report.result_output, "incremental");
        assert_eq!(
            layered_chunked_report.sampling.minimum_observation_count,
            layered_report.sampling.minimum_observation_count
        );
        assert_eq!(
            layered_chunked_report.sample_results[0].semi_major,
            layered_report.sample_results[0].semi_major
        );
        let layered_output = netcdf::open(&layered_output_path).expect("open layered output");
        assert_eq!(
            layered_output
                .attribute("result_output")
                .expect("incremental result-output metadata")
                .value()
                .expect("read result-output metadata"),
            netcdf::AttributeValue::Str("incremental".to_owned())
        );
        assert_eq!(
            layered_output
                .variable("siglay_index")
                .expect("layer coordinate")
                .get_values::<i64, _>(..)
                .expect("read layer coordinate"),
            [2, 0]
        );
        assert_eq!(
            layered_output
                .variable("element_index")
                .expect("element coordinate")
                .get_values::<i64, _>(..)
                .expect("read element coordinate"),
            [1, 0]
        );
        assert_eq!(
            layered_output
                .variable("observation_count")
                .expect("layered observation counts")
                .get_values::<i64, _>(..)
                .expect("read layered observation counts"),
            [46, 47, 47, 47]
        );
        assert_eq!(
            layered_output
                .variable("semi_major")
                .expect("layered semi-major")
                .dimensions()
                .iter()
                .map(netcdf::Dimension::name)
                .collect::<Vec<_>>(),
            ["siglay", "element", "constituent"]
        );
        assert_eq!(
            layered_output
                .variable("eastward_reconstruction")
                .expect("layered eastward reconstruction")
                .dimensions()
                .iter()
                .map(netcdf::Dimension::name)
                .collect::<Vec<_>>(),
            ["time", "siglay", "element"]
        );
        let eastward_means = layered_output
            .variable("eastward_mean")
            .expect("layered eastward means")
            .get_values::<f64, _>(..)
            .expect("read layered eastward means");
        let northward_means = layered_output
            .variable("northward_mean")
            .expect("layered northward means")
            .get_values::<f64, _>(..)
            .expect("read layered northward means");
        assert!((eastward_means[1] - eastward_means[3] - 2.0).abs() < 1e-6);
        assert!((northward_means[1] - northward_means[3] + 1.0).abs() < 1e-6);
        for name in [
            "semi_major",
            "semi_minor",
            "inclination",
            "phase",
            "robust_weight",
            "eastward_reconstruction",
            "northward_reconstruction",
        ] {
            let whole = layered_output
                .variable(name)
                .expect("whole layered variable")
                .get_values::<f64, _>(..)
                .expect("read whole layered variable");
            let chunked = netcdf::open(&layered_chunked_output_path)
                .expect("open chunked layered output")
                .variable(name)
                .expect("chunked layered variable")
                .get_values::<f64, _>(..)
                .expect("read chunked layered variable");
            assert_eq!(chunked, whole, "chunked layered {name} differs");
        }
        drop(layered_output);

        let inference_report = analyze_vector(&VectorAnalyzeConfig {
            input: input_path.clone(),
            output: inference_output_path.clone(),
            report: None,
            elements: NodeSelection::Indices(vec![1, 0]),
            layers: None,
            fixed_depths_meters: None,
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
                weight_function: RobustWeightFunction::Welsch,
                tolerance: 0.01,
                ..RobustOptions::default()
            }),
            constituent_diagnostics: None,
            reconstruction: Some(ReconstructionFilter::All),
            workers: 2,
            chunk_series: Some(1),
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
        assert_eq!(
            inference_report
                .robust_options
                .as_ref()
                .expect("robust options")
                .weight_function,
            "welsch"
        );
        assert!(!inference_report.trend_enabled);
        assert_eq!(inference_report.phase_reference, "raw");
        assert_eq!(inference_report.nodal_corrections, "linear-time");
        assert_eq!(inference_report.constituent_order, "signal-to-noise");
        assert_eq!(inference_report.confidence_interval, "monte-carlo");
        assert_eq!(inference_report.monte_carlo_realizations, Some(64));
        assert_eq!(inference_report.monte_carlo_seed, Some(99));
        assert_eq!(inference_report.chunk_count, 2);
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
                .attribute("robust_weight_function")
                .expect("robust weight metadata")
                .value()
                .expect("read robust weight metadata"),
            netcdf::AttributeValue::Str("welsch".to_owned())
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
        fs::remove_file(chunked_output_path).expect("remove chunked vector output");
        fs::remove_file(layered_output_path).expect("remove layered vector output");
        fs::remove_file(layered_chunked_output_path).expect("remove chunked layered vector output");
        fs::remove_file(inference_output_path).expect("remove inferred vector output");
    }
}
