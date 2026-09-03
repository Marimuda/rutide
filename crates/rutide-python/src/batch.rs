//! Time-major multi-series Python bindings.

mod persistence;

use std::sync::Arc;

use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
    ndarray::Array2,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rayon::{ThreadPool, ThreadPoolBuilder};
use rutide_core::{
    ConstituentDiagnosticsOptions, ConstituentSelectionDiagnostics, GreenwichNodalBatch,
    GreenwichNodalReconstructor, InferenceMode, ReconstructionFilter, RobustOptions,
    RobustTermination, ScalarInferenceBatch, ScalarSolution, SolverOptions, TidalConstituent,
    VectorInferenceBatch, VectorSolution, select_constituents_by_rayleigh,
};

use super::{
    Confidence, SolveConfig, constituent_order_from_diagnostics, normalized_confidence_name,
    parse_confidence, parse_constituents, parse_method_and_robust, parse_nodal_corrections,
    parse_phase_reference, reconstruction_filter, scalar_inference_relations,
    validate_empty_inference, vector_inference_relations,
};

enum PreparedBatchModel {
    Direct(Box<GreenwichNodalBatch>),
    ScalarInference(Box<ScalarInferenceBatch>),
    VectorInference(Box<VectorInferenceBatch>),
}

enum BatchSolutions {
    Scalar(Vec<ScalarSolution>),
    Vector(Vec<VectorSolution>),
}

/// Opaque shared model and multi-series solutions owned by a Python batch object.
#[pyclass(module = "rutide._native", frozen)]
pub(super) struct BatchFit {
    inner: Arc<BatchFitState>,
}

struct BatchFitState {
    model: PreparedBatchModel,
    solutions: BatchSolutions,
    time_mjd: Vec<f64>,
    config: SolveConfig,
    names: Vec<String>,
    frequencies: Vec<Vec<f64>>,
    presentation_order: Vec<Vec<usize>>,
    latitudes: Vec<f64>,
    valid_positions: Vec<Vec<usize>>,
    source_time_positions: Vec<usize>,
    original_time_count: usize,
    retained_time_count: usize,
    method: String,
    confidence: String,
    phase_reference: String,
    nodal_corrections: String,
    trend: bool,
    diagnostics: Option<Vec<ConstituentSelectionDiagnostics>>,
    worker_count: usize,
    chunk_series: usize,
    pool: Arc<ThreadPool>,
}

#[pymethods]
impl BatchFit {
    #[getter]
    fn is_vector(&self) -> bool {
        matches!(self.inner.solutions, BatchSolutions::Vector(_))
    }

    #[getter]
    fn series_count(&self) -> usize {
        self.inner.latitudes.len()
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.inner.summary(py)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        persistence::batch_snapshot(&self.inner, py)
    }
}

/// Fit time-major scalar or vector series through the shared Rust batch kernels.
#[allow(
    clippy::fn_params_excessive_bools,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "PyO3 exposes the UTide-compatible keyword surface directly"
)]
#[pyfunction]
pub(super) fn solve_many(
    py: Python<'_>,
    time_mjd: PyReadonlyArray1<'_, f64>,
    eastward: PyReadonlyArray2<'_, f64>,
    northward: Option<PyReadonlyArray2<'_, f64>>,
    latitudes: PyReadonlyArray1<'_, f64>,
    constituent_names: Option<Vec<String>>,
    rayleigh_min: f64,
    diagnostics: bool,
    diagnostic_min_signal_to_noise: f64,
    method_name: &str,
    confidence_name: &str,
    white: bool,
    trend: bool,
    phase_name: &str,
    nodal_name: &str,
    monte_carlo_realizations: usize,
    monte_carlo_seed: u64,
    robust_weight_name: &str,
    robust_tuning: Option<f64>,
    robust_tolerance: f64,
    robust_max_iterations: usize,
    inferred_names: Vec<String>,
    reference_names: Vec<String>,
    inference_ratios: Vec<f64>,
    inference_phase_offsets: Vec<f64>,
    approximate_inference: bool,
    order_name: &str,
    order_names: Vec<String>,
    workers: Option<usize>,
    memory_limit_bytes: Option<usize>,
) -> PyResult<Py<BatchFit>> {
    let eastward_shape = matrix_shape(&eastward, "u")?;
    let northward_shape = northward
        .as_ref()
        .map(|values| matrix_shape(values, "v"))
        .transpose()?;
    let time = contiguous_copy_1(&time_mjd, "time_mjd")?;
    let eastward = contiguous_copy_2(&eastward, "u")?;
    let northward = northward
        .as_ref()
        .map(|values| contiguous_copy_2(values, "v"))
        .transpose()?;
    let latitudes = contiguous_copy_1(&latitudes, "lat")?;
    let config = SolveConfig {
        latitude: 0.0,
        constituent_names,
        rayleigh_min,
        diagnostics,
        diagnostic_min_signal_to_noise,
        method_name: method_name.to_owned(),
        confidence_name: confidence_name.to_owned(),
        white,
        trend,
        phase_name: phase_name.to_owned(),
        nodal_name: nodal_name.to_owned(),
        monte_carlo_realizations,
        monte_carlo_seed,
        robust_weight_name: robust_weight_name.to_owned(),
        robust_tuning,
        robust_tolerance,
        robust_max_iterations,
        inferred_names,
        reference_names,
        inference_ratios,
        inference_phase_offsets,
        approximate_inference,
        order_name: order_name.to_owned(),
        order_names,
    };
    let fit = py
        .detach(move || {
            solve_many_native(
                time,
                eastward,
                northward,
                eastward_shape,
                northward_shape,
                latitudes,
                &config,
                workers,
                memory_limit_bytes,
            )
        })
        .map_err(PyValueError::new_err)?;
    Py::new(py, fit)
}

type ReconstructionMatrices<'py> = (
    Option<Bound<'py, PyArray2<f64>>>,
    Option<Bound<'py, PyArray2<f64>>>,
    Option<Bound<'py, PyArray2<f64>>>,
);

/// Reconstruct all solutions in a native batch at shared target timestamps.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub(super) fn reconstruct_many<'py>(
    py: Python<'py>,
    time_mjd: PyReadonlyArray1<'_, f64>,
    fit: PyRef<'_, BatchFit>,
    constituent_names: Option<Vec<String>>,
    minimum_signal_to_noise: Option<f64>,
    minimum_percent_energy: f64,
) -> PyResult<ReconstructionMatrices<'py>> {
    let time = contiguous_copy_1(&time_mjd, "time_mjd")?;
    let filter = reconstruction_filter(
        constituent_names,
        minimum_signal_to_noise,
        minimum_percent_energy,
    )
    .map_err(PyValueError::new_err)?;
    let state = Arc::clone(&fit.inner);
    drop(fit);
    let reconstructed = py
        .detach(move || state.reconstruct(&time, &filter))
        .map_err(PyValueError::new_err)?;
    match reconstructed {
        ReconstructedBatch::Scalar {
            time_count,
            series_count,
            values,
        } => Ok((
            Some(array2(time_count, series_count, values)?.into_pyarray(py)),
            None,
            None,
        )),
        ReconstructedBatch::Vector {
            time_count,
            series_count,
            eastward,
            northward,
        } => Ok((
            None,
            Some(array2(time_count, series_count, eastward)?.into_pyarray(py)),
            Some(array2(time_count, series_count, northward)?.into_pyarray(py)),
        )),
    }
}

/// Restore a versioned native batch snapshot without repeating the fit.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub(super) fn restore_batch(
    py: Python<'_>,
    snapshot: &Bound<'_, PyDict>,
    workers: Option<usize>,
) -> PyResult<Py<BatchFit>> {
    persistence::restore_batch(py, snapshot, workers)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn solve_many_native(
    time: Vec<f64>,
    eastward: Vec<f64>,
    northward: Option<Vec<f64>>,
    eastward_shape: (usize, usize),
    northward_shape: Option<(usize, usize)>,
    latitudes: Vec<f64>,
    config: &SolveConfig,
    workers: Option<usize>,
    memory_limit_bytes: Option<usize>,
) -> Result<BatchFit, String> {
    let (original_time_count, series_count) = eastward_shape;
    if time.len() != original_time_count {
        return Err(format!(
            "t contains {} rows but u contains {original_time_count}",
            time.len()
        ));
    }
    if series_count == 0 {
        return Err("u must contain at least one series".to_owned());
    }
    if latitudes.len() != series_count {
        return Err(format!(
            "lat contains {} values but u contains {series_count} series",
            latitudes.len()
        ));
    }
    if let Some(shape) = northward_shape
        && shape != eastward_shape
    {
        return Err(format!(
            "v shape ({}, {}) does not match u shape ({}, {})",
            shape.0, shape.1, eastward_shape.0, eastward_shape.1
        ));
    }
    if eastward.iter().any(|value| value.is_infinite())
        || northward
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| value.is_infinite()))
    {
        return Err("observations may contain NaN missing values, but not infinity".to_owned());
    }
    let (time, eastward, northward, source_time_positions) =
        compact_finite_time_rows(time, eastward, northward, original_time_count, series_count);
    if time.is_empty() {
        return Err("no finite timestamps remain".to_owned());
    }

    let constituents = match config.constituent_names.as_ref() {
        Some(names) => parse_constituents(names)?,
        None => {
            select_constituents_by_rayleigh(&time, config.rayleigh_min)
                .map_err(|error| error.to_string())?
                .constituents
        }
    };
    let phase_reference = parse_phase_reference(&config.phase_name)?;
    let nodal_corrections = parse_nodal_corrections(&config.nodal_name)?;
    let solver_options = SolverOptions::new(
        rutide_core::FitOptions {
            trend: config.trend,
        },
        phase_reference,
    )
    .with_nodal_corrections(nodal_corrections);
    let confidence = parse_confidence(
        &config.confidence_name,
        config.white,
        config.monte_carlo_realizations,
        config.monte_carlo_seed,
    )?;
    if config.diagnostics && matches!(confidence, Confidence::None) {
        return Err(
            "diagnostics=True requires confidence intervals (conf_int='linear' or 'MC')".to_owned(),
        );
    }
    let robust = parse_method_and_robust(config)?;
    let inference_mode = if config.approximate_inference {
        InferenceMode::Approximate
    } else {
        InferenceMode::Exact
    };
    let is_vector = northward.is_some();
    let model = if config.inferred_names.is_empty() {
        validate_empty_inference(config)?;
        PreparedBatchModel::Direct(Box::new(
            GreenwichNodalBatch::prepare_modified_julian_days_with_solver_options(
                &time,
                &constituents,
                solver_options,
            )
            .map_err(|error| error.to_string())?,
        ))
    } else if is_vector {
        let relationships = vector_inference_relations(config)?;
        PreparedBatchModel::VectorInference(Box::new(
            VectorInferenceBatch::prepare_modified_julian_days_with_solver_options(
                &time,
                &constituents,
                &relationships,
                inference_mode,
                solver_options,
            )
            .map_err(|error| error.to_string())?,
        ))
    } else {
        let relationships = scalar_inference_relations(config)?;
        PreparedBatchModel::ScalarInference(Box::new(
            ScalarInferenceBatch::prepare_modified_julian_days_with_solver_options(
                &time,
                &constituents,
                &relationships,
                inference_mode,
                solver_options,
            )
            .map_err(|error| error.to_string())?,
        ))
    };

    let worker_count = requested_worker_count(workers, series_count)?;
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|index| format!("rutide-python-{index}"))
            .build()
            .map_err(|error| format!("worker-pool error: {error}"))?,
    );
    let component_count = usize::from(is_vector) + 1;
    let chunk_series = working_chunk_series(
        time.len(),
        series_count,
        component_count,
        memory_limit_bytes,
    )?;
    let solutions = match (&model, northward.as_deref()) {
        (PreparedBatchModel::Direct(model), Some(northward)) => {
            BatchSolutions::Vector(solve_vector_chunks(
                model.as_ref(),
                &eastward,
                northward,
                &latitudes,
                robust,
                confidence,
                chunk_series,
                &pool,
            )?)
        }
        (PreparedBatchModel::Direct(model), None) => BatchSolutions::Scalar(solve_scalar_chunks(
            model.as_ref(),
            &eastward,
            &latitudes,
            robust,
            confidence,
            chunk_series,
            &pool,
        )?),
        (PreparedBatchModel::ScalarInference(model), None) => {
            BatchSolutions::Scalar(solve_scalar_chunks(
                model.as_ref(),
                &eastward,
                &latitudes,
                robust,
                confidence,
                chunk_series,
                &pool,
            )?)
        }
        (PreparedBatchModel::VectorInference(model), Some(northward)) => {
            BatchSolutions::Vector(solve_vector_chunks(
                model.as_ref(),
                &eastward,
                northward,
                &latitudes,
                robust,
                confidence,
                chunk_series,
                &pool,
            )?)
        }
        _ => return Err("internal batch model and observation type mismatch".to_owned()),
    };

    let names = model
        .tidal_constituents()
        .iter()
        .map(|constituent| constituent.name().to_owned())
        .collect::<Vec<_>>();
    let frequencies = solutions
        .reference_times()
        .iter()
        .map(|reference_time| {
            model
                .constituents_at_reference(*reference_time)
                .map(|constituents| {
                    constituents
                        .into_iter()
                        .map(|constituent| constituent.frequency_cph)
                        .collect()
                })
        })
        .collect::<Result<Vec<Vec<_>>, String>>()?;
    let presentation_order = solutions
        .diagnostics()
        .zip(&frequencies)
        .map(|((percent_energy, signal_to_noise), frequency)| {
            constituent_order_from_diagnostics(
                &config.order_name,
                &config.order_names,
                &names,
                frequency,
                percent_energy,
                signal_to_noise,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let valid_positions =
        valid_observation_positions(&eastward, northward.as_deref(), time.len(), series_count);
    let diagnostics = if config.diagnostics {
        let options = ConstituentDiagnosticsOptions::default()
            .with_rayleigh_minimum(config.rayleigh_min)
            .with_minimum_signal_to_noise(config.diagnostic_min_signal_to_noise);
        Some(pool.install(|| {
            model.diagnose(
                &eastward,
                northward.as_deref(),
                &latitudes,
                &solutions,
                options,
            )
        })?)
    } else {
        None
    };
    let mut stored_config = config.clone();
    stored_config.constituent_names = Some(
        constituents
            .iter()
            .map(|constituent| constituent.name().to_owned())
            .collect(),
    );

    let retained_time_count = time.len();
    Ok(BatchFit {
        inner: Arc::new(BatchFitState {
            model,
            solutions,
            time_mjd: time,
            config: stored_config,
            names,
            frequencies,
            presentation_order,
            latitudes,
            valid_positions,
            source_time_positions,
            original_time_count,
            retained_time_count,
            method: config.method_name.to_ascii_lowercase(),
            confidence: normalized_confidence_name(confidence).to_owned(),
            phase_reference: phase_reference.name().to_owned(),
            nodal_corrections: nodal_corrections.name().to_owned(),
            trend: config.trend,
            diagnostics,
            worker_count,
            chunk_series,
            pool,
        }),
    })
}

trait ScalarBatchModel {
    fn solve_missing(
        &self,
        observations: &[f64],
        latitudes: &[f64],
        robust: Option<RobustOptions>,
        confidence: Confidence,
        stream_offset: u64,
    ) -> Result<Vec<ScalarSolution>, String>;
}

macro_rules! impl_scalar_batch_model {
    ($model:ty) => {
        impl ScalarBatchModel for $model {
            fn solve_missing(
                &self,
                observations: &[f64],
                latitudes: &[f64],
                robust: Option<RobustOptions>,
                confidence: Confidence,
                stream_offset: u64,
            ) -> Result<Vec<ScalarSolution>, String> {
                let result = match (robust, confidence) {
                    (None, Confidence::None) => {
                        self.solve_time_major_with_missing(observations, latitudes)
                    }
                    (None, Confidence::Linear(noise)) => self
                        .solve_time_major_with_missing_and_linear_confidence(
                            observations,
                            latitudes,
                            noise,
                        ),
                    (None, Confidence::MonteCarlo(options, noise)) => self
                        .solve_time_major_with_missing_and_monte_carlo_confidence_with_stream_offset(
                            observations,
                            latitudes,
                            options,
                            noise,
                            stream_offset,
                        ),
                    (Some(options), Confidence::None) => self
                        .solve_time_major_with_missing_robust(observations, latitudes, options),
                    (Some(options), Confidence::Linear(noise)) => self
                        .solve_time_major_with_missing_robust_and_linear_confidence(
                            observations,
                            latitudes,
                            options,
                            noise,
                        ),
                    (Some(robust), Confidence::MonteCarlo(monte_carlo, noise)) => self
                        .solve_time_major_with_missing_robust_and_monte_carlo_confidence_with_stream_offset(
                            observations,
                            latitudes,
                            robust,
                            monte_carlo,
                            noise,
                            stream_offset,
                        ),
                };
                result.map_err(|error| error.to_string())
            }
        }
    };
}

impl_scalar_batch_model!(GreenwichNodalBatch);
impl_scalar_batch_model!(ScalarInferenceBatch);

trait VectorBatchModel {
    #[allow(clippy::too_many_arguments)]
    fn solve_missing_vector(
        &self,
        eastward: &[f64],
        northward: &[f64],
        latitudes: &[f64],
        robust: Option<RobustOptions>,
        confidence: Confidence,
        stream_offset: u64,
    ) -> Result<Vec<VectorSolution>, String>;
}

macro_rules! impl_vector_batch_model {
    ($model:ty) => {
        impl VectorBatchModel for $model {
            fn solve_missing_vector(
                &self,
                eastward: &[f64],
                northward: &[f64],
                latitudes: &[f64],
                robust: Option<RobustOptions>,
                confidence: Confidence,
                stream_offset: u64,
            ) -> Result<Vec<VectorSolution>, String> {
                let result = match (robust, confidence) {
                    (None, Confidence::None) => self.solve_vector_time_major_with_missing(
                        eastward, northward, latitudes,
                    ),
                    (None, Confidence::Linear(noise)) => self
                        .solve_vector_time_major_with_missing_and_linear_confidence(
                            eastward,
                            northward,
                            latitudes,
                            noise,
                        ),
                    (None, Confidence::MonteCarlo(options, noise)) => self
                        .solve_vector_time_major_with_missing_and_monte_carlo_confidence_with_stream_offset(
                            eastward,
                            northward,
                            latitudes,
                            options,
                            noise,
                            stream_offset,
                        ),
                    (Some(options), Confidence::None) => self
                        .solve_vector_time_major_with_missing_robust(
                            eastward,
                            northward,
                            latitudes,
                            options,
                        ),
                    (Some(options), Confidence::Linear(noise)) => self
                        .solve_vector_time_major_with_missing_robust_and_linear_confidence(
                            eastward,
                            northward,
                            latitudes,
                            options,
                            noise,
                        ),
                    (Some(robust), Confidence::MonteCarlo(monte_carlo, noise)) => self
                        .solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence_with_stream_offset(
                            eastward,
                            northward,
                            latitudes,
                            robust,
                            monte_carlo,
                            noise,
                            stream_offset,
                        ),
                };
                result.map_err(|error| error.to_string())
            }
        }
    };
}

impl_vector_batch_model!(GreenwichNodalBatch);
impl_vector_batch_model!(VectorInferenceBatch);

fn solve_scalar_chunks<M: ScalarBatchModel + Sync>(
    model: &M,
    observations: &[f64],
    latitudes: &[f64],
    robust: Option<RobustOptions>,
    confidence: Confidence,
    chunk_series: usize,
    pool: &ThreadPool,
) -> Result<Vec<ScalarSolution>, String> {
    let time_count = observations.len() / latitudes.len();
    let series_count = latitudes.len();
    let mut solutions = Vec::with_capacity(series_count);
    for start in (0..series_count).step_by(chunk_series) {
        let end = start.saturating_add(chunk_series).min(series_count);
        let storage = (end - start != series_count)
            .then(|| time_major_chunk(observations, time_count, series_count, start, end));
        let values = storage.as_deref().unwrap_or(observations);
        let mut chunk = pool.install(|| {
            model.solve_missing(
                values,
                &latitudes[start..end],
                robust,
                confidence,
                u64::try_from(start).expect("series index is representable as u64"),
            )
        })?;
        solutions.append(&mut chunk);
    }
    Ok(solutions)
}

#[allow(clippy::too_many_arguments)]
fn solve_vector_chunks<M: VectorBatchModel + Sync>(
    model: &M,
    eastward: &[f64],
    northward: &[f64],
    latitudes: &[f64],
    robust: Option<RobustOptions>,
    confidence: Confidence,
    chunk_series: usize,
    pool: &ThreadPool,
) -> Result<Vec<VectorSolution>, String> {
    let time_count = eastward.len() / latitudes.len();
    let series_count = latitudes.len();
    let mut solutions = Vec::with_capacity(series_count);
    for start in (0..series_count).step_by(chunk_series) {
        let end = start.saturating_add(chunk_series).min(series_count);
        let eastward_storage = (end - start != series_count)
            .then(|| time_major_chunk(eastward, time_count, series_count, start, end));
        let northward_storage = (end - start != series_count)
            .then(|| time_major_chunk(northward, time_count, series_count, start, end));
        let eastward_values = eastward_storage.as_deref().unwrap_or(eastward);
        let northward_values = northward_storage.as_deref().unwrap_or(northward);
        let mut chunk = pool.install(|| {
            model.solve_missing_vector(
                eastward_values,
                northward_values,
                &latitudes[start..end],
                robust,
                confidence,
                u64::try_from(start).expect("series index is representable as u64"),
            )
        })?;
        solutions.append(&mut chunk);
    }
    Ok(solutions)
}

impl PreparedBatchModel {
    fn tidal_constituents(&self) -> &[TidalConstituent] {
        match self {
            Self::Direct(model) => model.tidal_constituents(),
            Self::ScalarInference(model) => model.tidal_constituents(),
            Self::VectorInference(model) => model.tidal_constituents(),
        }
    }

    fn constituents_at_reference(
        &self,
        reference_time: f64,
    ) -> Result<Vec<rutide_core::Constituent>, String> {
        match self {
            Self::Direct(model) => {
                model.constituents_at_reference_modified_julian_day(reference_time)
            }
            Self::ScalarInference(model) => {
                model.constituents_at_reference_modified_julian_day(reference_time)
            }
            Self::VectorInference(model) => {
                model.constituents_at_reference_modified_julian_day(reference_time)
            }
        }
        .map_err(|error| error.to_string())
    }

    fn reconstructor(&self, time_mjd: &[f64]) -> Result<GreenwichNodalReconstructor, String> {
        match self {
            Self::Direct(model) => model.reconstructor_modified_julian_days(time_mjd),
            Self::ScalarInference(model) => model.reconstructor_modified_julian_days(time_mjd),
            Self::VectorInference(model) => model.reconstructor_modified_julian_days(time_mjd),
        }
        .map_err(|error| error.to_string())
    }

    fn diagnose(
        &self,
        eastward: &[f64],
        northward: Option<&[f64]>,
        latitudes: &[f64],
        solutions: &BatchSolutions,
        options: ConstituentDiagnosticsOptions,
    ) -> Result<Vec<ConstituentSelectionDiagnostics>, String> {
        match (self, solutions, northward) {
            (Self::Direct(model), BatchSolutions::Scalar(solutions), None) => {
                model.diagnose_time_major(eastward, latitudes, solutions, options)
            }
            (Self::Direct(model), BatchSolutions::Vector(solutions), Some(northward)) => {
                model.diagnose_vector_time_major(eastward, northward, latitudes, solutions, options)
            }
            (Self::ScalarInference(model), BatchSolutions::Scalar(solutions), None) => {
                model.diagnose_time_major(eastward, latitudes, solutions, options)
            }
            (Self::VectorInference(model), BatchSolutions::Vector(solutions), Some(northward)) => {
                model.diagnose_vector_time_major(eastward, northward, latitudes, solutions, options)
            }
            _ => return Err("internal batch model, solution, and observation mismatch".to_owned()),
        }
        .map_err(|error| error.to_string())
    }
}

impl BatchSolutions {
    fn len(&self) -> usize {
        match self {
            Self::Scalar(solutions) => solutions.len(),
            Self::Vector(solutions) => solutions.len(),
        }
    }

    fn reference_times(&self) -> Vec<f64> {
        match self {
            Self::Scalar(solutions) => solutions
                .iter()
                .map(|solution| solution.reference_time_days)
                .collect(),
            Self::Vector(solutions) => solutions
                .iter()
                .map(|solution| solution.reference_time_days)
                .collect(),
        }
    }

    fn diagnostics(&self) -> BatchDiagnostics<'_> {
        match self {
            Self::Scalar(solutions) => BatchDiagnostics::Scalar(solutions.iter()),
            Self::Vector(solutions) => BatchDiagnostics::Vector(solutions.iter()),
        }
    }
}

enum BatchDiagnostics<'a> {
    Scalar(std::slice::Iter<'a, ScalarSolution>),
    Vector(std::slice::Iter<'a, VectorSolution>),
}

impl<'a> Iterator for BatchDiagnostics<'a> {
    type Item = (&'a [f64], Option<&'a [f64]>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Scalar(solutions) => solutions.next().map(|solution| {
                (
                    solution.percent_energy.as_slice(),
                    solution.signal_to_noise.as_deref(),
                )
            }),
            Self::Vector(solutions) => solutions.next().map(|solution| {
                (
                    solution.percent_energy.as_slice(),
                    solution.signal_to_noise.as_deref(),
                )
            }),
        }
    }
}

enum ReconstructedBatch {
    Scalar {
        time_count: usize,
        series_count: usize,
        values: Vec<f64>,
    },
    Vector {
        time_count: usize,
        series_count: usize,
        eastward: Vec<f64>,
        northward: Vec<f64>,
    },
}

impl BatchFitState {
    fn reconstruct(
        &self,
        time_mjd: &[f64],
        filter: &ReconstructionFilter,
    ) -> Result<ReconstructedBatch, String> {
        let reconstructor = self.model.reconstructor(time_mjd)?;
        let series_count = self.latitudes.len();
        let time_count = time_mjd.len();
        match &self.solutions {
            BatchSolutions::Scalar(solutions) => {
                let series_major = self
                    .pool
                    .install(|| {
                        reconstructor.reconstruct_many_series_major(
                            solutions,
                            &self.latitudes,
                            filter,
                        )
                    })
                    .map_err(|error| error.to_string())?;
                Ok(ReconstructedBatch::Scalar {
                    time_count,
                    series_count,
                    values: series_to_time_major(&series_major, time_count),
                })
            }
            BatchSolutions::Vector(solutions) => {
                let series_major = self
                    .pool
                    .install(|| {
                        reconstructor.reconstruct_many_vectors_series_major(
                            solutions,
                            &self.latitudes,
                            filter,
                        )
                    })
                    .map_err(|error| error.to_string())?;
                let mut eastward = vec![0.0; time_count * series_count];
                let mut northward = vec![0.0; time_count * series_count];
                for (series, values) in series_major.iter().enumerate() {
                    for time in 0..time_count {
                        eastward[time * series_count + series] = values.eastward[time];
                        northward[time * series_count + series] = values.northward[time];
                    }
                }
                Ok(ReconstructedBatch::Vector {
                    time_count,
                    series_count,
                    eastward,
                    northward,
                })
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        let series_count = self.solutions.len();
        let constituent_count = self.names.len();
        let rank_count = self.presentation_order.first().map_or(0, Vec::len);
        output.set_item("name", &self.names)?;
        output.set_item(
            "frequency_cph",
            array2(
                series_count,
                constituent_count,
                self.frequencies.iter().flatten().copied().collect(),
            )?
            .into_pyarray(py),
        )?;
        output.set_item(
            "frequency_varies_by_series",
            self.frequencies
                .windows(2)
                .any(|pair| pair[0].as_slice() != pair[1].as_slice()),
        )?;
        output.set_item(
            "rank_index",
            array2(
                series_count,
                rank_count,
                self.presentation_order.iter().flatten().copied().collect(),
            )?
            .into_pyarray(py),
        )?;
        output.set_item("method", &self.method)?;
        output.set_item("confidence", &self.confidence)?;
        output.set_item("phase_reference", &self.phase_reference)?;
        output.set_item("nodal_corrections", &self.nodal_corrections)?;
        output.set_item("trend", self.trend)?;
        output.set_item("series_count", series_count)?;
        output.set_item("nobs_original", self.original_time_count)?;
        output.set_item(
            "nobs",
            self.valid_positions
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        output.set_item("worker_count", self.worker_count)?;
        output.set_item("chunk_series", self.chunk_series)?;

        let auxiliary = PyDict::new(py);
        auxiliary.set_item(
            "frq",
            array2(
                series_count,
                constituent_count,
                self.frequencies.iter().flatten().copied().collect(),
            )?
            .into_pyarray(py),
        )?;
        auxiliary.set_item("lat", self.latitudes.clone().into_pyarray(py))?;
        auxiliary.set_item("reftime", self.solutions.reference_times().into_pyarray(py))?;
        auxiliary.set_item("nobs_original", self.original_time_count)?;
        auxiliary.set_item(
            "nobs",
            self.valid_positions
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        auxiliary.set_item(
            "time_position",
            self.source_time_positions.clone().into_pyarray(py),
        )?;
        output.set_item("aux", auxiliary)?;

        match &self.solutions {
            BatchSolutions::Scalar(solutions) => {
                add_scalar_summary(py, &output, solutions, constituent_count)?;
            }
            BatchSolutions::Vector(solutions) => {
                add_vector_summary(py, &output, solutions, constituent_count)?;
            }
        }
        add_robust_summary(py, &output, self)?;
        add_batch_diagnostics_summary(py, &output, self.diagnostics.as_deref(), constituent_count)?;
        Ok(output)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "keeps the compact MATLAB-compatible diagnostic field mapping together"
)]
fn add_batch_diagnostics_summary(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    diagnostics: Option<&[ConstituentSelectionDiagnostics]>,
    constituent_count: usize,
) -> PyResult<()> {
    let Some(diagnostics) = diagnostics else {
        output.set_item("diagn", py.None())?;
        output.set_item("diagnostics", py.None())?;
        return Ok(());
    };
    let series_count = diagnostics.len();
    if diagnostics
        .iter()
        .any(|value| value.constituents.len() != constituent_count)
    {
        return Err(PyValueError::new_err(
            "internal diagnostic constituent counts are inconsistent",
        ));
    }
    let summary = PyDict::new(py);
    let lower = batch_neighbor_diagnostics_summary(
        py,
        diagnostics,
        series_count,
        constituent_count,
        false,
    )?;
    let higher =
        batch_neighbor_diagnostics_summary(py, diagnostics, series_count, constituent_count, true)?;
    summary.set_item("lo", &lower)?;
    summary.set_item("lower", lower)?;
    summary.set_item("hi", &higher)?;
    summary.set_item("higher", higher)?;
    set_diagnostic_vector(
        py,
        &summary,
        "K",
        diagnostics
            .iter()
            .map(|value| value.whole_model.basis_condition_number),
    )?;
    set_diagnostic_vector(
        py,
        &summary,
        "SNRallc",
        diagnostics.iter().map(|value| {
            value
                .whole_model
                .all_constituent_signal_to_noise
                .unwrap_or(f64::NAN)
        }),
    )?;
    set_diagnostic_vector(
        py,
        &summary,
        "SNRallc_over_K",
        diagnostics.iter().map(|value| {
            value
                .whole_model
                .condition_adjusted_signal_to_noise
                .unwrap_or(f64::NAN)
        }),
    )?;
    for (name, values) in [
        (
            "TVraw",
            diagnostics
                .iter()
                .map(|value| value.tidal_variance.raw_tidal_variance)
                .collect::<Vec<_>>(),
        ),
        (
            "TVallc",
            diagnostics
                .iter()
                .map(|value| value.tidal_variance.all_constituent_tidal_variance)
                .collect(),
        ),
        (
            "TVsnrc",
            diagnostics
                .iter()
                .map(|value| {
                    value
                        .tidal_variance
                        .significant_constituent_tidal_variance
                        .unwrap_or(f64::NAN)
                })
                .collect(),
        ),
        (
            "PTVallc",
            diagnostics
                .iter()
                .map(|value| {
                    value
                        .tidal_variance
                        .all_constituent_percent_tidal_variance
                        .unwrap_or(f64::NAN)
                })
                .collect(),
        ),
        (
            "PTVsnrc",
            diagnostics
                .iter()
                .map(|value| {
                    value
                        .tidal_variance
                        .significant_constituent_percent_tidal_variance
                        .unwrap_or(f64::NAN)
                })
                .collect(),
        ),
    ] {
        summary.set_item(name, values.into_pyarray(py))?;
    }
    set_diagnostic_vector(
        py,
        &summary,
        "Rayleigh_min",
        diagnostics.iter().map(|value| value.rayleigh_minimum),
    )?;
    set_diagnostic_vector(
        py,
        &summary,
        "min_SNR",
        diagnostics
            .iter()
            .map(|value| value.minimum_signal_to_noise),
    )?;
    output.set_item("diagn", &summary)?;
    output.set_item("diagnostics", summary)?;
    Ok(())
}

fn set_diagnostic_vector(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    name: &str,
    values: impl Iterator<Item = f64>,
) -> PyResult<()> {
    output.set_item(name, values.collect::<Vec<_>>().into_pyarray(py))
}

fn batch_neighbor_diagnostics_summary<'py>(
    py: Python<'py>,
    diagnostics: &[ConstituentSelectionDiagnostics],
    series_count: usize,
    constituent_count: usize,
    higher: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut indices = Vec::with_capacity(series_count * constituent_count);
    let mut frequencies = Vec::with_capacity(series_count * constituent_count);
    let mut rayleigh = Vec::with_capacity(series_count * constituent_count);
    let mut noise_modified_rayleigh = Vec::with_capacity(series_count * constituent_count);
    let mut maximum_correlation = Vec::with_capacity(series_count * constituent_count);
    for diagnostic in diagnostics {
        for independence in &diagnostic.constituents {
            let neighbor = if higher {
                independence.higher.as_ref()
            } else {
                independence.lower.as_ref()
            };
            if let Some(neighbor) = neighbor {
                indices.push(i64::try_from(neighbor.index).expect("constituent count fits i64"));
                frequencies.push(neighbor.frequency_cph);
                rayleigh.push(neighbor.rayleigh_criterion);
                noise_modified_rayleigh.push(
                    neighbor
                        .noise_modified_rayleigh_criterion
                        .unwrap_or(f64::NAN),
                );
                maximum_correlation.push(neighbor.maximum_correlation.unwrap_or(f64::NAN));
            } else {
                indices.push(-1);
                frequencies.push(f64::NAN);
                rayleigh.push(f64::NAN);
                noise_modified_rayleigh.push(f64::NAN);
                maximum_correlation.push(f64::NAN);
            }
        }
    }
    let summary = PyDict::new(py);
    summary.set_item(
        "index",
        array2(series_count, constituent_count, indices)?.into_pyarray(py),
    )?;
    summary.set_item(
        "frequency_cph",
        array2(series_count, constituent_count, frequencies)?.into_pyarray(py),
    )?;
    let rayleigh = array2(series_count, constituent_count, rayleigh)?.into_pyarray(py);
    summary.set_item("RR", &rayleigh)?;
    summary.set_item("rayleigh_criterion", rayleigh)?;
    let noise_modified_rayleigh =
        array2(series_count, constituent_count, noise_modified_rayleigh)?.into_pyarray(py);
    summary.set_item("RNM", &noise_modified_rayleigh)?;
    summary.set_item("noise_modified_rayleigh_criterion", noise_modified_rayleigh)?;
    let maximum_correlation =
        array2(series_count, constituent_count, maximum_correlation)?.into_pyarray(py);
    summary.set_item("CorMx", &maximum_correlation)?;
    summary.set_item("maximum_correlation", maximum_correlation)?;
    Ok(summary)
}

fn add_scalar_summary(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    solutions: &[ScalarSolution],
    constituent_count: usize,
) -> PyResult<()> {
    let series_count = solutions.len();
    let amplitude =
        matrix_from_scalar(solutions, constituent_count, |solution| &solution.amplitude)?
            .into_pyarray(py);
    let phase = matrix_from_scalar(solutions, constituent_count, |solution| {
        &solution.phase_degrees
    })?
    .into_pyarray(py);
    let percent_energy = matrix_from_scalar(solutions, constituent_count, |solution| {
        &solution.percent_energy
    })?
    .into_pyarray(py);
    output.set_item("A", &amplitude)?;
    output.set_item("amplitude", &amplitude)?;
    output.set_item("g", &phase)?;
    output.set_item("phase_degrees", &phase)?;
    output.set_item("PE", &percent_energy)?;
    output.set_item("percent_energy", &percent_energy)?;
    set_optional_scalar_matrix(
        py,
        output,
        solutions,
        constituent_count,
        "A_ci",
        "amplitude_ci",
        |solution| solution.amplitude_ci.as_deref(),
    )?;
    set_optional_scalar_matrix(
        py,
        output,
        solutions,
        constituent_count,
        "g_ci",
        "phase_ci_degrees",
        |solution| solution.phase_ci_degrees.as_deref(),
    )?;
    set_optional_scalar_matrix(
        py,
        output,
        solutions,
        constituent_count,
        "SNR",
        "signal_to_noise",
        |solution| solution.signal_to_noise.as_deref(),
    )?;
    output.set_item(
        "mean",
        solutions
            .iter()
            .map(|solution| solution.mean)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    output.set_item(
        "slope",
        solutions
            .iter()
            .map(|solution| solution.slope_per_day)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    output.set_item(
        "reference_time_mjd",
        solutions
            .iter()
            .map(|solution| solution.reference_time_days)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    debug_assert_eq!(amplitude.shape(), [series_count, constituent_count]);
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "keeps the UTide ellipse fields and their descriptive aliases together"
)]
fn add_vector_summary(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    solutions: &[VectorSolution],
    constituent_count: usize,
) -> PyResult<()> {
    let semi_major = matrix_from_vector(solutions, constituent_count, |solution| {
        &solution.semi_major
    })?
    .into_pyarray(py);
    let semi_minor = matrix_from_vector(solutions, constituent_count, |solution| {
        &solution.semi_minor
    })?
    .into_pyarray(py);
    let inclination = matrix_from_vector(solutions, constituent_count, |solution| {
        &solution.inclination_degrees
    })?
    .into_pyarray(py);
    let phase = matrix_from_vector(solutions, constituent_count, |solution| {
        &solution.phase_degrees
    })?
    .into_pyarray(py);
    let percent_energy = matrix_from_vector(solutions, constituent_count, |solution| {
        &solution.percent_energy
    })?
    .into_pyarray(py);
    for (short, descriptive, values) in [
        ("Lsmaj", "semi_major", &semi_major),
        ("Lsmin", "semi_minor", &semi_minor),
        ("theta", "inclination_degrees", &inclination),
        ("g", "phase_degrees", &phase),
        ("PE", "percent_energy", &percent_energy),
    ] {
        output.set_item(short, values)?;
        output.set_item(descriptive, values)?;
    }
    set_optional_vector_matrix(
        py,
        output,
        solutions,
        constituent_count,
        "Lsmaj_ci",
        "semi_major_ci",
        |solution| solution.semi_major_ci.as_deref(),
    )?;
    set_optional_vector_matrix(
        py,
        output,
        solutions,
        constituent_count,
        "Lsmin_ci",
        "semi_minor_ci",
        |solution| solution.semi_minor_ci.as_deref(),
    )?;
    set_optional_vector_matrix(
        py,
        output,
        solutions,
        constituent_count,
        "theta_ci",
        "inclination_ci_degrees",
        |solution| solution.inclination_ci_degrees.as_deref(),
    )?;
    set_optional_vector_matrix(
        py,
        output,
        solutions,
        constituent_count,
        "g_ci",
        "phase_ci_degrees",
        |solution| solution.phase_ci_degrees.as_deref(),
    )?;
    set_optional_vector_matrix(
        py,
        output,
        solutions,
        constituent_count,
        "SNR",
        "signal_to_noise",
        |solution| solution.signal_to_noise.as_deref(),
    )?;
    for (name, values) in [
        (
            "umean",
            solutions
                .iter()
                .map(|solution| solution.eastward_mean)
                .collect::<Vec<_>>(),
        ),
        (
            "vmean",
            solutions
                .iter()
                .map(|solution| solution.northward_mean)
                .collect(),
        ),
        (
            "uslope",
            solutions
                .iter()
                .map(|solution| solution.eastward_slope_per_day)
                .collect(),
        ),
        (
            "vslope",
            solutions
                .iter()
                .map(|solution| solution.northward_slope_per_day)
                .collect(),
        ),
        (
            "reference_time_mjd",
            solutions
                .iter()
                .map(|solution| solution.reference_time_days)
                .collect(),
        ),
    ] {
        output.set_item(name, values.into_pyarray(py))?;
    }
    Ok(())
}

fn add_robust_summary(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    state: &BatchFitState,
) -> PyResult<()> {
    let diagnostics = match &state.solutions {
        BatchSolutions::Scalar(solutions) => solutions
            .iter()
            .map(|solution| solution.robust.as_ref())
            .collect::<Vec<_>>(),
        BatchSolutions::Vector(solutions) => solutions
            .iter()
            .map(|solution| solution.robust.as_ref())
            .collect::<Vec<_>>(),
    };
    if diagnostics.iter().all(Option::is_none) {
        output.set_item("robust", py.None())?;
        output.set_item("weights", py.None())?;
        return Ok(());
    }
    if diagnostics.iter().any(Option::is_none) {
        return Err(PyValueError::new_err(
            "internal batch contains inconsistent robust diagnostics",
        ));
    }
    let series_count = state.latitudes.len();
    let mut weights = vec![f64::NAN; state.retained_time_count * series_count];
    let mut leverage = vec![f64::NAN; state.retained_time_count * series_count];
    for (series, (diagnostic, positions)) in
        diagnostics.iter().zip(&state.valid_positions).enumerate()
    {
        let diagnostic = diagnostic.expect("all diagnostics were checked above");
        if diagnostic.weights.len() != positions.len()
            || diagnostic.leverage.len() != positions.len()
        {
            return Err(PyValueError::new_err(
                "internal robust diagnostic length does not match retained observations",
            ));
        }
        for (retained, position) in positions.iter().copied().enumerate() {
            weights[position * series_count + series] = diagnostic.weights[retained];
            leverage[position * series_count + series] = diagnostic.leverage[retained];
        }
    }
    let weights = array2(state.retained_time_count, series_count, weights)?.into_pyarray(py);
    let robust = PyDict::new(py);
    robust.set_item("weights", &weights)?;
    robust.set_item(
        "leverage",
        array2(state.retained_time_count, series_count, leverage)?.into_pyarray(py),
    )?;
    robust.set_item(
        "iterations",
        diagnostics
            .iter()
            .map(|value| value.expect("checked").iterations)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    robust.set_item(
        "termination",
        diagnostics
            .iter()
            .map(|value| robust_termination_name(value.expect("checked").termination))
            .collect::<Vec<_>>(),
    )?;
    for (name, values) in [
        (
            "residual_scale",
            diagnostics
                .iter()
                .map(|value| value.expect("checked").residual_scale)
                .collect::<Vec<_>>(),
        ),
        (
            "ols_rms_residual",
            diagnostics
                .iter()
                .map(|value| value.expect("checked").ols_rms_residual)
                .collect(),
        ),
        (
            "rms_residual",
            diagnostics
                .iter()
                .map(|value| value.expect("checked").rms_residual)
                .collect(),
        ),
    ] {
        robust.set_item(name, values.into_pyarray(py))?;
    }
    output.set_item("weights", weights)?;
    output.set_item("robust", robust)?;
    Ok(())
}

fn set_optional_scalar_matrix<F>(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    solutions: &[ScalarSolution],
    constituent_count: usize,
    short: &str,
    descriptive: &str,
    field: F,
) -> PyResult<()>
where
    F: Fn(&ScalarSolution) -> Option<&[f64]>,
{
    let values = optional_matrix(
        solutions.iter().map(field),
        solutions.len(),
        constituent_count,
    )?;
    if let Some(values) = values {
        let values = values.into_pyarray(py);
        output.set_item(short, &values)?;
        output.set_item(descriptive, values)?;
    } else {
        output.set_item(short, py.None())?;
        output.set_item(descriptive, py.None())?;
    }
    Ok(())
}

fn set_optional_vector_matrix<F>(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    solutions: &[VectorSolution],
    constituent_count: usize,
    short: &str,
    descriptive: &str,
    field: F,
) -> PyResult<()>
where
    F: Fn(&VectorSolution) -> Option<&[f64]>,
{
    let values = optional_matrix(
        solutions.iter().map(field),
        solutions.len(),
        constituent_count,
    )?;
    if let Some(values) = values {
        let values = values.into_pyarray(py);
        output.set_item(short, &values)?;
        output.set_item(descriptive, values)?;
    } else {
        output.set_item(short, py.None())?;
        output.set_item(descriptive, py.None())?;
    }
    Ok(())
}

fn optional_matrix<'a>(
    rows: impl Iterator<Item = Option<&'a [f64]>>,
    row_count: usize,
    column_count: usize,
) -> PyResult<Option<Array2<f64>>> {
    let rows = rows.collect::<Vec<_>>();
    if rows.iter().all(Option::is_none) {
        return Ok(None);
    }
    if rows.iter().any(Option::is_none) {
        return Err(PyValueError::new_err(
            "internal batch contains inconsistent confidence fields",
        ));
    }
    let values = rows
        .into_iter()
        .flat_map(|row| row.expect("all rows were checked").iter().copied())
        .collect();
    array2(row_count, column_count, values).map(Some)
}

fn matrix_from_scalar<F>(
    solutions: &[ScalarSolution],
    constituent_count: usize,
    field: F,
) -> PyResult<Array2<f64>>
where
    F: Fn(&ScalarSolution) -> &[f64],
{
    array2(
        solutions.len(),
        constituent_count,
        solutions.iter().flat_map(field).copied().collect(),
    )
}

fn matrix_from_vector<F>(
    solutions: &[VectorSolution],
    constituent_count: usize,
    field: F,
) -> PyResult<Array2<f64>>
where
    F: Fn(&VectorSolution) -> &[f64],
{
    array2(
        solutions.len(),
        constituent_count,
        solutions.iter().flat_map(field).copied().collect(),
    )
}

fn array2<T>(rows: usize, columns: usize, values: Vec<T>) -> PyResult<Array2<T>> {
    Array2::from_shape_vec((rows, columns), values)
        .map_err(|error| PyValueError::new_err(format!("internal array shape error: {error}")))
}

fn compact_finite_time_rows(
    time: Vec<f64>,
    eastward: Vec<f64>,
    northward: Option<Vec<f64>>,
    time_count: usize,
    series_count: usize,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>, Vec<usize>) {
    if time.iter().all(|value| value.is_finite()) {
        return (time, eastward, northward, (0..time_count).collect());
    }
    let positions = time
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect::<Vec<_>>();
    let retained_time = positions.iter().map(|index| time[*index]).collect();
    let retained_eastward = retain_matrix_rows(&eastward, &positions, series_count);
    let retained_northward = northward
        .as_ref()
        .map(|values| retain_matrix_rows(values, &positions, series_count));
    (
        retained_time,
        retained_eastward,
        retained_northward,
        positions,
    )
}

fn retain_matrix_rows(values: &[f64], positions: &[usize], series_count: usize) -> Vec<f64> {
    let mut retained = Vec::with_capacity(positions.len() * series_count);
    for position in positions {
        let start = position * series_count;
        retained.extend_from_slice(&values[start..start + series_count]);
    }
    retained
}

fn valid_observation_positions(
    eastward: &[f64],
    northward: Option<&[f64]>,
    time_count: usize,
    series_count: usize,
) -> Vec<Vec<usize>> {
    (0..series_count)
        .map(|series| {
            (0..time_count)
                .filter(|time| {
                    let index = time * series_count + series;
                    eastward[index].is_finite()
                        && northward.is_none_or(|values| values[index].is_finite())
                })
                .collect()
        })
        .collect()
}

fn requested_worker_count(requested: Option<usize>, series_count: usize) -> Result<usize, String> {
    if requested == Some(0) {
        return Err("workers must be greater than zero".to_owned());
    }
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    Ok(requested.unwrap_or(available).min(series_count).max(1))
}

fn working_chunk_series(
    time_count: usize,
    series_count: usize,
    component_count: usize,
    memory_limit_bytes: Option<usize>,
) -> Result<usize, String> {
    let Some(limit) = memory_limit_bytes else {
        return Ok(series_count);
    };
    if limit == 0 {
        return Err("memory_limit_bytes must be greater than zero or None".to_owned());
    }
    let bytes_per_series = time_count
        .checked_mul(component_count)
        .and_then(|value| value.checked_mul(size_of::<f64>()))
        .ok_or("batch working-memory size overflow")?;
    Ok((limit / bytes_per_series.max(1)).max(1).min(series_count))
}

fn time_major_chunk(
    values: &[f64],
    time_count: usize,
    series_count: usize,
    start: usize,
    end: usize,
) -> Vec<f64> {
    let mut chunk = Vec::with_capacity(time_count * (end - start));
    for time in 0..time_count {
        let row = time * series_count;
        chunk.extend_from_slice(&values[row + start..row + end]);
    }
    chunk
}

fn series_to_time_major(series_major: &[Vec<f64>], time_count: usize) -> Vec<f64> {
    let series_count = series_major.len();
    let mut time_major = vec![0.0; time_count * series_count];
    for (series, values) in series_major.iter().enumerate() {
        for time in 0..time_count {
            time_major[time * series_count + series] = values[time];
        }
    }
    time_major
}

fn matrix_shape(array: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<(usize, usize)> {
    let shape = array.shape();
    match shape {
        [rows, columns] => Ok((*rows, *columns)),
        _ => Err(PyValueError::new_err(format!(
            "{name} must be two-dimensional"
        ))),
    }
}

fn contiguous_copy_1(array: &PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<Vec<f64>> {
    array
        .as_slice()
        .map(<[f64]>::to_vec)
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float64 array")))
}

fn contiguous_copy_2(array: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<f64>> {
    array
        .as_slice()
        .map(<[f64]>::to_vec)
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float64 array")))
}

const fn robust_termination_name(termination: RobustTermination) -> &'static str {
    match termination {
        RobustTermination::Tolerance => "tolerance",
        RobustTermination::ObjectiveIncrease => "objective_increase",
        RobustTermination::ExactFit => "exact_fit",
    }
}
