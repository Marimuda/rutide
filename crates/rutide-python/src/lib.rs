//! Native implementation behind `RUTide`'s `UTide`-inspired Python API.

mod batch;
mod persistence;

use std::{collections::HashSet, ops::Deref, sync::Arc};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyModule},
};
use rutide_core::{
    ConstituentDiagnosticsOptions, ConstituentSelectionDiagnostics, FitOptions, GreenwichNodalOls,
    InferenceMode, LinearConfidence, MonteCarloOptions, NodalCorrections, PhaseReference,
    ReconstructionFilter, RobustOptions, RobustTermination, RobustWeightFunction,
    ScalarInferenceOls, ScalarInferenceRelation, ScalarSolution, SolverOptions, TidalConstituent,
    VectorInferenceOls, VectorInferenceRelation, VectorSolution, select_constituents_by_rayleigh,
};

#[derive(Clone, Copy)]
enum Confidence {
    None,
    Linear(LinearConfidence),
    MonteCarlo(MonteCarloOptions, LinearConfidence),
}

enum PreparedModel {
    Direct(Box<GreenwichNodalOls>),
    ScalarInference(Box<ScalarInferenceOls>),
    VectorInference(Box<VectorInferenceOls>),
}

enum FittedSolution {
    Scalar(ScalarSolution),
    Vector(VectorSolution),
}

/// Opaque, reusable native model and solution owned by a Python coefficient object.
#[pyclass(module = "rutide._native", frozen)]
struct Fit {
    inner: Arc<FitState>,
}

struct FitState {
    model: PreparedModel,
    solution: FittedSolution,
    time_mjd: Vec<f64>,
    config: SolveConfig,
    names: Vec<String>,
    frequencies: Vec<f64>,
    presentation_order: Vec<usize>,
    latitude: f64,
    original_observations: usize,
    retained_observations: usize,
    method: String,
    confidence: String,
    phase_reference: String,
    nodal_corrections: String,
    trend: bool,
    diagnostics: Option<ConstituentSelectionDiagnostics>,
}

impl Deref for Fit {
    type Target = FitState;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[pymethods]
impl Fit {
    #[getter]
    fn is_vector(&self) -> bool {
        matches!(self.solution, FittedSolution::Vector(_))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        let names = reordered(&self.names, &self.presentation_order);
        let frequencies = reordered(&self.frequencies, &self.presentation_order);
        output.set_item("name", &names)?;
        output.set_item("frequency_cph", &frequencies)?;
        output.set_item("method", &self.method)?;
        output.set_item("confidence", &self.confidence)?;
        output.set_item("phase_reference", &self.phase_reference)?;
        output.set_item("nodal_corrections", &self.nodal_corrections)?;
        output.set_item("trend", self.trend)?;
        output.set_item("nobs", self.retained_observations)?;
        output.set_item("nobs_original", self.original_observations)?;

        let auxiliary = PyDict::new(py);
        auxiliary.set_item("frq", &frequencies)?;
        auxiliary.set_item("lat", self.latitude)?;
        auxiliary.set_item("nobs", self.retained_observations)?;
        auxiliary.set_item("nobs_original", self.original_observations)?;

        match &self.solution {
            FittedSolution::Scalar(solution) => {
                add_scalar_summary(&output, solution, &self.presentation_order)?;
                auxiliary.set_item("reftime", solution.reference_time_days)?;
                add_robust_summary(py, &output, solution.robust.as_ref())?;
            }
            FittedSolution::Vector(solution) => {
                add_vector_summary(&output, solution, &self.presentation_order)?;
                auxiliary.set_item("reftime", solution.reference_time_days)?;
                add_robust_summary(py, &output, solution.robust.as_ref())?;
            }
        }
        add_diagnostics_summary(
            py,
            &output,
            self.diagnostics.as_ref(),
            &self.presentation_order,
        )?;
        output.set_item("aux", auxiliary)?;
        Ok(output)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        persistence::fit_snapshot(self, py)
    }
}

#[allow(
    clippy::fn_params_excessive_bools,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "PyO3 exposes the UTide-compatible keyword surface directly"
)]
#[pyfunction]
fn solve(
    py: Python<'_>,
    time_mjd: PyReadonlyArray1<'_, f64>,
    eastward: PyReadonlyArray1<'_, f64>,
    northward: Option<PyReadonlyArray1<'_, f64>>,
    latitude: f64,
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
) -> PyResult<Py<Fit>> {
    let time = contiguous_copy(&time_mjd, "time_mjd")?;
    let eastward = contiguous_copy(&eastward, "u")?;
    let northward = northward
        .as_ref()
        .map(|array| contiguous_copy(array, "v"))
        .transpose()?;
    let config = SolveConfig {
        latitude,
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
        .detach(move || solve_native(&time, &eastward, northward.as_deref(), &config))
        .map_err(PyValueError::new_err)?;
    Py::new(py, fit)
}

type ReconstructionArrays<'py> = (
    Option<Bound<'py, PyArray1<f64>>>,
    Option<Bound<'py, PyArray1<f64>>>,
    Option<Bound<'py, PyArray1<f64>>>,
);

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn reconstruct<'py>(
    py: Python<'py>,
    time_mjd: PyReadonlyArray1<'_, f64>,
    fit: PyRef<'_, Fit>,
    constituent_names: Option<Vec<String>>,
    minimum_signal_to_noise: Option<f64>,
    minimum_percent_energy: f64,
) -> PyResult<ReconstructionArrays<'py>> {
    let time = contiguous_copy(&time_mjd, "time_mjd")?;
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
        Reconstructed::Scalar(heights) => Ok((Some(heights.into_pyarray(py)), None, None)),
        Reconstructed::Vector(eastward, northward) => Ok((
            None,
            Some(eastward.into_pyarray(py)),
            Some(northward.into_pyarray(py)),
        )),
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", rutide_core::VERSION)?;
    module.add_class::<Fit>()?;
    module.add_class::<batch::BatchFit>()?;
    module.add_function(wrap_pyfunction!(solve, module)?)?;
    module.add_function(wrap_pyfunction!(reconstruct, module)?)?;
    module.add_function(wrap_pyfunction!(batch::solve_many, module)?)?;
    module.add_function(wrap_pyfunction!(batch::reconstruct_many, module)?)?;
    module.add_function(wrap_pyfunction!(batch::restore_batch, module)?)?;
    module.add_function(wrap_pyfunction!(persistence::restore_fit, module)?)?;
    Ok(())
}

#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
struct SolveConfig {
    latitude: f64,
    constituent_names: Option<Vec<String>>,
    rayleigh_min: f64,
    diagnostics: bool,
    diagnostic_min_signal_to_noise: f64,
    method_name: String,
    confidence_name: String,
    white: bool,
    trend: bool,
    phase_name: String,
    nodal_name: String,
    monte_carlo_realizations: usize,
    monte_carlo_seed: u64,
    robust_weight_name: String,
    robust_tuning: Option<f64>,
    robust_tolerance: f64,
    robust_max_iterations: usize,
    inferred_names: Vec<String>,
    reference_names: Vec<String>,
    inference_ratios: Vec<f64>,
    inference_phase_offsets: Vec<f64>,
    approximate_inference: bool,
    order_name: String,
    order_names: Vec<String>,
}

#[allow(clippy::too_many_lines)]
fn solve_native(
    time: &[f64],
    eastward: &[f64],
    northward: Option<&[f64]>,
    config: &SolveConfig,
) -> Result<Fit, String> {
    let original_observations = time.len();
    let (time, eastward, northward) = retain_finite_rows(time, eastward, northward)?;
    let retained_observations = time.len();
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
        FitOptions {
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
    let robust_options = parse_method_and_robust(config)?;
    let inference_mode = if config.approximate_inference {
        InferenceMode::Approximate
    } else {
        InferenceMode::Exact
    };
    let is_vector = northward.is_some();

    let (model, solution, names, frequencies) = if config.inferred_names.is_empty() {
        validate_empty_inference(config)?;
        let model = GreenwichNodalOls::prepare_modified_julian_days_with_solver_options(
            &time,
            config.latitude,
            &constituents,
            solver_options,
        )
        .map_err(|error| error.to_string())?;
        let (names, frequencies) = constituent_metadata(model.constituents());
        let solution = match northward.as_deref() {
            Some(northward) => FittedSolution::Vector(solve_vector_model(
                &model,
                &eastward,
                northward,
                robust_options,
                confidence,
            )?),
            None => FittedSolution::Scalar(solve_scalar_model(
                &model,
                &eastward,
                robust_options,
                confidence,
            )?),
        };
        (
            PreparedModel::Direct(Box::new(model)),
            solution,
            names,
            frequencies,
        )
    } else if is_vector {
        let relationships = vector_inference_relations(config)?;
        let model = VectorInferenceOls::prepare_modified_julian_days_with_solver_options(
            &time,
            config.latitude,
            &constituents,
            &relationships,
            inference_mode,
            solver_options,
        )
        .map_err(|error| error.to_string())?;
        let (names, frequencies) = constituent_metadata(model.constituents());
        let northward = northward
            .as_deref()
            .ok_or("internal missing vector component")?;
        let solution = FittedSolution::Vector(solve_vector_model(
            &model,
            &eastward,
            northward,
            robust_options,
            confidence,
        )?);
        (
            PreparedModel::VectorInference(Box::new(model)),
            solution,
            names,
            frequencies,
        )
    } else {
        let relationships = scalar_inference_relations(config)?;
        let model = ScalarInferenceOls::prepare_modified_julian_days_with_solver_options(
            &time,
            config.latitude,
            &constituents,
            &relationships,
            inference_mode,
            solver_options,
        )
        .map_err(|error| error.to_string())?;
        let (names, frequencies) = constituent_metadata(model.constituents());
        let solution = FittedSolution::Scalar(solve_scalar_model(
            &model,
            &eastward,
            robust_options,
            confidence,
        )?);
        (
            PreparedModel::ScalarInference(Box::new(model)),
            solution,
            names,
            frequencies,
        )
    };

    let presentation_order = constituent_order(
        &config.order_name,
        &config.order_names,
        &names,
        &frequencies,
        &solution,
    )?;
    let diagnostics = if config.diagnostics {
        let options = ConstituentDiagnosticsOptions::default()
            .with_rayleigh_minimum(config.rayleigh_min)
            .with_minimum_signal_to_noise(config.diagnostic_min_signal_to_noise);
        Some(fit_diagnostics(
            &model,
            &solution,
            &time,
            &eastward,
            northward.as_deref(),
            options,
        )?)
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
    Ok(Fit {
        inner: Arc::new(FitState {
            model,
            solution,
            time_mjd: time,
            config: stored_config,
            names,
            frequencies,
            presentation_order,
            latitude: config.latitude,
            original_observations,
            retained_observations,
            method: config.method_name.to_ascii_lowercase(),
            confidence: normalized_confidence_name(confidence).to_owned(),
            phase_reference: phase_reference.name().to_owned(),
            nodal_corrections: nodal_corrections.name().to_owned(),
            trend: config.trend,
            diagnostics,
        }),
    })
}

fn fit_diagnostics(
    model: &PreparedModel,
    solution: &FittedSolution,
    time_mjd: &[f64],
    eastward: &[f64],
    northward: Option<&[f64]>,
    options: ConstituentDiagnosticsOptions,
) -> Result<ConstituentSelectionDiagnostics, String> {
    match (model, solution, northward) {
        (PreparedModel::Direct(model), FittedSolution::Scalar(solution), None) => {
            model.diagnose_scalar_solution(eastward, solution, options)
        }
        (PreparedModel::Direct(model), FittedSolution::Vector(solution), Some(northward)) => {
            model.diagnose_vector_solution(eastward, northward, solution, options)
        }
        (PreparedModel::ScalarInference(model), FittedSolution::Scalar(solution), None) => {
            model.diagnose_solution(eastward, solution, options)
        }
        (
            PreparedModel::VectorInference(model),
            FittedSolution::Vector(solution),
            Some(northward),
        ) => model.diagnose_vector_solution(time_mjd, eastward, northward, solution, options),
        _ => return Err("internal model, solution, and observation type mismatch".to_owned()),
    }
    .map_err(|error| error.to_string())
}

trait ScalarModel {
    fn ordinary(&self, observations: &[f64]) -> Result<ScalarSolution, String>;
    fn linear(
        &self,
        observations: &[f64],
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, String>;
    fn monte_carlo(
        &self,
        observations: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, String>;
    fn robust(
        &self,
        observations: &[f64],
        options: RobustOptions,
    ) -> Result<ScalarSolution, String>;
    fn robust_linear(
        &self,
        observations: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, String>;
    fn robust_monte_carlo(
        &self,
        observations: &[f64],
        robust: RobustOptions,
        monte_carlo: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, String>;
}

macro_rules! impl_scalar_model {
    ($model:ty) => {
        impl ScalarModel for $model {
            fn ordinary(&self, observations: &[f64]) -> Result<ScalarSolution, String> {
                self.solve(observations).map_err(|error| error.to_string())
            }
            fn linear(
                &self,
                observations: &[f64],
                noise: LinearConfidence,
            ) -> Result<ScalarSolution, String> {
                self.solve_with_linear_confidence(observations, noise)
                    .map_err(|error| error.to_string())
            }
            fn monte_carlo(
                &self,
                observations: &[f64],
                options: MonteCarloOptions,
                noise: LinearConfidence,
            ) -> Result<ScalarSolution, String> {
                self.solve_with_monte_carlo_confidence(observations, options, noise)
                    .map_err(|error| error.to_string())
            }
            fn robust(
                &self,
                observations: &[f64],
                options: RobustOptions,
            ) -> Result<ScalarSolution, String> {
                self.solve_robust(observations, options)
                    .map_err(|error| error.to_string())
            }
            fn robust_linear(
                &self,
                observations: &[f64],
                options: RobustOptions,
                noise: LinearConfidence,
            ) -> Result<ScalarSolution, String> {
                self.solve_robust_with_linear_confidence(observations, options, noise)
                    .map_err(|error| error.to_string())
            }
            fn robust_monte_carlo(
                &self,
                observations: &[f64],
                robust: RobustOptions,
                monte_carlo: MonteCarloOptions,
                noise: LinearConfidence,
            ) -> Result<ScalarSolution, String> {
                self.solve_robust_with_monte_carlo_confidence(
                    observations,
                    robust,
                    monte_carlo,
                    noise,
                )
                .map_err(|error| error.to_string())
            }
        }
    };
}

impl_scalar_model!(GreenwichNodalOls);
impl_scalar_model!(ScalarInferenceOls);

trait VectorModel {
    fn ordinary(&self, eastward: &[f64], northward: &[f64]) -> Result<VectorSolution, String>;
    fn linear(
        &self,
        eastward: &[f64],
        northward: &[f64],
        noise: LinearConfidence,
    ) -> Result<VectorSolution, String>;
    fn monte_carlo(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<VectorSolution, String>;
    fn robust(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: RobustOptions,
    ) -> Result<VectorSolution, String>;
    fn robust_linear(
        &self,
        eastward: &[f64],
        northward: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<VectorSolution, String>;
    fn robust_monte_carlo(
        &self,
        eastward: &[f64],
        northward: &[f64],
        robust: RobustOptions,
        monte_carlo: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<VectorSolution, String>;
}

macro_rules! impl_vector_model {
    ($model:ty) => {
        impl VectorModel for $model {
            fn ordinary(
                &self,
                eastward: &[f64],
                northward: &[f64],
            ) -> Result<VectorSolution, String> {
                self.solve_vector(eastward, northward)
                    .map_err(|error| error.to_string())
            }
            fn linear(
                &self,
                eastward: &[f64],
                northward: &[f64],
                noise: LinearConfidence,
            ) -> Result<VectorSolution, String> {
                self.solve_vector_with_linear_confidence(eastward, northward, noise)
                    .map_err(|error| error.to_string())
            }
            fn monte_carlo(
                &self,
                eastward: &[f64],
                northward: &[f64],
                options: MonteCarloOptions,
                noise: LinearConfidence,
            ) -> Result<VectorSolution, String> {
                self.solve_vector_with_monte_carlo_confidence(eastward, northward, options, noise)
                    .map_err(|error| error.to_string())
            }
            fn robust(
                &self,
                eastward: &[f64],
                northward: &[f64],
                options: RobustOptions,
            ) -> Result<VectorSolution, String> {
                self.solve_vector_robust(eastward, northward, options)
                    .map_err(|error| error.to_string())
            }
            fn robust_linear(
                &self,
                eastward: &[f64],
                northward: &[f64],
                options: RobustOptions,
                noise: LinearConfidence,
            ) -> Result<VectorSolution, String> {
                self.solve_vector_robust_with_linear_confidence(eastward, northward, options, noise)
                    .map_err(|error| error.to_string())
            }
            fn robust_monte_carlo(
                &self,
                eastward: &[f64],
                northward: &[f64],
                robust: RobustOptions,
                monte_carlo: MonteCarloOptions,
                noise: LinearConfidence,
            ) -> Result<VectorSolution, String> {
                self.solve_vector_robust_with_monte_carlo_confidence(
                    eastward,
                    northward,
                    robust,
                    monte_carlo,
                    noise,
                )
                .map_err(|error| error.to_string())
            }
        }
    };
}

impl_vector_model!(GreenwichNodalOls);
impl_vector_model!(VectorInferenceOls);

fn solve_scalar_model<M: ScalarModel>(
    model: &M,
    observations: &[f64],
    robust: Option<RobustOptions>,
    confidence: Confidence,
) -> Result<ScalarSolution, String> {
    match (robust, confidence) {
        (None, Confidence::None) => model.ordinary(observations),
        (None, Confidence::Linear(noise)) => model.linear(observations, noise),
        (None, Confidence::MonteCarlo(options, noise)) => {
            model.monte_carlo(observations, options, noise)
        }
        (Some(robust), Confidence::None) => model.robust(observations, robust),
        (Some(robust), Confidence::Linear(noise)) => {
            model.robust_linear(observations, robust, noise)
        }
        (Some(robust), Confidence::MonteCarlo(monte_carlo, noise)) => {
            model.robust_monte_carlo(observations, robust, monte_carlo, noise)
        }
    }
}

fn solve_vector_model<M: VectorModel>(
    model: &M,
    eastward: &[f64],
    northward: &[f64],
    robust: Option<RobustOptions>,
    confidence: Confidence,
) -> Result<VectorSolution, String> {
    match (robust, confidence) {
        (None, Confidence::None) => model.ordinary(eastward, northward),
        (None, Confidence::Linear(noise)) => model.linear(eastward, northward, noise),
        (None, Confidence::MonteCarlo(options, noise)) => {
            model.monte_carlo(eastward, northward, options, noise)
        }
        (Some(robust), Confidence::None) => model.robust(eastward, northward, robust),
        (Some(robust), Confidence::Linear(noise)) => {
            model.robust_linear(eastward, northward, robust, noise)
        }
        (Some(robust), Confidence::MonteCarlo(monte_carlo, noise)) => {
            model.robust_monte_carlo(eastward, northward, robust, monte_carlo, noise)
        }
    }
}

type RetainedRows = (Vec<f64>, Vec<f64>, Option<Vec<f64>>);

fn retain_finite_rows(
    time: &[f64],
    eastward: &[f64],
    northward: Option<&[f64]>,
) -> Result<RetainedRows, String> {
    if time.len() != eastward.len() {
        return Err("time_mjd and u must have the same length".to_owned());
    }
    if northward
        .as_ref()
        .is_some_and(|values| values.len() != time.len())
    {
        return Err("u and v must have the same length".to_owned());
    }
    if eastward.iter().any(|value| value.is_infinite())
        || northward
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| value.is_infinite()))
    {
        return Err("observations may contain NaN missing values, but not infinity".to_owned());
    }

    let retained = (0..time.len())
        .filter(|index| {
            time[*index].is_finite()
                && !eastward[*index].is_nan()
                && northward
                    .as_ref()
                    .is_none_or(|values| !values[*index].is_nan())
        })
        .collect::<Vec<_>>();
    let retained_time = retained.iter().map(|index| time[*index]).collect();
    let retained_eastward = retained.iter().map(|index| eastward[*index]).collect();
    let retained_northward = northward.map(|values| {
        retained
            .iter()
            .map(|index| values[*index])
            .collect::<Vec<_>>()
    });
    Ok((retained_time, retained_eastward, retained_northward))
}

fn parse_constituents(names: &[String]) -> Result<Vec<TidalConstituent>, String> {
    names.iter().map(|name| parse_constituent(name)).collect()
}

fn parse_constituent(name: &str) -> Result<TidalConstituent, String> {
    let trimmed = name.trim();
    TidalConstituent::from_name(trimmed)
        .or_else(|| TidalConstituent::from_name(&trimmed.to_ascii_uppercase()))
        .ok_or_else(|| format!("unknown tidal constituent '{name}'"))
}

fn parse_phase_reference(name: &str) -> Result<PhaseReference, String> {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "greenwich" => Ok(PhaseReference::Greenwich),
        "linear_time" => Ok(PhaseReference::LinearTime),
        "raw" => Ok(PhaseReference::Raw),
        _ => Err("phase must be 'Greenwich', 'linear_time', or 'raw'".to_owned()),
    }
}

fn parse_nodal_corrections(name: &str) -> Result<NodalCorrections, String> {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "exact" => Ok(NodalCorrections::Exact),
        "linear_time" => Ok(NodalCorrections::LinearTime),
        "disabled" => Ok(NodalCorrections::Disabled),
        _ => Err("nodal mode must be 'exact', 'linear_time', or 'disabled'".to_owned()),
    }
}

fn parse_confidence(
    name: &str,
    white: bool,
    realizations: usize,
    seed: u64,
) -> Result<Confidence, String> {
    let noise = if white {
        LinearConfidence::White
    } else {
        LinearConfidence::Colored
    };
    match name.to_ascii_lowercase().as_str() {
        "none" => Ok(Confidence::None),
        "linear" => Ok(Confidence::Linear(noise)),
        "mc" | "monte_carlo" | "monte-carlo" => Ok(Confidence::MonteCarlo(
            MonteCarloOptions { realizations, seed },
            noise,
        )),
        _ => Err("conf_int must be 'linear', 'MC', 'none', or None".to_owned()),
    }
}

fn normalized_confidence_name(confidence: Confidence) -> &'static str {
    match confidence {
        Confidence::None => "none",
        Confidence::Linear(_) => "linear",
        Confidence::MonteCarlo(_, _) => "MC",
    }
}

fn parse_method_and_robust(config: &SolveConfig) -> Result<Option<RobustOptions>, String> {
    match config.method_name.to_ascii_lowercase().as_str() {
        "ols" => Ok(None),
        "robust" => {
            let weight_function = parse_weight_function(&config.robust_weight_name)?;
            let mut options = RobustOptions::for_weight_function(weight_function);
            if let Some(tuning) = config.robust_tuning {
                options.tuning_constant = tuning;
            }
            options.tolerance = config.robust_tolerance;
            options.max_iterations = config.robust_max_iterations;
            Ok(Some(options))
        }
        _ => Err("method must be 'ols' or 'robust'".to_owned()),
    }
}

fn parse_weight_function(name: &str) -> Result<RobustWeightFunction, String> {
    match name.to_ascii_lowercase().as_str() {
        "andrews" => Ok(RobustWeightFunction::Andrews),
        "bisquare" | "bisq" => Ok(RobustWeightFunction::Bisquare),
        "cauchy" => Ok(RobustWeightFunction::Cauchy),
        "fair" => Ok(RobustWeightFunction::Fair),
        "huber" => Ok(RobustWeightFunction::Huber),
        "logistic" | "logist" => Ok(RobustWeightFunction::Logistic),
        "ols" => Ok(RobustWeightFunction::Ols),
        "talwar" => Ok(RobustWeightFunction::Talwar),
        "welsch" => Ok(RobustWeightFunction::Welsch),
        _ => Err(format!("unknown robust weight function '{name}'")),
    }
}

fn validate_empty_inference(config: &SolveConfig) -> Result<(), String> {
    if config.reference_names.is_empty()
        && config.inference_ratios.is_empty()
        && config.inference_phase_offsets.is_empty()
    {
        Ok(())
    } else {
        Err("inference arrays must all be empty or describe the same relationships".to_owned())
    }
}

fn scalar_inference_relations(
    config: &SolveConfig,
) -> Result<Vec<ScalarInferenceRelation>, String> {
    let count = config.inferred_names.len();
    validate_inference_shapes(config, count)?;
    config
        .inferred_names
        .iter()
        .zip(&config.reference_names)
        .zip(&config.inference_ratios)
        .zip(&config.inference_phase_offsets)
        .map(|(((inferred, reference), ratio), phase)| {
            Ok(ScalarInferenceRelation::new(
                parse_constituent(inferred)?,
                parse_constituent(reference)?,
                *ratio,
                *phase,
            ))
        })
        .collect()
}

fn vector_inference_relations(
    config: &SolveConfig,
) -> Result<Vec<VectorInferenceRelation>, String> {
    let count = config.inferred_names.len();
    validate_inference_shapes(config, count * 2)?;
    (0..count)
        .map(|index| {
            Ok(VectorInferenceRelation::new(
                parse_constituent(&config.inferred_names[index])?,
                parse_constituent(&config.reference_names[index])?,
                config.inference_ratios[index],
                config.inference_phase_offsets[index],
                config.inference_ratios[index + count],
                config.inference_phase_offsets[index + count],
            ))
        })
        .collect()
}

fn validate_inference_shapes(config: &SolveConfig, ratio_count: usize) -> Result<(), String> {
    if config.reference_names.len() != config.inferred_names.len() {
        return Err("inference reference_names must match inferred_names".to_owned());
    }
    if config.inference_ratios.len() != ratio_count
        || config.inference_phase_offsets.len() != ratio_count
    {
        return Err(format!(
            "inference ratios and phase offsets must each contain {ratio_count} values"
        ));
    }
    Ok(())
}

fn constituent_metadata(constituents: &[rutide_core::Constituent]) -> (Vec<String>, Vec<f64>) {
    (
        constituents.iter().map(|item| item.name.clone()).collect(),
        constituents.iter().map(|item| item.frequency_cph).collect(),
    )
}

fn constituent_order(
    order_name: &str,
    order_names: &[String],
    names: &[String],
    frequencies: &[f64],
    solution: &FittedSolution,
) -> Result<Vec<usize>, String> {
    constituent_order_from_diagnostics(
        order_name,
        order_names,
        names,
        frequencies,
        percent_energy(solution),
        signal_to_noise(solution),
    )
}

fn constituent_order_from_diagnostics(
    order_name: &str,
    order_names: &[String],
    names: &[String],
    frequencies: &[f64],
    percent_energy: &[f64],
    signal_to_noise: Option<&[f64]>,
) -> Result<Vec<usize>, String> {
    let mut indices = (0..names.len()).collect::<Vec<_>>();
    match order_name.to_ascii_lowercase().as_str() {
        "pe" => indices.sort_by(|left, right| {
            percent_energy[*right]
                .total_cmp(&percent_energy[*left])
                .then_with(|| left.cmp(right))
        }),
        "snr" => {
            let snr = signal_to_noise.ok_or(
                "order_constit='SNR' requires confidence intervals (conf_int='linear' or 'MC')",
            )?;
            indices.sort_by(|left, right| {
                snr[*right]
                    .total_cmp(&snr[*left])
                    .then_with(|| left.cmp(right))
            });
        }
        "frequency" => indices.sort_by(|left, right| {
            frequencies[*left]
                .total_cmp(&frequencies[*right])
                .then_with(|| left.cmp(right))
        }),
        "explicit" => {
            let mut seen = HashSet::new();
            indices = order_names
                .iter()
                .map(|requested| {
                    let normalized = parse_constituent(requested)?.name();
                    let index = names
                        .iter()
                        .position(|name| name == normalized)
                        .ok_or_else(|| {
                            format!("order_constit contains unfitted constituent '{requested}'")
                        })?;
                    if !seen.insert(index) {
                        return Err(format!(
                            "order_constit contains duplicate constituent '{requested}'"
                        ));
                    }
                    Ok(index)
                })
                .collect::<Result<Vec<_>, String>>()?;
            if indices.is_empty() {
                return Err("order_constit sequence must not be empty".to_owned());
            }
        }
        _ => {
            return Err(
                "order_constit must be 'PE', 'SNR', 'frequency', or a name sequence".to_owned(),
            );
        }
    }
    Ok(indices)
}

fn percent_energy(solution: &FittedSolution) -> &[f64] {
    match solution {
        FittedSolution::Scalar(solution) => &solution.percent_energy,
        FittedSolution::Vector(solution) => &solution.percent_energy,
    }
}

fn signal_to_noise(solution: &FittedSolution) -> Option<&[f64]> {
    match solution {
        FittedSolution::Scalar(solution) => solution.signal_to_noise.as_deref(),
        FittedSolution::Vector(solution) => solution.signal_to_noise.as_deref(),
    }
}

enum Reconstructed {
    Scalar(Vec<f64>),
    Vector(Vec<f64>, Vec<f64>),
}

impl FitState {
    fn reconstruct(
        &self,
        time_mjd: &[f64],
        filter: &ReconstructionFilter,
    ) -> Result<Reconstructed, String> {
        match (&self.model, &self.solution) {
            (PreparedModel::Direct(model), FittedSolution::Scalar(solution)) => model
                .reconstruct_modified_julian_days(time_mjd, solution, filter)
                .map(Reconstructed::Scalar)
                .map_err(|error| error.to_string()),
            (PreparedModel::Direct(model), FittedSolution::Vector(solution)) => model
                .reconstruct_vector_modified_julian_days(time_mjd, solution, filter)
                .map(|value| Reconstructed::Vector(value.eastward, value.northward))
                .map_err(|error| error.to_string()),
            (PreparedModel::ScalarInference(model), FittedSolution::Scalar(solution)) => model
                .reconstruct_modified_julian_days(time_mjd, solution, filter)
                .map(Reconstructed::Scalar)
                .map_err(|error| error.to_string()),
            (PreparedModel::VectorInference(model), FittedSolution::Vector(solution)) => model
                .reconstruct_vector_modified_julian_days(time_mjd, solution, filter)
                .map(|value| Reconstructed::Vector(value.eastward, value.northward))
                .map_err(|error| error.to_string()),
            _ => Err("internal model and solution type mismatch".to_owned()),
        }
    }
}

fn reconstruction_filter(
    constituent_names: Option<Vec<String>>,
    minimum_signal_to_noise: Option<f64>,
    minimum_percent_energy: f64,
) -> Result<ReconstructionFilter, String> {
    match constituent_names {
        Some(names) => Ok(ReconstructionFilter::Constituents(parse_constituents(
            &names,
        )?)),
        None if minimum_signal_to_noise.is_none() && minimum_percent_energy == 0.0 => {
            Ok(ReconstructionFilter::All)
        }
        None => Ok(ReconstructionFilter::Diagnostics {
            minimum_percent_energy,
            minimum_signal_to_noise,
        }),
    }
}

fn add_scalar_summary(
    output: &Bound<'_, PyDict>,
    solution: &ScalarSolution,
    order: &[usize],
) -> PyResult<()> {
    let amplitude = reordered(&solution.amplitude, order);
    let phase = reordered(&solution.phase_degrees, order);
    let percent_energy = reordered(&solution.percent_energy, order);
    let amplitude_ci = reordered_option(solution.amplitude_ci.as_deref(), order);
    let phase_ci = reordered_option(solution.phase_ci_degrees.as_deref(), order);
    let signal_to_noise = reordered_option(solution.signal_to_noise.as_deref(), order);
    output.set_item("A", &amplitude)?;
    output.set_item("g", &phase)?;
    output.set_item("A_ci", &amplitude_ci)?;
    output.set_item("g_ci", &phase_ci)?;
    output.set_item("PE", &percent_energy)?;
    output.set_item("SNR", &signal_to_noise)?;
    output.set_item("amplitude", &amplitude)?;
    output.set_item("phase_degrees", &phase)?;
    output.set_item("amplitude_ci", &amplitude_ci)?;
    output.set_item("phase_ci_degrees", &phase_ci)?;
    output.set_item("percent_energy", &percent_energy)?;
    output.set_item("signal_to_noise", &signal_to_noise)?;
    output.set_item("mean", solution.mean)?;
    output.set_item("slope", solution.slope_per_day)?;
    output.set_item("reference_time_mjd", solution.reference_time_days)?;
    Ok(())
}

fn add_vector_summary(
    output: &Bound<'_, PyDict>,
    solution: &VectorSolution,
    order: &[usize],
) -> PyResult<()> {
    let cartesian = solution
        .cartesian()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let semi_major = reordered(&solution.semi_major, order);
    let semi_minor = reordered(&solution.semi_minor, order);
    let inclination = reordered(&solution.inclination_degrees, order);
    let phase = reordered(&solution.phase_degrees, order);
    let percent_energy = reordered(&solution.percent_energy, order);
    let semi_major_ci = reordered_option(solution.semi_major_ci.as_deref(), order);
    let semi_minor_ci = reordered_option(solution.semi_minor_ci.as_deref(), order);
    let inclination_ci = reordered_option(solution.inclination_ci_degrees.as_deref(), order);
    let phase_ci = reordered_option(solution.phase_ci_degrees.as_deref(), order);
    let signal_to_noise = reordered_option(solution.signal_to_noise.as_deref(), order);
    output.set_item("Lsmaj", &semi_major)?;
    output.set_item("Lsmin", &semi_minor)?;
    output.set_item("theta", &inclination)?;
    output.set_item("g", &phase)?;
    output.set_item("Lsmaj_ci", &semi_major_ci)?;
    output.set_item("Lsmin_ci", &semi_minor_ci)?;
    output.set_item("theta_ci", &inclination_ci)?;
    output.set_item("g_ci", &phase_ci)?;
    output.set_item("PE", &percent_energy)?;
    output.set_item("SNR", &signal_to_noise)?;
    output.set_item("semi_major", &semi_major)?;
    output.set_item("semi_minor", &semi_minor)?;
    output.set_item("inclination_degrees", &inclination)?;
    output.set_item("phase_degrees", &phase)?;
    output.set_item("semi_major_ci", &semi_major_ci)?;
    output.set_item("semi_minor_ci", &semi_minor_ci)?;
    output.set_item("inclination_ci_degrees", &inclination_ci)?;
    output.set_item("phase_ci_degrees", &phase_ci)?;
    output.set_item("percent_energy", &percent_energy)?;
    output.set_item("signal_to_noise", &signal_to_noise)?;
    for (name, values) in [
        (
            "eastward_cosine_coefficient",
            &cartesian.eastward_cosine_coefficient,
        ),
        (
            "eastward_sine_coefficient",
            &cartesian.eastward_sine_coefficient,
        ),
        (
            "northward_cosine_coefficient",
            &cartesian.northward_cosine_coefficient,
        ),
        (
            "northward_sine_coefficient",
            &cartesian.northward_sine_coefficient,
        ),
    ] {
        output.set_item(name, reordered(values, order))?;
    }
    output.set_item("umean", solution.eastward_mean)?;
    output.set_item("vmean", solution.northward_mean)?;
    output.set_item("uslope", solution.eastward_slope_per_day)?;
    output.set_item("vslope", solution.northward_slope_per_day)?;
    output.set_item("reference_time_mjd", solution.reference_time_days)?;
    Ok(())
}

fn add_robust_summary(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    robust: Option<&rutide_core::RobustDiagnostics>,
) -> PyResult<()> {
    let Some(robust) = robust else {
        output.set_item("robust", py.None())?;
        output.set_item("weights", py.None())?;
        return Ok(());
    };
    let summary = PyDict::new(py);
    summary.set_item("iterations", robust.iterations)?;
    summary.set_item("termination", robust_termination_name(robust.termination))?;
    summary.set_item("residual_scale", robust.residual_scale)?;
    summary.set_item("ols_rms_residual", robust.ols_rms_residual)?;
    summary.set_item("rms_residual", robust.rms_residual)?;
    let weights = robust.weights.clone().into_pyarray(py);
    summary.set_item("weights", &weights)?;
    summary.set_item("leverage", robust.leverage.clone().into_pyarray(py))?;
    output.set_item("weights", weights)?;
    output.set_item("robust", summary)?;
    Ok(())
}

fn add_diagnostics_summary(
    py: Python<'_>,
    output: &Bound<'_, PyDict>,
    diagnostics: Option<&ConstituentSelectionDiagnostics>,
    order: &[usize],
) -> PyResult<()> {
    let Some(diagnostics) = diagnostics else {
        output.set_item("diagn", py.None())?;
        output.set_item("diagnostics", py.None())?;
        return Ok(());
    };
    let summary = diagnostics_summary(py, diagnostics, order)?;
    output.set_item("diagn", &summary)?;
    output.set_item("diagnostics", summary)?;
    Ok(())
}

fn diagnostics_summary<'py>(
    py: Python<'py>,
    diagnostics: &ConstituentSelectionDiagnostics,
    order: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    let summary = PyDict::new(py);
    let lower = neighbor_diagnostics_summary(py, diagnostics, order, false)?;
    let higher = neighbor_diagnostics_summary(py, diagnostics, order, true)?;
    summary.set_item("lo", &lower)?;
    summary.set_item("lower", lower)?;
    summary.set_item("hi", &higher)?;
    summary.set_item("higher", higher)?;
    summary.set_item("K", diagnostics.whole_model.basis_condition_number)?;
    summary.set_item(
        "SNRallc",
        diagnostics.whole_model.all_constituent_signal_to_noise,
    )?;
    summary.set_item(
        "SNRallc_over_K",
        diagnostics.whole_model.condition_adjusted_signal_to_noise,
    )?;
    summary.set_item("TVraw", diagnostics.tidal_variance.raw_tidal_variance)?;
    summary.set_item(
        "TVallc",
        diagnostics.tidal_variance.all_constituent_tidal_variance,
    )?;
    summary.set_item(
        "TVsnrc",
        diagnostics
            .tidal_variance
            .significant_constituent_tidal_variance,
    )?;
    summary.set_item(
        "PTVallc",
        diagnostics
            .tidal_variance
            .all_constituent_percent_tidal_variance,
    )?;
    summary.set_item(
        "PTVsnrc",
        diagnostics
            .tidal_variance
            .significant_constituent_percent_tidal_variance,
    )?;
    summary.set_item("Rayleigh_min", diagnostics.rayleigh_minimum)?;
    summary.set_item("min_SNR", diagnostics.minimum_signal_to_noise)?;
    Ok(summary)
}

fn neighbor_diagnostics_summary<'py>(
    py: Python<'py>,
    diagnostics: &ConstituentSelectionDiagnostics,
    order: &[usize],
    higher: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut displayed_index = vec![-1_i64; diagnostics.constituents.len()];
    for (index, source) in order.iter().copied().enumerate() {
        displayed_index[source] = i64::try_from(index).expect("constituent count fits i64");
    }
    let mut indices = Vec::with_capacity(order.len());
    let mut names = Vec::with_capacity(order.len());
    let mut frequencies = Vec::with_capacity(order.len());
    let mut rayleigh = Vec::with_capacity(order.len());
    let mut noise_modified_rayleigh = Vec::with_capacity(order.len());
    let mut maximum_correlation = Vec::with_capacity(order.len());
    for source in order.iter().copied() {
        let independence = &diagnostics.constituents[source];
        let neighbor = if higher {
            independence.higher.as_ref()
        } else {
            independence.lower.as_ref()
        };
        if let Some(neighbor) = neighbor {
            indices.push(displayed_index[neighbor.index]);
            names.push(neighbor.name.clone());
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
            names.push(String::new());
            frequencies.push(f64::NAN);
            rayleigh.push(f64::NAN);
            noise_modified_rayleigh.push(f64::NAN);
            maximum_correlation.push(f64::NAN);
        }
    }
    let summary = PyDict::new(py);
    summary.set_item("index", indices)?;
    summary.set_item("name", names)?;
    summary.set_item("frequency_cph", frequencies)?;
    summary.set_item("RR", &rayleigh)?;
    summary.set_item("rayleigh_criterion", rayleigh)?;
    summary.set_item("RNM", &noise_modified_rayleigh)?;
    summary.set_item("noise_modified_rayleigh_criterion", noise_modified_rayleigh)?;
    summary.set_item("CorMx", &maximum_correlation)?;
    summary.set_item("maximum_correlation", maximum_correlation)?;
    Ok(summary)
}

const fn robust_termination_name(termination: RobustTermination) -> &'static str {
    match termination {
        RobustTermination::Tolerance => "tolerance",
        RobustTermination::ObjectiveIncrease => "objective_increase",
        RobustTermination::ExactFit => "exact_fit",
    }
}

fn reordered<T: Clone>(values: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|index| values[*index].clone()).collect()
}

fn reordered_option(values: Option<&[f64]>, order: &[usize]) -> Option<Vec<f64>> {
    values.map(|values| reordered(values, order))
}

fn contiguous_copy(array: &PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<Vec<f64>> {
    array
        .as_slice()
        .map(<[f64]>::to_vec)
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float64 array")))
}
