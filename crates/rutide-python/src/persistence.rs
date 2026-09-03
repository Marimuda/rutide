//! Versioned, pickle-free snapshots for one-series native fits.

use std::collections::HashSet;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rutide_core::{
    ConstituentIndependenceDiagnostics, ConstituentSelectionDiagnostics, FitOptions,
    GreenwichNodalOls, InferenceMode, NeighboringConstituentDiagnostics, RobustDiagnostics,
    RobustTermination, ScalarInferenceOls, ScalarSolution, SolverOptions, TidalVarianceDiagnostics,
    VectorInferenceOls, VectorSolution, WholeModelIndependenceDiagnostics,
};

use super::{
    Fit, FitState, FittedSolution, PreparedModel, SolveConfig, constituent_metadata,
    normalized_confidence_name, parse_confidence, parse_constituents, parse_method_and_robust,
    parse_nodal_corrections, parse_phase_reference, scalar_inference_relations,
    validate_empty_inference, vector_inference_relations,
};

pub(crate) const SNAPSHOT_SCHEMA_VERSION: u32 = 2;

pub(super) fn fit_snapshot<'py>(state: &FitState, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let snapshot = snapshot_header(py, "single")?;
    snapshot.set_item("time_mjd", state.time_mjd.clone().into_pyarray(py))?;
    snapshot.set_item("config", config_snapshot(&state.config, py)?)?;
    snapshot.set_item("original_observations", state.original_observations)?;
    snapshot.set_item(
        "presentation_order",
        state.presentation_order.clone().into_pyarray(py),
    )?;
    snapshot.set_item(
        "diagnostics",
        diagnostics_snapshot(state.diagnostics.as_ref(), py)?,
    )?;
    match &state.solution {
        FittedSolution::Scalar(solution) => {
            snapshot.set_item("solution_kind", "scalar")?;
            snapshot.set_item("solution", scalar_solution_snapshot(solution, py)?)?;
        }
        FittedSolution::Vector(solution) => {
            snapshot.set_item("solution_kind", "vector")?;
            snapshot.set_item("solution", vector_solution_snapshot(solution, py)?)?;
        }
    }
    Ok(snapshot)
}

#[pyfunction]
pub(super) fn restore_fit(py: Python<'_>, snapshot: &Bound<'_, PyDict>) -> PyResult<Py<Fit>> {
    let schema_version = validate_header(snapshot, "single")?;
    let time_mjd = required_vec_f64(snapshot, "time_mjd")?;
    let retained_observations = time_mjd.len();
    let config = config_from_snapshot(&required_dict(snapshot, "config")?)?;
    let original_observations = required_extract::<usize>(snapshot, "original_observations")?;
    if original_observations < time_mjd.len() {
        return Err(snapshot_error(
            "original_observations is smaller than the retained timestamp count",
        ));
    }
    let presentation_order = required_vec_usize(snapshot, "presentation_order")?;
    let solution_kind = required_extract::<String>(snapshot, "solution_kind")?;
    let solution_dict = required_dict(snapshot, "solution")?;
    let constituents = parse_constituents(
        config
            .constituent_names
            .as_deref()
            .ok_or_else(|| snapshot_error("persisted constituent_names cannot be None"))?,
    )
    .map_err(snapshot_error)?;
    let phase_reference = parse_phase_reference(&config.phase_name).map_err(snapshot_error)?;
    let nodal_corrections = parse_nodal_corrections(&config.nodal_name).map_err(snapshot_error)?;
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
    )
    .map_err(snapshot_error)?;
    parse_method_and_robust(&config).map_err(snapshot_error)?;
    let inference_mode = if config.approximate_inference {
        InferenceMode::Approximate
    } else {
        InferenceMode::Exact
    };

    let (model, solution, names, frequencies) = restore_model_and_solution(
        &time_mjd,
        &config,
        &solution_kind,
        &solution_dict,
        &constituents,
        solver_options,
        inference_mode,
    )?;
    validate_presentation_order(&presentation_order, names.len())?;
    validate_method_diagnostics(&config.method_name, &solution, retained_observations)?;
    let diagnostics = diagnostics_from_snapshot(snapshot, schema_version, &names, &frequencies)?;
    if diagnostics.is_some() != config.diagnostics {
        return Err(snapshot_error(
            "diagnostic config and persisted diagnostic presence disagree",
        ));
    }
    Py::new(
        py,
        Fit {
            inner: std::sync::Arc::new(FitState {
                model,
                solution,
                time_mjd,
                config: config.clone(),
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
        },
    )
}

type RestoredFit = (PreparedModel, FittedSolution, Vec<String>, Vec<f64>);

#[allow(clippy::too_many_arguments)]
fn restore_model_and_solution(
    time_mjd: &[f64],
    config: &SolveConfig,
    solution_kind: &str,
    solution: &Bound<'_, PyDict>,
    constituents: &[rutide_core::TidalConstituent],
    solver_options: SolverOptions,
    inference_mode: InferenceMode,
) -> PyResult<RestoredFit> {
    match solution_kind {
        "scalar" => restore_scalar_model(
            time_mjd,
            config,
            solution,
            constituents,
            solver_options,
            inference_mode,
        ),
        "vector" => restore_vector_model(
            time_mjd,
            config,
            solution,
            constituents,
            solver_options,
            inference_mode,
        ),
        _ => Err(snapshot_error(
            "solution_kind must be either 'scalar' or 'vector'",
        )),
    }
}

fn restore_scalar_model(
    time_mjd: &[f64],
    config: &SolveConfig,
    solution: &Bound<'_, PyDict>,
    constituents: &[rutide_core::TidalConstituent],
    solver_options: SolverOptions,
    inference_mode: InferenceMode,
) -> PyResult<RestoredFit> {
    if config.inferred_names.is_empty() {
        validate_empty_inference(config).map_err(snapshot_error)?;
        let model = GreenwichNodalOls::prepare_modified_julian_days_with_solver_options(
            time_mjd,
            config.latitude,
            constituents,
            solver_options,
        )
        .map_err(|error| snapshot_error(error.to_string()))?;
        let (names, frequencies) = constituent_metadata(model.constituents());
        let solution = scalar_solution_from_snapshot(solution, names.len())?;
        return Ok((
            PreparedModel::Direct(Box::new(model)),
            FittedSolution::Scalar(solution),
            names,
            frequencies,
        ));
    }
    let relationships = scalar_inference_relations(config).map_err(snapshot_error)?;
    let model = ScalarInferenceOls::prepare_modified_julian_days_with_solver_options(
        time_mjd,
        config.latitude,
        constituents,
        &relationships,
        inference_mode,
        solver_options,
    )
    .map_err(|error| snapshot_error(error.to_string()))?;
    let (names, frequencies) = constituent_metadata(model.constituents());
    let solution = scalar_solution_from_snapshot(solution, names.len())?;
    Ok((
        PreparedModel::ScalarInference(Box::new(model)),
        FittedSolution::Scalar(solution),
        names,
        frequencies,
    ))
}

fn restore_vector_model(
    time_mjd: &[f64],
    config: &SolveConfig,
    solution: &Bound<'_, PyDict>,
    constituents: &[rutide_core::TidalConstituent],
    solver_options: SolverOptions,
    inference_mode: InferenceMode,
) -> PyResult<RestoredFit> {
    if config.inferred_names.is_empty() {
        validate_empty_inference(config).map_err(snapshot_error)?;
        let model = GreenwichNodalOls::prepare_modified_julian_days_with_solver_options(
            time_mjd,
            config.latitude,
            constituents,
            solver_options,
        )
        .map_err(|error| snapshot_error(error.to_string()))?;
        let (names, frequencies) = constituent_metadata(model.constituents());
        let solution = vector_solution_from_snapshot(solution, names.len())?;
        return Ok((
            PreparedModel::Direct(Box::new(model)),
            FittedSolution::Vector(solution),
            names,
            frequencies,
        ));
    }
    let relationships = vector_inference_relations(config).map_err(snapshot_error)?;
    let model = VectorInferenceOls::prepare_modified_julian_days_with_solver_options(
        time_mjd,
        config.latitude,
        constituents,
        &relationships,
        inference_mode,
        solver_options,
    )
    .map_err(|error| snapshot_error(error.to_string()))?;
    let (names, frequencies) = constituent_metadata(model.constituents());
    let solution = vector_solution_from_snapshot(solution, names.len())?;
    Ok((
        PreparedModel::VectorInference(Box::new(model)),
        FittedSolution::Vector(solution),
        names,
        frequencies,
    ))
}

pub(crate) fn snapshot_header<'py>(py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyDict>> {
    let snapshot = PyDict::new(py);
    snapshot.set_item("schema_version", SNAPSHOT_SCHEMA_VERSION)?;
    snapshot.set_item("kind", kind)?;
    snapshot.set_item("rutide_version", rutide_core::VERSION)?;
    Ok(snapshot)
}

pub(crate) fn validate_header(snapshot: &Bound<'_, PyDict>, kind: &str) -> PyResult<u32> {
    let schema_version = required_extract::<u32>(snapshot, "schema_version")?;
    if !(1..=SNAPSHOT_SCHEMA_VERSION).contains(&schema_version) {
        return Err(snapshot_error(format!(
            "unsupported coefficient snapshot schema {schema_version}; expected 1 or {SNAPSHOT_SCHEMA_VERSION}"
        )));
    }
    let actual_kind = required_extract::<String>(snapshot, "kind")?;
    if actual_kind != kind {
        return Err(snapshot_error(format!(
            "coefficient snapshot kind is '{actual_kind}', expected '{kind}'"
        )));
    }
    let writer_version = required_extract::<String>(snapshot, "rutide_version")?;
    let expected_major_minor = major_minor(rutide_core::VERSION);
    let writer_major_minor = major_minor(&writer_version);
    let compatible_legacy =
        schema_version == 1 && writer_major_minor == "0.2" && expected_major_minor == "0.3";
    if writer_major_minor != expected_major_minor && !compatible_legacy {
        return Err(snapshot_error(format!(
            "coefficient snapshot was written by incompatible RUTide {writer_version}; expected {expected_major_minor}.x"
        )));
    }
    Ok(schema_version)
}

fn major_minor(version: &str) -> String {
    version.split('.').take(2).collect::<Vec<_>>().join(".")
}

pub(crate) fn config_snapshot<'py>(
    config: &SolveConfig,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("latitude", config.latitude)?;
    output.set_item("constituent_names", &config.constituent_names)?;
    output.set_item("rayleigh_min", config.rayleigh_min)?;
    output.set_item("diagnostics", config.diagnostics)?;
    output.set_item(
        "diagnostic_min_signal_to_noise",
        config.diagnostic_min_signal_to_noise,
    )?;
    output.set_item("method_name", &config.method_name)?;
    output.set_item("confidence_name", &config.confidence_name)?;
    output.set_item("white", config.white)?;
    output.set_item("trend", config.trend)?;
    output.set_item("phase_name", &config.phase_name)?;
    output.set_item("nodal_name", &config.nodal_name)?;
    output.set_item("monte_carlo_realizations", config.monte_carlo_realizations)?;
    output.set_item("monte_carlo_seed", config.monte_carlo_seed)?;
    output.set_item("robust_weight_name", &config.robust_weight_name)?;
    output.set_item("robust_tuning", config.robust_tuning)?;
    output.set_item("robust_tolerance", config.robust_tolerance)?;
    output.set_item("robust_max_iterations", config.robust_max_iterations)?;
    output.set_item("inferred_names", &config.inferred_names)?;
    output.set_item("reference_names", &config.reference_names)?;
    output.set_item("inference_ratios", &config.inference_ratios)?;
    output.set_item("inference_phase_offsets", &config.inference_phase_offsets)?;
    output.set_item("approximate_inference", config.approximate_inference)?;
    output.set_item("order_name", &config.order_name)?;
    output.set_item("order_names", &config.order_names)?;
    Ok(output)
}

pub(crate) fn config_from_snapshot(input: &Bound<'_, PyDict>) -> PyResult<SolveConfig> {
    Ok(SolveConfig {
        latitude: required_extract(input, "latitude")?,
        constituent_names: Some(required_extract(input, "constituent_names")?),
        rayleigh_min: required_extract(input, "rayleigh_min")?,
        diagnostics: optional_extract(input, "diagnostics")?.unwrap_or(false),
        diagnostic_min_signal_to_noise: optional_extract(input, "diagnostic_min_signal_to_noise")?
            .unwrap_or(2.0),
        method_name: required_extract(input, "method_name")?,
        confidence_name: required_extract(input, "confidence_name")?,
        white: required_extract(input, "white")?,
        trend: required_extract(input, "trend")?,
        phase_name: required_extract(input, "phase_name")?,
        nodal_name: required_extract(input, "nodal_name")?,
        monte_carlo_realizations: required_extract(input, "monte_carlo_realizations")?,
        monte_carlo_seed: required_extract(input, "monte_carlo_seed")?,
        robust_weight_name: required_extract(input, "robust_weight_name")?,
        robust_tuning: optional_float(input, "robust_tuning")?,
        robust_tolerance: required_extract(input, "robust_tolerance")?,
        robust_max_iterations: required_extract(input, "robust_max_iterations")?,
        inferred_names: required_extract(input, "inferred_names")?,
        reference_names: required_extract(input, "reference_names")?,
        inference_ratios: required_extract(input, "inference_ratios")?,
        inference_phase_offsets: required_extract(input, "inference_phase_offsets")?,
        approximate_inference: required_extract(input, "approximate_inference")?,
        order_name: required_extract(input, "order_name")?,
        order_names: required_extract(input, "order_names")?,
    })
}

pub(crate) fn diagnostics_snapshot(
    diagnostics: Option<&ConstituentSelectionDiagnostics>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let Some(diagnostics) = diagnostics else {
        return Ok(py.None());
    };
    let output = PyDict::new(py);
    let mut lower_index = Vec::with_capacity(diagnostics.constituents.len());
    let mut lower_rayleigh = Vec::with_capacity(diagnostics.constituents.len());
    let mut lower_noise_modified = Vec::with_capacity(diagnostics.constituents.len());
    let mut lower_correlation = Vec::with_capacity(diagnostics.constituents.len());
    let mut higher_index = Vec::with_capacity(diagnostics.constituents.len());
    let mut higher_rayleigh = Vec::with_capacity(diagnostics.constituents.len());
    let mut higher_noise_modified = Vec::with_capacity(diagnostics.constituents.len());
    let mut higher_correlation = Vec::with_capacity(diagnostics.constituents.len());
    for constituent in &diagnostics.constituents {
        append_neighbor_snapshot(
            constituent.lower.as_ref(),
            &mut lower_index,
            &mut lower_rayleigh,
            &mut lower_noise_modified,
            &mut lower_correlation,
        );
        append_neighbor_snapshot(
            constituent.higher.as_ref(),
            &mut higher_index,
            &mut higher_rayleigh,
            &mut higher_noise_modified,
            &mut higher_correlation,
        );
    }
    for (name, values) in [
        ("lower_rayleigh", lower_rayleigh),
        ("lower_noise_modified_rayleigh", lower_noise_modified),
        ("lower_maximum_correlation", lower_correlation),
        ("higher_rayleigh", higher_rayleigh),
        ("higher_noise_modified_rayleigh", higher_noise_modified),
        ("higher_maximum_correlation", higher_correlation),
    ] {
        output.set_item(name, values.into_pyarray(py))?;
    }
    output.set_item("lower_index", lower_index.into_pyarray(py))?;
    output.set_item("higher_index", higher_index.into_pyarray(py))?;
    output.set_item(
        "basis_condition_number",
        diagnostics.whole_model.basis_condition_number,
    )?;
    output.set_item(
        "all_constituent_signal_to_noise",
        diagnostics.whole_model.all_constituent_signal_to_noise,
    )?;
    output.set_item(
        "condition_adjusted_signal_to_noise",
        diagnostics.whole_model.condition_adjusted_signal_to_noise,
    )?;
    output.set_item(
        "raw_tidal_variance",
        diagnostics.tidal_variance.raw_tidal_variance,
    )?;
    output.set_item(
        "all_constituent_tidal_variance",
        diagnostics.tidal_variance.all_constituent_tidal_variance,
    )?;
    output.set_item(
        "significant_constituent_tidal_variance",
        diagnostics
            .tidal_variance
            .significant_constituent_tidal_variance,
    )?;
    output.set_item(
        "all_constituent_percent_tidal_variance",
        diagnostics
            .tidal_variance
            .all_constituent_percent_tidal_variance,
    )?;
    output.set_item(
        "significant_constituent_percent_tidal_variance",
        diagnostics
            .tidal_variance
            .significant_constituent_percent_tidal_variance,
    )?;
    output.set_item("rayleigh_minimum", diagnostics.rayleigh_minimum)?;
    output.set_item(
        "minimum_signal_to_noise",
        diagnostics.minimum_signal_to_noise,
    )?;
    Ok(output.into_any().unbind())
}

#[allow(
    clippy::too_many_lines,
    reason = "writes the versioned dense batch diagnostic record in one auditable mapping"
)]
pub(crate) fn batch_diagnostics_snapshot(
    diagnostics: Option<&[ConstituentSelectionDiagnostics]>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let Some(diagnostics) = diagnostics else {
        return Ok(py.None());
    };
    let total_constituents = diagnostics
        .iter()
        .map(|value| value.constituents.len())
        .sum();
    let mut lower_index = Vec::with_capacity(total_constituents);
    let mut lower_rayleigh = Vec::with_capacity(total_constituents);
    let mut lower_noise_modified = Vec::with_capacity(total_constituents);
    let mut lower_correlation = Vec::with_capacity(total_constituents);
    let mut higher_index = Vec::with_capacity(total_constituents);
    let mut higher_rayleigh = Vec::with_capacity(total_constituents);
    let mut higher_noise_modified = Vec::with_capacity(total_constituents);
    let mut higher_correlation = Vec::with_capacity(total_constituents);
    for diagnostic in diagnostics {
        for constituent in &diagnostic.constituents {
            append_neighbor_snapshot(
                constituent.lower.as_ref(),
                &mut lower_index,
                &mut lower_rayleigh,
                &mut lower_noise_modified,
                &mut lower_correlation,
            );
            append_neighbor_snapshot(
                constituent.higher.as_ref(),
                &mut higher_index,
                &mut higher_rayleigh,
                &mut higher_noise_modified,
                &mut higher_correlation,
            );
        }
    }
    let output = PyDict::new(py);
    for (name, values) in [
        ("lower_rayleigh", lower_rayleigh),
        ("lower_noise_modified_rayleigh", lower_noise_modified),
        ("lower_maximum_correlation", lower_correlation),
        ("higher_rayleigh", higher_rayleigh),
        ("higher_noise_modified_rayleigh", higher_noise_modified),
        ("higher_maximum_correlation", higher_correlation),
    ] {
        output.set_item(name, values.into_pyarray(py))?;
    }
    output.set_item("lower_index", lower_index.into_pyarray(py))?;
    output.set_item("higher_index", higher_index.into_pyarray(py))?;
    for (name, values) in [
        (
            "basis_condition_number",
            diagnostics
                .iter()
                .map(|value| value.whole_model.basis_condition_number)
                .collect::<Vec<_>>(),
        ),
        (
            "all_constituent_signal_to_noise",
            diagnostics
                .iter()
                .map(|value| {
                    value
                        .whole_model
                        .all_constituent_signal_to_noise
                        .unwrap_or(f64::NAN)
                })
                .collect(),
        ),
        (
            "condition_adjusted_signal_to_noise",
            diagnostics
                .iter()
                .map(|value| {
                    value
                        .whole_model
                        .condition_adjusted_signal_to_noise
                        .unwrap_or(f64::NAN)
                })
                .collect(),
        ),
        (
            "raw_tidal_variance",
            diagnostics
                .iter()
                .map(|value| value.tidal_variance.raw_tidal_variance)
                .collect(),
        ),
        (
            "all_constituent_tidal_variance",
            diagnostics
                .iter()
                .map(|value| value.tidal_variance.all_constituent_tidal_variance)
                .collect(),
        ),
        (
            "significant_constituent_tidal_variance",
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
            "all_constituent_percent_tidal_variance",
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
            "significant_constituent_percent_tidal_variance",
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
        (
            "rayleigh_minimum",
            diagnostics
                .iter()
                .map(|value| value.rayleigh_minimum)
                .collect(),
        ),
        (
            "minimum_signal_to_noise",
            diagnostics
                .iter()
                .map(|value| value.minimum_signal_to_noise)
                .collect(),
        ),
    ] {
        output.set_item(name, values.into_pyarray(py))?;
    }
    Ok(output.into_any().unbind())
}

fn append_neighbor_snapshot(
    neighbor: Option<&NeighboringConstituentDiagnostics>,
    indices: &mut Vec<i64>,
    rayleigh: &mut Vec<f64>,
    noise_modified: &mut Vec<f64>,
    correlation: &mut Vec<f64>,
) {
    if let Some(neighbor) = neighbor {
        indices.push(i64::try_from(neighbor.index).expect("constituent count fits i64"));
        rayleigh.push(neighbor.rayleigh_criterion);
        noise_modified.push(
            neighbor
                .noise_modified_rayleigh_criterion
                .unwrap_or(f64::NAN),
        );
        correlation.push(neighbor.maximum_correlation.unwrap_or(f64::NAN));
    } else {
        indices.push(-1);
        rayleigh.push(f64::NAN);
        noise_modified.push(f64::NAN);
        correlation.push(f64::NAN);
    }
}

fn diagnostics_from_snapshot(
    snapshot: &Bound<'_, PyDict>,
    schema_version: u32,
    names: &[String],
    frequencies: &[f64],
) -> PyResult<Option<ConstituentSelectionDiagnostics>> {
    if schema_version == 1 {
        return Ok(None);
    }
    let value = required(snapshot, "diagnostics")?;
    if value.is_none() {
        return Ok(None);
    }
    let input = value
        .cast::<PyDict>()
        .map_err(|_| snapshot_error("diagnostics must be a dictionary or None"))?;
    diagnostics_value_from_snapshot(input, names, frequencies).map(Some)
}

#[allow(
    clippy::too_many_lines,
    reason = "validates every dense batch diagnostic field before native reconstruction"
)]
pub(crate) fn batch_diagnostics_from_snapshot(
    snapshot: &Bound<'_, PyDict>,
    schema_version: u32,
    names: &[String],
    frequencies: &[Vec<f64>],
) -> PyResult<Option<Vec<ConstituentSelectionDiagnostics>>> {
    if schema_version == 1 {
        return Ok(None);
    }
    let value = required(snapshot, "diagnostics")?;
    if value.is_none() {
        return Ok(None);
    }
    let input = value
        .cast::<PyDict>()
        .map_err(|_| snapshot_error("diagnostics must be a dictionary or None"))?;
    let series_count = frequencies.len();
    let constituent_count = names.len();
    let flattened_count = series_count.saturating_mul(constituent_count);
    let lower_index = required_vec_i64(input, "lower_index")?;
    let lower_rayleigh = required_vec_f64(input, "lower_rayleigh")?;
    let lower_noise_modified = required_vec_f64(input, "lower_noise_modified_rayleigh")?;
    let lower_correlation = required_vec_f64(input, "lower_maximum_correlation")?;
    let higher_index = required_vec_i64(input, "higher_index")?;
    let higher_rayleigh = required_vec_f64(input, "higher_rayleigh")?;
    let higher_noise_modified = required_vec_f64(input, "higher_noise_modified_rayleigh")?;
    let higher_correlation = required_vec_f64(input, "higher_maximum_correlation")?;
    for (field, length) in [
        ("lower_index", lower_index.len()),
        ("lower_rayleigh", lower_rayleigh.len()),
        ("lower_noise_modified_rayleigh", lower_noise_modified.len()),
        ("lower_maximum_correlation", lower_correlation.len()),
        ("higher_index", higher_index.len()),
        ("higher_rayleigh", higher_rayleigh.len()),
        (
            "higher_noise_modified_rayleigh",
            higher_noise_modified.len(),
        ),
        ("higher_maximum_correlation", higher_correlation.len()),
    ] {
        validate_diagnostic_length(field, length, flattened_count)?;
    }
    let basis_condition_number = required_vec_f64(input, "basis_condition_number")?;
    let all_constituent_signal_to_noise =
        required_vec_f64(input, "all_constituent_signal_to_noise")?;
    let condition_adjusted_signal_to_noise =
        required_vec_f64(input, "condition_adjusted_signal_to_noise")?;
    let raw_tidal_variance = required_vec_f64(input, "raw_tidal_variance")?;
    let all_constituent_tidal_variance = required_vec_f64(input, "all_constituent_tidal_variance")?;
    let significant_constituent_tidal_variance =
        required_vec_f64(input, "significant_constituent_tidal_variance")?;
    let all_constituent_percent_tidal_variance =
        required_vec_f64(input, "all_constituent_percent_tidal_variance")?;
    let significant_constituent_percent_tidal_variance =
        required_vec_f64(input, "significant_constituent_percent_tidal_variance")?;
    let rayleigh_minimum = required_vec_f64(input, "rayleigh_minimum")?;
    let minimum_signal_to_noise = required_vec_f64(input, "minimum_signal_to_noise")?;
    for (field, length) in [
        ("basis_condition_number", basis_condition_number.len()),
        (
            "all_constituent_signal_to_noise",
            all_constituent_signal_to_noise.len(),
        ),
        (
            "condition_adjusted_signal_to_noise",
            condition_adjusted_signal_to_noise.len(),
        ),
        ("raw_tidal_variance", raw_tidal_variance.len()),
        (
            "all_constituent_tidal_variance",
            all_constituent_tidal_variance.len(),
        ),
        (
            "significant_constituent_tidal_variance",
            significant_constituent_tidal_variance.len(),
        ),
        (
            "all_constituent_percent_tidal_variance",
            all_constituent_percent_tidal_variance.len(),
        ),
        (
            "significant_constituent_percent_tidal_variance",
            significant_constituent_percent_tidal_variance.len(),
        ),
        ("rayleigh_minimum", rayleigh_minimum.len()),
        ("minimum_signal_to_noise", minimum_signal_to_noise.len()),
    ] {
        validate_diagnostic_length(field, length, series_count)?;
    }
    if frequencies
        .iter()
        .any(|values| values.len() != constituent_count)
    {
        return Err(snapshot_error(
            "batch diagnostic frequency rows do not match constituent count",
        ));
    }

    (0..series_count)
        .map(|series| {
            let offset = series * constituent_count;
            let constituents = (0..constituent_count)
                .map(|source| {
                    let position = offset + source;
                    Ok(ConstituentIndependenceDiagnostics {
                        lower: neighbor_from_snapshot(
                            source,
                            lower_index[position],
                            lower_rayleigh[position],
                            lower_noise_modified[position],
                            lower_correlation[position],
                            names,
                            &frequencies[series],
                        )?,
                        higher: neighbor_from_snapshot(
                            source,
                            higher_index[position],
                            higher_rayleigh[position],
                            higher_noise_modified[position],
                            higher_correlation[position],
                            names,
                            &frequencies[series],
                        )?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            validated_diagnostics(
                constituents,
                basis_condition_number[series],
                nan_to_option(all_constituent_signal_to_noise[series]),
                nan_to_option(condition_adjusted_signal_to_noise[series]),
                raw_tidal_variance[series],
                all_constituent_tidal_variance[series],
                nan_to_option(significant_constituent_tidal_variance[series]),
                nan_to_option(all_constituent_percent_tidal_variance[series]),
                nan_to_option(significant_constituent_percent_tidal_variance[series]),
                rayleigh_minimum[series],
                minimum_signal_to_noise[series],
            )
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

fn validate_diagnostic_length(field: &str, actual: usize, expected: usize) -> PyResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(snapshot_error(format!(
            "diagnostic field '{field}' has length {actual}, expected {expected}"
        )))
    }
}

const fn nan_to_option(value: f64) -> Option<f64> {
    if value.is_nan() { None } else { Some(value) }
}

#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the compact, versioned diagnostic persistence record"
)]
fn validated_diagnostics(
    constituents: Vec<ConstituentIndependenceDiagnostics>,
    basis_condition_number: f64,
    all_constituent_signal_to_noise: Option<f64>,
    condition_adjusted_signal_to_noise: Option<f64>,
    raw_tidal_variance: f64,
    all_constituent_tidal_variance: f64,
    significant_constituent_tidal_variance: Option<f64>,
    all_constituent_percent_tidal_variance: Option<f64>,
    significant_constituent_percent_tidal_variance: Option<f64>,
    rayleigh_minimum: f64,
    minimum_signal_to_noise: f64,
) -> PyResult<ConstituentSelectionDiagnostics> {
    validate_neighbor_symmetry(&constituents)?;
    validate_positive_or_infinite("basis_condition_number", basis_condition_number)?;
    validate_optional_nonnegative(
        "all_constituent_signal_to_noise",
        all_constituent_signal_to_noise,
        true,
    )?;
    validate_optional_nonnegative(
        "condition_adjusted_signal_to_noise",
        condition_adjusted_signal_to_noise,
        true,
    )?;
    validate_nonnegative_finite("raw_tidal_variance", raw_tidal_variance)?;
    validate_nonnegative_finite(
        "all_constituent_tidal_variance",
        all_constituent_tidal_variance,
    )?;
    for (field, value) in [
        (
            "significant_constituent_tidal_variance",
            significant_constituent_tidal_variance,
        ),
        (
            "all_constituent_percent_tidal_variance",
            all_constituent_percent_tidal_variance,
        ),
        (
            "significant_constituent_percent_tidal_variance",
            significant_constituent_percent_tidal_variance,
        ),
    ] {
        validate_optional_nonnegative(field, value, false)?;
    }
    validate_positive_finite("rayleigh_minimum", rayleigh_minimum)?;
    validate_nonnegative_finite("minimum_signal_to_noise", minimum_signal_to_noise)?;
    Ok(ConstituentSelectionDiagnostics {
        constituents,
        whole_model: WholeModelIndependenceDiagnostics {
            basis_condition_number,
            all_constituent_signal_to_noise,
            condition_adjusted_signal_to_noise,
        },
        tidal_variance: TidalVarianceDiagnostics {
            raw_tidal_variance,
            all_constituent_tidal_variance,
            significant_constituent_tidal_variance,
            all_constituent_percent_tidal_variance,
            significant_constituent_percent_tidal_variance,
        },
        rayleigh_minimum,
        minimum_signal_to_noise,
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "validates every single-fit diagnostic field before native reconstruction"
)]
pub(crate) fn diagnostics_value_from_snapshot(
    input: &Bound<'_, PyDict>,
    names: &[String],
    frequencies: &[f64],
) -> PyResult<ConstituentSelectionDiagnostics> {
    let count = names.len();
    if frequencies.len() != count {
        return Err(snapshot_error(
            "diagnostic name and frequency counts do not match",
        ));
    }
    let lower_index = required_vec_i64(input, "lower_index")?;
    let lower_rayleigh = required_vec_f64(input, "lower_rayleigh")?;
    let lower_noise_modified = required_vec_f64(input, "lower_noise_modified_rayleigh")?;
    let lower_correlation = required_vec_f64(input, "lower_maximum_correlation")?;
    let higher_index = required_vec_i64(input, "higher_index")?;
    let higher_rayleigh = required_vec_f64(input, "higher_rayleigh")?;
    let higher_noise_modified = required_vec_f64(input, "higher_noise_modified_rayleigh")?;
    let higher_correlation = required_vec_f64(input, "higher_maximum_correlation")?;
    for (field, length) in [
        ("lower_index", lower_index.len()),
        ("lower_rayleigh", lower_rayleigh.len()),
        ("lower_noise_modified_rayleigh", lower_noise_modified.len()),
        ("lower_maximum_correlation", lower_correlation.len()),
        ("higher_index", higher_index.len()),
        ("higher_rayleigh", higher_rayleigh.len()),
        (
            "higher_noise_modified_rayleigh",
            higher_noise_modified.len(),
        ),
        ("higher_maximum_correlation", higher_correlation.len()),
    ] {
        if length != count {
            return Err(snapshot_error(format!(
                "diagnostic field '{field}' has length {length}, expected {count}"
            )));
        }
    }
    let constituents = (0..count)
        .map(|source| {
            Ok(ConstituentIndependenceDiagnostics {
                lower: neighbor_from_snapshot(
                    source,
                    lower_index[source],
                    lower_rayleigh[source],
                    lower_noise_modified[source],
                    lower_correlation[source],
                    names,
                    frequencies,
                )?,
                higher: neighbor_from_snapshot(
                    source,
                    higher_index[source],
                    higher_rayleigh[source],
                    higher_noise_modified[source],
                    higher_correlation[source],
                    names,
                    frequencies,
                )?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    validate_neighbor_symmetry(&constituents)?;

    let basis_condition_number = required_extract(input, "basis_condition_number")?;
    validate_positive_or_infinite("basis_condition_number", basis_condition_number)?;
    let all_constituent_signal_to_noise = optional_float(input, "all_constituent_signal_to_noise")?;
    validate_optional_nonnegative(
        "all_constituent_signal_to_noise",
        all_constituent_signal_to_noise,
        true,
    )?;
    let condition_adjusted_signal_to_noise =
        optional_float(input, "condition_adjusted_signal_to_noise")?;
    validate_optional_nonnegative(
        "condition_adjusted_signal_to_noise",
        condition_adjusted_signal_to_noise,
        true,
    )?;
    let raw_tidal_variance = required_extract(input, "raw_tidal_variance")?;
    let all_constituent_tidal_variance = required_extract(input, "all_constituent_tidal_variance")?;
    validate_nonnegative_finite("raw_tidal_variance", raw_tidal_variance)?;
    validate_nonnegative_finite(
        "all_constituent_tidal_variance",
        all_constituent_tidal_variance,
    )?;
    let significant_constituent_tidal_variance =
        optional_float(input, "significant_constituent_tidal_variance")?;
    let all_constituent_percent_tidal_variance =
        optional_float(input, "all_constituent_percent_tidal_variance")?;
    let significant_constituent_percent_tidal_variance =
        optional_float(input, "significant_constituent_percent_tidal_variance")?;
    for (field, value) in [
        (
            "significant_constituent_tidal_variance",
            significant_constituent_tidal_variance,
        ),
        (
            "all_constituent_percent_tidal_variance",
            all_constituent_percent_tidal_variance,
        ),
        (
            "significant_constituent_percent_tidal_variance",
            significant_constituent_percent_tidal_variance,
        ),
    ] {
        validate_optional_nonnegative(field, value, false)?;
    }
    let rayleigh_minimum = required_extract(input, "rayleigh_minimum")?;
    let minimum_signal_to_noise = required_extract(input, "minimum_signal_to_noise")?;
    validate_positive_finite("rayleigh_minimum", rayleigh_minimum)?;
    validate_nonnegative_finite("minimum_signal_to_noise", minimum_signal_to_noise)?;

    Ok(ConstituentSelectionDiagnostics {
        constituents,
        whole_model: WholeModelIndependenceDiagnostics {
            basis_condition_number,
            all_constituent_signal_to_noise,
            condition_adjusted_signal_to_noise,
        },
        tidal_variance: TidalVarianceDiagnostics {
            raw_tidal_variance,
            all_constituent_tidal_variance,
            significant_constituent_tidal_variance,
            all_constituent_percent_tidal_variance,
            significant_constituent_percent_tidal_variance,
        },
        rayleigh_minimum,
        minimum_signal_to_noise,
    })
}

fn neighbor_from_snapshot(
    source: usize,
    index: i64,
    rayleigh: f64,
    noise_modified: f64,
    correlation: f64,
    names: &[String],
    frequencies: &[f64],
) -> PyResult<Option<NeighboringConstituentDiagnostics>> {
    if index == -1 {
        if rayleigh.is_nan() && noise_modified.is_nan() && correlation.is_nan() {
            return Ok(None);
        }
        return Err(snapshot_error(
            "absent diagnostic neighbor must have NaN metric sentinels",
        ));
    }
    let index = usize::try_from(index)
        .ok()
        .filter(|index| *index < names.len() && *index != source)
        .ok_or_else(|| snapshot_error("diagnostic neighbor index is out of range"))?;
    validate_positive_finite("neighbor_rayleigh", rayleigh)?;
    let noise_modified_rayleigh_criterion = if noise_modified.is_nan() {
        None
    } else {
        validate_nonnegative("neighbor_noise_modified_rayleigh", noise_modified, true)?;
        Some(noise_modified)
    };
    let maximum_correlation = if correlation.is_nan() {
        None
    } else if correlation.is_finite() && (0.0..=1.0 + 1e-10).contains(&correlation) {
        Some(correlation)
    } else {
        return Err(snapshot_error(
            "neighbor maximum correlation must be between zero and one",
        ));
    };
    Ok(Some(NeighboringConstituentDiagnostics {
        index,
        name: names[index].clone(),
        frequency_cph: frequencies[index],
        rayleigh_criterion: rayleigh,
        noise_modified_rayleigh_criterion,
        maximum_correlation,
    }))
}

fn validate_neighbor_symmetry(constituents: &[ConstituentIndependenceDiagnostics]) -> PyResult<()> {
    for (source, value) in constituents.iter().enumerate() {
        for (neighbor, reverse_higher) in [(&value.lower, true), (&value.higher, false)] {
            if let Some(neighbor) = neighbor {
                let reverse = if reverse_higher {
                    constituents[neighbor.index].higher.as_ref()
                } else {
                    constituents[neighbor.index].lower.as_ref()
                };
                if reverse.is_none_or(|reverse| reverse.index != source) {
                    return Err(snapshot_error(
                        "diagnostic neighbor relationships are not symmetric",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_positive_finite(field: &str, value: f64) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(snapshot_error(format!(
            "diagnostic field '{field}' must be finite and greater than zero"
        )))
    }
}

fn validate_positive_or_infinite(field: &str, value: f64) -> PyResult<()> {
    if !value.is_nan() && value > 0.0 {
        Ok(())
    } else {
        Err(snapshot_error(format!(
            "diagnostic field '{field}' must be positive"
        )))
    }
}

fn validate_nonnegative_finite(field: &str, value: f64) -> PyResult<()> {
    validate_nonnegative(field, value, false)
}

fn validate_nonnegative(field: &str, value: f64, allow_infinite: bool) -> PyResult<()> {
    if !value.is_nan() && value >= 0.0 && (allow_infinite || value.is_finite()) {
        Ok(())
    } else {
        Err(snapshot_error(format!(
            "diagnostic field '{field}' must be nonnegative{}",
            if allow_infinite { "" } else { " and finite" }
        )))
    }
}

fn validate_optional_nonnegative(
    field: &str,
    value: Option<f64>,
    allow_infinite: bool,
) -> PyResult<()> {
    value.map_or(Ok(()), |value| {
        validate_nonnegative(field, value, allow_infinite)
    })
}

pub(crate) fn scalar_solution_snapshot<'py>(
    solution: &ScalarSolution,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    for (name, values) in [
        ("cosine_coefficient", &solution.cosine_coefficient),
        ("sine_coefficient", &solution.sine_coefficient),
        ("amplitude", &solution.amplitude),
        ("phase_degrees", &solution.phase_degrees),
        ("percent_energy", &solution.percent_energy),
    ] {
        output.set_item(name, values.clone().into_pyarray(py))?;
    }
    set_optional_vec(&output, py, "amplitude_ci", solution.amplitude_ci.as_ref())?;
    set_optional_vec(
        &output,
        py,
        "phase_ci_degrees",
        solution.phase_ci_degrees.as_ref(),
    )?;
    set_optional_vec(
        &output,
        py,
        "signal_to_noise",
        solution.signal_to_noise.as_ref(),
    )?;
    set_optional_vec(
        &output,
        py,
        "cosine_coefficient_variance",
        solution.cosine_coefficient_variance.as_ref(),
    )?;
    set_optional_vec(
        &output,
        py,
        "sine_coefficient_variance",
        solution.sine_coefficient_variance.as_ref(),
    )?;
    output.set_item("mean", solution.mean)?;
    output.set_item("slope_per_day", solution.slope_per_day)?;
    output.set_item("reference_time_days", solution.reference_time_days)?;
    output.set_item("robust", robust_snapshot(solution.robust.as_ref(), py)?)?;
    Ok(output)
}

pub(crate) fn vector_solution_snapshot<'py>(
    solution: &VectorSolution,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    for (name, values) in [
        ("semi_major", &solution.semi_major),
        ("semi_minor", &solution.semi_minor),
        ("inclination_degrees", &solution.inclination_degrees),
        ("phase_degrees", &solution.phase_degrees),
        ("percent_energy", &solution.percent_energy),
    ] {
        output.set_item(name, values.clone().into_pyarray(py))?;
    }
    set_optional_vec(
        &output,
        py,
        "semi_major_ci",
        solution.semi_major_ci.as_ref(),
    )?;
    set_optional_vec(
        &output,
        py,
        "semi_minor_ci",
        solution.semi_minor_ci.as_ref(),
    )?;
    set_optional_vec(
        &output,
        py,
        "inclination_ci_degrees",
        solution.inclination_ci_degrees.as_ref(),
    )?;
    set_optional_vec(
        &output,
        py,
        "phase_ci_degrees",
        solution.phase_ci_degrees.as_ref(),
    )?;
    set_optional_vec(
        &output,
        py,
        "signal_to_noise",
        solution.signal_to_noise.as_ref(),
    )?;
    output.set_item("eastward_mean", solution.eastward_mean)?;
    output.set_item("northward_mean", solution.northward_mean)?;
    output.set_item("eastward_slope_per_day", solution.eastward_slope_per_day)?;
    output.set_item("northward_slope_per_day", solution.northward_slope_per_day)?;
    output.set_item("reference_time_days", solution.reference_time_days)?;
    output.set_item("robust", robust_snapshot(solution.robust.as_ref(), py)?)?;
    Ok(output)
}

pub(crate) fn scalar_solution_from_snapshot(
    input: &Bound<'_, PyDict>,
    constituent_count: usize,
) -> PyResult<ScalarSolution> {
    let solution = ScalarSolution {
        cosine_coefficient: required_vec_f64(input, "cosine_coefficient")?,
        sine_coefficient: required_vec_f64(input, "sine_coefficient")?,
        amplitude: required_vec_f64(input, "amplitude")?,
        phase_degrees: required_vec_f64(input, "phase_degrees")?,
        percent_energy: required_vec_f64(input, "percent_energy")?,
        amplitude_ci: optional_vec_f64(input, "amplitude_ci")?,
        phase_ci_degrees: optional_vec_f64(input, "phase_ci_degrees")?,
        signal_to_noise: optional_vec_f64(input, "signal_to_noise")?,
        cosine_coefficient_variance: optional_vec_f64(input, "cosine_coefficient_variance")?,
        sine_coefficient_variance: optional_vec_f64(input, "sine_coefficient_variance")?,
        mean: required_extract(input, "mean")?,
        slope_per_day: required_extract(input, "slope_per_day")?,
        reference_time_days: required_extract(input, "reference_time_days")?,
        robust: robust_from_snapshot(&required(input, "robust")?)?,
    };
    validate_scalar_solution(&solution, constituent_count)?;
    Ok(solution)
}

pub(crate) fn vector_solution_from_snapshot(
    input: &Bound<'_, PyDict>,
    constituent_count: usize,
) -> PyResult<VectorSolution> {
    let solution = VectorSolution {
        semi_major: required_vec_f64(input, "semi_major")?,
        semi_minor: required_vec_f64(input, "semi_minor")?,
        inclination_degrees: required_vec_f64(input, "inclination_degrees")?,
        phase_degrees: required_vec_f64(input, "phase_degrees")?,
        percent_energy: required_vec_f64(input, "percent_energy")?,
        semi_major_ci: optional_vec_f64(input, "semi_major_ci")?,
        semi_minor_ci: optional_vec_f64(input, "semi_minor_ci")?,
        inclination_ci_degrees: optional_vec_f64(input, "inclination_ci_degrees")?,
        phase_ci_degrees: optional_vec_f64(input, "phase_ci_degrees")?,
        signal_to_noise: optional_vec_f64(input, "signal_to_noise")?,
        eastward_mean: required_extract(input, "eastward_mean")?,
        northward_mean: required_extract(input, "northward_mean")?,
        eastward_slope_per_day: required_extract(input, "eastward_slope_per_day")?,
        northward_slope_per_day: required_extract(input, "northward_slope_per_day")?,
        reference_time_days: required_extract(input, "reference_time_days")?,
        robust: robust_from_snapshot(&required(input, "robust")?)?,
    };
    validate_vector_solution(&solution, constituent_count)?;
    Ok(solution)
}

pub(crate) fn required_dict<'py>(
    input: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Bound<'py, PyDict>> {
    required(input, key)?
        .cast::<PyDict>()
        .cloned()
        .map_err(|_| snapshot_error(format!("snapshot field '{key}' must be a dictionary")))
}

pub(crate) fn required_vec_f64(input: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let value = required(input, key)?;
    let array = value.extract::<PyReadonlyArray1<'_, f64>>().map_err(|_| {
        snapshot_error(format!(
            "snapshot field '{key}' must be a contiguous float64 vector"
        ))
    })?;
    array
        .as_slice()
        .map(<[f64]>::to_vec)
        .map_err(|_| snapshot_error(format!("snapshot field '{key}' must be contiguous")))
}

pub(crate) fn required_vec_usize(input: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<usize>> {
    let value = required(input, key)?;
    let array = value
        .extract::<PyReadonlyArray1<'_, usize>>()
        .map_err(|_| {
            snapshot_error(format!(
                "snapshot field '{key}' must be a contiguous platform-integer vector"
            ))
        })?;
    array
        .as_slice()
        .map(<[usize]>::to_vec)
        .map_err(|_| snapshot_error(format!("snapshot field '{key}' must be contiguous")))
}

pub(crate) fn required_vec_i64(input: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<i64>> {
    let value = required(input, key)?;
    let array = value.extract::<PyReadonlyArray1<'_, i64>>().map_err(|_| {
        snapshot_error(format!(
            "snapshot field '{key}' must be a one-dimensional int64 array"
        ))
    })?;
    array
        .as_slice()
        .map(<[i64]>::to_vec)
        .map_err(|_| snapshot_error(format!("snapshot field '{key}' must be contiguous")))
}

pub(crate) fn required_extract<'py, T>(input: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    for<'a> T: FromPyObject<'a, 'py>,
{
    required(input, key)?.extract::<T>().map_err(|_| {
        snapshot_error(format!(
            "snapshot field '{key}' has an invalid type or value"
        ))
    })
}

fn optional_extract<'py, T>(input: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>>
where
    for<'a> T: FromPyObject<'a, 'py>,
{
    input
        .get_item(key)?
        .map(|value| {
            value.extract::<T>().map_err(|_| {
                snapshot_error(format!(
                    "snapshot field '{key}' has an invalid type or value"
                ))
            })
        })
        .transpose()
}

pub(crate) fn validate_presentation_order(
    order: &[usize],
    constituent_count: usize,
) -> PyResult<()> {
    if order.len() != constituent_count || order.iter().any(|index| *index >= constituent_count) {
        return Err(snapshot_error(
            "snapshot presentation order is incomplete or out of range",
        ));
    }
    if order.iter().copied().collect::<HashSet<_>>().len() != order.len() {
        return Err(snapshot_error(
            "snapshot presentation order contains duplicate indices",
        ));
    }
    Ok(())
}

pub(crate) fn snapshot_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(format!(
        "invalid RUTide coefficient snapshot: {}",
        message.into()
    ))
}

pub(crate) fn required<'py>(input: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    input
        .get_item(key)?
        .ok_or_else(|| snapshot_error(format!("missing field '{key}'")))
}

fn optional_float(input: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    let value = required(input, key)?;
    if value.is_none() {
        Ok(None)
    } else {
        value
            .extract::<f64>()
            .map(Some)
            .map_err(|_| snapshot_error(format!("snapshot field '{key}' must be a float or None")))
    }
}

fn optional_vec_f64(input: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<f64>>> {
    if required(input, key)?.is_none() {
        Ok(None)
    } else {
        required_vec_f64(input, key).map(Some)
    }
}

fn set_optional_vec(
    output: &Bound<'_, PyDict>,
    py: Python<'_>,
    key: &str,
    values: Option<&Vec<f64>>,
) -> PyResult<()> {
    if let Some(values) = values {
        output.set_item(key, values.clone().into_pyarray(py))
    } else {
        output.set_item(key, py.None())
    }
}

pub(crate) fn robust_snapshot(
    robust: Option<&RobustDiagnostics>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let Some(robust) = robust else {
        return Ok(py.None());
    };
    let output = PyDict::new(py);
    output.set_item("weights", robust.weights.clone().into_pyarray(py))?;
    output.set_item("leverage", robust.leverage.clone().into_pyarray(py))?;
    output.set_item("iterations", robust.iterations)?;
    output.set_item("termination", termination_name(robust.termination))?;
    output.set_item("residual_scale", robust.residual_scale)?;
    output.set_item("ols_rms_residual", robust.ols_rms_residual)?;
    output.set_item("rms_residual", robust.rms_residual)?;
    Ok(output.into_any().unbind())
}

pub(crate) fn robust_from_snapshot(
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<RobustDiagnostics>> {
    if value.is_none() {
        return Ok(None);
    }
    let input = value
        .cast::<PyDict>()
        .map_err(|_| snapshot_error("robust field must be a dictionary or None"))?;
    let termination = match required_extract::<String>(input, "termination")?.as_str() {
        "tolerance" => RobustTermination::Tolerance,
        "objective_increase" => RobustTermination::ObjectiveIncrease,
        "exact_fit" => RobustTermination::ExactFit,
        _ => return Err(snapshot_error("unknown robust termination value")),
    };
    let robust = RobustDiagnostics {
        weights: required_vec_f64(input, "weights")?,
        leverage: required_vec_f64(input, "leverage")?,
        iterations: required_extract(input, "iterations")?,
        termination,
        residual_scale: required_extract(input, "residual_scale")?,
        ols_rms_residual: required_extract(input, "ols_rms_residual")?,
        rms_residual: required_extract(input, "rms_residual")?,
    };
    if robust.weights.len() != robust.leverage.len() || robust.iterations == 0 {
        return Err(snapshot_error(
            "invalid robust diagnostic lengths or iteration count",
        ));
    }
    Ok(Some(robust))
}

pub(crate) fn validate_scalar_solution(solution: &ScalarSolution, count: usize) -> PyResult<()> {
    let lengths = [
        solution.cosine_coefficient.len(),
        solution.sine_coefficient.len(),
        solution.amplitude.len(),
        solution.phase_degrees.len(),
        solution.percent_energy.len(),
    ];
    if lengths.iter().any(|length| *length != count)
        || optional_lengths_scalar(solution)
            .into_iter()
            .flatten()
            .any(|length| length != count)
    {
        return Err(snapshot_error(
            "scalar solution shape does not match constituents",
        ));
    }
    if !solution.mean.is_finite()
        || !solution.slope_per_day.is_finite()
        || !solution.reference_time_days.is_finite()
    {
        return Err(snapshot_error(
            "scalar solution contains non-finite metadata",
        ));
    }
    Ok(())
}

fn optional_lengths_scalar(solution: &ScalarSolution) -> [Option<usize>; 5] {
    [
        solution.amplitude_ci.as_ref().map(Vec::len),
        solution.phase_ci_degrees.as_ref().map(Vec::len),
        solution.signal_to_noise.as_ref().map(Vec::len),
        solution.cosine_coefficient_variance.as_ref().map(Vec::len),
        solution.sine_coefficient_variance.as_ref().map(Vec::len),
    ]
}

pub(crate) fn validate_vector_solution(solution: &VectorSolution, count: usize) -> PyResult<()> {
    let lengths = [
        solution.semi_major.len(),
        solution.semi_minor.len(),
        solution.inclination_degrees.len(),
        solution.phase_degrees.len(),
        solution.percent_energy.len(),
    ];
    let optional = [
        solution.semi_major_ci.as_ref().map(Vec::len),
        solution.semi_minor_ci.as_ref().map(Vec::len),
        solution.inclination_ci_degrees.as_ref().map(Vec::len),
        solution.phase_ci_degrees.as_ref().map(Vec::len),
        solution.signal_to_noise.as_ref().map(Vec::len),
    ];
    if lengths.iter().any(|length| *length != count)
        || optional.into_iter().flatten().any(|length| length != count)
    {
        return Err(snapshot_error(
            "vector solution shape does not match constituents",
        ));
    }
    if [
        solution.eastward_mean,
        solution.northward_mean,
        solution.eastward_slope_per_day,
        solution.northward_slope_per_day,
        solution.reference_time_days,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(snapshot_error(
            "vector solution contains non-finite metadata",
        ));
    }
    Ok(())
}

fn validate_method_diagnostics(
    method: &str,
    solution: &FittedSolution,
    retained_observations: usize,
) -> PyResult<()> {
    let robust = match solution {
        FittedSolution::Scalar(solution) => solution.robust.as_ref(),
        FittedSolution::Vector(solution) => solution.robust.as_ref(),
    };
    if method.eq_ignore_ascii_case("robust") != robust.is_some() {
        return Err(snapshot_error(
            "method and robust diagnostic presence are inconsistent",
        ));
    }
    if let Some(robust) = robust
        && robust.weights.len() != retained_observations
    {
        return Err(snapshot_error(
            "robust diagnostic length does not match retained observations",
        ));
    }
    Ok(())
}

const fn termination_name(termination: RobustTermination) -> &'static str {
    match termination {
        RobustTermination::Tolerance => "tolerance",
        RobustTermination::ObjectiveIncrease => "objective_increase",
        RobustTermination::ExactFit => "exact_fit",
    }
}
