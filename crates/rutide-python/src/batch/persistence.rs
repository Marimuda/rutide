//! Versioned, pickle-free snapshots for multi-series native fits.

use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use rayon::ThreadPoolBuilder;
use rutide_core::{
    GreenwichNodalBatch, InferenceMode, ScalarInferenceBatch, SolverOptions, VectorInferenceBatch,
};

use super::{BatchFit, BatchFitState, BatchSolutions, PreparedBatchModel, requested_worker_count};
use crate::{
    SolveConfig, normalized_confidence_name, parse_confidence, parse_constituents,
    parse_method_and_robust, parse_nodal_corrections, parse_phase_reference,
    persistence::{
        batch_diagnostics_from_snapshot, batch_diagnostics_snapshot, config_from_snapshot,
        config_snapshot, required, required_dict, required_extract, required_vec_f64,
        required_vec_usize, scalar_solution_from_snapshot, scalar_solution_snapshot,
        snapshot_error, snapshot_header, validate_header, validate_presentation_order,
        vector_solution_from_snapshot, vector_solution_snapshot,
    },
    scalar_inference_relations, solver_options, validate_empty_inference,
    vector_inference_relations,
};

pub(super) fn batch_snapshot<'py>(
    state: &BatchFitState,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let snapshot = snapshot_header(py, "batch")?;
    snapshot.set_item("time_mjd", state.time_mjd.clone().into_pyarray(py))?;
    snapshot.set_item("config", config_snapshot(&state.config, py)?)?;
    snapshot.set_item("latitudes", state.latitudes.clone().into_pyarray(py))?;
    snapshot.set_item("original_time_count", state.original_time_count)?;
    snapshot.set_item(
        "source_time_positions",
        state.source_time_positions.clone().into_pyarray(py),
    )?;
    snapshot.set_item("worker_count", state.worker_count)?;
    snapshot.set_item("chunk_series", state.chunk_series)?;
    snapshot.set_item(
        "diagnostics",
        batch_diagnostics_snapshot(state.diagnostics.as_deref(), py)?,
    )?;

    let positions = PyList::empty(py);
    for values in &state.valid_positions {
        positions.append(values.clone().into_pyarray(py))?;
    }
    snapshot.set_item("valid_positions", positions)?;

    let solutions = PyList::empty(py);
    match &state.solutions {
        BatchSolutions::Scalar(values) => {
            snapshot.set_item("solution_kind", "scalar")?;
            for solution in values {
                solutions.append(scalar_solution_snapshot(solution, py)?)?;
            }
        }
        BatchSolutions::Vector(values) => {
            snapshot.set_item("solution_kind", "vector")?;
            for solution in values {
                solutions.append(vector_solution_snapshot(solution, py)?)?;
            }
        }
    }
    snapshot.set_item("solutions", solutions)?;

    let order = PyList::empty(py);
    for values in &state.presentation_order {
        order.append(values.clone().into_pyarray(py))?;
    }
    snapshot.set_item("presentation_order", order)?;
    Ok(snapshot)
}

pub(super) fn restore_batch(
    py: Python<'_>,
    snapshot: &Bound<'_, PyDict>,
    requested_workers: Option<usize>,
) -> PyResult<Py<BatchFit>> {
    let schema_version = validate_header(snapshot, "batch")?;
    let axes = restore_axes(snapshot)?;
    let RestoredAxes {
        time_mjd,
        config,
        latitudes,
        source_time_positions,
        valid_positions,
        original_time_count,
    } = axes;
    let retained_time_count = time_mjd.len();
    let series_count = latitudes.len();
    let solution_kind = required_extract::<String>(snapshot, "solution_kind")?;
    let is_vector = is_vector_kind(&solution_kind)?;
    let constituents = parse_constituents(
        config
            .constituent_names
            .as_deref()
            .ok_or_else(|| snapshot_error("persisted constituent_names cannot be None"))?,
    )
    .map_err(snapshot_error)?;
    let phase_reference = parse_phase_reference(&config.phase_name).map_err(snapshot_error)?;
    let nodal_corrections = parse_nodal_corrections(&config.nodal_name).map_err(snapshot_error)?;
    let solver_options =
        solver_options(&config, phase_reference, nodal_corrections).map_err(snapshot_error)?;
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
    let model = prepare_model(
        &time_mjd,
        &constituents,
        &config,
        solver_options,
        inference_mode,
        is_vector,
    )?;
    let names = model
        .tidal_constituents()
        .iter()
        .map(|constituent| constituent.name().to_owned())
        .collect::<Vec<_>>();
    let constituent_count = names.len();
    let solutions =
        solutions_from_snapshot(snapshot, &solution_kind, constituent_count, series_count)?;
    validate_robust_consistency(&config, &solutions, &valid_positions)?;
    let frequencies = restored_frequencies(&model, &solutions)?;
    let diagnostics =
        batch_diagnostics_from_snapshot(snapshot, schema_version, &names, &frequencies)?;
    if diagnostics.is_some() != config.diagnostics {
        return Err(snapshot_error(
            "diagnostic config and persisted diagnostic presence disagree",
        ));
    }
    let presentation_order = restored_orders(snapshot, series_count, constituent_count)?;
    let (worker_count, chunk_series, pool) =
        restored_execution(snapshot, requested_workers, series_count)?;
    Py::new(
        py,
        BatchFit {
            inner: Arc::new(BatchFitState {
                model,
                solutions,
                time_mjd,
                config: config.clone(),
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
                chunk_series: chunk_series.min(series_count),
                pool,
            }),
        },
    )
}

struct RestoredAxes {
    time_mjd: Vec<f64>,
    config: SolveConfig,
    latitudes: Vec<f64>,
    source_time_positions: Vec<usize>,
    valid_positions: Vec<Vec<usize>>,
    original_time_count: usize,
}

fn restore_axes(snapshot: &Bound<'_, PyDict>) -> PyResult<RestoredAxes> {
    let time_mjd = required_vec_f64(snapshot, "time_mjd")?;
    let config = config_from_snapshot(&required_dict(snapshot, "config")?)?;
    let latitudes = required_vec_f64(snapshot, "latitudes")?;
    if latitudes.is_empty() {
        return Err(snapshot_error("batch contains no series"));
    }
    let original_time_count = required_extract::<usize>(snapshot, "original_time_count")?;
    let source_time_positions = required_vec_usize(snapshot, "source_time_positions")?;
    validate_source_positions(&source_time_positions, time_mjd.len(), original_time_count)?;
    let valid_positions = vec_usize_list(snapshot, "valid_positions")?;
    if valid_positions.len() != latitudes.len() {
        return Err(snapshot_error(
            "valid_positions count does not match the batch series count",
        ));
    }
    for positions in &valid_positions {
        validate_positions(positions, time_mjd.len())?;
    }
    Ok(RestoredAxes {
        time_mjd,
        config,
        latitudes,
        source_time_positions,
        valid_positions,
        original_time_count,
    })
}

fn is_vector_kind(kind: &str) -> PyResult<bool> {
    match kind {
        "scalar" => Ok(false),
        "vector" => Ok(true),
        _ => Err(snapshot_error(
            "solution_kind must be either 'scalar' or 'vector'",
        )),
    }
}

fn restored_frequencies(
    model: &PreparedBatchModel,
    solutions: &BatchSolutions,
) -> PyResult<Vec<Vec<f64>>> {
    solutions
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
        .collect::<Result<Vec<Vec<_>>, String>>()
        .map_err(snapshot_error)
}

fn restored_orders(
    snapshot: &Bound<'_, PyDict>,
    series_count: usize,
    constituent_count: usize,
) -> PyResult<Vec<Vec<usize>>> {
    let orders = vec_usize_list(snapshot, "presentation_order")?;
    if orders.len() != series_count {
        return Err(snapshot_error(
            "presentation_order count does not match the batch series count",
        ));
    }
    for order in &orders {
        validate_presentation_order(order, constituent_count)?;
    }
    Ok(orders)
}

fn restored_execution(
    snapshot: &Bound<'_, PyDict>,
    requested_workers: Option<usize>,
    series_count: usize,
) -> PyResult<(usize, usize, Arc<rayon::ThreadPool>)> {
    let stored_workers = required_extract::<usize>(snapshot, "worker_count")?;
    let worker_count =
        requested_worker_count(requested_workers.or(Some(stored_workers)), series_count)
            .map_err(snapshot_error)?;
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|index| format!("rutide-python-{index}"))
            .build()
            .map_err(|error| snapshot_error(format!("worker-pool error: {error}")))?,
    );
    let chunk_series = required_extract::<usize>(snapshot, "chunk_series")?;
    if chunk_series == 0 {
        return Err(snapshot_error("chunk_series must be greater than zero"));
    }
    Ok((worker_count, chunk_series.min(series_count), pool))
}

fn prepare_model(
    time_mjd: &[f64],
    constituents: &[rutide_core::TidalConstituent],
    config: &SolveConfig,
    solver_options: SolverOptions,
    inference_mode: InferenceMode,
    is_vector: bool,
) -> PyResult<PreparedBatchModel> {
    if config.inferred_names.is_empty() {
        validate_empty_inference(config).map_err(snapshot_error)?;
        return GreenwichNodalBatch::prepare_modified_julian_days_with_solver_options(
            time_mjd,
            constituents,
            solver_options,
        )
        .map(|model| PreparedBatchModel::Direct(Box::new(model)))
        .map_err(|error| snapshot_error(error.to_string()));
    }
    if is_vector {
        let relationships = vector_inference_relations(config).map_err(snapshot_error)?;
        VectorInferenceBatch::prepare_modified_julian_days_with_solver_options(
            time_mjd,
            constituents,
            &relationships,
            inference_mode,
            solver_options,
        )
        .map(|model| PreparedBatchModel::VectorInference(Box::new(model)))
        .map_err(|error| snapshot_error(error.to_string()))
    } else {
        let relationships = scalar_inference_relations(config).map_err(snapshot_error)?;
        ScalarInferenceBatch::prepare_modified_julian_days_with_solver_options(
            time_mjd,
            constituents,
            &relationships,
            inference_mode,
            solver_options,
        )
        .map(|model| PreparedBatchModel::ScalarInference(Box::new(model)))
        .map_err(|error| snapshot_error(error.to_string()))
    }
}

fn solutions_from_snapshot(
    snapshot: &Bound<'_, PyDict>,
    kind: &str,
    constituent_count: usize,
    series_count: usize,
) -> PyResult<BatchSolutions> {
    let value = required(snapshot, "solutions")?;
    let inputs = value
        .cast::<PyList>()
        .map_err(|_| snapshot_error("solutions must be a list"))?;
    if inputs.len() != series_count {
        return Err(snapshot_error(
            "solution count does not match the batch series count",
        ));
    }
    if kind == "scalar" {
        inputs
            .iter()
            .map(|value| {
                value
                    .cast::<PyDict>()
                    .map_err(|_| snapshot_error("each solution must be a dictionary"))
                    .and_then(|input| scalar_solution_from_snapshot(input, constituent_count))
            })
            .collect::<PyResult<Vec<_>>>()
            .map(BatchSolutions::Scalar)
    } else {
        inputs
            .iter()
            .map(|value| {
                value
                    .cast::<PyDict>()
                    .map_err(|_| snapshot_error("each solution must be a dictionary"))
                    .and_then(|input| vector_solution_from_snapshot(input, constituent_count))
            })
            .collect::<PyResult<Vec<_>>>()
            .map(BatchSolutions::Vector)
    }
}

fn vec_usize_list(input: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Vec<usize>>> {
    let value = required(input, key)?;
    value
        .cast::<PyList>()
        .map_err(|_| snapshot_error(format!("{key} must be a list")))?
        .iter()
        .map(|value| {
            let array = value
                .extract::<PyReadonlyArray1<'_, usize>>()
                .map_err(|_| {
                    snapshot_error(format!(
                        "each {key} entry must be a platform-integer vector"
                    ))
                })?;
            array
                .as_slice()
                .map(<[usize]>::to_vec)
                .map_err(|_| snapshot_error(format!("each {key} entry must be contiguous")))
        })
        .collect()
}

fn validate_source_positions(
    positions: &[usize],
    retained_count: usize,
    original_count: usize,
) -> PyResult<()> {
    if positions.len() != retained_count
        || positions.iter().any(|position| *position >= original_count)
        || positions.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(snapshot_error(
            "source_time_positions are inconsistent with timestamp counts",
        ));
    }
    Ok(())
}

fn validate_positions(positions: &[usize], time_count: usize) -> PyResult<()> {
    if positions.is_empty()
        || positions.iter().any(|position| *position >= time_count)
        || positions.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(snapshot_error(
            "valid observation positions are empty, out of range, or unordered",
        ));
    }
    Ok(())
}

fn validate_robust_consistency(
    config: &SolveConfig,
    solutions: &BatchSolutions,
    valid_positions: &[Vec<usize>],
) -> PyResult<()> {
    let expected = config.method_name.eq_ignore_ascii_case("robust");
    let lengths = match solutions {
        BatchSolutions::Scalar(values) => values
            .iter()
            .map(|solution| {
                solution
                    .robust
                    .as_ref()
                    .map(|robust| (robust.weights.len(), robust.leverage.len()))
            })
            .collect::<Vec<_>>(),
        BatchSolutions::Vector(values) => values
            .iter()
            .map(|solution| {
                solution
                    .robust
                    .as_ref()
                    .map(|robust| (robust.weights.len(), robust.leverage.len()))
            })
            .collect(),
    };
    for (series, lengths) in lengths.into_iter().enumerate() {
        if lengths.is_some() != expected {
            return Err(snapshot_error(
                "method and robust diagnostic presence are inconsistent",
            ));
        }
        if let Some((weights, leverage)) = lengths
            && (weights != valid_positions[series].len() || leverage != weights)
        {
            return Err(snapshot_error(
                "robust diagnostic length does not match valid observation positions",
            ));
        }
    }
    Ok(())
}
