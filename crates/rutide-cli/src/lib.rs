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

use netcdf::{Variable, VariableMut};
use rayon::ThreadPoolBuilder;
use rutide_core::{
    AnalysisError, Constituent, GreenwichNodalBatch, ScalarSolution, TidalConstituent,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MILLISECONDS_PER_DAY: f64 = 86_400_000.0;
const OUTPUT_SCHEMA_VERSION: u32 = 1;
const CONSTITUENTS: [TidalConstituent; 5] = [
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

/// Configuration for one scalar FVCOM analysis run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalyzeConfig {
    /// Read-only source FVCOM `NetCDF` path.
    pub input: PathBuf,
    /// Destination `NetCDF` coefficient path.
    pub output: PathBuf,
    /// Optional JSON run-report path.
    pub report: Option<PathBuf>,
    /// Spatial subset to analyze.
    pub nodes: NodeSelection,
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
    /// Fitted constant offset.
    pub mean: f64,
    /// Fitted trend per day.
    pub slope_per_day: f64,
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
    /// Number of outer spatial workers.
    pub workers: usize,
    /// Constituent names in coefficient order.
    pub constituents: Vec<String>,
    /// Reference-time frequencies in cycles per hour.
    pub frequency_cph: Vec<f64>,
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
    input_file_bytes: u64,
    logical_input_bytes: u64,
}

/// Analyze an FVCOM `zeta(time, node)` field and write every coefficient.
///
/// # Errors
///
/// Returns [`AppError`] when configuration, source schema, observations,
/// numerical analysis, or output serialization fails.
pub fn analyze_scalar(config: &AnalyzeConfig) -> Result<RunReport, AppError> {
    validate_config(config)?;
    faer::set_global_parallelism(faer::Par::Seq);
    let total_start = Instant::now();

    let input_start = Instant::now();
    let input = read_fvcom_scalar(&config.input, &config.nodes)?;
    let input_seconds = input_start.elapsed().as_secs_f64();

    let preparation_start = Instant::now();
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(
        &input.modified_julian_days,
        &CONSTITUENTS,
    )?;
    let preparation_seconds = preparation_start.elapsed().as_secs_f64();

    let worker_pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()?;
    let solve_start = Instant::now();
    let solutions =
        worker_pool.install(|| batch.solve_time_major(&input.observations, &input.latitudes))?;
    let solve_seconds = solve_start.elapsed().as_secs_f64();

    let result_start = Instant::now();
    let result_sha256 = result_digest(
        &input.node_indices,
        &input.latitudes,
        batch.constituents(),
        &solutions,
    )?;
    let sample_results = retained_samples(&input.node_indices, &input.latitudes, &solutions);
    let result_processing_seconds = result_start.elapsed().as_secs_f64();

    let output_start = Instant::now();
    write_output(
        &config.output,
        config.overwrite,
        &input.node_indices,
        &input.latitudes,
        batch.constituents(),
        &solutions,
        &result_sha256,
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
        profile: "fixed-constituents-greenwich-nodal-ols",
        input_path: config.input.to_string_lossy().into_owned(),
        input_file_bytes: input.input_file_bytes,
        logical_input_bytes: input.logical_input_bytes,
        output_path: config.output.to_string_lossy().into_owned(),
        output_file_bytes,
        time_count: input.modified_julian_days.len(),
        series_count: input.node_indices.len(),
        workers: config.workers,
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
        result_sha256,
        timings: StageTimings {
            input_seconds,
            preparation_seconds,
            solve_seconds,
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
    let (latitude_values, observation_values) = if is_prefix {
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
    for (index, value) in observation_values.iter().copied().enumerate() {
        validate_source_value(
            "zeta",
            value,
            zeta_fill,
            index % series_count,
            index / series_count,
        )?;
    }

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

fn result_digest(
    node_indices: &[usize],
    latitudes: &[f64],
    constituents: &[Constituent],
    solutions: &[ScalarSolution],
) -> Result<String, AppError> {
    let mut digest = Sha256::new();
    digest.update(b"rutide-scalar-greenwich-nodal-v1\0");
    for constituent in constituents {
        digest.update(constituent.name.as_bytes());
        digest.update([0]);
        digest.update(constituent.frequency_cph.to_bits().to_le_bytes());
    }
    for ((node_index, latitude), solution) in node_indices
        .iter()
        .copied()
        .zip(latitudes.iter().copied())
        .zip(solutions)
    {
        let node_index = u64::try_from(node_index)
            .map_err(|_| AppError::Invalid("node index exceeds u64".to_owned()))?;
        digest.update(node_index.to_le_bytes());
        digest.update(latitude.to_bits().to_le_bytes());
        for value in &solution.amplitude {
            digest.update(value.to_bits().to_le_bytes());
        }
        for value in &solution.phase_degrees {
            digest.update(value.to_bits().to_le_bytes());
        }
        digest.update(solution.mean.to_bits().to_le_bytes());
        digest.update(solution.slope_per_day.to_bits().to_le_bytes());
    }
    Ok(encode_hex(&digest.finalize()))
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
    solutions: &[ScalarSolution],
) -> Vec<SampleResult> {
    node_indices
        .iter()
        .copied()
        .zip(latitudes.iter().copied())
        .zip(solutions)
        .take(3)
        .map(
            |((node_index, latitude_degrees_north), solution)| SampleResult {
                node_index,
                latitude_degrees_north,
                amplitude: solution.amplitude.clone(),
                phase_degrees: solution.phase_degrees.clone(),
                mean: solution.mean,
                slope_per_day: solution.slope_per_day,
            },
        )
        .collect()
}

fn write_output(
    path: &Path,
    overwrite: bool,
    node_indices: &[usize],
    latitudes: &[f64],
    constituents: &[Constituent],
    solutions: &[ScalarSolution],
    result_sha256: &str,
) -> Result<(), AppError> {
    let temporary = temporary_sibling(path)?;
    let write_result = write_output_file(
        &temporary,
        node_indices,
        latitudes,
        constituents,
        solutions,
        result_sha256,
    );
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

fn write_output_file(
    path: &Path,
    node_indices: &[usize],
    latitudes: &[f64],
    constituents: &[Constituent],
    solutions: &[ScalarSolution],
    result_sha256: &str,
) -> Result<(), AppError> {
    let mut output = netcdf::create(path)?;
    output.add_dimension("series", node_indices.len())?;
    output.add_dimension("constituent", constituents.len())?;
    output.add_attribute("title", "RUTide scalar harmonic coefficients")?;
    output.add_attribute("rutide_version", rutide_core::VERSION)?;
    output.add_attribute("profile", "fixed-constituents-greenwich-nodal-ols")?;
    output.add_attribute("constituent_names", "M2,S2,N2,K1,O1")?;
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
    let frequency = constituents
        .iter()
        .map(|constituent| constituent.frequency_cph)
        .collect::<Vec<_>>();
    write_variable(
        &mut output.add_variable::<f64>("frequency", &["constituent"])?,
        &frequency,
        "cycles per hour",
    )?;

    let mut amplitude = Vec::with_capacity(solutions.len() * constituents.len());
    let mut phase = Vec::with_capacity(solutions.len() * constituents.len());
    let mut mean = Vec::with_capacity(solutions.len());
    let mut slope = Vec::with_capacity(solutions.len());
    for solution in solutions {
        amplitude.extend_from_slice(&solution.amplitude);
        phase.extend_from_slice(&solution.phase_degrees);
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
        &mut output.add_variable::<f64>("mean", &["series"])?,
        &mean,
        "source variable units",
    )?;
    write_variable(
        &mut output.add_variable::<f64>("slope", &["series"])?,
        &slope,
        "source variable units per day",
    )?;
    output.close()?;
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

fn write_json_report(path: &Path, overwrite: bool, report: &RunReport) -> Result<(), AppError> {
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
        NodeSelection, encode_hex, read_fvcom_scalar, resolve_node_selection, temporary_sibling,
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
        fs::remove_file(path).expect("remove test NetCDF");
    }
}
