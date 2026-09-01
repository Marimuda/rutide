//! `RUTide` command-line entry point.

use std::{
    env,
    ffi::{OsStr, OsString},
    path::PathBuf,
    process::ExitCode,
};

use rutide_cli::{
    AnalysisMethod, AnalyzeConfig, ConfidenceInterval, ConstituentSelection, DEFAULT_CONSTITUENTS,
    NodeSelection, ScalarInferenceConfig, VectorAnalyzeConfig, VectorInferenceConfig,
    analyze_scalar, analyze_vector,
};
use rutide_core::{
    InferenceMode, LinearConfidence, ReconstructionFilter, RobustOptions, ScalarInferenceRelation,
    TidalConstituent, VectorInferenceRelation,
};

// The application repeatedly allocates short-lived QR storage across worker
// threads; this allocator keeps that full-field pattern bounded and reusable.
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const USAGE: &str = "\
Usage:
  rutide --version
  rutide analyze-scalar --input PATH --output PATH [OPTIONS]
  rutide analyze-vector --input PATH --output PATH [OPTIONS]

Options:
  --report PATH       Write the machine-readable JSON run report
  --workers N         Outer spatial workers (default: available CPUs)
  --node-count N      Analyze the first N nodes
  --nodes I,J,...     Analyze explicit zero-based node indices
  --element-count N   Analyze the first N elements (vector mode)
  --elements I,J,...  Analyze explicit zero-based element indices (vector mode)
  --constituents LIST Comma-separated names or 'auto' (default: M2,S2,N2,K1,O1)
  --rayleigh-min X    Automatic selection criterion (default with auto: 1.0)
  --infer SPEC        Repeatable inferred relationship:
                      scalar I:R:AMP:PHASE; vector I:R:AMP+:PHASE+:AMP-:PHASE-
  --infer-approximate Use Python-compatible reference-only approximate inference
  --method MODE       Least squares: ols or robust (default: ols)
  --robust-tuning X   Cauchy tuning constant (default: 2.385)
  --robust-tolerance X
                      Fractional IRLS tolerance (default: 0.001)
  --robust-max-iterations N
                      Maximum IRLS iterations (default: 50)
  --confidence MODE   Confidence intervals: none or linear (default: none)
  --white-noise       Use white noise instead of colored residual bands
  --reconstruct       Write complete original-time reconstruction to the output
  --reconstruct-constituents LIST
                      Reconstruct only these fitted constituent names
  --min-pe X          Reconstruct constituents with PE >= X percent
  --min-snr X         Reconstruct constituents with SNR >= X (requires confidence)
  --overwrite         Replace existing output and report files
  -h, --help          Show this help
";

enum Command {
    Help,
    Version,
    AnalyzeScalar(AnalyzeConfig),
    AnalyzeVector(VectorAnalyzeConfig),
}

fn main() -> ExitCode {
    let command = match parse_arguments(env::args_os().skip(1)) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    match command {
        Command::Help => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Command::Version => {
            println!("rutide {}", rutide_core::VERSION);
            ExitCode::SUCCESS
        }
        Command::AnalyzeScalar(config) => match analyze_scalar(&config) {
            Ok(report) => print_report(&report),
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Command::AnalyzeVector(config) => match analyze_vector(&config) {
            Ok(report) => print_report(&report),
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
    }
}

fn print_report(report: &impl serde::Serialize) -> ExitCode {
    match serde_json::to_string_pretty(report) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: could not serialize completed run report: {error}");
            ExitCode::FAILURE
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "single-pass parsing keeps option duplication and cross-option validation explicit"
)]
fn parse_arguments(arguments: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut arguments = arguments.into_iter();
    let Some(command) = arguments.next() else {
        return Ok(Command::Help);
    };
    if command == "--version" || command == "-V" {
        if arguments.next().is_some() {
            return Err("--version does not accept additional arguments".to_owned());
        }
        return Ok(Command::Version);
    }
    if command == "--help" || command == "-h" {
        return Ok(Command::Help);
    }
    let vector = match command.to_str() {
        Some("analyze-scalar") => false,
        Some("analyze-vector") => true,
        _ => return Err(format!("unknown command: {}", command.to_string_lossy())),
    };

    let mut input = None;
    let mut output = None;
    let mut report = None;
    let mut workers = std::thread::available_parallelism().map_or(1, usize::from);
    let mut selection = None;
    let mut constituents = None;
    let mut rayleigh_minimum = None;
    let mut scalar_inference_relationships = Vec::new();
    let mut vector_inference_relationships = Vec::new();
    let mut inference_approximate = false;
    let mut robust_requested = None;
    let mut robust_tuning = None;
    let mut robust_tolerance = None;
    let mut robust_max_iterations = None;
    let mut confidence_requested = None;
    let mut white_noise = false;
    let mut reconstruct = false;
    let mut reconstruction_constituents = None;
    let mut minimum_percent_energy = None;
    let mut minimum_signal_to_noise = None;
    let mut overwrite = false;
    while let Some(argument) = arguments.next() {
        let option = argument
            .to_str()
            .ok_or_else(|| "option names must be valid UTF-8".to_owned())?;
        match option {
            "--input" => input = Some(PathBuf::from(required_value(&mut arguments, option)?)),
            "--output" => output = Some(PathBuf::from(required_value(&mut arguments, option)?)),
            "--report" => report = Some(PathBuf::from(required_value(&mut arguments, option)?)),
            "--workers" => {
                workers = parse_positive_usize(&required_value(&mut arguments, option)?, option)?;
            }
            "--node-count" => {
                if vector {
                    return Err("--node-count is only valid for analyze-scalar".to_owned());
                }
                ensure_selection_is_unset(selection.as_ref())?;
                let count = parse_positive_usize(&required_value(&mut arguments, option)?, option)?;
                selection = Some(NodeSelection::Prefix(count));
            }
            "--nodes" => {
                if vector {
                    return Err("--nodes is only valid for analyze-scalar".to_owned());
                }
                ensure_selection_is_unset(selection.as_ref())?;
                selection = Some(parse_nodes(&required_value(&mut arguments, option)?)?);
            }
            "--element-count" => {
                if !vector {
                    return Err("--element-count is only valid for analyze-vector".to_owned());
                }
                ensure_selection_is_unset(selection.as_ref())?;
                let count = parse_positive_usize(&required_value(&mut arguments, option)?, option)?;
                selection = Some(NodeSelection::Prefix(count));
            }
            "--elements" => {
                if !vector {
                    return Err("--elements is only valid for analyze-vector".to_owned());
                }
                ensure_selection_is_unset(selection.as_ref())?;
                selection = Some(parse_nodes(&required_value(&mut arguments, option)?)?);
            }
            "--constituents" => {
                if constituents.is_some() {
                    return Err("--constituents may only be supplied once".to_owned());
                }
                constituents = Some(parse_constituents(&required_value(
                    &mut arguments,
                    option,
                )?)?);
            }
            "--rayleigh-min" => {
                if rayleigh_minimum.is_some() {
                    return Err("--rayleigh-min may only be supplied once".to_owned());
                }
                rayleigh_minimum = Some(parse_positive_f64(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
            }
            "--infer" => {
                let value = required_value(&mut arguments, option)?;
                if vector {
                    vector_inference_relationships.push(parse_vector_inference(&value)?);
                } else {
                    scalar_inference_relationships.push(parse_scalar_inference(&value)?);
                }
            }
            "--infer-approximate" => {
                if inference_approximate {
                    return Err("--infer-approximate may only be supplied once".to_owned());
                }
                inference_approximate = true;
            }
            "--method" => {
                if robust_requested.is_some() {
                    return Err("--method may only be supplied once".to_owned());
                }
                robust_requested = Some(parse_analysis_method(&required_value(
                    &mut arguments,
                    option,
                )?)?);
            }
            "--robust-tuning" => {
                if robust_tuning.is_some() {
                    return Err("--robust-tuning may only be supplied once".to_owned());
                }
                robust_tuning = Some(parse_positive_f64(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
            }
            "--robust-tolerance" => {
                if robust_tolerance.is_some() {
                    return Err("--robust-tolerance may only be supplied once".to_owned());
                }
                robust_tolerance = Some(parse_positive_f64(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
            }
            "--robust-max-iterations" => {
                if robust_max_iterations.is_some() {
                    return Err("--robust-max-iterations may only be supplied once".to_owned());
                }
                robust_max_iterations = Some(parse_positive_usize(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
            }
            "--confidence" => {
                if confidence_requested.is_some() {
                    return Err("--confidence may only be supplied once".to_owned());
                }
                let value = required_value(&mut arguments, option)?;
                confidence_requested = Some(parse_confidence(&value)?);
            }
            "--white-noise" => white_noise = true,
            "--reconstruct" => reconstruct = true,
            "--reconstruct-constituents" => {
                if reconstruction_constituents.is_some() {
                    return Err("--reconstruct-constituents may only be supplied once".to_owned());
                }
                reconstruction_constituents = Some(parse_explicit_constituents(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
            }
            "--min-pe" => {
                if minimum_percent_energy.is_some() {
                    return Err("--min-pe may only be supplied once".to_owned());
                }
                minimum_percent_energy = Some(parse_nonnegative_f64(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
            }
            "--min-snr" => {
                if minimum_signal_to_noise.is_some() {
                    return Err("--min-snr may only be supplied once".to_owned());
                }
                minimum_signal_to_noise = Some(parse_nonnegative_f64(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
            }
            "--overwrite" => overwrite = true,
            "--help" | "-h" => return Ok(Command::Help),
            _ => {
                let command = if vector {
                    "analyze-vector"
                } else {
                    "analyze-scalar"
                };
                return Err(format!("unknown option for {command}: {option}"));
            }
        }
    }

    let constituent_selection = resolve_constituent_selection(constituents, rayleigh_minimum)?;
    let analysis_method = resolve_analysis_method(
        robust_requested,
        robust_tuning,
        robust_tolerance,
        robust_max_iterations,
    )?;
    let confidence_interval = resolve_confidence_interval(confidence_requested, white_noise)?;
    if inference_approximate
        && scalar_inference_relationships.is_empty()
        && vector_inference_relationships.is_empty()
    {
        return Err("--infer-approximate requires at least one --infer relationship".to_owned());
    }
    let inference_mode = if inference_approximate {
        InferenceMode::Approximate
    } else {
        InferenceMode::Exact
    };
    let reconstruction = resolve_reconstruction(
        reconstruct,
        reconstruction_constituents,
        minimum_percent_energy,
        minimum_signal_to_noise,
        confidence_interval,
    )?;

    let command_name = if vector {
        "analyze-vector"
    } else {
        "analyze-scalar"
    };
    let input = input.ok_or_else(|| format!("{command_name} requires --input PATH"))?;
    let output = output.ok_or_else(|| format!("{command_name} requires --output PATH"))?;
    let selection = selection.unwrap_or(NodeSelection::All);
    if vector {
        let inference =
            (!vector_inference_relationships.is_empty()).then_some(VectorInferenceConfig {
                mode: inference_mode,
                relationships: vector_inference_relationships,
            });
        if inference.is_some() && matches!(analysis_method, AnalysisMethod::Robust(_)) {
            return Err("robust vector inference is not implemented; use --method ols".to_owned());
        }
        Ok(Command::AnalyzeVector(VectorAnalyzeConfig {
            input,
            output,
            report,
            elements: selection,
            constituent_selection,
            inference,
            confidence_interval,
            analysis_method,
            reconstruction,
            workers,
            overwrite,
        }))
    } else {
        let inference =
            (!scalar_inference_relationships.is_empty()).then_some(ScalarInferenceConfig {
                mode: inference_mode,
                relationships: scalar_inference_relationships,
            });
        Ok(Command::AnalyzeScalar(AnalyzeConfig {
            input,
            output,
            report,
            nodes: selection,
            constituent_selection,
            inference,
            confidence_interval,
            analysis_method,
            reconstruction,
            workers,
            overwrite,
        }))
    }
}

fn resolve_analysis_method(
    robust_requested: Option<bool>,
    tuning_constant: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> Result<AnalysisMethod, String> {
    if !robust_requested.unwrap_or(false) {
        if tuning_constant.is_some() || tolerance.is_some() || max_iterations.is_some() {
            return Err("robust options require --method robust".to_owned());
        }
        return Ok(AnalysisMethod::Ols);
    }
    let defaults = RobustOptions::default();
    Ok(AnalysisMethod::Robust(RobustOptions {
        tuning_constant: tuning_constant.unwrap_or(defaults.tuning_constant),
        tolerance: tolerance.unwrap_or(defaults.tolerance),
        max_iterations: max_iterations.unwrap_or(defaults.max_iterations),
    }))
}

fn resolve_reconstruction(
    enabled: bool,
    constituents: Option<Vec<TidalConstituent>>,
    minimum_percent_energy: Option<f64>,
    minimum_signal_to_noise: Option<f64>,
    confidence_interval: ConfidenceInterval,
) -> Result<Option<ReconstructionFilter>, String> {
    if !enabled {
        if constituents.is_some()
            || minimum_percent_energy.is_some()
            || minimum_signal_to_noise.is_some()
        {
            return Err("reconstruction filters require --reconstruct".to_owned());
        }
        return Ok(None);
    }
    if constituents.is_some()
        && (minimum_percent_energy.is_some() || minimum_signal_to_noise.is_some())
    {
        return Err(
            "--reconstruct-constituents is mutually exclusive with --min-pe and --min-snr"
                .to_owned(),
        );
    }
    if minimum_signal_to_noise.is_some() && confidence_interval == ConfidenceInterval::None {
        return Err("--min-snr requires --confidence linear".to_owned());
    }
    Ok(Some(match constituents {
        Some(constituents) => ReconstructionFilter::Constituents(constituents),
        None if minimum_percent_energy.is_some() || minimum_signal_to_noise.is_some() => {
            ReconstructionFilter::Diagnostics {
                minimum_percent_energy: minimum_percent_energy.unwrap_or(0.0),
                minimum_signal_to_noise,
            }
        }
        None => ReconstructionFilter::All,
    }))
}

fn resolve_constituent_selection(
    constituents: Option<ParsedConstituents>,
    rayleigh_minimum: Option<f64>,
) -> Result<ConstituentSelection, String> {
    Ok(match constituents {
        Some(ParsedConstituents::Auto) => ConstituentSelection::Rayleigh {
            minimum: rayleigh_minimum.unwrap_or(1.0),
        },
        Some(ParsedConstituents::Explicit(constituents)) => {
            if rayleigh_minimum.is_some() {
                return Err("--rayleigh-min requires --constituents auto".to_owned());
            }
            ConstituentSelection::Explicit(constituents)
        }
        None => {
            if rayleigh_minimum.is_some() {
                return Err("--rayleigh-min requires --constituents auto".to_owned());
            }
            ConstituentSelection::Explicit(DEFAULT_CONSTITUENTS.to_vec())
        }
    })
}

fn resolve_confidence_interval(
    confidence_requested: Option<bool>,
    white_noise: bool,
) -> Result<ConfidenceInterval, String> {
    if confidence_requested.unwrap_or(false) {
        Ok(ConfidenceInterval::Linear(if white_noise {
            LinearConfidence::White
        } else {
            LinearConfidence::Colored
        }))
    } else if white_noise {
        Err("--white-noise requires --confidence linear".to_owned())
    } else {
        Ok(ConfidenceInterval::None)
    }
}

fn required_value(
    arguments: &mut impl Iterator<Item = OsString>,
    option: &str,
) -> Result<OsString, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_positive_usize(value: &OsStr, option: &str) -> Result<usize, String> {
    let text = value
        .to_str()
        .ok_or_else(|| format!("{option} value must be valid UTF-8"))?;
    let parsed = text
        .parse::<usize>()
        .map_err(|_| format!("{option} requires a positive integer, received {text:?}"))?;
    if parsed == 0 {
        return Err(format!("{option} requires a positive integer"));
    }
    Ok(parsed)
}

fn parse_positive_f64(value: &OsStr, option: &str) -> Result<f64, String> {
    let text = value
        .to_str()
        .ok_or_else(|| format!("{option} value must be valid UTF-8"))?;
    let parsed = text
        .parse::<f64>()
        .map_err(|_| format!("{option} requires a positive number, received {text:?}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!("{option} requires a finite positive number"));
    }
    Ok(parsed)
}

fn parse_nonnegative_f64(value: &OsStr, option: &str) -> Result<f64, String> {
    let text = value
        .to_str()
        .ok_or_else(|| format!("{option} value must be valid UTF-8"))?;
    let parsed = text
        .parse::<f64>()
        .map_err(|_| format!("{option} requires a non-negative number, received {text:?}"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(format!("{option} requires a finite non-negative number"));
    }
    Ok(parsed)
}

fn parse_scalar_inference(value: &OsStr) -> Result<ScalarInferenceRelation, String> {
    let text = value
        .to_str()
        .ok_or_else(|| "--infer value must be valid UTF-8".to_owned())?;
    let fields = text.split(':').collect::<Vec<_>>();
    if fields.len() != 4 {
        return Err("scalar --infer requires INFERRED:REFERENCE:AMP_RATIO:PHASE_OFFSET".to_owned());
    }
    Ok(ScalarInferenceRelation::new(
        parse_inference_constituent(fields[0])?,
        parse_inference_constituent(fields[1])?,
        parse_inference_ratio(fields[2])?,
        parse_inference_phase(fields[3])?,
    ))
}

fn parse_vector_inference(value: &OsStr) -> Result<VectorInferenceRelation, String> {
    let text = value
        .to_str()
        .ok_or_else(|| "--infer value must be valid UTF-8".to_owned())?;
    let fields = text.split(':').collect::<Vec<_>>();
    if fields.len() != 6 {
        return Err(
            "vector --infer requires INFERRED:REFERENCE:AMP+:PHASE+:AMP-:PHASE-".to_owned(),
        );
    }
    Ok(VectorInferenceRelation::new(
        parse_inference_constituent(fields[0])?,
        parse_inference_constituent(fields[1])?,
        parse_inference_ratio(fields[2])?,
        parse_inference_phase(fields[3])?,
        parse_inference_ratio(fields[4])?,
        parse_inference_phase(fields[5])?,
    ))
}

fn parse_inference_constituent(value: &str) -> Result<TidalConstituent, String> {
    value
        .parse::<TidalConstituent>()
        .map_err(|error| error.to_string())
}

fn parse_inference_ratio(value: &str) -> Result<f64, String> {
    let ratio = value
        .parse::<f64>()
        .map_err(|_| format!("inference amplitude ratio must be numeric, received {value:?}"))?;
    if !ratio.is_finite() || ratio < 0.0 {
        return Err("inference amplitude ratio must be finite and non-negative".to_owned());
    }
    Ok(ratio)
}

fn parse_inference_phase(value: &str) -> Result<f64, String> {
    let phase = value
        .parse::<f64>()
        .map_err(|_| format!("inference phase offset must be numeric, received {value:?}"))?;
    if !phase.is_finite() {
        return Err("inference phase offset must be finite".to_owned());
    }
    Ok(phase)
}

fn parse_confidence(value: &OsStr) -> Result<bool, String> {
    match value.to_str() {
        Some("none") => Ok(false),
        Some("linear") => Ok(true),
        Some(value) => Err(format!(
            "--confidence must be 'none' or 'linear', received {value:?}"
        )),
        None => Err("--confidence value must be valid UTF-8".to_owned()),
    }
}

fn parse_analysis_method(value: &OsStr) -> Result<bool, String> {
    match value.to_str() {
        Some("ols") => Ok(false),
        Some("robust") => Ok(true),
        Some(value) => Err(format!(
            "--method must be 'ols' or 'robust', received {value:?}"
        )),
        None => Err("--method value must be valid UTF-8".to_owned()),
    }
}

fn ensure_selection_is_unset(selection: Option<&NodeSelection>) -> Result<(), String> {
    if selection.is_some() {
        return Err("--nodes and --node-count are mutually exclusive".to_owned());
    }
    Ok(())
}

fn parse_nodes(value: &OsStr) -> Result<NodeSelection, String> {
    let text = value
        .to_str()
        .ok_or_else(|| "--nodes value must be valid UTF-8".to_owned())?;
    if text == "all" {
        return Ok(NodeSelection::All);
    }
    let indices = text
        .split(',')
        .map(|item| {
            item.parse::<usize>()
                .map_err(|_| format!("invalid zero-based node index: {item:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if indices.is_empty() {
        return Err("--nodes requires 'all' or a comma-separated index list".to_owned());
    }
    Ok(NodeSelection::Indices(indices))
}

enum ParsedConstituents {
    Auto,
    Explicit(Vec<TidalConstituent>),
}

fn parse_constituents(value: &OsStr) -> Result<ParsedConstituents, String> {
    let text = value
        .to_str()
        .ok_or_else(|| "--constituents value must be valid UTF-8".to_owned())?;
    if text.is_empty() {
        return Err("--constituents requires a comma-separated name list".to_owned());
    }
    if text == "auto" {
        return Ok(ParsedConstituents::Auto);
    }
    let mut constituents = Vec::new();
    for raw_name in text.split(',') {
        let name = raw_name.trim();
        if name.is_empty() {
            return Err("--constituents contains an empty name".to_owned());
        }
        let constituent = name
            .parse::<TidalConstituent>()
            .map_err(|error| error.to_string())?;
        if constituents.contains(&constituent) {
            return Err(format!("constituent {name} appears more than once"));
        }
        constituents.push(constituent);
    }
    Ok(ParsedConstituents::Explicit(constituents))
}

fn parse_explicit_constituents(
    value: &OsStr,
    option: &str,
) -> Result<Vec<TidalConstituent>, String> {
    match parse_constituents(value)? {
        ParsedConstituents::Explicit(constituents) => Ok(constituents),
        ParsedConstituents::Auto => Err(format!("{option} requires explicit constituent names")),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{Command, parse_arguments};
    use rutide_cli::{
        AnalysisMethod, ConfidenceInterval, ConstituentSelection, DEFAULT_CONSTITUENTS,
        NodeSelection, ScalarInferenceConfig, VectorInferenceConfig,
    };
    use rutide_core::{
        InferenceMode, LinearConfidence, ReconstructionFilter, RobustOptions,
        ScalarInferenceRelation, TidalConstituent, VectorInferenceRelation,
    };

    fn args<'a>(values: &'a [&'a str]) -> impl Iterator<Item = OsString> + 'a {
        values.iter().map(OsString::from)
    }

    #[test]
    fn parses_explicit_analysis_selection() {
        let command = parse_arguments(args(&[
            "analyze-scalar",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--nodes",
            "4,1,3",
            "--workers",
            "8",
        ]))
        .expect("valid arguments");
        let Command::AnalyzeScalar(config) = command else {
            panic!("expected analyze command");
        };
        assert_eq!(config.nodes, NodeSelection::Indices(vec![4, 1, 3]));
        assert_eq!(
            config.constituent_selection,
            ConstituentSelection::Explicit(DEFAULT_CONSTITUENTS.to_vec())
        );
        assert_eq!(config.confidence_interval, ConfidenceInterval::None);
        assert_eq!(config.analysis_method, AnalysisMethod::Ols);
        assert_eq!(config.reconstruction, None);
        assert_eq!(config.workers, 8);
    }

    #[test]
    fn parses_robust_method_and_options() {
        let command = parse_arguments(args(&[
            "analyze-scalar",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--method",
            "robust",
            "--robust-tuning",
            "2.5",
            "--robust-tolerance",
            "0.0005",
            "--robust-max-iterations",
            "75",
        ]))
        .expect("valid robust options");
        let Command::AnalyzeScalar(config) = command else {
            panic!("expected scalar analyze command");
        };
        assert_eq!(
            config.analysis_method,
            AnalysisMethod::Robust(RobustOptions {
                tuning_constant: 2.5,
                tolerance: 0.0005,
                max_iterations: 75,
            })
        );
        assert!(
            parse_arguments(args(&[
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
                "--robust-tuning",
                "2.5",
            ]))
            .is_err()
        );
    }

    #[test]
    fn parses_vector_element_selection_and_shared_options() {
        let command = parse_arguments(args(&[
            "analyze-vector",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--elements",
            "4,1,3",
            "--confidence",
            "linear",
            "--reconstruct",
        ]))
        .expect("valid vector arguments");
        let Command::AnalyzeVector(config) = command else {
            panic!("expected vector analyze command");
        };
        assert_eq!(config.elements, NodeSelection::Indices(vec![4, 1, 3]));
        assert_eq!(
            config.confidence_interval,
            ConfidenceInterval::Linear(LinearConfidence::Colored)
        );
        assert_eq!(config.reconstruction, Some(ReconstructionFilter::All));
        assert!(
            parse_arguments(args(&[
                "analyze-vector",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
                "--nodes",
                "1",
            ]))
            .is_err()
        );
    }

    #[test]
    fn parses_scalar_and_vector_inference_and_rejects_invalid_combinations() {
        let scalar = parse_arguments(args(&[
            "analyze-scalar",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--infer",
            "S2:M2:0.35:20",
            "--infer-approximate",
        ]))
        .expect("valid scalar inference");
        let Command::AnalyzeScalar(config) = scalar else {
            panic!("expected scalar command");
        };
        assert_eq!(
            config.inference,
            Some(ScalarInferenceConfig {
                mode: InferenceMode::Approximate,
                relationships: vec![ScalarInferenceRelation::new(
                    TidalConstituent::S2,
                    TidalConstituent::M2,
                    0.35,
                    20.0,
                )],
            })
        );

        let vector = parse_arguments(args(&[
            "analyze-vector",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--infer",
            "O1:K1:0.5:45:0.4:30",
        ]))
        .expect("valid vector inference");
        let Command::AnalyzeVector(config) = vector else {
            panic!("expected vector command");
        };
        assert_eq!(
            config.inference,
            Some(VectorInferenceConfig {
                mode: InferenceMode::Exact,
                relationships: vec![VectorInferenceRelation::new(
                    TidalConstituent::O1,
                    TidalConstituent::K1,
                    0.5,
                    45.0,
                    0.4,
                    30.0,
                )],
            })
        );

        for invalid in [
            &["--infer-approximate"][..],
            &["--infer", "S2:M2:-0.1:20"][..],
            &["--infer", "S2:M2:0.3"][..],
            &["--infer", "S2:M2:0.3:20:0.2:10", "--method", "robust"][..],
        ] {
            let mut values = vec![
                "analyze-vector",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
            ];
            values.extend_from_slice(invalid);
            assert!(parse_arguments(args(&values)).is_err());
        }
    }

    #[test]
    fn rejects_conflicting_selections() {
        assert!(
            parse_arguments(args(&[
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
                "--nodes",
                "1",
                "--node-count",
                "2",
            ]))
            .is_err()
        );
    }

    #[test]
    fn parses_dynamic_base_and_shallow_constituents_in_input_order() {
        let command = parse_arguments(args(&[
            "analyze-scalar",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--constituents",
            "Q1,M2,M4,MK3",
        ]))
        .expect("valid arguments");
        let Command::AnalyzeScalar(config) = command else {
            panic!("expected analyze command");
        };
        assert_eq!(
            config.constituent_selection,
            ConstituentSelection::Explicit(
                ["Q1", "M2", "M4", "MK3"]
                    .map(|name| name.parse::<TidalConstituent>().expect("catalog name"))
                    .to_vec()
            )
        );
    }

    #[test]
    fn rejects_unknown_or_duplicate_constituents() {
        for names in ["M2,NOT_A_TIDE", "M2,S2,M2", "M2,,S2"] {
            assert!(
                parse_arguments(args(&[
                    "analyze-scalar",
                    "--input",
                    "input.nc",
                    "--output",
                    "output.nc",
                    "--constituents",
                    names,
                ]))
                .is_err()
            );
        }
    }

    #[test]
    fn parses_rayleigh_auto_selection_and_custom_minimum() {
        let command = parse_arguments(args(&[
            "analyze-scalar",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--constituents",
            "auto",
            "--rayleigh-min",
            "0.95",
        ]))
        .expect("valid arguments");
        let Command::AnalyzeScalar(config) = command else {
            panic!("expected analyze command");
        };
        assert_eq!(
            config.constituent_selection,
            ConstituentSelection::Rayleigh { minimum: 0.95 }
        );
    }

    #[test]
    fn rejects_rayleigh_minimum_without_valid_auto_selection() {
        for extra in [
            &["--rayleigh-min", "1.0"][..],
            &["--constituents", "M2,S2", "--rayleigh-min", "1.0"][..],
            &["--constituents", "auto", "--rayleigh-min", "NaN"][..],
        ] {
            let mut values = vec![
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
            ];
            values.extend_from_slice(extra);
            assert!(parse_arguments(args(&values)).is_err());
        }
    }

    #[test]
    fn parses_colored_and_white_linear_confidence() {
        for (extra, expected) in [
            (
                &["--confidence", "linear"][..],
                ConfidenceInterval::Linear(LinearConfidence::Colored),
            ),
            (
                &["--confidence", "linear", "--white-noise"][..],
                ConfidenceInterval::Linear(LinearConfidence::White),
            ),
        ] {
            let mut values = vec![
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
            ];
            values.extend_from_slice(extra);
            let command = parse_arguments(args(&values)).expect("valid confidence options");
            let Command::AnalyzeScalar(config) = command else {
                panic!("expected analyze command");
            };
            assert_eq!(config.confidence_interval, expected);
        }
    }

    #[test]
    fn rejects_white_noise_without_linear_confidence() {
        assert!(
            parse_arguments(args(&[
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
                "--white-noise",
            ]))
            .is_err()
        );
    }

    #[test]
    fn parses_all_explicit_and_diagnostic_reconstruction_filters() {
        for (extra, expected) in [
            (
                &["--reconstruct", "--reconstruct-constituents", "M2,K1"][..],
                ReconstructionFilter::Constituents(vec![
                    TidalConstituent::M2,
                    TidalConstituent::K1,
                ]),
            ),
            (
                &[
                    "--confidence",
                    "linear",
                    "--reconstruct",
                    "--min-pe",
                    "5",
                    "--min-snr",
                    "2",
                ][..],
                ReconstructionFilter::Diagnostics {
                    minimum_percent_energy: 5.0,
                    minimum_signal_to_noise: Some(2.0),
                },
            ),
            (&["--reconstruct"][..], ReconstructionFilter::All),
        ] {
            let mut values = vec![
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
            ];
            values.extend_from_slice(extra);
            let command = parse_arguments(args(&values)).expect("valid reconstruction options");
            let Command::AnalyzeScalar(config) = command else {
                panic!("expected analyze command");
            };
            assert_eq!(config.reconstruction, Some(expected));
        }
    }

    #[test]
    fn rejects_ambiguous_or_unavailable_reconstruction_filters() {
        for extra in [
            &["--min-pe", "5"][..],
            &["--reconstruct", "--min-snr", "2"][..],
            &[
                "--reconstruct",
                "--reconstruct-constituents",
                "M2",
                "--min-pe",
                "5",
            ][..],
            &["--reconstruct", "--reconstruct-constituents", "auto"][..],
        ] {
            let mut values = vec![
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
            ];
            values.extend_from_slice(extra);
            assert!(parse_arguments(args(&values)).is_err());
        }
    }
}
