//! `RUTide` command-line entry point.

use std::{
    env,
    ffi::{OsStr, OsString},
    path::PathBuf,
    process::ExitCode,
};

use rutide_cli::{
    AnalysisMethod, AnalyzeConfig, ConfidenceInterval, ConstituentOrder, ConstituentSelection,
    DEFAULT_CONSTITUENTS, NodeSelection, ScalarInferenceConfig, VectorAnalyzeConfig,
    VectorInferenceConfig, analyze_scalar, analyze_vector,
};
use rutide_core::{
    FitOptions, InferenceMode, LinearConfidence, MonteCarloOptions, NodalCorrections,
    PhaseReference, ReconstructionFilter, RobustOptions, RobustWeightFunction,
    ScalarInferenceRelation, TidalConstituent, VectorInferenceRelation,
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
  --chunk-series N    Spatial series held in memory per chunk (default: automatic)
  --node-count N      Analyze the first N nodes
  --nodes I,J,...     Analyze explicit zero-based node indices
  --element-count N   Analyze the first N elements (vector mode)
  --elements I,J,...  Analyze explicit zero-based element indices (vector mode)
  --layer-count N     Analyze the first N native sigma layers (vector mode)
  --layers I,J,...    Analyze explicit zero-based native sigma-layer indices
  --depths D,D,...    Interpolate currents to positive metres below the free surface
  --constituents LIST Comma-separated names or 'auto' (default: M2,S2,N2,K1,O1)
  --rayleigh-min X    Automatic selection criterion (default with auto: 1.0)
  --order ORDER       Presentation: selection, pe, snr, frequency, or a full name list
  --infer SPEC        Repeatable inferred relationship:
                      scalar I:R:AMP:PHASE; vector I:R:AMP+:PHASE+:AMP-:PHASE-
  --infer-approximate Use Python-compatible reference-only approximate inference
  --no-trend         Fit a mean without a linear trend (default: mean and trend)
  --phase MODE        Phase reference: greenwich, linear-time, or raw (default: greenwich)
  --nodal MODE        Nodal corrections: exact, linear-time, or disabled (default: exact)
  --method MODE       Least squares: ols or robust (default: ols)
  --robust-weight W   IRLS weight: andrews, bisquare, cauchy, fair, huber,
                      logistic, ols, talwar, or welsch (default: cauchy)
  --robust-tuning X   Weight-function tuning constant (default: conventional)
  --robust-tolerance X
                      Fractional IRLS tolerance (default: 0.001)
  --robust-max-iterations N
                      Maximum IRLS iterations (default: 50)
  --confidence MODE   Confidence intervals: none, linear, or monte-carlo (default: none)
  --white-noise       Use white noise instead of colored residual bands
  --mc-realizations N Coefficient draws for monte-carlo confidence (default: 200)
  --mc-seed N         Reproducible unsigned 64-bit root seed (default: 0)
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
    let mut chunk_series = None;
    let mut selection = None;
    let mut layer_selection = None;
    let mut fixed_depths_meters = None;
    let mut constituents = None;
    let mut rayleigh_minimum = None;
    let mut constituent_order = None;
    let mut scalar_inference_relationships = Vec::new();
    let mut vector_inference_relationships = Vec::new();
    let mut inference_approximate = false;
    let mut trend_disabled = false;
    let mut phase_reference = None;
    let mut nodal_corrections = None;
    let mut robust_requested = None;
    let mut robust_weight_function = None;
    let mut robust_tuning = None;
    let mut robust_tolerance = None;
    let mut robust_max_iterations = None;
    let mut confidence_requested = None;
    let mut white_noise = false;
    let mut monte_carlo_realizations = None;
    let mut monte_carlo_seed = None;
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
            "--chunk-series" => {
                if chunk_series.is_some() {
                    return Err("--chunk-series may only be supplied once".to_owned());
                }
                chunk_series = Some(parse_positive_usize(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
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
                selection = Some(parse_index_selection(
                    &required_value(&mut arguments, option)?,
                    option,
                    "element",
                )?);
            }
            "--layer-count" => {
                if !vector {
                    return Err("--layer-count is only valid for analyze-vector".to_owned());
                }
                if layer_selection.is_some() || fixed_depths_meters.is_some() {
                    return Err(
                        "--layers, --layer-count, and --depths are mutually exclusive".to_owned(),
                    );
                }
                let count = parse_positive_usize(&required_value(&mut arguments, option)?, option)?;
                layer_selection = Some(NodeSelection::Prefix(count));
            }
            "--layers" => {
                if !vector {
                    return Err("--layers is only valid for analyze-vector".to_owned());
                }
                if layer_selection.is_some() || fixed_depths_meters.is_some() {
                    return Err(
                        "--layers, --layer-count, and --depths are mutually exclusive".to_owned(),
                    );
                }
                layer_selection = Some(parse_index_selection(
                    &required_value(&mut arguments, option)?,
                    option,
                    "layer",
                )?);
            }
            "--depths" => {
                if !vector {
                    return Err("--depths is only valid for analyze-vector".to_owned());
                }
                if layer_selection.is_some() || fixed_depths_meters.is_some() {
                    return Err(
                        "--layers, --layer-count, and --depths are mutually exclusive".to_owned(),
                    );
                }
                fixed_depths_meters = Some(parse_fixed_depths(&required_value(
                    &mut arguments,
                    option,
                )?)?);
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
            "--order" => {
                if constituent_order.is_some() {
                    return Err("--order may only be supplied once".to_owned());
                }
                constituent_order = Some(parse_constituent_order(&required_value(
                    &mut arguments,
                    option,
                )?)?);
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
            "--no-trend" => {
                if trend_disabled {
                    return Err("--no-trend may only be supplied once".to_owned());
                }
                trend_disabled = true;
            }
            "--phase" => {
                if phase_reference.is_some() {
                    return Err("--phase may only be supplied once".to_owned());
                }
                phase_reference = Some(parse_phase_reference(&required_value(
                    &mut arguments,
                    option,
                )?)?);
            }
            "--nodal" => {
                if nodal_corrections.is_some() {
                    return Err("--nodal may only be supplied once".to_owned());
                }
                nodal_corrections = Some(parse_nodal_corrections(&required_value(
                    &mut arguments,
                    option,
                )?)?);
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
            "--robust-weight" => {
                if robust_weight_function.is_some() {
                    return Err("--robust-weight may only be supplied once".to_owned());
                }
                robust_weight_function = Some(parse_robust_weight_function(&required_value(
                    &mut arguments,
                    option,
                )?)?);
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
            "--mc-realizations" => {
                if monte_carlo_realizations.is_some() {
                    return Err("--mc-realizations may only be supplied once".to_owned());
                }
                monte_carlo_realizations = Some(parse_positive_usize(
                    &required_value(&mut arguments, option)?,
                    option,
                )?);
            }
            "--mc-seed" => {
                if monte_carlo_seed.is_some() {
                    return Err("--mc-seed may only be supplied once".to_owned());
                }
                monte_carlo_seed =
                    Some(parse_u64(&required_value(&mut arguments, option)?, option)?);
            }
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
        robust_weight_function,
        robust_tuning,
        robust_tolerance,
        robust_max_iterations,
    )?;
    let confidence_interval = resolve_confidence_interval(
        confidence_requested,
        white_noise,
        monte_carlo_realizations,
        monte_carlo_seed,
    )?;
    let constituent_order = constituent_order.unwrap_or_default();
    if constituent_order == ConstituentOrder::SignalToNoise
        && confidence_interval == ConfidenceInterval::None
    {
        return Err("--order snr requires confidence intervals".to_owned());
    }
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
    let fit_options = FitOptions {
        trend: !trend_disabled,
    };
    let phase_reference = phase_reference.unwrap_or_default();
    let nodal_corrections = nodal_corrections.unwrap_or_default();
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
        Ok(Command::AnalyzeVector(VectorAnalyzeConfig {
            input,
            output,
            report,
            elements: selection,
            layers: layer_selection,
            fixed_depths_meters,
            constituent_selection,
            constituent_order,
            inference,
            fit_options,
            phase_reference,
            nodal_corrections,
            confidence_interval,
            analysis_method,
            reconstruction,
            workers,
            chunk_series,
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
            constituent_order,
            inference,
            fit_options,
            phase_reference,
            nodal_corrections,
            confidence_interval,
            analysis_method,
            reconstruction,
            workers,
            chunk_series,
            overwrite,
        }))
    }
}

fn resolve_analysis_method(
    robust_requested: Option<bool>,
    weight_function: Option<RobustWeightFunction>,
    tuning_constant: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> Result<AnalysisMethod, String> {
    if !robust_requested.unwrap_or(false) {
        if weight_function.is_some()
            || tuning_constant.is_some()
            || tolerance.is_some()
            || max_iterations.is_some()
        {
            return Err("robust options require --method robust".to_owned());
        }
        return Ok(AnalysisMethod::Ols);
    }
    let defaults =
        RobustOptions::for_weight_function(weight_function.unwrap_or(RobustWeightFunction::Cauchy));
    Ok(AnalysisMethod::Robust(RobustOptions {
        weight_function: defaults.weight_function,
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
        return Err("--min-snr requires linear or monte-carlo confidence".to_owned());
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
    confidence_requested: Option<ParsedConfidence>,
    white_noise: bool,
    monte_carlo_realizations: Option<usize>,
    monte_carlo_seed: Option<u64>,
) -> Result<ConfidenceInterval, String> {
    let noise = if white_noise {
        LinearConfidence::White
    } else {
        LinearConfidence::Colored
    };
    match confidence_requested.unwrap_or(ParsedConfidence::None) {
        ParsedConfidence::None => {
            if white_noise || monte_carlo_realizations.is_some() || monte_carlo_seed.is_some() {
                return Err(
                    "--white-noise, --mc-realizations, and --mc-seed require an enabled --confidence mode"
                        .to_owned(),
                );
            }
            Ok(ConfidenceInterval::None)
        }
        ParsedConfidence::Linear => {
            if monte_carlo_realizations.is_some() || monte_carlo_seed.is_some() {
                return Err(
                    "--mc-realizations and --mc-seed require --confidence monte-carlo".to_owned(),
                );
            }
            Ok(ConfidenceInterval::Linear(noise))
        }
        ParsedConfidence::MonteCarlo => {
            let realizations = monte_carlo_realizations.unwrap_or(200);
            if realizations < 2 {
                return Err("--mc-realizations requires an integer of at least 2".to_owned());
            }
            Ok(ConfidenceInterval::MonteCarlo {
                options: MonteCarloOptions {
                    realizations,
                    seed: monte_carlo_seed.unwrap_or(0),
                },
                noise,
            })
        }
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

fn parse_u64(value: &OsStr, option: &str) -> Result<u64, String> {
    let text = value
        .to_str()
        .ok_or_else(|| format!("{option} value must be valid UTF-8"))?;
    text.parse::<u64>()
        .map_err(|_| format!("{option} requires an unsigned 64-bit integer, received {text:?}"))
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

fn parse_fixed_depths(value: &OsStr) -> Result<Vec<f64>, String> {
    let text = value
        .to_str()
        .ok_or_else(|| "--depths value must be valid UTF-8".to_owned())?;
    if text.is_empty() {
        return Err("--depths requires a comma-separated depth list".to_owned());
    }
    let mut depths = Vec::new();
    for item in text.split(',') {
        let depth = parse_positive_f64(OsStr::new(item), "--depths")?;
        if depths
            .iter()
            .any(|existing: &f64| existing.to_bits() == depth.to_bits())
        {
            return Err(format!("--depths contains duplicate depth {depth}"));
        }
        depths.push(depth);
    }
    Ok(depths)
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

#[derive(Clone, Copy)]
enum ParsedConfidence {
    None,
    Linear,
    MonteCarlo,
}

fn parse_confidence(value: &OsStr) -> Result<ParsedConfidence, String> {
    match value.to_str() {
        Some("none") => Ok(ParsedConfidence::None),
        Some("linear") => Ok(ParsedConfidence::Linear),
        Some("monte-carlo" | "mc") => Ok(ParsedConfidence::MonteCarlo),
        Some(value) => Err(format!(
            "--confidence must be 'none', 'linear', or 'monte-carlo', received {value:?}"
        )),
        None => Err("--confidence value must be valid UTF-8".to_owned()),
    }
}

fn parse_phase_reference(value: &OsStr) -> Result<PhaseReference, String> {
    match value.to_str() {
        Some("greenwich") => Ok(PhaseReference::Greenwich),
        Some("linear-time") => Ok(PhaseReference::LinearTime),
        Some("raw") => Ok(PhaseReference::Raw),
        Some(value) => Err(format!(
            "--phase must be greenwich, linear-time, or raw, got {value}"
        )),
        None => Err("--phase must be valid UTF-8".to_owned()),
    }
}

fn parse_nodal_corrections(value: &OsStr) -> Result<NodalCorrections, String> {
    match value.to_str() {
        Some("exact") => Ok(NodalCorrections::Exact),
        Some("linear-time") => Ok(NodalCorrections::LinearTime),
        Some("disabled") => Ok(NodalCorrections::Disabled),
        Some(value) => Err(format!(
            "--nodal must be exact, linear-time, or disabled, got {value}"
        )),
        None => Err("--nodal must be valid UTF-8".to_owned()),
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

fn parse_robust_weight_function(value: &OsStr) -> Result<RobustWeightFunction, String> {
    match value.to_str() {
        Some("andrews") => Ok(RobustWeightFunction::Andrews),
        Some("bisquare") => Ok(RobustWeightFunction::Bisquare),
        Some("cauchy") => Ok(RobustWeightFunction::Cauchy),
        Some("fair") => Ok(RobustWeightFunction::Fair),
        Some("huber") => Ok(RobustWeightFunction::Huber),
        Some("logistic") => Ok(RobustWeightFunction::Logistic),
        Some("ols") => Ok(RobustWeightFunction::Ols),
        Some("talwar") => Ok(RobustWeightFunction::Talwar),
        Some("welsch") => Ok(RobustWeightFunction::Welsch),
        Some(value) => Err(format!(
            "--robust-weight must be andrews, bisquare, cauchy, fair, huber, logistic, ols, talwar, or welsch; received {value:?}"
        )),
        None => Err("--robust-weight value must be valid UTF-8".to_owned()),
    }
}

fn ensure_selection_is_unset(selection: Option<&NodeSelection>) -> Result<(), String> {
    if selection.is_some() {
        return Err("--nodes and --node-count are mutually exclusive".to_owned());
    }
    Ok(())
}

fn parse_nodes(value: &OsStr) -> Result<NodeSelection, String> {
    parse_index_selection(value, "--nodes", "node")
}

fn parse_index_selection(
    value: &OsStr,
    option: &str,
    index_name: &str,
) -> Result<NodeSelection, String> {
    let text = value
        .to_str()
        .ok_or_else(|| format!("{option} value must be valid UTF-8"))?;
    if text == "all" {
        return Ok(NodeSelection::All);
    }
    let indices = text
        .split(',')
        .map(|item| {
            item.parse::<usize>()
                .map_err(|_| format!("invalid zero-based {index_name} index: {item:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if indices.is_empty() {
        return Err(format!(
            "{option} requires 'all' or a comma-separated index list"
        ));
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
    parse_explicit_constituents(value, "--constituents").map(ParsedConstituents::Explicit)
}

fn parse_constituent_order(value: &OsStr) -> Result<ConstituentOrder, String> {
    match value.to_str() {
        Some(text) if text.eq_ignore_ascii_case("selection") => Ok(ConstituentOrder::Selection),
        Some(text) if text.eq_ignore_ascii_case("pe") => Ok(ConstituentOrder::PercentEnergy),
        Some(text) if text.eq_ignore_ascii_case("snr") => Ok(ConstituentOrder::SignalToNoise),
        Some(text) if text.eq_ignore_ascii_case("frequency") => Ok(ConstituentOrder::Frequency),
        Some(_) => parse_explicit_constituents(value, "--order").map(ConstituentOrder::Explicit),
        None => Err("--order value must be valid UTF-8".to_owned()),
    }
}

fn parse_explicit_constituents(
    value: &OsStr,
    option: &str,
) -> Result<Vec<TidalConstituent>, String> {
    let text = value
        .to_str()
        .ok_or_else(|| format!("{option} value must be valid UTF-8"))?;
    if text.is_empty() || text == "auto" {
        return Err(format!(
            "{option} requires a comma-separated explicit constituent list"
        ));
    }
    let mut constituents = Vec::new();
    for raw_name in text.split(',') {
        let name = raw_name.trim();
        if name.is_empty() {
            return Err(format!("{option} contains an empty name"));
        }
        let constituent = name
            .parse::<TidalConstituent>()
            .map_err(|error| error.to_string())?;
        if constituents.contains(&constituent) {
            return Err(format!(
                "{option} constituent {name} appears more than once"
            ));
        }
        constituents.push(constituent);
    }
    Ok(constituents)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{Command, parse_arguments};
    use rutide_cli::{
        AnalysisMethod, ConfidenceInterval, ConstituentOrder, ConstituentSelection,
        DEFAULT_CONSTITUENTS, NodeSelection, ScalarInferenceConfig, VectorInferenceConfig,
    };
    use rutide_core::{
        FitOptions, InferenceMode, LinearConfidence, MonteCarloOptions, NodalCorrections,
        PhaseReference, ReconstructionFilter, RobustOptions, RobustWeightFunction,
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
            "--chunk-series",
            "1024",
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
        assert_eq!(config.nodal_corrections, NodalCorrections::Exact);
        assert_eq!(config.constituent_order, ConstituentOrder::Selection);
        assert_eq!(config.workers, 8);
        assert_eq!(config.chunk_series, Some(1024));
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
            "--robust-weight",
            "welsch",
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
                weight_function: RobustWeightFunction::Welsch,
                tuning_constant: 2.5,
                tolerance: 0.0005,
                max_iterations: 75,
            })
        );
        for robust_option in [
            &["--robust-tuning", "2.5"][..],
            &["--robust-weight", "welsch"][..],
        ] {
            let mut values = vec![
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
            ];
            values.extend_from_slice(robust_option);
            assert!(parse_arguments(args(&values)).is_err());
        }
        let command = parse_arguments(args(&[
            "analyze-vector",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--method",
            "robust",
            "--robust-weight",
            "bisquare",
        ]))
        .expect("valid conventional robust tuning");
        let Command::AnalyzeVector(config) = command else {
            panic!("expected vector analyze command");
        };
        assert_eq!(
            config.analysis_method,
            AnalysisMethod::Robust(RobustOptions::for_weight_function(
                RobustWeightFunction::Bisquare,
            ))
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
            "--layers",
            "2,0",
            "--confidence",
            "linear",
            "--phase",
            "linear-time",
            "--nodal",
            "linear-time",
            "--order",
            "frequency",
            "--reconstruct",
        ]))
        .expect("valid vector arguments");
        let Command::AnalyzeVector(config) = command else {
            panic!("expected vector analyze command");
        };
        assert_eq!(config.elements, NodeSelection::Indices(vec![4, 1, 3]));
        assert_eq!(config.layers, Some(NodeSelection::Indices(vec![2, 0])));
        assert_eq!(config.fixed_depths_meters, None);
        assert_eq!(
            config.confidence_interval,
            ConfidenceInterval::Linear(LinearConfidence::Colored)
        );
        assert_eq!(config.reconstruction, Some(ReconstructionFilter::All));
        assert_eq!(config.phase_reference, PhaseReference::LinearTime);
        assert_eq!(config.nodal_corrections, NodalCorrections::LinearTime);
        assert_eq!(config.constituent_order, ConstituentOrder::Frequency);
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
        assert!(
            parse_arguments(args(&[
                "analyze-vector",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
                "--layers",
                "0",
                "--layer-count",
                "2",
            ]))
            .is_err()
        );
        assert!(
            parse_arguments(args(&[
                "analyze-scalar",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
                "--layers",
                "0",
            ]))
            .is_err()
        );
    }

    #[test]
    fn parses_fixed_depths_and_rejects_invalid_combinations() {
        let Command::AnalyzeVector(fixed_depth) = parse_arguments(args(&[
            "analyze-vector",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--depths",
            "5,12.5,30",
        ]))
        .expect("valid fixed-depth arguments") else {
            panic!("expected fixed-depth vector command");
        };
        assert_eq!(fixed_depth.layers, None);
        assert_eq!(fixed_depth.fixed_depths_meters, Some(vec![5.0, 12.5, 30.0]));
        for arguments in [
            vec!["--depths", "5", "--layers", "0"],
            vec!["--depths", "5,5"],
            vec!["--depths", "0"],
        ] {
            let mut command = vec![
                "analyze-vector",
                "--input",
                "input.nc",
                "--output",
                "output.nc",
            ];
            command.extend(arguments);
            assert!(parse_arguments(args(&command)).is_err());
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "covers shared scalar/vector inference and solver-option parsing"
    )]
    fn parses_scalar_and_vector_inference_and_rejects_invalid_combinations() {
        let scalar = parse_arguments(args(&[
            "analyze-scalar",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--constituents",
            "M2",
            "--infer",
            "S2:M2:0.35:20",
            "--order",
            "S2,M2",
            "--infer-approximate",
            "--no-trend",
            "--phase",
            "raw",
            "--nodal",
            "disabled",
            "--confidence",
            "monte-carlo",
            "--mc-realizations",
            "33",
            "--mc-seed",
            "7",
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
        assert_eq!(config.fit_options, FitOptions { trend: false });
        assert_eq!(config.phase_reference, PhaseReference::Raw);
        assert_eq!(config.nodal_corrections, NodalCorrections::Disabled);
        assert_eq!(
            config.constituent_order,
            ConstituentOrder::Explicit(vec![TidalConstituent::S2, TidalConstituent::M2])
        );
        assert_eq!(
            config.confidence_interval,
            ConfidenceInterval::MonteCarlo {
                options: MonteCarloOptions {
                    realizations: 33,
                    seed: 7,
                },
                noise: LinearConfidence::Colored,
            }
        );

        let vector = parse_arguments(args(&[
            "analyze-vector",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--infer",
            "O1:K1:0.5:45:0.4:30",
            "--method",
            "robust",
            "--confidence",
            "monte-carlo",
            "--order",
            "SNR",
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
        assert!(matches!(config.analysis_method, AnalysisMethod::Robust(_)));
        assert_eq!(config.fit_options, FitOptions::default());
        assert_eq!(config.phase_reference, PhaseReference::Greenwich);
        assert_eq!(config.nodal_corrections, NodalCorrections::Exact);
        assert_eq!(config.constituent_order, ConstituentOrder::SignalToNoise);
        assert!(matches!(
            config.confidence_interval,
            ConfidenceInterval::MonteCarlo { .. }
        ));

        for invalid in [
            &["--infer-approximate"][..],
            &["--no-trend", "--no-trend"][..],
            &["--phase", "raw", "--phase", "greenwich"][..],
            &["--phase", "linear_time"][..],
            &["--nodal", "exact", "--nodal", "disabled"][..],
            &["--nodal", "linear_time"][..],
            &["--order", "snr"][..],
            &["--order", "pe", "--order", "frequency"][..],
            &["--order", "M2,M2"][..],
            &["--infer", "S2:M2:-0.1:20"][..],
            &["--infer", "S2:M2:0.3"][..],
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
    fn parses_seeded_monte_carlo_and_rejects_invalid_combinations() {
        let command = parse_arguments(args(&[
            "analyze-vector",
            "--input",
            "input.nc",
            "--output",
            "output.nc",
            "--confidence",
            "monte-carlo",
            "--white-noise",
            "--mc-realizations",
            "512",
            "--mc-seed",
            "18446744073709551615",
        ]))
        .expect("valid Monte Carlo options");
        let Command::AnalyzeVector(config) = command else {
            panic!("expected vector command");
        };
        assert_eq!(
            config.confidence_interval,
            ConfidenceInterval::MonteCarlo {
                options: MonteCarloOptions {
                    realizations: 512,
                    seed: u64::MAX,
                },
                noise: LinearConfidence::White,
            }
        );

        for extra in [
            &["--confidence", "linear", "--mc-seed", "1"][..],
            &["--confidence", "monte-carlo", "--mc-realizations", "1"][..],
        ] {
            let mut values = vec![
                "analyze-vector",
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
    fn rejects_white_noise_without_confidence() {
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
