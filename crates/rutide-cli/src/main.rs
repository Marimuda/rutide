//! `RUTide` command-line entry point.

use std::{
    env,
    ffi::{OsStr, OsString},
    path::PathBuf,
    process::ExitCode,
};

use rutide_cli::{
    AnalyzeConfig, ConfidenceInterval, ConstituentSelection, DEFAULT_CONSTITUENTS, NodeSelection,
    analyze_scalar,
};
use rutide_core::{LinearConfidence, TidalConstituent};

// The application repeatedly allocates short-lived QR storage across worker
// threads; this allocator keeps that full-field pattern bounded and reusable.
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const USAGE: &str = "\
Usage:
  rutide --version
  rutide analyze-scalar --input PATH --output PATH [OPTIONS]

Options:
  --report PATH       Write the machine-readable JSON run report
  --workers N         Outer spatial workers (default: available CPUs)
  --node-count N      Analyze the first N nodes
  --nodes I,J,...     Analyze explicit zero-based node indices
  --constituents LIST Comma-separated names or 'auto' (default: M2,S2,N2,K1,O1)
  --rayleigh-min X    Automatic selection criterion (default with auto: 1.0)
  --confidence MODE   Confidence intervals: none or linear (default: none)
  --white-noise       Use white noise instead of colored residual bands
  --overwrite         Replace existing output and report files
  -h, --help          Show this help
";

enum Command {
    Help,
    Version,
    Analyze(AnalyzeConfig),
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
        Command::Analyze(config) => match analyze_scalar(&config) {
            Ok(report) => match serde_json::to_string_pretty(&report) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("error: could not serialize completed run report: {error}");
                    ExitCode::FAILURE
                }
            },
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
    }
}

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
    if command != "analyze-scalar" {
        return Err(format!("unknown command: {}", command.to_string_lossy()));
    }

    let mut input = None;
    let mut output = None;
    let mut report = None;
    let mut workers = std::thread::available_parallelism().map_or(1, usize::from);
    let mut selection = None;
    let mut constituents = None;
    let mut rayleigh_minimum = None;
    let mut confidence_requested = None;
    let mut white_noise = false;
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
                ensure_selection_is_unset(selection.as_ref())?;
                let count = parse_positive_usize(&required_value(&mut arguments, option)?, option)?;
                selection = Some(NodeSelection::Prefix(count));
            }
            "--nodes" => {
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
            "--confidence" => {
                if confidence_requested.is_some() {
                    return Err("--confidence may only be supplied once".to_owned());
                }
                let value = required_value(&mut arguments, option)?;
                confidence_requested = Some(parse_confidence(&value)?);
            }
            "--white-noise" => white_noise = true,
            "--overwrite" => overwrite = true,
            "--help" | "-h" => return Ok(Command::Help),
            _ => return Err(format!("unknown option for analyze-scalar: {option}")),
        }
    }

    let constituent_selection = resolve_constituent_selection(constituents, rayleigh_minimum)?;
    let confidence_interval = resolve_confidence_interval(confidence_requested, white_noise)?;

    Ok(Command::Analyze(AnalyzeConfig {
        input: input.ok_or_else(|| "analyze-scalar requires --input PATH".to_owned())?,
        output: output.ok_or_else(|| "analyze-scalar requires --output PATH".to_owned())?,
        report,
        nodes: selection.unwrap_or(NodeSelection::All),
        constituent_selection,
        confidence_interval,
        workers,
        overwrite,
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

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{Command, parse_arguments};
    use rutide_cli::{
        ConfidenceInterval, ConstituentSelection, DEFAULT_CONSTITUENTS, NodeSelection,
    };
    use rutide_core::{LinearConfidence, TidalConstituent};

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
        let Command::Analyze(config) = command else {
            panic!("expected analyze command");
        };
        assert_eq!(config.nodes, NodeSelection::Indices(vec![4, 1, 3]));
        assert_eq!(
            config.constituent_selection,
            ConstituentSelection::Explicit(DEFAULT_CONSTITUENTS.to_vec())
        );
        assert_eq!(config.confidence_interval, ConfidenceInterval::None);
        assert_eq!(config.workers, 8);
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
        let Command::Analyze(config) = command else {
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
        let Command::Analyze(config) = command else {
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
            let Command::Analyze(config) = command else {
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
}
