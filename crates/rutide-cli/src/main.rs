//! `RUTide` command-line entry point.

use std::{
    env,
    ffi::{OsStr, OsString},
    path::PathBuf,
    process::ExitCode,
};

use rutide_cli::{AnalyzeConfig, NodeSelection, analyze_scalar};

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
            "--overwrite" => overwrite = true,
            "--help" | "-h" => return Ok(Command::Help),
            _ => return Err(format!("unknown option for analyze-scalar: {option}")),
        }
    }

    Ok(Command::Analyze(AnalyzeConfig {
        input: input.ok_or_else(|| "analyze-scalar requires --input PATH".to_owned())?,
        output: output.ok_or_else(|| "analyze-scalar requires --output PATH".to_owned())?,
        report,
        nodes: selection.unwrap_or(NodeSelection::All),
        workers,
        overwrite,
    }))
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

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{Command, parse_arguments};
    use rutide_cli::NodeSelection;

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
}
