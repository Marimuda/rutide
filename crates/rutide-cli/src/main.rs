//! `RUTide` command-line entry point.

use std::{env, process::ExitCode};

fn main() -> ExitCode {
    let arguments: Vec<_> = env::args_os().skip(1).collect();

    match arguments.as_slice() {
        [] => {
            println!(
                "RUTide {} is scaffolded; harmonic analysis is not implemented yet.",
                rutide_core::VERSION
            );
            ExitCode::SUCCESS
        }
        [argument] if argument == "--version" || argument == "-V" => {
            println!("rutide {}", rutide_core::VERSION);
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: rutide [--version]");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn core_version_is_available() {
        assert!(!rutide_core::VERSION.is_empty());
    }
}
