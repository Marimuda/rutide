# RUTide

RUTide is an experimental, performance-oriented Rust implementation of the
harmonic-analysis functionality needed from Python UTide for large FVCOM model
outputs.

The project is currently a feasibility exercise. Scientific compatibility comes
before speed, and measured results will determine whether this becomes a complete
rewrite, a smaller native kernel used from Python, or no rewrite at all. See the
[benchmark and decision plan](BENCHMARK_PLAN.md) for the fixture, comparison
protocol, correctness contract, and decision gates.

The implemented scalar kernels now cover fixed-constituent OLS with mean and
trend in both raw-phase mode and exact Greenwich/nodal mode. The corrected bulk
API shares latitude-independent astronomy, pre-aggregates satellite terms, and
parallelizes the latitude-specific factorizations across spatial series. Both
paths have real-FVCOM parity tests against the pinned Python oracle.

The current exact-correction catalog contains M2, S2, N2, K1, and O1. Automatic
selection, confidence intervals, missing observations, vector currents,
reconstruction, and end-to-end NetCDF processing are not implemented yet.

## Repository layout

```text
crates/rutide-core/  Numerical library; no file-format or CLI concerns
crates/rutide-cli/   Thin application and future NetCDF benchmark entry point
benchmarks/           Locked Python oracle harness and fixture manifests
BENCHMARK_PLAN.md    Frozen experimental intent and measurement protocol
```

The local `UTide/` checkout is deliberately ignored. It is the pinned Python
behavioral oracle described in the benchmark plan, not vendored RUTide source.
Large FVCOM data and generated benchmark results also remain outside Git.

## Development

Install [rustup](https://rustup.rs/) and run commands from the repository root.
The checked-in toolchain file selects Rust 1.98.0 with rustfmt and Clippy.

```console
cargo fmt --all -- --check
cargo ci
cargo test-all
cargo doc --workspace --all-features --no-deps --locked
cargo run --bin rutide -- --version
uv sync --project benchmarks/python --locked
uv run --project benchmarks/python --locked ruff check benchmarks/python
uv run --project benchmarks/python --locked \
  python -m unittest discover -s benchmarks/python/tests -v
```

`cargo ci` and `cargo test-all` are repository aliases defined in
`.cargo/config.toml`. CI runs the same gates on every push and pull request.

Performance measurements must use release or benchmark profiles. Machine-specific
compiler flags, CPU affinity, worker counts, and library thread settings belong in
the generated result manifest; they are intentionally not hidden in the default
Cargo configuration.

## Working principles

- Python UTide at the pinned revision is the initial compatibility oracle.
- Optimize only after a representative baseline and profiler evidence exist.
- Keep the numerical core independent of NetCDF and Python bindings.
- Preserve deterministic correctness fixtures and machine-readable benchmark
  results.
- Forbid unsafe Rust until a measured bottleneck and a documented safety argument
  justify reconsidering that policy.

## License

RUTide is licensed under the MIT License. See [LICENSE](LICENSE).
