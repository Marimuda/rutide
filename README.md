# RUTide

RUTide is an experimental, performance-oriented Rust implementation of the
harmonic-analysis functionality needed from Python UTide for large FVCOM model
outputs.

The project is currently a feasibility exercise. Scientific compatibility comes
before speed, and measured results will determine whether this becomes a complete
rewrite, a smaller native kernel used from Python, or no rewrite at all. See the
[benchmark and decision plan](BENCHMARK_PLAN.md) for the fixture, comparison
protocol, correctness contract, and decision gates.

The first complete scalar application measurement passes the provisional gate:
3.15 seconds median whole-process wall time for Rust versus 64.69 seconds for the
32-process Python baseline on all 75,160 nodes. See the
[application benchmark snapshot](benchmarks/results/fvcom-scalar-application-2026-08-31.md)
for the exact profile, correctness errors, worker scaling, resource caveats, and
scope limits.

The subsequent [memory optimization](benchmarks/results/memory-optimization-2026-08-31.md)
reduced 64-worker peak RSS from 5.40 GiB to 0.690 GiB and whole-process wall time
to 1.51 seconds without changing either correctness digest.

The implemented scalar kernels now cover fixed-constituent OLS with mean and
trend in both raw-phase mode and exact Greenwich/nodal mode. The exact-correction
catalog contains all 146 constituents, 162 satellite corrections, and 251
shallow-water relationships from the pinned Python oracle. The corrected bulk
API shares latitude-independent astronomy, pre-aggregates satellite terms, and
parallelizes the latitude-specific factorizations across spatial series. Both
paths have real-FVCOM parity tests against the pinned Python oracle.

Percent-energy diagnostics, linearized 95% amplitude/phase confidence intervals,
CI-derived signal-to-noise ratio, per-solution PE/SNR ranking, and exact scalar
reconstruction are available. Reconstruction supports arbitrary target times in
the core API and an opt-in complete original-time series in the FVCOM command.
Scalar `_FillValue` and `NaN` observations are omitted per series, with shared
valid-time masks grouped before fitting. Vector currents are not implemented yet.
The command-line application can analyze an FVCOM `zeta(time, node)` field with
an explicit or Rayleigh-selected constituent list and serialize its coefficients,
diagnostics, observation counts, per-series reference epochs, and optional
reconstruction to NetCDF.

## Repository layout

```text
crates/rutide-core/  Numerical library; no file-format or CLI concerns
crates/rutide-cli/   FVCOM NetCDF application and benchmark entry point
benchmarks/           Locked Python oracle harness and fixture manifests
BENCHMARK_PLAN.md    Frozen experimental intent and measurement protocol
```

The local `UTide/` checkout is deliberately ignored. It is the pinned Python
behavioral oracle described in the benchmark plan, not vendored RUTide source.
Large FVCOM data and generated benchmark results also remain outside Git.

## Development

Install [rustup](https://rustup.rs/) and the NetCDF C development library, then
run commands from the repository root. The checked-in toolchain file selects
Rust 1.98.0 with rustfmt and Clippy.

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

## FVCOM scalar analysis

The current application profile is deliberately frozen: M2, S2, N2, K1, and O1;
ordinary least squares; mean and trend; exact Greenwich phase and nodal
corrections; and no confidence intervals. Run it with:

```console
cargo run --release --bin rutide -- analyze-scalar \
  --input /path/to/fvcom.nc \
  --output coefficients.nc \
  --report run.json \
  --workers 64
```

Supply any unique catalog names in output order with, for example,
`--constituents Q1,O1,K1,M2,S2,K2,M4`. The default remains the frozen five-name
benchmark profile shown above. Use `--constituents auto` for UTide-compatible
record-length selection; its Rayleigh criterion defaults to `1.0` and can be
changed with `--rayleigh-min X`.

Add `--confidence linear` for UTide-compatible colored-noise linear confidence
intervals and SNR. `--white-noise` selects the white residual-noise alternative.
Colored confidence currently requires equidistant timestamps; the white model
also supports irregular timestamps. Missing observations on an originally
equidistant grid remain supported for colored confidence by linearly
interpolating fitted residuals onto the full grid before the FFT, matching the
pinned Python behavior. Truly irregular colored spectra are rejected explicitly
until a Lomb–Scargle implementation is added.

Add `--reconstruct` to write `reconstruction(time, series)` at every original
FVCOM timestamp. With no filter it includes every fitted constituent. Use
`--reconstruct-constituents M2,S2,K1` for an explicit subset, or diagnostic
thresholds such as `--min-pe 1 --min-snr 2`. PE and SNR thresholds are inclusive
and combine with logical AND; explicit names are the alternative selection mode,
matching Python UTide. `--min-snr` requires `--confidence linear`, while PE-only
filtering does not. For example:

```console
cargo run --release --bin rutide -- analyze-scalar \
  --input /path/to/fvcom.nc \
  --output coefficients-and-tide.nc \
  --constituents auto \
  --confidence linear \
  --reconstruct --min-pe 1 --min-snr 2 \
  --workers 64
```

The library-level `GreenwichNodalReconstructor` and model convenience method
accept arbitrary finite Modified Julian Days, including held-out and forecast
times, and always retain the fitted mean and trend.

Use `--node-count N` for a prefix or `--nodes 0,10,20` for an explicit
correctness sample. Existing destinations are preserved unless `--overwrite`
is supplied. The source is opened read-only, all coefficients are written to a
temporary sibling file before installation, and the JSON report records
per-stage timings and a canonical result digest.

The pinned Python comparison command checks every series in a Rust output and
returns nonzero if any frozen tolerance is exceeded:

```console
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.compare --rust-output coefficients.nc
```

Each series must retain enough finite observations to overdetermine its selected
model. Infinite observations remain invalid; missing samples are reported rather
than silently replaced in the fit.

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
