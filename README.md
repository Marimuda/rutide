# RUTide

[![CI](https://github.com/Marimuda/rutide/actions/workflows/ci.yml/badge.svg)](https://github.com/Marimuda/rutide/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rutide.svg)](https://pypi.org/project/rutide/)
[![Python](https://img.shields.io/pypi/pyversions/rutide.svg)](https://pypi.org/project/rutide/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Marimuda/rutide/blob/main/LICENSE)

RUTide is a high-performance tidal harmonic-analysis library written in Rust,
with a friendly Python API and a bounded-memory FVCOM NetCDF application. It
fits and reconstructs scalar elevations and vector-current ellipses for station,
ADCP, and model data.

Use RUTide when you need Python-UTide-style analysis with native batch
parallelism, irregular and missing observations, robust fitting, uncertainty,
and evidence that the selected constituent set is actually identifiable.

## Choose an interface

| Need | Interface |
|---|---|
| One station, one ADCP bin, or in-memory NumPy arrays | Python `solve` / `reconstruct` |
| Many stations, bins, nodes, or elements with a shared time axis | Python `solve_many` / `reconstruct_many` |
| Large FVCOM NetCDF files without loading the full field into Python | Rust `rutide analyze-scalar` / `analyze-vector` application |
| A reusable low-level numerical engine | `rutide-core` Rust crate from this repository |

The Python package is the installable public product in 0.3.0. The FVCOM
application and Rust crates are currently built from source; they are not yet
published to crates.io.

## Install

RUTide supports CPython 3.9 and newer on Linux x86-64/AArch64, macOS
x86-64/Apple Silicon, and Windows x86-64.

```console
python -m pip install rutide
python -c "import rutide; print(rutide.__version__)"
```

Binary wheels contain the Rust core. A source build additionally needs a Rust
toolchain. NumPy is the only runtime Python dependency.

## Five-minute Python example

The API is deliberately familiar to Python UTide users. Datetime arrays are
accepted directly; numeric time defaults to Modified Julian Days (MJD).

```python
import numpy as np
import rutide

# A copyable 90-day example; replace elevation with your one-dimensional series.
hours = np.arange(24 * 90)
time = np.datetime64("2024-01-01") + hours * np.timedelta64(1, "h")
rng = np.random.default_rng(7)
elevation = (
    1.2 * np.cos(2 * np.pi * hours / 12.4206012 - 0.4)
    + 0.35 * np.cos(2 * np.pi * hours / 23.9344721 + 0.7)
    + 0.03 * rng.standard_normal(hours.size)
)

coef = rutide.solve(
    time,
    elevation,
    lat=62.0,
    constit=["M2", "S2", "N2", "K1", "O1"],
    method="ols",
    conf_int="linear",
    diagnostics=True,
)

tide = rutide.reconstruct(time, coef, min_SNR=2.0)

print(coef.name)              # constituent names
print(coef.A, coef.g)         # amplitude and Greenwich phase
print(coef.A_ci, coef.SNR)    # 95% amplitude CI and signal-to-noise ratio
print(coef.diagnostic_table())
print(tide.h)
```

For eastward/northward currents, pass both components. RUTide jointly masks the
components and returns tidal-ellipse parameters.

```python
coef = rutide.solve(
    time,
    eastward_velocity,
    northward_velocity,
    lat=62.0,
    constit=["M2", "S2", "N2", "K1", "O1"],
    conf_int="linear",
)
current = rutide.reconstruct(time, coef, min_SNR=2.0)

print(coef.Lsmaj, coef.Lsmin)  # semi-major and signed semi-minor axes
print(coef.theta, coef.g)      # inclination and phase in degrees
print(current.u, current.v)
```

For a time-major `(time, series)` matrix, use the native batch endpoint instead
of a Python loop:

```python
coef = rutide.solve_many(
    time,
    values,
    lat=latitude_by_series,
    constit=["M2", "S2", "N2", "K1", "O1"],
    conf_int="linear",
    workers=16,
    memory_limit_mb=512,
)
tide = rutide.reconstruct_many(time, coef)
```

Batch coefficient arrays have shape `(series, constituent)`. Pass eastward and
northward matrices for vector currents. Fitted coefficients can be saved to a
versioned, pickle-free archive and reconstructed later without refitting:

```python
coef.save("analysis.rutide.npz")
restored = rutide.load("analysis.rutide.npz")
```

See the complete [Python API guide](https://github.com/Marimuda/rutide/blob/main/PYTHON_API.md)
for inference, Monte Carlo confidence, robust fitting, pre-filter correction,
epochs, ordering, xarray conversion, and persistence.

## FVCOM quick start

The FVCOM application reads large fields in bounded spatial chunks and writes
versioned NetCDF coefficients, optional reconstruction, sampling diagnostics,
provenance, and a machine-readable JSON run report.

Install Rust 1.98 and the NetCDF C development library, then build the optimized
binary from a source checkout:

```console
git clone https://github.com/Marimuda/rutide.git
cd rutide
cargo build --release --locked --bin rutide
./target/release/rutide --version
```

Analyze sea-surface elevation at FVCOM nodes:

```console
./target/release/rutide analyze-scalar \
  --input /path/to/fvcom.nc \
  --output elevation-harmonics.nc \
  --report elevation-run.json \
  --constituents auto \
  --confidence linear \
  --constituent-diagnostics \
  --reconstruct
```

Analyze depth-averaged `ua`/`va` currents at elements:

```console
./target/release/rutide analyze-vector \
  --input /path/to/fvcom.nc \
  --output current-ellipses.nc \
  --report current-run.json \
  --constituents M2,S2,N2,K1,O1 \
  --confidence linear \
  --constituent-diagnostics \
  --reconstruct
```

Native sigma layers use `--layers all` or `--layers 0,4,9`; fixed physical
depths use `--depths 5,20,50`. Omit `--chunk-series` for the recommended
automatic memory bound. Existing destinations are preserved unless
`--overwrite` is explicit, and output is installed atomically after a successful
analysis.

Read the [FVCOM application guide](https://github.com/Marimuda/rutide/blob/main/FVCOM_CLI.md)
before a whole-domain run. It documents required variables, vertical semantics,
selection, confidence, inference, robust fitting, output schemas, and memory
controls.

## What is supported

- 146-constituent catalog with exact Greenwich, nodal, satellite, and
  shallow-water corrections.
- Explicit or Rayleigh-selected constituents, plus exact or approximate
  scalar/vector inference.
- Ordinary least squares and nine robust IRLS weight functions.
- Mean with optional trend, missing values, and strictly increasing regular,
  gappy, or irregular timestamps.
- White or colored linear and Monte Carlo confidence intervals. Colored noise
  uses FFT spectra for regular grids and Lomb-Scargle spectra for truly irregular
  records.
- Scalar amplitudes/phases and vector tidal ellipses, PE, SNR, arbitrary-time
  reconstruction, and constituent filtering.
- Codiga-style RR, RNM, Corrmax, condition-number, and tidal-variance diagnostics
  for defensible constituent selection.
- Optional correction for a known real pre-processing filter response.
- Bounded, deterministic batch execution and atomic coefficient persistence.
- FVCOM elevation, depth-averaged current, native sigma-layer current, and
  fixed-physical-depth current products.

The complete option cross-product and intentional Python differences are
versioned in the [compatibility contract](https://github.com/Marimuda/rutide/blob/main/COMPATIBILITY.md).
RUTide is UTide-inspired, not a byte-for-byte drop-in replacement.

## Scientific-use checklist

A successful least-squares solve is not proof that a tidal model is suitable.
Before reporting results:

1. Verify time epoch, latitude, units, component orientation, vertical meaning,
   missing-data handling, and any upstream filtering.
2. Inspect retained duration, gaps, spectral-band coverage, and the FFT or
   Lomb-Scargle confidence route.
3. Review RR/RNM, adjacent-parameter correlation, whole-model conditioning,
   SNR, PE, and tidal variance together.
4. Inspect residuals and robust weights, then validate reconstruction on held-out
   data where possible.
5. Archive the RUTide version, options, constituent/inference lists, source
   identity, output schema, and result digest.

The [FVCOM and ADCP quality-control guide](https://github.com/Marimuda/rutide/blob/main/QUALITY_CONTROL.md)
explains these checks and the metadata needed for reproducible scientific use.

## Current boundaries

- RUTide does not automatically apply manufacturer-specific ADCP QC. Prepare
  instrument data before fitting and retain the deployment metadata.
- The FVCOM output retains source indices but is not a spatial mesh/current-atlas
  package. Spatial coefficient interpolation and OpenDrift integration are a
  separate future product.
- The linear trend is a local fitted term, not a safe long-range forecast.
- A known filter response can be corrected; a guessed response cannot recover
  information removed by undocumented preprocessing.
- The API is pre-1.0. Minor releases may contain documented breaking changes;
  patch releases preserve published 0.3 contracts.

## Performance

On the documented full-field FVCOM fixture and benchmark host, the bounded input
pipeline completed scalar analysis in 0.91 seconds and depth-averaged vector
analysis in 2.50 seconds—approximately 71.1x and 50.8x faster than the retained
tuned Python UTide processes. Irregular, robust, inference, Monte Carlo, and
installed-Python-binding profiles are measured separately.

These are reproducible measurements, not universal speed promises. Record size,
constituents, confidence method, missing masks, storage, CPU, and worker count all
matter. See the [benchmark plan](https://github.com/Marimuda/rutide/blob/main/BENCHMARK_PLAN.md)
and [versioned results](https://github.com/Marimuda/rutide/tree/main/benchmarks/results)
for commands, correctness tolerances, hardware, and scope.

## Documentation

| Guide | Purpose |
|---|---|
| [Python API](https://github.com/Marimuda/rutide/blob/main/PYTHON_API.md) | Complete endpoint, option, time, and persistence contract |
| [FVCOM application](https://github.com/Marimuda/rutide/blob/main/FVCOM_CLI.md) | NetCDF inputs, command profiles, outputs, and resource behavior |
| [Quality control](https://github.com/Marimuda/rutide/blob/main/QUALITY_CONTROL.md) | Scientific review sequence for FVCOM and ADCP products |
| [Constituent diagnostics](https://github.com/Marimuda/rutide/blob/main/DIAGNOSTICS.md) | RR, RNM, Corrmax, conditioning, and tidal variance |
| [Pre-filter correction](https://github.com/Marimuda/rutide/blob/main/PREFILTER_TRANSFER.md) | Known transfer-response semantics and safeguards |
| [Fixed-depth currents](https://github.com/Marimuda/rutide/blob/main/FIXED_DEPTH_INTERPOLATION.md) | FVCOM vertical interpolation and wet/dry contract |
| [Compatibility](https://github.com/Marimuda/rutide/blob/main/COMPATIBILITY.md) | Stable surfaces, schemas, and Python UTide differences |
| [Changelog](https://github.com/Marimuda/rutide/blob/main/CHANGELOG.md) | Release history and migration notes |
| [Roadmap](https://github.com/Marimuda/rutide/blob/main/ROADMAP.md) | Completed work and deliberately separate future products |

## Development and support

Bug reports, reproducible scientific discrepancies, and focused feature requests
are welcome in [GitHub Issues](https://github.com/Marimuda/rutide/issues). Please
include the RUTide version, platform, analysis options, time convention, array
shapes, and the smallest shareable reproducer. Do not attach restricted FVCOM or
instrument datasets.

Development requires the pinned Rust toolchain, NetCDF C library, and `uv`.
Run the local gates from the repository root:

```console
cargo fmt --all -- --check
cargo ci
cargo test-all
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --locked
uv sync --locked
uv run --locked ruff format --check python
uv run --locked ruff check python
uv run --locked python -m unittest discover -s python/tests -v
```

See [CONTRIBUTING.md](https://github.com/Marimuda/rutide/blob/main/CONTRIBUTING.md)
for scientific and performance evidence requirements.

## Citation and license

Use the repository's [`CITATION.cff`](https://github.com/Marimuda/rutide/blob/main/CITATION.cff)
entry to cite the software and include the exact release version in methods and
archived analysis metadata.

RUTide is licensed under the [MIT License](https://github.com/Marimuda/rutide/blob/main/LICENSE).
