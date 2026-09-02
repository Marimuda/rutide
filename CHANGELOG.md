# Changelog

All notable changes to RUTide are recorded here. The project follows Semantic
Versioning while the Rust and Python APIs remain pre-1.0: minor releases may
contain documented breaking API changes, while patch releases preserve the
published API and file-schema contracts.

## Unreleased

### Added

- An installed-wheel acceptance harness for a public NOAA/NDBC ADCP profile and
  a domain-spanning selection from the 25.78 GB FVCOM fixture, including
  reconstruction and coefficient-persistence validation.
- A matched Rust/Python benchmark profile for robust coupled-vector inference
  with irregular sampling, colored confidence, joint missing values, and
  isolated outliers.

### Changed

- Coalesced schema-1 coefficient snapshot arrays into bounded typed NPZ blobs.
  Existing schema-1 archives remain readable while large batch archives save
  and load with far fewer ZIP entries.

## 0.2.0 - 2026-09-02

### Added

- Broad scalar and vector Python-UTide-compatible harmonic analysis, including
  dynamic constituents, exact and approximate inference, robust fitting,
  linear and Monte Carlo confidence, diagnostics, and reconstruction.
- A typed UTide-inspired `rutide.solve` / `rutide.reconstruct` Python package,
  backed by a PyO3/NumPy stable-ABI extension with native model reuse and GIL-free
  solving and reconstruction.
- Time-major `solve_many` / `reconstruct_many` Python bindings with varying
  latitudes, joint vector masks, deterministic worker pools, bounded native
  chunks, stable coefficient axes, per-series rankings, and xarray conversion.
- Atomic, pickle-free schema-1 coefficient persistence for single and batch
  fits, including uncertainty, robust diagnostics, inference, and reusable
  native reconstruction state without source observations.
- A locked public-binding benchmark that compares Python UTide loops, RUTide
  one-series loops, and RUTide native batches for solve and reconstruction while
  enforcing numerical parity.
- Clean-environment wheel/source install smoke tests and separate protected
  TestPyPI/PyPI trusted-publishing gates.
- Irregular colored-confidence estimation through Lomb–Scargle residual spectra.
- Bounded and incremental FVCOM scalar, depth-averaged, sigma-layer, and
  fixed-physical-depth application workflows.
- Versioned JSON/NetCDF contracts, deterministic result digests, and a
  machine-readable solver-option compatibility matrix.

### Changed

- Prepared `rutide-core` and `rutide-cli` for registry packaging with explicit
  dependency versions and discoverability metadata.

## 0.1.0 - 2026-08-31

### Added

- Initial Rust harmonic-analysis kernel and FVCOM performance baseline.
