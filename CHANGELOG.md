# Changelog

All notable changes to RUTide are recorded here. The project follows Semantic
Versioning while the Rust and Python APIs remain pre-1.0: minor releases may
contain documented breaking API changes, while patch releases preserve the
published API and file-schema contracts.

## Unreleased

### Added

- Public Codiga equation-81 adjacent-frequency Rayleigh and equations-99–102
  scalar/vector tidal-variance kernels as the first RUTide 0.3.0 constituent-
  identifiability increment, with explicit inferred-constituent and zero-variance
  behavior.
- Opt-in scalar and current diagnostic results containing RNM, maximum adjacent-
  parameter correlation, cached whole-basis `K`, `SNRallc`, `SNRallc / K`, and
  full/SNR-filtered tidal variance for raw, Greenwich/nodal, robust, and scalar-
  inference fits. Condition numbers and unweighted normal inverses reuse the
  prepared pivoted QR's small triangular factor.
- Coupled-vector inference diagnostics using the full four-by-four Cartesian
  cross-covariance, plus parallel post-fit diagnostic orchestration for complete
  and missing-value scalar/vector batches with retained-record timing.
- Opt-in `diagnostics=True` Python results with MATLAB-inspired short fields,
  descriptive aliases, dense batch arrays, and compact single-series tables.
- An installed-wheel acceptance harness for a public NOAA/NDBC ADCP profile and
  a domain-spanning selection from the 25.78 GB FVCOM fixture, including
  reconstruction and coefficient-persistence validation.
- A matched Rust/Python benchmark profile for robust coupled-vector inference
  with irregular sampling, colored confidence, joint missing values, and
  isolated outliers.

### Changed

- Added native coefficient snapshot schema 2 for optional single and dense-batch
  identifiability diagnostics while retaining schema-1 reads from RUTide 0.2.
- Coalesced schema-1 coefficient snapshot arrays into bounded typed NPZ blobs.
  Existing schema-1 archives remain readable while large batch archives save
  and load with far fewer ZIP entries.
- Removed per-timestamp corrected-basis allocations and polar round trips,
  cached bounded shared-mask astronomical bases, and precomputed reconstruction
  phasors. Exact corrected throughput improves by 1.53–3.92x in the retained
  10,000-series worker matrix.
- Applied a lazily cached QR-derived least-squares projection to reusable fixed
  raw batches of at least 16 series, improving the retained 10,000-series
  throughput by 1.87–2.21x while preserving the direct QR path for small calls.
- Overlapped bounded automatic FVCOM input with scalar and regular-vector solves
  through a single-owner NetCDF reader and a strict two-buffer rendezvous. Full-
  field scalar/vector process wall improves by 1.15x/1.41x at 64 workers without
  changing result digests or exceeding the existing 512 MiB logical input bound.

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
