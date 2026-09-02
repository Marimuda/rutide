# Changelog

All notable changes to RUTide are recorded here. The project follows Semantic
Versioning while the Rust and Python APIs remain pre-1.0: minor releases may
contain documented breaking API changes, while patch releases preserve the
published API and file-schema contracts.

## Unreleased

## 0.2.0 - 2026-09-02

### Added

- Broad scalar and vector Python-UTide-compatible harmonic analysis, including
  dynamic constituents, exact and approximate inference, robust fitting,
  linear and Monte Carlo confidence, diagnostics, and reconstruction.
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
