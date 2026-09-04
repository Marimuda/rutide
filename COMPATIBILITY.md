# Compatibility and stability contracts

The current source tree is the RUTide `0.3.0` release line. The numerical and
file-format contracts are explicitly versioned, but the Rust crates remain
pre-1.0 and are not published to crates.io yet. This document defines what
downstream users can rely on in the 0.3 release line.

## Rust API

`rutide-core` is the dependency-light numerical API. Public types re-exported
from its crate root are the candidate stable surface. Within the `0.x` series,
breaking changes require a workspace minor-version increase and a migration note;
renames should use a deprecation cycle when practical. Patch releases may add
methods and fix behavior that contradicts documented invariants or the pinned
oracle.

`rutide-cli` exposes reusable FVCOM application configuration and report types,
but its structs may still gain fields before 1.0 as output strategies and product
interfaces settle. The command-line spelling and defaults are treated as stable
within a schema version. Both crates require Rust 1.98 and forbid unsafe code.
The workspace and Python distribution share one Cargo-sourced version.

## NetCDF and JSON reports

The compiled schema constants are public as
`rutide_cli::SCALAR_OUTPUT_SCHEMA_VERSION` and
`rutide_cli::VECTOR_OUTPUT_SCHEMA_VERSION`. They currently identify scalar v18
and vector v17. Scalar v17 and vector v15 added opt-in constituent-identifiability
configuration and complete per-series/per-constituent diagnostic fields. Vector
v16 added the four Cartesian eastward/northward cosine/sine coefficient fields.
Scalar v18 and vector v17 add complete preprocessing-filter response provenance,
source identity, and propagated source units. Declared vector-component units
must match because ellipse coefficients require a shared physical scale. When
extended diagnostics and pre-filtering are disabled, the application avoids
their work.

A schema version covers dimension names and order, required variables, variable
meaning and units, enumerated flag values, required global attributes, and JSON
report field meaning. The JSON `schema_version` and NetCDF
`rutide_schema_version` for the same analysis kind always match. Any incompatible
change increments the corresponding integer. Readers must reject versions they
do not understand instead of guessing from variable presence.

Additive metadata may be introduced without invalidating an older schema only
when existing readers can ignore it safely. Removing or renaming data, changing
dimension order, changing units or semantics, or making an optional field
required always creates a new schema version.

The additive `input_pipeline` report field and NetCDF attribute identify
`overlapped` or `sequential` input. In overlapped mode, `input_seconds` measures
active reader work that can coincide with solve and result processing, so stage
durations are not additive; `total_seconds` remains elapsed time. The existing
`maximum_observation_buffer_bytes` field covers all concurrently resident
promoted input arrays.

Scalar fields use a public `series` axis. Depth-averaged vector fields also use
`series`; native sigma-layer vector fields use `siglay, element` and retain the
requested source indices in coordinate variables. Native sigma layers are model
coordinates, not fixed physical depths. Fixed-depth vector fields use
`depth, element`, with positive metres below the instantaneous free surface and
the interpolation/masking contract in `FIXED_DEPTH_INTERPOLATION.md`.

## Determinism and digests

For identical normalized inputs, analysis options, constituent order, and schema
profile, results and `result_sha256` are independent of worker scheduling and
observation chunk size. A digest is an integrity and reproducibility identifier,
not a permanent cross-schema content address: consumers must compare the schema
version and profile along with the digest.

## Python compatibility

The compatibility oracle is the clean Python UTide revision
`8fabe121752bc317931472a10a42e306715106de`. "Python oracle" in the feature
matrix means a pinned comparison exists and passes its documented tolerance.
"Rust extension" means the behavior is scientifically defined and tested but
Python UTide has no complete executable path for direct end-to-end parity.

The machine-readable status is
[`crates/rutide-cli/compatibility/feature-matrix-v1.json`](crates/rutide-cli/compatibility/feature-matrix-v1.json),
validated against
[`crates/rutide-cli/compatibility/feature-matrix.schema.json`](crates/rutide-cli/compatibility/feature-matrix.schema.json)
and checked against compiled constants in the Rust test suite.

Rows marked `planned` are intentionally outside the current supported contract.
The remaining planned row is the separate spatial current-atlas/OpenDrift product
layer. MATLAB's optional known pre-filter response correction is supported; its
scope and limits are recorded in
[`PREFILTER_TRANSFER.md`](PREFILTER_TRANSFER.md).

The `rutide` Python distribution supports CPython 3.9 and newer through a
PyO3 `abi3-py39` extension. Its `solve` and `reconstruct` endpoints are shaped
after Python UTide and expose UTide coefficient names plus descriptive aliases,
but are not represented as a byte-for-byte drop-in interface. The supported
arguments, time behavior, read-only result snapshots, and current serialization
limitations are defined in [`PYTHON_API.md`](PYTHON_API.md). The batch endpoint
uses stable `(series, constituent)` result axes and separate per-series
presentation maps; missing masks never reorder the constituent dimension.

The pickle-free Python coefficient archive container remains schema 1. RUTide
0.3 writes native coefficient snapshot schema 2 inside that container and reads
both native schemas 1 and 2; schema-1 snapshots restore with no extended
constituent diagnostics. Archives store fitted native state and the retained
timestamp/model recipe, but not the source observations. Arbitrary Python UTide
coefficient dictionaries remain outside this contract.

Within the pre-1.0 series, Python breaking changes require a workspace minor
version and changelog migration note. Patch releases preserve documented
endpoint names and semantics. The native fit retained by a `Coefficient` is the
authoritative immutable reconstruction state; displayed NumPy coefficient arrays
cannot be mutated independently of it.

## Solver-option composition

Every valid Cartesian combination of the documented scalar/vector solver axes is
supported. This includes regular, gappy-grid, and irregular sampling; explicit
or Rayleigh selection; no, exact, or approximate inference; OLS or any robust
weight; optional trend; every phase and nodal mode; no, linear, or Monte Carlo
confidence with white or colored noise; every presentation order; and every
reconstruction filter.

The versioned
[`crates/rutide-cli/compatibility/solver-option-matrix-v1.json`](crates/rutide-cli/compatibility/solver-option-matrix-v1.json)
is the normative list of axis values, explicit rejection constraints, Python
parity exceptions, and retained evidence. Its schema is
[`crates/rutide-cli/compatibility/solver-option-matrix.schema.json`](crates/rutide-cli/compatibility/solver-option-matrix.schema.json).
The matrix uses a supported-by-default composition rule: combinations are not
silently omitted, and invalid combinations such as SNR without confidence are
named and rejected during validation.

The fundamental trend, phase, nodal, scope, and inference-mode cross-product is
executed as a test. Dense seam tests additionally combine robust fitting,
irregular missing observations, colored confidence, inference, alternative
astronomical conventions, pre-filter correction, Monte Carlo propagation,
ordering, and reconstruction.
Where Python UTide has an executable path, values are frozen against the pinned
oracle; otherwise the matrix identifies the path as a tested Rust extension.
