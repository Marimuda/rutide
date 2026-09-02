# Compatibility and stability contracts

RUTide is currently version `0.1.0`. The numerical and file-format contracts are
explicitly versioned, but the Rust crates remain pre-1.0 and are not published to
crates.io yet. This document defines what downstream users can rely on while the
remaining product work is completed.

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

## NetCDF and JSON reports

The compiled schema constants are public as
`rutide_cli::SCALAR_OUTPUT_SCHEMA_VERSION` and
`rutide_cli::VECTOR_OUTPUT_SCHEMA_VERSION`. They currently identify scalar v15
and vector v12.

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

Scalar fields use a public `series` axis. Depth-averaged vector fields also use
`series`; native sigma-layer vector fields use `siglay, element` and retain the
requested source indices in coordinate variables. Native sigma layers are model
coordinates, not fixed physical depths.

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
[`compatibility/feature-matrix-v1.json`](compatibility/feature-matrix-v1.json),
validated against
[`compatibility/feature-matrix.schema.json`](compatibility/feature-matrix.schema.json)
and checked against compiled constants in the Rust test suite.
