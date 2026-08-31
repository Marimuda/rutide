# Benchmark infrastructure

This directory contains the reproducible Python reference harness and small,
versioned fixture manifests. Raw FVCOM files and generated benchmark results stay
outside source control.

## Python oracle

The environment is locked with `uv`. The local `UTide/` checkout is intentionally
not an environment dependency: the runner verifies its Git revision and clean
state, puts that exact checkout first on `sys.path`, and rejects an installed copy
from elsewhere.

```console
uv sync --project benchmarks/python --locked
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.fixture
uv run --project benchmarks/python --locked \
  python -m rutide_baseline \
  --workload smoke \
  --profile full-compatible \
  --repetitions 5
```

Run these commands from the repository root. Generated run manifests are written
under `benchmark-results/`, which is ignored by Git.

The fixture inspector and runner reconstruct FVCOM time as
`Itime + Itime2 / 86_400_000`. The file's `float32 time` field does not retain
exact hourly intervals at its Modified Julian Date magnitude and must not be used
as the analysis time vector.

## Workloads

- `smoke`: one deterministic node;
- `correctness`: the 32 scalar nodes frozen in the fixture manifest;
- `scaling`: a deterministic prefix selected with `--series-count`; and
- `scalar-full`: all elevation nodes.

`canonical` executes the one-dimensional Python UTide API serially.
`multiprocessing` uses a Linux `fork` process pool and limits BLAS threads in each
worker. Both modes digest the same canonical result schema, allowing their outputs
to be compared without storing every coefficient in raw benchmark artifacts.

The `fixed-raw` solver profile is the first Rust parity target. It fits M2, S2,
N2, K1, and O1 with ordinary least squares, a mean and trend, raw phase, no nodal
corrections, and no confidence intervals. This deliberately isolates harmonic
basis construction and least squares before the Greenwich/nodal machinery is
ported.

The `fixed-constituents` profile is the next Rust parity target. It keeps the same
five constituents and OLS configuration while enabling exact Greenwich phase and
nodal/satellite corrections. The corresponding Rust batch path shares only the
time-dependent terms; it still constructs a distinct design and factorization for
each latitude.
