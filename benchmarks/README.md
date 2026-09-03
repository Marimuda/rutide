# Benchmark infrastructure

This directory contains the reproducible Python reference harness and small,
versioned fixture manifests. Raw FVCOM files and generated benchmark results stay
outside source control.

## Python oracle

The environment is locked with `uv`. The local `UTide/` checkout is intentionally
not an environment dependency: the runner verifies its Git revision and clean
state, puts that exact checkout first on `sys.path`, and rejects an installed copy
from elsewhere. The repository's RUTide package is an editable local dependency,
so the same environment can benchmark both public Python interfaces.

```console
uv sync --project benchmarks/python --locked
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.fixture
uv run --project benchmarks/python --locked \
  python -m rutide_baseline \
  --workload smoke \
  --profile full-compatible \
  --repetitions 5

# Depth-averaged current ellipses, fixed five-constituent parity profile.
uv run --project benchmarks/python --locked \
  python -m rutide_baseline \
  --field vector \
  --workload vector-full \
  --profile fixed-constituents \
  --mode multiprocessing --workers 32 --chunk-size 32 \
  --repetitions 5
```

Run these commands from the repository root. Generated run manifests are written
under `benchmark-results/`, which is ignored by Git.

The fixture inspector and runner reconstruct FVCOM time as
`Itime + Itime2 / 86_400_000`. The file's `float32 time` field does not retain
exact hourly intervals at its Modified Julian Date magnitude and must not be used
as the analysis time vector.

## Installed-package real-data acceptance

After building and installing a release wheel, exercise its public batch,
reconstruction, and persistence endpoints on both a genuine ADCP observation and
the largest FVCOM fixture:

```console
python -m rutide_baseline.real_data_acceptance \
  --adcp /path/to/OS_CCE1_11_D_ADCP.nc \
  --fvcom ../projects/fvcom/claude_scratchpad/baroclinic_vikc1701/run/frs2f_0001.nc \
  --fvcom-series 4096 --workers 16 --expected-version 0.2.0 \
  --output benchmark-results/real-data-acceptance.json
```

The ADCP file is the freely available NOAA/NDBC OceanSITES CCE1 record. The
harness analyzes all sufficiently populated depth cells, while the FVCOM
selection deterministically spans the complete element axis without loading its
three-dimensional native-layer currents. Classic-NetCDF current variables are
read as bounded contiguous time slabs before sparse columns are retained; this
avoids millions of scalar-shaped reads across record storage. Raw external data
remain outside Git; the report retains provenance, source identity, separate
input/solve/reconstruction timings, result digests, and a bitwise persistence/
reconstruction check.

## Workloads

- `smoke`: one deterministic node;
- `correctness`: the 32 scalar nodes frozen in the fixture manifest;
- `scaling`: a deterministic prefix selected with `--series-count`; and
- `scalar-full`: all elevation nodes.
- `vector-full`: all depth-averaged current elements when `--field vector` is
  selected.

`canonical` executes the one-dimensional Python UTide API serially.
`multiprocessing` uses a Linux `fork` process pool and limits BLAS threads in each
worker. Both modes digest the same canonical result schema, allowing their outputs
to be compared without storing every coefficient in raw benchmark artifacts.
Vector mode reads `ua(time, nele)`, `va(time, nele)`, and `latc(nele)`, applies a
joint finite-value mask, and digests the ellipse parameters returned by the same
pinned oracle.

Compare a RUTide vector output with the oracle using:

```console
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.compare_vector \
  --rust-output benchmark-results/rust-vector-correctness-32.nc
```

The comparator detects schema-v13 `vertical_mode`. For `sigma-layer` output it
expands each selected `(siglay, element)` coordinate in layer-major order and
compares it with the matching `u[:, layer, element]` / `v[:, layer, element]`
Python UTide fit. For `fixed-depth`, it independently applies the documented
triangle-centroid, moving-free-surface interpolation and wet/dry mask before
calling Python UTide. Depth-averaged output continues to use `ua` / `va`.

Benchmark scalar or vector colored confidence on the shared deterministic
irregular/gappy fixture with the dedicated Rust and pinned-Python probes:

```console
RUTIDE_BENCH_FIELD=scalar RUTIDE_BENCH_SERIES=100 \
  cargo bench -p rutide-core --bench irregular_confidence_throughput
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.irregular_benchmark \
  --field scalar --series-count 100
```

Record the worker count explicitly for both probes. With `--workers 1`, the
Python probe is canonical single-process UTide with BLAS limited to one thread.
Larger worker counts retain a Linux `fork` pool and still call the existing
one-series API once per series, matching the Rust batch worker-count comparison.

Benchmark the installed public bindings directly—including object construction
and Python/native boundary costs—with one matched command:

```console
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.binding_benchmark \
  --field scalar --sampling regular --profile ols \
  --samples 745 --series-count 100 --workers 16 \
  --output benchmark-results/python-bindings-scalar-ols.json

uv run --project benchmarks/python --locked \
  python -m rutide_baseline.binding_benchmark \
  --field vector --sampling irregular --profile linear-colored \
  --samples 745 --series-count 100 --workers 16 \
  --output benchmark-results/python-bindings-vector-lomb.json
```

Each run times UTide's one-series loop, RUTide's one-series loop, and RUTide's
time-major native batch for both solve and reconstruction. Inputs, options, and
one-thread BLAS are matched. The JSON report retains every repetition, three
pairwise speedups, software/hardware identity, output digests, and maximum
cross-language/batch numerical errors. Profiles `ols`, `linear-colored`, and
`robust-colored` separate the major computational regimes; irregular colored
profiles exercise the Lomb–Scargle path.

Benchmark robust Cauchy IRLS plus regular colored confidence on the matched
outlier fixtures with:

```console
RUTIDE_BENCH_FIELD=scalar RUTIDE_BENCH_SERIES=100 \
  cargo bench -p rutide-core --bench robust_throughput
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.robust_benchmark \
  --field scalar --series-count 100 --workers 1
```

Both probes print the CI checksum and IRLS iteration sum, minimum, mean, and
maximum. Use the same field, series count, warm-up count, repetitions, and worker
count in comparisons. The Python probe exposes `coef.rf.iterations` from the
pinned solver rather than inferring work from elapsed time.

The irregular and robust probes also accept an explicit Monte Carlo mode while
retaining their historical linear-confidence defaults. Both implementations use
200 realizations. Python UTide currently ignores configurable `MC_n`, so this
comparison deliberately fixes the oracle's effective realization count:

```console
RUTIDE_BENCH_CONFIDENCE=monte-carlo \
  cargo bench -p rutide-core --bench irregular_confidence_throughput
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.irregular_benchmark --confidence monte-carlo

RUTIDE_BENCH_CONFIDENCE=monte-carlo \
  cargo bench -p rutide-core --bench robust_throughput
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.robust_benchmark --confidence monte-carlo
```

Benchmark exact or approximate inferred-constituent OLS with colored confidence
on either the regular FFT route or the shared-mask irregular Lomb–Scargle route:

```console
RUTIDE_BENCH_FIELD=vector RUTIDE_BENCH_SAMPLING=irregular \
  RUTIDE_BENCH_INFERENCE_MODE=exact RUTIDE_BENCH_SERIES=100 \
  cargo bench -p rutide-core --bench inference_throughput
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.inference_benchmark \
  --field vector --sampling irregular --inference-mode exact \
  --series-count 100 --workers 1
```

The scalar probe constrains S2 from M2 and O1 from K1. Vector mode applies the
same relationships with independent positive/negative rotary ratios. Both sum
all reported amplitude or semi-major confidence intervals as a cross-language
checksum; preparation remains outside the repeated solve-only timing.

The inference probe defaults to linear confidence for the matched Python
comparison. Its Rust-only Monte Carlo extension can be measured with an
effective realization count and deterministic seed:

```console
RUTIDE_BENCH_CONFIDENCE=monte-carlo \
  RUTIDE_BENCH_MC_REALIZATIONS=200 RUTIDE_BENCH_MC_SEED=0 \
  cargo bench -p rutide-core --bench inference_throughput
```

There is no paired Python command for this profile because Python UTide raises
`NotImplementedError` when inference and Monte Carlo confidence are combined.

Use the same harness with an explicit outlier pair to benchmark robust coupled-
vector inference. The default OLS profile and its historical checksums are
unchanged:

```console
RUTIDE_BENCH_FIELD=vector RUTIDE_BENCH_SAMPLING=irregular \
  RUTIDE_BENCH_INFERENCE_MODE=exact RUTIDE_BENCH_METHOD=robust \
  RUTIDE_BENCH_SERIES=100 \
  cargo bench -p rutide-core --bench inference_throughput
uv run --project benchmarks/python --locked \
  python -m rutide_baseline.inference_benchmark \
  --field vector --sampling irregular --inference-mode exact --method robust \
  --series-count 100 --workers 1
```

The `fixed-raw` solver profile is the first Rust parity target. It fits M2, S2,
N2, K1, and O1 with ordinary least squares, a mean and trend, raw phase, no nodal
corrections, and no confidence intervals. This deliberately isolates harmonic
basis construction and least squares before the Greenwich/nodal machinery is
ported.

Run the reusable fixed-design and corrected-basis microbenchmarks with explicit
worker, series, warm-up, and repetition settings:

```console
RUSTFLAGS="-C target-cpu=native" \
  RUTIDE_BENCH_SERIES=10000 RUTIDE_BENCH_WORKERS=1 \
  RUTIDE_BENCH_WARMUPS=5 RUTIDE_BENCH_REPETITIONS=5 \
  cargo bench -p rutide-core --bench fixed_raw_throughput

RUSTFLAGS="-C target-cpu=native" \
  RUTIDE_BENCH_SERIES=10000 RUTIDE_BENCH_WORKERS=64 \
  RUTIDE_BENCH_PHASE=greenwich RUTIDE_BENCH_NODAL=exact \
  RUTIDE_BENCH_WARMUPS=2 RUTIDE_BENCH_REPETITIONS=5 \
  cargo bench -p rutide-core --bench greenwich_nodal_throughput
```

`RUTIDE_BENCH_PHASE` accepts `greenwich`, `linear-time`, or `raw`;
`RUTIDE_BENCH_NODAL` accepts `exact`, `linear-time`, or `disabled`. These knobs
isolate correction costs without changing the fixture. The fixed-raw probe also
accepts `RUTIDE_BENCH_WARMUPS=0` for measuring the lazy projection's cold first
call. Retained before/after results and memory bounds are recorded in
`benchmarks/results/compute-kernel-optimization-2026-09-03.md`.

The `fixed-constituents` profile is the next Rust parity target. It keeps the same
five constituents and OLS configuration while enabling exact Greenwich phase and
nodal/satellite corrections. The corresponding Rust batch path shares only the
time-dependent terms; it still constructs a distinct design and factorization for
each latitude.
