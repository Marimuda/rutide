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

The completed depth-averaged vector path also clears the gate: its
[whole-field snapshot](benchmarks/results/fvcom-vector-application-2026-09-01.md)
reports a 5.39-second Rust median versus 126.97 seconds for the 32-process Python
UTide baseline across all 144,860 current elements, a 23.6x process-boundary
speedup with Python-compatible ellipse and colored-CI correctness.

The subsequent [spatial-chunk snapshot](benchmarks/results/spatial-chunking-2026-09-02.md)
reduces the 64-worker vector peak RSS from 2.23 GiB to 0.94 GiB while retaining
the same canonical digest. Its 5.40-second median is 9.5% slower than the
one-chunk comparison and remains 23.5x faster than the previously measured
126.97-second Python baseline.

The focused [compute-kernel optimization](benchmarks/results/compute-kernel-optimization-2026-09-03.md)
removes measured allocation, repeated trigonometry, and wide-batch QR costs.
Exact corrected solves improve by 1.53–3.92x, fixed raw batches by 1.87–2.21x,
and installed-binding reconstruction by 1.29–1.90x. The snapshot also records
the current scaling ceiling and bounded cache costs.

The subsequent [bounded input pipeline](benchmarks/results/input-pipeline-2026-09-03.md)
overlaps one NetCDF read/conversion with the preceding chunk's solve while
retaining the 512 MiB logical input bound. Current complete-field process medians
are 0.91 seconds for scalar and 2.50 seconds for vector at their practical worker
counts—approximately 71.1x and 50.8x faster than the retained tuned Python UTide
processes.

The focused [irregular-confidence snapshot](benchmarks/results/irregular-confidence-2026-09-01.md)
records Lomb–Scargle scalar and vector parity and a 20.05–70.06x advantage over
pinned Python UTide on its 100-series observational workload, depending on field
and worker count.

The [robust-fitting snapshot](benchmarks/results/robust-fitting-2026-09-01.md)
records exact Cauchy-IRLS parity, including identical iteration counts, and a
7.66–19.46x advantage on 100 series. On the more sustained 1,000-series workload,
RUTide is 13.39–17.66x faster at 16 and 32 workers.

The controlled [inferred-constituent snapshot](benchmarks/results/inferred-constituents-2026-09-01.md)
records scalar and coupled-vector exact/approximate parity on both regular FFT
and irregular/gappy Lomb–Scargle confidence paths. RUTide is 48.97–228.45x faster
on the 100-series matrix and 71.93–140.44x faster in retained 1,000-series runs;
the report documents significant host contention and the CPU-affinity controls
used to exclude contaminated measurements.

The [Monte Carlo confidence snapshot](benchmarks/results/monte-carlo-confidence-2026-09-01.md)
records complete scalar/vector covariance sampling on irregular OLS and regular
robust profiles. RUTide is 17.24–61.09x faster on one worker and 10.80–35.08x
faster at 16 workers for the matched 200-realization workloads.

The focused [inferred Monte Carlo check](benchmarks/results/inferred-monte-carlo-2026-09-02.md)
records the Rust-only extension that Python UTide rejects. On the 100-series
irregular profile, nonlinear propagation adds 0.05–2.24% over scalar linear
confidence and 27.73–33.74% over vector linear confidence while retaining
bitwise-identical checksums across 1 and 16 workers.

The implemented scalar kernels now cover fixed-constituent OLS with a mean and
optional trend using raw, linear-time Greenwich, or exact Greenwich phase and
exact, midpoint-linearized, or disabled nodal corrections. The correction
catalog contains all 146 constituents, 162 satellite
corrections, and 251 shallow-water relationships from the pinned Python oracle.
The corrected bulk API shares latitude-independent astronomy, pre-aggregates
satellite terms, and parallelizes the latitude-specific factorizations across
spatial series. Both paths have real-FVCOM parity tests against the pinned
Python oracle.

Percent-energy diagnostics, linearized and Monte Carlo 95% amplitude/phase
confidence intervals, CI-derived signal-to-noise ratio, per-solution PE/SNR
ranking, and matching scalar reconstruction are available. Reconstruction
supports arbitrary target times in the core API and an opt-in complete
original-time series in the FVCOM command.
Scalar `_FillValue` and `NaN` observations are omitted per series, with shared
valid-time masks grouped before fitting. Depth-averaged, native sigma-layer,
and fixed-physical-depth vector currents use the same machinery with a joint
`ua`/`va` or `u`/`v`
validity mask and return current ellipses, all four linearized or Monte Carlo
ellipse confidence intervals, SNR, PE, and optional eastward/northward
reconstruction. The command-line application accepts either
FVCOM `zeta(time, node)`, `ua(time, nele)` plus `va(time, nele)`, or
`u(time, siglay, nele)` plus `v(time, siglay, nele)` and serializes
the corresponding analysis, diagnostics, observation counts, per-series
reference epochs, and optional reconstruction to NetCDF.

FVCOM observations are read and solved in bounded spatial chunks. By default,
the application targets at most 512 MiB across concurrently resident promoted
`f64` observations. Sufficiently large scalar, depth-averaged, and native-layer
inputs use a single-owner NetCDF reader to prepare chunk N+1 while the worker
pool solves chunk N; a rendezvous permits at most those two chunks. Explicit
`--chunk-series N` values remain sequential and provide an exact reproducibility
or memory-control override. Fixed-depth interpolation also remains sequential.
Contiguous selections use one NetCDF hyperslab per component. Sparse sigma-layer
selections coalesce nearby elements and traverse classic NetCDF record variables
in file order; requested layer and element order is still preserved. Chunk-local
masks are grouped exactly as before, and global Monte Carlo stream offsets make
results independent of chunk size and worker scheduling. JSON reports and
scalar/vector NetCDF schemas v18/v17 record the input-pipeline mode, actual chunk
count and series count, and maximum logical bytes across concurrent observation
buffers. When input is overlapped, active stage durations are not additive;
`total_seconds` remains the authoritative elapsed application time.

Robust fitting is available for scalar and vector analyses with Andrews,
bisquare, Cauchy, Fair, Huber, logistic, OLS, Talwar, or Welsch residual
weights, including missing and irregular records, confidence, inference, and
reconstruction. It returns auditable weights, leverage, iteration/stopping
diagnostics, robust scale, and OLS/final RMS residuals.

Monte Carlo confidence uses complete 2×2 scalar or 4×4 current coefficient
covariances, including eastward/northward cross-covariance. Colored current
noise uses the real co-spectrum from the FFT for regular records and the
phase-shifted Lomb–Scargle cross-spectrum for irregular records. Each
constituent's sampled amplitudes or ellipses are summarized with UTide's
clustered-angle median absolute deviation. Non-positive-definite colored
covariances are symmetrized, projected onto the positive-semidefinite cone, and
nudged to a sampleable positive-definite matrix. A pinned ChaCha12 RNG and
series/constituent-derived streams make a seed reproducible independently of
worker scheduling; NumPy and Rust draws are not expected to be bit-identical.
For inference, each independently fitted reference is sampled once per
realization and all of its inferred scalar or positive/negative rotary
coefficients are derived from that same draw. This preserves the exact
reference/inferred correlation instead of treating constrained constituents as
independent uncertain estimates.

## Repository layout

```text
crates/rutide-core/  Numerical library; no file-format or CLI concerns
crates/rutide-cli/   FVCOM NetCDF application and benchmark entry point
crates/rutide-python/ Native PyO3/NumPy extension
python/rutide/        Typed UTide-inspired public Python package
benchmarks/           Locked Python oracle harness and fixture manifests
BENCHMARK_PLAN.md    Frozen experimental intent and measurement protocol
```

The local `UTide/` checkout is deliberately ignored. It is the pinned Python
behavioral oracle described in the benchmark plan, not vendored RUTide source.
Large FVCOM data and generated benchmark results also remain outside Git.

The public compatibility contracts and machine-readable feature status are in
[`COMPATIBILITY.md`](COMPATIBILITY.md); remaining scientific, interface, and
resource tasks are tracked in [`ROADMAP.md`](ROADMAP.md). Irregular scalar and
vector colored confidence use Lomb–Scargle residual spectra, robust fitting and
Monte Carlo confidence are complete. Scalar and coupled-vector inferred
constituents pass exact, approximate, and gappy Python oracle fixtures and are
exposed by the FVCOM commands. Monte Carlo propagation through those
relationships supports OLS and robust fits, white and colored noise, complete,
missing, and irregular records, and deterministic parallel batches. Their
comparative linear-confidence benchmark is complete; Python UTide cannot supply
a direct Monte Carlo-inference timing because it rejects that combination.
The supported solver-option cross-product, validation constraints, parity
exceptions, and evidence are frozen in
[`compatibility/solver-option-matrix-v1.json`](compatibility/solver-option-matrix-v1.json).

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
uv sync --locked
uv run --locked ruff format --check python
uv run --locked ruff check python
uv run --locked python -m unittest discover -s python/tests -v
uv build --wheel
uv sync --project benchmarks/python --locked
uv run --project benchmarks/python --locked ruff check benchmarks/python
uv run --project benchmarks/python --locked \
  python -m unittest discover -s benchmarks/python/tests -v
```

`cargo ci` and `cargo test-all` are repository aliases defined in
`.cargo/config.toml`. CI runs the same gates on every push and pull request.

## Python

The mixed Python/Rust package provides `rutide.solve` / `rutide.reconstruct` for
individual scalar elevations and vector currents, plus time-major
`rutide.solve_many` / `rutide.reconstruct_many` batch endpoints. The endpoint and
result names follow Python UTide where practical, including `A`, `g`, `Lsmaj`,
`Lsmin`, `theta`, `PE`, and `SNR`. Dynamic constituents, missing observations,
irregular Lomb–Scargle confidence, robust fitting, exact/approximate inference,
Monte Carlo confidence, and filtered reconstruction all use the same Rust core
validated by the CLI.

Vector results additionally expose phase-wrap-safe Cartesian
`eastward_cosine_coefficient`, `eastward_sine_coefficient`,
`northward_cosine_coefficient`, and `northward_sine_coefficient` arrays. The
Rust core can cache that representation and evaluate many spatial solutions at
one prepared timestep with an explicit tide-only, mean, or mean-and-trend
policy. The temporal contract and restrictions for future spatial atlases are
defined in [`HARMONIC_CURRENT_ATLAS.md`](HARMONIC_CURRENT_ATLAS.md). The
[temporal-query benchmark](benchmarks/results/temporal-current-query-2026-09-03.md)
records isolated direct-current throughput and its batch-size-dependent worker
scaling; it does not include spatial interpolation or OpenDrift.

The optional MATLAB
[pre-filter transfer-function correction](PREFILTER_TRANSFER.md) is supported
through the Rust core, Python API, and both FVCOM commands. It is useful only
for records passed through a documented filter and should remain disabled for
raw FVCOM fields.

`diagnostics=True` adds the full Codiga constituent-identifiability suite as
structured `coef.diagn` arrays: adjacent RR/RNM/Corrmax, whole-basis `K` and
`SNRallc`, and full/SNR-subset tidal variance. Single and batch coefficient
objects also format a compact per-record diagnostic table.

The batch endpoint shares astronomy, grouped missing masks, irregular spectral
plans, and a dedicated worker pool inside Rust. Coefficients use stable
`(series, constituent)` shapes with per-series presentation rankings and optional
xarray conversion; a working-memory limit bounds native solve chunks. Single and
batch coefficient objects can be atomically saved to a versioned, pickle-free
NPZ archive and restored without refitting or storing source observations.
Packed typed-array blobs keep large batch archives compact and fast to restore;
the retained 4,096-series profile improved load time by 72.28x. Native snapshot
schema 2 stores optional diagnostic arrays densely while continuing to read
original schema-1 snapshots from RUTide 0.2. The current
[real-data product acceptance](benchmarks/results/real-data-product-acceptance-2026-09-03.md) and
[persistence profile](benchmarks/results/python-batch-persistence-2026-09-03.md)
record the installed-wheel ADCP/FVCOM evidence.

See the [Python API guide](PYTHON_API.md) for installation, examples, time
epochs, supported options, intentional compatibility differences, and the
native coefficient object's lifetime contract.

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

The automatic memory/performance balance is recommended. For a controlled
whole-field comparison, pass a `--chunk-series` value at least as large as the
selected node or element count; use a smaller value to enforce a tighter bound.

Supply any unique catalog names in output order with, for example,
`--constituents Q1,O1,K1,M2,S2,K2,M4`. The default remains the frozen five-name
benchmark profile shown above. Use `--constituents auto` for UTide-compatible
record-length selection; its Rayleigh criterion defaults to `1.0` and can be
changed with `--rayleigh-min X`.

Add `--confidence linear` for UTide-compatible colored-noise linear confidence
intervals and SNR. `--white-noise` selects the white residual-noise alternative.
Colored and white scalar confidence support regular and irregular timestamps.
Missing observations on an originally equidistant grid use Python-compatible
linear interpolation of fitted residuals onto the full grid before the FFT;
truly irregular timestamps use a Lomb–Scargle residual spectrum.

Use `--confidence monte-carlo` for nonlinear coefficient-sampling intervals.
The defaults are 200 realizations and root seed 0; override them with
`--mc-realizations N` and `--mc-seed N`. Unlike the pinned Python `MC_n` option,
which is ignored and always draws 200 realizations, both Rust options are
effective. Add `--white-noise` to either confidence method to bypass residual
band coloring. Monte Carlo works with OLS or robust fitting, regular, missing,
and truly irregular scalar/current records, including inferred constituents.
Inferred estimates reuse the reference's sampled realization so their
constraint and correlation are retained exactly.

Add `--method robust` for iteratively reweighted least squares. Cauchy remains
the Python UTide analysis default; select another function with
`--robust-weight`, and override its conventional tuning constant with
`--robust-tuning`. Fractional tolerance defaults to `0.001` and the iteration
limit to 50; override them with `--robust-tolerance` and
`--robust-max-iterations`.

| Weight | Default tuning | Pinned Python status |
|---|---:|---|
| Andrews | 1.339 | Rust extension; upstream array implementation raises |
| Bisquare | 4.685 | Oracle matched |
| Cauchy | 2.385 | Oracle matched; analysis default |
| Fair | 1.400 | Oracle matched |
| Huber | 1.345 | Rust extension; upstream array implementation raises |
| Logistic | 1.205 | Rust extension; upstream array implementation raises |
| OLS | 1.000 | Oracle matched |
| Talwar | 2.795 | Oracle matched |
| Welsch | 2.985 | Oracle matched |

Robust NetCDF outputs record the selected weight function and include per-series
convergence diagnostics plus ragged `robust_weight` / `robust_leverage` arrays.
For a series with missing values, each ragged row follows the finite observations
in original timestamp order. The three Rust-extension functions use their
standard scalar definitions; pinned Python UTide applies Python's scalar `max`
to residual arrays and fails before fitting them.

Add `--no-trend` to match Python UTide's `trend=False` model. The mean remains
fitted, but the linear-time column is omitted from the solve. This option works
for scalar and vector analyses, ordinary and inferred constituents, OLS and
robust fitting, all confidence methods, and complete, gappy, or irregular
records. JSON reports and NetCDF metadata record `trend_enabled`; NetCDF keeps
its stable slope variables and writes exact zeros when the trend is disabled.

Use `--phase greenwich`, `--phase linear-time`, or `--phase raw` to select
Python UTide's phase-reference convention. `greenwich` is the default and
evaluates the astronomical argument at every timestamp. `linear-time` evaluates
it once at the fitted record midpoint and advances it using the constituent's
reference-time frequency. `raw` references phase directly to that midpoint.

Use `--nodal exact`, `--nodal linear-time`, or `--nodal disabled` independently
of `--phase`. `exact` is the default and evaluates nodal/satellite amplitude and
phase corrections at every timestamp. `linear-time` evaluates them once at the
fitted record midpoint and holds them constant; this is Python UTide's
`nodal="linear_time"` approximation. `disabled` applies unit nodal amplitude and
zero nodal phase, matching Python's `nodal=False`. For an irregular series with
missing observations, the midpoint belongs to that series' retained record.
Fit and reconstruction always use the same selected mode. Exact mode remains
the scientifically preferred default, especially for long records; the other
modes are primarily compatibility and sensitivity controls.

Both phase and nodal choices are retained independently in JSON reports, NetCDF
metadata, result digests, profile names, and reconstruction. Existing default
profile names keep their `nodal` component; alternatives use
`nodal-linear-time` or `no-nodal`.

Use `--prefilter-response response.json` only when the input was passed through
a known temporal filter. The file supplies `frequency_cph`, `gain`,
`acceptable_gain_range`, and an optional `fallback` (`error` by default).
MATLAB-style `frq`, `P`, and `rng` names are accepted as aliases. Fitted
coefficients estimate the pre-filter physical harmonics; reconstruction
reapplies the response so it remains comparable with the filtered observations.
The complete response is embedded in NetCDF/JSON provenance and the result
digest. See [PREFILTER_TRANSFER.md](PREFILTER_TRANSFER.md) for the exact contract.

Use `--order selection`, `--order pe`, `--order snr`, or `--order frequency`
to choose a constituent presentation view. The default `selection` view retains
the fitted-model order; PE and SNR rank from largest to smallest, while frequency
ranks from smallest to largest. Ties retain the fitted-model order. SNR ordering
requires either linear or Monte Carlo confidence. An explicit comma-separated
list, such as `--order K1,M2,O1,S2,N2`, is also accepted, but it must be a
complete permutation of every fitted (including inferred) constituent. Use
`--constituents` or inference options to change which constituents are fitted.

Presentation never changes coefficient-array identity or reconstruction. NetCDF
coefficient and diagnostic arrays always retain their stable `constituent` axis;
`constituent_index_by_rank(series,presentation_rank)` maps each requested rank to
its zero-based stable index. The mapping is per series because PE, SNR, and even
the fitted reference-time frequency can vary between records. Retained JSON
samples carry the same mapping, and reports state the requested order and whether
it actually varies by series. Shared selection/explicit maps are stored compactly
in memory and expanded in bounded chunks only while writing NetCDF output.

Add `--constituent-diagnostics` to compute Codiga's extended constituent-
selection evidence: adjacent RR/RNM/Corrmax, whole-model `K`, `SNRallc`, and
condition-adjusted SNR, plus raw, all-constituent, and SNR-filtered tidal
variance. This is deliberately opt-in and requires `--confidence linear` or
`--confidence monte-carlo`; `--diagnostic-min-snr` changes the inclusive
significant-subset threshold from its default of 2. For automatic selection the
configured Rayleigh minimum is reused as the RR denominator, while explicit
lists use the conventional value 1. For example:

```console
cargo run --release --bin rutide -- analyze-scalar \
  --input /path/to/fvcom.nc \
  --output diagnostic-coefficients.nc \
  --constituents auto --confidence linear \
  --constituent-diagnostics --diagnostic-min-snr 2 \
  --workers 64
```

Schemas 18/17 store the configuration as global attributes and the complete
per-series and per-constituent result arrays under descriptive
`diagnostic_*` variable names. Missing neighbors and unavailable fixed-depth
records use index `-1` and `NaN` floating fields. `K` and Corrmax use the actual
retained design matrix, but conventional RR/RNM reduce irregular sampling to an
effective record length and remain advisory for heavily gapped records. See
[`DIAGNOSTICS.md`](DIAGNOSTICS.md) for definitions and the measured full-field
cost.

Add one repeatable `--infer INFERRED:REFERENCE:AMPLITUDE_RATIO:PHASE_OFFSET`
for each constrained scalar constituent. Exact astronomical inference is the
default; `--infer-approximate` selects Python UTide's reference-only approximate
basis. Inferred relationships are retained in both JSON and NetCDF metadata and
participate in confidence intervals, PE/SNR, and reconstruction. For example:

```console
cargo run --release --bin rutide -- analyze-scalar \
  --input /path/to/fvcom.nc \
  --output inferred-scalar.nc \
  --constituents M2,K1 \
  --infer S2:M2:0.35:20 \
  --infer O1:K1:0.50:45 \
  --confidence linear --reconstruct
```

Add `--reconstruct` to write `reconstruction(time, series)` at every original
FVCOM timestamp. With no filter it includes every fitted constituent. Use
`--reconstruct-constituents M2,S2,K1` for an explicit subset, or diagnostic
thresholds such as `--min-pe 1 --min-snr 2`. PE and SNR thresholds are inclusive
and combine with logical AND; explicit names are the alternative selection mode,
matching Python UTide. `--min-snr` requires an enabled confidence method, while PE-only
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
times, and retain the fitted mean plus the trend when it was enabled.

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

The solver core uses strictly increasing Modified Julian Days. Boundary code can
normalize numeric days declared as MJD, Unix, Python Gregorian, MATLAB, or a
custom proleptic-Gregorian epoch with `normalize_numeric_time`; validated
millisecond-resolution `GregorianDateTime` and Rust `SystemTime` conversion are
also available. These conversions reproduce the pinned Python UTide epoch
offsets before the MJD-based astronomical kernel is entered.

Matching Python UTide's safe timestamp behavior, non-finite numeric timestamps
are removed together with their corresponding observations. FVCOM `_FillValue`
entries in either `Itime` or `Itime2` therefore discard that complete time row
for every scalar node or both vector components. Reports and NetCDF metadata
retain the source, analyzed, and discarded timestamp counts; reconstruction is
written only at retained finite timestamps. RUTide deliberately rejects duplicate
or decreasing retained timestamps instead of accepting the ambiguous ordering as
Python currently does, and it never sorts observations implicitly.

Every scalar and vector run also records sampling-quality diagnostics rather
than treating a successful Lomb–Scargle calculation as proof of adequate data.
Per series these include retained count, record span, mean interval, largest gap,
the actual FFT or Lomb–Scargle residual-spectrum route, its even sample count,
and coverage of all nine Python UTide colored-noise bands. Both the total
frequency-bin count and the usable count after excluding fitted-constituent bins
are retained. A zero usable count means that band cannot supply a background
noise estimate for that series.

NetCDF stores the complete per-series fields as `sampling_record_span`,
`sampling_mean_interval`, `sampling_largest_gap`, `residual_spectrum_method`,
`residual_spectrum_time_count`, `spectral_band_frequency_bin_count`, and
`spectral_band_usable_bin_count`, with explicit band-edge coordinate variables.
JSON reports provide worst-case band coverage, FFT/Lomb series counts, extrema,
and full details for retained sample results. These diagnostics are always
produced, including white-noise or no-confidence runs, so data adequacy can be
reviewed before enabling colored confidence; they inform judgment but do not
silently reject a fit.

The practical review order, metadata checklist, and separate ADCP/FVCOM cautions
are maintained in [QUALITY_CONTROL.md](QUALITY_CONTROL.md). A successful solve
is not by itself evidence that the selected constituents are identifiable.

## FVCOM vector-current analysis

Use `analyze-vector` for FVCOM eastward/northward currents. With no layer option,
the source schema is depth-averaged `ua(time, nele)`, `va(time, nele)`, and
`latc(nele)`. Add `--layers all`, `--layers 0,4,9`, or `--layer-count N` to read
native `u(time, siglay, nele)` and `v(time, siglay, nele)` instead. A sample is
removed from both components when either is `_FillValue` or `NaN`; this preserves
a single current ellipse fit on a joint time base.

Use `--depths 5,20,50` instead to interpolate native currents to positive metres
below the instantaneous free surface. Fixed depths require `siglay`, `h`,
`zeta`, triangular `nv`, and the authoritative `wet_cells` mask in addition to
`u`/`v`. Depths, layers, and layer counts are mutually exclusive.

```console
cargo run --release --bin rutide -- analyze-vector \
  --input /path/to/fvcom.nc \
  --output current-ellipses.nc \
  --report current-run.json \
  --constituents M2,S2,N2,K1,O1 \
  --confidence linear \
  --reconstruct \
  --workers 64
```

A depth-resolved example is:

```console
cargo run --release --bin rutide -- analyze-vector \
  --input /path/to/fvcom.nc \
  --output sigma-layer-ellipses.nc \
  --layers 0,4,9 \
  --constituents M2,S2,N2,K1,O1 \
  --workers 64
```

A fixed-physical-depth example is:

```console
cargo run --release --bin rutide -- analyze-vector \
  --input /path/to/fvcom.nc \
  --output fixed-depth-ellipses.nc \
  --depths 5,20,50 \
  --constituents M2,S2,N2,K1,O1 \
  --workers 64
```

The output includes semi-major and signed semi-minor axes, inclination, the
configured phase convention, PE, and—when enabled—the four ellipse CIs and SNR.
Depth-averaged reconstruction is written as
`eastward_reconstruction(time, series)` and
`northward_reconstruction(time, series)`; resolved layouts are described below.
Use `--element-count N` or `--elements 0,10,20` for subsets. Dynamic constituent
selection and reconstruction filters behave as in scalar mode.

Sigma-layer output preserves the selected geometry: coefficient and diagnostic
variables use `(siglay, element[, constituent])`, while reconstruction uses
`(time, siglay, element)`. `siglay_index` and `element_index` retain the exact
zero-based requested source order. These are native terrain-following model
layers, not fixed physical depths; physical depth varies with bathymetry and the
free surface.

Fixed-depth coefficient and diagnostic variables use
`(depth, element[, constituent])`, while reconstruction uses
`(time, depth, element)`. The layer-centre position is evaluated from all three
nodes of each FVCOM triangle at every retained time, and both current components
are linearly interpolated with one shared weight. Dry cells and missing geometry
or bracket currents are jointly missing; targets above the shallowest or below
the deepest physical layer centre are not extrapolated. Shallow or always-dry
coordinates without enough observations remain in the rectangular product with
`analysis_status=unavailable` and NaN harmonic fields. The complete frozen
semantics are in
[`FIXED_DEPTH_INTERPOLATION.md`](FIXED_DEPTH_INTERPOLATION.md).
Real-file Python-oracle correctness and full-domain 10 m performance are
recorded in
[`benchmarks/results/fixed-depth-fvcom-2026-09-02.md`](benchmarks/results/fixed-depth-fvcom-2026-09-02.md).
On the documented 251 GiB host, 64 workers with one 144,860-element block
completed that 144,860-series product in 10.96 s at a 10.64 GiB peak; omit
`--chunk-series` to retain the portable automatic-memory profile.

Vertically resolved results are written incrementally to a temporary NetCDF file
and atomically installed only after all chunks and the canonical digest complete.
This includes confidence fields, ragged robust weights/leverage, and complete
reconstruction. The JSON report and NetCDF `result_output` attribute identify
this path as `incremental`; depth-averaged output remains `buffered`. On the
1,448,600-series benchmark, incremental output preserved the v12 digest while
reducing peak RSS from 2.35 GiB to 1.13 GiB; see
[`benchmarks/results/incremental-layered-output-2026-09-02.md`](benchmarks/results/incremental-layered-output-2026-09-02.md).

For compatibility, colored two-dimensional linear intervals reproduce the
pinned Python UTide implementation exactly, including its asymmetric variance
rescaling: the eastward coefficient pair retains the white estimate while the
northward pair uses its colored residual band. White and colored intervals support
regular timestamps, gaps on an originally regular grid, and truly irregular
timestamps; the last use a Lomb–Scargle residual spectrum.

Monte Carlo current intervals instead use the complete 4×4 covariance and the
eastward/northward co-spectrum, as Python UTide intends for its nonlinear path.
This yields joint semi-major, signed semi-minor, inclination, and phase
realizations rather than independently linearizing the two components.

Vector inference uses separate positive- and negative-rotary constraints:
`--infer INFERRED:REFERENCE:AMP+:PHASE+:AMP-:PHASE-`. It supports exact or
approximate OLS or robust fitting, missing values, white/colored linear or Monte
Carlo confidence, PE/SNR, and reconstruction. Robust inference solves the coupled
complex rotary model with one shared configured weight per retained timestamp; NetCDF
outputs retain those shared weights, complex-model leverage, convergence
diagnostics, and reconstruction metadata. Add `--method robust` alongside the
vector `--infer` relationships to enable it.

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
