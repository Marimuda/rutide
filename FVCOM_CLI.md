# FVCOM NetCDF application

RUTide's command-line application performs bounded-memory tidal harmonic
analysis directly from FVCOM NetCDF output. Use it when a field is too large to
materialize as a NumPy matrix or when you need a self-describing, versioned
NetCDF analysis product.

The application reads its source file without modification, processes spatial
series in deterministic chunks, writes to a temporary sibling file, and installs
the result only after the complete analysis and digest succeed. Existing output
and report files are preserved unless `--overwrite` is supplied.

## Build

Install Rust 1.98 and the NetCDF C development library. Common package names are
`libnetcdf-dev` on Debian/Ubuntu and `netcdf` with Homebrew.

```console
git clone https://github.com/Marimuda/rutide.git
cd rutide
cargo build --release --locked --bin rutide
./target/release/rutide --help
```

Use a release build for production or performance work. Debug builds are
intentionally not representative.

## Supported FVCOM inputs

| Analysis | Required variables | Spatial location |
|---|---|---|
| Scalar elevation | `zeta(time,node)`, node latitude, FVCOM time | Node |
| Depth-averaged current | `ua(time,nele)`, `va(time,nele)`, `latc(nele)`, FVCOM time | Element |
| Native-layer current | `u(time,siglay,nele)`, `v(time,siglay,nele)`, `latc(nele)`, FVCOM time | Sigma layer and element |
| Fixed-depth current | Native-layer fields plus `siglay`, `h`, `zeta`, triangular `nv`, and authoritative `wet_cells` | Positive metres below instantaneous surface and element |

FVCOM `Itime`/`Itime2` timestamps are normalized to Modified Julian Days.
Non-finite or fill-valued timestamps remove that complete row. Retained times
must be strictly increasing; RUTide does not silently sort duplicate or reversed
input.

Scalar `_FillValue` and NaN observations are omitted per series. Vector samples
use one joint mask: if either eastward or northward velocity is missing, both are
omitted at that timestamp. Infinite observations are rejected.

Declared vector component units must agree. Source identity, size, units,
indices, latitude, vertical selection, analyzed/discarded time counts, options,
schema version, and a canonical result digest are retained in the output and
JSON report.

## Scalar elevation

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

Use `--node-count N` for a prefix or `--nodes 0,10,20` for explicit zero-based
node indices.

## Vector currents

With no vertical option, `analyze-vector` reads depth-averaged `ua` and `va`:

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

Use `--element-count N` for a prefix or `--elements 0,10,20` for explicit
zero-based element indices.

Native model layers preserve terrain-following coordinates:

```console
./target/release/rutide analyze-vector \
  --input /path/to/fvcom.nc \
  --output sigma-layer-ellipses.nc \
  --layers 0,4,9 \
  --constituents M2,S2,N2,K1,O1
```

`--layers all` selects every layer and `--layer-count N` selects a prefix.
Layer, layer-count, and physical-depth selections are mutually exclusive.

Fixed-depth analysis evaluates layer-centre positions from all three element
nodes at every retained timestamp and jointly interpolates both current
components:

```console
./target/release/rutide analyze-vector \
  --input /path/to/fvcom.nc \
  --output fixed-depth-ellipses.nc \
  --depths 5,20,50 \
  --constituents M2,S2,N2,K1,O1
```

Depths are positive metres below the instantaneous free surface. RUTide does not
extrapolate above the shallowest or below the deepest layer centre. Dry cells,
missing geometry, and missing bracket velocities are jointly unavailable. Read
[FIXED_DEPTH_INTERPOLATION.md](FIXED_DEPTH_INTERPOLATION.md) before interpreting
this product.

## Constituent selection and inference

The default benchmark profile is `M2,S2,N2,K1,O1`. Supply a unique ordered list
with `--constituents Q1,O1,K1,M2,S2,K2,M4`, or use
`--constituents auto --rayleigh-min 1.0` for record-length selection.

Presentation order can be `selection`, `pe`, `snr`, `frequency`, or a complete
explicit permutation. Presentation never changes coefficient identity or
reconstruction; output stores a rank-to-stable-index map.

Scalar inference is repeatable:

```console
--infer S2:M2:0.35:20
```

Vector inference uses separate positive/negative rotary relationships:

```console
--infer INFERRED:REFERENCE:AMP+:PHASE+:AMP-:PHASE-
```

Exact astronomical inference is the default. `--infer-approximate` selects the
Python-compatible reference-only approximation. Inference participates in
confidence, diagnostics, reconstruction, robust fitting, and output provenance.

## Model and astronomical conventions

- `--no-trend` retains the fitted mean and omits the linear trend.
- `--phase greenwich` is the default exact per-time Greenwich argument.
  `linear-time` and `raw` are compatibility/sensitivity alternatives.
- `--nodal exact` is the preferred default. `linear-time` holds midpoint nodal
  terms constant and `disabled` applies unit amplitude/zero phase corrections.
- Fit and reconstruction always use the same conventions, which are retained in
  reports, metadata, profiles, and digests.

A fitted trend is local to the analyzed interval. Exclude it from long-range
prediction unless persistence has independent physical support.

## Confidence and diagnostics

`--confidence linear` enables 95% intervals and SNR. Colored residual noise is
the default: regular and gappy-grid samples use the Python-compatible FFT path,
while truly irregular timestamps use Lomb-Scargle spectra. Add `--white-noise`
only when white-noise uncertainty is scientifically justified.

`--confidence monte-carlo` uses 200 nonlinear covariance draws and root seed 0
by default. Change them with `--mc-realizations N` and `--mc-seed N`. Monte
Carlo confidence supports scalar/vector, OLS/robust, regular/irregular, and
ordinary/inferred models.

Add `--constituent-diagnostics` to write adjacent RR/RNM/Corrmax, whole-model
`K`, `SNRallc`, `SNRallc/K`, and full/SNR-subset tidal variance. Confidence is
required; `--diagnostic-min-snr` defaults to 2. These diagnostics are opt-in so
their runtime and storage costs do not affect ordinary solves.

Read [DIAGNOSTICS.md](DIAGNOSTICS.md) for definitions and
[QUALITY_CONTROL.md](QUALITY_CONTROL.md) for interpretation. RR and RNM reduce
irregular records to an effective duration and remain advisory for heavily
gapped data; `K` and Corrmax use the actual retained design matrix.

## Robust fitting

Add `--method robust` for iteratively reweighted least squares. Cauchy is the
default robust weight. Available weights are Andrews, bisquare, Cauchy, Fair,
Huber, logistic, OLS, Talwar, and Welsch.

Use `--robust-weight`, `--robust-tuning`, `--robust-tolerance`, and
`--robust-max-iterations` to control the fit. Output records the configuration,
termination reason, scale, OLS/final RMS residuals, and retained-time weights and
leverage. Review weights against known instrument/model events rather than
assuming every low-weight sample is disposable.

## Reconstruction

`--reconstruct` writes the model at every retained source timestamp. Select a
subset with either `--reconstruct-constituents M2,S2,K1` or inclusive diagnostic
thresholds such as `--min-pe 1 --min-snr 2`. SNR filtering requires confidence.

Scalar reconstruction uses `(time,series)`. Depth-averaged vector output stores
eastward/northward `(time,series)` arrays. Native-layer and fixed-depth output
preserve `(time,siglay,element)` or `(time,depth,element)` geometry.

## Known pre-filter response

Use `--prefilter-response response.json` only when the source was passed through
a documented real temporal filter. The response contains frequency, gain,
acceptable range, and an explicit error or unity fallback. Fitted coefficients
estimate pre-filter physical harmonics; reconstruction reapplies the filter for
comparison with observations. The complete response is retained in provenance.

Do not enable correction for raw FVCOM fields or a guessed filter. See
[PREFILTER_TRANSFER.md](PREFILTER_TRANSFER.md) for the exact JSON contract and
limits.

## Memory and parallelism

`--workers N` controls independent spatial solves. By default the application
targets at most 512 MiB across concurrently resident promoted `f64` observation
buffers and overlaps one NetCDF read with the preceding solve where supported.
Omit `--chunk-series` for this recommended automatic policy.

Set `--chunk-series N` to enforce a known series count per chunk and sequential
input. This provides a reproducibility and tighter-memory override. Fixed-depth
input is sequential. Stage durations can overlap in automatic mode;
`total_seconds` is the authoritative wall time.

Large native-layer/fixed-depth products are written incrementally while
retaining deterministic order and digest behavior. Monte Carlo streams remain
stable across worker and chunk counts.

## Output contract

Scalar NetCDF schema 18 and vector schema 17 retain stable coefficient axes,
source-position maps, sampling diagnostics, optional confidence and
identifiability diagnostics, robust details, pre-filter provenance,
reconstruction metadata, and result digest. Readers must reject unknown schema
versions rather than guessing from variable presence.

Complete schema and compatibility guarantees are in
[COMPATIBILITY.md](COMPATIBILITY.md). The JSON report contains the matching
schema, configuration, aggregate sampling/resource statistics, timings, and
canonical digest.

## Before a whole-field analysis

1. Run a small explicit node/element sample spanning open boundary, shelf,
   strait, fjord, shallow/deep, weak-flow, and wet/dry regimes.
2. Verify units, time epoch, excluded spin-up, latitude, current orientation,
   vertical semantics, and missing/dry masks.
3. Enable confidence and constituent diagnostics; inspect resolution,
   conditioning, band coverage, residuals, and reconstruction.
4. Choose workers and automatic memory behavior on the target host.
5. Preserve the JSON report next to the NetCDF result.

The full domain-specific checklist is in [QUALITY_CONTROL.md](QUALITY_CONTROL.md).
