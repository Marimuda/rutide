# Python API

RUTide's Python package is a mixed Python/Rust distribution built with PyO3 and
maturin. It exposes familiar `utide.solve` and `utide.reconstruct`-shaped
endpoints while the fit, confidence calculation, and reconstruction remain in
the optimized Rust core.

The API is intentionally **UTide-inspired, not a drop-in replacement**. It uses
UTide field names where they are useful, adds descriptive aliases, rejects
unknown or inconsistent options, and documents the few deliberate differences
below.

## Install and develop

Published releases will install with:

```console
python -m pip install rutide
```

For a source checkout, `uv` creates an isolated environment and invokes maturin:

```console
uv sync --locked
uv run --locked python -m unittest discover -s python/tests -v
```

The package supports CPython 3.9 and newer. Wheels use PyO3's `abi3-py39`
stable ABI, so one wheel per supported operating-system/architecture pair can
serve several CPython versions. NumPy inputs are normalized to contiguous
`float64` buffers and copied once into GIL-independent Rust-owned memory; native
output vectors transfer into NumPy ownership. Both solve and reconstruction
release the Python GIL during numerical work.

## Scalar elevations

```python
import numpy as np
from rutide import reconstruct, solve

time_mjd = 60_000.0 + np.arange(24 * 90) / 24
height = get_observed_height()

coef = solve(
    time_mjd,
    height,
    lat=62.0,
    constit="auto",
    method="ols",
    conf_int="linear",
)
tide = reconstruct(time_mjd, coef)

print(coef.name, coef.A, coef.g, coef.A_ci, coef.SNR)
print(tide.h)
```

`coef.A` and `coef.g` are amplitude and Greenwich phase. Descriptive aliases
are available as `coef.amplitude` and `coef.phase_degrees`. `coef.PE` /
`coef.percent_energy` and `coef.SNR` / `coef.signal_to_noise` contain the
diagnostics used for ordering and reconstruction filters.

## Vector currents

Pass northward velocity as the third positional argument:

```python
coef = solve(time_mjd, eastward, northward, lat=62.0, constit=["M2", "S2"])
currents = reconstruct(time_mjd, coef, min_SNR=2.0, min_PE=0.1)

print(coef.Lsmaj, coef.Lsmin, coef.theta, coef.g)
print(currents.u, currents.v)
```

The ellipse fields also have `semi_major`, `semi_minor`,
`inclination_degrees`, and `phase_degrees` aliases. Confidence fields follow the
same convention with `_ci` suffixes.

Vector results also expose `eastward_cosine_coefficient`,
`eastward_sine_coefficient`, `northward_cosine_coefficient`, and
`northward_sine_coefficient`. These are algebraically equivalent to the ellipse
fields but remain continuous across phase and inclination wrap boundaries, so
they are the preferred representation for spatial interpolation. Batch values
have shape `(series, constituent)` and are included by `to_xarray()`. They are
derived from the authoritative stored ellipse solution when a saved coefficient
object is loaded, preserving compatibility with existing archives.

## Batched arrays

`solve_many` fits a time-major `(time, series)` matrix without a Python loop.
It shares astronomical preparation and repeated missing-value masks, executes
independent fits on a dedicated Rayon worker pool, and retains deterministic
series order and Monte Carlo streams across worker and chunk counts.

```python
from rutide import reconstruct_many, solve_many

# height.shape == (time, station); one latitude per station is optional.
coef = solve_many(
    time_mjd,
    height,
    lat=station_latitude,
    constit=["M2", "S2", "N2", "K1", "O1"],
    workers=16,
    memory_limit_mb=512,
)
tide = reconstruct_many(time_mjd, coef)

print(coef.A.shape)          # (station, constituent)
print(coef.rank_index.shape) # (station, presentation_rank)
print(tide.h.shape)          # (time, station)
```

Pass eastward and northward `(time, series)` arrays to use the vector path.
`lat` may be one scalar shared by all series or a one-dimensional per-series
array. Masked arrays and NaNs are accepted; vector components use a joint mask.
Non-finite timestamps remove that row globally, with the retained-to-source row
mapping available as `coef.aux.time_position`.

The coefficient matrices always preserve one stable fitted-constituent axis.
`rank_index[series]` maps requested PE, SNR, frequency, or explicit presentation
rank to that stable axis. Reference-time frequencies have shape
`(series, constituent)` because different missing masks can produce different
fit epochs. The `dims` mapping names array dimensions, and `coef.to_xarray()`
creates an `xarray.Dataset` when the optional `xarray` package is installed.

`workers=None` uses available parallelism, capped at the series count.
`memory_limit_mb` bounds each temporary native component chunk, not the caller's
NumPy arrays, the single Rust-owned input copy needed for GIL-free execution, or
the retained solution arrays. Set it to `None` to process the entire matrix in
one native chunk. Both fitting and reconstruction release the GIL.

## Important options

| Python option | Supported values and behavior |
|---|---|
| `constit` | `"auto"`, `None`, or a sequence of catalog names; automatic selection uses `Rayleigh_min` |
| `method` | `"ols"` or `"robust"` |
| `conf_int` | `"linear"`, `"MC"`, `"none"`, or `None` |
| `white` | `False` uses FFT or Lomb–Scargle colored residual spectra; `True` uses white noise |
| `trend` | Include or omit the linear trend; the mean is always fitted |
| `phase` | `"Greenwich"`, `"linear_time"`, or `"raw"` |
| `nodal` | `True`, `"linear_time"`, or `False` |
| `order_constit` | `"PE"`, `"SNR"`, `"frequency"`, or an explicit name sequence |
| `robust_kw` | Weight function/name plus `tune`, `tol`, and `maxit`, or their descriptive aliases |
| `infer` | UTide-shaped inferred/reference names, amplitude ratios, phase offsets, and optional `approximate` |
| `MC_n`, `MC_seed` | Effective realization count and deterministic root seed |
| `diagnostics` | Opt in to Codiga's extended constituent-identifiability suite; requires confidence/SNR |
| `diagnostic_min_SNR` | Inclusive SNR threshold for the diagnostic significant-subset reconstruction; default `2.0` |
| `prefilt` | Known real preprocessing-filter response with frequency, gain, acceptable range, and fallback |

Automatic constituent selection and confidence estimation use only retained
rows. NumPy/NetCDF4 masked arrays are normalized to NaN. NaN observations and
non-finite timestamps are omitted jointly; vector currents use one joint
component mask. Infinite observations, duplicate or decreasing retained times,
invalid relationships, and unsupported option values raise `ValueError` instead
of being silently adjusted.

Irregular colored confidence uses the implemented Lomb–Scargle path. It is not
silently approximated with an FFT.

### Pre-filter response correction

Pass `prefilt=` only when a documented temporal filter was applied to the
observations. Descriptive keys and MATLAB aliases are interchangeable:

```python
coef = solve(
    time_mjd,
    filtered_velocity,
    lat=62.0,
    prefilt={
        "frequency_cph": [0.0, 0.04, 0.10, 0.20],  # alias: frq
        "gain": [1.0, 0.98, 0.74, 0.30],           # alias: P
        "acceptable_gain_range": [0.05, 2.0],      # alias: rng
        "fallback": "error",
    },
)
```

Fitted amplitudes estimate the physical harmonics before filtering.
`reconstruct` reapplies the response and returns the filtered observation-domain
signal. `fallback="error"` rejects out-of-grid and unacceptable gains;
`fallback="unity"` requests MATLAB-compatible substitution. Complex responses
and different eastward/northward filters are rejected because they require a
coupled phase-changing vector formulation. Single/batch saves retain the
response and reconstruction behavior. See
[PREFILTER_TRANSFER.md](PREFILTER_TRANSFER.md).

### Constituent identifiability

Set `diagnostics=True` on `solve` or `solve_many` to evaluate RR, RNM, Corrmax,
whole-model `K` and `SNRallc`, and raw/all/significant tidal variance. The result
is available through both `coef.diagn` and `coef.diagnostics`; the short fields
follow MATLAB UTide where practical:

```python
coef = solve(
    time_mjd,
    height,
    lat=62.0,
    constit=["M2", "S2", "N2", "K1", "O1"],
    diagnostics=True,
)

print(coef.diagn.lo.RR, coef.diagn.hi.RNM, coef.diagn.hi.CorMx)
print(coef.diagn.K, coef.diagn.SNRallc, coef.diagn.SNRallc_over_K)
print(coef.diagn.TVraw, coef.diagn.PTVallc, coef.diagn.PTVsnrc)
print(coef.diagnostic_table())
```

Missing neighbors use index `-1` and NaN metrics. Single-fit diagnostic vectors
follow the requested presentation order; dense batch fields have shape
`(series, constituent)` on the stable coefficient axis. Use
`batch.diagnostic_table(series)` for a readable one-series table. Diagnostics
are opt-in so ordinary solve performance and memory remain unchanged.

### Inference

Scalar relationships contain one ratio and phase offset per inferred name.
Vector relationships contain `2N` entries: all positive-rotary values followed
by all negative-rotary values, matching Python UTide.

```python
infer = {
    "inferred_names": ["S2"],
    "reference_names": ["M2"],
    "amp_ratios": [0.2],
    "phase_offsets": [15.0],
    "approximate": False,
}
coef = solve(time_mjd, height, lat=62.0, constit=["M2", "S2"], infer=infer)
```

Monte Carlo confidence is valid with inference in RUTide. Each reference draw is
shared with its inferred constituents so the constraint and correlation are
preserved. The pinned Python UTide implementation rejects that combination.

## Time conventions

NumPy `datetime64`, Python `date`/`datetime` sequences, and numeric arrays are
accepted. Datetimes are converted to Modified Julian Days (MJD) before entering
the core. Numeric input defaults to MJD, which is convenient for FVCOM and is a
documented difference from Python UTide.

Use `epoch="python"` for Python Gregorian ordinal days, `epoch="matlab"` for
MATLAB datenums, `epoch="unix"` for Unix seconds, or pass a date/datetime/string
origin when values are days relative to a custom epoch. The same rules apply to
`reconstruct` target times. Missing target timestamps produce NaN output rows.

## Reconstruction and object lifetime

`Coefficient` and `Tide` inherit from `Bunch`, so values support both mapping and
attribute access. A `Coefficient` owns an immutable native fitted model and
solution; reconstruction does not repeat the fit and does not translate the
model back through Python. Coefficient arrays are therefore read-only snapshots.
Robust fits expose the retained-row weights as `coef.weights` and under
`coef.robust.weights`. OLS fits return `coef.weights is None` rather than
allocating an all-ones array proportional to the input length.

## Saving fitted coefficients

Both single and batched fits can be saved and restored without repeating the
analysis:

```python
import rutide

coef.save("analysis.rutide.npz")
restored = rutide.load("analysis.rutide.npz", workers=8)  # workers: batches only
tide = rutide.reconstruct_many(target_time, restored)
```

`rutide.save(coef, path)` is the equivalent function form. It writes atomically
and uses compressed NPZ by default; pass `compressed=False` when write speed is
more important than storage. The archive is pickle-free and stores a schema-1
container with a schema-2 native coefficient snapshot plus typed NumPy arrays.
Arrays are coalesced into bounded typed
blobs instead of creating one ZIP entry for every field of every batch member;
loading exposes validated zero-copy views of those blobs to the native restore
path. It contains the normalized retained
timestamps, fitted solution, uncertainty and robust diagnostics, constituent
selection, optional identifiability diagnostics, inference graph, and every
option needed to rebuild the immutable native reconstruction model. Dense batch
diagnostics use a fixed set of flattened typed arrays rather than per-series
metadata. Original observation values are deliberately not stored.

RUTide `0.3.x` writes native snapshot schema 2 and continues to load the
per-array and packed-blob schema-1 snapshots written by `0.2.x`; legacy fits
restore with `diagn=None`. Unknown schemas are rejected rather than guessed.
Loading a batch recreates its dedicated native worker pool; `workers=` may
override the saved worker count for the current machine. A loaded object has the
same read-only arrays and reconstruction behavior as the original object.

By default `reconstruct` uses `min_SNR=2.0` and `min_PE=0.0`, like UTide.
SNR filtering requires a fit with confidence intervals. Set `min_SNR=None` for
PE-only filtering or to reconstruct a fit made with `conf_int="none"`. An
explicit `constit=[...]` selection takes precedence over diagnostic thresholds.

RUTide does not import arbitrary Python UTide coefficient dictionaries.
`solve_many` covers in-memory station and model matrices. The CLI remains the
appropriate interface when FVCOM NetCDF input and incremental output must stay
bounded rather than materializing the complete array in Python.
