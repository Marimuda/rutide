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

Automatic constituent selection and confidence estimation use only retained
rows. NumPy/NetCDF4 masked arrays are normalized to NaN. NaN observations and
non-finite timestamps are omitted jointly; vector currents use one joint
component mask. Infinite observations, duplicate or decreasing retained times,
invalid relationships, and unsupported option values raise `ValueError` instead
of being silently adjusted.

Irregular colored confidence uses the implemented Lomb–Scargle path. It is not
silently approximated with an FFT.

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

By default `reconstruct` uses `min_SNR=2.0` and `min_PE=0.0`, like UTide.
SNR filtering requires a fit with confidence intervals. Set `min_SNR=None` for
PE-only filtering or to reconstruct a fit made with `conf_int="none"`. An
explicit `constit=[...]` selection takes precedence over diagnostic thresholds.

Version `0.2` accepts only coefficients created by the same RUTide process; it
does not import arbitrary Python UTide coefficient dictionaries and does not yet
serialize native fit objects. `solve_many` covers in-memory station and model
matrices. The CLI remains the appropriate interface when FVCOM NetCDF input and
incremental output must stay bounded rather than materializing the complete
array in Python.
