# Pre-filter transfer-function correction

RUTide supports the real preprocessing-filter response correction described by
MATLAB UTide. It is optional and should not be enabled for raw FVCOM fields or
raw ADCP ensembles. It is intended for records that were passed through a known
filter whose gain appreciably changes one or more fitted tidal frequencies.

## Scientific meaning

For a known response `P(f)`, RUTide linearly interpolates one real gain for each
constituent and includes it in the astronomical design basis:

```text
observed harmonic(f) = P(f) * physical harmonic(f)
```

The fitted amplitudes and Cartesian coefficients therefore estimate the
pre-filter physical harmonics. Reconstruction deliberately reapplies `P(f)` and
returns the signal in the observation domain, matching the input and MATLAB
UTide convention. This makes residuals and held-out comparisons internally
consistent.

The correction cannot restore information destroyed by a stop band, identify an
unknown processing chain, repair aliasing, or compensate for undocumented ADCP
averaging. A near-zero response makes physical coefficient recovery
ill-conditioned. RUTide therefore requires an explicit acceptable gain-magnitude
range and rejects values outside it by default.

## Rust and Python APIs

The Rust core uses `PreFilterCorrection` and `PreFilterFallback`. The response is
validated once and the interpolated constituent gains are folded into cached
bases outside the sample and spatial-series loops. Leaving the option unset
retains the original fast path.

Python accepts `prefilt=` on `solve` and `solve_many`:

```python
prefilt = {
    "frequency_cph": [0.0, 0.04, 0.10, 0.20],
    "gain": [1.0, 0.98, 0.74, 0.30],
    "acceptable_gain_range": [0.05, 2.0],
    "fallback": "error",
}

coef = rutide.solve(time, velocity, lat=62.0, prefilt=prefilt)
```

MATLAB-style `frq`, `P`, and `rng` keys are aliases. `fallback="unity"`
reproduces MATLAB's permissive behavior for a constituent outside the response
grid or acceptable range; `fallback="error"` is the safer default. Complex
responses are explicitly rejected. Saved coefficient objects retain the full
response and reconstruct identically after loading.

## FVCOM command

The scalar and vector commands accept `--prefilter-response PATH`. The JSON file
uses the same descriptive or MATLAB-style keys:

```json
{
  "frequency_cph": [0.0, 0.04, 0.10, 0.20],
  "gain": [1.0, 0.98, 0.74, 0.30],
  "acceptable_gain_range": [0.05, 2.0],
  "fallback": "error"
}
```

```console
rutide analyze-vector \
  --input filtered-currents.nc \
  --output coefficients.nc \
  --prefilter-response instrument-response.json
```

Scalar NetCDF schema 18 and vector schema 17 record the response samples,
accepted range, fallback, interpolation method, and response convention. JSON
reports contain the same values, profile names include `prefilter`, and result
digests include the complete response. The path itself is not authoritative:
the values used by the solve are embedded in the result.

## Supported scope and limits

- Scalar elevation or one-component velocity records are supported.
- Vector records are supported when the same real response was applied to both
  eastward and northward components.
- Ordinary and inferred constituents, OLS and robust fits, complete and missing
  batches, confidence intervals, diagnostics, persistence, and reconstruction
  use the same corrected basis.
- A complex response, or different component-specific responses, needs a
  coupled formulation because phase-changing component filters alter the current
  ellipse. RUTide rejects that unsupported representation instead of treating it
  as a real gain.
- Response frequencies are cycles per hour, finite, non-negative, and strictly
  increasing. At least two response samples are required.
- Gains may be signed but must be finite. Their magnitudes must lie within the
  configured positive range at every fitted constituent unless the explicit
  unity fallback is selected.

Raw FVCOM `zeta`, `ua`/`va`, and native `u`/`v` normally require no correction.
For ADCP products, only enable it when the exact temporal processing response is
known and applies to the provided velocity series.

## Validation

Independent MATLAB-equation tests cover linear interpolation, strict rejection,
and the legacy unity fallback. End-to-end scalar/vector tests cover physical
coefficient recovery, observation-domain reconstruction, exact inference,
complete/gappy batches, Python persistence, and FVCOM NetCDF provenance. The
uncorrected control path remains separately exercised.

Primary implementation references are MATLAB UTide `ut_solv.m`/`ut_E` revision
`4a6354f` and Python UTide `harmonics.py` revision `9b60caf`, where the retained
pre-filter argument is marked unimplemented.
