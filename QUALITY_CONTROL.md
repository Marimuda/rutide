# FVCOM and ADCP analysis quality control

This guide defines the checks needed to interpret a successful RUTide fit as a
scientifically usable tidal analysis. Numerical convergence alone does not show
that a constituent set is identifiable, that the record samples the relevant
periods, or that instrument/model metadata are correct.

RUTide reports evidence and rejects malformed numerical inputs. It does not
silently remove scientifically questionable constituents or invent missing
deployment metadata. Thresholds below are starting points for review, not
universal acceptance rules.

## Product boundary

The current product fits and reconstructs temporal scalar and vector harmonics:

- in-memory scalar, ADCP, station, and model arrays through Rust or Python;
- FVCOM elevation at nodes;
- FVCOM depth-averaged currents at elements;
- FVCOM currents at native sigma layers; and
- FVCOM currents interpolated to fixed physical depths.

It does not spatially interpolate coefficients, package an FVCOM mesh as a
current atlas, or drive OpenDrift. Those are a separate downstream product.

## Metadata checklist before fitting

Record these items with the analysis even when the source format does not make
them machine-readable:

| Area | Required interpretation |
|---|---|
| Time | Epoch, calendar, timezone treatment, sampling interval, clock corrections, and deployment/model interval |
| Position | Latitude used for astronomy; station/node/element identity; horizontal datum and coordinate reference system for mapped products |
| Quantity | Elevation or eastward/northward velocity; physical units; sign and component convention |
| Vertical | Datum for elevation; ADCP bin depth and orientation, or FVCOM depth-averaged/sigma/fixed-depth definition |
| Missing data | Fill values, wet/dry masks, rejected ensembles/bins, and whether gaps are natural or QC-generated |
| Processing | Averaging, filtering, detiding, coordinate rotation, magnetic-declination correction, and the exact transfer response when `prefilt` is used |
| Model context | FVCOM run/configuration identifier, forcing interval, spin-up excluded, output cadence, and relevant boundary forcing |

Python accepts numeric arrays and therefore cannot infer their units or
coordinate reference system. Preserve those alongside the saved RUTide archive
in the surrounding dataset or analysis manifest. The FVCOM CLI retains source
path and size, propagates a declared `zeta` unit, requires declared vector
component units to match, and retains source indices, latitudes, vertical
selection, analysis options, schema/version, and a canonical result digest.

## Recommended review sequence

### 1. Inspect retained sampling

Review `observation_count`, record span, mean interval, largest gap, and the
reported FFT or Lomb–Scargle spectrum route. Confirm that timestamps are in the
intended epoch and strictly increasing. A plausible tidal answer from dates
interpreted in the wrong epoch is still wrong.

For colored confidence, inspect all nine spectral-band bin counts after fitted
constituent exclusions. A band with zero usable bins has no local background
noise estimate. Lomb–Scargle supports irregular times but does not replace
missing record duration or make clustered sampling equivalent to continuous
coverage.

### 2. Check constituent resolution

Enable constituent diagnostics and review adjacent pairs in frequency order:

- `RR >= 1` is the conventional Rayleigh starting point.
- `RNM >= 1` includes the record's estimated signal-to-noise level.
- large `Corrmax` means the pair is sharing parameter variance; values near or
  below 0.2 have historically been treated as comfortable, but the complete
  context matters.
- `SNRallc / K > 1` is the whole-model condition-bound criterion.

`K` and `Corrmax` use the actual retained design matrix and see gaps directly.
RR and RNM reduce the record to an effective duration and are only approximate
for heavily irregular sampling. When a pair is unresolved, prefer a longer
record, remove one constituent, or use a scientifically justified inference
relationship; do not select whichever amplitude looks more familiar.

### 3. Check significance and explanatory power

Review amplitude/ellipse confidence, SNR, and percent energy together. `SNR >=
2` is the conventional RUTide significant-subset default, not a guarantee of
physical importance. Compare `PTVallc` with `PTVsnrc`: a large difference means
the full fit gains substantial variance from constituents that do not pass the
chosen SNR threshold. Inspect residual time series and spectra for remaining
tidal peaks, drift, bursts, and nonstationarity.

Percent tidal variance can increase when more basis functions are added. It
must be interpreted alongside resolution, conditioning, and held-out residuals,
not as a model-selection objective by itself.

### 4. Review robust fits

For IRLS, retain the selected weight function, tuning constant, tolerance,
iteration limit, termination reason, final scale, and OLS/final RMS residuals.
Plot weights against time and deployment/model events. Many low weights may
indicate real intermittent physics or a bad stationary harmonic model rather
than disposable sensor outliers. Treat iteration exhaustion or objective
rollback as a review flag.

### 5. Review vector-current physics

Confirm eastward/northward orientation before fitting. Inspect signed semi-minor
axes, inclination, phase, and their confidence intervals. Near-circular or
near-linear ellipses naturally make some angles poorly determined even when the
Cartesian current prediction is stable. Use the Cartesian coefficient fields
for interpolation or wrap-sensitive comparisons.

For depth-resolved products, compare adjacent bins/layers for discontinuities
and check that dry, side-lobe-contaminated, above-surface, and below-bottom
samples did not enter the fit.

### 6. Validate reconstruction

Compare reconstruction with source values on retained times and, where
possible, a withheld interval. Report bias, RMS error, residual variance, and
extreme-event behavior. A linear trend is a fitted local term, not a safe
long-range forecast; use tide-only or mean-inclusive reconstruction for
extrapolation unless trend persistence has an independent physical basis.

When pre-filter correction is enabled, default reconstruction is in the
filtered observation domain. The reported harmonic coefficients estimate the
pre-filter physical amplitudes.

## ADCP-specific checks

- Apply manufacturer and deployment QC before harmonic analysis: correlation,
  percent-good, error velocity, tilt, blanking distance, side-lobe exclusion,
  and impossible-speed checks as appropriate to the instrument.
- Verify whether velocities are beam, instrument, magnetic, or true east/north.
  Record heading correction and magnetic declination.
- Preserve ensemble duration and any temporal averaging/filter response. Do not
  claim pre-filter recovery from a guessed response.
- Use one joint mask for eastward/northward velocity; RUTide does this for vector
  input. Treat isolated depth-bin gaps separately from deployment-wide time
  gaps.
- Record transducer depth, bin-centre convention, changing pressure/tide effects,
  upward/downward orientation, and moving-platform corrections where relevant.
- Compare stable depth ranges separately; pooling bins with different dynamics
  can produce a precise but physically meaningless ellipse.

## FVCOM-specific checks

- Exclude spin-up and select a time span appropriate to the constituents being
  tested. Output cadence must resolve the fastest selected constituent.
- Verify `zeta` units/datum and that `ua`/`va` or `u`/`v` are geographic
  eastward/northward components in consistent units.
- Respect `wet_cells` for depth-resolved/fixed-depth work. RUTide jointly masks
  dry cells and missing vector components and never extrapolates fixed-depth
  currents above the shallowest or below the deepest layer centre.
- Remember that elevation lives at nodes and FVCOM velocity at elements. Source
  indices are retained precisely; a harmonic-analysis output is not yet a
  self-contained spatial mesh product.
- Inspect shallow-water/compound constituents where nonlinear coastal dynamics
  warrant them. Automatic Rayleigh selection answers frequency separation, not
  physical relevance.
- Compare representative open-boundary, shelf, strait, fjord, shallow, deep,
  weak-flow, and wet/dry locations before accepting a full-domain run.

## Minimum evidence for a reported analysis

Archive the RUTide version, output schema, options/profile, exact constituent
and inference lists, confidence/noise method, sampling diagnostics, optional
identifiability diagnostics, source identity, and result digest. Also retain the
external instrument/model metadata that RUTide cannot infer.

For publications or operational products, report the record interval and
duration, missing fraction, constituent-selection rationale, phase/nodal
convention, latitude, units, vertical definition, confidence method, robust
method if used, and reconstruction validation. If pre-filter correction is
enabled, publish the complete response table and acceptable range.

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for equations and field definitions and
[PREFILTER_TRANSFER.md](PREFILTER_TRANSFER.md) for filter-response semantics.
