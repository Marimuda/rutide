# Future harmonic-current atlas product

This document defines a possible downstream spatial-current product. It is not
part of the RUTide FVCOM/ADCP temporal harmonic-analysis release scope. The
analysis engine already supplies its temporal coefficient and prediction
foundation; this future product would add mesh interpolation and, optionally,
an OpenDrift adapter.

## Cartesian representation

For constituent `k`, RUTide vector results expose four component coefficients:

```text
eastward_cosine_coefficient[k]
eastward_sine_coefficient[k]
northward_cosine_coefficient[k]
northward_sine_coefficient[k]
```

If `z_k(latitude, time)` is RUTide's complex astronomical basis including the
configured nodal correction, the harmonic current is

```text
u_k = eastward_cosine_coefficient[k] * real(z_k)
    + eastward_sine_coefficient[k]   * imag(z_k)

v_k = northward_cosine_coefficient[k] * real(z_k)
    + northward_sine_coefficient[k]   * imag(z_k)
```

These coefficients are algebraically equivalent to `semi_major`, `semi_minor`,
`inclination`, and `phase`. They are the atlas payload because Cartesian values
remain continuous where ellipse phase or inclination wraps.

The public Rust `VectorSolution::cartesian` conversion validates all source
array shapes and returns a `CartesianVectorSolution`. Existing ellipse outputs
remain the compatibility and domain-reporting view.

## Temporal query contract

`GreenwichNodalReconstructor` prepares astronomical terms once for a target time
axis. It can then:

- reconstruct complete Cartesian current time series;
- reconstruct many cached Cartesian solutions in parallel; or
- evaluate many cached Cartesian solutions at one prepared time index, returning
  one compact `VectorCurrent` per input location.

The last operation is intended for a particle model. The caller retains
Cartesian coefficients across timesteps and avoids rebuilding ellipse-derived
scalar solutions for every query.

Every new Cartesian reconstruction call requires an explicit
`NonHarmonicTerms` policy:

- `TidesOnly` includes only selected tidal constituents;
- `Mean` adds the fitted constant current;
- `MeanAndTrend` also extrapolates the fitted linear trend.

The existing compatibility reconstruction continues to include the fitted mean
and trend. A transport reader should default to `TidesOnly` or `Mean`; it should
only expose trend extrapolation as an explicit expert choice.

## Reference epochs and spatial interpolation

With exact Greenwich phase and exact nodal corrections, the harmonic basis is
evaluated directly at every prediction timestamp. The harmonic Cartesian
coefficients therefore do not need to be rephased merely because missing-value
masks gave neighboring fits different record midpoints.

Non-harmonic terms still carry a per-series `reference_time`. Before spatially
interpolating a fitted mean together with a trend, normalize the intercept to a
shared atlas epoch `t_a`:

```text
mean_atlas = mean + slope_per_day * (t_a - reference_time)
```

Raw phase, linear-time Greenwich phase, and midpoint-linearized nodal modes do
depend on their fitted reference epoch. Until an explicit rephasing transform is
implemented and validated, an atlas must do one of the following:

1. require exact Greenwich phase and exact nodal corrections;
2. reconstruct at every spatial support location before interpolating `u, v`; or
3. retain distinct coefficient epochs and never interpolate their coefficients.

The first option is the canonical production profile.

## Serialized fields

FVCOM vector NetCDF/JSON schema 17 contains the four Cartesian fields while retaining
all ellipse, confidence, diagnostic, mean, trend, and per-series epoch fields.
Python single and batch coefficient summaries expose the same descriptive names.
The Cartesian arrays are derived from authoritative ellipse solutions, so
existing Python coefficient archives remain readable without a persistence
schema change.

## Remaining atlas work

The following work is intentionally outside this temporal increment:

- choose and serialize a common epoch for non-harmonic atlas terms;
- store FVCOM coordinates, connectivity, boundaries, bathymetry, and wet masks;
- implement boundary-safe element lookup and spatial interpolation;
- add a vectorized Python `(time, lon, lat, depth) -> (u, v)` query;
- implement the optional OpenDrift reader and trajectory validation;
- benchmark the complete particle-query path rather than only harmonic kernels.
