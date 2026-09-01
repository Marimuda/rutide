# RUTide roadmap

This roadmap tracks the remaining work required to move from the validated FVCOM
application profile toward broad Python UTide compatibility. It is deliberately
ordered by scientific dependency: each increment must pass a pinned-Python oracle
gate before it becomes part of the performance comparison or public interface.

Status values are **complete**, **next**, and **planned**. A feature is complete
only when its scalar and vector scope is stated explicitly, invalid inputs have
defined behavior, and representative oracle tests are versioned.

## Validated foundation — complete

- Full pinned catalog: 146 constituents, 162 satellite corrections, and 251
  shallow-water relationships.
- Explicit and Rayleigh-selected dynamic constituent lists.
- Ordinary least-squares scalar and vector fits with mean and trend.
- Raw-phase and exact Greenwich/nodal numerical kernels.
- Percent energy, linearized 95% confidence intervals, SNR, and ranking views.
- Reconstruction at arbitrary target times with constituent, PE, and SNR filters.
- Per-series missing observations, grouped valid-time masks, and joint eastward /
  northward masks for vector currents.
- Python-compatible residual interpolation plus FFT colored spectra when samples
  are missing from an otherwise equidistant source grid.
- FVCOM `zeta` and depth-averaged `ua` / `va` NetCDF applications.
- Whole-field scalar and vector correctness and performance benchmarks.

## 1. Irregular colored-noise confidence — in progress

The harmonic least-squares fit already supports irregular timestamps. This item
concerns only the colored residual spectrum used to estimate confidence intervals.
Until the vector increment is complete, truly irregular vector colored-confidence
requests must continue to return `UnevenTimeForColoredConfidence` rather than use
an unverified component path.

### 1a. Scalar Lomb–Scargle spectrum — complete

- Reproduce UTide's nine residual-noise frequency bands and oversampled frequency
  grid, including the per-band frequency cap.
- Remove the residual mean, map the periodic Hann window onto irregular sample
  positions, and match SciPy/UTide one-sided PSD normalization.
- Exclude the spectral sample nearest every fitted constituent before averaging
  each band and convert spectral density to record-length power.
- Integrate the result with scalar linearized colored confidence intervals.
- Preserve the FFT route for equidistant and gappy-equidistant records.

Acceptance requires scalar oracle fixtures for deterministic jitter, random
jitter, isolated gaps, clustered gaps, and at least one band-boundary constituent.
Tests must compare band powers, coefficient variances, amplitude/phase intervals,
and SNR, not only the final amplitude.

### 1b. Vector Lomb–Scargle spectrum — next

- Compute eastward and northward residual auto-spectra using the scalar kernel.
- Add the UTide-compatible irregular cross-spectrum needed by full vector
  covariance and future Monte Carlo intervals.
- Integrate vector linearized colored intervals while recording any intentional
  Python-compatibility behavior separately from scientifically improved behavior.

Acceptance requires rotary/ellipse fixtures with correlated and uncorrelated
component noise, asymmetric missingness resolved through the joint mask, and
comparison of all four ellipse confidence intervals plus SNR.

### 1c. Performance and sampling diagnostics

- Cache frequency grids, windows, and reusable trigonometric work by valid-time
  mask where mathematically valid.
- Benchmark representative observation records separately from the regular FVCOM
  whole-field profile; Lomb–Scargle is an `O(samples * frequencies)` path.
- Report record span, retained observation count, largest gap, and spectral-band
  coverage so Lomb–Scargle is not presented as a cure for inadequate sampling.

## 2. Robust fitting — planned

- Implement iteratively reweighted least squares with the pinned Python default
  Cauchy weight function.
- Define convergence tolerance, iteration limit, scale estimate, zero-weight
  behavior, and non-convergence errors explicitly.
- Return final weights and robust diagnostics.
- Support scalar and complex vector fits, missing observations, exact corrections,
  confidence intervals, and reconstruction from the robust solution.

Acceptance requires clean-data convergence to OLS, injected spikes, sustained
outliers, near-zero residual scale, non-convergence, and Python weight/coefficient
comparisons. Performance results must state iteration counts.

## 3. Inferred constituents — planned

- Support inferred/reference names, amplitude ratios, phase offsets, and exact and
  approximate inference modes.
- Validate scalar ratios separately from the two-ratio vector convention.
- Propagate inferred constituents through diagnostics, linear confidence intervals,
  ordering, serialization, and reconstruction.

Acceptance requires resolved/unresolved scalar pairs, vector ellipses, invalid
reference graphs, and exact/approximate Python-oracle fixtures. Python UTide does
not implement Monte Carlo confidence combined with inference; RUTide must either
retain that explicit boundary or specify and validate an extension.

## 4. Monte Carlo confidence intervals — planned

- Build the complete coefficient covariance and pseudo-covariance matrices.
- Include eastward/northward cross-covariance and colored cross-spectral power.
- Make realization count and random seed effective, reproducible API options.
- Reproduce UTide's angle clustering and median-absolute-deviation intervals.
- Repair non-positive-definite covariance matrices using a documented method.

Acceptance requires deterministic seeded scalar/vector fixtures, white and colored
noise, near-degenerate ellipses, covariance repair, and distribution-level checks.
The pinned Python `MC_n` option is documented as not yet implemented even though
its Monte Carlo route uses an internal 200 realizations; compatibility tests must
therefore freeze the actual oracle behavior rather than its nominal option.

## 5. Solver-option parity — planned

- Make the linear trend optional while retaining a fitted mean.
- Expose exact Greenwich, linear-time, and raw phase modes through one API.
- Expose exact, linear-time, and disabled nodal/satellite corrections.
- Support result presentation by PE, SNR, frequency, or explicit order without
  losing stable constituent identity in bulk results.
- Define missing/non-finite timestamp handling and datetime/epoch conversion at
  application or binding boundaries.

Each option combination needs focused oracle coverage. Unsupported combinations
must be rejected during validation rather than partially applied.

## 6. Product and resource work — planned

- Read and solve NetCDF spatial chunks so complete `f64` scalar/vector fields are
  not simultaneously resident. Preserve deterministic output order and mask
  grouping across chunks.
- Add depth-resolved FVCOM current variables after defining their output schema.
- Stabilize the Rust library and NetCDF schemas, then add Python bindings only if
  a drop-in or mixed Python/Rust workflow is an actual user requirement.
- Publish reproducible release artifacts, compatibility documentation, and a
  machine-readable feature matrix.

For the current vector benchmark, chunked input is the main remaining resource
opportunity: the complete promoted `ua` and `va` arrays dominate the approximately
2.10 GiB Rust high-water mark. It is independent of the scientific feature order
above and can be scheduled between oracle-heavy increments.

## Definition of broad compatibility

RUTide may claim broad Python UTide compatibility only after items 1 through 5
have explicit supported/unsupported matrices and the supported cases pass the
pinned oracle. Performance claims remain profile-specific: regular FVCOM OLS,
irregular Lomb–Scargle confidence, robust fitting, and Monte Carlo intervals must
be reported separately.
