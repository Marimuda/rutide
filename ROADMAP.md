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

## 1. Irregular colored-noise confidence — complete

The harmonic least-squares fit already supports irregular timestamps. This item
concerns only the colored residual spectrum used to estimate confidence intervals.

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

### 1b. Vector Lomb–Scargle spectrum — complete

- Compute eastward and northward residual auto-spectra using the scalar kernel.
- Integrate vector linearized colored intervals while recording any intentional
  Python-compatibility behavior separately from scientifically improved behavior.

Acceptance requires rotary/ellipse fixtures with correlated and uncorrelated
component noise, asymmetric missingness resolved through the joint mask, and
comparison of all four ellipse confidence intervals plus SNR.

### 1c. Performance — complete

- The dedicated scalar/vector probes benchmark representative irregular records
  separately from the regular FVCOM whole-field profile; Lomb–Scargle remains an
  `O(samples * frequencies)` path.
- Reusable records now cache the timestamp-only Hann window, frequency grid,
  phase-shifted trigonometric bases, and basis energies. Batch plans require at
  least 16 series with the same valid-time mask and are constructed in parallel.
- Cache growth is bounded: at most four mask groups per batch, at most 16 MiB of
  trigonometric bases per plan, and a direct low-memory fallback for unique,
  lightly reused, or longer records.
- The retained measurements and environment are recorded in
  `benchmarks/results/irregular-confidence-2026-09-01.md`.
- On the 100-series comparison the planned kernel is 20.05–70.06x faster than
  pinned Python UTide and 5.54–10.82x faster than the original direct Rust
  kernel. At 1,000 series, 16/32-worker speedups over Python are 31.61–61.64x.
- Batched matrix projection remains a possible optimization for larger shared-mask
  workloads, but is no longer a viability blocker.

### 1d. Sampling diagnostics — planned

- Report record span, retained observation count, largest gap, and spectral-band
  coverage so Lomb–Scargle is not presented as a cure for inadequate sampling.

## 2. Robust fitting — complete

- Cauchy IRLS reproduces the pinned Python default (`tune=2.385`,
  `tol=0.001`, `maxit=50`) and permits explicit validated overrides.
- Convergence, objective-increase rollback, exact fits, degenerate MAD scale,
  leverage validation, and iteration exhaustion have defined outcomes.
- Scalar fits and complex vector fits return the final shared per-time weights,
  leverage, iteration count, stopping reason, scale, and OLS/final RMS residuals.
- Complete, missing, and irregular records support exact corrections, white or
  colored linear confidence intervals, SNR, PE, and reconstruction from the
  robust coefficients.
- Scalar and vector FVCOM commands expose `--method robust` plus tuning,
  tolerance, and iteration-limit options. NetCDF outputs retain robust settings
  and ragged per-observation diagnostics for gappy series.

The acceptance suite covers exact clean data, injected spikes, a sustained
outlier block, non-convergence, near-zero non-exact scale, missing irregular
scalar/vector records, and pinned Python comparisons of coefficients, ellipses,
weights, confidence intervals, and SNR. The dedicated performance snapshot
records all iteration counts in
`benchmarks/results/robust-fitting-2026-09-01.md`; RUTide is 7.66–19.46x faster
on the 100-series worker matrix and 13.39–17.66x faster on 1,000 series at 16/32
workers.

## 3. Inferred constituents — in progress

- Support inferred/reference names, amplitude ratios, phase offsets, and exact and
  approximate inference modes.
- Validate scalar ratios separately from the two-ratio vector convention.
- Propagate inferred constituents through diagnostics, linear confidence intervals,
  ordering, serialization, and reconstruction.

Acceptance requires resolved/unresolved scalar pairs, vector ellipses, invalid
reference graphs, and exact/approximate Python-oracle fixtures. Python UTide does
not implement Monte Carlo confidence combined with inference; RUTide must either
retain that explicit boundary or specify and validate an extension.

The scalar and vector fixed-latitude kernels are complete. Scalar inference
includes robust and bulk solves; the vector kernel uses one coupled complex solve
for the independent positive/negative rotary ratios. Exact and
Python-compatible approximate bases, grouped references, white and colored
linear confidence, PE/SNR, reconstruction, and invalid graph/value handling pass
resolved and unresolved pinned-oracle fixtures. Inference confidence preserves a
pinned Python column-order/indexing quirk inside these models only. Scalar and
joint-mask vector batches now support varying latitudes, complete or missing
records, and white/colored confidence while retaining bounded shared Lomb plans.
The scalar and vector FVCOM commands now accept repeatable relationships and an
exact/approximate switch; JSON reports and versioned NetCDF schemas retain every
ratio, phase offset, convention, and mode, while the canonical digest includes
the complete inference configuration. End-to-end vector coverage combines
inference with joint missing-value masks, Lomb–Scargle colored confidence, and
reconstruction. Robust vector inference remains before this item can be marked
complete and is rejected explicitly at both configuration and CLI boundaries.

The comparative measurements are now retained in
`benchmarks/results/inferred-constituents-2026-09-01.md`. Across controlled
100-series exact-inference runs, RUTide is 48.97–228.45x faster than pinned
Python; retained 1,000-series comparisons are 71.93–140.44x faster. Approximate
mode is 69.30–86.55x faster at 16 workers. Robust coupled-vector inference is now
the only implementation gap in this item.

## 4. Monte Carlo confidence intervals — planned

- Build the complete coefficient covariance and pseudo-covariance matrices.
- Add the UTide-compatible irregular eastward/northward cross-spectrum and include
  cross-covariance and colored cross-spectral power.
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
