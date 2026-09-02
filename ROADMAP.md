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
- Ordinary least-squares scalar and vector fits with a mean and optional trend.
- Raw, linear-time Greenwich, and exact Greenwich phase with independently
  selectable exact, midpoint-linearized, or disabled nodal terms.
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

## 3. Inferred constituents — complete

- Support inferred/reference names, amplitude ratios, phase offsets, and exact and
  approximate inference modes.
- Validate scalar ratios separately from the two-ratio vector convention.
- Propagate inferred constituents through diagnostics, linear and Monte Carlo
  confidence intervals, ordering, serialization, and reconstruction.

Acceptance requires resolved/unresolved scalar pairs, vector ellipses, invalid
reference graphs, exact/approximate Python-oracle fixtures, and independent
invariants for the Monte Carlo extension that Python UTide does not implement.

The scalar and vector fixed-latitude kernels are complete. Scalar inference
includes robust and bulk solves; the vector kernel uses one coupled complex solve
for the independent positive/negative rotary ratios. Exact and
Python-compatible approximate bases, grouped references, white and colored
linear confidence, PE/SNR, reconstruction, and invalid graph/value handling pass
resolved and unresolved pinned-oracle fixtures. Inference confidence preserves a
pinned Python column-order/indexing quirk inside these models only. Scalar and
joint-mask vector batches now support varying latitudes, complete or missing
records, and white/colored confidence while retaining bounded shared Lomb plans.
Monte Carlo batches sample every independent reference coefficient once per
realization and transform that shared draw through all scalar or rotary inference
relationships. Exact/approximate, OLS/robust, white/colored, complete/missing,
and regular/irregular paths are covered, including bitwise worker-count
reproducibility. Python-oracle tests freeze the underlying fitted coefficients,
covariances, and linear confidence; transformation invariants and cross-solver
tests cover the unsupported-in-Python nonlinear propagation.
The scalar and vector FVCOM commands now accept repeatable relationships and an
exact/approximate switch; JSON reports and versioned NetCDF schemas retain every
ratio, phase offset, convention, and mode, while the canonical digest includes
the complete inference configuration. End-to-end vector coverage combines
inference with joint missing-value masks, Lomb–Scargle colored confidence,
reconstruction, and Cauchy robust fitting. The robust vector path solves one
coupled complex IRLS problem, preserves one shared weight per timestamp, and
propagates UTide-compatible weighted covariance and colored residuals into
linear intervals. Pinned-oracle coverage freezes ellipse coefficients, white
and colored intervals, leverage, weights, iteration count, residual diagnostics,
and reconstruction; the CLI writes the same ragged diagnostics for complete or
gappy batches.

The comparative measurements are now retained in
`benchmarks/results/inferred-constituents-2026-09-01.md`. Across controlled
100-series exact-inference runs, RUTide is 48.97–228.45x faster than pinned
Python; retained 1,000-series comparisons are 71.93–140.44x faster. Approximate
mode is 69.30–86.55x faster at 16 workers. These measurements predate the robust
coupled-vector implementation; robust inference performance has not yet been
benchmarked separately.

## 4. Monte Carlo confidence intervals — complete

- Complete 2×2 scalar and 4×4 vector coefficient covariances are sampled for OLS
  and final-weight Cauchy robust fits.
- Regular FFT and irregular Lomb–Scargle residual paths now include the
  UTide-compatible real eastward/northward co-spectrum. Direct and cached Lomb
  kernels pass the same pinned-Python band-power fixtures.
- Realization count and root seed are effective API and CLI options. Derived
  series/constituent streams make output independent of outer worker scheduling
  when the inner matrix implementation is held sequential, as in the FVCOM app.
- Scalar amplitudes and vector ellipses reproduce UTide's angle clustering and
  median-absolute-deviation interval calculation.
- Finite non-positive-definite covariance matrices are symmetrized, projected
  onto the positive-semidefinite cone with a Jacobi eigendecomposition, and
  diagonally nudged until Cholesky sampling is valid.
- Missing and truly irregular scalar/vector batches, robust fits, CLI options,
  result digests, JSON reports, and versioned NetCDF metadata are integrated.
- Scalar and vector inference preserve perfect reference/inferred dependence by
  applying every constrained relationship to the same joint reference draw.
  Exact/approximate, OLS/robust, white/colored, and regular/irregular batches are
  supported with deterministic per-series streams.

The acceptance suite now covers deterministic seeded scalar/vector fixtures,
white and colored noise, near-degenerate ellipses, covariance repair, statistical
cross-covariance recovery, robust fits, irregular masks, worker-count invariance,
and distribution-level pinned-Python comparisons. The pinned Python `MC_n`
option is ignored and always uses an internal 200 realizations; RUTide preserves
that default but deliberately makes its configured count effective.

There is no theoretical restriction on combining Monte Carlo confidence with
inferred constituents. RUTide samples each independently fitted
ordinary/reference coefficient jointly, applies exact or approximate inference
relations to every realization, and retains reference/inferred correlation
before forming amplitudes or ellipses. Python UTide does not implement that path,
so validation combines pinned-Python parity for the fitted model and covariance
inputs with analytic scaling/rotation invariants, shared-draw dependence checks,
cross-solver equivalence, and batch reproducibility tests.

The dedicated
`benchmarks/results/monte-carlo-confidence-2026-09-01.md` snapshot covers
regular robust and irregular/gappy OLS scalar/vector workloads. On matched
200-realization comparisons, RUTide is 17.24–61.09x faster than pinned Python on
one worker and 10.80–35.08x faster at 16 workers. Those historical measurements
cover the shared non-inferred surface. The Rust-only inferred profile is recorded
in `benchmarks/results/inferred-monte-carlo-2026-09-02.md`: its focused irregular
comparison adds 0.05–2.24% over scalar linear-confidence time and 27.73–33.74%
over vector time, with deterministic 1/16-worker checksums. Python speedup is not
reported because the Python implementation rejects the combination.

## 5. Solver-option parity — in progress

- **Complete:** make the linear trend optional while retaining a fitted mean.
  The core and FVCOM scalar/vector APIs cover ordinary and inferred
  constituents, OLS and robust fitting, every confidence method, and complete,
  missing, and irregular records. Pinned Python `trend=False` fixtures freeze
  scalar and vector coefficients and colored intervals. JSON reports and the
  versioned NetCDF schemas expose `trend_enabled`; stable slope fields are exact
  zeros when the trend is omitted.
- **Complete:** expose exact Greenwich, linear-time Greenwich, and raw phase
  through one `SolverOptions` / `PhaseReference` API. The choice propagates
  through ordinary and inferred scalar/vector fits, OLS and robust confidence
  paths, mask-specific irregular epochs, reconstruction, CLI profiles, canonical
  digests, reports, and versioned NetCDF metadata. Pinned Python fixtures cover
  scalar coefficients and colored intervals, vector ellipses, exact and
  approximate inference, and held-out reconstruction.
- **Complete:** expose exact, linear-time, and disabled nodal/satellite
  corrections independently of the phase-reference convention. The option is
  shared by ordinary and inferred scalar/vector fits, retained-record epochs,
  reconstruction, CLI profiles, canonical digests, JSON reports, and versioned
  NetCDF metadata. Alternative modes avoid per-timestamp satellite evaluation;
  the exact default path retains its prior behavior. Pinned Python fixtures cover
  scalar coefficients, rotary ellipses, exact/approximate inference, batching,
  missing irregular records, and held-out reconstruction.
- **Complete:** support result presentation by descending PE, descending SNR,
  ascending fitted frequency, stable selection order, or an explicit complete
  permutation. Bulk coefficient arrays retain one stable constituent axis;
  versioned scalar/vector NetCDF schemas and retained JSON samples expose a
  per-series rank-to-index mapping. SNR ordering requires confidence, ties are
  deterministic, and focused fixtures freeze the pinned Python UTide PE/SNR/
  frequency views. Shared maps avoid per-series allocation, while diagnostic
  maps use one contiguous buffer.
- **Complete:** normalize numeric MJD, Unix, Python Gregorian, MATLAB, and custom
  Gregorian epochs plus civil/Rust datetimes at application or binding
  boundaries while retaining an MJD-only numerical core. Non-finite times and
  corresponding scalar/vector rows are removed as in Python UTide; source and
  discarded counts are serialized. Duplicate or decreasing retained times are
  explicitly rejected rather than silently sorted or passed into an ambiguous
  Python fit. Pinned epoch constants, leap dates, invalid civil components,
  missing-time oracle parity, and FVCOM scalar/vector fill rows are covered.

Each remaining option combination needs focused oracle coverage. Unsupported
combinations must be rejected during validation rather than partially applied.

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
