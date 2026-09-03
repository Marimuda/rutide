# Constituent-selection diagnostics

RUTide 0.3.0 restores the constituent-selection diagnostics described in
section II.D of Codiga (2011), *Unified Tidal Analysis and Prediction Using the
UTide Matlab Functions*. Python UTide retains PE and SNR but not the broader
independence and reconstructed-fit suite.

The diagnostics are advisory. They explain whether a chosen model appears
identifiable and how much detrended variance it captures; they do not silently
change the fitted constituent set. A later caller can use the evidence to make an
explicit second fit with a constituent removed or inferred.

## Definitions and conventions

For adjacent directly modeled constituents `q1` and `q2`, conventional Rayleigh
resolution is Codiga equation 81:

```text
RR(q1, q2) = 24 * effective_record_length_days
             * abs(frequency_q2_cph - frequency_q1_cph) / Rmin
```

Neighbor relationships are constructed in ascending frequency order and mapped
back to stable fitted-constituent order. Reference constituents participate;
inferred constituents do not, because their amplitudes are constrained rather
than independently identified. A value of one is the conventional boundary, but
RUTide reports the value instead of treating it as a new automatic-selection
rule.

Tidal variance follows equations 99–102 and the original MATLAB implementation.
After subtracting the fitted mean and optional trend:

```text
TVraw   = mean(raw_east^2 + raw_north^2)
TVallc  = mean(all_fit_east^2 + all_fit_north^2)
TVsnrc  = mean(SNR_subset_east^2 + SNR_subset_north^2)
PTV*    = 100 * TV* / TVraw
```

The northward terms are omitted for a scalar record. If `TVraw` is zero, the
percentage is mathematically undefined and the typed API returns `None` rather
than propagating an implicit NaN or infinity.

RNM follows equation 82 by multiplying RR by the square root of the adjacent
constituents' mean SNR. `K` is the two-norm condition number of the actual basis
matrix, and `SNRallc / K > 1` is the report's whole-model error-bound criterion.
Corrmax is the greatest absolute correlation among the two scalar or four vector
Cartesian parameters belonging to an adjacent constituent pair.

The high-level Rust calculation is opt-in through
`ConstituentDiagnosticsOptions`. Its defaults reproduce MATLAB UTide's
`Rmin = 1` and inclusive `SNR >= 2` significant-subset threshold. Prepared raw
and Greenwich/nodal models expose scalar diagnostics, while Greenwich/nodal
models also expose vector diagnostics. Scalar and coupled-vector inference are
supported: ordinary and reference constituents participate in the neighbor
graph, while constrained inferred outputs remain aligned with the solution but
have no independent neighbors. The coupled-vector path evaluates Corrmax from
the complete four-by-four Cartesian parameter block rather than treating the
eastward and northward fits as independent.

Prepared batch models provide matching post-fit diagnostic calls for complete
and `NaN`-gappy scalar or jointly masked vector series. Records sharing a mask
reuse retained-position metadata and independent series run in parallel. Each
diagnostic model still uses its own retained timestamps, midpoint, astronomical
terms, and covariance; an exact time-axis fingerprint guards the coupled-vector
interface against a solution/model mismatch.

Python callers opt in with `diagnostics=True` on `rutide.solve` or
`rutide.solve_many`. Results are exposed as `coef.diagn` / `coef.diagnostics`,
with MATLAB-style `lo.RR`, `hi.RNM`, `hi.CorMx`, `K`, `SNRallc`, `TVraw`, and
`PTVsnrc` names plus descriptive aliases for neighbor fields. Single and batch
coefficient objects provide `diagnostic_table()` presentation; batch numeric
fields remain dense on the stable `(series, constituent)` axis.

`K` does not refactor or copy the tall basis. RUTide applies the complex-basis
normalization to the prepared pivoted QR's small triangular factor and caches
its singular-value ratio. Unweighted coefficient normal inverses are likewise
formed from and cached behind that QR factor. The reconstructed variance and
whole-model energy calculations stream over the observations without retaining
reconstruction arrays; vector inputs are not interleaved into a temporary copy.
Coupled inference necessarily materializes the complex covariance matrices used
by Corrmax and performs one filtered reconstruction for the independently
thresholded inferred subset. These choices keep the default solve path unchanged
and bound opt-in diagnostic memory by the fitted output and small coefficient
matrices.

## Irregular records

The suite is evaluated for the retained record. Basis-derived `K` and `Corrmax`
use its actual design matrix and therefore see the effects of gaps and clusters.
RR and RNM instead reduce the sampling pattern to an effective record length,
so the report warns that constituent selection for irregularly distributed
timestamps is not rigorously characterized by these conventional thresholds.
RUTide will serialize the existing sampling diagnostics beside these values and
will not claim that RR or RNM alone proves identifiability for a heavily gapped
record.

## 0.3.0 implementation sequence

1. Public equation-81 neighbor and equation-99–102 tidal-variance kernels — done.
2. Cached `K`, `SNRallc`, RNM, and Corrmax from the fitted design/covariance —
   done for the real scalar/vector basis.
3. Scalar/vector ordinary and scalar/coupled-vector inference OLS/robust
   integration, including batch and missing-value orchestration — done.
4. Structured Rust and Python results plus compact human-readable presentation
   — done.
5. Scalar NetCDF schema 18, vector schema 17, and backward-readable Python
   coefficient snapshot schema 2 — done.
6. MATLAB-derived equation fixtures and measured whole-field overhead — done.

The suite remains opt-in. On the retained full FVCOM field it changed scalar
process wall from 1.07 s to 2.01 s and vector wall from 3.03 s to 4.92 s while
adding roughly 82 MiB and 67 MiB of median peak RSS, respectively. The default
path retains its prior digest and avoids diagnostic work. Detailed commands,
repetitions, storage costs, and stage timings are recorded in
[`benchmarks/results/constituent-diagnostics-2026-09-03.md`](benchmarks/results/constituent-diagnostics-2026-09-03.md).

## Sources

- [Codiga, D. L. (2011), GSO Technical Report 2011-01](https://www.po.gso.uri.edu/~codiga/utide/2011Codiga-UTide-Report.pdf),
  section II.D.
- [MATLAB UTide `ut_diagntable`](https://github.com/OceanMetSEPA/utide_toolbox/blob/master/ut_solv.m),
  used to resolve implementation details such as mean-square normalization,
  effective record length, and inferred-neighbor exclusion.
- [Python UTide `utide/diagnostics.py`](https://github.com/wesleybowman/UTide/blob/master/utide/diagnostics.py),
  which computes PE and SNR only.
