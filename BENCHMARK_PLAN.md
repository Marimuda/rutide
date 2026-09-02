# RUTide feasibility and benchmark plan

Status: implementation underway

Date: 2026-08-31

## Progress snapshot

The baseline, fixed raw-kernel, and fixed corrected bulk gates have been reached
for the initial five-constituent scalar profile. Real FVCOM node-zero results
match the pinned Python oracle, and the corrected field API scales across cores
by separating shared time-only satellite terms from latitude-specific designs.

The initial fixed-profile application gate has passed. The Rust path reads
distinct FVCOM `zeta` values, reconstructs exact time, fits varying latitudes,
writes every coefficient to NetCDF, and emits per-stage timings and a result
digest. The 32 frozen nodes pass an automated comparison against Python UTide.
On the full 75,160-node field, retained whole-process medians were 3.15 seconds
for Rust and 64.69 seconds for the tuned 32-process Python baseline. The 20.5x
application speedup passes the provisional 3x target for this narrow profile.
A follow-up allocator and input-buffer optimization reduced peak Rust RSS from
about 5.4 GiB to 0.690 GiB and the whole-process median to 1.51 seconds without
changing results. See the curated snapshots under `benchmarks/results/`.

## Objective

Determine whether reimplementing UTide in optimized, modern Rust can preserve the
scientific results of Python UTide while making large FVCOM tidal analyses
substantially faster and more resource-efficient on this machine.

This is a decision experiment, not yet a commitment to a complete rewrite. The
first implementation should cover the narrowest useful path, benchmark it fairly,
and only expand if the evidence supports doing so.

The initial success criteria are:

1. scientifically equivalent harmonic coefficients and reconstructions;
2. at least 5x higher in-memory solve throughput than canonical single-process
   Python UTide on the full scalar workload;
3. at least 3x lower wall time than the best practical Python baseline, including
   NetCDF input, on the same full workload; and
4. bounded memory use that permits the workload to run without swapping.

These thresholds are provisional and should be revisited after measuring where
the Python baseline actually spends its time. A Rust rewrite is not justified if
the workload is dominated by storage bandwidth or if tuned Python can obtain
similar performance with batching and multiprocessing.

## Baseline implementation

The canonical reference is the sibling checkout at `UTide/`, currently:

- package: Python UTide;
- Git revision: `8fabe121752bc317931472a10a42e306715106de`;
- branch: `master`;
- public operations: `utide.solve` and `utide.reconstruct`;
- numerical dependencies: NumPy 2 or newer and SciPy; and
- current solve API limitation: a time vector and scalar/vector observations must
  be one-dimensional, so a spatial field is processed as many independent calls.

The exact Python, NumPy, SciPy, NetCDF library, BLAS/LAPACK implementation, and
thread settings used in a benchmark run must be recorded in its result manifest.
The reference checkout must remain unmodified while it serves as the oracle.

## Primary FVCOM fixture

The largest NetCDF file found under the sibling FVCOM tree is:

```text
../projects/fvcom/claude_scratchpad/baroclinic_vikc1701/run/frs2f_0001.nc
```

Observed metadata on 2026-08-31:

| Property | Value |
|---|---:|
| File size | 25,778,391,080 bytes (about 25.78 GB) |
| NetCDF format | 64-bit offset (CDF-2) |
| Time samples | 745 |
| Time range | MJD 58113 to 58144 |
| Nominal sampling | hourly |
| Mesh nodes | 75,160 |
| Mesh elements | 144,860 |
| Vertical sigma layers | 10 |
| Scalar tidal candidate | `zeta(time, node)` |
| Vector tidal candidates | `ua(time, nele)`, `va(time, nele)` |
| Per-location latitude | `lat(node)` or `latc(nele)` |

The primary benchmark is harmonic analysis of every `zeta` node. A secondary
benchmark analyzes every depth-averaged `(ua, va)` element as a current vector.
These variables are physically appropriate for tidal analysis and exercise both
the scalar and vector UTide paths.

The total file size is not the amount of data consumed by these tests. In raw
32-bit values, `zeta` is about 224 MB and `ua` plus `va` are about 863 MB. Most of
the 25.78 GB file consists of unrelated three-dimensional fields. Reports must
therefore include the variable, shape, and bytes read rather than presenting the
container's size as the analyzed data volume.

Do not copy this fixture into the RUTide repository. The workspace filesystem was
95% full with about 108 GB available during planning. Treat the file as read-only,
and identify it in each run by path, size, modification time, header metadata, and
eventually a checksum.

## Machine under test

The initial machine observed during planning has:

- 2 x AMD EPYC 7713 processors;
- 128 physical CPU cores, one hardware thread per core;
- AVX2 and FMA support;
- 251 GiB RAM;
- one NUMA node exposed by the Xen virtual machine; and
- the data and workspace on the same mounted filesystem.

The machine was not idle when inspected. Every publishable run must capture CPU
model/topology, memory, virtualization, kernel, compiler flags, CPU affinity,
thread counts, and competing system load. Benchmarks should run on an otherwise
quiet host. The Rust toolchain and dedicated Python benchmark environment were not
installed or selected at planning time, so their versions remain to be pinned.

## What is being compared

Three implementations are needed to make the decision fair:

1. **Canonical Python:** an ordinary loop over `utide.solve`, one spatial series
   per call. This is the compatibility reference, not necessarily the strongest
   performance competitor.
2. **Practical Python:** the same calls distributed across a fixed process pool,
   with BLAS thread counts controlled to avoid oversubscription. Process count is
   swept and the best stable configuration is reported.
3. **RUTide:** single-thread and parallel executions of the Rust implementation,
   using exactly the same observations, latitude, constituent set, options, and
   output requirements.

Report speedup against both Python modes. Comparing a 128-core Rust run only with
single-process Python would measure parallelism and implementation together and
would overstate the case for a rewrite.

## Benchmark layers

Keep input costs separate from harmonic-analysis costs.

### Layer A: solve-only

Load and normalize the selected arrays before starting the timer. Time only
harmonic analysis and result construction. This reveals computational scaling and
is the main measurement for optimization work.

### Layer B: application throughput

Time opening NetCDF, reading the required coordinates and observations,
normalizing missing values, solving all requested series, and writing or
serializing an agreed minimal result. This is the user-visible measurement and
the primary go/no-go metric.

### Layer C: isolated components

Instrument, without changing results:

- NetCDF read and layout conversion;
- time normalization and valid-value filtering;
- constituent selection;
- astronomical and nodal corrections;
- design-matrix construction;
- least-squares solve;
- confidence intervals and diagnostics; and
- reconstruction, when requested.

Warm-cache and uncontrolled-cache results must be labeled separately. Do not
claim a cold-cache result unless cache state was actually controlled and recorded.

## Workloads

Use deterministic spatial indices shared by every implementation. Include wet,
dry/masked, shallow, deep, low-variance, and energetic locations in correctness
samples.

| Workload | Series | Purpose |
|---|---:|---|
| Smoke | 1 | Startup, profiler, and basic parity |
| Correctness sample | 32 fixed locations | Edge cases and detailed coefficient comparison |
| Scaling | 100, 1,000, 10,000 | Find overheads and parallel saturation |
| Scalar full field | all 75,160 `zeta` nodes | Primary decision workload |
| Vector full field | all 144,860 `(ua, va)` elements | Secondary throughput workload |

Run at least these solver profiles:

- **Full-compatible:** Python defaults unless explicitly frozen: automatic
  constituents, OLS, trend enabled, Greenwich phase, exact nodal corrections,
  linear confidence intervals, and `Rayleigh_min=1`.
- **Core OLS:** the same fit with `conf_int="none"` and diagnostics disabled by
  that option. This isolates the reusable harmonic fit from uncertainty work.
- **Fixed constituents:** an explicitly recorded constituent list. This prevents
  automatic selection from hiding behavioral differences and enables efficient
  reuse across spatial series.

Robust regression, irregular/gappy time, inferred constituents, Monte Carlo
confidence intervals, and reconstruction are required for eventual API coverage,
but they should not block the first performance decision. Add dedicated synthetic
correctness cases for them before claiming broad UTide compatibility.

## The main optimization hypothesis

All locations in an FVCOM output share the same 745 timestamps. Most also share
the same solver options, while latitude varies spatially.

Python UTide's 1-D API repeats setup and solves each location independently. A
Rust field API can instead:

- group locations for which the harmonic basis is identical or reusable;
- cache constituent selection and time-dependent astronomical quantities;
- construct design terms once where mathematically valid;
- use matrix factorizations or batched least-squares operations across many right
  hand sides;
- schedule chunks across cores without per-series Python overhead; and
- stream spatial blocks so memory use is predictable.

Latitude-dependent nodal corrections mean that blindly sharing one complete
design matrix may be incorrect. Reuse must occur only after separating the
time-only and latitude-dependent terms or grouping truly equivalent inputs. Every
optimization is subordinate to parity.

This hypothesis also means the benchmark is evaluating an improved bulk API and
algorithm, not only a language translation. If a batched NumPy/SciPy prototype
captures most of the same gain, that is evidence against maintaining a full Rust
rewrite and in favor of a smaller native kernel or improved Python API.

The public-binding harness now makes this distinction explicit. It measures a
matched Python UTide one-series loop, RUTide one-series loop, and RUTide native
time-major batch for both fitting and reconstruction. This isolates the gain from
the Rust scalar kernel from the larger gain due to shared preparation, native
scheduling, and fewer Python calls. See `benchmarks/README.md` for commands and
the versioned JSON report contract.

## Correctness contract

Python UTide at the pinned revision is the initial behavioral oracle. For each
case, compare:

- selected constituent names and order;
- frequencies and auxiliary metadata;
- scalar amplitude and phase;
- vector major/minor axes, inclination, and phase;
- means and trends;
- confidence intervals and diagnostics when enabled;
- weights for robust fits;
- reconstructed values at original and held-out timestamps; and
- failure behavior for invalid inputs.

Phase error must use circular distance, not ordinary subtraction. Constituents
should be matched by name before comparing numeric arrays. Use both absolute and
relative tolerances, with explicit handling near zero amplitude. Record maximum,
median, and high-percentile errors; do not rely only on a single aggregate norm.
For scalar phase, also compare the equivalent complex coefficient and expand the
angular tolerance by the geometric bound `asin(coefficient tolerance / amplitude)`
near zero, where phase is intrinsically ill-conditioned.

Exact floating-point identity is not required across LAPACK implementations or
parallel reduction orders. Initial acceptance tolerances should be derived from
well-conditioned Python reference cases, then frozen before the full benchmark.
Poorly conditioned or scientifically ambiguous cases must be reported rather than
silently loosened until they pass. Reconstruction error relative to the signal
scale is the final scientific backstop.

## Measurement protocol

For each implementation/profile/workload combination:

1. validate inputs and outputs outside the measured region;
2. perform one unreported warm-up;
3. run at least five measured repetitions, or three for a full run if each is
   expensive;
4. report every sample plus median and spread, not only the best sample;
5. keep affinity, thread counts, allocator, BLAS settings, and cache policy fixed;
6. sweep concurrency separately, including 1, 2, 4, 8, 16, 32, 64, and 128
   workers where supported; and
7. stop scaling when throughput plateaus or regresses, but retain those results.

Capture at minimum:

- wall-clock time;
- series solved per second and sample-values processed per second;
- CPU time and average effective core utilization;
- peak resident memory;
- bytes read/written and observed I/O throughput;
- result size; and
- correctness summary and any rejected series.

Use release builds with native CPU optimization for the production Rust result,
but also report the exact compiler and flags. Control common BLAS environment
variables in Python. A process pool with multithreaded BLAS can otherwise create
severe oversubscription and an invalid comparison.

## Result manifest

Each run should emit a machine-readable manifest alongside a concise table. It
must include:

- UTC timestamp and benchmark-suite Git revision;
- fixture identity and selected variables/indices;
- implementation and dependency versions;
- complete solver options and constituent list;
- hardware/software environment and relevant environment variables;
- warm-up and repetition policy;
- raw timing/resource samples; and
- numeric comparison statistics against the oracle.

Generated benchmark outputs should live outside source control, except for small
curated summaries used to document decisions.

## Decision gates

Proceed incrementally:

1. **Baseline gate:** Python measurements are reproducible and component timings
   identify a meaningful compute bottleneck.
2. **Kernel gate:** a Rust scalar OLS prototype matches the frozen correctness
   contract and demonstrates a credible solve-only advantage.
3. **Bulk gate:** the field API scales across cores and retains its advantage over
   tuned multiprocess Python.
4. **Application gate:** full NetCDF-to-results speedup meets the 3x target without
   excessive memory or operational complexity.
5. **Coverage gate:** only then expand toward robust fitting, confidence modes,
   inference, irregular series, reconstruction, packaging, and Python bindings.

Stop or narrow the project if parity requires prohibitive complexity, if I/O
dominates application time, if optimized Python approaches the Rust throughput,
or if the maintenance cost outweighs the measured operational saving. A valid
outcome may be a small Rust kernel exposed to Python rather than a complete UTide
replacement.

## Coverage increments

The first coverage sequence is complete and retained as separate Git increments:

1. the complete pinned 146-constituent, 162-satellite, and 251-shallow-term
   catalog;
2. explicit dynamic constituent lists;
3. Python-compatible Rayleigh automatic selection;
4. percent-energy diagnostics and ranking;
5. linear amplitude/phase confidence intervals and CI-derived SNR; and
6. exact complete-series reconstruction with explicit constituent, PE, and SNR
   filtering, including held-out-time oracle tests.

Missing-value grouping and scalar/vector current support are now complete. Scalar
and vector fits use per-series valid-time masks, vector fits use the joint
component mask, and equidistant gaps retain Python-compatible colored FFT
behavior. Scalar and vector colored intervals use Lomb–Scargle spectra for truly
irregular timestamps. [`ROADMAP.md`](ROADMAP.md) is the authoritative ordered task
list and acceptance contract for the remaining performance, robust, inference,
Monte Carlo, option-parity, and resource increments. Each expansion remains
subject to the same parity and resource gates above.
