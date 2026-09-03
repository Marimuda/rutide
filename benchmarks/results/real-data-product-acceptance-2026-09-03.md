# Temporal-product real-data acceptance: 2026-09-03

The freshly built `rutide-0.3.0-cp39-abi3-linux_x86_64.whl` passed the public
Python fit, diagnostics, reconstruction, persistence, and optional pre-filter
control workflow on an observational ADCP deployment and a domain-spanning
selection from the largest local FVCOM fixture.

This is an installed-wheel test. The imported native extension came from an
isolated temporary installation, not the editable repository checkout.

## Artifact and environment

- wheel: 1,965,842 bytes;
- wheel SHA-256:
  `0d20f8fb738d3cefea7f543e57f088d7e3168aa4b30710389afcff81d83a5b1c`;
- source base when built: `5120a91d5744`;
- Python 3.10.12, NumPy 2.2.6, netCDF4 1.7.2;
- Linux 6.8 x86-64, 128 logical CPUs, 16 RUTide workers;
- complete acceptance process: 7.72 seconds wall and 603,980 KiB maximum
  resident set size.

The elapsed time includes NetCDF input, two principal fits with diagnostics,
representative unity-response control fits, reconstruction, archive save/load,
and restored reconstruction. It is acceptance timing, not a clean kernel
benchmark.

## Observational ADCP

The source is the public NOAA/NDBC OceanSITES CCE1 record
`OS_CCE1_11_D_ADCP.nc`, SHA-256
`47214feeaa974f3c4dd5e6dc3cabe2189f4cce8e5cafe13d6559b0691bfbbde7`.
Its `UCUR` and `VCUR` variables both declare `m/s`.

- 8,967 timestamps over 375.63 days and all 55 retained depth cells;
- depths 14.3–554.3 m, 489,386 jointly valid vector observations, 0.7703%
  jointly missing;
- nominal one-hour sampling; the largest recorded gap is 39.59 hours;
- M2, S2, N2, K2, K1, O1, P1, and Q1 with Greenwich/nodal astronomy,
  trend, colored linear confidence, and constituent diagnostics;
- fit: 2.717 seconds; reconstruction: 0.059 seconds;
- all 55 series pass `SNRallc / K > 1`;
- basis condition number: 3.592–3.764, median 3.593;
- all 385 adjacent constituent pairs pass RR and RNM, and all have
  `Corrmax <= 0.2`; maximum observed Corrmax is 0.0778;
- 397 of 440 constituent/depth tests have `SNR >= 2`;
- median `PTVallc` is 6.86% and median `PTVsnrc` is 6.84%.

The modest percent tidal variance is scientifically plausible for this
observational current record and is not treated as a software failure. It
demonstrates why significance, identifiability, and explanatory power need to
be reported separately.

## FVCOM simulation

The source is the 25,778,391,080-byte CDF-2 `frs2f_0001.nc` FVCOM 5.1 output.
The deterministic 4,096-element selection spans element 0 through 144,859.
Both depth-averaged current components declare `meters s-1`.

- 745 hourly records over 31.0 days and 3,051,520 vector observations;
- M2, S2, N2, K1, and O1 with the same analysis settings;
- bounded NetCDF input: 0.741 seconds;
- diagnostic fit: 0.383 seconds, or 10,702 series/s;
- reconstruction: 0.105 seconds, or 39,043 series/s;
- all 4,096 series pass `SNRallc / K > 1`;
- basis condition number: 3.80108–3.80113;
- all 16,384 adjacent pairs pass conventional RR and have `Corrmax <= 0.2`;
  maximum observed Corrmax is 0.0966;
- 16,369 of 16,384 pair/series cases pass RNM. The 15 failures are retained as
  scientifically useful weak-signal warnings rather than silently changing the
  constituent model;
- 19,751 of 20,480 constituent/element tests have `SNR >= 2`;
- median `PTVallc` and `PTVsnrc` are both 79.87%.

This depth-averaged test complements the previously recorded real-fixture
[native sigma-layer](depth-resolved-fvcom-2026-09-02.md) and
[fixed-depth](fixed-depth-fvcom-2026-09-02.md) validations.

## Native FVCOM command integration

The release-mode `rutide` executable was also run against four nodes and four
elements from the same FVCOM file with linear white confidence, constituent
diagnostics, reconstruction, and the unity response enabled through
`--prefilter-response`.

- scalar schema 18 propagated source `zeta` units `meters` to amplitude, mean,
  trend, diagnostic variance, and reconstruction fields;
- vector schema 17 required matching `ua`/`va` units `meters s-1` and propagated
  them to ellipses, Cartesian coefficients, affine terms, diagnostics, and both
  reconstruction components;
- both NetCDF products and JSON reports retain the absolute source path,
  25,778,391,080-byte source size, exact component-variable names, complete
  response table, fallback policy, analysis options, and schema/version;
- scalar digest:
  `0bd03be0417355f5670293b04eac4b59eea944213d7ef4d5ac80fb1c105e298e`;
- vector digest:
  `e8fb575c021f69ec5552ac07c1b2501ea31129762f85a210955fe533d916419c`.

The scalar and vector processes completed in 0.135 and 0.168 seconds,
respectively. These small selections are integration checks, not throughput
claims.

## Persistence and pre-filter gates

For both sources, saving and reloading the coefficient batch produces a
bitwise-identical reconstruction. A unity real transfer response was then
applied to four representative series from each source. Ellipse coefficients,
confidence values, SNR, condition diagnostics, and percent tidal variance are
bitwise identical to the uncorrected batch result.

The unity control is deliberately a software invariant, not a claim that these
source currents were filtered. Non-unity correction is covered by equation and
synthetic recovery tests. A scientifically honest field validation of a
non-unity response requires the documented transfer function from the ADCP or
model post-processing chain; RUTide does not invent one.

## Reproduction

The harness is `rutide_baseline.real_data_acceptance` and report schema 2. It
records source metadata, units, sampling/gaps, missing fraction, diagnostic
ranges and threshold counts, timings, array digests, and persistence controls.
Generated JSON and external fixtures remain outside Git.

```console
PYTHONPATH=/tmp/isolated-rutide-wheel \
uv run --project benchmarks/python --locked --no-sync \
  python -m rutide_baseline.real_data_acceptance \
  --adcp /path/to/OS_CCE1_11_D_ADCP.nc \
  --fvcom /path/to/frs2f_0001.nc \
  --fvcom-series 4096 --workers 16 --memory-limit-mb 512 \
  --expected-version 0.3.0 \
  --output benchmark-results/real-data-product-acceptance.json
```
