# Python UTide scaling snapshot: 2026-08-31

This is an exploratory scaling snapshot used to select the tuned Python comparison
for the first Rust kernel. It is not the final full-field benchmark.

## Configuration

- RUTide revision: `4ccf75edec4cef4004f567679e532c4cb8e62fd2`
- Python UTide revision: `8fabe121752bc317931472a10a42e306715106de`
- workload: first 1,000 FVCOM `zeta` nodes, 745 hourly observations each
- profile: full-compatible OLS with exact nodal/Greenwich corrections and linear
  confidence intervals
- execution: Linux `fork` process pool, BLAS limited to one thread per process
- chunk size: 16 during the broad sweep; 8 for the repeated 16/32/64-worker runs
- raw run manifests: generated locally under ignored `benchmark-results/`

## Results

| Workers | Median solve time (s) | Throughput (series/s) | Repetitions |
|---:|---:|---:|---:|
| 1 | 28.304 | 35.3 | 1 |
| 2 | 16.221 | 61.6 | 1 |
| 4 | 8.468 | 118.1 | 1 |
| 8 | 4.441 | 225.2 | 1 |
| 16 | 2.456 | 407.1 | 3 |
| 32 | 1.961 | 510.1 | 3 |
| 64 | 2.443 | 409.3 | 3 |
| 128 | 3.570 | 280.1 | 1 |

Every run produced result digest
`a406f0e88c5b26eb31c005b2b27d2a071038ed854c3ba291772a932c83c776dc`.

On this workload, 32 processes are the practical Python baseline. More processes
regress because scheduling, process, memory, and result-collection overhead exceed
the additional parallel work. The 32-process result is about 14.4 times the
one-process throughput, so Rust must be compared with this tuned mode as well as
canonical serial Python.

## Profile evidence

A `cProfile` run over 100 canonical full-compatible solves attributed about 2.82
of 4.67 seconds inside `utide.solve` to `ut_E`/`FUV`, primarily exact
astronomical/nodal basis generation. This supports the planned optimization of
factoring or caching shared time-dependent work across spatial series.
