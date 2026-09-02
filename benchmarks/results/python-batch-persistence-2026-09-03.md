# Python batch-persistence optimization: 2026-09-03

Profiling the installed-package 4,096-series FVCOM acceptance run showed that
the harmonic solve was already fast (approximately 0.3 s), but restoring the
compressed schema-1 archive took 47.59 s. The original NPZ layout created a
separate ZIP member for every small array nested under every batch solution.
Opening and decompressing thousands of entries, rather than numerical model
restoration, dominated the end-to-end workflow.

The optimized writer coalesces contiguous numeric arrays into typed blobs of at
most 64 MiB. JSON markers retain each array's blob, offset, byte length, dtype,
and shape. The loader validates all bounds and dtypes, decompresses each blob
once, and gives the native restore path contiguous views. It still accepts the
original per-array schema-1 markers, so no schema increment or `0.2.x` archive
migration is required.

## Matched result

| Metric | Per-array NPZ | Packed-blob NPZ | Improvement |
|---|---:|---:|---:|
| Save | 6.414 s | 0.822 s | 7.80x faster |
| Load/native restore | 47.594 s | 0.658 s | 72.28x faster |
| Archive size | 16,593,264 B | 2,178,694 B | 7.62x smaller |

Both layouts were produced from the same 4,096-series fit. Coefficient digest
`773cb8759908b883a9a873b3931013cf89ed629858c26037f831a165c5b1ad72`
and reconstruction digest
`da9d3fd8b76428f78f77665b2430900e63333f10cb469c2027374275d14803d8`
are unchanged, and reconstruction after restore is bitwise identical. A legacy
schema-1 archive written by the pre-optimization installed wheel was also loaded
and reconstructed successfully by the new reader.
