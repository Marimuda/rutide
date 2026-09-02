# Incremental sigma-layer result output — 2026-09-02

Native sigma-layer vector analyses now write each solved spatial chunk directly
to a temporary NetCDF transaction. Whole-domain coefficient, confidence,
sampling, robust, and reconstruction collections are no longer retained. The
file is closed and read back in bounded blocks to reproduce the established v12
digest byte order, the digest attribute is added, and only then is the temporary
file atomically installed.

## Correctness and transaction contract

The all-options integration fixture covers joint missing masks, robust fitting,
Monte Carlo confidence, complete reconstruction, frequency presentation order,
out-of-order layers/elements, and a three-series chunk crossing a two-element
layer boundary. One-chunk and boundary-crossing outputs are bit-identical. The
layered fixture retains its pre-change digest
`037de642b316829ecbc498e261c5f8e95c73fe880be1dd5a5dd3e092716ca065`.
The real all-layer field likewise retained
`94940a8ab5b77529775dad46259dff24983bdce49b7b4ba71b1df49f958fb505`.
The 96-series, three-layer real-FVCOM comparator was rerun against pinned Python
UTide and passed. Its largest component-coefficient error remained
`1.0797e-12` against a `5e-12` tolerance, and its digest remained
`80566f6facd6f186729594d363ff2765beadda5bed2c8d517f720becf0650c18`.

Output remains transactional: errors close and remove the temporary sibling;
`result_sha256` is attached only after every chunk has been written and the
bounded read-back succeeds; the destination is then replaced by one rename.
The NetCDF attribute and JSON report field `result_output` distinguish
`incremental` sigma-layer output from the existing `buffered` depth-averaged
path.

## Full-field memory result

The measurement used the same 25,778,391,080-byte `frs2f_0001.nc` fixture,
release build with `-C target-cpu=native`, 64 workers, ten native layers,
1,448,600 series, and explicit `M2,S2,N2,K1,O1` OLS analysis as the prior
depth-resolved snapshot. GNU `time -v` measured the complete process.

| Path | Series | Chunks | Report total | Result processing | Output | Peak RSS | Output size |
|---|---:|---:|---:|---:|---:|---:|---:|
| previous buffered all-layer | 1,448,600 | 33 | 75.28 s | 2.82 s | 1.13 s | 2,460,120 KiB | 744,030,280 B |
| incremental all-layer | 1,448,600 | 33 | 50.79 s | 2.83 s | 1.58 s | 1,182,068 KiB | 744,030,375 B |
| incremental layer 0 control | 144,860 | 4 | 5.20 s | 0.28 s | 0.16 s | 1,180,260 KiB | 76,515,495 B |

Peak RSS fell by 1,278,052 KiB, or **52.0%**, from 2.35 GiB to 1.13 GiB. The
ten-layer run peaked only 1,808 KiB above the one-layer control, demonstrating
that retained results no longer scale with the number of fitted layers. The
extra 95 output bytes are the new `result_output` attribute.

The wall-time rows are not a controlled before/after speed comparison: the
previous all-layer read took 49.37 s, while this warm-cache observation took
23.14 s. The directly relevant added work is small. Result processing remained
2.83 s and output rose from 1.13 s to 1.58 s, so streaming plus the digest-safe
read-back added about 0.45 s on this 710 MiB result. The digest and solver output
were unchanged.

Automatic robust sigma-layer chunks additionally account for the two
time-major robust diagnostic rows retained beside input or reconstruction.
Their series count is halved relative to OLS when necessary, while the reported
`maximum_observation_buffer_bytes` continues to describe only promoted `u`/`v`
source arrays. Explicit `--chunk-series` remains an intentional user override.
