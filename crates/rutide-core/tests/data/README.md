# Cross-language test data

`fvcom_node_0_zeta_f32.hex` contains the 745 values from `zeta[:, 0]` in the
manifested FVCOM fixture. Each line is the exact hexadecimal bit pattern of one
IEEE-754 `float32`; comments begin with `#`.

The integration test reconstructs time from hourly `Itime`/`Itime2` values and
compares against Python UTide revision
`8fabe121752bc317931472a10a42e306715106de`. The oracle profile is:

- constituents: M2, S2, N2, K1, O1;
- OLS with mean and trend;
- raw phase;
- nodal corrections disabled; and
- confidence intervals disabled.

The Python result artifact used to freeze the expected constants had SHA-256
digest `d69134442fa79591e2a88c7789c3f72f9711f5668d94181f22f2a8cc7515d23c`.
Frequencies, amplitudes, phases, mean, and slope are recorded at full displayed
precision in `../python_oracle_fixed_raw.rs`.

The data file is intentionally a compact single-series fixture. Full FVCOM data
remain external and are validated through `benchmarks/fixtures/`.
