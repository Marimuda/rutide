# Installed-package real-data acceptance: 2026-09-03

The locally built `rutide-0.2.0-cp39-abi3-linux_x86_64.whl` passes complete
public-API fit, reconstruction, save, reload, and reconstructed-result checks on
both a genuine observational ADCP deployment and the largest FVCOM fixture.

## Observational ADCP

The input is the freely available NOAA/NDBC OceanSITES CCE1 mooring record,
`OS_CCE1_11_D_ADCP.nc`. It contains manually reviewed eastward and northward
currents from a downward-looking 75 kHz RDI Workhorse ADCP operated by Scripps
Institution of Oceanography. The retained file covers 2017-11-04 through
2018-11-14 at latitude 33.4592 degrees and depths from 14.3 to 554.3 m.

- source URL:
  `https://dods.ndbc.noaa.gov/thredds/fileServer/data/oceansites/DATA/CCE1/OS_CCE1_11_D_ADCP.nc`;
- local file: 15,861,060 bytes, SHA-256
  `47214feeaa974f3c4dd5e6dc3cabe2189f4cce8e5cafe13d6559b0691bfbbde7`;
- 8,967 timestamps and all 55 sufficiently populated depth cells;
- 489,386 jointly valid eastward/northward observations;
- M2, S2, N2, K2, K1, O1, P1, and Q1 with colored linear confidence;
- 16 workers: 2.788 s solve, 0.096 s reconstruction;
- coefficient archive: 823,835 bytes, 0.312 s save, 0.045 s load;
- saved/restored reconstruction: bitwise identical.

## FVCOM simulation

The 25,778,391,080-byte CDF-2 `frs2f_0001.nc` fixture contains 745 hourly
records and 144,860 current elements. The installed-package acceptance uses
4,096 deterministic elements spanning the complete domain, or 3,051,520 joint
current observations, and the established M2/S2/N2/K1/O1 profile.

- bounded contiguous classic-NetCDF input: 0.742 s;
- 16-worker solve: 0.291 s, or 14,084 series/s;
- reconstruction: 0.199 s, or 20,577 series/s;
- coefficient archive: 2,178,694 bytes;
- save: 0.822 s; load: 0.658 s;
- saved/restored reconstruction: bitwise identical;
- complete process: 7.27 s wall time and 587,568 KiB peak RSS.

Raw external files and generated JSON remain outside Git. The harness is
`rutide_baseline.real_data_acceptance`; it records source metadata, timings,
digests, environment, and persistence checks in a machine-readable report.
