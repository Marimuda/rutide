"""Shared immutable settings for the Python reference benchmark."""

from pathlib import Path

SCHEMA_VERSION = 1
EXPECTED_UTIDE_REVISION = "8fabe121752bc317931472a10a42e306715106de"
MILLISECONDS_PER_DAY = 86_400_000.0
CORRECTNESS_SERIES_COUNT = 32
FIXED_CONSTITUENTS = ("M2", "S2", "N2", "K1", "O1")

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = (
    REPOSITORY_ROOT / "../projects/fvcom/claude_scratchpad/baroclinic_vikc1701/run/frs2f_0001.nc"
).resolve()
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks/fixtures/fvcom-baroclinic-vikc1701-frs2f-0001.json"
DEFAULT_UTIDE_ROOT = REPOSITORY_ROOT / "UTide"
