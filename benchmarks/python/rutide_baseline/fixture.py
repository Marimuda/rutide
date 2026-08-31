"""Inspect the external FVCOM fixture and produce a deterministic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from .constants import (
    CORRECTNESS_SERIES_COUNT,
    DEFAULT_FIXTURE,
    DEFAULT_MANIFEST,
    MILLISECONDS_PER_DAY,
    REPOSITORY_ROOT,
    SCHEMA_VERSION,
)

_INSPECTED_VARIABLES = (
    "time",
    "Itime",
    "Itime2",
    "lat",
    "latc",
    "h",
    "zeta",
    "ua",
    "va",
)
_INSPECTED_ATTRIBUTES = (
    "long_name",
    "standard_name",
    "units",
    "positive",
    "coordinates",
    "location",
    "_FillValue",
)


def reconstruct_mjd(itime: np.ndarray, itime2: np.ndarray) -> np.ndarray:
    """Reconstruct precise Modified Julian Dates from FVCOM integer fields."""
    days = np.asarray(itime, dtype=np.float64)
    milliseconds = np.asarray(itime2, dtype=np.float64)
    if days.shape != milliseconds.shape:
        raise ValueError("Itime and Itime2 shapes differ")
    return days + milliseconds / MILLISECONDS_PER_DAY


def deterministic_indices(
    size: int,
    count: int,
    anchors: list[int] | tuple[int, ...] = (),
) -> list[int]:
    """Return sorted, stable indices without relying on a random generator."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 < count <= size:
        raise ValueError("count must be in the inclusive range [1, size]")

    selected = {int(index) for index in anchors}
    if any(index < 0 or index >= size for index in selected):
        raise ValueError("anchor index is out of bounds")
    if len(selected) > count:
        raise ValueError("more unique anchors were provided than requested indices")

    evenly_spaced = np.linspace(0, size - 1, min(count, 16), dtype=np.int64)
    selected.update(int(index) for index in evenly_spaced)

    candidate = 1
    while len(selected) < count:
        selected.add((candidate * 2_654_435_761) % size)
        candidate += 1

    return sorted(selected)[:count]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_digest(values: np.ndarray | np.ma.MaskedArray) -> str:
    """Hash array shape, normalized little-endian values, and mask."""
    masked = np.ma.asarray(values)
    dtype = masked.dtype.newbyteorder("<")
    data = np.ascontiguousarray(masked.filled(0), dtype=dtype)
    mask = np.ascontiguousarray(np.ma.getmaskarray(masked), dtype=np.uint8)

    digest = hashlib.sha256()
    digest.update(dtype.str.encode("ascii"))
    digest.update(json.dumps(data.shape, separators=(",", ":")).encode("ascii"))
    digest.update(data.tobytes(order="C"))
    digest.update(mask.tobytes(order="C"))
    return digest.hexdigest()


def _array_summary(values: np.ndarray | np.ma.MaskedArray) -> dict[str, Any]:
    masked = np.ma.asarray(values)
    compressed = masked.compressed()
    return {
        "dtype": masked.dtype.str,
        "shape": list(masked.shape),
        "masked_values": int(np.ma.count_masked(masked)),
        "finite_values": int(np.isfinite(compressed).sum()),
        "minimum": float(compressed.min()),
        "maximum": float(compressed.max()),
        "sha256": array_digest(masked),
    }


def _variable_metadata(variable: netCDF4.Variable) -> dict[str, Any]:
    attributes = {
        name: _json_value(variable.getncattr(name))
        for name in _INSPECTED_ATTRIBUTES
        if name in variable.ncattrs()
    }
    return {
        "dtype": variable.dtype.str,
        "dimensions": list(variable.dimensions),
        "shape": list(variable.shape),
        "attributes": attributes,
    }


def _relative_fixture_path(path: Path) -> str:
    return os.path.relpath(path, start=REPOSITORY_ROOT)


def inspect_fixture(path: Path) -> dict[str, Any]:
    """Return the canonical manifest for an FVCOM benchmark fixture."""
    resolved = path.resolve(strict=True)
    stat = resolved.stat()

    with netCDF4.Dataset(resolved, mode="r") as dataset:
        missing = [name for name in _INSPECTED_VARIABLES if name not in dataset.variables]
        if missing:
            raise ValueError(f"fixture is missing required variables: {missing}")

        dimensions = {
            name: {
                "length": len(dimension),
                "unlimited": dimension.isunlimited(),
            }
            for name, dimension in sorted(dataset.dimensions.items())
        }
        variables = {
            name: _variable_metadata(dataset.variables[name]) for name in _INSPECTED_VARIABLES
        }
        schema_metadata = {
            "file_format": dataset.file_format,
            "dimensions": dimensions,
            "variables": variables,
        }

        raw_time = np.ma.asarray(dataset.variables["time"][:])
        itime = np.ma.asarray(dataset.variables["Itime"][:])
        itime2 = np.ma.asarray(dataset.variables["Itime2"][:])
        exact_time = reconstruct_mjd(itime, itime2)
        interval_seconds = np.diff(exact_time) * 86_400.0

        depth = np.ma.asarray(dataset.variables["h"][:])
        valid_depth_order = np.ma.asarray(depth).filled(np.inf).argsort(kind="stable")
        quantile_positions = np.linspace(
            0,
            len(valid_depth_order) - 1,
            10,
            dtype=np.int64,
        )
        depth_anchors = [int(valid_depth_order[position]) for position in quantile_positions]
        node_indices = deterministic_indices(
            len(dataset.dimensions["node"]),
            CORRECTNESS_SERIES_COUNT,
            depth_anchors,
        )
        element_indices = deterministic_indices(
            len(dataset.dimensions["nele"]),
            CORRECTNESS_SERIES_COUNT,
        )

        latitude = np.ma.asarray(dataset.variables["lat"][:])
        element_latitude = np.ma.asarray(dataset.variables["latc"][:])
        sampled_zeta = np.ma.asarray(dataset.variables["zeta"][:, node_indices])
        sampled_ua = np.ma.asarray(dataset.variables["ua"][:, element_indices])
        sampled_va = np.ma.asarray(dataset.variables["va"][:, element_indices])

        global_attributes = {
            name: _json_value(dataset.getncattr(name))
            for name in ("title", "source", "Tidal_Forcing")
            if name in dataset.ncattrs()
        }

    raw_interval_seconds = np.diff(np.asarray(raw_time, dtype=np.float64)) * 86_400.0
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture": {
            "path_from_repository": _relative_fixture_path(resolved),
            "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
            "modified_utc": modified,
            "full_file_sha256": None,
            "format": schema_metadata["file_format"],
            "schema_metadata_sha256": _canonical_digest(schema_metadata),
        },
        "global_attributes": global_attributes,
        "dimensions": dimensions,
        "variables": variables,
        "time": {
            "analysis_source": "Itime + Itime2 / 86400000",
            "units": "days since 1858-11-17 00:00:00",
            "samples": int(exact_time.size),
            "first_mjd": float(exact_time[0]),
            "last_mjd": float(exact_time[-1]),
            "nominal_interval_seconds": float(np.round(np.median(interval_seconds))),
            "minimum_interval_seconds": float(interval_seconds.min()),
            "maximum_interval_seconds": float(interval_seconds.max()),
            "utide_detects_reconstructed_time_as_equally_spaced": bool(
                np.var(np.unique(np.diff(exact_time))) < np.finfo(np.float64).eps
            ),
            "utide_detects_float32_time_as_equally_spaced": bool(
                np.var(np.unique(np.diff(np.asarray(raw_time, dtype=np.float64))))
                < np.finfo(np.float64).eps
            ),
            "float32_time_interval_seconds": sorted(
                float(value) for value in np.unique(raw_interval_seconds)
            ),
            "exact_time_sha256": array_digest(exact_time),
            "raw_time_sha256": array_digest(raw_time),
        },
        "correctness_selection": {
            "algorithm": "depth-decile anchors, 16 linear anchors, then multiplicative fill",
            "node_indices": node_indices,
            "element_indices": element_indices,
            "node_depth": _array_summary(depth[node_indices]),
            "node_latitude": _array_summary(latitude[node_indices]),
            "element_latitude": _array_summary(element_latitude[element_indices]),
            "zeta": _array_summary(sampled_zeta),
            "ua": _array_summary(sampled_ua),
            "va": _array_summary(sampled_va),
        },
    }


def write_manifest(manifest: dict[str, Any], destination: Path) -> None:
    """Write a stable, human-readable JSON fixture manifest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse fixture-inspector command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output already matches instead of rewriting it",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run the fixture inspector."""
    args = parse_args(arguments)
    observed = inspect_fixture(args.fixture)

    if args.check:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit(f"fixture manifest is stale: {args.output}")
        print(f"fixture manifest matches: {args.output}")
        return 0

    write_manifest(observed, args.output)
    print(f"wrote fixture manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
