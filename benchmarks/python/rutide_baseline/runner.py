"""Canonical and multiprocessing Python UTide benchmark runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import resource
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .constants import (
    DEFAULT_FIXTURE,
    DEFAULT_MANIFEST,
    DEFAULT_UTIDE_ROOT,
    EXPECTED_UTIDE_REVISION,
    FIXED_CONSTITUENTS,
    REPOSITORY_ROOT,
    SCHEMA_VERSION,
)
from .fixture import array_digest, reconstruct_mjd

_WORKER_STATE: dict[str, Any] | None = None


def _git_output(arguments: list[str], directory: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_oracle(utide_root: Path) -> Any:
    """Verify and import the exact clean Python UTide oracle checkout."""
    root = utide_root.resolve(strict=True)
    revision = _git_output(["rev-parse", "HEAD"], root)
    if revision != EXPECTED_UTIDE_REVISION:
        raise RuntimeError(
            f"UTide revision mismatch: expected {EXPECTED_UTIDE_REVISION}, got {revision}"
        )
    if _git_output(["status", "--porcelain"], root):
        raise RuntimeError(f"UTide oracle checkout is dirty: {root}")

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    for module_name in list(sys.modules):
        if module_name == "utide" or module_name.startswith("utide."):
            del sys.modules[module_name]
    oracle = importlib.import_module("utide")
    imported_path = Path(oracle.__file__).resolve()
    if not imported_path.is_relative_to(root):
        raise RuntimeError(f"imported UTide from unexpected path: {imported_path}")
    return oracle


def _profile_options(profile: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "constit": "auto",
        "conf_int": "linear",
        "method": "ols",
        "trend": True,
        "phase": "Greenwich",
        "nodal": True,
        "infer": None,
        "order_constit": "PE",
        "Rayleigh_min": 1.0,
        "white": False,
        "verbose": False,
        "epoch": "1858-11-17",
    }
    if profile == "core-ols":
        options["conf_int"] = "none"
    elif profile == "fixed-constituents":
        options["conf_int"] = "none"
        options["constit"] = list(FIXED_CONSTITUENTS)
    elif profile == "fixed-raw":
        options["conf_int"] = "none"
        options["constit"] = list(FIXED_CONSTITUENTS)
        options["nodal"] = False
        options["phase"] = "raw"
    elif profile != "full-compatible":
        raise ValueError(f"unknown profile: {profile}")
    return options


def _json_number(value: Any) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return number


def _json_numbers(values: Any) -> list[float | str]:
    return [_json_number(value) for value in np.asarray(values).ravel()]


def _coefficient_summary(
    coefficient: Any,
    spatial_index: int,
    latitude: float,
    valid_observations: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "spatial_index": spatial_index,
        "latitude_degrees_north": latitude,
        "valid_observations": valid_observations,
        "constituents": [str(value) for value in coefficient.name.tolist()],
        "frequency_cph": _json_numbers(coefficient.aux.frq),
        "amplitude": _json_numbers(coefficient.A),
        "phase_degrees": _json_numbers(coefficient.g),
        "mean": _json_number(coefficient.mean),
        "slope_per_day": _json_number(coefficient.slope),
    }
    for output_name, coefficient_name in (
        ("amplitude_ci", "A_ci"),
        ("phase_ci_degrees", "g_ci"),
        ("percent_energy", "PE"),
        ("signal_to_noise", "SNR"),
    ):
        if coefficient_name in coefficient:
            summary[output_name] = _json_numbers(coefficient[coefficient_name])
    return summary


def _summary_digest(summary: dict[str, Any]) -> str:
    payload = json.dumps(
        summary,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def aggregate_result_digest(results: list[tuple[int, str, Any]]) -> str:
    """Combine per-series digests in spatial-index order."""
    digest = hashlib.sha256()
    for spatial_index, series_digest, _ in sorted(results):
        digest.update(struct.pack("<Q", spatial_index))
        digest.update(bytes.fromhex(series_digest))
    return digest.hexdigest()


def _solve_position(position: int, state: dict[str, Any]) -> tuple[int, str, Any]:
    oracle = state["oracle"]
    observations = state["observations"][:, position]
    spatial_index = int(state["indices"][position])
    latitude = float(state["latitudes"][position])
    coefficient = oracle.solve(
        state["time"],
        observations,
        lat=latitude,
        **state["options"],
    )
    summary = _coefficient_summary(
        coefficient,
        spatial_index,
        latitude,
        int(np.isfinite(observations).sum()),
    )
    retained = summary if spatial_index in state["retained_indices"] else None
    return spatial_index, _summary_digest(summary), retained


def _solve_worker_batch(bounds: tuple[int, int]) -> list[tuple[int, str, Any]]:
    if _WORKER_STATE is None:
        raise RuntimeError("worker state is not initialized")
    start, stop = bounds
    with threadpool_limits(limits=_WORKER_STATE["blas_threads"]):
        return [_solve_position(position, _WORKER_STATE) for position in range(start, stop)]


def _solve_once(
    state: dict[str, Any],
    mode: str,
    workers: int,
    chunk_size: int,
    blas_threads: int,
) -> list[tuple[int, str, Any]]:
    if mode == "canonical":
        with threadpool_limits(limits=blas_threads):
            return [_solve_position(position, state) for position in range(len(state["indices"]))]

    if mode != "multiprocessing":
        raise ValueError(f"unknown execution mode: {mode}")
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("multiprocessing baseline requires the Linux fork start method")

    global _WORKER_STATE
    state["blas_threads"] = blas_threads
    _WORKER_STATE = state
    bounds = [
        (start, min(start + chunk_size, len(state["indices"])))
        for start in range(0, len(state["indices"]), chunk_size)
    ]
    context = multiprocessing.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            batches = executor.map(_solve_worker_batch, bounds)
            return [result for batch in batches for result in batch]
    finally:
        _WORKER_STATE = None


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported fixture-manifest schema: {manifest.get('schema_version')}")
    return manifest


def _verify_fixture_identity(path: Path, manifest: dict[str, Any]) -> None:
    stat = path.stat()
    expected = manifest["fixture"]
    if stat.st_size != expected["size_bytes"] or stat.st_mtime_ns != expected["modified_time_ns"]:
        raise RuntimeError("fixture size or modification time differs from its manifest")


def _workload_indices(
    workload: str,
    series_count: int | None,
    manifest: dict[str, Any],
) -> np.ndarray:
    node_count = int(manifest["dimensions"]["node"]["length"])
    correctness = manifest["correctness_selection"]["node_indices"]
    if workload == "smoke":
        return np.asarray(correctness[:1], dtype=np.int64)
    if workload == "correctness":
        return np.asarray(correctness, dtype=np.int64)
    if workload == "scalar-full":
        if series_count is not None:
            raise ValueError("--series-count is incompatible with scalar-full")
        return np.arange(node_count, dtype=np.int64)
    if workload == "scaling":
        if series_count is None:
            raise ValueError("scaling workload requires --series-count")
        if not 0 < series_count <= node_count:
            raise ValueError("--series-count is outside the node dimension")
        return np.arange(series_count, dtype=np.int64)
    raise ValueError(f"unknown workload: {workload}")


def _load_scalar_inputs(path: Path, indices: np.ndarray) -> dict[str, Any]:
    prefix_selection = np.array_equal(indices, np.arange(len(indices), dtype=np.int64))
    selector: slice | np.ndarray = slice(0, len(indices)) if prefix_selection else indices
    with netCDF4.Dataset(path, mode="r") as dataset:
        exact_time = reconstruct_mjd(dataset.variables["Itime"][:], dataset.variables["Itime2"][:])
        latitudes = np.asarray(dataset.variables["lat"][selector], dtype=np.float64)
        values = np.ma.asarray(dataset.variables["zeta"][:, selector])
    observations = np.asarray(values.filled(np.nan), dtype=np.float64)
    return {
        "time": np.asarray(exact_time, dtype=np.float64),
        "latitudes": latitudes,
        "observations": observations,
        "source_observations_sha256": array_digest(values),
        "indices": indices,
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _environment_manifest(oracle: Any) -> dict[str, Any]:
    repository_revision = _git_output(["rev-parse", "HEAD"], REPOSITORY_ROOT)
    repository_dirty = bool(_git_output(["status", "--porcelain"], REPOSITORY_ROOT))
    relevant_environment = {
        name: os.environ[name]
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if name in os.environ
    }
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "packages": {
            name: _package_version(name) for name in ("netCDF4", "numpy", "scipy", "threadpoolctl")
        },
        "netcdf_c": netCDF4.getlibversion(),
        "hdf5": netCDF4.__hdf5libversion__,
        "thread_pools": threadpool_info(),
        "environment": relevant_environment,
        "rutide_repository_revision": repository_revision,
        "rutide_repository_dirty": repository_dirty,
        "utide_revision": EXPECTED_UTIDE_REVISION,
        "utide_import_path": str(Path(oracle.__file__).resolve()),
    }


def _peak_rss_kib() -> dict[str, int]:
    return {
        "self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one configured benchmark and return its result manifest."""
    oracle = load_oracle(args.utide_root)
    fixture = args.fixture.resolve(strict=True)
    manifest = _load_manifest(args.manifest)
    _verify_fixture_identity(fixture, manifest)
    indices = _workload_indices(args.workload, args.series_count, manifest)

    load_start = time.perf_counter()
    inputs = _load_scalar_inputs(fixture, indices)
    load_seconds = time.perf_counter() - load_start
    if array_digest(inputs["time"]) != manifest["time"]["exact_time_sha256"]:
        raise RuntimeError("reconstructed fixture time differs from its manifest")
    if (
        args.workload == "correctness"
        and inputs["source_observations_sha256"]
        != manifest["correctness_selection"]["zeta"]["sha256"]
    ):
        raise RuntimeError("correctness observations differ from the fixture manifest")
    retained_indices = {int(index) for index in indices[: min(3, len(indices))]}
    state = {
        **inputs,
        "oracle": oracle,
        "options": _profile_options(args.profile),
        "retained_indices": retained_indices,
    }

    for _ in range(args.warmups):
        warmup_state = {
            **state,
            "latitudes": state["latitudes"][:1],
            "observations": state["observations"][:, :1],
            "indices": state["indices"][:1],
        }
        _solve_once(warmup_state, "canonical", 1, 1, args.blas_threads)

    measurements = []
    reference_digest = None
    retained_summaries: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        solve_start = time.perf_counter()
        results = _solve_once(
            state,
            args.mode,
            args.workers,
            args.chunk_size,
            args.blas_threads,
        )
        solve_seconds = time.perf_counter() - solve_start
        result_digest = aggregate_result_digest(results)
        if reference_digest is None:
            reference_digest = result_digest
            retained_summaries = [result[2] for result in sorted(results) if result[2] is not None]
        elif result_digest != reference_digest:
            raise RuntimeError("result digest changed between repetitions")
        measurements.append(
            {
                "repetition": repetition,
                "solve_seconds": solve_seconds,
                "series_per_second": len(indices) / solve_seconds,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "implementation": "python-utide",
        "layer": "solve-only",
        "configuration": {
            "mode": args.mode,
            "profile": args.profile,
            "workload": args.workload,
            "series": len(indices),
            "workers": args.workers,
            "chunk_size": args.chunk_size,
            "blas_threads_per_process": args.blas_threads,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "solver_options": state["options"],
        },
        "fixture": {
            "path": str(fixture),
            "manifest": str(args.manifest.resolve()),
            "time_sha256": array_digest(state["time"]),
            "source_observations_sha256": state["source_observations_sha256"],
            "observations_sha256": array_digest(state["observations"]),
            "indices_sha256": array_digest(state["indices"]),
            "load_seconds": load_seconds,
        },
        "environment": _environment_manifest(oracle),
        "measurements": measurements,
        "peak_rss_kib": _peak_rss_kib(),
        "result_sha256": reference_digest,
        "sample_results": retained_summaries,
    }


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark-runner command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("canonical", "multiprocessing"),
        default="canonical",
    )
    parser.add_argument(
        "--profile",
        choices=("full-compatible", "core-ols", "fixed-constituents", "fixed-raw"),
        default="full-compatible",
    )
    parser.add_argument(
        "--workload",
        choices=("smoke", "correctness", "scaling", "scalar-full"),
        default="smoke",
    )
    parser.add_argument("--series-count", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--utide-root", type=Path, default=DEFAULT_UTIDE_ROOT)
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args(arguments)
    for name in ("workers", "chunk_size", "blas_threads", "repetitions"):
        if getattr(parsed, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if parsed.warmups < 0:
        parser.error("--warmups must not be negative")
    if parsed.mode == "canonical" and parsed.workers != 1:
        parser.error("canonical mode requires --workers 1")
    return parsed


def _default_output() -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPOSITORY_ROOT / "benchmark-results" / f"python-baseline-{timestamp}.json"


def main(arguments: list[str] | None = None) -> int:
    """Run a Python UTide reference benchmark."""
    args = parse_args(arguments)
    result = run_benchmark(args)
    output = args.output or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    median_seconds = float(np.median([item["solve_seconds"] for item in result["measurements"]]))
    print(f"wrote benchmark result: {output}")
    print(
        f"{result['configuration']['series']} series; median solve "
        f"{median_seconds:.6f} s; sha256 {result['result_sha256']}"
    )
    return 0
