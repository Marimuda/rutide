"""Pinned-Python throughput probe for inferred colored-confidence analysis."""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from .constants import DEFAULT_UTIDE_ROOT
from .runner import load_oracle

_CONSTITUENTS = ["N2", "M2", "S2", "K1", "O1"]
_LATITUDE = 60.957_717_895_507_81
_HEX_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "crates"
    / "rutide-core"
    / "tests"
    / "data"
    / "fvcom_node_0_zeta_f32.hex"
)
_WORKER_STATE: dict[str, Any] | None = None


def _times(sampling: str) -> np.ndarray:
    index = np.arange(745, dtype=np.float64)
    times = 58_113.0 + index / 24.0
    if sampling == "irregular":
        times[1:-1] += 0.002 * np.sin(index[1:-1] * 0.37)
        times[1:-1] += 0.0007 * np.cos(index[1:-1] * 0.11)
    return times


def _scalar_observations(sampling: str) -> np.ndarray:
    bits = np.array(
        [
            int(line, 16)
            for line in _HEX_FIXTURE.read_text(encoding="ascii").splitlines()
            if not line.startswith("#")
        ],
        dtype=np.uint32,
    )
    values = bits.view(np.float32).astype(np.float64)
    if sampling == "irregular":
        values[[0, 137, 411]] = np.nan
    return values


def _vector_observations(times: np.ndarray, sampling: str) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(len(times), dtype=np.float64)
    reference = (times[0] + times[-1]) / 2
    eastward = (
        0.15
        + 0.0008 * (times - reference)
        + 0.42 * np.sin(index / 11)
        + 0.17 * np.cos(index / 37)
        + 0.05 * np.sin(index / 3.7)
    )
    northward = (
        -0.08
        - 0.0003 * (times - reference)
        + 0.31 * np.cos(index / 13)
        - 0.12 * np.sin(index / 29)
        + 0.04 * np.cos(index / 4.1)
    )
    if sampling == "irregular":
        eastward[[0, 137]] = np.nan
        northward[[2, 411]] = np.nan
    return eastward, northward


def _inference(field: str, mode: str) -> dict[str, Any]:
    amplitude_ratios = [0.35, 0.5]
    phase_offsets = [20.0, 45.0]
    if field == "vector":
        amplitude_ratios.extend([0.25, 0.4])
        phase_offsets.extend([-10.0, 30.0])
    return {
        "inferred_names": ["S2", "O1"],
        "reference_names": ["M2", "K1"],
        "amp_ratios": amplitude_ratios,
        "phase_offsets": phase_offsets,
        "approximate": mode == "approximate",
    }


def _set_worker_state(
    oracle: Any,
    field: str,
    inference: Any,
    times: np.ndarray,
    scalar: np.ndarray,
    eastward: np.ndarray,
    northward: np.ndarray,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "oracle": oracle,
        "field": field,
        "times": times,
        "scalar": scalar,
        "eastward": eastward,
        "northward": northward,
        "options": {
            "constit": _CONSTITUENTS,
            "order_constit": _CONSTITUENTS,
            "conf_int": "linear",
            "method": "ols",
            "trend": True,
            "phase": "Greenwich",
            "nodal": True,
            "white": False,
            "verbose": False,
            "epoch": "1858-11-17",
            "infer": inference,
        },
    }


def _solve_series(series: int) -> float:
    if _WORKER_STATE is None:
        raise RuntimeError("inference benchmark worker was not initialized")
    state = _WORKER_STATE
    latitude = _LATITUDE + series * 1e-5
    if state["field"] == "scalar":
        coefficient = state["oracle"].solve(
            state["times"],
            state["scalar"],
            lat=latitude,
            **state["options"],
        )
        return float(np.sum(coefficient.A_ci))
    coefficient = state["oracle"].solve(
        state["times"],
        state["eastward"],
        state["northward"],
        lat=latitude,
        **state["options"],
    )
    return float(np.sum(coefficient.Lsmaj_ci))


class _SerialPool:
    """Small context-compatible map adapter for the one-process baseline."""

    def __enter__(self) -> _SerialPool:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @staticmethod
    def map(function: Any, values: range, *, chunksize: int) -> list[float]:
        del chunksize
        return [function(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("scalar", "vector"), default="scalar")
    parser.add_argument("--sampling", choices=("regular", "irregular"), default="irregular")
    parser.add_argument("--inference-mode", choices=("exact", "approximate"), default="exact")
    parser.add_argument("--series-count", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--utide-root", type=Path, default=DEFAULT_UTIDE_ROOT)
    args = parser.parse_args()
    if (
        args.series_count < 1
        or args.warmups < 0
        or args.repetitions < 1
        or args.workers < 1
        or args.chunk_size < 1
    ):
        parser.error(
            "series-count, repetitions, workers, and chunk-size must be positive; "
            "warmups must be non-negative",
        )

    oracle = load_oracle(args.utide_root)
    bunch = importlib.import_module("utide.utilities").Bunch
    times = _times(args.sampling)
    scalar = _scalar_observations(args.sampling)
    eastward, northward = _vector_observations(times, args.sampling)
    _set_worker_state(
        oracle,
        args.field,
        bunch(_inference(args.field, args.inference_mode)),
        times,
        scalar,
        eastward,
        northward,
    )

    with threadpool_limits(limits=1):
        context = multiprocessing.get_context("fork")
        pool_context = (
            context.Pool(processes=args.workers) if args.workers > 1 else _SerialPool()
        )
        with pool_context as pool:

            def run() -> float:
                return sum(
                    pool.map(
                        _solve_series,
                        range(args.series_count),
                        chunksize=args.chunk_size,
                    ),
                )

            for _ in range(args.warmups):
                run()
            samples = []
            checksums = []
            for repetition in range(args.repetitions):
                start = time.perf_counter()
                checksum = run()
                elapsed = time.perf_counter() - start
                samples.append(elapsed)
                checksums.append(checksum)
                print(
                    f"repetition={repetition} seconds={elapsed:.9f} "
                    f"checksum={checksum:.12e}",
                    flush=True,
                )
    median = float(np.median(samples))
    print(
        json.dumps(
            {
                "field": args.field,
                "sampling": args.sampling,
                "inference_mode": args.inference_mode,
                "series_count": args.series_count,
                "workers": args.workers,
                "chunk_size": args.chunk_size,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "seconds": samples,
                "median_seconds": median,
                "median_series_per_second": args.series_count / median,
                "checksums": checksums,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
