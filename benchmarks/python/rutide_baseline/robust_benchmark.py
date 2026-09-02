"""Pinned-Python throughput probe for robust colored-confidence analysis."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from .constants import DEFAULT_UTIDE_ROOT
from .runner import load_oracle

_CONSTITUENTS = ["M2", "S2", "N2", "K1", "O1"]
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


def _times() -> np.ndarray:
    return 58_113.0 + np.arange(745, dtype=np.float64) / 24.0


def _scalar_observations() -> np.ndarray:
    bits = np.array(
        [
            int(line, 16)
            for line in _HEX_FIXTURE.read_text(encoding="ascii").splitlines()
            if not line.startswith("#")
        ],
        dtype=np.uint32,
    )
    values = bits.view(np.float32).astype(np.float64)
    values[[71, 218, 503]] += [5.0, -4.0, 6.0]
    return values


def _vector_observations(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    eastward[71] += 5.0
    northward[218] -= 4.0
    eastward[503] += 4.0
    northward[503] += 3.0
    return eastward, northward


def _set_worker_state(
    oracle: Any,
    field: str,
    confidence: str,
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
            "conf_int": "linear" if confidence == "linear" else "MC",
            "MC_n": 200,
            "method": "robust",
            "trend": True,
            "phase": "Greenwich",
            "nodal": True,
            "white": False,
            "verbose": False,
            "epoch": "1858-11-17",
        },
    }


def _solve_series(series: int) -> tuple[float, int]:
    if _WORKER_STATE is None:
        raise RuntimeError("robust benchmark worker was not initialized")
    state = _WORKER_STATE
    latitude = _LATITUDE + series * 1e-5
    if state["field"] == "scalar":
        coefficient = state["oracle"].solve(
            state["times"],
            state["scalar"],
            lat=latitude,
            **state["options"],
        )
        return float(coefficient.A_ci[0]), int(coefficient.rf.iterations)
    coefficient = state["oracle"].solve(
        state["times"],
        state["eastward"],
        state["northward"],
        lat=latitude,
        **state["options"],
    )
    return float(coefficient.Lsmaj_ci[0]), int(coefficient.rf.iterations)


class _SerialPool:
    """Small context-compatible map adapter for the one-process baseline."""

    def __enter__(self) -> _SerialPool:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @staticmethod
    def map(function: Any, values: range, *, chunksize: int) -> list[tuple[float, int]]:
        del chunksize
        return [function(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("scalar", "vector"), default="scalar")
    parser.add_argument(
        "--confidence",
        choices=("linear", "monte-carlo"),
        default="linear",
    )
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
    times = _times()
    scalar = _scalar_observations()
    eastward, northward = _vector_observations(times)
    _set_worker_state(
        oracle,
        args.field,
        args.confidence,
        times,
        scalar,
        eastward,
        northward,
    )

    with threadpool_limits(limits=1):
        context = multiprocessing.get_context("fork")
        pool_context = context.Pool(processes=args.workers) if args.workers > 1 else _SerialPool()
        with pool_context as pool:

            def run() -> tuple[float, int, int, int]:
                results = pool.map(
                    _solve_series,
                    range(args.series_count),
                    chunksize=args.chunk_size,
                )
                iterations = [result[1] for result in results]
                return (
                    sum(result[0] for result in results),
                    sum(iterations),
                    min(iterations),
                    max(iterations),
                )

            for _ in range(args.warmups):
                run()
            samples = []
            checksums = []
            iteration_sums = []
            iteration_minimum = 0
            iteration_maximum = 0
            for repetition in range(args.repetitions):
                start = time.perf_counter()
                checksum, iteration_sum, iteration_minimum, iteration_maximum = run()
                elapsed = time.perf_counter() - start
                samples.append(elapsed)
                checksums.append(checksum)
                iteration_sums.append(iteration_sum)
                print(
                    f"repetition={repetition} seconds={elapsed:.9f} "
                    f"checksum={checksum:.12e} iteration_sum={iteration_sum} "
                    f"iteration_min={iteration_minimum} iteration_max={iteration_maximum}",
                    flush=True,
                )
    median = float(np.median(samples))
    print(
        json.dumps(
            {
                "field": args.field,
                "confidence": args.confidence,
                "monte_carlo_realizations": (200 if args.confidence == "monte-carlo" else 0),
                "series_count": args.series_count,
                "workers": args.workers,
                "chunk_size": args.chunk_size,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "seconds": samples,
                "median_seconds": median,
                "median_series_per_second": args.series_count / median,
                "checksums": checksums,
                "iteration_sums": iteration_sums,
                "iteration_mean": iteration_sums[-1] / args.series_count,
                "iteration_min": iteration_minimum,
                "iteration_max": iteration_maximum,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
