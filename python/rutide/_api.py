"""Public Python API; numerical work is delegated to :mod:`rutide._native`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any, Literal, Union

import numpy as np
import numpy.typing as npt

from . import _native

ArrayLike = npt.ArrayLike
Epoch = Union[str, date, datetime]

_MILLISECONDS_PER_DAY = 86_400_000.0
_MJD_EPOCH = np.datetime64("1858-11-17T00:00:00", "ms")


class Bunch(dict[str, Any]):
    """Dictionary whose keys are also available as attributes, like UTide's Bunch."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class Coefficient(Bunch):
    """Harmonic coefficients and fit metadata returned by :func:`solve`."""

    _fit: _native.Fit

    def __init__(self, fit: _native.Fit) -> None:
        super().__init__(_arrays_from_lists(fit.summary()))
        object.__setattr__(self, "_fit", fit)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_fit":
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)


class CoefficientBatch(Bunch):
    """Time-major multi-series coefficients returned by :func:`solve_many`."""

    _fit: _native.BatchFit

    def __init__(self, fit: _native.BatchFit) -> None:
        values = _arrays_from_lists(fit.summary())
        series = np.arange(fit.series_count, dtype=np.intp)
        series.setflags(write=False)
        values["series"] = series
        values["dims"] = Bunch(
            coefficients=("series", "constituent"),
            frequency=("series", "constituent"),
            ranking=("series", "presentation_rank"),
            series=("series",),
            robust_weights=("retained_time", "series"),
        )
        super().__init__(values)
        object.__setattr__(self, "_fit", fit)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_fit":
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    def to_xarray(self) -> Any:
        """Return an ``xarray.Dataset`` when the optional xarray package is installed."""

        try:
            import xarray as xr
        except ImportError as error:
            raise ImportError(
                "CoefficientBatch.to_xarray() requires the optional xarray package"
            ) from error

        coordinates = {
            "series": self.series,
            "constituent": self.name,
            "presentation_rank": np.arange(self.rank_index.shape[1], dtype=np.intp),
            "retained_time": self.aux.time_position,
        }
        variables: dict[str, Any] = {
            "frequency_cph": (self.dims.frequency, self.frequency_cph),
            "rank_index": (self.dims.ranking, self.rank_index),
            "latitude": (self.dims.series, self.aux.lat),
            "nobs": (self.dims.series, self.nobs),
            "reference_time_mjd": (self.dims.series, self.reference_time_mjd),
        }
        harmonic_fields = (
            (
                "semi_major",
                "semi_minor",
                "inclination_degrees",
                "phase_degrees",
                "semi_major_ci",
                "semi_minor_ci",
                "inclination_ci_degrees",
                "phase_ci_degrees",
                "percent_energy",
                "signal_to_noise",
            )
            if self._fit.is_vector
            else (
                "amplitude",
                "phase_degrees",
                "amplitude_ci",
                "phase_ci_degrees",
                "percent_energy",
                "signal_to_noise",
            )
        )
        for name in harmonic_fields:
            if self[name] is not None:
                variables[name] = (self.dims.coefficients, self[name])
        series_fields = (
            ("umean", "vmean", "uslope", "vslope") if self._fit.is_vector else ("mean", "slope")
        )
        for name in series_fields:
            variables[name] = (self.dims.series, self[name])
        if self.robust is not None:
            variables["robust_weight"] = (self.dims.robust_weights, self.weights)
            variables["robust_leverage"] = (
                self.dims.robust_weights,
                self.robust.leverage,
            )
            for name in ("iterations", "residual_scale", "ols_rms_residual", "rms_residual"):
                variables[f"robust_{name}"] = (self.dims.series, self.robust[name])
        return xr.Dataset(
            data_vars=variables,
            coords=coordinates,
            attrs={
                "rutide_version": _native.__version__,
                "method": self.method,
                "confidence": self.confidence,
                "phase_reference": self.phase_reference,
                "nodal_corrections": self.nodal_corrections,
                "trend": self.trend,
            },
        )


class Tide(Bunch):
    """Scalar heights or vector currents returned by :func:`reconstruct`."""


def solve(
    t: ArrayLike,
    u: ArrayLike,
    v: ArrayLike | None = None,
    lat: float | None = None,
    *,
    constit: Literal["auto"] | Sequence[str] | None = "auto",
    conf_int: Literal["linear", "MC", "none"] | None = "linear",
    method: Literal["ols", "robust"] = "ols",
    trend: bool = True,
    phase: Literal["Greenwich", "linear_time", "raw"] = "Greenwich",
    nodal: bool | Literal["linear_time"] = True,
    infer: Mapping[str, Any] | None = None,
    order_constit: Literal["PE", "SNR", "frequency"] | Sequence[str] | None = "PE",
    MC_n: int = 200,
    MC_seed: int = 0,
    robust_kw: Mapping[str, Any] | None = None,
    Rayleigh_min: float = 1.0,
    white: bool = False,
    epoch: Epoch | None = None,
    verbose: bool = True,
) -> Coefficient:
    """Fit one scalar elevation or two-component current series.

    The endpoint intentionally follows :func:`utide.solve`. Numeric times are
    interpreted as Modified Julian Days when ``epoch`` is omitted. NaN
    observations and timestamps are removed jointly; infinite observations are
    rejected. ``verbose`` is accepted for source compatibility and is silent.
    """

    del verbose
    time_mjd = _time_to_mjd(t, epoch)
    eastward = _as_vector(u, "u")
    northward = None if v is None else _as_vector(v, "v")
    if time_mjd.size != eastward.size:
        raise ValueError("t and u must have the same length")
    if northward is not None and northward.size != eastward.size:
        raise ValueError("u and v must have the same length")
    if lat is None:
        raise ValueError("lat is required for astronomical and nodal corrections")

    fit = _native.solve(
        time_mjd,
        eastward,
        northward,
        float(lat),
        *_solver_arguments(
            constit,
            conf_int,
            method,
            trend,
            phase,
            nodal,
            infer,
            order_constit,
            MC_n,
            MC_seed,
            robust_kw,
            Rayleigh_min,
            white,
            northward is not None,
        ),
    )
    return Coefficient(fit)


def solve_many(
    t: ArrayLike,
    u: ArrayLike,
    v: ArrayLike | None = None,
    lat: float | ArrayLike | None = None,
    *,
    constit: Literal["auto"] | Sequence[str] | None = "auto",
    conf_int: Literal["linear", "MC", "none"] | None = "linear",
    method: Literal["ols", "robust"] = "ols",
    trend: bool = True,
    phase: Literal["Greenwich", "linear_time", "raw"] = "Greenwich",
    nodal: bool | Literal["linear_time"] = True,
    infer: Mapping[str, Any] | None = None,
    order_constit: Literal["PE", "SNR", "frequency"] | Sequence[str] | None = "PE",
    MC_n: int = 200,
    MC_seed: int = 0,
    robust_kw: Mapping[str, Any] | None = None,
    Rayleigh_min: float = 1.0,
    white: bool = False,
    epoch: Epoch | None = None,
    workers: int | None = None,
    memory_limit_mb: float | None = 512.0,
    verbose: bool = True,
) -> CoefficientBatch:
    """Fit multiple ``(time, series)`` scalar or vector series.

    Astronomy, missing-value masks, irregular spectral plans, and worker setup
    are shared inside Rust. ``memory_limit_mb`` bounds the temporary native
    component buffer used by each solve chunk; it does not include the caller's
    arrays, the one owned native input copy, or retained coefficient results.
    """

    del verbose
    time_mjd = _time_to_mjd(t, epoch)
    eastward = _as_matrix(u, "u")
    northward = None if v is None else _as_matrix(v, "v")
    if time_mjd.size != eastward.shape[0]:
        raise ValueError("t length must match the first dimension of u")
    if northward is not None and northward.shape != eastward.shape:
        raise ValueError("u and v must have the same shape")
    if lat is None:
        raise ValueError("lat is required for astronomical and nodal corrections")
    latitudes = _as_latitudes(lat, eastward.shape[1])
    if workers is not None and int(workers) <= 0:
        raise ValueError("workers must be greater than zero or None")
    if memory_limit_mb is None:
        memory_limit_bytes = None
    else:
        memory_limit_mb = float(memory_limit_mb)
        if not np.isfinite(memory_limit_mb) or memory_limit_mb <= 0.0:
            raise ValueError("memory_limit_mb must be finite and greater than zero or None")
        memory_limit_bytes = int(memory_limit_mb * 1024 * 1024)
    fit = _native.solve_many(
        time_mjd,
        eastward,
        northward,
        latitudes,
        *_solver_arguments(
            constit,
            conf_int,
            method,
            trend,
            phase,
            nodal,
            infer,
            order_constit,
            MC_n,
            MC_seed,
            robust_kw,
            Rayleigh_min,
            white,
            northward is not None,
        ),
        None if workers is None else int(workers),
        memory_limit_bytes,
    )
    return CoefficientBatch(fit)


def reconstruct(
    t: ArrayLike,
    coef: Coefficient,
    epoch: Epoch | None = None,
    verbose: bool = True,
    constit: Sequence[str] | None = None,
    min_SNR: float | None = 2.0,
    min_PE: float = 0.0,
) -> Tide:
    """Reconstruct elevations or currents at arbitrary target timestamps."""

    del verbose
    if not isinstance(coef, Coefficient):
        raise TypeError("coef must be the Coefficient returned by rutide.solve")
    target = _time_to_mjd(t, epoch)
    finite = np.isfinite(target)
    if finite.any():
        result = _native.reconstruct(
            np.ascontiguousarray(target[finite]),
            coef._fit,
            None if constit is None else [str(name) for name in constit],
            None if min_SNR is None else float(min_SNR),
            float(min_PE),
        )
    else:
        empty = np.empty(0, dtype=np.float64)
        result = (None, empty, empty) if coef._fit.is_vector else (empty, None, None)
    if coef._fit.is_vector:
        _, eastward, northward = result
        return Tide(
            u=_restore_missing(target.size, finite, eastward),
            v=_restore_missing(target.size, finite, northward),
        )
    heights, _, _ = result
    return Tide(h=_restore_missing(target.size, finite, heights))


def reconstruct_many(
    t: ArrayLike,
    coef: CoefficientBatch,
    epoch: Epoch | None = None,
    verbose: bool = True,
    constit: Sequence[str] | None = None,
    min_SNR: float | None = 2.0,
    min_PE: float = 0.0,
) -> Tide:
    """Reconstruct all series in a :class:`CoefficientBatch`."""

    del verbose
    if not isinstance(coef, CoefficientBatch):
        raise TypeError("coef must be the CoefficientBatch returned by rutide.solve_many")
    target = _time_to_mjd(t, epoch)
    finite = np.isfinite(target)
    if finite.any():
        result = _native.reconstruct_many(
            np.ascontiguousarray(target[finite]),
            coef._fit,
            None if constit is None else [str(name) for name in constit],
            None if min_SNR is None else float(min_SNR),
            float(min_PE),
        )
    else:
        empty = np.empty((0, coef._fit.series_count), dtype=np.float64)
        result = (None, empty, empty) if coef._fit.is_vector else (empty, None, None)
    if coef._fit.is_vector:
        _, eastward, northward = result
        return Tide(
            u=_restore_missing_matrix(target.size, finite, eastward),
            v=_restore_missing_matrix(target.size, finite, northward),
        )
    heights, _, _ = result
    return Tide(h=_restore_missing_matrix(target.size, finite, heights))


def _as_vector(value: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    masked = np.ma.isMaskedArray(value)
    array = np.asarray(np.ma.getdata(value) if masked else value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if masked:
        array = array.copy()
        array[np.ma.getmaskarray(value)] = np.nan
    return np.ascontiguousarray(array)


def _as_matrix(value: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    masked = np.ma.isMaskedArray(value)
    array = np.asarray(np.ma.getdata(value) if masked else value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional with shape (time, series)")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one series")
    if masked:
        array = array.copy()
        array[np.ma.getmaskarray(value)] = np.nan
    return np.ascontiguousarray(array)


def _as_latitudes(value: float | ArrayLike, series_count: int) -> npt.NDArray[np.float64]:
    latitudes = np.asarray(value, dtype=np.float64)
    if latitudes.ndim == 0:
        latitudes = np.full(series_count, float(latitudes), dtype=np.float64)
    elif latitudes.ndim != 1 or latitudes.size != series_count:
        raise ValueError(
            f"lat must be a scalar or contain one value for each of {series_count} series"
        )
    if not np.all(np.isfinite(latitudes)):
        raise ValueError("lat values must be finite")
    return np.ascontiguousarray(latitudes)


def _solver_arguments(
    constit: Literal["auto"] | Sequence[str] | None,
    conf_int: Literal["linear", "MC", "none"] | None,
    method: Literal["ols", "robust"],
    trend: bool,
    phase: Literal["Greenwich", "linear_time", "raw"],
    nodal: bool | Literal["linear_time"],
    infer: Mapping[str, Any] | None,
    order_constit: Literal["PE", "SNR", "frequency"] | Sequence[str] | None,
    MC_n: int,
    MC_seed: int,
    robust_kw: Mapping[str, Any] | None,
    Rayleigh_min: float,
    white: bool,
    vector: bool,
) -> tuple[Any, ...]:
    constituents = _constituent_selection(constit)
    inferred, references, ratios, phase_offsets, approximate = _parse_inference(infer, vector)
    robust = dict(robust_kw or {})
    robust_weight = str(_pop_alias(robust, "weight_function", "weight", "cauchy"))
    robust_tuning = _optional_float(_pop_alias(robust, "tuning_constant", "tune", None))
    robust_tolerance = float(_pop_alias(robust, "tolerance", "tol", 0.001))
    robust_max_iterations = int(_pop_alias(robust, "max_iterations", "maxit", 50))
    if robust:
        unknown = ", ".join(sorted(robust))
        raise TypeError(f"unknown robust_kw option(s): {unknown}")
    if order_constit is None:
        order_constit = "PE"
    return (
        constituents,
        float(Rayleigh_min),
        method,
        "none" if conf_int is None else conf_int,
        bool(white),
        bool(trend),
        phase,
        _nodal_name(nodal),
        int(MC_n),
        int(MC_seed),
        robust_weight,
        robust_tuning,
        robust_tolerance,
        robust_max_iterations,
        inferred,
        references,
        ratios,
        phase_offsets,
        approximate,
        order_constit if isinstance(order_constit, str) else "explicit",
        [] if isinstance(order_constit, str) else [str(name) for name in order_constit],
    )


def _constituent_selection(
    value: Literal["auto"] | Sequence[str] | None,
) -> list[str] | None:
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return None
    if isinstance(value, str):
        raise ValueError("constit must be 'auto' or a sequence of constituent names")
    return [str(name) for name in value]


def _time_to_mjd(value: ArrayLike, epoch: Epoch | None) -> npt.NDArray[np.float64]:
    masked = np.ma.isMaskedArray(value)
    mask = np.ma.getmaskarray(value) if masked else None
    array = np.asarray(np.ma.getdata(value) if masked else value)

    def finish(result: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        result = np.ascontiguousarray(result)
        if mask is not None:
            result = result.copy()
            result[mask] = np.nan
        return result

    if epoch is None and array.dtype.kind in ("O", "S", "U"):
        try:
            array = array.astype("datetime64[ms]")
        except (TypeError, ValueError):
            pass
    if np.issubdtype(array.dtype, np.datetime64):
        if epoch is not None:
            raise ValueError("epoch cannot be supplied with datetime64 timestamps")
        if array.ndim != 1:
            raise ValueError("t must be one-dimensional")
        milliseconds = array.astype("datetime64[ms]").astype(np.int64)
        nat = milliseconds == np.iinfo(np.int64).min
        mjd = (array.astype("datetime64[ms]") - _MJD_EPOCH).astype("timedelta64[ms]").astype(
            np.float64
        ) / _MILLISECONDS_PER_DAY
        mjd[nat] = np.nan
        return finish(mjd)

    numeric = _as_vector(array, "t")
    if epoch is None or epoch == "mjd":
        return finish(numeric)
    if epoch == "unix":
        return finish(numeric / 86_400.0 + 40_587.0)
    if epoch == "python":
        return finish(numeric - 678_576.0)
    if epoch == "matlab":
        return finish(numeric - 678_942.0)
    origin = _datetime_to_mjd(epoch)
    return finish(numeric + origin)


def _datetime_to_mjd(value: Epoch) -> float:
    if isinstance(value, datetime) and value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    instant = np.datetime64(value, "ms")
    return float((instant - _MJD_EPOCH) / np.timedelta64(1, "D"))


def _parse_inference(
    infer: Mapping[str, Any] | None, vector: bool
) -> tuple[list[str], list[str], list[float], list[float], bool]:
    if infer is None:
        return [], [], [], [], False
    inferred = _string_list(infer, "inferred_names")
    references = _string_list(infer, "reference_names")
    ratios = [float(item) for item in infer.get("amp_ratios", ())]
    phase_offsets = [float(item) for item in infer.get("phase_offsets", ())]
    expected = len(inferred) * (2 if vector else 1)
    if len(references) != len(inferred):
        raise ValueError("inference reference_names must match inferred_names")
    if len(ratios) != expected or len(phase_offsets) != expected:
        kind = "2N" if vector else "N"
        raise ValueError(f"inference amp_ratios and phase_offsets must each contain {kind} values")
    return inferred, references, ratios, phase_offsets, bool(infer.get("approximate", False))


def _string_list(mapping: Mapping[str, Any], key: str) -> list[str]:
    value = mapping.get(key, ())
    if isinstance(value, str):
        raise TypeError(f"inference {key} must be a sequence of names")
    return [str(item) for item in value]


def _nodal_name(value: bool | Literal["linear_time"]) -> str:
    if isinstance(value, (bool, np.bool_)) and bool(value):
        return "exact"
    if isinstance(value, (bool, np.bool_)) and not bool(value):
        return "disabled"
    if value == "linear_time":
        return "linear_time"
    raise ValueError("nodal must be True, False, or 'linear_time'")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _pop_alias(options: dict[str, Any], primary: str, alias: str, default: Any) -> Any:
    if primary in options and alias in options:
        raise TypeError(f"robust_kw cannot contain both '{primary}' and '{alias}'")
    if primary in options:
        return options.pop(primary)
    if alias in options:
        return options.pop(alias)
    return default


def _arrays_from_lists(value: Any) -> Any:
    if isinstance(value, dict):
        return Bunch({key: _arrays_from_lists(item) for key, item in value.items()})
    if isinstance(value, list):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    return value


def _restore_missing(
    size: int, finite: npt.NDArray[np.bool_], values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    output = np.full(size, np.nan, dtype=np.float64)
    output[finite] = values
    return output


def _restore_missing_matrix(
    size: int, finite: npt.NDArray[np.bool_], values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    output = np.full((size, values.shape[1]), np.nan, dtype=np.float64)
    output[finite, :] = values
    return output
