"""Validated projected environmental analysis landscapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import CRS


@dataclass(frozen=True)
class Landscape:
    cells: pd.DataFrame
    feature_names: tuple[str, ...]
    crs: str
    feature_means: Mapping[str, float]
    feature_scales: Mapping[str, float]

    def __post_init__(self) -> None:
        required = {"cell_id", "x", "y", "area_weight", *self.feature_names}
        missing = required.difference(self.cells.columns)
        if missing:
            raise ValueError(f"landscape cells are missing columns: {sorted(missing)}")
        if not self.cells.cell_id.is_unique:
            raise ValueError("landscape cell_id values must be unique")
        if len(self.feature_names) < 3:
            raise ValueError("landscape requires at least three environmental features")
        try:
            projected = CRS.from_user_input(self.crs).is_projected
        except Exception as exc:
            raise ValueError("landscape CRS must be a valid projected CRS") from exc
        if not projected:
            raise ValueError("landscape CRS must be projected")

        numeric = self.cells.loc[:, ["x", "y", "area_weight", *self.feature_names]]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("landscape coordinates, areas, and features must be finite")
        if self.cells.area_weight.le(0).any():
            raise ValueError("landscape area weights must be strictly positive")


def landscape_from_arrays(
    predictors: Mapping[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray,
    crs: str,
) -> Landscape:
    """Create a projected landscape and standardize environmental predictors."""

    if len(predictors) < 3:
        raise ValueError("landscape requires at least three environmental features")
    coordinate_x = np.asarray(x, dtype=float)
    coordinate_y = np.asarray(y, dtype=float)
    area_weight = np.asarray(area, dtype=float)
    if coordinate_x.ndim != 1 or coordinate_x.size == 0:
        raise ValueError("landscape arrays must be non-empty and one-dimensional")
    expected_length = coordinate_x.size
    if coordinate_y.shape != (expected_length,) or area_weight.shape != (
        expected_length,
    ):
        raise ValueError("landscape arrays must have the same length")
    if not np.isfinite(coordinate_x).all() or not np.isfinite(coordinate_y).all():
        raise ValueError("landscape coordinates must be finite")
    if not np.isfinite(area_weight).all():
        raise ValueError("landscape area weights must be finite")
    if np.any(area_weight <= 0):
        raise ValueError("landscape area weights must be strictly positive")

    feature_names = tuple(predictors)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    standardized: dict[str, np.ndarray] = {}
    for name, values in predictors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("environmental feature names must be non-empty strings")
        feature = np.asarray(values, dtype=float)
        if feature.shape != (expected_length,):
            raise ValueError(f"environmental feature {name!r} has the wrong length")
        if not np.isfinite(feature).all():
            raise ValueError(f"environmental feature {name!r} must be finite")
        mean = float(feature.mean())
        scale = float(feature.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"environmental feature {name!r} must not be constant")
        means[name] = mean
        scales[name] = scale
        standardized[name] = (feature - mean) / scale

    cells = pd.DataFrame(
        {
            "cell_id": np.arange(expected_length, dtype=np.int64),
            "x": coordinate_x,
            "y": coordinate_y,
            "area_weight": area_weight,
            **standardized,
        }
    )
    return Landscape(
        cells=cells,
        feature_names=feature_names,
        crs=crs,
        feature_means=means,
        feature_scales=scales,
    )
