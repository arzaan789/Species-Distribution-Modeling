"""A fixed-basis, regularized MaxEnt-equivalent presence-background model."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from provenance_sdm.landscape import Landscape


FEATURE_BASIS = "linear"
LOWER_LOG_INTENSITY_CLIP = -50.0
PRIMARY_REGULARIZATION = 2.0


@dataclass(frozen=True)
class PredictionResult:
    suitability: np.ndarray
    feature_basis: str
    max_cell_mass: float
    effective_cell_count: float
    log_intensity_range: float
    lower_clip_cells: int
    lower_clip_fraction: float
    solver_converged: bool


@dataclass(frozen=True)
class MaxentModel:
    feature_names: tuple[str, ...]
    feature_means: np.ndarray
    feature_scales: np.ndarray
    estimator: LogisticRegression
    solver_converged: bool

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names).difference(frame.columns)
        if missing:
            raise ValueError(f"prediction data are missing features: {sorted(missing)}")
        linear = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(linear).all():
            raise ValueError("prediction features must be finite")
        linear = (linear - self.feature_means) / self.feature_scales
        return linear

    def predict_log_intensity(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.decision_function(self.transform(frame)))

    def predict_with_diagnostics(
        self,
        landscape: Landscape | pd.DataFrame,
        batch_size: int = 50_000,
    ) -> PredictionResult:
        frame = landscape.cells if isinstance(landscape, Landscape) else landscape
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        log_intensity = np.empty(len(frame), dtype=float)
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            log_intensity[start:stop] = self.predict_log_intensity(
                frame.iloc[start:stop]
            )
        log_intensity_range = float(np.ptp(log_intensity))
        log_intensity -= float(log_intensity.max())
        lower_clip = log_intensity < LOWER_LOG_INTENSITY_CLIP
        intensity = np.exp(
            np.clip(log_intensity, LOWER_LOG_INTENSITY_CLIP, 0.0)
        )
        if "area_weight" in frame:
            intensity *= frame.area_weight.to_numpy(dtype=float)
        total = float(intensity.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("predicted landscape intensity must be finite and positive")
        suitability = intensity / total
        return PredictionResult(
            suitability=suitability,
            feature_basis=FEATURE_BASIS,
            max_cell_mass=float(suitability.max()),
            effective_cell_count=float(1.0 / np.square(suitability).sum()),
            log_intensity_range=log_intensity_range,
            lower_clip_cells=int(lower_clip.sum()),
            lower_clip_fraction=float(lower_clip.mean()),
            solver_converged=self.solver_converged,
        )

    def predict_suitability(
        self,
        landscape: Landscape | pd.DataFrame,
        batch_size: int = 50_000,
    ) -> np.ndarray:
        return self.predict_with_diagnostics(
            landscape,
            batch_size,
        ).suitability


def fit_maxent(
    presence: pd.DataFrame,
    background: pd.DataFrame,
    feature_names: Sequence[str],
    regularization: float,
    seed: int,
) -> MaxentModel:
    """Fit a deterministic presence-background logistic approximation.

    With an intercept and a common feature basis, logistic presence-background
    slopes approximate a Poisson point-process/MaxEnt intensity model as
    background sampling becomes dense. Returned scores are relative
    suitability masses, not occurrence probabilities.
    """

    names = tuple(feature_names)
    if presence.empty or background.empty:
        raise ValueError("presence and background data must be non-empty")
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be non-empty and unique")
    missing = set(names).difference(presence.columns).union(
        set(names).difference(background.columns)
    )
    if missing:
        raise ValueError(f"model data are missing features: {sorted(missing)}")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")

    combined = pd.concat(
        [presence.loc[:, names], background.loc[:, names]],
        ignore_index=True,
    )
    raw = combined.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("model features must be finite")
    means = raw.mean(axis=0)
    scales = raw.std(axis=0, ddof=0)
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("model features must be non-constant")

    transform_model = MaxentModel(
        feature_names=names,
        feature_means=means,
        feature_scales=scales,
        estimator=LogisticRegression(),
        solver_converged=False,
    )
    design = transform_model.transform(combined)
    labels = np.concatenate(
        [np.ones(len(presence), dtype=int), np.zeros(len(background), dtype=int)]
    )
    sample_weight = np.concatenate(
        [
            np.full(len(presence), 0.5 / len(presence)),
            np.full(len(background), 0.5 / len(background)),
        ]
    )
    estimator = LogisticRegression(
        C=1.0 / regularization,
        solver="lbfgs",
        max_iter=1_000,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(design, labels, sample_weight=sample_weight)
    converged = not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    return MaxentModel(
        feature_names=names,
        feature_means=means,
        feature_scales=scales,
        estimator=estimator,
        solver_converged=converged,
    )
