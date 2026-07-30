"""A fixed-basis, regularized MaxEnt-equivalent presence-background model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from provenance_sdm.landscape import Landscape


@dataclass(frozen=True)
class MaxentModel:
    feature_names: tuple[str, ...]
    feature_means: np.ndarray
    feature_scales: np.ndarray
    estimator: LogisticRegression

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names).difference(frame.columns)
        if missing:
            raise ValueError(f"prediction data are missing features: {sorted(missing)}")
        linear = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(linear).all():
            raise ValueError("prediction features must be finite")
        linear = (linear - self.feature_means) / self.feature_scales
        quadratic = linear**2
        interactions = [
            linear[:, left] * linear[:, right]
            for left in range(linear.shape[1])
            for right in range(left + 1, linear.shape[1])
        ]
        return np.column_stack([linear, quadratic, *interactions])

    def predict_log_intensity(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.decision_function(self.transform(frame)))

    def predict_suitability(
        self,
        landscape: Landscape | pd.DataFrame,
    ) -> np.ndarray:
        frame = landscape.cells if isinstance(landscape, Landscape) else landscape
        log_intensity = self.predict_log_intensity(frame)
        log_intensity -= float(log_intensity.max())
        intensity = np.exp(np.clip(log_intensity, -50.0, 0.0))
        if "area_weight" in frame:
            intensity *= frame.area_weight.to_numpy(dtype=float)
        total = float(intensity.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("predicted landscape intensity must be finite and positive")
        return intensity / total


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
    )
    design = transform_model.transform(combined)
    labels = np.concatenate(
        [np.ones(len(presence), dtype=int), np.zeros(len(background), dtype=int)]
    )
    estimator = LogisticRegression(
        C=1.0 / regularization,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1_000,
        random_state=seed,
    )
    estimator.fit(design, labels)
    return MaxentModel(
        feature_names=names,
        feature_means=means,
        feature_scales=scales,
        estimator=estimator,
    )
