"""Truth-based simulation and descriptive empirical metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score

from provenance_sdm.virtual_species import SpeciesTruth


@dataclass(frozen=True)
class BoyceResult:
    value: float | None
    defined: bool
    reason: str | None


def _validate_surface(
    values: np.ndarray,
    expected_length: int,
    field: str,
) -> np.ndarray:
    surface = np.asarray(values, dtype=float)
    if surface.shape != (expected_length,):
        raise ValueError(f"{field} has the wrong length")
    if not np.isfinite(surface).all():
        raise ValueError(f"{field} must be finite")
    if np.any(surface < 0):
        raise ValueError(f"{field} must be non-negative")
    if surface.sum() <= 0:
        raise ValueError(f"{field} must contain positive mass")
    return surface


def integrated_normalized_error(
    predicted: np.ndarray,
    truth: np.ndarray,
    area_weight: np.ndarray,
) -> float:
    """Return total-variation error between area-normalized surfaces."""

    prediction = np.asarray(predicted, dtype=float)
    target = np.asarray(truth, dtype=float)
    area = np.asarray(area_weight, dtype=float)
    if prediction.shape != target.shape or prediction.shape != area.shape:
        raise ValueError("metric surfaces and area weights must have the same length")
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(target).all()
        and np.isfinite(area).all()
    ):
        raise ValueError("metric surfaces and area weights must be finite")
    if np.any(prediction < 0) or np.any(target < 0):
        raise ValueError("metric surfaces must be non-negative")
    if np.any(area <= 0):
        raise ValueError("area weights must be positive")
    prediction = prediction / np.sum(prediction * area)
    target = target / np.sum(target * area)
    return float(0.5 * np.sum(np.abs(prediction - target) * area))


def top_quantile_overlap(
    predicted: np.ndarray,
    truth: np.ndarray,
    area_weight: np.ndarray,
    quantile: float = 0.90,
) -> float:
    """Return area-weighted Jaccard overlap above separate score quantiles."""

    prediction = np.asarray(predicted, dtype=float)
    target = np.asarray(truth, dtype=float)
    area = np.asarray(area_weight, dtype=float)
    if prediction.shape != target.shape or prediction.shape != area.shape:
        raise ValueError("metric surfaces and area weights must have the same length")
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(target).all()
        and np.isfinite(area).all()
    ):
        raise ValueError("metric surfaces and area weights must be finite")
    predicted_mask = prediction >= np.quantile(prediction, quantile)
    truth_mask = target >= np.quantile(target, quantile)
    union_area = area[predicted_mask | truth_mask].sum()
    if union_area <= 0:
        raise ValueError("upper-quantile union must have positive area")
    return float(area[predicted_mask & truth_mask].sum() / union_area)


def continuous_boyce(
    presence_scores: np.ndarray,
    landscape_scores: np.ndarray,
    n_bins: int = 10,
) -> BoyceResult:
    """Calculate an equal-frequency-bin continuous Boyce index."""

    presence = np.asarray(presence_scores, dtype=float)
    landscape = np.asarray(landscape_scores, dtype=float)
    if presence.ndim != 1 or landscape.ndim != 1 or not len(presence):
        raise ValueError("Boyce score arrays must be non-empty and one-dimensional")
    if not np.isfinite(presence).all() or not np.isfinite(landscape).all():
        raise ValueError("Boyce scores must be finite")
    if n_bins < 3:
        raise ValueError("Boyce index requires at least three bins")
    if np.ptp(landscape) == 0:
        return BoyceResult(None, False, "landscape scores are constant")

    edges = np.unique(np.quantile(landscape, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 4:
        return BoyceResult(None, False, "fewer than three populated score bins")
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected, _ = np.histogram(landscape, bins=edges)
    observed, _ = np.histogram(presence, bins=edges)
    valid = expected > 0
    ratio = (observed[valid] / len(presence)) / (
        expected[valid] / len(landscape)
    )
    if len(ratio) < 3 or np.ptp(ratio) == 0:
        return BoyceResult(None, False, "observed-to-expected ratios are constant")
    value = float(scipy.stats.spearmanr(np.arange(len(ratio)), ratio).statistic)
    if not np.isfinite(value):
        return BoyceResult(None, False, "Boyce rank correlation is undefined")
    return BoyceResult(value, True, None)


def response_curve_rmse(
    predicted: np.ndarray,
    truth: SpeciesTruth,
    n_bins: int = 20,
) -> float:
    """Compare predicted and true marginal responses on common quantile bins."""

    target = truth.suitability
    prediction = _validate_surface(predicted, len(target), "predicted suitability")
    if n_bins < 3:
        raise ValueError("response curves require at least three bins")
    errors: list[float] = []
    for feature in truth.landscape.feature_names:
        values = truth.landscape.cells[feature].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 4:
            continue
        bins = np.digitize(values, edges[1:-1], right=True)
        predicted_curve = np.bincount(
            bins,
            weights=prediction,
            minlength=len(edges) - 1,
        )
        truth_curve = np.bincount(
            bins,
            weights=target,
            minlength=len(edges) - 1,
        )
        predicted_range = float(np.ptp(predicted_curve))
        truth_range = float(np.ptp(truth_curve))
        if predicted_range == 0 or truth_range == 0:
            continue
        predicted_curve = (
            predicted_curve - predicted_curve.min()
        ) / predicted_range
        truth_curve = (truth_curve - truth_curve.min()) / truth_range
        errors.extend(np.square(predicted_curve - truth_curve))
    if not errors:
        raise ValueError("response curves are undefined for constant marginal curves")
    return float(np.sqrt(np.mean(errors)))


def generate_unbiased_evaluation(
    truth: SpeciesTruth,
    n_presence: int,
    n_background: int,
    seed: int,
) -> pd.DataFrame:
    """Sample evaluation rows from truth and area, independently of training."""

    if n_presence <= 0 or n_background <= 0:
        raise ValueError("evaluation sample sizes must be positive")
    generator = np.random.default_rng(seed)
    cells = truth.landscape.cells
    cell_ids = cells.cell_id.to_numpy(dtype=np.int64)
    presence_cells = generator.choice(
        cell_ids,
        size=n_presence,
        replace=True,
        p=truth.suitability,
    )
    area_probability = cells.area_weight.to_numpy(dtype=float)
    area_probability = area_probability / area_probability.sum()
    background_cells = generator.choice(
        cell_ids,
        size=n_background,
        replace=True,
        p=area_probability,
    )
    return pd.DataFrame(
        {
            "sample_id": np.arange(n_presence + n_background, dtype=np.int64),
            "cell_id": np.concatenate([presence_cells, background_cells]),
            "label": np.concatenate(
                [
                    np.ones(n_presence, dtype=np.int8),
                    np.zeros(n_background, dtype=np.int8),
                ]
            ),
        }
    )


def truth_metrics(
    predicted: np.ndarray,
    truth: SpeciesTruth,
    unbiased_y: np.ndarray,
    unbiased_score: np.ndarray,
) -> dict[str, float]:
    """Evaluate a prediction against ecological truth and an unbiased sample."""

    target = truth.suitability
    prediction = _validate_surface(predicted, len(target), "predicted suitability")
    if np.ptp(prediction) == 0:
        raise ValueError("predicted suitability must not be constant")
    labels = np.asarray(unbiased_y)
    scores = np.asarray(unbiased_score, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("unbiased evaluation labels and scores have the wrong length")
    if not np.isfinite(scores).all():
        raise ValueError("unbiased evaluation scores must be finite")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("unbiased evaluation requires both classes")

    area = truth.landscape.cells.area_weight.to_numpy(dtype=float)
    correlation = float(scipy.stats.spearmanr(prediction, target).statistic)
    return {
        "suitability_spearman": correlation,
        "integrated_error": integrated_normalized_error(prediction, target, area),
        "top10_overlap": top_quantile_overlap(prediction, target, area, 0.90),
        "unbiased_auc": float(roc_auc_score(labels, scores)),
        "response_curve_error": response_curve_rmse(prediction, truth),
    }
