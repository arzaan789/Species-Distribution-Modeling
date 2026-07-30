from __future__ import annotations

import numpy as np
import pytest

from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.metrics import (
    continuous_boyce,
    generate_unbiased_evaluation,
    integrated_normalized_error,
    response_curve_rmse,
    top_quantile_overlap,
    truth_metrics,
)
from provenance_sdm.virtual_species import simulate_species_truth


@pytest.fixture
def toy_truth():
    side = np.linspace(-2.0, 2.0, 10)
    x, y = np.meshgrid(side * 1_000, side * 1_000)
    landscape = landscape_from_arrays(
        {
            "env_1": x.ravel(),
            "env_2": y.ravel(),
            "env_3": np.sin(x.ravel() / 1_000) + np.cos(y.ravel() / 1_000),
        },
        x=x.ravel(),
        y=y.ravel(),
        area=np.ones(x.size),
        crs="EPSG:27700",
    )
    return simulate_species_truth(landscape, n_species=1, seed=4)[0]


def test_identical_prediction_has_optimal_truth_metrics(toy_truth) -> None:
    values = truth_metrics(
        toy_truth.suitability,
        toy_truth,
        unbiased_y=np.array([1, 1, 0, 0]),
        unbiased_score=np.array([0.9, 0.8, 0.2, 0.1]),
    )

    assert values["suitability_spearman"] == pytest.approx(1.0)
    assert values["integrated_error"] == pytest.approx(0.0)
    assert values["top10_overlap"] == pytest.approx(1.0)
    assert values["unbiased_auc"] == pytest.approx(1.0)
    assert values["response_curve_error"] == pytest.approx(0.0)


def test_integrated_error_uses_area_weights_after_normalization() -> None:
    predicted = np.array([1.0, 1.0, 0.0])
    truth = np.array([1.0, 0.0, 0.0])
    area = np.array([1.0, 3.0, 1.0])

    assert integrated_normalized_error(predicted, truth, area) == pytest.approx(
        0.75
    )


def test_top_quantile_overlap_is_area_weighted_jaccard() -> None:
    predicted = np.arange(10, dtype=float)
    truth = np.array([0, 1, 2, 3, 4, 5, 6, 9, 8, 7], dtype=float)
    area = np.ones(10)

    assert top_quantile_overlap(predicted, truth, area, quantile=0.8) == pytest.approx(
        1 / 3
    )


def test_continuous_boyce_detects_increasing_presence_preference() -> None:
    landscape_scores = np.linspace(0.0, 1.0, 1_000)
    presence_scores = np.repeat(np.linspace(0.55, 1.0, 10), np.arange(1, 11))

    result = continuous_boyce(presence_scores, landscape_scores, n_bins=10)

    assert result.defined
    assert result.value > 0.9
    assert result.reason is None


def test_continuous_boyce_marks_constant_scores_undefined() -> None:
    result = continuous_boyce(np.ones(10), np.ones(100), n_bins=10)

    assert not result.defined
    assert result.value is None
    assert "constant" in result.reason


def test_response_curve_error_increases_for_reversed_surface(toy_truth) -> None:
    reversed_prediction = toy_truth.suitability[::-1].copy()

    assert response_curve_rmse(toy_truth.suitability, toy_truth) == pytest.approx(
        0.0
    )
    assert response_curve_rmse(reversed_prediction, toy_truth) > 0.1


def test_unbiased_evaluation_is_deterministic_and_separate_from_training(
    toy_truth,
) -> None:
    first = generate_unbiased_evaluation(
        toy_truth, n_presence=2_000, n_background=2_000, seed=44
    )
    second = generate_unbiased_evaluation(
        toy_truth, n_presence=2_000, n_background=2_000, seed=44
    )

    assert first.equals(second)
    assert first.label.value_counts().to_dict() == {1: 2_000, 0: 2_000}
    assert first.sample_id.is_unique
    presence_mean = first.query("label == 1").cell_id.mean()
    background_mean = first.query("label == 0").cell_id.mean()
    assert abs(presence_mean - background_mean) > 5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("non_finite", "finite"),
        ("negative", "non-negative"),
        ("wrong_length", "length"),
    ],
)
def test_invalid_truth_metric_inputs_are_rejected(
    toy_truth,
    mutation: str,
    message: str,
) -> None:
    predicted = toy_truth.suitability.copy()
    if mutation == "non_finite":
        predicted[0] = np.nan
    elif mutation == "negative":
        predicted[0] = -1.0
    elif mutation == "wrong_length":
        predicted = predicted[:-1]

    with pytest.raises(ValueError, match=message):
        truth_metrics(
            predicted,
            toy_truth,
            unbiased_y=np.array([1, 0]),
            unbiased_score=np.array([0.8, 0.2]),
        )


def test_auc_requires_both_evaluation_classes(toy_truth) -> None:
    with pytest.raises(ValueError, match="classes"):
        truth_metrics(
            toy_truth.suitability,
            toy_truth,
            unbiased_y=np.ones(3),
            unbiased_score=np.array([0.9, 0.8, 0.7]),
        )
