from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.maxent import fit_maxent


@pytest.fixture
def gradient_data():
    env_1 = np.linspace(-3.0, 3.0, 301)
    landscape = landscape_from_arrays(
        {
            "env_1": env_1,
            "env_2": np.sin(env_1),
            "env_3": np.cos(env_1),
        },
        x=np.arange(env_1.size, dtype=float) * 1_000,
        y=np.zeros(env_1.size),
        area=np.ones(env_1.size),
        crs="EPSG:27700",
    )
    frame = landscape.cells
    presence = pd.concat(
        [
            frame.query("env_1 > 0.7"),
            frame.query("env_1 > 1.2"),
        ],
        ignore_index=True,
    )
    background = frame.iloc[::3].copy()
    truth = np.exp(landscape.cells.env_1.to_numpy())
    truth /= truth.sum()
    return landscape, presence, background, truth


def test_maxent_recovers_simple_environmental_gradient(gradient_data) -> None:
    landscape, presence, background, truth = gradient_data

    model = fit_maxent(
        presence,
        background,
        feature_names=("env_1",),
        regularization=1.0,
        seed=5,
    )
    predicted = model.predict_suitability(landscape)

    assert scipy.stats.spearmanr(predicted, truth).statistic > 0.8
    assert predicted.sum() == pytest.approx(1.0)
    assert np.all(predicted > 0)


def test_fit_is_deterministic_under_fixed_seed(gradient_data) -> None:
    _, presence, background, _ = gradient_data

    first = fit_maxent(presence, background, ("env_1",), 1.0, seed=8)
    second = fit_maxent(presence, background, ("env_1",), 1.0, seed=8)

    np.testing.assert_allclose(first.estimator.coef_, second.estimator.coef_)
    np.testing.assert_allclose(first.estimator.intercept_, second.estimator.intercept_)


def test_batched_landscape_prediction_matches_single_batch(gradient_data) -> None:
    landscape, presence, background, _ = gradient_data
    model = fit_maxent(presence, background, ("env_1",), 1.0, seed=8)

    batched = model.predict_suitability(landscape, batch_size=17)
    single = model.predict_suitability(landscape, batch_size=len(landscape.cells))

    np.testing.assert_allclose(batched, single, rtol=1e-12, atol=1e-15)
    assert batched.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_feature", "missing"),
        ("non_finite", "finite"),
        ("zero_regularization", "positive"),
        ("empty_presence", "non-empty"),
    ],
)
def test_invalid_model_inputs_are_rejected(
    gradient_data,
    mutation: str,
    message: str,
) -> None:
    _, presence, background, _ = gradient_data
    features = ("env_1",)
    regularization = 1.0
    if mutation == "missing_feature":
        features = ("missing",)
    elif mutation == "non_finite":
        presence = presence.copy()
        presence.loc[presence.index[0], "env_1"] = np.nan
    elif mutation == "zero_regularization":
        regularization = 0.0
    elif mutation == "empty_presence":
        presence = presence.iloc[0:0]

    with pytest.raises(ValueError, match=message):
        fit_maxent(
            presence,
            background,
            features,
            regularization,
            seed=1,
        )
