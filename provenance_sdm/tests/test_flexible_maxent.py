from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.flexible_maxent import fit_flexible_maxent


FEATURES = ("a", "b", "c")


@pytest.fixture
def training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    values = np.linspace(-2.0, 2.0, 20)
    presence = pd.DataFrame(
        {
            "a": values,
            "b": np.sin(values) + 0.25,
            "c": np.square(values) - 0.5,
        }
    )
    background = pd.DataFrame(
        {
            "a": values[::-1] + 0.5,
            "b": np.cos(values),
            "c": values * 0.75,
        }
    )
    return presence, background


def test_flexible_design_has_linear_square_and_pairwise_columns(
    training_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    presence, background = training_frames
    model = fit_flexible_maxent(
        presence,
        background,
        FEATURES,
        regularization=2.0,
        seed=1,
    )

    assert model.transform(presence).shape == (20, 9)
    assert model.feature_basis == "clamped_linear_quadratic_interactions"


def test_prediction_clamps_raw_values_before_polynomial_expansion(
    training_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    presence, background = training_frames
    model = fit_flexible_maxent(
        presence,
        background,
        FEATURES,
        regularization=2.0,
        seed=1,
    )
    extreme = pd.DataFrame(
        {"a": [1e12], "b": [-1e12], "c": [1e12]}
    )
    boundary = pd.DataFrame(
        {
            "a": [model.raw_maxs[0]],
            "b": [model.raw_mins[1]],
            "c": [model.raw_maxs[2]],
        }
    )

    np.testing.assert_allclose(
        model.transform(extreme),
        model.transform(boundary),
    )


def test_flexible_prediction_is_finite_and_exports_stability(
    training_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    presence, background = training_frames
    model = fit_flexible_maxent(
        presence,
        background,
        FEATURES,
        regularization=2.0,
        seed=2,
    )
    landscape = pd.concat(
        [presence, background, pd.DataFrame({"a": [1e9], "b": [-1e9], "c": [1e9]})],
        ignore_index=True,
    )
    landscape["area_weight"] = 1.0

    result = model.predict_with_diagnostics(landscape, batch_size=7)

    assert np.isfinite(result.suitability).all()
    assert result.suitability.sum() == pytest.approx(1.0)
    assert 0 < result.max_cell_mass <= 1
    assert result.effective_cell_count >= 1
    assert result.log_intensity_range >= 0
    assert 0 <= result.lower_clip_fraction <= 1
    assert isinstance(result.solver_converged, bool)


@pytest.mark.parametrize("regularization", (0.0, -1.0, np.nan))
def test_flexible_fit_rejects_invalid_regularization(
    training_frames: tuple[pd.DataFrame, pd.DataFrame],
    regularization: float,
) -> None:
    with pytest.raises(ValueError, match="regularization"):
        fit_flexible_maxent(
            *training_frames,
            FEATURES,
            regularization=regularization,
            seed=3,
        )


def test_flexible_fit_rejects_empty_or_nonfinite_training_rows(
    training_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    presence, background = training_frames
    with pytest.raises(ValueError, match="non-empty"):
        fit_flexible_maxent(
            presence.iloc[:0], background, FEATURES, 2.0, seed=3
        )
    invalid = presence.copy()
    invalid.loc[0, "a"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_flexible_maxent(invalid, background, FEATURES, 2.0, seed=3)


def test_flexible_fit_rejects_missing_duplicate_and_constant_features(
    training_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    presence, background = training_frames
    with pytest.raises(ValueError, match="missing"):
        fit_flexible_maxent(presence, background, ("a", "missing"), 2.0, 3)
    with pytest.raises(ValueError, match="unique"):
        fit_flexible_maxent(presence, background, ("a", "a"), 2.0, 3)
    constant_presence = presence.assign(a=1.0)
    constant_background = background.assign(a=1.0)
    with pytest.raises(ValueError, match="non-constant"):
        fit_flexible_maxent(
            constant_presence,
            constant_background,
            FEATURES,
            2.0,
            3,
        )


def test_flexible_prediction_rejects_invalid_batch_size(
    training_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    model = fit_flexible_maxent(*training_frames, FEATURES, 2.0, seed=4)

    with pytest.raises(ValueError, match="batch_size"):
        model.predict_with_diagnostics(training_frames[0], batch_size=0)
