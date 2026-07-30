from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.provenance import pm_tgb_weights, source_distribution_distance


def test_supported_sources_match_focal_source_mass() -> None:
    focal = pd.Series(["A"] * 8 + ["B"] * 2)
    candidates = pd.Series(["A", "A", "B", "B"], index=[10, 11, 12, 13])

    result = pm_tgb_weights(focal, candidates)

    assert result.weights.index.tolist() == [10, 11, 12, 13]
    assert result.weights.groupby(candidates).sum().to_dict() == pytest.approx(
        {"A": 0.8, "B": 0.2}
    )
    assert result.supported_mass == pytest.approx(1.0)
    assert result.unsupported_mass == pytest.approx(0.0)


def test_unsupported_mass_uses_conventional_pooled_record_fallback() -> None:
    focal = pd.Series(["A"] * 6 + ["C"] * 4)
    candidates = pd.Series(["A", "A", "B", "B"], index=[10, 11, 12, 13])

    result = pm_tgb_weights(focal, candidates)

    assert result.weights.to_numpy() == pytest.approx([0.4, 0.4, 0.1, 0.1])
    assert result.focal_mass.to_dict() == pytest.approx({"A": 0.6, "C": 0.4})
    assert result.supported_mass == pytest.approx(0.6)
    assert result.unsupported_mass == pytest.approx(0.4)
    assert result.weights.sum() == pytest.approx(1.0)


def test_fully_unsupported_focal_sources_reduce_to_pooled_tgb() -> None:
    focal = pd.Series(["C", "C"])
    candidates = pd.Series(["A", "A", "B"], index=[4, 5, 6])

    result = pm_tgb_weights(focal, candidates)

    assert result.weights.to_numpy() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert result.supported_mass == pytest.approx(0.0)
    assert result.unsupported_mass == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("focal", "candidates", "message"),
    [
        (pd.Series(dtype=object), pd.Series(["A"]), "non-empty"),
        (pd.Series(["A"]), pd.Series(dtype=object), "non-empty"),
        (pd.Series(["A", None]), pd.Series(["A"]), "complete"),
        (pd.Series(["A"]), pd.Series(["A", None]), "complete"),
        (
            pd.Series(["A"]),
            pd.Series(["A", "B"], index=[1, 1]),
            "unique",
        ),
    ],
)
def test_invalid_source_inputs_are_rejected(
    focal: pd.Series,
    candidates: pd.Series,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pm_tgb_weights(focal, candidates)


def test_result_weights_are_finite_and_non_negative() -> None:
    focal = pd.Series(["A", "B", "C"])
    candidates = pd.Series(["A", "B", "D"], index=[1, 2, 3])

    result = pm_tgb_weights(focal, candidates)

    assert np.isfinite(result.weights).all()
    assert result.weights.ge(0).all()


def test_source_distribution_distance_is_total_variation() -> None:
    focal = pd.Series(["A"] * 8 + ["B"] * 2)
    candidates = pd.Series(["A", "B"])

    assert source_distribution_distance(focal, candidates) == pytest.approx(0.3)
