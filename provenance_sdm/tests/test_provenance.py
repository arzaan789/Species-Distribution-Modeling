from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.provenance import (
    pm_tgb_weights,
    source_distribution_distance,
    source_target_weights,
)


def test_explicit_source_target_mass_uses_same_supported_and_fallback_rules() -> None:
    target = pd.Series({"source_a": 0.6, "source_b": 0.3, "missing": 0.1})
    candidates = pd.Series(
        ["source_a", "source_a", "source_b"],
        index=[10, 11, 12],
    )

    result = source_target_weights(target, candidates)

    grouped = result.weights.groupby(candidates).sum()
    assert grouped.to_dict() == pytest.approx(
        {"source_a": 2 / 3, "source_b": 1 / 3}
    )
    assert result.focal_mass.to_dict() == pytest.approx(target.to_dict())
    assert result.supported_mass == pytest.approx(0.9)
    assert result.unsupported_mass == pytest.approx(0.1)


def test_observed_source_wrapper_matches_explicit_target_mass() -> None:
    focal = pd.Series(["a", "a", "b"])
    candidates = pd.Series(["a", "b", "b"], index=[4, 5, 6])

    observed = pm_tgb_weights(focal, candidates)
    explicit = source_target_weights(
        pd.Series({"a": 2 / 3, "b": 1 / 3}),
        candidates,
    )

    pd.testing.assert_series_equal(observed.weights, explicit.weights)


@pytest.mark.parametrize(
    "target",
    (
        pd.Series(dtype=float),
        pd.Series({"a": 0.4, "b": 0.4}),
        pd.Series({"a": -0.1, "b": 1.1}),
        pd.Series({"a": np.nan, "b": 1.0}),
        pd.Series([0.5, 0.5], index=["a", "a"]),
    ),
)
def test_explicit_target_mass_must_be_a_labelled_probability(
    target: pd.Series,
) -> None:
    candidates = pd.Series(["a", "b"], index=[1, 2])

    with pytest.raises(ValueError, match="target source mass"):
        source_target_weights(target, candidates)


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
