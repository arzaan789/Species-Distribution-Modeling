from __future__ import annotations

import numpy as np
import pytest

from provenance_sdm.landscape import landscape_from_arrays


def valid_arrays() -> dict[str, object]:
    return {
        "predictors": {
            "temperature": np.array([1.0, 2.0, 3.0, 4.0]),
            "rainfall": np.array([8.0, 6.0, 4.0, 2.0]),
            "woodland": np.array([0.0, 1.0, 0.0, 1.0]),
        },
        "x": np.array([0.0, 1_000.0, 0.0, 1_000.0]),
        "y": np.array([0.0, 0.0, 1_000.0, 1_000.0]),
        "area": np.ones(4),
        "crs": "EPSG:27700",
    }


def test_landscape_rejects_geographic_coordinates() -> None:
    values = valid_arrays()
    values["crs"] = "EPSG:4326"

    with pytest.raises(ValueError, match="projected"):
        landscape_from_arrays(**values)


def test_landscape_standardizes_only_predictors() -> None:
    landscape = landscape_from_arrays(**valid_arrays())

    for feature in landscape.feature_names:
        assert landscape.cells[feature].mean() == pytest.approx(0.0, abs=1e-12)
        assert landscape.cells[feature].std(ddof=0) == pytest.approx(1.0)
    assert landscape.cells.x.tolist() == [0.0, 1_000.0, 0.0, 1_000.0]
    assert landscape.cells.area_weight.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert landscape.feature_means == pytest.approx(
        {"temperature": 2.5, "rainfall": 5.0, "woodland": 0.5}
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("too_few_features", "three"),
        ("constant_feature", "constant"),
        ("mismatched_length", "length"),
        ("non_finite_coordinate", "finite"),
        ("non_finite_feature", "finite"),
        ("non_positive_area", "positive"),
    ],
)
def test_invalid_landscape_inputs_are_rejected(
    mutation: str,
    message: str,
) -> None:
    values = valid_arrays()
    if mutation == "too_few_features":
        values["predictors"].pop("woodland")
    elif mutation == "constant_feature":
        values["predictors"]["woodland"] = np.zeros(4)
    elif mutation == "mismatched_length":
        values["y"] = np.array([0.0, 1.0])
    elif mutation == "non_finite_coordinate":
        values["x"][2] = np.nan
    elif mutation == "non_finite_feature":
        values["predictors"]["temperature"][1] = np.inf
    elif mutation == "non_positive_area":
        values["area"][0] = 0.0

    with pytest.raises(ValueError, match=message):
        landscape_from_arrays(**values)
