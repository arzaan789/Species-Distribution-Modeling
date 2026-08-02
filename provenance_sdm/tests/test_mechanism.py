from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.mechanism import (
    expected_source_composition,
    mechanism_row,
    total_variation,
)
from provenance_sdm.observation import ObservedCommunity, ProgrammeEffort
from provenance_sdm.virtual_species import SpeciesTruth


@pytest.fixture
def manual_observed() -> ObservedCommunity:
    landscape = landscape_from_arrays(
        {
            "env_1": np.array([0.0, 1.0]),
            "env_2": np.array([1.0, 0.0]),
            "env_3": np.array([-1.0, 1.0]),
        },
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 0.0]),
        area=np.ones(2),
        crs="EPSG:27700",
    )
    truth = SpeciesTruth(
        species_id="sp_000",
        taxonomic_group=0,
        coefficients=np.array([0.0]),
        niche_breadth=1.25,
        suitability=np.array([1.0, 0.0], dtype=np.float32),
        landscape=landscape,
    )
    programmes = (
        ProgrammeEffort(
            programme_id="programme_0",
            intensity=np.array([0.8, 0.2]),
            bias_level="moderate",
        ),
        ProgrammeEffort(
            programme_id="programme_1",
            intensity=np.array([0.2, 0.8]),
            bias_level="moderate",
        ),
    )
    records = pd.DataFrame(
        {
            "record_id": range(4),
            "species_id": ["sp_000"] * 4,
            "taxonomic_group": [0] * 4,
            "cell_id": [0] * 4,
            "programme_id": ["programme_0"] * 3 + ["programme_1"],
        }
    )
    source_mixtures = pd.DataFrame(
        {
            "species_id": ["sp_000", "sp_000"],
            "taxonomic_group": [0, 0],
            "programme_id": ["programme_0", "programme_1"],
            "weight": [0.5, 0.5],
        }
    )
    return ObservedCommunity(
        records=records,
        species_effort={"sp_000": np.array([0.5, 0.5])},
        programme_effort=programmes,
        source_mixtures=source_mixtures,
        truth=(truth,),
        landscape=landscape,
    )


def test_expected_composition_includes_ecological_overlap(
    manual_observed: ObservedCommunity,
) -> None:
    expected = expected_source_composition(manual_observed, "sp_000")

    assert expected.sum() == pytest.approx(1.0)
    assert expected.to_dict() == pytest.approx(
        {"programme_0": 0.8, "programme_1": 0.2}
    )


def test_total_variation_aligns_source_labels_and_is_symmetric() -> None:
    left = pd.Series({"a": 0.75, "b": 0.25})
    right = pd.Series({"a": 0.25, "c": 0.75})

    assert total_variation(left, right) == pytest.approx(0.75)
    assert total_variation(left, right) == total_variation(right, left)


@pytest.mark.parametrize(
    "invalid",
    (
        pd.Series({"a": 0.4, "b": 0.4}),
        pd.Series({"a": -0.1, "b": 1.1}),
        pd.Series({"a": np.nan, "b": 1.0}),
    ),
)
def test_total_variation_rejects_invalid_probability_mass(
    invalid: pd.Series,
) -> None:
    with pytest.raises(ValueError, match="mass"):
        total_variation(invalid, pd.Series({"a": 0.5, "b": 0.5}))


def test_mechanism_row_reports_distinct_distortion_components(
    manual_observed: ObservedCommunity,
) -> None:
    row = mechanism_row(manual_observed, "sp_000")

    assert row == pytest.approx(
        {
            "record_count": 4,
            "niche_breadth": 1.25,
            "ecological_overlap_tv": 0.30,
            "finite_record_tv": 0.05,
            "total_composition_tv": 0.25,
        }
    )


def test_mechanism_functions_reject_unknown_focal_species(
    manual_observed: ObservedCommunity,
) -> None:
    with pytest.raises(ValueError, match="unknown focal species"):
        expected_source_composition(manual_observed, "sp_999")
    with pytest.raises(ValueError, match="unknown focal species"):
        mechanism_row(manual_observed, "sp_999")
