from __future__ import annotations

import numpy as np
import pytest

from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.virtual_species import simulate_species_truth


@pytest.fixture
def toy_landscape():
    side = np.linspace(-2.0, 2.0, 12)
    x, y = np.meshgrid(side * 10_000, side * 10_000)
    predictors = {
        "env_1": x.ravel() / 10_000,
        "env_2": y.ravel() / 10_000,
        "env_3": np.sin(x.ravel() / 10_000) + np.cos(y.ravel() / 10_000),
    }
    return landscape_from_arrays(
        predictors,
        x=x.ravel(),
        y=y.ravel(),
        area=np.full(x.size, 1_000_000.0),
        crs="EPSG:27700",
    )


def test_species_truth_is_deterministic(toy_landscape) -> None:
    first = simulate_species_truth(toy_landscape, n_species=20, seed=7)
    second = simulate_species_truth(toy_landscape, n_species=20, seed=7)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.suitability, right.suitability)
        np.testing.assert_allclose(left.coefficients, right.coefficients)
        assert left.niche_breadth == right.niche_breadth


def test_species_truth_is_normalized_positive_and_non_constant(toy_landscape) -> None:
    truths = simulate_species_truth(toy_landscape, n_species=20, seed=7)

    assert all(np.isclose(truth.suitability.sum(), 1.0) for truth in truths)
    assert all(np.all(truth.suitability > 0) for truth in truths)
    assert all(np.ptp(truth.suitability) > 0 for truth in truths)
    assert all(0.5 <= truth.niche_breadth <= 2.0 for truth in truths)


def test_species_ids_and_taxonomic_groups_are_balanced(toy_landscape) -> None:
    truths = simulate_species_truth(toy_landscape, n_species=20, seed=9)

    assert [truth.species_id for truth in truths[:3]] == [
        "sp_000",
        "sp_001",
        "sp_002",
    ]
    assert np.bincount([truth.taxonomic_group for truth in truths]).tolist() == [
        2
    ] * 10


@pytest.mark.parametrize("n_species", [0, -1])
def test_non_positive_species_count_is_rejected(toy_landscape, n_species: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        simulate_species_truth(toy_landscape, n_species=n_species, seed=1)
