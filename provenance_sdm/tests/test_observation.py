from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.observation import (
    SpeciesEffortMapping,
    simulate_observations,
    simulate_programmes,
)
from provenance_sdm.virtual_species import simulate_species_truth


@pytest.fixture
def toy_landscape():
    side = np.linspace(-2.5, 2.5, 15)
    x, y = np.meshgrid(side * 10_000.0, side * 10_000.0)
    return landscape_from_arrays(
        {
            "env_1": x.ravel() / 10_000.0,
            "env_2": y.ravel() / 10_000.0,
            "env_3": np.sin(x.ravel() / 10_000.0)
            + np.cos(y.ravel() / 10_000.0),
        },
        x=x.ravel(),
        y=y.ravel(),
        area=np.ones(x.size),
        crs="EPSG:27700",
    )


def test_programme_effort_is_positive_normalized_and_deterministic(
    toy_landscape,
) -> None:
    first = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="strong", seed=3
    )
    second = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="strong", seed=3
    )

    assert [item.programme_id for item in first] == [f"programme_{i}" for i in range(6)]
    for left, right in zip(first, second, strict=True):
        assert np.all(left.intensity > 0)
        assert left.intensity.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(left.intensity, right.intensity)


def test_strong_programme_bias_is_more_concentrated_than_moderate(
    toy_landscape,
) -> None:
    moderate = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="moderate", seed=4
    )
    strong = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="strong", seed=4
    )

    moderate_concentration = np.mean(
        [np.square(item.intensity).sum() for item in moderate]
    )
    strong_concentration = np.mean(
        [np.square(item.intensity).sum() for item in strong]
    )
    assert strong_concentration > moderate_concentration


def test_observed_cells_follow_suitability_times_species_effort(
    toy_landscape,
) -> None:
    truths = simulate_species_truth(toy_landscape, n_species=20, seed=7)
    programmes = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="strong", seed=8
    )

    observed = simulate_observations(
        truths,
        programmes,
        alignment="partial",
        bias_level="strong",
        min_records=10_000,
        max_records=10_000,
        seed=9,
    )

    assert observed.species_effort["sp_000"].dtype == np.float32
    expected = truths[0].suitability * observed.species_effort["sp_000"]
    expected /= expected.sum()
    actual = (
        observed.records.query("species_id == 'sp_000'")
        .cell_id.value_counts(normalize=True)
        .reindex(range(len(expected)), fill_value=0.0)
        .to_numpy()
    )
    assert np.corrcoef(actual, expected)[0, 1] > 0.95


def test_species_effort_is_computed_lazily_without_retaining_cell_arrays(
    toy_landscape,
) -> None:
    truths = simulate_species_truth(toy_landscape, n_species=20, seed=7)
    programmes = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="strong", seed=8
    )
    observed = simulate_observations(
        truths,
        programmes,
        alignment="partial",
        bias_level="strong",
        min_records=20,
        max_records=20,
        seed=9,
    )

    assert isinstance(observed.species_effort, SpeciesEffortMapping)
    assert not any(
        isinstance(value, np.ndarray)
        for value in vars(observed.species_effort).values()
    )

    mixture = (
        observed.source_mixtures.query("species_id == 'sp_000'")
        .set_index("programme_id")
        .reindex([item.programme_id for item in programmes])
        .weight.to_numpy(dtype=float)
    )
    expected = mixture @ np.vstack([item.intensity for item in programmes])
    expected /= expected.sum()
    expected = np.asarray(expected, dtype=np.float32)

    np.testing.assert_array_equal(observed.species_effort["sp_000"], expected)
    np.testing.assert_array_equal(
        observed.species_effort["sp_000"],
        observed.species_effort["sp_000"],
    )


def test_long_tail_counts_stay_within_declared_bounds(toy_landscape) -> None:
    truths = simulate_species_truth(toy_landscape, n_species=100, seed=1)
    programmes = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="moderate", seed=2
    )

    observed = simulate_observations(
        truths,
        programmes,
        alignment="low",
        bias_level="moderate",
        min_records=20,
        max_records=2_000,
        seed=3,
    )

    counts = observed.records.groupby("species_id").size()
    assert counts.min() >= 20
    assert counts.max() <= 2_000
    assert counts.quantile(0.9) > counts.median()


def test_high_alignment_increases_within_taxonomic_group_similarity(
    toy_landscape,
) -> None:
    truths = simulate_species_truth(toy_landscape, n_species=100, seed=10)
    programmes = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="moderate", seed=11
    )
    low = simulate_observations(
        truths, programmes, "low", "moderate", 20, 20, seed=12
    )
    high = simulate_observations(
        truths, programmes, "high", "moderate", 20, 20, seed=12
    )

    def within_group_distance(mixtures: pd.DataFrame) -> float:
        wide = mixtures.pivot(
            index=["species_id", "taxonomic_group"],
            columns="programme_id",
            values="weight",
        )
        distances = []
        for _, group in wide.groupby(level="taxonomic_group"):
            rows = group.to_numpy()
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    distances.append(np.abs(rows[left] - rows[right]).sum() / 2)
        return float(np.mean(distances))

    assert within_group_distance(high.source_mixtures) < within_group_distance(
        low.source_mixtures
    )


@pytest.mark.parametrize(
    ("alignment", "bias_level", "message"),
    [
        ("unknown", "moderate", "alignment"),
        ("low", "extreme", "bias"),
    ],
)
def test_unknown_observation_scenarios_are_rejected(
    toy_landscape,
    alignment: str,
    bias_level: str,
    message: str,
) -> None:
    truths = simulate_species_truth(toy_landscape, n_species=20, seed=1)
    programmes = simulate_programmes(
        toy_landscape, n_programmes=6, bias_level="moderate", seed=2
    )

    with pytest.raises(ValueError, match=message):
        simulate_observations(
            truths, programmes, alignment, bias_level, 20, 30, seed=3
        )
