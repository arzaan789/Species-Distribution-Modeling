from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.backgrounds import (
    make_backgrounds,
    make_latent_mixture_background,
)
from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.observation import simulate_observations, simulate_programmes
from provenance_sdm.virtual_species import simulate_species_truth


@pytest.fixture
def observed_community():
    side = np.linspace(-3.0, 3.0, 20)
    x, y = np.meshgrid(side * 10_000.0, side * 10_000.0)
    landscape = landscape_from_arrays(
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
    truths = simulate_species_truth(landscape, n_species=20, seed=20)
    programmes = simulate_programmes(
        landscape, n_programmes=6, bias_level="strong", seed=21
    )
    return simulate_observations(
        truths, programmes, "partial", "strong", 500, 500, seed=22
    )


def test_all_arms_have_same_unique_cell_budget(observed_community) -> None:
    arms = make_backgrounds(
        observed_community,
        focal_species="sp_000",
        n_cells=20,
        seed=3,
    )

    assert set(arms) == {
        "uniform",
        "conventional_tgb",
        "pm_tgb",
        "oracle_effort",
    }
    assert {len(frame) for frame in arms.values()} == {20}
    assert all(frame.cell_id.is_unique for frame in arms.values())
    assert all(set(observed_community.landscape.feature_names) <= set(frame) for frame in arms.values())


def test_non_oracle_arms_do_not_expose_truth_columns(observed_community) -> None:
    arms = make_backgrounds(
        observed_community,
        focal_species="sp_000",
        n_cells=20,
        seed=3,
    )

    for name in ("uniform", "conventional_tgb", "pm_tgb"):
        assert "true_effort" not in arms[name]
        assert "suitability" not in arms[name]
    assert "true_effort" in arms["oracle_effort"]


def test_taxonomic_background_uses_other_species_candidate_cells(
    observed_community,
) -> None:
    arms = make_backgrounds(
        observed_community,
        focal_species="sp_000",
        n_cells=20,
        seed=5,
    )
    candidate_cells = set(
        observed_community.records.query(
            "taxonomic_group == 0 and species_id != 'sp_000'"
        ).cell_id
    )

    assert set(arms["conventional_tgb"].cell_id) <= candidate_cells
    assert set(arms["pm_tgb"].cell_id) <= candidate_cells


def test_budget_adapts_to_common_target_group_support(
    observed_community,
) -> None:
    candidate_count = observed_community.records.query(
        "taxonomic_group == 0 and species_id != 'sp_000'"
    ).cell_id.nunique()

    arms = make_backgrounds(
        observed_community,
        focal_species="sp_000",
        n_cells=candidate_count + 1,
        minimum_cells=5,
        seed=5,
    )

    budgets = {len(frame) for frame in arms.values()}
    assert len(budgets) == 1
    assert budgets.pop() <= candidate_count


def test_common_support_below_minimum_raises_diagnostic(
    observed_community,
) -> None:
    candidate_count = observed_community.records.query(
        "taxonomic_group == 0 and species_id != 'sp_000'"
    ).cell_id.nunique()

    with pytest.raises(ValueError, match="unique target-group cells"):
        make_backgrounds(
            observed_community,
            focal_species="sp_000",
            n_cells=candidate_count + 1,
            minimum_cells=candidate_count + 1,
            seed=5,
        )


def test_latent_mixture_background_uses_candidate_cells_and_exact_budget(
    observed_community,
) -> None:
    frame = make_latent_mixture_background(
        observed_community,
        focal_species="sp_000",
        n_cells=20,
        seed=17,
    )
    candidate_cells = set(
        observed_community.records.query(
            "taxonomic_group == 0 and species_id != 'sp_000'"
        ).cell_id
    )

    assert len(frame) == 20
    assert frame.cell_id.is_unique
    assert set(frame.cell_id) <= candidate_cells
    assert set(frame.background_arm) == {"latent_mixture_tgb"}
    assert "unsupported_mass" in frame
    assert "true_effort" not in frame
    assert "suitability" not in frame


def test_latent_mixture_background_is_seed_deterministic(
    observed_community,
) -> None:
    first = make_latent_mixture_background(
        observed_community, "sp_000", 20, seed=31
    )
    second = make_latent_mixture_background(
        observed_community, "sp_000", 20, seed=31
    )

    pd.testing.assert_frame_equal(first, second)
