from __future__ import annotations

import pandas as pd
import pytest

from provenance_sdm.summaries import (
    diagnostic_arm_effects,
    hierarchical_bootstrap,
    hierarchical_effect_bootstrap,
    mechanism_correlations,
    oriented_paired_effects,
    paired_effects,
)


@pytest.fixture
def fake_metrics() -> pd.DataFrame:
    rows = []
    for community_seed in (101, 202):
        for species_index in range(2):
            for arm in ("conventional_tgb", "pm_tgb"):
                conventional = 0.50 + species_index * 0.05
                rows.append(
                    {
                        "community_seed": community_seed,
                        "alignment": "low",
                        "bias_level": "strong",
                        "species_id": f"sp_{species_index:03d}",
                        "background_arm": arm,
                        "suitability_spearman": conventional
                        + (0.10 if arm == "pm_tgb" else 0.0),
                        "integrated_error": 0.40
                        - (0.05 if arm == "pm_tgb" else 0.0),
                        "unbiased_auc": 0.65
                        + (0.04 if arm == "pm_tgb" else 0.0),
                        "response_curve_error": 0.30
                        - (0.03 if arm == "pm_tgb" else 0.0),
                        "top10_overlap": 0.45
                        + (0.08 if arm == "pm_tgb" else 0.0),
                        "record_count": 100 + species_index * 50,
                        "niche_breadth": 0.8 + species_index * 0.4,
                        "source_distribution_distance": 0.6,
                        "unsupported_mass": 0.1 if arm == "pm_tgb" else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_primary_effect_is_pm_minus_conventional(fake_metrics) -> None:
    effects = paired_effects(fake_metrics)

    correlations = effects.query("metric == 'suitability_spearman'")
    errors = effects.query("metric == 'integrated_error'")
    assert correlations.effect.tolist() == pytest.approx([0.1] * 4)
    assert errors.effect.tolist() == pytest.approx([-0.05] * 4)
    assert set(effects.contrast) == {"pm_tgb_minus_conventional_tgb"}


def test_incomplete_primary_pair_is_rejected(fake_metrics) -> None:
    incomplete = fake_metrics.drop(index=fake_metrics.index[-1])

    with pytest.raises(ValueError, match="complete"):
        paired_effects(incomplete)


def test_hierarchical_bootstrap_is_deterministic(fake_metrics) -> None:
    first = hierarchical_bootstrap(fake_metrics, n_boot=100, seed=4)
    second = hierarchical_bootstrap(fake_metrics, n_boot=100, seed=4)

    pd.testing.assert_frame_equal(first, second)
    assert set(first) >= {
        "metric",
        "alignment",
        "bias_level",
        "estimate",
        "lower",
        "upper",
        "n_pairs",
    }
    assert first.lower.le(first.estimate).all()
    assert first.upper.ge(first.estimate).all()


def test_oriented_effects_make_lower_errors_positive(fake_metrics) -> None:
    effects = oriented_paired_effects(fake_metrics)

    correlations = effects.query("metric == 'suitability_spearman'")
    integrated = effects.query("metric == 'integrated_error'")
    curves = effects.query("metric == 'response_curve_error'")
    assert correlations.oriented_effect.tolist() == pytest.approx([0.1] * 4)
    assert integrated.oriented_effect.tolist() == pytest.approx([0.05] * 4)
    assert curves.oriented_effect.tolist() == pytest.approx([0.03] * 4)
    assert set(effects.orientation) == {-1, 1}


def test_diagnostic_arm_effects_align_latent_with_both_tgb_arms(
    fake_metrics,
) -> None:
    conventional = fake_metrics.query(
        "background_arm == 'conventional_tgb'"
    ).copy()
    latent = conventional.assign(background_arm="latent_mixture_tgb")
    latent["suitability_spearman"] += 0.04
    latent["integrated_error"] -= 0.02

    effects = diagnostic_arm_effects(fake_metrics, latent)

    suitability = effects.query("metric == 'suitability_spearman'")
    integrated = effects.query("metric == 'integrated_error'")
    assert set(effects.contrast) == {
        "latent_mixture_tgb_minus_conventional_tgb",
        "latent_mixture_tgb_minus_pm_tgb",
    }
    assert suitability.query(
        "contrast == 'latent_mixture_tgb_minus_conventional_tgb'"
    ).oriented_effect.tolist() == pytest.approx([0.04] * 4)
    assert integrated.query(
        "contrast == 'latent_mixture_tgb_minus_conventional_tgb'"
    ).oriented_effect.tolist() == pytest.approx([0.02] * 4)

    first = hierarchical_effect_bootstrap(effects, n_boot=40, seed=8)
    second = hierarchical_effect_bootstrap(effects, n_boot=40, seed=8)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 10
    assert first.lower.le(first.estimate).all()
    assert first.upper.ge(first.estimate).all()


@pytest.fixture
def fake_diagnostics(fake_metrics) -> pd.DataFrame:
    pairs = fake_metrics.loc[
        :, ["community_seed", "alignment", "bias_level", "species_id"]
    ].drop_duplicates()
    pairs["ecological_overlap_tv"] = pairs.species_id.map(
        {"sp_000": 0.1, "sp_001": 0.8}
    )
    return pairs


def test_mechanism_correlations_are_deterministic_and_hierarchy_labelled(
    fake_metrics,
    fake_diagnostics,
) -> None:
    variable = fake_metrics.copy()
    species_gain = variable.species_id.map({"sp_000": 0.02, "sp_001": 0.12})
    pm = variable.background_arm.eq("pm_tgb")
    variable.loc[pm, "suitability_spearman"] = (
        variable.loc[pm, "suitability_spearman"] - 0.1 + species_gain.loc[pm]
    )
    first = mechanism_correlations(
        variable,
        fake_diagnostics,
        n_boot=40,
        seed=7,
    )
    second = mechanism_correlations(
        variable,
        fake_diagnostics,
        n_boot=40,
        seed=7,
    )

    pd.testing.assert_frame_equal(first, second)
    suitability = first.query("metric == 'suitability_spearman'").iloc[0]
    assert suitability.estimate == pytest.approx(1.0)
    assert suitability.n_pairs == 4
    assert suitability.bootstrap_draws == 40
    assert suitability.bootstrap_seed == 7


def test_mechanism_correlations_require_unique_complete_diagnostics(
    fake_metrics,
    fake_diagnostics,
) -> None:
    duplicate = pd.concat(
        [fake_diagnostics, fake_diagnostics.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="unique"):
        mechanism_correlations(fake_metrics, duplicate, n_boot=20)
    with pytest.raises(ValueError, match="complete"):
        mechanism_correlations(
            fake_metrics,
            fake_diagnostics.iloc[:-1],
            n_boot=20,
        )
