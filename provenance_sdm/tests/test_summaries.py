from __future__ import annotations

import pandas as pd
import pytest

from provenance_sdm.summaries import hierarchical_bootstrap, paired_effects


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
