from __future__ import annotations

from pathlib import Path

import pandas as pd

from provenance_sdm.figures import write_simulation_figures


def figure_metrics() -> pd.DataFrame:
    rows = []
    for community_seed in (11, 22):
        for alignment_index, alignment in enumerate(("low", "partial", "high")):
            for species_index in range(3):
                for arm in ("conventional_tgb", "pm_tgb"):
                    adjustment = 0.08 if arm == "pm_tgb" else 0.0
                    rows.append(
                        {
                            "community_seed": community_seed,
                            "alignment": alignment,
                            "bias_level": "strong",
                            "species_id": f"sp_{species_index:03d}",
                            "background_arm": arm,
                            "suitability_spearman": 0.4
                            + alignment_index * 0.05
                            + adjustment,
                            "integrated_error": 0.4 - adjustment,
                            "unbiased_auc": 0.6 + adjustment,
                            "response_curve_error": 0.3 - adjustment,
                            "top10_overlap": 0.4 + adjustment,
                            "record_count": 50 + species_index * 100,
                            "niche_breadth": 0.7 + species_index * 0.4,
                            "source_distribution_distance": 0.7
                            - alignment_index * 0.2,
                            "unsupported_mass": 0.1
                            if arm == "pm_tgb"
                            else 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_simulation_figures_are_nonempty_neutral_pngs(tmp_path: Path) -> None:
    paths = write_simulation_figures(figure_metrics(), tmp_path)

    assert len(paths) == 3
    assert all(path.suffix == ".png" for path in paths)
    assert all(path.stat().st_size > 1_000 for path in paths)
    assert all(
        word not in path.name
        for path in paths
        for word in ("improvement", "superior", "failure")
    )
