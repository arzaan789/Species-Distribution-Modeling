from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from provenance_sdm.figures import (
    _workflow_geometry,
    empirical_map_difference,
    write_empirical_figures,
    write_mechanism_figures,
    write_simulation_figures,
)


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

    assert {path.name for path in paths} == {
        "simulation_workflow.png",
        "paired_truth_contrasts.png",
    }
    assert all(path.suffix == ".png" for path in paths)
    assert all(path.stat().st_size > 1_000 for path in paths)
    assert all(
        word not in path.name
        for path in paths
        for word in ("improvement", "superior", "failure")
    )


def test_workflow_geometry_keeps_boxes_separate_and_arrows_forward() -> None:
    boxes, arrows = _workflow_geometry(7)

    gaps = boxes[1:, 0] - boxes[:-1, 1]
    assert (gaps > 0.01).all()
    assert (arrows[:, 1] > arrows[:, 0]).all()
    assert (arrows[:, 0] >= boxes[:-1, 1]).all()
    assert (arrows[:, 1] <= boxes[1:, 0]).all()


def test_empirical_figures_export_source_and_map_panels(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "species": species,
                "block_width_m": 50_000,
                "fold_id": fold,
                "provenance_level": "dataset",
                "background_arm": arm,
                "auc": 0.6 + 0.02 * fold + (0.03 if arm == "pm_tgb" else 0),
                "source_distance": 0.2 + 0.1 * fold,
                "unsupported_mass": 0.1 if arm == "pm_tgb" else 0.0,
            }
            for species in ("hare", "squirrel")
            for fold in range(3)
            for arm in ("uniform", "conventional_tgb", "pm_tgb")
        ]
    )
    maps = pd.DataFrame(
        [
            {
                "cell_id": cell_id,
                "x": cell_id % 4,
                "y": cell_id // 4,
                "species": "hare",
                "background_arm": arm,
                "predicted_suitability": (cell_id + 1)
                * (1.2 if arm == "pm_tgb" else 1.0),
            }
            for arm in ("uniform", "conventional_tgb", "pm_tgb")
            for cell_id in range(16)
        ]
    )

    paths = write_empirical_figures(metrics, maps, tmp_path)

    assert len(paths) == 2
    assert all(path.stat().st_size > 1_000 for path in paths)


def test_mechanism_figures_export_distortion_and_latent_contrasts(
    tmp_path: Path,
) -> None:
    primary = figure_metrics()
    pairs = primary.loc[
        :, ["community_seed", "alignment", "bias_level", "species_id"]
    ].drop_duplicates()
    pairs["ecological_overlap_tv"] = (
        pairs.species_id.str.extract(r"(\d+)$")[0].astype(int) + 1
    ) / 10
    conventional = primary.query(
        "background_arm == 'conventional_tgb'"
    ).copy()
    latent = conventional.assign(background_arm="latent_mixture_tgb")
    latent["suitability_spearman"] += 0.03

    paths = write_mechanism_figures(
        primary,
        pairs,
        latent,
        tmp_path,
        n_boot=20,
        seed=4,
    )

    assert {path.name for path in paths} == {
        "source_composition_mechanism.png",
        "latent_mixture_contrasts.png",
    }
    assert all(path.stat().st_size > 1_000 for path in paths)


def test_empirical_map_difference_selects_first_species_and_scales_mass() -> None:
    rows = []
    for species in ("zeta_species", "alpha_species"):
        for arm, values in (
            ("conventional_tgb", (0.6, 0.4)),
            ("pm_tgb", (0.5, 0.5)),
        ):
            for cell_id, value in enumerate(values):
                rows.append(
                    {
                        "cell_id": cell_id,
                        "x": cell_id,
                        "y": 0,
                        "species": species,
                        "background_arm": arm,
                        "predicted_suitability": value,
                    }
                )

    difference = empirical_map_difference(pd.DataFrame(rows))

    assert set(difference.species) == {"alpha_species"}
    assert difference.difference_per_million.tolist() == pytest.approx(
        [-100_000, 100_000]
    )
