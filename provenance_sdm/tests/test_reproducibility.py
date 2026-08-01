from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.reproducibility import (
    build_reproducibility_audit,
    export_manuscript,
)
from provenance_sdm.simulation_runner import (
    PRIMARY_METRICS,
    expected_simulation_keys,
)


def complete_tiny_run(
    root: Path,
    study_config,
) -> object:
    output = root / "outputs"
    figures = root / "manuscript" / "figures"
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    simulation = replace(
        study_config.simulation,
        n_species=2,
        n_communities=1,
        alignments=("low",),
        bias_levels=("moderate",),
        background_cells=10,
        minimum_background_cells=5,
    )
    config = replace(study_config, simulation=simulation, output_dir=output)
    rows = expected_simulation_keys(config)
    for metric in PRIMARY_METRICS:
        rows[metric] = 0.5
    rows["background_cells"] = 10
    rows["landscape_hash"] = "landscape-hash"
    rows["record_count"] = 100
    rows["niche_breadth"] = 1.0
    rows["source_distribution_distance"] = 0.4
    rows["feature_basis"] = "linear"
    rows["max_cell_mass"] = 0.01
    rows["effective_cell_count"] = 100.0
    rows["log_intensity_range"] = 4.0
    rows["lower_clip_cells"] = 0
    rows["lower_clip_fraction"] = 0.0
    rows["solver_converged"] = True
    rows["model_regularization"] = 2.0
    rows["unsupported_mass"] = np.where(
        rows.background_arm.eq("pm_tgb"),
        0.1,
        0.0,
    )
    rows.to_parquet(output / "simulation_metrics.parquet", index=False)

    empirical_rows = []
    for species in config.empirical_species:
        for width in (25_000, 50_000, 100_000):
            for fold in range(5):
                for provenance in ("dataset", "publisher"):
                    for arm in ("uniform", "conventional_tgb", "pm_tgb"):
                        empirical_rows.append(
                            {
                                "species": species.key,
                                "scientific_name": species.scientific_name,
                                "block_width_m": width,
                                "fold_id": fold,
                                "provenance_level": provenance,
                                "background_arm": arm,
                                "background_cells": 10,
                                "evaluation_hash": f"{species.key}-{width}-{fold}",
                                "auc": 0.65,
                                "boyce": 0.3,
                                "boyce_defined": True,
                                "map_spearman": 0.8,
                                "upper_area_overlap": 0.7,
                                "upper_area_shift": 0.0,
                                "centroid_shift_m": 1_000.0,
                                "source_distance": 0.4,
                                "unsupported_mass": 0.1
                                if arm == "pm_tgb"
                                else 0.0,
                                "feature_basis": "linear",
                                "max_cell_mass": 0.01,
                                "effective_cell_count": 100.0,
                                "log_intensity_range": 4.0,
                                "lower_clip_cells": 0,
                                "lower_clip_fraction": 0.0,
                                "solver_converged": True,
                                "model_regularization": 2.0,
                            }
                        )
    pd.DataFrame(empirical_rows).to_parquet(
        output / "empirical_metrics.parquet",
        index=False,
    )
    map_rows = []
    for arm in ("uniform", "conventional_tgb", "pm_tgb"):
        for cell_id in range(16):
            map_rows.append(
                {
                    "cell_id": cell_id,
                    "x": cell_id % 4,
                    "y": cell_id // 4,
                    "species": config.empirical_species[0].key,
                    "background_arm": arm,
                    "predicted_suitability": (cell_id + 1)
                    * (1.01 if arm == "pm_tgb" else 1.0),
                }
            )
    pd.DataFrame(map_rows).to_parquet(
        output / "empirical_maps.parquet",
        index=False,
    )
    assignment_rows = []
    block_rows = []
    for species in config.empirical_species:
        for width in (25_000, 50_000, 100_000):
            for fold in range(5):
                assignment_rows.extend(
                    {
                        "species": species.key,
                        "block_width_m": width,
                        "row_index": fold * 2 + label,
                        "block_id": f"{fold}:0",
                        "fold_id": fold,
                        "label": label,
                        "cell_id": fold * 2 + label,
                    }
                    for label in (0, 1)
                )
                block_rows.append(
                    {
                        "species": species.key,
                        "block_width_m": width,
                        "fold_id": fold,
                        "block_id": f"{fold}:0",
                        "rows": 2,
                        "positive_rows": 1,
                        "negative_rows": 1,
                    }
                )
    pd.DataFrame(assignment_rows).to_parquet(
        output / "spatial_fold_assignments.parquet",
        index=False,
    )
    pd.DataFrame(block_rows).to_csv(
        output / "spatial_block_class_audit.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "stage": [
                "input",
                "allowed_taxa",
                "gb_year_status",
                "finite_coordinates",
                "no_geospatial_issue",
                "deduplicated",
                "valid_predictor_cell",
                "complete_provenance",
            ],
            "records": [1_000, 1_000, 980, 970, 960, 900, 820, 800],
            "removed": [0, 0, 20, 10, 10, 60, 80, 20],
        }
    ).to_csv(output / "occurrence_cleaning_audit.csv", index=False)
    (output / "gbif_archive.json").write_text(
        json.dumps(
            {
                "download_key": "0018113-260721160103020",
                "doi": "10.15468/dl.example",
                "total_records": 1_000,
                "archive_sha256": "a" * 64,
                "retrieved_at": "2026-07-30T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (output / "gb_grid.manifest.json").write_text(
        json.dumps({"source_sha256": "b" * 64, "retained_cells": 100}),
        encoding="utf-8",
    )
    (output / "deepmaxent_gate.json").write_text(
        json.dumps({"include": True, "reasons": []}),
        encoding="utf-8",
    )
    for name in (
        "simulation_workflow.png",
        "paired_truth_contrasts.png",
        "empirical_source_contrasts.png",
        "empirical_map_contrast.png",
    ):
        (figures / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 2_000)
    return config


def test_reproducibility_audit_requires_core_artifacts(
    tmp_path: Path,
    study_config,
) -> None:
    with pytest.raises(FileNotFoundError, match="simulation_metrics"):
        build_reproducibility_audit(tmp_path, study_config)


def test_bat_tokens_are_absent_from_submission_artifacts(
    tmp_path: Path,
    study_config,
) -> None:
    config = complete_tiny_run(tmp_path, study_config)

    audit = build_reproducibility_audit(tmp_path, config)

    assert audit["core_status"] == "passed"
    assert audit["excluded_taxon_scan"]["matches"] == []
    assert audit["deepmaxent"]["status"] == "included"


def test_manuscript_export_writes_four_auditable_tables(
    tmp_path: Path,
    study_config,
) -> None:
    config = complete_tiny_run(tmp_path, study_config)

    paths = export_manuscript(
        tmp_path,
        config,
        tmp_path / "submission",
        n_boot=20,
    )

    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    for path in paths:
        table = pd.read_csv(path)
        assert {"status", "sample_count", "units", "method"} <= set(table)


def test_reproducibility_audit_rejects_stale_nonlinear_results(
    tmp_path: Path,
    study_config,
) -> None:
    config = complete_tiny_run(tmp_path, study_config)
    path = tmp_path / "outputs" / "simulation_metrics.parquet"
    rows = pd.read_parquet(path)
    rows["feature_basis"] = "linear_quadratic_interactions"
    rows.to_parquet(path, index=False)

    audit = build_reproducibility_audit(tmp_path, config)

    assert audit["core_status"] == "failed"
    assert audit["linear_feature_basis"] is False


def test_reproducibility_audit_rejects_stale_regularization(
    tmp_path: Path,
    study_config,
) -> None:
    config = complete_tiny_run(tmp_path, study_config)
    path = tmp_path / "outputs" / "empirical_metrics.parquet"
    rows = pd.read_parquet(path)
    rows.loc[0, "model_regularization"] = 1.0
    rows.to_parquet(path, index=False)

    audit = build_reproducibility_audit(tmp_path, config)

    assert audit["core_status"] == "failed"
    assert audit["checks"]["primary_regularization"] is False
