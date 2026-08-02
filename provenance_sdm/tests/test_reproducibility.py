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
from provenance_sdm.flexible_runner import expected_flexible_keys
from provenance_sdm.mechanism_runner import expected_mechanism_keys
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
        "source_composition_mechanism.png",
        "latent_mixture_contrasts.png",
        "empirical_source_contrasts.png",
        "empirical_map_contrast.png",
    ):
        (figures / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 2_000)

    mechanism = expected_mechanism_keys(config)
    mechanism["record_count"] = 100
    mechanism["niche_breadth"] = 1.0
    mechanism["source_distribution_distance"] = [0.2, 0.6]
    mechanism["ecological_overlap_tv"] = [0.1, 0.5]
    mechanism["finite_record_tv"] = [0.02, 0.04]
    mechanism["total_composition_tv"] = [0.11, 0.52]
    mechanism["landscape_hash"] = "landscape-hash"
    mechanism.to_parquet(
        output / "mechanism_diagnostics.parquet",
        index=False,
    )
    latent = mechanism.loc[
        :,
        ["community_seed", "alignment", "bias_level", "species_id"],
    ].copy()
    latent["background_arm"] = "latent_mixture_tgb"
    for metric in PRIMARY_METRICS:
        latent[metric] = 0.5
    latent["record_count"] = 100
    latent["niche_breadth"] = 1.0
    latent["background_cells"] = 10
    latent["unsupported_mass"] = 0.0
    latent["landscape_hash"] = "landscape-hash"
    latent["evaluation_hash"] = "c" * 64
    latent["feature_basis"] = "linear"
    latent["model_regularization"] = 2.0
    latent["max_cell_mass"] = 0.01
    latent["effective_cell_count"] = 100.0
    latent["log_intensity_range"] = 4.0
    latent["lower_clip_cells"] = 0
    latent["lower_clip_fraction"] = 0.0
    latent["solver_converged"] = True
    latent.to_parquet(output / "latent_mixture_metrics.parquet", index=False)

    flexible_keys = expected_flexible_keys(config, community_indices=(0,))
    pilot = flexible_keys.copy()
    pilot["model_regularization"] = 10.0
    pilot["background_cells"] = 10
    pilot["landscape_hash"] = "landscape-hash"
    pilot["feature_basis"] = "clamped_linear_quadratic_interactions"
    pilot["fit_succeeded"] = True
    pilot["max_cell_mass"] = 0.01
    pilot["effective_cell_count"] = 100.0
    pilot["log_intensity_range"] = 4.0
    pilot["lower_clip_cells"] = 0
    pilot["lower_clip_fraction"] = 0.0
    pilot["solver_converged"] = True
    pilot["failure_type"] = None
    pilot["failure_message"] = None
    pilot.to_parquet(output / "flexible_pilot.parquet", index=False)
    (output / "flexible_gate.json").write_text(
        json.dumps(
            {
                "include": True,
                "regularization": 10.0,
                "reason": "smallest fully stable candidate",
                "regularizations": [10.0],
                "tested_regularizations": [10.0],
                "expected_rows_per_candidate": len(flexible_keys),
                "landscape_hash": "landscape-hash",
                "feature_basis": "clamped_linear_quadratic_interactions",
            }
        ),
        encoding="utf-8",
    )
    flexible = flexible_keys.copy()
    for metric in PRIMARY_METRICS:
        flexible[metric] = 0.5
    flexible["record_count"] = 100
    flexible["niche_breadth"] = 1.0
    flexible["background_cells"] = 10
    flexible["source_distribution_distance"] = 0.4
    flexible["unsupported_mass"] = np.where(
        flexible.background_arm.eq("pm_tgb"), 0.1, 0.0
    )
    flexible["landscape_hash"] = "landscape-hash"
    flexible["evaluation_hash"] = "d" * 64
    flexible["feature_basis"] = "clamped_linear_quadratic_interactions"
    flexible["model_regularization"] = 10.0
    flexible["max_cell_mass"] = 0.01
    flexible["effective_cell_count"] = 100.0
    flexible["log_intensity_range"] = 4.0
    flexible["lower_clip_cells"] = 0
    flexible["lower_clip_fraction"] = 0.0
    flexible["solver_converged"] = True
    flexible.to_parquet(
        output / "flexible_sensitivity_metrics.parquet",
        index=False,
    )
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
    assert audit["deepmaxent"]["status"] == "gate_passed_no_complete_run"
    assert "flexible_sensitivity_metrics.parquet" in audit["artifact_hashes"]
    json.dumps(audit)


def test_manuscript_export_writes_six_auditable_tables(
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

    assert len(paths) == 6
    assert {path.name for path in paths} >= {
        "table_5_mechanism.csv",
        "table_6_flexible_sensitivity.csv",
    }
    assert all(path.is_file() for path in paths)
    for path in paths:
        table = pd.read_csv(path)
        assert {"status", "sample_count", "units", "method"} <= set(table)
    manifest = pd.read_csv(
        tmp_path
        / "submission"
        / "tables"
        / "table_4_reproducibility_manifest.csv"
    )
    assert "flexible_sensitivity_metrics.parquet" in set(manifest.artifact)


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


def test_reproducibility_audit_rejects_tampered_mechanism_artifact(
    tmp_path: Path,
    study_config,
) -> None:
    config = complete_tiny_run(tmp_path, study_config)
    path = tmp_path / "outputs" / "mechanism_diagnostics.parquet"
    rows = pd.read_parquet(path).iloc[:-1]
    rows.to_parquet(path, index=False)

    audit = build_reproducibility_audit(tmp_path, config)

    assert audit["core_status"] == "failed"
    assert audit["checks"]["mechanism_complete"] is False


def test_excluded_taxon_scan_includes_extension_species_labels(
    tmp_path: Path,
    study_config,
) -> None:
    config = complete_tiny_run(tmp_path, study_config)
    path = tmp_path / "outputs" / "mechanism_diagnostics.parquet"
    rows = pd.read_parquet(path)
    rows.loc[0, "species_id"] = "bat_species"
    rows.to_parquet(path, index=False)

    audit = build_reproducibility_audit(tmp_path, config)

    assert audit["checks"]["excluded_taxa_absent"] is False
    assert audit["excluded_taxon_scan"]["matches"]
