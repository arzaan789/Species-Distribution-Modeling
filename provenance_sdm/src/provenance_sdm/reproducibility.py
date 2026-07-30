"""Submission-table exports and fail-closed reproducibility audit."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from provenance_sdm.config import StudyConfig
from provenance_sdm.manifests import sha256_file
from provenance_sdm.simulation_runner import PRIMARY_METRICS, audit_simulation
from provenance_sdm.summaries import hierarchical_bootstrap


EXCLUDED_TAXON_PATTERN = re.compile(r"\bbat\w*\b|pipistrell\w*", re.IGNORECASE)
SUBMISSION_FIGURES = (
    "simulation_workflow.png",
    "paired_truth_contrasts.png",
    "empirical_source_contrasts.png",
    "empirical_map_contrast.png",
)
CORE_OUTPUTS = (
    "simulation_metrics.parquet",
    "empirical_metrics.parquet",
    "occurrence_cleaning_audit.csv",
    "gbif_archive.json",
    "gb_grid.manifest.json",
)
EXPECTED_CLEANING_STAGES = (
    "input",
    "allowed_taxa",
    "gb_year_status",
    "finite_coordinates",
    "no_geospatial_issue",
    "deduplicated",
    "valid_predictor_cell",
    "complete_provenance",
)


def _configuration_hash(config: StudyConfig) -> str:
    return hashlib.sha256(repr(config).encode("utf-8")).hexdigest()


def _required_path(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path.name}")
    return path


def _scan_excluded_taxa(
    empirical: pd.DataFrame,
    gbif_manifest: dict[str, object],
) -> list[dict[str, str]]:
    values = []
    for column in ("species", "scientific_name"):
        if column in empirical:
            values.extend(empirical[column].dropna().astype(str).unique())
    values.append(json.dumps(gbif_manifest, sort_keys=True))
    matches = []
    for value in values:
        for match in EXCLUDED_TAXON_PATTERN.finditer(value):
            matches.append({"value": value, "match": match.group(0)})
    return matches


def build_reproducibility_audit(
    root: Path,
    config: StudyConfig,
) -> dict[str, object]:
    """Validate every core result needed for the submission bundle."""

    study_root = Path(root)
    output = study_root / "outputs"
    paths = {
        name: _required_path(output / name)
        for name in CORE_OUTPUTS
    }
    simulation = pd.read_parquet(paths["simulation_metrics.parquet"])
    empirical = pd.read_parquet(paths["empirical_metrics.parquet"])
    cleaning = pd.read_csv(paths["occurrence_cleaning_audit.csv"])
    gbif = json.loads(paths["gbif_archive.json"].read_text(encoding="utf-8"))
    grid = json.loads(
        paths["gb_grid.manifest.json"].read_text(encoding="utf-8")
    )

    simulation_audit = audit_simulation(
        paths["simulation_metrics.parquet"],
        config,
    )
    budget_groups = simulation.groupby(
        ["community_seed", "alignment", "bias_level", "species_id"]
    ).background_cells.nunique()
    simulation_budgets_paired = bool(budget_groups.eq(1).all())

    expected_species = {
        species.scientific_name
        for species in config.empirical_species
    }
    actual_species = set(empirical.scientific_name.dropna().astype(str))
    empirical_widths = set(empirical.block_width_m.astype(int))
    empirical_finite_columns = (
        "auc",
        "map_spearman",
        "upper_area_overlap",
        "upper_area_shift",
        "centroid_shift_m",
        "source_distance",
        "unsupported_mass",
    )
    empirical_finite = all(
        column in empirical
        and np.isfinite(empirical[column].to_numpy(dtype=float)).all()
        for column in empirical_finite_columns
    )
    evaluation_groups = empirical.groupby(
        ["species", "block_width_m", "fold_id"]
    ).evaluation_hash.nunique()
    empirical_common_evaluation = bool(evaluation_groups.eq(1).all())
    empirical_budget_groups = empirical.groupby(
        ["species", "block_width_m", "fold_id", "provenance_level"]
    ).background_cells.nunique()
    empirical_budgets_paired = bool(empirical_budget_groups.eq(1).all())
    empirical_arms_exact = set(empirical.background_arm) == {
        "uniform",
        "conventional_tgb",
        "pm_tgb",
    }
    empirical_provenance_exact = set(empirical.provenance_level) == {
        "dataset",
        "publisher",
    }
    empirical_fold_counts = empirical.groupby(
        ["species", "block_width_m", "provenance_level", "background_arm"]
    ).fold_id.nunique()
    empirical_folds_complete = bool(
        empirical_fold_counts.eq(5).all()
    )

    gbif_valid = (
        isinstance(gbif.get("download_key"), str)
        and bool(gbif.get("doi"))
        and isinstance(gbif.get("archive_sha256"), str)
        and len(str(gbif["archive_sha256"])) == 64
        and isinstance(gbif.get("total_records"), int)
        and int(gbif["total_records"]) > 0
    )
    timestamp_valid = False
    try:
        retrieved = datetime.fromisoformat(str(gbif["retrieved_at"]))
        timestamp_valid = retrieved.tzinfo is not None
    except (KeyError, TypeError, ValueError):
        pass
    cleaning_valid = (
        {"stage", "records", "removed"} <= set(cleaning)
        and tuple(cleaning.stage) == EXPECTED_CLEANING_STAGES
        and cleaning.stage.is_unique
        and cleaning.records.iloc[-1] > 0
    )
    grid_valid = (
        isinstance(grid.get("source_sha256"), str)
        and len(str(grid["source_sha256"])) == 64
        and int(grid.get("retained_cells", 0)) > 0
    )

    figure_root = study_root / "manuscript" / "figures"
    figure_checks = {
        name: {
            "exists": (figure_root / name).is_file(),
            "bytes": (figure_root / name).stat().st_size
            if (figure_root / name).is_file()
            else 0,
            "png_signature": (
                (figure_root / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
                if (figure_root / name).is_file()
                else False
            ),
        }
        for name in SUBMISSION_FIGURES
    }
    figures_valid = all(
        check["exists"] and check["bytes"] > 1_000 and check["png_signature"]
        for check in figure_checks.values()
    )
    excluded_matches = _scan_excluded_taxa(empirical, gbif)

    gate_path = output / "deepmaxent_gate.json"
    if not gate_path.is_file():
        gate_path = study_root / "manifests" / "deepmaxent_gate.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        deepmaxent = {
            "status": "included" if gate.get("include") is True else "excluded",
            "reasons": gate.get("reasons", []),
            "gate_sha256": sha256_file(gate_path),
        }
    else:
        deepmaxent = {
            "status": "not_evaluated",
            "reasons": ["gate artifact is unavailable"],
        }

    checks = {
        "simulation_complete": simulation_audit["status"] == "passed",
        "simulation_budgets_paired": simulation_budgets_paired,
        "empirical_species_exact": actual_species == expected_species,
        "empirical_widths_exact": empirical_widths
        == {25_000, 50_000, 100_000},
        "empirical_metrics_finite": empirical_finite,
        "empirical_common_evaluation": empirical_common_evaluation,
        "empirical_budgets_paired": empirical_budgets_paired,
        "empirical_arms_exact": empirical_arms_exact,
        "empirical_provenance_exact": empirical_provenance_exact,
        "empirical_folds_complete": empirical_folds_complete,
        "cleaning_audit_valid": cleaning_valid,
        "gbif_manifest_valid": gbif_valid,
        "gbif_timestamp_valid": timestamp_valid,
        "grid_manifest_valid": grid_valid,
        "submission_figures_valid": figures_valid,
        "excluded_taxa_absent": not excluded_matches,
    }
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
    }
    return {
        "core_status": "passed" if all(checks.values()) else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration_hash": _configuration_hash(config),
        "checks": checks,
        "simulation_audit": simulation_audit,
        "expected_empirical_species": sorted(expected_species),
        "actual_empirical_species": sorted(actual_species),
        "excluded_taxon_scan": {"matches": excluded_matches},
        "figures": figure_checks,
        "deepmaxent": deepmaxent,
        "artifact_hashes": artifact_hashes,
    }


def _units_for_metric(metric: str) -> str:
    return {
        "suitability_spearman": "Spearman correlation",
        "integrated_error": "total-variation distance",
        "unbiased_auc": "AUC",
        "response_curve_error": "RMSE",
        "top10_overlap": "area-weighted Jaccard",
    }.get(metric, "metric-specific")


def export_manuscript(
    root: Path,
    config: StudyConfig,
    output_dir: Path,
    *,
    n_boot: int = 2_000,
) -> tuple[Path, ...]:
    """Export the four predeclared submission tables from tidy artifacts."""

    study_root = Path(root)
    source = study_root / "outputs"
    destination = Path(output_dir) / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    simulation_path = _required_path(source / "simulation_metrics.parquet")
    empirical_path = _required_path(source / "empirical_metrics.parquet")
    cleaning_path = _required_path(source / "occurrence_cleaning_audit.csv")
    gbif_path = _required_path(source / "gbif_archive.json")
    grid_path = _required_path(source / "gb_grid.manifest.json")
    simulation = pd.read_parquet(simulation_path)
    empirical = pd.read_parquet(empirical_path)
    configuration_hash = _configuration_hash(config)

    design = pd.DataFrame(
        [
            {
                "design": "virtual-species paired background experiment",
                "species": config.simulation.n_species,
                "communities": config.simulation.n_communities,
                "observation_scenarios": len(config.simulation.alignments)
                * len(config.simulation.bias_levels),
                "background_arms": len(config.background_arms),
                "planned_fits": len(simulation),
                "status": "primary",
                "sample_count": len(simulation),
                "units": "model fits",
                "method": "paired MaxEnt-equivalent simulation",
                "configuration_hash": configuration_hash,
                "input_hash": sha256_file(simulation_path),
            }
        ]
    )
    table_1 = destination / "table_1_simulation_design.csv"
    design.to_csv(table_1, index=False)

    effects = hierarchical_bootstrap(
        simulation,
        n_boot=n_boot,
        seed=config.simulation.seed,
    )
    effects["status"] = "primary"
    effects["sample_count"] = effects.n_pairs
    effects["units"] = effects.metric.map(_units_for_metric)
    effects["method"] = "PM-TGB minus conventional TGB hierarchical bootstrap"
    effects["configuration_hash"] = configuration_hash
    effects["input_hash"] = sha256_file(simulation_path)
    table_2 = destination / "table_2_primary_effects.csv"
    effects.to_csv(table_2, index=False)

    aggregations: dict[str, tuple[str, str]] = {
        "auc_mean": ("auc", "mean"),
        "source_distance_mean": ("source_distance", "mean"),
        "centroid_shift_m_mean": ("centroid_shift_m", "mean"),
        "folds": ("fold_id", "nunique"),
    }
    if "boyce" in empirical:
        aggregations["boyce_mean"] = ("boyce", "mean")
    composition = (
        empirical.groupby(
            [
                "species",
                "scientific_name",
                "block_width_m",
                "provenance_level",
                "background_arm",
            ],
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
    )
    composition["status"] = np.where(
        composition.block_width_m.eq(50_000)
        & composition.provenance_level.eq("dataset"),
        "primary",
        "supplementary",
    )
    composition["sample_count"] = composition.folds
    composition["units"] = "AUC/Boyce; source distance; centroid metres"
    composition["method"] = composition.background_arm
    composition["configuration_hash"] = configuration_hash
    composition["input_hash"] = sha256_file(empirical_path)
    table_3 = destination / "table_3_empirical_composition_metrics.csv"
    composition.to_csv(table_3, index=False)

    manifest_rows = []
    for path, method in (
        (simulation_path, "simulation metrics"),
        (empirical_path, "empirical metrics"),
        (cleaning_path, "occurrence cleaning audit"),
        (gbif_path, "GBIF DOI archive manifest"),
        (grid_path, "projected predictor-grid manifest"),
    ):
        manifest_rows.append(
            {
                "artifact": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "status": "required",
                "sample_count": 1,
                "units": "file",
                "method": method,
                "configuration_hash": configuration_hash,
                "input_hash": sha256_file(path),
            }
        )
    table_4 = destination / "table_4_reproducibility_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(table_4, index=False)
    return table_1, table_2, table_3, table_4
