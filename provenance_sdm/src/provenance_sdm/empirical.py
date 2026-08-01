"""Audited empirical occurrence preparation."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from provenance_sdm.config import StudyConfig
from provenance_sdm.landscape import Landscape
from provenance_sdm.maxent import PRIMARY_REGULARIZATION, fit_maxent
from provenance_sdm.metrics import continuous_boyce, top_quantile_overlap
from provenance_sdm.provenance import pm_tgb_weights, source_distribution_distance
from provenance_sdm.simulation_runner import seed_for
from provenance_sdm.spatial import (
    projected_block_folds,
    spatial_assignment_frames,
)


REQUIRED_COLUMNS = {
    "taxonKey",
    "scientificName",
    "countryCode",
    "occurrenceStatus",
    "decimalLongitude",
    "decimalLatitude",
    "year",
    "eventDate",
    "datasetKey",
    "publishingOrgKey",
    "hasGeospatialIssues",
    "cell_id",
}
EXCLUDED_TOKENS = ("bat", "pipistrell")


@dataclass(frozen=True)
class CleanedOccurrences:
    records: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class EmpiricalInputs:
    config: StudyConfig
    records: pd.DataFrame
    landscape: Landscape
    taxon_keys: Mapping[str, int]
    block_widths: tuple[int, ...] = (25_000, 50_000, 100_000)
    n_folds: int = 5


def _verified_simple_csv_request(
    request: Mapping[str, object] | None,
    archive_path: Path,
) -> bool:
    if request is None:
        return False
    predicate = request.get("predicate")
    if not isinstance(predicate, Mapping):
        return False
    predicates = predicate.get("predicates")
    if not isinstance(predicates, list):
        return False
    geospatial_filter = any(
        isinstance(item, Mapping)
        and item.get("type") == "equals"
        and item.get("key") == "HAS_GEOSPATIAL_ISSUE"
        and str(item.get("value")).casefold() == "false"
        for item in predicates
    )
    return (
        request.get("format") == "SIMPLE_CSV"
        and request.get("download_key") == archive_path.stem
        and geospatial_filter
    )


def read_gbif_archive(
    path: Path,
    request: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Read the occurrence table from a verified GBIF SIMPLE_CSV archive."""

    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            occurrence_members = [
                member
                for member in archive.namelist()
                if Path(member).name.casefold() == "occurrence.txt"
            ]
            simple_csv_members = [
                member
                for member in archive.namelist()
                if Path(member).suffix.casefold() == ".csv"
            ]
            if len(occurrence_members) == 1 and not simple_csv_members:
                member = occurrence_members[0]
                server_filtered = False
            elif len(simple_csv_members) == 1 and not occurrence_members:
                if not _verified_simple_csv_request(request, archive_path):
                    raise ValueError(
                        "GBIF SIMPLE_CSV requires a verified geospatial predicate"
                    )
                member = simple_csv_members[0]
                server_filtered = True
            else:
                raise ValueError(
                    "GBIF archive must contain exactly one occurrence.txt or CSV member"
                )
            with archive.open(member) as stream:
                records = pd.read_csv(stream, sep="\t", low_memory=False)
    except zipfile.BadZipFile as exc:
        raise ValueError("GBIF archive is not a valid ZIP file") from exc
    if records.empty:
        raise ValueError("GBIF occurrence table is empty")
    if server_filtered and "hasGeospatialIssues" not in records:
        records["hasGeospatialIssues"] = False
    return records


def attach_nearest_grid_cells(
    records: pd.DataFrame,
    landscape: Landscape,
    max_distance_m: float,
) -> pd.DataFrame:
    """Project WGS84 occurrences and attach the nearest valid grid cell."""

    missing = {"decimalLongitude", "decimalLatitude"}.difference(records.columns)
    if missing:
        raise ValueError(f"occurrence coordinates are missing: {sorted(missing)}")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("max_distance_m must be finite and positive")
    coordinates = records.loc[
        :, ["decimalLongitude", "decimalLatitude"]
    ].to_numpy(dtype=float)
    output = records.copy()
    output["x"] = np.nan
    output["y"] = np.nan
    output["cell_distance_m"] = np.nan
    output["cell_id"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    finite = np.isfinite(coordinates).all(axis=1)
    if not finite.any():
        return output

    transformer = Transformer.from_crs(
        "EPSG:4326",
        landscape.crs,
        always_xy=True,
    )
    projected_x, projected_y = transformer.transform(
        coordinates[finite, 0],
        coordinates[finite, 1],
    )
    projected = np.column_stack([projected_x, projected_y])
    tree = cKDTree(landscape.cells.loc[:, ["x", "y"]].to_numpy(dtype=float))
    distance, position = tree.query(projected, k=1)
    finite_index = output.index[finite]
    output.loc[finite_index, "x"] = projected[:, 0]
    output.loc[finite_index, "y"] = projected[:, 1]
    output.loc[finite_index, "cell_distance_m"] = distance
    accepted = distance <= max_distance_m
    accepted_index = finite_index[accepted]
    cell_ids = landscape.cells.iloc[position[accepted]].cell_id.to_numpy(dtype=np.int64)
    output.loc[accepted_index, "cell_id"] = cell_ids
    return output


def clean_occurrences(
    records: pd.DataFrame,
    valid_cell_ids: Collection[int],
    allowed_taxa: Collection[int],
) -> CleanedOccurrences:
    """Apply the frozen occurrence filters and return a staged count audit."""

    missing = REQUIRED_COLUMNS.difference(records.columns)
    if missing:
        raise ValueError(f"occurrence data are missing columns: {sorted(missing)}")
    scientific_names = records.scientificName.fillna("").astype(str).str.casefold()
    if scientific_names.str.contains("|".join(EXCLUDED_TOKENS), regex=True).any():
        raise ValueError("occurrence data contain an excluded taxon")

    current = records.copy()
    audit_rows: list[dict[str, object]] = []

    def record_stage(stage: str, before: int) -> None:
        audit_rows.append(
            {
                "stage": stage,
                "records": len(current),
                "removed": before - len(current),
            }
        )

    audit_rows.append({"stage": "input", "records": len(current), "removed": 0})

    before = len(current)
    current = current[current.taxonKey.isin(set(allowed_taxa))].copy()
    record_stage("allowed_taxa", before)

    before = len(current)
    current = current[
        current.countryCode.eq("GB")
        & current.occurrenceStatus.eq("PRESENT")
        & current.year.between(2022, 2025, inclusive="both")
    ].copy()
    record_stage("gb_year_status", before)

    before = len(current)
    coordinates = current.loc[
        :,
        ["decimalLongitude", "decimalLatitude"],
    ].to_numpy(dtype=float)
    current = current[np.isfinite(coordinates).all(axis=1)].copy()
    record_stage("finite_coordinates", before)

    before = len(current)
    issue = current.hasGeospatialIssues
    no_issue = issue.isna() | issue.eq(False) | issue.astype(str).str.casefold().eq(
        "false"
    )
    current = current[no_issue].copy()
    record_stage("no_geospatial_issue", before)

    before = len(current)
    current = current.drop_duplicates(
        subset=["taxonKey", "cell_id", "eventDate", "datasetKey"],
        keep="first",
    ).copy()
    record_stage("deduplicated", before)

    before = len(current)
    current = current[current.cell_id.isin(set(valid_cell_ids))].copy()
    record_stage("valid_predictor_cell", before)

    before = len(current)
    current = current[
        current.datasetKey.notna() & current.publishingOrgKey.notna()
    ].copy()
    record_stage("complete_provenance", before)

    if current.empty:
        raise ValueError("occurrence cleaning removed every allowed record")
    return CleanedOccurrences(
        records=current.reset_index(drop=True),
        audit=pd.DataFrame(audit_rows),
    )


def _sample_cells(
    weights: pd.Series,
    n_cells: int,
    seed: int,
) -> np.ndarray:
    cell_weights = weights.groupby(level=0).sum().astype(float)
    cell_weights = cell_weights[cell_weights > 0]
    if len(cell_weights) < n_cells:
        raise ValueError("background support is smaller than the paired budget")
    probability = cell_weights.to_numpy(dtype=float, copy=True)
    probability /= probability.sum()
    return np.random.default_rng(seed).choice(
        cell_weights.index.to_numpy(dtype=np.int64),
        size=n_cells,
        replace=False,
        p=probability,
    )


def _evaluation_hash(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(
        frame.loc[:, ["cell_id", "label"]].reset_index(drop=True),
        index=True,
    ).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _centroid_shift(
    prediction: np.ndarray,
    baseline: np.ndarray,
    landscape: Landscape,
) -> float:
    cells = landscape.cells
    area = cells.area_weight.to_numpy(dtype=float)

    def centroid(surface: np.ndarray) -> np.ndarray:
        mask = surface >= np.quantile(surface, 0.90)
        weights = area[mask] * surface[mask]
        if weights.sum() <= 0:
            weights = area[mask]
        return np.average(
            cells.loc[mask, ["x", "y"]].to_numpy(dtype=float),
            axis=0,
            weights=weights,
        )

    return float(np.linalg.norm(centroid(prediction) - centroid(baseline)))


def _backgrounds_for_fold(
    inputs: EmpiricalInputs,
    focal: pd.DataFrame,
    candidates: pd.DataFrame,
    train_cell_ids: set[int],
    source_column: str,
    semantic_key: tuple[object, ...],
) -> tuple[dict[str, pd.DataFrame], float, float, int]:
    landscape = inputs.landscape.cells
    focal_train = focal[focal.cell_id.isin(train_cell_ids)]
    candidate_train = candidates[candidates.cell_id.isin(train_cell_ids)].copy()
    if focal_train.empty or candidate_train.empty:
        raise ValueError("training fold lacks focal or target-group records")

    candidate_sources = candidate_train.set_index("record_id")[source_column]
    provenance = pm_tgb_weights(
        focal_train[source_column],
        candidate_sources,
    )
    uniform_weights = pd.Series(
        landscape.loc[
            landscape.cell_id.isin(train_cell_ids),
            "area_weight",
        ].to_numpy(dtype=float),
        index=landscape.loc[
            landscape.cell_id.isin(train_cell_ids),
            "cell_id",
        ].to_numpy(dtype=np.int64),
    )
    conventional_weights = pd.Series(
        1.0,
        index=candidate_train.cell_id.to_numpy(dtype=np.int64),
    )
    pm_weights = provenance.weights.groupby(
        candidate_train.set_index("record_id").cell_id
    ).sum()
    requested = inputs.config.simulation.background_cells
    budget = min(
        requested,
        int(uniform_weights.groupby(level=0).sum().gt(0).sum()),
        int(conventional_weights.groupby(level=0).sum().gt(0).sum()),
        int(pm_weights.gt(0).sum()),
    )
    minimum = inputs.config.simulation.minimum_background_cells
    if budget < minimum:
        raise ValueError(
            f"paired background budget is {budget}, below required minimum {minimum}"
        )

    weights_by_arm = {
        "uniform": uniform_weights,
        "conventional_tgb": conventional_weights,
        "pm_tgb": pm_weights,
    }
    backgrounds = {}
    for arm, weights in weights_by_arm.items():
        selected = _sample_cells(
            weights,
            budget,
            seed_for(inputs.config.simulation.seed, *semantic_key, arm),
        )
        backgrounds[arm] = (
            landscape.set_index("cell_id").loc[selected].reset_index()
        )
    distance = source_distribution_distance(
        focal_train[source_column],
        candidate_train[source_column],
    )
    return backgrounds, distance, provenance.unsupported_mass, budget


def run_empirical(inputs: EmpiricalInputs, output_dir: Path) -> Path:
    """Run paired spatial-fold empirical comparisons and export map summaries."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    landscape = inputs.landscape
    landscape_cells = landscape.cells
    feature_names = landscape.feature_names
    result_rows: list[dict[str, object]] = []
    map_sums: dict[tuple[str, str], np.ndarray] = {}
    map_counts: dict[tuple[str, str], int] = {}
    assignment_tables: list[pd.DataFrame] = []
    block_audit_tables: list[pd.DataFrame] = []

    for species in inputs.config.empirical_species:
        try:
            focal_key = int(inputs.taxon_keys[species.scientific_name])
            target_keys = {
                int(inputs.taxon_keys[name])
                for name in species.target_group
            }
        except KeyError as exc:
            raise ValueError(f"taxon key is missing for {exc.args[0]!r}") from exc
        focal = inputs.records[inputs.records.taxonKey.eq(focal_key)].copy()
        candidates = inputs.records[inputs.records.taxonKey.isin(target_keys)].copy()
        if focal.empty or candidates.empty:
            raise ValueError(f"{species.key} lacks focal or target-group records")
        required = {
            "record_id",
            "taxonKey",
            "cell_id",
            "datasetKey",
            "publishingOrgKey",
        }
        missing = required.difference(inputs.records.columns)
        if missing:
            raise ValueError(f"empirical records are missing columns: {sorted(missing)}")

        presence_cells = set(focal.cell_id.astype(int))
        evaluation_grid = landscape_cells.loc[:, ["cell_id", "x", "y"]].copy()
        evaluation_grid["label"] = evaluation_grid.cell_id.isin(
            presence_cells
        ).astype(np.int8)

        for width in inputs.block_widths:
            folds = projected_block_folds(
                evaluation_grid,
                width,
                inputs.n_folds,
                seed_for(
                    inputs.config.simulation.seed,
                    "empirical-folds",
                    species.key,
                    width,
                ),
            )
            assignments, block_audit = spatial_assignment_frames(
                evaluation_grid,
                width,
                folds,
            )
            assignments.insert(0, "species", species.key)
            assignments.insert(1, "block_width_m", width)
            assignments["cell_id"] = evaluation_grid.cell_id.to_numpy(
                dtype=np.int64
            )
            block_audit.insert(0, "species", species.key)
            block_audit.insert(1, "block_width_m", width)
            assignment_tables.append(assignments)
            block_audit_tables.append(block_audit)
            for fold in folds:
                train_cells = set(
                    evaluation_grid.loc[list(fold.train_row_indices), "cell_id"].astype(int)
                )
                test_cells = set(
                    evaluation_grid.loc[list(fold.test_row_indices), "cell_id"].astype(int)
                )
                train_presence = focal[focal.cell_id.isin(train_cells)].merge(
                    landscape_cells,
                    on="cell_id",
                    how="inner",
                )
                held_out_presence = focal[focal.cell_id.isin(test_cells)].loc[
                    :, ["cell_id"]
                ]
                test_background = landscape_cells[
                    landscape_cells.cell_id.isin(test_cells)
                    & ~landscape_cells.cell_id.isin(presence_cells)
                ].loc[:, ["cell_id"]]
                evaluation = pd.concat(
                    [
                        held_out_presence.assign(label=np.int8(1)),
                        test_background.assign(label=np.int8(0)),
                    ],
                    ignore_index=True,
                )
                if set(evaluation.label) != {0, 1}:
                    raise ValueError("an empirical fold lacks both evaluation classes")
                evaluation_hash = _evaluation_hash(evaluation)
                position_by_cell = pd.Series(
                    np.arange(len(landscape_cells), dtype=int),
                    index=landscape_cells.cell_id,
                )
                evaluation_scores_index = evaluation.cell_id.map(
                    position_by_cell
                ).to_numpy(dtype=int)

                for provenance_level, source_column in (
                    ("dataset", "datasetKey"),
                    ("publisher", "publishingOrgKey"),
                ):
                    backgrounds, distance, unsupported, budget = _backgrounds_for_fold(
                        inputs,
                        focal,
                        candidates,
                        train_cells,
                        source_column,
                        (
                            "empirical-background",
                            species.key,
                            width,
                            fold.fold_id,
                            provenance_level,
                        ),
                    )
                    prediction_results = {}
                    for arm, background in backgrounds.items():
                        model = fit_maxent(
                            train_presence,
                            background,
                            feature_names,
                            regularization=PRIMARY_REGULARIZATION,
                            seed=seed_for(
                                inputs.config.simulation.seed,
                                "empirical-model",
                                species.key,
                                width,
                                fold.fold_id,
                                provenance_level,
                                arm,
                            ),
                        )
                        prediction_results[arm] = (
                            model.predict_with_diagnostics(landscape)
                        )

                    baseline = prediction_results[
                        "conventional_tgb"
                    ].suitability
                    held_out_ids = held_out_presence.cell_id.map(
                        position_by_cell
                    ).to_numpy(dtype=int)
                    test_ids = landscape_cells.loc[
                        landscape_cells.cell_id.isin(test_cells),
                        "cell_id",
                    ].map(position_by_cell).to_numpy(dtype=int)
                    for arm, prediction_result in prediction_results.items():
                        prediction = prediction_result.suitability
                        evaluation_score = prediction[evaluation_scores_index]
                        boyce = continuous_boyce(
                            prediction[held_out_ids],
                            prediction[test_ids],
                        )
                        correlation = float(
                            scipy.stats.spearmanr(prediction, baseline).statistic
                        )
                        area = landscape_cells.area_weight.to_numpy(dtype=float)
                        predicted_upper = prediction >= np.quantile(prediction, 0.90)
                        baseline_upper = baseline >= np.quantile(baseline, 0.90)
                        result_rows.append(
                            {
                                "species": species.key,
                                "scientific_name": species.scientific_name,
                                "block_width_m": width,
                                "fold_id": fold.fold_id,
                                "provenance_level": provenance_level,
                                "background_arm": arm,
                                "background_cells": budget,
                                "evaluation_hash": evaluation_hash,
                                "auc": float(
                                    roc_auc_score(evaluation.label, evaluation_score)
                                ),
                                "boyce": boyce.value,
                                "boyce_defined": boyce.defined,
                                "boyce_reason": boyce.reason,
                                "map_spearman": correlation,
                                "upper_area_overlap": top_quantile_overlap(
                                    prediction,
                                    baseline,
                                    area,
                                ),
                                "upper_area_shift": float(
                                    area[predicted_upper].sum()
                                    - area[baseline_upper].sum()
                                ),
                                "centroid_shift_m": _centroid_shift(
                                    prediction,
                                    baseline,
                                    landscape,
                                ),
                                "source_distance": distance,
                                "unsupported_mass": (
                                    unsupported if arm == "pm_tgb" else 0.0
                                ),
                                "feature_basis": (
                                    prediction_result.feature_basis
                                ),
                                "model_regularization": PRIMARY_REGULARIZATION,
                                "max_cell_mass": (
                                    prediction_result.max_cell_mass
                                ),
                                "effective_cell_count": (
                                    prediction_result.effective_cell_count
                                ),
                                "log_intensity_range": (
                                    prediction_result.log_intensity_range
                                ),
                                "lower_clip_cells": (
                                    prediction_result.lower_clip_cells
                                ),
                                "lower_clip_fraction": (
                                    prediction_result.lower_clip_fraction
                                ),
                                "solver_converged": (
                                    prediction_result.solver_converged
                                ),
                            }
                        )
                        if width == 50_000 and provenance_level == "dataset":
                            key = (species.key, arm)
                            map_sums.setdefault(key, np.zeros(len(prediction), dtype=float))
                            map_sums[key] += prediction
                            map_counts[key] = map_counts.get(key, 0) + 1

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise ValueError("empirical study produced no model results")
    result_path = destination / "empirical_metrics.parquet"
    results.to_parquet(result_path, index=False)
    maps = []
    for (species_key, arm), total in map_sums.items():
        frame = landscape_cells.loc[:, ["cell_id", "x", "y"]].copy()
        frame["species"] = species_key
        frame["background_arm"] = arm
        frame["predicted_suitability"] = total / map_counts[(species_key, arm)]
        maps.append(frame)
    pd.concat(maps, ignore_index=True).to_parquet(
        destination / "empirical_maps.parquet",
        index=False,
    )
    pd.concat(assignment_tables, ignore_index=True).to_parquet(
        destination / "spatial_fold_assignments.parquet",
        index=False,
    )
    pd.concat(block_audit_tables, ignore_index=True).to_csv(
        destination / "spatial_block_class_audit.csv",
        index=False,
    )
    return result_path
