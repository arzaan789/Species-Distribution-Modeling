from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from provenance_sdm.config import EmpiricalSpecies
from provenance_sdm.empirical import (
    EmpiricalInputs,
    attach_nearest_grid_cells,
    clean_occurrences,
    read_gbif_archive,
    run_empirical,
)
from provenance_sdm.landscape import landscape_from_arrays


@pytest.fixture
def raw_occurrences() -> pd.DataFrame:
    rows = []
    for index in range(8):
        rows.append(
            {
                "gbifID": str(index),
                "taxonKey": 10 if index < 4 else 20,
                "scientificName": "Focal species" if index < 4 else "Target species",
                "countryCode": "GB",
                "occurrenceStatus": "PRESENT",
                "decimalLongitude": -2.0 + index * 0.01,
                "decimalLatitude": 52.0 + index * 0.01,
                "year": 2023,
                "eventDate": f"2023-06-{index + 1:02d}",
                "datasetKey": "dataset-a" if index % 2 else "dataset-b",
                "publishingOrgKey": "publisher-a",
                "hasGeospatialIssues": False,
                "cell_id": index,
            }
        )
    return pd.DataFrame(rows)


def test_cleaning_retains_provenance_and_audits_each_stage(
    raw_occurrences,
) -> None:
    raw = pd.concat([raw_occurrences, raw_occurrences.iloc[[0]]], ignore_index=True)
    raw.loc[8, "gbifID"] = "duplicate"
    raw.loc[8, "decimalLongitude"] = raw.loc[0, "decimalLongitude"]

    cleaned = clean_occurrences(raw, set(range(8)), {10, 20})

    assert {"datasetKey", "publishingOrgKey", "cell_id"} <= set(cleaned.records)
    assert cleaned.audit.stage.is_unique
    assert cleaned.audit.iloc[0].records == 9
    assert cleaned.audit.iloc[-1].records == 8
    assert cleaned.audit.set_index("stage").loc["deduplicated", "removed"] == 1


def test_cleaning_rejects_any_bat_before_allowed_taxon_filter(
    raw_occurrences,
) -> None:
    bat = raw_occurrences.iloc[[0]].copy()
    bat["taxonKey"] = 999
    bat["scientificName"] = "Pipistrellus pygmaeus"
    raw = pd.concat([raw_occurrences, bat], ignore_index=True)

    with pytest.raises(ValueError, match="excluded taxon"):
        clean_occurrences(raw, set(range(8)), {10, 20})


def test_cleaning_filters_invalid_rows_in_declared_order(raw_occurrences) -> None:
    raw = raw_occurrences.copy()
    raw.loc[0, "year"] = 2021
    raw.loc[1, "decimalLongitude"] = np.nan
    raw.loc[2, "hasGeospatialIssues"] = True
    raw.loc[3, "cell_id"] = 999
    raw.loc[4, "datasetKey"] = None

    cleaned = clean_occurrences(raw, set(range(8)), {10, 20})
    audit = cleaned.audit.set_index("stage")

    assert audit.loc["gb_year_status", "removed"] == 1
    assert audit.loc["finite_coordinates", "removed"] == 1
    assert audit.loc["no_geospatial_issue", "removed"] == 1
    assert audit.loc["valid_predictor_cell", "removed"] == 1
    assert audit.loc["complete_provenance", "removed"] == 1
    assert len(cleaned.records) == 3


def test_missing_required_schema_is_rejected(raw_occurrences) -> None:
    with pytest.raises(ValueError, match="datasetKey"):
        clean_occurrences(
            raw_occurrences.drop(columns="datasetKey"),
            set(range(8)),
            {10, 20},
        )


def test_gbif_simple_csv_archive_is_read_from_occurrence_member(
    raw_occurrences: pd.DataFrame,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "gbif.zip"
    payload = raw_occurrences.drop(columns="cell_id").to_csv(
        sep="\t",
        index=False,
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "GBIF download")
        archive.writestr("occurrence.txt", payload)

    restored = read_gbif_archive(archive_path)

    assert len(restored) == len(raw_occurrences)
    assert {"taxonKey", "datasetKey", "publishingOrgKey"} <= set(restored)


def test_current_gbif_simple_csv_member_uses_verified_request_filter(
    raw_occurrences: pd.DataFrame,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "0018113-260721160103020.zip"
    payload = raw_occurrences.drop(
        columns=["cell_id", "hasGeospatialIssues"]
    ).to_csv(sep="\t", index=False)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("0018113-260721160103020.csv", payload)
    request = {
        "download_key": "0018113-260721160103020",
        "format": "SIMPLE_CSV",
        "predicate": {
            "type": "and",
            "predicates": [
                {
                    "type": "equals",
                    "key": "HAS_GEOSPATIAL_ISSUE",
                    "value": "false",
                }
            ],
        },
    }

    restored = read_gbif_archive(archive_path, request)

    assert len(restored) == len(raw_occurrences)
    assert restored.hasGeospatialIssues.eq(False).all()
    assert restored.taxonKey.tolist() == raw_occurrences.taxonKey.tolist()


def test_simple_csv_without_verified_geospatial_predicate_is_rejected(
    raw_occurrences: pd.DataFrame,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "gbif.zip"
    payload = raw_occurrences.drop(
        columns=["cell_id", "hasGeospatialIssues"]
    ).to_csv(sep="\t", index=False)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("download.csv", payload)

    with pytest.raises(ValueError, match="verified geospatial predicate"):
        read_gbif_archive(archive_path)


def test_occurrence_coordinates_attach_to_nearest_projected_grid_cell() -> None:
    longitude = np.array((-2.0, -1.9, -2.0, -1.9))
    latitude = np.array((52.0, 52.0, 52.1, 52.1))
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    landscape = landscape_from_arrays(
        {
            "env_1": np.arange(4, dtype=float),
            "env_2": np.array((0.0, 2.0, 1.0, 3.0)),
            "env_3": np.array((3.0, 1.0, 0.0, 2.0)),
        },
        np.asarray(x),
        np.asarray(y),
        np.ones(4),
        "EPSG:27700",
    )
    records = pd.DataFrame(
        {
            "decimalLongitude": longitude[[0, 3]],
            "decimalLatitude": latitude[[0, 3]],
        }
    )

    attached = attach_nearest_grid_cells(records, landscape, max_distance_m=10)

    assert attached.cell_id.tolist() == [0, 3]
    assert attached.cell_distance_m.max() < 0.01
    assert np.isfinite(attached.loc[:, ["x", "y"]].to_numpy()).all()


@pytest.fixture
def tiny_empirical_inputs(study_config, tmp_path) -> EmpiricalInputs:
    side = np.arange(24, dtype=float) * 25_000
    x, y = np.meshgrid(side, side)
    landscape = landscape_from_arrays(
        {
            "env_1": x.ravel(),
            "env_2": y.ravel(),
            "env_3": np.sin(x.ravel() / 100_000)
            + np.cos(y.ravel() / 100_000),
        },
        x=x.ravel(),
        y=y.ravel(),
        area=np.ones(x.size),
        crs="EPSG:27700",
    )
    focal_cells = [
        y_index * 24 + x_index
        for y_index in range(0, 24, 4)
        for x_index in range(0, 24, 4)
    ]
    rows = []
    for record_id, cell_id in enumerate(focal_cells * 3):
        rows.append(
            {
                "record_id": record_id,
                "taxonKey": 10,
                "cell_id": cell_id,
                "datasetKey": "dataset-a" if record_id % 4 else "dataset-b",
                "publishingOrgKey": "publisher-a"
                if record_id % 3
                else "publisher-b",
            }
        )
    next_id = len(rows)
    for offset, cell_id in enumerate(range(0, len(landscape.cells), 2)):
        rows.append(
            {
                "record_id": next_id + offset,
                "taxonKey": 20,
                "cell_id": cell_id,
                "datasetKey": ("dataset-a", "dataset-b", "dataset-c")[offset % 3],
                "publishingOrgKey": ("publisher-a", "publisher-b")[offset % 2],
            }
        )
    simulation = replace(
        study_config.simulation,
        background_cells=10,
        minimum_background_cells=5,
    )
    config = replace(
        study_config,
        simulation=simulation,
        empirical_species=(
            EmpiricalSpecies(
                key="focal",
                scientific_name="Focal species",
                target_group=("Target species",),
            ),
        ),
        output_dir=tmp_path,
    )
    return EmpiricalInputs(
        config=config,
        records=pd.DataFrame(rows),
        landscape=landscape,
        taxon_keys={"Focal species": 10, "Target species": 20},
        block_widths=(50_000,),
        n_folds=5,
    )


def test_empirical_arms_share_evaluation_rows_and_background_budget(
    tiny_empirical_inputs: EmpiricalInputs,
    tmp_path: Path,
) -> None:
    path = run_empirical(tiny_empirical_inputs, tmp_path)
    rows = pd.read_parquet(path)

    assert len(rows) == 30
    hashes = rows.groupby(["species", "block_width_m", "fold_id"]).evaluation_hash.nunique()
    assert hashes.eq(1).all()
    budgets = rows.groupby(
        ["species", "block_width_m", "fold_id", "provenance_level"]
    ).background_cells.nunique()
    assert budgets.eq(1).all()
    assert set(rows.background_arm) == {"uniform", "conventional_tgb", "pm_tgb"}
    assert set(rows.provenance_level) == {"dataset", "publisher"}
    required = {
        "feature_basis",
        "max_cell_mass",
        "effective_cell_count",
        "log_intensity_range",
        "lower_clip_cells",
        "lower_clip_fraction",
        "solver_converged",
    }
    assert required <= set(rows)
    assert rows.feature_basis.eq("linear").all()
    assert rows.solver_converged.all()
    assert rows.model_regularization.eq(2.0).all()

    assignments = pd.read_parquet(
        tmp_path / "spatial_fold_assignments.parquet"
    )
    block_audit = pd.read_csv(tmp_path / "spatial_block_class_audit.csv")
    assert set(assignments.species) == {"focal"}
    assert set(assignments.block_width_m) == {50_000}
    assert assignments.groupby("block_id").fold_id.nunique().eq(1).all()
    fold_counts = block_audit.groupby("fold_id")[
        ["positive_rows", "negative_rows"]
    ].sum()
    assert fold_counts.positive_rows.gt(0).all()
    assert fold_counts.negative_rows.gt(0).all()


def test_empirical_comparison_exports_common_map_metrics(
    tiny_empirical_inputs: EmpiricalInputs,
    tmp_path: Path,
) -> None:
    rows = pd.read_parquet(run_empirical(tiny_empirical_inputs, tmp_path))

    assert rows.map_spearman.between(-1, 1).all()
    assert rows.upper_area_overlap.between(0, 1).all()
    assert np.isfinite(rows.upper_area_shift).all()
    assert np.isfinite(rows.centroid_shift_m).all()
    assert (tmp_path / "empirical_maps.parquet").is_file()
