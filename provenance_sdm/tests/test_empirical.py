from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.empirical import clean_occurrences


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
