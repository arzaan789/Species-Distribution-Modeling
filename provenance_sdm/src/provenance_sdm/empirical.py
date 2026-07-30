"""Audited empirical occurrence preparation."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd


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
