from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.simulation_runner import (
    RESULT_KEY_COLUMNS,
    audit_simulation,
    expected_simulation_keys,
    run_simulation,
    seed_for,
)


def test_full_design_has_14400_unique_fit_keys(study_config) -> None:
    keys = expected_simulation_keys(study_config)

    assert len(keys) == 14_400
    assert not keys.duplicated().any()
    assert tuple(keys.columns) == RESULT_KEY_COLUMNS


def test_child_seed_depends_on_key_not_scheduling_order() -> None:
    key = ("community", 2, "low", "strong", "sp_019", "pm_tgb")

    assert seed_for(20260730, *key) == seed_for(20260730, *key)
    assert seed_for(20260730, *key) != seed_for(20260730, *reversed(key))


@pytest.fixture
def tiny_config(study_config, tmp_path):
    simulation = replace(
        study_config.simulation,
        n_species=20,
        n_communities=1,
        alignments=("low",),
        bias_levels=("moderate",),
        min_records=100,
        max_records=100,
        background_cells=10,
        minimum_background_cells=5,
    )
    return replace(study_config, simulation=simulation, output_dir=tmp_path)


@pytest.fixture
def runner_landscape():
    side = np.linspace(-2.5, 2.5, 18)
    x, y = np.meshgrid(side * 10_000, side * 10_000)
    return landscape_from_arrays(
        {
            "env_1": x.ravel(),
            "env_2": y.ravel(),
            "env_3": np.sin(x.ravel() / 10_000) + np.cos(y.ravel() / 10_000),
        },
        x=x.ravel(),
        y=y.ravel(),
        area=np.ones(x.size),
        crs="EPSG:27700",
    )


def test_runner_resumes_without_duplicate_rows(
    tiny_config,
    runner_landscape,
    tmp_path: Path,
) -> None:
    path = run_simulation(tiny_config, runner_landscape, tmp_path)
    first = pd.read_parquet(path)

    resumed_path = run_simulation(tiny_config, runner_landscape, tmp_path)
    second = pd.read_parquet(resumed_path)

    assert path == resumed_path
    pd.testing.assert_frame_equal(first, second)
    assert len(second) == 80
    assert not second.duplicated(list(RESULT_KEY_COLUMNS)).any()
    assert second.landscape_hash.nunique() == 1
    assert second.source_distribution_distance.between(0, 1).all()


def test_audit_reports_exact_missing_fit_keys(
    tiny_config,
    runner_landscape,
    tmp_path: Path,
) -> None:
    path = run_simulation(tiny_config, runner_landscape, tmp_path)
    rows = pd.read_parquet(path).iloc[:-1]
    rows.to_parquet(path, index=False)

    audit = audit_simulation(path, tiny_config)

    assert audit["status"] == "failed"
    assert audit["expected"] == 80
    assert audit["completed"] == 79
    assert audit["missing"] == 1
    assert audit["failed"] == 0
    assert len(audit["missing_keys"]) == 1


def test_audit_rejects_unexpected_fit_keys(
    tiny_config,
    runner_landscape,
    tmp_path: Path,
) -> None:
    path = run_simulation(tiny_config, runner_landscape, tmp_path)
    rows = pd.read_parquet(path)
    unexpected = rows.iloc[[0]].copy()
    unexpected["species_id"] = "sp_999"
    pd.concat([rows, unexpected], ignore_index=True).to_parquet(path, index=False)

    audit = audit_simulation(path, tiny_config)

    assert audit["status"] == "failed"
    assert audit["unexpected"] == 1
    assert audit["unexpected_keys"][0]["species_id"] == "sp_999"
