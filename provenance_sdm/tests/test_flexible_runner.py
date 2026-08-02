from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.flexible_runner import (
    audit_flexible_sensitivity,
    expected_flexible_keys,
    run_flexible_pilot,
    run_flexible_sensitivity,
    select_flexible_regularization,
)
from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.simulation_runner import PRIMARY_METRICS


def fake_pilot_rows(stability: dict[float, bool]) -> pd.DataFrame:
    rows = []
    for regularization, stable in stability.items():
        for fit_id in range(2):
            rows.append(
                {
                    "model_regularization": regularization,
                    "fit_id": fit_id,
                    "fit_succeeded": True,
                    "max_cell_mass": 0.01 if stable else 0.50,
                    "effective_cell_count": 100.0 if stable else 4.0,
                    "log_intensity_range": 5.0,
                    "lower_clip_cells": 0,
                    "lower_clip_fraction": 0.0,
                    "solver_converged": stable,
                    "suitability_spearman": -999.0 if stable else 999.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def flexible_config(study_config, tmp_path: Path):
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
def flexible_landscape():
    side = np.linspace(-2.5, 2.5, 18)
    x, y = np.meshgrid(side * 10_000, side * 10_000)
    return landscape_from_arrays(
        {
            "env_1": x.ravel(),
            "env_2": y.ravel(),
            "env_3": np.sin(x.ravel() / 10_000)
            + np.cos(y.ravel() / 10_000),
        },
        x=x.ravel(),
        y=y.ravel(),
        area=np.ones(x.size),
        crs="EPSG:27700",
    )


def test_flexible_design_has_1800_exact_subset_keys(study_config) -> None:
    keys = expected_flexible_keys(study_config)

    assert len(keys) == 1_800
    assert not keys.duplicated().any()
    assert set(keys.species_id) == {
        f"sp_{index:03d}" for index in range(0, 200, 4)
    }
    assert set(keys.background_arm) == {"conventional_tgb", "pm_tgb"}


def test_gate_selects_smallest_fully_stable_regularization() -> None:
    rows = fake_pilot_rows({2.0: False, 5.0: True, 10.0: True})

    gate = select_flexible_regularization(rows, (2.0, 5.0, 10.0))

    assert gate["include"] is True
    assert gate["regularization"] == 5.0
    assert gate["reason"] == "smallest fully stable candidate"


def test_gate_selection_does_not_depend_on_truth_metrics() -> None:
    rows = fake_pilot_rows({2.0: True})
    without_metrics = rows.drop(columns=list(PRIMARY_METRICS), errors="ignore")

    assert select_flexible_regularization(rows, (2.0,)) == (
        select_flexible_regularization(without_metrics, (2.0,))
    )


def test_gate_excludes_when_every_candidate_is_unstable() -> None:
    gate = select_flexible_regularization(
        fake_pilot_rows({2.0: False, 5.0: False}),
        (2.0, 5.0),
    )

    assert gate["include"] is False
    assert gate["regularization"] is None
    assert gate["reason"] == "no candidate passed every stability check"


def test_pilot_full_run_and_audit_are_restartable(
    flexible_config,
    flexible_landscape,
    tmp_path: Path,
) -> None:
    pilot_path, gate_path = run_flexible_pilot(
        flexible_config,
        flexible_landscape,
        tmp_path,
        regularizations=(10.0,),
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["include"] is True
    assert gate["regularization"] == 10.0
    pilot = pd.read_parquet(pilot_path)
    assert len(pilot) == 10
    assert not set(PRIMARY_METRICS).intersection(pilot.columns)

    result_path = run_flexible_sensitivity(
        flexible_config,
        flexible_landscape,
        gate_path,
        tmp_path,
    )
    first = pd.read_parquet(result_path)
    resumed = pd.read_parquet(
        run_flexible_sensitivity(
            flexible_config,
            flexible_landscape,
            gate_path,
            tmp_path,
        )
    )

    pd.testing.assert_frame_equal(first, resumed)
    assert len(first) == 10
    assert first.evaluation_hash.str.len().eq(64).all()
    audit = audit_flexible_sensitivity(tmp_path, flexible_config)
    assert audit["status"] == "passed"
    assert audit["full_run_included"] is True
    json.dumps(audit)


def test_full_runner_rejects_excluded_gate(
    flexible_config,
    flexible_landscape,
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "flexible_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "include": False,
                "regularization": None,
                "reason": "no candidate passed every stability check",
                "regularizations": [2.0, 5.0, 10.0],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="excluded"):
        run_flexible_sensitivity(
            flexible_config,
            flexible_landscape,
            gate_path,
            tmp_path,
        )
