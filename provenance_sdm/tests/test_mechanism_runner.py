from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.landscape import landscape_from_arrays
from provenance_sdm.mechanism_runner import (
    audit_mechanism,
    expected_mechanism_keys,
    run_mechanism,
)


@pytest.fixture
def mechanism_config(study_config, tmp_path: Path):
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
def mechanism_landscape():
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


def test_expected_mechanism_design_has_3600_unique_pairs(study_config) -> None:
    keys = expected_mechanism_keys(study_config)

    assert len(keys) == 3_600
    assert not keys.duplicated().any()
    assert tuple(keys.columns) == (
        "community_seed",
        "alignment",
        "bias_level",
        "species_id",
    )


def test_mechanism_runner_resumes_two_complete_artifacts(
    mechanism_config,
    mechanism_landscape,
    tmp_path: Path,
) -> None:
    first_paths = run_mechanism(
        mechanism_config,
        mechanism_landscape,
        tmp_path,
    )
    first = tuple(pd.read_parquet(path) for path in first_paths)

    second_paths = run_mechanism(
        mechanism_config,
        mechanism_landscape,
        tmp_path,
    )
    second = tuple(pd.read_parquet(path) for path in second_paths)

    for left, right in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(left, right)
    assert [len(frame) for frame in second] == [20, 20]
    assert second[0].ecological_overlap_tv.between(0, 1).all()
    assert set(second[1].background_arm) == {"latent_mixture_tgb"}
    assert second[1].evaluation_hash.str.len().eq(64).all()
    audit = audit_mechanism(tmp_path, mechanism_config)
    assert audit["status"] == "passed"
    json.dumps(audit)


def test_mechanism_audit_rejects_missing_and_unstable_rows(
    mechanism_config,
    mechanism_landscape,
    tmp_path: Path,
) -> None:
    diagnostics_path, latent_path = run_mechanism(
        mechanism_config,
        mechanism_landscape,
        tmp_path,
    )
    diagnostics = pd.read_parquet(diagnostics_path).iloc[:-1]
    diagnostics.to_parquet(diagnostics_path, index=False)
    latent = pd.read_parquet(latent_path)
    latent.loc[0, "max_cell_mass"] = 0.5
    latent.to_parquet(latent_path, index=False)

    audit = audit_mechanism(tmp_path, mechanism_config)

    assert audit["missing_diagnostics"] == 1
    assert audit["stable_latent_predictions"] is False
    assert audit["status"] == "failed"


def test_mechanism_runner_rejects_stale_latent_checkpoint(
    mechanism_config,
    mechanism_landscape,
    tmp_path: Path,
) -> None:
    _, latent_path = run_mechanism(
        mechanism_config,
        mechanism_landscape,
        tmp_path,
    )
    latent = pd.read_parquet(latent_path)
    latent["model_regularization"] = 1.0
    latent.to_parquet(latent_path, index=False)

    with pytest.raises(ValueError, match="regularization"):
        run_mechanism(mechanism_config, mechanism_landscape, tmp_path)


def test_successful_resume_removes_historical_failure_file(
    mechanism_config,
    mechanism_landscape,
    tmp_path: Path,
) -> None:
    run_mechanism(mechanism_config, mechanism_landscape, tmp_path)
    failure_path = tmp_path / "mechanism_failures.csv"
    pd.DataFrame(
        [{"species_id": "sp_000", "message": "historical failure"}]
    ).to_csv(failure_path, index=False)

    run_mechanism(mechanism_config, mechanism_landscape, tmp_path)

    assert not failure_path.exists()
    assert audit_mechanism(tmp_path, mechanism_config)["status"] == "passed"
