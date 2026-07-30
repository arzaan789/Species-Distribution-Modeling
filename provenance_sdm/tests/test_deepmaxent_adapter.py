from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.cli import main
from provenance_sdm.deepmaxent_adapter import (
    DeepMaxentAdapter,
    GateReport,
    evaluate_deepmaxent_gate,
    normalized_poisson_reference_loss,
    run_deepmaxent_pilot,
)


def test_reference_loss_matches_manual_normalized_poisson() -> None:
    counts = np.array([[2.0, 0.0], [0.0, 1.0]])
    logits = np.log(np.array([[0.8, 0.25], [0.2, 0.75]]))

    actual = normalized_poisson_reference_loss(counts, logits)
    expected = -(2.0 * np.log(0.8) + np.log(0.75)) / 3.0

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("shape", "shape"),
        ("negative", "non-negative"),
        ("empty_species", "positive count"),
        ("non_finite", "finite"),
    ],
)
def test_reference_loss_rejects_invalid_tensors(mutation: str, message: str) -> None:
    counts = np.array([[1.0, 0.0], [0.0, 1.0]])
    logits = np.zeros((2, 2))
    if mutation == "shape":
        logits = np.zeros((3, 2))
    elif mutation == "negative":
        counts[0, 0] = -1
    elif mutation == "empty_species":
        counts[:, 0] = 0
    elif mutation == "non_finite":
        logits[0, 0] = np.nan

    with pytest.raises(ValueError, match=message):
        normalized_poisson_reference_loss(counts, logits)


def test_gate_excludes_when_any_required_check_fails(
    study_config,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "official"
    checkout.mkdir()
    pilot = tmp_path / "pilot.parquet"
    pd.DataFrame(
        {
            "seed": [1, 2, 3],
            "passed": [True, True, True],
            "runtime_seconds": [1.0, 1.1, 0.9],
            "comparable_predictions": [True, True, True],
        }
    ).to_parquet(pilot, index=False)

    report = evaluate_deepmaxent_gate(checkout, pilot, study_config)

    assert isinstance(report, GateReport)
    assert not report.include
    assert report.official_commit == "unavailable"
    assert report.reasons
    json.dumps(asdict(report))


def test_pilot_writes_three_seed_runtime_and_surface_checks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_fit(self, features, counts, seed, epochs):
        prediction = np.full(
            counts.shape,
            1.0 / len(counts),
            dtype=float,
        )
        return prediction, 2.0, 1.0

    monkeypatch.setattr(DeepMaxentAdapter, "fit_pilot", fake_fit)

    path = run_deepmaxent_pilot(
        tmp_path / "official",
        tmp_path / "pilot.parquet",
        seeds=(3, 5, 7),
        n_sites=64,
        n_species=4,
        epochs=2,
        full_site_count=1_000,
        full_epochs=100,
    )
    rows = pd.read_parquet(path)

    assert rows.seed.tolist() == [3, 5, 7]
    assert rows.passed.all()
    assert rows.comparable_predictions.all()
    assert rows.runtime_seconds.gt(0).all()
    assert rows.full_site_count.eq(1_000).all()


def test_failed_gate_refuses_full_run(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "official_commit": "abc",
                "formula_check_passed": True,
                "repository_example_passed": False,
                "multi_seed_pilot_passed": True,
                "projected_calendar_days": 0.2,
                "comparable_predictions_passed": True,
                "include": False,
                "reasons": ["repository example failed"],
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-deepmaxent", "--gate", str(gate)]) == 1
