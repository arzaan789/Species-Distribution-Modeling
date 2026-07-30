"""Restartable orchestration and completeness auditing for simulations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from provenance_sdm.backgrounds import make_backgrounds
from provenance_sdm.config import StudyConfig
from provenance_sdm.landscape import Landscape
from provenance_sdm.maxent import fit_maxent
from provenance_sdm.metrics import generate_unbiased_evaluation, truth_metrics
from provenance_sdm.observation import simulate_observations, simulate_programmes
from provenance_sdm.virtual_species import simulate_species_truth


RESULT_KEY_COLUMNS = (
    "community_seed",
    "alignment",
    "bias_level",
    "species_id",
    "background_arm",
)
PRIMARY_METRICS = (
    "suitability_spearman",
    "integrated_error",
    "unbiased_auc",
    "response_curve_error",
    "top10_overlap",
)


def seed_for(study_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit child seed from a semantic key."""

    encoded = json.dumps([study_seed, *parts], separators=(",", ":"), default=str)
    digest = hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def expected_simulation_keys(config: StudyConfig) -> pd.DataFrame:
    """Return the frozen composite-key matrix for all planned primary fits."""

    rows = []
    simulation = config.simulation
    for community_index in range(simulation.n_communities):
        community_seed = seed_for(simulation.seed, "community", community_index)
        for alignment in simulation.alignments:
            for bias_level in simulation.bias_levels:
                for species_index in range(simulation.n_species):
                    for arm in config.background_arms:
                        rows.append(
                            (
                                community_seed,
                                alignment,
                                bias_level,
                                f"sp_{species_index:03d}",
                                arm,
                            )
                        )
    return pd.DataFrame(rows, columns=RESULT_KEY_COLUMNS)


def _atomic_parquet(rows: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    rows.to_parquet(temporary, index=False)
    check = pd.read_parquet(temporary)
    if len(check) != len(rows) or check.duplicated(list(RESULT_KEY_COLUMNS)).any():
        temporary.unlink(missing_ok=True)
        raise ValueError("incremental simulation artifact failed key validation")
    os.replace(temporary, path)


def _landscape_hash(landscape: Landscape) -> str:
    cell_digest = pd.util.hash_pandas_object(
        landscape.cells,
        index=True,
    ).to_numpy().tobytes()
    metadata = json.dumps(
        [landscape.crs, landscape.feature_names],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(metadata + cell_digest).hexdigest()


def run_simulation(
    config: StudyConfig,
    landscape: Landscape,
    output_dir: Path,
) -> Path:
    """Run only missing fits and atomically update the primary result artifact."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "simulation_metrics.parquet"
    failure_path = destination / "simulation_failures.csv"
    expected = expected_simulation_keys(config)
    if result_path.exists():
        completed = pd.read_parquet(result_path)
        if completed.duplicated(list(RESULT_KEY_COLUMNS)).any():
            raise ValueError("existing simulation results contain duplicate keys")
    else:
        completed = pd.DataFrame()
    completed_keys = (
        set(map(tuple, completed.loc[:, RESULT_KEY_COLUMNS].itertuples(index=False, name=None)))
        if not completed.empty
        else set()
    )
    missing_keys = set(
        map(tuple, expected.itertuples(index=False, name=None))
    ).difference(completed_keys)
    if not missing_keys:
        return result_path

    new_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    simulation = config.simulation
    landscape_hash = _landscape_hash(landscape)
    for community_index in range(simulation.n_communities):
        community_seed = seed_for(simulation.seed, "community", community_index)
        truth = simulate_species_truth(
            landscape,
            simulation.n_species,
            seed_for(simulation.seed, "truth", community_seed),
        )
        for bias_level in simulation.bias_levels:
            programmes = simulate_programmes(
                landscape,
                simulation.n_programmes,
                bias_level,
                seed_for(simulation.seed, "programmes", community_seed, bias_level),
            )
            for alignment in simulation.alignments:
                scenario_keys = [
                    key
                    for key in missing_keys
                    if key[0] == community_seed
                    and key[1] == alignment
                    and key[2] == bias_level
                ]
                if not scenario_keys:
                    continue
                observed = simulate_observations(
                    truth,
                    programmes,
                    alignment,
                    bias_level,
                    simulation.min_records,
                    simulation.max_records,
                    seed_for(
                        simulation.seed,
                        "observations",
                        community_seed,
                        alignment,
                        bias_level,
                    ),
                )
                for species in truth:
                    requested_arms = [
                        arm
                        for arm in config.background_arms
                        if (
                            community_seed,
                            alignment,
                            bias_level,
                            species.species_id,
                            arm,
                        )
                        in missing_keys
                    ]
                    if not requested_arms:
                        continue
                    try:
                        backgrounds = make_backgrounds(
                            observed,
                            species.species_id,
                            simulation.background_cells,
                            seed_for(
                                simulation.seed,
                                "backgrounds",
                                community_seed,
                                alignment,
                                bias_level,
                                species.species_id,
                            ),
                            minimum_cells=simulation.minimum_background_cells,
                        )
                        presence = observed.records.query(
                            "species_id == @species.species_id"
                        ).merge(landscape.cells, on="cell_id", how="left")
                        evaluation = generate_unbiased_evaluation(
                            species,
                            n_presence=500,
                            n_background=500,
                            seed=seed_for(
                                simulation.seed,
                                "evaluation",
                                community_seed,
                                alignment,
                                bias_level,
                                species.species_id,
                            ),
                        )
                        for arm in requested_arms:
                            model = fit_maxent(
                                presence,
                                backgrounds[arm],
                                landscape.feature_names,
                                regularization=1.0,
                                seed=seed_for(
                                    simulation.seed,
                                    "model",
                                    community_seed,
                                    alignment,
                                    bias_level,
                                    species.species_id,
                                    arm,
                                ),
                            )
                            prediction = model.predict_suitability(landscape)
                            evaluation_score = prediction[
                                evaluation.cell_id.to_numpy(dtype=int)
                            ]
                            metrics = truth_metrics(
                                prediction,
                                species,
                                evaluation.label.to_numpy(),
                                evaluation_score,
                            )
                            new_rows.append(
                                {
                                    "community_seed": community_seed,
                                    "alignment": alignment,
                                    "bias_level": bias_level,
                                    "species_id": species.species_id,
                                    "background_arm": arm,
                                    "record_count": len(presence),
                                    "niche_breadth": species.niche_breadth,
                                    "background_cells": len(backgrounds[arm]),
                                    "landscape_hash": landscape_hash,
                                    "unsupported_mass": (
                                        float(backgrounds[arm].unsupported_mass.iloc[0])
                                        if arm == "pm_tgb"
                                        else 0.0
                                    ),
                                    **metrics,
                                }
                            )
                    except Exception as exc:
                        for arm in requested_arms:
                            failures.append(
                                {
                                    "community_seed": community_seed,
                                    "alignment": alignment,
                                    "bias_level": bias_level,
                                    "species_id": species.species_id,
                                    "background_arm": arm,
                                    "exception_type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            )
                        continue

                    combined = pd.concat(
                        [completed, pd.DataFrame(new_rows)],
                        ignore_index=True,
                    )
                    _atomic_parquet(combined, result_path)
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    return result_path


def audit_simulation(path: Path, config: StudyConfig) -> dict[str, object]:
    """Compare actual fits with the exact expected key matrix."""

    result_path = Path(path)
    actual = pd.read_parquet(result_path)
    expected = expected_simulation_keys(config)
    duplicates = int(actual.duplicated(list(RESULT_KEY_COLUMNS)).sum())
    comparison = expected.merge(
        actual.loc[:, RESULT_KEY_COLUMNS].drop_duplicates(),
        on=list(RESULT_KEY_COLUMNS),
        how="left",
        indicator=True,
    )
    missing_rows = comparison.query("_merge == 'left_only'").drop(columns="_merge")
    unexpected_comparison = (
        actual.loc[:, RESULT_KEY_COLUMNS]
        .drop_duplicates()
        .merge(
            expected,
            on=list(RESULT_KEY_COLUMNS),
            how="left",
            indicator=True,
        )
    )
    unexpected_rows = unexpected_comparison.query(
        "_merge == 'left_only'"
    ).drop(columns="_merge")
    finite = all(
        metric in actual and np.isfinite(actual[metric].to_numpy(dtype=float)).all()
        for metric in PRIMARY_METRICS
    )
    arm_counts = actual.groupby(
        ["community_seed", "alignment", "bias_level", "species_id"]
    ).background_arm.nunique()
    complete_arms = bool(arm_counts.eq(len(config.background_arms)).all())
    failure_path = result_path.parent / "simulation_failures.csv"
    failed = len(pd.read_csv(failure_path)) if failure_path.exists() else 0
    status = (
        "passed"
        if missing_rows.empty
        and unexpected_rows.empty
        and duplicates == 0
        and finite
        and complete_arms
        else "failed"
    )
    audit: dict[str, object] = {
        "status": status,
        "expected": len(expected),
        "completed": len(actual),
        "failed": failed,
        "missing": len(missing_rows),
        "unexpected": len(unexpected_rows),
        "duplicates": duplicates,
        "finite_primary_metrics": finite,
        "complete_background_arms": complete_arms,
        "missing_keys": missing_rows.to_dict(orient="records"),
        "unexpected_keys": unexpected_rows.to_dict(orient="records"),
        "configuration_hash": hashlib.sha256(
            repr(config).encode("utf-8")
        ).hexdigest(),
        "landscape_hashes": sorted(actual.landscape_hash.unique().tolist())
        if "landscape_hash" in actual
        else [],
    }
    audit_path = result_path.parent / "simulation_audit.json"
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit
